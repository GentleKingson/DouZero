"""Model V2: shared state-action model with multi-head outputs (P05).

This is the unified model that replaces the three role-specific legacy MLPs with
one shared backbone plus role conditioning. It is selected by
``model_version=v2`` and consumes an :class:`~douzero.observation.encode_v2.ObservationV2`
(public inputs only).

Architecture (mirrors the spec in ``docs/model_v2.md``)::

    state block (once) ──┐
    public context ──────┼── StateEncoder ──► state_trunk ──────────────┐
                         │                                                ├──►
    history tokens+mask ── HistoryEncoder ──► history_summary ───────────┤    StateActionFusion
                                                                        ├──► (per action) ──► ValueHeads ──► ModelOutput
    action feature rows ── ActionEncoder ──► action_embeddings (N) ──────┤      (+ role embed)
                                                                        │
    acting role ───────────────────────────────────────────────────────────┘

Contract
--------
- The state and history are encoded **once per decision** (not per legal
  action). Only the action path and the final fusion run per candidate. This
  is the P04 factorized property, generalized to the V2 inputs.
- Variable legal-action counts are handled natively: the action encoder takes
  ``(N, action_width)`` and the fusion broadcasts the shared trunk across N.
  No fixed maximum action count is assumed.
- Masks are respected: the history encoder takes a padding mask so padded
  history slots never affect the output; padded action rows are excluded from
  selection via :meth:`ModelOutput.argmax_win`.
- The model is deterministic under ``eval()`` (no BatchNorm; dropout is a
  configured no-op at the default 0.0).

Checkpoint compatibility
------------------------
V2 weights are NOT compatible with the legacy / factorized models (different
parameter names and shapes). The checkpoint manifest records
``model_version="v2"``; the strict V2 loader (P05: ``load_v2_checkpoint`` /
``load_v2_position_weights`` / ``load_v2_model``) validates the manifest's
schema/config/ruleset identity against runtime expectations and rejects a
mismatch rather than permissively partial-loading. Only the legacy
per-position loader remains permissive until P16. Legacy model files are
untouched.

Imperfect-information boundary
------------------------------
This module imports ONLY from :mod:`douzero.observation` (public) and the
sibling V2 modules. It MUST NOT import :mod:`douzero.observation.privileged`
or accept a :class:`PrivilegedObservation`. The deployment guard
(:class:`~douzero.evaluation.deep_agent.DeepAgentV2`) enforces this by type at
the boundary; this model itself consumes only the tensor blocks of
``ObservationV2``.
"""

from __future__ import annotations

import torch
from torch import nn

from douzero.observation.schema import (
    FeatureSchemaManifest,
    action_width as schema_action_width,
    context_width as schema_context_width,
    history_token_width as schema_history_token_width,
    state_width as schema_state_width,
)

from .action_encoder import ActionEncoder
from .config import HISTORY_ENCODER_TRANSFORMER, ModelV2Config, SUPPORTED_ROLES
from .fusion import StateActionFusion
from .heads import BiddingHeads, PriorHead, StrategyAuxiliaryHeads, ValueHeads
from .history_encoder import build_history_encoder
from .output import BatchedModelOutput, BiddingModelOutput, ModelOutput
from .state_encoder import StateEncoder

#: The number of card-set sub-blocks the state encoder consumes from the state
#: block. Must match the schema's card-vector state fields (see
#: ``build_v2_schema``). Asserted at construction.
_NUM_STATE_CARD_FIELDS = 6  # my, other, landlord/landlord_down/landlord_up played, last_move
#: The number of card-set sub-blocks in the public context block.
_NUM_CONTEXT_CARD_FIELDS = 2  # bottom_cards_revealed, bottom_cards_unplayed


class ModelV2(nn.Module):
    """Shared state-action value model (P05).

    Parameters
    ----------
    schema:
        The :class:`FeatureSchemaManifest` the model is constructed against.
        Every input width is derived from it, so a schema change surfaces as a
        shape mismatch at construction (or at forward) rather than a silent
        misconfiguration. Bind this to the checkpoint manifest for loading.
    config:
        Architecture hyperparameters. Defaults match ``ModelV2Config()`` and
        ``configs/enhanced.yaml``.
    """

    model_access = "public"

    def __init__(
        self,
        schema: FeatureSchemaManifest,
        config: ModelV2Config | None = None,
    ) -> None:
        super().__init__()
        self.schema = schema
        self.config = config or ModelV2Config()
        cfg = self.config

        # --- Derive every input width from the schema (no magic numbers). ---
        self._state_width = schema_state_width(schema)
        self._action_width = schema_action_width(schema)
        self._history_token_width = schema_history_token_width(schema)
        self._context_width = schema_context_width(schema)
        self._card_vector_dim = schema.card_vector_dim
        self._max_history_len = schema.max_history_len

        # The non-card portion of the state block (one-hots + counts) feeds the
        # state encoder's flat-context path. Its width is the state width minus
        # the card-vector fields' total width.
        state_card_total = self._card_vector_dim * _NUM_STATE_CARD_FIELDS
        if state_card_total > self._state_width:
            raise ValueError(
                f"schema state_width {self._state_width} is smaller than the "
                f" {_NUM_STATE_CARD_FIELDS} card-vector fields "
                f"({state_card_total}); the schema layout is unexpected"
            )
        # The flat context passed to the state encoder is: the non-card state
        # fields + the ENTIRE public-context block (both its card fields and
        # its small fields). The state encoder embeds the state card fields and
        # the context card fields through the shared card projection, so only
        # the non-card state + non-card context go into context_flat. We pass
        # the context block's non-card fields as context too, to keep the input
        # width stable when a future public field is added.
        context_card_total = self._card_vector_dim * _NUM_CONTEXT_CARD_FIELDS
        non_card_state_width = self._state_width - state_card_total
        non_card_context_width = self._context_width - context_card_total
        if non_card_context_width < 0:
            raise ValueError(
                f"schema context_width {self._context_width} is smaller than "
                f"the {_NUM_CONTEXT_CARD_FIELDS} context card-vector fields "
                f"({context_card_total}); the schema layout is unexpected"
            )
        flat_context_width = non_card_state_width + non_card_context_width

        # --- Build the sub-modules. ---
        # The state encoder receives BOTH the state card fields and the
        # context card fields (bottom revealed + bottom unplayed), in a fixed
        # order: [state card fields..., context card fields...]. The total
        # count is passed so the encoder can size its per-field projection and
        # enforce the count at forward time (a wrong count is a caller bug, not
        # a silent misconfiguration).
        total_card_fields = _NUM_STATE_CARD_FIELDS + _NUM_CONTEXT_CARD_FIELDS
        self.state_encoder = StateEncoder(
            card_vector_dim=self._card_vector_dim,
            num_card_fields=total_card_fields,
            flat_context_width=flat_context_width,
            hidden_size=cfg.hidden_size,
        )
        self.history_encoder = build_history_encoder(
            token_width=self._history_token_width,
            hidden_size=cfg.hidden_size,
            max_history_len=self._max_history_len,
            backend=cfg.history_encoder,
            num_layers=cfg.history_layers,
            num_heads=cfg.history_heads,
            dropout=cfg.history_dropout,
        )
        strategy_width = 0
        if cfg.strategy_features_enabled:
            from douzero.strategy.features import STRATEGY_FEATURE_WIDTH

            strategy_width = STRATEGY_FEATURE_WIDTH
        self.action_encoder = ActionEncoder(
            action_width=self._action_width,
            hidden_size=cfg.hidden_size,
            strategy_width=strategy_width,
        )
        self.fusion = StateActionFusion(
            hidden_size=cfg.hidden_size,
            role_embedding_dim=cfg.role_embedding_dim,
            num_roles=len(SUPPORTED_ROLES),
            num_layers=cfg.mlp_layers,
            dropout=cfg.mlp_dropout,
        )
        self.heads = ValueHeads(
            hidden_size=cfg.hidden_size,
            score_clamp=cfg.score_clamp,
        )

        # Bidding is a distinct neutral-seat decision space. It never calls
        # ActionEncoder (which represents candidate card moves) and is absent
        # from the module/state_dict when disabled.
        self.bidding_schema = None
        self.bidding_heads: BiddingHeads | None = None
        if cfg.bidding_enabled:
            from douzero.observation.bidding import BIDDING_ACTIONS, build_bidding_schema

            self.bidding_schema = build_bidding_schema()
            self.bidding_heads = BiddingHeads(
                self.bidding_schema.input_width,
                cfg.bidding_hidden_size,
                num_bid_actions=len(BIDDING_ACTIONS),
                score_clamp=cfg.score_clamp,
                uncertainty_enabled=cfg.bidding_uncertainty_enabled,
            )

        # P07: optional belief fusion. When ``belief_enabled`` the model gains
        # a projection that maps the (frozen, externally-computed) belief
        # posterior features into the trunk and adds them to the state
        # representation before fusion. The belief FEATURES are produced by a
        # separate :class:`~douzero.belief.model.BeliefModel` (pretrained then
        # frozen); this model never owns or reads true hidden hands. The
        # architecture delta is exactly ``belief_enabled`` (already an identity
        # axis in :meth:`ModelV2Config.compatibility_dict`), so no checkpoint
        # identity-version bump is required and existing belief-disabled
        # checkpoints remain loadable unchanged.
        self.belief_proj: nn.Module | None = None
        if cfg.belief_enabled:
            from douzero.belief.model import BELIEF_FEATURE_DIM

            self.belief_proj = nn.Linear(BELIEF_FEATURE_DIM, cfg.hidden_size)

        self.style_encoder: nn.Module | None = None
        if cfg.style_enabled:
            from douzero.style.encoder import StyleEncoder

            self.style_encoder = StyleEncoder(
                output_dim=cfg.hidden_size,
                hidden_dim=cfg.style_embedding_dim,
            )

        # P08: optional listwise policy-prior head. When ``human_prior_enabled``
        # the model gains a :class:`~douzero.models_v2.heads.PriorHead` that
        # emits one prior logit per legal action, trained by listwise BC. The
        # head reads only the fused public action representation; it never sees
        # hidden hands. ``human_prior_enabled`` is already an identity axis in
        # :meth:`ModelV2Config.compatibility_dict`, so flipping it changes the
        # checkpoint hash and the strict loader rejects a cross-load (a
        # prior-enabled checkpoint has extra ``prior_head.*`` keys).
        self.prior_head: PriorHead | None = None
        if cfg.human_prior_enabled:
            self.prior_head = PriorHead(hidden_size=cfg.hidden_size)

        self.strategy_aux_heads: StrategyAuxiliaryHeads | None = None
        if cfg.strategy_aux_enabled:
            self.strategy_aux_heads = StrategyAuxiliaryHeads(
                hidden_size=cfg.hidden_size
            )

        # Cache the role-name -> index map so forward() does no dict lookup.
        self._role_to_index = {role: i for i, role in enumerate(SUPPORTED_ROLES)}

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def role_index(self, role: str) -> int:
        """Return the integer index of ``role`` for the role embedding table.

        Raises ``ValueError`` for an unknown role so a malformed observation
        fails at the boundary rather than producing a silent default.
        """
        try:
            return self._role_to_index[role]
        except KeyError as exc:
            raise ValueError(
                f"Unknown acting role {role!r}. Supported roles: {SUPPORTED_ROLES}"
            ) from exc

    def forward_bidding(self, observation) -> BiddingModelOutput:
        """Run the public bidding path without encoding card-play actions."""
        from douzero.observation.bidding import BiddingObservationV2

        if self.bidding_heads is None or self.bidding_schema is None:
            raise RuntimeError(
                "forward_bidding requires ModelV2Config(bidding_enabled=True)"
            )
        if not isinstance(observation, BiddingObservationV2):
            raise TypeError("forward_bidding requires a public BiddingObservationV2")
        expected_hash = self.bidding_schema.stable_hash()
        if observation.feature_schema_hash != expected_hash:
            raise ValueError(
                "bidding feature schema mismatch: observation was encoded under "
                f"{observation.feature_schema_hash}, model expects {expected_hash}"
            )
        parameter = next(self.bidding_heads.parameters())
        features = observation.to_tensor(parameter.device).to(parameter.dtype)
        head_out = self.bidding_heads(features)
        mask = torch.from_numpy(observation.bid_action_mask.copy()).to(
            device=parameter.device, dtype=torch.bool
        )
        if self.config.nan_guard:
            from .numerical import assert_finite

            for name in (
                "bid_logits", "landlord_win_logit", "expected_landlord_score"
            ):
                assert_finite(head_out[name], name)
            if head_out["uncertainty"] is not None:
                assert_finite(head_out["uncertainty"], "bidding_uncertainty")
        return BiddingModelOutput(
            bid_logits=head_out["bid_logits"],
            bid_action_mask=mask,
            landlord_win_logit=head_out["landlord_win_logit"],
            expected_landlord_score=head_out["expected_landlord_score"],
            uncertainty=head_out["uncertainty"],
        )

    def strategy_feature_config(self):
        """Return the public feature config, or ``None`` for the P08 path."""

        if not self.config.strategy_features_enabled:
            return None
        from douzero.strategy.config import StrategyFeatureConfig

        return StrategyFeatureConfig(
            hand_enabled=self.config.strategy_hand_enabled,
            structure_enabled=self.config.strategy_structure_enabled,
            control_enabled=self.config.strategy_control_enabled,
            cooperation_enabled=self.config.strategy_cooperation_enabled,
            risk_enabled=self.config.strategy_risk_enabled,
            node_budget=self.config.strategy_node_budget,
            time_budget_ms=self.config.strategy_time_budget_ms,
        )

    def forward(
        self,
        state_card_vectors: tuple[torch.Tensor, ...],
        state_context_flat: torch.Tensor,
        context_card_vectors: tuple[torch.Tensor, ...],
        context_flat: torch.Tensor,
        history_tokens: torch.Tensor,
        history_key_padding_mask: torch.Tensor,
        action_features: torch.Tensor,
        action_mask: torch.Tensor,
        acting_role: str,
        belief_features: torch.Tensor | None = None,
        belief_stop_gradient: bool = True,
        allow_missing_belief_features: bool = False,
        strategy_features: torch.Tensor | None = None,
        style_features: torch.Tensor | None = None,
    ) -> ModelOutput:
        """Run a full forward pass for one decision.

        All inputs come from an :class:`ObservationV2`; the caller (or
        :class:`~douzero.models_v2.batch.observation_to_model_inputs`) is
        responsible for splitting the state/context blocks into their
        card-vector and flat-field portions. This explicit split keeps the
        model's tensor contract inspectable and testable.

        Parameters
        ----------
        state_card_vectors:
            The card-vector fields of the state block, in schema order (my,
            other, landlord/landlord_down/landlord_up played, last_move). Each
            ``(card_vector_dim,)`` float.
        state_context_flat:
            The non-card state fields, flattened in schema order.
        context_card_vectors:
            The card-vector fields of the public-context block (bottom revealed
            + bottom unplayed). Each ``(card_vector_dim,)`` float.
        context_flat:
            The non-card context fields (bid, phase, rocket, multiplier,
            ruleset), flattened in schema order.
        history_tokens:
            Shape ``(max_history_len, history_token_width)`` float. Padded
            slots should be zero.
        history_key_padding_mask:
            Shape ``(max_history_len,)`` bool, ``True`` for PADDING.
        action_features:
            Shape ``(N, action_width)`` float, one row per legal action (N >= 1).
        action_mask:
            Shape ``(N,)`` bool, ``True`` for a valid action. For the common
            case of no padding, pass all-True.
        acting_role:
            The acting role name (``"landlord"`` / ``"landlord_up"`` /
            ``"landlord_down"``).
        belief_features:
            Optional ``(BELIEF_FEATURE_DIM,)`` float tensor — the **constrained**
            belief posterior features from a
            :class:`~douzero.belief.model.BeliefModel`. Only consumed when
            ``config.belief_enabled``; passing it to a belief-disabled model
            raises ``ValueError``. When ``belief_enabled`` and ``None``, the
            model **fails closed** by default (a belief-trained checkpoint must
            not silently degrade to a zero-feature baseline) — pass
            ``allow_missing_belief_features=True`` to opt into the zero-vector
            behaviour for ablations only. The features are cast to the trunk's
            device/dtype and detached before fusion only when
            ``belief_stop_gradient`` is true.
        belief_stop_gradient:
            ``True`` preserves the pretrained/frozen path. ``False`` keeps the
            incoming feature graph intact, allowing value loss to update the
            belief encoder when the features came from the differentiable
            PyTorch constrained-marginal path.
        allow_missing_belief_features:
            When ``belief_enabled`` and ``belief_features`` is None, the model
            raises by default; set this True only for explicit ablations that
            want the zero-vector baseline.
        strategy_features:
            Optional ``(N, STRATEGY_FEATURE_WIDTH)`` public tactical feature
            matrix. Required exactly when ``strategy_features_enabled`` is
            true; rejected by the strategy-disabled action encoder.
        style_features:
            Optional public other-player style statistics. Required when the
            checkpoint was built with ``style_enabled=True`` and rejected
            otherwise. The vector is derived solely from public action history.

        Returns
        -------
        ModelOutput
            The multi-head output over the N actions.

        Raises
        ------
        ValueError
            If there are zero legal actions (the model cannot select from an
            empty action set).
        NumericalError
            If ``nan_guard`` is enabled and any fused/head tensor contains NaN
            or Inf (catches both bad inputs and bad weights).
        """
        # Bug #6: zero legal actions is a caller error. The model must not
        # return an empty output (downstream argmax/selection would silently
        # pick index 0 of an empty tensor). Fail loudly here.
        if action_features.shape[0] == 0:
            raise ValueError(
                "ModelV2.forward received zero legal actions (action_features "
                "has zero rows). A decision with no legal actions is undefined; "
                "the caller must short-circuit before calling the model."
            )

        # --- Encode the shared state once. ---
        all_card_vectors = state_card_vectors + context_card_vectors
        full_context_flat = torch.cat([state_context_flat, context_flat], dim=-1)
        state_trunk = self.state_encoder(all_card_vectors, full_context_flat)

        # P07: optional belief fusion. The belief features (posterior expected
        # counts, entropy, key-card probabilities) are produced by a separate,
        # pretrained-and-frozen BeliefModel from the PUBLIC observation only;
        # they are projected into the trunk and added. ``belief_stop_gradient``
        # (default True) detaches the features so value loss never updates the
        # frozen belief weights — the "pretrain belief then freeze" path. A
        # caller doing joint training passes ``belief_stop_gradient=False``.
        # When ``belief_enabled`` but no features are supplied, the model
        # FAILS CLOSED by default (a belief-trained checkpoint must not silently
        # degrade to a zero-feature baseline at deployment). Pass
        # ``allow_missing_belief_features=True`` to opt into the zero-vector
        # behaviour for ablations / unit tests.
        if self.belief_proj is not None:
            from douzero.belief.model import BELIEF_FEATURE_DIM

            if belief_features is None:
                if not allow_missing_belief_features:
                    raise ValueError(
                        "belief_features were not supplied to a belief-ENABLED "
                        "ModelV2. A checkpoint trained with belief fusion must "
                        "receive the frozen belief posterior at every forward; "
                        "pass allow_missing_belief_features=True only for "
                        "explicit ablations."
                    )
                belief_features = state_trunk.new_zeros(BELIEF_FEATURE_DIM)
            else:
                # Exact shape check (review: only the trailing dim was checked,
                # and the device/dtype were not aligned with the trunk).
                if tuple(belief_features.shape) != (BELIEF_FEATURE_DIM,):
                    raise ValueError(
                        f"belief_features must have shape "
                        f"({BELIEF_FEATURE_DIM},), got "
                        f"{tuple(belief_features.shape)}"
                    )
                # Align device/dtype with the trunk so a GPU value model
                # receiving CPU belief features does not mismatch.
                belief_features = belief_features.to(
                    device=state_trunk.device, dtype=state_trunk.dtype
                )
                if belief_stop_gradient:
                    belief_features = belief_features.detach()
            state_trunk = state_trunk + self.belief_proj(belief_features)
        elif belief_features is not None:
            raise ValueError(
                "belief_features were passed to a belief-DISABLED ModelV2 "
                "(config.belief_enabled is False). Drop belief_features or "
                "rebuild the model with belief_enabled=True."
            )
        del belief_stop_gradient  # consumed; guard against accidental reuse

        if self.style_encoder is not None:
            from douzero.style.features import STYLE_FEATURE_WIDTH

            if style_features is None:
                raise ValueError(
                    "style_features were not supplied to a style-enabled ModelV2"
                )
            if tuple(style_features.shape) != (STYLE_FEATURE_WIDTH,):
                raise ValueError(
                    f"style_features must have shape ({STYLE_FEATURE_WIDTH},), "
                    f"got {tuple(style_features.shape)}"
                )
            style_features = style_features.to(
                device=state_trunk.device, dtype=state_trunk.dtype
            )
            state_trunk = state_trunk + self.style_encoder(style_features)
        elif style_features is not None:
            raise ValueError(
                "style_features were passed to a style-disabled ModelV2"
            )

        # --- Encode the history once. ---
        history_summary = self.history_encoder(history_tokens, history_key_padding_mask)

        # --- Encode each action. ---
        action_embeddings = self.action_encoder(action_features, strategy_features)

        # --- Fuse shared trunk + history + per-action + role. ---
        role_idx = self.role_index(acting_role)
        fused = self.fusion(state_trunk, history_summary, action_embeddings, role_idx)

        # Bug #5: runtime NaN/Inf guard. Imported once and applied to the fused
        # representation AND every head output. The fused check catches bad
        # inputs and bad fusion/encoder weights; the head-output checks catch a
        # NaN/Inf originating in a head (e.g. a NaN weight in a score head,
        # which the score clamp cannot remove — torch.clamp(nan) is nan).
        if self.config.nan_guard:
            from .numerical import assert_finite
            assert_finite(fused, "fused")

        # --- Heads. ---
        head_out = self.heads(fused)

        if self.config.nan_guard:
            # Guard EVERY head output, including the conditional score heads
            # and the derived score_mean. The heads apply a clamp to the score
            # outputs, but clamp only bounds finite values — it does NOT catch
            # NaN (torch.clamp(nan, -c, c) is nan), and a NaN weight in
            # score_win_head / score_loss_head produces a NaN score output that
            # the clamp cannot remove. Checking all five outputs catches a NaN
            # or Inf wherever it originates (win head, either score head, or
            # the derived mean), so a corrupted score head cannot silently
            # propagate into the decision policy / loss.
            assert_finite(head_out["win_logit"], "win_logit")
            assert_finite(head_out["score_if_win"], "score_if_win")
            assert_finite(head_out["score_if_loss"], "score_if_loss")
            assert_finite(head_out["p_win"], "p_win")
            assert_finite(head_out["score_mean"], "score_mean")

        # P08: optional listwise prior head. Only computed when the model was
        # built with ``human_prior_enabled``; the head reads only the fused
        # public action representation (no hidden hands, no privileged label).
        prior_logit: torch.Tensor | None = None
        if self.prior_head is not None:
            prior_logit = self.prior_head(fused)
            if self.config.nan_guard:
                assert_finite(prior_logit, "prior_logit")

        aux_out: dict[str, torch.Tensor | None] = {
            "min_turns_after": None,
            "regain_initiative_logit": None,
            "teammate_finish_logit": None,
            "spring_probability_logit": None,
            "structure_cost": None,
        }
        if self.strategy_aux_heads is not None:
            aux_out.update(self.strategy_aux_heads(fused))
            if self.config.nan_guard:
                for name, tensor in aux_out.items():
                    assert_finite(tensor, name)

        return ModelOutput(
            win_logit=head_out["win_logit"],
            score_if_win=head_out["score_if_win"],
            score_if_loss=head_out["score_if_loss"],
            p_win=head_out["p_win"],
            score_mean=head_out["score_mean"],
            action_mask=action_mask,
            prior_logit=prior_logit,
            **aux_out,
        )

    def forward_batched(
        self,
        state_card_vectors: tuple[torch.Tensor, ...],
        state_context_flat: torch.Tensor,
        context_card_vectors: tuple[torch.Tensor, ...],
        context_flat: torch.Tensor,
        history_tokens: torch.Tensor,
        history_key_padding_mask: torch.Tensor,
        action_features: torch.Tensor,
        action_mask: torch.Tensor,
        acting_role: torch.Tensor,
        belief_features: torch.Tensor | None = None,
        belief_stop_gradient: bool = True,
        allow_missing_belief_features: bool = False,
        strategy_features: torch.Tensor | None = None,
        style_features: torch.Tensor | None = None,
    ) -> BatchedModelOutput:
        """Run one vectorized forward over padded ``[B, Amax, ...]`` inputs.

        This method adds no modules or parameters, so the model configuration
        hash and ``state_dict`` identity are exactly those of the scalar API.
        Padding is carried through the cheap action path and is never eligible
        for selection or gathered learner loss.
        """
        if action_features.ndim != 3:
            raise ValueError("action_features must have shape (B, A, action_width)")
        batch, actions, _ = action_features.shape
        if batch == 0 or actions == 0:
            raise ValueError("ModelV2.forward_batched requires a non-empty batch")
        if action_mask.shape != (batch, actions) or action_mask.dtype != torch.bool:
            raise ValueError("action_mask must be bool with shape (B, A)")
        legal_rows = action_mask.any(dim=1).all()
        if action_mask.device.type == "cuda":
            torch._assert_async(
                legal_rows, "every batched decision must contain a legal action"
            )
        elif not bool(legal_rows):
            raise ValueError("every batched decision must contain a legal action")
        if acting_role.shape != (batch,) or acting_role.dtype != torch.long:
            raise ValueError("acting_role must be long with shape (B,)")

        all_card_vectors = state_card_vectors + context_card_vectors
        full_context_flat = torch.cat([state_context_flat, context_flat], dim=-1)
        state_trunk = self.state_encoder(all_card_vectors, full_context_flat)
        if state_trunk.ndim != 2 or state_trunk.shape[0] != batch:
            raise ValueError("batched state inputs must share leading dimension B")

        if self.belief_proj is not None:
            from douzero.belief.model import BELIEF_FEATURE_DIM

            if belief_features is None:
                if not allow_missing_belief_features:
                    raise ValueError("belief_features are required by this model")
                belief_features = state_trunk.new_zeros((batch, BELIEF_FEATURE_DIM))
            if belief_features.shape != (batch, BELIEF_FEATURE_DIM):
                raise ValueError(
                    f"belief_features must have shape ({batch}, {BELIEF_FEATURE_DIM})"
                )
            belief_features = belief_features.to(state_trunk.device, state_trunk.dtype)
            if belief_stop_gradient:
                belief_features = belief_features.detach()
            state_trunk = state_trunk + self.belief_proj(belief_features)
        elif belief_features is not None:
            raise ValueError("belief_features were passed to a belief-disabled model")

        if self.style_encoder is not None:
            from douzero.style.features import STYLE_FEATURE_WIDTH

            if style_features is None or style_features.shape != (
                batch, STYLE_FEATURE_WIDTH
            ):
                raise ValueError(
                    f"style_features must have shape ({batch}, {STYLE_FEATURE_WIDTH})"
                )
            state_trunk = state_trunk + self.style_encoder(
                style_features.to(state_trunk.device, state_trunk.dtype)
            )
        elif style_features is not None:
            raise ValueError("style_features were passed to a style-disabled model")

        history_summary = self.history_encoder(
            history_tokens, history_key_padding_mask
        )
        action_embeddings = self.action_encoder(action_features, strategy_features)
        fused = self.fusion.forward_batched(
            state_trunk, history_summary, action_embeddings, acting_role
        )
        head_out = self.heads(fused)
        if self.config.nan_guard:
            from .numerical import assert_finite

            assert_finite(fused, "batched_fused")
            for name, value in head_out.items():
                assert_finite(value, f"batched_{name}")
        prior_logit = self.prior_head(fused) if self.prior_head is not None else None
        aux_out: dict[str, torch.Tensor | None] = {
            "min_turns_after": None,
            "regain_initiative_logit": None,
            "teammate_finish_logit": None,
            "spring_probability_logit": None,
            "structure_cost": None,
        }
        if self.strategy_aux_heads is not None:
            aux_out.update(self.strategy_aux_heads(fused))
        if self.config.nan_guard:
            if prior_logit is not None:
                assert_finite(prior_logit, "batched_prior_logit")
            for name, value in aux_out.items():
                if value is not None:
                    assert_finite(value, f"batched_{name}")
        return BatchedModelOutput(
            win_logit=head_out["win_logit"],
            score_if_win=head_out["score_if_win"],
            score_if_loss=head_out["score_if_loss"],
            p_win=head_out["p_win"],
            score_mean=head_out["score_mean"],
            action_mask=action_mask,
            prior_logit=prior_logit,
            **aux_out,
        )

    # ------------------------------------------------------------------ #
    # Introspection / reporting helpers
    # ------------------------------------------------------------------ #
    def parameter_count(self) -> dict[str, int]:
        """Return per-submodule and total parameter counts.

        Useful for the model-card / architecture report. Counts trainable
        parameters only (buffers like LayerNorm running stats are excluded).
        """
        counts: dict[str, int] = {}
        # Build the submodule list, including the optional belief projection
        # when present, so a belief-enabled model's parameter report is not
        # silently undercounted.
        submodules: list[tuple[str, nn.Module]] = [
            ("state_encoder", self.state_encoder),
            ("history_encoder", self.history_encoder),
            ("action_encoder", self.action_encoder),
            ("fusion", self.fusion),
            ("heads", self.heads),
        ]
        if self.belief_proj is not None:
            submodules.append(("belief_proj", self.belief_proj))
        if self.style_encoder is not None:
            submodules.append(("style_encoder", self.style_encoder))
        if self.prior_head is not None:
            submodules.append(("prior_head", self.prior_head))
        if self.strategy_aux_heads is not None:
            submodules.append(("strategy_aux_heads", self.strategy_aux_heads))
        if self.bidding_heads is not None:
            submodules.append(("bidding_heads", self.bidding_heads))
        for name, module in submodules:
            counts[name] = sum(p.numel() for p in module.parameters() if p.requires_grad)
        counts["total"] = sum(counts.values())
        return counts
