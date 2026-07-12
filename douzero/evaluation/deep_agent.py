import torch
import numpy as np

from douzero.env.env import get_obs

# Supported deployment backends. ``legacy`` is the original per-row forward
# (the LSTM runs once per legal action). ``legacy_factorized`` (P04) encodes
# the shared history/state once per decision and is numerically equivalent to
# ``legacy`` under the same weights; it loads the SAME per-position .ckpt with
# no conversion. The default stays ``legacy`` so existing behavior is unchanged
# until a caller explicitly opts into the factorized path.
SUPPORTED_BACKENDS = ('legacy', 'legacy_factorized')


def _load_model(position, model_path, backend='legacy'):
    """Load a per-position role model for a given backend.

    Both backends consume the same legacy per-position sidecar (a bare
    ``state_dict``): the factorized models declare identical submodule names
    and shapes, so ``load_legacy_position_ckpt`` + the key filter loads either
    without conversion. P16 replaces this permissive filter with a strict
    manifest load.
    """
    if backend not in SUPPORTED_BACKENDS:
        raise ValueError(
            f"Unknown backend {backend!r}; expected one of {SUPPORTED_BACKENDS}."
        )
    from douzero.checkpoint import load_legacy_position_ckpt
    if backend == 'legacy_factorized':
        from douzero.dmc.models_factorized import factorized_model_dict
        model = factorized_model_dict[position]()
    else:
        from douzero.dmc.models import model_dict
        model = model_dict[position]()
    model_state_dict = model.state_dict()
    # Legacy per-position sidecar: bare state_dict. The permissive key filter
    # below is pinned by P00 tests; P16 replaces it with a strict manifest load.
    pretrained = load_legacy_position_ckpt(model_path)
    pretrained = {k: v for k, v in pretrained.items() if k in model_state_dict}
    model_state_dict.update(pretrained)
    model.load_state_dict(model_state_dict)
    if torch.cuda.is_available():
        model.cuda()
    model.eval()
    return model


class DeepAgent:

    def __init__(self, position, model_path, backend='legacy'):
        if backend not in SUPPORTED_BACKENDS:
            raise ValueError(
                f"Unknown backend {backend!r}; expected one of {SUPPORTED_BACKENDS}."
            )
        self.backend = backend
        self.model = _load_model(position, model_path, backend=backend)

    def act(self, infoset):
        if len(infoset.legal_actions) == 1:
            return infoset.legal_actions[0]

        if self.backend == 'legacy_factorized':
            return self._act_factorized(infoset)
        return self._act_legacy(infoset)

    def _act_legacy(self, infoset):
        """Legacy per-row forward (unchanged from the original DeepAgent).

        Builds the tiled (N, ...) batches via get_obs and forwards them. The
        default backend; behavior is identical to pre-P04 DeepAgent.
        """
        obs = get_obs(infoset)

        z_batch = torch.from_numpy(obs['z_batch']).float()
        x_batch = torch.from_numpy(obs['x_batch']).float()
        if torch.cuda.is_available():
            z_batch, x_batch = z_batch.cuda(), x_batch.cuda()
        with torch.inference_mode():
            y_pred = self.model.forward(z_batch, x_batch, return_value=True)['values']
        y_pred = y_pred.detach().cpu().numpy()

        best_action_index = np.argmax(y_pred, axis=0)[0]
        best_action = infoset.legal_actions[best_action_index]

        return best_action

    def _act_factorized(self, infoset):
        """Factorized forward: consume the split observation directly.

        Uses get_obs_factorized, which encodes the shared history/state ONCE
        (never tiling them across the N legal-action rows) and produces only
        the per-action (N, 54) matrix with an N dimension. The model's
        forward_factorized runs the LSTM once on the singleton history and
        broadcasts across the per-action rows.
        """
        from douzero.env.env import get_obs_factorized

        obs = get_obs_factorized(infoset)
        z_single = torch.from_numpy(obs['z_single']).float()
        x_state_single = torch.from_numpy(obs['x_state_single']).float()
        x_action = torch.from_numpy(obs['x_action']).float()
        if torch.cuda.is_available():
            z_single = z_single.cuda()
            x_state_single = x_state_single.cuda()
            x_action = x_action.cuda()
        with torch.inference_mode():
            y_pred = self.model.forward_factorized(
                z_single, x_state_single, x_action, return_value=True
            )['values']
        y_pred = y_pred.detach().cpu().numpy()

        best_action_index = np.argmax(y_pred, axis=0)[0]
        best_action = infoset.legal_actions[best_action_index]

        return best_action


# --------------------------------------------------------------------------- #
# DeepAgentV2 (P05): public-only deployment agent for Model V2.
# --------------------------------------------------------------------------- #
class DeepAgentV2:
    """Deployment agent backed by :class:`~douzero.models_v2.model.ModelV2`.

    This is the P05 public-only deployment path. It consumes a
    :class:`~douzero.observation.encode_v2.ObservationV2` (public inputs only)
    and selects the legal action with the highest win probability by default.

    Imperfect-information boundary (the most important safety property):

    - :meth:`act_v2` accepts ONLY an :class:`ObservationV2`. Passing a
      :class:`~douzero.observation.privileged.PrivilegedObservation raises
      :class:`TypeError` BEFORE any model call. This is the canonical type
      guard required by the P03/P05 acceptance criteria.
    - This class imports the V2 model and the V2 observation *public* modules.
      It MUST NOT import :mod:`douzero.observation.privileged` except to perform
      the isinstance rejection (the import is local to :meth:`act_v2` so the
      production import graph never depends on the privileged module).
    - The model itself only ever sees the tensor blocks of the observation; it
      has no field for hidden hands.

    Identity closure (blocker #3): the agent binds to the model's verified
    ruleset identity (attached by :func:`load_v2_model`) and requires an
    explicit :class:`~douzero.env.rules.RuleSet` at construction. The agent's
    RuleSet MUST match the model's checkpoint identity (id + version + hash),
    and every observation's ruleset identity must match too. This closes the
    loophole where a standard-policy model could be served under a legacy
    observation context (the ruleset family/bid/multiplier are observation
    data values that do not necessarily change the schema layout, so the
    schema-hash check alone cannot catch the mismatch).

    The legacy ``act(infoset)`` method is also provided so this agent can drop
    into the existing :mod:`douzero.evaluation.simulation` harness, which passes
    a ``GameEnv`` infoset. It builds an :class:`ObservationV2` internally via
    :func:`~douzero.observation.encode_v2.get_obs_v2`, which recomputes the
    public unseen pool and never reads the true hidden hands.

    Parameters
    ----------
    position:
        The acting role (``"landlord"`` / ``"landlord_up"`` /
        ``"landlord_down"``). Used to build the V2 observation from an infoset.
    model:
        A constructed :class:`~douzero.models_v2.model.ModelV2`. If loaded via
        :func:`load_v2_model`, it carries ``expected_ruleset_identity`` which
        the agent validates against ``ruleset``.
    ruleset:
        REQUIRED :class:`~douzero.env.rules.RuleSet`. Used to build the V2
        observation from an infoset AND validated against the model's
        checkpoint ruleset identity (if present). Passing ``None`` is rejected
        so a standard-policy model can never silently run under a legacy
        observation context.
    decision_mode:
        How to convert the multi-head output to a single action. ``"win"``
        (default) picks the highest-``p_win`` valid action. ``"score"`` picks
        the highest expected score. P06 adds the full multi-objective policy
        set (``pure_win``, ``pure_score``, ``win_then_score``,
        ``score_then_win``, ``risk_aware``); ``win`` and ``score`` are kept
        as aliases for ``pure_win`` / ``pure_score``.
    decision_config:
        Optional :class:`~douzero.training.decision_policy.DecisionConfig`
        carrying the mode AND the tolerance/risk-penalty knobs (P06 r1 fix:
        ``decision_mode`` alone drops ``abs_tol`` / ``rel_tol`` /
        ``risk_penalty`` to their defaults, which silently disables the
        lexicographic and risk-aware modes). When supplied, this takes
        precedence over ``decision_mode``; when omitted, a
        :class:`DecisionConfig` is built from ``decision_mode`` with default
        tolerances (preserving the P05 contract for callers that only pass
        ``decision_mode``).
    """

    def __init__(
        self,
        position,
        model,
        ruleset,
        decision_mode=None,
        decision_config=None,
    ):
        from douzero.models_v2.model import ModelV2  # local import: keep the
        # production import graph (evaluation.simulation) free of a hard torch
        # model dependency at module load, mirroring the lazy imports above.
        from douzero.env.rules import RuleSet
        from douzero.training.decision_policy import (
            DecisionConfig,
            SUPPORTED_DECISION_MODES,
            canonical_mode,
        )
        if not isinstance(model, ModelV2):
            raise TypeError(
                f"DeepAgentV2 requires a ModelV2 instance, got {type(model).__name__}"
            )
        if ruleset is None:
            raise ValueError(
                "DeepAgentV2 requires an explicit RuleSet. Passing ruleset=None "
                "is rejected so a standard-policy model cannot silently run "
                "under a legacy observation context. Pass RuleSet.legacy() or "
                "RuleSet.standard() explicitly."
            )
        if not isinstance(ruleset, RuleSet):
            raise TypeError(
                f"ruleset must be a RuleSet instance, got {type(ruleset).__name__}"
            )
        # P06 r1: accept a full DecisionConfig so the tolerance and
        # risk-penalty knobs actually reach selection. Precedence:
        #   1. decision_config (if supplied) — carries mode + tolerances.
        #   2. decision_mode (if supplied) — build a DecisionConfig with
        #      default tolerances (preserves the P05 caller contract).
        #   3. Neither — default to pure_win with default tolerances.
        if decision_config is not None:
            if not isinstance(decision_config, DecisionConfig):
                raise TypeError(
                    "decision_config must be a DecisionConfig instance, got "
                    f"{type(decision_config).__name__}"
                )
            # If the caller ALSO passed a non-None decision_mode, require it
            # to agree with the config's canonical mode — silent disagreement
            # is a caller bug. A None decision_mode (the default) is inferred
            # from the config so callers don't have to repeat the mode.
            if decision_mode is not None:
                if canonical_mode(decision_mode) != decision_config.mode:
                    raise ValueError(
                        f"decision_mode {decision_mode!r} disagrees with "
                        f"decision_config.mode {decision_config.mode!r}; pass only "
                        f"one or make them agree."
                    )
            self.decision_config = decision_config
        else:
            # No DecisionConfig supplied; build one from decision_mode.
            mode = decision_mode if decision_mode is not None else "win"
            if mode not in SUPPORTED_DECISION_MODES:
                raise ValueError(
                    f"decision_mode must be one of {SUPPORTED_DECISION_MODES}, "
                    f"got {mode!r}"
                )
            self.decision_config = DecisionConfig(mode=canonical_mode(mode))
        self.decision_mode = self.decision_config.mode
        # Blocker #3: if the model carries a verified checkpoint ruleset
        # identity (attached by load_v2_model), the agent's RuleSet MUST match
        # it exactly (id + version + hash). This prevents serving a
        # standard-policy model under a legacy agent context.
        model_identity = getattr(model, "expected_ruleset_identity", None)
        if model_identity is not None:
            mid, mver, mhash = model_identity
            if (ruleset.ruleset_id, ruleset.ruleset_version, ruleset.stable_hash()) != (mid, mver, mhash):
                raise ValueError(
                    f"DeepAgentV2 ruleset identity mismatch: the model's "
                    f"checkpoint was verified under ruleset "
                    f"(id={mid!r}, version={mver!r}, hash={mhash!r}), but the "
                    f"agent's RuleSet is "
                    f"(id={ruleset.ruleset_id!r}, version={ruleset.ruleset_version!r}, "
                    f"hash={ruleset.stable_hash()!r}). A model trained under one "
                    f"ruleset must not be served under another."
                )
        self.position = position
        self.model = model
        self.ruleset = ruleset
        self.backend = "v2"
        # Bug #3: bind the agent to the model's feature schema hash. Every
        # observation forwarded through act_v2 must carry the SAME schema hash,
        # so a model trained under schema A cannot silently consume an
        # observation encoded under schema B (even if the shapes match).
        self._feature_schema_hash = model.schema.stable_hash()
        # Blocker #3: cache the agent's ruleset identity triple for the
        # per-observation check in act_v2.
        self._ruleset_identity = (
            ruleset.ruleset_id,
            ruleset.ruleset_version,
            ruleset.stable_hash(),
        )
        if torch.cuda.is_available():
            self.model.cuda()
        self.model.eval()

    # --- The canonical public-only entry point ------------------------------ #
    def act_v2(self, obs):
        """Select an action from an :class:`ObservationV2`.

        This is the type-guarded public entry point. A
        :class:`PrivilegedObservation` is rejected by type before any model
        call. Returns the selected legal action (a tuple of card ints).
        """
        # LOCAL import of the privileged module: used ONLY for the isinstance
        # rejection. It is not imported at module top level so the production
        # import graph never depends on the privileged module.
        from douzero.observation.privileged import PrivilegedObservation
        from douzero.observation.encode_v2 import ObservationV2

        if isinstance(obs, PrivilegedObservation):
            raise TypeError(
                "DeepAgentV2.act_v2 received a PrivilegedObservation. "
                "Production act() must accept public data only; privileged "
                "observations are training-only and may never reach a "
                "deployment agent."
            )
        if not isinstance(obs, ObservationV2):
            raise TypeError(
                f"DeepAgentV2.act_v2 expects an ObservationV2, got "
                f"{type(obs).__name__}"
            )

        # Bug #3 / blocker: schema-hash binding, with recompute.
        # ``ObservationV2.__post_init__`` already binds ``feature_schema_hash``
        # to ``schema.stable_hash()`` at construction, but a value forged after
        # construction (e.g. via ``object.__setattr__`` or pickle) would bypass
        # it. Recompute from the attached schema here so the defense holds
        # without trusting the carried string. Two distinct failures:
        #   (a) obs carries a FALSE hash (hash != schema.stable_hash()) — the
        #       container is lying about which schema encoded it;
        #   (b) the schema the observation was actually encoded under differs
        #       from the model's — same shapes are NOT enough.
        actual_schema_hash = obs.schema.stable_hash()
        if obs.feature_schema_hash != actual_schema_hash:
            raise ValueError(
                f"DeepAgentV2 received an ObservationV2 carrying a false "
                f"schema hash: feature_schema_hash="
                f"{obs.feature_schema_hash!r} but schema.stable_hash()="
                f"{actual_schema_hash!r}. Refusing to forward."
            )
        if actual_schema_hash != self._feature_schema_hash:
            raise ValueError(
                f"DeepAgentV2 schema-hash mismatch: observation was encoded "
                f"under schema {actual_schema_hash!r}, model expects "
                f"{self._feature_schema_hash!r}. The observation was encoded "
                f"under a different feature schema than the model was trained "
                f"against. Refusing to forward."
            )

        # Blocker #3: per-observation ruleset identity check. The observation's
        # ruleset identity (carried in obs.public) must match the agent's
        # (and therefore the model's checkpoint) identity. The schema-hash
        # check above does NOT catch this: the ruleset family/bid/multiplier
        # are observation data values that do not necessarily change the schema
        # layout, so a standard-policy model could otherwise run under a legacy
        # observation context. Compare the full triple (id + version + hash).
        obs_ruleset_identity = (
            obs.public.ruleset_id,
            obs.public.ruleset_version,
            obs.public.ruleset_hash,
        )
        if obs_ruleset_identity != self._ruleset_identity:
            raise ValueError(
                f"DeepAgentV2 ruleset identity mismatch: observation carries "
                f"(id={obs.public.ruleset_id!r}, version={obs.public.ruleset_version!r}, "
                f"hash={obs.public.ruleset_hash!r}), but the agent/model "
                f"expects {self._ruleset_identity}. A model trained under one "
                f"ruleset must not consume an observation encoded under another."
            )

        # Source the return action from ``obs.actions.legal_actions`` — the
        # SAME container whose order matches ``obs.actions.features`` rows. The
        # ``public.legal_actions`` list is verified to agree in order by
        # ``ObservationV2.__post_init__``, but reading from the actions block
        # guarantees the returned action is the one the model just scored even
        # if a post-construction forge bypassed that check.
        legal_actions = obs.actions.legal_actions
        # Bug #6: zero legal actions is a caller error. The model cannot select
        # from an empty action set; returning an empty tuple would let a silent
        # bug propagate downstream. Fail loudly.
        if len(legal_actions) == 0:
            raise ValueError(
                "DeepAgentV2.act_v2 received an observation with zero legal "
                "actions. A decision with no legal actions is undefined."
            )
        # Single legal action: short-circuit without inference (matches the
        # legacy DeepAgent behaviour and avoids a degenerate forward).
        if len(legal_actions) == 1:
            return legal_actions[0]

        return self._select_from_observation(obs)

    # --- Legacy-compatible entry point (for evaluation.simulation) ---------- #
    def act(self, infoset):
        """Select an action from a legacy ``GameEnv`` infoset.

        Builds a public :class:`ObservationV2` via ``get_obs_v2`` (which never
        reads the true hidden hands) and delegates to :meth:`act_v2`. This lets
        ``DeepAgentV2`` drop into the existing evaluation harness without
        changes.

        The selected action is mapped back onto the infoset's own canonical
        action object (a list) so downstream code that compares by identity /
        type against ``infoset.legal_actions`` keeps working — the legacy
        ``DeepAgent.act`` returns the infoset's list, and this path matches.
        """
        if len(infoset.legal_actions) == 1:
            return infoset.legal_actions[0]
        from douzero.observation.encode_v2 import get_obs_v2

        obs = get_obs_v2(infoset, ruleset=self.ruleset)
        chosen = self.act_v2(obs)
        # The V2 obs stores actions as sorted tuples (``actions.legal_actions``)
        # and the infoset stores them as lists whose internal order may differ.
        # Map the chosen sorted tuple back onto the infoset's canonical object
        # by comparing as sorted tuples, so the returned action is the same
        # type/identity the caller passed in (the legacy ``DeepAgent.act``
        # returns the infoset's list, and this path matches).
        chosen_sorted = tuple(sorted(chosen))
        for la in infoset.legal_actions:
            if tuple(sorted(la)) == chosen_sorted:
                return la
        # Defensive: should be unreachable because get_obs_v2 preserves the
        # legal-action set. Fail loudly rather than returning a foreign object.
        raise RuntimeError(
            f"DeepAgentV2.act selected an action {chosen!r} that is not in "
            f"infoset.legal_actions. This indicates an observation/infoset "
            f"legal-action mismatch."
        )

    # --- Internal selection ------------------------------------------------- #
    def _select_from_observation(self, obs):
        from douzero.models_v2.batch import observation_to_model_inputs

        bundle = observation_to_model_inputs(obs)
        if torch.cuda.is_available():
            bundle.to("cuda")
        with torch.inference_mode():
            out = self.model(
                bundle.state_card_vectors,
                bundle.state_context_flat,
                bundle.context_card_vectors,
                bundle.context_flat,
                bundle.history_tokens,
                bundle.history_key_padding_mask,
                bundle.action_features,
                bundle.action_mask,
                bundle.acting_role,
            )
        # P06 r1: route through the unified decision policy using the FULL
        # DecisionConfig (carrying abs_tol / rel_tol / risk_penalty), not a
        # freshly-constructed one with default tolerances. The agent's
        # decision_config was built once at construction and carries the
        # caller's tolerance / risk-penalty choices into deployment.
        from douzero.training.decision_policy import select_action

        idx = select_action(out, self.decision_config)
        legal_actions = obs.actions.legal_actions
        # The action_mask may include padding rows beyond the real actions;
        # the decision policy respects the mask, so the index is a
        # real-action index as long as the observation's action block was
        # not padded. observation_to_model_inputs does not pad, so idx < len.
        if idx >= len(legal_actions):
            # Defensive: should never happen because we do not pad here.
            raise RuntimeError(
                f"selected action index {idx} >= len(legal_actions) "
                f"{len(legal_actions)}; the observation action block was "
                f"unexpectedly padded."
            )
        return legal_actions[idx]


def load_v2_model(model_path, schema, ruleset, config=None):
    """Load a :class:`~douzero.models_v2.model.ModelV2` from a V2 sidecar.

    The sidecar MUST be a manifest-bearing V2 bundle (written by
    :func:`douzero.checkpoint.save_v2_position_weights`). The manifest's
    model_version, schema hash, model-config hash, ruleset identity, and
    checkpoint_kind are validated against RUNTIME expectations, and the
    state_dict is loaded with ``strict=True``.

    Blocker #3 fix: the verified ruleset identity is ATTACHED to the returned
    model (``model.expected_ruleset_identity``) so a downstream
    :class:`DeepAgentV2` can enforce that the agent's RuleSet matches the
    checkpoint's, and that every observation's ruleset identity matches too.

    A bare state_dict sidecar, a legacy/factorized ``.ckpt``, a same-shape
    different-schema sidecar, a same-shape different-config sidecar, or a
    wrong-ruleset sidecar is rejected with a precise error.

    Parameters
    ----------
    model_path:
        Path to a V2 sidecar ``.ckpt`` (manifest-bearing).
    schema:
        The :class:`~douzero.observation.schema.FeatureSchemaManifest` the
        runtime expects. The sidecar's schema hash must equal
        ``schema.stable_hash()``.
    ruleset:
        The :class:`~douzero.env.rules.RuleSet` the runtime expects. The full
        identity (id + version + hash) is validated, supporting custom rule
        families and rejecting an unknown id.
    config:
        Optional :class:`~douzero.models_v2.config.ModelV2Config`. Defaults to
        ``ModelV2Config()``; must match the config the weights were saved under
        (the model-config hash is validated).

    Returns
    -------
    ModelV2
        The loaded model in eval mode, with ``expected_ruleset_identity`` and
        ``expected_model_config_hash`` attached as attributes for DeepAgentV2.
    """
    from douzero.models_v2.config import ModelV2Config
    from douzero.models_v2.model import ModelV2
    from douzero.checkpoint import load_v2_position_weights

    cfg = config or ModelV2Config()
    model = ModelV2(schema, cfg)
    expected_schema_hash = schema.stable_hash()
    expected_cfg_hash = cfg.stable_hash()

    # Load + validate the manifest-bearing sidecar. weights_only=True is the
    # default inside load_v2_position_weights. This rejects a bare state_dict,
    # a legacy .ckpt, a wrong-schema sidecar, a wrong-config sidecar, and a
    # wrong-ruleset sidecar (including a custom rule family with a mismatched
    # hash).
    pretrained, manifest = load_v2_position_weights(
        model_path,
        expected_schema_hash=expected_schema_hash,
        expected_model_config_hash=expected_cfg_hash,
        expected_ruleset=ruleset,
        # P06 r6: pass the runtime config so P05-format checkpoints can be
        # migrated via the v1 hash + raw-transform check.
        runtime_model_config=cfg,
    )

    # STRICT load (strict=True is the default for load_state_dict). A key/shape
    # mismatch means the config the caller passed does not match the config the
    # weights were saved under (e.g. a different hidden_size).
    model.load_state_dict(pretrained, strict=True)
    model.eval()

    # Blocker #3: attach the VERIFIED ruleset identity to the model so a
    # downstream DeepAgentV2 can enforce agent-vs-model and obs-vs-model
    # ruleset consistency without re-deriving it. The identity triple is the
    # full (id, version, hash) from the manifest, which was just validated
    # against the caller's RuleSet.
    model.expected_ruleset_identity = (
        manifest.ruleset_id,
        manifest.ruleset_version,
        manifest.ruleset_hash,
    )
    model.expected_model_config_hash = expected_cfg_hash
    return model
