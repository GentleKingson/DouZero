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
        A constructed :class:`~douzero.models_v2.model.ModelV2`. The caller is
        responsible for loading weights (see :func:`load_v2_model`).
    ruleset:
        Optional :class:`~douzero.env.rules.RuleSet` used when building the V2
        observation from an infoset. Defaults to the legacy ruleset.
    decision_mode:
        How to convert the multi-head output to a single action. ``"win"``
        (default) picks the highest-``p_win`` valid action. ``"score"`` picks
        the highest expected score. P06 adds lexicographic modes; P05 keeps
        these two simple, fully-tested modes.
    """

    def __init__(
        self,
        position,
        model,
        ruleset=None,
        decision_mode="win",
    ):
        from douzero.models_v2.model import ModelV2  # local import: keep the
        # production import graph (evaluation.simulation) free of a hard torch
        # model dependency at module load, mirroring the lazy imports above.
        if not isinstance(model, ModelV2):
            raise TypeError(
                f"DeepAgentV2 requires a ModelV2 instance, got {type(model).__name__}"
            )
        if decision_mode not in ("win", "score"):
            raise ValueError(
                f"decision_mode must be 'win' or 'score', got {decision_mode!r}"
            )
        self.position = position
        self.model = model
        self.ruleset = ruleset
        self.decision_mode = decision_mode
        self.backend = "v2"
        # Bug #3: bind the agent to the model's feature schema hash. Every
        # observation forwarded through act_v2 must carry the SAME schema hash,
        # so a model trained under schema A cannot silently consume an
        # observation encoded under schema B (even if the shapes match).
        self._feature_schema_hash = model.schema.stable_hash()
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

        # Bug #3: schema-hash binding. The observation's feature_schema_hash
        # must equal the model's. A mismatch means the observation was encoded
        # under a different schema (e.g. a field reordered/resized) than the
        # model was trained against — same shapes are NOT enough. Reject before
        # forwarding so a mis-encoded observation cannot produce a silent
        # garbage action.
        if obs.feature_schema_hash != self._feature_schema_hash:
            raise ValueError(
                f"DeepAgentV2 schema-hash mismatch: observation has "
                f"{obs.feature_schema_hash!r}, model expects "
                f"{self._feature_schema_hash!r}. The observation was encoded "
                f"under a different feature schema than the model was trained "
                f"against. Refusing to forward."
            )

        legal_actions = obs.public.legal_actions
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
        # The V2 obs stores actions as tuples; the infoset stores them as lists.
        # Map the chosen tuple back onto the infoset's canonical object so the
        # returned action is the same type/identity the caller passed in.
        chosen_key = tuple(chosen) if not isinstance(chosen, tuple) else chosen
        for la in infoset.legal_actions:
            if tuple(la) == chosen_key:
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
        if self.decision_mode == "score":
            # Highest expected score among valid actions.
            scores = out.score_mean.squeeze(-1).clone()
            scores[~out.action_mask] = float("-inf")
            idx = int(torch.argmax(scores).item())
        else:
            idx = out.argmax_win()
        legal_actions = obs.public.legal_actions
        # The action_mask may include padding rows beyond the real actions;
        # argmax over the win logit already respects the mask, so the index is
        # a real-action index as long as the observation's action block was not
        # padded. observation_to_model_inputs does not pad, so idx < len.
        if idx >= len(legal_actions):
            # Defensive: should never happen because we do not pad here.
            raise RuntimeError(
                f"selected action index {idx} >= len(legal_actions) "
                f"{len(legal_actions)}; the observation action block was "
                f"unexpectedly padded."
            )
        return legal_actions[idx]


def load_v2_model(model_path, schema, config=None, *, ruleset_id="legacy"):
    """Load a :class:`~douzero.models_v2.model.ModelV2` from a V2 sidecar.

    Bug #3 fix: the sidecar MUST be a manifest-bearing V2 bundle (written by
    :func:`douzero.checkpoint.save_v2_position_weights`). The manifest's
    model_version, schema hash, ruleset identity, and checkpoint_kind are
    validated against RUNTIME expectations (the ``schema`` and ``ruleset_id``
    the caller passes), and the state_dict is loaded with ``strict=True``.

    A bare state_dict sidecar, a legacy/factorized ``.ckpt``, or a
    same-shape different-schema sidecar is rejected with a precise error.

    Parameters
    ----------
    model_path:
        Path to a V2 sidecar ``.ckpt`` (manifest-bearing).
    schema:
        The :class:`~douzero.observation.schema.FeatureSchemaManifest` the
        runtime expects. The sidecar's schema hash must equal
        ``schema.stable_hash()``.
    config:
        Optional :class:`~douzero.models_v2.config.ModelV2Config`. Defaults to
        ``ModelV2Config()``; must match the config the weights were saved under.
    ruleset_id:
        The rule-engine identity the runtime expects (``"legacy"`` or
        ``"standard"``). Defaults to ``"legacy"``.

    Returns
    -------
    ModelV2
        The loaded model in eval mode.
    """
    from douzero.models_v2.config import ModelV2Config
    from douzero.models_v2.model import ModelV2
    from douzero.checkpoint import (
        CheckpointCompatibilityError,
        load_v2_position_weights,
    )

    cfg = config or ModelV2Config()
    model = ModelV2(schema, cfg)
    expected_schema_hash = schema.stable_hash()

    # Load + validate the manifest-bearing sidecar. weights_only=True is the
    # default inside load_v2_position_weights. This rejects a bare state_dict,
    # a legacy .ckpt, a wrong-schema sidecar, and a wrong-ruleset sidecar.
    try:
        pretrained, manifest = load_v2_position_weights(
            model_path,
            expected_schema_hash=expected_schema_hash,
            expected_ruleset_id=ruleset_id,
        )
    except CheckpointCompatibilityError:
        # Re-raise as-is: the message is already precise and actionable.
        raise

    # STRICT load (strict=True is the default for load_state_dict). A key/shape
    # mismatch means the config the caller passed does not match the config the
    # weights were saved under (e.g. a different hidden_size).
    model.load_state_dict(pretrained, strict=True)
    model.eval()
    return model
