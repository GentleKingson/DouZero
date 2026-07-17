"""Model V2: shared state-action value model with multi-head outputs (P05).

This package implements the unified model selected by ``model_version=v2``. It
replaces the three role-specific legacy MLPs with one shared backbone plus a
learned role embedding, and exposes multi-head outputs (win probability +
conditional scores) for the P06 multi-objective training loop.

Public API
----------
- :class:`ModelV2` — the top-level model.
- :class:`ModelV2Config` — frozen architecture configuration.
- :class:`ModelOutput` — structured multi-head return value.
- :class:`ModelInputBundle` + :func:`observation_to_model_inputs` — bridge from
  an :class:`~douzero.observation.encode_v2.ObservationV2` to the model's tensor
  contract.

Sub-modules
-----------
- :mod:`card_encoder` — shared card-set projection.
- :mod:`history_encoder` — Transformer / LSTM history summarizer with padding mask.
- :mod:`action_encoder` — per-legal-action embedding.
- :mod:`state_encoder` — once-per-decision state trunk.
- :mod:`fusion` — state + history + action + role fusion (residual MLP).
- :mod:`heads` — win/score multi-head output.
- :mod:`batch` — observation -> tensor conversion.

Imperfect-information boundary
------------------------------
This package imports only from :mod:`douzero.observation` (public) and the
standard library / torch. It does NOT import
:mod:`douzero.observation.privileged` and does NOT accept a
:class:`PrivilegedObservation`. The :class:`~douzero.evaluation.deep_agent.DeepAgentV2`
guard enforces this at the deployment boundary.
"""

from __future__ import annotations

from douzero.models_v2.action_encoder import ActionEncoder
from douzero.models_v2.batch import ModelInputBundle, observation_to_model_inputs
from douzero.models_v2.card_encoder import CardSetEncoder, MultiCardSetEncoder
from douzero.models_v2.config import (
    HISTORY_ENCODER_LSTM,
    HISTORY_ENCODER_TRANSFORMER,
    ModelV2Config,
    SUPPORTED_ROLES,
)
from douzero.models_v2.fusion import StateActionFusion
from douzero.models_v2.heads import BiddingHeads, ValueHeads
from douzero.models_v2.history_encoder import (
    LSTMHistoryEncoder,
    TransformerHistoryEncoder,
    build_history_encoder,
)
from douzero.models_v2.model import ModelV2
from douzero.models_v2.numerical import NumericalError, assert_finite
from douzero.models_v2.output import BiddingModelOutput, ModelOutput
from douzero.models_v2.state_encoder import StateEncoder

__all__ = [
    # model + config + output
    "ModelV2",
    "ModelV2Config",
    "ModelOutput",
    "BiddingModelOutput",
    "SUPPORTED_ROLES",
    "HISTORY_ENCODER_TRANSFORMER",
    "HISTORY_ENCODER_LSTM",
    # batch bridge
    "ModelInputBundle",
    "observation_to_model_inputs",
    # sub-modules (for tests / introspection)
    "CardSetEncoder",
    "MultiCardSetEncoder",
    "ActionEncoder",
    "StateEncoder",
    "StateActionFusion",
    "ValueHeads",
    "BiddingHeads",
    "TransformerHistoryEncoder",
    "LSTMHistoryEncoder",
    "build_history_encoder",
    # numerical safety
    "NumericalError",
    "assert_finite",
]
