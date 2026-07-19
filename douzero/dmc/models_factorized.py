"""Factorized legacy forward: state/history encoded once per decision (P04).

Background
----------
The legacy models (:class:`~douzero.dmc.models.LandlordLstmModel` /
:class:`~douzero.dmc.models.FarmerLstmModel`) score ``N`` legal actions by
*tiling* the shared state ``x_no_action`` and the shared history ``z`` across
all ``N`` rows, then running the LSTM once per row:

::

    z_batch : (N, 5, 162)   # N identical copies of the same history (5, 162)
    x_batch : (N, D_state + 54)   # D_state columns identical; last 54 per-action
    lstm_out, _ = self.lstm(z_batch)         # N identical (N, 5, 128)
    lstm_out = lstm_out[:, -1, :]            # (N, 128), identical across rows
    h = cat([lstm_out, x_batch], dim=-1)     # (N, 128 + D_state + 54)
    values = dense1..6(h)                    # (N, 1)

Because every row of ``z_batch`` is the SAME history and every row's state
block is the SAME ``x_no_action``, the LSTM output is identical across rows
(the LSTM has no per-row state in eval mode, and the legacy models use no
dropout/BatchNorm). Only the trailing 54-dim ``my_action`` block varies.

Factorized forward
------------------
This module provides drop-in models that compute the same result while running
the shared history LSTM and the shared state projection exactly ONCE per
decision:

::

    z_single : (1, 5, 162)              # the shared history, encoded once
    x_state  : (1, D_state)             # the shared state, encoded once
    x_action : (N, 54)                  # per-action card vectors
    lstm_out, _ = self.lstm(z_single)   # (1, 128) — ONE LSTM call
    h_state = lstm_out[:, -1, :]        # (1, 128)
    h = cat([h_state.expand(N, 128),
             x_state.expand(N, D_state),
             x_action], dim=-1)         # (N, 128 + D_state + 54)
    values = dense1..6(h)               # (N, 1)

``expand`` creates a view (no copy), so the per-action rows share the same
memory for the state/history block. The MLP then maps each row to a value.

Checkpoint compatibility
------------------------
These models intentionally declare the SAME submodule names and shapes as the
legacy models (``lstm``, ``dense1`` ... ``dense6``) so that a legacy
per-position ``.ckpt`` (a bare ``state_dict``) loads with NO conversion via the
existing :func:`~douzero.checkpoint.compat.load_legacy_position_ckpt` path.
``state_dict()`` keys and shapes are byte-for-byte identical to the legacy
models; only :meth:`forward` differs.

Scope
-----
Deployment only (``DeepAgent`` with ``backend="legacy_factorized"``). Training
(actor/learner) integration arrives in P05/P06 alongside Model V2; the
training gate in :func:`~douzero.dmc.dmc.train` continues to reject
non-legacy training.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

# Legacy state widths, imported from the single source of truth (P03 adapter).
# These are NOT magic numbers: they are derived from the schema constants in
# douzero/observation/legacy_adapter.py and match douzero/env/env.py exactly.
from douzero.observation.legacy_adapter import (
    FARMER_X_NO_ACTION_WIDTH,
    LANDLORD_X_NO_ACTION_WIDTH,
)

# Per-action card-vector width (the trailing my_action block in x_batch).
_ACTION_WIDTH: int = 54

# LSTM input/hidden sizes, matching the legacy models exactly.
_LSTM_INPUT_SIZE: int = 162
_LSTM_HIDDEN_SIZE: int = 128
# History is the last 15 moves reshaped to (5, 162) = 5 timesteps x 3 moves x 54.
_LSTM_SEQ_LEN: int = 5

# MLP widths, matching the legacy models exactly.
_MLP_WIDTH: int = 512


class _LegacyFactorizedBase(nn.Module):
    """Shared factorized forward for a single legacy role model.

    Subclasses set ``self._state_width`` (319 for landlord, 430 for farmers)
    and construct ``dense1`` with the matching input width (state + action +
    lstm_hidden). All other layers are identical to the legacy models.

    The factorized :meth:`forward` accepts the LEGACY-batched tensors
    ``(z_batch, x_batch)`` — the same tensors ``DeepAgent`` already builds via
    :func:`~douzero.env.env.get_obs` — so no observation-side change is needed.
    Internally it slices the shared history/state once and broadcasts across the
    per-action rows, yielding numerically identical values to the legacy
    per-row forward.
    """

    # Set by subclasses.
    _state_width: int

    def _validate_legacy_batch(self, z_batch, x_batch):
        """Validate the legacy-batched tensors before slicing.

        Because :meth:`forward` consumes only row 0 of ``z_batch``/``x_state``
        (the legacy encoder guarantees all rows are identical), a caller that
        passes rows that are NOT identical would get a silently wrong result
        (the per-action variation in the state block would be dropped). This
        guards against that: shape errors raise immediately, and in debug mode
        (or when ``DUZERO_FACTORIZED_STRICT`` is set) the shared-row invariant
        is verified too.

        See issue: a drop-in model must not silently accept non-shared input.
        """
        if z_batch.ndim != 3:
            raise ValueError(
                f"factorized forward expects z_batch.ndim==3 (N, 5, 162), "
                f"got {z_batch.ndim} (shape {tuple(z_batch.shape)})."
            )
        if x_batch.ndim != 2:
            raise ValueError(
                f"factorized forward expects x_batch.ndim==2 "
                f"(N, state_width+54), got {x_batch.ndim} "
                f"(shape {tuple(x_batch.shape)})."
            )
        n_z = z_batch.shape[0]
        n_x = x_batch.shape[0]
        if n_z != n_x:
            raise ValueError(
                f"factorized forward expects z_batch and x_batch to share the "
                f"row count (N); got z_batch N={n_z}, x_batch N={n_x}."
            )
        if n_z < 1:
            raise ValueError(
                "factorized forward expects N>=1 legal actions; got N=0."
            )
        if z_batch.shape[1:] != (_LSTM_SEQ_LEN, _LSTM_INPUT_SIZE):
            raise ValueError(
                f"factorized forward expects z_batch.shape[1:]=="
                f"{(_LSTM_SEQ_LEN, _LSTM_INPUT_SIZE)}, got "
                f"{tuple(z_batch.shape[1:])}."
            )
        expected_x_width = self._state_width + _ACTION_WIDTH
        if x_batch.shape[1] != expected_x_width:
            raise ValueError(
                f"factorized forward expects x_batch.shape[1]=="
                f"{expected_x_width} (state_width {self._state_width} + "
                f"action {_ACTION_WIDTH}), got {x_batch.shape[1]}."
            )
        # Shared-row invariant: all rows of z_batch must be identical, and all
        # rows of the state block of x_batch must be identical. This is cheap
        # to check and catches the silent-failure case. Opt out via the env
        # var only for hot paths that have already validated upstream.
        import os
        if os.environ.get("DUZERO_FACTORIZED_STRICT", "") == "0":
            return
        z_single = z_batch[:1]
        if not torch.equal(z_batch, z_single.expand_as(z_batch)):
            raise ValueError(
                "factorized forward received a z_batch whose rows are NOT "
                "identical. The factorized path encodes the shared history "
                "once (row 0); non-identical rows would be silently dropped. "
                "Pass a tiled batch from the legacy encoder, or use "
                "forward_factorized() with a true singleton."
            )
        x_state = x_batch[:, : self._state_width]
        x_state_single = x_state[:1]
        if not torch.equal(x_state, x_state_single.expand_as(x_state)):
            raise ValueError(
                "factorized forward received an x_batch whose state block "
                "rows are NOT identical. The factorized path encodes the "
                "shared state once (row 0); per-row state variation would be "
                "silently dropped. Pass a tiled batch from the legacy encoder, "
                "or use forward_factorized() with a true singleton."
            )

    def _validate_factorized_inputs(self, z_single, x_state_single, x_action):
        """Validate the split (singleton) inputs to forward_factorized."""
        if z_single.ndim != 3 or z_single.shape[0] != 1:
            raise ValueError(
                f"forward_factorized expects z_single of shape (1, 5, 162), "
                f"got shape {tuple(z_single.shape)}."
            )
        if z_single.shape[1:] != (5, _LSTM_INPUT_SIZE):
            raise ValueError(
                f"forward_factorized expects z_single.shape[1:]==(5, 162), "
                f"got {tuple(z_single.shape[1:])}."
            )
        if x_state_single.ndim != 2 or x_state_single.shape[0] != 1:
            raise ValueError(
                f"forward_factorized expects x_state_single of shape "
                f"(1, {self._state_width}), got shape "
                f"{tuple(x_state_single.shape)}."
            )
        if x_state_single.shape[1] != self._state_width:
            raise ValueError(
                f"forward_factorized expects x_state_single.shape[1]=="
                f"{self._state_width}, got {x_state_single.shape[1]}."
            )
        if x_action.ndim != 2:
            raise ValueError(
                f"forward_factorized expects x_action.ndim==2 (N, 54), got "
                f"{x_action.ndim} (shape {tuple(x_action.shape)})."
            )
        if x_action.shape[0] < 1:
            raise ValueError(
                "forward_factorized expects N>=1 legal actions; got N=0."
            )
        if x_action.shape[1] != _ACTION_WIDTH:
            raise ValueError(
                f"forward_factorized expects x_action.shape[1]==54, got "
                f"{x_action.shape[1]}."
            )

    def forward(self, z_batch, x_batch, return_value=False, flags=None):
        """Factorized forward over legacy-batched tensors.

        Args:
            z_batch: ``(N, 5, 162)`` legacy history batch. All rows MUST be
                identical (the legacy encoder guarantees this); only row 0 is
                consumed by the LSTM. Non-identical rows raise ValueError.
            x_batch: ``(N, state_width + 54)`` legacy state+action batch. The
                first ``state_width`` columns MUST be identical across rows;
                only the last 54 are per-action card vectors. Shape and
                shared-row invariants are validated before slicing.
            return_value: when True, return ``{'values': (N, 1)}``. When False,
                return ``{'action': index}`` via argmax (or epsilon-greedy),
                matching the legacy contract exactly.
            flags: optional flags with ``exp_epsilon`` for exploration.

        Returns:
            dict with ``'values'`` (when ``return_value``) or ``'action'``.
        """
        self._validate_legacy_batch(z_batch, x_batch)
        n = x_batch.shape[0]

        # --- Shared history: ONE LSTM call over the single shared history. ---
        # z_batch rows are identical by construction; row 0 suffices. Using
        # z_batch[:1] keeps the (1, 5, 162) shape the LSTM expects.
        z_single = z_batch[:1]
        lstm_out, _ = self.lstm(z_single)          # (1, 5, 128)
        h_lstm = lstm_out[:, -1, :]                  # (1, 128)

        # --- Shared state: slice once, broadcast (view, no copy). ---
        x_state_single = x_batch[:1, : self._state_width]  # (1, state_width)

        # --- Per-action card vectors. ---
        x_action = x_batch[:, self._state_width :]    # (N, 54)

        # Broadcast the shared blocks to N rows (expand -> view, no copy) and
        # concatenate with the per-action block. The concatenation is the first
        # point at which per-row data physically diverges.
        h = torch.cat([
            h_lstm.expand(n, _LSTM_HIDDEN_SIZE),
            x_state_single.expand(n, self._state_width),
            x_action,
        ], dim=-1)                                     # (N, 128 + state + 54)

        h = self.dense1(h)
        h = torch.relu(h)
        h = self.dense2(h)
        h = torch.relu(h)
        h = self.dense3(h)
        h = torch.relu(h)
        h = self.dense4(h)
        h = torch.relu(h)
        h = self.dense5(h)
        h = torch.relu(h)
        values = self.dense6(h)                        # (N, 1)

        if return_value:
            return dict(values=values)
        if flags is not None and flags.exp_epsilon > 0 and np.random.rand() < flags.exp_epsilon:
            action = torch.randint(values.shape[0], (1,))[0]
        else:
            action = torch.argmax(values, dim=0)[0]
        return dict(action=action)

    def forward_factorized(self, z_single, x_state_single, x_action,
                           return_value=False, flags=None):
        """Pure factorized forward taking already-split inputs.

        This is the canonical factorized interface: the shared history and
        shared state are passed as singletons ``(1, ...)`` and only the
        per-action card vectors carry the ``N`` dimension. It exists so callers
        that already hold split tensors (e.g. from
        :func:`~douzero.env.env.get_obs_factorized`) can skip the legacy-batched
        representation entirely — no tiling, no tiled tensor allocation, no
        tiled CPU->GPU transfer.

        Args:
            z_single: ``(1, 5, 162)`` shared history.
            x_state_single: ``(1, state_width)`` shared state.
            x_action: ``(N, 54)`` per-action card vectors.
            return_value, flags: as in :meth:`forward`.
        """
        self._validate_factorized_inputs(z_single, x_state_single, x_action)
        n = x_action.shape[0]
        lstm_out, _ = self.lstm(z_single)             # (1, 5, 128)
        h_lstm = lstm_out[:, -1, :]                     # (1, 128)
        h = torch.cat([
            h_lstm.expand(n, _LSTM_HIDDEN_SIZE),
            x_state_single.expand(n, self._state_width),
            x_action,
        ], dim=-1)
        h = self.dense1(h)
        h = torch.relu(h)
        h = self.dense2(h)
        h = torch.relu(h)
        h = self.dense3(h)
        h = torch.relu(h)
        h = self.dense4(h)
        h = torch.relu(h)
        h = self.dense5(h)
        h = torch.relu(h)
        values = self.dense6(h)
        if return_value:
            return dict(values=values)
        if flags is not None and flags.exp_epsilon > 0 and np.random.rand() < flags.exp_epsilon:
            action = torch.randint(values.shape[0], (1,))[0]
        else:
            action = torch.argmax(values, dim=0)[0]
        return dict(action=action)

    def select_actions_packed(self, z_batch, x_state_batch, x_action,
                              action_counts):
        """Score a microbatch of decisions packed by role and return argmaxes."""
        if z_batch.ndim != 3 or x_state_batch.ndim != 2 or x_action.ndim != 2:
            raise ValueError("invalid packed factorized input ranks")
        if len(action_counts) != z_batch.shape[0] or z_batch.shape[0] < 1:
            raise ValueError("action_counts must match the decision batch")
        if any(count < 1 for count in action_counts):
            raise ValueError("every packed decision requires a legal action")
        if sum(action_counts) != x_action.shape[0]:
            raise ValueError("packed action rows do not match action_counts")
        lstm_out, _ = self.lstm(z_batch)
        h_lstm = lstm_out[:, -1, :]
        repeats = torch.as_tensor(
            action_counts, dtype=torch.long, device=z_batch.device
        )
        h = torch.cat([
            torch.repeat_interleave(h_lstm, repeats, dim=0),
            torch.repeat_interleave(x_state_batch, repeats, dim=0),
            x_action,
        ], dim=-1)
        h = torch.relu(self.dense1(h))
        h = torch.relu(self.dense2(h))
        h = torch.relu(self.dense3(h))
        h = torch.relu(self.dense4(h))
        h = torch.relu(self.dense5(h))
        values = self.dense6(h).squeeze(-1)
        return torch.stack([
            chunk.argmax() for chunk in values.split(list(action_counts))
        ])


class LegacyFactorizedLandlordModel(_LegacyFactorizedBase):
    """Factorized landlord model — loads legacy landlord checkpoints unchanged.

    ``dense1`` input width = ``373 + 128 = 501`` (landlord x_batch 373 + lstm
    hidden 128), identical to :class:`~douzero.dmc.models.LandlordLstmModel`.
    """

    def __init__(self):
        super().__init__()
        self._state_width = LANDLORD_X_NO_ACTION_WIDTH  # 319
        self.lstm = nn.LSTM(_LSTM_INPUT_SIZE, _LSTM_HIDDEN_SIZE, batch_first=True)
        self.dense1 = nn.Linear(LANDLORD_X_NO_ACTION_WIDTH + _ACTION_WIDTH + _LSTM_HIDDEN_SIZE, _MLP_WIDTH)
        self.dense2 = nn.Linear(_MLP_WIDTH, _MLP_WIDTH)
        self.dense3 = nn.Linear(_MLP_WIDTH, _MLP_WIDTH)
        self.dense4 = nn.Linear(_MLP_WIDTH, _MLP_WIDTH)
        self.dense5 = nn.Linear(_MLP_WIDTH, _MLP_WIDTH)
        self.dense6 = nn.Linear(_MLP_WIDTH, 1)


class LegacyFactorizedFarmerModel(_LegacyFactorizedBase):
    """Factorized farmer model — loads legacy farmer checkpoints unchanged.

    ``dense1`` input width = ``484 + 128 = 612`` (farmer x_batch 484 + lstm
    hidden 128), identical to :class:`~douzero.dmc.models.FarmerLstmModel`.
    Used for both ``landlord_up`` and ``landlord_down``.
    """

    def __init__(self):
        super().__init__()
        self._state_width = FARMER_X_NO_ACTION_WIDTH  # 430
        self.lstm = nn.LSTM(_LSTM_INPUT_SIZE, _LSTM_HIDDEN_SIZE, batch_first=True)
        self.dense1 = nn.Linear(FARMER_X_NO_ACTION_WIDTH + _ACTION_WIDTH + _LSTM_HIDDEN_SIZE, _MLP_WIDTH)
        self.dense2 = nn.Linear(_MLP_WIDTH, _MLP_WIDTH)
        self.dense3 = nn.Linear(_MLP_WIDTH, _MLP_WIDTH)
        self.dense4 = nn.Linear(_MLP_WIDTH, _MLP_WIDTH)
        self.dense5 = nn.Linear(_MLP_WIDTH, _MLP_WIDTH)
        self.dense6 = nn.Linear(_MLP_WIDTH, 1)


# Evaluation/dispatch dict, parallel to douzero.dmc.models.model_dict.
# A legacy per-position .ckpt loads into these models with NO key conversion.
factorized_model_dict = {
    'landlord': LegacyFactorizedLandlordModel,
    'landlord_up': LegacyFactorizedFarmerModel,
    'landlord_down': LegacyFactorizedFarmerModel,
}


def split_legacy_batch(position, z_batch, x_batch):
    """Split legacy-batched tensors into the factorized (z_single, x_state, x_action).

    Convenience used by ``DeepAgent`` and parity tests. The widths are derived
    from the role (landlord 319, farmers 430), never hard-coded at call sites.

    Args:
        position: one of ``'landlord'`` / ``'landlord_up'`` / ``'landlord_down'``.
        z_batch: ``(N, 5, 162)`` legacy history batch (rows identical).
        x_batch: ``(N, state_width + 54)`` legacy state+action batch.

    Returns:
        ``(z_single, x_state_single, x_action)`` where ``z_single`` is
        ``(1, 5, 162)``, ``x_state_single`` is ``(1, state_width)`` and
        ``x_action`` is ``(N, 54)``.
    """
    if position == 'landlord':
        state_width = LANDLORD_X_NO_ACTION_WIDTH
    elif position in ('landlord_up', 'landlord_down'):
        state_width = FARMER_X_NO_ACTION_WIDTH
    else:
        raise ValueError(
            f"Unknown position {position!r}; expected 'landlord', "
            f"'landlord_up', or 'landlord_down'."
        )
    z_single = z_batch[:1]
    x_state_single = x_batch[:1, :state_width]
    x_action = x_batch[:, state_width:]
    return z_single, x_state_single, x_action


class LegacyFactorizedModel:
    """Three-role wrapper parallel to :class:`~douzero.dmc.models.Model`.

    Provided for symmetry with the legacy training wrapper. It accepts the
    legacy-batched ``(z, x)`` tensors and dispatches to the factorized role
    model, which internally encodes the shared history/state once.

    NOT wired into training in P04 (the actor/learner loop stays on the legacy
    ``Model``); training integration arrives in P05/P06.
    """

    def __init__(self, device=0):
        self.models = {}
        if not device == "cpu":
            device = 'cuda:' + str(device)
        self.models['landlord'] = LegacyFactorizedLandlordModel().to(torch.device(device))
        self.models['landlord_up'] = LegacyFactorizedFarmerModel().to(torch.device(device))
        self.models['landlord_down'] = LegacyFactorizedFarmerModel().to(torch.device(device))

    def forward(self, position, z, x, training=False, flags=None):
        model = self.models[position]
        return model.forward(z, x, return_value=not training, flags=flags)

    def share_memory(self):
        self.models['landlord'].share_memory()
        self.models['landlord_up'].share_memory()
        self.models['landlord_down'].share_memory()

    def eval(self):
        self.models['landlord'].eval()
        self.models['landlord_up'].eval()
        self.models['landlord_down'].eval()

    def parameters(self, position):
        return self.models[position].parameters()

    def get_model(self, position):
        return self.models[position]

    def get_models(self):
        return self.models
