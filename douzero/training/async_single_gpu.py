"""Spawn-safe shared-memory protocol for V2 centralized GPU inference.

Only slot identifiers and compact integer metadata cross queues.  Observation
tensors live in a fixed CPU shared-memory slab owned by the main process.  The
state machine is deliberately independent of CUDA so timeout, crash and
quiescence behavior can be tested on CPU hosts.
"""

from __future__ import annotations

import multiprocessing as mp
import queue
import time
import ctypes
import math
from collections import defaultdict, deque
from dataclasses import dataclass
from enum import IntEnum

import torch
import numpy as np

from douzero.belief.constraints import NUM_BELIEF_RANKS
from douzero.belief.features import BELIEF_INPUT_DIM, BeliefInput
from douzero.models_v2.batch import (
    BatchedBiddingInput,
    BatchedModelInputBundle,
    ModelInputBundle,
)
from douzero.models_v2.config import SUPPORTED_ROLES
from douzero.observation.seats import ALL_ROLES
from douzero.observation.schema import (
    action_width,
    context_width,
    history_token_width,
    state_width,
)
from douzero.strategy.features import STRATEGY_FEATURE_WIDTH
from douzero.style.features import STYLE_FEATURE_WIDTH
from douzero.training.seed_stream import (
    FORMAL_SEED_DERIVATION_V1,
    TOPOLOGY_LOCAL_SEED_DERIVATION_V1,
    derive_formal_stream_seed,
)
from douzero.training.v2_buffer import compact_model_input_shapes


# Inference has a different padding/launch tradeoff from learner replay.  The
# learner keeps its fine-grained action buckets, while centralized inference
# deliberately uses two broad buckets so a small actor pool does not fragment
# into singleton GPU launches.
INFERENCE_ACTION_BUCKET_LIMITS: tuple[int, ...] = (64, 512)


def _formal_action_seed(root_seed: int, actor_id: int, episode_id: int) -> int:
    return derive_formal_stream_seed(
        root_seed, "action", actor_id, episode_id
    )


def _async_redeal_seed(
    root_seed: int,
    environment_seed_derivation: str,
    actor_id: int,
    episode_id: int,
    redeal_count: int,
) -> int:
    """Derive a redeal seed that cannot depend on interleaved game ordering."""

    if environment_seed_derivation not in {
        TOPOLOGY_LOCAL_SEED_DERIVATION_V1,
        FORMAL_SEED_DERIVATION_V1,
    }:
        raise ValueError("unsupported async environment seed derivation")
    worker_id = (
        0
        if environment_seed_derivation == FORMAL_SEED_DERIVATION_V1
        else actor_id
    )
    return derive_formal_stream_seed(
        root_seed,
        f"environment-redeal-{redeal_count}",
        worker_id,
        episode_id,
    )


def _check_actor_step_limit(steps: int, max_steps: int) -> None:
    """Fail at the shared card-play/bidding actor step boundary."""

    for name, value in (("steps", steps), ("max_steps", max_steps)):
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"{name} must be an int")
    if steps < 0 or max_steps < 1:
        raise ValueError("actor step counters are out of range")
    if steps >= max_steps:
        raise RuntimeError(f"actor episode exceeded max_steps={max_steps}")


def inference_action_count_bucket(action_count: int) -> int | str:
    """Return the coarse centralized-inference padding bucket."""
    if isinstance(action_count, bool) or not isinstance(action_count, int):
        raise TypeError("action_count must be an int")
    if action_count <= 0:
        raise ValueError("action_count must be positive")
    for limit in INFERENCE_ACTION_BUCKET_LIMITS:
        if action_count <= limit:
            return limit
    return "overflow"


def _shared_tensor(shape, dtype=torch.float32):
    """Create a tensor view over stdlib shared memory (no torch_shm_manager)."""
    numel = math.prod(shape)
    sizes = {
        torch.float32: 4,
        torch.bool: 1,
        torch.int8: 1,
        torch.int32: 4,
        torch.int64: 8,
    }
    if dtype not in sizes:
        raise TypeError(f"unsupported shared tensor dtype {dtype}")
    owner = mp.get_context("spawn").RawArray(ctypes.c_ubyte, numel * sizes[dtype])
    tensor = torch.frombuffer(owner, dtype=dtype, count=numel).reshape(shape)
    tensor.zero_()
    return tensor, owner


def _restore_shared_tensor(owner, shape, dtype):
    return torch.frombuffer(
        owner, dtype=dtype, count=math.prod(shape)
    ).reshape(shape)


class SlotState(IntEnum):
    FREE = 0
    WRITING = 1
    READY = 2
    RUNNING = 3
    DONE = 4
    FAILED = 5
    SHUTDOWN = 6


class RequestKind(IntEnum):
    CARD_PLAY = 0
    BIDDING = 1


@dataclass(frozen=True)
class RequestMetadata:
    slot_id: int
    actor_id: int
    request_id: int
    policy_snapshot: int
    action_count: int
    acting_role: int
    submitted_ns: int
    request_kind: int = int(RequestKind.CARD_PLAY)

    @property
    def grouping_key(self) -> tuple[int, int | str]:
        if self.request_kind == int(RequestKind.BIDDING):
            return (self.policy_snapshot, "bidding")
        return (
            self.policy_snapshot,
            inference_action_count_bucket(self.action_count),
        )


@dataclass(frozen=True)
class AsyncReplayKey:
    """Stable actor-local transition identity shared by public and sidecar paths."""

    actor_id: int
    episode_id: int
    trace_index: int

    def __post_init__(self) -> None:
        for name in ("actor_id", "episode_id", "trace_index"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"async replay key {name} must be non-negative")


class SharedObservationSlots:
    """Preallocated CPU shared tensors for inference requests and responses."""

    def __init__(
        self,
        schema,
        num_slots: int,
        max_actions: int = 256,
        *,
        output_width: int = 5,
        strategy_features: bool = False,
        style_features: bool = False,
    ) -> None:
        if num_slots < 1 or max_actions < 1:
            raise ValueError("shared slot dimensions must be positive")
        if output_width not in {5, 6}:
            raise ValueError("shared output width must be 5 (V2) or 6 (V3)")
        card_dim = schema.card_vector_dim
        state_flat = state_width(schema) - 6 * card_dim
        context_flat = context_width(schema) - 2 * card_dim
        history_width = history_token_width(schema)
        action_feature_width = action_width(schema)
        self.num_slots = int(num_slots)
        self.max_actions = int(max_actions)
        self.output_width = int(output_width)
        for name, value in (
            ("strategy_features", strategy_features),
            ("style_features", style_features),
        ):
            if not isinstance(value, bool):
                raise TypeError(f"{name} must be bool")
        self.strategy_features_enabled = strategy_features
        self.style_features_enabled = style_features

        self._shared_owners = []
        self._shared_specs = []
        self._tensor_fields = list(self._TENSOR_FIELDS)

        def shared(shape, dtype=torch.float32):
            tensor, owner = _shared_tensor(shape, dtype)
            self._shared_owners.append(owner)
            self._shared_specs.append((tuple(shape), dtype))
            return tensor

        self.state_cards = shared((num_slots, 6, card_dim))
        self.state_flat = shared((num_slots, state_flat))
        self.context_cards = shared((num_slots, 2, card_dim))
        self.context_flat = shared((num_slots, context_flat))
        self.history = shared(
            (num_slots, schema.max_history_len, history_width)
        )
        self.history_padding = shared(
            (num_slots, schema.max_history_len), torch.bool
        )
        self.actions = shared(
            (num_slots, max_actions, action_feature_width)
        )
        self.action_mask = shared((num_slots, max_actions), torch.bool)
        # Keep all response heads adjacent per action.  The inference service
        # publishes one contiguous row copy; compatibility views retain the
        # named fields used by the actor and protocol tests.
        self.output_values = shared((num_slots, max_actions, output_width))
        self._bind_output_views()
        self.action_counts = shared((num_slots,), torch.int32)
        self.roles = shared((num_slots,), torch.int64)
        self.strategy_features = (
            shared((num_slots, max_actions, STRATEGY_FEATURE_WIDTH))
            if strategy_features
            else None
        )
        if strategy_features:
            self._tensor_fields.append("strategy_features")
        self.style_features = (
            shared((num_slots, STYLE_FEATURE_WIDTH))
            if style_features
            else None
        )
        if style_features:
            self._tensor_fields.append("style_features")

    _TENSOR_FIELDS = (
        "state_cards", "state_flat", "context_cards", "context_flat",
        "history", "history_padding", "actions", "action_mask",
        "output_values", "action_counts", "roles",
    )

    _OUTPUT_VIEW_FIELDS = (
        "output_win", "output_score_win", "output_score_loss",
        "output_p_win", "output_score", "output_dmc_q",
    )

    def _bind_output_views(self) -> None:
        self.output_win = self.output_values[..., 0]
        self.output_score_win = self.output_values[..., 1]
        self.output_score_loss = self.output_values[..., 2]
        self.output_p_win = self.output_values[..., 3]
        self.output_score = self.output_values[..., 4]
        self.output_dmc_q = (
            self.output_values[..., 5] if self.output_width == 6 else None
        )

    def __getstate__(self):
        state = self.__dict__.copy()
        for name in tuple(self._tensor_fields) + self._OUTPUT_VIEW_FIELDS:
            state.pop(name, None)
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        for name, owner, (shape, dtype) in zip(
            self._tensor_fields, self._shared_owners, self._shared_specs
        ):
            setattr(self, name, _restore_shared_tensor(owner, shape, dtype))
        if not self.strategy_features_enabled:
            self.strategy_features = None
        if not self.style_features_enabled:
            self.style_features = None
        self._bind_output_views()

    def write(self, slot_id: int, bundle: ModelInputBundle) -> None:
        count = int(bundle.action_features.shape[0])
        if count < 1:
            raise ValueError("inference request has zero legal actions")
        if count > self.max_actions:
            raise ValueError(
                f"inference request has {count} actions; shared max is {self.max_actions}"
            )
        self.state_cards[slot_id].copy_(torch.stack(bundle.state_card_vectors))
        self.state_flat[slot_id].copy_(bundle.state_context_flat)
        self.context_cards[slot_id].copy_(torch.stack(bundle.context_card_vectors))
        self.context_flat[slot_id].copy_(bundle.context_flat)
        self.history[slot_id].copy_(bundle.history_tokens)
        self.history_padding[slot_id].copy_(bundle.history_key_padding_mask)
        # Only mask state can make stale padded rows observable.  Clearing the
        # previous live range avoids touching the full 4096-row slot on every
        # decision; action rows in the new live range are overwritten below.
        previous_count = int(self.action_counts[slot_id])
        if previous_count:
            self.action_mask[slot_id, :previous_count].zero_()
        self.actions[slot_id, :count].copy_(bundle.action_features)
        self.action_mask[slot_id, :count].copy_(bundle.action_mask)
        if self.strategy_features_enabled != (bundle.strategy_features is not None):
            raise ValueError(
                "shared slot strategy feature contract does not match the request"
            )
        if self.strategy_features is not None:
            if bundle.strategy_features.shape != (
                count, STRATEGY_FEATURE_WIDTH
            ):
                raise ValueError("shared slot strategy feature layout mismatch")
            self.strategy_features[slot_id, :count].copy_(
                bundle.strategy_features
            )
        if self.style_features_enabled != (bundle.style_features is not None):
            raise ValueError(
                "shared slot style feature contract does not match the request"
            )
        if self.style_features is not None:
            if bundle.style_features.shape != (STYLE_FEATURE_WIDTH,):
                raise ValueError("shared slot style feature layout mismatch")
            self.style_features[slot_id].copy_(bundle.style_features)
        self.action_counts[slot_id] = count
        try:
            role = SUPPORTED_ROLES.index(bundle.acting_role)
        except ValueError as exc:
            raise ValueError("unsupported acting role") from exc
        self.roles[slot_id] = role

    def read_bundle(self, slot_id: int, feature_schema_hash: str) -> ModelInputBundle:
        count = int(self.action_counts[slot_id])
        role = SUPPORTED_ROLES[int(self.roles[slot_id])]
        return ModelInputBundle(
            state_card_vectors=tuple(self.state_cards[slot_id, i] for i in range(6)),
            state_context_flat=self.state_flat[slot_id],
            context_card_vectors=tuple(self.context_cards[slot_id, i] for i in range(2)),
            context_flat=self.context_flat[slot_id],
            history_tokens=self.history[slot_id],
            history_key_padding_mask=self.history_padding[slot_id],
            action_features=self.actions[slot_id, :count],
            action_mask=self.action_mask[slot_id, :count],
            acting_role=role,
            feature_schema_hash=feature_schema_hash,
            strategy_features=(
                None
                if self.strategy_features is None
                else self.strategy_features[slot_id, :count]
            ),
            style_features=(
                None
                if self.style_features is None
                else self.style_features[slot_id]
            ),
        )


class SharedBeliefInputSlots:
    """Public-only belief inputs transported beside existing inference slots."""

    _TENSOR_FIELDS = (
        "feature_vectors",
        "unseen_counts",
        "opponent_totals",
        "roles",
        "style_features",
        "valid",
    )

    def __init__(self, num_slots: int) -> None:
        if isinstance(num_slots, bool) or not isinstance(num_slots, int) or num_slots < 1:
            raise ValueError("belief input slots require a positive slot count")
        self.num_slots = num_slots
        self._shared_owners = []
        self._shared_specs = []

        def shared(shape, dtype):
            tensor, owner = _shared_tensor(shape, dtype)
            self._shared_owners.append(owner)
            self._shared_specs.append((tuple(shape), dtype))
            return tensor

        self.feature_vectors = shared(
            (num_slots, BELIEF_INPUT_DIM), torch.float32
        )
        self.unseen_counts = shared(
            (num_slots, NUM_BELIEF_RANKS), torch.int64
        )
        self.opponent_totals = shared((num_slots, 2), torch.int64)
        self.roles = shared((num_slots, 3), torch.int64)
        self.style_features = shared(
            (num_slots, STYLE_FEATURE_WIDTH), torch.float32
        )
        self.valid = shared((num_slots,), torch.bool)

    def __getstate__(self):
        state = self.__dict__.copy()
        for name in self._TENSOR_FIELDS:
            state.pop(name, None)
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        for name, owner, (shape, dtype) in zip(
            self._TENSOR_FIELDS, self._shared_owners, self._shared_specs
        ):
            setattr(self, name, _restore_shared_tensor(owner, shape, dtype))

    def clear(self, slot_id: int) -> None:
        self.valid[slot_id] = False

    def write(self, slot_id: int, belief_input: BeliefInput) -> None:
        if not isinstance(belief_input, BeliefInput):
            raise TypeError("async belief input slot requires BeliefInput")
        self.feature_vectors[slot_id].copy_(
            torch.from_numpy(belief_input.feature_vector.copy())
        )
        self.unseen_counts[slot_id].copy_(
            torch.from_numpy(belief_input.unseen_counts.copy())
        )
        self.opponent_totals[slot_id, 0] = belief_input.opponent_a_total
        self.opponent_totals[slot_id, 1] = belief_input.opponent_b_total
        for index, role in enumerate((
            belief_input.acting_role,
            belief_input.opponent_a_role,
            belief_input.opponent_b_role,
        )):
            try:
                self.roles[slot_id, index] = ALL_ROLES.index(role)
            except ValueError as exc:
                raise ValueError("async belief input contains an unknown role") from exc
        self.style_features[slot_id].copy_(
            torch.from_numpy(belief_input.style_features.copy())
        )
        self.valid[slot_id] = True

    def read(self, slot_id: int) -> BeliefInput:
        if not bool(self.valid[slot_id]):
            raise RuntimeError("async belief input slot was not written")
        return BeliefInput(
            feature_vector=self.feature_vectors[slot_id].numpy().copy(),
            unseen_counts=self.unseen_counts[slot_id].numpy().copy(),
            opponent_a_total=int(self.opponent_totals[slot_id, 0]),
            opponent_b_total=int(self.opponent_totals[slot_id, 1]),
            acting_role=ALL_ROLES[int(self.roles[slot_id, 0])],
            opponent_a_role=ALL_ROLES[int(self.roles[slot_id, 1])],
            opponent_b_role=ALL_ROLES[int(self.roles[slot_id, 2])],
            style_features=self.style_features[slot_id].numpy().copy(),
        )


class SharedBiddingSlots:
    """Public 0/1/2/3 bidding inputs and logits on the shared request slab."""

    _TENSOR_FIELDS = ("features", "legal_mask", "output_logits", "valid")

    def __init__(self, num_slots: int, feature_width: int) -> None:
        if (
            isinstance(num_slots, bool)
            or not isinstance(num_slots, int)
            or num_slots < 1
            or isinstance(feature_width, bool)
            or not isinstance(feature_width, int)
            or feature_width < 1
        ):
            raise ValueError("bidding slots require positive dimensions")
        self.num_slots = num_slots
        self.feature_width = feature_width
        self._shared_owners = []
        self._shared_specs = []

        def shared(shape, dtype):
            tensor, owner = _shared_tensor(shape, dtype)
            self._shared_owners.append(owner)
            self._shared_specs.append((tuple(shape), dtype))
            return tensor

        self.features = shared((num_slots, feature_width), torch.float32)
        self.legal_mask = shared((num_slots, 4), torch.bool)
        self.output_logits = shared((num_slots, 4), torch.float32)
        self.valid = shared((num_slots,), torch.bool)

    def __getstate__(self):
        state = self.__dict__.copy()
        for name in self._TENSOR_FIELDS:
            state.pop(name, None)
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        for name, owner, (shape, dtype) in zip(
            self._TENSOR_FIELDS, self._shared_owners, self._shared_specs
        ):
            setattr(self, name, _restore_shared_tensor(owner, shape, dtype))

    def clear(self, slot_id: int) -> None:
        self.valid[slot_id] = False

    def write(self, slot_id: int, observation) -> None:
        from douzero.observation.bidding import BiddingObservationV2

        if not isinstance(observation, BiddingObservationV2):
            raise TypeError("async bidding slot requires BiddingObservationV2")
        features = observation.to_tensor()
        legal_mask = torch.from_numpy(observation.bid_action_mask.copy())
        if features.shape != (self.feature_width,) or legal_mask.shape != (4,):
            raise ValueError("async bidding observation layout mismatch")
        self.features[slot_id].copy_(features)
        self.legal_mask[slot_id].copy_(legal_mask)
        self.valid[slot_id] = True

    def batch(
        self, slot_ids: list[int], feature_schema_hash: str
    ) -> BatchedBiddingInput:
        if not slot_ids:
            raise ValueError("async bidding batch must not be empty")
        indices = torch.tensor(slot_ids, dtype=torch.long)
        if not bool(self.valid.index_select(0, indices).all()):
            raise RuntimeError("async bidding batch contains an unwritten slot")
        return BatchedBiddingInput(
            features=self.features.index_select(0, indices),
            legal_mask=self.legal_mask.index_select(0, indices),
            feature_schema_hash=feature_schema_hash,
        )


class PinnedObservationBatchStager:
    """Reusable shared-SoA to pinned-batch staging for one action capacity."""

    def __init__(
        self,
        slots: SharedObservationSlots,
        *,
        max_batch_size: int,
        action_capacity: int,
    ) -> None:
        if max_batch_size < 1:
            raise ValueError("max_batch_size must be positive")
        if action_capacity < 1 or action_capacity > slots.max_actions:
            raise ValueError("action_capacity is outside the shared slot range")
        self.slots = slots
        self.max_batch_size = int(max_batch_size)
        self.action_capacity = int(action_capacity)

        def pinned(shape, dtype):
            return torch.empty(shape, dtype=dtype, pin_memory=True)

        batch = self.max_batch_size
        self.state_cards = pinned(
            (batch, *slots.state_cards.shape[1:]), slots.state_cards.dtype
        )
        self.state_flat = pinned(
            (batch, *slots.state_flat.shape[1:]), slots.state_flat.dtype
        )
        self.context_cards = pinned(
            (batch, *slots.context_cards.shape[1:]), slots.context_cards.dtype
        )
        self.context_flat = pinned(
            (batch, *slots.context_flat.shape[1:]), slots.context_flat.dtype
        )
        self.history = pinned(
            (batch, *slots.history.shape[1:]), slots.history.dtype
        )
        self.history_padding = pinned(
            (batch, *slots.history_padding.shape[1:]), slots.history_padding.dtype
        )
        self.actions = pinned(
            (batch, action_capacity, slots.actions.shape[-1]), slots.actions.dtype
        )
        self.action_mask = pinned(
            (batch, action_capacity), slots.action_mask.dtype
        )
        self.roles = pinned((batch,), slots.roles.dtype)
        self.strategy_features = (
            None
            if slots.strategy_features is None
            else pinned(
                (batch, action_capacity, STRATEGY_FEATURE_WIDTH),
                slots.strategy_features.dtype,
            )
        )
        self.style_features = (
            None
            if slots.style_features is None
            else pinned(
                (batch, STYLE_FEATURE_WIDTH), slots.style_features.dtype
            )
        )
        self.output_values = pinned(
            (batch, action_capacity, slots.output_width), torch.float32
        )

    @staticmethod
    def _gather(source: torch.Tensor, indices: torch.Tensor, destination) -> None:
        torch.index_select(source, 0, indices, out=destination)

    def gather_slots(self, slot_ids: list[int]) -> int:
        """Copy shared slot rows directly into the reusable pinned buffers."""
        batch_size = len(slot_ids)
        if not 1 <= batch_size <= self.max_batch_size:
            raise ValueError("staged request count is outside batch capacity")
        indices = torch.tensor(slot_ids, dtype=torch.long)
        counts = self.slots.action_counts.index_select(0, indices)
        if bool((counts < 1).any()) or bool((counts > self.action_capacity).any()):
            raise ValueError("request action count is outside staging capacity")

        self._gather(self.slots.state_cards, indices, self.state_cards[:batch_size])
        self._gather(self.slots.state_flat, indices, self.state_flat[:batch_size])
        self._gather(
            self.slots.context_cards, indices, self.context_cards[:batch_size]
        )
        self._gather(self.slots.context_flat, indices, self.context_flat[:batch_size])
        self._gather(self.slots.history, indices, self.history[:batch_size])
        self._gather(
            self.slots.history_padding,
            indices,
            self.history_padding[:batch_size],
        )
        self._gather(
            self.slots.actions[:, :self.action_capacity],
            indices,
            self.actions[:batch_size],
        )
        self._gather(
            self.slots.action_mask[:, :self.action_capacity],
            indices,
            self.action_mask[:batch_size],
        )
        self._gather(self.slots.roles, indices, self.roles[:batch_size])
        if self.strategy_features is not None:
            self._gather(
                self.slots.strategy_features[:, :self.action_capacity],
                indices,
                self.strategy_features[:batch_size],
            )
        if self.style_features is not None:
            self._gather(
                self.slots.style_features,
                indices,
                self.style_features[:batch_size],
            )
        return batch_size

    def batch_view(
        self,
        batch_size: int,
        feature_schema_hash: str,
    ) -> BatchedModelInputBundle:
        """Build the model-facing views over an already gathered batch."""
        if not 1 <= batch_size <= self.max_batch_size:
            raise ValueError("staged request count is outside batch capacity")
        state_cards = self.state_cards[:batch_size]
        context_cards = self.context_cards[:batch_size]
        return BatchedModelInputBundle(
            state_card_vectors=tuple(
                state_cards[:, index] for index in range(state_cards.shape[1])
            ),
            state_context_flat=self.state_flat[:batch_size],
            context_card_vectors=tuple(
                context_cards[:, index] for index in range(context_cards.shape[1])
            ),
            context_flat=self.context_flat[:batch_size],
            history_tokens=self.history[:batch_size],
            history_key_padding_mask=self.history_padding[:batch_size],
            action_features=self.actions[:batch_size],
            action_mask=self.action_mask[:batch_size],
            acting_role=self.roles[:batch_size],
            chosen_action_index=None,
            feature_schema_hashes=(feature_schema_hash,) * batch_size,
            strategy_features=(
                None
                if self.strategy_features is None
                else self.strategy_features[:batch_size]
            ),
            style_features=(
                None
                if self.style_features is None
                else self.style_features[:batch_size]
            ),
        )

    def stage_inputs(
        self,
        slot_ids: list[int],
        feature_schema_hash: str,
    ) -> BatchedModelInputBundle:
        """Gather shared rows and return their model-facing pinned views."""
        batch_size = self.gather_slots(slot_ids)
        return self.batch_view(batch_size, feature_schema_hash)

    def stage_outputs(self, values: torch.Tensor) -> torch.Tensor:
        expected = (
            values.shape[0], self.action_capacity, self.output_values.shape[-1]
        )
        if tuple(values.shape) != expected:
            raise ValueError(
                f"packed inference output must have shape {expected}, "
                f"got {tuple(values.shape)}"
            )
        destination = self.output_values[:values.shape[0]]
        destination.copy_(values, non_blocking=True)
        return destination


class PendingRequestScheduler:
    """FIFO request groups retained across service iterations."""

    def __init__(
        self,
        *,
        max_batch_size: int,
        target_batch_size: int = 4,
        max_delay_seconds: float = 0.002,
    ) -> None:
        if max_batch_size < 1 or target_batch_size < 1:
            raise ValueError("scheduler batch sizes must be positive")
        if max_delay_seconds < 0:
            raise ValueError("max_delay_seconds must be non-negative")
        self.max_batch_size = int(max_batch_size)
        self.target_batch_size = min(int(target_batch_size), self.max_batch_size)
        self.max_delay_seconds = float(max_delay_seconds)
        self._groups: dict[tuple[int, int | str], deque[RequestMetadata]] = (
            defaultdict(deque)
        )

    def add(self, requests: list[RequestMetadata]) -> None:
        for request in requests:
            self._groups[request.grouping_key].append(request)

    @property
    def pending_count(self) -> int:
        return sum(len(group) for group in self._groups.values())

    def pop_ready(
        self, *, now_ns: int | None = None
    ) -> tuple[tuple[int, int | str], list[RequestMetadata]] | None:
        if not self._groups:
            return None
        now_ns = time.monotonic_ns() if now_ns is None else int(now_ns)
        eligible = []
        for key, group in self._groups.items():
            oldest_wait = max(0.0, (now_ns - group[0].submitted_ns) / 1e9)
            if len(group) >= self.target_batch_size or (
                oldest_wait >= self.max_delay_seconds
            ):
                eligible.append((key, group, oldest_wait))
        if not eligible:
            return None
        # Prefer a launch that fills the GPU, then the oldest request.  FIFO is
        # preserved within every (snapshot, inference bucket) group.
        key, group, _ = max(
            eligible,
            key=lambda item: (
                min(len(item[1]), self.max_batch_size), item[2]
            ),
        )
        requests = [
            group.popleft()
            for _ in range(min(len(group), self.max_batch_size))
        ]
        if not group:
            del self._groups[key]
        return key, requests


class SharedReplaySlots:
    """Shared tensor handoff for completed, terminal-labelled transitions."""

    TARGET_NAMES = (
        "target_win", "target_score", "target_log_score",
        "target_min_turns_after", "target_min_turns_exact_mask",
        "target_regain_initiative", "target_teammate_finish",
        "target_teammate_finish_mask", "target_spring_probability",
        "target_structure_cost",
    )

    def __init__(
        self,
        schema,
        num_slots: int,
        max_actions: int = 256,
        *,
        v3_provenance: bool = False,
        strategy_features: bool = False,
        style_features: bool = False,
    ) -> None:
        self.context = mp.get_context("spawn")
        self.observations = SharedObservationSlots(
            schema,
            num_slots,
            max_actions,
            strategy_features=strategy_features,
            style_features=style_features,
        )
        self.v3_provenance = bool(v3_provenance)
        self._validation_shapes = compact_model_input_shapes(schema)
        self._shared_owners = []
        self._shared_specs = []
        self.labels, owner = _shared_tensor((num_slots, len(self.TARGET_NAMES)))
        self._shared_owners.append(owner)
        self._shared_specs.append(((num_slots, len(self.TARGET_NAMES)), torch.float32))
        self.labels.fill_(float("nan"))
        self.action_indices, owner = _shared_tensor((num_slots,), torch.int64)
        self._shared_owners.append(owner)
        self._shared_specs.append(((num_slots,), torch.int64))
        self.action_indices.fill_(-1)
        self.trace_indices, owner = _shared_tensor((num_slots,), torch.int64)
        self._shared_owners.append(owner)
        self._shared_specs.append(((num_slots,), torch.int64))
        self.trace_indices.fill_(-1)
        self.policy_steps, owner = _shared_tensor((num_slots,), torch.int64)
        self._shared_owners.append(owner)
        self._shared_specs.append(((num_slots,), torch.int64))
        self.policy_steps.fill_(-1)
        self.q_old, owner = _shared_tensor((num_slots,), torch.float32)
        self._shared_owners.append(owner)
        self._shared_specs.append(((num_slots,), torch.float32))
        self.q_old.fill_(float("nan"))
        self.actor_ids, owner = _shared_tensor((num_slots,), torch.int32)
        self._shared_owners.append(owner)
        self._shared_specs.append(((num_slots,), torch.int32))
        self.actor_ids.fill_(-1)
        self.episode_ids, owner = _shared_tensor((num_slots,), torch.int64)
        self._shared_owners.append(owner)
        self._shared_specs.append(((num_slots,), torch.int64))
        self.episode_ids.fill_(-1)
        self.free_queue = self.context.Queue()
        self.ready_queue = self.context.Queue()
        for slot_id in range(num_slots):
            self.free_queue.put(slot_id)

    _TENSOR_FIELDS = (
        "labels", "action_indices", "trace_indices", "policy_steps",
        "q_old", "actor_ids", "episode_ids",
    )

    def __getstate__(self):
        state = self.__dict__.copy()
        for name in self._TENSOR_FIELDS:
            state.pop(name, None)
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        for name, owner, (shape, dtype) in zip(
            self._TENSOR_FIELDS, self._shared_owners, self._shared_specs
        ):
            setattr(self, name, _restore_shared_tensor(owner, shape, dtype))

    def write_transition(
        self, transition, bundle: ModelInputBundle, policy_step: int,
        timeout_seconds: float, abort_event=None, shutdown_event=None,
    ) -> None:
        deadline = time.monotonic() + timeout_seconds
        while True:
            if abort_event is not None and abort_event.is_set():
                raise RuntimeError("async runtime aborted while waiting for replay slot")
            if shutdown_event is not None and shutdown_event.is_set():
                raise RuntimeError("async runtime shut down while waiting for replay slot")
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("timed out waiting for a shared replay slot")
            try:
                slot_id = int(self.free_queue.get(timeout=min(0.05, remaining)))
                break
            except queue.Empty:
                continue
        self.observations.write(slot_id, bundle)
        self.action_indices[slot_id] = transition.action_index
        self.trace_indices[slot_id] = transition.trace_index
        self.policy_steps[slot_id] = policy_step
        if self.v3_provenance:
            q_old = getattr(transition, "actor_q_old", None)
            actor_id = getattr(transition, "actor_id", None)
            episode_id = getattr(transition, "episode_id", None)
            if (
                isinstance(q_old, bool)
                or not isinstance(q_old, (int, float))
                or not math.isfinite(q_old)
            ):
                raise ValueError("V3 async transition requires finite actor q_old")
            for name, value in (("actor_id", actor_id), ("episode_id", episode_id)):
                if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                    raise ValueError(f"V3 async transition requires non-negative {name}")
            self.q_old[slot_id] = float(q_old)
            self.actor_ids[slot_id] = actor_id
            self.episode_ids[slot_id] = episode_id
        for index, name in enumerate(self.TARGET_NAMES):
            self.labels[slot_id, index] = float(getattr(transition, name))
        self.ready_queue.put(slot_id)

    def read_ready(self, feature_schema_hash: str, policy_version: str):
        from douzero.training.v2_buffer import (
            COMPACT_REPLAY_SCHEMA_VERSION,
            CompactTensorTransition,
        )

        records = []
        while True:
            try:
                slot_id = int(self.ready_queue.get_nowait())
            except queue.Empty:
                break
            source = self.observations.read_bundle(slot_id, feature_schema_hash)
            bundle = ModelInputBundle(
                state_card_vectors=tuple(value.to(torch.int8) for value in source.state_card_vectors),
                state_context_flat=source.state_context_flat.to(torch.int8),
                context_card_vectors=tuple(value.to(torch.int8) for value in source.context_card_vectors),
                context_flat=source.context_flat.to(torch.int32),
                history_tokens=source.history_tokens.to(torch.int8),
                history_key_padding_mask=source.history_key_padding_mask.clone(),
                action_features=source.action_features.to(torch.int8),
                action_mask=source.action_mask.clone(),
                acting_role=source.acting_role,
                feature_schema_hash=feature_schema_hash,
            )
            targets = {
                name: float(self.labels[slot_id, index].item())
                for index, name in enumerate(self.TARGET_NAMES)
            }
            records.append(CompactTensorTransition(
                model_inputs=bundle,
                action_index=int(self.action_indices[slot_id]),
                position=source.acting_role,
                targets=targets,
                trace_index=int(self.trace_indices[slot_id]),
                policy_id=policy_version,
                teammate_policy_id=None,
                policy_version=policy_version,
                policy_step=int(self.policy_steps[slot_id]),
                schema_version=COMPACT_REPLAY_SCHEMA_VERSION,
            ))
            records[-1].validate(
                feature_schema_hash,
                expected_tensor_shapes=self._validation_shapes,
            )
            self.free_queue.put(slot_id)
        return records

    def read_ready_v3_aligned(
        self,
        *,
        feature_schema_hash: str,
        target_transform: str,
        ruleset_identity,
        include_strategy_targets: bool = False,
    ):
        """Drain public V3 rows with exact actor-snapshot Q provenance."""
        if not self.v3_provenance:
            raise RuntimeError("shared replay was not configured for V3 provenance")
        if not isinstance(include_strategy_targets, bool):
            raise TypeError("include_strategy_targets must be bool")
        if (
            include_strategy_targets
            and not self.observations.strategy_features_enabled
        ):
            raise ValueError(
                "strategy targets require strategy-enriched public replay"
            )
        from douzero.v3_hybrid.replay import (
            AdaptiveSnapshotProvenance,
            V3ReplayTransition,
        )

        records = []
        while True:
            try:
                slot_id = int(self.ready_queue.get_nowait())
            except queue.Empty:
                break
            source = self.observations.read_bundle(slot_id, feature_schema_hash)
            episode_id = int(self.episode_ids[slot_id])
            actor_id = int(self.actor_ids[slot_id])
            policy_step = int(self.policy_steps[slot_id])
            record = V3ReplayTransition(
                model_inputs=ModelInputBundle(
                    state_card_vectors=tuple(value.clone() for value in source.state_card_vectors),
                    state_context_flat=source.state_context_flat.clone(),
                    context_card_vectors=tuple(value.clone() for value in source.context_card_vectors),
                    context_flat=source.context_flat.clone(),
                    history_tokens=source.history_tokens.clone(),
                    history_key_padding_mask=source.history_key_padding_mask.clone(),
                    action_features=source.action_features.clone(),
                    action_mask=source.action_mask.clone(),
                    acting_role=source.acting_role,
                    feature_schema_hash=feature_schema_hash,
                    strategy_features=(
                        None
                        if source.strategy_features is None
                        else source.strategy_features.clone()
                    ),
                    style_features=(
                        None
                        if source.style_features is None
                        else source.style_features.clone()
                    ),
                ),
                selected_action_index=int(self.action_indices[slot_id]),
                role=source.acting_role,
                episode_id=f"actor-{actor_id}-episode-{episode_id}",
                deal_id=f"async-deal-{episode_id}",
                target_transform=target_transform,
                mc_return=float(self.labels[slot_id, 1]),
                adaptive_provenance=AdaptiveSnapshotProvenance(
                    q_old=float(self.q_old[slot_id]),
                    policy_version=policy_step,
                    snapshot_slot=0,
                    owner_id=actor_id,
                    generation=policy_step,
                ),
                **dict(ruleset_identity),
            )
            record.validate(
                expected_schema_hash=feature_schema_hash,
                expected_target_transform=target_transform,
                expected_ruleset_identity=ruleset_identity,
                adaptive_required=True,
                strategy_features_allowed=(
                    self.observations.strategy_features_enabled
                ),
                style_features_allowed=self.observations.style_features_enabled,
            )
            key = AsyncReplayKey(
                    actor_id=actor_id,
                    episode_id=episode_id,
                    trace_index=int(self.trace_indices[slot_id]),
                )
            if include_strategy_targets:
                target_names = (
                    "min_turns_after",
                    "min_turns_exact_mask",
                    "regain_initiative",
                    "teammate_finish",
                    "teammate_finish_mask",
                    "spring_probability",
                    "structure_cost",
                )
                targets = {
                    name: float(self.labels[slot_id, index + 3].item())
                    for index, name in enumerate(target_names)
                }
                if not all(math.isfinite(value) for value in targets.values()):
                    raise ValueError(
                        "async strategy replay contains non-finite labels"
                    )
                records.append((record, key, targets))
            else:
                records.append((record, key))
            self.free_queue.put(slot_id)
        return records

    def read_ready_v3(
        self,
        *,
        feature_schema_hash: str,
        target_transform: str,
        ruleset_identity,
    ):
        return [
            record
            for record, _key in self.read_ready_v3_aligned(
                feature_schema_hash=feature_schema_hash,
                target_transform=target_transform,
                ruleset_identity=ruleset_identity,
            )
        ]

    def close(self) -> None:
        self.free_queue.close()
        self.ready_queue.close()


class AsyncRequestCoordinator:
    """Fail-fast shared request state machine using a ``spawn`` context."""

    def __init__(
        self,
        schema,
        *,
        num_slots: int,
        max_actions: int = 256,
        output_width: int = 5,
        request_timeout_seconds: float = 30.0,
        belief_inputs: bool = False,
        strategy_features: bool = False,
        style_features: bool = False,
        bidding_feature_width: int | None = None,
    ) -> None:
        if request_timeout_seconds <= 0:
            raise ValueError("request_timeout_seconds must be positive")
        for name, value in (
            ("belief_inputs", belief_inputs),
            ("strategy_features", strategy_features),
            ("style_features", style_features),
        ):
            if not isinstance(value, bool):
                raise TypeError(f"{name} must be bool")
        self.context = mp.get_context("spawn")
        if self.context.get_start_method() != "spawn":
            raise RuntimeError("async V2 requires multiprocessing spawn")
        self.slots = SharedObservationSlots(
            schema,
            num_slots,
            max_actions,
            output_width=output_width,
            strategy_features=strategy_features,
            style_features=style_features,
        )
        self.belief_inputs = (
            SharedBeliefInputSlots(num_slots) if belief_inputs else None
        )
        self.bidding_inputs = (
            None
            if bidding_feature_width is None
            else SharedBiddingSlots(num_slots, bidding_feature_width)
        )
        self._shared_owners = []
        self._shared_specs = []
        self.states, owner = _shared_tensor((num_slots,), torch.int8)
        self._shared_owners.append(owner)
        self._shared_specs.append(((num_slots,), torch.int8))
        self.request_ids, owner = _shared_tensor((num_slots,), torch.int64)
        self._shared_owners.append(owner)
        self._shared_specs.append(((num_slots,), torch.int64))
        self.request_ids.fill_(-1)
        self.policy_snapshots, owner = _shared_tensor((num_slots,), torch.int64)
        self._shared_owners.append(owner)
        self._shared_specs.append(((num_slots,), torch.int64))
        self.policy_snapshots.fill_(-1)
        self.actor_ids, owner = _shared_tensor((num_slots,), torch.int32)
        self._shared_owners.append(owner)
        self._shared_specs.append(((num_slots,), torch.int32))
        self.actor_ids.fill_(-1)
        self.submitted_ns, owner = _shared_tensor((num_slots,), torch.int64)
        self._shared_owners.append(owner)
        self._shared_specs.append(((num_slots,), torch.int64))
        self.request_kinds, owner = _shared_tensor((num_slots,), torch.int8)
        self._shared_owners.append(owner)
        self._shared_specs.append(((num_slots,), torch.int8))
        self.request_kinds.fill_(int(RequestKind.CARD_PLAY))
        self.ready_queue = self.context.Queue()
        self.free_queue = self.context.Queue()
        # Shared result tensors use RawArray storage, so publishing DONE in a
        # separate RawArray is not a synchronization boundary.  A per-slot
        # Event provides the release/acquire hand-off: the coordinator writes
        # every result tensor before set(), and the actor waits before reading.
        self.response_events = [
            self.context.Event() for _ in range(num_slots)
        ]
        for slot_id in range(num_slots):
            self.free_queue.put(slot_id)
        self.request_timeout_seconds = float(request_timeout_seconds)
        self._submitted_at: dict[int, float] = {}
        self.abort_event = self.context.Event()
        self.shutdown_event = self.context.Event()
        self.failure_message = self.context.Array(ctypes.c_char, 1024, lock=True)
        self.active_games = 0
        self.completed_episodes_pending = 0

    _TENSOR_FIELDS = (
        "states", "request_ids", "policy_snapshots", "actor_ids", "submitted_ns",
        "request_kinds",
    )

    def __getstate__(self):
        state = self.__dict__.copy()
        for name in self._TENSOR_FIELDS:
            state.pop(name, None)
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        for name, owner, (shape, dtype) in zip(
            self._TENSOR_FIELDS, self._shared_owners, self._shared_specs
        ):
            setattr(self, name, _restore_shared_tensor(owner, shape, dtype))

    def _raise_if_failed(self) -> None:
        if self.abort_event.is_set():
            with self.failure_message.get_lock():
                reason = bytes(self.failure_message.value).decode("utf-8", "replace")
            raise RuntimeError(
                f"async actor runtime failed: {reason or 'unknown worker failure'}"
            )
        if self.shutdown_event.is_set():
            raise RuntimeError("async actor runtime is shut down")

    def acquire(self, actor_id: int, timeout: float | None = None) -> int:
        self._raise_if_failed()
        timeout = self.request_timeout_seconds if timeout is None else timeout
        deadline = time.monotonic() + timeout
        while True:
            self._raise_if_failed()
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("timed out waiting for a free inference slot")
            try:
                slot_id = int(self.free_queue.get(timeout=min(0.05, remaining)))
                break
            except queue.Empty:
                continue
        if int(self.states[slot_id]) != SlotState.FREE:
            raise RuntimeError("free queue returned a non-FREE slot")
        self.response_events[slot_id].clear()
        if self.belief_inputs is not None:
            self.belief_inputs.clear(slot_id)
        if self.bidding_inputs is not None:
            self.bidding_inputs.clear(slot_id)
        self.states[slot_id] = int(SlotState.WRITING)
        self.actor_ids[slot_id] = actor_id
        return slot_id

    def submit(
        self, slot_id: int, *, request_id: int, policy_snapshot: int
    ) -> None:
        self._raise_if_failed()
        if int(self.states[slot_id]) != SlotState.WRITING:
            raise RuntimeError("only a WRITING slot may be submitted")
        count = int(self.slots.action_counts[slot_id])
        if count < 1 or not bool(self.slots.action_mask[slot_id, :count].any()):
            self.states[slot_id] = int(SlotState.FAILED)
            raise ValueError("request has zero legal actions")
        if self.belief_inputs is not None:
            try:
                belief = self.belief_inputs.read(slot_id)
            except Exception:
                self.states[slot_id] = int(SlotState.FAILED)
                raise
            role = SUPPORTED_ROLES[int(self.slots.roles[slot_id])]
            if belief.acting_role != role:
                self.states[slot_id] = int(SlotState.FAILED)
                raise ValueError("public policy and belief acting roles differ")
        self.request_ids[slot_id] = request_id
        self.policy_snapshots[slot_id] = policy_snapshot
        self.request_kinds[slot_id] = int(RequestKind.CARD_PLAY)
        self.states[slot_id] = int(SlotState.READY)
        self._submitted_at[slot_id] = time.monotonic()
        self.submitted_ns[slot_id] = time.monotonic_ns()
        self.ready_queue.put(slot_id)

    def submit_bidding(
        self, slot_id: int, *, request_id: int, policy_snapshot: int
    ) -> None:
        self._raise_if_failed()
        if self.bidding_inputs is None:
            raise RuntimeError("async bidding transport is disabled")
        if int(self.states[slot_id]) != SlotState.WRITING:
            raise RuntimeError("only a WRITING slot may be submitted")
        if not bool(self.bidding_inputs.valid[slot_id]):
            self.states[slot_id] = int(SlotState.FAILED)
            raise ValueError("bidding request slot was not written")
        if not bool(self.bidding_inputs.legal_mask[slot_id].any()):
            self.states[slot_id] = int(SlotState.FAILED)
            raise ValueError("bidding request has zero legal bids")
        self.request_ids[slot_id] = request_id
        self.policy_snapshots[slot_id] = policy_snapshot
        self.request_kinds[slot_id] = int(RequestKind.BIDDING)
        self.states[slot_id] = int(SlotState.READY)
        self._submitted_at[slot_id] = time.monotonic()
        self.submitted_ns[slot_id] = time.monotonic_ns()
        self.ready_queue.put(slot_id)

    def claim_ready(self, max_items: int, wait_seconds: float = 0.0) -> list[RequestMetadata]:
        self._raise_if_failed()
        if max_items < 1:
            raise ValueError("max_items must be positive")
        slot_ids: list[int] = []
        wait_seconds = max(0.0, wait_seconds)
        # Wait for the first request, then give its peers a complete
        # coalescing window.  Starting the deadline before the first request
        # arrives leaves almost no batching opportunity when the queue was
        # initially empty.
        try:
            if wait_seconds:
                slot_ids.append(int(self.ready_queue.get(timeout=wait_seconds)))
            else:
                slot_ids.append(int(self.ready_queue.get_nowait()))
        except queue.Empty:
            return []
        deadline = time.monotonic() + wait_seconds
        while len(slot_ids) < max_items:
            timeout = max(0.0, deadline - time.monotonic()) if wait_seconds else 0.0
            try:
                if wait_seconds:
                    slot_ids.append(int(self.ready_queue.get(timeout=timeout)))
                else:
                    slot_ids.append(int(self.ready_queue.get_nowait()))
            except queue.Empty:
                break
        metadata = []
        for slot_id in slot_ids:
            if int(self.states[slot_id]) != SlotState.READY:
                raise RuntimeError("ready queue returned a non-READY slot")
            self.states[slot_id] = int(SlotState.RUNNING)
            request_kind = int(self.request_kinds[slot_id])
            if request_kind not in {
                int(RequestKind.CARD_PLAY), int(RequestKind.BIDDING)
            }:
                raise RuntimeError("ready queue contains an unknown request kind")
            metadata.append(RequestMetadata(
                slot_id=slot_id,
                actor_id=int(self.actor_ids[slot_id]),
                request_id=int(self.request_ids[slot_id]),
                policy_snapshot=int(self.policy_snapshots[slot_id]),
                action_count=(
                    4
                    if request_kind == int(RequestKind.BIDDING)
                    else int(self.slots.action_counts[slot_id])
                ),
                acting_role=(
                    -1
                    if request_kind == int(RequestKind.BIDDING)
                    else int(self.slots.roles[slot_id])
                ),
                submitted_ns=int(self.submitted_ns[slot_id]),
                request_kind=request_kind,
            ))
        return metadata

    def complete(self, slot_id: int) -> None:
        if int(self.states[slot_id]) != SlotState.RUNNING:
            raise RuntimeError("only a RUNNING slot may complete")
        self.states[slot_id] = int(SlotState.DONE)
        self.response_events[slot_id].set()

    def wait_done(self, slot_id: int, request_id: int) -> None:
        deadline = time.monotonic() + self.request_timeout_seconds
        response_event = self.response_events[slot_id]
        while True:
            self._raise_if_failed()
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            if not response_event.wait(timeout=min(0.05, remaining)):
                continue
            # Event.wait() is the acquire side of the result publication
            # barrier.  Only inspect state and RawArray-backed outputs after it
            # has observed complete(), fail(), or shutdown().
            self._raise_if_failed()
            state = SlotState(int(self.states[slot_id]))
            if state == SlotState.DONE:
                if int(self.request_ids[slot_id]) != request_id:
                    raise RuntimeError("inference response request_id mismatch")
                return
            if state in {SlotState.FAILED, SlotState.SHUTDOWN}:
                raise RuntimeError(f"inference request ended in state {state.name}")
            raise RuntimeError(
                f"inference response event published unexpected state {state.name}"
            )
        self.fail(f"request {request_id} timed out")
        raise TimeoutError(f"inference request {request_id} timed out")

    def release(self, slot_id: int) -> None:
        if int(self.states[slot_id]) != SlotState.DONE:
            raise RuntimeError("only a DONE slot may be released")
        self.states[slot_id] = int(SlotState.FREE)
        if self.belief_inputs is not None:
            self.belief_inputs.clear(slot_id)
        if self.bidding_inputs is not None:
            self.bidding_inputs.clear(slot_id)
        self.response_events[slot_id].clear()
        self._submitted_at.pop(slot_id, None)
        self.free_queue.put(slot_id)

    def fail(self, reason: str) -> None:
        message = (str(reason) or "unknown worker failure").encode("utf-8")[:1023]
        with self.failure_message.get_lock():
            if not self.failure_message.value:
                self.failure_message.value = message
        self.abort_event.set()
        for slot_id in range(len(self.states)):
            if int(self.states[slot_id]) not in {
                SlotState.FREE, SlotState.DONE, SlotState.SHUTDOWN
            }:
                self.states[slot_id] = int(SlotState.FAILED)
            self.response_events[slot_id].set()

    def quiesce(self) -> dict[str, int]:
        self._raise_if_failed()
        counts = {
            state.name.lower(): int((self.states == int(state)).sum().item())
            for state in SlotState
        }
        if self.active_games or self.completed_episodes_pending:
            raise RuntimeError("cannot quiesce with active or uncommitted episodes")
        if counts["ready"] or counts["running"] or counts["writing"]:
            raise RuntimeError("cannot quiesce with in-flight inference requests")
        return counts

    def request_shutdown(self) -> None:
        """Publish shutdown before joins without closing shared queues."""
        if self.shutdown_event.is_set():
            return
        self.shutdown_event.set()
        for slot_id in range(len(self.states)):
            self.states[slot_id] = int(SlotState.SHUTDOWN)
            self.response_events[slot_id].set()

    def shutdown(self) -> None:
        self.request_shutdown()
        self.ready_queue.close()
        self.free_queue.close()


def async_actor_main(
    actor_id: int,
    task_queue,
    event_queue,
    coordinator: AsyncRequestCoordinator,
    replay_slots: SharedReplaySlots,
    *,
    environment_seed: int,
    environment_seed_derivation: str = TOPOLOGY_LOCAL_SEED_DERIVATION_V1,
    action_rng_seed: int,
    epsilon: float,
    max_steps: int,
    decision_config,
    ruleset,
    feature_schema_hash: str,
    policy_version: str,
    policy_step,
    games_per_actor: int,
    runtime_kind: str = "v2",
    belief_sidecar_queue=None,
    oracle_sidecar_queue=None,
    cooperation_sidecar_queue=None,
    strategy_config=None,
    style_enabled: bool = False,
    strategy_targets_enabled: bool = False,
    strategy_node_budget: int = 500,
    strategy_time_budget_ms: int = 0,
    bidding_replay_queue=None,
    bidding_policy_config=None,
    first_bidder_mode: str = "rotate",
) -> None:
    """CPU-only interleaved-game actor. CUDA is never initialized here."""
    import random

    from douzero.env.env import Env
    from douzero.belief.features import build_belief_input
    from douzero.models_v2.batch import observation_to_model_inputs
    from douzero.models_v2.output import ModelOutput
    from douzero.observation.encode_v2 import get_obs_v2
    from douzero.observation.bidding import get_bidding_obs_v2
    from douzero.training.bidding import (
        BiddingPolicyConfig,
        BiddingTransition,
        select_bidding_action,
    )
    from douzero.training.decision_policy import select_action
    from douzero.training.v2_buffer import Episode, Transition

    rng = (
        random.Random()
        if action_rng_seed == 0
        else random.Random(action_rng_seed + actor_id)
    )
    if environment_seed_derivation not in {
        TOPOLOGY_LOCAL_SEED_DERIVATION_V1,
        FORMAL_SEED_DERIVATION_V1,
    }:
        raise ValueError("unsupported async environment seed derivation")
    if (
        environment_seed
        and environment_seed_derivation == TOPOLOGY_LOCAL_SEED_DERIVATION_V1
    ):
        np.random.seed((environment_seed + actor_id) % (1 << 32))
    if games_per_actor < 1:
        raise ValueError("games_per_actor must be positive")
    if runtime_kind not in {"v2", "v3_hybrid"}:
        raise ValueError("runtime_kind must be v2 or v3_hybrid")
    belief_async = coordinator.belief_inputs is not None
    oracle_async = oracle_sidecar_queue is not None
    cooperation_async = cooperation_sidecar_queue is not None
    public_aux_async = (
        coordinator.slots.strategy_features_enabled
        or coordinator.slots.style_features_enabled
    )
    bidding_async = coordinator.bidding_inputs is not None
    for name, value in (
        ("style_enabled", style_enabled),
        ("strategy_targets_enabled", strategy_targets_enabled),
    ):
        if not isinstance(value, bool):
            raise TypeError(f"async {name} must be bool")
    if coordinator.slots.style_features_enabled != style_enabled:
        raise ValueError("async style feature transport and actor config disagree")
    if (
        coordinator.slots.strategy_features_enabled
        != (strategy_config is not None)
    ):
        raise ValueError(
            "async strategy feature transport and actor config disagree"
        )
    if strategy_targets_enabled != coordinator.slots.strategy_features_enabled:
        raise ValueError(
            "async strategy target and public feature transport must match"
        )
    if belief_async != (belief_sidecar_queue is not None):
        raise ValueError(
            "async belief inference and training sidecar must be enabled together"
        )
    if belief_async and runtime_kind != "v3_hybrid":
        raise ValueError("async belief is supported only by the V3 runtime")
    if oracle_async and runtime_kind != "v3_hybrid":
        raise ValueError("async Oracle is supported only by the V3 runtime")
    if cooperation_async and runtime_kind != "v3_hybrid":
        raise ValueError("async cooperation is supported only by the V3 runtime")
    if bidding_async != (bidding_replay_queue is not None):
        raise ValueError(
            "async bidding inference and replay must be enabled together"
        )
    if bidding_async:
        if runtime_kind != "v3_hybrid":
            raise ValueError("async bidding is supported only by the V3 runtime")
        if ruleset is None or ruleset.ruleset_id != "standard":
            raise ValueError("async bidding requires the standard ruleset")
        if not isinstance(bidding_policy_config, BiddingPolicyConfig):
            raise TypeError("async bidding requires BiddingPolicyConfig")
        if first_bidder_mode not in {"rotate", "seeded_random"}:
            raise ValueError("async first_bidder_mode is unsupported")
    elif bidding_policy_config is not None:
        raise ValueError("bidding policy config requires async bidding transport")
    transport_flags = (
        belief_async,
        oracle_async,
        cooperation_async,
        public_aux_async,
        bidding_async,
    )
    full_hybrid_transport = transport_flags in {
        (True, True, True, True, False),
        (True, True, True, True, True),
    }
    if sum(transport_flags) > 1 and not full_hybrid_transport:
        raise NotImplementedError(
            "partial combined async H7.1 capability transports are not supported"
        )
    if belief_async or oracle_async:
        from douzero.observation.privileged import PrivilegedObservation
    if belief_async:
        from douzero.v3_hybrid.training.h4_learner import (
            build_v3_h4_belief_sidecar,
        )
    if oracle_async:
        from douzero.v3_hybrid.training.h3_learner import (
            build_v3_h3_oracle_sidecar,
        )
    if cooperation_async:
        from douzero.v3_hybrid.training.cooperation import (
            FARMER_ROLES,
            build_v3_h5_async_decision_sidecar,
        )
    request_id = actor_id << 48

    def publish_belief_sidecar(key, sidecar) -> None:
        deadline = time.monotonic() + coordinator.request_timeout_seconds
        while True:
            coordinator._raise_if_failed()
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(
                    "async belief sidecar queue remained full past its deadline"
                )
            try:
                belief_sidecar_queue.put(
                    (key, sidecar), timeout=min(0.05, remaining)
                )
                return
            except queue.Full:
                continue

    def publish_oracle_sidecar(key, sidecar) -> None:
        deadline = time.monotonic() + coordinator.request_timeout_seconds
        while True:
            coordinator._raise_if_failed()
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(
                    "async Oracle sidecar queue remained full past its deadline"
                )
            try:
                oracle_sidecar_queue.put(
                    (key, sidecar), timeout=min(0.05, remaining)
                )
                return
            except queue.Full:
                continue

    def publish_cooperation_sidecar(key, sidecar) -> None:
        deadline = time.monotonic() + coordinator.request_timeout_seconds
        while True:
            coordinator._raise_if_failed()
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(
                    "async cooperation sidecar queue remained full past its deadline"
                )
            try:
                cooperation_sidecar_queue.put(
                    (key, sidecar), timeout=min(0.05, remaining)
                )
                return
            except queue.Full:
                continue

    def publish_bidding_transitions(transitions) -> None:
        deadline = time.monotonic() + coordinator.request_timeout_seconds
        payload = tuple(transitions)
        while True:
            coordinator._raise_if_failed()
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(
                    "async bidding replay queue remained full past its deadline"
                )
            try:
                bidding_replay_queue.put(
                    payload, timeout=min(0.05, remaining)
                )
                return
            except queue.Full:
                continue

    def start_game(task):
        episode_id = int(task)
        if environment_seed_derivation == FORMAL_SEED_DERIVATION_V1:
            np.random.seed(derive_formal_stream_seed(
                environment_seed,
                "environment",
                0,
                episode_id,
            ))
            action_rng = random.Random(
                _formal_action_seed(action_rng_seed, actor_id, episode_id)
            )
        else:
            action_rng = rng
        snapshot = int(policy_step.value)
        event_queue.put(("started", actor_id, episode_id, snapshot))
        env = Env("adp", ruleset=ruleset)
        bidding_order = None
        if bidding_async:
            seats = ["0", "1", "2"]
            if first_bidder_mode == "rotate":
                offset = episode_id % len(seats)
            else:
                offset = action_rng.randrange(len(seats))
            bidding_order = seats[offset:] + seats[:offset]
        env.reset(bidding_order=bidding_order)
        return {
            "episode_id": episode_id,
            "snapshot": snapshot,
            "env": env,
            "episode": Episode(
                policy_version_at_start=policy_version,
                policy_step_at_start=snapshot,
            ),
            "steps": 0,
            "pending": None,
            "started_at": time.monotonic(),
            "blocked_seconds": 0.0,
            "action_rng": action_rng,
            "bidding_transitions": [],
            "bidding_decisions": 0,
            "abandoned_bidding_transitions": 0,
            "redeals": 0,
        }

    def finish_game(game) -> None:
        episode = game["episode"]
        if episode.max_redeals_exceeded:
            episode.excluded_from_training = True
            episode.exclusion_reason = "redeal_cap_guard"
            episode.transitions.clear()
            game["bidding_transitions"].clear()
        episode.label_from_terminal()
        if bidding_async:
            seat_to_role = game["env"]._env._seat_to_role
            for transition in game["bidding_transitions"]:
                if not transition.actor_role:
                    transition.assign_actor_role(seat_to_role)
                transition.label_from_terminal(episode.terminal_result)
                transition.validate()
            publish_bidding_transitions(game["bidding_transitions"])
        if strategy_targets_enabled:
            episode.label_strategy_auxiliary(
                node_budget=strategy_node_budget,
                time_budget_ms=strategy_time_budget_ms,
            )
        for transition in episode.transitions:
            public_inputs = getattr(transition, "public_model_inputs", None)
            if public_inputs is None:
                public_inputs = observation_to_model_inputs(
                    transition.obs,
                    strategy_config,
                    style_enabled=style_enabled,
                )
            replay_slots.write_transition(
                transition,
                public_inputs,
                game["snapshot"],
                coordinator.request_timeout_seconds,
                coordinator.abort_event,
                coordinator.shutdown_event,
            )
            if belief_async:
                sidecar = getattr(transition, "belief_sidecar", None)
                if sidecar is None:
                    raise RuntimeError(
                        "async belief transition is missing its training sidecar"
                    )
                publish_belief_sidecar(
                    AsyncReplayKey(
                        actor_id=actor_id,
                        episode_id=game["episode_id"],
                        trace_index=transition.trace_index,
                    ),
                    sidecar,
                )
            if oracle_async:
                sidecar = getattr(transition, "oracle_sidecar", None)
                if sidecar is None:
                    raise RuntimeError(
                        "async Oracle transition is missing its training sidecar"
                    )
                publish_oracle_sidecar(
                    AsyncReplayKey(
                        actor_id=actor_id,
                        episode_id=game["episode_id"],
                        trace_index=transition.trace_index,
                    ),
                    sidecar,
                )
            if cooperation_async and transition.position in FARMER_ROLES:
                sidecar = getattr(transition, "cooperation_sidecar", None)
                if sidecar is None:
                    raise RuntimeError(
                        "async farmer transition is missing its cooperation sidecar"
                    )
                publish_cooperation_sidecar(
                    AsyncReplayKey(
                        actor_id=actor_id,
                        episode_id=game["episode_id"],
                        trace_index=transition.trace_index,
                    ),
                    sidecar,
                )
        team = episode.terminal_result.get("winner_team", "landlord")
        farmer_counts = {
            role: sum(
                transition.position == role for transition in episode.transitions
            )
            for role in ("landlord_up", "landlord_down")
        }
        completed_event = (
            "completed", actor_id, game["episode_id"], len(episode.transitions),
            0 if team == "landlord" else 1, game["snapshot"],
            len(episode.action_trace),
            float(game["blocked_seconds"]),
            float(time.monotonic() - game["started_at"]),
            farmer_counts,
        )
        if bidding_async:
            completed_event += (
                len(game["bidding_transitions"]),
                int(game["bidding_decisions"]),
                int(game["abandoned_bidding_transitions"]),
                int(game["redeals"]),
            )
        event_queue.put(completed_event)

    def apply_action(
        game,
        action_index,
        obs,
        position,
        legal_actions,
        *,
        q_old=None,
        belief_sidecar=None,
        public_inputs=None,
    ) -> bool:
        episode = game["episode"]
        if obs is not None:
            transition = Transition(
                obs=obs,
                action_index=action_index,
                position=position,
                trace_index=len(episode.action_trace),
                policy_id=policy_version,
                policy_version=policy_version,
                policy_step=game["snapshot"],
            )
            if runtime_kind == "v3_hybrid":
                if q_old is None or not math.isfinite(float(q_old)):
                    raise ValueError("V3 actor decision requires finite snapshot q_old")
                transition.actor_q_old = float(q_old)
                transition.actor_id = int(actor_id)
                transition.episode_id = int(game["episode_id"])
                if public_inputs is None:
                    raise ValueError(
                        "V3 actor decision requires its served public inputs"
                    )
                transition.public_model_inputs = public_inputs
            if belief_async:
                if belief_sidecar is None:
                    raise ValueError(
                        "async belief decision requires a training sidecar"
                    )
                transition.belief_sidecar = belief_sidecar
            if oracle_async:
                privileged = PrivilegedObservation(
                    all_handcards=dict(game["env"].infoset.all_handcards),
                    acting_role=position,
                )
                transition.oracle_sidecar = build_v3_h3_oracle_sidecar(
                    obs,
                    privileged,
                    action_index=action_index,
                    public_inputs=public_inputs,
                )
            if cooperation_async and position in FARMER_ROLES:
                provenance = f"{policy_version}@{game['snapshot']}"
                transition.cooperation_sidecar = (
                    build_v3_h5_async_decision_sidecar(
                        obs,
                        selected_action_index=action_index,
                        trace_index=transition.trace_index,
                        public_inputs=public_inputs,
                        snapshot_policy_version=int(game["snapshot"]),
                        policy_id=provenance,
                        teammate_policy_id=provenance,
                    )
                )
            episode.transitions.append(transition)
        action = legal_actions[action_index]
        episode.action_trace.append((position, tuple(sorted(action))))
        _obs, _reward, done, info = game["env"].step(action)
        game["steps"] += 1
        if done:
            episode.terminal_result = info or {}
            finish_game(game)
            return True
        if game["steps"] >= max_steps:
            raise RuntimeError(f"actor episode exceeded max_steps={max_steps}")
        return False

    def apply_bidding_action(game, observation, bid: int, source: str) -> None:
        if bid not in observation.legal_bids:
            raise ValueError("async bidding selected an illegal environment bid")
        game["bidding_transitions"].append(BiddingTransition(
            obs=observation,
            bid_action=bid,
            policy_version=policy_version,
            source_policy=source,
        ))
        game["bidding_decisions"] += 1
        _obs, _reward, done, info = game["env"].step(None, bid_value=bid)
        game["steps"] += 1
        if done and info.get("redeal"):
            abandoned = len(game["bidding_transitions"])
            game["abandoned_bidding_transitions"] += abandoned
            game["bidding_transitions"].clear()
            game["redeals"] = int(info["redeal_count"])
            np.random.seed(_async_redeal_seed(
                environment_seed,
                environment_seed_derivation,
                actor_id,
                int(game["episode_id"]),
                game["redeals"],
            ))
            game["env"].redeal()
            return
        if info.get("max_redeals_exceeded"):
            if game["env"].bidding_obs is not None:
                raise RuntimeError(
                    "environment redeal-cap transition did not enter card play"
                )
            game["abandoned_bidding_transitions"] += len(
                game["bidding_transitions"]
            )
            game["bidding_transitions"].clear()
            game["episode"].max_redeals_exceeded = True
            return
        if done:
            raise RuntimeError(
                "async bidding ended without a terminal card-play result"
            )
        if game["env"].bidding_obs is None:
            seat_to_role = game["env"]._env._seat_to_role
            for transition in game["bidding_transitions"]:
                transition.assign_actor_role(seat_to_role)

    def advance_until_request_or_done(game) -> bool:
        nonlocal request_id
        while True:
            if bidding_async and game["env"].bidding_obs is not None:
                bid_obs = get_bidding_obs_v2(
                    game["env"].bidding_obs,
                    ruleset=ruleset,
                    redeal_count=game["env"]._redeal_count,
                )
                action_rng = game["action_rng"]
                use_learned = (
                    bidding_policy_config.policy == "learned"
                    and action_rng.random()
                    < bidding_policy_config.learned_probability
                )
                if use_learned:
                    slot_id = coordinator.acquire(actor_id)
                    coordinator.bidding_inputs.write(slot_id, bid_obs)
                    request_id += 1
                    coordinator.submit_bidding(
                        slot_id,
                        request_id=request_id,
                        policy_snapshot=game["snapshot"],
                    )
                    game["pending"] = (
                        "bidding",
                        slot_id,
                        request_id,
                        bid_obs,
                    )
                    return False
                policy = bidding_policy_config
                if policy.policy == "learned":
                    policy = BiddingPolicyConfig(
                        policy=policy.warm_start_policy,
                        warm_start_policy=policy.warm_start_policy,
                    )
                bid, source = select_bidding_action(
                    bid_obs, policy, action_rng
                )
                apply_bidding_action(game, bid_obs, bid, source)
                _check_actor_step_limit(game["steps"], max_steps)
                continue
            position = game["env"]._acting_player_position
            infoset = game["env"].infoset
            legal_actions = infoset.legal_actions
            if len(legal_actions) == 1:
                if apply_action(game, 0, None, position, legal_actions):
                    return True
                continue

            obs = get_obs_v2(infoset, ruleset=ruleset)
            action_rng = game["action_rng"]
            if (
                runtime_kind == "v2"
                and epsilon > 0
                and action_rng.random() < epsilon
            ):
                action_index = action_rng.randrange(len(legal_actions))
                if apply_action(
                    game, action_index, obs, position, legal_actions
                ):
                    return True
                continue

            bundle = observation_to_model_inputs(
                obs,
                strategy_config,
                style_enabled=style_enabled,
            )
            slot_id = coordinator.acquire(actor_id)
            coordinator.slots.write(slot_id, bundle)
            belief_sidecar = None
            if belief_async:
                public_belief_input = build_belief_input(obs.public)
                coordinator.belief_inputs.write(slot_id, public_belief_input)
                privileged = PrivilegedObservation(
                    all_handcards=dict(infoset.all_handcards),
                    acting_role=position,
                )
                belief_sidecar = build_v3_h4_belief_sidecar(
                    obs, privileged, public_inputs=bundle
                )
            request_id += 1
            coordinator.submit(
                slot_id,
                request_id=request_id,
                policy_snapshot=game["snapshot"],
            )
            game["pending"] = (
                "card_play",
                slot_id,
                request_id,
                obs,
                position,
                legal_actions,
                belief_sidecar,
                bundle,
            )
            return False

    def resolve_request(game):
        pending = game["pending"]
        if pending[0] == "bidding":
            _, slot_id, pending_id, bid_obs = pending
            blocked_started = time.monotonic()
            coordinator.wait_done(slot_id, pending_id)
            game["blocked_seconds"] += time.monotonic() - blocked_started
            logits = coordinator.bidding_inputs.output_logits[slot_id].clone()
            legal_mask = coordinator.bidding_inputs.legal_mask[slot_id].clone()
            coordinator.release(slot_id)
            game["pending"] = None
            bid = int(
                torch.argmax(
                    logits.masked_fill(~legal_mask, float("-inf"))
                ).item()
            )
            return ("bidding", bid, bid_obs)
        (
            _kind,
            slot_id,
            pending_id,
            obs,
            position,
            legal_actions,
            belief_sidecar,
            public_inputs,
        ) = game["pending"]
        blocked_started = time.monotonic()
        coordinator.wait_done(slot_id, pending_id)
        game["blocked_seconds"] += time.monotonic() - blocked_started
        count = int(coordinator.slots.action_counts[slot_id])
        mask = coordinator.slots.action_mask[slot_id, :count].clone()
        packed = coordinator.slots.output_values[slot_id, :count].clone()
        coordinator.release(slot_id)
        game["pending"] = None
        output = ModelOutput(
            win_logit=packed[:, 0:1],
            score_if_win=packed[:, 1:2],
            score_if_loss=packed[:, 2:3],
            p_win=packed[:, 3:4],
            score_mean=packed[:, 4:5],
            action_mask=mask,
        )
        if runtime_kind == "v3_hybrid":
            if packed.shape[1] != 6:
                raise RuntimeError("V3 async response is missing dmc_q")
            q_values = packed[:, 5].masked_fill(~mask, float("-inf"))
            valid_indices = torch.nonzero(mask, as_tuple=False).flatten().tolist()
            action_index = (
                int(game["action_rng"].choice(valid_indices))
                if epsilon > 0 and game["action_rng"].random() < epsilon
                else int(torch.argmax(q_values).item())
            )
            q_old = float(q_values[action_index].item())
        else:
            action_index = select_action(output, decision_config)
            q_old = None
        return (
            "card_play",
            action_index,
            obs,
            position,
            legal_actions,
            q_old,
            belief_sidecar,
            public_inputs,
        )

    try:
        active = []
        while True:
            if coordinator.shutdown_event.is_set():
                return
            coordinator._raise_if_failed()
            while len(active) < games_per_actor:
                try:
                    task = (
                        task_queue.get(timeout=0.1)
                        if not active else task_queue.get_nowait()
                    )
                except queue.Empty:
                    break
                if task is None:
                    return
                active.append(start_game(task))
            if not active:
                continue

            completed = []
            for game in tuple(active):
                if advance_until_request_or_done(game):
                    completed.append(game)
            for game in completed:
                active.remove(game)

            pending_games = [
                game for game in active if game["pending"] is not None
            ]
            # All games submit before this actor waits.  Responses are cloned
            # and slots released for the whole wave before terminal replay
            # publication can block on replay capacity.
            resolved = [
                (game, *resolve_request(game)) for game in pending_games
            ]
            for item in resolved:
                game, kind, *payload = item
                if kind == "bidding":
                    bid, bid_obs = payload
                    apply_bidding_action(game, bid_obs, bid, "learned")
                    _check_actor_step_limit(game["steps"], max_steps)
                    continue
                (
                    action_index,
                    obs,
                    position,
                    legal_actions,
                    q_old,
                    belief_sidecar,
                    public_inputs,
                ) = payload
                if apply_action(
                    game,
                    action_index,
                    obs,
                    position,
                    legal_actions,
                    q_old=q_old,
                    belief_sidecar=belief_sidecar,
                    public_inputs=public_inputs,
                ):
                    active.remove(game)
    except BaseException as exc:
        if coordinator.shutdown_event.is_set() and not coordinator.abort_event.is_set():
            event_queue.put(("stopped", actor_id))
            return
        message = f"actor {actor_id}: {type(exc).__name__}: {exc}"
        coordinator.fail(message)
        event_queue.put(("failed", actor_id, message))
        raise
