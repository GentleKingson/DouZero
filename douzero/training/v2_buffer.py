"""Per-episode V2 transition store for the single-process trainer (P06).

The legacy multiprocessing buffer (``douzero/dmc/utils.py``) is keyed by six
fixed-shape tensors per decision, which cannot represent the V2 model's
variable-N-action decisions. Rewriting it would overlap with P14's
high-throughput work and risk regressing the legacy path.

This module provides a minimal, single-process transition store that:

- Records one :class:`~douzero.observation.encode_v2.ObservationV2` per
  decision along with the action index the actor chose and the acting
  position.
- At terminal, looks up the team-perspective labels from the env's
  ``info['team_targets']`` (populated by
  :meth:`~douzero.env.env.Env._attach_team_perspective_labels`) and attaches
  them to every transition of the same position in that episode. This is
  the Monte-Carlo return: every decision the landlord made in a game the
  landlord ultimately won gets ``target_win = 1`` and the same final
  ``target_score``; ditto for farmers.
- Exposes :meth:`V2ReplayBuffer.sample_minibatch` which returns ``B``
  transitions (with their labels) for the trainer's forward+backward pass.

This is intentionally NOT a high-throughput data structure. It exists to
satisfy the P06 acceptance criterion ("极短训练能完成一次优化且参数变化")
on CPU while preserving the legacy path. P14 will introduce the
multiprocessing/shmem equivalent.
"""

from __future__ import annotations

import math
import random
from collections import deque
from dataclasses import dataclass, field
from typing import Iterable

import torch

from douzero.observation.encode_v2 import ObservationV2
from douzero.models_v2.batch import ModelInputBundle, observation_to_model_inputs

COMPACT_REPLAY_SCHEMA_VERSION = 1
ACTION_BUCKET_LIMITS: tuple[int, ...] = (8, 16, 32, 64, 128)


def action_count_bucket(action_count: int) -> int | str:
    """Return the smallest configured action bucket, or ``"overflow"``."""
    if isinstance(action_count, bool) or not isinstance(action_count, int):
        raise TypeError("action_count must be an int")
    if action_count <= 0:
        raise ValueError("action_count must be positive")
    for limit in ACTION_BUCKET_LIMITS:
        if action_count <= limit:
            return limit
    return "overflow"

#: Valid acting positions (mirrors douzero.env.scoring.ALL_POSITIONS).
_VALID_POSITIONS: frozenset[str] = frozenset(
    {"landlord", "landlord_up", "landlord_down"}
)


@dataclass
class Transition:
    """One decision in a V2 self-play episode.

    ``obs`` is the public :class:`ObservationV2` captured at the decision
    point. ``action_index`` is the row index (into
    ``obs.actions.features``) the actor took. ``position`` is the acting
    role. The four ``target_*`` fields are filled in at episode terminal
    from the team-perspective label (Monte-Carlo return).
    """

    obs: ObservationV2
    action_index: int
    position: str
    target_win: float = float("nan")
    target_score: float = float("nan")
    target_log_score: float = float("nan")
    target_min_turns_after: float = float("nan")
    target_min_turns_exact_mask: float = float("nan")
    target_regain_initiative: float = float("nan")
    target_teammate_finish: float = float("nan")
    target_teammate_finish_mask: float = float("nan")
    target_spring_probability: float = float("nan")
    target_structure_cost: float = float("nan")
    # Index in Episode.action_trace. -1 is accepted for pre-P09/manual tests.
    trace_index: int = -1
    # P11 provenance. Empty values preserve pre-P11/manual transitions.
    policy_id: str = ""
    teammate_policy_id: str | None = None
    # Exact inference snapshot provenance.  ``policy_id`` remains the league
    # identity; these fields identify the immutable snapshot used throughout
    # the game and survive compact replay serialization.
    policy_version: str = ""
    policy_step: int = -1

    def has_labels(self) -> bool:
        """Quick NaN check for the label-stamping loop.

        :meth:`validate` is the comprehensive check (finiteness, binary,
        action-index range, position); this method is a cheaper pre-check
        used to decide whether :meth:`Episode.label_from_terminal` should
        be called.
        """
        for value in (self.target_win, self.target_score, self.target_log_score):
            if isinstance(value, float) and math.isnan(value):
                return False
        return True

    def validate(self) -> None:
        """Comprehensive integrity check (P06 r4).

        Raises :class:`ValueError` / :class:`TypeError` for:
        - ``position`` not in the three valid roles;
        - ``position`` does not match ``obs.public.acting_role`` (P06 r6);
        - ``action_index`` is a bool, not an int, or outside
          ``[0, len(obs.actions.legal_actions))``;
        - ``target_win`` non-finite or not in ``{0.0, 1.0}``;
        - ``target_score`` / ``target_log_score`` non-finite.

        This is called at buffer entry (:meth:`V2ReplayBuffer.add_episode`)
        so a corrupted transition is caught immediately rather than
        bypassing the non-finite-loss guard downstream (e.g.
        ``target_score=+inf`` clamped to ``score_clamp`` produces a finite
        loss, silently training on corrupted data).
        """
        if self.position not in _VALID_POSITIONS:
            raise ValueError(
                f"Transition.position must be one of {sorted(_VALID_POSITIONS)}, "
                f"got {self.position!r}"
            )
        # P06 r6: the position that determines the team-perspective label
        # must match the acting role encoded in the observation. Without
        # this check, a transition could carry a farmer observation with a
        # landlord position: the model sees a farmer view, but the terminal
        # label is stamped from the landlord's team target — a silent
        # label/observation mismatch that poisons training without producing
        # any non-finite value.
        obs_acting_role = self.obs.public.acting_role
        if self.position != obs_acting_role:
            raise ValueError(
                f"Transition.position ({self.position!r}) does not match the "
                f"observation's acting_role ({obs_acting_role!r}). The "
                f"position that selects the team-perspective label must "
                f"match the role encoded in the observation, or the model "
                f"trains on a farmer view with a landlord label (or vice "
                f"versa)."
            )
        # bool is a subclass of int; reject it so True/False do not pass as
        # an action index.
        if isinstance(self.action_index, bool) or not isinstance(self.action_index, int):
            raise TypeError(
                f"Transition.action_index must be int, got "
                f"{type(self.action_index).__name__}: {self.action_index!r}"
            )
        if isinstance(self.trace_index, bool) or not isinstance(self.trace_index, int):
            raise TypeError("Transition.trace_index must be int")
        if self.trace_index < -1:
            raise ValueError("Transition.trace_index must be -1 or non-negative")
        if not isinstance(self.policy_version, str):
            raise TypeError("Transition.policy_version must be a string")
        if isinstance(self.policy_step, bool) or not isinstance(self.policy_step, int):
            raise TypeError("Transition.policy_step must be an int")
        if self.policy_step < -1:
            raise ValueError("Transition.policy_step must be -1 or non-negative")
        n_actions = len(self.obs.actions.legal_actions)
        if not (0 <= self.action_index < n_actions):
            raise ValueError(
                f"Transition.action_index {self.action_index} is outside the "
                f"observation's legal-action range [0, {n_actions})."
            )
        for name, value in (
            ("target_win", self.target_win),
            ("target_score", self.target_score),
            ("target_log_score", self.target_log_score),
        ):
            if not math.isfinite(value):
                raise ValueError(
                    f"Transition.{name} must be finite, got {value!r}"
                )
        if self.target_win not in (0.0, 1.0):
            raise ValueError(
                f"Transition.target_win must be 0.0 or 1.0, got {self.target_win!r}"
            )
        aux_values = (
            self.target_min_turns_after,
            self.target_min_turns_exact_mask,
            self.target_regain_initiative,
            self.target_teammate_finish,
            self.target_teammate_finish_mask,
            self.target_spring_probability,
            self.target_structure_cost,
        )
        if any(math.isfinite(value) for value in aux_values):
            if not all(math.isfinite(value) for value in aux_values):
                raise ValueError("Transition strategy auxiliary labels are partially populated")
            for name in (
                "target_min_turns_exact_mask", "target_regain_initiative", "target_teammate_finish",
                "target_teammate_finish_mask", "target_spring_probability",
            ):
                if getattr(self, name) not in (0.0, 1.0):
                    raise ValueError(f"Transition.{name} must be binary")
            if self.target_min_turns_after < 0 or self.target_structure_cost < 0:
                raise ValueError("strategy regression labels must be non-negative")


@dataclass
class CompactTensorTransition:
    """Versioned CPU-tensor replay record for one card-play decision.

    It stores every public tensor consumed by ``ModelV2`` plus every learner
    label and provenance field.  Legal-action descriptors are intentionally
    absent: training consumes the versioned action feature rows, while bidding
    remains in its independent replay type.
    """

    model_inputs: ModelInputBundle
    action_index: int
    position: str
    targets: dict[str, float]
    trace_index: int
    policy_id: str
    teammate_policy_id: str | None
    policy_version: str
    policy_step: int
    schema_version: int = COMPACT_REPLAY_SCHEMA_VERSION

    @classmethod
    def from_transition(
        cls,
        transition: Transition,
        *,
        strategy_config=None,
        style_enabled: bool = False,
    ) -> "CompactTensorTransition":
        transition.validate()
        bundle = observation_to_model_inputs(
            transition.obs,
            strategy_config=strategy_config,
            style_enabled=style_enabled,
        )
        bundle = ModelInputBundle(
            state_card_vectors=tuple(value.to(torch.int8) for value in bundle.state_card_vectors),
            state_context_flat=bundle.state_context_flat.to(torch.int8),
            context_card_vectors=tuple(value.to(torch.int8) for value in bundle.context_card_vectors),
            context_flat=bundle.context_flat.to(torch.int32),
            history_tokens=bundle.history_tokens.to(torch.int8),
            history_key_padding_mask=bundle.history_key_padding_mask,
            action_features=bundle.action_features.to(torch.int8),
            action_mask=bundle.action_mask,
            acting_role=bundle.acting_role,
            feature_schema_hash=bundle.feature_schema_hash,
            strategy_features=bundle.strategy_features,
            style_features=bundle.style_features,
        )
        # Compact records are always detached CPU tensors.  Integral public
        # encodings use their narrow source dtype where practical; model input
        # conversion remains centralized in the batch bridge.
        for tensor in (
            *bundle.state_card_vectors,
            bundle.state_context_flat,
            *bundle.context_card_vectors,
            bundle.context_flat,
            bundle.history_tokens,
            bundle.history_key_padding_mask,
            bundle.action_features,
            bundle.action_mask,
        ):
            if tensor.device.type != "cpu":
                raise ValueError("compact replay accepts CPU model inputs only")
        target_names = (
            "target_win", "target_score", "target_log_score",
            "target_min_turns_after", "target_min_turns_exact_mask",
            "target_regain_initiative", "target_teammate_finish",
            "target_teammate_finish_mask", "target_spring_probability",
            "target_structure_cost",
        )
        return cls(
            model_inputs=bundle,
            action_index=transition.action_index,
            position=transition.position,
            targets={name: float(getattr(transition, name)) for name in target_names},
            trace_index=transition.trace_index,
            policy_id=transition.policy_id,
            teammate_policy_id=transition.teammate_policy_id,
            policy_version=transition.policy_version,
            policy_step=transition.policy_step,
        )

    @property
    def action_count(self) -> int:
        return int(self.model_inputs.action_features.shape[0])

    @property
    def bucket(self) -> int | str:
        return action_count_bucket(self.action_count)

    def state_dict(self) -> dict:
        bundle = self.model_inputs
        tensor_fields = {
            "state_card_vectors": bundle.state_card_vectors,
            "state_context_flat": bundle.state_context_flat,
            "context_card_vectors": bundle.context_card_vectors,
            "context_flat": bundle.context_flat,
            "history_tokens": bundle.history_tokens,
            "history_key_padding_mask": bundle.history_key_padding_mask,
            "action_features": bundle.action_features,
            "action_mask": bundle.action_mask,
            "strategy_features": bundle.strategy_features,
            "style_features": bundle.style_features,
        }
        return {
            "schema_version": self.schema_version,
            "model_inputs": tensor_fields,
            "acting_role": bundle.acting_role,
            "feature_schema_hash": bundle.feature_schema_hash,
            "action_index": self.action_index,
            "position": self.position,
            "targets": dict(self.targets),
            "trace_index": self.trace_index,
            "policy_id": self.policy_id,
            "teammate_policy_id": self.teammate_policy_id,
            "policy_version": self.policy_version,
            "policy_step": self.policy_step,
        }

    @classmethod
    def from_state_dict(cls, raw: dict) -> "CompactTensorTransition":
        if not isinstance(raw, dict) or raw.get("schema_version") != COMPACT_REPLAY_SCHEMA_VERSION:
            raise ValueError("unknown compact replay schema_version")
        values = raw["model_inputs"]
        bundle = ModelInputBundle(
            state_card_vectors=tuple(values["state_card_vectors"]),
            state_context_flat=values["state_context_flat"],
            context_card_vectors=tuple(values["context_card_vectors"]),
            context_flat=values["context_flat"],
            history_tokens=values["history_tokens"],
            history_key_padding_mask=values["history_key_padding_mask"],
            action_features=values["action_features"],
            action_mask=values["action_mask"],
            acting_role=raw["acting_role"],
            feature_schema_hash=raw["feature_schema_hash"],
            strategy_features=values.get("strategy_features"),
            style_features=values.get("style_features"),
        )
        return cls(
            model_inputs=bundle,
            action_index=int(raw["action_index"]),
            position=str(raw["position"]),
            targets={str(k): float(v) for k, v in raw["targets"].items()},
            trace_index=int(raw["trace_index"]),
            policy_id=str(raw["policy_id"]),
            teammate_policy_id=raw["teammate_policy_id"],
            policy_version=str(raw["policy_version"]),
            policy_step=int(raw["policy_step"]),
        )


@dataclass
class Episode:
    """All transitions of one game, plus the terminal result dict."""

    transitions: list[Transition] = field(default_factory=list)
    # P17 bidding observations use neutral seats; terminal labelling adds the
    # resolved actor role for policy credit and landlord-side value targets.
    bidding_transitions: list[object] = field(default_factory=list)
    redeal_count: int = 0
    max_redeals_exceeded: bool = False
    excluded_from_training: bool = False
    exclusion_reason: str = ""
    abandoned_bidding_transitions: int = 0
    terminal_result: dict = field(default_factory=dict)
    # Complete public action trace, including forced single-action decisions.
    # P09 trajectory labels use it so spring/finisher targets are not biased by
    # the replay buffer's deliberate omission of trivial decisions.
    action_trace: list[tuple[str, tuple[int, ...]]] = field(default_factory=list)
    policy_ids_by_seat: dict[str, str] = field(default_factory=dict)
    learner_controlled_seats: tuple[str, ...] = ()
    # P12 policy identity frozen before the opening is sampled. Coach labels
    # must use this snapshot rather than mutable trainer counters at terminal.
    policy_version_at_start: str = ""
    policy_step_at_start: int = -1

    def label_from_terminal(self) -> None:
        """Apply per-position team-perspective labels to every transition.

        Looks up ``self.terminal_result['team_targets']`` (populated by the
        env at terminal) and stamps each transition's labels based on its
        ``position``. Idempotent.
        """
        team_targets = self.terminal_result.get("team_targets")
        if team_targets is None:
            raise ValueError(
                "Episode.terminal_result is missing 'team_targets'; the env "
                "must populate it at terminal (see Env._attach_team_perspective_labels)."
            )
        for tr in self.transitions:
            labels = team_targets[tr.position]
            tr.target_win = float(labels["target_win"])
            tr.target_score = float(labels["target_score"])
            tr.target_log_score = float(labels["target_log_score"])

    def label_strategy_auxiliary(self, *, node_budget: int, time_budget_ms: int) -> None:
        """Stamp direct and future-trajectory P09 targets on every transition."""

        from douzero.strategy.hand_decomposition import hand_decomposition
        from douzero.strategy.structure import action_structure_cost

        winner_position = str(self.terminal_result.get("winner_position", ""))
        winner_team = str(self.terminal_result.get("winner_team", ""))
        if winner_position in ("", "farmer") and self.action_trace:
            winner_position = self.action_trace[-1][0]
        ruleset_id = str(self.terminal_result.get("ruleset_id", ""))
        if ruleset_id != "legacy" and (
            "spring" in self.terminal_result
            or "anti_spring" in self.terminal_result
        ):
            spring = float(
                bool(self.terminal_result.get("spring", False))
                or bool(self.terminal_result.get("anti_spring", False))
            )
        else:
            non_pass_by_role = {role: 0 for role in _VALID_POSITIONS}
            trace = self.action_trace or [
                (
                    transition.position,
                    transition.obs.actions.legal_actions[transition.action_index],
                )
                for transition in self.transitions
            ]
            for position, action in trace:
                if action:
                    non_pass_by_role[position] += 1
            spring = float(
                (winner_team == "landlord" and non_pass_by_role["landlord_up"] == 0
                 and non_pass_by_role["landlord_down"] == 0)
                or (winner_team == "farmer" and non_pass_by_role["landlord"] <= 1)
            )

        def team(role: str) -> str:
            return "landlord" if role == "landlord" else "farmer"

        for index, transition in enumerate(self.transitions):
            action = transition.obs.actions.legal_actions[transition.action_index]
            remaining = list(transition.obs.public.my_handcards)
            for card in action:
                remaining.remove(card)
            decomposition = hand_decomposition(
                remaining,
                node_budget=node_budget,
                time_budget_ms=time_budget_ms,
            )
            transition.target_min_turns_after = float(decomposition.min_turns)
            transition.target_min_turns_exact_mask = float(decomposition.exact)
            transition.target_structure_cost = action_structure_cost(
                transition.obs.public.my_handcards, action
            ).total
            transition.target_spring_probability = spring
            transition.target_teammate_finish_mask = float(
                transition.position != "landlord"
            )
            transition.target_teammate_finish = float(
                transition.position != "landlord"
                and winner_team == "farmer"
                and winner_position not in ("", transition.position)
            )
            transition.target_regain_initiative = 0.0
            if transition.trace_index >= 0 and self.action_trace:
                for trace_index in range(transition.trace_index + 1, len(self.action_trace)):
                    future_position, future_action = self.action_trace[trace_index]
                    prior_two_passed = (
                        trace_index >= 2
                        and not self.action_trace[trace_index - 1][1]
                        and not self.action_trace[trace_index - 2][1]
                    )
                    if (
                        future_action
                        and prior_two_passed
                        and team(future_position) == team(transition.position)
                    ):
                        transition.target_regain_initiative = 1.0
                        break
            else:
                for future in self.transitions[index + 1:]:
                    if team(future.position) != team(transition.position):
                        continue
                    future_action = future.obs.actions.legal_actions[future.action_index]
                    if future_action and not future.obs.public.last_move:
                        transition.target_regain_initiative = 1.0
                        break


@dataclass
class Minibatch:
    """A minibatch of ``B`` labelled transitions.

    Tensors are built by the trainer from ``observations`` (which are run
    through the model one decision at a time, since :meth:`ModelV2.forward`
    handles a single decision). The four label tensors are stacked here so
    the loss module can consume them directly.
    """

    observations: list[ObservationV2]
    action_indices: torch.Tensor  # (B,) long
    target_win: torch.Tensor  # (B,) float
    target_score: torch.Tensor  # (B,) float
    target_log_score: torch.Tensor  # (B,) float
    target_min_turns_after: torch.Tensor | None = None
    target_min_turns_exact_mask: torch.Tensor | None = None
    target_regain_initiative: torch.Tensor | None = None
    target_teammate_finish: torch.Tensor | None = None
    target_teammate_finish_mask: torch.Tensor | None = None
    target_spring_probability: torch.Tensor | None = None
    target_structure_cost: torch.Tensor | None = None
    model_inputs: list[ModelInputBundle] | None = None

    @property
    def batch_size(self) -> int:
        return int(self.action_indices.shape[0])

    def validate(self) -> None:
        """Check batch-length consistency across all fields (P06 r4).

        Ensures the observations list, the action_indices tensor, and the
        three label tensors all have the same length ``B``, so the trainer's
        per-decision gather loop cannot index out of bounds.
        """
        b_obs = len(self.model_inputs) if self.model_inputs is not None else len(self.observations)
        b_act = int(self.action_indices.shape[0])
        b_win = int(self.target_win.shape[0])
        b_score = int(self.target_score.shape[0])
        b_log = int(self.target_log_score.shape[0])
        lengths = {
            "observations": b_obs,
            "action_indices": b_act,
            "target_win": b_win,
            "target_score": b_score,
            "target_log_score": b_log,
        }
        for name in (
            "target_min_turns_after", "target_min_turns_exact_mask",
            "target_regain_initiative",
            "target_teammate_finish", "target_teammate_finish_mask",
            "target_spring_probability", "target_structure_cost",
        ):
            value = getattr(self, name)
            if value is not None:
                lengths[name] = int(value.shape[0])
        if len(set(lengths.values())) != 1:
            raise ValueError(
                f"Minibatch batch lengths disagree: {lengths}"
            )


class V2ReplayBuffer:
    """A bounded, label-checked replay buffer for the V2 trainer.

    Stores completed, labelled :class:`Episode` objects and samples
    minibatches uniformly across all labelled transitions. The buffer is
    intentionally simple (``collections.deque`` of episodes); it is not
    shared-memory and not thread-safe. The single-process trainer owns it.
    """

    def __init__(self, capacity_transitions: int = 4096) -> None:
        if capacity_transitions <= 0:
            raise ValueError(
                f"capacity_transitions must be positive, got {capacity_transitions}"
            )
        self._capacity = int(capacity_transitions)
        self._episodes: deque[Episode] = deque()
        self._size = 0
        self._buckets: dict[int | str, list[Transition]] = {
            **{limit: [] for limit in ACTION_BUCKET_LIMITS},
            "overflow": [],
        }

    def __len__(self) -> int:
        return self._size

    def clear(self) -> None:
        """Drop replay at an explicit checkpoint-safe boundary."""
        self._episodes.clear()
        self._size = 0
        for transitions in self._buckets.values():
            transitions.clear()

    @property
    def capacity(self) -> int:
        return self._capacity

    def add_episode(self, episode: Episode) -> None:
        """Append a labelled episode, validating every transition (P06 r4).

        Each transition is :meth:`Transition.validate`-d so corrupted
        labels (e.g. ``target_win=2.0``, ``target_score=+inf``) or an
        out-of-range ``action_index`` is caught at buffer entry — before
        it can bypass the non-finite-loss guard downstream.
        """
        if not episode.transitions:
            return
        # Ensure labels are stamped.
        if any(not tr.has_labels() for tr in episode.transitions):
            episode.label_from_terminal()
        # Comprehensive validation: every transition must pass the
        # integrity check before entering the buffer.
        for tr in episode.transitions:
            tr.validate()
        self._episodes.append(episode)
        self._size += len(episode.transitions)
        while self._size > self._capacity and self._episodes:
            evicted = self._episodes.popleft()
            self._size -= len(evicted.transitions)
        self._rebuild_buckets()

    def _rebuild_buckets(self) -> None:
        for transitions in self._buckets.values():
            transitions.clear()
        for episode in self._episodes:
            for transition in episode.transitions:
                bucket = action_count_bucket(len(transition.obs.actions.legal_actions))
                self._buckets[bucket].append(transition)

    def bucket_occupancy(self) -> dict[int | str, int]:
        return {name: len(values) for name, values in self._buckets.items()}

    def extend(self, episodes: Iterable[Episode]) -> None:
        for ep in episodes:
            self.add_episode(ep)

    def sample_minibatch(
        self,
        batch_size: int,
        rng: random.Random | None = None,
    ) -> Minibatch | None:
        """Sample ``batch_size`` labelled transitions uniformly at random.

        Returns ``None`` if the buffer has fewer than ``batch_size``
        transitions. The trainer treats ``None`` as "not enough data yet".
        The returned :class:`Minibatch` is :meth:`Minibatch.validate`-d as
        a boundary defense.
        """
        if self._size < batch_size:
            return None
        rng = rng or random
        # Prefer one action-count bucket to minimize padding.  Selecting among
        # eligible buckets by occupancy preserves transition-level weighting;
        # if none can fill the batch, retain the established uniform fallback.
        eligible = [values for values in self._buckets.values() if len(values) >= batch_size]
        if eligible:
            total = sum(len(values) for values in eligible)
            needle = rng.randrange(total)
            selected = eligible[-1]
            for values in eligible:
                if needle < len(values):
                    selected = values
                    break
                needle -= len(values)
            picks = rng.sample(selected, batch_size)
        else:
            flat: list[Transition] = []
            for ep in self._episodes:
                flat.extend(ep.transitions)
            picks = rng.sample(flat, batch_size)
        has_aux = all(math.isfinite(p.target_min_turns_after) for p in picks)
        aux_tensor = lambda name: torch.tensor(
            [getattr(p, name) for p in picks], dtype=torch.float32
        ) if has_aux else None
        minibatch = Minibatch(
            observations=[p.obs for p in picks],
            action_indices=torch.tensor(
                [p.action_index for p in picks], dtype=torch.long
            ),
            target_win=torch.tensor([p.target_win for p in picks], dtype=torch.float32),
            target_score=torch.tensor(
                [p.target_score for p in picks], dtype=torch.float32
            ),
            target_log_score=torch.tensor(
                [p.target_log_score for p in picks], dtype=torch.float32
            ),
            target_min_turns_after=aux_tensor("target_min_turns_after"),
            target_min_turns_exact_mask=aux_tensor("target_min_turns_exact_mask"),
            target_regain_initiative=aux_tensor("target_regain_initiative"),
            target_teammate_finish=aux_tensor("target_teammate_finish"),
            target_teammate_finish_mask=aux_tensor("target_teammate_finish_mask"),
            target_spring_probability=aux_tensor("target_spring_probability"),
            target_structure_cost=aux_tensor("target_structure_cost"),
        )
        minibatch.validate()
        return minibatch


class CompactTensorReplayBuffer:
    """Bounded bucketed replay containing no environment Python objects."""

    def __init__(self, capacity_transitions: int = 4096) -> None:
        if capacity_transitions < 1:
            raise ValueError("capacity_transitions must be positive")
        self._capacity = int(capacity_transitions)
        self._records: deque[CompactTensorTransition] = deque()
        self._buckets: dict[int | str, list[CompactTensorTransition]] = {
            **{limit: [] for limit in ACTION_BUCKET_LIMITS}, "overflow": []
        }

    def __len__(self) -> int:
        return len(self._records)

    @property
    def capacity(self) -> int:
        return self._capacity

    def clear(self) -> None:
        self._records.clear()
        for records in self._buckets.values():
            records.clear()

    def add(self, record: CompactTensorTransition) -> None:
        if record.schema_version != COMPACT_REPLAY_SCHEMA_VERSION:
            raise ValueError("unknown compact replay schema version")
        self._records.append(record)
        if len(self._records) > self._capacity:
            self._records.popleft()
        self._rebuild_buckets()

    def _rebuild_buckets(self) -> None:
        for records in self._buckets.values():
            records.clear()
        for record in self._records:
            self._buckets[record.bucket].append(record)

    def bucket_occupancy(self) -> dict[int | str, int]:
        return {name: len(records) for name, records in self._buckets.items()}

    def sample_minibatch(
        self, batch_size: int, rng: random.Random | None = None
    ) -> Minibatch | None:
        if len(self) < batch_size:
            return None
        rng = rng or random
        eligible = [records for records in self._buckets.values() if len(records) >= batch_size]
        source = rng.choice(eligible) if eligible else list(self._records)
        picks = rng.sample(source, batch_size)
        target = lambda name: torch.tensor(
            [record.targets[name] for record in picks], dtype=torch.float32
        )
        has_aux = all(
            math.isfinite(record.targets["target_min_turns_after"])
            for record in picks
        )
        auxiliary = lambda name: target(name) if has_aux else None
        batch = Minibatch(
            observations=[],
            model_inputs=[record.model_inputs for record in picks],
            action_indices=torch.tensor(
                [record.action_index for record in picks], dtype=torch.long
            ),
            target_win=target("target_win"),
            target_score=target("target_score"),
            target_log_score=target("target_log_score"),
            target_min_turns_after=auxiliary("target_min_turns_after"),
            target_min_turns_exact_mask=auxiliary("target_min_turns_exact_mask"),
            target_regain_initiative=auxiliary("target_regain_initiative"),
            target_teammate_finish=auxiliary("target_teammate_finish"),
            target_teammate_finish_mask=auxiliary("target_teammate_finish_mask"),
            target_spring_probability=auxiliary("target_spring_probability"),
            target_structure_cost=auxiliary("target_structure_cost"),
        )
        batch.validate()
        return batch
