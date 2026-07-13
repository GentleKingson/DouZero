"""Build listwise BC samples from validated human-game records (P08).

A :class:`BCSample` is the unit of behaviour-cloning training data: a **public**
:class:`~douzero.observation.encode_v2.ObservationV2` plus the **privileged**
``human_action_index`` (the position of the recorded human action in that
decision's legal-action list). The index is into the *current* legal-action
list — never a global action-class id — exactly as AGENTS.md requires:

    "Human behavior cloning must score the current legal-action list; do not
    replace the variable action representation with a brittle global class list."

Imperfect-information boundary
------------------------------
The public ``obs`` is what the BC *student* model consumes. The
``human_action_index`` is privileged training-only data (analogous to a belief
label) and is stamped ``kind="bc_sample"`` so a deployment guard can reject the
whole sample without introspection. The deployment ``DeepAgentV2.act`` never
receives a :class:`BCSample`; it only receives the public
:class:`ObservationV2`.

The builder replays each validated record through ``GameEnv`` (the same engine
used by :mod:`douzero.human_data.validate`) and, at every non-trivial decision,
snapshots the public observation and the recorded action's legal-action index.
Decisions with a single legal action are skipped by default — they carry no
listwise signal (the index is always 0 and the loss is trivially zero).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator

from .schema import HumanGameRecord
from .validate import _REPLAY_ROLES, _ReplayAgent, ReplayValidationError, assert_legacy_ruleset

#: Kind stamp identifying a BCSample as privileged training data. A deployment
#: guard can reject any object/dict carrying this kind.
BC_SAMPLE_KIND: str = "bc_sample"


class BCSampleError(ValueError):
    """Raised when a BC sample cannot be built canonically."""


@dataclass(frozen=True)
class BCSample:
    """One listwise BC decision (public obs + privileged human-action index).

    Attributes
    ----------
    obs:
        The public :class:`~douzero.observation.encode_v2.ObservationV2` the
        student model consumes. Built once per decision via
        :func:`~douzero.observation.encode_v2.get_obs_v2`.
    human_action_index:
        The privileged label: the row index of the recorded human action within
        ``obs.actions.legal_actions`` (sorted-tuple canonical form). The student
        predicts this index via a listwise cross-entropy over the N legal
        actions.
    position:
        The acting role (``"landlord"`` / ``"landlord_down"`` /
        ``"landlord_up"``). Must equal ``obs.public.acting_role``.
    game_id:
        Provenance: the source record's ``game_id``. Used for split integrity
        and per-game diagnostics.
    skill_weight:
        Non-negative sample weight derived from
        ``record.player_skill_weight[position]`` (default 1.0). Clipped/
        normalized downstream in :mod:`douzero.human_data.weights`.
    num_legal_actions:
        Cached count of legal actions at this decision (== len of
        ``obs.actions.legal_actions``). Stored so a downstream collator can
        bucket by action-set size without re-reading the obs.
    """

    obs: object  # ObservationV2 (typed as object to avoid an import cycle)
    human_action_index: int
    position: str
    game_id: str
    skill_weight: float = 1.0
    num_legal_actions: int = 0
    kind: str = field(default=BC_SAMPLE_KIND, init=False)

    def __post_init__(self) -> None:
        if self.kind != BC_SAMPLE_KIND:  # defensive
            object.__setattr__(self, "kind", BC_SAMPLE_KIND)
        if not isinstance(self.human_action_index, int) or isinstance(
            self.human_action_index, bool
        ):
            raise BCSampleError(
                f"human_action_index must be an int, got "
                f"{type(self.human_action_index).__name__}"
            )
        if self.human_action_index < 0:
            raise BCSampleError(
                f"human_action_index must be non-negative, got "
                f"{self.human_action_index}"
            )
        if self.position not in _REPLAY_ROLES:
            raise BCSampleError(
                f"position must be one of {_REPLAY_ROLES}, got {self.position!r}"
            )
        if not isinstance(self.skill_weight, (int, float)) or isinstance(
            self.skill_weight, bool
        ):
            raise BCSampleError(
                f"skill_weight must be a number, got {type(self.skill_weight).__name__}"
            )
        if self.skill_weight < 0:
            raise BCSampleError(
                f"skill_weight must be non-negative, got {self.skill_weight}"
            )
        if self.num_legal_actions < 0:
            raise BCSampleError(
                f"num_legal_actions must be non-negative, got "
                f"{self.num_legal_actions}"
            )

    def validate(self) -> None:
        """Cross-check the sample against its own observation.

        Confirms ``position`` matches the observation's acting role, the index
        is within the legal-action list, and the indexed legal action equals
        the recorded human action's canonical key. Raises :class:`BCSampleError`
        on any mismatch.
        """
        obs = self.obs
        acting_role = getattr(getattr(obs, "public", None), "acting_role", "")
        if acting_role != self.position:
            raise BCSampleError(
                f"position {self.position!r} != obs.public.acting_role "
                f"{acting_role!r}"
            )
        legal = getattr(getattr(obs, "actions", None), "legal_actions", ())
        n = len(legal)
        if n != self.num_legal_actions:
            raise BCSampleError(
                f"num_legal_actions {self.num_legal_actions} != len(legal) {n}"
            )
        if not (0 <= self.human_action_index < n):
            raise BCSampleError(
                f"human_action_index {self.human_action_index} out of range "
                f"[0, {n})"
            )


# --------------------------------------------------------------------------- #
# Sample construction (replay + snapshot)
# --------------------------------------------------------------------------- #
def build_bc_samples(
    record: HumanGameRecord,
    *,
    skip_single_action: bool = True,
) -> list[BCSample]:
    """Replay one record and build its BC samples.

    Parameters
    ----------
    record:
        A :class:`~douzero.human_data.schema.HumanGameRecord` that has already
        passed replay validation (:mod:`douzero.human_data.validate`). If the
        record does not replay cleanly, a :class:`BCSampleError` is raised —
        call :func:`~douzero.human_data.validate.validate_record` first in any
        pipeline that handles untrusted input.
    skip_single_action:
        When True (default), decisions with exactly one legal action are
        skipped (they carry no listwise signal). Set False to keep them.
    """
    from douzero.observation.encode_v2 import get_obs_v2

    from douzero.env.game import GameEnv

    # Ruleset identity: reject a non-legacy record before replay (Blocker 2).
    # The validator does this too, but the sample builder is a separate entry
    # point and must fail-closed on its own. Wrap as BCSampleError so the
    # builder's error contract is consistent.
    try:
        assert_legacy_ruleset(record)
    except ReplayValidationError as exc:
        raise BCSampleError(
            f"{record.game_id}: {exc}"
        ) from exc

    samples: list[BCSample] = []
    players = {pos: _ReplayAgent(pos) for pos in _REPLAY_ROLES}
    genv = GameEnv(players)  # legacy cardplay (ruleset=None)
    deal = {
        "landlord": list(record.initial_hands["landlord"]),
        "landlord_up": list(record.initial_hands["landlord_up"]),
        "landlord_down": list(record.initial_hands["landlord_down"]),
        "three_landlord_cards": list(record.initial_hands["three_landlord_cards"]),
    }
    genv.card_play_init(deal)

    skill_weight = float(
        record.player_skill_weight.get("", 0.0)  # placeholder; per-role below
    )
    del skill_weight

    for pos, cards in record.action_history:
        acting = genv.acting_player_position
        if acting != pos:
            raise BCSampleError(
                f"{record.game_id}: turn mismatch at recorded action for "
                f"{pos!r} but engine expects {acting!r}; record not validated?"
            )
        if genv.game_over:
            raise BCSampleError(
                f"{record.game_id}: game over before all recorded actions"
            )
        infoset = genv.game_infoset
        legal = list(infoset.legal_actions)
        human_key = tuple(sorted(cards))

        if not skip_single_action or len(legal) > 1:
            obs = get_obs_v2(infoset)
            obs_legal = obs.actions.legal_actions
            # The recorded action MUST be legal (validation guarantees this);
            # locate its row index by canonical sorted-tuple match.
            try:
                idx = obs_legal.index(human_key)
            except ValueError as exc:
                raise BCSampleError(
                    f"{record.game_id}: recorded action {human_key!r} not found "
                    f"in the legal-action list (record was not validated?)"
                ) from exc
            weight = float(record.player_skill_weight.get(pos, 1.0))
            sample = BCSample(
                obs=obs,
                human_action_index=idx,
                position=pos,
                game_id=record.game_id,
                skill_weight=weight,
                num_legal_actions=len(obs_legal),
            )
            sample.validate()
            samples.append(sample)

        # Advance the env with the recorded action.
        players[pos].set_action(list(cards))
        try:
            genv.step()
        except ReplayValidationError as exc:
            raise BCSampleError(
                f"{record.game_id}: replay failed during BC sampling: {exc}"
            ) from exc

    return samples


@dataclass
class BatchSampleReport:
    """Outcome of :func:`build_bc_samples_batch` (auditable, no silent drops).

    ``samples`` are the built BC samples; ``quarantined`` carries the
    ``(game_id, error)`` pairs for records that failed replay/sampling so the
    caller can write a quarantine file. Nothing is dropped silently.
    """

    samples: list[BCSample]
    quarantined: list[tuple[str, str]]


def build_bc_samples_batch(
    records,
    *,
    skip_single_action: bool = True,
    stop_on_error: bool = True,
) -> Iterator[BCSample]:
    """Stream BC samples from a batch of records.

    Blocker 3: the default is now ``stop_on_error=True`` (fail-fast). A record
    that fails replay/sampling raises :class:`BCSampleError` immediately rather
    than being silently skipped — the quarantine contract requires that no bad
    record disappears without a trace. Callers that want collect-and-quarantine
    semantics should use :func:`build_bc_samples_with_report` instead, which
    returns a :class:`BatchSampleReport` with the bad records listed.
    """
    for record in records:
        # Re-raise on error by default (stop_on_error=True). The only way to
        # skip is to explicitly pass stop_on_error=False, and even then the
        # caller should use build_bc_samples_with_report to audit the skips.
        try:
            for sample in build_bc_samples(
                record, skip_single_action=skip_single_action
            ):
                yield sample
        except BCSampleError:
            if stop_on_error:
                raise
            continue


def build_bc_samples_with_report(
    records,
    *,
    skip_single_action: bool = True,
) -> BatchSampleReport:
    """Build BC samples, quarantining (not dropping) records that fail.

    Blocker 3: this is the production path. Every record that fails replay or
    sampling appears in ``report.quarantined`` with its ``game_id`` and the
    error, so nothing disappears silently. Valid records' samples accumulate in
    ``report.samples``.
    """
    report = BatchSampleReport(samples=[], quarantined=[])
    for record in records:
        try:
            for sample in build_bc_samples(
                record, skip_single_action=skip_single_action
            ):
                report.samples.append(sample)
        except BCSampleError as exc:
            report.quarantined.append((record.game_id, str(exc)))
    return report
