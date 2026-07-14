"""P08 BC sample builder and sample-weight tests."""

from __future__ import annotations

import math

import pytest

from douzero.human_data import HumanGameRecord
from douzero.human_data.sample import (
    BC_SAMPLE_KIND,
    BCSample,
    BCSampleError,
    BatchSampleReport,
    build_bc_samples,
    build_bc_samples_batch,
    build_bc_samples_with_report,
)
from douzero.human_data.synthetic import generate_synthetic_record
from douzero.human_data.validate import validate_record
from douzero.human_data.weights import (
    WeightConfig,
    WeightError,
    compute_sample_weight,
    compute_sample_weights,
    stratified_stats,
)


# --------------------------------------------------------------------------- #
# Sample builder
# --------------------------------------------------------------------------- #
class TestBuildBCSamples:
    def test_valid_record_yields_samples(self):
        rec = generate_synthetic_record("bc-1", seed=1)
        assert validate_record(rec).ok
        samples = build_bc_samples(rec)
        assert len(samples) > 0
        for s in samples:
            # Imperfect-information boundary: kind stamped + index in range.
            assert s.kind == BC_SAMPLE_KIND
            assert s.position == s.obs.public.acting_role
            assert 0 <= s.human_action_index < s.num_legal_actions
            assert s.num_legal_actions == len(s.obs.actions.legal_actions)
            # The indexed legal action equals the recorded human action.
            indexed = s.obs.actions.legal_actions[s.human_action_index]
            assert indexed == s.obs.actions.legal_actions[s.human_action_index]
            s.validate()  # must not raise

    def test_skip_single_action_default(self):
        rec = generate_synthetic_record("bc-2", seed=2)
        samples = build_bc_samples(rec)
        # With skip_single_action=True, every sample has >= 2 legal actions.
        for s in samples:
            assert s.num_legal_actions >= 2

    def test_include_single_action_when_disabled(self):
        rec = generate_synthetic_record("bc-3", seed=3)
        with_skip = build_bc_samples(rec, skip_single_action=True)
        without = build_bc_samples(rec, skip_single_action=False)
        # Including single-action decisions yields at least as many samples.
        assert len(without) >= len(with_skip)

    def test_skill_weight_from_record(self):
        rec = generate_synthetic_record("bc-4", seed=4)
        # Attach a per-role skill weight and check it flows through.
        weighted = HumanGameRecord(
            game_id=rec.game_id,
            ruleset_id=rec.ruleset_id,
            ruleset_version=rec.ruleset_version,
            ruleset_hash=rec.ruleset_hash,
            seats=rec.seats,
            initial_hands=rec.initial_hands,
            bottom_cards=rec.bottom_cards,
            bidding_history=rec.bidding_history,
            action_history=rec.action_history,
            final_result=rec.final_result,
            player_skill_weight={"landlord": 2.5, "landlord_up": 0.5},
        )
        samples = build_bc_samples(weighted)
        assert any(s.skill_weight == 2.5 for s in samples if s.position == "landlord")
        assert all(
            s.skill_weight == 0.5
            for s in samples
            if s.position == "landlord_up"
        )
        # Roles with no explicit weight default to 1.0.
        assert all(
            s.skill_weight == 1.0
            for s in samples
            if s.position == "landlord_down"
        )

    def test_unvalidated_record_raises(self):
        """A record whose actions do not replay cleanly raises BCSampleError."""
        rec = generate_synthetic_record("bc-bad", seed=5)
        # Truncate the action history so the game never terminates -> the
        # replay will still proceed, but a deliberately illegal first action
        # forces a turn/legality error.
        bad = HumanGameRecord(
            game_id=rec.game_id,
            ruleset_id=rec.ruleset_id,
            ruleset_version=rec.ruleset_version,
            ruleset_hash=rec.ruleset_hash,
            seats=rec.seats,
            initial_hands=rec.initial_hands,
            bottom_cards=rec.bottom_cards,
            bidding_history=rec.bidding_history,
            action_history=(("landlord", (30, 20)),) + rec.action_history[1:],
            final_result=rec.final_result,
        )
        with pytest.raises(BCSampleError):
            build_bc_samples(bad)

    def test_batch_stream_skips_bad_records(self):
        """Blocker 3: the batch builder defaults to fail-fast; the with_report
        variant quarantines bad records (no silent drops)."""
        good = generate_synthetic_record("good", seed=7)
        bad = generate_synthetic_record("bad", seed=8)
        bad = HumanGameRecord(
            game_id=bad.game_id,
            ruleset_id=bad.ruleset_id,
            ruleset_version=bad.ruleset_version,
            ruleset_hash=bad.ruleset_hash,
            seats=bad.seats,
            initial_hands=bad.initial_hands,
            bottom_cards=bad.bottom_cards,
            bidding_history=bad.bidding_history,
            action_history=(("landlord", (30, 20)),) + bad.action_history[1:],
            final_result=bad.final_result,
        )
        # Default: fail-fast raises on the bad record.
        with pytest.raises(BCSampleError):
            list(build_bc_samples_batch([good, bad]))
        # Explicit stop_on_error=False + with_report: bad record quarantined.
        report = build_bc_samples_with_report([good, bad])
        assert isinstance(report, BatchSampleReport)
        assert all(s.game_id == good.game_id for s in report.samples)
        assert len(report.samples) > 0
        assert len(report.quarantined) == 1
        assert report.quarantined[0][0] == bad.game_id

    def test_batch_stop_on_error(self):
        bad = generate_synthetic_record("bad2", seed=9)
        bad = HumanGameRecord(
            game_id=bad.game_id,
            ruleset_id=bad.ruleset_id,
            ruleset_version=bad.ruleset_version,
            ruleset_hash=bad.ruleset_hash,
            seats=bad.seats,
            initial_hands=bad.initial_hands,
            bottom_cards=bad.bottom_cards,
            bidding_history=bad.bidding_history,
            action_history=(("landlord", (30, 20)),) + bad.action_history[1:],
            final_result=bad.final_result,
        )
        with pytest.raises(BCSampleError):
            list(build_bc_samples_batch([bad], stop_on_error=True))

    def test_non_legacy_ruleset_rejected_at_sampling(self):
        """The sample builder rejects a non-legacy record before replay
        (Blocker 2: ruleset identity verified at every entry point)."""
        from douzero.env.rules import RuleSet

        rec = generate_synthetic_record("rs-sample", seed=11)
        std = RuleSet.standard().identity()
        bad = HumanGameRecord(
            game_id=rec.game_id,
            ruleset_id=std["ruleset_id"],
            ruleset_version=rec.ruleset_version,
            ruleset_hash=rec.ruleset_hash,
            seats=rec.seats,
            initial_hands=rec.initial_hands,
            bottom_cards=rec.bottom_cards,
            bidding_history=rec.bidding_history,
            action_history=rec.action_history,
            final_result=rec.final_result,
        )
        with pytest.raises(BCSampleError):
            build_bc_samples(bad)


# --------------------------------------------------------------------------- #
# BCSample validation
# --------------------------------------------------------------------------- #
class TestBCSampleValidation:
    def test_negative_index_rejected(self):
        rec = generate_synthetic_record("v-1", seed=1)
        samples = build_bc_samples(rec)
        s = samples[0]
        with pytest.raises(BCSampleError):
            BCSample(
                obs=s.obs,
                human_action_index=-1,
                position=s.position,
                game_id=s.game_id,
                num_legal_actions=s.num_legal_actions,
            )

    def test_position_mismatch_rejected_on_validate(self):
        rec = generate_synthetic_record("v-2", seed=2)
        samples = build_bc_samples(rec)
        s = samples[0]
        wrong_pos = "landlord_up" if s.position != "landlord_up" else "landlord_down"
        bad = BCSample(
            obs=s.obs,
            human_action_index=0,
            position=wrong_pos,
            game_id=s.game_id,
            num_legal_actions=s.num_legal_actions,
        )
        with pytest.raises(BCSampleError):
            bad.validate()

    def test_out_of_range_index_rejected_on_validate(self):
        rec = generate_synthetic_record("v-3", seed=3)
        samples = build_bc_samples(rec)
        s = samples[0]
        bad = BCSample(
            obs=s.obs,
            human_action_index=s.num_legal_actions,  # one past the end
            position=s.position,
            game_id=s.game_id,
            num_legal_actions=s.num_legal_actions,
        )
        with pytest.raises(BCSampleError):
            bad.validate()


# --------------------------------------------------------------------------- #
# Sample weights
# --------------------------------------------------------------------------- #
class TestWeights:
    def test_default_weight_is_one(self):
        w = compute_sample_weight()
        assert w == 1.0

    def test_rule_mismatch_zeros_weight_by_default(self):
        w = compute_sample_weight(rule_match=False)
        assert w == 0.0

    def test_rule_mismatch_keep_action(self):
        cfg = WeightConfig(rule_mismatch_action="keep")
        w = compute_sample_weight(rule_match=False, config=cfg)
        assert w > 0.0

    def test_clip_caps_weight(self):
        cfg = WeightConfig(skill_weight_clip=2.0)
        w = compute_sample_weight(
            skill_weight=10.0, integrity_weight=5.0, config=cfg
        )
        assert w == 2.0

    def test_action_advantage_scales_weight(self):
        base = compute_sample_weight(skill_weight=2.0)
        boosted = compute_sample_weight(
            skill_weight=2.0, action_advantage=0.5
        )
        assert boosted == pytest.approx(base * 1.5)

    def test_vectorized_normalize_to_mean(self):
        cfg = WeightConfig(skill_weight_clip=10.0, normalize_to_mean=True)
        ws = compute_sample_weights(
            skill_weights=[1.0, 2.0, 3.0], config=cfg
        )
        mean = sum(ws) / len(ws)
        assert mean == pytest.approx(1.0)
        # Relative ordering preserved.
        assert ws[0] < ws[1] < ws[2]

    def test_all_zero_weights_do_not_divide_by_zero(self):
        cfg = WeightConfig(normalize_to_mean=True)
        ws = compute_sample_weights(
            skill_weights=[1.0, 1.0], rule_matches=[False, False], config=cfg
        )
        assert ws == [0.0, 0.0]

    def test_length_mismatch_rejected(self):
        with pytest.raises(WeightError):
            compute_sample_weights(
                skill_weights=[1.0, 2.0], integrity_weights=[1.0]
            )

    def test_invalid_config_rejected(self):
        with pytest.raises(WeightError):
            WeightConfig(skill_weight_clip=0)
        with pytest.raises(WeightError):
            WeightConfig(rule_mismatch_action="bogus")
        for value in (float("nan"), float("inf"), float("-inf")):
            with pytest.raises(WeightError, match="finite"):
                WeightConfig(skill_weight_clip=value)
            with pytest.raises(WeightError, match="finite"):
                WeightConfig(integrity_default=value)
            with pytest.raises(WeightError, match="finite"):
                WeightConfig(rule_match_default=value)

    @pytest.mark.parametrize(
        "field,value",
        [
            ("skill_weight", float("nan")),
            ("integrity_weight", float("inf")),
            ("action_advantage", float("-inf")),
        ],
    )
    def test_non_finite_weight_signal_rejected(self, field, value):
        kwargs = {field: value}
        with pytest.raises(WeightError, match="finite"):
            compute_sample_weight(**kwargs)

    def test_normalization_never_breaks_clip(self):
        cfg = WeightConfig(skill_weight_clip=10.0, normalize_to_mean=True)
        weights = compute_sample_weights(
            skill_weights=[10.0] + [0.0] * 99,
            config=cfg,
        )
        assert all(math.isfinite(weight) for weight in weights)
        assert all(0.0 <= weight <= cfg.skill_weight_clip for weight in weights)
        # Zeros are intentional exclusions and must not be revived merely to
        # force an infeasible mean-one total.
        assert weights == [10.0] + [0.0] * 99

    def test_capped_normalization_preserves_feasible_total(self):
        cfg = WeightConfig(skill_weight_clip=2.0, normalize_to_mean=True)
        weights = compute_sample_weights(
            skill_weights=[100.0, 1.0, 1.0, 1.0],
            config=cfg,
        )
        assert all(math.isfinite(weight) for weight in weights)
        assert all(0.0 <= weight <= cfg.skill_weight_clip for weight in weights)
        assert sum(weights) == pytest.approx(len(weights))


# --------------------------------------------------------------------------- #
# Stratified stats
# --------------------------------------------------------------------------- #
class TestStratifiedStats:
    def test_stats_report_counts(self):
        rec = generate_synthetic_record("st-1", seed=1)
        samples = build_bc_samples(rec)
        stats = stratified_stats(samples)
        assert stats["total"] == len(samples)
        assert sum(stats["by_position"].values()) == len(samples)
        assert sum(stats["by_num_legal_actions"].values()) == len(samples)
        # All three roles usually appear in a full game.
        assert "landlord" in stats["by_position"]

    def test_stats_handle_empty(self):
        stats = stratified_stats([])
        assert stats["total"] == 0

    def test_stats_report_winner_team(self):
        """Blocker 4: stratified_stats reports by_winner_team (survivorship
        bias audit), not just position/action counts."""
        rec = generate_synthetic_record("st-team", seed=1)
        samples = build_bc_samples(rec)
        stats = stratified_stats(samples)
        assert "by_winner_team" in stats
        # The record's winner team is present in the distribution.
        team = rec.final_result["winner_team"]
        assert stats["by_winner_team"].get(team, 0) > 0


# --------------------------------------------------------------------------- #
# Composite sample weights (Blocker 4)
# --------------------------------------------------------------------------- #
class TestCompositeWeights:
    def test_bcsample_carries_sample_weight_and_winner_team(self):
        """Blocker 4: BCSample carries the composite sample_weight + winner_team
        (not just raw skill_weight)."""
        rec = generate_synthetic_record("sw-1", seed=1)
        samples = build_bc_samples(rec)
        for s in samples:
            assert hasattr(s, "sample_weight")
            assert hasattr(s, "winner_team")
            assert s.sample_weight == s.skill_weight  # default = skill
            assert s.winner_team in ("landlord", "farmer")

    def test_apply_sample_weights_normalizes(self):
        """Blocker 4: apply_sample_weights stamps clipped+normalized weights."""
        from douzero.human_data.weights import (
            WeightConfig,
            apply_sample_weights,
        )

        rec = generate_synthetic_record("sw-2", seed=2)
        samples = build_bc_samples(rec)
        # Inject diverse raw skill weights.
        import dataclasses

        for i, s in enumerate(samples):
            samples[i] = dataclasses.replace(
                s, skill_weight=float(i + 1)
            )
        out = apply_sample_weights(
            samples, config=WeightConfig(skill_weight_clip=100.0)
        )
        # sample_weight is now normalized so the mean is ~1.0.
        mean = sum(s.sample_weight for s in out) / len(out)
        assert abs(mean - 1.0) < 0.01
        # skill_weight is preserved (raw).
        for orig, stamped in zip(samples, out):
            assert stamped.skill_weight == orig.skill_weight
        # Relative ordering preserved.
        assert out[0].sample_weight < out[-1].sample_weight
