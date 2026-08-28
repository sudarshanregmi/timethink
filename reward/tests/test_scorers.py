"""
Tests for reward scorers — edge cases, boundary conditions, self-consistency.
"""
import pytest

from reward.config import RewardConfig, score_numeric_proximity, get_type_similarity
from reward.evidence import (
    set_context_seq_len,
    score_evidence_quality,
    score_verification_intermediate,
    _score_local_events,
    _score_trend_segments,
    _score_key_points,
    _score_params,
    _score_event_pair,
    _score_single_metric,
    _consistency_multiplier,
)
from reward.scorers.core import (
    score_mts_verdict,
    score_mts_set,
    score_enumeration,
    score_yes_no,
    score_description,
    score_trend_dominance,
    score_anti_judgment,
    _score_condition_steps,
)
from reward.scorers.segments import (
    score_compound_judgment,
    _score_judgment_conditions,
    _extract_outcome,
)
from reward.scorers.statistical import (
    score_stat_numerical,
    _parse_float_verdict,
    _parse_list_verdict,
)


@pytest.fixture(autouse=True)
def set_seq_len():
    """Set a default seq_len context for all tests."""
    set_context_seq_len(256)
    yield
    set_context_seq_len(0)


# ══════════════════════════════════════════════════════════════════════
# config.py helpers
# ══════════════════════════════════════════════════════════════════════

class TestScoreNumericProximity:
    def test_exact_match(self):
        assert score_numeric_proximity(5.0, 5.0, 10.0, 0.1) == 1.0

    def test_far_away(self):
        score = score_numeric_proximity(100.0, 0.0, 10.0, 0.1)
        assert score < 0.01

    def test_symmetry(self):
        s1 = score_numeric_proximity(3.0, 5.0, 10.0, 0.1)
        s2 = score_numeric_proximity(5.0, 3.0, 10.0, 0.1)
        assert abs(s1 - s2) < 1e-10

    def test_zero_scale_safety(self):
        # scale=0 should not cause div by zero
        score = score_numeric_proximity(1.0, 0.0, 0.0, 0.1)
        assert 0.0 <= score <= 1.0

    def test_negative_values(self):
        score = score_numeric_proximity(-5.0, -5.0, 10.0, 0.1)
        assert score == 1.0

    def test_small_k_more_sensitive(self):
        s_sensitive = score_numeric_proximity(6.0, 5.0, 10.0, 0.01)
        s_lenient = score_numeric_proximity(6.0, 5.0, 10.0, 0.5)
        assert s_sensitive < s_lenient


class TestGetTypeSimilarity:
    def test_exact_match(self):
        assert get_type_similarity("increase", "increase") == 1.0

    def test_known_pair(self):
        assert get_type_similarity("linear increase", "increase") == 0.9

    def test_symmetry(self):
        s1 = get_type_similarity("upward spike", "sudden increase")
        s2 = get_type_similarity("sudden increase", "upward spike")
        assert s1 == s2

    def test_unknown_pair(self):
        assert get_type_similarity("foo", "bar") == 0.0

    def test_substring_match(self):
        score = get_type_similarity("spike", "upward spike")
        assert score == 0.4  # substring fallback

    def test_case_insensitive(self):
        assert get_type_similarity("Smooth", "smooth") == 1.0

    def test_underscore_to_space(self):
        assert get_type_similarity("linear_increase", "linear increase") == 1.0


# ══════════════════════════════════════════════════════════════════════
# statistical.py
# ══════════════════════════════════════════════════════════════════════

class TestParseFloatVerdict:
    def test_normal(self):
        assert _parse_float_verdict("3.14") == pytest.approx(3.14)

    def test_nan(self):
        assert _parse_float_verdict("nan") is None

    def test_NaN(self):
        assert _parse_float_verdict("NaN") is None

    def test_inf(self):
        assert _parse_float_verdict("inf") is None

    def test_neg_inf(self):
        assert _parse_float_verdict("-inf") is None

    def test_integer(self):
        assert _parse_float_verdict("42") == 42.0

    def test_scientific(self):
        assert _parse_float_verdict("1.5e-3") == pytest.approx(0.0015)

    def test_garbage(self):
        assert _parse_float_verdict("abc") is None

    def test_trailing_dot(self):
        # '71.251.' — the bug that crashed us
        assert _parse_float_verdict("71.251.") is None

    def test_double_dot(self):
        assert _parse_float_verdict("1..5") is None

    def test_empty(self):
        assert _parse_float_verdict("") is None

    def test_none(self):
        assert _parse_float_verdict(None) is None


class TestParseListVerdict:
    def test_normal(self):
        result = _parse_list_verdict("[1.0, 2.0, 3.0]")
        assert result == [1.0, 2.0, 3.0]

    def test_nan_in_list(self):
        assert _parse_list_verdict("[1.0, nan, 2.0]") is None

    def test_inf_in_list(self):
        assert _parse_list_verdict("[inf]") is None

    def test_empty_list(self):
        result = _parse_list_verdict("[]")
        assert result == []

    def test_no_brackets(self):
        assert _parse_list_verdict("1.0, 2.0") is None

    def test_single_element(self):
        result = _parse_list_verdict("[42.0]")
        assert result == [42.0]

    def test_negative_values(self):
        result = _parse_list_verdict("[-1.5, -2.3]")
        assert result == [-1.5, -2.3]


class TestScoreStatNumerical:
    def _wrap(self, think_content):
        return think_content  # scorers take raw think content, not wrapped

    def test_self_consistency_float(self):
        think = "metric: M\nstats: mean=5.000\n===\nanswer: 5.000"
        assert score_stat_numerical(think, think) >= 0.99

    def test_self_consistency_list(self):
        think = "metric: M\nstats: mean=5.0\n===\nanswer: [1.0, 2.0, 3.0]"
        assert score_stat_numerical(think, think) >= 0.99

    def test_self_consistency_categorical(self):
        think = "metric: M\nfirst_half [0-127]: mean=6.00\nsecond_half [128-255]: mean=4.00\n===\n6.00 > 4.00\nanswer: first half"
        assert score_stat_numerical(think, think) >= 0.99

    def test_nan_verdict_low_score(self):
        gt = "metric: M\nstats: mean=5.000\n===\nanswer: 5.000"
        pred = "metric: M\nstats: mean=5.000\n===\nanswer: nan"
        assert score_stat_numerical(pred, gt) < 0.5

    def test_no_answer_line(self):
        gt = "metric: M\nstats: mean=5.000\n===\nanswer: 5.000"
        pred = "metric: M\nstats: mean=5.000\n==="
        score = score_stat_numerical(pred, gt)
        # No verdict → partial credit from evidence only, well below correct.
        assert 0.0 < score < 0.65

    def test_wrong_float(self):
        gt = "metric: M\nstats: mean=5.000\n===\nanswer: 5.000"
        pred = "metric: M\nstats: mean=5.000\n===\nanswer: 100.000"
        score = score_stat_numerical(pred, gt)
        assert score < 0.5

    def test_list_length_mismatch(self):
        gt = "metric: M\n===\nanswer: [1.0, 2.0, 3.0]"
        pred = "metric: M\n===\nanswer: [1.0, 2.0]"
        score = score_stat_numerical(pred, gt)
        assert score < 0.9  # length penalty


# ══════════════════════════════════════════════════════════════════════
# core.py scorers
# ══════════════════════════════════════════════════════════════════════

class TestScoreYesNo:
    def test_self_consistency(self):
        think = (
            "metric: M\nnoise=smooth, strength=0.00\n"
            "===\ntarget: PASS.\nanswer: yes"
        )
        assert score_yes_no(think, think) >= 0.99

    def test_wrong_verdict(self):
        gt = "metric: M\nnoise=smooth, strength=0.00\n===\ntarget: PASS."
        pred = "metric: M\nnoise=smooth, strength=0.00\n===\ntarget: FAIL."
        score = score_yes_no(pred, gt)
        assert score < 0.5

    def test_missing_verdict(self):
        gt = "metric: M\nnoise=smooth, strength=0.00\n===\ntarget: PASS."
        pred = "metric: M\nnoise=smooth, strength=0.00\n==="
        score = score_yes_no(pred, gt)
        assert 0.0 < score < 0.6  # no verdict → low partial credit


class TestScoreMtsVerdict:
    def test_self_consistency(self):
        think = (
            "metric: A\nnoise=smooth, strength=0.00\n"
            "---\nmetric: B\nnoise=smooth, strength=0.00\n"
            "===\nvs B: PASS.\nanswer: similar"
        )
        assert score_mts_verdict(think, think) >= 0.99

    def test_wrong_answer(self):
        gt = "metric: A\n---\nmetric: B\n===\nanswer: positive"
        pred = "metric: A\n---\nmetric: B\n===\nanswer: negative"
        assert score_mts_verdict(pred, gt) < 0.5


class TestScoreMtsSet:
    def test_self_consistency(self):
        think = (
            "metric: A\n---\nmetric: B\n---\nmetric: C\n"
            "===\nvs B: PASS.\nvs C: FAIL.\nanswer: (A, B)"
        )
        assert score_mts_set(think, think) >= 0.99

    def test_partial_match(self):
        gt = "metric: A\n---\nmetric: B\n===\nanswer: (A, B)"
        pred = "metric: A\n---\nmetric: B\n===\nanswer: (A)"
        score = score_mts_set(pred, gt)
        assert 0.3 < score < 0.9  # partial overlap → meaningful middle range

    def test_empty_gt_pred_hallucinated(self):
        gt = "metric: A\n===\nanswer: ()"
        pred = "metric: A\n===\nanswer: (X, Y)"
        assert score_mts_set(pred, gt) < 0.8


class TestScoreEnumeration:
    def test_self_consistency(self):
        think = "metric: M\nnoise=smooth, strength=0.00\n===\nanswer: 3"
        assert score_enumeration(think, think) >= 0.99

    def test_wrong_answer(self):
        gt = "metric: M\n===\nanswer: 3"
        pred = "metric: M\n===\nanswer: 5"
        assert score_enumeration(pred, gt) < 0.5

    def test_numeric_tolerance_int_float(self):
        """'3' should match '3.0' via numeric fallback."""
        gt = "metric: M\n===\nanswer: 3"
        pred = "metric: M\n===\nanswer: 3.0"
        assert score_enumeration(pred, gt) >= 0.60  # verdict=1.0, partial evidence

    def test_float_proportion_tolerance(self):
        """'0.50' should match '0.5' via numeric fallback."""
        gt = "metric: M\n===\nanswer: 0.50"
        pred = "metric: M\n===\nanswer: 0.5"
        assert score_enumeration(pred, gt) >= 0.60

    def test_arrow_transition_verdict(self):
        """Arrow-notation transition verdict like 'increase→decrease' should match."""
        think = "metric: M\n===\ncounts: increase→decrease=2\nanswer: increase→decrease"
        assert score_enumeration(think, think) >= 0.60

    def test_segment_display_verdict(self):
        """Segment display verdict like '(increase, 0, 128)' should match."""
        think = "metric: M\n===\nmax events = 3, segment: (increase, 0, 128)\nanswer: (increase, 0, 128)"
        assert score_enumeration(think, think) >= 0.60

    def test_comma_separated_tie_verdict(self):
        """Comma-separated ties like 'A, B' should match."""
        think = "metric: A\n---\nmetric: B\n===\nanswer: A, B"
        assert score_enumeration(think, think) >= 0.60

    def test_position_integer_verdict(self):
        """Position integer verdict for temporal_position."""
        think = "metric: M\n===\nsorted positions: 10, 80\nearliest = 10\nanswer: 10"
        assert score_enumeration(think, think) >= 0.60

    def test_trend_type_verdict(self):
        """Type name verdict for trend_type_with_most_events."""
        think = "metric: M\n===\ncounts: increase=3, decrease=1\nmax events = 3, type: increase\nanswer: increase"
        assert score_enumeration(think, think) >= 0.60


class TestScoreDescription:
    def test_self_consistency(self):
        think = (
            "metric: TestMetric\n"
            "noise=smooth, strength=0.00\n"
            "start=10.0, end=20.0, amp=10.0, overall trend=increase\n"
            "trend_segments=[(increase, 1, 256)]\n"
            "season=no periodic fluctuation, period=0.00, amp=0.00"
        )
        assert score_description(think, think) >= 0.99


class TestScoreTrendDominance:
    def test_self_consistency(self):
        think = (
            "metric: M\nnoise=smooth, strength=0.00\n"
            "===\n∩ segments: 3\naggregate durations:\n"
            "- increase: 150\n- decrease: 106\n"
            "dominant trend comparison: increase > decrease\nanswer: increase"
        )
        assert score_trend_dominance(think, think) >= 0.99

    def test_missing_structural_markers(self):
        think = "metric: M\nnoise=smooth, strength=0.00\n===\nanswer: increase"
        # Without ∩, aggregate durations, dominant trend comparison
        # → structural_score=0, coherency=0.5 (neutral)
        # = 0.55 + 0.05*0.5 + 0.15*0 + 0.25*evidence
        score = score_trend_dominance(think, think)
        assert 0.75 < score < 0.95

    def test_wrong_trend(self):
        gt = "metric: M\n===\nanswer: increase"
        pred = "metric: M\n===\nanswer: decrease"
        assert score_trend_dominance(pred, gt) < 0.5


class TestScoreAntiJudgment:
    def test_self_consistency(self):
        think = (
            "metric: A\n---\nmetric: B\n"
            "===\ncondition 1: check\ncondition 2: check\n"
            "all conditions: all met\nanswer: yes"
        )
        assert score_anti_judgment(think, think) >= 0.95

    def test_wrong_verdict(self):
        gt = "===\nanswer: yes"
        pred = "===\nanswer: no"
        assert score_anti_judgment(pred, gt) < 0.5


# ══════════════════════════════════════════════════════════════════════
# segments.py
# ══════════════════════════════════════════════════════════════════════

class TestScoreJudgmentConditions:
    def test_format_a_basic(self):
        text = (
            "condition 1: check X\nvalue 5.0 > 3.0 -> met\n"
            "condition 2: check Y\nvalue 2.0 <= 4.0 -> met\n"
            "all conditions: all met"
        )
        assert _score_judgment_conditions(text) >= 0.99

    def test_format_a_missing_summary(self):
        text = (
            "condition 1: check X\nvalue -> met\n"
            "condition 2: check Y\nvalue -> met\n"
        )
        # Missing 'all conditions:' with 2 conditions → penalty
        score = _score_judgment_conditions(text)
        assert score < 0.95  # missing summary penalty

    def test_format_b_events(self):
        text = "event 1: spike at 50, pass\nevent 2: dip at 100, fail\nsummary: 1 of 2 pass"
        assert _score_judgment_conditions(text) >= 0.99

    def test_format_c_met(self):
        text = "ratio = 1.600 / 1.000 = 1.600\n1.600 >= 1.50 -> met"
        assert _score_judgment_conditions(text) >= 0.99

    def test_format_c_pass_fail(self):
        text = "pair (spike at 10, dip at 20): gap = 10, 10 <= 15 -> pass"
        assert _score_judgment_conditions(text) >= 0.99

    def test_format_c_parenthesized(self):
        text = "-31.848 >= -31.929 (fail)"
        assert _score_judgment_conditions(text) >= 0.99

    def test_bypass_no_false_positive(self):
        """'bypass' should NOT match \\bpass\\b."""
        text = (
            "condition 1: check value\nresult bypass the threshold\n"
            "condition 2: noise check\nvalue met\n"
            "all conditions: all met"
        )
        score = _score_judgment_conditions(text)
        # condition 1 has 'bypass' not 'pass' → should only get 0.5 for that condition
        assert score < 0.95  # bypass != pass penalty

    def test_empty_text(self):
        assert _score_judgment_conditions("") == 0.0

    def test_single_condition_no_summary_needed(self):
        text = "condition 1: trend check\ntrend is increasing -> met"
        assert _score_judgment_conditions(text) >= 0.99


class TestScoreCompoundJudgment:
    def test_self_consistency(self):
        think = (
            "metric: A\nstats: max=10.000, min=2.000\n---\n"
            "metric: B\nstats: max=8.000, min=3.000\n===\n"
            "A range = 8.000\nB range = 5.000\n"
            "ratio = 1.600\n1.600 >= 1.50 -> met\nanswer: yes"
        )
        assert score_compound_judgment(think, think) >= 0.99


# ══════════════════════════════════════════════════════════════════════
# evidence.py
# ══════════════════════════════════════════════════════════════════════

class TestEvidenceScoring:
    def test_empty_gt(self):
        assert score_evidence_quality("anything", "") == 1.0

    def test_self_consistency_single_metric(self):
        think = (
            "metric: M\n"
            "noise=smooth, strength=0.00\n"
            "start=10.0, end=20.0, amp=10.0, overall trend=increase\n"
            "trend_segments=[(increase, 1, 256)]\n"
            "season=no periodic fluctuation, period=0.00, amp=0.00"
        )
        assert score_evidence_quality(think, think) >= 0.99

    def test_hallucinated_extra_metric(self):
        gt = "metric: A\nnoise=smooth, strength=0.00"
        pred = (
            "metric: A\nnoise=smooth, strength=0.00\n"
            "---\nmetric: B\nnoise=noisy, strength=0.5"
        )
        score = score_evidence_quality(pred, gt)
        assert score < 0.95  # penalty for extra metric

    def test_missing_metric_in_pred(self):
        gt = (
            "metric: A\nnoise=smooth, strength=0.00\n"
            "---\nmetric: B\nnoise=noisy, strength=0.5"
        )
        pred = "metric: A\nnoise=smooth, strength=0.00"
        score = score_evidence_quality(pred, gt)
        assert score < 0.85  # missing metric B

    def test_seq_len_not_set(self):
        set_context_seq_len(0)
        gt = "metric: M\nnoise=smooth, strength=0.00"
        with pytest.raises(ValueError, match="seq_len not set"):
            score_evidence_quality(gt, gt)


class TestVerificationIntermediate:
    def test_all_correct(self):
        think = "data\n===\nvs A: PASS.\nvs B: FAIL."
        assert score_verification_intermediate(think, think) == 1.0

    def test_one_wrong(self):
        gt = "data\n===\nvs A: PASS.\nvs B: FAIL."
        pred = "data\n===\nvs A: PASS.\nvs B: PASS."
        score = score_verification_intermediate(pred, gt)
        assert score == 0.5  # one right, one wrong

    def test_missing_metric_in_pred(self):
        gt = "data\n===\nvs A: PASS.\nvs B: FAIL."
        pred = "data\n===\nvs A: PASS."
        score = score_verification_intermediate(pred, gt)
        # A correct (1.0), B absent (0.0) → avg 0.5
        assert score == pytest.approx(0.5)

    def test_no_vs_lines(self):
        assert score_verification_intermediate("data\n===\njust text", "data\n===\njust text") == 1.0


class TestLocalEventScoring:
    def test_both_empty(self):
        assert _score_local_events([], [], 256) == 1.0

    def test_hallucinated_events(self):
        pred = [{"type": "spike"}]
        score = _score_local_events(pred, [], 256)
        assert score < 0.85  # hallucinated event penalty

    def test_missing_events(self):
        gt = [{"type": "spike"}]
        assert _score_local_events([], gt, 256) == 0.0

    def test_perfect_match(self):
        ev = {"type": "upward spike", "amplitude": 5.0, "position_start": 50.0}
        assert _score_local_events([ev], [ev], 256) >= 0.99

    def test_extra_pred_events_penalty(self):
        gt = [{"type": "spike"}]
        pred = [{"type": "spike"}, {"type": "dip"}, {"type": "flat"}]
        score = _score_local_events(pred, gt, 256)
        assert score < 0.85  # extra events penalty


class TestTrendSegmentScoring:
    def test_perfect_match(self):
        segs = [{"type": "increase", "start": 1, "end": 128},
                {"type": "decrease", "start": 129, "end": 256}]
        assert _score_trend_segments(segs, segs, 256) >= 0.99

    def test_count_mismatch(self):
        gt = [{"type": "increase", "start": 1, "end": 256}]
        pred = [{"type": "increase", "start": 1, "end": 128},
                {"type": "decrease", "start": 129, "end": 256}]
        score = _score_trend_segments(pred, gt, 256)
        assert score < 0.85  # segment count mismatch

    def test_both_empty(self):
        assert _score_trend_segments([], [], 256) == 1.0


class TestKeyPointScoring:
    def test_perfect_match(self):
        kps = [{"label": "start", "index": 10, "value": 1.0}]
        assert _score_key_points(kps, kps, 256) >= 0.99

    def test_missing_label(self):
        gt = [{"label": "start", "index": 10}, {"label": "end", "index": 50}]
        pred = [{"label": "start", "index": 10}]
        score = _score_key_points(pred, gt, 256)
        assert score == pytest.approx(0.5, abs=0.05)  # one matched, one missing

    def test_empty_gt(self):
        assert _score_key_points([], [], 256) == 1.0

    def test_empty_pred(self):
        gt = [{"label": "peak", "index": 50}]
        assert _score_key_points([], gt, 256) == 0.0


class TestParamScoring:
    def test_perfect_match(self):
        params = {"width": 10.0, "sub_amplitudes": [0.5, 0.3]}
        assert _score_params(params, params) >= 0.99

    def test_missing_key(self):
        gt = {"width": 10.0, "height": 5.0}
        pred = {"width": 10.0}
        score = _score_params(pred, gt)
        assert score == pytest.approx(0.5, abs=0.05)

    def test_empty_gt(self):
        assert _score_params({}, {}) == 1.0

    def test_empty_pred(self):
        assert _score_params({}, {"width": 10.0}) == 0.0

    def test_list_length_mismatch(self):
        gt = {"vals": [1.0, 2.0, 3.0]}
        pred = {"vals": [1.0, 2.0]}
        assert _score_params(pred, gt) == 0.0

    def test_string_param(self):
        gt = {"mode": "linear"}
        pred = {"mode": "LINEAR"}
        assert _score_params(pred, gt) == 1.0  # case-insensitive


# ══════════════════════════════════════════════════════════════════════
# compute_score integration (end-to-end)
# ══════════════════════════════════════════════════════════════════════

class TestComputeScoreIntegration:
    """End-to-end tests for the top-level compute_score function."""

    def test_self_consistency_description(self):
        from reward import compute_score
        gt = (
            "<think>\n"
            "metric: Attribution Metrics\n"
            "noise=smooth, strength=0.00\n"
            "start=79.85, end=79.85, amp=0.00, overall trend=keep steady\n"
            "trend_segments=[(keep steady, 1, 256)]\n"
            "season=no periodic fluctuation, period=0.00, amp=0.00\n"
            "</think>\nSome text."
        )
        # description is free-form text: rule-based reward returns neutral 0.5
        # (LLM-as-judge in evaluation/ is the proper scorer for this type).
        # See memory/reward_verdict_only_val_monitoring.md.
        score = compute_score(gt, gt, eval_type="description", eval_metadata={"length": 256})
        assert abs(score - 0.5) < 0.01

    def test_self_consistency_yes_no(self):
        from reward import compute_score
        gt = (
            "<think>\n"
            "metric: M\nnoise=smooth, strength=0.00\n"
            "===\ntarget: PASS.\nanswer: yes\n"
            "</think>\nAnswer."
        )
        score = compute_score(gt, gt, eval_type="yes_no", eval_metadata={"length": 256})
        assert score >= 0.99

    def test_self_consistency_stat_numerical(self):
        from reward import compute_score
        gt = (
            "<think>\n"
            "metric: M\nstats: mean=5.000\n"
            "===\nanswer: 5.000\n"
            "</think>"
        )
        score = compute_score(gt, gt, eval_type="stat_numerical", eval_metadata={"length": 256})
        assert score >= 0.99

    def test_no_think_block_in_pred(self):
        from reward import compute_score
        gt = "<think>metric: M\nnoise=smooth, strength=0.00</think>"
        pred = "no think block here"
        score = compute_score(pred, gt, eval_type="description", eval_metadata={"length": 256})
        assert score == RewardConfig.PENALTY_PARSE_ERROR

    def test_missing_eval_type(self):
        from reward import compute_score
        with pytest.raises(ValueError, match="eval_type"):
            compute_score("<think>x</think>", "<think>x</think>")

    def test_missing_eval_metadata_length(self):
        from reward import compute_score
        with pytest.raises(ValueError, match="eval_metadata"):
            compute_score("<think>x</think>", "<think>x</think>",
                          eval_type="description", eval_metadata={})

    def test_unknown_eval_type(self):
        from reward import compute_score
        with pytest.raises(ValueError, match="Unknown eval_type"):
            compute_score("<think>x</think>", "<think>x</think>",
                          eval_type="nonexistent", eval_metadata={"length": 256})

    def test_eval_type_from_extra_info(self):
        """Verify compute_score can read eval_type from extra_info kwarg."""
        from reward import compute_score
        gt = (
            "<think>\n"
            "metric: M\nnoise=smooth, strength=0.00\n"
            "start=10.0, end=10.0, amp=0.0, overall trend=keep steady\n"
            "trend_segments=[(keep steady, 1, 256)]\n"
            "season=no periodic fluctuation, period=0.00, amp=0.00\n"
            "</think>"
        )
        score = compute_score(
            gt, gt,
            extra_info={
                "eval_type": "description",
                "eval_metadata": {"length": 256},
            },
        )
        # description returns neutral 0.5 — see note above.
        assert abs(score - 0.5) < 0.01

    def test_all_eval_types_self_consistent(self):
        """Every eval_type should score >= 0.99 on self-comparison with valid input."""
        from reward import compute_score

        test_cases = {
            "description": (
                "<think>metric: M\nnoise=smooth, strength=0.00\n"
                "start=10.0, end=10.0, amp=0.0, overall trend=keep steady\n"
                "trend_segments=[(keep steady, 1, 256)]\n"
                "season=no periodic fluctuation, period=0.00, amp=0.00</think>"
            ),
            "correlation": (
                "<think>metric: A\nnoise=smooth, strength=0.00\n"
                "---\nmetric: B\nnoise=smooth, strength=0.00\n"
                "===\nvs B: PASS.\nanswer: positive</think>"
            ),
            "anticorrelation": (
                "<think>metric: A\nnoise=smooth, strength=0.00\n"
                "---\nmetric: B\nnoise=smooth, strength=0.00\n"
                "===\nvs B: FAIL.\nanswer: negative</think>"
            ),
            "clustering": (
                "<think>metric: A\n---\nmetric: B\n---\nmetric: C\n"
                "===\nanswer: (A, B)</think>"
            ),
            "anticlustering": (
                "<think>metric: A\n---\nmetric: B\n"
                "===\nanswer: (A)</think>"
            ),
            "yes_no": (
                "<think>metric: M\nnoise=smooth, strength=0.00\n"
                "===\ntarget: PASS.</think>"
            ),
            "stat_numerical": (
                "<think>metric: M\nstats: mean=5.000\n"
                "===\nanswer: 5.000</think>"
            ),
            "cross_stat_judgment": (
                "<think>metric: A\nstats: max=10.000\n---\n"
                "metric: B\nstats: max=8.000\n===\n"
                "ratio = 1.25\n1.25 >= 1.0 -> met\nanswer: yes</think>"
            ),
            "segment_judgment": (
                "<think>metric: A\nstats: max=10.000\n---\n"
                "metric: B\nstats: max=8.000\n===\n"
                "condition 1: check\n-> met\nanswer: yes</think>"
            ),
            "anti_judgment": (
                "<think>metric: A\n---\nmetric: B\n===\n"
                "condition 1: X\ncondition 2: Y\n"
                "all conditions: all met\nanswer: yes</think>"
            ),
            "local_enumeration": (
                "<think>metric: M\nlocal=[spike, amp=5.0]\n"
                "===\nanswer: 1</think>"
            ),
            "segment_enumeration": (
                "<think>metric: M\ntrend_segments=[(increase, 1, 128), (decrease, 129, 256)]\n"
                "===\nanswer: 2</think>"
            ),
            "transition_enumeration": (
                "<think>metric: M\n"
                "trend_segments=[(increase, 0, 50), (decrease, 51, 100), (increase, 101, 200)]\n"
                "===\ntransitions: 2\nanswer: 2</think>"
            ),
            "event_segment_enumeration": (
                "<think>metric: M\n"
                "trend_segments=[(increase, 0, 128), (decrease, 129, 256)]\n"
                "  event 1: upward spike, pos=50\n"
                "===\nfilter: events in increase segments\n"
                "  event 1: pos=50 in (increase, 0, 128) -> increase == increase, pass\n"
                "summary: 1 of 1 pass\nanswer: 1</think>"
            ),
            "temporal_position": (
                "<think>metric: M\n"
                "  event 1: spike, pos=10\n  event 2: dip, pos=80\n"
                "===\nsorted positions: 10, 80\nearliest = 10\nanswer: 10</think>"
            ),
            "duration_proportion": (
                "<think>metric: M\n"
                "trend_segments=[(increase, 0, 100), (decrease, 101, 200)]\n"
                "===\ndurations:\n  (increase, 0, 100): duration = 100 - 0 = 100\n"
                "  (decrease, 101, 200): duration = 200 - 101 = 99\n"
                "total = 199\nproportions:\n  increase = 100 / 199 = 0.50\n"
                "answer: 0.50</think>"
            ),
            "cross_metric_enumeration": (
                "<think>metric: A\n---\nmetric: B\n===\nanswer: 2</think>"
            ),
            "cross_trend_query": (
                "<think>metric: M\nnoise=smooth, strength=0.00\n"
                "===\n∩ segments: 2\naggregate durations: increase=150\n"
                "dominant trend comparison: increase\nanswer: increase</think>"
            ),
            "segment_trend_dominance": (
                "<think>metric: M\nnoise=smooth, strength=0.00\n"
                "===\n∩ segments: 2\naggregate durations: decrease=200\n"
                "dominant trend comparison: decrease\nanswer: decrease</think>"
            ),
            # --- Periodicity / Anomaly / Change-point ---
            "periodicity": (
                "<think>metric: M\nseason=sine, period=32.00, amp=1.50\n"
                "===\nperiod = 32.00\nanswer: 32.00</think>"
            ),
            "change_point": (
                "<think>metric: M\n"
                "trend_segments=[(increase, 0, 100), (decrease, 101, 200)]\n"
                "===\nchange points = 1\nanswer: 1</think>"
            ),
        }

        # eval_metadata overrides for types that need sub_type/verdict
        meta_overrides = {
            "periodicity": {"sub_type": "period_value", "verdict": "32.00"},

            "change_point": {"sub_type": "change_count", "verdict": "1"},
        }

        # description/tsevol are free-form text — rule-based reward returns
        # neutral 0.5 (LLM-as-judge in evaluation/ is the proper scorer).
        NEUTRAL_TYPES = {'description', 'tsevol'}
        for eval_type, think_text in test_cases.items():
            meta = {"length": 256}
            if eval_type in meta_overrides:
                meta.update(meta_overrides[eval_type])
            score = compute_score(
                think_text, think_text,
                eval_type=eval_type,
                eval_metadata=meta,
            )
            if eval_type in NEUTRAL_TYPES:
                assert abs(score - 0.5) < 0.01, (
                    f"{eval_type} neutral check failed: {score}"
                )
            else:
                assert score >= 0.95, f"{eval_type} self-consistency failed: {score}"

    def test_score_bounded_0_1(self):
        """Score should always be in [0, 1]."""
        from reward import compute_score
        gt = "<think>metric: M\nnoise=smooth, strength=0.00</think>"
        pred = "<think>totally wrong content here\nmetric: Wrong\nnoise=chaotic, strength=99.0</think>"
        score = compute_score(pred, gt, eval_type="description", eval_metadata={"length": 256})
        assert 0.0 <= score <= 1.0


# ══════════════════════════════════════════════════════════════════════
# Semantic fix tests (Changes #1–#10)
# ══════════════════════════════════════════════════════════════════════

class TestConsistencyMultiplier:
    """Change #1: cross-field consistency multiplier."""

    def test_contradictory_trend_direction(self):
        # increase type but end < start (actually decreasing)
        attrs = {'trend': {'type': 'increase', 'amplitude': 5, 'start': 10, 'end': 0}}
        assert _consistency_multiplier(attrs) < 1.0

    def test_contradictory_noise_smooth_nonzero_strength(self):
        # Pipeline always sets strength=0.0 for smooth; any nonzero is contradictory
        attrs = {'noise': {'type': 'smooth', 'strength': 0.03}}
        assert _consistency_multiplier(attrs) < 1.0

    def test_contradictory_stats_min_gt_max(self):
        attrs = {'stats': {'min': 100, 'max': 50}}
        assert _consistency_multiplier(attrs) < 1.0

    def test_consistent_trend(self):
        attrs = {'trend': {'type': 'increase', 'amplitude': 5, 'start': 0, 'end': 5}}
        assert _consistency_multiplier(attrs) == 1.0

    def test_consistent_decrease(self):
        attrs = {'trend': {'type': 'decrease', 'amplitude': -5, 'start': 10, 'end': 5}}
        assert _consistency_multiplier(attrs) == 1.0

    def test_consistent_steady(self):
        attrs = {'trend': {'type': 'keep steady', 'amplitude': 0, 'start': 5, 'end': 5}}
        assert _consistency_multiplier(attrs) == 1.0

    def test_steady_small_amp_within_pipeline_threshold(self):
        # Pipeline uses 10% of data_range as threshold for "steady".
        # amp=9 with start=100 means 9% of ref — within the 10% tolerance.
        attrs = {'trend': {'type': 'keep steady', 'amplitude': 9, 'start': 100, 'end': 109}}
        assert _consistency_multiplier(attrs) == 1.0

    def test_steady_large_amp_violation(self):
        # amp=20 with start=100 means 20% of ref — exceeds 10% tolerance.
        attrs = {'trend': {'type': 'keep steady', 'amplitude': 20, 'start': 100, 'end': 120}}
        assert _consistency_multiplier(attrs) < 1.0

    def test_no_periodic_with_period(self):
        attrs = {'seasonal': {'type': 'no periodic fluctuation', 'period': 10, 'amplitude': 0}}
        assert _consistency_multiplier(attrs) < 1.0

    def test_stats_mean_below_min(self):
        attrs = {'stats': {'min': 10, 'max': 100, 'mean': 5}}
        assert _consistency_multiplier(attrs) < 1.0

    def test_negative_std(self):
        attrs = {'stats': {'min': 0, 'max': 100, 'std': -1.0}}
        assert _consistency_multiplier(attrs) < 1.0

    def test_empty_attrs(self):
        assert _consistency_multiplier({}) == 1.0

    def test_floor_at_0_7(self):
        # Many violations should not go below 0.7
        attrs = {
            'trend': {'type': 'increase', 'amplitude': -5, 'start': 0, 'end': 10},
            'noise': {'type': 'smooth', 'strength': 0.5},
            'stats': {'min': 100, 'max': 50, 'mean': 200, 'std': -1},
        }
        assert _consistency_multiplier(attrs) == 0.7


class TestEventPositionWeighting:
    """Change #2: event position weighted by type correctness."""

    def test_wrong_type_right_position_penalized(self):
        gt = {"type": "upward spike", "position_start": 100.0}
        pred_wrong = {"type": "downward spike", "position_start": 100.0}
        pred_right = {"type": "upward spike", "position_start": 100.0}
        score_wrong = _score_event_pair(pred_wrong, gt, 256)
        score_right = _score_event_pair(pred_right, gt, 256)
        assert score_wrong < score_right


class TestVerificationAbsentZero:
    """Change #6: absent metric verification scores 0, not 0.5."""

    def test_absent_metric_scores_zero(self):
        gt = "data\n===\nvs A: PASS.\nvs B: FAIL."
        pred = "data\n==="  # no verification at all
        score = score_verification_intermediate(pred, gt)
        assert score == 0.0


class TestEnumerationNumericTolerance:
    """Change #7: '3' == '3.0' in enumeration."""

    def test_integer_vs_float_format(self):
        gt = "metric: M\nlocal=[spike]\n===\nanswer: 3"
        pred = "metric: M\nlocal=[spike]\n===\nanswer: 3.0"
        score = score_enumeration(pred, gt)
        # verdict should be 1.0 (numeric tolerance), evidence ~1.0
        assert score >= 0.95

    def test_float_trailing_zero(self):
        gt = "metric: M\n===\nanswer: 5.46"
        pred = "metric: M\n===\nanswer: 5.460"
        assert score_enumeration(pred, gt) >= 0.95


class TestConditionOutcomeCorrectness:
    """Change #8: wrong condition outcome should be penalized."""

    def test_wrong_outcome_penalized(self):
        gt = "condition 1: check X\nvalue 5.0 > 3.0 -> met"
        pred = "condition 1: check X\nvalue 2.0 < 3.0 -> fail"
        score = _score_judgment_conditions(pred, gt)
        assert score < 0.75  # wrong outcome penalty

    def test_correct_outcome_full_score(self):
        gt = "condition 1: check X\nvalue 5.0 > 3.0 -> met"
        pred = "condition 1: check X\nvalue 5.0 > 3.0 -> met"
        score = _score_judgment_conditions(pred, gt)
        assert score >= 0.99

    def test_not_met_vs_met_distinguished(self):
        """'not met' and 'met' must be different outcomes — not conflated."""
        gt = "condition 1: check X\nvalue 2.0 < 3.0 -> not met"
        pred = "condition 1: check X\nvalue 5.0 > 3.0 -> met"
        score = _score_judgment_conditions(pred, gt)
        assert score < 0.75  # wrong outcome penalty

    def test_not_met_self_consistent(self):
        gt = "condition 1: check X\nvalue 2.0 < 3.0 -> not met"
        pred = "condition 1: check X\nvalue 2.0 < 3.0 -> not met"
        score = _score_judgment_conditions(pred, gt)
        assert score >= 0.99

    def test_format_c_not_met_vs_met(self):
        """Format C: direct computation with 'not met' vs 'met'."""
        gt = "ratio = 0.5\n0.5 < 1.0 -> not met"
        pred = "ratio = 1.5\n1.5 >= 1.0 -> met"
        score = _score_judgment_conditions(pred, gt)
        assert score < 0.75  # wrong outcome penalty


class TestExtractOutcome:
    """Unit tests for _extract_outcome — negation handling."""

    def test_bare_met(self):
        assert _extract_outcome("value 5.0 > 3.0 -> met") == 'met'

    def test_not_met(self):
        assert _extract_outcome("value 2.0 < 3.0 -> not met") == 'not met'

    def test_pass(self):
        assert _extract_outcome("amplitude check -> pass") == 'pass'

    def test_fail(self):
        assert _extract_outcome("amplitude check -> fail") == 'fail'

    def test_no_outcome(self):
        assert _extract_outcome("just some text here") is None

    def test_not_met_not_confused_with_met(self):
        # This was the critical bug — "not met" was extracted as "met"
        result = _extract_outcome("condition not met")
        assert result == 'not met'
        assert result != 'met'


class TestTypeSimilarityMinLength:
    """Change #10: substring match requires len >= 4."""

    def test_short_substring_no_match(self):
        assert get_type_similarity("no", "no trend") == 0.0

    def test_long_substring_still_matches(self):
        assert get_type_similarity("spike", "upward spike") == 0.4

    def test_short_both_ways(self):
        # "in" should not match "linear increase"
        assert get_type_similarity("in", "linear increase") == 0.0


# ══════════════════════════════════════════════════════════════════════
# Baby-step anti-cliff tests
# ══════════════════════════════════════════════════════════════════════

class TestLocalDetectionFloor:
    """Change 1: wrong-type local events get a detection floor, not 0."""

    def test_all_types_wrong_gets_floor(self):
        gt = [{"type": "upward convex", "position_start": 50.0}]
        pred = [{"type": "downward spike", "position_start": 50.0}]
        score = _score_local_events(pred, gt, 256)
        assert score >= RewardConfig.LOCAL_DETECTION_FLOOR

    def test_correct_count_gets_bonus(self):
        gt = [{"type": "upward convex"}, {"type": "shake"}]
        pred = [{"type": "downward spike"}, {"type": "upward spike"}]
        score = _score_local_events(pred, gt, 256)
        assert score >= RewardConfig.LOCAL_DETECTION_FLOOR + RewardConfig.LOCAL_COUNT_BONUS

    def test_wrong_count_no_bonus(self):
        gt = [{"type": "upward convex"}, {"type": "shake"}]
        pred = [{"type": "downward spike"}]
        score = _score_local_events(pred, gt, 256)
        assert score >= RewardConfig.LOCAL_DETECTION_FLOOR
        # But less than floor + bonus
        score_with_count = _score_local_events(
            [{"type": "downward spike"}, {"type": "upward spike"}], gt, 256
        )
        assert score < score_with_count

    def test_perfect_match_unaffected(self):
        ev = {"type": "upward spike", "amplitude": 5.0, "position_start": 50.0}
        assert _score_local_events([ev], [ev], 256) >= 0.99

    def test_empty_pred_still_zero(self):
        gt = [{"type": "upward spike"}]
        assert _score_local_events([], gt, 256) == 0.0


class TestCategoryKeywordFallback:
    """Change 2: shared event keywords give 0.15 partial credit."""

    def test_shared_spike_keyword(self):
        # "upward spike" vs "decrease after downward spike" — both contain "spike"
        assert get_type_similarity("upward spike", "decrease after downward spike") == 0.15

    def test_shared_sudden_keyword(self):
        assert get_type_similarity("sudden jump", "sudden drop") == 0.15

    def test_no_shared_keywords(self):
        assert get_type_similarity("linear increase", "keep steady") == 0.0

    def test_table_takes_precedence(self):
        # "upward spike" vs "sudden increase" is in the table at 0.6
        assert get_type_similarity("upward spike", "sudden increase") == 0.6


class TestOppositeDirectionSimilarity:
    """Change 3: opposite-direction pairs get 0.10 instead of 0.0."""

    def test_upward_vs_downward_spike(self):
        assert get_type_similarity("upward spike", "downward spike") == 0.10

    def test_sudden_increase_vs_decrease(self):
        assert get_type_similarity("sudden increase", "sudden decrease") == 0.10

    def test_below_same_direction_score(self):
        opp = get_type_similarity("upward spike", "downward spike")
        same = get_type_similarity("upward spike", "sudden increase")
        assert opp < same


class TestSteeperHallucinationPenalty:
    """Change 4: steeper per-event hallucination penalty (0.20 instead of 0.15)."""

    def test_one_hallucinated_event(self):
        score = _score_local_events([{"type": "spike"}], [], 256)
        assert score == pytest.approx(0.80)

    def test_three_hallucinated_events(self):
        pred = [{"type": "a"}, {"type": "b"}, {"type": "c"}]
        score = _score_local_events(pred, [], 256)
        assert score == pytest.approx(0.40)

    def test_five_hallucinated_events(self):
        pred = [{"type": str(i)} for i in range(5)]
        score = _score_local_events(pred, [], 256)
        assert score == pytest.approx(0.0)


class TestUnnecessaryLocalPenalty:
    """Change 5: penalty when GT has no local field but pred hallucinated events."""

    def test_no_gt_local_pred_has_events(self):
        gt_attrs = {"trend": {"type": "increase", "amplitude": 5, "start": 0, "end": 5}}
        pred_attrs = {
            "trend": {"type": "increase", "amplitude": 5, "start": 0, "end": 5},
            "local": [{"type": "spike"}],
        }
        score_with = _score_single_metric(pred_attrs, gt_attrs, 256)
        score_without = _score_single_metric(
            {"trend": {"type": "increase", "amplitude": 5, "start": 0, "end": 5}},
            gt_attrs, 256,
        )
        assert score_with < score_without

    def test_gt_has_local_no_extra_penalty(self):
        # GT has local=[] (explicitly empty) — unnecessary penalty should NOT fire
        gt_attrs = {"local": []}
        pred_attrs = {"local": [{"type": "spike"}]}
        # This goes through _score_local_events (hallucination penalty), not unnecessary
        score = _score_single_metric(pred_attrs, gt_attrs, 256)
        assert score < 0.85  # hallucination penalty from _score_local_events

    def test_no_pred_local_no_penalty(self):
        gt_attrs = {"trend": {"type": "increase", "amplitude": 5, "start": 0, "end": 5}}
        pred_attrs = {"trend": {"type": "increase", "amplitude": 5, "start": 0, "end": 5}}
        score = _score_single_metric(pred_attrs, gt_attrs, 256)
        assert score >= 0.99

    def test_penalty_capped_at_085(self):
        gt_attrs = {"trend": {"type": "increase", "amplitude": 5, "start": 0, "end": 5}}
        pred_attrs = {
            "trend": {"type": "increase", "amplitude": 5, "start": 0, "end": 5},
            "local": [{"type": f"ev{i}"} for i in range(10)],  # many unnecessary
        }
        score = _score_single_metric(pred_attrs, gt_attrs, 256)
        # base * max(0.85, ...) * consistency → at least 0.85 * base
        score_clean = _score_single_metric(
            {"trend": {"type": "increase", "amplitude": 5, "start": 0, "end": 5}},
            gt_attrs, 256,
        )
        assert score >= score_clean * 0.84  # floor is 0.85


class TestSegmentDetectionFloor:
    """Change 6: wrong-type trend segments get a detection floor."""

    def test_wrong_types_get_floor(self):
        gt = [{"type": "increase", "start": 1, "end": 128},
              {"type": "decrease", "start": 129, "end": 256}]
        pred = [{"type": "steady", "start": 1, "end": 256}]
        score = _score_trend_segments(pred, gt, 256)
        assert score >= RewardConfig.SEGMENT_DETECTION_FLOOR

    def test_correct_count_gets_bonus(self):
        gt = [{"type": "increase", "start": 1, "end": 128},
              {"type": "decrease", "start": 129, "end": 256}]
        pred = [{"type": "steady", "start": 1, "end": 128},
                {"type": "steady", "start": 129, "end": 256}]
        score = _score_trend_segments(pred, gt, 256)
        assert score >= RewardConfig.SEGMENT_DETECTION_FLOOR + RewardConfig.SEGMENT_COUNT_BONUS

    def test_perfect_match_unaffected(self):
        segs = [{"type": "increase", "start": 1, "end": 128},
                {"type": "decrease", "start": 129, "end": 256}]
        assert _score_trend_segments(segs, segs, 256) >= 0.99

    def test_empty_pred_still_zero(self):
        gt = [{"type": "increase", "start": 1, "end": 256}]
        assert _score_trend_segments([], gt, 256) == 0.0


class TestVerdictWrongFloor:
    """Change 7: wrong-but-present verdict gets floor instead of 0."""

    def test_verdict_match_wrong_floor(self):
        from reward.parsing import extract_answer_scalar
        gt = "metric: A\n---\nmetric: B\n===\nanswer: positive"
        pred = "metric: A\n---\nmetric: B\n===\nanswer: negative"
        score = score_mts_verdict(pred, gt)
        # Wrong verdict: total capped at WRONG_VERDICT_CAP
        assert score > 0.0
        assert score <= RewardConfig.WRONG_VERDICT_CAP
        # But still less than correct
        assert score < score_mts_verdict(gt, gt)

    def test_yes_no_wrong_floor(self):
        gt = "metric: M\nnoise=smooth, strength=0.00\n===\ntarget: PASS."
        pred = "metric: M\nnoise=smooth, strength=0.00\n===\ntarget: FAIL."
        score = score_yes_no(pred, gt)
        # Should include verdict floor contribution
        assert score >= RewardConfig.W_VERDICT * RewardConfig.VERDICT_WRONG_FLOOR

    def test_trend_dominance_wrong_floor(self):
        gt = "metric: M\n===\nanswer: increase"
        pred = "metric: M\n===\nanswer: decrease"
        score = score_trend_dominance(pred, gt)
        assert score >= 0.60 * RewardConfig.VERDICT_WRONG_FLOOR

    def test_anti_judgment_wrong_floor(self):
        gt = "===\nanswer: yes"
        pred = "===\nanswer: no"
        score = score_anti_judgment(pred, gt)
        assert score >= 0.65 * RewardConfig.VERDICT_WRONG_FLOOR

    def test_compound_judgment_wrong_floor(self):
        gt = "metric: A\n===\nanswer: yes"
        pred = "metric: A\n===\nanswer: no"
        score = score_compound_judgment(pred, gt)
        assert score >= 0.65 * RewardConfig.VERDICT_WRONG_FLOOR

    def test_stat_numerical_categorical_wrong_floor(self):
        gt = "metric: M\nstats: mean=5.0\n===\nanswer: first half"
        pred = "metric: M\nstats: mean=5.0\n===\nanswer: second half"
        score = score_stat_numerical(pred, gt)
        assert score >= 0.80 * RewardConfig.VERDICT_WRONG_FLOOR

    def test_missing_verdict_unchanged(self):
        # Missing verdict should still use PARTIAL_NO_VERDICT, not VERDICT_WRONG_FLOOR
        # But since verdict is wrong (missing != gt), WRONG_VERDICT_CAP also applies
        gt = "metric: A\n---\nmetric: B\n===\nanswer: positive"
        pred = "metric: A\n---\nmetric: B\n==="
        score = score_mts_verdict(pred, gt)
        assert score > 0.0
        assert score <= RewardConfig.WRONG_VERDICT_CAP

    def test_self_consistency_preserved(self):
        think = (
            "metric: M\nnoise=smooth, strength=0.00\n"
            "===\ntarget: PASS.\nanswer: yes"
        )
        assert score_yes_no(think, think) >= 0.99

    def test_wrong_higher_than_zero(self):
        """Wrong verdict should score > 0 but capped at WRONG_VERDICT_CAP."""
        gt = "metric: M\nnoise=smooth, strength=0.00\n===\ntarget: PASS."
        pred = "metric: M\nnoise=smooth, strength=0.00\n===\ntarget: FAIL."
        score = score_yes_no(pred, gt)
        assert score > 0.0
        assert score <= RewardConfig.WRONG_VERDICT_CAP

