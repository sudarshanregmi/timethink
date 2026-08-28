"""Tests for reward/coherency.py — internal reasoning consistency checks."""
import pytest

from reward.coherency import (
    coherency_condition_verdict,
    coherency_duration_dominance,
    coherency_stats_categorical,
    coherency_mts_verdict,
    coherency_mts_cluster,
    coherency_yes_no,
)


# ═══════════════════════════════════════════════════════════════════════
# 1. coherency_condition_verdict
# ═══════════════════════════════════════════════════════════════════════

class TestCoherencyConditionVerdict:
    def test_all_met_yes_consistent(self):
        text = (
            "data\n===\n"
            "condition 1: check X\nvalue -> met\n"
            "condition 2: check Y\nvalue -> met\n"
            "all conditions: all met\nanswer: yes"
        )
        assert coherency_condition_verdict(text) == 1.0

    def test_not_all_met_no_consistent(self):
        text = (
            "data\n===\n"
            "condition 1: check X\nvalue -> met\n"
            "condition 2: check Y\nvalue -> not met\n"
            "all conditions: not all met\nanswer: no"
        )
        assert coherency_condition_verdict(text) == 1.0

    def test_all_met_but_answer_no_detected(self):
        text = (
            "data\n===\n"
            "condition 1: check X\nvalue -> met\n"
            "condition 2: check Y\nvalue -> met\n"
            "all conditions: all met\nanswer: no"
        )
        # Summary says "all met" but answer is "no" — inconsistency detected
        assert coherency_condition_verdict(text) <= 0.5

    def test_unparseable_returns_neutral(self):
        assert coherency_condition_verdict("some random text") == 0.5

    def test_format_c_met_to_yes(self):
        text = "data\n===\nratio = 1.5\n1.5 >= 1.0 -> met\nanswer: yes"
        assert coherency_condition_verdict(text) == 1.0

    def test_format_c_not_met_to_no(self):
        text = "data\n===\nratio = 0.5\n0.5 < 1.0 -> not met\nanswer: no"
        assert coherency_condition_verdict(text) == 1.0


# ═══════════════════════════════════════════════════════════════════════
# 2. coherency_duration_dominance
# ═══════════════════════════════════════════════════════════════════════

class TestCoherencyDurationDominance:
    def test_increase_dominant_consistent(self):
        text = (
            "data\n===\n"
            "aggregate durations:\n"
            "- increase: 150\n"
            "- decrease: 100\n"
            "dominant trend comparison: increase\n"
            "answer: increase"
        )
        assert coherency_duration_dominance(text) == 1.0

    def test_wrong_dominant_inconsistent(self):
        text = (
            "data\n===\n"
            "aggregate durations:\n"
            "- increase: 150\n"
            "- decrease: 100\n"
            "answer: decrease"
        )
        assert coherency_duration_dominance(text) == 0.0

    def test_tie_equal_consistent(self):
        text = (
            "data\n===\n"
            "aggregate durations:\n"
            "- increase: 100\n"
            "- decrease: 100\n"
            "answer: equal"
        )
        assert coherency_duration_dominance(text) == 1.0

    def test_unparseable_returns_neutral(self):
        assert coherency_duration_dominance("data\n===\nno durations here") == 0.5

    def test_summation_format(self):
        text = (
            "data\n===\n"
            "aggregate durations:\n"
            "- increase: 69 + 15 = 84\n"
            "- decrease: 60\n"
            "answer: increase"
        )
        assert coherency_duration_dominance(text) == 1.0


# ═══════════════════════════════════════════════════════════════════════
# 3. coherency_stats_categorical
# ═══════════════════════════════════════════════════════════════════════

class TestCoherencyStatsCategorical:
    def test_first_half_higher_consistent(self):
        text = (
            "stats: first_half_mean=10.00, second_half_mean=5.00\n"
            "===\nanswer: first half"
        )
        assert coherency_stats_categorical(text) == 1.0

    def test_second_half_higher_consistent(self):
        text = (
            "stats: first_half_mean=3.00, second_half_mean=8.00\n"
            "===\nanswer: second half"
        )
        assert coherency_stats_categorical(text) == 1.0

    def test_wrong_half_inconsistent(self):
        text = (
            "stats: first_half_mean=10.00, second_half_mean=5.00\n"
            "===\nanswer: second half"
        )
        assert coherency_stats_categorical(text) == 0.0

    def test_approximately_equal_any_answer_ok(self):
        text = (
            "stats: first_half_mean=10.00, second_half_mean=10.01\n"
            "===\nanswer: first half"
        )
        assert coherency_stats_categorical(text) == 1.0

    def test_no_separator_returns_neutral(self):
        assert coherency_stats_categorical("no separator here") == 0.5


# ═══════════════════════════════════════════════════════════════════════
# 4. coherency_mts_verdict
# ═══════════════════════════════════════════════════════════════════════

class TestCoherencyMTSVerdict:
    def test_pass_similar_consistent(self):
        text = "data\n===\nvs B: PASS.\nanswer: similar"
        assert coherency_mts_verdict(text) == 1.0

    def test_fail_different_consistent(self):
        text = "data\n===\nvs B: FAIL.\nanswer: different"
        assert coherency_mts_verdict(text) == 1.0

    def test_pass_different_inconsistent(self):
        text = "data\n===\nvs B: PASS.\nanswer: different"
        assert coherency_mts_verdict(text) == 0.0

    def test_anti_pass_opposite_consistent(self):
        text = "data\n===\nvs B: PASS.\nanswer: opposite"
        assert coherency_mts_verdict(text) == 1.0

    def test_anti_fail_not_opposite_consistent(self):
        text = "data\n===\nvs B: FAIL.\nanswer: not_opposite"
        assert coherency_mts_verdict(text) == 1.0

    def test_no_vs_line_returns_neutral(self):
        assert coherency_mts_verdict("data\n===\nanswer: similar") == 0.5

    def test_no_separator_returns_neutral(self):
        assert coherency_mts_verdict("just text") == 0.5


# ═══════════════════════════════════════════════════════════════════════
# 5. coherency_mts_cluster
# ═══════════════════════════════════════════════════════════════════════

class TestCoherencyMTSCluster:
    def test_pass_in_answer_consistent(self):
        text = (
            "data\n===\n"
            "vs b: PASS.\n"
            "vs c: FAIL.\n"
            "answer: (a, b)"
        )
        score = coherency_mts_cluster(text)
        # b passed and is in answer (1.0), c failed and is NOT in answer (1.0)
        assert score == 1.0

    def test_pass_not_in_answer_inconsistent(self):
        text = (
            "data\n===\n"
            "vs b: PASS.\n"
            "vs c: FAIL.\n"
            "answer: (a, c)"
        )
        score = coherency_mts_cluster(text)
        # b passed but NOT in answer (0.0), c failed but IS in answer (0.0)
        assert score == 0.0

    def test_no_vs_lines_returns_neutral(self):
        text = "data\n===\nanswer: (a, b)"
        assert coherency_mts_cluster(text) == 0.5

    def test_empty_answer_set_all_fail_coherent(self):
        text = "data\n===\nvs b: FAIL.\nanswer: ()"
        score = coherency_mts_cluster(text)
        # Empty answer set is coherent when all metrics FAIL
        assert score == 1.0


# ═══════════════════════════════════════════════════════════════════════
# 6. coherency_yes_no
# ═══════════════════════════════════════════════════════════════════════

class TestCoherencyYesNo:
    def test_pass_yes_consistent(self):
        text = "data\n===\ntarget: PASS.\nanswer: yes"
        assert coherency_yes_no(text) == 1.0

    def test_fail_no_consistent(self):
        text = "data\n===\ntarget: FAIL.\nanswer: no"
        assert coherency_yes_no(text) == 1.0

    def test_pass_no_inconsistent(self):
        text = "data\n===\ntarget: PASS.\nanswer: no"
        assert coherency_yes_no(text) == 0.0

    def test_fail_yes_inconsistent(self):
        text = "data\n===\ntarget: FAIL.\nanswer: yes"
        assert coherency_yes_no(text) == 0.0

    def test_no_pass_fail_returns_neutral(self):
        text = "data\n===\nsome reasoning\nanswer: yes"
        assert coherency_yes_no(text) == 0.5

    def test_no_separator_returns_neutral(self):
        assert coherency_yes_no("just text with PASS.") == 0.5
