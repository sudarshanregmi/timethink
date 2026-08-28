"""
Tests for evaluation pipeline: aggregation helpers, parser logic,
results_to_latex round-trip, and ALL_EVAL_TYPES completeness.

Covers gaps identified in the 2026-03-23 eval pipeline audit.
"""
import math
import re
import tempfile
from pathlib import Path

import pandas as pd
import pytest


# ===================================================================
# SECTION 1: _as_list edge cases
# ===================================================================

class TestAsList:
    @staticmethod
    def _as_list(val):
        from evaluation.eval.aggregation import _as_list
        return _as_list(val)

    def test_actual_list(self):
        assert self._as_list(["A", "B"]) == ["A", "B"]

    def test_string_serialized_list(self):
        assert self._as_list("['A', 'B']") == ["A", "B"]

    def test_none(self):
        assert self._as_list(None) == []

    def test_nan(self):
        assert self._as_list(float("nan")) == []

    def test_empty_string(self):
        assert self._as_list("") == []

    def test_empty_list_string(self):
        assert self._as_list("[]") == []

    def test_malformed_string(self):
        assert self._as_list("not a list") == []

    def test_nested_list_returns_empty(self):
        """Nested lists would crash set() downstream — must return []."""
        assert self._as_list("[[1,2],[3,4]]") == []

    def test_numeric_list_returns_empty(self):
        """Entity lists must be strings, not ints."""
        assert self._as_list("[1, 2, 3]") == []

    def test_entity_with_quotes(self):
        assert self._as_list("['Metric A', \"Metric B\"]") == ["Metric A", "Metric B"]

    def test_entity_with_comma_in_name(self):
        assert self._as_list("['Metric A, B']") == ["Metric A, B"]

    def test_actual_empty_list(self):
        assert self._as_list([]) == []


# ===================================================================
# SECTION 2: _compute_micro_f1 edge cases
# ===================================================================

class TestComputeMicroF1:
    @staticmethod
    def _compute(df):
        from evaluation.eval.aggregation import _compute_micro_f1
        return _compute_micro_f1(df)

    def test_all_empty_entities(self):
        """Both gt and pred empty for all rows → perfect score."""
        df = pd.DataFrame({"gt_entities": [[], []], "pred_entities": [[], []]})
        mp, mr, mf1 = self._compute(df)
        assert (mp, mr, mf1) == (1.0, 1.0, 1.0)

    def test_mixed_with_none(self):
        """None rows should be skipped."""
        df = pd.DataFrame({
            "gt_entities": [["A", "B"], None, ["C"]],
            "pred_entities": [["A"], None, ["C", "D"]],
        })
        mp, mr, mf1 = self._compute(df)
        # row0: tp=1, gt=2, pred=1; row2: tp=1, gt=1, pred=2
        # total: tp=2, gt=3, pred=3 → p=2/3, r=2/3, f1=2/3
        assert mp == round(2 / 3, 4)
        assert mr == round(2 / 3, 4)

    def test_from_csv_strings(self):
        """String-serialized lists (from pd.read_csv) should parse correctly."""
        df = pd.DataFrame({
            "gt_entities": ["['A', 'B']", "['C']"],
            "pred_entities": ["['A']", "['C', 'D']"],
        })
        mp, mr, mf1 = self._compute(df)
        assert mp == round(2 / 3, 4)

    def test_empty_dataframe(self):
        df = pd.DataFrame({"gt_entities": [], "pred_entities": []})
        assert self._compute(df) == (None, None, None)

    def test_missing_columns(self):
        df = pd.DataFrame({"some_col": [1, 2]})
        assert self._compute(df) == (None, None, None)

    def test_all_nan_entities(self):
        """All rows have NaN entities → no valid rows → None."""
        df = pd.DataFrame({
            "gt_entities": [float("nan"), float("nan")],
            "pred_entities": [float("nan"), float("nan")],
        })
        assert self._compute(df) == (None, None, None)

    def test_zero_overlap(self):
        """No overlap → micro F1 = 0."""
        df = pd.DataFrame({
            "gt_entities": [["A", "B"]],
            "pred_entities": [["C", "D"]],
        })
        mp, mr, mf1 = self._compute(df)
        assert mf1 == 0.0


# ===================================================================
# SECTION 3: Reasoning score zeroing (parser.py)
# ===================================================================

class TestReasoningScoreZeroing:
    """Verify that wrong answers zero reasoning_score for applicable types."""

    @staticmethod
    def _make_results(task_type, gt_entities, pred_entities, gt_verdict, pred_verdict,
                      reasoning_score=0.8):
        """Build minimal raw_results for parse_and_score."""
        import json
        eval_metadata = {}
        if task_type in ("clustering", "anticlustering"):
            eval_metadata["verdict"] = ""
        judge_json = json.dumps({"score": reasoning_score, "reasoning": "good"})
        return [
            (
                {
                    "idx": 1,
                    "question": "test question",
                    "type": "reasoning_judge",
                    "eval_type": task_type,
                    "eval_metadata": eval_metadata,
                    "pred_answer_text": None,
                },
                judge_json,
            ),
            (
                {
                    "idx": 1,
                    "question": "test question",
                    "type": "gt",
                    "eval_type": task_type,
                    "eval_metadata": eval_metadata,
                    "gt_answer_text": "",
                    "pred_answer_text": "",
                },
                json.dumps({
                    "related_entities": gt_entities,
                    "verdict": gt_verdict,
                }),
            ),
            (
                {
                    "idx": 1,
                    "question": "test question",
                    "type": "pred",
                    "eval_type": task_type,
                    "eval_metadata": eval_metadata,
                    "gt_answer_text": "",
                    "pred_answer_text": "",
                },
                json.dumps({
                    "related_entities": pred_entities,
                    "verdict": pred_verdict,
                }),
            ),
        ]

    def test_clustering_f1_zero_zeros_reasoning(self):
        from evaluation.eval.parser import parse_and_score
        raw = self._make_results("clustering", ["A"], ["B"], True, True)
        rows = parse_and_score(raw)
        assert rows[0]["f1"] == 0.0
        assert rows[0]["reasoning_score"] == 0.0

    def test_clustering_f1_nonzero_keeps_reasoning(self):
        from evaluation.eval.parser import parse_and_score
        raw = self._make_results("clustering", ["A"], ["A"], True, True)
        rows = parse_and_score(raw)
        assert rows[0]["f1"] == 1.0
        assert rows[0]["reasoning_score"] == 0.8

    def test_correlation_wrong_verdict_zeros_reasoning(self):
        from evaluation.eval.parser import parse_and_score
        raw = self._make_results("correlation", [], [], True, False)
        rows = parse_and_score(raw)
        assert rows[0]["binary_accuracy"] == 0.0
        assert rows[0]["reasoning_score"] == 0.0

    def test_correlation_correct_verdict_keeps_reasoning(self):
        from evaluation.eval.parser import parse_and_score
        raw = self._make_results("correlation", [], [], True, True)
        rows = parse_and_score(raw)
        assert rows[0]["binary_accuracy"] == 1.0
        assert rows[0]["reasoning_score"] == 0.8

    def test_anticlustering_f1_zero_zeros_reasoning(self):
        from evaluation.eval.parser import parse_and_score
        raw = self._make_results("anticlustering", ["X"], ["Y"], True, True)
        rows = parse_and_score(raw)
        assert rows[0]["f1"] == 0.0
        assert rows[0]["reasoning_score"] == 0.0


# ===================================================================
# SECTION 4: results_to_latex round-trip parsing
# ===================================================================

class TestResultsToLatexParsing:
    """Verify parse_summary correctly reads write_summary_txt output."""

    @staticmethod
    def _round_trip(df):
        from evaluation.eval.aggregation import write_summary_txt
        import sys
        import os
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts", "utils"))
        from results_to_latex import parse_summary

        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False, mode="w") as f:
            path = f.name
        try:
            write_summary_txt(df, path)
            return parse_summary(Path(path))
        finally:
            Path(path).unlink()

    @pytest.fixture
    def sample_df(self):
        return pd.DataFrame([
            {"idx": 1, "task_type": "clustering", "f1": 0.5, "precision": 0.5,
             "recall": 0.5, "binary_accuracy": None, "reasoning_score": 0.8,
             "gt_entities": ["A", "B"], "pred_entities": ["A"], "gt_len": 2},
            {"idx": 2, "task_type": "clustering", "f1": 1.0, "precision": 1.0,
             "recall": 1.0, "binary_accuracy": None, "reasoning_score": 0.9,
             "gt_entities": ["C"], "pred_entities": ["C"], "gt_len": 1},
            {"idx": 3, "task_type": "correlation", "f1": None, "precision": None,
             "recall": None, "binary_accuracy": 1.0, "reasoning_score": 0.7,
             "gt_entities": None, "pred_entities": None, "gt_len": None},
            {"idx": 4, "task_type": "yes_no", "f1": None, "precision": None,
             "recall": None, "binary_accuracy": 0.0, "reasoning_score": 0.0,
             "gt_entities": None, "pred_entities": None, "gt_len": None},
        ])

    def test_main_table_columns_correct(self, sample_df):
        """micro_p must NOT be confused with description_overall."""
        result = self._round_trip(sample_df)
        clust = result["main"]["clustering"]
        assert abs(clust["f1"] - 0.75) < 0.01
        assert abs(clust["reasoning_score"] - 0.85) < 0.01
        assert clust["count"] == 2.0
        assert abs(clust["micro_p"] - 1.0) < 0.01
        assert abs(clust["micro_r"] - 0.6667) < 0.01

    def test_main_table_non_clustering_has_nan_micro(self, sample_df):
        result = self._round_trip(sample_df)
        corr = result["main"]["correlation"]
        assert math.isnan(corr["micro_p"])

    def test_clustering_by_len_parsed(self, sample_df):
        result = self._round_trip(sample_df)
        assert 1 in result["clustering_by_len"]
        assert 2 in result["clustering_by_len"]
        clu1 = result["clustering_by_len"][1]
        assert abs(clu1["f1"] - 1.0) < 0.01
        assert clu1["sample_count"] == 1

    def test_backward_compat_no_micro_columns(self):
        """Old summary files without micro columns parse correctly."""
        import sys
        import os
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts", "utils"))
        from results_to_latex import parse_summary

        old_summary = (
            "\n==================================================\n"
            "FINAL EVALUATION SUMMARY\n"
            "==================================================\n"
            "               f1  binary_accuracy  reasoning_score  count\n"
            "task_type\n"
            "clustering   0.75              NaN             0.85      2\n"
            "correlation   NaN              1.0             0.70      1\n"
            "\n==================================================\n"
            "CLUSTERING INSIGHTS BY GT LENGTH\n"
            "==================================================\n"
            "No clustering tasks found.\n"
            "\n==================================================\n"
            "REASONING SCORE BREAKDOWN BY TASK TYPE\n"
            "==================================================\n"
        )
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False, mode="w") as f:
            f.write(old_summary)
            path = f.name
        try:
            result = parse_summary(Path(path))
            clust = result["main"]["clustering"]
            assert abs(clust["f1"] - 0.75) < 0.01
            assert "micro_p" not in clust
        finally:
            Path(path).unlink()


# ===================================================================
# SECTION 5: ALL_EVAL_TYPES completeness
# ===================================================================

class TestAllEvalTypes:
    def test_all_eval_types_covers_reward_routing(self):
        """ALL_EVAL_TYPES must contain every type routed in reward/__init__.py."""
        from evaluation.eval.config import ALL_EVAL_TYPES

        src = Path(__file__).resolve().parent.parent / "reward" / "__init__.py"
        text = src.read_text()
        reward_types = set()
        for m in re.finditer(r"qa_type\s*==\s*'([^']+)'", text):
            reward_types.add(m.group(1))
        for m in re.finditer(r"qa_type\s+in\s+\(([^)]+)\)", text):
            for t in re.findall(r"'([^']+)'", m.group(1)):
                reward_types.add(t)
        # Expand prefix patterns (e.g. rl_* → check at least one match)
        prefix_types = set()
        for m in re.finditer(r"qa_type\.startswith\(\(([^)]+)\)", text):
            for prefix in re.findall(r"'([^']+)'", m.group(1)):
                matching = {t for t in ALL_EVAL_TYPES if t.startswith(prefix)}
                assert matching, f"prefix '{prefix}*' has no match in ALL_EVAL_TYPES"
                prefix_types |= matching

        missing = reward_types - ALL_EVAL_TYPES
        assert not missing, f"eval_types in reward routing but NOT in ALL_EVAL_TYPES: {missing}"

    def test_all_eval_types_is_superset_of_ragas(self):
        from evaluation.eval.config import ALL_EVAL_TYPES, RAGAS_QUESTION_BY_TASK
        assert set(RAGAS_QUESTION_BY_TASK.keys()).issubset(ALL_EVAL_TYPES)

    def test_description_and_tsevol_in_all_eval_types(self):
        from evaluation.eval.config import ALL_EVAL_TYPES
        assert "description" in ALL_EVAL_TYPES
        assert "tsevol" in ALL_EVAL_TYPES

# ===================================================================
# SECTION 6: List-valued verdict scoring
# ===================================================================

class TestNumberListExtraction:
    @staticmethod
    def _extract(text):
        from evaluation.eval.parser import _extract_number_list
        return _extract_number_list(text)

    def test_float_list(self):
        assert self._extract('[-14.48, -7.75, 3.19]') == [-14.48, -7.75, 3.19]

    def test_int_list(self):
        assert self._extract('[106, 151]') == [106.0, 151.0]

    def test_single_element(self):
        assert self._extract('[42]') == [42.0]

    def test_empty_list(self):
        assert self._extract('[]') == []

    def test_no_brackets_returns_none(self):
        assert self._extract('3.14') is None

    def test_embedded_in_sentence(self):
        # Model might answer 'the values are [1.0, 2.0, 3.0].'
        assert self._extract('the values are [1.0, 2.0, 3.0].') == [1.0, 2.0, 3.0]

    def test_non_numeric_element_returns_none(self):
        assert self._extract('[1, banana, 3]') is None

    def test_none_input(self):
        assert self._extract(None) is None


class TestListRelativeAccuracy:
    @staticmethod
    def _acc(pred, gt, seq_len=None):
        from evaluation.eval.parser import _list_relative_accuracy
        return _list_relative_accuracy(pred, gt, seq_len)

    def test_exact_match(self):
        assert self._acc([-14.48, -7.75, 3.19], [-14.48, -7.75, 3.19]) == 1.0

    def test_close_values_high_score(self):
        # Each element near-perfect → average near-perfect
        score = self._acc([-14.5, -7.8, 3.2], [-14.48, -7.75, 3.19])
        assert score > 0.99

    def test_missing_elements_penalized(self):
        # Model outputs only first of 3 — should score ~1/3, not 1.0
        score = self._acc([-14.48], [-14.48, -7.75, 3.19])
        assert 0.3 < score < 0.34

    def test_extra_elements_penalized(self):
        # Model outputs 6 elements for a 3-element GT — extra 3 contribute 0
        score = self._acc([-14.48, -7.75, 3.19, 0.0, 0.0, 0.0],
                          [-14.48, -7.75, 3.19])
        assert 0.49 < score < 0.51

    def test_empty_prediction(self):
        assert self._acc([], [1.0, 2.0]) == 0.0

    def test_positional_list_uses_seq_len(self):
        # change_point: GT=[106, 151], pred=[108, 149], seq_len=256
        # per-element: 1 - 2/256 ≈ 0.992 each → avg ≈ 0.992
        score = self._acc([108, 149], [106, 151], seq_len=256)
        assert abs(score - (1 - 2/256)) < 1e-6


class TestListVerdictIntegration:
    '''Integration: GT list with matching-only-first-element prediction no longer scores 1.0.'''

    @staticmethod
    def _make_raw(task_type, gt_verdict, pred_text, length=256):
        import json
        md = {'verdict': gt_verdict, 'length': length}
        return [
            (
                {'idx': 1, 'question': 'q', 'type': 'reasoning_judge',
                 'eval_type': task_type, 'eval_metadata': md,
                 'pred_answer_text': None},
                json.dumps({'score': 0.8, 'reasoning': 'ok'}),
            ),
            (
                {'idx': 1, 'question': 'q', 'type': 'gt',
                 'eval_type': task_type, 'eval_metadata': md,
                 'gt_answer_text': '', 'pred_answer_text': pred_text},
                json.dumps({'verdict_match': True, 'score': 0.8}),
            ),
            (
                {'idx': 1, 'question': 'q', 'type': 'pred',
                 'eval_type': task_type, 'eval_metadata': md,
                 'gt_answer_text': '', 'pred_answer_text': pred_text},
                json.dumps({'verdict_match': True, 'score': 0.8}),
            ),
        ]

    def test_matching_only_first_element_scored_low(self):
        # Before fix: _extract_first_number would pick -14.48 from both
        # GT and pred, scoring 1.0 despite pred dropping 2/3 of the answer.
        # After fix: element-wise, missing elements score 0 → ~0.33.
        from evaluation.eval.parser import parse_and_score
        raw = self._make_raw(
            'atomic_chunked_means',
            gt_verdict='[-14.48, -7.75, 3.19]',
            pred_text='[-14.48]',
        )
        rows = parse_and_score(raw)
        assert rows[0]['binary_accuracy'] < 0.4

    def test_exact_list_match_scored_perfect(self):
        from evaluation.eval.parser import parse_and_score
        raw = self._make_raw(
            'atomic_chunked_means',
            gt_verdict='[-14.48, -7.75, 3.19]',
            pred_text='The chunk means are [-14.48, -7.75, 3.19].',
        )
        rows = parse_and_score(raw)
        assert rows[0]['binary_accuracy'] == 1.0

    def test_change_point_list_seq_len_scaling(self):
        # GT=[106, 151], pred=[108, 149], seq_len=256.
        # per-element accuracy ≈ 0.992 each, avg ≈ 0.992.
        from evaluation.eval.parser import parse_and_score
        raw = self._make_raw(
            'change_point',
            gt_verdict='[106, 151]',
            pred_text='[108, 149]',
            length=256,
        )
        rows = parse_and_score(raw)
        assert rows[0]['binary_accuracy'] > 0.98

    def test_singleton_numeric_still_works(self):
        # Non-list numeric GT still uses _numeric_relative_accuracy.
        from evaluation.eval.parser import parse_and_score
        raw = self._make_raw(
            'atomic_global_mean',
            gt_verdict='-0.19',
            pred_text='The mean is -0.18.',
        )
        rows = parse_and_score(raw)
        assert rows[0]['binary_accuracy'] > 0.98

