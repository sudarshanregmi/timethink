import glob
import json
import math
import os

from reward.config import RewardConfig
from reward.evidence import set_context_seq_len
from reward.parsing import extract_think_content
from reward.scorers import (
    score_mts_verdict,
    score_mts_set,
    score_enumeration,
    score_yes_no,
    score_description,
    score_trend_dominance,
    score_anti_judgment,
    score_compound_judgment,
    score_stat_numerical,
)
from reward.scorers.verdict_scorers import (
    score_verdict_numeric,
    score_verdict_binary,
    score_verdict_categorical,
    score_verdict_enumeration,
    score_verdict_set,
    score_verdict_polymorphic,
)

# ---------------------------------------------------------------------------
# Taxonomy type → scorer mapping
# Each type routes to the scorer matching its verdict format.
# OOD types are eval-only and never reach reward (safe fallback if they do).
# ---------------------------------------------------------------------------

# Atomic SFT types (single-metric) — taught with full think blocks
_ATOMIC_BINARY = frozenset({
    'atomic_cross_stat_compare', 'atomic_cross_trend_align',
})
_ATOMIC_NUMERIC = frozenset({
    'atomic_global_mean', 'atomic_global_std',
    'atomic_interval_mean', 'atomic_interval_std',
    'atomic_min_value', 'atomic_max_value',
    'atomic_min_position', 'atomic_max_position',
    'atomic_percentile',
    'atomic_event_enumeration', 'atomic_trend_enumeration',
    'atomic_cross_counting',
})
_ATOMIC_CATEGORICAL = frozenset({
    'atomic_cross_ranking', 'atomic_periodic_description',
})
_ATOMIC_ENUMERATION = frozenset({
    'atomic_chunked_means', 'atomic_cross_filtering',
})

# RL composition types (single-metric) — model discovers reasoning via reward
_RL_BINARY = frozenset({
    'rl_amplitude_vs_range', 'rl_amplitude_vs_std', 'rl_condition_recovery',
    'rl_event_in_trend_type', 'rl_event_near_extremum', 'rl_extrema_same_half',
    'rl_half_mean_compare', 'rl_has_periodicity', 'rl_interval_comparison',
    'rl_max_in_first_half', 'rl_max_in_trend_type', 'rl_mean_shift',
    'rl_mean_stability', 'rl_median_mean_close', 'rl_monotonic_chunks',
    'rl_segment_mean_compare', 'rl_volatility_change',
    # Cross-metric binary
    'rl_cross_stat_ratio', 'rl_cross_event_sync',
    'rl_cross_period_compare', 'rl_cross_attribute_corr',
})
_RL_NUMERIC = frozenset({
    'rl_cycle_count', 'rl_event_count', 'rl_event_count_by_type',
    'rl_half_mean_diff', 'rl_longest_segment',
    'rl_normalized_range', 'rl_period_estimate', 'rl_range',
    'rl_segment_count', 'rl_segment_duration', 'rl_type_duration_fraction',
    # Cross-metric numeric
    'rl_cross_trend_concordance', 'rl_cluster_count',
    'rl_cross_asymmetric_behavior',
})
_RL_CATEGORICAL = frozenset({
    'rl_dominant_trend_type', 'rl_event_type_at_pos', 'rl_max_amplitude_event',
    'rl_segment_type_at_pos', 'rl_type_of_longest',
    # Cross-metric categorical
    'rl_cross_conditional_query', 'rl_cluster_dominant', 'rl_corr_count',
})

# Cross-metric types (used across SFT + RL tiers, grouped by scorer)
_CROSS_BINARY = frozenset({
    'atomic_cross_stat_compare', 'atomic_cross_trend_align',
    'rl_cross_stat_ratio', 'rl_cross_event_sync',
    'rl_cross_period_compare', 'rl_cross_attribute_corr',
    'rl_corr_conditional',
})
_CROSS_NUMERIC = frozenset({
    'atomic_cross_counting', 'rl_cross_trend_concordance',
    'rl_cluster_count', 'rl_cross_asymmetric_behavior',
})
_CROSS_CATEGORICAL = frozenset({
    'atomic_cross_ranking', 'rl_cross_conditional_query', 'rl_cluster_dominant',
    'rl_corr_count',
})
_CROSS_ENUMERATION = frozenset({
    'atomic_cross_filtering', 'rl_cross_full_ordering',
})


def compute_score(solution_str: str, ground_truth: str, **kwargs) -> float:
    """
    Reward in [0, 1] comparing the <think> block of solution_str against ground_truth.

    Args:
        solution_str: Model prediction (with <think> block).
        ground_truth: Ground truth output (with <think> block).
        **kwargs: Optional metadata from verl's reward_model dict:
            eval_type: QA type string for direct routing (skips think-block parsing).
            eval_metadata: Dict with sub_type, verdict, seq_len, etc.
    """
    assert isinstance(ground_truth, str), f"ground_truth must be str, got {type(ground_truth)}"

    gt_think = extract_think_content(ground_truth)
    if not gt_think:
        raise ValueError("CRITICAL: Ground truth has no parseable <think> block!")

    pred_think = extract_think_content(solution_str)
    if not pred_think:
        return RewardConfig.PENALTY_PARSE_ERROR

    # eval_type and eval_metadata can come as direct kwargs or inside extra_info
    extra_info = kwargs.get('extra_info', {})
    qa_type = kwargs.get('eval_type') or extra_info.get('eval_type')
    if not qa_type:
        raise ValueError(
            "eval_type must be passed to compute_score (via kwargs or extra_info). "
            "Got kwargs: " + str(list(kwargs.keys()))
        )

    eval_meta = kwargs.get('eval_metadata') or extra_info.get('eval_metadata') or {}
    if not isinstance(eval_meta, dict) or 'length' not in eval_meta:
        raise ValueError(
            "eval_metadata must contain 'length'. "
            f"Got keys: {list(eval_meta.keys()) if isinstance(eval_meta, dict) else type(eval_meta)}"
        )
    set_context_seq_len(eval_meta['length'])

    # --- Type-tag routing (verdict-only, 2026-04-22) ---
    #
    # Every branch below uses a VERDICT-ONLY scorer — reads only the
    # `answer: ...` line, ignores data block / verification / coherency /
    # evidence. Rationale: during RL, the model explores reasoning
    # formats via GRPO, and full-structure scorers (score_yes_no,
    # score_mts_verdict, score_compound_judgment, score_stat_numerical,
    # score_enumeration, score_trend_dominance, score_description) would
    # penalize valid but non-SFT-formatted reasoning — conflating format
    # adherence with capability and producing misleading val curves.
    # Legacy types don't appear in train_rl.parquet (force-SFT routing),
    # so changing these scorers affects val metrics only, not gradients.
    # For structure-quality analysis use the evaluation/ pipeline or call
    # the legacy scorers directly offline; they remain importable.
    # See memory/reward_verdict_only_val_monitoring.md for full rationale.
    if qa_type == 'stat_numerical':
        # Polymorphic: answer may be float, int, categorical, or list.
        raw = score_verdict_polymorphic(pred_think, gt_think)
    elif qa_type in ('correlation', 'anticorrelation'):
        raw = score_verdict_categorical(pred_think, gt_think)
    elif qa_type in ('clustering', 'anticlustering'):
        raw = score_verdict_set(pred_think, gt_think)
    elif qa_type == 'cross_stat_judgment':
        raw = score_verdict_binary(pred_think, gt_think)
    elif qa_type == 'anti_judgment':
        raw = score_verdict_binary(pred_think, gt_think)
    elif qa_type in ('local_enumeration', 'segment_enumeration', 'transition_enumeration', 'event_segment_enumeration', 'temporal_position', 'duration_proportion', 'cross_metric_enumeration'):
        raw = score_verdict_enumeration(pred_think, gt_think)
    elif qa_type in ('cross_trend_query', 'segment_trend_dominance'):
        raw = score_verdict_categorical(pred_think, gt_think)
    elif qa_type == 'segment_judgment':
        raw = score_verdict_binary(pred_think, gt_think)
    elif qa_type == 'yes_no':
        raw = score_verdict_binary(pred_think, gt_think)
    elif qa_type in ('description', 'tsevol'):
        # Free-form text types. The reward function is rule-based and has
        # no meaningful verdict to extract for these. The evaluation/
        # pipeline uses LLM-as-judge for description quality — that's the
        # right place to measure these. Here we return neutral 0.5 so
        # val curves don't move based on unmeasurable signal.
        raw = 0.5
    elif qa_type == 'periodicity':
        # Polymorphic: answer may be float (period) or categorical (e.g. "none").
        raw = score_verdict_polymorphic(pred_think, gt_think)
    elif qa_type == 'change_point':
        # Polymorphic: answer may be int position or list of positions.
        raw = score_verdict_polymorphic(pred_think, gt_think)
    # --- OOD types: eval-only, never trained via RL ---
    # OOD samples should not reach compute_score during training.
    # Safe fallback if they do (don't crash, return neutral score).
    elif qa_type.startswith('ood_'):
        raw = 0.5
    # --- Taxonomy: atomic SFT types (force-SFT, kept for val monitoring) ---
    elif qa_type in _ATOMIC_BINARY:
        raw = score_verdict_binary(pred_think, gt_think)
    elif qa_type in _ATOMIC_NUMERIC:
        raw = score_verdict_numeric(pred_think, gt_think)
    elif qa_type in _ATOMIC_CATEGORICAL:
        raw = score_verdict_categorical(pred_think, gt_think)
    elif qa_type in _ATOMIC_ENUMERATION:
        raw = score_verdict_enumeration(pred_think, gt_think)
    # --- Taxonomy: RL composition types (single-metric) ---
    elif qa_type in _RL_BINARY:
        raw = score_verdict_binary(pred_think, gt_think)
    elif qa_type in _RL_NUMERIC:
        raw = score_verdict_numeric(pred_think, gt_think)
    elif qa_type in _RL_CATEGORICAL:
        raw = score_verdict_categorical(pred_think, gt_think)
    # --- Taxonomy: cross-metric types (SFT + RL) ---
    elif qa_type in _CROSS_BINARY:
        raw = score_verdict_binary(pred_think, gt_think)
    elif qa_type in _CROSS_NUMERIC:
        raw = score_verdict_numeric(pred_think, gt_think)
    elif qa_type in _CROSS_CATEGORICAL:
        raw = score_verdict_categorical(pred_think, gt_think)
    elif qa_type in _CROSS_ENUMERATION:
        raw = score_verdict_enumeration(pred_think, gt_think)
    else:
        raise ValueError(f"Unknown eval_type: {qa_type!r}")

    return 0.0 if not math.isfinite(raw) else max(0.0, min(1.0, raw))


class VerificationSuite:
    def __init__(self, data_dir: str = "./data"):
        self.data_dir = data_dir
        self.files = glob.glob(os.path.join(data_dir, "**", "*.jsonl"), recursive=True)
        self.stats = {
            "total_files": 0,
            "total_lines": 0,
            "broken_gt": 0,
            "perfect_reconstructions": 0,
            "unit_tests_passed": 0,
            "unit_tests_failed": 0,
        }

    # ------------------------------------------------------------------
    # Unit tests: scorer edge cases
    # ------------------------------------------------------------------

    def _run_unit_tests(self):
        """Run scorer-level unit tests. Returns (passed, failed, details)."""
        from reward.scorers.statistical import (
            _parse_float_verdict, _parse_list_verdict, score_stat_numerical,
        )
        from reward.scorers.segments import (
            score_compound_judgment, _score_judgment_conditions,
        )
        from reward.scorers.core import _score_condition_steps
        from reward.coherency import (
            coherency_condition_verdict,
            coherency_duration_dominance,
            coherency_stats_categorical,
            coherency_mts_verdict,
            coherency_mts_cluster,
            coherency_yes_no,
        )

        # Scorers that call score_evidence_quality need seq_len context
        set_context_seq_len(256)

        passed = failed = 0
        details = []

        def check(name, condition):
            nonlocal passed, failed
            if condition:
                passed += 1
            else:
                failed += 1
                details.append(name)

        # --- NaN / Inf rejection ---
        check("parse_float rejects nan", _parse_float_verdict("nan") is None)
        check("parse_float rejects NaN", _parse_float_verdict("NaN") is None)
        check("parse_float rejects inf", _parse_float_verdict("inf") is None)
        check("parse_float rejects -inf", _parse_float_verdict("-inf") is None)
        check("parse_float accepts 3.14", _parse_float_verdict("3.14") is not None)
        check("parse_list rejects [nan]", _parse_list_verdict("[1.0, nan, 2.0]") is None)
        check("parse_list rejects [inf]", _parse_list_verdict("[inf]") is None)
        check("parse_list accepts [1,2]", _parse_list_verdict("[1.0, 2.0]") is not None)

        # NaN verdict → zero reward (not perfect score)
        gt = "stats: mean=5.000\n===\nanswer: 5.000"
        pred_nan = "stats: mean=5.000\n===\nanswer: nan"
        check("nan verdict < 0.5", score_stat_numerical(pred_nan, gt) < 0.5)

        # --- Compound judgment: min=/max= routing ---
        # min=/max= in DATA block should NOT trigger stat_threshold scorer
        gt_ratio = (
            "metric: A\nstats: max=10.000, min=2.000\n---\n"
            "metric: B\nstats: max=8.000, min=3.000\n===\n"
            "A range = 10.000 - 2.000 = 8.000\n"
            "B range = 8.000 - 3.000 = 5.000\n"
            "ratio = 8.000 / 5.000 = 1.600\n"
            "1.600 >= 1.50 -> met\nanswer: yes"
        )
        check("data-block min/max self-consistency",
              score_compound_judgment(gt_ratio, gt_ratio) >= 0.99)

        # --- Word boundary: 'bypass' should NOT match \bpass\b ---
        bypass_text = (
            "condition 1: check value\nresult bypass the threshold\n"
            "condition 2: noise check\nvalue met\n"
            "all conditions: all met"
        )
        bypass_score = _score_judgment_conditions(bypass_text)
        check("bypass no false positive (< 1.0 for cond 1)",
              bypass_score < 1.0)

        # Real pass/fail SHOULD match
        real_text = (
            "condition 1: trend check\ntrend is increasing -> pass\n"
            "condition 2: noise check\nnoise level high -> fail\n"
            "all conditions: not all met"
        )
        check("real pass/fail matches", _score_judgment_conditions(real_text) >= 0.99)

        # --- Format A/B/C scoring ---
        fmt_a = (
            "condition 1: check X\nvalue 5.0 > 3.0 -> met\n"
            "condition 2: check Y\nvalue 2.0 <= 4.0 -> met\n"
            "all conditions: all met"
        )
        check("format A scoring", _score_judgment_conditions(fmt_a) >= 0.99)

        fmt_b = "event 1: spike at 50, pass\nevent 2: dip at 100, fail\nsummary: 1 of 2 pass"
        check("format B scoring", _score_judgment_conditions(fmt_b) >= 0.99)

        fmt_c = "ratio = 1.600 / 1.000 = 1.600\n1.600 >= 1.50 -> met"
        check("format C met scoring", _score_judgment_conditions(fmt_c) >= 0.99)

        # Format C with -> pass/fail (sequential_events format)
        fmt_c_pass = "pair (spike at 10, dip at 20): gap = 20 - 10 = 10, 10 <= 15 -> pass"
        check("format C pass scoring", _score_judgment_conditions(fmt_c_pass) >= 0.99)

        fmt_c_fail = "pair (spike at 10, dip at 200): gap = 200 - 10 = 190, 190 > 15 -> fail"
        check("format C fail scoring", _score_judgment_conditions(fmt_c_fail) >= 0.99)

        # Format C with (pass)/(fail) — windowed_monotonicity format
        fmt_c_parens = "-31.848 >= -31.929 (fail)"
        check("format C (fail) scoring", _score_judgment_conditions(fmt_c_parens) >= 0.99)

        # --- 1-condition vs 2-condition ---
        single = "condition 1: trend opposite\nseg 1: increase vs decrease -> met"
        check("1-cond no summary = 1.0",
              _score_condition_steps(single, single) >= 0.99)

        two_with = "condition 1: X\ncondition 2: Y\nall conditions: all met"
        check("2-cond with summary = 1.0",
              _score_condition_steps(two_with, two_with) >= 0.99)

        two_without = "condition 1: X\ncondition 2: Y"
        check("2-cond missing summary < 1.0",
              _score_condition_steps(two_without, two_with) < 1.0)

        # --- Coherency: condition-verdict chain ---
        check("coh: all met -> yes = coherent",
              coherency_condition_verdict(
                  "===\ncondition 1: check X\n5.0 > 3.0 -> met\n"
                  "condition 2: check Y\n2.0 > 1.0 -> met\n"
                  "all conditions: all met\nanswer: yes") >= 0.99)
        check("coh: all met but summary + answer both wrong",
              coherency_condition_verdict(
                  "===\ncondition 1: check X\n5.0 > 3.0 -> met\n"
                  "condition 2: check Y\n2.0 > 1.0 -> met\n"
                  "all conditions: not all met\nanswer: yes") < 0.5)
        check("coh: not met -> no = coherent",
              coherency_condition_verdict(
                  "===\ncondition 1: check X\n5.0 > 3.0 -> met\n"
                  "condition 2: check Y\n0.5 <= 1.0 -> not met\n"
                  "all conditions: not all met\nanswer: no") >= 0.99)
        check("coh: not met but summary + answer both wrong",
              coherency_condition_verdict(
                  "===\ncondition 1: check X\n5.0 > 3.0 -> met\n"
                  "condition 2: check Y\n0.5 <= 1.0 -> not met\n"
                  "all conditions: all met\nanswer: no") < 0.5)
        check("coh: single cond met -> yes = coherent",
              coherency_condition_verdict(
                  "===\ncondition 1: trend check\nincrease -> met\nanswer: yes") >= 0.99)
        check("coh: format C met -> yes = coherent",
              coherency_condition_verdict(
                  "===\nratio = 1.600\n1.600 >= 1.50 -> met\nanswer: yes") >= 0.99)

        # --- Coherency: duration-dominance chain ---
        check("coh: max increase -> answer increase = coherent",
              coherency_duration_dominance(
                  "===\naggregate durations:\n- increase: 144\n- decrease: 62\n"
                  "dominant trend comparison: increase (144) > decrease (62)\n"
                  "answer: increase") >= 0.99)
        check("coh: max decrease -> answer increase = incoherent",
              coherency_duration_dominance(
                  "===\naggregate durations:\n- increase: 62\n- decrease: 144\n"
                  "answer: increase") < 0.5)

        # --- Coherency: stats-categorical chain ---
        check("coh: first > second -> first half = coherent",
              coherency_stats_categorical(
                  "stats: first_half_mean=8.68, second_half_mean=8.27\n"
                  "===\n8.68 > 8.27\nanswer: first half") >= 0.99)
        check("coh: first > second -> second half = incoherent",
              coherency_stats_categorical(
                  "stats: first_half_mean=8.68, second_half_mean=5.20\n"
                  "===\nanswer: second half") < 0.5)

        # --- Coherency: MTS verdict (correlation/anticorrelation) ---
        check("coh: corr PASS -> similar = coherent",
              coherency_mts_verdict(
                  "===\nvs MetricB: |10-10|=0 <= 5. PASS.\nanswer: similar") >= 0.99)
        check("coh: corr FAIL -> different = coherent",
              coherency_mts_verdict(
                  "===\nvs MetricB: |100-10|=90 > 5. FAIL.\nanswer: different") >= 0.99)
        check("coh: corr PASS -> different = incoherent",
              coherency_mts_verdict(
                  "===\nvs MetricB: |10-10|=0 <= 5. PASS.\nanswer: different") < 0.5)
        check("coh: anticorr PASS -> opposite = coherent",
              coherency_mts_verdict(
                  "===\nvs MetricB: opposite OK. PASS.\nanswer: opposite") >= 0.99)
        check("coh: anticorr FAIL -> not_opposite = coherent",
              coherency_mts_verdict(
                  "===\nvs MetricB: not opposite. FAIL.\nanswer: not_opposite") >= 0.99)

        # --- Coherency: MTS cluster (clustering/anticlustering) ---
        check("coh: cluster PASS in answer = coherent",
              coherency_mts_cluster(
                  "===\nvs MetricB: ... PASS.\nvs MetricC: ... FAIL.\n"
                  "answer: (Anchor, MetricB)") >= 0.99)
        check("coh: cluster FAIL in answer = incoherent",
              coherency_mts_cluster(
                  "===\nvs MetricB: ... PASS.\nvs MetricC: ... FAIL.\n"
                  "answer: (Anchor, MetricC)") < 0.99)

        # --- Coherency: yes_no PASS/FAIL → answer chain ---
        check("coh: PASS + answer yes = coherent",
              coherency_yes_no(
                  "===\nMetric: |477-471|=6 <= 59. PASS.\nanswer: yes") >= 0.99)
        check("coh: FAIL + answer no = coherent",
              coherency_yes_no(
                  "===\nMetric: |477-471|=6 > 3. FAIL.\nanswer: no") >= 0.99)
        check("coh: PASS + answer no = incoherent",
              coherency_yes_no(
                  "===\nMetric: |477-471|=6 <= 59. PASS.\nanswer: no") < 0.5)
        check("coh: FAIL + answer yes = incoherent",
              coherency_yes_no(
                  "===\nMetric: |477-471|=6 > 3. FAIL.\nanswer: yes") < 0.5)
        check("coh: yes_no unparseable = 0.5",
              abs(coherency_yes_no("===\nsome random text") - 0.5) < 0.01)

        # --- Coherency: neutral defaults ---
        check("coh: unparseable cond = 0.5",
              abs(coherency_condition_verdict("random garbage text") - 0.5) < 0.01)
        check("coh: empty duration = 0.5",
              abs(coherency_duration_dominance("") - 0.5) < 0.01)
        check("coh: no stats = 0.5",
              abs(coherency_stats_categorical("just some text") - 0.5) < 0.01)

        # --- Unknown / missing eval_type ---
        try:
            compute_score("<think>t</think>", "<think>t</think>",
                          eval_type="nonexistent", eval_metadata={"length": 256})
            check("unknown eval_type raises", False)
        except ValueError:
            check("unknown eval_type raises", True)

        try:
            compute_score("<think>t</think>", "<think>t</think>")
            check("missing eval_type raises", False)
        except ValueError:
            check("missing eval_type raises", True)

        return passed, failed, details

    # ------------------------------------------------------------------
    # Self-consistency: compute_score(gt, gt) >= 0.99
    # ------------------------------------------------------------------

    def run(self):
        # --- Unit tests ---
        print("Running unit tests...")
        ut_passed, ut_failed, ut_details = self._run_unit_tests()
        self.stats["unit_tests_passed"] = ut_passed
        self.stats["unit_tests_failed"] = ut_failed
        if ut_failed:
            for d in ut_details:
                print(f"   FAIL: {d}")
        print(f"   Unit tests: {ut_passed} passed, {ut_failed} failed\n")

        # --- Self-consistency on JSONL files ---
        print(f"Verifying Reward Model in: {self.data_dir}")
        for filepath in self.files:
            print(f"   Processing {os.path.basename(filepath)}...", end="", flush=True)
            file_ok = file_fail = 0

            with open(filepath, 'r', encoding='utf-8') as f:
                for line in f:
                    self.stats["total_lines"] += 1
                    try:
                        data = json.loads(line)
                        output_str = data.get("output", "")
                        reward_kwargs = {}
                        if "eval_type" in data:
                            reward_kwargs["eval_type"] = data["eval_type"]
                        if "eval_metadata" in data:
                            reward_kwargs["eval_metadata"] = data["eval_metadata"]
                        score = compute_score(output_str, output_str, **reward_kwargs)
                        if score >= 0.99:
                            file_ok += 1
                            self.stats["perfect_reconstructions"] += 1
                        else:
                            file_fail += 1
                    except (ValueError, AssertionError):
                        self.stats["broken_gt"] += 1
                        file_fail += 1
                    except json.JSONDecodeError:
                        file_fail += 1

            status = (
                f" OK ({file_ok} samples)"
                if file_fail == 0
                else f" WARN ({file_ok} pass, {file_fail} fail)"
            )
            print(status)
            self.stats["total_files"] += 1

        print("\n" + "=" * 60)
        print("VERIFICATION RESULTS")
        print("=" * 60)
        print(f"Unit Tests:                 {self.stats['unit_tests_passed']} passed, {self.stats['unit_tests_failed']} failed")
        print(f"Files Scanned:              {self.stats['total_files']}")
        print(f"Total Lines:                {self.stats['total_lines']}")
        print(f"Critical GT Failures:       {self.stats['broken_gt']}")
        print(f"Self-Consistency Successes: {self.stats['perfect_reconstructions']}")
        valid = self.stats['total_lines'] - self.stats['broken_gt']
        if valid > 0:
            rate = self.stats['perfect_reconstructions'] / valid * 100
            print(f"Self-Consistency Rate:      {rate:.2f}%")
        if self.stats['broken_gt'] > 0 or self.stats['unit_tests_failed'] > 0:
            print("\nCRITICAL: Issues found!")
        else:
            print("\nDATASET INTEGRITY CONFIRMED")
        print("=" * 60)


if __name__ == "__main__":
    verifier = VerificationSuite(data_dir="./think_data")
    verifier.run()
