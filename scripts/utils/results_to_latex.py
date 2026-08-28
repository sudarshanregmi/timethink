#!/usr/bin/env python3
"""Convert evaluation results from exp/ folders into LaTeX tables for Overleaf.

Usage:
    python scripts/utils/results_to_latex.py exp/sft exp/rl
    python scripts/utils/results_to_latex.py exp/sft exp/rl -o tables.tex
    python scripts/utils/results_to_latex.py exp/sft exp/rl --names "SFT" "RL"
"""

import argparse
import math
import re
import sys
from pathlib import Path

# ── Task type descriptions ───────────────────────────────────────────────
TASK_DESCRIPTIONS = {
    "anti_judgment": "Two-metric: anticorrelated trends + noise condition?",
    "anticlustering": "Which metrics have trends opposite to the target?",
    "anticorrelation": "Do two metrics show opposite trend patterns?",
    "clustering": "Which metrics cluster with the target (similar trends)?",
    "correlation": "Do two metrics show correlated fluctuations?",
    "cross_metric_enumeration": "Filter/argmax across metrics (e.g.\\ noisiest)",
    "cross_stat_judgment": "Cross-metric statistical condition (ratio, shift)?",
    "cross_trend_query": "When metric A trends X, what does metric B do?",
    "description": "Describe trend, seasonality, noise, local events",
    "local_enumeration": "Count/filter local events by type, direction, amplitude",
    "segment_enumeration": "Count/analyze trend segments (longest, by type, etc.)",
    "segment_judgment": "Multi-condition judgment (trend+noise, phase, etc.)",
    "segment_trend_dominance": "Which trend type dominates in a time range?",
    "stat_numerical": "Compute exact statistic (mean, range, extrema pos, etc.)",
    "yes_no": "Is there a local event at this specific point?",
    # OOD compositional evaluation types
    "ood_conditional_stat": "Statistic restricted to a specific trend type's segments",
    "ood_nested_extrema": "Extreme event amplitude within extreme segment",
    "ood_event_density": "Trend type with highest/lowest event density",
    "ood_conditional_count": "Segments of a type whose mean exceeds a threshold",
    "ood_trend_reversal": "Does dominant trend reverse after largest change point?",
    "ood_range_normalized_amplitude": "Largest event amplitude vs.\\ half the value range?",
    "ood_segment_stat_compare": "First vs.\\ last segment mean comparison",
    # ── Periodicity / change-point families ────────────────────────────────
    "periodicity": "Periodicity question (estimated period, cycle count, regularity)",
    "change_point": "Change-point question (number of behavioral shifts, positions, largest shift)",
    # ── Atomic primitives (single-metric SFT taxonomy) ──────────────────────
    "atomic_global_mean": "Atomic: mean of the full series (chunked averaging)",
    "atomic_chunked_means": "Atomic: per-chunk means across the series",
    "atomic_interval_mean": "Atomic: mean of a specified sub-range",
    "atomic_global_std": "Atomic: standard deviation of the full series",
    "atomic_interval_std": "Atomic: standard deviation of a specified sub-range",
    "atomic_min_value": "Atomic: minimum value",
    "atomic_max_value": "Atomic: maximum value",
    "atomic_min_position": "Atomic: position of the minimum",
    "atomic_max_position": "Atomic: position of the maximum",
    "atomic_percentile": "Atomic: percentile of the series",
    "atomic_trend_enumeration": "Atomic: enumerate all trend segments",
    "atomic_event_enumeration": "Atomic: enumerate all local events",
    "atomic_periodic_description": "Atomic: describe periodic behavior",
    # ── Atomic cross-metric primitives (Category H) ────────────────────────
    "atomic_cross_stat_compare": "Cross: compare a statistic between two metrics (yes/no)",
    "atomic_cross_ranking": "Cross: which metric has the highest/lowest stat",
    "atomic_cross_filtering": "Cross: list metrics satisfying a property",
    "atomic_cross_counting": "Cross: count metrics satisfying a property",
    "atomic_cross_trend_align": "Cross: do two metrics share the same trend direction (yes/no)",
    # ── RL single-metric compositional ─────────────────────────────────────
    "rl_amplitude_vs_range": "RL: largest event amplitude vs.\\ a fraction of the value range (yes/no)",
    "rl_amplitude_vs_std": "RL: largest event amplitude vs.\\ a multiple of the std (yes/no)",
    "rl_condition_recovery": "RL: does the series recover to pre-event level after the largest event (yes/no)",
    "rl_cycle_count": "RL: count of complete periodic cycles",
    "rl_dominant_trend_type": "RL: which trend type covers the largest fraction",
    "rl_event_count": "RL: total number of local events",
    "rl_event_count_by_type": "RL: count of events of a specific type",
    "rl_event_in_trend_type": "RL: event occurs within a specific trend segment (yes/no)",
    "rl_event_near_extremum": "RL: event near global max/min (yes/no)",
    "rl_event_type_at_pos": "RL: event type at a specific position",
    "rl_extrema_same_half": "RL: global max and min in the same half (yes/no)",
    "rl_half_mean_compare": "RL: first-half mean exceeds second-half mean (yes/no)",
    "rl_half_mean_diff": "RL: absolute difference between halves' means",
    "rl_has_periodicity": "RL: does the series have a periodic component (yes/no)",
    "rl_interval_comparison": "RL: mean of one interval exceeds another (yes/no)",
    "rl_longest_segment": "RL: duration of the longest trend segment",
    "rl_max_amplitude_event": "RL: event type of the largest-amplitude event",
    "rl_max_in_first_half": "RL: global max in the first half (yes/no)",
    "rl_max_in_trend_type": "RL: global max within a specific trend type (yes/no)",
    "rl_mean_shift": "RL: mean shifts significantly between halves (yes/no)",
    "rl_mean_stability": "RL: chunked means are stable (low CV) (yes/no)",
    "rl_median_mean_close": "RL: median and mean are within threshold (yes/no)",
    "rl_monotonic_chunks": "RL: chunk means are monotonic (yes/no)",
    "rl_normalized_range": "RL: range normalized by std",
    "rl_period_estimate": "RL: estimated period of the periodic component",
    "rl_range": "RL: value range (max minus min)",
    "rl_segment_count": "RL: count of trend segments",
    "rl_segment_duration": "RL: duration of a specific trend segment",
    "rl_segment_mean_compare": "RL: mean of one segment exceeds another (yes/no)",
    "rl_segment_type_at_pos": "RL: trend type at a specific position",
    "rl_type_duration_fraction": "RL: duration fraction covered by a specific trend type",
    "rl_type_of_longest": "RL: trend type of the longest segment",
    "rl_volatility_change": "RL: volatility changes significantly between halves (yes/no)",
    # ── RL cross-metric compositional ──────────────────────────────────────
    "rl_cross_stat_ratio": "RL cross: stat ratio between two metrics exceeds threshold (yes/no)",
    "rl_cross_full_ordering": "RL cross: rank all metrics by a statistic",
    "rl_cross_event_sync": "RL cross: two metrics have events within a proximity (yes/no)",
    "rl_cross_period_compare": "RL cross: compare periodic components of two metrics (yes/no)",
    "rl_cross_trend_concordance": "RL cross: fraction of metrics in the dominant trend direction",
    "rl_cross_conditional_query": "RL cross: among metrics satisfying a condition, which has extreme stat",
    "rl_cross_attribute_corr": "RL cross: top on one attribute also top on another (yes/no)",
    "rl_cross_asymmetric_behavior": "RL cross: count of metrics with upward (or downward) mean shift",
    "rl_cluster_count": "RL cross: number of distinct trend clusters",
    "rl_cluster_dominant": "RL cross: which cluster has the most metrics",
    "rl_corr_count": "RL cross: most common trend direction across metrics",
    "rl_corr_conditional": "RL cross: two same-trend metrics also have nearby events (yes/no)",
    # ── OOD single-metric (expanded taxonomy, eval-only) ───────────────────
    "ood_max_before_min": "OOD: does global max occur before global min (yes/no)",
    "ood_std_exceeds_half_range": "OOD: std exceeds half the value range (yes/no)",
    "ood_max_in_highest_mean_quarter": "OOD: max falls in the highest-mean quarter (yes/no)",
    "ood_quarter_mean_ordering": "OOD: quarter means follow a monotonic ordering (yes/no)",
    "ood_max_mean_chunk_pos": "OOD: chunk index of the highest-mean chunk",
    "ood_chunk_above_proportion": "OOD: proportion of chunks whose mean exceeds overall mean",
    "ood_symmetric_recovery": "OOD: symmetric recovery around the extremum (yes/no)",
    "ood_cycle_mean_trend": "OOD: cycle-level means show a monotonic trend (yes/no)",
    "ood_trend_follows_mean": "OOD: dominant trend aligns with overall mean shift (yes/no)",
    "ood_conditional_mean_by_type": "OOD: mean of the series restricted to segments of a type",
    "ood_longest_type_fraction": "OOD: duration fraction of the longest trend segment",
    "ood_event_amplitude_vs_std": "OOD: largest event amplitude vs.\\ a multiple of std (yes/no)",
    "ood_amplitude_vs_segment_std": "OOD: event amplitude vs.\\ local-segment std (yes/no)",
    "ood_symmetric_trend_sequence": "OOD: trend-type sequence is palindromic (yes/no)",
    "ood_event_density_by_trend": "OOD: event density within segments of a specific trend type",
    "ood_max_amp_in_longest_segment": "OOD: max event amplitude inside the longest segment",
    # ── OOD cross-metric (expanded taxonomy) ──────────────────────────────
    "ood_cross_corr_count": "OOD cross: number of metric pairs sharing overall trend direction",
    "ood_cross_trend_convergence": "OOD cross: trend agreement stronger in second half than first (yes/no)",
    "ood_cross_extrema_alignment": "OOD cross: any two metrics have max at similar positions (yes/no)",
    "ood_cross_range_overlap": "OOD cross: value ranges of two metrics overlap (yes/no)",
    "ood_cross_concordant_shift": "OOD cross: count of metrics with higher mean in the second half",
    "ood_cluster_singleton": "OOD cross: any metric has a unique trend (yes/no)",
    "ood_corr_transitivity": "OOD cross: trend-sharing transitivity among three metrics (yes/no)",
    "ood_mixed_corr_anti": "OOD cross: count of metric pairs with opposite trends",
    # ── OOD compositional probes (atomic + RL primitive compositions, eval-only) ──
    "ood_peak_in_longest_segment": "OOD probe: does the global max lie inside the longest trend segment (yes/no)",
    "ood_duration_weighted_mean": "OOD probe: duration-weighted mean of per-segment means",
    "ood_dominant_trend_in_cluster": "OOD probe: dominant trend type among upper-half-mean metrics",
    "ood_cross_event_causality": "OOD probe: a local event in metric A precedes one in metric B within a scaled window (yes/no)",
    # ── Legacy MTS families (older taxonomy, still wired) ──────────────────
    "temporal_position": "Temporal position of an event or extremum",
    "duration_proportion": "Proportion of the series covered by a condition",
    "event_segment_enumeration": "Enumerate local events conditioned on a trend segment",
    "transition_enumeration": "Enumerate trend-type transitions across the series",
}

# ── Category grouping ────────────────────────────────────────────────────
TASK_CATEGORIES = {
    "Binary Judgment": {
        "tasks": ["yes_no", "correlation", "anticorrelation", "anti_judgment",
                  "segment_judgment", "cross_stat_judgment"],
        "caption": (
            "Binary judgment tasks: the model must produce a yes/no or true/false verdict "
            "about time series properties (e.g.\\ correlation, local events, multi-condition checks). "
            "Accuracy = fraction of correct verdicts. "
            "Reasoning = LLM-judged quality of the chain-of-thought explanation (0--1)."
        ),
        "label": "tab:binary-judgment",
        "metric_name": "Accuracy",
    },
    "Trend Query": {
        "tasks": ["cross_trend_query", "segment_trend_dominance"],
        "caption": (
            "Trend query tasks: the model must identify cross-metric trend relationships "
            "or determine which trend type dominates a time range. "
            "Accuracy = fraction of correct answers. "
            "Reasoning = LLM-judged quality of the explanation (0--1)."
        ),
        "label": "tab:trend-query",
        "metric_name": "Accuracy",
    },
    "Enumeration": {
        "tasks": ["local_enumeration", "segment_enumeration", "cross_metric_enumeration",
                  "event_segment_enumeration", "transition_enumeration"],
        "caption": (
            "Enumeration tasks: the model must count, filter, or identify specific elements "
            "(e.g.\\ number of downward spikes, longest trend segment, noisiest metric). "
            "Accuracy = exact-match rate. "
            "Reasoning = LLM-judged quality of the explanation (0--1)."
        ),
        "label": "tab:enumeration",
        "metric_name": "Accuracy",
    },
    "Numerical \\& Temporal": {
        "tasks": ["stat_numerical", "temporal_position", "duration_proportion"],
        "caption": (
            "Numerical and temporal tasks: the model must compute an exact statistic "
            "(mean, range, extrema position, windowed aggregation) or identify temporal positions/durations. "
            "Accuracy = fraction within tolerance of the ground truth. "
            "Reasoning = LLM-judged quality of the working shown (0--1)."
        ),
        "label": "tab:numerical",
        "metric_name": "Accuracy",
    },
    "OOD Compositional": {
        "tasks": ["ood_conditional_stat", "ood_nested_extrema", "ood_event_density",
                  "ood_conditional_count", "ood_trend_reversal",
                  "ood_range_normalized_amplitude", "ood_segment_stat_compare"],
        "caption": (
            "Out-of-distribution compositional tasks: questions that compose 2+ skills "
            "never combined during training (e.g.\\ conditional statistics, nested extrema, "
            "event density, trend reversal). These test whether the model generalizes beyond imitation. "
            "Accuracy = fraction of correct verdicts. "
            "Reasoning = LLM-judged quality of the explanation (0--1)."
        ),
        "label": "tab:ood-compositional",
        "metric_name": "Accuracy",
    },
    # ── Expanded taxonomy (atomic primitives, RL compositional, expanded OOD) ─
    "Atomic — Single-Metric": {
        "tasks": [
            "atomic_global_mean", "atomic_chunked_means", "atomic_interval_mean",
            "atomic_global_std", "atomic_interval_std",
            "atomic_min_value", "atomic_max_value", "atomic_min_position", "atomic_max_position",
            "atomic_percentile",
            "atomic_trend_enumeration", "atomic_event_enumeration", "atomic_periodic_description",
        ],
        "caption": (
            "Atomic single-metric tasks: minimum-complexity primitives that compute a single "
            "statistic (mean, std, min/max, percentile) or enumerate structural elements "
            "(segments, events, periodic behavior) of one time series. These are taught in "
            "SFT and are the building blocks for the compositional RL tasks."
        ),
        "label": "tab:atomic-single",
        "metric_name": "Accuracy",
    },
    "Atomic — Cross-Metric": {
        "tasks": [
            "atomic_cross_stat_compare", "atomic_cross_ranking", "atomic_cross_filtering",
            "atomic_cross_counting", "atomic_cross_trend_align",
        ],
        "caption": (
            "Atomic cross-metric tasks: primitives over multiple metrics "
            "(compare, rank, filter, count, trend-align). Taught in SFT via MTS generators; "
            "supports the cross-metric RL and OOD families."
        ),
        "label": "tab:atomic-cross",
        "metric_name": "Accuracy",
    },
    "RL Compositional — Single-Metric": {
        "tasks": [
            "rl_amplitude_vs_range", "rl_amplitude_vs_std", "rl_condition_recovery",
            "rl_cycle_count", "rl_dominant_trend_type", "rl_event_count", "rl_event_count_by_type",
            "rl_event_in_trend_type", "rl_event_near_extremum", "rl_event_type_at_pos",
            "rl_extrema_same_half", "rl_half_mean_compare", "rl_half_mean_diff",
            "rl_has_periodicity", "rl_interval_comparison", "rl_longest_segment",
            "rl_max_amplitude_event", "rl_max_in_first_half", "rl_max_in_trend_type",
            "rl_mean_shift", "rl_mean_stability", "rl_median_mean_close", "rl_monotonic_chunks",
            "rl_normalized_range", "rl_period_estimate", "rl_range",
            "rl_segment_count", "rl_segment_duration", "rl_segment_mean_compare",
            "rl_segment_type_at_pos", "rl_type_duration_fraction", "rl_type_of_longest",
            "rl_volatility_change",
        ],
        "caption": (
            "RL compositional single-metric tasks: compositions of two or more atomic primitives "
            "within one metric (e.g.\\ largest event amplitude vs.\\ a fraction of the value range, "
            "first-half mean compared to second-half mean). Trained with verifiable reward "
            "on the `<think>` block. "
            "Accuracy = fraction of correct verdicts. Reasoning = LLM-judged explanation quality."
        ),
        "label": "tab:rl-single",
        "metric_name": "Accuracy",
    },
    "RL Compositional — Cross-Metric": {
        "tasks": [
            "rl_cross_stat_ratio", "rl_cross_full_ordering", "rl_cross_event_sync",
            "rl_cross_period_compare", "rl_cross_trend_concordance",
            "rl_cross_conditional_query", "rl_cross_attribute_corr", "rl_cross_asymmetric_behavior",
            "rl_cluster_count", "rl_cluster_dominant", "rl_corr_count", "rl_corr_conditional",
        ],
        "caption": (
            "RL compositional cross-metric tasks: compositions of atomic primitives across "
            "two or more metrics (e.g.\\ stat-ratio thresholds, event synchronization, "
            "conditional ranking, cluster counting). Trained with verifiable reward."
        ),
        "label": "tab:rl-cross",
        "metric_name": "Accuracy",
    },
    "OOD Expanded — Single-Metric": {
        "tasks": [
            "ood_max_before_min", "ood_std_exceeds_half_range",
            "ood_max_in_highest_mean_quarter", "ood_quarter_mean_ordering",
            "ood_max_mean_chunk_pos", "ood_chunk_above_proportion", "ood_symmetric_recovery",
            "ood_cycle_mean_trend", "ood_trend_follows_mean", "ood_conditional_mean_by_type",
            "ood_longest_type_fraction", "ood_event_amplitude_vs_std", "ood_amplitude_vs_segment_std",
            "ood_symmetric_trend_sequence", "ood_event_density_by_trend", "ood_max_amp_in_longest_segment",
        ],
        "caption": (
            "OOD single-metric tasks (eval-only): novel compositions not seen during training "
            "(e.g.\\ max-before-min ordering, symmetric recovery, quarter-mean ordering). "
            "Tests whether the model generalizes the atomic primitives to unseen compositions."
        ),
        "label": "tab:ood-single-expanded",
        "metric_name": "Accuracy",
    },
    "OOD Expanded — Cross-Metric": {
        "tasks": [
            "ood_cross_corr_count", "ood_cross_trend_convergence", "ood_cross_extrema_alignment",
            "ood_cross_range_overlap", "ood_cross_concordant_shift",
            "ood_cluster_singleton", "ood_corr_transitivity", "ood_mixed_corr_anti",
        ],
        "caption": (
            "OOD cross-metric tasks (eval-only): novel multi-metric compositions such as "
            "transitivity of trend sharing, convergence between halves, singleton clusters. "
            "Tests cross-metric generalization."
        ),
        "label": "tab:ood-cross-expanded",
        "metric_name": "Accuracy",
    },
    "Periodicity \\& Change Point": {
        "tasks": ["periodicity", "change_point"],
        "caption": (
            "Periodicity and change-point detection tasks. "
            "Periodicity covers estimated period, cycle count, and regularity. "
            "Change-point covers detection count, position, and largest-shift location."
        ),
        "label": "tab:periodicity-changepoint",
        "metric_name": "Accuracy",
    },
    "OOD Compositional Probes": {
        "tasks": [
            "ood_peak_in_longest_segment",
            "ood_duration_weighted_mean",
            "ood_dominant_trend_in_cluster",
            "ood_cross_event_causality",
        ],
        "caption": (
            "OOD compositional probes (eval-only): four targeted compositions of "
            "strongly-demonstrated atomic + RL primitives, never seen during training. "
            "(i) \\texttt{peak\\_in\\_longest\\_segment} — composes $\\arg\\max$ position with "
            "longest trend segment (containment test). "
            "(ii) \\texttt{duration\\_weighted\\_mean} — composes segment duration with "
            "interval mean (weighted average). "
            "(iii) \\texttt{dominant\\_trend\\_in\\_cluster} — composes cross-metric clustering "
            "with dominant-trend identification. "
            "(iv) \\texttt{cross\\_event\\_causality} — extends cross-metric event synchronization "
            "from simultaneous to temporally ordered. "
            "A positive $\\Delta$ vs SFT is held-out evidence of compositional transfer."
        ),
        "label": "tab:ood-compositional-probes",
        "metric_name": "Accuracy",
    },
}

# Eval types that vary meaningfully with sequence length — used by the
# cross-length-bucket summary table (one row per probe, one column per
# length bucket in-SFT ≤256 / near 257-768 / far 769-1536 / extreme >1536).
COMPOSITIONAL_PROBES_CROSS_LENGTH = [
    "ood_peak_in_longest_segment",
    "ood_duration_weighted_mean",
    "ood_dominant_trend_in_cluster",
    "ood_cross_event_causality",
]

SET_MATCHING_TASKS = ["clustering", "anticlustering"]


def parse_summary(path: Path) -> dict:
    """Parse an evaluation_report_summary.txt into structured data."""
    text = path.read_text()
    result = {"main": {}, "clustering_by_len": {}, "description_breakdown": {},
              "trend_detail": {}, "reasoning_breakdown": {}, "tsevol_by_strategy": {}}

    # ── Main table (FINAL EVALUATION SUMMARY) ──────────────────────────
    main_match = re.search(
        r"FINAL EVALUATION SUMMARY\n=+\n(.*?)\n\n=+",
        text, re.DOTALL,
    )
    if main_match:
        lines = main_match.group(1).strip().split("\n")
        # Parse column names from header (first line of pandas .to_string())
        col_names = None
        for line in lines:
            stripped = line.strip()
            if stripped and not stripped.startswith("task_type"):
                # First non-empty, non-index-name line is the header
                col_names = stripped.split()
                break
        if col_names is None:
            col_names = ["f1", "binary_accuracy", "reasoning_score", "count"]
        for line in lines:
            stripped = line.strip()
            if not stripped:
                continue
            # Skip header and index-name lines
            if stripped.split()[0] in col_names or stripped.startswith("task_type"):
                continue
            parts = stripped.split()
            task = parts[0]
            vals = parts[1:]
            row = {}
            for i, col in enumerate(col_names):
                if i < len(vals):
                    try:
                        row[col] = float(vals[i])
                    except ValueError:
                        row[col] = None
                else:
                    row[col] = None
            result["main"][task] = row

    # ── Description QA breakdown ───────────────────────────────────────
    desc_match = re.search(
        r"DESCRIPTION QA BREAKDOWN.*?\n=+\n(.*?)\n\n=+",
        text, re.DOTALL,
    )
    if desc_match:
        for line in desc_match.group(1).strip().split("\n"):
            line = line.strip()
            m = re.match(r"(\w+)\s+cls=([\d.]+)\s+quant=([\d.]+)(?:\s+\[(.+?)\])?\s+n=(\d+)", line)
            if m:
                perspective = m.group(1)
                row = {"cls": float(m.group(2)), "quant": float(m.group(3)), "n": int(m.group(5))}
                if m.group(4):
                    for kv in m.group(4).split():
                        k, v = kv.split("=")
                        row[k] = float(v)
                result["description_breakdown"][perspective] = row

    # ── Clustering insights by GT length ─────────────────────────────────
    clust_match = re.search(
        r"CLUSTERING INSIGHTS BY GT LENGTH\n=+\n(.*?)\n\n=+",
        text, re.DOTALL,
    )
    if clust_match:
        clu_lines = clust_match.group(1).strip().split("\n")
        # Parse column names from header
        clu_col_names = None
        for line in clu_lines:
            stripped = line.strip()
            if stripped and not stripped.startswith("gt_len"):
                clu_col_names = stripped.split()
                break
        if clu_col_names is None:
            clu_col_names = ["f1", "precision", "recall", "sample_count"]
        for line in clu_lines:
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.split()[0] in clu_col_names or stripped.startswith("gt_len"):
                continue
            parts = stripped.split()
            try:
                gt_len = int(float(parts[0]))
            except (ValueError, IndexError):
                continue  # skip non-data lines (e.g. "clustering micro: ...")
            row = {}
            for i, col in enumerate(clu_col_names):
                if i < len(parts) - 1:
                    try:
                        v = float(parts[i + 1])
                        row[col] = int(v) if col == "sample_count" else v
                    except ValueError:
                        row[col] = None
            result["clustering_by_len"][gt_len] = row

    # ── Trend direction detail ─────────────────────────────────────────
    trend_match = re.search(
        r"TREND DIRECTION DETAIL.*?\n=+\n(.*?)(?:\n\n|\Z)",
        text, re.DOTALL,
    )
    if trend_match:
        block = trend_match.group(1)
        m = re.search(r"direction_cls\s+acc=([\d.]+)\s+n=(\d+)", block)
        if m:
            result["trend_detail"]["direction_acc"] = float(m.group(1))
            result["trend_detail"]["direction_n"] = int(m.group(2))
        m = re.search(r"segments\s+P=([\d.]+)\s+R=([\d.]+)\s+F1=([\d.]+)\s+boundary_pos_acc=([\d.]+)\s+n=(\d+)", block)
        if m:
            result["trend_detail"]["seg_P"] = float(m.group(1))
            result["trend_detail"]["seg_R"] = float(m.group(2))
            result["trend_detail"]["seg_F1"] = float(m.group(3))
            result["trend_detail"]["seg_boundary"] = float(m.group(4))
            result["trend_detail"]["seg_n"] = int(m.group(5))
        m = re.search(r"reasoning\s+score=([\d.]+)\s+n=(\d+)", block)
        if m:
            result["trend_detail"]["reasoning"] = float(m.group(1))
            result["trend_detail"]["reasoning_n"] = int(m.group(2))
        m = re.search(r"OVERALL.*?score=([\d.]+)\s+n=(\d+)", block)
        if m:
            result["trend_detail"]["overall"] = float(m.group(1))
            result["trend_detail"]["overall_n"] = int(m.group(2))

    # ── TSEvol by strategy ──────────────────────────────────────────────
    tsevol_match = re.search(
        r"TSEVOL PERFORMANCE BY STRATEGY\n=+\n(.*?)(?:\n\n|\Z)",
        text, re.DOTALL,
    )
    if tsevol_match:
        for line in tsevol_match.group(1).strip().split("\n"):
            m = re.match(r"\s*(\w+)\s+score=([\d.]+)\s+n=(\d+)", line)
            if m:
                result["tsevol_by_strategy"][m.group(1)] = {
                    "score": float(m.group(2)), "n": int(m.group(3)),
                }

    return result


def parse_length_bucket_overall(path: Path) -> dict:
    """Parse length_bucket_overall.csv → {bucket: {'mean_score', 'n_samples'}}.

    Produced by evaluation/eval/aggregation.py::build_length_bucket_overall().
    Columns: length_bucket, mean_score, n_samples.
    Returns {} if file doesn't exist (non-synthetic datasets).
    """
    if not path.exists():
        return {}
    result = {}
    with path.open() as f:
        header = f.readline().strip().split(",")
        try:
            i_bucket = header.index("length_bucket")
            i_score = header.index("mean_score")
            i_n = header.index("n_samples")
        except ValueError:
            return {}
        for line in f:
            parts = line.strip().split(",")
            if len(parts) < 3:
                continue
            bucket = parts[i_bucket]
            try:
                result[bucket] = {
                    "mean_score": float(parts[i_score]),
                    "n_samples": int(float(parts[i_n])),
                }
            except ValueError:
                continue
    return result


LENGTH_BUCKET_LABELS = {
    "in_sft": "In-envelope ($\\leq$256, both trained)",
    "rl_only": "RL-only (257--768, SFT didn't see)",
    "ood_near": "OOD near (769--1536, neither trained)",
    "ood_far": "OOD far ($>$1536, extreme)",
    "unknown": "Unknown length",
}
LENGTH_BUCKET_ORDER = ["in_sft", "rl_only", "ood_near", "ood_far", "unknown"]


def full_document_preamble(title: str = "Evaluation Results") -> str:
    """Return a standalone-compilable LaTeX preamble ending at \\begin{document}.

    Packages cover everything the generated tables + examples appendix use:
      booktabs     -> \\toprule / \\midrule / \\bottomrule / \\cmidrule
      adjustbox    -> \\begin{adjustbox}{max width=...}
      xcolor[table] -> \\cellcolor and \\definecolor
      tcolorbox    -> example boxes in the appendix
      amsmath      -> \\text{...} inside math-mode formulas
      hyperref     -> optional but harmless; cross-references
      geometry     -> sensible margins so wide tables fit
    """
    return (
        "\\documentclass[11pt]{article}\n"
        "\\usepackage[margin=1in]{geometry}\n"
        "\\usepackage{booktabs}\n"
        "\\usepackage{adjustbox}\n"
        "\\usepackage[table]{xcolor}\n"
        "\\usepackage{tcolorbox}\n"
        "\\usepackage{amsmath}\n"
        "\\usepackage{placeins}\n"
        "\\usepackage{hyperref}\n"
        "\\definecolor{improv}{RGB}{200,235,200}\n"
        "\n"
        f"\\title{{{title}}}\n"
        "\\date{}\n"
        "\\begin{document}\n"
        "\\maketitle\n"
    )


def full_document_postamble() -> str:
    return "\n\\end{document}\n"


def _is_none(v):
    return v is None or (isinstance(v, float) and math.isnan(v))


def fmt(val, bold=False, highlight=False, is_pct=True):
    """Format a value for LaTeX."""
    if val is None or (isinstance(val, float) and math.isnan(val)):
        return "—"
    if is_pct:
        s = f"{val:.2f}"
    else:
        s = str(int(val))
    if bold:
        s = f"\\textbf{{{s}}}"
    if highlight:
        s = f"\\cellcolor{{improv}}{s}"
    return s


def bold_better(v1, v2, higher_is_better=True, is_pct=True):
    """Return (formatted_v1, formatted_v2). Better value is bolded.

    Only the v2 (RL) cell is highlighted green when v2 does NOT degrade
    vs. v1 (SFT) — i.e. v2 >= v1 for higher_is_better metrics, v2 <= v1
    for lower_is_better. Plain (uncolored) v2 is therefore a visual flag
    for RL regressions, which is the only thing that needs attention
    during a scan-through.
    """
    return tuple(bold_better_n([v1, v2], baseline_idx=0,
                                higher_is_better=higher_is_better,
                                is_pct=is_pct))


def bold_better_n(values, baseline_idx=0, higher_is_better=True, is_pct=True):
    """N-way version of bold_better. Returns list of N formatted strings.

    - The single best value among non-None values is bolded.
    - Every NON-baseline cell that beats the baseline (values[baseline_idx])
      gets the green highlight, signalling improvement over baseline.
      The baseline cell itself never highlights.
    - None / NaN values render as em-dashes.

    With N=2 and baseline_idx=0 this matches the legacy bold_better
    (highlight on v2 if v2 ≥ v1).
    """
    n = len(values)
    valid_mask = [not _is_none(v) for v in values]
    valid_vals = [v for v, m in zip(values, valid_mask) if m]
    if not valid_vals:
        return ["—"] * n
    best = max(valid_vals) if higher_is_better else min(valid_vals)

    baseline = values[baseline_idx] if (
        0 <= baseline_idx < n and valid_mask[baseline_idx]
    ) else None

    out = []
    for i, v in enumerate(values):
        if not valid_mask[i]:
            out.append("—")
            continue
        is_best = (v == best)
        is_baseline_row = (i == baseline_idx)
        if baseline is None or is_baseline_row:
            highlight = False
        else:
            highlight = (v >= baseline) if higher_is_better else (v <= baseline)
        out.append(fmt(v, bold=is_best, highlight=highlight, is_pct=is_pct))
    return out


def escape_latex(s: str) -> str:
    """Escape underscores and other special chars for LaTeX."""
    return s.replace("_", "\\_").replace("&", "\\&").replace("%", "\\%")


def _two_metric_header_n(metric_left: str, metric_right: str, names: list[str],
                          extra_label: str | None = None,
                          row_label: str = "Task",
                          add_delta: bool = False):
    """Build LaTeX header lines for an N-way table with TWO metric groups
    (e.g. Accuracy + Reasoning). Each metric group spans N columns, one per
    model. Returns (col_spec, header_lines).

    extra_label: header text for an extra row-label column inserted between
    the row label and the metric groups (e.g. "Type" for OOD merged table).
    None = no extra column.
    row_label: label of the leftmost column (default "Task").
    add_delta: if True (and N>=2), append a final "Δ" column showing
    last-minus-first model on the LEFT metric. Header is "Δ ({last_name} −
    {first_name})".
    """
    N = len(names)
    extra_left_cols = 1 if extra_label is not None else 0

    # Column spec: l + extra l + N c (group1) + N c (group2) + c (n) + optional c (Δ)
    cols = ["l"] + ["l"] * extra_left_cols + ["c"] * N + ["c"] * N + ["c"]
    if add_delta:
        cols.append("c")
    col_spec = " ".join(cols)

    # multicolumn header: empty cells for label cols, then 2 multicolumns, then & for $n$ + optional & for Δ
    leading_blanks = "& " * (1 + extra_left_cols)
    multicol = (
        leading_blanks
        + f"\\multicolumn{{{N}}}{{c}}{{{metric_left}}} & "
        + f"\\multicolumn{{{N}}}{{c}}{{{metric_right}}} & "
        + ("& \\\\" if add_delta else "\\\\")
    )

    start_left = 2 + extra_left_cols
    end_left = start_left + N - 1
    start_right = end_left + 1
    end_right = start_right + N - 1
    cmidrules = (
        f"\\cmidrule(lr){{{start_left}-{end_left}}} "
        f"\\cmidrule(lr){{{start_right}-{end_right}}}"
    )

    # row header
    parts = [row_label]
    if extra_label is not None:
        parts.append(extra_label)
    parts.extend(escape_latex(n) for n in names)
    parts.extend(escape_latex(n) for n in names)
    parts.append("$n$")
    if add_delta and N >= 2:
        parts.append(
            f"$\\Delta$ ({escape_latex(names[-1])}--{escape_latex(names[0])})"
        )
    row_header = " & ".join(parts) + " \\\\"

    return col_spec, [multicol, cmidrules, row_header]


def _delta_cell_pp(values, higher_is_better=True, dp=2):
    """Format the last-minus-first delta as a percentage-point string with sign,
    color-coded green/italic for positive/negative. Returns "—" if either is None."""
    if len(values) < 2:
        return "—"
    v_first = values[0]
    v_last = values[-1]
    if v_first is None or v_last is None:
        return "—"
    if isinstance(v_first, float) and math.isnan(v_first): return "—"
    if isinstance(v_last, float) and math.isnan(v_last): return "—"
    d = v_last - v_first
    sign = "+" if d >= 0 else ""
    s = f"{sign}{d*100:.1f}"
    if (d > 0.005 and higher_is_better) or (d < -0.005 and not higher_is_better):
        return f"\\cellcolor{{improv}}\\textbf{{{s}}}"
    elif (d < -0.005 and higher_is_better) or (d > 0.005 and not higher_is_better):
        return f"\\textit{{{s}}}"
    return s


def render_length_bucket_table(
    bucket_data: list[tuple[str, list[dict]]], names: list[str], label_suffix: str = ""
) -> list[str]:
    """Render length-generalization table across one or more datasets, N-way.

    bucket_data: list of (dataset_label, [model1_buckets, model2_buckets, ...])
                 where each buckets dict is {bucket: {mean_score, n_samples}}.
    """
    N = len(names)
    lines = []
    lines.append("% ═══ Length Generalization ═══")
    lines.append("\\begin{table}[!htbp]")
    lines.append("\\centering")
    lines.append(
        "\\caption{Length-generalization performance by training-envelope bucket. "
        "\\texttt{in\\_sft} ($\\leq$256) is the SFT/RL-shared envelope; "
        "\\texttt{rl\\_only} (257--768) is seen by RL only; "
        "\\texttt{ood\\_near} (769--1536) and \\texttt{ood\\_far} ($>$1536) are "
        "eval-only out-of-distribution buckets. "
        "Score = mean of binary\\_accuracy (fallback: reasoning\\_score, f1) averaged "
        "across all task types within the bucket, weighted by sample count.}"
    )
    lines.append(f"\\label{{tab:length-generalization{label_suffix}}}")
    lines.append("\\begin{adjustbox}{max width=\\columnwidth}")
    lines.append(f"\\begin{{tabular}}{{l l {'c '*N}c}}")
    lines.append("\\toprule")
    names_row = " & ".join(escape_latex(n) for n in names)
    lines.append(f"Dataset & Bucket & {names_row} & $n$ \\\\")
    lines.append("\\midrule")
    for dataset_label, model_buckets_list in bucket_data:
        present = set()
        for mb in model_buckets_list:
            present |= set(mb.keys())
        buckets_in_order = [b for b in LENGTH_BUCKET_ORDER if b in present]
        if not buckets_in_order:
            continue
        first = True
        for b in buckets_in_order:
            vals = [mb.get(b, {}).get("mean_score") for mb in model_buckets_list]
            cells = bold_better_n(vals)
            n_val = next((mb.get(b, {}).get("n_samples")
                          for mb in model_buckets_list
                          if mb.get(b, {}).get("n_samples")), "")
            ds_label = escape_latex(dataset_label) if first else ""
            first = False
            b_tex = escape_latex(b)
            row_cells = [ds_label, f"\\texttt{{{b_tex}}}"] + cells + [str(n_val) if n_val else ""]
            lines.append(" & ".join(row_cells) + " \\\\")
        lines.append("\\midrule")
    if lines[-1] == "\\midrule":
        lines[-1] = "\\bottomrule"
    else:
        lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{adjustbox}")
    lines.append("\\end{table}")
    lines.append("")
    return lines


def render_ood_summary_consolidated(
    dirs, names: list[str],
) -> list[str]:
    """Single OOD summary table: every OOD eval_type × every length bucket.

    For each OOD type we show Δ = (last model - first model) per length
    bucket, plus mean Δ and a wins-out-of-buckets count. Subheaders separate
    the OOD subcategories. For N>2 models, the delta is computed between the
    LAST and FIRST model in `dirs` (e.g. SFT+RL vs Baseline), since this is
    the headline comparison; intermediate models contribute via the merged
    per-dataset OOD table elsewhere in the report.

    `dirs` accepts either a 2-tuple/list (legacy) or any N-element list of
    dir paths. `names` is matched length.
    """
    import csv

    # Backward-compat: accept (dir1, dir2) positional too. Prefer list.
    if isinstance(dirs, (tuple, list)):
        dirs_list = list(dirs)
    else:  # legacy: 2-arg invocation
        dirs_list = [dirs]
    # When >2 models we render delta = last - first.
    dir_first = dirs_list[0]
    dir_last = dirs_list[-1]
    name_first = names[0]
    name_last = names[-1]

    BUCKETS = [
        ("test",             "in\\_sft"),
        ("test_len_near",    "near"),
        ("test_len_far",     "far"),
        ("test_len_extreme", "extreme"),
    ]

    OOD_GROUPS = [
        ("OOD Compositional (legacy)",
         TASK_CATEGORIES["OOD Compositional"]["tasks"]),
        ("OOD Single-Metric (expanded)",
         TASK_CATEGORIES["OOD Expanded — Single-Metric"]["tasks"]),
        ("OOD Cross-Metric (expanded)",
         TASK_CATEGORIES["OOD Expanded — Cross-Metric"]["tasks"]),
        ("OOD Compositional Probes",
         TASK_CATEGORIES["OOD Compositional Probes"]["tasks"]),
    ]

    def _load(dir_root: Path, dataset: str) -> dict:
        path = dir_root / dataset / "summary_by_task_type.csv"
        out = {}
        if not path.exists():
            return out
        with open(path) as f:
            r = csv.DictReader(f)
            for row in r:
                tt = row.get("task_type")
                if not tt:
                    continue
                try:
                    ba = float(row.get("binary_accuracy") or "nan")
                    cnt = int(float(row.get("count") or 0))
                except (ValueError, TypeError):
                    continue
                if ba == ba:  # not NaN
                    out[tt] = (ba, cnt)
        return out

    data1 = {ds: _load(dir_first, ds) for ds, _ in BUCKETS}
    data2 = {ds: _load(dir_last, ds) for ds, _ in BUCKETS}

    # Skip if no OOD data anywhere
    all_ood = {t for _, ts in OOD_GROUPS for t in ts}
    has_data = any(
        any(t in d for t in all_ood)
        for d in list(data1.values()) + list(data2.values())
    )
    if not has_data:
        return []

    n1 = name_first
    n2 = name_last

    def _delta_cell(probe: str, ds: str) -> tuple[str, float]:
        """Return (formatted_cell, delta_value or NaN)."""
        v1, _ = data1[ds].get(probe, (None, 0))
        v2, _ = data2[ds].get(probe, (None, 0))
        if v1 is None or v2 is None:
            return "—", float("nan")
        d = v2 - v1
        sign = "+" if d >= 0 else ""
        s = f"{sign}{d*100:.1f}"
        if d > 0.005:
            s = f"\\cellcolor{{improv}}\\textbf{{{s}}}"
        elif d < -0.005:
            s = f"\\textit{{{s}}}"
        return s, d

    lines = []
    lines.append("% ═══ OOD Summary (consolidated) ═══")
    lines.append("\\begin{table}[!htbp]")
    lines.append("\\centering")
    lines.append(
        "\\caption{OOD summary across all evaluation types and length buckets. "
        "Each cell is the absolute-percentage-point $\\Delta = $ "
        f"({escape_latex(n2)} $-$ {escape_latex(n1)}) accuracy at that length bucket. "
        "Green-highlighted bold cells: "
        f"{escape_latex(n2)} improves; italics: regression; em-dash: type not present "
        "in that bucket. \\textbf{Wins/4} counts the buckets where "
        f"{escape_latex(n2)} beats {escape_latex(n1)} for that type. "
        "Mean $\\Delta$ averages over present buckets only.}"
    )
    lines.append("\\label{tab:ood-summary}")
    lines.append("\\begin{adjustbox}{max width=\\textwidth}")
    lines.append("\\begin{tabular}{l " + "c " * len(BUCKETS) + "c c}")
    lines.append("\\toprule")
    header = ["Task"]
    for _, label in BUCKETS:
        header.append(f"$\\Delta$ {label}")
    header.append("Mean $\\Delta$")
    header.append("Wins")
    lines.append(" & ".join(header) + " \\\\")
    lines.append("\\midrule")

    overall_wins = 0
    overall_total = 0
    type_wins = 0   # types where mean Δ > 0
    type_total = 0  # types with any data
    for group_label, tasks in OOD_GROUPS:
        present_tasks = [
            t for t in tasks
            if any(t in d for d in list(data1.values()) + list(data2.values()))
        ]
        if not present_tasks:
            continue
        lines.append(
            f"\\multicolumn{{{len(BUCKETS)+3}}}{{l}}{{\\textit{{{escape_latex(group_label)}}}}} \\\\"
        )
        for t in present_tasks:
            row = [f"\\texttt{{{escape_latex(t)}}}"]
            deltas = []
            wins = 0
            for ds, _ in BUCKETS:
                cell, d = _delta_cell(t, ds)
                row.append(cell)
                if d == d:  # not NaN
                    deltas.append(d)
                    if d > 0.005:
                        wins += 1
            if deltas:
                mean_d = sum(deltas) / len(deltas)
                sign = "+" if mean_d >= 0 else ""
                mean_s = f"{sign}{mean_d*100:.1f}"
                if mean_d > 0.005:
                    mean_s = f"\\cellcolor{{improv}}\\textbf{{{mean_s}}}"
                    type_wins += 1
                elif mean_d < -0.005:
                    mean_s = f"\\textit{{{mean_s}}}"
                row.append(mean_s)
                row.append(f"{wins}/{len(deltas)}")
                overall_wins += wins
                overall_total += len(deltas)
                type_total += 1
            else:
                row.extend(["—", "—"])
            lines.append(" & ".join(row) + " \\\\")
        lines.append("\\midrule")
    if lines[-1] == "\\midrule":
        lines[-1] = "\\bottomrule"
    else:
        lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{adjustbox}")
    lines.append("\\end{table}")
    lines.append("")
    lines.append(
        f"\\noindent\\textit{{Overall: {type_wins} of {type_total} OOD types "
        f"have positive mean $\\Delta$ "
        f"(SFT+RL beats SFT). "
        f"Cell-level: {overall_wins} of {overall_total} (type $\\times$ length-bucket) "
        f"combinations show $\\Delta > 0$.}}"
    )
    lines.append("")
    return lines


def render_composition_probes_by_length(
    dirs, names: list[str],
) -> list[str]:
    """Render the compositional-probe scores stratified by length bucket, N-way.

    One row per probe. For each length-bucket dataset (test, test_len_near,
    test_len_far, test_len_extreme), emit N columns (one per model).
    Pulls binary_accuracy and count from each dataset's summary_by_task_type.csv.

    `dirs` accepts either a 2-tuple (legacy: dir1, dir2) or any list of dir paths.
    """
    import csv

    if isinstance(dirs, (tuple, list)):
        dirs_list = list(dirs)
    else:
        dirs_list = [dirs]

    # Map dataset_subdir → human-readable length label
    BUCKETS = [
        ("test",             "in\\_sft ($\\leq$256)"),
        ("test_len_near",    "near (257--768)"),
        ("test_len_far",     "far (769--1536)"),
        ("test_len_extreme", "extreme ($>$1536)"),
    ]

    def _load(dir_root: Path, dataset: str) -> dict:
        """Return {task_type: (binary_accuracy, count)}."""
        path = dir_root / dataset / "summary_by_task_type.csv"
        out = {}
        if not path.exists():
            return out
        with open(path) as f:
            r = csv.DictReader(f)
            for row in r:
                tt = row.get("task_type")
                if not tt:
                    continue
                try:
                    ba = float(row.get("binary_accuracy") or "nan")
                    cnt = int(float(row.get("count") or 0))
                except (ValueError, TypeError):
                    continue
                out[tt] = (ba, cnt)
        return out

    # Preload every (model, dataset) → task map.
    N = len(dirs_list)
    data_per_model = [
        {ds: _load(d, ds) for ds, _ in BUCKETS}
        for d in dirs_list
    ]

    # Skip the table if no bucket has a probe at all.
    any_probe = any(
        any(probe in d for probe in COMPOSITIONAL_PROBES_CROSS_LENGTH)
        for dpm in data_per_model
        for d in dpm.values()
    )
    if not any_probe:
        return []

    lines = []
    lines.append("% ═══ Compositional Probes × Length Bucket ═══")
    lines.append("\\begin{table}[!htbp]")
    lines.append("\\centering")
    lines.append(
        "\\caption{Compositional-probe accuracy stratified by training-envelope "
        "length bucket. Each probe pairs two RL-demonstrated primitives in a "
        "novel composition not seen during training; positive accuracy gap vs.\\ "
        "the leftmost (baseline) column at any length is held-out evidence of "
        "compositional transfer. The in\\_sft bucket ($\\leq$256) is the "
        "shared training envelope; near/far/extreme are progressively "
        "out-of-distribution. Highlighted cells: model meets/beats the baseline.}"
    )
    lines.append("\\label{tab:composition-probes-by-length}")
    lines.append("\\begin{adjustbox}{max width=\\textwidth}")

    # Column spec: l (probe) | then (c × N) per bucket
    col_spec = "l " + " ".join([" ".join(["c"] * N)] * len(BUCKETS))
    lines.append(f"\\begin{{tabular}}{{{col_spec}}}")
    lines.append("\\toprule")

    # Header row 1: bucket labels spanning N columns each
    header1 = ["Probe"]
    for _, label in BUCKETS:
        header1.append(f"\\multicolumn{{{N}}}{{c}}{{{label}}}")
    lines.append(" & ".join(header1) + " \\\\")

    # Header row 2: model names under each bucket
    header2 = [""]
    for _ in BUCKETS:
        for nm in names:
            header2.append(escape_latex(nm))
    lines.append(" & ".join(header2) + " \\\\")
    lines.append("\\midrule")

    for probe in COMPOSITIONAL_PROBES_CROSS_LENGTH:
        row_cells = [f"\\texttt{{{escape_latex(probe)}}}"]
        for ds, _ in BUCKETS:
            vals = []
            counts = []
            for dpm in data_per_model:
                v, c = dpm[ds].get(probe, (None, 0))
                vals.append(v)
                counts.append(c)
            if all(v is None for v in vals):
                row_cells.extend(["—"] * N)
            else:
                cells = bold_better_n(vals)
                n_both = max(c or 0 for c in counts)
                if n_both and n_both < 20:
                    cells[0] = f"{cells[0]}$^{{{n_both}}}$"
                row_cells.extend(cells)
        lines.append(" & ".join(row_cells) + " \\\\")
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{adjustbox}")
    lines.append("\\end{table}")
    lines.append("")
    return lines


def generate_latex(
    exps: list[dict],
    names: list[str],
    section_title: str | None = None,
    include_preamble: bool = True,
    include_examples: bool = True,
) -> str:
    """Generate full LaTeX content comparing experiments.

    section_title: if set, wrap the output in \\section*{...} (used by --combine
                   mode to nest per-dataset tables under a heading).
    include_preamble: emit the \\definecolor line at the top (only once when
                      combining multiple datasets).
    include_examples: emit the \\clearpage examples appendix at the end.
    """
    assert len(exps) >= 2, "Need at least 2 experiments"
    assert len(exps) == len(names), "exps and names must match length"
    N = len(exps)
    # Backward-compat aliases used in a few places below
    e1, e2 = exps[0], exps[-1]
    n1, n2 = names[0], names[-1]

    lines = []

    if include_preamble:
        lines.append("% Auto-generated by results_to_latex.py")
        lines.append("% \\usepackage{booktabs, adjustbox}")
        lines.append("% \\usepackage[table]{xcolor}  % or add 'table' option if xcolor already loaded")
        lines.append("\\definecolor{improv}{RGB}{200,235,200}")
        lines.append("")

    if section_title:
        lines.append(f"\\section*{{{section_title}}}")
        lines.append("")

    all_tasks_union: set[str] = set()
    for e in exps:
        all_tasks_union |= set(e["main"].keys())
    all_tasks = sorted(all_tasks_union)

    # OOD subcategories are merged into a single table (with a Category
    # column) instead of rendering one table per subcategory. Order = the
    # 4 subcategory names in the order they should appear, with the short
    # tag shown in the Category column.
    OOD_MERGE = [
        ("OOD Compositional",            "compositional"),
        ("OOD Compositional Probes",     "compositional"),
        ("OOD Expanded — Single-Metric", "single-metric"),
        ("OOD Expanded — Cross-Metric",  "cross-metric"),
    ]
    OOD_MERGE_KEYS = {k for k, _ in OOD_MERGE}

    # ══════════════════════════════════════════════════════════════════════
    # Per-category tables (Binary Judgment, Trend Query, Enumeration, Numerical)
    # OOD subcategories are skipped here and rendered as a single merged table below.
    # ══════════════════════════════════════════════════════════════════════
    for cat_name, cat_info in TASK_CATEGORIES.items():
        if cat_name in OOD_MERGE_KEYS:
            continue
        tasks_in_cat = [t for t in cat_info["tasks"] if t in all_tasks]
        if not tasks_in_cat:
            continue

        metric_name = cat_info["metric_name"]
        lines.append(f"% ═══ {cat_name} ═══")
        lines.append("\\begin{table}[!htbp]")
        lines.append("\\centering")
        lines.append(f"\\caption{{{cat_info['caption']}}}")
        lines.append(f"\\label{{{cat_info['label']}}}")
        lines.append("\\begin{adjustbox}{max width=\\columnwidth}")
        col_spec, header_lines = _two_metric_header_n(
            metric_name, "Reasoning", names, add_delta=True)
        lines.append(f"\\begin{{tabular}}{{{col_spec}}}")
        lines.append("\\toprule")
        lines.extend(header_lines)
        lines.append("\\midrule")

        for task in tasks_in_cat:
            rs = [e["main"].get(task, {}) for e in exps]
            acc_vals = [r.get("binary_accuracy") for r in rs]
            acc_cells = bold_better_n(acc_vals)
            rea_cells = bold_better_n([r.get("reasoning_score") for r in rs])
            count = next((r.get("count") for r in rs if r.get("count")), None)
            count_s = str(int(count)) if count else "—"
            task_display = escape_latex(task)
            delta_cell = _delta_cell_pp(acc_vals)
            row_cells = ([f"\\texttt{{{task_display}}}"]
                         + acc_cells + rea_cells
                         + [count_s, delta_cell])
            lines.append(" & ".join(row_cells) + " \\\\")

        lines.append("\\bottomrule")
        lines.append("\\end{tabular}")
        lines.append("\\end{adjustbox}")
        lines.append("\\end{table}")
        lines.append("")

    # ══════════════════════════════════════════════════════════════════════
    # OOD merged: all 4 OOD subcategories in one table with a Category column.
    # ══════════════════════════════════════════════════════════════════════
    ood_rows: list[tuple[str, str, str]] = []  # (group_tag, task, line)
    for cat_name, tag in OOD_MERGE:
        cat_info = TASK_CATEGORIES.get(cat_name)
        if not cat_info: continue
        for task in cat_info["tasks"]:
            if task not in all_tasks: continue
            ood_rows.append((tag, task, ""))
    if ood_rows:
        lines.append("% ═══ OOD (all subcategories merged) ═══")
        lines.append("\\begin{table}[!htbp]")
        lines.append("\\centering")
        lines.append(
            "\\caption{Out-of-distribution evaluation, all subcategories merged. "
            "The \\textbf{Type} column tags each task by its OOD subcategory: "
            "\\textit{compositional} = compositional probes (legacy + length-targeted); "
            "\\textit{single-metric} = OOD primitives over a single time series; "
            "\\textit{cross-metric} = OOD compositions across two or more series. "
            "Accuracy = fraction of correct verdicts (or relative accuracy for numeric); "
            "Reasoning = LLM-judged quality of the explanation (0--1). "
            "Highlighted cells: a non-baseline model meets/beats the leftmost (baseline) column.}"
        )
        lines.append("\\label{tab:ood-merged}")
        lines.append("\\begin{adjustbox}{max width=\\columnwidth}")
        col_spec, header_lines = _two_metric_header_n(
            "Accuracy", "Reasoning", names, extra_label="Type", add_delta=True)
        lines.append(f"\\begin{{tabular}}{{{col_spec}}}")
        lines.append("\\toprule")
        lines.extend(header_lines)
        lines.append("\\midrule")
        last_tag = None
        for tag, task, _ in ood_rows:
            rs = [e["main"].get(task, {}) for e in exps]
            acc_vals = [r.get("binary_accuracy") for r in rs]
            acc_cells = bold_better_n(acc_vals)
            rea_cells = bold_better_n([r.get("reasoning_score") for r in rs])
            count = next((r.get("count") for r in rs if r.get("count")), None)
            count_s = str(int(count)) if count else "—"
            task_display = escape_latex(task)
            tag_display = escape_latex(tag) if tag != last_tag else ""
            if tag != last_tag and last_tag is not None:
                lines.append("\\midrule")
            last_tag = tag
            delta_cell = _delta_cell_pp(acc_vals)
            row_cells = ([f"\\texttt{{{task_display}}}", tag_display]
                         + acc_cells + rea_cells
                         + [count_s, delta_cell])
            lines.append(" & ".join(row_cells) + " \\\\")
        lines.append("\\bottomrule")
        lines.append("\\end{tabular}")
        lines.append("\\end{adjustbox}")
        lines.append("\\end{table}")
        lines.append("")

    # ══════════════════════════════════════════════════════════════════════
    # Set Matching (clustering / anticlustering) — F1 + Reasoning combined
    # ══════════════════════════════════════════════════════════════════════
    set_tasks = [t for t in SET_MATCHING_TASKS if t in all_tasks]
    if set_tasks:
        lines.append("% ═══ Set Matching ═══")
        lines.append("\\begin{table}[!htbp]")
        lines.append("\\centering")
        lines.append("\\caption{Set matching tasks: the model must identify which metrics belong to a cluster "
                     "(similar or opposite trends to a target). "
                     "F1 = harmonic mean of precision and recall over predicted vs.\\ ground-truth metric sets. "
                     "Reasoning = LLM-judged quality of the explanation (0--1).}")
        lines.append("\\label{tab:set-matching}")
        lines.append("\\begin{adjustbox}{max width=\\columnwidth}")
        col_spec, header_lines = _two_metric_header_n(
            "F1", "Reasoning", names, add_delta=True)
        lines.append(f"\\begin{{tabular}}{{{col_spec}}}")
        lines.append("\\toprule")
        lines.extend(header_lines)
        lines.append("\\midrule")

        for task in set_tasks:
            rs = [e["main"].get(task, {}) for e in exps]
            f1_vals = [r.get("f1") for r in rs]
            f1_cells = bold_better_n(f1_vals)
            rea_cells = bold_better_n([r.get("reasoning_score") for r in rs])
            count = next((r.get("count") for r in rs if r.get("count")), None)
            count_s = str(int(count)) if count else "—"
            task_display = escape_latex(task)
            delta_cell = _delta_cell_pp(f1_vals)
            row_cells = ([f"\\texttt{{{task_display}}}"]
                         + f1_cells + rea_cells
                         + [count_s, delta_cell])
            lines.append(" & ".join(row_cells) + " \\\\")

        lines.append("\\bottomrule")
        lines.append("\\end{tabular}")
        lines.append("\\end{adjustbox}")
        lines.append("\\end{table}")
        lines.append("")

    # ══════════════════════════════════════════════════════════════════════
    # Description Task Summary
    # ══════════════════════════════════════════════════════════════════════
    desc_rs = [e["main"].get("description", {}) for e in exps]
    if any(desc_rs):
        lines.append("% ═══ Description Task Summary ═══")
        lines.append("\\begin{table}[!htbp]")
        lines.append("\\centering")
        lines.append("\\caption{Description task: the model must describe the trend, seasonality, noise level, "
                     "and local events of a univariate time series. "
                     "Reasoning = LLM-judged quality of the explanation. "
                     "Overall = composite of classification accuracy, quantitative accuracy, "
                     "and reasoning quality across all four perspectives.}")
        lines.append("\\label{tab:description-summary}")
        lines.append(f"\\begin{{tabular}}{{l {'c '*N}}}")
        lines.append("\\toprule")
        names_row = " & ".join(escape_latex(n) for n in names)
        lines.append(f"Metric & {names_row} \\\\")
        lines.append("\\midrule")

        rea_cells = bold_better_n([r.get("reasoning_score") for r in desc_rs])
        ov_cells = bold_better_n([r.get("description_overall") for r in desc_rs])
        desc_n = next((r.get("count") for r in desc_rs if r.get("count")), 0)
        lines.append("Reasoning Score & " + " & ".join(rea_cells) + " \\\\")
        lines.append("Overall Score & " + " & ".join(ov_cells) + " \\\\")
        lines.append("\\midrule")
        lines.append(f"$n$ & \\multicolumn{{{N}}}{{c}}{{{int(desc_n)}}} \\\\")

        lines.append("\\bottomrule")
        lines.append("\\end{tabular}")
        lines.append("\\end{table}")
        lines.append("")

    # ══════════════════════════════════════════════════════════════════════
    # Description QA Breakdown by perspective
    # ══════════════════════════════════════════════════════════════════════
    lines.append("% ═══ Description QA Breakdown ═══")
    lines.append("\\begin{table}[!htbp]")
    lines.append("\\centering")
    lines.append("\\caption{Description task breakdown by perspective. "
                 "Classification = whether the model correctly identifies the category "
                 "(e.g.\\ trend type, noise level, presence of periodicity). "
                 "Quantitative = accuracy of predicted numerical values "
                 "(e.g.\\ amplitude, period, event position).}")
    lines.append("\\label{tab:description-breakdown}")
    lines.append("\\begin{adjustbox}{max width=\\columnwidth}")
    col_spec, header_lines = _two_metric_header_n(
        "Classification", "Quantitative", names, row_label="Perspective")
    lines.append(f"\\begin{{tabular}}{{{col_spec}}}")
    lines.append("\\toprule")
    lines.extend(header_lines)
    lines.append("\\midrule")

    for p in ["trend", "seasonal", "noise", "local"]:
        ds = [e["description_breakdown"].get(p, {}) for e in exps]
        cls_cells = bold_better_n([d.get("cls") for d in ds])
        q_cells = bold_better_n([d.get("quant") for d in ds])
        n_val = next((d.get("n") for d in ds if d.get("n")), 0)
        row_cells = [p.capitalize()] + cls_cells + q_cells + [str(int(n_val))]
        lines.append(" & ".join(row_cells) + " \\\\")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{adjustbox}")
    lines.append("\\end{table}")
    lines.append("")

    # ══════════════════════════════════════════════════════════════════════
    # Description Sub-Metrics
    # ══════════════════════════════════════════════════════════════════════
    lines.append("% ═══ Description Sub-Metric Details ═══")
    lines.append("\\begin{table}[!htbp]")
    lines.append("\\centering")
    lines.append("\\caption{Description quantitative sub-metrics. "
                 "Each value is a relative accuracy (1 = perfect match to ground truth). "
                 "Amplitude = magnitude accuracy, Position = temporal location accuracy, "
                 "Period = seasonal cycle length accuracy, "
                 "Strength = noise strength accuracy.}")
    lines.append("\\label{tab:description-submetrics}")
    lines.append(f"\\begin{{tabular}}{{l l {'c '*N}}}")
    lines.append("\\toprule")
    names_row = " & ".join(escape_latex(n) for n in names)
    lines.append(f"Perspective & Sub-metric & {names_row} \\\\")
    lines.append("\\midrule")

    sub_metrics = {
        "trend": [("amp", "Amplitude"), ("pos", "Position")],
        "seasonal": [("amp", "Amplitude"), ("period", "Period")],
        "noise": [("str", "Strength")],
        "local": [("amp", "Amplitude"), ("pos", "Position")],
    }
    for p, subs in sub_metrics.items():
        ds = [e["description_breakdown"].get(p, {}) for e in exps]
        first = True
        for key, label in subs:
            cells = bold_better_n([d.get(key) for d in ds])
            p_label = p.capitalize() if first else ""
            first = False
            row_cells = [p_label, label] + cells
            lines.append(" & ".join(row_cells) + " \\\\")
        lines.append("\\midrule")

    lines[-1] = "\\bottomrule"
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")
    lines.append("")

    # ══════════════════════════════════════════════════════════════════════
    # Trend Direction Detail
    # ══════════════════════════════════════════════════════════════════════
    lines.append("% ═══ Trend Direction Detail ═══")
    lines.append("\\begin{table}[!htbp]")
    lines.append("\\centering")
    lines.append("\\caption{Trend analysis detail. "
                 "Direction Accuracy = correct overall trend type classification. "
                 "Segment F1 = harmonic mean of precision and recall over matched trend segments. "
                 "Boundary Position = temporal accuracy of segment start/end points.}")
    lines.append("\\label{tab:trend-detail}")
    lines.append(f"\\begin{{tabular}}{{l {'c '*N}c}}")
    lines.append("\\toprule")
    names_row = " & ".join(escape_latex(n) for n in names)
    lines.append(f"Metric & {names_row} & $n$ \\\\")
    lines.append("\\midrule")

    trend_metrics = [
        ("direction_acc", "Direction Accuracy", "direction_n"),
        ("seg_F1", "Segment F1", "seg_n"),
        ("seg_boundary", "Boundary Position Acc", "seg_n"),
        ("reasoning", "Reasoning Score", "reasoning_n"),
        ("overall", "Overall Score", "overall_n"),
    ]
    for key, label, n_key in trend_metrics:
        cells = bold_better_n([e["trend_detail"].get(key) for e in exps])
        n_val = next((e["trend_detail"].get(n_key) for e in exps
                      if e["trend_detail"].get(n_key)), "")
        n_str = str(n_val) if n_val else ""
        row_cells = [label] + cells + [n_str]
        lines.append(" & ".join(row_cells) + " \\\\")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")
    lines.append("")

    # ══════════════════════════════════════════════════════════════════════
    # TSEvol by Strategy
    # ══════════════════════════════════════════════════════════════════════
    all_strategies_union: set[str] = set()
    for e in exps:
        all_strategies_union |= set(e["tsevol_by_strategy"].keys())
    all_strategies = sorted(all_strategies_union)
    if all_strategies:
        lines.append("% ═══ TSEvol by Strategy ═══")
        lines.append("\\begin{table}[!htbp]")
        lines.append("\\centering")
        lines.append("\\caption{TSEvol (time series evolution) reasoning scores by strategy. "
                     "Each strategy tests a different reasoning skill "
                     "(causal, deductive, constraint satisfaction, etc.). "
                     "Score = LLM-judged quality of the evolved response (0--1).}")
        lines.append("\\label{tab:tsevol}")
        lines.append(f"\\begin{{tabular}}{{l {'c '*N}c}}")
        lines.append("\\toprule")
        names_row = " & ".join(escape_latex(n) for n in names)
        lines.append(f"Strategy & {names_row} & $n$ \\\\")
        lines.append("\\midrule")

        for strat in all_strategies:
            if strat == "OVERALL":
                continue
            ss = [e["tsevol_by_strategy"].get(strat, {}) for e in exps]
            cells = bold_better_n([s.get("score") for s in ss])
            n_val = next((s.get("n") for s in ss if s.get("n")), "")
            row_cells = [strat] + cells + [str(n_val) if n_val else ""]
            lines.append(" & ".join(row_cells) + " \\\\")

        # Overall row
        lines.append("\\midrule")
        ovs = [e["tsevol_by_strategy"].get("OVERALL", {}) for e in exps]
        ov_cells = bold_better_n([o.get("score") for o in ovs])
        ov_n = next((o.get("n") for o in ovs if o.get("n")), "")
        row_cells = ["\\textbf{Overall}"] + ov_cells + [str(ov_n) if ov_n else ""]
        lines.append(" & ".join(row_cells) + " \\\\")

        lines.append("\\bottomrule")
        lines.append("\\end{tabular}")
        lines.append("\\end{table}")
        lines.append("")

    # ══════════════════════════════════════════════════════════════════════
    # Clustering by GT Length
    # ══════════════════════════════════════════════════════════════════════
    all_gt_lens_union: set = set()
    for e in exps:
        all_gt_lens_union |= set(e["clustering_by_len"].keys())
    all_gt_lens = sorted(all_gt_lens_union)
    if all_gt_lens:
        lines.append("% ═══ Clustering by GT Length (hidden — uncomment \\iffalse/\\fi to show) ═══")
        lines.append("\\iffalse")
        lines.append("\\begin{table}[!htbp]")
        lines.append("\\centering")
        lines.append("\\caption{Clustering F1 stratified by ground-truth cluster size. "
                     "Smaller clusters are harder to identify; "
                     "GT Len~0 means no metrics should be selected.}")
        lines.append("\\label{tab:clustering-by-len}")
        lines.append(f"\\begin{{tabular}}{{r {'c '*N}c}}")
        lines.append("\\toprule")
        names_row = " & ".join(escape_latex(n) for n in names)
        lines.append(f"GT Len & {names_row} & $n$ \\\\")
        lines.append("\\midrule")

        for gl in all_gt_lens:
            cs = [e["clustering_by_len"].get(gl, {}) for e in exps]
            cells = bold_better_n([c.get("f1") for c in cs])
            n_val = next((c.get("sample_count") for c in cs if c.get("sample_count")), "")
            row_cells = [str(gl)] + cells + [str(n_val) if n_val else ""]
            lines.append(" & ".join(row_cells) + " \\\\")

        lines.append("\\bottomrule")
        lines.append("\\end{tabular}")
        lines.append("\\end{table}")
        lines.append("\\fi")
        lines.append("")

    # ══════════════════════════════════════════════════════════════════════
    # Task Type Examples — one Q&A per category
    # ══════════════════════════════════════════════════════════════════════
    if include_examples:
        lines.append("% ═══ Task Type Examples ═══")
        lines.append(_generate_examples_section())
        lines.append("")

    return "\n".join(lines)


# ── Hand-curated short examples per task category ────────────────────
# Keep these concise — one sentence Q, one sentence A.
# <ts> stands for omitted time series data.

TASK_EXAMPLES = {
    "Binary Judgment": [
        {
            "type": "yes\\_no",
            "q": "Is there a downward spike ending near point 219 in Grid Load?",
            "a": "Yes, Grid Load has a downward spike ending near point 219.",
            "acc": "1.0 if the yes/no verdict matches ground truth, else 0.0.",
            "rea": "LLM judge rates how well the model explains \\emph{why} it answered yes or no (0--1).",
        },
        {
            "type": "correlation",
            "q": "Do Revenue Growth and Market Share show correlated trends?",
            "a": "Yes, both show a decreasing trend from point 0 to 2784, indicating correlation.",
            "acc": "1.0 if the correlation verdict matches, else 0.0.",
            "rea": "Quality of the trend comparison and evidence cited.",
        },
        {
            "type": "anti\\_judgment",
            "q": "Do Resource Utilization and Throughput show anticorrelated trends, "
                 "AND does Throughput's noise strength exceed 0.08?",
            "a": "Yes. They show anticorrelated trends, and noise strength (0.14) exceeds 0.08.",
            "acc": "1.0 if the compound yes/no verdict is correct, else 0.0.",
            "rea": "Quality of reasoning about both conditions.",
        },
    ],
    "Trend Query": [
        {
            "type": "cross\\_trend\\_query",
            "q": "When Social Media Ad Performance decreases, what does Cost Per Click (CPC) do?",
            "a": "CPC predominantly shows a decreasing trend.",
            "acc": "1.0 if the trend direction (increase/decrease/steady) is correct.",
            "rea": "Quality of cross-metric trend analysis.",
        },
        {
            "type": "segment\\_trend\\_dominance",
            "q": "In the range [42, 2169], which trend type dominates Sales Trends?",
            "a": "Increasing (upward) trend dominates.",
            "acc": "1.0 if the dominant trend type is correct.",
            "rea": "Quality of segment duration analysis.",
        },
    ],
    "Enumeration": [
        {
            "type": "local\\_enumeration",
            "q": "In Tablespace Usage, identify the widest local event.",
            "a": "The widest local event is a decrease-after-upward-spike spanning 10 timesteps.",
            "acc": "1.0 if the exact answer (type, span, count) matches ground truth.",
            "rea": "Quality of the event identification reasoning.",
        },
        {
            "type": "segment\\_enumeration",
            "q": "How many trend segments are in Latency?",
            "a": "There are 2 trend segments in Latency.",
            "acc": "1.0 if the count is exactly correct.",
            "rea": "Quality of segment identification reasoning.",
        },
    ],
    "Numerical \\& Temporal": [
        {
            "type": "stat\\_numerical",
            "q": "What is the mean of Click-Through Rates in each 16-step window?",
            "a": "[51.07, 52.13, 53.12, \\ldots, 59.58] (16 windowed means).",
            "acc": "1.0 if each value is within tolerance of ground truth.",
            "rea": "Quality of the statistical computation shown.",
        },
        {
            "type": "temporal\\_position",
            "q": "At what position does the first local event occur in Engagement Metrics?",
            "a": "Position 2096.",
            "acc": "Relative accuracy: $\\max(0,\\; 1 - |\\text{pred} - \\text{gt}|\\,/\\,\\text{seq\\_len})$.",
            "rea": "Quality of the position identification reasoning.",
        },
    ],
    "Set Matching": [
        {
            "type": "clustering",
            "q": "Which metrics in this Marketing system cluster with Sales Pipeline Metrics (similar trends)?",
            "a": "Ad Click Rates, Customer Lifetime Value, Return on Ad Spend, Sales Pipeline Metrics.",
            "acc": "F1 over predicted vs.\\ ground-truth metric sets.",
            "rea": "Quality of the trend similarity analysis.",
        },
        {
            "type": "anticlustering",
            "q": "Which metrics show opposite trends to Television Ratings?",
            "a": "Live Event Attendance, Revenue from Ads, Virtual Event Participation.",
            "acc": "F1 over predicted vs.\\ ground-truth opposite-trend sets.",
            "rea": "Quality of the anti-trend analysis.",
        },
    ],
    "Description": [
        {
            "type": "description",
            "q": "Examine the behavior of Dividend Yields focusing on local characteristics.",
            "a": "Downward spike ends around point 37, suggesting reduced dividend payments or "
                 "increased stock price.",
            "acc": "Classification: correct event type. Quantitative: amplitude \\& position accuracy.",
            "rea": "Quality of the domain-specific explanation.",
        },
    ],
    "TSEvol": [
        {
            "type": "tsevol",
            "q": "(Multi-turn evolved question about Post Frequency patterns in Social Media.)",
            "a": "The smooth noise level indicates consistent posting activity. "
                 "The upward spike suggests a viral event\\ldots",
            "acc": "N/A (reasoning-only evaluation).",
            "rea": "LLM judge rates factual accuracy, grounding in data, and reasoning depth (0--1).",
        },
    ],
}


def _generate_examples_section() -> str:
    """Generate the LaTeX for the task-type examples appendix."""
    lines = []
    lines.append("\\clearpage")
    lines.append("\\section*{Appendix: Task Type Examples}")
    lines.append("\\addcontentsline{toc}{section}{Appendix: Task Type Examples}")
    lines.append("")
    lines.append("Each evaluation task tests a different aspect of time series understanding. "
                 "Below we show one representative question--answer pair per category, "
                 "with explanations of how \\textbf{Accuracy} and \\textbf{Reasoning} are measured. "
                 "Time series data (\\texttt{<ts>}) is omitted for brevity.")
    lines.append("")

    for cat_name, examples in TASK_EXAMPLES.items():
        lines.append(f"\\subsection*{{{cat_name}}}")
        for ex in examples:
            lines.append("\\begin{tcolorbox}[colback=gray!5, colframe=gray!60, "
                         "title={\\texttt{" + ex["type"] + "}}, fonttitle=\\bfseries\\small, "
                         "boxrule=0.4pt, arc=2pt, left=4pt, right=4pt, top=2pt, bottom=2pt]")
            lines.append("\\small")
            lines.append(f"\\textbf{{Q:}} {ex['q']} \\\\[3pt]")
            lines.append(f"\\textbf{{A:}} {ex['a']} \\\\[3pt]")
            lines.append(f"\\textcolor{{teal}}{{\\textbf{{Accuracy:}}}} {ex['acc']} \\\\[1pt]")
            lines.append(f"\\textcolor{{violet}}{{\\textbf{{Reasoning:}}}} {ex['rea']}")
            lines.append("\\end{tcolorbox}")
            lines.append("\\vspace{2pt}")
        lines.append("")

    return "\n".join(lines)


def _run_single_pair(
    dirs,
    names: list[str],
    section_title: str | None,
    include_preamble: bool,
    include_examples: bool,
) -> tuple[str, list[dict]]:
    """Parse summary files from N exp dirs and render the comparison.

    Returns (latex_text, length_bucket_per_model_list).
    Length buckets are empty dicts for non-synthetic datasets.

    Backward-compat: accepts (dir1, dir2) as separate args via *args spread,
    or a list of dirs.
    """
    # Normalize: if positional 2-arg style was used, dirs is a Path
    if isinstance(dirs, Path):
        # Caller passed (dir1, dir2, names, ...) — old signature; combine.
        # This branch is only entered if mistakenly called old-style.
        raise TypeError("_run_single_pair now takes a list of dirs as first arg")
    dirs_list = list(dirs)
    exps = []
    for d in dirs_list:
        summary = d / "evaluation_report_summary.txt"
        if not summary.exists():
            raise FileNotFoundError(f"Missing {summary}")
        exps.append(parse_summary(summary))
    latex = generate_latex(
        exps,
        names,
        section_title=section_title,
        include_preamble=include_preamble,
        include_examples=include_examples,
    )
    lb_per_model = [
        parse_length_bucket_overall(d / "length_bucket_overall.csv")
        for d in dirs_list
    ]
    return latex, lb_per_model


def main():
    parser = argparse.ArgumentParser(description="Convert evaluation results to LaTeX tables")
    parser.add_argument(
        "dirs", nargs="+",
        help=(
            "Two or more experiment directories. "
            "Default mode: each dir contains evaluation_report_summary.txt directly "
            "(e.g. exp/sft/test_lite exp/rl/test_lite). "
            "With --combine: each dir is a model root containing per-dataset subdirs "
            "(e.g. exp/sft_nothink exp/sft exp/rl). 3+ dirs render side-by-side."
        ),
    )
    parser.add_argument("-o", "--output", default=None, help="Output .tex file (default: <dir1-parent>/comparison_tables.tex)")
    parser.add_argument(
        "--names", nargs="+", default=None,
        help="Display names for experiments (default: folder names)"
    )
    parser.add_argument(
        "--combine", action="store_true",
        help=(
            "Auto-discover per-dataset subdirs under the two model roots and emit "
            "one LaTeX document with a section per dataset plus a consolidated "
            "length-generalization table across synthetic datasets."
        ),
    )
    parser.add_argument(
        "--datasets", nargs="+", default=None,
        help="With --combine: restrict to named dataset subdirs (default: all present in both roots).",
    )
    parser.add_argument(
        "--fragment", action="store_true",
        help=(
            "Emit a LaTeX fragment only (no \\documentclass / \\begin{document}). "
            "Default is a self-contained compilable document with the required "
            "\\usepackage{booktabs,adjustbox,xcolor,tcolorbox,amsmath,hyperref}."
        ),
    )
    parser.add_argument(
        "--title", default=None,
        help="Document title when emitting a self-contained document (default: 'Evaluation Results: <names>').",
    )
    args = parser.parse_args()

    def _wrap(latex: str) -> str:
        if args.fragment:
            return latex
        if args.title:
            title = args.title
        else:
            sep = " vs.\\ "
            title = "Evaluation Results: " + sep.join(escape_latex(n) for n in names)
        return full_document_preamble(title=title) + latex + full_document_postamble()

    if len(args.dirs) < 2:
        print("Error: at least 2 experiment directories required for comparison", file=sys.stderr)
        sys.exit(1)

    dirs = [Path(d) for d in args.dirs]
    names = args.names or [d.name.replace("_", " ").title() for d in dirs]
    if len(names) != len(dirs):
        print(f"Error: {len(dirs)} dirs but {len(names)} names", file=sys.stderr)
        sys.exit(1)

    if not args.combine:
        # Original single-dataset comparison mode.
        latex, _ = _run_single_pair(
            dirs, names,
            section_title=None,
            include_preamble=args.fragment,  # inner preamble only when not wrapping
            include_examples=True,
        )
        latex = _wrap(latex)
        if args.output:
            out_path = Path(args.output)
        else:
            out_path = dirs[0].parent / "comparison_tables.tex"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(latex)
        print(f"LaTeX written to {out_path}", file=sys.stderr)
        print(latex)
        return

    # Combine mode: discover dataset subdirs.
    def _datasets_in(root: Path) -> set[str]:
        if not root.is_dir():
            return set()
        return {
            p.name
            for p in root.iterdir()
            if p.is_dir() and (p / "evaluation_report_summary.txt").exists()
        }

    # Common = intersection of all model dirs' datasets
    per_root_sets = [_datasets_in(d) for d in dirs]
    common_set = set.intersection(*per_root_sets)
    common = sorted(common_set)
    if args.datasets:
        common = [d for d in common if d in args.datasets]
    if not common:
        print(
            f"Error: no shared dataset subdirs with evaluation_report_summary.txt "
            f"under all of {[str(d) for d in dirs]}",
            file=sys.stderr,
        )
        sys.exit(1)
    union_sets = set.union(*per_root_sets)
    missing_on_some = union_sets - common_set
    if missing_on_some:
        print(
            f"WARN: datasets present in some but not all roots (skipped): {sorted(missing_on_some)}",
            file=sys.stderr,
        )

    # Preferred ordering: lite first, then length buckets in order, then length buckets.
    order_hint = [
        "test_lite", "test", "test_len_near", "test_len_far", "test_len_extreme",
    ]
    common.sort(key=lambda s: (order_hint.index(s) if s in order_hint else len(order_hint), s))

    all_parts: list[str] = []
    # length_bucket_rows: list of (dataset_label, [model1_buckets, model2_buckets, ...])
    length_bucket_rows: list[tuple[str, list[dict]]] = []
    first = True
    for ds in common:
        ds_dirs = [d / ds for d in dirs]
        latex, lb_per_model = _run_single_pair(
            ds_dirs, names,
            section_title=f"Results: \\texttt{{{ds.replace('_', chr(92)+'_')}}}",
            # In wrapped mode the outer preamble already has \definecolor etc.
            # In fragment mode we need it in the first inner section.
            include_preamble=(first and args.fragment),
            include_examples=False,
        )
        all_parts.append(latex)
        first = False
        if any(lb_per_model):
            length_bucket_rows.append((ds, lb_per_model))

    # Prepend consolidated length-generalization table if we have any buckets.
    consolidated = []
    if length_bucket_rows:
        consolidated = render_length_bucket_table(length_bucket_rows, names)

    # Consolidated OOD summary table — pass full dirs list (function uses
    # last vs first for delta computation in N-way mode).
    ood_summary = render_ood_summary_consolidated(dirs, names)

    # Compositional-probes × length-bucket table.
    probes_table = render_composition_probes_by_length(dirs, names)

    # Append examples appendix once at the end.
    appendix = ["% ═══ Task Type Examples ═══", _generate_examples_section(), ""]

    # Flush floats between dataset sections — many tables × many datasets overflows
    # LaTeX's float queue otherwise ("Output loop — dead cycles" error).
    parts = [all_parts[0]] + consolidated + ood_summary + probes_table + ["\\FloatBarrier"]
    for p in all_parts[1:]:
        parts.append(p)
        parts.append("\\FloatBarrier")
    parts.extend(appendix)
    latex = "\n".join(parts)
    latex = _wrap(latex)

    if args.output:
        out_path = Path(args.output)
    else:
        out_path = dirs[0].parent / "comparison_tables.tex"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(latex)
    print(f"LaTeX written to {out_path}", file=sys.stderr)
    print(f"Datasets included: {common}", file=sys.stderr)


if __name__ == "__main__":
    main()
