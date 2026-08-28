"""Build docs/qa_taxonomy.jsonl: structured taxonomy of all 115 QA eval_types.

Schema per line:
  eval_type        — string identifier (matches reward/eval routing)
  regime           — "atomic" | "easy_composite" | "hard_composite" | "ood_composite"
  subgroup         — descriptive bucket within regime
  atoms            — list of atom names (from F_atom)
  subskills        — list of sub-skill names (from F_op)
  in_sft           — bool (verified against data/train_sft.jsonl on 2026-05-03)
  in_rl_public     — bool (verified against data/train_rl_public.jsonl)
  bridge_present   — bool (true if at least one sample has eval_metadata.bridge=True)
  example_question — one representative question text
  notes            — free-form notes for special cases

Verified against data on disk dated 2026-04-23. Re-run if data is regenerated.
"""
import json
import os
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent
os.chdir(str(REPO))

OUT = Path("docs/qa_taxonomy.jsonl")

# ---------------------------------------------------------------------------
# Taxonomy table — one tuple per eval_type
# Format: (eval_type, regime, subgroup, atoms, subskills, example, notes)
# ---------------------------------------------------------------------------
ROWS = [
    # ===== ATOMIC (13) =====
    ("atomic_global_mean", "atomic", "single-metric", ["Mean"], [],
     "What mean value does {metric} take?", ""),
    ("atomic_interval_mean", "atomic", "single-metric", ["Mean"], [],
     "What is the average value of {metric} from position a to b?", ""),
    ("atomic_chunked_means", "atomic", "single-metric", ["Mean"], [],
     "What are the means of consecutive 16-point chunks of {metric}?", ""),
    ("atomic_global_std", "atomic", "single-metric", ["Std"], [],
     "What is the spread of {metric} measured by standard deviation?", ""),
    ("atomic_interval_std", "atomic", "single-metric", ["Std"], [],
     "What is the standard deviation of {metric} over [a, b]?", ""),
    ("atomic_percentile", "atomic", "single-metric", ["Percentile"], [],
     "Report the 50th percentile value for {metric}.", ""),
    ("atomic_max_value", "atomic", "single-metric", ["ExtVal"], [],
     "What is the largest value of {metric}?", ""),
    ("atomic_min_value", "atomic", "single-metric", ["ExtVal"], [],
     "What is the minimum value of {metric}?", ""),
    ("atomic_max_position", "atomic", "single-metric", ["ExtPos"], [],
     "Where is the highest value in {metric}?", ""),
    ("atomic_min_position", "atomic", "single-metric", ["ExtPos"], [],
     "At which index does {metric} attain its minimum?", ""),
    ("atomic_event_enumeration", "atomic", "single-metric", ["EventEnum"], [],
     "Enumerate the local events in {metric}, including type, position, amplitude.", ""),
    ("atomic_trend_enumeration", "atomic", "single-metric", ["SegEnum"], [],
     "Describe the segments of the trend in {metric} over time.", ""),
    ("atomic_periodic_description", "atomic", "single-metric", ["Period"], [],
     "Describe the periodic behavior of {metric}.", ""),

    # ===== EASY COMPOSITE (36) =====
    # Cross-metric atomic reductions (5) — strictly 2-step (atom + cross-metric reduce)
    ("atomic_cross_stat_compare", "easy_composite", "cross-metric atomic",
     ["Mean", "Mean"], ["Compare"],
     "Is {metricB}'s mean higher than {metricA}'s?",
     "atomic_cross_* moved into easy_composite because they apply per-metric atom + cross-metric reduction (2 ops)."),
    ("atomic_cross_ranking", "easy_composite", "cross-metric atomic",
     ["Mean"], ["Argmax"],
     "Identify the metric with the largest standard deviation.", ""),
    ("atomic_cross_filtering", "easy_composite", "cross-metric atomic",
     ["TrendClassify"], ["Filter"],
     "Which metrics show an increase trend?", ""),
    ("atomic_cross_counting", "easy_composite", "cross-metric atomic",
     ["Mean"], ["Count"],
     "Count the metrics whose mean exceeds tau.", ""),
    ("atomic_cross_trend_align", "easy_composite", "cross-metric atomic",
     ["TrendClassify", "TrendClassify"], ["Compare"],
     "Is the overall trend of {metricA} the same as {metricB}?", ""),

    # Legacy single-metric (12)
    ("yes_no", "easy_composite", "legacy single-metric",
     ["EventEnum"], ["Locate", "Threshold"],
     "Does {metric} have a noticeable event around point p within tolerance tau?", ""),
    ("periodicity", "easy_composite", "legacy single-metric",
     ["Period"], ["Count"],
     "Compute the estimated period for {metric}.", ""),
    ("change_point", "easy_composite", "legacy single-metric",
     ["SegEnum"], ["Argmax", "Count"],
     "How many times does {metric} change its dominant trend?", ""),
    ("stat_numerical", "easy_composite", "legacy single-metric",
     ["Mean", "Std", "Percentile"], ["Arith", "Compare"],
     "How does the std of {metric} change from the first half to the second half?", ""),
    ("segment_judgment", "easy_composite", "legacy single-metric",
     ["Mean"], ["Arith", "Threshold"],
     "If a 'significant upward shift' is when (mean(Q3)-mean(Q1))/range > 15%, does {metric} qualify?", ""),
    ("segment_trend_dominance", "easy_composite", "legacy single-metric",
     ["SegEnum"], ["Filter", "Sum", "Argmax"],
     "Between points a and b in {metric}, which trend behavior dominates?", ""),
    ("local_enumeration", "easy_composite", "legacy single-metric",
     ["EventEnum"], ["Filter", "Count", "Argmax"],
     "What is the shortest local event in {metric} in terms of duration?", ""),
    ("segment_enumeration", "easy_composite", "legacy single-metric",
     ["SegEnum"], ["Filter", "Count", "Argmax"],
     "How many trend segments show an increase in {metric}?", ""),
    ("transition_enumeration", "easy_composite", "legacy single-metric",
     ["SegEnum"], ["Filter", "Count", "Argmax"],
     "Which adjacent segment pair pattern is most common in {metric}?", ""),
    ("event_segment_enumeration", "easy_composite", "legacy single-metric",
     ["EventEnum", "SegEnum"], ["Filter", "Locate", "Count", "Argmax"],
     "Count the local events in {metric} that fall within decrease segments.", ""),
    ("temporal_position", "easy_composite", "legacy single-metric",
     ["EventEnum"], ["Locate"],
     "Where does the last local event appear in {metric}?", ""),
    ("duration_proportion", "easy_composite", "legacy single-metric",
     ["SegEnum"], ["Filter", "Sum", "Compare"],
     "In {metric}, which has a longer total duration: increase or decrease?", ""),

    # Legacy cross-metric (8)
    ("correlation", "easy_composite", "legacy cross-metric",
     ["TrendClassify", "TrendClassify"], ["Compare"],
     "Evaluate the trend similarity between {metricA} and {metricB}; transitions within 2 timesteps are aligned.", ""),
    ("anticorrelation", "easy_composite", "legacy cross-metric",
     ["TrendClassify", "TrendClassify"], ["Compare"],
     "Determine whether {metricA} and {metricB} exhibit opposite trend behavior.", ""),
    ("clustering", "easy_composite", "legacy cross-metric",
     ["TrendClassify", "EventEnum"], ["Filter"],
     "Identify metrics whose local behavior near point p resembles {metricAnchor}.", ""),
    ("anticlustering", "easy_composite", "legacy cross-metric",
     ["TrendClassify"], ["Filter"],
     "Find metrics that move opposite to {metricAnchor}.", ""),
    ("cross_metric_enumeration", "easy_composite", "legacy cross-metric",
     ["Mean", "Std"], ["Filter", "Argmax"],
     "Which metric has the highest standard deviation?", ""),
    ("cross_stat_judgment", "easy_composite", "legacy cross-metric",
     ["Std", "Std"], ["Arith", "Threshold"],
     "An 'instability gap' occurs when std({metricA}) is at least r-times std({metricB}); does it occur?", ""),
    ("anti_judgment", "easy_composite", "legacy cross-metric",
     ["TrendClassify", "TrendClassify", "Std"], ["Compare", "Threshold"],
     "An 'anticorrelated high-noise state' is when {metricA} and {metricB} are anti-correlated AND noise of {metricB} exceeds tau; does it hold?", ""),
    ("cross_trend_query", "easy_composite", "legacy cross-metric",
     ["TrendClassify", "TrendClassify"], ["Filter"],
     "When {metricA} is increasing, what does {metricB}'s trend look like?", ""),

    # Bridge-RL single-metric (8)
    ("rl_half_mean_compare", "easy_composite", "bridge-RL single-metric",
     ["Mean", "Mean"], ["Compare"],
     "Does the first-half mean of {metric} exceed the second-half mean?",
     "BRIDGE-ONLY: 1803 samples in train_sft (all bridge=True), 0 in train_rl_public. The only easy composite without an RL counterpart in the realized data."),
    ("rl_volatility_change", "easy_composite", "bridge-RL single-metric",
     ["Std", "Std"], ["Compare"],
     "Is the std of the second half of {metric} higher than the first?", ""),
    ("rl_event_in_trend_type", "easy_composite", "bridge-RL single-metric",
     ["EventEnum", "SegEnum"], ["Locate", "Filter", "Threshold"],
     "Does any local event of {metric} fall in an increase segment?", ""),
    ("rl_max_in_trend_type", "easy_composite", "bridge-RL single-metric",
     ["ExtPos", "SegEnum"], ["Locate", "Compare"],
     "Does the global max of {metric} fall within a decrease segment?", ""),
    ("rl_amplitude_vs_std", "easy_composite", "bridge-RL single-metric",
     ["EventEnum", "Std"], ["Argmax", "Arith", "Threshold"],
     "Is the seasonal amplitude of {metric} greater than k-sigma?", ""),
    ("rl_event_count_by_type", "easy_composite", "bridge-RL single-metric",
     ["EventEnum"], ["Filter", "Count"],
     "How many decrease events are there in {metric}?", ""),
    ("rl_type_of_longest", "easy_composite", "bridge-RL single-metric",
     ["SegEnum"], ["Argmax"],
     "Is the longest segment in {metric} an increase, decrease, or steady segment?", ""),
    ("rl_type_duration_fraction", "easy_composite", "bridge-RL single-metric",
     ["SegEnum"], ["Filter", "Sum", "Arith"],
     "What share of {metric} is decreasing?", ""),

    # Bridge-RL cross-metric (3)
    ("rl_cross_stat_ratio", "easy_composite", "bridge-RL cross-metric",
     ["Std", "Std"], ["Arith", "Threshold"],
     "Does {metricA} have a std at least r-times that of {metricB}?", ""),
    ("rl_cross_full_ordering", "easy_composite", "bridge-RL cross-metric",
     ["Mean"], ["Rank"],
     "Rank the metrics by mean from highest to lowest.", ""),
    ("rl_cross_trend_concordance", "easy_composite", "bridge-RL cross-metric",
     ["TrendClassify"], ["Filter", "Count", "Argmax", "Arith"],
     "How dominant is the most common trend? Report the mode share.", ""),

    # ===== HARD COMPOSITE (32) =====
    ("rl_event_count", "hard_composite", "single-metric",
     ["EventEnum"], ["Count"],
     "How many spikes, dips, or sudden changes are in {metric}?", ""),
    ("rl_segment_count", "hard_composite", "single-metric",
     ["SegEnum"], ["Count"],
     "How many distinct trend segments does {metric} have?", ""),
    ("rl_segment_duration", "hard_composite", "single-metric",
     ["SegEnum"], ["Locate"],
     "How many points are in the segment of {metric} containing index p?", ""),
    ("rl_segment_type_at_pos", "hard_composite", "single-metric",
     ["SegEnum"], ["Locate"],
     "What is happening in {metric} at index p?", ""),
    ("rl_segment_mean_compare", "hard_composite", "single-metric",
     ["SegEnum", "Mean"], ["Locate", "Compare"],
     "Is the mean of the segment at index p greater than at q?", ""),
    ("rl_longest_segment", "hard_composite", "single-metric",
     ["SegEnum"], ["Argmax"],
     "What is the duration of the longest trend segment in {metric}?", ""),
    ("rl_dominant_trend_type", "hard_composite", "single-metric",
     ["SegEnum"], ["Filter", "Count", "Argmax"],
     "What is the dominant trend type by segment count in {metric}?", ""),
    ("rl_max_amplitude_event", "hard_composite", "single-metric",
     ["EventEnum"], ["Argmax"],
     "Find the local event of {metric} with the largest amplitude; report its type.", ""),
    ("rl_event_type_at_pos", "hard_composite", "single-metric",
     ["EventEnum"], ["Locate"],
     "What is the nearest event type to position p in {metric}?", ""),
    ("rl_event_near_extremum", "hard_composite", "single-metric",
     ["ExtPos", "EventEnum"], ["Locate", "Threshold"],
     "Does a local event of {metric} occur near its global max or min?", ""),
    ("rl_extrema_same_half", "hard_composite", "single-metric",
     ["ExtPos", "ExtPos"], ["Locate", "Compare"],
     "Are the global max and min of {metric} in the same half?", ""),
    ("rl_max_in_first_half", "hard_composite", "single-metric",
     ["ExtPos"], ["Locate"],
     "Does the global max of {metric} occur before the midpoint?", ""),
    ("rl_range", "hard_composite", "single-metric",
     ["ExtVal", "ExtVal"], ["Arith"],
     "What is the spread of {metric} (max minus min)?", ""),
    ("rl_normalized_range", "hard_composite", "single-metric",
     ["ExtVal", "ExtVal", "Std"], ["Arith"],
     "What is the normalized range of {metric}?", ""),
    ("rl_amplitude_vs_range", "hard_composite", "single-metric",
     ["EventEnum", "ExtVal", "ExtVal"], ["Argmax", "Arith", "Threshold"],
     "Does the peak event amplitude of {metric} exceed half the value range?", ""),
    ("rl_half_mean_diff", "hard_composite", "single-metric",
     ["Mean", "Mean"], ["Arith"],
     "Compute |mean(first half) - mean(second half)|.", ""),
    ("rl_mean_shift", "hard_composite", "single-metric",
     ["Mean", "Mean"], ["Arith", "Threshold"],
     "Is there a mean shift greater than tau between the first and last thirds of {metric}?", ""),
    ("rl_condition_recovery", "hard_composite", "single-metric",
     ["Mean", "Mean"], ["Arith", "Threshold"],
     "Is |mean(Q1)-mean(Q4)| < tau for {metric}?", ""),
    ("rl_mean_stability", "hard_composite", "single-metric",
     ["Mean", "Mean"], ["Filter", "Compare"],
     "Are all 16-point chunk means of {metric} within tau of the global mean?", ""),
    ("rl_median_mean_close", "hard_composite", "single-metric",
     ["Percentile", "Mean"], ["Arith", "Threshold"],
     "Is {metric}'s median within tau of its mean?", ""),
    ("rl_interval_comparison", "hard_composite", "single-metric",
     ["Mean", "Mean"], ["Compare"],
     "Is the mean of {metric} on [a1,b1] greater than on [a2,b2]?", ""),
    ("rl_cycle_count", "hard_composite", "single-metric",
     ["Period"], ["Count"],
     "Count the approximate number of complete cycles in {metric}.", ""),
    ("rl_period_estimate", "hard_composite", "single-metric",
     ["Period"], [],
     "Estimate the period of {metric} in time steps.", ""),
    ("rl_has_periodicity", "hard_composite", "single-metric",
     ["Period"], ["Threshold"],
     "Does {metric} contain a periodic component?", ""),
    ("rl_corr_count", "hard_composite", "cross-metric",
     ["TrendClassify"], ["Filter", "Count", "Argmax"],
     "Which trend pattern is most common across the metrics?",
     "Reframed 2026-04-17 from pair-count to most-common-trend-direction (mode of trend labels)."),
    ("rl_cluster_count", "hard_composite", "cross-metric",
     ["TrendClassify"], ["Filter", "Count"],
     "Into how many groups do the metrics cluster by trend?", ""),
    ("rl_cluster_dominant", "hard_composite", "cross-metric",
     ["TrendClassify"], ["Filter", "Count", "Argmax"],
     "What is the dominant trend cluster?", ""),
    ("rl_cross_event_sync", "hard_composite", "cross-metric",
     ["EventEnum", "EventEnum"], ["Locate", "Threshold"],
     "Do {metricA} and {metricB} have temporally close events?", ""),
    ("rl_cross_period_compare", "hard_composite", "cross-metric",
     ["Period", "Period"], ["Compare"],
     "Does {metricA} oscillate faster than {metricB}?", ""),
    ("rl_cross_attribute_corr", "hard_composite", "cross-metric",
     ["Mean", "Std"], ["Argmax", "Compare"],
     "Is the metric with the highest mean also the one with the highest std?", ""),
    ("rl_cross_conditional_query", "hard_composite", "cross-metric",
     ["TrendClassify", "Mean", "Std"], ["Filter", "Argmax"],
     "Among metrics with an increase trend, which has the largest range?", ""),
    ("rl_cross_asymmetric_behavior", "hard_composite", "cross-metric",
     ["Mean", "Mean"], ["Compare", "Filter", "Count"],
     "How many metrics show an upward shift in mean between halves?",
     "Reframed 2026-04-17 from random-pair XOR to count-of-shifted-metrics."),

    # ===== OOD COMPOSITE (34) =====
    ("ood_max_before_min", "ood_composite", "single-metric",
     ["ExtPos", "ExtPos"], ["Compare"],
     "Is the index of the global max before the index of the global min?", ""),
    ("ood_max_in_highest_mean_quarter", "ood_composite", "single-metric",
     ["Mean", "ExtPos"], ["Argmax", "Locate", "Compare"],
     "Does the global max fall in the quarter with the highest mean?", ""),
    ("ood_quarter_mean_ordering", "ood_composite", "single-metric",
     ["Mean"], ["Rank"],
     "Do the quarter means of {metric} strictly increase (or decrease)?", ""),
    ("ood_max_mean_chunk_pos", "ood_composite", "single-metric",
     ["Mean"], ["Argmax"],
     "Find the 16-point chunk with the largest mean and report its index.", ""),
    ("ood_chunk_above_proportion", "ood_composite", "single-metric",
     ["Mean", "Mean"], ["Filter", "Count", "Arith"],
     "What fraction of 16-point chunks have a mean above the global mean?", ""),
    ("ood_std_exceeds_half_range", "ood_composite", "single-metric",
     ["Std", "ExtVal", "ExtVal"], ["Arith", "Threshold"],
     "Is the standard deviation greater than 1/3 of the range?", ""),
    ("ood_symmetric_recovery", "ood_composite", "single-metric",
     ["Mean"], ["Arith", "Threshold"],
     "Does {metric} exhibit symmetric recovery (Q1~Q4 and Q2~Q3)?", ""),
    ("ood_cycle_mean_trend", "ood_composite", "single-metric",
     ["Period", "Mean", "TrendClassify"], [],
     "Do the cycle-level means show an increasing or decreasing trend?", ""),
    ("ood_amplitude_vs_segment_std", "ood_composite", "single-metric",
     ["EventEnum", "SegEnum", "Std"], ["Argmax", "Locate", "Threshold"],
     "Does the largest event amplitude exceed the std of its containing segment?", ""),
    ("ood_event_amplitude_vs_std", "ood_composite", "single-metric",
     ["EventEnum", "Std"], ["Argmax", "Arith", "Threshold"],
     "Does the largest event amplitude exceed k-sigma?", ""),
    ("ood_range_normalized_amplitude", "ood_composite", "single-metric",
     ["EventEnum", "ExtVal", "ExtVal"], ["Argmax", "Arith", "Threshold"],
     "Does the largest event amplitude exceed half the value range?", ""),
    ("ood_event_density", "ood_composite", "single-metric",
     ["EventEnum", "SegEnum"], ["Filter", "Count", "Sum", "Arith", "Argmax"],
     "Which trend type has the lowest density of local events?", ""),
    ("ood_event_density_by_trend", "ood_composite", "single-metric",
     ["EventEnum", "SegEnum"], ["Filter", "Count", "Sum", "Arith"],
     "What is the event density in increase segments?", ""),
    ("ood_max_amp_in_longest_segment", "ood_composite", "single-metric",
     ["SegEnum", "EventEnum"], ["Argmax", "Filter"],
     "What is the largest event amplitude in the longest trend segment?", ""),
    ("ood_peak_in_longest_segment", "ood_composite", "single-metric",
     ["SegEnum", "ExtPos"], ["Argmax", "Locate"],
     "Does the global max fall inside the longest trend segment?", ""),
    ("ood_duration_weighted_mean", "ood_composite", "single-metric",
     ["SegEnum"], ["Sum", "Arith"],
     "Compute the duration-weighted mean across trend segments.", ""),
    ("ood_longest_type_fraction", "ood_composite", "single-metric",
     ["SegEnum"], ["Argmax", "Filter", "Sum", "Arith"],
     "What fraction of {metric} is covered by the trend type of the longest segment?", ""),
    ("ood_conditional_stat", "ood_composite", "single-metric",
     ["SegEnum", "Mean"], ["Filter"],
     "Compute the mean of {metric} restricted to increase segments.", ""),
    ("ood_conditional_mean_by_type", "ood_composite", "single-metric",
     ["SegEnum"], ["Filter", "Sum", "Arith"],
     "What is the average of segment means for all increase segments?", ""),
    ("ood_conditional_count", "ood_composite", "single-metric",
     ["SegEnum", "Mean"], ["Filter", "Threshold", "Count"],
     "Count increase segments whose mean exceeds the global mean.", ""),
    ("ood_segment_stat_compare", "ood_composite", "single-metric",
     ["SegEnum"], ["Locate", "Compare"],
     "Is the mean of the first segment greater than the last segment?", ""),
    ("ood_trend_follows_mean", "ood_composite", "single-metric",
     ["SegEnum", "Mean"], ["Filter", "Compare"],
     "Do all increase segments have above-average means?", ""),
    ("ood_trend_reversal", "ood_composite", "single-metric",
     ["SegEnum"], ["Argmax", "Compare"],
     "Does the dominant trend reverse after the largest change point?", ""),
    ("ood_symmetric_trend_sequence", "ood_composite", "single-metric",
     ["SegEnum"], ["Compare"],
     "Is the trend pattern of {metric} palindromic?", ""),
    ("ood_cross_corr_count", "ood_composite", "cross-metric",
     ["TrendClassify"], ["Compare", "Count"],
     "How many pairs of metrics share the same overall trend direction?", ""),
    ("ood_cross_trend_convergence", "ood_composite", "cross-metric",
     ["TrendClassify"], ["Count", "Compare"],
     "Do the metrics become more aligned in trend in the second half than the first?", ""),
    ("ood_cross_extrema_alignment", "ood_composite", "cross-metric",
     ["ExtPos"], ["Locate"],
     "Do {metricA} and {metricB} have their maxima at similar positions?", ""),
    ("ood_cross_range_overlap", "ood_composite", "cross-metric",
     ["ExtVal", "ExtVal"], ["Compare"],
     "Is there any overlap between the value ranges of two metrics?", ""),
    ("ood_cross_concordant_shift", "ood_composite", "cross-metric",
     ["Mean", "Mean"], ["Compare", "Count"],
     "How many metrics show an upward shift in mean between halves?", ""),
    ("ood_cluster_singleton", "ood_composite", "cross-metric",
     ["TrendClassify"], ["Filter", "Count"],
     "Is any metric in its own unique trend cluster?", ""),
    ("ood_corr_transitivity", "ood_composite", "cross-metric",
     ["TrendClassify"], ["Compare"],
     "If A's trend matches B's and B's matches C's, does A's match C's?", ""),
    ("ood_mixed_corr_anti", "ood_composite", "cross-metric",
     ["TrendClassify"], ["Compare", "Count"],
     "Count the anti-correlated pairs among the metrics.", ""),
    ("ood_dominant_trend_in_cluster", "ood_composite", "cross-metric",
     ["Mean", "TrendClassify"], ["Filter", "Count", "Argmax"],
     "Among metrics in the upper-half mean group, identify the dominant trend type.", ""),
    ("ood_cross_event_causality", "ood_composite", "cross-metric",
     ["EventEnum", "EventEnum"], ["Locate"],
     "Does an event in {metricA} precede an event in {metricB} within w timesteps?", ""),
]


def verify_against_data():
    """Cross-check eval_type presence and bridge flag against the realized data."""
    sft_count = Counter()
    sft_bridge = Counter()
    rl_count = Counter()
    if Path("data/train_sft.jsonl").exists():
        with open("data/train_sft.jsonl") as f:
            for line in f:
                r = json.loads(line)
                et = r.get("eval_type", "?")
                sft_count[et] += 1
                em = r.get("eval_metadata") or {}
                if em.get("bridge"):
                    sft_bridge[et] += 1
    if Path("data/train_rl_public.jsonl").exists():
        with open("data/train_rl_public.jsonl") as f:
            for line in f:
                et = json.loads(line).get("eval_type", "?")
                rl_count[et] += 1
    return sft_count, sft_bridge, rl_count


def main():
    sft, sft_bridge, rl = verify_against_data()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w") as fh:
        for (et, regime, subgroup, atoms, subskills, example, notes) in ROWS:
            entry = {
                "eval_type": et,
                "regime": regime,
                "subgroup": subgroup,
                "atoms": atoms,
                "subskills": subskills,
                "in_sft": sft.get(et, 0) > 0,
                "in_rl_public": rl.get(et, 0) > 0,
                "bridge_present": sft_bridge.get(et, 0) > 0,
                "sft_rows": sft.get(et, 0),
                "rl_public_rows": rl.get(et, 0),
                "example_question": example,
                "notes": notes,
            }
            fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
    print(f"Wrote {len(ROWS)} entries to {OUT}")
    # Sanity-check counts
    by_regime = Counter(r[1] for r in ROWS)
    print("Regime counts:", dict(by_regime))


if __name__ == "__main__":
    main()
