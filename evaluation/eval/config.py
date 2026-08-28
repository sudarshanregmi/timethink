import os
import resource
from loguru import logger
from evaluation.ragas.config import config

_eval_config = config.get('eval', {})

MTYPE = _eval_config.get('mtype', 'think1000_now')
CONCURRENT_REQUESTS = _eval_config.get('concurrent_requests', 800)
API_BASE = os.environ.get('JUDGE_API_BASE') or config.get('models', {}).get('openai_api_base', "http://localhost:8000/v1")
API_KEY = config.get('models', {}).get('openai_api_key', "EMPTY")
MODEL_NAME = config.get('models', {}).get('llm_model', "Qwen/Qwen2.5-72B-Instruct-GPTQ-Int4")

try:
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (hard, hard))
    logger.info(f"System open file limit raised to: {hard}")
except Exception as e:
    logger.warning(f"Could not raise system file limit: {e}. If you see 'Too many open files' errors, run 'ulimit -n 65000' in terminal.")

JUDGE_PROMPT_TEMPLATE = """
You are an expert time-series analyst evaluator.
Your task is to rate the quality of a generated explanation compared to a ground truth explanation.
Question: {question}
Ground Truth: {gt}
Prediction: {pred}
Critique the prediction based on physical significance and accuracy.
Finally, provide a score from 0.0 to 1.0.
Output ONLY the JSON object. Do not include any introductory text, markdown headers, or explanations outside of the JSON block:
{{
    "reasoning": "your critique here",
    "score": 0.85
}}
"""

MCQ_JUDGE_PROMPT_TEMPLATE = """
You are grading a multiple-choice answer.

Question:
{question}

Options:
{options_block}

Correct answer: {correct_letter}) {correct_option}

Model response:
{pred}

Task:
1. Identify which option (A/B/C/D) the model ultimately picked. Acceptable forms:
   - Starts with the letter (e.g. "A) ...", "Answer: B", "The answer is C").
   - Paraphrases or quotes the option text without the letter — match by content.
   - Rambles or revises — use the final committed choice (the answer it ends on).
   - Does NOT commit to any option → picked_letter = null.
2. Set verdict_match = 1.0 iff picked_letter == correct_letter, else 0.0. If picked_letter is null, verdict_match = 0.0.
3. Rate reasoning quality in score (0.0–1.0) INDEPENDENT of correctness — how clear, grounded in the time series, and well-structured the justification is.

Output ONLY the JSON object. No preface, no markdown:
{{
    "picked_letter": "A",
    "verdict_match": 1.0,
    "reasoning": "brief justification for the grading",
    "score": 0.85
}}
"""

SEGMENT_JUDGE_PROMPT_TEMPLATE = """
You are an expert time-series analyst evaluator.
Your task is to evaluate a model's response to a structured time-series question.

Question: {question}
Expected Answer: {expected_verdict}
Ground Truth Response: {gt}
Model Response: {pred}

Evaluate FOUR things:
1. **verdict_match**: Does the model's response convey the same answer/verdict as the expected answer? Use 1.0 for correct, 0.0 for incorrect. Be lenient with formatting — "yes", "Yes.", and "Yes, because..." all match an expected verdict of "yes". For numeric verdicts, allow small rounding differences (e.g. "5.23" matches "5.24").
2. **correctness_score**: A graded measure of ANSWER CORRECTNESS only — ignore explanation quality, depth, or verbosity. Range 0.0–1.0. Use this rubric:
   - 1.0 = the answer fully matches the expected answer (any acceptable phrasing)
   - 0.7–0.9 = mostly correct with a small/specific error (e.g., one item out of place in an ordered list of 6, a numeric value off by a small relative amount, a partially-overlapping set)
   - 0.4–0.6 = roughly half right (e.g., half of a list correct, an estimate at the right order of magnitude but off, a categorical with a related-but-wrong label)
   - 0.1–0.3 = mostly wrong but some element is recognizable
   - 0.0 = wrong, or no answer
   For ordered lists, use Kendall's tau intuition (more swaps = lower score). For sets, use Jaccard intuition (|intersection|/|union|).
3. **score**: Rate the overall quality of the explanation (reasoning, accuracy, completeness) from 0.0 to 1.0. This is for diagnostic purposes; it can differ from correctness_score (a terse-but-correct answer can have low score and high correctness_score).
4. **extracted_number**: If the Expected Answer is numeric (a single integer or float), identify the final numeric verdict the model is asserting — the number it is CLAIMING as its answer, not an intermediate calculation. For prose like "covering 132 timesteps, which accounts for 0.52 of the total length", the verdict is 0.52, not 132. If the response expresses the answer as a ratio in words (e.g. "shared by 3 of 7 metrics" or "4 out of 5"), compute the ratio to 2 decimal places (3/7 ≈ 0.43, 4/5 = 0.80) and use that. Output the number as a JSON number (e.g. 0.52). If the model response contains no clear numeric verdict or the Expected Answer is not numeric, set this field to null.

Output ONLY the JSON object:
{{
    "reasoning": "your critique here",
    "verdict_match": 1.0,
    "correctness_score": 0.85,
    "score": 0.85,
    "extracted_number": 0.52
}}
"""

# Standardized judge questions for MTS relationship types.
# These types need a generic judge question because their raw questions contain
# specific metric names, point references, and tolerance values that would bias
# the judge toward surface-level keyword matching rather than reasoning quality.
RAGAS_QUESTION_BY_TASK = {
    # ── Core MTS families ──────────────────────────────────────────────────────
    "correlation": "Explain why these two time series are or are not correlated in terms of their fluctuations.",
    "clustering": "Based on the time series behavior (e.g., trend patterns, local fluctuations), identify which time series (if any) are related to the target metric and explain why.",
    "yes_no": "Is there a local event at the specified point? Please answer Yes or No and explain it in detail.",
    "anticorrelation": "Explain whether these two time series show opposite trend patterns — where one increases the other decreases and vice versa, while steady segments are shared.",
    "anticlustering": "Based on their trend patterns, identify which time series (if any) show opposite behavior to the target metric and explain why.",
    # NOTE: "description" is intentionally NOT here. resolve_judge_question() falls back
    # to the actual question text, which specifies the exact perspectives asked about
    # (e.g., trend + noise only). A generic 4-perspective question would mislead the judge.

    # ── Segment family (Families 3–8) ─────────────────────────────────────────
    "segment_judgment": "Based on the given multi-condition definition and threshold, does the time series satisfy the stated condition? Answer yes or no with evidence.",
    "segment_trend_dominance": "In the specified time range, which trend direction (increasing, decreasing, or steady) is dominant? Justify with the proportion each covers.",

    # ── MTS shape compound families ────────────────────────────────────────────
    "anti_judgment": "Based on the given definition combining anticorrelation and noise strength conditions, does the described pattern exist? Answer yes or no with reasoning.",
    "cross_trend_query": "When the first metric shows the specified trend type, what trend does the second metric predominantly exhibit? State the dominant direction and supporting evidence.",

    # ── Statistical numerical (Family 9) ──────────────────────────────────────
    "stat_numerical": "Based on the computed global statistics, provide the exact numerical answer to the given statistical query (e.g. minimum, mean, range, crossing count, windowed mean).",

    # ── Periodicity ──────────────────────────────────────────────────────────
    "periodicity": "Answer the periodicity question about the time series (e.g. estimated period, cycle count, regularity) with the exact value.",

    # ── Change point detection ──────────────────────────────────────────────
    "change_point": "Answer the change point detection question about the time series (e.g. number of behavioral changes, positions, largest shift) with the exact value.",

    # ── Enumeration families ─────────────────────────────────────────────────
    "local_enumeration": "Answer the enumeration question about local events (e.g. count, types, positions, amplitudes) with the exact value from the time series.",
    "segment_enumeration": "Answer the enumeration question about trend segments (e.g. count, longest, shortest, types) with the exact value from the time series.",
    "cross_metric_enumeration": "Answer the cross-metric enumeration question (e.g. which metrics share a property, highest amplitude across metrics) with the exact value.",
    "transition_enumeration": "Answer the enumeration question about trend transitions (e.g. count, types, most common transition) with the exact value from the time series.",
    "event_segment_enumeration": "Answer the enumeration question about the relationship between local events and trend segments (e.g. events within a trend type, segment with most events) with the exact value.",
    "temporal_position": "Answer the temporal positioning question about local events (e.g. first/last event, largest/smallest gap between events) with the exact value.",
    "duration_proportion": "Answer the duration question about trend segments (e.g. total duration of a trend type in timesteps, which type has the longest/shortest total duration, comparison of durations between two types, or their absolute difference) with the exact value.",

    # ── Cross-metric statistical judgment ──────────────────────────────────
    "cross_stat_judgment": "Based on the given cross-metric statistical condition (e.g. volatility ratio, range dominance, synchronized shift), does the data satisfy the stated definition? Answer yes or no with evidence.",

    # NOTE: "tsevol" is intentionally NOT here. resolve_judge_question() falls back
    # to the actual question text, which the judge needs to assess answer correctness.

    # ── OOD compositional evaluation types (eval-only, never in training) ─────
    "ood_conditional_stat": "Compute the requested statistic (mean, std, or range) of the time series restricted to segments of a specific trend type. Provide the exact numerical value.",
    "ood_nested_extrema": "Within the specified trend segment (longest or shortest), identify the local event with the extreme amplitude. Provide the exact amplitude value.",
    "ood_event_density": "Determine which trend type has the highest or lowest density of local events (events per unit time). State the trend type.",
    "ood_conditional_count": "Count how many segments of the specified trend type satisfy the statistical condition (mean above or below the overall series mean). Provide the exact count.",
    "ood_trend_reversal": "Determine whether the dominant trend direction reverses after the largest change point. Answer yes or no with evidence.",
    "ood_range_normalized_amplitude": "Determine whether the largest local event amplitude exceeds half the total value range. Answer yes or no with evidence.",
    "ood_segment_stat_compare": "Compare the mean values of the first and last trend segments. Answer yes or no to whether the first exceeds the last.",
    # --- New taxonomy types (single-metric) ---
    "atomic_global_mean": "Compute the mean of the full time series using chunked averaging.",
    "atomic_chunked_means": "Compute chunk-level means across the time series.",
    "atomic_interval_mean": "Compute the mean of a specified sub-range.",
    "atomic_global_std": "Compute the standard deviation of the full time series.",
    "atomic_interval_std": "Compute the standard deviation of a specified sub-range.",
    "atomic_min_value": "Find the minimum value in the time series.",
    "atomic_max_value": "Find the maximum value in the time series.",
    "atomic_min_position": "Find the position of the minimum value.",
    "atomic_max_position": "Find the position of the maximum value.",
    "atomic_percentile": "Compute the specified percentile of the time series.",
    "atomic_trend_enumeration": "Enumerate all trend segments in the time series.",
    "atomic_event_enumeration": "Enumerate all local events in the time series.",
    "atomic_periodic_description": "Describe the periodic behavior of the time series.",
    # --- Cross-metric taxonomy (Category H) ---
    "atomic_cross_stat_compare": "Compare a specific statistic between two metrics. Answer yes or no.",
    "atomic_cross_ranking": "Identify which metric has the highest or lowest value of a specific statistic.",
    "atomic_cross_filtering": "List the metrics that satisfy a given property.",
    "atomic_cross_counting": "Count how many metrics satisfy a given property.",
    "atomic_cross_trend_align": "Determine whether two metrics share the same overall trend direction. Answer yes or no.",
    "rl_cross_stat_ratio": "Determine whether the ratio of a statistic between two metrics exceeds a threshold. Answer yes or no.",
    "rl_cross_full_ordering": "Rank all metrics by a specified statistic from highest to lowest.",
    "rl_cross_event_sync": "Determine whether two metrics have local events occurring within a specified proximity. Answer yes or no.",
    "rl_cross_period_compare": "Compare the periodic components of two metrics. Answer yes or no.",
    "rl_cross_trend_concordance": "Compute the fraction of metrics in the dominant trend direction (mode count divided by total metric count).",
    "rl_cross_conditional_query": "Among metrics satisfying a condition, identify which has the extreme value of a statistic.",
    "rl_cross_attribute_corr": "Determine whether the metric ranked highest on one attribute also ranks highest on another. Answer yes or no.",
    "rl_cross_asymmetric_behavior": "Count how many metrics shifted their mean upward (or downward) between the first and second halves. Provide the exact integer count.",
    "ood_cross_corr_count": "Count the number of metric pairs that share the same overall trend direction.",
    "ood_cross_trend_convergence": "Determine whether trend agreement among metrics is stronger/weaker in the second half than the first. Answer yes or no with evidence.",
    "ood_cross_extrema_alignment": "Determine whether any two metrics have their maximum at similar positions. Answer yes or no.",
    "ood_cross_range_overlap": "Determine whether the value ranges of two metrics overlap. Answer yes or no.",
    "ood_cross_concordant_shift": "Count how many metrics show a higher mean in the second half than the first.",
    # --- Cross-metric RL cluster/correlation types ---
    "rl_cluster_count": "Count how many distinct behavior clusters (groups of metrics sharing the same trend) exist.",
    "rl_cluster_dominant": "Identify which trend group (cluster) contains the most metrics.",
    "rl_corr_count": "Identify which trend direction (increase, decrease, or keep steady) is most common among the metrics.",
    "rl_corr_conditional": "Determine whether two metrics with the same trend also have nearby local events. Answer yes or no.",
    # --- Cross-metric OOD cluster/correlation types ---
    "ood_cluster_singleton": "Determine whether any metric has a unique trend not shared by any other metric. Answer yes or no.",
    "ood_corr_transitivity": "Determine whether trend-sharing is transitive among three specified metrics. Answer yes or no.",
    "ood_mixed_corr_anti": "Count how many metric pairs show opposite trend directions.",
    # --- OOD single-metric taxonomy types (eval-only) ---
    "ood_max_before_min": "Determine whether the global maximum occurs before the global minimum. Answer yes or no.",
    "ood_std_exceeds_half_range": "Determine whether the standard deviation exceeds half the value range. Answer yes or no.",
    "ood_max_in_highest_mean_quarter": "Determine whether the global maximum falls in the quarter with the highest mean. Answer yes or no.",
    "ood_quarter_mean_ordering": "Determine whether the quarter means follow a monotonic ordering. Answer yes or no.",
    "ood_max_mean_chunk_pos": "Identify the position (chunk index) of the chunk with the highest mean.",
    "ood_chunk_above_proportion": "Compute the proportion of chunks whose mean exceeds the overall mean.",
    "ood_symmetric_recovery": "Determine whether the series shows symmetric recovery around its extremum. Answer yes or no.",
    "ood_cycle_mean_trend": "Determine whether the cycle-level means show an increasing or decreasing trend. Answer yes or no.",
    "ood_trend_follows_mean": "Determine whether the dominant trend direction aligns with the overall mean shift. Answer yes or no.",
    "ood_conditional_mean_by_type": "Compute the mean of the series restricted to segments of a specific trend type.",
    "ood_longest_type_fraction": "Compute the fraction of total duration covered by the longest trend segment.",
    "ood_event_amplitude_vs_std": "Determine whether the largest event amplitude exceeds a multiple of the standard deviation. Answer yes or no.",
    "ood_amplitude_vs_segment_std": "Determine whether the event amplitude exceeds the local segment standard deviation. Answer yes or no.",
    "ood_symmetric_trend_sequence": "Determine whether the trend type sequence is palindromic/symmetric. Answer yes or no.",
    "ood_event_density_by_trend": "Compute the event density within segments of a specific trend type.",
    "ood_max_amp_in_longest_segment": "Find the maximum event amplitude within the longest trend segment.",
    # --- OOD compositional probes (composition of atomic + rl primitives) ---
    "ood_peak_in_longest_segment": "Determine whether the global maximum lies within the longest trend segment. Answer yes or no.",
    "ood_duration_weighted_mean": "Compute the duration-weighted mean: sum of each trend segment's mean weighted by its duration, divided by total duration.",
    "ood_dominant_trend_in_cluster": "Among metrics whose mean falls in the upper half across metrics, identify the dominant trend type.",
    "ood_cross_event_causality": "Determine whether at least one local event in one metric precedes a local event in another metric within a scaled window. Answer yes or no.",
    # --- RL single-metric taxonomy types ---
    "rl_amplitude_vs_range": "Determine whether the largest event amplitude exceeds a fraction of the value range. Answer yes or no.",
    "rl_amplitude_vs_std": "Determine whether the largest event amplitude exceeds a multiple of the standard deviation. Answer yes or no.",
    "rl_condition_recovery": "Determine whether the series recovers to its pre-event level after the largest event. Answer yes or no.",
    "rl_cycle_count": "Count the number of complete cycles in the periodic component.",
    "rl_dominant_trend_type": "Identify which trend type covers the largest fraction of the series.",
    "rl_event_count": "Count the total number of local events in the series.",
    "rl_event_count_by_type": "Count the number of local events of a specific type.",
    "rl_event_in_trend_type": "Determine whether a local event occurs within a segment of a specific trend type. Answer yes or no.",
    "rl_event_near_extremum": "Determine whether a local event occurs near the global maximum or minimum. Answer yes or no.",
    "rl_event_type_at_pos": "Identify the type of local event at a specific position.",
    "rl_extrema_same_half": "Determine whether the global max and min are in the same half of the series. Answer yes or no.",
    "rl_half_mean_compare": "Determine whether the first-half mean exceeds the second-half mean. Answer yes or no.",
    "rl_half_mean_diff": "Compute the absolute difference between first-half and second-half means.",
    "rl_has_periodicity": "Determine whether the series has a periodic component. Answer yes or no.",
    "rl_interval_comparison": "Determine whether the mean of one interval exceeds another. Answer yes or no.",
    "rl_longest_segment": "Compute the duration of the longest trend segment.",
    "rl_max_amplitude_event": "Identify the type of the local event with the largest amplitude.",
    "rl_max_in_first_half": "Determine whether the global maximum is in the first half. Answer yes or no.",
    "rl_max_in_trend_type": "Determine whether the global maximum falls within a segment of a specific trend type. Answer yes or no.",
    "rl_mean_shift": "Determine whether the mean shifts significantly between halves. Answer yes or no.",
    "rl_mean_stability": "Determine whether the chunked means are stable (low coefficient of variation). Answer yes or no.",
    "rl_median_mean_close": "Determine whether the median and mean are within a threshold of each other. Answer yes or no.",
    "rl_monotonic_chunks": "Determine whether the chunk means are monotonically increasing or decreasing. Answer yes or no.",
    "rl_normalized_range": "Compute the range normalized by the standard deviation.",
    "rl_period_estimate": "Estimate the period of the periodic component.",
    "rl_range": "Compute the value range (max minus min) of the series.",
    "rl_segment_count": "Count the number of trend segments.",
    "rl_segment_duration": "Compute the duration of a specific trend segment.",
    "rl_segment_mean_compare": "Determine whether the mean of one segment exceeds another. Answer yes or no.",
    "rl_segment_type_at_pos": "Identify the trend type at a specific position.",
    "rl_type_duration_fraction": "Compute the fraction of total duration covered by a specific trend type.",
    "rl_type_of_longest": "Identify the trend type of the longest segment.",
    "rl_volatility_change": "Determine whether volatility changes significantly between halves. Answer yes or no.",
}

# Canonical set of all eval_types the pipeline handles.
# `description` and `tsevol` are intentionally absent from RAGAS_QUESTION_BY_TASK
# (they use the sample's actual question text as the judge question).
ALL_EVAL_TYPES = (
    frozenset(RAGAS_QUESTION_BY_TASK.keys())
    | frozenset({"description", "tsevol"})
)

SEGMENT_FAMILY_TYPES = frozenset({
    'segment_judgment',
    'segment_trend_dominance', 'anti_judgment', 'cross_trend_query',
    'stat_numerical', 'periodicity', 'change_point',
    'local_enumeration', 'segment_enumeration', 'cross_metric_enumeration',
    'transition_enumeration', 'event_segment_enumeration', 'temporal_position', 'duration_proportion',
    'cross_stat_judgment',
    # OOD compositional types (eval-only)
    'ood_conditional_stat', 'ood_nested_extrema', 'ood_event_density',
    'ood_conditional_count', 'ood_trend_reversal',
    'ood_range_normalized_amplitude', 'ood_segment_stat_compare',
    # New taxonomy types (single-metric)
    'atomic_global_mean', 'atomic_chunked_means', 'atomic_interval_mean',
    'atomic_global_std', 'atomic_interval_std',
    'atomic_min_value', 'atomic_max_value', 'atomic_min_position', 'atomic_max_position',
    'atomic_percentile', 'atomic_trend_enumeration', 'atomic_event_enumeration',
    'atomic_periodic_description',
    # Cross-metric taxonomy (Category H)
    'atomic_cross_stat_compare', 'atomic_cross_ranking', 'atomic_cross_filtering',
    'atomic_cross_counting', 'atomic_cross_trend_align',
    'rl_cross_stat_ratio', 'rl_cross_full_ordering', 'rl_cross_event_sync',
    'rl_cross_period_compare', 'rl_cross_trend_concordance',
    'rl_cross_conditional_query', 'rl_cross_attribute_corr', 'rl_cross_asymmetric_behavior',
    'rl_cluster_count', 'rl_cluster_dominant', 'rl_corr_count', 'rl_corr_conditional',
    'ood_cross_corr_count', 'ood_cross_trend_convergence', 'ood_cross_extrema_alignment',
    'ood_cross_range_overlap', 'ood_cross_concordant_shift',
    'ood_cluster_singleton', 'ood_corr_transitivity', 'ood_mixed_corr_anti',
    # OOD single-metric taxonomy
    'ood_max_before_min', 'ood_std_exceeds_half_range',
    'ood_max_in_highest_mean_quarter', 'ood_quarter_mean_ordering',
    'ood_max_mean_chunk_pos', 'ood_chunk_above_proportion', 'ood_symmetric_recovery',
    'ood_cycle_mean_trend', 'ood_trend_follows_mean', 'ood_conditional_mean_by_type',
    'ood_longest_type_fraction', 'ood_event_amplitude_vs_std', 'ood_amplitude_vs_segment_std',
    'ood_symmetric_trend_sequence', 'ood_event_density_by_trend', 'ood_max_amp_in_longest_segment',
    # OOD compositional probes
    'ood_peak_in_longest_segment', 'ood_duration_weighted_mean',
    'ood_dominant_trend_in_cluster', 'ood_cross_event_causality',
    # RL single-metric taxonomy
    'rl_amplitude_vs_range', 'rl_amplitude_vs_std', 'rl_condition_recovery',
    'rl_cycle_count', 'rl_dominant_trend_type', 'rl_event_count', 'rl_event_count_by_type',
    'rl_event_in_trend_type', 'rl_event_near_extremum', 'rl_event_type_at_pos',
    'rl_extrema_same_half', 'rl_half_mean_compare', 'rl_half_mean_diff',
    'rl_has_periodicity', 'rl_interval_comparison', 'rl_longest_segment',
    'rl_max_amplitude_event', 'rl_max_in_first_half', 'rl_max_in_trend_type',
    'rl_mean_shift', 'rl_mean_stability', 'rl_median_mean_close', 'rl_monotonic_chunks',
    'rl_normalized_range', 'rl_period_estimate', 'rl_range',
    'rl_segment_count', 'rl_segment_duration', 'rl_segment_mean_compare',
    'rl_segment_type_at_pos', 'rl_type_duration_fraction', 'rl_type_of_longest',
    'rl_volatility_change',
})


def resolve_task_type(eval_type, question_text=""):
    if eval_type is not None:
        return eval_type
    # Fallback: infer task type from question text (for old data without eval_type)
    q_lower = question_text.lower()
    if "anticorrelat" in q_lower:
        return "anticorrelation"
    if "opposite" in q_lower and "trend" in q_lower:
        return "anticlustering"
    if "correlation" in q_lower:
        return "correlation"
    if "find" in q_lower and "related" in q_lower:
        return "clustering"
    if "is there" in q_lower and "fluctuation" in q_lower:
        return "yes_no"
    return "unknown"


def resolve_judge_question(eval_type, question_text=""):
    task = resolve_task_type(eval_type, question_text)
    return RAGAS_QUESTION_BY_TASK.get(task, question_text)