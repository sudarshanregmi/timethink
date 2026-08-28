"""Compositional QA generators for RL training (mean category).

These compose 2+ SFT atomic skills in ways never seen during SFT.
The model generates its own reasoning during RL rollout; reward checks the answer.

Format:
    <think>
    answer: REWARD_ANSWER
    </think>
    Rich natural language explanation with computed values.

Ground truth is computed using the same chunk-and-average method as SFT,
so the model's learned computation approach produces reward-matching answers.
"""

import random
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from loguru import logger

from synth.align.generators.atomic_qa import _compute_chunks, _fmt, CHUNK_SIZE

# ============================================================
# Question templates — per task, ~6-8 each
# ============================================================

CONDITION_RECOVERY_QUESTIONS = [
    "Is the condition recovered for {metric}? A condition is recovered if the absolute difference between the mean of the first quarter and the mean of the last quarter is less than {threshold}.",
    "Check whether {metric} shows recovery: the absolute difference between the first-quarter mean and last-quarter mean must be below {threshold}.",
    "For {metric}, determine if recovery occurred. Recovery means |mean(first quarter) - mean(last quarter)| < {threshold}.",
    "Does {metric} exhibit condition recovery? Recovery is defined as the absolute difference between the first quarter's mean and the last quarter's mean being under {threshold}.",
    "Assess whether the condition is recovered for {metric}. The condition is met when the absolute difference between the mean of the first 25% and last 25% of the data is less than {threshold}.",
    "For the {metric} timeseries, is the condition recovered? That is, is |mean(first quarter) - mean(last quarter)| below {threshold}?",
]

MEAN_SHIFT_QUESTIONS = [
    "Is there a significant mean shift in {metric}? A mean shift is detected if the absolute difference between the mean of the first third and the mean of the last third exceeds {threshold}.",
    "Check whether {metric} exhibits a mean shift: |mean(first third) - mean(last third)| > {threshold}.",
    "For {metric}, determine if there is a mean shift. A shift is present when the absolute difference between the first-third mean and the last-third mean is greater than {threshold}.",
    "Does {metric} show a mean shift? A shift means |mean(first 1/3) - mean(last 1/3)| exceeds {threshold}.",
    "Assess whether a mean shift occurred in {metric}. The shift is detected when the absolute difference between the mean of the first third and last third of the data exceeds {threshold}.",
    "For the {metric} timeseries, is there a mean shift greater than {threshold} between the first and last thirds?",
]

INTERVAL_COMPARISON_QUESTIONS = [
    "For {metric}, is the mean of the range [{a_start}, {a_end}] greater than the mean of the range [{b_start}, {b_end}]?",
    "Compare two intervals of {metric}: is the mean from index {a_start} to {a_end} higher than the mean from index {b_start} to {b_end}?",
    "In the {metric} timeseries, does the interval [{a_start}, {a_end}] have a greater mean than the interval [{b_start}, {b_end}]?",
    "For {metric}, which interval has the higher mean: [{a_start}, {a_end}] or [{b_start}, {b_end}]? Specifically, is the first interval's mean greater?",
    "Is the average of {metric} from index {a_start} to {a_end} larger than the average from index {b_start} to {b_end}?",
    "Check whether the mean of {metric} over [{a_start}, {a_end}] exceeds the mean over [{b_start}, {b_end}].",
]

HALF_MEAN_DIFF_QUESTIONS = [
    "What is the absolute difference between the first-half mean and second-half mean of {metric}?",
    "Compute |mean(first half) - mean(second half)| for {metric}.",
    "For {metric}, find the absolute difference between the mean of the first half and the mean of the second half.",
    "Calculate the absolute difference in means between the two halves of the {metric} timeseries.",
    "How large is the difference between the first-half average and second-half average of {metric}?",
    "For the {metric} data, what is the magnitude of the difference between the first-half mean and the second-half mean?",
]

MEAN_STABILITY_QUESTIONS = [
    "Is {metric} stable? Stability means every chunk mean (chunk size 16) falls within {threshold} of the global mean.",
    "Check whether {metric} is stable: all 16-point chunk means must be within {threshold} of the overall mean.",
    "For {metric}, determine if the series is stable. Stable means no chunk mean (at granularity 16) deviates from the global mean by more than {threshold}.",
    "Does {metric} exhibit stability? That is, does every 16-point chunk's mean lie within {threshold} of the global mean?",
    "Assess the stability of {metric}: are all chunk means (chunk size 16) within {threshold} of the series mean?",
    "For the {metric} timeseries, is it stable? Stability requires that every 16-point chunk mean stays within {threshold} of the global mean.",
]

MONOTONIC_CHUNKS_QUESTIONS = [
    "Do the chunk means of {metric} (chunk size 16) monotonically {direction} from index {start} to {end}?",
    "For {metric} between index {start} and {end}, are the 16-point chunk means monotonically {direction}?",
    "Check whether the chunk means of {metric} from index {start} to {end} (chunk size 16) are strictly {direction}.",
    "Is there a monotonic {direction} trend in the chunk means of {metric} from index {start} to {end} (using 16-point chunks)?",
    "For {metric} from index {start} to {end}, do the means of consecutive 16-point chunks strictly {direction}?",
    "Assess whether the 16-point chunk means of {metric} between index {start} and {end} form a monotonically {direction} sequence.",
]

# ============================================================
# Answer templates (after </think>) — per task × verdict
# ============================================================

CONDITION_RECOVERY_ANSWERS = {
    'yes': [
        "Yes, the condition is recovered for {metric}.",
        "Recovery is confirmed for {metric}.",
        "The condition is recovered for {metric}.",
    ],
    'no': [
        "No, the condition is not recovered for {metric}.",
        "Recovery has not occurred for {metric}.",
        "The condition is not recovered for {metric}.",
    ],
}

MEAN_SHIFT_ANSWERS = {
    'yes': [
        "Yes, a mean shift is detected in {metric}.",
        "{metric} shows a significant mean shift.",
        "There is a mean shift in {metric}.",
    ],
    'no': [
        "No significant mean shift is detected in {metric}.",
        "No, {metric} does not show a significant mean shift.",
        "No mean shift is detected in {metric}.",
    ],
}

INTERVAL_COMPARISON_ANSWERS = {
    'yes': [
        "Yes, the mean of {metric} over the first interval is greater than over the second.",
        "The first interval's mean exceeds the second for {metric}.",
        "Yes, the first interval has a higher mean than the second for {metric}.",
    ],
    'no': [
        "No, the mean of {metric} over the first interval is not greater than over the second.",
        "The first interval's mean does not exceed the second for {metric}.",
        "No, the first interval does not have a higher mean for {metric}.",
    ],
}

HALF_MEAN_DIFF_ANSWERS = [
    "The absolute difference in half means of {metric} is {diff}.",
    "For {metric}, the half-mean difference is {diff}.",
    "The half-mean absolute difference of {metric} is {diff}.",
]

MEAN_STABILITY_ANSWERS = {
    'yes': [
        "Yes, {metric} is stable.",
        "The {metric} timeseries is stable.",
        "Stability confirmed for {metric}.",
    ],
    'no': [
        "No, {metric} is not stable.",
        "The {metric} timeseries is not stable.",
        "{metric} does not pass the stability check.",
    ],
}

MONOTONIC_CHUNKS_ANSWERS = {
    'yes': [
        "Yes, the chunk means of {metric} are monotonically {direction} in that range.",
        "The chunk means of {metric} monotonically {direction} from index {start} to {end}.",
        "Confirmed, the chunk means are monotonically {direction}.",
    ],
    'no': [
        "No, the chunk means of {metric} are not monotonically {direction} in that range.",
        "The chunk means of {metric} do not monotonically {direction} from index {start} to {end}.",
        "The chunk means of {metric} are not monotonically {direction} in that range.",
    ],
}


# ============================================================
# Helpers
# ============================================================

def _interval_mean(ts: np.ndarray, start: int, end: int) -> float:
    """Compute interval mean using chunk-and-average (same as SFT method)."""
    chunks = _compute_chunks(ts, start, end)
    chunk_means = [cm for _, _, cm, _ in chunks]
    return round(float(np.mean(chunk_means)), 2)


def _pick_threshold(actual: float, comparison: str = '<') -> Tuple[float, bool]:
    """Pick a threshold for balanced yes/no verdicts.

    For comparison='<': question is "is actual < threshold?"
    For comparison='>': question is "is actual > threshold?"

    Returns (threshold, verdict_is_yes).
    ~50/50 balanced by construction (coin flip).
    """
    want_yes = random.random() < 0.5
    margin = max(0.5, abs(actual) * 0.3)

    if comparison == '<':
        if want_yes:   # want actual < T → T above actual
            threshold = actual + random.uniform(0.2, 1.0) * margin
        else:          # want actual >= T → T at or below actual
            threshold = max(0.01, actual - random.uniform(0.2, 1.0) * margin)
    else:  # '>'
        if want_yes:   # want actual > T → T below actual
            threshold = max(0.01, actual - random.uniform(0.2, 1.0) * margin)
        else:          # want actual <= T → T above actual
            threshold = actual + random.uniform(0.2, 1.0) * margin

    return round(threshold, 2), want_yes


def _pick_balanced_param(compute_verdict, candidate_params):
    """Pick a candidate param whose verdict matches a fair-coin choice.

    Guarantees 50/50 verdict distribution by construction. Use when the
    generator has a fixed set of discrete candidate parameters (thresholds,
    multipliers, etc.) and you want the question to target a desired
    verdict rather than accept whatever verdict the data naturally produces.

    Args:
        compute_verdict: callable(param) → verdict (any hashable value)
        candidate_params: iterable of params to consider

    Returns (param, verdict) on success, or None if the sample is degenerate
    (every candidate produces the same verdict — cannot be balanced, skip).
    """
    param_verdicts = [(p, compute_verdict(p)) for p in candidate_params]
    unique = sorted({v for _, v in param_verdicts}, key=str)
    if len(unique) < 2:
        return None
    desired = random.choice(unique)
    candidates = [p for p, v in param_verdicts if v == desired]
    return random.choice(candidates), desired


from collections import Counter as _Counter
from collections import defaultdict as _defaultdict

# Per-(eval_type, verdict) running counters for stateful rebalancing.
# Uses process-local state — works naturally under multiprocess generation
# because each worker balances its own stream independently.
_VERDICT_COUNTS: "dict[str, _Counter]" = _defaultdict(_Counter)


def _coin_flip_keep(actual_verdict, allowed_verdicts, eval_type: str = "") -> bool:
    """Stateful inverse-frequency balancer — forces emitted verdict distribution
    toward uniform across `allowed_verdicts`.

    We track INCOMING counts per eval_type (every candidate sample, regardless
    of whether it was kept). The keep probability for a given verdict is
    `min_incoming / this_incoming`: the rarest incoming verdict is always
    kept, the most frequent is subsampled proportionally. Because the ratio
    is computed against actual stream frequencies rather than kept counts,
    the rebalancing disengages at the right steady state.

    Warm-up: until every allowed verdict has been seen at least once, every
    sample is kept (we can't estimate ratios without data from all classes).

    Args:
        actual_verdict: the verdict the candidate sample would produce
        allowed_verdicts: iterable of verdicts to balance over
        eval_type: key for per-eval-type counter (required when one generator
            serves multiple eval types)

    Returns True if the sample should be kept, False otherwise.
    """
    allowed = list(allowed_verdicts)
    counts = _VERDICT_COUNTS[eval_type]
    counts[actual_verdict] += 1  # always record incoming
    seen = [counts.get(v, 0) for v in allowed]
    this_incoming = counts[actual_verdict]
    if min(seen) < 1:
        # Warm-up: not all classes seen yet. Use harmonic-decay keep
        # probability (1/k for kth incoming of this class) so we don't
        # over-emit a single dominant class before a rare class appears.
        keep_prob = 1.0 / this_incoming
    else:
        min_incoming = min(seen)
        keep_prob = min_incoming / this_incoming
    return random.random() < keep_prob


def _make_rl_result(
    question: str,
    verdict,
    eval_type: str,
    answer_text: str,
    seq_len: int,
    **extra_meta,
) -> Dict[str, Any]:
    """Build RL output dict with minimal think block + rich answer."""
    reward_answer = str(verdict)
    think_block = f"<think>\nanswer: {reward_answer}\n</think>"
    meta = {'length': seq_len, 'verdict': reward_answer}
    meta.update(extra_meta)
    return {
        'question': question,
        'answer': f"{think_block}\n{answer_text}",
        'eval_type': eval_type,
        'eval_metadata': meta,
    }


# ============================================================
# Generator
# ============================================================

class CompositionalMeanGenerator:
    """Generates compositional mean QA pairs for RL training.

    Each task composes 2+ SFT atomic skills (A1-A3).
    Output has minimal think block; model generates own reasoning during RL.

    Tasks:
        R1: condition_recovery  — |mean(Q1) - mean(Q4)| < T
        R2: mean_shift          — |mean(first third) - mean(last third)| > T
        R3: interval_comparison — mean([a,b]) > mean([c,d])?
        R4: half_mean_diff      — |mean(first half) - mean(second half)|
        R5: mean_stability      — all chunk means within T of global?
        R6: monotonic_chunks    — chunk means monotonically increase/decrease?
    """

    MIN_SEQ_LEN = 64  # need room for quarters/thirds

    def __init__(self, timeseries: np.ndarray, metric: str, seq_len: int):
        self.ts = np.asarray(timeseries, dtype=float)
        self.metric = metric
        self.seq_len = seq_len

    # ----------------------------------------------------------
    # R1: Condition recovery
    # ----------------------------------------------------------

    def generate_condition_recovery(self) -> Optional[Dict[str, Any]]:
        """R1: |mean(Q1) - mean(Q4)| < T → yes/no.
        Atoms: A3 (interval mean) × 2 + subtraction + abs + threshold.
        """
        if self.seq_len < self.MIN_SEQ_LEN:
            return None

        q_len = self.seq_len // 4
        q1s, q1e = 0, q_len - 1
        q4s, q4e = self.seq_len - q_len, self.seq_len - 1

        q1_mean = _interval_mean(self.ts, q1s, q1e)
        q4_mean = _interval_mean(self.ts, q4s, q4e)
        diff = round(abs(q1_mean - q4_mean), 2)

        threshold, verdict_yes = _pick_threshold(diff, '<')
        verdict = "yes" if verdict_yes else "no"

        question = random.choice(CONDITION_RECOVERY_QUESTIONS).format(
            metric=self.metric, threshold=_fmt(threshold),
        )
        answer_text = random.choice(CONDITION_RECOVERY_ANSWERS[verdict]).format(
            metric=self.metric,
            q1s=q1s, q1e=q1e, q1_mean=_fmt(q1_mean),
            q4s=q4s, q4e=q4e, q4_mean=_fmt(q4_mean),
            diff=_fmt(diff), threshold=_fmt(threshold),
        )

        return _make_rl_result(
            question, verdict, 'rl_condition_recovery', answer_text, self.seq_len,
        )

    # ----------------------------------------------------------
    # R2: Mean shift
    # ----------------------------------------------------------

    def generate_mean_shift(self) -> Optional[Dict[str, Any]]:
        """R2: |mean(first third) - mean(last third)| > T → yes/no.
        Atoms: A3 × 2 + subtraction + abs + threshold.
        """
        if self.seq_len < 48:
            return None

        third = self.seq_len // 3
        t1s, t1e = 0, third - 1
        t3s, t3e = self.seq_len - third, self.seq_len - 1

        t1_mean = _interval_mean(self.ts, t1s, t1e)
        t3_mean = _interval_mean(self.ts, t3s, t3e)
        diff = round(abs(t1_mean - t3_mean), 2)

        threshold, verdict_yes = _pick_threshold(diff, '>')
        verdict = "yes" if verdict_yes else "no"

        question = random.choice(MEAN_SHIFT_QUESTIONS).format(
            metric=self.metric, threshold=_fmt(threshold),
        )
        answer_text = random.choice(MEAN_SHIFT_ANSWERS[verdict]).format(
            metric=self.metric,
            t1_mean=_fmt(t1_mean), t3_mean=_fmt(t3_mean),
            diff=_fmt(diff), threshold=_fmt(threshold),
        )

        return _make_rl_result(
            question, verdict, 'rl_mean_shift', answer_text, self.seq_len,
        )

    # ----------------------------------------------------------
    # R3: Interval comparison
    # ----------------------------------------------------------

    def generate_interval_comparison(self) -> Optional[Dict[str, Any]]:
        """R3: mean([a,b]) > mean([c,d])? → yes/no.
        Atoms: A3 × 2 + comparison.
        """
        if self.seq_len < self.MIN_SEQ_LEN:
            return None

        # Pick two non-overlapping ranges (each 32-128 points)
        max_range = min(128, self.seq_len // 2 - 8)
        min_range = min(32, max_range)
        if min_range > max_range:
            return None

        a_len = random.randint(min_range, max_range)
        b_len = random.randint(min_range, max_range)
        # Place them with a gap
        a_start = random.randint(0, self.seq_len - a_len - b_len - 1)
        a_end = a_start + a_len - 1
        b_start = random.randint(a_end + 1, self.seq_len - b_len)
        b_end = b_start + b_len - 1

        a_mean = _interval_mean(self.ts, a_start, a_end)
        b_mean = _interval_mean(self.ts, b_start, b_end)

        # Avoid trivially equal means
        if a_mean == b_mean:
            return None

        verdict = "yes" if a_mean > b_mean else "no"

        question = random.choice(INTERVAL_COMPARISON_QUESTIONS).format(
            metric=self.metric,
            a_start=a_start, a_end=a_end, b_start=b_start, b_end=b_end,
        )
        answer_text = random.choice(INTERVAL_COMPARISON_ANSWERS[verdict]).format(
            metric=self.metric,
            a_start=a_start, a_end=a_end, a_mean=_fmt(a_mean),
            b_start=b_start, b_end=b_end, b_mean=_fmt(b_mean),
        )

        return _make_rl_result(
            question, verdict, 'rl_interval_comparison', answer_text, self.seq_len,
        )

    # ----------------------------------------------------------
    # R4: Half-mean difference
    # ----------------------------------------------------------

    def generate_half_mean_diff(self) -> Optional[Dict[str, Any]]:
        """R4: |mean(first half) - mean(second half)| → float.
        Atoms: A3 × 2 + subtraction + abs.
        """
        if self.seq_len < 32:
            return None

        mid = self.seq_len // 2
        first_mean = _interval_mean(self.ts, 0, mid - 1)
        second_mean = _interval_mean(self.ts, mid, self.seq_len - 1)
        diff = round(abs(first_mean - second_mean), 2)

        question = random.choice(HALF_MEAN_DIFF_QUESTIONS).format(
            metric=self.metric,
        )
        answer_text = random.choice(HALF_MEAN_DIFF_ANSWERS).format(
            metric=self.metric,
            first_mean=_fmt(first_mean), second_mean=_fmt(second_mean),
            diff=_fmt(diff),
        )

        return _make_rl_result(
            question, _fmt(diff), 'rl_half_mean_diff', answer_text, self.seq_len,
        )

    # ----------------------------------------------------------
    # R5: Mean stability
    # ----------------------------------------------------------

    def generate_mean_stability(self) -> Optional[Dict[str, Any]]:
        """R5: All chunk means within T of global mean? → yes/no.
        Atoms: A2 (chunked means) + A1 (global mean) + threshold.
        """
        if self.seq_len < 32:
            return None

        # Global mean via chunk-and-average
        chunks = _compute_chunks(self.ts, 0, self.seq_len - 1)
        chunk_means = [cm for _, _, cm, _ in chunks]
        global_mean = round(float(np.mean(chunk_means)), 2)

        # Max deviation of any chunk from global
        max_dev = round(max(abs(cm - global_mean) for cm in chunk_means), 2)

        threshold, verdict_yes = _pick_threshold(max_dev, '<')
        verdict = "yes" if verdict_yes else "no"

        question = random.choice(MEAN_STABILITY_QUESTIONS).format(
            metric=self.metric, threshold=_fmt(threshold),
        )
        answer_text = random.choice(MEAN_STABILITY_ANSWERS[verdict]).format(
            metric=self.metric, threshold=_fmt(threshold),
            global_mean=_fmt(global_mean),
            min_chunk=_fmt(min(chunk_means)),
            max_chunk=_fmt(max(chunk_means)),
        )

        return _make_rl_result(
            question, verdict, 'rl_mean_stability', answer_text, self.seq_len,
        )

    # ----------------------------------------------------------
    # R6: Monotonic chunks
    # ----------------------------------------------------------

    def generate_monotonic_chunks(self) -> Optional[Dict[str, Any]]:
        """R6: Chunk means monotonically increase/decrease over a range? → yes/no.
        Atoms: A2 (chunked means) + ordering check.
        """
        if self.seq_len < self.MIN_SEQ_LEN:
            return None

        # Pick a sub-range with 4-12 chunks
        max_len = min(192, self.seq_len)
        min_len = 64
        if min_len > max_len:
            return None

        range_len = random.randint(min_len, max_len)
        # Snap to multiple of 16
        range_len = (range_len // CHUNK_SIZE) * CHUNK_SIZE
        range_len = max(min_len, range_len)

        start = random.randint(0, self.seq_len - range_len)
        end = start + range_len - 1

        chunks = _compute_chunks(self.ts, start, end)
        chunk_means = [cm for _, _, cm, _ in chunks]

        direction = random.choice(['increasing', 'decreasing'])

        if direction == 'increasing':
            is_monotonic = all(a < b for a, b in zip(chunk_means, chunk_means[1:]))
        else:
            is_monotonic = all(a > b for a, b in zip(chunk_means, chunk_means[1:]))

        verdict = "yes" if is_monotonic else "no"
        chunk_means_str = "[" + ", ".join(_fmt(m) for m in chunk_means) + "]"

        question = random.choice(MONOTONIC_CHUNKS_QUESTIONS).format(
            metric=self.metric, start=start, end=end, direction=direction,
        )
        answer_text = random.choice(MONOTONIC_CHUNKS_ANSWERS[verdict]).format(
            metric=self.metric, start=start, end=end,
            direction=direction, chunk_means_str=chunk_means_str,
        )

        return _make_rl_result(
            question, verdict, 'rl_monotonic_chunks', answer_text, self.seq_len,
        )

    # ----------------------------------------------------------
    # Generate all
    # ----------------------------------------------------------

    def generate_all(self) -> List[Dict[str, Any]]:
        """Generate all compositional mean QAs for this timeseries."""
        results = []
        generators = [
            self.generate_condition_recovery,
            self.generate_mean_shift,
            self.generate_interval_comparison,
            self.generate_half_mean_diff,
            self.generate_mean_stability,
            self.generate_monotonic_chunks,
        ]
        for gen_fn in generators:
            try:
                r = gen_fn()
                if r is not None:
                    results.append(r)
            except Exception:
                logger.warning(f"RL {gen_fn.__name__} failed", exc_info=True)
        return results
