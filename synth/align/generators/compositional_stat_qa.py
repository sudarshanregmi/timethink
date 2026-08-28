"""Cross-category compositional QA generators for RL training.

These compose atoms from DIFFERENT SFT categories (A-D):
    - A: mean (chunk-and-average)
    - B: std
    - C: extrema (value + position)
    - D: percentiles

The key difference from compositional_qa.py (mean-only): these tasks
require the model to chain skills it learned in separate SFT atoms.

Tasks:
    R7:  range              — max - min (C1 × 2)
    R9:  volatility_change  — std(H2) > std(H1)? (B2 × 2)
    R10: max_in_first_half  — is argmax in first half? (C2)
    R11: normalized_range   — (max - min) / std (C1 + B1)
    R12: median_mean_close  — |median - mean| < T? (D1@50 + A1)
    R13: extrema_same_half  — argmin and argmax in same half? (C2 × 2)
"""

import random
from typing import Any, Dict, List, Optional

import numpy as np
from loguru import logger

from synth.align.generators.atomic_qa import _compute_chunks, _fmt
from synth.align.generators.compositional_qa import (
    _interval_mean, _pick_threshold, _make_rl_result,
)

# ============================================================
# Question templates
# ============================================================

RANGE_QUESTIONS = [
    "What is the range of {metric}? The range is the difference between the maximum and minimum values.",
    "Compute the range (max - min) of the {metric} timeseries.",
    "For {metric}, find the difference between the largest and smallest values.",
    "What is the spread of {metric} measured as max minus min?",
    "Calculate the range of {metric}: its maximum value minus its minimum value.",
    "How large is the range of {metric} (i.e., max - min)?",
]

VOLATILITY_CHANGE_QUESTIONS = [
    "Does the standard deviation of {metric} increase from the first half to the second half?",
    "For {metric}, is the second half more volatile than the first half? Compare std(first half) vs std(second half).",
    "Check whether the standard deviation of {metric} in the second half exceeds that of the first half.",
    "Is std(second half) > std(first half) for {metric}?",
    "For the {metric} timeseries, does volatility (standard deviation) increase from the first to the second half?",
    "Compare the variability of {metric} across its two halves: is the second half's std greater?",
]

MAX_IN_FIRST_HALF_QUESTIONS = [
    "Is the global maximum of {metric} located in the first half of the data?",
    "Does the maximum value of {metric} occur in the first half (indices 0 to {mid})?",
    "For {metric}, is the position of the maximum value in the first half of the timeseries?",
    "Check whether the global maximum of {metric} falls within the first half of the series.",
    "Is the peak value of {metric} found in the first half of the data?",
    "For the {metric} timeseries, does the global maximum occur before the midpoint?",
]

NORMALIZED_RANGE_QUESTIONS = [
    "What is the normalized range of {metric}? Normalized range = (max - min) / std.",
    "Compute (max - min) / std for the {metric} timeseries.",
    "For {metric}, what is the ratio of the range to the standard deviation?",
    "Calculate the range of {metric} divided by its standard deviation.",
    "What is the range-to-std ratio for {metric}?",
    "Find (max - min) / standard deviation for {metric}.",
]

MEDIAN_MEAN_CLOSE_QUESTIONS = [
    "Is the median of {metric} close to its mean? Closeness means |median - mean| < {threshold}.",
    "For {metric}, check whether the absolute difference between the median and the mean is less than {threshold}.",
    "Does {metric} have a median close to its mean? That is, |median - mean| < {threshold}.",
    "Check if the median and mean of {metric} are within {threshold} of each other.",
    "For the {metric} timeseries, is |median - mean| below {threshold}?",
    "Assess whether {metric}'s median is close to its mean, with tolerance {threshold}.",
]

EXTREMA_SAME_HALF_QUESTIONS = [
    "Are the minimum and maximum values of {metric} located in the same half of the data?",
    "For {metric}, do the global min and max occur in the same half of the timeseries?",
    "Check whether the positions of the minimum and maximum of {metric} are both in the same half.",
    "Do the extrema of {metric} fall within the same half of the series?",
    "For the {metric} data, are both the min and max positions in the first half, or both in the second half?",
    "Is it true that the minimum and maximum of {metric} are co-located in the same half?",
]

# ============================================================
# Answer templates
# ============================================================

RANGE_ANSWERS = [
    "The range of {metric} is {value}.",
    "For {metric}, the range is {value}.",
    "The range of the {metric} data is {value}.",
]

VOLATILITY_CHANGE_ANSWERS = {
    'yes': [
        "Yes, {metric} becomes more volatile in the second half.",
        "Volatility increases for {metric} in the second half.",
        "The second half of {metric} is more volatile.",
    ],
    'no': [
        "No, {metric} does not become more volatile in the second half.",
        "Volatility does not increase for {metric}.",
        "The second half of {metric} is not more volatile.",
    ],
}

MAX_IN_FIRST_HALF_ANSWERS = {
    'yes': [
        "Yes, the global maximum of {metric} is in the first half.",
        "The maximum of {metric} falls in the first half.",
        "Confirmed, the peak of {metric} is in the first half.",
    ],
    'no': [
        "No, the global maximum of {metric} is in the second half.",
        "The maximum of {metric} falls in the second half.",
        "The peak of {metric} is not in the first half.",
    ],
}

NORMALIZED_RANGE_ANSWERS = [
    "The normalized range of {metric} is {value}.",
    "For {metric}, the range-to-std ratio is {value}.",
    "The range-to-std ratio of {metric} is {value}.",
]

MEDIAN_MEAN_CLOSE_ANSWERS = {
    'yes': [
        "Yes, the median and mean of {metric} are close.",
        "The median and mean of {metric} are within tolerance.",
        "For {metric}, the median and mean are close.",
    ],
    'no': [
        "No, the median and mean of {metric} are not close.",
        "The median and mean of {metric} are not within tolerance.",
        "For {metric}, the median and mean are not close.",
    ],
}

EXTREMA_SAME_HALF_ANSWERS = {
    'yes': [
        "Yes, both extrema of {metric} are in the same half.",
        "The min and max of {metric} are co-located in the same half.",
        "Both extrema fall in the same half of {metric}.",
    ],
    'no': [
        "No, the extrema of {metric} are in different halves.",
        "The min and max of {metric} are in opposite halves.",
        "The extrema span both halves of {metric}.",
    ],
}


# ============================================================
# Generator
# ============================================================

class CompositionalStatGenerator:
    """Cross-category RL compositions: mean + std + extrema + percentiles.

    Each task chains atoms from at least 2 different SFT categories.
    """

    MIN_SEQ_LEN = 32

    def __init__(self, timeseries: np.ndarray, metric: str, seq_len: int):
        self.ts = np.asarray(timeseries, dtype=float)
        self.metric = metric
        self.seq_len = seq_len

    # --- R7: Range ---

    def generate_range(self) -> Dict[str, Any]:
        """R7: max - min → numeric. (C1 × 2)"""
        max_val = round(float(np.max(self.ts)), 2)
        min_val = round(float(np.min(self.ts)), 2)
        range_val = round(max_val - min_val, 2)

        answer_text = random.choice(RANGE_ANSWERS).format(
            metric=self.metric, value=_fmt(range_val),
            max_val=_fmt(max_val), min_val=_fmt(min_val),
            length=self.seq_len,
        )
        return _make_rl_result(
            random.choice(RANGE_QUESTIONS).format(metric=self.metric),
            _fmt(range_val), 'rl_range', answer_text, self.seq_len,
        )

    # --- R9: Volatility change ---

    def generate_volatility_change(self) -> Optional[Dict[str, Any]]:
        """R9: std(H2) > std(H1)? → yes/no. (B2 × 2)"""
        if self.seq_len < self.MIN_SEQ_LEN:
            return None

        mid = self.seq_len // 2
        std1 = round(float(np.std(self.ts[:mid])), 2)
        std2 = round(float(np.std(self.ts[mid:])), 2)

        verdict = "yes" if std2 > std1 else "no"

        answer_text = random.choice(VOLATILITY_CHANGE_ANSWERS[verdict]).format(
            metric=self.metric, std1=_fmt(std1), std2=_fmt(std2),
        )
        return _make_rl_result(
            random.choice(VOLATILITY_CHANGE_QUESTIONS).format(metric=self.metric),
            verdict, 'rl_volatility_change', answer_text, self.seq_len,
        )

    # --- R10: Max in first half ---

    def generate_max_in_first_half(self) -> Optional[Dict[str, Any]]:
        """R10: is argmax in first half? → yes/no. (C2)"""
        if self.seq_len < self.MIN_SEQ_LEN:
            return None

        mid = self.seq_len // 2
        pos = int(np.argmax(self.ts))
        value = round(float(self.ts[pos]), 2)

        verdict = "yes" if pos < mid else "no"

        if verdict == "yes":
            question = random.choice(MAX_IN_FIRST_HALF_QUESTIONS).format(
                metric=self.metric, mid=mid - 1,
            )
            answer_text = random.choice(MAX_IN_FIRST_HALF_ANSWERS['yes']).format(
                metric=self.metric, pos=pos, value=_fmt(value),
                mid=mid - 1, length=self.seq_len,
            )
        else:
            question = random.choice(MAX_IN_FIRST_HALF_QUESTIONS).format(
                metric=self.metric, mid=mid - 1,
            )
            answer_text = random.choice(MAX_IN_FIRST_HALF_ANSWERS['no']).format(
                metric=self.metric, pos=pos, value=_fmt(value),
                mid_plus=mid, last=self.seq_len - 1, length=self.seq_len,
            )

        return _make_rl_result(
            question, verdict, 'rl_max_in_first_half', answer_text, self.seq_len,
        )

    # --- R11: Normalized range ---

    def generate_normalized_range(self) -> Optional[Dict[str, Any]]:
        """R11: (max - min) / std → numeric. (C1 + B1)
        Guard: only when std > 0.01 (avoid division by ~0).
        """
        std_val = round(float(np.std(self.ts)), 2)
        if std_val < 0.01:
            return None

        max_val = round(float(np.max(self.ts)), 2)
        min_val = round(float(np.min(self.ts)), 2)
        range_val = round(max_val - min_val, 2)
        normalized = round(range_val / std_val, 2)

        answer_text = random.choice(NORMALIZED_RANGE_ANSWERS).format(
            metric=self.metric, value=_fmt(normalized),
            range_val=_fmt(range_val), std_val=_fmt(std_val),
            max_val=_fmt(max_val), min_val=_fmt(min_val),
        )
        return _make_rl_result(
            random.choice(NORMALIZED_RANGE_QUESTIONS).format(metric=self.metric),
            _fmt(normalized), 'rl_normalized_range', answer_text, self.seq_len,
        )

    # --- R12: Median vs mean proximity ---

    def generate_median_mean_close(self) -> Optional[Dict[str, Any]]:
        """R12: |median - mean| < T? → yes/no. (D1@50 + A1)"""
        median = round(float(np.percentile(self.ts, 50)), 2)
        # Mean via chunk-and-average for consistency
        chunks = _compute_chunks(self.ts, 0, self.seq_len - 1)
        mean = round(float(np.mean([cm for _, _, cm, _ in chunks])), 2)
        diff = round(abs(median - mean), 2)

        threshold, verdict_yes = _pick_threshold(diff, '<')
        verdict = "yes" if verdict_yes else "no"

        question = random.choice(MEDIAN_MEAN_CLOSE_QUESTIONS).format(
            metric=self.metric, threshold=_fmt(threshold),
        )
        answer_text = random.choice(MEDIAN_MEAN_CLOSE_ANSWERS[verdict]).format(
            metric=self.metric,
            median=_fmt(median), mean=_fmt(mean),
            diff=_fmt(diff), threshold=_fmt(threshold),
        )
        return _make_rl_result(
            question, verdict, 'rl_median_mean_close', answer_text, self.seq_len,
        )

    # --- R13: Extrema same half ---

    def generate_extrema_same_half(self) -> Optional[Dict[str, Any]]:
        """R13: are argmin and argmax in the same half? → yes/no. (C2 × 2)"""
        if self.seq_len < self.MIN_SEQ_LEN:
            return None

        mid = self.seq_len // 2
        min_pos = int(np.argmin(self.ts))
        max_pos = int(np.argmax(self.ts))
        min_val = round(float(self.ts[min_pos]), 2)
        max_val = round(float(self.ts[max_pos]), 2)

        min_in_first = min_pos < mid
        max_in_first = max_pos < mid
        same_half = min_in_first == max_in_first

        verdict = "yes" if same_half else "no"
        min_half = "first" if min_in_first else "second"
        max_half = "first" if max_in_first else "second"
        which_half = min_half  # only used when same_half is True

        if verdict == "yes":
            answer_text = random.choice(EXTREMA_SAME_HALF_ANSWERS['yes']).format(
                metric=self.metric, which_half=which_half,
                min_val=_fmt(min_val), min_pos=min_pos,
                max_val=_fmt(max_val), max_pos=max_pos,
            )
        else:
            answer_text = random.choice(EXTREMA_SAME_HALF_ANSWERS['no']).format(
                metric=self.metric,
                min_val=_fmt(min_val), min_pos=min_pos, min_half=min_half,
                max_val=_fmt(max_val), max_pos=max_pos, max_half=max_half,
            )

        return _make_rl_result(
            random.choice(EXTREMA_SAME_HALF_QUESTIONS).format(metric=self.metric),
            verdict, 'rl_extrema_same_half', answer_text, self.seq_len,
        )

    # --- Generate all ---

    def generate_all(self) -> List[Dict[str, Any]]:
        """Generate all cross-category RL compositions."""
        results = []
        generators = [
            self.generate_range,
            self.generate_volatility_change,
            self.generate_max_in_first_half,
            self.generate_normalized_range,
            self.generate_median_mean_close,
            self.generate_extrema_same_half,
        ]
        for gen_fn in generators:
            try:
                r = gen_fn()
                if r is not None:
                    results.append(r)
            except Exception:
                logger.warning(f"RL {gen_fn.__name__} failed", exc_info=True)
        return results
