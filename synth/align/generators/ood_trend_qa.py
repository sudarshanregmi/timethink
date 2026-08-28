"""Trend OOD evaluation QA generators.

Novel compositions using trend atoms that RL never explicitly rewards.

Why each is OOD:
    - conditional_mean_by_type: RL compares adjacent segment means (R18),
      but never filters by type then aggregates
    - trend_follows_mean: RL checks max-in-type (R19), but never asks
      whether trend direction matches mean movement
    - longest_type_fraction: RL knows longest duration (R15) and type fraction
      (R17) separately, never combines them
    - symmetric_trend_sequence: RL checks monotonicity and ordering,
      never checks palindromic structure
"""

import random
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from loguru import logger

from synth.align.generators.atomic_qa import _fmt, _compute_chunks
from synth.align.generators.compositional_qa import (
    _coin_flip_keep, _interval_mean, _make_rl_result,
)

ALL_TREND_TYPES = ['increase', 'decrease', 'keep steady']

# ============================================================
# Question templates
# ============================================================

CONDITIONAL_MEAN_BY_TYPE_QUESTIONS = [
    "What is the average of the segment means for all {trend_type} segments in {metric}?",
    "For {metric}, compute the mean across all {trend_type} segments: take each {trend_type} segment's mean, then average them.",
    "In {metric}, what is the average mean of segments with {trend_type} trend?",
    "Compute the conditional mean for {metric}: average the means of all segments that follow a {trend_type} trend.",
    "For the {trend_type} segments of {metric}, what is the average of their means?",
]

TREND_FOLLOWS_MEAN_QUESTIONS = [
    "For each trend segment in {metric}, does the trend direction match the mean movement? That is, do increase segments have mean > global mean, decrease segments have mean < global mean? Report yes if ALL segments match, no otherwise.",
    "Check trend-mean consistency for {metric}: do all increase segments have above-average means and all decrease segments have below-average means?",
    "In {metric}, is the trend direction consistent with mean position? Increase segments should be above the global mean, decrease segments below.",
    "For {metric}, does every increase segment have a mean above the global mean, and every decrease segment have a mean below? Yes or no.",
]

LONGEST_TYPE_FRACTION_QUESTIONS = [
    "What fraction of {metric} is covered by the trend type that has the longest individual segment? Report as a decimal.",
    "For {metric}, find the trend type with the longest single segment, then compute what fraction of the total series that trend type covers.",
    "In {metric}, which trend type has the longest segment? What is that type's total fraction of the timeseries?",
    "Identify the trend type of the longest segment in {metric}, then report its total duration fraction.",
]

SYMMETRIC_TREND_SEQUENCE_QUESTIONS = [
    "Is the sequence of trend types in {metric} a palindrome? That is, does the sequence read the same forwards and backwards?",
    "For {metric}, check whether the trend type sequence is symmetric: does the pattern of segments mirror from start to end?",
    "Does the {metric} timeseries have a palindromic trend structure? The sequence of trend types should be the same forwards and backwards.",
    "Check whether the trend segments of {metric} form a symmetric sequence.",
    "Is the trend pattern of {metric} palindromic (same sequence of types from both ends)?",
]

# ============================================================
# Answer templates
# ============================================================

CONDITIONAL_MEAN_BY_TYPE_ANSWERS = [
    "The average mean of {trend_type} segments in {metric} is {verdict}.",
    "For {metric}, the conditional mean of {trend_type} segments is {verdict}.",
    "The conditional mean is {verdict} for {trend_type} segments in {metric}.",
]

TREND_FOLLOWS_MEAN_ANSWERS = {
    'yes': [
        "Yes, trend direction matches mean position in {metric}.",
        "Trend-mean consistency holds for {metric}.",
        "All segments in {metric} are consistent with mean position.",
    ],
    'no': [
        "No, trend direction does not fully match mean position in {metric}.",
        "Trend-mean consistency fails for {metric}.",
        "At least one segment in {metric} breaks the trend-mean pattern.",
    ],
}

LONGEST_TYPE_FRACTION_ANSWERS = [
    "The duration fraction of the trend type with the longest segment in {metric} is {verdict}.",
    "For {metric}, the longest segment's trend type covers a fraction of {verdict} of the series.",
    "In {metric}, the trend type of the longest segment accounts for {verdict} of the total duration.",
]

SYMMETRIC_TREND_SEQUENCE_ANSWERS = {
    'yes': [
        "Yes, the trend sequence of {metric} is a palindrome.",
        "The trend pattern of {metric} is symmetric.",
        "Confirmed, {metric} has a palindromic trend structure.",
    ],
    'no': [
        "No, the trend sequence of {metric} is not a palindrome.",
        "The trend pattern of {metric} is not symmetric.",
        "The trend structure of {metric} is not palindromic.",
    ],
}


# ============================================================
# Generator
# ============================================================

class OODTrendGenerator:
    """Trend OOD compositions — eval only.

    Novel uses of trend structure never seen during RL training.
    """

    MIN_SEGMENTS = 2

    def __init__(
        self,
        timeseries: np.ndarray,
        metric: str,
        seq_len: int,
        trend_list: List[Tuple[str, int, int]],
    ):
        self.ts = np.asarray(timeseries, dtype=float)
        self.metric = metric
        self.seq_len = seq_len
        self.trend_list = trend_list

    # --- O10: Conditional mean by trend type ---

    def generate_conditional_mean_by_type(self) -> Optional[Dict[str, Any]]:
        if len(self.trend_list) < self.MIN_SEGMENTS:
            return None

        # Pick a trend type that has at least 1 segment
        present_types = list(set(t for t, _, _ in self.trend_list))
        trend_type = random.choice(present_types)

        # Compute mean for each segment of that type
        seg_means = []
        for t, s, e in self.trend_list:
            if t == trend_type:
                seg_means.append(_interval_mean(self.ts, s, e))

        if not seg_means:
            return None

        avg = round(float(np.mean(seg_means)), 2)
        means_str = "[" + ", ".join(_fmt(m) for m in seg_means) + "]"

        answer_text = random.choice(CONDITIONAL_MEAN_BY_TYPE_ANSWERS).format(
            metric=self.metric, trend_type=trend_type,
            verdict=_fmt(avg), count=len(seg_means), means_str=means_str,
        )
        return _make_rl_result(
            random.choice(CONDITIONAL_MEAN_BY_TYPE_QUESTIONS).format(
                metric=self.metric, trend_type=trend_type,
            ),
            _fmt(avg), 'ood_conditional_mean_by_type', answer_text, self.seq_len,
        )

    # --- O11: Trend follows mean ---

    def generate_trend_follows_mean(self) -> Optional[Dict[str, Any]]:
        if len(self.trend_list) < self.MIN_SEGMENTS:
            return None

        # Global mean
        chunks = _compute_chunks(self.ts, 0, self.seq_len - 1)
        global_mean = round(float(np.mean([cm for _, _, cm, _ in chunks])), 2)

        # Check each segment
        all_match = True
        details_parts = []
        for i, (t, s, e) in enumerate(self.trend_list):
            seg_mean = _interval_mean(self.ts, s, e)
            if t == 'increase':
                match = seg_mean > global_mean
            elif t == 'decrease':
                match = seg_mean < global_mean
            else:  # keep steady
                continue  # no directional expectation

            status = "match" if match else "mismatch"
            details_parts.append(
                f"seg {i} ({t}, mean {_fmt(seg_mean)}): {status}"
            )
            if not match:
                all_match = False

        verdict = "yes" if all_match else "no"
        # Natural rate is ~95% yes — drop majority-class samples via fair coin
        # over {yes, no} to force 50/50 in the emitted set.
        if not _coin_flip_keep(verdict, ['yes', 'no'], 'ood_trend_follows_mean'):
            return None
        details = "; ".join(details_parts) if details_parts else "no increase/decrease segments"

        answer_text = random.choice(TREND_FOLLOWS_MEAN_ANSWERS[verdict]).format(
            metric=self.metric, global_mean=_fmt(global_mean), details=details,
        )
        return _make_rl_result(
            random.choice(TREND_FOLLOWS_MEAN_QUESTIONS).format(metric=self.metric),
            verdict, 'ood_trend_follows_mean', answer_text, self.seq_len,
        )

    # --- O12: Longest type fraction ---

    def generate_longest_type_fraction(self) -> Optional[Dict[str, Any]]:
        if len(self.trend_list) < self.MIN_SEGMENTS:
            return None

        durations = [(e - s, i) for i, (_, s, e) in enumerate(self.trend_list)]
        max_dur, max_idx = max(durations)
        seg_type = self.trend_list[max_idx][0]

        type_dur = sum(e - s for t, s, e in self.trend_list if t == seg_type)
        total_dur = sum(e - s for _, s, e in self.trend_list)

        if total_dur == 0:
            return None

        fraction = round(type_dur / total_dur, 2)

        # Natural distribution locks to 0.53 (~95% of 2-segment trend lists).
        # Quantize into 3 equal-width buckets and coin-flip over buckets so
        # emitted samples spread across the fraction range without
        # collapsing to a single value. Previously 5 buckets, which was
        # too aggressive — retention dropped to ~20% and final test split
        # carried only 11 samples. 3 buckets keeps the spread objective
        # (low/mid/high fractions covered) while tripling retention.
        bucket = min(int(fraction * 3), 2)  # 0..2
        if not _coin_flip_keep(bucket, [0, 1, 2], 'ood_longest_type_fraction'):
            return None

        answer_text = random.choice(LONGEST_TYPE_FRACTION_ANSWERS).format(
            metric=self.metric, seg_type=seg_type, seg_idx=max_idx,
            max_dur=max_dur, verdict=_fmt(fraction),
            type_dur=type_dur, total_dur=total_dur,
        )
        return _make_rl_result(
            random.choice(LONGEST_TYPE_FRACTION_QUESTIONS).format(
                metric=self.metric,
            ),
            _fmt(fraction), 'ood_longest_type_fraction', answer_text, self.seq_len,
        )

    # --- O13: Symmetric trend sequence ---

    def generate_symmetric_trend_sequence(self) -> Optional[Dict[str, Any]]:
        if len(self.trend_list) < 3:  # palindrome check needs ≥3 to be interesting
            return None

        types = [t for t, _, _ in self.trend_list]
        is_palindrome = types == types[::-1]
        verdict = "yes" if is_palindrome else "no"

        sequence_str = " → ".join(types)

        answer_text = random.choice(SYMMETRIC_TREND_SEQUENCE_ANSWERS[verdict]).format(
            metric=self.metric, sequence_str=sequence_str,
        )
        return _make_rl_result(
            random.choice(SYMMETRIC_TREND_SEQUENCE_QUESTIONS).format(
                metric=self.metric,
            ),
            verdict, 'ood_symmetric_trend_sequence', answer_text, self.seq_len,
        )

    # --- Generate all ---

    def generate_all(self) -> List[Dict[str, Any]]:
        results = []
        generators = [
            self.generate_conditional_mean_by_type,
            self.generate_trend_follows_mean,
            self.generate_longest_type_fraction,
            self.generate_symmetric_trend_sequence,
        ]
        for gen_fn in generators:
            try:
                r = gen_fn()
                if r is not None:
                    results.append(r)
            except Exception:
                logger.warning(f"OOD {gen_fn.__name__} failed", exc_info=True)
        return results
