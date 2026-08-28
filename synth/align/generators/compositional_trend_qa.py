"""Trend compositional QA generators for RL training.

Compose the single trend enumeration atom (E1) with simple reasoning
or cross with stat atoms (A-D).

The trend SFT atom teaches enumeration. These RL tasks require the model
to enumerate (learned) then do a derived operation (count, lookup, filter,
compare) — basic LLM capabilities composed with the recognition skill.

Tasks:
    R14: segment_count         — how many segments? (enumerate + count)
    R15: segment_type_at_pos   — what trend at index p? (enumerate + lookup)
    R16: segment_duration      — duration of segment at index p? (enumerate + subtract)
    R17: dominant_trend_type   — most segments by type? (enumerate + count + argmax)
    R18: longest_segment       — duration of longest segment (enumerate + argmax)
    R19: type_of_longest       — trend type of longest (enumerate + argmax + read)
    R20: type_duration_fraction — fraction in trend type X (enumerate + sum + divide)
    R21: segment_mean_compare  — mean(seg_k) > mean(seg_k+1)? (E + A, cross-category)
    R22: max_in_trend_type     — global max in a given trend type? (C + E, cross-category)
"""

import random
from typing import Any, Dict, List, Optional, Tuple
from collections import Counter

import numpy as np
from loguru import logger

from synth.align.generators.atomic_qa import _fmt, _compute_chunks
from synth.align.generators.compositional_qa import (
    _interval_mean, _make_rl_result,
)

ALL_TREND_TYPES = ['increase', 'decrease', 'keep steady']

# ============================================================
# Question templates
# ============================================================

# R14: segment count
SEGMENT_COUNT_QUESTIONS = [
    "How many distinct trend segments does the {metric} timeseries have?",
    "Count the number of trend segments in {metric}.",
    "For {metric}, how many separate trend phases are there?",
    "What is the total number of trend segments in {metric}?",
    "How many trend changes occur in {metric}? Report the segment count.",
    "Determine the number of distinct trend segments in {metric}.",
]

# R15: type at position
SEGMENT_TYPE_AT_POS_QUESTIONS = [
    "What is the trend at index {pos} in the {metric} timeseries?",
    "For {metric}, what trend type is occurring at position {pos}?",
    "At index {pos} of {metric}, is the data increasing, decreasing, or keeping steady?",
    "What is the trend direction of {metric} at index {pos}?",
    "Identify the trend type at position {pos} in {metric}.",
    "For {metric}, what is happening at index {pos}?",
]

# R16: segment duration
SEGMENT_DURATION_QUESTIONS = [
    "What is the duration of the trend segment containing index {pos} in {metric}?",
    "For {metric}, how long is the trend segment at position {pos}?",
    "How many data points are in the trend segment of {metric} that includes index {pos}?",
    "Find the length of the segment containing position {pos} in {metric}.",
    "For {metric}, what is the duration of the trend phase at index {pos}?",
]

# R17: dominant type
DOMINANT_TYPE_QUESTIONS = [
    "Which trend type has the most segments in {metric}?",
    "For {metric}, which trend direction appears in the most segments?",
    "What is the dominant trend type by segment count in {metric}?",
    "In {metric}, which trend type has the highest segment count?",
    "Which of increase, decrease, or keep steady occurs most frequently in {metric}?",
]

# R18: longest segment
LONGEST_SEGMENT_QUESTIONS = [
    "What is the duration of the longest trend segment in {metric}?",
    "For {metric}, how many data points does the longest segment span?",
    "Find the duration of the longest trend segment in {metric}.",
    "What is the maximum segment duration in {metric}?",
    "How long is the longest trend phase in {metric}?",
]

# R19: type of longest
TYPE_OF_LONGEST_QUESTIONS = [
    "What trend type does the longest segment of {metric} follow?",
    "For {metric}, what is the trend direction of the longest segment?",
    "What type of trend characterizes the longest segment in {metric}?",
    "In {metric}, is the longest segment increasing, decreasing, or keeping steady?",
    "Identify the trend type of the longest segment in {metric}.",
]

# R20: type duration fraction
TYPE_DURATION_FRACTION_QUESTIONS = [
    "What fraction of {metric} is spent in {trend_type} segments? Report as a decimal.",
    "For {metric}, what proportion of the data follows a {trend_type} trend?",
    "Compute the fraction of time {metric} spends in {trend_type} phases.",
    "What share of {metric} is {trend_type}? Express as a decimal rounded to 2 places.",
    "How much of {metric} is {trend_type}? Report the fraction of total duration.",
]

# R21: segment mean compare (cross-category)
SEGMENT_MEAN_COMPARE_QUESTIONS = [
    "For {metric}, is the mean of the segment at index {pos_a} greater than the mean of the segment at index {pos_b}?",
    "Compare adjacent segment means in {metric}: is the mean around index {pos_a} higher than around index {pos_b}?",
    "In {metric}, does the trend segment containing index {pos_a} have a greater mean than the segment containing index {pos_b}?",
    "For {metric}, which segment has the higher mean: the one at index {pos_a} or the one at index {pos_b}? Is the first greater?",
]

# R22: max in trend type (cross-category)
MAX_IN_TREND_TYPE_QUESTIONS = [
    "Is the global maximum of {metric} located in an {trend_type} segment?",
    "For {metric}, does the global max fall within a {trend_type} trend phase?",
    "Check whether the peak of {metric} occurs in an {trend_type} segment.",
    "Does the maximum value of {metric} lie in an {trend_type} segment?",
    "Is the global maximum of {metric} in a segment with {trend_type} trend?",
]

# ============================================================
# Answer templates
# ============================================================

SEGMENT_COUNT_ANSWERS_SINGULAR = [
    "The {metric} timeseries has {count} distinct trend segment.",
    "There is {count} trend segment in {metric}.",
    "{metric} has {count} trend segment.",
]
SEGMENT_COUNT_ANSWERS_PLURAL = [
    "The {metric} timeseries has {count} distinct trend segments.",
    "There are {count} trend segments in {metric}.",
    "{metric} has {count} trend segments.",
]

SEGMENT_TYPE_AT_POS_ANSWERS = [
    "The trend at index {pos} in {metric} is {verdict}.",
    "At index {pos}, {metric} is in a {verdict} phase.",
    "{metric} shows a {verdict} trend at position {pos}.",
]

SEGMENT_DURATION_ANSWERS = [
    "The segment duration is {duration} data points.",
    "The trend segment containing that index in {metric} spans {duration} data points.",
    "For {metric}, the segment at that index has a duration of {duration} data points.",
]

DOMINANT_TYPE_ANSWERS = [
    "The dominant trend type in {metric} is {verdict}.",
    "{verdict} is the most frequent trend type in {metric}.",
    "The dominant trend is {verdict} for {metric}.",
]

LONGEST_SEGMENT_ANSWERS = [
    "The longest segment in {metric} has a duration of {duration} data points.",
    "The longest segment spans {duration} data points in {metric}.",
    "The longest segment duration in {metric} is {duration}.",
]

TYPE_OF_LONGEST_ANSWERS = [
    "The longest segment in {metric} follows a {verdict} trend.",
    "The trend type of the longest segment in {metric} is {verdict}.",
    "For {metric}, the longest segment is {verdict}.",
]

TYPE_DURATION_FRACTION_ANSWERS = [
    "The fraction of {metric} spent in {trend_type} segments is {verdict}.",
    "For {metric}, {trend_type} segments account for a fraction of {verdict} of the total duration.",
    "In {metric}, the {trend_type} trend type covers {verdict} of the series.",
]

SEGMENT_MEAN_COMPARE_ANSWERS = {
    'yes': [
        "Yes, the segment at the first index has a greater mean than the segment at the second index in {metric}.",
        "The first segment's mean exceeds the second's in {metric}.",
    ],
    'no': [
        "No, the segment at the first index does not have a greater mean than the segment at the second index in {metric}.",
        "The first segment's mean does not exceed the second's in {metric}.",
    ],
}

MAX_IN_TREND_TYPE_ANSWERS = {
    'yes': [
        "Yes, the global maximum of {metric} is in a {trend_type} segment.",
        "The peak of {metric} falls within a {trend_type} segment.",
    ],
    'no': [
        "No, the global maximum of {metric} is not in a {trend_type} segment.",
        "The peak of {metric} does not fall in a {trend_type} segment.",
    ],
}


# ============================================================
# Generator
# ============================================================

class CompositionalTrendGenerator:
    """RL compositions using trend enumeration atom + simple reasoning.

    R14-R16: derived reads (count, lookup, duration) — enumeration + basic ops
    R17-R20: aggregations (dominant, longest, fraction) — enumeration + reduce
    R21-R22: cross-category (segment mean, max in type) — E + A/C atoms
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

    def _segment_at_position(self, pos: int):
        for i, (t, s, e) in enumerate(self.trend_list):
            if s <= pos <= e:
                return i, t, s, e
        return None

    def _segment_summary(self) -> str:
        parts = []
        for seg_type, start, end in self.trend_list:
            parts.append(f"{seg_type} ({start}-{end})")
        return ", ".join(parts)

    def _pick_position_in_segment(self, seg_idx: int) -> int:
        _, start, end = self.trend_list[seg_idx]
        return random.randint(start, end)

    # --- R14: Segment count ---

    def generate_segment_count(self) -> Optional[Dict[str, Any]]:
        if not self.trend_list:
            return None

        count = len(self.trend_list)
        summary = self._segment_summary()

        templates = SEGMENT_COUNT_ANSWERS_SINGULAR if count == 1 else SEGMENT_COUNT_ANSWERS_PLURAL
        answer_text = random.choice(templates).format(
            metric=self.metric, count=count, summary=summary,
        )
        return _make_rl_result(
            random.choice(SEGMENT_COUNT_QUESTIONS).format(metric=self.metric),
            str(count), 'rl_segment_count', answer_text, self.seq_len,
        )

    # --- R15: Type at position ---

    def generate_segment_type_at_pos(self) -> Optional[Dict[str, Any]]:
        if not self.trend_list:
            return None

        seg_idx = random.randrange(len(self.trend_list))
        pos = self._pick_position_in_segment(seg_idx)
        seg_type, start, end = self.trend_list[seg_idx]

        answer_text = random.choice(SEGMENT_TYPE_AT_POS_ANSWERS).format(
            metric=self.metric, pos=pos, verdict=seg_type,
            start=start, end=end,
        )
        return _make_rl_result(
            random.choice(SEGMENT_TYPE_AT_POS_QUESTIONS).format(
                metric=self.metric, pos=pos,
            ),
            seg_type, 'rl_segment_type_at_pos', answer_text, self.seq_len,
        )

    # --- R16: Segment duration ---

    def generate_segment_duration(self) -> Optional[Dict[str, Any]]:
        if not self.trend_list:
            return None

        seg_idx = random.randrange(len(self.trend_list))
        pos = self._pick_position_in_segment(seg_idx)
        seg_type, start, end = self.trend_list[seg_idx]
        duration = end - start

        answer_text = random.choice(SEGMENT_DURATION_ANSWERS).format(
            metric=self.metric, pos=pos, duration=duration,
            seg_type=seg_type, start=start, end=end,
        )
        return _make_rl_result(
            random.choice(SEGMENT_DURATION_QUESTIONS).format(
                metric=self.metric, pos=pos,
            ),
            str(duration), 'rl_segment_duration', answer_text, self.seq_len,
        )

    # --- R17: Dominant trend type ---

    def generate_dominant_trend_type(self) -> Optional[Dict[str, Any]]:
        if len(self.trend_list) < self.MIN_SEGMENTS:
            return None

        types = [t for t, _, _ in self.trend_list]
        counts = Counter(types)
        dominant = counts.most_common(1)[0][0]

        # Tie-break: first occurrence order
        max_count = counts[dominant]
        tied = [t for t, c in counts.items() if c == max_count]
        if len(tied) > 1:
            for t in types:
                if t in tied:
                    dominant = t
                    break

        counts_str = ", ".join(
            f"{t}: {counts.get(t, 0)}" for t in ALL_TREND_TYPES if counts.get(t, 0) > 0
        )

        answer_text = random.choice(DOMINANT_TYPE_ANSWERS).format(
            metric=self.metric, verdict=dominant,
            counts_str=counts_str, count=counts[dominant],
            total=len(types),
        )
        return _make_rl_result(
            random.choice(DOMINANT_TYPE_QUESTIONS).format(metric=self.metric),
            dominant, 'rl_dominant_trend_type', answer_text, self.seq_len,
        )

    # --- R18: Longest segment duration ---

    def generate_longest_segment(self) -> Optional[Dict[str, Any]]:
        if len(self.trend_list) < self.MIN_SEGMENTS:
            return None

        durations = [(e - s, i) for i, (_, s, e) in enumerate(self.trend_list)]
        max_dur, max_idx = max(durations)
        seg_type, start, end = self.trend_list[max_idx]

        answer_text = random.choice(LONGEST_SEGMENT_ANSWERS).format(
            metric=self.metric, seg_idx=max_idx, seg_type=seg_type,
            start=start, end=end, duration=max_dur,
        )
        return _make_rl_result(
            random.choice(LONGEST_SEGMENT_QUESTIONS).format(metric=self.metric),
            str(max_dur), 'rl_longest_segment', answer_text, self.seq_len,
        )

    # --- R19: Type of longest ---

    def generate_type_of_longest(self) -> Optional[Dict[str, Any]]:
        if len(self.trend_list) < self.MIN_SEGMENTS:
            return None

        durations = [(e - s, i) for i, (_, s, e) in enumerate(self.trend_list)]
        max_dur, max_idx = max(durations)
        seg_type, start, end = self.trend_list[max_idx]

        answer_text = random.choice(TYPE_OF_LONGEST_ANSWERS).format(
            metric=self.metric, verdict=seg_type,
            seg_idx=max_idx, start=start, end=end, duration=max_dur,
        )
        return _make_rl_result(
            random.choice(TYPE_OF_LONGEST_QUESTIONS).format(metric=self.metric),
            seg_type, 'rl_type_of_longest', answer_text, self.seq_len,
        )

    # --- R20: Duration fraction by type ---

    def generate_type_duration_fraction(self) -> Optional[Dict[str, Any]]:
        if len(self.trend_list) < self.MIN_SEGMENTS:
            return None

        present_types = list(set(t for t, _, _ in self.trend_list))
        trend_type = random.choice(present_types)

        type_dur = sum(e - s for t, s, e in self.trend_list if t == trend_type)
        total_dur = sum(e - s for _, s, e in self.trend_list)

        if total_dur == 0:
            return None

        fraction = round(type_dur / total_dur, 2)

        answer_text = random.choice(TYPE_DURATION_FRACTION_ANSWERS).format(
            metric=self.metric, trend_type=trend_type,
            verdict=_fmt(fraction), type_dur=type_dur, total_dur=total_dur,
        )
        return _make_rl_result(
            random.choice(TYPE_DURATION_FRACTION_QUESTIONS).format(
                metric=self.metric, trend_type=trend_type,
            ),
            _fmt(fraction), 'rl_type_duration_fraction', answer_text, self.seq_len,
        )

    # --- R21: Segment mean comparison (cross-category: E + A) ---

    def generate_segment_mean_compare(self) -> Optional[Dict[str, Any]]:
        if len(self.trend_list) < self.MIN_SEGMENTS:
            return None

        idx_a = random.randrange(len(self.trend_list) - 1)
        idx_b = idx_a + 1
        a_type, a_start, a_end = self.trend_list[idx_a]
        b_type, b_start, b_end = self.trend_list[idx_b]

        if (a_end - a_start) < 4 or (b_end - b_start) < 4:
            return None

        a_mean = _interval_mean(self.ts, a_start, a_end)
        b_mean = _interval_mean(self.ts, b_start, b_end)

        if a_mean == b_mean:
            return None

        # Pick representative positions inside each segment
        pos_a = self._pick_position_in_segment(idx_a)
        pos_b = self._pick_position_in_segment(idx_b)

        verdict = "yes" if a_mean > b_mean else "no"

        question = random.choice(SEGMENT_MEAN_COMPARE_QUESTIONS).format(
            metric=self.metric, pos_a=pos_a, pos_b=pos_b,
        )
        answer_text = random.choice(SEGMENT_MEAN_COMPARE_ANSWERS[verdict]).format(
            metric=self.metric,
            pos_a=pos_a, pos_b=pos_b,
            a_mean=_fmt(a_mean), b_mean=_fmt(b_mean),
            a_type=a_type, a_start=a_start, a_end=a_end,
            b_type=b_type, b_start=b_start, b_end=b_end,
        )
        return _make_rl_result(
            question, verdict, 'rl_segment_mean_compare', answer_text, self.seq_len,
        )

    # --- R22: Max in trend type (cross-category: C + E) ---

    def generate_max_in_trend_type(self) -> Optional[Dict[str, Any]]:
        if len(self.trend_list) < self.MIN_SEGMENTS:
            return None

        max_pos = int(np.argmax(self.ts))
        max_val = round(float(self.ts[max_pos]), 2)

        result = self._segment_at_position(max_pos)
        if result is None:
            return None
        seg_idx, seg_type, start, end = result

        trend_type = random.choice(ALL_TREND_TYPES)
        verdict = "yes" if seg_type == trend_type else "no"

        question = random.choice(MAX_IN_TREND_TYPE_QUESTIONS).format(
            metric=self.metric, trend_type=trend_type,
        )
        answer_text = random.choice(MAX_IN_TREND_TYPE_ANSWERS[verdict]).format(
            metric=self.metric,
            max_val=_fmt(max_val), max_pos=max_pos,
            seg_idx=seg_idx, seg_type=seg_type,
            trend_type=trend_type, start=start, end=end,
        )
        return _make_rl_result(
            question, verdict, 'rl_max_in_trend_type', answer_text, self.seq_len,
        )

    # --- Generate all ---

    def generate_all(self) -> List[Dict[str, Any]]:
        results = []
        generators = [
            self.generate_segment_count,
            self.generate_segment_type_at_pos,
            self.generate_segment_duration,
            self.generate_dominant_trend_type,
            self.generate_longest_segment,
            self.generate_type_of_longest,
            self.generate_type_duration_fraction,
            self.generate_segment_mean_compare,
            self.generate_max_in_trend_type,
        ]
        for gen_fn in generators:
            try:
                r = gen_fn()
                if r is not None:
                    results.append(r)
            except Exception:
                logger.warning(f"RL {gen_fn.__name__} failed", exc_info=True)
        return results
