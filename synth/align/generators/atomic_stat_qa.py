"""Atomic stat QA generators: std, extrema, percentiles.

These are simpler atoms than mean — direct reads from the model's
representation, not multi-step chunk decompositions. The think blocks
are honest about this: shorter reasoning, consistent answer format.

Atoms:
    B1: Global std
    B2: Interval std [x, y]
    C1: Extremum value (min or max)
    C2: Extremum position (min or max), 0-indexed
    D1: Percentile value at k
"""

import random
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from loguru import logger

from synth.align.generators.atomic_qa import _fmt, _compute_chunks

# ============================================================
# Shared openers (reuse across all atoms)
# ============================================================

_OPENERS = [
    "Let me find this.",
    "I'll work through this step by step.",
    "Let me analyze the data.",
    "I need to compute this from the timeseries.",
    "Let me determine this.",
    "I'll examine the data to find this.",
    "Let me check the timeseries.",
    "I need to extract this from the data.",
    "Let me look at the timeseries carefully.",
    "I'll compute this from the data.",
]

# ============================================================
# B1: Global std
# ============================================================

B1_QUESTIONS = [
    "What is the standard deviation of the {metric} timeseries?",
    "Compute the standard deviation of {metric}.",
    "Find the standard deviation of {metric} across the entire series.",
    "What is the overall standard deviation of {metric}?",
    "Determine the standard deviation of the full {metric} data.",
    "How much does {metric} vary? Report the standard deviation.",
    "Calculate the standard deviation for {metric}.",
    "What is the spread of {metric} as measured by standard deviation?",
]

B1_ANSWERS = [
    "The standard deviation of {metric} is {std}.",
    "For {metric}, the standard deviation is {std}.",
    "The {metric} data has a standard deviation of {std}.",
]

# ============================================================
# B2: Interval std — two framings share the same computation
#     (1) explicit [start, end] index range
#     (2) the Nth / first / last trend segment
# ============================================================

B2_QUESTIONS = [
    "What is the standard deviation of {metric} from index {start} to index {end}?",
    "Compute the standard deviation of {metric} over the range [{start}, {end}].",
    "Find the standard deviation of {metric} between index {start} and {end}.",
    "For {metric}, what is the standard deviation from position {start} to {end}?",
    "Calculate the standard deviation of {metric} in the interval [{start}, {end}].",
    "What is the spread of {metric} from index {start} to {end}, measured by standard deviation?",
    "Determine the standard deviation of {metric} over positions {start} through {end}.",
]

B2_ANSWERS = [
    "The standard deviation of {metric} from index {start} to {end} is {std}.",
    "For {metric} between index {start} and {end}, the standard deviation is {std}.",
    "The {metric} data from index {start} to {end} has a standard deviation of {std}.",
]

B2_SEGMENT_QUESTIONS = [
    "What is the standard deviation of the {ordinal} trend segment of {metric}?",
    "Compute the standard deviation within the {ordinal} segment of {metric}.",
    "For {metric}, find the standard deviation over the {ordinal} trend segment.",
    "What is the spread of {metric} within the {ordinal} trend segment?",
    "Calculate the standard deviation of {metric} restricted to the {ordinal} segment.",
]

B2_SEGMENT_ANSWERS = [
    "The standard deviation of the {ordinal} segment of {metric} is {std}.",
    "For the {ordinal} segment of {metric}, the standard deviation is {std}.",
    "Within the {ordinal} trend segment of {metric}, the standard deviation is {std}.",
]


def _segment_ordinal(idx: int, n_segs: int) -> str:
    """Natural-language label for a segment at 0-indexed position idx (of n_segs total)."""
    if idx == 0:
        return random.choice(["first", "initial", "1st"])
    if idx == n_segs - 1:
        return random.choice(["last", "final", f"{n_segs}th"])
    ordinals = {1: "2nd", 2: "3rd", 3: "4th", 4: "5th", 5: "6th", 6: "7th"}
    return random.choice([
        ordinals.get(idx, f"{idx + 1}th"),
        f"segment {idx + 1}",
    ])

# ============================================================
# C1: Extremum value (min or max)
# ============================================================

C1_MIN_QUESTIONS = [
    "What is the minimum value of the {metric} timeseries?",
    "Find the minimum value in {metric}.",
    "What is the lowest value in the {metric} data?",
    "Compute the minimum of {metric} across the entire series.",
    "What is the smallest value in {metric}?",
    "Determine the minimum value of {metric}.",
    "Report the minimum value of the {metric} timeseries.",
]

C1_MAX_QUESTIONS = [
    "What is the maximum value of the {metric} timeseries?",
    "Find the maximum value in {metric}.",
    "What is the highest value in the {metric} data?",
    "Compute the maximum of {metric} across the entire series.",
    "What is the largest value in {metric}?",
    "Determine the maximum value of {metric}.",
    "Report the maximum value of the {metric} timeseries.",
]

C1_MIN_ANSWERS = [
    "The minimum value of {metric} is {value}.",
    "For {metric}, the lowest value is {value}.",
    "The smallest value in {metric} is {value}.",
]

C1_MAX_ANSWERS = [
    "The maximum value of {metric} is {value}.",
    "For {metric}, the highest value is {value}.",
    "The largest value in {metric} is {value}.",
]

# ============================================================
# C2: Extremum position (min or max), 0-indexed
# ============================================================

C2_MIN_QUESTIONS = [
    "At what index does the minimum value of {metric} occur? Report 0-indexed.",
    "Find the position of the minimum value in {metric} (0-indexed).",
    "Where is the lowest value in {metric}? Give the 0-indexed position.",
    "What is the 0-indexed position of the minimum in {metric}?",
    "At which index does {metric} reach its minimum?",
    "Locate the minimum value of {metric} and report its index (0-based).",
]

C2_MAX_QUESTIONS = [
    "At what index does the maximum value of {metric} occur? Report 0-indexed.",
    "Find the position of the maximum value in {metric} (0-indexed).",
    "Where is the highest value in {metric}? Give the 0-indexed position.",
    "What is the 0-indexed position of the maximum in {metric}?",
    "At which index does {metric} reach its maximum?",
    "Locate the maximum value of {metric} and report its index (0-based).",
]

C2_MIN_ANSWERS = [
    "The minimum of {metric} occurs at index {pos}.",
    "For {metric}, the lowest value is at position {pos}.",
    "The minimum of {metric} is at index {pos}.",
]

C2_MAX_ANSWERS = [
    "The maximum of {metric} occurs at index {pos}.",
    "For {metric}, the highest value is at position {pos}.",
    "The maximum of {metric} is at index {pos}.",
]

# ============================================================
# D1: Percentile value
# ============================================================

D1_QUESTIONS = [
    "What is the {k}th percentile value of {metric}?",
    "Compute the {k}th percentile of the {metric} timeseries.",
    "Find the {k}th percentile value in {metric}.",
    "For {metric}, what value corresponds to the {k}th percentile?",
    "Determine the {k}th percentile of {metric}.",
    "What is the value at the {k}th percentile of {metric}?",
    "Report the {k}th percentile value for {metric}.",
    "Calculate the {k}th percentile of the {metric} data.",
]

D1_ANSWERS = [
    "The {k}th percentile of {metric} is {value}.",
    "For {metric}, the {k}th percentile value is {value}.",
    "The value at the {k}th percentile of {metric} is {value}.",
]

PERCENTILE_KS = [10, 25, 50, 75, 95]


# ============================================================
# Think block builder
# ============================================================

def _build_simple_think(lines: List[str], reward_answer: str) -> Tuple[str, str]:
    """Build a think block from a list of reasoning lines.

    Returns (full_think_str_with_tags, reward_answer).
    """
    opener = random.choice(_OPENERS)
    parts = [opener] + lines + [f"answer: {reward_answer}"]
    content = "\n".join(parts)
    return f"<think>\n{content}\n</think>", reward_answer


# ============================================================
# Generator
# ============================================================

class AtomicStatGenerator:
    """Generates atomic stat QA pairs: std, extrema, percentiles.

    These are direct reads from the model's representation —
    simpler think blocks than the chunk-based mean atoms.
    """

    def __init__(
        self, timeseries: np.ndarray, metric: str, seq_len: int,
        trend_list: Optional[List[Tuple[str, int, int]]] = None,
    ):
        self.ts = np.asarray(timeseries, dtype=float)
        self.metric = metric
        self.seq_len = seq_len
        self.trend_list = trend_list or []

    def _interval_mean_chunked(self, start: int, end: int) -> float:
        """Mean via chunk-and-average for consistency with SFT mean atoms."""
        chunks = _compute_chunks(self.ts, start, end)
        return round(float(np.mean([cm for _, _, cm, _ in chunks])), 2)

    # --- B1: Global std ---

    def generate_global_std(self) -> Optional[Dict[str, Any]]:
        # Std via naive np.std — correct at any length. The think block shows
        # a single scalar step (no chunk decomposition, since chunk std would
        # require within+between variance). Works at any seq_len.
        std_val = round(float(np.std(self.ts)), 2)
        mean_val = self._interval_mean_chunked(0, self.seq_len - 1)

        think, ans = _build_simple_think([
            f"The {self.metric} timeseries has {self.seq_len} data points.",
            f"The mean is {_fmt(mean_val)}.",
            f"The standard deviation is {_fmt(std_val)}.",
        ], _fmt(std_val))

        answer_text = random.choice(B1_ANSWERS).format(
            metric=self.metric, std=_fmt(std_val),
            mean=_fmt(mean_val), length=self.seq_len,
        )
        return {
            'question': random.choice(B1_QUESTIONS).format(metric=self.metric),
            'answer': f"{think}\n{answer_text}",
            'eval_type': 'atomic_global_std',
            'eval_metadata': {'length': self.seq_len, 'verdict': ans},
        }

    # --- B2: Interval std (supports [start, end], Nth-segment, and union framings) ---

    def generate_interval_std(self) -> Optional[Dict[str, Any]]:
        if self.seq_len < 32:
            return None

        # Union framing: 25% when seq_len supports two disjoint 32-point
        # intervals with a 16-point gap. Plants stat-over-union primitive for
        # OOD conditional_stat transfer.
        if self.seq_len >= 80 and random.random() < 0.25:
            return self._interval_std_union()

        # Segment-framed branch: fires 50% when trend_list has ≥2 usable segments.
        usable_segments = [
            (t, s, e) for (t, s, e) in self.trend_list if (e - s + 1) >= 16
        ]
        if len(usable_segments) >= 2 and random.random() < 0.5:
            return self._interval_std_segment(usable_segments)

        # Explicit [start, end] framing (original behaviour).
        range_len = random.randint(32, min(256, self.seq_len))
        start = random.randint(0, self.seq_len - range_len)
        end = start + range_len - 1
        n = end - start + 1

        subset = self.ts[start:end + 1]
        std_val = round(float(np.std(subset)), 2)
        mean_val = self._interval_mean_chunked(start, end)

        think, ans = _build_simple_think([
            f"Looking at {self.metric} from index {start} to {end} ({n} data points).",
            f"The mean of this range is {_fmt(mean_val)}.",
            f"The standard deviation of this range is {_fmt(std_val)}.",
        ], _fmt(std_val))

        answer_text = random.choice(B2_ANSWERS).format(
            metric=self.metric, std=_fmt(std_val),
            mean=_fmt(mean_val), start=start, end=end, n=n,
        )
        return {
            'question': random.choice(B2_QUESTIONS).format(
                metric=self.metric, start=start, end=end,
            ),
            'answer': f"{think}\n{answer_text}",
            'eval_type': 'atomic_interval_std',
            'eval_metadata': {
                'length': self.seq_len, 'verdict': ans,
                'start': start, 'end': end, 'framing': 'interval',
            },
        }

    def _interval_std_segment(
        self, usable_segments: List[Tuple[str, int, int]],
    ) -> Optional[Dict[str, Any]]:
        """Nth-segment framing for interval std. Underlying computation is
        identical to the explicit-interval branch; only the question and the
        think block's first line change (look up segment → use its [s, e]).
        """
        # Pick one segment uniformly at random
        idx_in_usable = random.randint(0, len(usable_segments) - 1)
        seg_type, start, end = usable_segments[idx_in_usable]

        # Map back to index in the full trend_list (for ordinal labeling)
        full_idx = self.trend_list.index((seg_type, start, end))
        n_segs = len(self.trend_list)
        ordinal = _segment_ordinal(full_idx, n_segs)

        n = end - start + 1
        subset = self.ts[start:end + 1]
        std_val = round(float(np.std(subset)), 2)
        mean_val = self._interval_mean_chunked(start, end)

        seg_list_str = ", ".join(
            f"{t} [{s}, {e}]" for (t, s, e) in self.trend_list
        )

        think, ans = _build_simple_think([
            f"The trend segments of {self.metric} are: {seg_list_str}.",
            f"The {ordinal} segment is {seg_type} [{start}, {end}] ({n} data points).",
            f"The mean of this range is {_fmt(mean_val)}.",
            f"The standard deviation of this range is {_fmt(std_val)}.",
        ], _fmt(std_val))

        answer_text = random.choice(B2_SEGMENT_ANSWERS).format(
            metric=self.metric, ordinal=ordinal, std=_fmt(std_val),
        )
        return {
            'question': random.choice(B2_SEGMENT_QUESTIONS).format(
                metric=self.metric, ordinal=ordinal,
            ),
            'answer': f"{think}\n{answer_text}",
            'eval_type': 'atomic_interval_std',
            'eval_metadata': {
                'length': self.seq_len, 'verdict': ans,
                'start': start, 'end': end, 'framing': 'segment',
                'segment_index': full_idx, 'segment_type': seg_type,
            },
        }

    def _interval_std_union(self) -> Optional[Dict[str, Any]]:
        """Union-of-intervals framing for interval std. Computation:
        concatenate data from disjoint intervals, then np.std over the union.
        Plants the stat-over-union primitive for OOD conditional_stat transfer.
        """
        n_intervals = random.choice([2, 3])
        per_len = 32
        gap = 16
        needed = n_intervals * per_len + (n_intervals - 1) * gap
        if self.seq_len < needed:
            return None

        ranges: List[Tuple[int, int]] = []
        cursor = random.randint(0, self.seq_len - needed)
        for _ in range(n_intervals):
            s = cursor
            e = s + per_len - 1
            ranges.append((s, e))
            cursor = e + 1 + gap

        pieces = [self.ts[s:e + 1] for s, e in ranges]
        combined = np.concatenate(pieces)
        std_val = round(float(np.std(combined)), 2)
        total_n = sum(e - s + 1 for s, e in ranges)

        pretty = " and ".join(f"[{s}, {e}]" for s, e in ranges)
        counts_str = " + ".join(str(e - s + 1) for s, e in ranges)

        think, ans = _build_simple_think([
            f"Looking at {self.metric} over the union of intervals {pretty}.",
            f"Points per interval: {counts_str} = {total_n} total.",
            f"Concatenating the values from each interval into one set of {total_n} points.",
            f"The standard deviation of the combined values is {_fmt(std_val)}.",
        ], _fmt(std_val))

        answer_text = (
            f"The standard deviation of {self.metric} over the union of "
            f"intervals {pretty} is {_fmt(std_val)}."
        )
        return {
            'question': (
                f"What is the standard deviation of {self.metric} over the "
                f"union of intervals {pretty}?"
            ),
            'answer': f"{think}\n{answer_text}",
            'eval_type': 'atomic_interval_std',
            'eval_metadata': {
                'length': self.seq_len, 'verdict': ans,
                'framing': 'union', 'ranges': list(ranges),
            },
        }

    # --- C1: Extremum value ---

    def generate_extremum_value(self) -> Dict[str, Any]:
        direction = random.choice(['min', 'max'])
        if direction == 'min':
            value = round(float(np.min(self.ts)), 2)
            pos = int(np.argmin(self.ts))
            questions = C1_MIN_QUESTIONS
            answers = C1_MIN_ANSWERS
            eval_type = 'atomic_min_value'
        else:
            value = round(float(np.max(self.ts)), 2)
            pos = int(np.argmax(self.ts))
            questions = C1_MAX_QUESTIONS
            answers = C1_MAX_ANSWERS
            eval_type = 'atomic_max_value'

        think, ans = _build_simple_think([
            f"Looking at the {self.metric} timeseries ({self.seq_len} data points).",
            f"The {direction}imum value is {_fmt(value)} at index {pos}.",
        ], _fmt(value))

        answer_text = random.choice(answers).format(
            metric=self.metric, value=_fmt(value),
            pos=pos, length=self.seq_len,
        )
        return {
            'question': random.choice(questions).format(metric=self.metric),
            'answer': f"{think}\n{answer_text}",
            'eval_type': eval_type,
            'eval_metadata': {'length': self.seq_len, 'verdict': ans, 'direction': direction},
        }

    # --- C2: Extremum position ---

    def generate_extremum_position(self) -> Dict[str, Any]:
        direction = random.choice(['min', 'max'])
        if direction == 'min':
            pos = int(np.argmin(self.ts))
            value = round(float(self.ts[pos]), 2)
            questions = C2_MIN_QUESTIONS
            answers = C2_MIN_ANSWERS
            eval_type = 'atomic_min_position'
        else:
            pos = int(np.argmax(self.ts))
            value = round(float(self.ts[pos]), 2)
            questions = C2_MAX_QUESTIONS
            answers = C2_MAX_ANSWERS
            eval_type = 'atomic_max_position'

        think, ans = _build_simple_think([
            f"Looking at the {self.metric} timeseries ({self.seq_len} data points).",
            f"The {direction}imum value of {_fmt(value)} occurs at index {pos}.",
        ], str(pos))

        answer_text = random.choice(answers).format(
            metric=self.metric, value=_fmt(value), pos=pos,
        )
        return {
            'question': random.choice(questions).format(metric=self.metric),
            'answer': f"{think}\n{answer_text}",
            'eval_type': eval_type,
            'eval_metadata': {'length': self.seq_len, 'verdict': ans, 'direction': direction},
        }

    # --- D1: Percentile value ---

    def generate_percentile(self) -> Dict[str, Any]:
        k = random.choice(PERCENTILE_KS)
        value = round(float(np.percentile(self.ts, k)), 2)

        think, ans = _build_simple_think([
            f"The {self.metric} timeseries has {self.seq_len} data points.",
            f"The {k}th percentile value is {_fmt(value)}.",
        ], _fmt(value))

        answer_text = random.choice(D1_ANSWERS).format(
            metric=self.metric, k=k, value=_fmt(value), length=self.seq_len,
        )
        return {
            'question': random.choice(D1_QUESTIONS).format(metric=self.metric, k=k),
            'answer': f"{think}\n{answer_text}",
            'eval_type': 'atomic_percentile',
            'eval_metadata': {'length': self.seq_len, 'verdict': ans, 'k': k},
        }

    # --- Generate all ---

    def generate_all(self) -> List[Dict[str, Any]]:
        results = []
        generators = [
            self.generate_global_std,
            self.generate_interval_std,
            self.generate_extremum_value,
            self.generate_extremum_position,
            self.generate_percentile,
        ]
        for gen_fn in generators:
            try:
                r = gen_fn()
                if r is not None:
                    results.append(r)
            except Exception:
                logger.warning(f"Atom {gen_fn.__name__} failed", exc_info=True)
        return results
