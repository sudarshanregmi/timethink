"""Atomic skill QA generators.

Generates SFT training data for irreducible timeseries computation skills.
Each atom teaches the model ONE operation with step-by-step thinking.

Current: Mean category (A1, A2, A3).
All three follow the same pattern: chunk at 16 → compute chunk means → aggregate.

Format:
    <think>
    step-by-step reasoning (diverse phrasing)
    answer: REWARD_ANSWER
    </think>
    Natural language answer.
"""

import random
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from loguru import logger

from synth.align.templates.atomic_templates import (
    GLOBAL_MEAN_QUESTIONS,
    CHUNKED_MEAN_QUESTIONS,
    INTERVAL_MEAN_QUESTIONS,
    THINK_OPENERS,
    RANGE_DESCRIPTIONS,
    CHUNK_INTROS,
    CHUNK_LINE_STYLES,
    PARTIAL_CHUNK_NOTES,
    AGGREGATION_PHRASES,
    GLOBAL_MEAN_ANSWERS,
    CHUNKED_MEAN_ANSWERS,
    INTERVAL_MEAN_ANSWERS,
)

CHUNK_SIZE = 16


def _fmt(val: float, dp: int = 2) -> str:
    """Format float to dp decimal places."""
    return f"{round(val, dp):.{dp}f}"


def _compute_chunks(
    timeseries: np.ndarray,
    start: int,
    end: int,
) -> List[Tuple[int, int, float, int]]:
    """Compute chunk means within [start, end] inclusive.

    Returns list of (chunk_start, chunk_end_inclusive, mean, chunk_length).
    """
    chunks = []
    i = start
    while i <= end:
        chunk_end = min(i + CHUNK_SIZE - 1, end)
        chunk_data = timeseries[i : chunk_end + 1]
        chunk_mean = round(float(np.mean(chunk_data)), 2)
        chunk_len = chunk_end - i + 1
        chunks.append((i, chunk_end, chunk_mean, chunk_len))
        i = chunk_end + 1
    return chunks


def _build_think_block(
    metric: str,
    start: int,
    end: int,
    chunks: List[Tuple[int, int, float, int]],
    aggregate: bool,
) -> Tuple[str, str]:
    """Build a diverse think block for mean computation.

    Args:
        metric: Metric name for the timeseries.
        start: Start index (inclusive).
        end: End index (inclusive).
        chunks: Output of _compute_chunks.
        aggregate: If True, compute and show the mean of chunk means (A1/A3).
                   If False, answer is the list of chunk means (A2).

    Returns:
        (full_think_str, reward_answer_str)
        full_think_str includes <think>...</think> wrapper.
        reward_answer_str is the formatted answer for reward parsing.
    """
    parts = []
    n = end - start + 1
    n_chunks = len(chunks)

    # 1. Opener
    parts.append(random.choice(THINK_OPENERS))

    # 2. Range description
    range_desc = random.choice(RANGE_DESCRIPTIONS).format(
        start=start, end=end, n=n, metric=metric,
    )
    parts.append(range_desc)

    # 3. Chunk intro
    chunk_intro = random.choice(CHUNK_INTROS).format(n_chunks=n_chunks)
    parts.append(chunk_intro)
    parts.append("")  # blank line

    # 4. Chunk lines — pick a random style, use it consistently
    style_name = random.choice(list(CHUNK_LINE_STYLES.keys()))
    style_fn = CHUNK_LINE_STYLES[style_name]

    for i, (cs, ce, cm, cl) in enumerate(chunks):
        if cl < CHUNK_SIZE:
            partial_note = random.choice(PARTIAL_CHUNK_NOTES).format(n=cl)
        else:
            partial_note = ""
        parts.append(style_fn(i, cs, ce, _fmt(cm), partial_note))

    parts.append("")  # blank line

    # 5. Build answer
    if aggregate:
        chunk_means = [cm for _, _, cm, _ in chunks]
        final_mean = round(float(np.mean(chunk_means)), 2)
        reward_answer = _fmt(final_mean)

        # Show the averaging computation
        means_strs = [_fmt(m) for m in chunk_means]
        sum_str = " + ".join(means_strs)
        computation = f"({sum_str}) / {n_chunks}"

        agg_phrase = random.choice(AGGREGATION_PHRASES).format(
            computation=computation,
            sum_str=sum_str,
            n=n_chunks,
            result=_fmt(final_mean),
        )
        parts.append(agg_phrase)
    else:
        means_strs = [_fmt(cm) for _, _, cm, _ in chunks]
        reward_answer = "[" + ", ".join(means_strs) + "]"

    parts.append(f"answer: {reward_answer}")

    think_content = "\n".join(parts)
    full_think = f"<think>\n{think_content}\n</think>"
    return full_think, reward_answer


class AtomicMeanGenerator:
    """Generates atomic mean QA pairs from a timeseries.

    Takes a timeseries array and metric name, produces QA dicts
    with diverse think blocks and consistent answer formats.

    Atoms:
        A1 (global_mean): Chunk full series at 16 → mean of chunk means.
        A2 (chunked_means): Chunk sub-range at 16 → list of chunk means.
        A3 (interval_mean): Chunk sub-range at 16 → mean of chunk means.
    """

    # Range limits for A2/A3. MAX_RANGE kept at 256 for sub-range picking so
    # A2/A3 stay focused on a short window; A1 (global) now runs at any
    # seq_len and may enumerate more chunks. See generate_global_mean.
    MIN_RANGE = 32   # at least 2 chunks
    MAX_RANGE = 256  # sub-range cap for A2/A3 only

    def __init__(
        self,
        timeseries: np.ndarray,
        metric: str,
        seq_len: int,
    ):
        self.ts = np.asarray(timeseries, dtype=float)
        self.metric = metric
        self.seq_len = seq_len

    def _pick_range(self) -> Optional[Tuple[int, int]]:
        """Pick a random sub-range for A2/A3.

        Returns (start, end) inclusive, or None if seq_len too short.
        Range length is snapped to nearest multiple of 16 when possible
        (avoids partial last chunk in most cases).
        """
        if self.seq_len < self.MIN_RANGE:
            return None

        max_len = min(self.MAX_RANGE, self.seq_len)
        min_len = self.MIN_RANGE

        # Pick a length, prefer multiples of 16
        raw_len = random.randint(min_len, max_len)
        # Snap to nearest multiple of 16 (but keep within bounds)
        snapped = round(raw_len / CHUNK_SIZE) * CHUNK_SIZE
        range_len = max(min_len, min(snapped, max_len, self.seq_len))

        start = random.randint(0, self.seq_len - range_len)
        end = start + range_len - 1
        return start, end

    def generate_global_mean(self) -> Optional[Dict[str, Any]]:
        """A1: Global mean of full timeseries via chunk-and-average.

        Works at any seq_len. At long seq_len the think block enumerates
        more chunks (n_chunks = seq_len / 16), which is intentional for RL
        length-generalization training.
        """
        start, end = 0, self.seq_len - 1
        chunks = _compute_chunks(self.ts, start, end)
        think_str, reward_answer = _build_think_block(
            self.metric, start, end, chunks, aggregate=True,
        )

        answer_text = random.choice(GLOBAL_MEAN_ANSWERS).format(
            metric=self.metric, value=reward_answer, length=self.seq_len,
        )

        return {
            'question': random.choice(GLOBAL_MEAN_QUESTIONS).format(
                metric=self.metric,
            ),
            'answer': f"{think_str}\n{answer_text}",
            'eval_type': 'atomic_global_mean',
            'eval_metadata': {
                'length': self.seq_len,
                'verdict': reward_answer,
            },
        }

    def generate_chunked_means(self) -> Optional[Dict[str, Any]]:
        """A2: List of chunk means over a sub-range."""
        rng = self._pick_range()
        if rng is None:
            return None
        start, end = rng

        chunks = _compute_chunks(self.ts, start, end)
        think_str, reward_answer = _build_think_block(
            self.metric, start, end, chunks, aggregate=False,
        )

        answer_text = random.choice(CHUNKED_MEAN_ANSWERS).format(
            metric=self.metric, value=reward_answer, start=start, end=end,
        )

        return {
            'question': random.choice(CHUNKED_MEAN_QUESTIONS).format(
                metric=self.metric, start=start, end=end,
            ),
            'answer': f"{think_str}\n{answer_text}",
            'eval_type': 'atomic_chunked_means',
            'eval_metadata': {
                'length': self.seq_len,
                'verdict': reward_answer,
                'start': start,
                'end': end,
            },
        }

    def _pick_disjoint_ranges(
        self, n_intervals: int,
    ) -> Optional[List[Tuple[int, int]]]:
        """Pick n_intervals disjoint sub-ranges, each snapped to CHUNK_SIZE.

        Returns sorted list of (start, end) pairs or None if infeasible.
        """
        per_len = CHUNK_SIZE * 2  # 32-point intervals (2 chunks each) for tractable think blocks
        gap = CHUNK_SIZE  # require at least one chunk of gap between intervals
        needed = n_intervals * per_len + (n_intervals - 1) * gap
        if self.seq_len < needed:
            return None

        # Split the series into n_intervals * 2 + (n_intervals - 1) "slots",
        # randomly choose starting positions ensuring no overlap.
        ranges = []
        cursor = random.randint(0, self.seq_len - needed)
        for i in range(n_intervals):
            s = cursor
            e = s + per_len - 1
            ranges.append((s, e))
            cursor = e + 1 + gap
        return ranges

    def _build_union_think_mean(
        self, ranges: List[Tuple[int, int]],
    ) -> Tuple[str, str]:
        """Think block for mean over union of disjoint intervals."""
        parts = [random.choice(THINK_OPENERS)]
        pretty = " and ".join(f"[{s}, {e}]" for s, e in ranges)
        parts.append(
            f"Looking at {self.metric} over the union of intervals {pretty}."
        )

        all_chunk_means: List[float] = []
        for s, e in ranges:
            interval_chunks = _compute_chunks(self.ts, s, e)
            parts.append(f"Chunking [{s}, {e}]: {len(interval_chunks)} chunks.")
            for (cs, ce, cm, _) in interval_chunks:
                parts.append(f"  chunk [{cs}, {ce}] mean = {_fmt(cm)}")
                all_chunk_means.append(cm)
            parts.append("")

        means_strs = [_fmt(m) for m in all_chunk_means]
        combined_list = "[" + ", ".join(means_strs) + "]"
        parts.append(f"Combined chunk means: {combined_list}")

        final_mean = round(float(np.mean(all_chunk_means)), 2)
        sum_str = " + ".join(means_strs)
        parts.append(
            f"Mean over union: ({sum_str}) / {len(all_chunk_means)} "
            f"= {_fmt(final_mean)}"
        )
        parts.append(f"answer: {_fmt(final_mean)}")
        return (
            f"<think>\n" + "\n".join(parts) + "\n</think>",
            _fmt(final_mean),
        )

    def generate_interval_mean(self) -> Optional[Dict[str, Any]]:
        """A3: Mean of a sub-range via chunk-and-average.

        Two framings share the same eval_type:
          contiguous: "mean over [start, end]" (existing)
          union:      "mean over [s1, e1] and [s2, e2]" — plants stat-over-union
                      primitive that transfers to OOD conditional_stat.
        """
        # Union framing fires 30% of the time when seq_len supports ≥2 disjoint
        # intervals of 32 points each with a 16-point gap.
        if random.random() < 0.3:
            n_intervals = random.choice([2, 3])
            ranges = self._pick_disjoint_ranges(n_intervals)
            if ranges is not None:
                think_str, reward_answer = self._build_union_think_mean(ranges)
                pretty = " and ".join(f"[{s}, {e}]" for s, e in ranges)
                question = (
                    f"What is the mean of {self.metric} over the union "
                    f"of intervals {pretty}?"
                )
                answer_text = (
                    f"The mean of {self.metric} over the union of intervals "
                    f"{pretty} is {reward_answer}."
                )
                return {
                    'question': question,
                    'answer': f"{think_str}\n{answer_text}",
                    'eval_type': 'atomic_interval_mean',
                    'eval_metadata': {
                        'length': self.seq_len,
                        'verdict': reward_answer,
                        'framing': 'union',
                        'ranges': list(ranges),
                    },
                }

        rng = self._pick_range()
        if rng is None:
            return None
        start, end = rng

        chunks = _compute_chunks(self.ts, start, end)
        think_str, reward_answer = _build_think_block(
            self.metric, start, end, chunks, aggregate=True,
        )

        answer_text = random.choice(INTERVAL_MEAN_ANSWERS).format(
            metric=self.metric, value=reward_answer, start=start, end=end,
        )

        return {
            'question': random.choice(INTERVAL_MEAN_QUESTIONS).format(
                metric=self.metric, start=start, end=end,
            ),
            'answer': f"{think_str}\n{answer_text}",
            'eval_type': 'atomic_interval_mean',
            'eval_metadata': {
                'length': self.seq_len,
                'verdict': reward_answer,
                'framing': 'contiguous',
                'start': start,
                'end': end,
            },
        }

    def generate_all(self) -> List[Dict[str, Any]]:
        """Generate all mean atomic QAs for this timeseries."""
        results = []

        # A1
        try:
            r = self.generate_global_mean()
            if r is not None:
                results.append(r)
        except Exception:
            logger.warning("A1 global_mean failed", exc_info=True)

        # A2: needs seq_len >= 32
        try:
            r = self.generate_chunked_means()
            if r is not None:
                results.append(r)
        except Exception:
            logger.warning("A2 chunked_means failed", exc_info=True)

        # A3: needs seq_len >= 32
        try:
            r = self.generate_interval_mean()
            if r is not None:
                results.append(r)
        except Exception:
            logger.warning("A3 interval_mean failed", exc_info=True)

        return results
