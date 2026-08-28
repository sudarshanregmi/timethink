"""OOD evaluation QA generators (mean category).

Novel compositions that RL never explicitly rewards.
These go ONLY to the test split — never seen during training.

Why each is OOD:
    - symmetric_recovery: RL checks ONE pair of intervals, never TWO simultaneous
    - max_mean_chunk_pos: RL checks ordering/stability, never asks "which chunk is max"
    - chunk_above_proportion: RL checks stability (all within T), never counts/proportions
    - quarter_mean_ordering: RL checks monotonicity of chunk-16 granularity, never quarters
"""

import random
from typing import Any, Dict, List, Optional

import numpy as np
from loguru import logger

from synth.align.generators.atomic_qa import _compute_chunks, _fmt, CHUNK_SIZE
from synth.align.generators.compositional_qa import (
    _coin_flip_keep, _interval_mean, _make_rl_result,
)

# ============================================================
# Question templates
# ============================================================

SYMMETRIC_RECOVERY_QUESTIONS = [
    "Does {metric} show symmetric recovery? Symmetric recovery means BOTH |mean(Q1) - mean(Q4)| < {threshold} AND |mean(Q2) - mean(Q3)| < {threshold}, where Q1-Q4 are the four quarters.",
    "Check whether {metric} exhibits symmetric recovery: the absolute difference between the first and last quarter means AND between the second and third quarter means must both be below {threshold}.",
    "For {metric}, is there symmetric recovery? That requires |mean(first quarter) - mean(last quarter)| < {threshold} and |mean(second quarter) - mean(third quarter)| < {threshold} simultaneously.",
    "Assess symmetric recovery for {metric}. Both the outer pair (Q1 vs Q4) and inner pair (Q2 vs Q3) must have mean differences below {threshold}.",
    "Does the {metric} timeseries exhibit symmetric recovery with threshold {threshold}? Both |mean(Q1)-mean(Q4)| and |mean(Q2)-mean(Q3)| must be under {threshold}.",
]

MAX_MEAN_CHUNK_POS_QUESTIONS = [
    "Which chunk of 16 in the {metric} timeseries has the highest mean? Report the 0-indexed chunk number.",
    "For {metric}, find the chunk (size 16) with the largest mean and report its 0-indexed position.",
    "Identify which 16-point chunk of {metric} has the greatest average value. Give the chunk index (0-indexed).",
    "Among all 16-point chunks of {metric}, which chunk number (0-indexed) has the highest mean?",
    "What is the index (0-based) of the chunk with the maximum mean in the {metric} timeseries (chunk size 16)?",
]

CHUNK_ABOVE_PROPORTION_QUESTIONS = [
    "What fraction of the 16-point chunks of {metric} have a mean above the global mean? Report as a decimal rounded to 2 places.",
    "For {metric}, what proportion of chunk means (chunk size 16) exceed the overall series mean?",
    "Compute the fraction of 16-point chunks in {metric} whose mean is above the global mean of the series.",
    "What share of the 16-point chunk means of {metric} are above the timeseries average? Express as a decimal.",
    "For the {metric} data, what fraction of 16-point chunks have a mean exceeding the global average?",
]

QUARTER_MEAN_ORDERING_QUESTIONS = [
    "Do the quarter means of {metric} strictly increase from the first to the last quarter?",
    "For {metric}, are the means of the four quarters in strictly increasing order (Q1 < Q2 < Q3 < Q4)?",
    "Check whether the four quarter means of {metric} form a strictly increasing sequence.",
    "Is it true that mean(Q1) < mean(Q2) < mean(Q3) < mean(Q4) for {metric}?",
    "Does the {metric} timeseries have strictly increasing quarter means from first to last?",
]

# ============================================================
# Answer templates
# ============================================================

SYMMETRIC_RECOVERY_ANSWERS = {
    'yes': [
        "Yes, {metric} exhibits symmetric recovery.",
        "Symmetric recovery is confirmed for {metric}.",
        "{metric} shows symmetric recovery.",
    ],
    'no': [
        "No, {metric} does not exhibit symmetric recovery.",
        "Symmetric recovery is not present for {metric}.",
        "{metric} does not show symmetric recovery.",
    ],
}

MAX_MEAN_CHUNK_POS_ANSWERS = [
    "The chunk with the highest mean in {metric} is chunk {pos}.",
    "For {metric}, chunk {pos} has the maximum mean.",
    "Chunk {pos} of {metric} has the highest mean.",
]

CHUNK_ABOVE_PROPORTION_ANSWERS = [
    "The proportion of chunks above the global mean in {metric} is {proportion}.",
    "For {metric}, the fraction of chunks exceeding the global mean is {proportion}.",
    "In {metric}, {proportion} of the chunks have a mean above the global average.",
]

QUARTER_MEAN_ORDERING_ANSWERS = {
    'yes': [
        "Yes, the quarter means of {metric} are strictly increasing.",
        "The quarter means of {metric} form a strictly increasing sequence.",
        "Confirmed, the quarter means increase monotonically for {metric}.",
    ],
    'no': [
        "No, the quarter means of {metric} are not strictly increasing.",
        "The quarter means of {metric} do not form a strictly increasing sequence.",
        "The quarter means of {metric} are not strictly increasing.",
    ],
}


# ============================================================
# Generator
# ============================================================

class OODMeanGenerator:
    """Generates OOD evaluation QA pairs for the mean category.

    Each task composes mean atoms in ways RL never explicitly rewards.
    Output format matches RL (minimal think block + rich answer).

    Tasks:
        O1: symmetric_recovery     — both outer AND inner quarter pairs within T
        O2: max_mean_chunk_pos     — argmax over chunk means
        O3: chunk_above_proportion — fraction of chunks above global mean
        O4: quarter_mean_ordering  — strict Q1 < Q2 < Q3 < Q4
    """

    MIN_SEQ_LEN = 64

    def __init__(self, timeseries: np.ndarray, metric: str, seq_len: int):
        self.ts = np.asarray(timeseries, dtype=float)
        self.metric = metric
        self.seq_len = seq_len

    def _quarter_boundaries(self):
        """Return (start, end) inclusive for each quarter."""
        q = self.seq_len // 4
        return [
            (0, q - 1),
            (q, 2 * q - 1),
            (2 * q, 3 * q - 1),
            (3 * q, self.seq_len - 1),
        ]

    # ----------------------------------------------------------
    # O1: Symmetric recovery
    # ----------------------------------------------------------

    def generate_symmetric_recovery(self) -> Optional[Dict[str, Any]]:
        """O1: Both |mean(Q1)-mean(Q4)| < T AND |mean(Q2)-mean(Q3)| < T.
        OOD because RL only checks ONE pair, never two simultaneously.
        """
        if self.seq_len < self.MIN_SEQ_LEN:
            return None

        quarters = self._quarter_boundaries()
        q_means = [_interval_mean(self.ts, s, e) for s, e in quarters]
        outer_diff = round(abs(q_means[0] - q_means[3]), 2)
        inner_diff = round(abs(q_means[1] - q_means[2]), 2)

        # Threshold must make both conditions jointly true or false
        max_diff = max(outer_diff, inner_diff)

        # Coin flip for desired verdict
        want_yes = random.random() < 0.5
        margin = max(0.5, max_diff * 0.3)

        if want_yes:
            # T must be above BOTH diffs
            threshold = round(max_diff + random.uniform(0.2, 1.0) * margin, 2)
        else:
            # T must be below at least one diff
            threshold = round(max(0.01, max_diff - random.uniform(0.2, 1.0) * margin), 2)

        is_met = outer_diff < threshold and inner_diff < threshold
        verdict = "yes" if is_met else "no"

        question = random.choice(SYMMETRIC_RECOVERY_QUESTIONS).format(
            metric=self.metric, threshold=_fmt(threshold),
        )
        answer_text = random.choice(SYMMETRIC_RECOVERY_ANSWERS[verdict]).format(
            metric=self.metric,
            q1m=_fmt(q_means[0]), q2m=_fmt(q_means[1]),
            q3m=_fmt(q_means[2]), q4m=_fmt(q_means[3]),
            outer_diff=_fmt(outer_diff), inner_diff=_fmt(inner_diff),
            threshold=_fmt(threshold),
        )

        return _make_rl_result(
            question, verdict, 'ood_symmetric_recovery', answer_text, self.seq_len,
        )

    # ----------------------------------------------------------
    # O2: Max-mean chunk position
    # ----------------------------------------------------------

    def generate_max_mean_chunk_pos(self) -> Optional[Dict[str, Any]]:
        """O2: Which chunk has the highest mean? → int (0-indexed).
        OOD because RL checks ordering/stability but never asks argmax.
        """
        if self.seq_len < 48:
            return None

        chunks = _compute_chunks(self.ts, 0, self.seq_len - 1)
        chunk_means = [cm for _, _, cm, _ in chunks]
        pos = int(np.argmax(chunk_means))
        max_mean = chunk_means[pos]

        chunk_means_str = "[" + ", ".join(_fmt(m) for m in chunk_means) + "]"

        question = random.choice(MAX_MEAN_CHUNK_POS_QUESTIONS).format(
            metric=self.metric,
        )
        answer_text = random.choice(MAX_MEAN_CHUNK_POS_ANSWERS).format(
            metric=self.metric, pos=pos,
            max_mean=_fmt(max_mean), chunk_means_str=chunk_means_str,
        )

        return _make_rl_result(
            question, pos, 'ood_max_mean_chunk_pos', answer_text, self.seq_len,
        )

    # ----------------------------------------------------------
    # O3: Chunk above proportion
    # ----------------------------------------------------------

    def generate_chunk_above_proportion(self) -> Optional[Dict[str, Any]]:
        """O3: Fraction of chunks with mean above global mean → float.
        OOD because RL checks stability (all within T) but never counts/proportions.
        """
        if self.seq_len < 48:
            return None

        chunks = _compute_chunks(self.ts, 0, self.seq_len - 1)
        chunk_means = [cm for _, _, cm, _ in chunks]
        global_mean = round(float(np.mean(chunk_means)), 2)

        count = sum(1 for m in chunk_means if m > global_mean)
        total = len(chunk_means)
        proportion = round(count / total, 2)

        question = random.choice(CHUNK_ABOVE_PROPORTION_QUESTIONS).format(
            metric=self.metric,
        )
        answer_text = random.choice(CHUNK_ABOVE_PROPORTION_ANSWERS).format(
            metric=self.metric,
            count=count, total=total,
            global_mean=_fmt(global_mean), proportion=_fmt(proportion),
        )

        return _make_rl_result(
            question, _fmt(proportion), 'ood_chunk_above_proportion',
            answer_text, self.seq_len,
        )

    # ----------------------------------------------------------
    # O4: Quarter mean ordering
    # ----------------------------------------------------------

    def generate_quarter_mean_ordering(self) -> Optional[Dict[str, Any]]:
        """O4: Q1 < Q2 < Q3 < Q4? → yes/no.
        OOD because RL checks monotonicity at chunk-16 granularity,
        never at quarter granularity.
        """
        if self.seq_len < self.MIN_SEQ_LEN:
            return None

        quarters = self._quarter_boundaries()
        q_means = [_interval_mean(self.ts, s, e) for s, e in quarters]

        is_increasing = all(a < b for a, b in zip(q_means, q_means[1:]))
        verdict = "yes" if is_increasing else "no"
        # Natural rate is ~83% "no" (strict monotonic increase is rare in
        # random series). Coin-flip over {yes, no} to force 50/50 emitted.
        if not _coin_flip_keep(verdict, ['yes', 'no'], 'ood_quarter_mean_ordering'):
            return None

        question = random.choice(QUARTER_MEAN_ORDERING_QUESTIONS).format(
            metric=self.metric,
        )
        answer_text = random.choice(QUARTER_MEAN_ORDERING_ANSWERS[verdict]).format(
            metric=self.metric,
            q1m=_fmt(q_means[0]), q2m=_fmt(q_means[1]),
            q3m=_fmt(q_means[2]), q4m=_fmt(q_means[3]),
        )

        return _make_rl_result(
            question, verdict, 'ood_quarter_mean_ordering',
            answer_text, self.seq_len,
        )

    # ----------------------------------------------------------
    # Generate all
    # ----------------------------------------------------------

    def generate_all(self) -> List[Dict[str, Any]]:
        """Generate all OOD mean QAs for this timeseries."""
        results = []
        generators = [
            self.generate_symmetric_recovery,
            self.generate_max_mean_chunk_pos,
            self.generate_chunk_above_proportion,
            self.generate_quarter_mean_ordering,
        ]
        for gen_fn in generators:
            try:
                r = gen_fn()
                if r is not None:
                    results.append(r)
            except Exception:
                logger.warning(f"OOD {gen_fn.__name__} failed", exc_info=True)
        return results
