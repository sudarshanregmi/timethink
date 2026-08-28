"""Periodicity OOD evaluation QA generators.

Novel compositions involving periodicity that RL never rewards.

Why each is OOD:
    - cycle_mean_trend: RL knows cycle count (R31) and trend (E),
      but never asks whether the mean shifts across cycles
    - amplitude_vs_segment_std: RL compares amplitude to global std (R33),
      but never to the std of a specific trend segment
"""

import random
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from loguru import logger

from synth.align.generators.atomic_qa import _fmt
from synth.align.generators.compositional_qa import (
    _coin_flip_keep, _interval_mean, _make_rl_result, _pick_balanced_param,
)

# ============================================================
# Question templates
# ============================================================

CYCLE_MEAN_TREND_QUESTIONS = [
    "Does the mean of {metric} shift across cycles? Compare the mean of the first half to the mean of the second half — is the difference greater than the seasonal amplitude ({amplitude})?",
    "For {metric}, is the mean shift between the first and second halves larger than the periodic amplitude ({amplitude})?",
    "Check whether {metric}'s overall mean trends across cycles: is |mean(first half) - mean(second half)| > amplitude ({amplitude})?",
    "Does {metric} show a long-term mean shift larger than its periodic amplitude ({amplitude})?",
]

AMPLITUDE_VS_SEGMENT_STD_QUESTIONS = [
    "Is the seasonal amplitude of {metric} greater than {m}x the standard deviation of the longest trend segment?",
    "For {metric}, does the periodic amplitude exceed {m}x the std within the longest trend segment?",
    "Compare {metric}'s periodic amplitude to {m}x the std of its longest segment: is the amplitude larger?",
    "Check if the seasonal amplitude of {metric} exceeds {m}x the standard deviation computed over the longest trend segment.",
]

# ============================================================
# Answer templates
# ============================================================

CYCLE_MEAN_TREND_ANSWERS = {
    'yes': [
        "Yes, {metric} shows a significant mean shift across cycles.",
        "The mean of {metric} shifts significantly across cycles.",
    ],
    'no': [
        "No, {metric}'s mean does not shift significantly across cycles.",
        "The mean shift is within the periodic amplitude for {metric}.",
    ],
}

AMPLITUDE_VS_SEGMENT_STD_ANSWERS = {
    'yes': [
        "Yes, the seasonal amplitude exceeds {m}x the std of the longest segment in {metric}.",
        "The periodic amplitude of {metric} is greater than {m}x the longest segment's std.",
    ],
    'no': [
        "No, the seasonal amplitude does not exceed {m}x the std of the longest segment in {metric}.",
        "The periodic amplitude of {metric} is not greater than {m}x the longest segment's std.",
    ],
}


# ============================================================
# Generator
# ============================================================

class OODPeriodicGenerator:
    """Periodicity OOD compositions — eval only."""

    def __init__(
        self,
        timeseries: np.ndarray,
        metric: str,
        seq_len: int,
        seasonal: Dict,
        trend_list: Optional[List[Tuple[str, int, int]]] = None,
    ):
        self.ts = np.asarray(timeseries, dtype=float)
        self.metric = metric
        self.seq_len = seq_len
        self.seasonal = seasonal or {}
        self.trend_list = trend_list or []
        self.period = self.seasonal.get('period', 0)
        self.amplitude = round(self.seasonal.get('amplitude', 0), 2)
        self.has_period = self.period > 0 and self.seasonal.get('frequency_type') != 'no periodicity'

    # --- O17: Cycle mean trend ---

    def generate_cycle_mean_trend(self) -> Optional[Dict[str, Any]]:
        if not self.has_period or self.seq_len < 32:
            return None

        mid = self.seq_len // 2
        h1_mean = _interval_mean(self.ts, 0, mid - 1)
        h2_mean = _interval_mean(self.ts, mid, self.seq_len - 1)
        diff = round(abs(h1_mean - h2_mean), 2)

        verdict = "yes" if diff > self.amplitude else "no"

        if not _coin_flip_keep(verdict, ['yes', 'no'], 'ood_cycle_mean_trend'):
            return None

        answer_text = random.choice(CYCLE_MEAN_TREND_ANSWERS[verdict]).format(
            metric=self.metric,
            h1_mean=_fmt(h1_mean), h2_mean=_fmt(h2_mean),
            diff=_fmt(diff), amplitude=_fmt(self.amplitude),
        )
        return _make_rl_result(
            random.choice(CYCLE_MEAN_TREND_QUESTIONS).format(
                metric=self.metric, amplitude=_fmt(self.amplitude),
            ),
            verdict, 'ood_cycle_mean_trend', answer_text, self.seq_len,
        )

    # --- O18: Amplitude vs longest segment std ---

    def generate_amplitude_vs_segment_std(self) -> Optional[Dict[str, Any]]:
        if not self.has_period or len(self.trend_list) < 2:
            return None

        durations = [(e - s, i) for i, (_, s, e) in enumerate(self.trend_list)]
        max_dur, max_idx = max(durations)
        seg_type, seg_start, seg_end = self.trend_list[max_idx]

        if max_dur < 4:
            return None

        seg_std = round(float(np.std(self.ts[seg_start:seg_end + 1])), 2)
        # Pick multiplier m via _pick_balanced_param to guarantee 50/50:
        # given a fixed candidate set {1..5}, the coin-flip chooses a verdict
        # first and picks an m achieving it. Returns None if the amplitude/std
        # ratio is too extreme to produce both verdicts across {1..5}.
        def verdict_for(m):
            return "yes" if self.amplitude > round(m * seg_std, 2) else "no"

        picked = _pick_balanced_param(verdict_for, [1, 2, 3, 4, 5])
        if picked is None:
            return None
        m, verdict = picked
        threshold = round(m * seg_std, 2)

        answer_text = random.choice(AMPLITUDE_VS_SEGMENT_STD_ANSWERS[verdict]).format(
            metric=self.metric,
            amplitude=_fmt(self.amplitude), seg_std=_fmt(seg_std),
            seg_type=seg_type, seg_start=seg_start, seg_end=seg_end,
            m=m, threshold=_fmt(threshold),
        )
        return _make_rl_result(
            random.choice(AMPLITUDE_VS_SEGMENT_STD_QUESTIONS).format(
                metric=self.metric, m=m,
            ),
            verdict, 'ood_amplitude_vs_segment_std', answer_text, self.seq_len,
        )

    # --- Generate all ---

    def generate_all(self) -> List[Dict[str, Any]]:
        results = []
        generators = [
            self.generate_cycle_mean_trend,
            self.generate_amplitude_vs_segment_std,
        ]
        for gen_fn in generators:
            try:
                r = gen_fn()
                if r is not None:
                    results.append(r)
            except Exception:
                logger.warning(f"OOD {gen_fn.__name__} failed", exc_info=True)
        return results
