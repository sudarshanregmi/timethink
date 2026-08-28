"""Periodicity compositional QA generators for RL training.

Compose the periodicity description atom (G1) with reasoning
or cross with stat/trend atoms.

Tasks:
    R29: has_periodicity       — yes/no (recognize + report)
    R30: period_estimate       — what is the period? (recognize + read)
    R31: cycle_count           — how many complete cycles? (G + arithmetic)
    R33: amplitude_vs_std      — seasonal amplitude > std? (G + B, cross-cat)
    R34: amplitude_vs_range    — amplitude > range/4? (G + C, cross-cat)
"""

import random
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from loguru import logger

from synth.align.generators.atomic_qa import _fmt, _compute_chunks
from synth.align.generators.compositional_qa import _make_rl_result

# ============================================================
# Question templates
# ============================================================

HAS_PERIODICITY_QUESTIONS = [
    "Does {metric} exhibit periodic fluctuations?",
    "Is there periodicity in the {metric} timeseries?",
    "For {metric}, are periodic patterns present?",
    "Check whether {metric} shows periodic behavior.",
    "Does the {metric} data have any periodic or seasonal component?",
    "Is {metric} periodic?",
]

PERIOD_ESTIMATE_QUESTIONS = [
    "What is the approximate period of the periodic fluctuations in {metric}?",
    "Estimate the period of {metric} in time steps.",
    "For {metric}, what is the length of one complete cycle?",
    "How many time steps per cycle does {metric} exhibit?",
    "What is the period of the periodic pattern in {metric}?",
]

CYCLE_COUNT_QUESTIONS = [
    "How many complete cycles does {metric} contain?",
    "For {metric}, how many full periodic cycles fit in the timeseries?",
    "Count the approximate number of complete cycles in {metric}.",
    "What is the approximate cycle count for {metric}?",
    "How many times does {metric} complete a full periodic cycle?",
]

AMPLITUDE_VS_STD_QUESTIONS = [
    "Is the seasonal amplitude of {metric} greater than the standard deviation of the timeseries?",
    "For {metric}, does the periodic amplitude exceed the overall std?",
    "Compare the seasonal amplitude to the std of {metric}: is the amplitude larger?",
    "Is the amplitude of the periodic component in {metric} greater than its standard deviation?",
    "Check if {metric}'s seasonal amplitude exceeds its std.",
]

AMPLITUDE_VS_RANGE_QUESTIONS = [
    "Is the seasonal amplitude of {metric} greater than one quarter of the range?",
    "For {metric}, does the periodic amplitude exceed (max - min) / 4?",
    "Compare the seasonal amplitude to a quarter of the range of {metric}: is the amplitude larger?",
    "Is the periodic amplitude in {metric} greater than 25% of the full range?",
    "Check if {metric}'s seasonal amplitude exceeds range/4.",
]

# ============================================================
# Answer templates
# ============================================================

HAS_PERIODICITY_ANSWERS = {
    'yes': [
        "Yes, {metric} exhibits periodic fluctuations.",
        "{metric} shows periodic behavior.",
        "Periodic fluctuations are present in {metric}.",
    ],
    'no': [
        "No, {metric} does not exhibit periodic fluctuations.",
        "{metric} shows no periodic behavior.",
        "No periodicity is detected in {metric}.",
    ],
}

PERIOD_ESTIMATE_ANSWERS = [
    "The period of {metric} is approximately {period} time steps.",
    "For {metric}, the periodic pattern repeats every ~{period} time steps.",
    "The estimated period of {metric} is {period}.",
]

CYCLE_COUNT_ANSWERS = [
    "{metric} contains approximately {n_cycles} complete cycles.",
    "For {metric}, there are about {n_cycles} complete cycles.",
    "Approximately {n_cycles} full cycles are present in {metric}.",
]

AMPLITUDE_VS_STD_ANSWERS = {
    'yes': [
        "Yes, the seasonal amplitude of {metric} exceeds its standard deviation.",
        "The periodic amplitude of {metric} is greater than its std.",
        "For {metric}, seasonal amplitude exceeds std.",
    ],
    'no': [
        "No, the seasonal amplitude of {metric} does not exceed its standard deviation.",
        "The periodic amplitude of {metric} is not greater than its std.",
        "For {metric}, seasonal amplitude does not exceed std.",
    ],
}

AMPLITUDE_VS_RANGE_ANSWERS = {
    'yes': [
        "Yes, the seasonal amplitude of {metric} exceeds a quarter of the range.",
        "The periodic amplitude of {metric} is greater than range/4.",
        "For {metric}, amplitude exceeds range/4.",
    ],
    'no': [
        "No, the seasonal amplitude of {metric} does not exceed a quarter of the range.",
        "The periodic amplitude of {metric} is not greater than range/4.",
        "For {metric}, amplitude does not exceed range/4.",
    ],
}


# ============================================================
# Generator
# ============================================================

class CompositionalPeriodicGenerator:
    """RL compositions using periodicity atom + reasoning.

    R29-R32: derived queries from periodicity description
    R33-R34: cross-category (periodicity + stats)
    """

    def __init__(
        self,
        timeseries: np.ndarray,
        metric: str,
        seq_len: int,
        seasonal: Dict,
    ):
        self.ts = np.asarray(timeseries, dtype=float)
        self.metric = metric
        self.seq_len = seq_len
        self.seasonal = seasonal or {}
        self.period = self.seasonal.get('period', 0)
        self.amplitude = round(self.seasonal.get('amplitude', 0), 2)
        self.freq_type = self.seasonal.get('frequency_type', 'no periodicity')
        self.has_period = self.period > 0 and self.freq_type != 'no periodicity'

    # --- R29: Has periodicity ---

    def generate_has_periodicity(self) -> Dict[str, Any]:
        verdict = "yes" if self.has_period else "no"

        if verdict == "yes":
            answer_text = random.choice(HAS_PERIODICITY_ANSWERS['yes']).format(
                metric=self.metric, period=round(self.period, 1),
                amplitude=_fmt(self.amplitude), freq_type=self.freq_type,
            )
        else:
            answer_text = random.choice(HAS_PERIODICITY_ANSWERS['no']).format(
                metric=self.metric,
            )

        return _make_rl_result(
            random.choice(HAS_PERIODICITY_QUESTIONS).format(metric=self.metric),
            verdict, 'rl_has_periodicity', answer_text, self.seq_len,
        )

    # --- R30: Period estimate ---

    def generate_period_estimate(self) -> Optional[Dict[str, Any]]:
        if not self.has_period:
            return None

        period_r = round(self.period, 1)

        answer_text = random.choice(PERIOD_ESTIMATE_ANSWERS).format(
            metric=self.metric, period=period_r,
            amplitude=_fmt(self.amplitude), freq_type=self.freq_type,
        )
        return _make_rl_result(
            random.choice(PERIOD_ESTIMATE_QUESTIONS).format(metric=self.metric),
            str(period_r), 'rl_period_estimate', answer_text, self.seq_len,
        )

    # --- R31: Cycle count ---

    def generate_cycle_count(self) -> Optional[Dict[str, Any]]:
        if not self.has_period:
            return None

        n_cycles = round(self.seq_len / self.period, 1)
        period_r = round(self.period, 1)

        answer_text = random.choice(CYCLE_COUNT_ANSWERS).format(
            metric=self.metric, n_cycles=n_cycles,
            length=self.seq_len, period=period_r,
        )
        return _make_rl_result(
            random.choice(CYCLE_COUNT_QUESTIONS).format(metric=self.metric),
            str(n_cycles), 'rl_cycle_count', answer_text, self.seq_len,
        )

    # --- R33: Amplitude vs std (cross-category: G + B) ---

    def generate_amplitude_vs_std(self) -> Optional[Dict[str, Any]]:
        if not self.has_period:
            return None

        std_val = round(float(np.std(self.ts)), 2)
        verdict = "yes" if self.amplitude > std_val else "no"

        answer_text = random.choice(AMPLITUDE_VS_STD_ANSWERS[verdict]).format(
            metric=self.metric,
            amplitude=_fmt(self.amplitude), std=_fmt(std_val),
        )
        return _make_rl_result(
            random.choice(AMPLITUDE_VS_STD_QUESTIONS).format(metric=self.metric),
            verdict, 'rl_amplitude_vs_std', answer_text, self.seq_len,
        )

    # --- R34: Amplitude vs range/4 (cross-category: G + C) ---

    def generate_amplitude_vs_range(self) -> Optional[Dict[str, Any]]:
        if not self.has_period:
            return None

        max_val = round(float(np.max(self.ts)), 2)
        min_val = round(float(np.min(self.ts)), 2)
        quarter_range = round((max_val - min_val) / 4, 2)

        verdict = "yes" if self.amplitude > quarter_range else "no"

        answer_text = random.choice(AMPLITUDE_VS_RANGE_ANSWERS[verdict]).format(
            metric=self.metric,
            amplitude=_fmt(self.amplitude), quarter_range=_fmt(quarter_range),
        )
        return _make_rl_result(
            random.choice(AMPLITUDE_VS_RANGE_QUESTIONS).format(metric=self.metric),
            verdict, 'rl_amplitude_vs_range', answer_text, self.seq_len,
        )

    # --- Generate all ---

    def generate_all(self) -> List[Dict[str, Any]]:
        results = []
        generators = [
            self.generate_has_periodicity,
            self.generate_period_estimate,
            self.generate_cycle_count,
            self.generate_amplitude_vs_std,
            self.generate_amplitude_vs_range,
        ]
        for gen_fn in generators:
            try:
                r = gen_fn()
                if r is not None:
                    results.append(r)
            except Exception:
                logger.warning(f"RL {gen_fn.__name__} failed", exc_info=True)
        return results
