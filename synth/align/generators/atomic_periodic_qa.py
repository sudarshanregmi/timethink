"""Atomic periodicity QA generator (Category G).

ONE atom: describe the periodic behavior.

The model recognizes whether there's periodicity and reports
its characteristics (period, amplitude, frequency type).
70% of generated timeseries have NO periodicity — the atom
also teaches recognizing absence.

Derived queries (cycle count, regularity, amplitude comparisons)
become RL compositions.
"""

import random
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from loguru import logger

from synth.align.generators.atomic_qa import _fmt

# ============================================================
# Question templates
# ============================================================

ENUMERATE_QUESTIONS = [
    "Describe the periodic behavior of {metric}.",
    "Does {metric} exhibit periodicity? If so, describe it.",
    "Analyze the periodic characteristics of the {metric} timeseries.",
    "What periodic patterns, if any, are present in {metric}?",
    "For {metric}, describe any seasonal or periodic fluctuations.",
    "Examine the {metric} data for periodic behavior and report your findings.",
    "Is there periodicity in {metric}? Describe the periodic structure.",
    "What is the periodic behavior of the {metric} timeseries?",
    "Characterize the periodicity of {metric}, including period and amplitude.",
    "For {metric}, assess whether periodic fluctuations are present and describe them.",
]

# ============================================================
# Think block diversity
# ============================================================

_OPENERS = [
    "Let me examine the periodic behavior.",
    "I'll check for periodic fluctuations.",
    "Let me analyze the timeseries for periodicity.",
    "I need to assess the periodic characteristics.",
    "Let me look for cyclic patterns in the data.",
    "I'll examine whether there are periodic fluctuations.",
    "Let me check for seasonal patterns.",
    "I need to identify any periodic behavior.",
]

_PERIODIC_DESCRIPTIONS = [
    "The timeseries shows periodic fluctuations with an approximate period of {period} time steps.",
    "There is periodicity present. The period is approximately {period} time steps.",
    "Periodic behavior detected: period ≈ {period} time steps.",
    "The data exhibits periodic fluctuations at a period of roughly {period} time steps.",
    "I can identify a periodic pattern with period approximately {period}.",
]

_AMPLITUDE_DESCRIPTIONS = [
    "The seasonal amplitude is {amplitude}.",
    "The amplitude of the periodic component is {amplitude}.",
    "The fluctuations have an amplitude of approximately {amplitude}.",
    "The periodic oscillations have amplitude {amplitude}.",
]

_FREQ_DESCRIPTIONS = {
    'high frequency': [
        "This is a high frequency pattern — many cycles within the series.",
        "The periodicity is high frequency relative to the series length.",
        "This represents high frequency oscillations.",
    ],
    'low frequency': [
        "This is a low frequency pattern — few cycles within the series.",
        "The periodicity is low frequency relative to the series length.",
        "This represents low frequency oscillations.",
    ],
}

_NO_PERIODIC_DESCRIPTIONS = [
    "No periodic fluctuations are present in the data.",
    "The timeseries does not exhibit periodicity.",
    "No periodic or seasonal pattern detected.",
    "The data shows no periodic behavior.",
    "There are no cyclic patterns in this timeseries.",
]

# ============================================================
# Answer templates
# ============================================================

PERIODIC_ANSWERS = [
    "The {metric} timeseries exhibits periodic fluctuations with a period of approximately {period} time steps and an amplitude of {amplitude}. This is a {freq_type} pattern, yielding roughly {n_cycles} complete cycles across the {length}-point series.",
    "For {metric}, periodic behavior is present: period ≈ {period} time steps, amplitude ≈ {amplitude}, {freq_type}. The series contains approximately {n_cycles} complete cycles.",
    "{metric} shows {freq_type} periodicity. The period is about {period} time steps with amplitude {amplitude}, giving approximately {n_cycles} cycles over {length} data points.",
    "Periodic fluctuations are present in {metric}: approximately {n_cycles} cycles with period {period} and amplitude {amplitude} ({freq_type}).",
]

NO_PERIODIC_ANSWERS = [
    "The {metric} timeseries does not exhibit periodic fluctuations.",
    "For {metric}, no periodic or seasonal pattern is present in the data.",
    "No periodicity is detected in {metric}. The timeseries does not show cyclic behavior.",
    "{metric} has no periodic fluctuations.",
]


# ============================================================
# Generator
# ============================================================

class AtomicPeriodicGenerator:
    """Generates the single periodicity description atom.

    ONE atom: describe the periodic behavior.
    Handles both periodic and non-periodic timeseries.
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

    def generate_enumeration(self) -> Optional[Dict[str, Any]]:
        """G1: Describe periodic behavior.

        The single SFT atom for periodicity recognition.
        """
        period = self.seasonal.get('period', 0)
        amplitude = self.seasonal.get('amplitude', 0)
        freq_type = self.seasonal.get('frequency_type', 'no periodicity')
        has_periodicity = period > 0 and freq_type != 'no periodicity'

        opener = random.choice(_OPENERS)

        if has_periodicity:
            period_r = round(period, 1)
            amplitude_r = round(amplitude, 2)
            n_cycles = round(self.seq_len / period, 1)

            parts = [opener, ""]
            parts.append(random.choice(_PERIODIC_DESCRIPTIONS).format(
                period=period_r,
            ))
            parts.append(random.choice(_AMPLITUDE_DESCRIPTIONS).format(
                amplitude=_fmt(amplitude_r),
            ))
            if freq_type in _FREQ_DESCRIPTIONS:
                parts.append(random.choice(_FREQ_DESCRIPTIONS[freq_type]))
            parts.append(f"Number of complete cycles: approximately {n_cycles}.")
            parts.append("")

            reward_answer = str(period_r)
            parts.append(f"answer: {reward_answer}")

            content = "\n".join(parts)
            think = f"<think>\n{content}\n</think>"

            answer_text = random.choice(PERIODIC_ANSWERS).format(
                metric=self.metric, period=period_r,
                amplitude=_fmt(amplitude_r), freq_type=freq_type,
                n_cycles=n_cycles, length=self.seq_len,
            )

            return {
                'question': random.choice(ENUMERATE_QUESTIONS).format(
                    metric=self.metric,
                ),
                'answer': f"{think}\n{answer_text}",
                'eval_type': 'atomic_periodic_description',
                'eval_metadata': {
                    'length': self.seq_len,
                    'verdict': reward_answer,
                    'has_periodicity': True,
                    'period': period_r,
                    'amplitude': amplitude_r,
                },
            }
        else:
            parts = [opener, ""]
            parts.append(random.choice(_NO_PERIODIC_DESCRIPTIONS))
            parts.append("")

            reward_answer = "none"
            parts.append(f"answer: {reward_answer}")

            content = "\n".join(parts)
            think = f"<think>\n{content}\n</think>"

            answer_text = random.choice(NO_PERIODIC_ANSWERS).format(
                metric=self.metric,
            )

            return {
                'question': random.choice(ENUMERATE_QUESTIONS).format(
                    metric=self.metric,
                ),
                'answer': f"{think}\n{answer_text}",
                'eval_type': 'atomic_periodic_description',
                'eval_metadata': {
                    'length': self.seq_len,
                    'verdict': reward_answer,
                    'has_periodicity': False,
                },
            }

    def generate_all(self) -> List[Dict[str, Any]]:
        results = []
        try:
            r = self.generate_enumeration()
            if r is not None:
                results.append(r)
        except Exception:
            logger.warning("G1 periodic_description failed", exc_info=True)
        return results
