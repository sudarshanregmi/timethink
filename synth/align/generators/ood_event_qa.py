"""Local event OOD evaluation QA generators.

Novel compositions using events + trends + stats that RL never rewards.

Why each is OOD:
    - event_density_by_trend: RL checks event-in-type (R28), but never
      computes density (events per unit duration within a trend type)
    - max_amplitude_in_longest_segment: RL knows max amplitude (R25) and
      longest segment (R18) separately, never nests them
    - event_amplitude_vs_std: RL knows event amplitudes and std separately,
      never compares them
"""

import random
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from loguru import logger

from synth.align.generators.atomic_qa import _fmt
from synth.align.generators.atomic_event_qa import _normalize_event
from synth.align.generators.compositional_qa import (
    _make_rl_result, _pick_balanced_param,
)

# ============================================================
# Question templates
# ============================================================

EVENT_DENSITY_BY_TREND_QUESTIONS = [
    "What is the event density in {trend_type} segments of {metric}? Report events per 100 data points, rounded to 2 decimal places.",
    "For {metric}, compute the density of local events in {trend_type} segments (events per 100 points).",
    "How many local events per 100 data points occur in the {trend_type} phases of {metric}?",
    "Calculate the event density within {trend_type} segments of {metric} (events per 100 points).",
]

MAX_AMP_IN_LONGEST_SEGMENT_QUESTIONS = [
    "What is the amplitude of the highest-amplitude event in the longest trend segment of {metric}?",
    "For {metric}, find the maximum event amplitude within the longest trend segment.",
    "In the longest segment of {metric}, what is the peak event amplitude?",
    "What is the largest event amplitude occurring in the longest trend segment of {metric}?",
]

EVENT_AMPLITUDE_VS_STD_QUESTIONS = [
    "Does the maximum event amplitude in {metric} exceed {m}x the standard deviation of the timeseries?",
    "For {metric}, is the largest event amplitude greater than {m}x the overall std?",
    "Is the peak event amplitude in {metric} larger than {m}x the timeseries standard deviation?",
    "Compare the maximum event amplitude to {m}x the std of {metric}: is the amplitude greater?",
    "For {metric}, check if any event amplitude exceeds {m}x the standard deviation.",
]

# ============================================================
# Answer templates
# ============================================================

EVENT_DENSITY_BY_TREND_ANSWERS = [
    "The event density in {trend_type} segments of {metric} is {verdict} events per 100 data points.",
    "For {metric}, the {trend_type} event density is {verdict} per 100 points.",
    "The density is {verdict} events per 100 points in {trend_type} phases of {metric}.",
]

MAX_AMP_IN_LONGEST_SEGMENT_ANSWERS_FOUND = [
    "The largest event amplitude in the longest segment of {metric} is {verdict}.",
    "The peak amplitude in the longest segment of {metric} is {verdict}.",
]

MAX_AMP_IN_LONGEST_SEGMENT_ANSWERS_NONE = [
    "The longest segment of {metric} contains no local events.",
    "There are no local events in the longest segment of {metric}.",
]

EVENT_AMPLITUDE_VS_STD_ANSWERS = {
    'yes': [
        "Yes, the maximum event amplitude exceeds {m}x the std of {metric}.",
        "The peak event amplitude in {metric} is greater than {m}x its std.",
        "The largest event amplitude in {metric} exceeds {m}x its standard deviation.",
    ],
    'no': [
        "No, the maximum event amplitude does not exceed {m}x the std of {metric}.",
        "The peak event amplitude in {metric} is not greater than {m}x its std.",
        "The largest event amplitude in {metric} does not exceed {m}x its standard deviation.",
    ],
}


# ============================================================
# Generator
# ============================================================

class OODEventGenerator:
    """Event OOD compositions — eval only.

    Cross-category tasks combining events + trends + stats in novel ways.
    """

    def __init__(
        self,
        timeseries: np.ndarray,
        metric: str,
        seq_len: int,
        local_events: List[Dict],
        trend_list: Optional[List[Tuple[str, int, int]]] = None,
    ):
        self.ts = np.asarray(timeseries, dtype=float)
        self.metric = metric
        self.seq_len = seq_len
        self.trend_list = trend_list or []
        self.events = []
        for ev in local_events:
            norm = _normalize_event(ev)
            if norm is not None:
                self.events.append(norm)
        self.events.sort(key=lambda e: e['position'])

    # --- O14: Event density by trend type ---

    def generate_event_density_by_trend(self) -> Optional[Dict[str, Any]]:
        if not self.events or len(self.trend_list) < 2:
            return None

        # Pick a trend type that exists
        present_types = list(set(t for t, _, _ in self.trend_list))
        trend_type = random.choice(present_types)

        # Total duration of this trend type
        dur = sum(e - s for t, s, e in self.trend_list if t == trend_type)
        if dur == 0:
            return None

        # Count events in segments of this type
        n_events = 0
        for ev in self.events:
            for t, s, e in self.trend_list:
                if t == trend_type and s <= ev['position'] <= e:
                    n_events += 1
                    break

        density = round(n_events / dur * 100, 2)

        answer_text = random.choice(EVENT_DENSITY_BY_TREND_ANSWERS).format(
            metric=self.metric, trend_type=trend_type,
            verdict=_fmt(density), n_events=n_events, dur=dur,
        )
        return _make_rl_result(
            random.choice(EVENT_DENSITY_BY_TREND_QUESTIONS).format(
                metric=self.metric, trend_type=trend_type,
            ),
            _fmt(density), 'ood_event_density_by_trend', answer_text, self.seq_len,
        )

    # --- O15: Max amplitude in longest segment ---

    def generate_max_amp_in_longest_segment(self) -> Optional[Dict[str, Any]]:
        if not self.events or len(self.trend_list) < 2:
            return None

        # Find longest segment
        durations = [(e - s, i) for i, (_, s, e) in enumerate(self.trend_list)]
        max_dur, max_idx = max(durations)
        seg_type, seg_start, seg_end = self.trend_list[max_idx]

        # Events in longest segment
        seg_events = [
            ev for ev in self.events
            if seg_start <= ev['position'] <= seg_end
        ]

        # Natural rate is 95% "none" (longest segment usually has no events).
        # Skip rather than emit a collapsed "none" verdict — only generate when
        # the longest segment actually contains events.
        if not seg_events:
            return None

        max_ev = max(seg_events, key=lambda e: abs(e['amplitude']))

        answer_text = random.choice(MAX_AMP_IN_LONGEST_SEGMENT_ANSWERS_FOUND).format(
            metric=self.metric,
            event_type=max_ev['type'], event_pos=max_ev['position'],
            verdict=_fmt(max_ev['amplitude']),
            seg_type=seg_type, seg_start=seg_start, seg_end=seg_end,
            seg_dur=max_dur,
        )
        return _make_rl_result(
            random.choice(MAX_AMP_IN_LONGEST_SEGMENT_QUESTIONS).format(
                metric=self.metric,
            ),
            _fmt(max_ev['amplitude']), 'ood_max_amp_in_longest_segment',
            answer_text, self.seq_len,
        )

    # --- O16: Event amplitude vs std ---

    def generate_event_amplitude_vs_std(self) -> Optional[Dict[str, Any]]:
        if not self.events:
            return None

        max_amp = max(abs(ev['amplitude']) for ev in self.events)
        std_val = round(float(np.std(self.ts)), 2)
        max_amp_rounded = round(max_amp, 2)

        # Pick multiplier m via _pick_balanced_param for guaranteed 50/50:
        # flip a coin on the verdict first, pick an m ∈ {1..5} matching it,
        # skip sample if the amp/std ratio is too extreme to produce both.
        def verdict_for(m):
            return "yes" if max_amp_rounded > round(m * std_val, 2) else "no"

        picked = _pick_balanced_param(verdict_for, [1, 2, 3, 4, 5])
        if picked is None:
            return None
        m, verdict = picked
        threshold = round(m * std_val, 2)

        answer_text = random.choice(EVENT_AMPLITUDE_VS_STD_ANSWERS[verdict]).format(
            metric=self.metric,
            max_amp=_fmt(max_amp_rounded), std=_fmt(std_val),
            m=m, threshold=_fmt(threshold),
        )
        return _make_rl_result(
            random.choice(EVENT_AMPLITUDE_VS_STD_QUESTIONS).format(
                metric=self.metric, m=m,
            ),
            verdict, 'ood_event_amplitude_vs_std', answer_text, self.seq_len,
        )

    # --- Generate all ---

    def generate_all(self) -> List[Dict[str, Any]]:
        results = []
        generators = [
            self.generate_event_density_by_trend,
            self.generate_max_amp_in_longest_segment,
            self.generate_event_amplitude_vs_std,
        ]
        for gen_fn in generators:
            try:
                r = gen_fn()
                if r is not None:
                    results.append(r)
            except Exception:
                logger.warning(f"OOD {gen_fn.__name__} failed", exc_info=True)
        return results
