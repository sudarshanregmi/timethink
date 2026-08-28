"""Local event compositional QA generators for RL training.

Compose the single event enumeration atom (F1) with simple reasoning
or cross with stat/trend atoms.

Tasks:
    R23: event_count            — how many events? (enumerate + count)
    R24: event_type_at_pos      — what event near index p? (enumerate + lookup)
    R25: max_amplitude_event    — which event has largest amplitude? (enumerate + argmax)
    R26: event_count_by_type    — how many events of type X? (enumerate + filter + count)
    R27: event_near_extremum    — is there an event near the global max/min? (F + C, cross-cat)
    R28: event_in_trend_type    — is there an event in a given trend type? (F + E, cross-cat)
"""

import random
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from loguru import logger

from synth.align.generators.atomic_qa import _fmt
from synth.align.generators.atomic_event_qa import _normalize_event
from synth.align.generators.compositional_qa import _make_rl_result

# ============================================================
# Question templates
# ============================================================

EVENT_COUNT_QUESTIONS = [
    "How many local events does the {metric} timeseries have?",
    "Count the number of local events in {metric}.",
    "For {metric}, how many local events are present?",
    "What is the total number of local events in {metric}?",
    "How many spikes, dips, or sudden changes are there in {metric}?",
]

EVENT_TYPE_AT_POS_QUESTIONS = [
    "What type of local event occurs nearest to index {pos} in {metric}?",
    "For {metric}, what local event is at or near position {pos}?",
    "Identify the local event type closest to index {pos} in {metric}.",
    "What is happening at index {pos} in {metric}? Identify the local event.",
    "At position {pos} in {metric}, what type of local event occurs?",
]

MAX_AMPLITUDE_EVENT_QUESTIONS = [
    "Which local event in {metric} has the largest amplitude?",
    "For {metric}, what is the type and position of the highest-amplitude event?",
    "Find the local event with the maximum amplitude in {metric}.",
    "What is the most prominent local event in {metric} by amplitude?",
    "Identify the local event with the greatest amplitude in {metric}.",
]

EVENT_COUNT_BY_TYPE_QUESTIONS = [
    "How many {event_type} events are there in {metric}?",
    "Count the number of {event_type} events in the {metric} timeseries.",
    "For {metric}, how many events are of type {event_type}?",
    "What is the count of {event_type} events in {metric}?",
    "How many times does a {event_type} occur in {metric}?",
]

EVENT_NEAR_EXTREMUM_QUESTIONS = [
    "Is there a local event within {threshold} positions of the global {direction} in {metric}?",
    "For {metric}, does a local event occur near the global {direction} (within {threshold} indices)?",
    "Check if any local event in {metric} is within {threshold} positions of the {direction}.",
    "Is there a local event close to the global {direction} of {metric}? Use a distance of {threshold}.",
    "Does {metric} have a local event within {threshold} indices of its global {direction}?",
]

EVENT_IN_TREND_TYPE_QUESTIONS = [
    "Is there a local event in an {trend_type} segment of {metric}?",
    "For {metric}, does any local event fall within a {trend_type} trend segment?",
    "Check whether {metric} has a local event in a segment with {trend_type} trend.",
    "Does any local event of {metric} occur during an {trend_type} phase?",
    "Is there a local event located in an {trend_type} segment of {metric}?",
]

# ============================================================
# Answer templates
# ============================================================

EVENT_COUNT_ANSWERS_SINGULAR = [
    "The {metric} timeseries has {count} local event.",
    "There is {count} local event in {metric}.",
    "{metric} has {count} local event.",
]
EVENT_COUNT_ANSWERS_PLURAL = [
    "The {metric} timeseries has {count} local events.",
    "There are {count} local events in {metric}.",
    "{metric} has {count} local events.",
]

EVENT_TYPE_AT_POS_ANSWERS = [
    "The local event nearest to that index in {metric} is a {verdict}.",
    "The closest event to that position in {metric} is a {verdict}.",
    "Near that position in {metric}, the event type is {verdict}.",
]

MAX_AMPLITUDE_EVENT_ANSWERS = [
    "The highest-amplitude event in {metric} is a {event_type}.",
    "The most prominent event in {metric} is a {event_type}.",
    "The local event with the largest amplitude in {metric} is a {event_type}.",
]

EVENT_COUNT_BY_TYPE_ANSWERS_SINGULAR = [
    "There is {count} {event_type} event in {metric}.",
    "For {metric}, {count} event is of type {event_type}.",
    "{metric} has {count} {event_type} event.",
]
EVENT_COUNT_BY_TYPE_ANSWERS_PLURAL = [
    "There are {count} {event_type} events in {metric}.",
    "For {metric}, {count} events are of type {event_type}.",
    "{metric} has {count} {event_type} events.",
]

EVENT_NEAR_EXTREMUM_ANSWERS = {
    'yes': [
        "Yes, there is a local event near the global {direction} in {metric}.",
        "A local event is near the global {direction} of {metric}.",
    ],
    'no': [
        "No, no local event in {metric} is near the global {direction}.",
        "There is no local event near the global {direction} in {metric}.",
    ],
}

EVENT_IN_TREND_TYPE_ANSWERS = {
    'yes': [
        "Yes, {metric} has a local event within an {trend_type} segment.",
        "A local event falls in an {trend_type} segment of {metric}.",
    ],
    'no': [
        "No, no local event in {metric} falls within an {trend_type} segment.",
        "There are no local events in {trend_type} segments of {metric}.",
    ],
}


# ============================================================
# Generator
# ============================================================

class CompositionalEventGenerator:
    """RL compositions using event enumeration atom + simple reasoning.

    R23-R26: derived queries (count, lookup, filter)
    R27-R28: cross-category (events + extrema, events + trends)
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
        # Normalize and sort events
        self.events = []
        for ev in local_events:
            norm = _normalize_event(ev)
            if norm is not None:
                self.events.append(norm)
        self.events.sort(key=lambda e: e['position'])

    def _event_summary(self) -> str:
        parts = [f"{ev['type']} at {ev['position']}" for ev in self.events]
        return ", ".join(parts)

    # --- R23: Event count ---

    def generate_event_count(self) -> Optional[Dict[str, Any]]:
        if not self.events:
            return None

        count = len(self.events)
        summary = self._event_summary()

        templates = EVENT_COUNT_ANSWERS_SINGULAR if count == 1 else EVENT_COUNT_ANSWERS_PLURAL
        answer_text = random.choice(templates).format(
            metric=self.metric, count=count, summary=summary,
        )
        return _make_rl_result(
            random.choice(EVENT_COUNT_QUESTIONS).format(metric=self.metric),
            str(count), 'rl_event_count', answer_text, self.seq_len,
        )

    # --- R24: Event type at position ---

    def generate_event_type_at_pos(self) -> Optional[Dict[str, Any]]:
        if not self.events:
            return None

        # Pick a random event, ask about a position near it
        ev = random.choice(self.events)
        # Add some noise to position to make lookup non-trivial
        noise = random.randint(-5, 5)
        pos = max(0, min(self.seq_len - 1, ev['position'] + noise))

        # Find nearest event to this position
        nearest = min(self.events, key=lambda e: abs(e['position'] - pos))

        answer_text = random.choice(EVENT_TYPE_AT_POS_ANSWERS).format(
            metric=self.metric, pos=pos,
            verdict=nearest['type'], event_pos=nearest['position'],
            amplitude=_fmt(nearest['amplitude']),
        )
        return _make_rl_result(
            random.choice(EVENT_TYPE_AT_POS_QUESTIONS).format(
                metric=self.metric, pos=pos,
            ),
            nearest['type'], 'rl_event_type_at_pos', answer_text, self.seq_len,
        )

    # --- R25: Max amplitude event ---

    def generate_max_amplitude_event(self) -> Optional[Dict[str, Any]]:
        if not self.events:
            return None

        # Find event with max absolute amplitude
        max_ev = max(self.events, key=lambda e: abs(e['amplitude']))

        answer_text = random.choice(MAX_AMPLITUDE_EVENT_ANSWERS).format(
            metric=self.metric,
            event_type=max_ev['type'], pos=max_ev['position'],
            amplitude=_fmt(max_ev['amplitude']),
        )
        return _make_rl_result(
            random.choice(MAX_AMPLITUDE_EVENT_QUESTIONS).format(
                metric=self.metric,
            ),
            max_ev['type'], 'rl_max_amplitude_event', answer_text, self.seq_len,
        )

    # --- R26: Event count by type ---

    def generate_event_count_by_type(self) -> Optional[Dict[str, Any]]:
        if not self.events:
            return None

        # Pick a type that exists
        present_types = list(set(ev['type'] for ev in self.events))
        event_type = random.choice(present_types)
        count = sum(1 for ev in self.events if ev['type'] == event_type)

        templates = EVENT_COUNT_BY_TYPE_ANSWERS_SINGULAR if count == 1 else EVENT_COUNT_BY_TYPE_ANSWERS_PLURAL
        answer_text = random.choice(templates).format(
            metric=self.metric, count=count, event_type=event_type,
        )
        return _make_rl_result(
            random.choice(EVENT_COUNT_BY_TYPE_QUESTIONS).format(
                metric=self.metric, event_type=event_type,
            ),
            str(count), 'rl_event_count_by_type', answer_text, self.seq_len,
        )

    # --- R27: Event near extremum (cross-category: F + C) ---

    def generate_event_near_extremum(self) -> Optional[Dict[str, Any]]:
        if not self.events:
            return None

        direction = random.choice(['maximum', 'minimum'])
        if direction == 'maximum':
            ext_pos = int(np.argmax(self.ts))
        else:
            ext_pos = int(np.argmin(self.ts))

        threshold = random.choice([10, 15, 20, 25])

        # Find nearest event
        distances = [(abs(ev['position'] - ext_pos), ev) for ev in self.events]
        nearest_dist, nearest_ev = min(distances)

        has_near = nearest_dist <= threshold
        verdict = "yes" if has_near else "no"

        if verdict == "yes":
            answer_text = random.choice(EVENT_NEAR_EXTREMUM_ANSWERS['yes']).format(
                metric=self.metric, direction=direction,
                event_type=nearest_ev['type'], event_pos=nearest_ev['position'],
                ext_pos=ext_pos, threshold=threshold, distance=nearest_dist,
            )
        else:
            answer_text = random.choice(EVENT_NEAR_EXTREMUM_ANSWERS['no']).format(
                metric=self.metric, direction=direction,
                ext_pos=ext_pos, threshold=threshold,
                nearest_pos=nearest_ev['position'], nearest_dist=nearest_dist,
            )

        return _make_rl_result(
            random.choice(EVENT_NEAR_EXTREMUM_QUESTIONS).format(
                metric=self.metric, direction=direction, threshold=threshold,
            ),
            verdict, 'rl_event_near_extremum', answer_text, self.seq_len,
        )

    # --- R28: Event in trend type (cross-category: F + E) ---

    def generate_event_in_trend_type(self) -> Optional[Dict[str, Any]]:
        if not self.events or len(self.trend_list) < 2:
            return None

        trend_type = random.choice(['increase', 'decrease', 'keep steady'])

        # Check if any event falls in a segment of this trend type
        found_event = None
        found_seg = None
        for ev in self.events:
            for t, s, e in self.trend_list:
                if t == trend_type and s <= ev['position'] <= e:
                    found_event = ev
                    found_seg = (s, e)
                    break
            if found_event:
                break

        verdict = "yes" if found_event else "no"

        if verdict == "yes":
            answer_text = random.choice(EVENT_IN_TREND_TYPE_ANSWERS['yes']).format(
                metric=self.metric, trend_type=trend_type,
                event_type=found_event['type'],
                event_pos=found_event['position'],
                seg_start=found_seg[0], seg_end=found_seg[1],
            )
        else:
            answer_text = random.choice(EVENT_IN_TREND_TYPE_ANSWERS['no']).format(
                metric=self.metric, trend_type=trend_type,
            )

        return _make_rl_result(
            random.choice(EVENT_IN_TREND_TYPE_QUESTIONS).format(
                metric=self.metric, trend_type=trend_type,
            ),
            verdict, 'rl_event_in_trend_type', answer_text, self.seq_len,
        )

    # --- Generate all ---

    def generate_all(self) -> List[Dict[str, Any]]:
        results = []
        generators = [
            self.generate_event_count,
            self.generate_event_type_at_pos,
            self.generate_max_amplitude_event,
            self.generate_event_count_by_type,
            self.generate_event_near_extremum,
            self.generate_event_in_trend_type,
        ]
        for gen_fn in generators:
            try:
                r = gen_fn()
                if r is not None:
                    results.append(r)
            except Exception:
                logger.warning(f"RL {gen_fn.__name__} failed", exc_info=True)
        return results
