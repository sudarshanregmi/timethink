"""Atomic local event QA generator (Category F).

ONE atom: enumerate local events.

Same reasoning as trend (Category E): the hard part is RECOGNITION —
identifying events from the timeseries representation. Once the model
can enumerate events, derived queries (count, type lookup, amplitude
comparison) are basic LLM capabilities → RL compositions.

The think block teaches the model to identify and list local events
with diverse phrasings.
"""

import random
from typing import Any, Dict, List, Optional

import numpy as np
from loguru import logger

from synth.align.generators.atomic_qa import _fmt

# ============================================================
# Question templates — diverse phrasings of "enumerate events"
# ============================================================

ENUMERATE_QUESTIONS = [
    "Describe the local events in the {metric} timeseries.",
    "What local events are present in {metric}?",
    "List all local events in {metric} with their types and positions.",
    "Identify the local events occurring in {metric}.",
    "What anomalies or local changes are present in the {metric} data?",
    "Enumerate the local events in {metric}, including their types, positions, and amplitudes.",
    "For {metric}, what local events can you identify?",
    "Break down the local events in {metric}.",
    "Describe any spikes, dips, or sudden changes in {metric}.",
    "What local fluctuations or events does {metric} contain?",
    "Provide a list of all local events in {metric} with details.",
    "For {metric}, identify all local changes with their positions and characteristics.",
]

# ============================================================
# Think block diversity
# ============================================================

_OPENERS = [
    "Let me identify the local events.",
    "I'll examine the timeseries for local events.",
    "Let me look for spikes, dips, and other local changes.",
    "I need to identify the local events in this data.",
    "Let me find all local events and their characteristics.",
    "I'll scan for local anomalies and changes.",
    "Let me examine the local event structure.",
    "I need to list the local events present.",
]

_EVENT_STYLES = {
    'arrow': lambda i, t, p, a: f"  {p} → {t} (amplitude {a})",
    'bracket': lambda i, t, p, a: f"  [{p}] {t}, amplitude {a}",
    'numbered': lambda i, t, p, a: f"  Event {i}: {t} at index {p} (amplitude {a})",
    'dash': lambda i, t, p, a: f"  {i} — {t} at position {p}, amplitude {a}",
    'descriptive': lambda i, t, p, a: f"  At index {p} there is a {t} with amplitude {a}",
    'colon': lambda i, t, p, a: f"  {t}: position {p}, amplitude {a}",
}

_SUMMARY_STYLES = [
    "That gives {n} local events total.",
    "Total: {n} events.",
    "{n} local events identified.",
    "In total there are {n} local events.",
    "The series has {n} local events.",
]


def _normalize_event(event: Dict) -> Optional[Dict]:
    """Extract (type, position, amplitude) from an event dict.

    Handles the various field names in the pipeline.
    Returns normalized dict or None if essential fields missing.
    """
    e_type = event.get('type', '')
    if isinstance(e_type, list):
        e_type = e_type[0] if e_type else ''
    if not e_type:
        return None

    pos = event.get('position_start', event.get('position'))
    if pos is None:
        return None

    amplitude = event.get('amplitude', 0.0)
    if amplitude is None:
        amplitude = 0.0

    return {
        'type': e_type,
        'position': int(pos),
        'amplitude': round(float(amplitude), 2),
    }


def _build_event_enumeration_think(
    events: List[Dict],
) -> str:
    """Build a diverse think block for event enumeration.

    Returns (full_think_str, reward_answer).
    """
    n = len(events)
    opener = random.choice(_OPENERS)
    style_name = random.choice(list(_EVENT_STYLES.keys()))
    style_fn = _EVENT_STYLES[style_name]

    parts = [opener, ""]

    for i, ev in enumerate(events):
        parts.append(style_fn(i, ev['type'], ev['position'], _fmt(ev['amplitude'])))

    parts.append("")
    parts.append(random.choice(_SUMMARY_STYLES).format(n=n))

    reward_answer = str(n)
    parts.append(f"answer: {reward_answer}")

    content = "\n".join(parts)
    return f"<think>\n{content}\n</think>", reward_answer


# ============================================================
# Answer templates
# ============================================================

ENUMERATE_ANSWERS_SINGULAR = [
    "The {metric} timeseries contains {n} local event: {description}.",
    "For {metric}, {n} local event is present: {description}.",
    "The {metric} data has {n} local event: {description}.",
    "{metric} contains {n} local event. {description}.",
    "There is {n} local event in {metric}: {description}.",
]
ENUMERATE_ANSWERS_PLURAL = [
    "The {metric} timeseries contains {n} local events: {description}.",
    "For {metric}, {n} local events are present: {description}.",
    "The {metric} data has {n} local events: {description}.",
    "{metric} contains {n} local events. {description}.",
    "There are {n} local events in {metric}: {description}.",
]


def _build_event_description(events: List[Dict]) -> str:
    """Build natural language description of events."""
    if len(events) == 1:
        ev = events[0]
        return f"a {ev['type']} at index {ev['position']} with amplitude {_fmt(ev['amplitude'])}"

    style = random.choice(['list', 'sequential'])

    parts = []
    for ev in events:
        if style == 'list':
            parts.append(f"{ev['type']} at index {ev['position']} (amplitude {_fmt(ev['amplitude'])})")
        else:
            parts.append(f"a {ev['type']} at index {ev['position']}")

    return ", ".join(parts)


# ============================================================
# Generator
# ============================================================

class AtomicEventGenerator:
    """Generates the single local event enumeration atom.

    ONE atom: enumerate all local events.
    Derived queries (count, type at position, max amplitude, etc.)
    become RL compositions.
    """

    def __init__(
        self,
        timeseries: np.ndarray,
        metric: str,
        seq_len: int,
        local_events: List[Dict],
    ):
        self.ts = np.asarray(timeseries, dtype=float)
        self.metric = metric
        self.seq_len = seq_len
        # Normalize events upfront
        self.events = []
        for ev in local_events:
            norm = _normalize_event(ev)
            if norm is not None:
                self.events.append(norm)
        # Sort by position for consistent ordering
        self.events.sort(key=lambda e: e['position'])

    def generate_enumeration(self) -> Optional[Dict[str, Any]]:
        """F1: Enumerate all local events.

        The single SFT atom for local event recognition.
        """
        if not self.events:
            return None

        think, ans = _build_event_enumeration_think(self.events)

        description = _build_event_description(self.events)
        n_ev = len(self.events)
        templates = ENUMERATE_ANSWERS_SINGULAR if n_ev == 1 else ENUMERATE_ANSWERS_PLURAL
        answer_text = random.choice(templates).format(
            metric=self.metric, n=n_ev,
            description=description,
        )

        return {
            'question': random.choice(ENUMERATE_QUESTIONS).format(
                metric=self.metric,
            ),
            'answer': f"{think}\n{answer_text}",
            'eval_type': 'atomic_event_enumeration',
            'eval_metadata': {
                'length': self.seq_len,
                'verdict': ans,
                'n_events': len(self.events),
            },
        }

    def generate_all(self) -> List[Dict[str, Any]]:
        results = []
        try:
            r = self.generate_enumeration()
            if r is not None:
                results.append(r)
        except Exception:
            logger.warning("F1 event_enumeration failed", exc_info=True)
        return results
