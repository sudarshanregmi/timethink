"""Atomic trend QA generator (Category E).

ONE atom: enumerate the trend breakdown.

The hard part is RECOGNITION — identifying structure from the timeseries
representation. Once the model can enumerate segments, derived operations
(count, lookup, filter, compare) are basic LLM capabilities.

Derived queries (count, type at position, boundaries, duration) become
RL compositions — the model chains enumeration + simple reasoning.

The think block teaches the model to articulate structural understanding
with diverse phrasings, so it doesn't memorize one rigid format.
"""

import random
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from loguru import logger

from synth.align.generators.atomic_qa import _fmt

# ============================================================
# Question templates — diverse phrasings of "enumerate"
# ============================================================

ENUMERATE_QUESTIONS = [
    "Describe the trend structure of {metric}.",
    "What are the trend segments of {metric}?",
    "Break down the trend phases of {metric} with their boundaries.",
    "List the trend segments of the {metric} timeseries.",
    "How does {metric} evolve over time? Describe the trend segments.",
    "Provide a structural breakdown of the {metric} timeseries by trend.",
    "What trend phases make up the {metric} data? Report each segment's type and boundaries.",
    "Enumerate the distinct trend segments in {metric}.",
    "For {metric}, identify all trend segments with their types, start indices, and end indices.",
    "Decompose the {metric} timeseries into its constituent trend segments.",
    "What is the trend breakdown of {metric}?",
    "Describe how {metric} changes across its full duration — list each trend segment.",
]

# ============================================================
# Think block diversity — different ways to present the enumeration
# ============================================================

_OPENERS = [
    "Let me examine the trend structure.",
    "I'll identify the trend segments.",
    "Let me look at how the timeseries evolves.",
    "I need to analyze the structural composition.",
    "Let me break down the trend phases.",
    "I'll identify the distinct trend periods.",
    "Let me trace the trend changes through the data.",
    "I need to examine the structural breakdown.",
]

# Different segment line styles (like chunk line styles in atomic_qa)
_SEGMENT_STYLES = {
    'arrow': lambda i, t, s, e, d: f"  {s} → {e}: {t} (duration {d})",
    'bracket': lambda i, t, s, e, d: f"  [{s}, {e}] {t} ({d} points)",
    'numbered': lambda i, t, s, e, d: f"  Segment {i}: {t} from index {s} to {e} ({d} points)",
    'dash': lambda i, t, s, e, d: f"  {i} — {t}, indices {s}-{e}, length {d}",
    'colon': lambda i, t, s, e, d: f"  {t}: {s} to {e} (duration {d})",
    'descriptive': lambda i, t, s, e, d: f"  From {s} to {e} the data is {t} ({d} data points)",
}

_SUMMARY_STYLES = [
    "That gives {n} segments total.",
    "Total: {n} segments.",
    "{n} distinct trend segments identified.",
    "In total there are {n} trend phases.",
    "The series has {n} segments.",
]


def _build_enumeration_think(
    trend_list: List[Tuple[str, int, int]],
) -> Tuple[str, str]:
    """Build a diverse think block for trend enumeration.

    Returns (full_think_str, reward_answer).
    The reward_answer is the segment count as string.
    """
    n = len(trend_list)
    opener = random.choice(_OPENERS)
    style_name = random.choice(list(_SEGMENT_STYLES.keys()))
    style_fn = _SEGMENT_STYLES[style_name]

    parts = [opener, ""]

    for i, (seg_type, start, end) in enumerate(trend_list):
        duration = end - start
        parts.append(style_fn(i, seg_type, start, end, duration))

    parts.append("")
    parts.append(random.choice(_SUMMARY_STYLES).format(n=n))

    # Build the reward answer: segment count
    reward_answer = str(n)
    parts.append(f"answer: {reward_answer}")

    content = "\n".join(parts)
    return f"<think>\n{content}\n</think>", reward_answer


# ============================================================
# Answer templates — rich descriptions of the structure
# ============================================================

ENUMERATE_ANSWERS = [
    "The {metric} timeseries consists of {n} distinct trend segments: {description}.",
    "For {metric}, the structural breakdown reveals {n} trend phases: {description}.",
    "The {metric} data evolves through {n} trend segments: {description}.",
    "{metric} has {n} trend phases. {description}.",
    "The trend structure of {metric} comprises {n} segments: {description}.",
]


def _build_description(
    trend_list: List[Tuple[str, int, int]],
) -> str:
    """Build a natural language description of the trend segments."""
    if len(trend_list) == 1:
        t, s, e = trend_list[0]
        return f"a single {t} phase from index {s} to {e}"

    # Pick a random connector style
    style = random.choice(['sequential', 'then', 'followed'])

    parts = []
    for i, (seg_type, start, end) in enumerate(trend_list):
        duration = end - start

        if style == 'sequential':
            parts.append(f"{seg_type} from index {start} to {end} ({duration} points)")
        elif style == 'then':
            if i == 0:
                parts.append(f"{seg_type} from {start} to {end}")
            else:
                parts.append(f"then {seg_type} from {start} to {end}")
        else:  # followed
            if i == 0:
                parts.append(f"an {seg_type} phase ({start}-{end})")
            else:
                parts.append(f"followed by {seg_type} ({start}-{end})")

    if style == 'sequential':
        return ", ".join(parts)
    else:
        return ", ".join(parts)


# ============================================================
# Generator
# ============================================================

class AtomicTrendGenerator:
    """Generates the single trend enumeration atom.

    ONE atom: enumerate the trend breakdown.
    Derived queries (count, type lookup, boundaries, duration)
    become RL compositions.

    The think block shows diverse enumeration styles.
    The answer after </think> is a rich structural description.
    """

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

    def generate_enumeration(self) -> Optional[Dict[str, Any]]:
        """E1: Enumerate all trend segments.

        The single SFT atom for trend structure.
        Teaches the model to recognize and articulate trend phases.
        """
        if not self.trend_list:
            return None

        think, ans = _build_enumeration_think(self.trend_list)

        description = _build_description(self.trend_list)
        answer_text = random.choice(ENUMERATE_ANSWERS).format(
            metric=self.metric, n=len(self.trend_list),
            description=description,
        )

        return {
            'question': random.choice(ENUMERATE_QUESTIONS).format(
                metric=self.metric,
            ),
            'answer': f"{think}\n{answer_text}",
            'eval_type': 'atomic_trend_enumeration',
            'eval_metadata': {
                'length': self.seq_len,
                'verdict': ans,
                'n_segments': len(self.trend_list),
            },
        }

    def generate_all(self) -> List[Dict[str, Any]]:
        results = []
        try:
            r = self.generate_enumeration()
            if r is not None:
                results.append(r)
        except Exception:
            logger.warning("E1 trend_enumeration failed", exc_info=True)
        return results
