"""Decomposition bridge QA generators.

SFT examples teaching the meta-skill of breaking compositional
questions into atomic steps — bridging atomic SFT → RL compositions.

5 representative decomposition patterns:
    B1: compare_interval_means — "compute X, compute Y, compare"
    B2: stat_cross_compare     — "derive from different stats, compare"
    B3: event_in_context       — "enumerate + filter by context"
    B4: extremum_in_context    — "find + locate in context"
    B5: periodicity_cross      — "cross-category comparison"

Each produces a full think block showing the decomposition structure.
eval_metadata['bridge'] = True distinguishes from regular RL samples.

Also provides rich Analysis Context for TSEvol seed evolution —
structured decomposition steps give TSEvol much more to ground on
than minimal RL think blocks.
"""

import random
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from loguru import logger

from synth.align.generators.atomic_qa import _fmt, _compute_chunks, CHUNK_SIZE
from synth.align.generators.compositional_qa import _interval_mean
from synth.align.generators.atomic_event_qa import _normalize_event


def _chunked_mean_lines(
    ts: np.ndarray, start: int, end: int, label: str,
) -> Tuple[List[str], float]:
    """Build step-by-step chunk-mean breakdown lines for interval [start, end].

    Returns (lines, mean_of_chunk_means).
    """
    chunks = _compute_chunks(ts, start, end)
    lines = [f"{label} [{start}, {end}]:"]
    for cs, ce, cm, _ in chunks:
        lines.append(f"  chunk [{cs}, {ce}] mean = {_fmt(cm)}")
    chunk_means = [cm for _, _, cm, _ in chunks]
    mean_of_means = round(float(np.mean(chunk_means)), 2)
    sum_expr = " + ".join(_fmt(cm) for cm in chunk_means)
    lines.append(
        f"  {label.lower()} mean = ({sum_expr}) / {len(chunk_means)} "
        f"= {_fmt(mean_of_means)}"
    )
    return lines, mean_of_means

# ============================================================
# Shared think-block templates
# ============================================================

_OPENERS = [
    # Decomposition framing
    "Let me break this down step by step.",
    "I'll decompose this into sub-problems.",
    "This requires multiple computations.",
    "Let me work through this systematically.",
    "I need to combine several analyses.",
    "Let me approach this methodically.",
    "I'll solve this in stages.",
    "This is a multi-step problem.",
    # Hypothesis-testing framing
    "Let me check whether the condition holds.",
    "I need to verify if the stated claim is true.",
    "The question is whether this property applies — let me test it.",
    # Exploratory framing
    "First, let me look at what the data shows.",
    "Starting from what's observable in the series.",
    "Let me examine the time series carefully.",
    # Comparative framing
    "Comparing the two pieces requires a few steps.",
    "I'll put these side by side and work it out.",
    "To compare these properly, I need to compute each first.",
    # Condition-checking framing
    "Does this meet the threshold? Let me work it out.",
    "I need to determine if the condition is satisfied.",
    "The check requires computing and comparing.",
]

_PLAN_FORMATS = [
    "To answer this, I need to: {steps}",
    "My approach: {steps}.",
    "This involves: {steps}.",
    "Steps: {steps}.",
    "Plan — {steps}.",
]

_STEP_NUMBERING = [
    lambda i, s: f"{i + 1}) {s}",
    lambda i, s: f"({i + 1}) {s}",
]

_BACKTRACK_NUDGES = [
    "Let me double-check these intermediate values before combining.",
    "Actually, wait — let me re-verify the numbers above.",
    "Before concluding, let me make sure I have the right values.",
    "Hmm, let me re-read the computed values to be sure.",
]


def _format_plan(plan_steps: List[str]) -> str:
    """Format a decomposition plan from step descriptions."""
    fmt = random.choice(_PLAN_FORMATS)
    numberer = random.choice(_STEP_NUMBERING)
    steps_text = "; ".join(numberer(i, s) for i, s in enumerate(plan_steps))
    return fmt.format(steps=steps_text)


def _build_bridge_think(
    plan_steps: List[str],
    step_results: List[str],
    combination: str,
    verdict: str,
) -> str:
    """Build a decomposition bridge think block.

    Shows: opener → plan → step results → [optional backtrack] → combination → answer.

    A small fraction of samples insert a self-check line between the step results
    and the combination. This plants a self-verification template that RL
    rollout can reuse when exploring alternative reasoning paths.
    """
    opener = random.choice(_OPENERS)
    plan = _format_plan(plan_steps)

    parts = [opener, "", plan, ""]
    parts.extend(step_results)
    if random.random() < 0.08:
        parts.extend(["", random.choice(_BACKTRACK_NUDGES)])
    parts.extend(["", combination, ""])
    parts.append(f"answer: {verdict}")

    content = "\n".join(parts)
    return f"<think>\n{content}\n</think>"


# ============================================================
# B1: Compare interval means ("compute X, compute Y, compare")
# ============================================================

B1_QUESTIONS = [
    "Is the mean of the first half of {metric} greater than the mean of the second half?",
    "For {metric}, does the first-half mean exceed the second-half mean?",
    "Compare the mean of {metric} in the first half vs the second half: is the first larger?",
    "Is the average of {metric} higher in the first half than the second half?",
    "Does {metric} have a higher mean in its first half compared to its second half?",
]

B1_ANSWERS = {
    'yes': [
        "Yes, the first-half mean of {metric} exceeds the second-half mean.",
        "The mean of {metric} is higher in the first half.",
    ],
    'no': [
        "No, the first-half mean of {metric} does not exceed the second-half mean.",
        "The mean of {metric} is not higher in the first half.",
    ],
}


# ============================================================
# B2: Stat cross-compare ("derive from different stats, compare")
# ============================================================

B2_QUESTIONS = [
    "Is the standard deviation of the first half of {metric} greater than the second half?",
    "For {metric}, does the first-half std exceed the second-half std?",
    "Compare the volatility of {metric} across halves: is the first half more volatile?",
    "Is the first half of {metric} more variable (higher std) than the second half?",
    "Does the standard deviation of {metric} decrease from the first half to the second half?",
]

B2_ANSWERS = {
    'yes': [
        "Yes, the first-half std of {metric} exceeds the second-half std.",
        "The first half of {metric} is more volatile.",
    ],
    'no': [
        "No, the first-half std of {metric} does not exceed the second-half std.",
        "The first half of {metric} is not more volatile.",
    ],
}


# ============================================================
# B3: Event in context ("enumerate + filter by context")
# ============================================================

B3_QUESTIONS = [
    "Is there a local event in an {trend_type} segment of {metric}?",
    "For {metric}, does any local event fall within a {trend_type} trend segment?",
    "Check whether {metric} has a local event during a {trend_type} phase.",
    "Does {metric} have any local events in {trend_type} segments?",
    "Is there a local event occurring within an {trend_type} segment of {metric}?",
]

B3_ANSWERS = {
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
# B4: Extremum in context ("find + locate in context")
# ============================================================

B4_QUESTIONS = [
    "Does the global maximum of {metric} fall in a {trend_type} segment?",
    "Is the maximum value of {metric} located within a {trend_type} trend phase?",
    "For {metric}, is the global peak found in a {trend_type} segment?",
    "Check whether the maximum of {metric} occurs during a {trend_type} phase.",
    "Does the global max of {metric} lie in a segment with {trend_type} trend?",
]

B4_ANSWERS = {
    'yes': [
        "Yes, the global maximum of {metric} falls in a {trend_type} segment.",
        "The peak of {metric} is in a {trend_type} segment.",
    ],
    'no': [
        "No, the global maximum of {metric} is not in a {trend_type} segment.",
        "The peak of {metric} does not fall in a {trend_type} segment.",
    ],
}


# ============================================================
# B5: Periodicity cross ("cross-category comparison")
# ============================================================

B5_QUESTIONS = [
    "Is the seasonal amplitude of {metric} greater than its standard deviation?",
    "For {metric}, does the periodic amplitude exceed the overall std?",
    "Compare the seasonal amplitude to the std of {metric}: is the amplitude larger?",
    "Is the amplitude of the periodic component in {metric} greater than its standard deviation?",
    "Check if {metric}'s seasonal amplitude exceeds its std.",
]

B5_ANSWERS = {
    'yes': [
        "Yes, the seasonal amplitude of {metric} exceeds its std.",
        "The periodic amplitude of {metric} is greater than its standard deviation.",
    ],
    'no': [
        "No, the seasonal amplitude of {metric} does not exceed its std.",
        "The periodic amplitude of {metric} is not greater than its standard deviation.",
    ],
}


# ============================================================
# B6: Enumerate → filter → count ("count the subset matching a predicate")
# ============================================================

B6_QUESTIONS = [
    "How many local events of type {event_type} occur in {metric}?",
    "Count the number of {event_type} events in {metric}.",
    "For {metric}, how many local events are of type {event_type}?",
    "What is the count of {event_type}-type local events in {metric}?",
    "Tally the {event_type} events in {metric}.",
]


# ============================================================
# B7: Enumerate → argmax by property → extract ("longest, dominant, biggest")
# ============================================================

B7_QUESTIONS = [
    "What is the trend type of the longest segment in {metric}?",
    "Which trend type has the longest segment in {metric}?",
    "For {metric}, what type is the longest trend segment?",
    "Identify the trend type of the longest segment in {metric}.",
    "In {metric}, what is the type of the longest trend segment?",
]


# ============================================================
# B8: Chunk → per-chunk stat → meta-stat ("monotonic, stable, shifted")
# ============================================================

B8_QUESTIONS = [
    "Are the chunked means of {metric} monotonically increasing or decreasing?",
    "Do the chunk-level means of {metric} form a monotonic sequence?",
    "For {metric}, are the chunk means monotonic (strictly increasing or decreasing)?",
    "Check whether the chunked mean sequence of {metric} is monotonic.",
    "Is the sequence of chunk means in {metric} monotonically changing?",
]

B8_ANSWERS = {
    'yes': [
        "Yes, the chunk means of {metric} are monotonically changing.",
        "The chunked means of {metric} form a monotonic sequence.",
    ],
    'no': [
        "No, the chunk means of {metric} are not monotonic.",
        "The chunked means of {metric} do not form a monotonic sequence.",
    ],
}


# ============================================================
# B9: Enumerate → filter → sum → divide by total
#     ("aggregate / aggregate = ratio" — plants the rate primitive)
#
# Two question framings share identical computation:
#   fraction: "what fraction of {metric}'s duration..." (accurate terminology)
#   density:  "what is the density of {trend_type} in {metric}..."
#             (anchors the density/rate vocabulary to the division primitive
#              so OOD event-density questions have a word-level hook)
# ============================================================

B9_FRACTION_QUESTIONS = [
    "What fraction of {metric}'s total duration is covered by {trend_type} trend segments?",
    "Compute the proportion of the series where {metric} shows {trend_type} behavior.",
    "Over the full series, what fraction of positions are in {trend_type} segments of {metric}?",
    "For {metric}, what fraction of the total duration is spent in {trend_type} segments?",
    "What proportion of {metric}'s timeline is {trend_type}?",
]

B9_FRACTION_ANSWERS = [
    "Approximately {fraction} of {metric}'s duration is covered by {trend_type} segments.",
    "About {fraction} of {metric}'s timeline is in {trend_type} segments.",
]

B9_DENSITY_QUESTIONS = [
    "What is the density of {trend_type} segments in {metric} (as a fraction of positions they cover)?",
    "Compute the density of {trend_type} behavior in {metric}: what portion of the timeline does it occupy?",
    "For {metric}, what is the coverage density of {trend_type} segments over the full series?",
    "How densely is {metric} covered by {trend_type} segments, as a ratio of covered to total duration?",
    "Measure the density of {trend_type} segments in {metric} — covered duration divided by total duration.",
]

B9_DENSITY_ANSWERS = [
    "The density of {trend_type} segments in {metric} is {fraction} (covered / total).",
    "For {metric}, {trend_type} segments have a coverage density of {fraction}.",
]


# ============================================================
# Generator
# ============================================================

_ALL_TREND_TYPES = ['increase', 'decrease', 'keep steady']


class DecompositionBridgeGenerator:
    """Generates decomposition bridge SFT examples.

    5 representative patterns teaching compositional reasoning.
    Each has a full think block showing the decomposition structure:
    plan → step results → combination → verdict.
    """

    def __init__(
        self,
        timeseries: np.ndarray,
        metric: str,
        seq_len: int,
        trend_list: Optional[List[Tuple[str, int, int]]] = None,
        local_events: Optional[List[Dict]] = None,
        seasonal: Optional[Dict] = None,
    ):
        self.ts = np.asarray(timeseries, dtype=float)
        self.metric = metric
        self.seq_len = seq_len
        self.trend_list = trend_list or []
        self.events = []
        if local_events:
            for ev in local_events:
                norm = _normalize_event(ev)
                if norm is not None:
                    self.events.append(norm)
            self.events.sort(key=lambda e: e['position'])
        self.seasonal = seasonal or {}
        self.period = self.seasonal.get('period', 0)
        self.amplitude = round(self.seasonal.get('amplitude', 0), 2)
        self.has_period = (
            self.period > 0
            and self.seasonal.get('frequency_type') != 'no periodicity'
        )

    def _make_result(
        self, question, think, answer_text, eval_type, verdict,
    ) -> Dict[str, Any]:
        return {
            'question': question,
            'answer': f"{think}\n{answer_text}",
            'eval_type': eval_type,
            'eval_metadata': {
                'length': self.seq_len,
                'verdict': str(verdict),
                'bridge': True,
            },
        }

    # --- B1: Compare interval means ---

    def generate_compare_interval_means(self) -> Optional[Dict[str, Any]]:
        """B1: compute X, compute Y, compare. (A1+A3 → R1 pattern)"""
        if self.seq_len < 32:
            return None

        mid = self.seq_len // 2
        h1_lines, h1_mean = _chunked_mean_lines(self.ts, 0, mid - 1, "First half")
        h2_lines, h2_mean = _chunked_mean_lines(
            self.ts, mid, self.seq_len - 1, "Second half",
        )
        verdict = "yes" if h1_mean > h2_mean else "no"

        think = _build_bridge_think(
            plan_steps=[
                f"chunk the first half [0, {mid - 1}] and compute its mean",
                f"chunk the second half [{mid}, {self.seq_len - 1}] and compute its mean",
                "compare the two means",
            ],
            step_results=[*h1_lines, "", *h2_lines],
            combination=(
                f"Comparison: {_fmt(h1_mean)} "
                f"{'>' if h1_mean > h2_mean else '<='} "
                f"{_fmt(h2_mean)} → {verdict}"
            ),
            verdict=verdict,
        )

        answer_text = random.choice(B1_ANSWERS[verdict]).format(
            metric=self.metric,
            h1_mean=_fmt(h1_mean), h2_mean=_fmt(h2_mean),
        )
        return self._make_result(
            random.choice(B1_QUESTIONS).format(metric=self.metric),
            think, answer_text, 'rl_half_mean_compare', verdict,
        )

    # --- B2: Stat cross-compare ---

    def generate_stat_cross_compare(self) -> Optional[Dict[str, Any]]:
        """B2: derive from different stats, compare. (B2×2 → R9 pattern)"""
        if self.seq_len < 32:
            return None

        mid = self.seq_len // 2
        std1 = round(float(np.std(self.ts[:mid])), 2)
        std2 = round(float(np.std(self.ts[mid:])), 2)
        verdict = "yes" if std1 > std2 else "no"

        think = _build_bridge_think(
            plan_steps=[
                f"compute the std of the first half [0, {mid - 1}]",
                f"compute the std of the second half [{mid}, {self.seq_len - 1}]",
                "compare the two standard deviations",
            ],
            step_results=[
                f"Std of first half [0, {mid - 1}]: {_fmt(std1)}",
                f"Std of second half [{mid}, {self.seq_len - 1}]: {_fmt(std2)}",
            ],
            combination=(
                f"Comparison: {_fmt(std1)} "
                f"{'>' if std1 > std2 else '<='} "
                f"{_fmt(std2)} → {verdict}"
            ),
            verdict=verdict,
        )

        answer_text = random.choice(B2_ANSWERS[verdict]).format(
            metric=self.metric, std1=_fmt(std1), std2=_fmt(std2),
        )
        # Uses rl_volatility_change eval_type (same computation, flipped direction)
        return self._make_result(
            random.choice(B2_QUESTIONS).format(metric=self.metric),
            think, answer_text, 'rl_volatility_change', verdict,
        )

    # --- B3: Event in context ---

    def generate_event_in_context(self) -> Optional[Dict[str, Any]]:
        """B3: enumerate + filter by context. (F+E → R28 pattern)"""
        if not self.events or len(self.trend_list) < 2:
            return None

        trend_type = random.choice(_ALL_TREND_TYPES)

        # Summarize events for think block
        event_summary = ", ".join(
            f"{ev['type']} at {ev['position']}" for ev in self.events
        )

        # Summarize segments
        seg_summary = ", ".join(
            f"{t} [{s}-{e}]" for t, s, e in self.trend_list
        )

        # Find match
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

        if found_event:
            check_line = (
                f"Check: {found_event['type']} at {found_event['position']} "
                f"is in {trend_type} [{found_seg[0]}-{found_seg[1]}] → yes"
            )
        else:
            check_line = f"Check: no events fall in {trend_type} segments → no"

        think = _build_bridge_think(
            plan_steps=[
                "identify all local events",
                "identify all trend segments",
                f"check if any event falls in a {trend_type} segment",
            ],
            step_results=[
                f"Events: {event_summary}",
                f"Segments: {seg_summary}",
            ],
            combination=check_line,
            verdict=verdict,
        )

        if verdict == "yes":
            answer_text = random.choice(B3_ANSWERS['yes']).format(
                metric=self.metric, trend_type=trend_type,
                event_type=found_event['type'],
                event_pos=found_event['position'],
                seg_start=found_seg[0], seg_end=found_seg[1],
            )
        else:
            answer_text = random.choice(B3_ANSWERS['no']).format(
                metric=self.metric, trend_type=trend_type,
            )

        return self._make_result(
            random.choice(B3_QUESTIONS).format(
                metric=self.metric, trend_type=trend_type,
            ),
            think, answer_text, 'rl_event_in_trend_type', verdict,
        )

    # --- B4: Extremum in context ---

    def generate_extremum_in_context(self) -> Optional[Dict[str, Any]]:
        """B4: find + locate in context. (C+E → R22 pattern)"""
        if len(self.trend_list) < 2:
            return None

        max_pos = int(np.argmax(self.ts))
        max_val = round(float(self.ts[max_pos]), 2)

        # Find containing segment
        seg_type, start, end = None, None, None
        for t, s, e in self.trend_list:
            if s <= max_pos <= e:
                seg_type, start, end = t, s, e
                break
        if seg_type is None:
            return None

        trend_type = random.choice(_ALL_TREND_TYPES)
        verdict = "yes" if seg_type == trend_type else "no"

        # Segment summary for think block
        seg_summary = ", ".join(
            f"{t} [{s}-{e}]" for t, s, e in self.trend_list
        )

        think = _build_bridge_think(
            plan_steps=[
                "find the global maximum and its position",
                "identify which trend segment contains that position",
                f"check if that segment is {trend_type}",
            ],
            step_results=[
                f"Global maximum: {_fmt(max_val)} at index {max_pos}",
                f"Segments: {seg_summary}",
                f"Index {max_pos} is in segment: {seg_type} [{start}-{end}]",
            ],
            combination=(
                f"Is {seg_type} == {trend_type}? → {verdict}"
            ),
            verdict=verdict,
        )

        answer_text = random.choice(B4_ANSWERS[verdict]).format(
            metric=self.metric, max_val=_fmt(max_val), max_pos=max_pos,
            seg_type=seg_type, trend_type=trend_type,
            start=start, end=end,
        )
        return self._make_result(
            random.choice(B4_QUESTIONS).format(
                metric=self.metric, trend_type=trend_type,
            ),
            think, answer_text, 'rl_max_in_trend_type', verdict,
        )

    # --- B5: Periodicity cross ---

    def generate_periodicity_cross(self) -> Optional[Dict[str, Any]]:
        """B5: cross-category comparison. (G+B → R33 pattern)"""
        if not self.has_period:
            return None

        std_val = round(float(np.std(self.ts)), 2)
        verdict = "yes" if self.amplitude > std_val else "no"

        freq_type = self.seasonal.get('frequency_type', 'unknown')
        period_r = round(self.period, 1)

        think = _build_bridge_think(
            plan_steps=[
                "determine the periodic amplitude",
                "compute the overall standard deviation",
                "compare amplitude vs std",
            ],
            step_results=[
                f"Periodic amplitude: {_fmt(self.amplitude)} (period {period_r}, {freq_type})",
                f"Standard deviation: {_fmt(std_val)}",
            ],
            combination=(
                f"Comparison: {_fmt(self.amplitude)} "
                f"{'>' if self.amplitude > std_val else '<='} "
                f"{_fmt(std_val)} → {verdict}"
            ),
            verdict=verdict,
        )

        answer_text = random.choice(B5_ANSWERS[verdict]).format(
            metric=self.metric,
            amplitude=_fmt(self.amplitude), std=_fmt(std_val),
        )
        return self._make_result(
            random.choice(B5_QUESTIONS).format(metric=self.metric),
            think, answer_text, 'rl_amplitude_vs_std', verdict,
        )

    # --- B6: Enumerate → filter → count ---

    def generate_enumerate_filter_count(self) -> Optional[Dict[str, Any]]:
        """B6: enumerate → filter → count. (F → rl_event_count_by_type pattern)

        Teaches the archetype: enumerate a collection, apply a predicate,
        count the matching subset. Covers event/segment counting RL types.
        """
        if not self.events or len(self.events) < 2:
            return None

        types_present = sorted({ev['type'] for ev in self.events})
        if not types_present:
            return None
        event_type = random.choice(types_present)

        filtered = [ev for ev in self.events if ev['type'] == event_type]
        count = len(filtered)

        event_summary = ", ".join(
            f"{ev['type']} at {ev['position']}" for ev in self.events
        )
        if filtered:
            filtered_summary = ", ".join(
                f"{ev['type']} at {ev['position']}" for ev in filtered
            )
        else:
            filtered_summary = "(none)"

        think = _build_bridge_think(
            plan_steps=[
                "enumerate all local events",
                f"filter to events of type {event_type}",
                "count the filtered set",
            ],
            step_results=[
                f"All events: {event_summary}",
                f"Filtered ({event_type}): {filtered_summary}",
            ],
            combination=f"Count of {event_type} events: {count}",
            verdict=str(count),
        )

        answer_text = (
            f"There are {count} local events of type {event_type} in {self.metric}."
        )
        return self._make_result(
            random.choice(B6_QUESTIONS).format(
                metric=self.metric, event_type=event_type,
            ),
            think, answer_text, 'rl_event_count_by_type', count,
        )

    # --- B7: Enumerate → argmax by property → extract ---

    def generate_enumerate_argmax_property(self) -> Optional[Dict[str, Any]]:
        """B7: enumerate → argmax by duration → extract type.
        (E → rl_type_of_longest pattern)

        Teaches the archetype: enumerate items, compute a sub-property,
        select the item maximizing it, extract a different attribute.
        """
        if len(self.trend_list) < 2:
            return None

        seg_with_dur = [(t, s, e, e - s) for t, s, e in self.trend_list]
        longest = max(seg_with_dur, key=lambda x: x[3])
        longest_type = longest[0]
        max_dur = longest[3]

        duration_lines = [
            f"segment [{s}-{e}]: type={t}, duration={dur}"
            for t, s, e, dur in seg_with_dur
        ]
        argmax_line = (
            f"Longest: [{longest[1]}-{longest[2]}] "
            f"with duration={max_dur}"
        )

        think = _build_bridge_think(
            plan_steps=[
                "enumerate all trend segments with their durations",
                "find the segment with the largest duration",
                "extract its trend type",
            ],
            step_results=[
                *duration_lines,
                argmax_line,
            ],
            combination=f"Type of longest segment: {longest_type}",
            verdict=longest_type,
        )

        answer_text = (
            f"The longest trend segment in {self.metric} is of type {longest_type}."
        )
        return self._make_result(
            random.choice(B7_QUESTIONS).format(metric=self.metric),
            think, answer_text, 'rl_type_of_longest', longest_type,
        )

    # --- B8: Chunk → per-chunk stat → meta-stat ---

    def generate_chunk_monotonic(self) -> Optional[Dict[str, Any]]:
        """B8: chunk → per-chunk mean → check monotonic.
        (A → rl_monotonic_chunks pattern)

        Teaches the archetype: chunk a series, compute a per-chunk stat,
        apply a meta-stat (monotonicity / stability / argmax) over them.
        """
        if self.seq_len < 64:
            return None

        chunks = _compute_chunks(self.ts, 0, self.seq_len - 1)
        chunk_means = [cm for _, _, cm, _ in chunks]
        if len(chunk_means) < 3:
            return None

        increasing = all(
            chunk_means[i] < chunk_means[i + 1]
            for i in range(len(chunk_means) - 1)
        )
        decreasing = all(
            chunk_means[i] > chunk_means[i + 1]
            for i in range(len(chunk_means) - 1)
        )
        is_monotonic = increasing or decreasing
        verdict = "yes" if is_monotonic else "no"

        chunk_lines = [
            f"chunk [{cs}, {ce}] mean = {_fmt(cm)}"
            for cs, ce, cm, _ in chunks
        ]

        n_show = min(len(chunk_means) - 1, 5)
        diff_lines = []
        for i in range(n_show):
            a, b = chunk_means[i], chunk_means[i + 1]
            diff = round(b - a, 2)
            if diff > 0:
                direction = "up"
            elif diff < 0:
                direction = "down"
            else:
                direction = "flat"
            diff_lines.append(
                f"chunk {i} → {i + 1}: {_fmt(a)} → {_fmt(b)} "
                f"({direction}, diff={_fmt(diff)})"
            )
        if len(chunk_means) - 1 > n_show:
            diff_lines.append(
                f"(... {len(chunk_means) - 1 - n_show} more transitions, same pattern ...)"
            )

        if is_monotonic:
            direction_name = "increasing" if increasing else "decreasing"
            meta_line = (
                f"All transitions go the same way ({direction_name}) → monotonic"
            )
        else:
            meta_line = "Transitions mix directions → not monotonic"

        think = _build_bridge_think(
            plan_steps=[
                "chunk the series and compute per-chunk means",
                "check whether consecutive chunk means strictly increase or strictly decrease",
                "determine monotonicity",
            ],
            step_results=[
                *chunk_lines,
                "",
                *diff_lines,
            ],
            combination=meta_line,
            verdict=verdict,
        )

        answer_text = random.choice(B8_ANSWERS[verdict]).format(metric=self.metric)
        return self._make_result(
            random.choice(B8_QUESTIONS).format(metric=self.metric),
            think, answer_text, 'rl_monotonic_chunks', verdict,
        )

    # --- B9: Enumerate → filter → sum → divide (plants ratio-of-aggregates) ---

    def generate_type_duration_fraction_bridge(self) -> Optional[Dict[str, Any]]:
        """B9: enumerate segments → filter by type → sum filtered durations → divide by total.
        (rl_type_duration_fraction pattern — plants the 'aggregate / aggregate = ratio'
        primitive that transfers to OOD event-density reasoning.)
        """
        if len(self.trend_list) < 2:
            return None

        present_types = sorted({t for t, _, _ in self.trend_list})
        if not present_types:
            return None
        trend_type = random.choice(present_types)

        # Match rl_type_duration_fraction's convention: duration = e - s
        seg_with_dur = [(t, s, e, e - s) for t, s, e in self.trend_list]
        total_dur = sum(d for _, _, _, d in seg_with_dur)
        if total_dur == 0:
            return None

        filtered = [(t, s, e, d) for (t, s, e, d) in seg_with_dur if t == trend_type]
        type_dur = sum(d for _, _, _, d in filtered)
        fraction = round(type_dur / total_dur, 2)

        enumeration_lines = [
            f"segment [{s}, {e}]: type={t}, duration={d}"
            for (t, s, e, d) in seg_with_dur
        ]
        if filtered:
            filtered_parts = ", ".join(
                f"[{s}, {e}] duration={d}" for (_, s, e, d) in filtered
            )
        else:
            filtered_parts = "(none)"
        sum_terms = " + ".join(str(d) for _, _, _, d in filtered) if filtered else "0"

        # Dual-vocabulary framing: "fraction" (mathematically exact) or
        # "density" (anchors the density/rate word to the division primitive
        # so OOD event-density questions have a word-level hook). Same
        # computation, same verdict, same eval_type — only vocabulary shifts.
        framing = random.choice(['fraction', 'density'])
        label = "Fraction" if framing == 'fraction' else "Density"
        plan_close = (
            f"divide by the total series duration to get the {framing}"
        )

        think = _build_bridge_think(
            plan_steps=[
                "enumerate all trend segments with their types and durations",
                f"filter to segments of type {trend_type}",
                "sum the durations of the filtered segments",
                plan_close,
            ],
            step_results=[
                *enumeration_lines,
                "",
                f"Filtered ({trend_type}): {filtered_parts}",
                f"Sum of {trend_type} durations: {sum_terms} = {type_dur}",
                f"Total duration: {total_dur}",
            ],
            combination=f"{label}: {type_dur} / {total_dur} = {_fmt(fraction)}",
            verdict=_fmt(fraction),
        )

        if framing == 'fraction':
            question = random.choice(B9_FRACTION_QUESTIONS)
            answer_text = random.choice(B9_FRACTION_ANSWERS).format(
                metric=self.metric, trend_type=trend_type, fraction=_fmt(fraction),
            )
        else:
            question = random.choice(B9_DENSITY_QUESTIONS)
            answer_text = random.choice(B9_DENSITY_ANSWERS).format(
                metric=self.metric, trend_type=trend_type, fraction=_fmt(fraction),
            )

        return self._make_result(
            question.format(metric=self.metric, trend_type=trend_type),
            think, answer_text, 'rl_type_duration_fraction', _fmt(fraction),
        )

    # --- Generate all ---

    def generate_all(self) -> List[Dict[str, Any]]:
        results = []
        generators = [
            self.generate_compare_interval_means,
            self.generate_stat_cross_compare,
            self.generate_event_in_context,
            self.generate_extremum_in_context,
            self.generate_periodicity_cross,
            self.generate_enumerate_filter_count,
            self.generate_enumerate_argmax_property,
            self.generate_chunk_monotonic,
            self.generate_type_duration_fraction_bridge,
        ]
        for gen_fn in generators:
            try:
                r = gen_fn()
                if r is not None:
                    results.append(r)
            except Exception:
                logger.warning(
                    f"Bridge {gen_fn.__name__} failed", exc_info=True,
                )
        return results
