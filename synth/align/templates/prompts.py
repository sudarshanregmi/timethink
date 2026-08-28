"""Prompt template registry for diverse instruction generation.

Each template is a Python format string. Placeholders use named fields
like {metric}, {perspectives}, {point}, {tolerance}, {metric_a}, {metric_b}.
The PromptRegistry selects a random template and fills it with the
provided context dictionary.
"""

import random
import re
from typing import Any, Dict, Optional

from synth.ts_generator.utils.probability_utils import QAType
from synth.align.templates.perturbations import TextAugmenter

_A_BEFORE_VOWEL = re.compile(r'\b(a)\s+([aeiou])', re.IGNORECASE)
_AN_BEFORE_CONSONANT = re.compile(r'\b(an)\s+([bcdfghjklmnpqrstvwxyz])', re.IGNORECASE)


def _fix_article(text: str) -> str:
    """Fix 'a'/'an' mismatches: 'a' → 'an' before vowels, 'an' → 'a' before consonants."""
    def _to_an(m: re.Match) -> str:
        an = 'An' if m.group(1)[0].isupper() else 'an'
        return f'{an} {m.group(2)}'
    def _to_a(m: re.Match) -> str:
        a = 'A' if m.group(1)[0].isupper() else 'a'
        return f'{a} {m.group(2)}'
    text = _A_BEFORE_VOWEL.sub(_to_an, text)
    text = _AN_BEFORE_CONSONANT.sub(_to_a, text)
    return text


def pluralize(count, singular: str, plural: str = None) -> str:
    """Return ``"is {count} {singular}"`` or ``"are {count} {plural}"``.

    Used in count-answer templates to keep is/are + singular/plural in sync
    with the verdict value. Accepts int or numeric string for count.
    """
    plural = plural or (singular + 's')
    try:
        n = int(count)
    except (TypeError, ValueError):
        n = None
    if n == 1:
        return f"is {count} {singular}"
    return f"are {count} {plural}"


DESCRIPTION_TEMPLATES = [
    # Formal / standard
    (
        "Now, please analyze the characteristics of {metric} from the "
        "perspectives of {perspectives}, and conclude the physical meaning "
        "of each in one sentence."
    ),
    (
        "Examine the behavior of {metric} focusing on {perspectives}. "
        "For each aspect, provide a one-sentence explanation of its physical significance."
    ),
    (
        "Provide a detailed analysis of {metric} covering {perspectives}. "
        "Summarize the physical meaning of each characteristic in one sentence."
    ),
    # Concise
    (
        "Describe {metric} in terms of {perspectives}. "
        "Explain the physical meaning of each in one sentence."
    ),
    (
        "Analyze {metric}: focus on {perspectives}. "
        "One sentence per aspect explaining its physical significance."
    ),
    # Casual / conversational
    (
        "What can you tell me about {metric}? Look at {perspectives} "
        "and explain the physical meaning of each in one sentence."
    ),
    (
        "Break down the behavior of {metric} — specifically {perspectives}. "
        "What does each aspect mean physically? Keep it to one sentence each."
    ),
    (
        "I'd like to understand {metric} better. Analyze {perspectives} "
        "and summarize what each means in physical terms (one sentence each)."
    ),
    # Technical
    (
        "Conduct a multi-perspective analysis of {metric} covering {perspectives}. "
        "Conclude each perspective with a single sentence on its physical interpretation."
    ),
    (
        "Evaluate the signal properties of {metric} with respect to {perspectives}. "
        "Provide a one-sentence physical interpretation for each."
    ),
]


# Compound description templates: all chosen metrics share the same perspectives.
# {perspectives} is a natural-language list (e.g. "trend and noise"),
# {metrics} is a natural-language list of metric names.
COMPOUND_DESCRIPTION_UNIFORM_TEMPLATES = [
    (
        "Describe the {perspectives} of {metrics}, and explain the physical "
        "meaning of each in one sentence."
    ),
    (
        "Analyze {metrics} across {perspectives}. For each aspect, give a "
        "one-sentence physical interpretation."
    ),
    (
        "Examine {perspectives} for {metrics}. Summarize the physical meaning "
        "of each in one sentence."
    ),
    (
        "Break down {metrics} focusing on {perspectives}, and note what each "
        "aspect means physically in one sentence."
    ),
    (
        "Characterize {metrics} in terms of {perspectives}, explaining the "
        "physical significance of each characteristic in a single sentence."
    ),
    (
        "Walk through the {perspectives} of {metrics}, and conclude each "
        "aspect with a one-sentence physical interpretation."
    ),
]


# Compound description templates: different perspectives per metric.
# {parts} is already a natural-language list like "the trend of A, the noise of B".
COMPOUND_DESCRIPTION_MIXED_TEMPLATES = [
    (
        "Describe {parts}, and explain the physical meaning of each in one sentence."
    ),
    (
        "Analyze {parts}. For each aspect, provide a one-sentence physical interpretation."
    ),
    (
        "Examine {parts}, and summarize the physical significance of each in one sentence."
    ),
    (
        "Break down {parts}, noting what each aspect means physically in a single sentence."
    ),
    (
        "Walk through {parts}, concluding each with a one-sentence physical interpretation."
    ),
    (
        "Give a focused analysis of {parts}, and explain each aspect's physical meaning in one sentence."
    ),
]

# Additional hint for questions that include local events
LOCAL_FORMAT_HINTS = [
    (
        " Answer format for local fluctuations: shake, position around point 125, "
        "amplitude 135.03. A sudden surge in public interest, likely due to significant news, "
        "a major event, or a trending topic related to the platform that rapidly captured "
        "user attention; small sudden decrease, position around point 102, amplitude 31.05. "
        "A slight increase in interest, possibly driven by minor news, promotions, or social media "
        "discussions that briefly captured attention without indicating a significant trend."
    ),
    (
        " When describing local fluctuations, use this format: [type], position around "
        "point [N], amplitude [value]. Then explain the physical meaning in one sentence."
    ),
    (
        " For each local event, state its type, approximate position, amplitude, "
        "and give a one-sentence explanation of the physical cause."
    ),
]

YES_NO_TEMPLATES = [
    # Standard
    (
        "Is there a local event starting around point {point} "
        "in {metric}{tolerance}?"
    ),
    # Variations
    (
        "Does {metric} show any local fluctuation near point {point}{tolerance}?"
    ),
    (
        "Check whether {metric} has a noticeable event around point {point}{tolerance}."
    ),
    (
        "Can you confirm if there is a fluctuation at or near point {point} "
        "in {metric}{tolerance}?"
    ),
    (
        "Around point {point}, does {metric} exhibit any local event or "
        "change{tolerance}?"
    ),
    (
        "Is there any significant local change in {metric} near point {point}{tolerance}?"
    ),
    (
        "Please determine if {metric} has a spike, dip, or other fluctuation "
        "around point {point}{tolerance}."
    ),
    (
        "Verify whether a local event occurs in {metric} at approximately "
        "point {point}{tolerance}."
    ),
]

# Type-specific yes/no questions — require {event_type} in context.
YES_NO_TYPED_TEMPLATES = [
    (
        "Is there a {event_type} near point {point} in {metric}{tolerance}?"
    ),
    (
        "Does {metric} exhibit a {event_type} around point {point}{tolerance}?"
    ),
    (
        "Can you confirm if a {event_type} occurs in {metric} at approximately "
        "point {point}{tolerance}?"
    ),
    (
        "Around point {point}, does {metric} show a {event_type}{tolerance}?"
    ),
    (
        "Check whether {metric} has a {event_type} near point {point}{tolerance}."
    ),
    (
        "Is a {event_type} detectable in {metric} around point {point}{tolerance}?"
    ),
    (
        "Verify if {metric} displays a {event_type} at or near point {point}{tolerance}."
    ),
    (
        "Does a {event_type} occur in {metric} around point {point}{tolerance}?"
    ),
]

# Local event property questions — richer than binary yes/no.
# Count variant: needs only {metric}.
# Direction/type variant: needs {metric} and {point}.
LOCAL_PROPERTY_TEMPLATES = [
    # Count
    (
        "How many distinct local fluctuations does {metric} exhibit in total?"
    ),
    (
        "Count the local events or spikes present in {metric}."
    ),
    (
        "How many local changes can you identify in {metric}?"
    ),
    # Direction near a point
    (
        "What is the direction of the local fluctuation in {metric} near point {point}{tolerance}? "
        "Is it upward or downward?"
    ),
    (
        "Is the local change in {metric} around point {point}{tolerance} an upward or a downward movement?"
    ),
    # Type identification near a point
    (
        "What type of local event occurs in {metric} near point {point}{tolerance}?"
    ),
    (
        "Describe the kind of local change that {metric} exhibits around point {point}{tolerance}."
    ),
    (
        "Identify the specific local fluctuation pattern in {metric} near point {point}{tolerance}."
    ),
]

CORRELATION_TEMPLATES = [
    # Standard (local correlation)
    (
        "Based on the characteristics of the time series, please describe "
        "the local events of {metric_a} and {metric_b}. "
        "And analyze whether there may be a correlation "
        "of fluctuation between them around point {point}{tolerance}. "
        "Conclude the physical meaning "
        "of the fluctuation correlation (or no correlation) in one sentence."
    ),
    (
        "Describe the local behavior of {metric_a} and {metric_b} near point {point}. "
        "Are their fluctuations correlated{tolerance}? "
        "Explain the physical significance in one sentence."
    ),
    (
        "Compare the local fluctuations of {metric_a} and {metric_b} around "
        "point {point}{tolerance}. Is there a correlation? "
        "Summarize the physical meaning in one sentence."
    ),
    (
        "Do {metric_a} and {metric_b} show synchronized fluctuations near "
        "point {point}{tolerance}? Describe each metric's local events "
        "and provide a one-sentence physical explanation."
    ),
    (
        "Analyze whether {metric_a} and {metric_b} exhibit correlated local "
        "changes around point {point}{tolerance}. "
        "Conclude with one sentence on the physical meaning."
    ),
    (
        "Look at {metric_a} and {metric_b} near point {point}. "
        "Do they fluctuate together{tolerance}? "
        "Explain why or why not in one sentence."
    ),
    (
        "Examine the local events in {metric_a} and {metric_b} around "
        "point {point}{tolerance}. Are they correlated? "
        "State the physical implication in one sentence."
    ),
    (
        "Are the fluctuations of {metric_a} and {metric_b} around "
        "point {point} related{tolerance}? Describe each and conclude "
        "the physical meaning in one sentence."
    ),
]

# Correlation templates with explicit position-based definition in the question.
CORRELATION_EXPLICIT_TEMPLATES = [
    (
        "Define correlation as: both {metric_a} and {metric_b} have local fluctuations "
        "starting near point {point}{tolerance}. Do they satisfy this definition?"
    ),
    (
        "Considering two metrics correlated when both show local changes starting around "
        "point {point}{tolerance}: are {metric_a} and {metric_b} correlated?"
    ),
    (
        "Assuming fluctuations starting at similar timestamps around point {point} are "
        "correlated{tolerance}: do {metric_a} and {metric_b} meet this criterion?"
    ),
    (
        "Under the criterion that correlated metrics both exhibit local events near the "
        "same point: does this hold for {metric_a} and {metric_b} around point {point}{tolerance}?"
    ),
    (
        "If we define correlation as synchronized local events near point {point}{tolerance}, "
        "are {metric_a} and {metric_b} correlated based on their behavior?"
    ),
    (
        "Two metrics are correlated if both show a local fluctuation starting around "
        "point {point}{tolerance}. Does this apply to {metric_a} and {metric_b}?"
    ),
]

# Type-aware correlation templates — require matching both timing AND event type.
CORRELATION_TYPED_TEMPLATES = [
    (
        "Consider {metric_a} and {metric_b} correlated only if both show local "
        "fluctuations of the same type near point {point}{tolerance}. "
        "Are they correlated under this stricter definition?"
    ),
    (
        "Under a strict correlation criterion requiring matching both timestep AND "
        "type of fluctuation near point {point}{tolerance}: are {metric_a} and {metric_b} correlated?"
    ),
    (
        "Analyze whether {metric_a} and {metric_b} are type-correlated around point {point}, "
        "where correlation requires both similar occurrence time and the same kind of local "
        "change{tolerance}."
    ),
    (
        "Do {metric_a} and {metric_b} show the same kind of local event near "
        "point {point}{tolerance}? Are they type-correlated at this point?"
    ),
    (
        "Determine if {metric_a} and {metric_b} have correlated local changes near "
        "point {point}, where correlation requires both similar timing and matching "
        "event type{tolerance}."
    ),
    (
        "Are the local events in {metric_a} and {metric_b} near point {point} "
        "correlated in both timing and type{tolerance}?"
    ),
]

SHAPE_CORRELATION_TEMPLATES = [
    # Standard (trend correlation)
    (
        "Based on the trend characteristics analyze whether there may be "
        "a correlation of trend between {metric_a} and {metric_b}.{tolerance}"
    ),
    (
        "Do {metric_a} and {metric_b} show similar trend patterns?{tolerance} "
        "Analyze and explain."
    ),
    (
        "Compare the overall trends of {metric_a} and {metric_b}.{tolerance} "
        "Are they correlated?"
    ),
    (
        "Analyze whether the trends of {metric_a} and {metric_b} move "
        "together.{tolerance}"
    ),
    (
        "Is there a trend correlation between {metric_a} and {metric_b}?{tolerance} "
        "Examine their trend characteristics."
    ),
    (
        "Look at how {metric_a} and {metric_b} trend over time.{tolerance} "
        "Are their trends related?"
    ),
    (
        "Evaluate the trend similarity between {metric_a} and {metric_b}.{tolerance}"
    ),
    (
        "Determine if {metric_a} and {metric_b} exhibit correlated trends "
        "based on their characteristics.{tolerance}"
    ),
]

CLUSTERING_LOCAL_TEMPLATES = [
    # Standard (local similarity)
    (
        "Based on the fluctuations in the metrics around point {point}, "
        "please find other metric(s) that may be related to {metric}"
        "{tolerance}, "
        "output their numbers, and explain the reasons. If related metrics are found, "
        "explain why they have similar **local fluctuations** in one sentence."
    ),
    (
        "Which other metrics fluctuate similarly to {metric} near point {point}"
        "{tolerance}? List them and explain the connection."
    ),
    (
        "Find metrics related to {metric} based on local events around "
        "point {point}{tolerance}. Explain why they are related."
    ),
    (
        "Are there other metrics that show fluctuations similar to {metric} "
        "near point {point}{tolerance}? Identify them and give a reason."
    ),
    (
        "Look at the fluctuations around point {point} and find metrics "
        "that behave like {metric}{tolerance}. Explain the similarity."
    ),
    (
        "Identify metrics whose local behavior near point {point} "
        "resembles {metric}{tolerance}. Output their numbers and explain."
    ),
    (
        "Near point {point}, which metrics are related to {metric} "
        "in terms of fluctuations{tolerance}? List and explain."
    ),
    (
        "Find other time series related to {metric} based on their "
        "behavior around point {point}{tolerance}. Explain the relationship."
    ),
]

# Type-aware local clustering — requires matching both position and event type.
CLUSTERING_TYPED_TEMPLATES = [
    (
        "Find other metrics that show the same type of local fluctuation as {metric} "
        "near point {point}, where both timing and event type must match{tolerance}. "
        "List them and explain."
    ),
    (
        "Identify metrics whose local events near point {point} match {metric} in "
        "both timing and type{tolerance}. Output their names and explain the connection."
    ),
    (
        "Which metrics exhibit the same kind of local change as {metric} around "
        "point {point}{tolerance}? Consider only metrics with matching event type as related."
    ),
    (
        "Looking at both position and type of local events near point {point}: which "
        "metrics show fluctuations similar in kind to {metric}{tolerance}?"
    ),
    (
        "Find time series related to {metric} based on their local events near "
        "point {point}, where relatedness requires matching both timing and event "
        "type{tolerance}. Explain."
    ),
    (
        "Near point {point}, which metrics share the same type of local fluctuation "
        "as {metric}{tolerance}? List and explain why they are related."
    ),
]

# End-position yes/no — checks whether a local event *ends* near a given point.
YES_NO_END_TEMPLATES = [
    (
        "Is there a local fluctuation ending around point {point} "
        "in {metric}{tolerance}?"
    ),
    (
        "Does {metric} have a local event that concludes near point {point}{tolerance}?"
    ),
    (
        "Can you confirm if a local change in {metric} finishes around "
        "point {point}{tolerance}?"
    ),
    (
        "Around point {point}, does any local fluctuation in {metric} end{tolerance}?"
    ),
    (
        "Check whether a local event in {metric} wraps up at or near "
        "point {point}{tolerance}."
    ),
    (
        "Is there a local change in {metric} whose ending is near "
        "point {point}{tolerance}?"
    ),
]

# End-position correlation — both metrics have events *ending* near the same point.
CORRELATION_END_TEMPLATES = [
    (
        "Do both {metric_a} and {metric_b} have local events ending near "
        "point {point}{tolerance}? Are their event endpoints correlated?"
    ),
    (
        "Compare whether {metric_a} and {metric_b} have local fluctuations "
        "that conclude near point {point}{tolerance}. Are their endings synchronized?"
    ),
    (
        "Looking at when local events end: do {metric_a} and {metric_b} both "
        "finish their fluctuations near point {point}{tolerance}?"
    ),
    (
        "Analyze whether {metric_a} and {metric_b} are correlated based on "
        "their local events ending near point {point}{tolerance}."
    ),
    (
        "Do the local fluctuations in {metric_a} and {metric_b} both conclude "
        "around point {point}{tolerance}? Assess their endpoint correlation."
    ),
    (
        "Check if local events in {metric_a} and {metric_b} both end near "
        "point {point}{tolerance}. Is there a correlation in their endpoints?"
    ),
]

# End-position clustering — find metrics with events *ending* near the same point.
CLUSTERING_END_TEMPLATES = [
    (
        "Find other metrics that have local events ending near point {point}, "
        "similar to {metric}{tolerance}. List them and explain."
    ),
    (
        "Which metrics have local fluctuations concluding near point {point}, "
        "like {metric}{tolerance}? Identify them and explain the connection."
    ),
    (
        "Based on when local events end near point {point}: which metrics "
        "show endings similar to {metric}{tolerance}?"
    ),
    (
        "Identify time series whose local events end around the same point as "
        "{metric} near point {point}{tolerance}. Explain the relationship."
    ),
    (
        "Near point {point}, which metrics share similar local event endpoints "
        "with {metric}{tolerance}? List them and explain."
    ),
    (
        "Find metrics related to {metric} based on their local events concluding "
        "near point {point}{tolerance}. Output their names and explain."
    ),
]

# ---------------------------------------------------------------------------
# Param-specific question templates (ask about structural parameters stored
# in context_info["params"] by each Change class in local_changes.py).
# ---------------------------------------------------------------------------

# Timing phase length — {phase_name} is filled with the param key label
# (e.g. "rise", "fall", "plateau", "drop", "spike").
LOCAL_PARAM_TIMING_TEMPLATES = [
    (
        "How many timesteps does the {phase_name} phase last in the local "
        "event near point {point}{tolerance} in {metric}?"
    ),
    (
        "What is the length of the {phase_name} phase for the event "
        "near point {point}{tolerance} in {metric}?"
    ),
    (
        "Near point {point}{tolerance} in {metric}, how long does the {phase_name} "
        "phase of the local change last?"
    ),
    (
        "How many data points span the {phase_name} phase of the event "
        "around point {point}{tolerance} in {metric}?"
    ),
    (
        "Estimate the duration of the {phase_name} phase in the local "
        "fluctuation near point {point}{tolerance} in {metric}."
    ),
    (
        "How long is the {phase_name} phase of the local event near "
        "point {point}{tolerance} in {metric}?"
    ),
]

# Amplitude / spike amplitude of a single local event.
LOCAL_PARAM_AMPLITUDE_TEMPLATES = [
    (
        "What is the amplitude of the local event near point {point}{tolerance} in {metric}?"
    ),
    (
        "Estimate the amplitude of the fluctuation around point {point}{tolerance} in {metric}."
    ),
    (
        "Near point {point}{tolerance} in {metric}, how large is the local change in terms "
        "of amplitude?"
    ),
    (
        "What is the magnitude of the local fluctuation near point {point}{tolerance} in {metric}?"
    ),
    (
        "How big is the amplitude of the event around point {point}{tolerance} in {metric}?"
    ),
    (
        "What amplitude does the local event near point {point}{tolerance} in {metric} exhibit?"
    ),
]

# Followup type — for SpikeFollowedBy* events where a spike is followed by
# a specific type of change (e.g. rise, fall, convex, concave).
LOCAL_PARAM_FOLLOWUP_TEMPLATES = [
    (
        "After the spike near point {point}{tolerance} in {metric}, what type of change "
        "follows it?"
    ),
    (
        "What happens to {metric} immediately after the spike around "
        "point {point}{tolerance}?"
    ),
    (
        "Near point {point}{tolerance} in {metric}, what kind of movement follows "
        "the spike in the local event?"
    ),
    (
        "Describe the followup change after the spike in the local event "
        "near point {point}{tolerance} in {metric}."
    ),
    (
        "What type of pattern does {metric} exhibit after the local spike "
        "around point {point}{tolerance}?"
    ),
    (
        "After the local spike at approximately point {point}{tolerance} in {metric}, "
        "is the followup a rise, fall, or plateau?"
    ),
]

# Number of spikes in a multi-spike / continuous-spike local event.
LOCAL_PARAM_COUNT_TEMPLATES = [
    (
        "How many spikes are present in the local event near point {point}{tolerance} "
        "in {metric}?"
    ),
    (
        "Count the spikes in the local fluctuation around point {point}{tolerance} "
        "in {metric}."
    ),
    (
        "Near point {point}{tolerance} in {metric}, how many individual spikes make "
        "up the local event?"
    ),
    (
        "What is the number of spikes contained in the local change near "
        "point {point}{tolerance} in {metric}?"
    ),
]

CLUSTERING_SHAPE_TEMPLATES = [
    # Standard (trend similarity)
    (
        "Based on the trends, find time series related to {metric}, "
        "output numbers and explain.{tolerance}"
    ),
    (
        "Which time series have trends similar to {metric}?{tolerance} "
        "List them and explain."
    ),
    (
        "Find other metrics whose trend patterns resemble {metric}.{tolerance} "
        "Output their numbers and explain."
    ),
    (
        "Identify time series that are trend-correlated with {metric}.{tolerance} "
        "Explain the relationship."
    ),
    (
        "Are there other time series with trends similar to {metric}?{tolerance} "
        "List them and give a reason."
    ),
    (
        "Look at the trend behavior and find metrics related to {metric}.{tolerance} "
        "Explain the connection."
    ),
    (
        "Which metrics show trend patterns similar to {metric}?{tolerance} "
        "Output numbers and reasoning."
    ),
    (
        "Based on trend analysis, identify time series resembling {metric}.{tolerance}"
    ),
]

SHAPE_ANTICORRELATION_TEMPLATES = [
    (
        "Based on the trend characteristics, analyze whether {metric_a} and {metric_b} "
        "show opposite trend patterns — where one increases, the other decreases, and "
        "steady segments are shared.{tolerance}"
    ),
    (
        "Do {metric_a} and {metric_b} exhibit anticorrelated trends?{tolerance} "
        "Examine each trend segment for opposite directions."
    ),
    (
        "Compare the trend segments of {metric_a} and {metric_b}.{tolerance} "
        "Are their trends consistently opposite (increase vs decrease)?"
    ),
    (
        "Analyze whether the trend of {metric_a} mirrors the opposite of {metric_b} "
        "segment by segment.{tolerance}"
    ),
    (
        "Is there an anticorrelation in trend between {metric_a} and {metric_b}?{tolerance} "
        "A segment that increases in one should decrease in the other."
    ),
    (
        "Look at how {metric_a} and {metric_b} trend over time.{tolerance} "
        "Do their trend directions oppose each other?"
    ),
    (
        "Evaluate whether {metric_a} and {metric_b} are trend-anticorrelated — "
        "opposite directions in each segment.{tolerance}"
    ),
    (
        "Determine if {metric_a} and {metric_b} exhibit opposite trend behavior "
        "based on their segment characteristics.{tolerance}"
    ),
]

ANTICLUSTERING_SHAPE_TEMPLATES = [
    (
        "Based on the trends, find time series with opposite trend patterns to {metric}, "
        "output numbers and explain.{tolerance}"
    ),
    (
        "Which time series have trends opposite to {metric}?{tolerance} "
        "List them and explain."
    ),
    (
        "Find other metrics whose trend directions are reversed compared to {metric}.{tolerance} "
        "Output their numbers and explain."
    ),
    (
        "Identify time series that are trend-anticorrelated with {metric}.{tolerance} "
        "Explain the relationship."
    ),
    (
        "Are there other time series with trends opposite to {metric}?{tolerance} "
        "List them and give a reason."
    ),
    (
        "Look at the trend behavior and find metrics that move opposite to {metric}.{tolerance} "
        "Explain the connection."
    ),
    (
        "Which metrics show trend patterns opposite to {metric}?{tolerance} "
        "Output numbers and reasoning."
    ),
    (
        "Based on trend analysis, identify time series that anticorrelate with {metric}.{tolerance}"
    ),
]

TREND_TEMPLATES = [
    "Analyze the trend of {metric} in detail.",
    "Describe the trend behavior of {metric}.",
    "What is the overall trend of {metric}? Analyze in detail.",
    "Examine the trend pattern of {metric} and describe it.",
    "How does {metric} trend over time? Provide a detailed analysis.",
    "Break down the trend of {metric} step by step.",
    "Look at {metric} and describe its trend characteristics.",
    "Characterize the trend of {metric} in detail.",
]


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

# Map QAType (and mode-specific sub-keys) to template pools
_TEMPLATE_MAP = {
    QAType.DESCRIPTION: DESCRIPTION_TEMPLATES,
    QAType.YES_NO: YES_NO_TEMPLATES,
    QAType.CORRELATION: CORRELATION_TEMPLATES,
    QAType.CLUSTERING: CLUSTERING_LOCAL_TEMPLATES,
    # Local event specific sub-keys
    "local_yes_no_typed": YES_NO_TYPED_TEMPLATES,
    "local_property": LOCAL_PROPERTY_TEMPLATES,
    "local_property_count": LOCAL_PROPERTY_TEMPLATES[:3],
    "local_property_direction": LOCAL_PROPERTY_TEMPLATES[3:5],
    "local_property_type_id": LOCAL_PROPERTY_TEMPLATES[5:],
    "local_corr_explicit": CORRELATION_EXPLICIT_TEMPLATES,
    "local_corr_typed": CORRELATION_TYPED_TEMPLATES,
    "local_cluster_typed": CLUSTERING_TYPED_TEMPLATES,
    "local_yes_no_end": YES_NO_END_TEMPLATES,
    "local_corr_end": CORRELATION_END_TEMPLATES,
    "local_cluster_end": CLUSTERING_END_TEMPLATES,
    # Param-specific question sub-keys
    "local_param_timing": LOCAL_PARAM_TIMING_TEMPLATES,
    "local_param_amplitude": LOCAL_PARAM_AMPLITUDE_TEMPLATES,
    "local_param_followup": LOCAL_PARAM_FOLLOWUP_TEMPLATES,
    "local_param_count": LOCAL_PARAM_COUNT_TEMPLATES,
    # Shape/trend sub-keys
    "shape_correlation": SHAPE_CORRELATION_TEMPLATES,
    "shape_clustering": CLUSTERING_SHAPE_TEMPLATES,
    "shape_anticorrelation": SHAPE_ANTICORRELATION_TEMPLATES,
    "shape_anticlustering": ANTICLUSTERING_SHAPE_TEMPLATES,
    "trend": TREND_TEMPLATES,
}


class PromptRegistry:
    """Central registry for selecting diverse prompt templates."""

    @staticmethod
    def get_prompt(
        qa_type: str,
        context: Dict[str, Any],
        augment: bool = True,
        sub_key: Optional[str] = None,
    ) -> str:
        """Select a random template and fill it with context.

        Args:
            qa_type: QAType enum value (e.g. QAType.DESCRIPTION)
            context: Dict of named fields to fill into the template
            augment: Whether to apply text perturbation
            sub_key: Optional sub-key for mode-specific template pools
                     (e.g. 'shape_correlation', 'shape_clustering', 'trend')

        Returns:
            Formatted prompt string
        """
        key = sub_key if sub_key and sub_key in _TEMPLATE_MAP else qa_type
        templates = _TEMPLATE_MAP.get(key, DESCRIPTION_TEMPLATES)
        template = random.choice(templates)

        try:
            prompt = template.format(**context)
        except KeyError:
            # Fallback: return template with unfilled placeholders stripped
            prompt = template

        prompt = _fix_article(prompt)

        if augment:
            prompt = TextAugmenter.augment(prompt, prob=0.2)

        return prompt

    @staticmethod
    def get_local_hint() -> str:
        """Get a random local fluctuation format hint string."""
        return random.choice(LOCAL_FORMAT_HINTS)
