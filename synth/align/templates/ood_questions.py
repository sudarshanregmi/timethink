"""Question templates for out-of-distribution (OOD) evaluation types.

These questions compose 2+ trained skills in novel combinations.
Used for eval-only — never in training data.
"""

# --- ood_conditional_stat ---
OOD_CONDITIONAL_STAT_MEAN = [
    "What is the mean of {metric} considering only the time points where the trend is {trend_type}?",
    "Compute the mean value of {metric} restricted to {trend_type} segments.",
    "If we isolate only the {trend_type} portions of {metric}, what is the mean?",
    "What average value does {metric} take during {trend_type} segments?",
]

OOD_CONDITIONAL_STAT_STD = [
    "What is the standard deviation of {metric} during {trend_type} segments only?",
    "Compute the std of {metric} restricted to the {trend_type} portions.",
    "If we consider only the {trend_type} segments of {metric}, what is the standard deviation?",
]

OOD_CONDITIONAL_STAT_RANGE = [
    "What is the value range of {metric} during {trend_type} segments only?",
    "Compute the range (max minus min) of {metric} restricted to {trend_type} portions.",
    "If we isolate only the {trend_type} segments of {metric}, what is the range?",
]

OOD_CONDITIONAL_STAT_TEMPLATES = {
    'conditional_mean': OOD_CONDITIONAL_STAT_MEAN,
    'conditional_std': OOD_CONDITIONAL_STAT_STD,
    'conditional_range': OOD_CONDITIONAL_STAT_RANGE,
}

# --- ood_nested_extrema ---
OOD_NESTED_EXTREMA_MAX_IN_LONGEST = [
    "What is the maximum amplitude of any local event that occurs within the longest trend segment of {metric}?",
    "In the longest trend segment of {metric}, what is the highest local event amplitude?",
    "Find the longest trend segment in {metric} and report the maximum event amplitude within it.",
]

OOD_NESTED_EXTREMA_MIN_IN_SHORTEST = [
    "What is the minimum amplitude of any local event within the shortest trend segment of {metric}?",
    "In the shortest trend segment of {metric}, what is the lowest local event amplitude?",
    "Find the shortest trend segment in {metric} and report the minimum event amplitude within it.",
]

OOD_NESTED_EXTREMA_TEMPLATES = {
    'max_amp_in_longest': OOD_NESTED_EXTREMA_MAX_IN_LONGEST,
    'min_amp_in_shortest': OOD_NESTED_EXTREMA_MIN_IN_SHORTEST,
}

# --- ood_event_density ---
OOD_EVENT_DENSITY_HIGHEST = [
    "Which trend type (increase, decrease, or keep steady) has the highest density of local events in {metric}? Density is events per unit time.",
    "In {metric}, during which trend type do local events occur most frequently relative to the duration of that trend?",
    "Considering events per unit time, which trend direction in {metric} has the densest concentration of local events?",
]

OOD_EVENT_DENSITY_LOWEST = [
    "Which trend type (increase, decrease, or keep steady) has the lowest density of local events in {metric}?",
    "In {metric}, during which trend type do local events occur least frequently per unit time?",
    "Considering events per unit time, which trend direction in {metric} has the sparsest local events?",
]

OOD_EVENT_DENSITY_TEMPLATES = {
    'highest_density': OOD_EVENT_DENSITY_HIGHEST,
    'lowest_density': OOD_EVENT_DENSITY_LOWEST,
}

# --- ood_conditional_count ---
OOD_CONDITIONAL_COUNT_ABOVE = [
    "How many {trend_type} segments of {metric} have a segment mean above the overall series mean?",
    "Count the number of {trend_type} segments in {metric} whose mean exceeds the global mean.",
    "In {metric}, how many {trend_type} segments have an average value higher than the overall mean?",
]

OOD_CONDITIONAL_COUNT_BELOW = [
    "How many {trend_type} segments of {metric} have a segment mean below the overall series mean?",
    "Count the number of {trend_type} segments in {metric} whose mean is lower than the global mean.",
    "In {metric}, how many {trend_type} segments have an average value lower than the overall mean?",
]

OOD_CONDITIONAL_COUNT_TEMPLATES = {
    'count_above_mean': OOD_CONDITIONAL_COUNT_ABOVE,
    'count_below_mean': OOD_CONDITIONAL_COUNT_BELOW,
}

# --- ood_trend_reversal ---
OOD_TREND_REVERSAL = [
    "After the largest behavioral change point in {metric}, does the dominant trend direction reverse compared to before the change point?",
    "In {metric}, find the largest level shift between adjacent segments. Does the dominant trend change from before to after that point?",
    "Does the dominant trend in {metric} reverse after its most significant change point?",
    "At the point of greatest behavioral change in {metric}, is the dominant trend before the change point different from after?",
]

# --- ood_range_normalized_amplitude ---
OOD_RANGE_NORMALIZED_AMPLITUDE = [
    "Is the largest local event amplitude in {metric} greater than half of the total value range?",
    "In {metric}, does the maximum event amplitude exceed 50% of the overall range (max minus min)?",
    "Considering {metric}, is the peak local event amplitude more than half the signal's total range?",
    "Does the strongest local event in {metric} have an amplitude exceeding half the value range?",
]

# --- ood_segment_stat_compare ---
OOD_SEGMENT_STAT_COMPARE = [
    "Is the mean value of the first trend segment in {metric} greater than the mean of the last trend segment?",
    "In {metric}, does the first trend segment have a higher average than the final trend segment?",
    "Compare the first and last trend segments of {metric}: is the mean of the first segment greater?",
    "For {metric}, is the average value during the first trend segment higher than during the last?",
]
