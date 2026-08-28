"""UTS question banks.

All question banks for UTS QA families (1–9) live here so that logic files
(base_generator.py) stay free of inline data.  Follow the same pattern as
mts_questions.py.
"""

# ── Families 1–3: segment-dominance, transition, count, type-change ────────

_TREND_DOMINANCE_QUESTIONS = [
    "In {metric}, between points {a} and {b}, which trend behavior dominates: increasing, decreasing, or steady?",
    "For {metric} from point {a} to point {b}, what is the dominant trend direction?",
    "Looking at {metric} in the range [{a}, {b}], which trend type covers the most: increasing, decreasing, or steady?",
    "In the range from point {a} to {b} of {metric}, does an upward, downward, or flat trend dominate?",
    "Considering {metric} between points {a} and {b}, which trend pattern is most prevalent?",
]

# ── Family 6: compound judgment question banks ──────────────────────────────

_JUDGMENT_TREND_LOCAL_QUESTIONS = [
    "A '{label}' is defined as a {direction} event with amplitude greater than {threshold:.2f} occurring during an overall {req_trend} trend. Based on this definition, does {metric} contain a '{label}'?",
    "In {metric}, does a '{label}' occur? A '{label}' requires: (1) overall {req_trend} trend AND (2) a {direction} event with amplitude > {threshold:.2f}.",
    "If a '{label}' is defined as any {direction} activity exceeding amplitude {threshold:.2f} during a {req_trend} trend, is there evidence of a '{label}' in {metric}?",
]

_JUDGMENT_MULTI_PHASE_QUESTIONS = [
    "A '{label}' is defined as a time series whose trend follows the phase sequence: {phase_seq}. Does {metric} exhibit a '{label}'?",
    "In {metric}, does the trend follow a '{label}' pattern, defined as phases progressing in order: {phase_seq}?",
    "If '{label}' requires a trend progression of {phase_seq}, does {metric} match this pattern?",
]

_JUDGMENT_NOISE_TREND_QUESTIONS = [
    "A '{label}' is defined as a {noise_cat} signal (noise strength > {threshold:.2f}) during an overall {req_trend} trend. Does {metric} exhibit '{label}'?",
    "In {metric}, define '{label}' as noise strength exceeding {threshold:.2f} AND overall {req_trend} trend. Does {metric} satisfy this definition?",
    "If '{label}' requires noise strength > {threshold:.2f} during a {req_trend} trend, does {metric} qualify?",
]

_JUDGMENT_STAT_QUESTIONS = [
    "If any reading {direction} {threshold:.2f} in {metric} is considered anomalous, does the data contain an anomaly?",
    "For {metric}, define an anomaly as any value {direction} {threshold:.2f}. Is there an anomalous reading in the data?",
    "Does {metric} contain any data point {direction} {threshold:.2f}?",
]

# ── Family 6b: cross-attribute judgment question banks ──────────────────────

_JUDGMENT_AMPLITUDE_VS_NOISE_QUESTIONS = [
    "A '{label}' is a local event whose amplitude exceeds {K} times the background noise strength of {noise_strength}. Does {metric} contain a '{label}'?",
    "In {metric}, does any local event qualify as a '{label}'? A '{label}' requires amplitude > {K} x noise strength ({noise_strength}).",
    "If a '{label}' is defined as a local event with amplitude exceeding {K} times the noise strength of {noise_strength}, does {metric} contain one?",
]

_JUDGMENT_AMPLITUDE_VS_STD_QUESTIONS = [
    "An '{label}' is a local event whose amplitude exceeds {K} times the series standard deviation of {std}. Does {metric} contain an '{label}'?",
    "In {metric}, does any local event qualify as an '{label}'? An '{label}' requires amplitude > {K} x std ({std}).",
    "If an '{label}' is defined as a local event with amplitude exceeding {K} times the standard deviation of {std}, does {metric} contain one?",
]

_JUDGMENT_RANGE_VS_STD_QUESTIONS = [
    "A '{label}' is a series whose value range exceeds {K} times its standard deviation. Based on the statistics, does {metric} qualify?",
    "Does {metric} exhibit a '{label}'? A '{label}' requires the value range (max - min) to exceed {K} times the standard deviation.",
    "If a '{label}' is when the total value range is more than {K} times the standard deviation, does {metric} qualify?",
]

_JUDGMENT_HALF_VOLATILITY_QUESTIONS = [
    "A '{label}' is when the second-half standard deviation is at least {K} times the first-half standard deviation. Does {metric} exhibit a '{label}'?",
    "Does {metric} show a '{label}'? A '{label}' requires the second-half std to be at least {K} times the first-half std.",
    "If a '{label}' means the second-half std must reach {K} times the first-half std, does {metric} qualify?",
]

_JUDGMENT_HALF_MEAN_SHIFT_QUESTIONS = [
    "A '{label}' occurs when the second-half mean exceeds the first-half mean by more than {K}% of the total value range. Does {metric} exhibit a '{label}'?",
    "Does {metric} show a '{label}'? A '{label}' requires the half-mean shift to exceed {K}% of the value range.",
    "If a '{label}' is when the difference between second-half and first-half means is more than {K}% of the total range, does {metric} qualify?",
]

_JUDGMENT_WINDOWED_MONOTONICITY_QUESTIONS = [
    "A '{label}' is a signal where each successive 16-step window mean is strictly {direction} than the previous. Does {metric} exhibit a '{label}'?",
    "Does {metric} show a '{label}'? A '{label}' requires all consecutive 16-step window means to be strictly {direction}.",
    "If a '{label}' means every 16-step window mean is {direction} than the one before it, does {metric} qualify?",
]

_JUDGMENT_PHASE_EVENT_QUESTIONS = [
    "A '{label}' is defined as an {direction} event occurring during a {trend_type} trend phase. Does {metric} contain a '{label}'?",
    "In {metric}, does a '{label}' occur? A '{label}' requires an {direction} local event within a {trend_type} trend segment.",
    "If a '{label}' is an {direction} fluctuation during a {trend_type} phase, does {metric} exhibit one?",
]

_JUDGMENT_SEQUENTIAL_EVENTS_QUESTIONS = [
    "A '{label}' is a {event_a} followed within {gap} timesteps by a {event_b}. Does {metric} exhibit a '{label}'?",
    "Does {metric} contain a '{label}'? A '{label}' requires a {event_a} followed by a {event_b} within {gap} steps.",
    "If a '{label}' is defined as a {event_a} occurring within {gap} timesteps before a {event_b}, does {metric} exhibit one?",
]

# ── Family 8: segment-comparison question banks ─────────────────────────────

# ── Family 9: Statistical Numerical QA question banks ──────────────────────

_STAT_MIN_VAL_QUESTIONS = [
    "What is the minimum value recorded in {metric}?",
    "What is the global minimum of {metric} across the entire series?",
    "Report the lowest value that {metric} reaches.",
    "What is the smallest data point in {metric}?",
    "What is the floor value of {metric} across all time steps?",
]

_STAT_MAX_VAL_QUESTIONS = [
    "What is the maximum value recorded in {metric}?",
    "What is the global maximum of {metric} across the entire series?",
    "Report the highest value that {metric} reaches.",
    "What is the largest data point in {metric}?",
    "What is the peak value of {metric} across all time steps?",
]

_STAT_MEAN_VAL_QUESTIONS = [
    "What is the mean (average) value of {metric} across the entire series?",
    "Compute the average of {metric} over all time steps.",
    "What is the overall mean of {metric}?",
    "Report the global average value of {metric}.",
    "What is the arithmetic mean of {metric} from start to end?",
]

_STAT_STD_VAL_QUESTIONS = [
    "What is the standard deviation of {metric} across the entire series?",
    "Compute the overall standard deviation of {metric}.",
    "Report the spread of {metric}'s values expressed as standard deviation.",
    "What is the global standard deviation of {metric}?",
    "How volatile is {metric} on average, measured by standard deviation?",
]

_STAT_PEAK_QUESTIONS = [
    "At which index does {metric} reach its global peak (maximum value)?",
    "At what point in the series does {metric} achieve its highest value?",
    "Identify the position (0-indexed) of the maximum value in {metric}.",
    "At which step does {metric} hit its highest recorded value?",
    "Where does {metric} reach its global maximum? Give the 0-indexed position.",
]

_STAT_TROUGH_QUESTIONS = [
    "At which index does {metric} reach its global trough (minimum value)?",
    "At what point in the series does {metric} achieve its lowest value?",
    "Identify the position (0-indexed) of the minimum value in {metric}.",
    "At which step does {metric} hit its lowest recorded value?",
    "Where does {metric} reach its global minimum? Give the 0-indexed position.",
]

_STAT_WINDOWED_MEAN_FULL_QUESTIONS = [
    "What is the mean of {metric} for each consecutive {window}-step window from start to end?",
    "Break {metric} into non-overlapping chunks of {window} steps and report the mean of each chunk.",
    "Compute the average value of {metric} in every {window}-step interval across the full series.",
    "Report the mean of {metric} for each {window}-step segment spanning the entire series.",
    "What is the average of {metric} in each {window}-step block from the beginning to the end?",
]

_STAT_WINDOWED_MEAN_PARTIAL_QUESTIONS = [
    "What is the mean of {metric} for each {window}-step window from point {start} to point {end}?",
    "Between points {start} and {end} in {metric}, report the mean for each {window}-step segment.",
    "From point {start} to {end}, compute the average of {metric} in every {window}-step chunk.",
    "For {metric} in the range [{start}, {end}], what is the mean of each {window}-step interval?",
]

_STAT_THRESHOLD_ABOVE_MEAN_QUESTIONS = [
    "For how many time steps does {metric} stay above its mean value of {threshold}?",
    "How many data points in {metric} exceed the mean of {threshold}?",
    "In {metric}, count the number of steps where the value is above the mean ({threshold}).",
    "What is the count of time steps in {metric} that are strictly greater than the mean {threshold}?",
    "How many of the {seq_len} steps in {metric} have a value above {threshold} (the series mean)?",
]

_STAT_THRESHOLD_ABOVE_STD_QUESTIONS = [
    "How many time steps in {metric} exceed {threshold} (mean plus one standard deviation)?",
    "For how many steps does {metric} rise above {threshold}, which is its mean + 1 standard deviation?",
    "Count the number of data points in {metric} that exceed {threshold} (μ + σ).",
    "In {metric}, how many values surpass the threshold of {threshold} (one std above the mean)?",
]

_STAT_THRESHOLD_BELOW_STD_QUESTIONS = [
    "How many time steps in {metric} fall below {threshold} (mean minus one standard deviation)?",
    "For how many steps does {metric} drop below {threshold}, which is its mean - 1 standard deviation?",
    "Count the number of data points in {metric} below {threshold} (μ - σ).",
    "In {metric}, how many values are below the threshold of {threshold} (one std below the mean)?",
]

_STAT_MEAN_CROSSING_QUESTIONS = [
    "How many times does {metric} cross its mean value of {mean} throughout the series?",
    "Count the number of times {metric} transitions across its mean ({mean}).",
    "How often does {metric} switch from above-mean to below-mean (or vice versa)?",
    "What is the total number of mean crossings in {metric} (mean = {mean})?",
    "In {metric}, how many sign changes occur relative to the series mean of {mean}?",
]

_STAT_RANGE_QUESTIONS = [
    "What is the total value range (maximum minus minimum) of {metric}?",
    "Compute the range of {metric}: the difference between its highest and lowest recorded values.",
    "What is the peak-to-trough amplitude of {metric}?",
    "How large is the spread of {metric}'s values from its minimum to its maximum?",
    "For {metric}, what is max - min?",
]

_STAT_SEGMENT_COMPARE_MEAN_QUESTIONS = [
    "Does the first half or the second half of {metric} have a higher average value?",
    "Comparing the first and second halves of {metric}, which half has a greater mean?",
    "Between the first half (points 0-{mid}) and the second half (points {mid1}-{end}) of {metric}, which has a higher average?",
    "Is the mean of {metric} higher in its first half or its second half?",
    "Split {metric} into two equal halves. Which half has the larger mean?",
]

_STAT_SEGMENT_COMPARE_STD_QUESTIONS = [
    "Is the first half or the second half of {metric} more volatile (higher standard deviation)?",
    "Which half of {metric} shows more variability: the first half or the second half?",
    "Comparing spread: does the first half or second half of {metric} have a higher standard deviation?",
    "Between the first half and second half of {metric}, which is more noisy (higher std)?",
    "Split {metric} into two equal halves. Which half exhibits greater volatility?",
]

# ── Family 11E: Local event enumeration question banks ─────────────────────

_LOCAL_COUNT_BY_TYPE_QUESTIONS = [
    "How many {target_type} events are present in {metric}?",
    "In {metric}, count the number of local events whose type is {target_type}.",
    "How many local fluctuations of type {target_type} occur in {metric}?",
    "For {metric}, what is the total count of {target_type} events?",
]

_LOCAL_COUNT_BY_DIRECTION_QUESTIONS = [
    "How many local events in {metric} have a {target_direction} direction?",
    "In {metric}, count the local fluctuations with direction {target_direction}.",
    "How many {target_direction}-directed local events are found in {metric}?",
    "For {metric}, what is the count of local events whose direction is {target_direction}?",
]

_LOCAL_COUNT_BY_AMPLITUDE_QUESTIONS = [
    "How many local events in {metric} have amplitude greater than {threshold}?",
    "In {metric}, count the local fluctuations with amplitude exceeding {threshold}.",
    "How many local events in {metric} surpass an amplitude of {threshold}?",
    "For {metric}, how many local events have amplitude above {threshold}?",
]

_LOCAL_COUNT_BY_TYPE_AND_AMP_QUESTIONS = [
    "How many {target_type} events in {metric} have amplitude greater than {threshold}?",
    "In {metric}, count the {target_type} events with amplitude exceeding {threshold}.",
    "How many {target_type} local fluctuations in {metric} surpass amplitude {threshold}?",
    "For {metric}, how many {target_type} events have amplitude above {threshold}?",
]

_LOCAL_HIGHEST_AMPLITUDE_QUESTIONS = [
    "Which local event in {metric} has the highest amplitude?",
    "In {metric}, identify the local fluctuation with the greatest amplitude.",
    "What is the peak amplitude among all local events in {metric}, and which event achieves it?",
    "For {metric}, which local event exhibits the largest amplitude?",
]

_LOCAL_LOWEST_AMPLITUDE_QUESTIONS = [
    "Which local event in {metric} has the lowest amplitude?",
    "In {metric}, identify the local fluctuation with the smallest amplitude.",
    "What is the minimum amplitude among all local events in {metric}, and which event achieves it?",
    "For {metric}, which local event exhibits the smallest amplitude?",
]

_LOCAL_WIDEST_SPAN_QUESTIONS = [
    "Which local event in {metric} spans the most timesteps?",
    "In {metric}, identify the widest local fluctuation by duration.",
    "What is the longest local event in {metric} in terms of timestep span?",
    "For {metric}, which local event covers the greatest number of timesteps?",
]

_LOCAL_NARROWEST_SPAN_QUESTIONS = [
    "Which local event in {metric} spans the fewest timesteps?",
    "In {metric}, identify the narrowest local fluctuation by duration.",
    "What is the shortest local event in {metric} in terms of timestep span?",
    "For {metric}, which local event covers the fewest number of timesteps?",
]

# ── Family 11E: Segment enumeration question banks ────────────────────────

_SEGMENT_COUNT_ALL_QUESTIONS = [
    "How many trend segments are present in {metric}?",
    "In {metric}, what is the total number of distinct trend segments?",
    "Count the number of trend phases in {metric}.",
    "For {metric}, how many separate trend segments exist?",
]

_SEGMENT_COUNT_BY_TYPE_QUESTIONS = [
    "How many {target_type} trend segments are present in {metric}?",
    "In {metric}, count the number of trend segments with type {target_type}.",
    "How many trend phases of type {target_type} occur in {metric}?",
    "For {metric}, what is the total count of {target_type} trend segments?",
]

_SEGMENT_LONGEST_QUESTIONS = [
    "Which trend segment in {metric} has the longest duration?",
    "In {metric}, identify the trend segment that spans the most timesteps.",
    "What is the longest trend phase in {metric} by duration?",
    "For {metric}, which trend segment covers the greatest number of timesteps?",
]

_SEGMENT_SHORTEST_QUESTIONS = [
    "Which trend segment in {metric} has the shortest duration?",
    "In {metric}, identify the trend segment that spans the fewest timesteps.",
    "What is the shortest trend phase in {metric} by duration?",
    "For {metric}, which trend segment covers the fewest number of timesteps?",
]

# ── Family 11E: Transition enumeration question banks ──────────────────────

_TRANSITION_COUNT_QUESTIONS = [
    "How many trend transitions occur in {metric}?",
    "In {metric}, how many times does the trend direction change?",
    "Count the number of trend transitions in {metric}.",
    "For {metric}, how many adjacent segment boundaries are there?",
]

_TRANSITION_COUNT_BY_TYPE_QUESTIONS = [
    "How many times does {metric} transition from {type_a} to {type_b}?",
    "In {metric}, count the transitions where the trend changes from {type_a} to {type_b}.",
    "How many {type_a} to {type_b} trend transitions occur in {metric}?",
    "For {metric}, how often does a {type_a} phase lead into a {type_b} phase?",
]

_TRANSITION_MOST_COMMON_QUESTIONS = [
    "What is the most common trend transition pattern in {metric}?",
    "In {metric}, which type of trend transition occurs most frequently?",
    "Identify the most prevalent trend transition in {metric}.",
    "For {metric}, which adjacent segment pair pattern is most common?",
]

# ── Family 11E: Event-segment enumeration question banks ───────────────────

_EVENT_SEG_COUNT_IN_TREND_TYPE_QUESTIONS = [
    "How many local events in {metric} occur during {target_type} trend phases?",
    "In {metric}, count the local events that fall within {target_type} trend segments.",
    "How many local fluctuations in {metric} happen during a {target_type} trend?",
    "For {metric}, how many local events are positioned within {target_type} segments?",
]

_EVENT_SEG_MOST_EVENTS_QUESTIONS = [
    "Which trend segment in {metric} contains the most local events?",
    "In {metric}, identify the trend phase with the greatest number of local events.",
    "Which trend segment in {metric} has the highest concentration of local events?",
    "For {metric}, which trend segment contains the most local fluctuations?",
]

_EVENT_SEG_TREND_TYPE_MOST_EVENTS_QUESTIONS = [
    "During which type of trend phase do the most local events occur in {metric}?",
    "In {metric}, which trend type (increasing, decreasing, or steady) contains the most local events?",
    "Which trend direction in {metric} has the highest number of local events?",
    "For {metric}, do local events concentrate during increasing, decreasing, or steady phases?",
]

# ── Temporal position ────────────────────────────────────────────────────

_FIRST_EVENT_QUESTIONS = [
    "At what position does the first local event occur in {metric}?",
    "What is the earliest position of a local event in {metric}?",
    "Where does the first local event appear in {metric}?",
    "In {metric}, at which timestep does the earliest local event happen?",
]

_LAST_EVENT_QUESTIONS = [
    "At what position does the last local event occur in {metric}?",
    "What is the latest position of a local event in {metric}?",
    "Where does the last local event appear in {metric}?",
    "In {metric}, at which timestep does the latest local event happen?",
]

_LARGEST_GAP_QUESTIONS = [
    "What is the largest gap between consecutive local events in {metric}?",
    "In {metric}, what is the maximum spacing between adjacent local events?",
    "How far apart are the two most separated consecutive events in {metric}?",
    "What is the longest stretch without a local event between consecutive events in {metric}?",
]

_SMALLEST_GAP_QUESTIONS = [
    "What is the smallest gap between consecutive local events in {metric}?",
    "In {metric}, what is the minimum spacing between adjacent local events?",
    "How close together are the two nearest consecutive events in {metric}?",
    "What is the shortest distance between consecutive local events in {metric}?",
]

# ── Segment durations (count-based, no ratio) ────────────────────────────

_DURATION_BY_TYPE_QUESTIONS = [
    "How many timesteps of {metric} are spent in {target_type} trend segments?",
    "What is the total duration of {target_type} trend segments in {metric}?",
    "For how many timesteps does {metric} remain in {target_type} trends overall?",
    "Across all {target_type} segments in {metric}, what is the combined duration?",
]

_LONGEST_DURATION_QUESTIONS = [
    "Which trend type has the longest total duration in {metric}?",
    "Across all trend segments in {metric}, which type spans the most timesteps?",
    "Which trend phase occupies the greatest total duration in {metric}?",
    "In {metric}, which trend type accumulates the longest total time?",
]

_SHORTEST_DURATION_QUESTIONS = [
    "Which trend type has the shortest total duration in {metric}?",
    "Across all trend segments in {metric}, which type spans the fewest timesteps?",
    "Which trend phase occupies the smallest total duration in {metric}?",
    "In {metric}, which trend type accumulates the shortest total time?",
]

_DURATION_COMPARISON_QUESTIONS = [
    "Does {metric} spend more time in {type_a} or {type_b} trend segments?",
    "In {metric}, which has a longer total duration: {type_a} or {type_b}?",
    "Across {metric}, is the total duration of {type_a} trends greater than {type_b} trends, or vice versa?",
    "Between {type_a} and {type_b} trend segments in {metric}, which accumulates more timesteps?",
]

_DURATION_DIFFERENCE_QUESTIONS = [
    "What is the difference in total duration between {type_a} and {type_b} trend segments in {metric}?",
    "By how many timesteps do {type_a} and {type_b} trend durations differ in {metric}?",
    "In {metric}, what is the absolute difference between total duration of {type_a} and {type_b} segments?",
    "How many timesteps separate the total duration of {type_a} trends from {type_b} trends in {metric}?",
]

# ── Change point detection question banks ─────────────────────────────────

_CHANGE_POINT_COUNT_QUESTIONS = [
    "How many significant behavioral changes occur in {metric}?",
    "How many change points does {metric} exhibit?",
    "Count the number of times the behavior of {metric} shifts significantly.",
    "How many distinct regime changes are there in {metric}?",
    "How many times does {metric} change its dominant trend?",
]

_CHANGE_POINT_POSITIONS_QUESTIONS = [
    "At what positions do behavioral changes occur in {metric}?",
    "List the positions where {metric} exhibits significant changes in behavior.",
    "Where are the change points located in {metric}?",
    "Report the positions of regime transitions in {metric}.",
    "At which indices does {metric} shift its behavior?",
]

_CHANGE_POINT_LARGEST_SHIFT_QUESTIONS = [
    "Which behavioral change in {metric} shows the largest shift in level?",
    "At what position does the most significant level change occur in {metric}?",
    "Where is the largest regime shift in {metric}?",
    "Report the position of the change point with the biggest level difference in {metric}.",
    "Which change point in {metric} has the greatest difference in mean values between adjacent segments?",
]

# ── Periodicity QA question banks ──────────────────────────────────────────

_PERIODICITY_PERIOD_QUESTIONS = [
    "What is the estimated period of {metric}?",
    "How many time steps does one complete cycle of {metric} span?",
    "What is the length of a single period in {metric}?",
    "Report the periodicity length of {metric} in time steps.",
    "How long is one full oscillation cycle of {metric}?",
]

_PERIODICITY_CYCLE_COUNT_QUESTIONS = [
    "How many complete cycles does {metric} exhibit?",
    "Approximately how many full periods fit within {metric}?",
    "Count the number of complete oscillation cycles in {metric}.",
    "How many times does {metric} repeat its pattern?",
    "What is the total number of full cycles in {metric}?",
]

# ── Family 9 (extended): Statistical Numerical QA question banks ──────────

_STAT_VOLATILITY_CHANGE_QUESTIONS = [
    "Is {metric} becoming more or less volatile over time?",
    "Does the volatility of {metric} increase or decrease from the first half to the second half?",
    "Compare the variability of {metric} between its first and second halves.",
    "Is {metric} more volatile in the second half compared to the first?",
    "How does the standard deviation of {metric} change from the first half to the second half?",
]

_STAT_HALF_MEAN_DIFF_QUESTIONS = [
    "What is the absolute difference between the first-half and second-half means of {metric}?",
    "How much does the mean of {metric} change between the first and second halves?",
    "Compute the absolute mean shift between the two halves of {metric}.",
    "What is the magnitude of the mean difference between the first and second halves of {metric}?",
    "Report the absolute difference in mean values between the first half and second half of {metric}.",
]

_STAT_MEDIAN_QUESTIONS = [
    "What is the median value of {metric}?",
    "Report the median of {metric} across all time steps.",
    "What is the middle value of {metric} when sorted?",
    "Compute the 50th percentile of {metric}.",
    "What is the central tendency of {metric} as measured by the median?",
]

_STAT_WINDOWED_TREND_QUESTIONS = [
    "Do the windowed means of {metric} show an overall upward, downward, or flat trend?",
    "What is the general direction of the windowed means of {metric}?",
    "Looking at the windowed averages, does {metric} trend upward, downward, or stay flat?",
    "Characterize the overall trend of the windowed means of {metric}.",
    "Based on the windowed means, is {metric} trending upward, downward, or flat?",
]

