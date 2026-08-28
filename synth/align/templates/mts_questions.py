"""MTS question template banks.

Placeholders (pairwise):
  {metric_a}    — first metric name
  {metric_b}    — second metric name
  {trend_type}  — trend type string (e.g. "increase", "decrease", "keep steady")
  {trend_type_verb} — verb form (e.g. "increasing", "decreasing", "keeping steady")
  {label}       — judgment label string
  {threshold}   — noise threshold float

Placeholders (cross-metric):
  {target_type} — trend type to filter by (which_have_trend_type only)
"""

# ---------------------------------------------------------------------------
# Cross-trend questions (MTSShape)
# ---------------------------------------------------------------------------
CROSS_TREND_QUESTIONS = [
    "During the time period when {metric_a} is {trend_type}, what trend behavior does {metric_b} predominantly show?",
    "In the range where {metric_a} exhibits a {trend_type} trend, what is {metric_b}'s dominant trend direction?",
    "When {metric_a} is {trend_type_verb}, what does {metric_b}'s trend look like in that same range?",
    "For the segments where {metric_a} shows a {trend_type} trend, which trend type covers most of {metric_b}'s behavior?",
    "Identify the dominant trend of {metric_b} during the period when {metric_a} is {trend_type}.",
]

TREND_TYPE_VERBS = {
    "increase": "increasing",
    "decrease": "decreasing",
    "steady": "steady",
    "keep steady": "keeping steady",
}

# ---------------------------------------------------------------------------
# Anti-correlation judgment questions (MTSShape)
# ---------------------------------------------------------------------------
ANTI_JUDGMENT_LABELS = [
    "anticorrelated high-noise state",
    "inverse trend with signal disturbance",
    "opposing trend noise anomaly",
    "diverging noisy behavior",
]

ANTI_JUDGMENT_QUESTIONS = [
    "A '{label}' is defined as: {metric_a} and {metric_b} have anticorrelated trends AND {metric_b} noise strength exceeds {threshold:.2f}. Does this condition hold?",
    "In this system, define '{label}' as: {metric_a} trends opposite to {metric_b} AND {metric_b} exhibits noise strength above {threshold:.2f}. Is '{label}' present?",
    "If '{label}' requires (1) anticorrelated trends between {metric_a} and {metric_b}, AND (2) {metric_b} noise strength > {threshold:.2f}, does the data satisfy '{label}'?",
    "Does '{label}' apply here? '{label}' means {metric_a} and {metric_b} show opposite trend segments while {metric_b}'s noise strength exceeds {threshold:.2f}.",
]

# ---------------------------------------------------------------------------
# Cross-metric enumeration questions (MTS LOCAL + SHAPE)
# ---------------------------------------------------------------------------

_CROSS_WHICH_HAVE_LOCAL_QUESTIONS = [
    "Which of the time series have local fluctuations or anomalies?",
    "Identify all metrics that contain local events.",
    "Do any of the time series exhibit localized changes? If so, which ones?",
    "List the metrics that have at least one local event.",
]

_CROSS_WHICH_HAVE_TREND_TYPE_QUESTIONS = [
    "Which metrics exhibit an overall {target_type} trend?",
    "Identify all time series whose overall trend is {target_type}.",
    "Which of the metrics show a {target_type} trend pattern?",
    "List every metric that has an overall {target_type} trend.",
]

_CROSS_ALL_SAME_TREND_QUESTIONS = [
    "Do all of the time series share the same overall trend direction?",
    "Is the overall trend identical across all metrics?",
    "Are all metrics trending in the same direction?",
    "Check whether every time series has the same overall trend type.",
]

_CROSS_MOST_LOCAL_EVENTS_QUESTIONS = [
    "Which metric has the most local events?",
    "Identify the time series with the highest number of local fluctuations.",
    "Which metric contains the greatest number of local anomalies?",
    "Of all the metrics, which one has the most local changes?",
]

_CROSS_HIGHEST_AMPLITUDE_QUESTIONS = [
    "Which metric has the local event with the highest amplitude?",
    "Identify the time series whose local events reach the greatest amplitude.",
    "Across all metrics, which one contains the largest-amplitude local event?",
    "Which metric exhibits the most intense local fluctuation by amplitude?",
]

_CROSS_MOST_TREND_SEGMENTS_QUESTIONS = [
    "Which metric has the most trend segments?",
    "Identify the time series with the greatest number of trend phases.",
    "Which metric's trend is divided into the most segments?",
    "Of all the metrics, which one has the highest number of trend segments?",
]

_CROSS_LONGEST_SEGMENT_QUESTIONS = [
    "Which metric has the longest single trend segment?",
    "Identify the time series containing the trend segment with the greatest duration.",
    "Across all metrics, which one has the longest continuous trend phase?",
    "Which metric exhibits the single longest trend segment?",
]

_CROSS_NOISIEST_METRIC_QUESTIONS = [
    "Which metric is the noisiest?",
    "Identify the time series with the highest noise strength.",
    "Which metric has the most noise?",
    "Of all the metrics, which one exhibits the greatest noise level?",
]

_CROSS_QUIETEST_METRIC_QUESTIONS = [
    "Which metric is the quietest?",
    "Identify the time series with the lowest noise strength.",
    "Which metric has the least noise?",
    "Of all the metrics, which one exhibits the lowest noise level?",
]

_CROSS_WIDEST_RANGE_QUESTIONS = [
    "Which metric has the widest value range?",
    "Identify the time series with the largest range between its minimum and maximum values.",
    "Across all metrics, which one spans the widest range?",
    "Which metric covers the greatest range of values?",
]

_CROSS_HIGHEST_MEAN_QUESTIONS = [
    "Which metric has the highest mean value?",
    "Identify the time series with the greatest average value.",
    "Across all metrics, which one has the highest mean?",
    "Which metric has the largest average over all timesteps?",
]

_CROSS_LOWEST_MEAN_QUESTIONS = [
    "Which metric has the lowest mean value?",
    "Identify the time series with the smallest average value.",
    "Across all metrics, which one has the lowest mean?",
    "Which metric has the smallest average over all timesteps?",
]

_CROSS_HIGHEST_MAX_QUESTIONS = [
    "Which metric reaches the highest peak value?",
    "Identify the time series with the largest maximum value.",
    "Across all metrics, which one has the highest maximum?",
    "Which metric achieves the greatest peak value?",
]

_CROSS_LOWEST_MIN_QUESTIONS = [
    "Which metric reaches the lowest trough value?",
    "Identify the time series with the smallest minimum value.",
    "Across all metrics, which one dips to the lowest point?",
    "Which metric has the smallest minimum value?",
]

_CROSS_HIGHEST_STD_QUESTIONS = [
    "Which metric has the highest standard deviation?",
    "Identify the time series with the greatest variability.",
    "Across all metrics, which one has the highest std?",
    "Which metric shows the most statistical spread?",
]

_CROSS_LOWEST_STD_QUESTIONS = [
    "Which metric has the lowest standard deviation?",
    "Identify the time series with the least variability.",
    "Across all metrics, which one has the lowest std?",
    "Which metric shows the least statistical spread?",
]

# ── Cross-metric temporal position ────────────────────────────────────────

_CROSS_EARLIEST_EVENT_QUESTIONS = [
    "Which metric has the earliest local event?",
    "Across all metrics, which one has a local event occurring first?",
    "In which time series does the first local event appear earliest?",
    "Which metric has the smallest first-event position?",
]

_CROSS_LATEST_EVENT_QUESTIONS = [
    "Which metric has the latest local event?",
    "Across all metrics, which one has a local event occurring last?",
    "In which time series does the last local event appear latest?",
    "Which metric has the largest last-event position?",
]

_CROSS_LARGEST_GAP_QUESTIONS = [
    "Which metric has the largest gap between consecutive local events?",
    "Across all metrics, which one has the most spread-out consecutive events?",
    "In which time series is the maximum spacing between adjacent events the largest?",
    "Which metric shows the widest inter-event gap?",
]

# ---------------------------------------------------------------------------
# Cross-metric statistical judgment questions (MTS LOCAL + SHAPE)
# ---------------------------------------------------------------------------

_CROSS_STAT_STD_RATIO_QUESTIONS = [
    "A '{label}' occurs when the standard deviation of {metric_a} is at least {K} times that of {metric_b}. Is there a '{label}'?",
    "Does a '{label}' exist? A '{label}' requires {metric_a}'s std to be at least {K} times {metric_b}'s std.",
    "If a '{label}' means {metric_a}'s std must reach {K} times {metric_b}'s std, does it hold?",
]

_CROSS_STAT_RANGE_RATIO_QUESTIONS = [
    "A '{label}' occurs when the value range of {metric_a} is at least {K} times that of {metric_b}. Is there a '{label}'?",
    "Does a '{label}' exist? A '{label}' requires {metric_a}'s range to be at least {K} times {metric_b}'s range.",
    "If a '{label}' means {metric_a}'s value range must reach {K} times {metric_b}'s range, does it hold?",
]

_CROSS_HALF_SHIFT_QUESTIONS = [
    "A '{label}' requires BOTH {metric_a} AND {metric_b} to have second-half means exceeding first-half means by more than {K}% of their individual ranges. Is there a '{label}'?",
    "Does a '{label}' exist? A '{label}' means both {metric_a} and {metric_b} show a half-mean shift exceeding {K}% of their respective ranges.",
    "If a '{label}' requires each metric's second-half mean to exceed its first-half mean by more than {K}% of its range, do {metric_a} and {metric_b} both qualify?",
]

_CROSS_HALF_VOLATILITY_QUESTIONS = [
    "An '{label}' occurs when {metric_a} shows increasing volatility (second-half std > first-half std) while {metric_b} maintains stable or decreasing volatility. Is there an '{label}'?",
    "Does an '{label}' exist? An '{label}' requires {metric_a}'s volatility to increase while {metric_b}'s stays flat or decreases.",
    "If an '{label}' means {metric_a}'s second-half std exceeds its first-half std AND {metric_b}'s second-half std does not exceed its first-half std, does it hold?",
]