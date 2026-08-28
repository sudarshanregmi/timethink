import math
from typing import Any, Dict, List, Tuple
from synth.ts_generator.utils.common_utils import format_float
from synth.ts_generator.utils.thinking.core import (
    ThoughtBuilder, format_add_sub_calc, format_window_calc, format_cmp,
    format_mul_calc,
)

# ---------------------------------------------------------------------------
# Statistical threshold constants (used across builders in this module)
# ---------------------------------------------------------------------------
VOLATILITY_INCREASE_THRESHOLD = 1.1   # std ratio above this → more_volatile
VOLATILITY_DECREASE_THRESHOLD = 0.9   # std ratio below this → less_volatile
WINDOWED_TREND_TOLERANCE = 0.05       # ±5% of first window mean for trend detection

# ---------------------------------------------------------------------------
# Statistical numerical QA thought builder (Family 9)
# ---------------------------------------------------------------------------

def build_stat_numerical_thought(
    attributes: Dict[str, Any],
    metric: str,
    sub_type: str,
    params: Dict[str, Any],
) -> Tuple[str, str]:
    """
    Statistical numerical QA thought builder.

    Sub-types: stat_extrema_pos, stat_windowed_mean, stat_threshold_duration,
               stat_mean_crossing, stat_range, stat_segment_compare,
               stat_min_val, stat_max_val, stat_mean_val, stat_std_val

    Returns (think_str, verdict_str).
    """
    tb = ThoughtBuilder()
    stats   = attributes.get('statistics', {}) or {}
    seq_len = attributes['seq_len']

    if sub_type == 'stat_min_val':
        val = params['value']
        tb.data(f"metric: {metric}")
        tb.data(f"stats: min={format_float(val, 2)}")
        verdict = format_float(val, 2)

    elif sub_type == 'stat_max_val':
        val = params['value']
        tb.data(f"metric: {metric}")
        tb.data(f"stats: max={format_float(val, 2)}")
        verdict = format_float(val, 2)

    elif sub_type == 'stat_mean_val':
        val = params['value']
        chunks = params['chunks']  # List[(cs, ce, cm, cl)]
        tb.data(f"metric: {metric}")
        for cs, ce, cm, _ in chunks:
            tb.line(f"chunk [{cs}, {ce}] mean = {format_float(cm, 2)}")
        chunk_means = [cm for _, _, cm, _ in chunks]
        sum_expr = " + ".join(format_float(cm, 2) for cm in chunk_means)
        tb.line(f"mean = ({sum_expr}) / {len(chunk_means)} = {format_float(val, 2)}")
        verdict = format_float(val, 2)

    elif sub_type == 'stat_std_val':
        val = params['value']
        tb.data(f"metric: {metric}")
        tb.data(f"stats: std={format_float(val, 2)}")
        verdict = format_float(val, 2)

    elif sub_type == 'stat_extrema_pos':
        direction = params['direction']
        pos       = params['pos']
        val       = params['val']
        tb.data(f"metric: {metric}")
        if direction == 'peak':
            tb.data(f"stats: max={format_float(val, 2)}")
        else:
            tb.data(f"stats: min={format_float(val, 2)}")
        tb.line(f"index={pos}")
        verdict = str(pos)

    elif sub_type == 'stat_windowed_mean':
        window_size  = params['window_size']
        window_means = params['window_means']
        start_win    = params['start_window']
        n_windows    = len(window_means)
        win_start    = params['win_start']
        win_end      = params['win_end']
        tb.data(f"metric: {metric}")
        tb.line(format_window_calc(win_start, win_end, window_size, n_windows))
        for i, wm in enumerate(window_means):
            abs_win  = start_win + i
            ws       = abs_win * window_size
            we       = min((abs_win + 1) * window_size - 1, seq_len - 1)
            tb.line(f"window {i} [{ws}-{we}]: mean={format_float(wm, 2)}")
        verdict = "[" + ", ".join(format_float(v, 2) for v in window_means) + "]"

    elif sub_type == 'stat_threshold_duration':
        direction = params['direction']
        threshold = params['threshold']
        count     = params['count']
        variant   = params['variant']
        mean_val  = params['mean_val']
        std_val   = params['std_val']
        op        = 'above' if direction == 'above' else 'below'
        tb.data(f"metric: {metric}")
        if variant == 'above_mean':
            tb.data(f"stats: mean={format_float(mean_val, 2)}")
        else:
            tb.data(f"stats: mean={format_float(mean_val, 2)}, std={format_float(std_val, 2)}")
            if variant == 'above_std':
                tb.line(format_add_sub_calc(mean_val, std_val, threshold, op='+', label='threshold'))
            else:
                tb.line(format_add_sub_calc(mean_val, std_val, threshold, op='-', label='threshold'))
        tb.line(f"count {op} {format_float(threshold, 2)} = {count}")
        verdict = str(count)

    elif sub_type == 'stat_mean_crossing':
        mean_val         = stats['mean']
        crossings        = params['crossings']
        crossing_indices = params['crossing_indices']
        tb.data(f"metric: {metric}")
        tb.data(f"stats: mean={format_float(mean_val, 2)}")
        if crossing_indices:
            indices_str = ", ".join(str(idx) for idx in crossing_indices)
            tb.line(f"crossings at: [{indices_str}]")
        tb.line(f"count = {crossings}")
        verdict = str(crossings)

    elif sub_type == 'stat_range':
        min_val = stats['min']
        max_val = stats['max']
        r       = params['range']
        tb.data(f"metric: {metric}")
        tb.data(f"stats: max={format_float(max_val, 2)}, min={format_float(min_val, 2)}")
        tb.line(f"min = {format_float(min_val, 2)}")
        tb.line(f"max = {format_float(max_val, 2)}")
        tb.line(format_add_sub_calc(max_val, min_val, r, op='-', label='range'))
        verdict = format_float(r, 2)

    elif sub_type == 'stat_segment_compare':
        compare_by = params['compare_by']
        first_val  = params['first_val']
        second_val = params['second_val']
        mid        = params['mid']
        verdict    = params['verdict']
        tb.data(f"metric: {metric}")
        tb.data(f"first_half [0-{mid - 1}]: {compare_by}={format_float(first_val, 2)}")
        tb.data(f"second_half [{mid}-{seq_len - 1}]: {compare_by}={format_float(second_val, 2)}")
        first_val = round(first_val, 2)
        second_val = round(second_val, 2)
        if verdict == 'equal':
            tb.line(f"{format_float(first_val, 2)} ≈ {format_float(second_val, 2)} -> equal")
        elif verdict == 'first half':
            cmp_text, _ = format_cmp(first_val, second_val, '>', dp_val=2)
            tb.line(f"{cmp_text} -> first half higher")
        else:
            cmp_text, _ = format_cmp(first_val, second_val, '<', dp_val=2)
            tb.line(f"{cmp_text} -> second half higher")

    elif sub_type == 'stat_volatility_change':
        first_half_std  = params['first_half_std']
        second_half_std = params['second_half_std']
        tb.data(f"metric: {metric}")
        tb.data(f"stats: first_half_std={format_float(first_half_std, 2)}, second_half_std={format_float(second_half_std, 2)}")
        if first_half_std == 0.0:
            if second_half_std > 0.0:
                tb.line(f"first_half_std=0, second_half_std > 0 -> more_volatile")
                verdict = 'more_volatile'
            else:
                tb.line(f"first_half_std=0, second_half_std=0 -> stable")
                verdict = 'stable'
        else:
            # Compare via pre-computed thresholds (no division)
            inc_thresh = round(first_half_std * VOLATILITY_INCREASE_THRESHOLD, 2)
            dec_thresh = round(first_half_std * VOLATILITY_DECREASE_THRESHOLD, 2)
            tb.line(format_mul_calc(VOLATILITY_INCREASE_THRESHOLD, first_half_std, inc_thresh, label='increase_threshold'))
            cmp_text_hi, is_more = format_cmp(second_half_std, inc_thresh, '>', dp_val=2)
            tb.line(cmp_text_hi)
            if is_more:
                verdict = 'more_volatile'
            else:
                tb.line(format_mul_calc(VOLATILITY_DECREASE_THRESHOLD, first_half_std, dec_thresh, label='decrease_threshold'))
                cmp_text_lo, is_less = format_cmp(second_half_std, dec_thresh, '<', dp_val=2)
                tb.line(cmp_text_lo)
                if is_less:
                    verdict = 'less_volatile'
                else:
                    verdict = 'stable'
            tb.line(f"-> {verdict}")

    elif sub_type == 'stat_half_mean_diff':
        first_half_mean  = params['first_half_mean']
        second_half_mean = params['second_half_mean']
        diff = second_half_mean - first_half_mean
        tb.data(f"metric: {metric}")
        tb.data(f"stats: first_half_mean={format_float(first_half_mean, 2)}, second_half_mean={format_float(second_half_mean, 2)}")
        tb.line(format_add_sub_calc(second_half_mean, first_half_mean, diff, op='-', label='half_mean_diff'))
        abs_diff = abs(diff)
        tb.line(f"|half_mean_diff| = {format_float(abs_diff, 2)}")
        verdict = format_float(abs_diff, 2)

    elif sub_type == 'stat_median':
        median_val = params['median']
        tb.data(f"metric: {metric}")
        tb.data(f"stats: median={format_float(median_val, 2)}")
        verdict = format_float(median_val, 2)

    elif sub_type == 'stat_windowed_trend':
        segment_means = params['segment_means_16']
        tb.data(f"metric: {metric}")
        means_str = ", ".join(format_float(v, 2) for v in segment_means)
        tb.data(f"segment_means_16=[{means_str}]")
        first_wm = round(segment_means[0], 2)
        last_wm  = round(segment_means[-1], 2)
        delta = round(abs(first_wm) * WINDOWED_TREND_TOLERANCE, 2)
        upper_thresh = round(first_wm + delta, 2)
        lower_thresh = round(first_wm - delta, 2)
        tb.line(format_mul_calc(abs(first_wm), WINDOWED_TREND_TOLERANCE, delta,
                                dp=2, label='tolerance'))
        tb.line(format_add_sub_calc(first_wm, delta, upper_thresh,
                                    op='+', label='upper_thresh'))
        cmp_text_up, is_up = format_cmp(last_wm, upper_thresh, '>', dp_val=2)
        tb.line(cmp_text_up)
        if is_up:
            verdict = 'upward'
        else:
            tb.line(format_add_sub_calc(first_wm, delta, lower_thresh,
                                        op='-', label='lower_thresh'))
            cmp_text_dn, is_dn = format_cmp(last_wm, lower_thresh, '<', dp_val=2)
            tb.line(cmp_text_dn)
            if is_dn:
                verdict = 'downward'
            else:
                verdict = 'flat'
        tb.line(f"-> {verdict}")

    else:
        raise ValueError(f"Unknown stat_numerical sub_type: {sub_type!r}")

    tb.verdict(verdict)
    return tb.build()


# ---------------------------------------------------------------------------
# Periodicity QA thought builder
# ---------------------------------------------------------------------------

def build_periodicity_thought(
    attributes: Dict[str, Any],
    metric: str,
    sub_type: str,
    params: Dict[str, Any],
) -> Tuple[str, str]:
    """
    Periodicity QA thought builder.

    Sub-types: period_estimate, cycle_count

    Returns (think_str, verdict_str).
    """
    tb = ThoughtBuilder()
    seasonal = attributes.get('seasonal', {}) or {}
    period = seasonal.get('period', 0.0)
    seq_len = attributes['seq_len']

    if sub_type == 'period_estimate':
        tb.data(f"metric: {metric}")
        tb.data(f"seasonal: period={format_float(period, 2)}")
        verdict = format_float(period, 2)

    elif sub_type == 'cycle_count':
        floored = int(math.floor(seq_len / period))
        covered = round(floored * period, 2)
        tb.data(f"metric: {metric}")
        tb.data(f"seasonal: period={format_float(period, 2)}")
        tb.data(f"seq_len={seq_len}")
        tb.line(format_mul_calc(floored, period, covered, label='covered_length'))
        cmp_text, fits = format_cmp(covered, seq_len, '<=', dp_val=2, dp_thresh=0)
        tb.line(cmp_text)
        verdict = str(floored)

    else:
        raise ValueError(f"Unknown periodicity sub_type: {sub_type!r}")

    tb.verdict(verdict)
    return tb.build()
