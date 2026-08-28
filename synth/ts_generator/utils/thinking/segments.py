from typing import Any, Dict, List, Tuple
from synth.ts_generator.utils.common_utils import format_float
from synth.ts_generator.utils.thinking.core import (
    ThoughtBuilder, met_str, pass_str, format_seg_display,
    format_add_sub_calc, format_mul_calc, format_cmp,
    format_window_calc, format_str_eq,
)

_UPWARD_EVENT_TYPES = frozenset({
    'upward spike', 'wide upward spike', 'continuous upward spike',
    'sudden increase', 'increase after downward spike', 'increase after upward spike',
    'upward convex', 'slow rise followed by rapid decline', 'rapid rise followed by slow decline',
})
_DOWNWARD_EVENT_TYPES = frozenset({
    'downward spike', 'wide downward spike', 'continuous downward spike',
    'sudden decrease', 'decrease after upward spike', 'decrease after downward spike',
    'downward convex', 'slow decline followed by rapid rise', 'rapid decline followed by slow rise',
})


def _render_local_events_simple(local: list, limit: int = 6) -> str:
    """Render local events in simplified compound-judgment format."""
    return '; '.join(
        f"{e['type']}, amplitude={format_float(e['amplitude'], 2)}, "
        f"direction={'upward' if e['type'] in _UPWARD_EVENT_TYPES else 'downward' if e['type'] in _DOWNWARD_EVENT_TYPES else 'none'}"
        for e in local[:limit]
    )


def build_compound_judgment_thought(
    attributes: Dict[str, Any],
    metric: str,
    judgment_type: str,
    params: Dict[str, Any],
) -> Tuple[str, str]:
    """
    Family 7: Compound judgment combining multiple attribute conditions.

    judgment_type:
      'trend_local'          -- local event with amp > threshold during trend type
      'multi_phase'          -- trend follows a given phase sequence
      'noise_trend'          -- noise strength + type during a trend type
      'stat_threshold'       -- statistical threshold anomaly detection
      'amplitude_vs_noise'   -- local event amp > K × noise strength
      'amplitude_vs_std'     -- local event amp > K × std
      'range_vs_std'         -- value range > K × std
      'half_volatility'      -- second-half std / first-half std >= K
      'half_mean_shift'      -- |half-mean shift| > K% of range
      'windowed_monotonicity' -- window means strictly increasing/decreasing
      'phase_event'          -- event direction during specific trend phase
      'sequential_events'    -- event A followed by event B within N steps

    params: type-specific dict with all pre-computed values including 'verdict'.
    verdict: 'yes' | 'no'
    """
    tb = ThoughtBuilder()
    verdict = params['verdict']

    # ── Data section ──────────────────────────────────────────────────────────
    if judgment_type in ('trend_local', 'noise_trend'):
        tb.metric(metric, attributes, ['trend_overall'])
    elif judgment_type == 'multi_phase':
        tb.metric(metric, attributes, ['trend_list'])
    elif judgment_type == 'stat_threshold':
        tb.data(f"metric: {metric}")
        direction = params['direction']
        if direction == 'below':
            tb.data(f"stats: min={format_float(params['actual_min'], 2)}")
        else:
            tb.data(f"stats: max={format_float(params['actual_max'], 2)}")
    elif judgment_type in ('amplitude_vs_noise', 'amplitude_vs_std'):
        tb.data(f"metric: {metric}")
        if judgment_type == 'amplitude_vs_noise':
            ns = attributes.get('noise', {})
            tb.data(f"noise: strength={format_float(ns['strength'], 2)}")
        else:
            stats = attributes['statistics']
            tb.data(f"stats: std={format_float(stats['std'], 2)}")
        local = attributes.get('local', [])
        if local:
            tb.data(f"local=[{_render_local_events_simple(local)}]")
    elif judgment_type == 'range_vs_std':
        tb.data(f"metric: {metric}")
        stats = attributes['statistics']
        tb.data(
            f"stats: max={format_float(stats['max'], 2)}, "
            f"min={format_float(stats['min'], 2)}, "
            f"std={format_float(stats['std'], 2)}"
        )
    elif judgment_type == 'half_volatility':
        tb.metric(metric, attributes, ['statistic:halves_std'])
    elif judgment_type == 'half_mean_shift':
        tb.data(f"metric: {metric}")
        stats = attributes['statistics']
        tb.data(
            f"stats: max={format_float(stats['max'], 2)}, "
            f"min={format_float(stats['min'], 2)}, "
            f"first_half_mean={format_float(stats['first_half_mean'], 2)}, "
            f"second_half_mean={format_float(stats['second_half_mean'], 2)}"
        )
    elif judgment_type == 'windowed_monotonicity':
        tb.data(f"metric: {metric}")
    elif judgment_type == 'phase_event':
        tb.metric(metric, attributes, ['trend_list'])
        local = attributes.get('local', [])
        if local:
            ev_parts = []
            for e in local[:6]:
                d = ('upward' if e['type'] in _UPWARD_EVENT_TYPES
                     else 'downward' if e['type'] in _DOWNWARD_EVENT_TYPES
                     else 'none')
                ev_parts.append(
                    f"{e['type']}, pos={e['position_start']}, direction={d}"
                )
            tb.data(f"local=[{'; '.join(ev_parts)}]")
    elif judgment_type == 'sequential_events':
        tb.data(f"metric: {metric}")
        local = attributes.get('local', [])
        if local:
            ev_parts = []
            for e in local[:6]:
                ev_parts.append(
                    f"{e['type']}, start@({e['position_start']}, "
                    f"{format_float(e['value_start'], 2)})"
                )
            tb.data(f"local=[{'; '.join(ev_parts)}]")
    else:
        tb.data(f"metric: {metric}")

    if judgment_type == 'noise_trend':
        ns = attributes.get('noise', {})
        tb.data(f"noise={ns['type']}, strength={format_float(ns['strength'], 2)}")

    if judgment_type == 'trend_local':
        local = attributes.get('local', [])
        if local:
            tb.data(f"local=[{_render_local_events_simple(local)}]")

    # ── Reasoning section ─────────────────────────────────────────────────────
    if judgment_type == 'trend_local':
        direction  = params['event_direction']
        threshold  = params['threshold']
        req_trend  = params['required_trend']
        act_trend  = params['actual_trend']
        qual_evts  = params['qualifying_events']

        trend_text, c1_met = format_str_eq(act_trend, req_trend)
        tb.condition(1, f"overall trend is {req_trend}")
        tb.detail(f"actual: {trend_text} -> {met_str(c1_met)}")

        if c1_met:
            all_evts = params['all_events']
            n_checked = min(len(all_evts), 6)
            tb.condition(2, f"{direction} events with amplitude > {format_float(threshold, 2)}")
            for idx, (t, a, d) in enumerate(all_evts[:6], 1):
                dir_text, dir_match = format_str_eq(d, direction)
                if not dir_match:
                    tb.detail(
                        f"event {idx}: {t}, amp={format_float(a, 2)}, direction={d} "
                        f"-> direction {dir_text}, {pass_str(False)}"
                    )
                else:
                    amp_text, amp_pass = format_cmp(round(a, 2), round(threshold, 2), '>', dp_val=2)
                    tb.detail(
                        f"event {idx}: {t}, amp={format_float(a, 2)}, direction={d} "
                        f"-> direction {dir_text}, {pass_str(True)} "
                        f"-> amplitude {amp_text}, {pass_str(amp_pass)}"
                    )
            tb.summary(len(qual_evts), n_checked, indented=True)
            all_met = len(qual_evts) > 0
            tb.all_conditions(all_met)
        # c1 failed: only 1 condition shown, no all_conditions line

    elif judgment_type == 'multi_phase':
        target_seq = params['target_sequence']
        actual_seq = params['actual_sequence']

        target_str = ' -> '.join(target_seq)
        n_phases = len(target_seq)
        actual_str = ' -> '.join(actual_seq)
        match = len(actual_seq) >= n_phases and actual_seq[:n_phases] == target_seq
        tb.condition(1, f"trend follows phase sequence [{target_str}]")
        tb.detail(f"actual: {actual_str}")
        first_n = ' -> '.join(actual_seq[:n_phases])
        tb.detail(f"first {n_phases} phases: {first_n} -> {met_str(match)}")

    elif judgment_type == 'noise_trend':
        threshold  = params['threshold']
        req_trend  = params['required_trend']
        act_trend  = params['actual_trend']
        act_n_type = params['actual_noise_type']
        act_n_amp  = params['actual_noise_amp']

        trend_text, c1_met = format_str_eq(act_trend, req_trend)
        tb.condition(1, f"overall trend is {req_trend}")
        tb.detail(f"actual: {trend_text} -> {met_str(c1_met)}")

        if c1_met:
            act_n_amp = round(act_n_amp, 2)
            tb.condition(2, f"noise strength > {format_float(threshold, 2)}")
            cmp_text, c2_met = format_cmp(act_n_amp, threshold, '>', dp_val=2)
            tb.detail(
                f"actual noise: {act_n_type}, strength={format_float(act_n_amp, 2)} -> "
                f"{cmp_text} -> {met_str(c2_met)}"
            )
            all_met = c2_met
            tb.all_conditions(all_met)
        # c1 failed: only 1 condition shown, no all_conditions line

    elif judgment_type == 'stat_threshold':
        direction = params['direction']
        threshold = params['threshold']
        act_min   = params['actual_min']
        act_max   = params['actual_max']

        if direction == 'below':
            tb.condition(1, f"min value < {format_float(threshold, 2)}")
            cmp_text, c1_met = format_cmp(round(act_min, 2), round(threshold, 2), '<', dp_val=2)
            tb.detail(f"{cmp_text} -> {met_str(c1_met)}")
        else:
            tb.condition(1, f"max value > {format_float(threshold, 2)}")
            cmp_text, c1_met = format_cmp(round(act_max, 2), round(threshold, 2), '>', dp_val=2)
            tb.detail(f"{cmp_text} -> {met_str(c1_met)}")

    elif judgment_type in ('amplitude_vs_noise', 'amplitude_vs_std'):
        K = params['K']
        ref_val = params['ref_val']
        threshold = params['threshold']
        all_evts = params['all_events']
        qual_count = params['qualifying_count']
        dp = 2
        n_checked = min(len(all_evts), 6)

        tb.line(format_mul_calc(K, ref_val, threshold, dp=dp))
        for idx, (t, a, d) in enumerate(all_evts[:6], 1):
            cmp_text, passed = format_cmp(round(a, 2), round(threshold, 2), '>', dp_val=2, dp_thresh=dp)
            tb.line(f"event {idx}: {t}, amp={format_float(a, 2)} -> {cmp_text}, {pass_str(passed)}")
        tb.summary(qual_count, n_checked)

    elif judgment_type == 'range_vs_std':
        K = params['K']
        act_max = params['actual_max']
        act_min = params['actual_min']
        act_std = params['actual_std']
        rng = params['range']
        threshold = params['threshold']

        tb.line(format_add_sub_calc(act_max, act_min, rng, op='-', label='range'))
        tb.line(format_mul_calc(K, act_std, threshold))
        rng = round(rng, 2)
        threshold = round(threshold, 2)
        cmp_text, is_met = format_cmp(rng, threshold, '>')
        tb.line(f"{cmp_text} -> {met_str(is_met)}")

    elif judgment_type == 'half_volatility':
        K = params['K']
        fhs = params['first_half_std']
        shs = params['second_half_std']
        threshold = round(K * fhs, 2)

        tb.line(format_mul_calc(K, fhs, threshold, label='threshold'))
        cmp_text, is_met = format_cmp(shs, threshold, '>=', dp_val=2)
        tb.line(f"{cmp_text} -> {met_str(is_met)}")

    elif judgment_type == 'half_mean_shift':
        K_pct = params['K_pct']
        act_max = params['actual_max']
        act_min = params['actual_min']
        fhm = params['first_half_mean']
        shm = params['second_half_mean']
        rng = params['range']
        shift = params['shift']
        threshold = params['threshold']

        tb.line(format_add_sub_calc(act_max, act_min, rng, op='-', label='range'))
        signed_shift = round(shift, 2)
        tb.line(format_add_sub_calc(shm, fhm, signed_shift, op='-', label='shift'))
        abs_shift = round(abs(signed_shift), 2)
        tb.line(f"|shift| = {format_float(abs_shift, 2)}")
        tb.line(format_mul_calc(K_pct, rng, threshold, is_pct=True))
        threshold = round(threshold, 2)
        cmp_text, is_met = format_cmp(abs_shift, threshold, '>')
        tb.line(f"{cmp_text} -> {met_str(is_met)}")

    elif judgment_type == 'windowed_monotonicity':
        means = params['means']
        direction = params['direction']
        wm_seq_len = params['seq_len']
        op = '<' if direction == 'higher' else '>'

        tb.line(format_window_calc(0, wm_seq_len - 1, 16, len(means)))
        wm_vals = ', '.join(format_float(v, 2) for v in means)
        tb.line(f"means: [{wm_vals}]")
        parts = [format_float(means[0], 2)]
        for i in range(1, len(means)):
            cmp_text, passed = format_cmp(round(means[i - 1], 2), round(means[i], 2), op, dp_val=2)
            parts.append(f"{cmp_text} ({pass_str(passed)})")
            if not passed:
                break
        tb.line(' '.join(parts))

    elif judgment_type == 'phase_event':
        req_direction = params['event_direction']
        req_trend_type = params['required_trend_type']
        all_evts = params['all_events']
        trend_list = params['trend_list']
        n_checked = min(len(all_evts), 6)

        # condition 1: find events with matching direction
        tb.condition(1, f"find {req_direction} events")
        dir_evts = []
        for idx, (t, pos, d) in enumerate(all_evts[:6], 1):
            dir_text, passed = format_str_eq(d, req_direction)
            tb.detail(f"event {idx}: {t}, direction={d} -> {dir_text}, {pass_str(passed)}")
            if passed:
                dir_evts.append((t, pos, d))
        tb.summary(len(dir_evts), n_checked, indented=True)

        if dir_evts:
            # condition 2: check if any qualifying event falls in required phase
            tb.condition(2, f"qualifying events during {req_trend_type} phase")
            found = False
            for t, pos, d in dir_evts:
                in_phase = False
                phase_info = "(no segment)"
                type_text = None
                for seg_type, seg_start, seg_end in trend_list:
                    if seg_start <= pos <= seg_end:
                        phase_info = format_seg_display(seg_type, seg_start, seg_end)
                        type_text, type_match = format_str_eq(seg_type, req_trend_type)
                        in_phase = type_match
                        break
                if in_phase:
                    tb.detail(f"{t} at {pos}: {phase_info} -> {type_text}, {pass_str(True)}")
                    found = True
                    break
                elif type_text is not None:
                    # segment found but wrong type
                    tb.detail(f"{t} at {pos}: {phase_info} -> {type_text}, {pass_str(False)}")
                else:
                    # no segment contains this position
                    tb.detail(f"{t} at {pos}: {phase_info}, {pass_str(False)}")
            all_met = found
            tb.all_conditions(all_met)
        # no dir_evts: only 1 condition shown, no all_conditions line

    elif judgment_type == 'sequential_events':
        pairs = params['pairs']
        gap_limit = params['gap_limit']

        for t_a, pos_a, t_b, pos_b in pairs:
            gap = pos_b - pos_a
            gap_text = format_add_sub_calc(pos_b, pos_a, op='-', label='gap', int_mode=True)
            cmp_text, passed = format_cmp(gap, gap_limit, '<=', dp_val=0)
            tb.line(
                f"pair ({t_a} at {pos_a}, {t_b} at {pos_b}): "
                f"{gap_text}, "
                f"{cmp_text} -> {pass_str(passed)}"
            )
            if passed:
                break

    tb.verdict(verdict)
    return tb.build()


def build_cross_stat_judgment_thought(
    attributes_list: List[Dict[str, Any]],
    metrics: List[str],
    idx_a: int,
    idx_b: int,
    judgment_type: str,
    params: Dict[str, Any],
) -> Tuple[str, str]:
    """
    Cross-metric statistical judgment (E1–E3).

    judgment_type:
      'cross_stat_ratio'     -- std or range ratio between two metrics
      'cross_half_shift'     -- both metrics show half-mean shift > K%
      'cross_half_volatility' -- asymmetric volatility change

    params: type-specific dict with all pre-computed values including 'verdict'.
    """
    tb = ThoughtBuilder()
    verdict = params['verdict']
    attr_a = attributes_list[idx_a]
    attr_b = attributes_list[idx_b]
    m_a = metrics[idx_a]
    m_b = metrics[idx_b]

    # ── Data section ──────────────────────────────────────────────────────────
    if judgment_type == 'cross_stat_ratio':
        sub_variant = params['sub_variant']
        stats_a = attr_a['statistics']
        stats_b = attr_b['statistics']
        if sub_variant == 'std_ratio':
            tb.data(f"metric: {m_a}")
            tb.data(f"stats: std={format_float(stats_a['std'], 2)}")
            tb.metric_separator()
            tb.data(f"metric: {m_b}")
            tb.data(f"stats: std={format_float(stats_b['std'], 2)}")
        else:  # range_ratio
            tb.data(f"metric: {m_a}")
            tb.data(f"stats: max={format_float(stats_a['max'], 2)}, min={format_float(stats_a['min'], 2)}")
            tb.metric_separator()
            tb.data(f"metric: {m_b}")
            tb.data(f"stats: max={format_float(stats_b['max'], 2)}, min={format_float(stats_b['min'], 2)}")

    elif judgment_type == 'cross_half_shift':
        for m, attr in [(m_a, attr_a), (m_b, attr_b)]:
            stats = attr['statistics']
            tb.data(f"metric: {m}")
            tb.data(
                f"stats: max={format_float(stats['max'], 2)}, "
                f"min={format_float(stats['min'], 2)}, "
                f"first_half_mean={format_float(stats['first_half_mean'], 2)}, "
                f"second_half_mean={format_float(stats['second_half_mean'], 2)}"
            )
            if m == m_a:
                tb.metric_separator()

    elif judgment_type == 'cross_half_volatility':
        for m, attr in [(m_a, attr_a), (m_b, attr_b)]:
            tb.metric(m, attr, ['statistic:halves_std'])
            if m == m_a:
                tb.metric_separator()

    # ── Reasoning section ─────────────────────────────────────────────────────
    if judgment_type == 'cross_stat_ratio':
        sub_variant = params['sub_variant']
        K = params['K']
        val_a = params['val_a']
        val_b = params['val_b']

        if sub_variant != 'std_ratio':
            max_a, min_a = stats_a['max'], stats_a['min']
            max_b, min_b = stats_b['max'], stats_b['min']
            tb.line(f"{m_a} {format_add_sub_calc(max_a, min_a, val_a, op='-', label='range')}")
            tb.line(f"{m_b} {format_add_sub_calc(max_b, min_b, val_b, op='-', label='range')}")
        threshold = round(K * val_b, 2)
        tb.line(format_mul_calc(K, val_b, threshold, label='threshold'))
        val_a_r = round(val_a, 2)
        cmp_text, is_met = format_cmp(val_a_r, threshold, '>=', dp_val=2)
        tb.line(f"{cmp_text} -> {met_str(is_met)}")

    elif judgment_type == 'cross_half_shift':
        K_pct = params['K_pct']
        metrics_data = params['metrics_data']
        all_met = True
        cond_idx = 0
        for cond_idx, (m, d) in enumerate(metrics_data, 1):
            tb.condition(cond_idx, f"{m} shift")
            rng = d['range']
            abs_shift = d['shift']  # already absolute from generator
            threshold = d['threshold']
            act_max = d['max']
            act_min = d['min']
            fhm = d['first_half_mean']
            shm = d['second_half_mean']
            signed_shift = round(shm - fhm, 2)
            tb.detail(format_add_sub_calc(act_max, act_min, rng, op='-', label='range'))
            tb.detail(format_add_sub_calc(shm, fhm, signed_shift, op='-', label='shift'))
            abs_shift = round(abs(signed_shift), 2)
            tb.detail(f"|shift| = {format_float(abs_shift, 2)}")
            tb.detail(format_mul_calc(K_pct, rng, threshold, is_pct=True))
            threshold = round(threshold, 2)
            cmp_text, is_met = format_cmp(abs_shift, threshold, '>')
            tb.detail(f"{cmp_text} -> {met_str(is_met)}")
            if not is_met:
                all_met = False
                break
        # Only emit all_conditions when both conditions were shown
        if cond_idx >= 2:
            tb.all_conditions(all_met)

    elif judgment_type == 'cross_half_volatility':
        fhs_a = params['fhs_a']
        shs_a = params['shs_a']
        fhs_b = params['fhs_b']
        shs_b = params['shs_b']

        tb.condition(1, f"{m_a} volatility increases")
        cmp_text, c1_met = format_cmp(round(shs_a, 2), round(fhs_a, 2), '>')
        tb.detail(f"{cmp_text} -> {met_str(c1_met)}")

        if c1_met:
            tb.condition(2, f"{m_b} volatility stable or decreasing")
            cmp_text, c2_met = format_cmp(round(shs_b, 2), round(fhs_b, 2), '<=')
            tb.detail(f"{cmp_text} -> {met_str(c2_met)}")
            tb.all_conditions(c2_met)

    tb.verdict(verdict)
    return tb.build()
