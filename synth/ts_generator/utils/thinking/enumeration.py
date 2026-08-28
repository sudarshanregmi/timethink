"""
synth.ts_generator.utils.thinking.enumeration
================================================
CoT builders for enumeration QA families:
  - Local event enumeration (count/filter/extreme over local events)
  - Segment enumeration (count/filter/extreme over trend segments)
  - Transition enumeration (count/filter/argmax over adjacent segment transitions)
  - Cross-metric enumeration (filter/argmax/argmin across all metrics)
"""

from collections import Counter
from typing import Any, Dict, List, Tuple

from synth.ts_generator.utils.common_utils import format_float
from synth.ts_generator.utils.thinking.core import (
    ThoughtBuilder,
    _format_trend_segs,
    pass_str,
    format_add_sub_calc,
    format_cmp,

    format_seg_display,
    format_str_eq,
    format_tie,
)


def build_local_enumeration_thought(
    attributes: Dict[str, Any],
    metric: str,
    sub_type: str,
    params: Dict[str, Any],
) -> Tuple[str, str]:
    """Build CoT for local event enumeration questions.

    Sub-types:
      count_by_type, count_by_direction, count_by_amplitude,
      count_by_type_and_amp, highest_amplitude, lowest_amplitude,
      widest_span, narrowest_span

    Returns (think_str, verdict_string).
    """
    tb = ThoughtBuilder()
    local_events = attributes.get('local', [])
    verdict = str(params['verdict'])

    # ── Data block ─────────────────────────────────────────────────────
    tb.data(f"metric: {metric}")

    if sub_type in ('count_by_type',):
        ev_parts = [ev['type'] for ev in local_events]
        tb.data(f"local=[{'; '.join(ev_parts)}]")

    elif sub_type in ('count_by_direction',):
        ev_parts = [
            f"{ev['type']}, direction={ev['params']['direction']}"
            for ev in local_events
        ]
        tb.data(f"local=[{'; '.join(ev_parts)}]")

    elif sub_type in ('count_by_amplitude', 'count_by_type_and_amp',
                       'highest_amplitude', 'lowest_amplitude'):
        ev_parts = [
            f"{ev['type']}, amplitude={format_float(ev['amplitude'], 2)}"
            for ev in local_events
        ]
        tb.data(f"local=[{'; '.join(ev_parts)}]")

    elif sub_type in ('widest_span', 'narrowest_span'):
        ev_parts = [
            f"{ev['type']}, start={ev['position_start']}, end={ev['position_end']}"
            for ev in local_events if 'position_end' in ev
        ]
        tb.data(f"local=[{'; '.join(ev_parts)}]")

    # ── Computation block ──────────────────────────────────────────────

    if sub_type == 'count_by_type':
        target_type = params['target_type']
        tb.line(f"filter: type == {target_type}")
        pass_count = 0
        for i, ev in enumerate(local_events, 1):
            t = ev['type']
            eq_text, matched = format_str_eq(t, target_type)
            tb.detail(f"event {i}: {eq_text} -> {pass_str(matched)}")
            if matched:
                pass_count += 1
        tb.summary(pass_count, len(local_events))
        tb.verdict(verdict)

    elif sub_type == 'count_by_direction':
        target_dir = params['target_direction']
        tb.line(f"filter: direction == {target_dir}")
        pass_count = 0
        for i, ev in enumerate(local_events, 1):
            p = ev['params']
            d = p['direction']
            eq_text, matched = format_str_eq(d, target_dir)
            tb.detail(f"event {i}: {ev['type']}, direction {eq_text} -> {pass_str(matched)}")
            if matched:
                pass_count += 1
        tb.summary(pass_count, len(local_events))
        tb.verdict(verdict)

    elif sub_type == 'count_by_amplitude':
        threshold = params['threshold']
        threshold_str = format_float(threshold, 2)
        tb.line(f"filter: amplitude > {threshold_str}")
        pass_count = 0
        for i, ev in enumerate(local_events, 1):
            amp = ev['amplitude']
            cmp_text, passed = format_cmp(round(amp, 2), round(threshold, 2), '>', dp_val=2)
            tb.detail(f"event {i}: {ev['type']}, amplitude={cmp_text}, {pass_str(passed)}")
            if passed:
                pass_count += 1
        tb.summary(pass_count, len(local_events))
        tb.verdict(verdict)

    elif sub_type == 'count_by_type_and_amp':
        target_type = params['target_type']
        threshold = params['threshold']
        threshold_str = format_float(threshold, 2)
        tb.line(f"filter: type == {target_type} AND amplitude > {threshold_str}")
        pass_count = 0
        for i, ev in enumerate(local_events, 1):
            t = ev['type']
            amp = ev['amplitude']
            type_text, type_match = format_str_eq(t, target_type)
            if not type_match:
                # AND-compound early exit: type fails, skip amplitude check
                tb.detail(f"event {i}: type {type_text}, {pass_str(False)}")
            else:
                amp_text, amp_pass = format_cmp(round(amp, 2), round(threshold, 2), '>', dp_val=2)
                tb.detail(f"event {i}: type {type_text}, {pass_str(True)} -> amplitude {amp_text}, {pass_str(amp_pass)}")
                if amp_pass:
                    pass_count += 1
        tb.summary(pass_count, len(local_events))
        tb.verdict(verdict)

    elif sub_type in ('highest_amplitude', 'lowest_amplitude'):
        amps = []
        for i, ev in enumerate(local_events, 1):
            if 'amplitude' in ev:
                amps.append((i, ev['type'], round(ev['amplitude'], 2)))
        if sub_type == 'highest_amplitude':
            sorted_amps = sorted(amps, key=lambda x: x[2], reverse=True)
            label = "highest"
            best_val = sorted_amps[0][2]
            cmp_op = '>='
        else:
            sorted_amps = sorted(amps, key=lambda x: x[2])
            label = "lowest"
            best_val = sorted_amps[0][2]
            cmp_op = '<='
        # Pairwise comparison via format_cmp to avoid "X.XX > X.XX" contradictions
        cmp_parts = [format_float(sorted_amps[0][2], 2)]
        for k in range(1, len(sorted_amps)):
            cmp_text, _ = format_cmp(sorted_amps[k - 1][2], sorted_amps[k][2], cmp_op, dp_val=2)
            # Extract just the operator from format_cmp's text (e.g., "1.24 >= 1.24")
            cmp_parts.append(format_float(sorted_amps[k][2], 2))
        # Build display: use format_cmp for the full line to get correct operators
        if len(sorted_amps) == 1:
            tb.line(f"comparison: {format_float(sorted_amps[0][2], 2)}")
        else:
            pair_texts = []
            for k in range(1, len(sorted_amps)):
                cmp_text, _ = format_cmp(sorted_amps[k - 1][2], sorted_amps[k][2], cmp_op, dp_val=2)
                pair_texts.append(cmp_text)
            tb.line(f"comparison: {'; '.join(pair_texts)}")
        winners = [(idx, t) for idx, t, a in amps if a == best_val]
        if len(winners) > 1:
            tb.line(format_tie([f"event {idx} ({format_float(best_val, 2)})" for idx, _ in winners]))
            winner_strs = ", ".join(f"event {idx} ({t})" for idx, t in winners)
            tb.line(f"{label} amplitude = {format_float(best_val, 2)}, {winner_strs}")
        else:
            tb.line(f"{label} amplitude = {format_float(best_val, 2)}, event {winners[0][0]}, {winners[0][1]}")
        tb.verdict(verdict)

    elif sub_type in ('widest_span', 'narrowest_span'):
        spans = []
        for i, ev in enumerate(local_events, 1):
            if 'position_end' not in ev:
                continue
            display_start = ev['position_start']
            display_end = ev['position_end']
            span = display_end - display_start
            spans.append((i, ev['type'], span, display_start, display_end))
            tb.detail(f"event {i}: {format_add_sub_calc(display_end, display_start, op='-', label='duration', int_mode=True)}")
        if sub_type == 'widest_span':
            best_val = max(s[2] for s in spans)
            label = "longest duration"
        else:
            best_val = min(s[2] for s in spans)
            label = "shortest duration"
        winners = [(idx, t) for idx, t, s, _, _ in spans if s == best_val]
        if len(winners) > 1:
            tb.line(format_tie([f"event {idx} ({best_val})" for idx, _ in winners]))
            winner_strs = ", ".join(f"event {idx} ({t})" for idx, t in winners)
            tb.line(f"{label} = {best_val}, {winner_strs}")
        else:
            tb.line(f"{label} = {best_val}, event {winners[0][0]}, {winners[0][1]}")
        tb.verdict(verdict)

    return tb.build()


def build_segment_enumeration_thought(
    attributes: Dict[str, Any],
    metric: str,
    sub_type: str,
    params: Dict[str, Any],
) -> Tuple[str, str]:
    """Build CoT for trend segment enumeration questions.

    Sub-types:
      segment_count_all, segment_count_by_type,
      segment_longest, segment_shortest

    Returns (think_str, verdict_string).
    """
    tb = ThoughtBuilder()
    verdict = str(params['verdict'])

    # ── Data block — render trend segments ─────────────────────────────
    tb.metric(metric, attributes, ['trend_list'])

    # ── Computation block ──────────────────────────────────────────────
    trend_list = [s for s in attributes.get('trend_list', [])
                  if isinstance(s, (list, tuple)) and len(s) >= 3]

    if sub_type == 'segment_count_all':
        tb.line(f"count: {len(trend_list)}")
        tb.verdict(verdict)

    elif sub_type == 'segment_count_by_type':
        target_type = params['target_type']
        tb.line(f"filter: type={target_type}")
        pass_count = 0
        for seg in trend_list:
            s_type, s_start, s_end = seg[0], seg[1], seg[2]
            eq_text, matched = format_str_eq(s_type, target_type)
            tb.detail(f"({s_type}, {s_start}, {s_end}) -> {eq_text}, {pass_str(matched)}")
            if matched:
                pass_count += 1
        tb.summary(pass_count, len(trend_list))
        tb.verdict(verdict)

    elif sub_type in ('segment_longest', 'segment_shortest'):
        tb.line("durations:")
        durations = []
        for seg in trend_list:
            s_type, s_start, s_end = seg[0], seg[1], seg[2]
            dur = s_end - s_start
            durations.append((s_type, s_start, s_end, dur))
            tb.detail(f"({s_type}, {s_start}, {s_end}): {format_add_sub_calc(s_end, s_start, op='-', label='duration', int_mode=True)}")

        if sub_type == 'segment_longest':
            best_val = max(d[3] for d in durations)
            label = "longest duration"
        else:
            best_val = min(d[3] for d in durations)
            label = "shortest duration"
        winners = [(t, s, e) for t, s, e, d in durations if d == best_val]
        if len(winners) > 1:
            tb.line(format_tie([f"({t}, {s}, {e}) ({best_val})" for t, s, e in winners]))
            winner_strs = ", ".join(f"({t}, {s}, {e})" for t, s, e in winners)
            tb.line(f"{label} = {best_val}, {winner_strs}")
        else:
            w = winners[0]
            tb.line(f"{label} = {best_val}, ({w[0]}, {w[1]}, {w[2]})")
        tb.verdict(verdict)

    return tb.build()


def _find_segment_for_position(trend_list: list, pos: int) -> tuple:
    """Return (seg_type, seg_start, seg_end) for the segment containing pos."""
    for seg in trend_list:
        if seg[1] <= pos <= seg[2]:
            return seg[0], seg[1], seg[2]
    return None, None, None


def build_event_segment_enumeration_thought(
    attributes: Dict[str, Any],
    metric: str,
    sub_type: str,
    params: Dict[str, Any],
) -> Tuple[str, str]:
    """Build CoT for event-segment co-occurrence enumeration questions.

    Cross-references local event positions against trend segment boundaries.
    An event belongs to the segment containing its position_start.

    Sub-types:
      events_in_trend_type, segment_with_most_events, trend_type_with_most_events

    Returns (think_str, verdict_string).
    """
    tb = ThoughtBuilder()
    verdict = str(params['verdict'])
    local_events = attributes.get('local', [])
    trend_list = [s for s in attributes.get('trend_list', [])
                  if isinstance(s, (list, tuple)) and len(s) >= 3]

    # ── Data block — trend segments + stripped local events ────────────
    tb.metric(metric, attributes, ['trend_list'])
    ev_parts = [
        f"{ev['type']}, pos={ev['position_start']}"
        for ev in local_events
    ]
    tb.data(f"local=[{'; '.join(ev_parts)}]")

    # ── Computation block ──────────────────────────────────────────────

    if sub_type == 'events_in_trend_type':
        target_type = params['target_type']
        tb.line(f"filter: events in {target_type} segments")
        pass_count = 0
        for i, ev in enumerate(local_events, 1):
            pos = ev['position_start']
            seg_type, seg_s, seg_e = _find_segment_for_position(trend_list, pos)
            if seg_type is not None:
                seg_ref = format_seg_display(seg_type, seg_s, seg_e)
                eq_text, matched = format_str_eq(seg_type, target_type)
                tb.detail(f"event {i}: pos={pos} in {seg_ref} -> {eq_text}, {pass_str(matched)}")
                if matched:
                    pass_count += 1
            else:
                tb.detail(f"event {i}: pos={pos} -> no segment, {pass_str(False)}")
        tb.summary(pass_count, len(local_events))
        tb.verdict(verdict)

    elif sub_type == 'segment_with_most_events':
        # Count events per segment
        seg_counts = []
        for seg in trend_list:
            s_type, s_start, s_end = seg[0], seg[1], seg[2]
            count = sum(
                1 for ev in local_events
                if s_start <= ev['position_start'] <= s_end
            )
            seg_ref = format_seg_display(s_type, s_start, s_end)
            tb.line(f"{seg_ref}: {count} events")
            seg_counts.append((s_type, s_start, s_end, count))

        best_count = max(c for _, _, _, c in seg_counts)
        winners = [(t, s, e) for t, s, e, c in seg_counts if c == best_count]
        if len(winners) > 1:
            tb.line(format_tie([
                f"{format_seg_display(t, s, e)} ({best_count})"
                for t, s, e in winners
            ]))
            winner_strs = ", ".join(format_seg_display(t, s, e) for t, s, e in winners)
            tb.line(f"max events = {best_count}, segments: {winner_strs}")
        else:
            w = winners[0]
            tb.line(f"max events = {best_count}, segment: {format_seg_display(w[0], w[1], w[2])}")
        tb.verdict(verdict)

    elif sub_type == 'trend_type_with_most_events':
        # Count events per trend type (aggregate across segments of same type)
        type_counts: Dict[str, int] = {}
        for ev in local_events:
            pos = ev['position_start']
            seg_type, _, _ = _find_segment_for_position(trend_list, pos)
            if seg_type is not None:
                type_counts[seg_type] = type_counts.get(seg_type, 0) + 1
        # Show all trend types (including those with 0 events)
        all_types = list(dict.fromkeys(seg[0] for seg in trend_list))
        for t in all_types:
            if t not in type_counts:
                type_counts[t] = 0
        tb.line("counts: " + ", ".join(
            f"{t}={type_counts[t]}" for t in all_types
        ))
        best_count = max(type_counts.values())
        winners = [t for t in all_types if type_counts[t] == best_count]
        if len(winners) > 1:
            tb.line(format_tie([f"{t} ({best_count})" for t in winners]))
            tb.line(f"max events = {best_count}, types: {', '.join(winners)}")
        else:
            tb.line(f"max events = {best_count}, type: {winners[0]}")
        tb.verdict(verdict)

    return tb.build()


def build_temporal_position_thought(
    attributes: Dict[str, Any],
    metric: str,
    sub_type: str,
    params: Dict[str, Any],
) -> Tuple[str, str]:
    """Build CoT for temporal position / inter-event gap questions.

    Sub-types:
      first_event, last_event, largest_gap, smallest_gap

    Returns (think_str, verdict_string).
    """
    tb = ThoughtBuilder()
    verdict = str(params['verdict'])
    local_events = attributes['local']

    # ── Data block — events with type + position only ─────────────────
    tb.data(f"metric: {metric}")
    ev_parts = [
        f"{ev['type']}, pos={ev['position_start']}"
        for ev in local_events
    ]
    tb.data(f"local=[{'; '.join(ev_parts)}]")

    # ── Computation ───────────────────────────────────────────────────
    positions = sorted(set(ev['position_start'] for ev in local_events))
    tb.line(f"sorted positions: {', '.join(str(p) for p in positions)}")

    if sub_type in ('first_event', 'last_event'):
        label = "earliest" if sub_type == 'first_event' else "latest"
        tb.line(f"{label} = {verdict}")
        tb.verdict(verdict)

    elif sub_type in ('largest_gap', 'smallest_gap'):
        tb.line("gaps:")
        gaps = []
        for j in range(len(positions) - 1):
            a, b = positions[j], positions[j + 1]
            gap = b - a
            gaps.append(gap)
            tb.detail(format_add_sub_calc(b, a, gap, op='-', label='gap', int_mode=True))
        label = "largest gap" if sub_type == 'largest_gap' else "smallest gap"
        best = max(gaps) if sub_type == 'largest_gap' else min(gaps)
        tb.line(f"{label} = {best}")
        tb.verdict(verdict)

    return tb.build()


def build_duration_proportion_thought(
    attributes: Dict[str, Any],
    metric: str,
    sub_type: str,
    params: Dict[str, Any],
) -> Tuple[str, str]:
    """Build CoT for segment duration questions (no ratio arithmetic).

    Computes per-segment durations, aggregates by type. Dispatches on sub_type
    for duration lookup, ranking, comparison, or difference.

    Sub-types:
      duration_by_type, longest_total_duration, shortest_total_duration,
      duration_comparison, duration_difference

    Returns (think_str, verdict_string).
    """
    tb = ThoughtBuilder()
    verdict = str(params['verdict'])
    trend_list = [s for s in attributes['trend_list']
                  if isinstance(s, (list, tuple)) and len(s) >= 3]

    # ── Data block ────────────────────────────────────────────────────
    tb.metric(metric, attributes, ['trend_list'])

    # ── Computation — per-segment durations ───────────────────────────
    tb.line("durations:")
    durations = []
    for seg in trend_list:
        s_type, s_start, s_end = seg[0], seg[1], seg[2]
        dur = s_end - s_start
        durations.append((s_type, s_start, s_end, dur))
        seg_ref = format_seg_display(s_type, s_start, s_end)
        tb.detail(f"{seg_ref}: {format_add_sub_calc(s_end, s_start, dur, op='-', label='duration', int_mode=True)}")

    # ── Aggregate by type ─────────────────────────────────────────────
    all_types = list(dict.fromkeys(seg[0] for seg in trend_list))
    type_parts: Dict[str, List[int]] = {}
    for s_type, _, _, dur in durations:
        type_parts.setdefault(s_type, []).append(dur)

    type_totals = {t: sum(type_parts[t]) for t in all_types}

    tb.line("aggregate:")
    for ttype in all_types:
        durs = type_parts[ttype]
        if len(durs) == 1:
            tb.detail(f"{ttype}: {durs[0]}")
        else:
            sum_expr = " + ".join(str(d) for d in durs)
            tb.detail(f"{ttype}: {sum_expr} = {sum(durs)}")

    # ── Sub-type dispatch ─────────────────────────────────────────────
    if sub_type == 'duration_by_type':
        tb.verdict(verdict)

    elif sub_type == 'longest_total_duration':
        best = max(type_totals.values())
        winners = [t for t in all_types if type_totals[t] == best]
        if len(winners) > 1:
            tb.line(format_tie([f"{t} ({type_totals[t]})" for t in winners]))
        else:
            tb.line(f"largest total = {winners[0]} ({best})")
        tb.verdict(verdict)

    elif sub_type == 'shortest_total_duration':
        worst = min(type_totals.values())
        winners = [t for t in all_types if type_totals[t] == worst]
        if len(winners) > 1:
            tb.line(format_tie([f"{t} ({type_totals[t]})" for t in winners]))
        else:
            tb.line(f"smallest total = {winners[0]} ({worst})")
        tb.verdict(verdict)

    elif sub_type == 'duration_comparison':
        a = params['type_a']
        b = params['type_b']
        ta = type_totals[a]
        tb_val = type_totals[b]
        if ta > tb_val:
            cmp_text, _ = format_cmp(ta, tb_val, '>', dp_val=0)
            tb.line(f"{a}: {cmp_text} -> winner = {a}")
        elif tb_val > ta:
            cmp_text, _ = format_cmp(tb_val, ta, '>', dp_val=0)
            tb.line(f"{b}: {cmp_text} -> winner = {b}")
        else:
            tb.line(f"{a} == {b} ({ta}) -> winner = equal")
        tb.verdict(verdict)

    elif sub_type == 'duration_difference':
        a = params['type_a']
        b = params['type_b']
        ta = type_totals[a]
        tb_val = type_totals[b]
        hi, lo = max(ta, tb_val), min(ta, tb_val)
        diff = hi - lo
        tb.line(format_add_sub_calc(hi, lo, diff, op='-', label='|diff|', int_mode=True))
        tb.verdict(verdict)

    return tb.build()


def build_transition_enumeration_thought(
    attributes: Dict[str, Any],
    metric: str,
    sub_type: str,
    params: Dict[str, Any],
) -> Tuple[str, str]:
    """Build CoT for trend transition enumeration questions.

    Analyses adjacent segment pairs (A→B transitions) in trend_list.

    Sub-types:
      transition_count, transition_count_by_type, most_common_transition

    Returns (think_str, verdict_string).
    """
    tb = ThoughtBuilder()
    verdict = str(params['verdict'])

    # ── Data block — render trend segments ─────────────────────────────
    tb.metric(metric, attributes, ['trend_list'])

    # ── Computation block ──────────────────────────────────────────────
    trend_list = [s for s in attributes.get('trend_list', [])
                  if isinstance(s, (list, tuple)) and len(s) >= 3]

    # Build transition list from adjacent pairs
    transitions = []
    for i in range(len(trend_list) - 1):
        t_from = trend_list[i][0]
        t_to = trend_list[i + 1][0]
        transitions.append(f"{t_from}→{t_to}")

    if sub_type == 'transition_count':
        n_segs = len(trend_list)
        seg_types = ", ".join(s[0] for s in trend_list)
        tb.line(f"segments: {seg_types} ({n_segs} total)")
        tb.line(format_add_sub_calc(n_segs, 1, n_segs - 1, op='-',
                                    label='transitions', int_mode=True))
        tb.verdict(verdict)

    elif sub_type == 'transition_count_by_type':
        target = params['target_transition']
        tb.line(f"filter: {target}")
        pass_count = 0
        for i, tr in enumerate(transitions, 1):
            eq_text, matched = format_str_eq(tr, target)
            tb.detail(f"transition {i}: {eq_text} -> {pass_str(matched)}")
            if matched:
                pass_count += 1
        tb.summary(pass_count, len(transitions))
        tb.verdict(verdict)

    elif sub_type == 'most_common_transition':
        if not transitions:
            raise ValueError("most_common_transition requires at least 1 transition (2+ segments)")
        counts = Counter(transitions)
        tb.line("counts: " + ", ".join(
            f"{tr}={c}" for tr, c in counts.most_common()
        ))
        best_count = counts.most_common(1)[0][1]
        winners = [tr for tr, c in counts.items() if c == best_count]
        if len(winners) > 1:
            tb.line(format_tie([f"{tr} ({best_count})" for tr in winners]))
            tb.line(f"most common = {best_count}, transitions: {', '.join(winners)}")
        else:
            tb.line(f"most common = {best_count}, transition: {winners[0]}")
        tb.verdict(verdict)

    return tb.build()


def build_cross_metric_enumeration_thought(
    attributes_list: List[Dict[str, Any]],
    metrics: List[str],
    sub_type: str,
    params: Dict[str, Any],
) -> Tuple[str, str]:
    """Build CoT for cross-metric enumeration questions.

    Scans ALL metrics and compares/filters across the whole set.
    Data block uses canonical formats from _render_metric_attrs (via
    tb.metric()) — local, trend_overall, trend_list, noise.
    Computation block does NOT re-list data — directly computes result.

    Sub-types:
      which_have_local, which_have_trend_type, all_same_trend,
      most_local_events, highest_amplitude_across, most_trend_segments,
      longest_segment_across, noisiest_metric, quietest_metric, widest_range,
      highest_mean, lowest_mean, highest_max, lowest_min,
      highest_std, lowest_std,
      earliest_event_across, latest_event_across, largest_gap_across

    Returns (think_str, verdict_string).
    """
    tb = ThoughtBuilder()
    verdict = str(params['verdict'])
    per_metric = params['per_metric']

    # ── Data block — canonical format per metric ──────────────────────
    for idx, (attrs, name) in enumerate(zip(attributes_list, metrics)):
        if idx > 0:
            tb.metric_separator()

        if sub_type in ('which_have_local', 'most_local_events'):
            # local=[type1; type2; ...] — type only (stripped)
            stripped = dict(attrs)
            stripped['local'] = [{'type': ev['type']}
                                 for ev in attrs.get('local', [])]
            tb.metric(name, stripped, ['local'])

        elif sub_type == 'highest_amplitude_across':
            # local=[type, amplitude=X; ...] — type + amplitude (stripped)
            stripped = dict(attrs)
            stripped['local'] = [
                {'type': ev['type'], 'amplitude': ev['amplitude']}
                for ev in attrs.get('local', []) if 'amplitude' in ev
            ]
            tb.metric(name, stripped, ['local'])

        elif sub_type in ('which_have_trend_type', 'all_same_trend'):
            # start=X, end=Y, amp=Z, overall trend=TYPE
            tb.metric(name, attrs, ['trend_overall'])

        elif sub_type in ('most_trend_segments', 'longest_segment_across'):
            # trend_segments=[(type, s, e), ...]
            tb.metric(name, attrs, ['trend_list'])

        elif sub_type in ('noisiest_metric', 'quietest_metric'):
            strength = attrs['noise']['strength']
            tb.data(f"metric: {name}")
            tb.data(f"noise strength={format_float(strength, 2)}")

        elif sub_type == 'widest_range':
            max_v = attrs['statistics']['max']
            min_v = attrs['statistics']['min']
            tb.data(f"metric: {name}")
            tb.data(f"stats: max={format_float(max_v, 2)}, min={format_float(min_v, 2)}")

        elif sub_type in ('highest_mean', 'lowest_mean'):
            mean_v = attrs['statistics']['mean']
            tb.data(f"metric: {name}")
            tb.data(f"stats: mean={format_float(mean_v, 2)}")

        elif sub_type == 'highest_max':
            max_v = attrs['statistics']['max']
            tb.data(f"metric: {name}")
            tb.data(f"stats: max={format_float(max_v, 2)}")

        elif sub_type == 'lowest_min':
            min_v = attrs['statistics']['min']
            tb.data(f"metric: {name}")
            tb.data(f"stats: min={format_float(min_v, 2)}")

        elif sub_type in ('highest_std', 'lowest_std'):
            std_v = attrs['statistics']['std']
            tb.data(f"metric: {name}")
            tb.data(f"stats: std={format_float(std_v, 2)}")

        elif sub_type in ('earliest_event_across', 'latest_event_across',
                          'largest_gap_across'):
            tb.data(f"metric: {name}")
            ev_parts = [
                f"{ev['type']}, pos={ev['position_start']}"
                for ev in attrs['local']
            ]
            tb.data(f"local=[{'; '.join(ev_parts)}]")

    # ── Computation block — no re-listing, no headers, no indentation ──

    if sub_type == 'which_have_local':
        pass_names = []
        for attrs, name in zip(attributes_list, metrics):
            count = len(attrs.get('local', []))
            passed = count > 0
            tb.line(f"{name}: {count} events -> {pass_str(passed)}")
            if passed:
                pass_names.append(name)
        tb.summary(len(pass_names), len(metrics))
        tb.verdict(verdict)

    elif sub_type == 'which_have_trend_type':
        target_type = params['target_type']
        pass_names = []
        for attrs, name in zip(attributes_list, metrics):
            trend = attrs.get('trend', {})
            t = trend['type'] if isinstance(trend, dict) and trend else '?'
            eq_text, passed = format_str_eq(t, target_type)
            tb.line(f"{name}: {eq_text} -> {pass_str(passed)}")
            if passed:
                pass_names.append(name)
        tb.summary(len(pass_names), len(metrics))
        tb.verdict(verdict)

    elif sub_type == 'all_same_trend':
        # Early exit: stop as soon as 2 distinct types found
        types_seen = set()
        for attrs, name in zip(attributes_list, metrics):
            trend = attrs.get('trend', {})
            t = trend['type'] if isinstance(trend, dict) and trend else '?'
            types_seen.add(t)
            tb.line(f"{name}: {t} -> types_seen={{{', '.join(sorted(types_seen))}}}")
            if len(types_seen) > 1:
                tb.line(f"{len(types_seen)} distinct types found -> not all same")
                break
        else:
            tb.line("1 type across all metrics -> all same")
        tb.verdict(verdict)

    elif sub_type == 'most_local_events':
        counts = [(name, len(attrs.get('local', [])))
                   for attrs, name in zip(attributes_list, metrics)]
        counts_str = ", ".join(f"{n}={c}" for n, c in counts)
        tb.line(f"event counts: {counts_str}")
        _emit_extremum(tb, "max events", counts, verdict)

    elif sub_type == 'highest_amplitude_across':
        values = []
        for idx, (attrs, name) in enumerate(zip(attributes_list, metrics)):
            info = per_metric[idx]
            if info['type'] == 'none':
                tb.line(f"{name}: no events")
            else:
                tb.line(f"{name}: max amplitude = {format_float(info['amp'], 2)} ({info['type']})")
                values.append((name, info['amp']))
        _emit_extremum(tb, "max amplitude", values, verdict,
                       fmt_fn=lambda v: format_float(v, 2))

    elif sub_type == 'most_trend_segments':
        counts = []
        for attrs, name in zip(attributes_list, metrics):
            tl = [s for s in attrs.get('trend_list', [])
                  if isinstance(s, (list, tuple)) and len(s) >= 3]
            counts.append((name, len(tl)))
        counts_str = ", ".join(f"{n}={c}" for n, c in counts)
        tb.line(f"segment counts: {counts_str}")
        _emit_extremum(tb, "max segments", counts, verdict)

    elif sub_type == 'longest_segment_across':
        values = []
        for attrs, name in zip(attributes_list, metrics):
            tl = [s for s in attrs.get('trend_list', [])
                  if isinstance(s, (list, tuple)) and len(s) >= 3]
            if not tl:
                tb.line(f"{name}: no segments")
                continue
            # Show all segment durations on one line with math shown
            parts = []
            best_dur = 0
            for i, seg in enumerate(tl, 1):
                s_start, s_end = seg[1], seg[2]
                dur = s_end - s_start
                parts.append(f"seg{i}: {format_add_sub_calc(s_end, s_start, op='-', label='duration', int_mode=True)}")
                if dur > best_dur:
                    best_dur = dur
            tb.line(f"{name}: {', '.join(parts)} -> longest={best_dur}")
            values.append((name, best_dur))
        _emit_extremum(tb, "max duration", values, verdict)

    elif sub_type == 'noisiest_metric':
        strengths = [(name, attrs['noise']['strength'])
                     for attrs, name in zip(attributes_list, metrics)]
        tb.line("values: " + ", ".join(f"{n}={format_float(v, 2)}" for n, v in strengths))
        _emit_extremum(tb, "max noise strength", strengths, verdict,
                       fmt_fn=lambda v: format_float(v, 2))

    elif sub_type == 'quietest_metric':
        strengths = [(name, attrs['noise']['strength'])
                     for attrs, name in zip(attributes_list, metrics)]
        tb.line("values: " + ", ".join(f"{n}={format_float(v, 2)}" for n, v in strengths))
        _emit_extremum(tb, "min noise strength", strengths, verdict,
                       fmt_fn=lambda v: format_float(v, 2), mode='min')

    elif sub_type == 'widest_range':
        ranges = []
        for attrs, name in zip(attributes_list, metrics):
            max_v = attrs['statistics']['max']
            min_v = attrs['statistics']['min']
            rng = attrs['statistics']['range']
            tb.line(f"{name}: {format_add_sub_calc(max_v, min_v, rng, op='-', label='range')}")
            ranges.append((name, rng))
        _emit_extremum(tb, "max range", ranges, verdict,
                       fmt_fn=lambda v: format_float(v, 2))

    elif sub_type == 'highest_mean':
        values = [(name, attrs['statistics']['mean'])
                  for attrs, name in zip(attributes_list, metrics)]
        tb.line("values: " + ", ".join(f"{n}={format_float(v, 2)}" for n, v in values))
        _emit_extremum(tb, "max mean", values, verdict,
                       fmt_fn=lambda v: format_float(v, 2))

    elif sub_type == 'lowest_mean':
        values = [(name, attrs['statistics']['mean'])
                  for attrs, name in zip(attributes_list, metrics)]
        tb.line("values: " + ", ".join(f"{n}={format_float(v, 2)}" for n, v in values))
        _emit_extremum(tb, "min mean", values, verdict,
                       fmt_fn=lambda v: format_float(v, 2), mode='min')

    elif sub_type == 'highest_max':
        values = [(name, attrs['statistics']['max'])
                  for attrs, name in zip(attributes_list, metrics)]
        tb.line("values: " + ", ".join(f"{n}={format_float(v, 2)}" for n, v in values))
        _emit_extremum(tb, "max peak", values, verdict,
                       fmt_fn=lambda v: format_float(v, 2))

    elif sub_type == 'lowest_min':
        values = [(name, attrs['statistics']['min'])
                  for attrs, name in zip(attributes_list, metrics)]
        tb.line("values: " + ", ".join(f"{n}={format_float(v, 2)}" for n, v in values))
        _emit_extremum(tb, "min trough", values, verdict,
                       fmt_fn=lambda v: format_float(v, 2), mode='min')

    elif sub_type == 'highest_std':
        values = [(name, attrs['statistics']['std'])
                  for attrs, name in zip(attributes_list, metrics)]
        tb.line("values: " + ", ".join(f"{n}={format_float(v, 2)}" for n, v in values))
        _emit_extremum(tb, "max std", values, verdict,
                       fmt_fn=lambda v: format_float(v, 2))

    elif sub_type == 'lowest_std':
        values = [(name, attrs['statistics']['std'])
                  for attrs, name in zip(attributes_list, metrics)]
        tb.line("values: " + ", ".join(f"{n}={format_float(v, 2)}" for n, v in values))
        _emit_extremum(tb, "min std", values, verdict,
                       fmt_fn=lambda v: format_float(v, 2), mode='min')

    elif sub_type == 'earliest_event_across':
        values = []
        for attrs, name in zip(attributes_list, metrics):
            local = attrs['local']
            if not local:
                tb.line(f"{name}: no events")
                continue
            positions = sorted(ev['position_start'] for ev in local)
            tb.line(f"{name}: first event at {positions[0]}")
            values.append((name, positions[0]))
        _emit_extremum(tb, "min first-event position", values, verdict, mode='min')

    elif sub_type == 'latest_event_across':
        values = []
        for attrs, name in zip(attributes_list, metrics):
            local = attrs['local']
            if not local:
                tb.line(f"{name}: no events")
                continue
            positions = sorted(ev['position_start'] for ev in local)
            tb.line(f"{name}: last event at {positions[-1]}")
            values.append((name, positions[-1]))
        _emit_extremum(tb, "max last-event position", values, verdict)

    elif sub_type == 'largest_gap_across':
        values = []
        for attrs, name in zip(attributes_list, metrics):
            local = attrs['local']
            positions = sorted(set(ev['position_start'] for ev in local))
            if len(positions) < 2:
                tb.line(f"{name}: <2 positions, skip")
                continue
            gaps = [positions[j + 1] - positions[j] for j in range(len(positions) - 1)]
            max_gap = max(gaps)
            gap_strs = [format_add_sub_calc(positions[j + 1], positions[j],
                                            op='-', label='gap', int_mode=True)
                        for j in range(len(positions) - 1)]
            tb.line(f"{name}: {', '.join(gap_strs)} -> max gap={max_gap}")
            values.append((name, max_gap))
        _emit_extremum(tb, "max inter-event gap", values, verdict)

    return tb.build()


def _emit_extremum(
    tb: ThoughtBuilder,
    label: str,
    values: List[Tuple[str, Any]],
    verdict: str,
    fmt_fn=None,
    mode: str = 'max',
) -> None:
    """Emit extremum scan + result line + verdict. References all values."""
    if not values:
        raise ValueError(f"_emit_extremum called with empty values for label={label!r}")
    if fmt_fn is None:
        fmt_fn = str
    op = '>' if mode == 'max' else '<'
    # Scan all values so every data-block entry is referenced in reasoning
    if len(values) > 1:
        best_name, best_val = values[0]
        tb.line(f"start: {best_name} ({fmt_fn(best_val)})")
        for name, v in values[1:]:
            cmp_text, is_better = format_cmp(round(v, 2), round(best_val, 2), op, dp_val=2)
            if is_better:
                tb.line(f"{name}: {cmp_text} -> new {mode}")
                best_name, best_val = name, v
            else:
                tb.line(f"{name}: {cmp_text} -> keep current")
    else:
        best_val = values[0][1]
    winners = [name for name, v in values if v == best_val]
    if len(winners) > 1:
        tb.line(format_tie([f"{n} ({fmt_fn(best_val)})" for n in winners]))
        tb.line(f"{label} = {fmt_fn(best_val)}, metrics: {', '.join(winners)}")
    else:
        tb.line(f"{label} = {fmt_fn(best_val)}, metric: {winners[0]}")
    tb.verdict(verdict)
