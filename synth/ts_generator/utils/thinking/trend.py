from typing import Any, Dict, List, Tuple
from synth.ts_generator.utils.common_utils import format_float
from synth.ts_generator.utils.thinking.core import _anti_trend_type, ThoughtBuilder, met_str, format_cmp, format_add_sub_calc, format_seg_display, format_str_eq, format_abs_diff_cmp


# ---------------------------------------------------------------------------
# Shared helpers for trend dominance / cross-trend builders
# ---------------------------------------------------------------------------

def _emit_aggregate_durations(tb: ThoughtBuilder, type_parts: Dict[str, List[int]]) -> None:
    """Emit ``Aggregate Durations:`` block with per-type sums."""
    tb.line("aggregate durations:")
    for ttype in sorted(type_parts.keys()):
        durs = type_parts[ttype]
        if len(durs) == 1:
            tb.detail(f"- {ttype}: {durs[0]}")
        else:
            sum_expr = " + ".join(str(d) for d in durs)
            tb.detail(f"- {ttype}: {sum_expr} = {sum(durs)}")


def _emit_dominant_trend(tb: ThoughtBuilder, coverage: Dict[str, int]) -> str:
    """Emit ``Dominant trend comparison:`` block. Returns verdict string."""
    sorted_items = sorted(coverage.items(), key=lambda x: x[1], reverse=True)
    if len(sorted_items) == 1:
        verdict = sorted_items[0][0]
        tb.line(f"dominant trend comparison: {verdict} ({format_float(sorted_items[0][1], 0)})")
    else:
        verdict = "equal" if sorted_items[0][1] == sorted_items[1][1] else sorted_items[0][0]
        cmp_parts = [f"{sorted_items[0][0]} ({format_float(sorted_items[0][1], 0)})"]
        for i in range(1, len(sorted_items)):
            op = "==" if sorted_items[i - 1][1] == sorted_items[i][1] else ">"
            cmp_parts.append(f"{op} {sorted_items[i][0]} ({format_float(sorted_items[i][1], 0)})")
        tb.line(f"dominant trend comparison: {' '.join(cmp_parts)}")
    return verdict


# ---------------------------------------------------------------------------
# Family 8: Trend dominance in range (single metric)
# ---------------------------------------------------------------------------

def build_trend_dominance_thought(
    attributes: Dict[str, Any],
    metric: str,
    query_a: int,
    query_b: int,
) -> Tuple[str, str]:
    """
    Family 8: In the range [query_a, query_b], which trend type dominates?

    Structure:
      Data: trend_segments=
      ===
      Intersecting [qa, qb] with metric trend segments:
        - type[seg_s, seg_e] ∩ [qa, qb] = [clip_s, clip_e] -> duration = clip_e - clip_s = dur
      Aggregate Durations:
        - type: dur  (or  type: d1 + d2 = total  when multiple segments)
      Dominant trend comparison: type1 (d1) > type2 (d2) > ...
      verdict: increase | decrease | keep steady | equal

    Uses exclusive-right endpoint convention: duration = end - start.
    Returns (think_str, verdict).
    """
    tb = ThoughtBuilder()
    trend_segs = [s for s in attributes.get('trend_list', []) if isinstance(s, (list, tuple)) and len(s) >= 3]

    # Data
    tb.metric(metric, attributes, ['trend_list'])

    # Reasoning
    tb.line(f"intersecting [{query_a}, {query_b}] with {metric} trend segments:")

    coverage: Dict[str, int] = {}
    type_parts: Dict[str, List[int]] = {}
    intersection_lines: List[str] = []

    for seg in trend_segs:
        seg_type, seg_s, seg_e = seg[0], seg[1], seg[2]
        clip_s = max(query_a, seg_s)
        clip_e = min(query_b, seg_e)
        dur = max(0, clip_e - clip_s)
        if dur > 0:
            intersection_lines.append(
                f"- {format_seg_display(seg_type, seg_s, seg_e)} ∩ [{query_a}, {query_b}]"
                f" = [{clip_s}, {clip_e}] -> {format_add_sub_calc(clip_e, clip_s, op='-', label='duration', int_mode=True)}"
            )
            coverage[seg_type] = coverage.get(seg_type, 0) + dur
            type_parts.setdefault(seg_type, []).append(dur)

    if not intersection_lines:
        tb.detail("(no segments overlap the query range)")
        tb.blank()
        tb.verdict("equal")
        return tb.build()

    for il in intersection_lines:
        tb.detail(il)

    _emit_aggregate_durations(tb, type_parts)
    verdict = _emit_dominant_trend(tb, coverage)

    tb.verdict(verdict)
    return tb.build()


# ---------------------------------------------------------------------------
# Multi-metric anti-trend compound judgment
# ---------------------------------------------------------------------------

def build_anti_judgment_thought(
    attrs_a: Dict[str, Any],
    attrs_b: Dict[str, Any],
    name_a: str,
    name_b: str,
    judgment_type: str,
    params: Dict[str, Any],
) -> Tuple[str, str]:
    """
    Multi-metric compound judgment for anticorrelation scenarios.

    judgment_type: 'anti_trend_noise'
        condition 1: A and B have anticorrelated trend segments (all segments match the flip rule)
        condition 2: B's noise strength > threshold

    verdict: 'yes' | 'no'

    Returns (think_str, verdict).
    """
    tb = ThoughtBuilder()
    verdict = params['verdict']
    tl_a = [s for s in attrs_a.get('trend_list', []) if isinstance(s, (list, tuple)) and len(s) >= 3]
    tl_b = [s for s in attrs_b.get('trend_list', []) if isinstance(s, (list, tuple)) and len(s) >= 3]

    # Data
    tb.metric(name_a, attrs_a, ['trend_list'])
    tb.metric_separator()
    tb.metric(name_b, attrs_b, ['trend_list'])
    ns_b = attrs_b.get('noise', {})
    if isinstance(ns_b, dict) and ns_b:
        tb.data(f"noise={ns_b['type']}, strength={format_float(ns_b['strength'], 2)}")

    # Reasoning
    if judgment_type == 'anti_trend_noise':
        threshold = params['threshold']
        act_noise_amp = params['actual_noise_amp']
        act_noise_type = params['actual_noise_type']

        # Condition 1: anticorrelated trend segments
        tb.condition(1, f"{name_a} and {name_b} have anticorrelated trend segments")
        if len(tl_a) != len(tl_b):
            tb.detail(f"segment count: {len(tl_a)} vs {len(tl_b)} -> mismatch -> {met_str(False)}")
            c1_met = False
        else:
            tb.detail(f"segment count: {len(tl_a)} vs {len(tl_b)} -> {met_str(True)}")
            c1_met = True
            for i, (sa, sb) in enumerate(zip(tl_a, tl_b)):
                ta, tb_type = sa[0], sb[0]
                expected = _anti_trend_type(ta)
                eq_text, match_ok = format_str_eq(tb_type, expected)
                if match_ok:
                    tb.detail(f"seg {i}: {ta} vs {tb_type} (opposite {eq_text}, {met_str(True)})")
                else:
                    tb.detail(f"seg {i}: {ta} vs {tb_type} (expected {expected}) -> {eq_text} -> {met_str(False)}")
                    c1_met = False
                    break
            if c1_met:
                tb.detail(f"-> {met_str(True)}")

        if not c1_met:
            tb.verdict(verdict)
            return tb.build()

        # Condition 2: B noise strength > threshold (only reached if c1 passed)
        tb.condition(2, f"{name_b} noise strength > {format_float(threshold, 2)}")
        cmp_text, c2_met = format_cmp(act_noise_amp, threshold, '>', dp_val=2)
        tb.detail(
            f"actual: {act_noise_type}, strength={format_float(act_noise_amp, 2)} -> "
            f"{cmp_text} -> {met_str(c2_met)}"
        )

        all_met = c2_met
        tb.all_conditions(all_met)

    tb.verdict(verdict)
    return tb.build()


# ---------------------------------------------------------------------------
# Multi-metric cross-trend range query
# ---------------------------------------------------------------------------

def build_cross_trend_thought(
    attrs_a: Dict[str, Any],
    attrs_b: Dict[str, Any],
    name_a: str,
    name_b: str,
    target_trend_type: str,
) -> Tuple[str, str]:
    """
    Multi-metric cross-trend query: when metric A shows [target_trend_type],
    what does metric B's trend look like?

    Structure:
      Data: trend_segments= for both metrics
      ===
      Intersecting A [s, e] with B segments:
        - type[seg_s, seg_e] ∩ [a_s, a_e] = [clip_s, clip_e] -> duration = ...
      Aggregate Durations:
      Dominant trend comparison:
      verdict: increase | decrease | keep steady | equal | none

    Returns (think_str, verdict).
    """
    tb = ThoughtBuilder()
    tl_a = [s for s in attrs_a.get('trend_list', []) if isinstance(s, (list, tuple)) and len(s) >= 3]
    tl_b = [s for s in attrs_b.get('trend_list', []) if isinstance(s, (list, tuple)) and len(s) >= 3]

    # Data
    tb.metric(name_a, attrs_a, ['trend_list'])
    tb.metric_separator()
    tb.metric(name_b, attrs_b, ['trend_list'])

    # Reasoning
    # Collect A's target-type segment ranges using exclusive-right endpoint convention
    a_ranges = []
    for seg in tl_a:
        seg_type, seg_s1, seg_e1 = seg[0], seg[1], seg[2]
        if seg_type == target_trend_type:
            a_ranges.append((seg_s1, seg_e1))

    if not a_ranges:
        tb.line(f"{name_a} has no {target_trend_type} segments")
        tb.verdict("none")
        return tb.build()

    # Header
    if len(a_ranges) == 1:
        a_s, a_e = a_ranges[0]
        tb.line(f"intersecting {name_a} [{a_s}, {a_e}] with {name_b} segments:")
    else:
        ranges_str = ", ".join(f"[{s}, {e}]" for s, e in a_ranges)
        tb.line(f"intersecting {name_a} {target_trend_type} segments ({ranges_str}) with {name_b} segments:")

    # Compute B's overlap with each A range using exclusive-right endpoint logic
    coverage: Dict[str, int] = {}
    type_parts: Dict[str, List[int]] = {}
    intersection_lines: List[str] = []

    for seg in tl_b:
        seg_type, seg_s, seg_e = seg[0], seg[1], seg[2]
        for a_s, a_e in a_ranges:
            clip_s = max(a_s, seg_s)
            clip_e = min(a_e, seg_e)
            dur = max(0, clip_e - clip_s)
            if dur > 0:
                intersection_lines.append(
                    f"- {format_seg_display(seg_type, seg_s, seg_e)} ∩ [{a_s}, {a_e}]"
                    f" = [{clip_s}, {clip_e}] -> {format_add_sub_calc(clip_e, clip_s, op='-', label='duration', int_mode=True)}"
                )
                coverage[seg_type] = coverage.get(seg_type, 0) + dur
                type_parts.setdefault(seg_type, []).append(dur)

    if not intersection_lines:
        tb.detail("(no overlap)")
        tb.verdict("none")
        return tb.build()

    for il in intersection_lines:
        tb.detail(il)

    _emit_aggregate_durations(tb, type_parts)
    verdict = _emit_dominant_trend(tb, coverage)

    tb.verdict(verdict)
    return tb.build()


# ---------------------------------------------------------------------------
# Change point detection QA thought builder
# ---------------------------------------------------------------------------

def build_change_point_thought(
    attributes: Dict[str, Any],
    metric: str,
    sub_type: str,
    params: Dict[str, Any],
) -> Tuple[str, str]:
    """
    Change point detection QA thought builder.

    Sub-types:
      change_point_count     — how many behavioral changes
      change_point_positions — at what positions do changes occur
      largest_level_shift    — which change has the largest shift in level

    Returns (think_str, verdict_str).
    """
    tb = ThoughtBuilder()
    trend_list = params['trend_list']
    # Filter to valid segments (tuples with >= 3 elements)
    segs = [s for s in trend_list if isinstance(s, (list, tuple)) and len(s) >= 3]

    if sub_type == 'change_point_count':
        # Data: show trend segments
        tb.metric(metric, attributes, ['trend_list'])

        # Reasoning: change points = n_segments - 1
        n_segments = len(segs)
        count = n_segments - 1
        tb.line(format_add_sub_calc(n_segments, 1, count, op='-', label='change_points', int_mode=True))
        verdict = str(count)

    elif sub_type == 'change_point_positions':
        # Data: show trend segments
        tb.metric(metric, attributes, ['trend_list'])

        # Reasoning: boundary at each adjacent pair
        positions = []
        cap = min(len(segs) - 1, 8)
        for j in range(cap):
            pos = segs[j][2]  # end of segment j = boundary
            positions.append(pos)
            tb.line(f"boundary {j}: segment {j} ends at {pos}")
        if len(segs) - 1 > 8:
            tb.line(f"(capped at 8 of {len(segs) - 1} boundaries)")
        verdict = "[" + ", ".join(str(p) for p in positions) + "]"

    elif sub_type == 'largest_level_shift':
        segment_means = params['segment_means']

        # Data: show trend segments + per-segment means
        tb.metric(metric, attributes, ['trend_list'])
        means_str = ", ".join(format_float(v, 2) for v in segment_means)
        tb.data(f"segment_means=[{means_str}]")

        # Reasoning: compute |mean_after - mean_before| for each change point
        n_boundaries = len(segs) - 1
        diffs: List[Tuple[int, float, int]] = []  # (boundary_idx, abs_diff, position)
        for j in range(n_boundaries):
            mean_before = segment_means[j]
            mean_after = segment_means[j + 1]
            signed_diff = round(mean_after - mean_before, 2)
            diff = round(abs(signed_diff), 2)
            pos = segs[j][2]
            tb.line(f"boundary {j} at {pos}:")
            tb.line(format_add_sub_calc(mean_after, mean_before, signed_diff,
                                        op='-', label='shift'))
            tb.line(f"|shift| = {format_float(diff, 2)}")
            diffs.append((j, diff, pos))

        # Iterative scan for maximum
        if len(diffs) == 1:
            best_pos = diffs[0][2]
        else:
            best_idx = 0
            best_diff = diffs[0][1]
            tb.line(f"max scan: start with boundary 0 (shift={format_float(best_diff, 2)})")
            for k in range(1, len(diffs)):
                cmp_text, is_greater = format_cmp(diffs[k][1], best_diff, '>', dp_val=2)
                if is_greater:
                    tb.line(f"boundary {k}: {cmp_text} -> new max")
                    best_idx = k
                    best_diff = diffs[k][1]
                else:
                    tb.line(f"boundary {k}: {cmp_text} -> keep current")
            best_pos = diffs[best_idx][2]
        verdict = str(best_pos)

    else:
        raise ValueError(f"Unknown change_point sub_type: {sub_type!r}")

    tb.verdict(verdict)
    return tb.build()
