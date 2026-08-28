from typing import Any, Dict, List, Optional, Tuple, Union
from synth.ts_generator.utils.common_utils import format_float


# ---------------------------------------------------------------------------
# Atomic vocabulary helpers — use these EVERYWHERE instead of inline strings.
# New builders MUST compose from these atoms to guarantee consistency.
# ---------------------------------------------------------------------------

def met_str(is_met: bool) -> str:
    """Canonical condition-level outcome: 'met' / 'not met'."""
    return "met" if is_met else "not met"


def pass_str(passed: bool) -> str:
    """Canonical item-level check outcome: 'pass' / 'fail'."""
    return "pass" if passed else "fail"


def format_seg_display(seg_type: str, start: int, end: int) -> str:
    """Canonical trend segment reference: (type, start, end) — 0-indexed."""
    return f"({seg_type}, {start}, {end})"


# ---------------------------------------------------------------------------
# Molecular helpers — multi-line reusable patterns composed from atoms.
# New builders SHOULD use these instead of inlining range/ratio/shift/cmp math.
# ---------------------------------------------------------------------------

_CMP_COMPLEMENTS = {'>': '<=', '>=': '<', '<': '>=', '<=': '>'}
_CMP_OPS = {
    '>':  lambda a, b: a > b,
    '>=': lambda a, b: a >= b,
    '<':  lambda a, b: a < b,
    '<=': lambda a, b: a <= b,
}


def format_window_calc(start: int, end: int, window_size: int, n_windows: int) -> str:
    """Format window count derivation: span from inclusive endpoints + ceil division."""
    span = end - start + 1
    raw_div = span / window_size
    return (f"window size={window_size}, "
            f"span = {end} - {start} + 1 = {span}, "
            f"number of windows = ⌈{span} / {window_size}⌉ = ⌈{format_float(raw_div, 2)}⌉ = {n_windows}")


def format_cmp(val: float, threshold: float, op: str,
               dp_val: int = 2, dp_thresh: int = None) -> Tuple[str, bool]:
    """Format a comparison and compute result.

    op: '>', '>=', '<', '<='
    Returns (text, is_met) where text = ``"val_str OP thresh_str"``
    and the displayed OP reflects the actual outcome
    (true → op, false → complement).

    Both values are rounded to display precision BEFORE comparing,
    ensuring the comparison result always matches the displayed text.
    Without this, ``0.034 > 0.032`` displays as ``0.03 > 0.03 = true``
    which is a visual contradiction.
    """
    if dp_thresh is None:
        dp_thresh = dp_val
    val_r = round(val, dp_val)
    thresh_r = round(threshold, dp_thresh)
    is_met = _CMP_OPS[op](val_r, thresh_r)
    shown_op = op if is_met else _CMP_COMPLEMENTS[op]
    return f"{format_float(val_r, dp_val)} {shown_op} {format_float(thresh_r, dp_thresh)}", is_met


def format_abs_diff_cmp(a: int, b: int, threshold: int) -> Tuple[str, bool]:
    """Format ``|a - b| = diff OP threshold`` and compute result.

    Returns (text, is_within) where is_within = (diff <= threshold).
    """
    diff = abs(a - b)
    is_within = diff <= threshold
    op = "<=" if is_within else ">"
    return f"|{a}-{b}|={diff} {op} {threshold}", is_within


def format_str_eq(actual: str, expected: str) -> Tuple[str, bool]:
    """Format categorical equality: ``actual == expected`` or ``actual ≠ expected``.

    Returns (text, matched).
    """
    matched = actual == expected
    op = "==" if matched else "≠"
    return f"{actual} {op} {expected}", matched


def format_add_sub_calc(a, b, result=None,
                        op: str = '+', dp: int = 2,
                        label: str = None, int_mode: bool = False) -> str:
    """Single-source primitive for addition/subtraction display.

    Format ``[label = ] a ± b = result``.
    Wraps *b* in parens when it is negative (avoids ambiguous ``+ -2`` or ``- -2`` rendering).
    Pass ``int_mode=True`` for integer operands (no decimal formatting).
    *result* is auto-computed when omitted.

    Examples::

        format_add_sub_calc(10.5, 2.3, 8.2, op='-', label='range')
        # → "range = 10.50 - 2.30 = 8.20"

        format_add_sub_calc(100, 30, op='-', label='duration', int_mode=True)
        # → "duration = 100 - 30 = 70"
    """
    if result is None:
        result = (a + b) if op == '+' else (a - b)
    if int_mode:
        a_str, b_str, r_str = str(int(a)), str(int(b)), str(int(result))
    else:
        a_str = format_float(a, dp)
        b_str = format_float(b, dp)
        r_str = format_float(result, dp)
    if b < 0:
        b_str = f"({b_str})"
    prefix = f"{label} = " if label else ""
    return f"{prefix}{a_str} {op} {b_str} = {r_str}"


def format_div_calc(a: float, b: float, result: float,
                    dp: int = 2, label: str = 'ratio') -> str:
    """Single-source primitive for division display.

    Format ``[label = ] a / b = result``.
    """
    prefix = f"{label} = " if label else ""
    return f"{prefix}{format_float(a, dp)} / {format_float(b, dp)} = {format_float(result, dp)}"


def format_mul_calc(k, value: float, result: float,
                    dp: int = 2, label: str = 'threshold',
                    is_pct: bool = False) -> str:
    """Single-source primitive for multiplication display.

    Format ``[label = ] K × value = result`` (or ``K% ×`` when *is_pct*).
    """
    k_str = f"{k}%" if is_pct else str(k)
    prefix = f"{label} = " if label else ""
    return f"{prefix}{k_str} × {format_float(value, dp)} = {format_float(result, dp)}"


def format_tie(parts: List[str]) -> str:
    """Format tied items: ``item1 == item2 -> tied``."""
    return f"{' == '.join(parts)} -> tied"


def _anti_trend_type(t: str) -> str:
    """Returns the opposite trend type: increase <-> decrease.

    Steady has no meaningful opposite — return a sentinel that never matches
    any real trend type, so two steady segments are NOT counted as anticorrelated.
    """
    if t == 'increase':
        return 'decrease'
    if t == 'decrease':
        return 'increase'
    return '__no_anti__'


def get_threshold_note(threshold_provided: bool, threshold: Optional[int]) -> str:
    if threshold_provided:
        return f"threshold={threshold}."
    else:
        return f"threshold not provided, using default threshold={threshold}."


def _format_trend_segs(trend_list: list) -> str:
    """Format trend_segments list for the Data section.
    Returns the trend_segments=[...] line (no trailing newline), or '' if nothing to show."""
    segs = [s for s in trend_list if isinstance(s, (list, tuple)) and len(s) >= 3]
    if not segs:
        return ""
    segs_str = ', '.join(f"({t[0]}, {t[1]}, {t[2]})" for t in segs)
    return f"trend_segments=[{segs_str}]"


def _assemble_thought(lines: List[str]) -> str:
    """Wrap lines in <think>...</think> block with trailing newlines."""
    return f"<think>\n{chr(10).join(lines)}\n</think>\n\n"


def _render_metric_attrs(name: str, attr: Dict, includes: List[str]) -> List[str]:
    """
    Render the display lines for a single metric's attributes.
    Returns a list of lines starting with 'metric: {name}'.
    """
    if not attr:
        attr = {}
    result = [f"metric: {name}"]

    for key in includes:
        if key == 'periodicity':
            sea = attr.get('seasonal', {})
            if isinstance(sea, dict) and sea:
                result.append(f"season={sea['type']}, period={format_float(sea['period'])}, amp={format_float(sea['amplitude'])}")
            else:
                result.append(f"season={str(sea)}")

        elif key == 'trend' or key == 'trend_list':
            if key == 'trend':
                tr = attr.get('trend', {})
                if isinstance(tr, dict) and tr:
                    result.append(f"start={format_float(tr['start'])}, end={format_float(tr['end'])}, amp={format_float(tr['amplitude'])}, overall trend={tr['type']}")
            tl = attr.get('trend_list', [])
            ts_line = _format_trend_segs(tl)
            if ts_line:
                result.append(ts_line)

        elif key == 'local':
            locals_list = attr.get('local', [])
            changes = []
            for l in locals_list:
                if not isinstance(l, dict):
                    continue

                parts = []
                if 'type' in l:
                    parts.append(f"{l['type']}")

                # Canonical local event order: type, start@, end@, amplitude, params.
                meta_parts = []

                # Key Points — resolved first (with values), then intent (indices only)
                kp_list = l.get('key_points_resolved') or l.get('key_points_intent')
                if kp_list:
                    for kp in kp_list:
                        # Token optimization: remove "_context" (e.g. "start_context" -> "start")
                        label = kp['label'].replace('_context', '')
                        idx = kp.get('index')
                        val = kp.get('value')
                        display_idx = idx
                        if val is not None:
                            meta_parts.append(f"{label}@({display_idx}, {format_float(val, 2)})")
                        else:
                            meta_parts.append(f"{label}@{display_idx}")
                else:
                    # Sanitized events (no key_points): must have value_start/value_end.
                    if 'position_start' in l:
                        if 'value_start' not in l:
                            raise KeyError(f"value_start missing from sanitized local event: {l}")
                        meta_parts.append(f"start@({l['position_start']}, {format_float(l['value_start'], 2)})")
                    if 'position_end' in l:
                        if 'value_end' not in l:
                            raise KeyError(f"value_end missing from sanitized local event: {l}")
                        meta_parts.append(f"end@({l['position_end']}, {format_float(l['value_end'], 2)})")

                if 'amplitude' in l:
                    meta_parts.append(f"amplitude={format_float(l['amplitude'])}")

                # Params (direction, rise_length, etc.)
                if 'params' in l and l['params']:
                    for p_k, p_v in l['params'].items():
                        if isinstance(p_v, list):
                            val_str = str(p_v).replace(' ', '')
                        else:
                            val_str = str(p_v)
                        meta_parts.append(f"{p_k}={val_str}")

                parts.extend(meta_parts)

                changes.append(", ".join(parts))

            result.append(f"local=[{'; '.join(changes)}]")

        elif key == 'noise':
            ns = attr.get('noise', {})
            if isinstance(ns, dict) and ns:
                strength = ns['strength']
                result.append(f"noise={ns['type']}, strength={format_float(strength, 2)}")
            else:
                result.append(f"noise={str(ns)}")

        elif key == 'statistic':
            stats = attr.get('statistics', {}) or {}
            if stats:
                result.append(
                    f"stats: min={format_float(stats['min'], 2)}, "
                    f"max={format_float(stats['max'], 2)}, "
                    f"mean={format_float(stats['mean'], 2)}, "
                    f"std={format_float(stats['std'], 2)}"
                )
                # Extended statistics
                if stats.get('range') is not None:
                    result.append(
                        f"stats: range={format_float(stats['range'], 2)}, "
                        f"mean_crossings={stats['mean_crossings']}, "
                        f"peak_pos={stats['max_pos']}, "
                        f"trough_pos={stats['min_pos']}, "
                        f"above_mean_count={stats['above_mean_count']}, "
                        f"above_mean_std_count={stats['above_mean_std_count']}"
                    )
                mid = stats.get('split_mid')
                if mid:
                    result.append(
                        f"stats: first_mean={format_float(stats['first_half_mean'], 2)}, "
                        f"second_mean={format_float(stats['second_half_mean'], 2)}, "
                        f"first_std={format_float(stats['first_half_std'], 2)}, "
                        f"second_std={format_float(stats['second_half_std'], 2)}, "
                        f"split_mid={mid}"
                    )
                win16 = stats.get('segment_means_16', [])
                if win16:
                    vals = ', '.join(format_float(v, 2) for v in win16)
                    result.append(f"stats_win16: [{vals}]")

        elif key == 'length':
            l_val = attr.get('seq_len') or attr.get('length')
            if l_val:
                result.append(f"len={l_val}")

        elif key == 'trend_overall':
            tr = attr.get('trend', {})
            if isinstance(tr, dict) and tr:
                result.append(
                    f"start={format_float(tr['start'])}, "
                    f"end={format_float(tr['end'])}, "
                    f"amp={format_float(tr['amplitude'])}, "
                    f"overall trend={tr['type']}"
                )

        elif key == 'statistic:halves_std':
            stats = attr.get('statistics', {}) or {}
            fhs = format_float(stats['first_half_std'], 2)
            shs = format_float(stats['second_half_std'], 2)
            result.append(f"stats: first_half_std={fhs}, second_half_std={shs}")

    return result


class ThoughtBuilder:
    """Fluent builder that enforces the data → === → reasoning → verdict skeleton."""

    def __init__(self):
        self._data_lines: List[str] = []
        self._reason_lines: List[str] = []
        self._verdict_value: Optional[str] = None
        self._verdict_line: Optional[str] = None

    def metric(self, name: str, attrs: Dict, keys: List[str]) -> 'ThoughtBuilder':
        """Render metric data block via _render_metric_attrs."""
        self._data_lines.extend(_render_metric_attrs(name, attrs or {}, keys))
        return self

    def data(self, line: str) -> 'ThoughtBuilder':
        """Add a custom data line (before ===)."""
        self._data_lines.append(line)
        return self

    def metric_separator(self) -> 'ThoughtBuilder':
        """Add --- separator between metrics."""
        self._data_lines.append("---")
        return self

    def detail(self, text: str) -> 'ThoughtBuilder':
        """Add indented detail line."""
        self._reason_lines.append(f"  {text}")
        return self

    def line(self, text: str) -> 'ThoughtBuilder':
        """Add raw reasoning line."""
        self._reason_lines.append(text)
        return self

    def blank(self) -> 'ThoughtBuilder':
        """Add empty line."""
        self._reason_lines.append("")
        return self

    def condition(self, num: int, text: str) -> 'ThoughtBuilder':
        """Add 'condition N: text' reasoning line. Use for numbered conditions."""
        self._reason_lines.append(f"condition {num}: {text}")
        return self

    def all_conditions(self, all_met: bool) -> 'ThoughtBuilder':
        """Add 'all conditions: all met / not all met'. Use ONLY with 2+ conditions."""
        self._reason_lines.append(f"all conditions: {'all met' if all_met else 'not all met'}")
        return self

    def summary(self, pass_count: int, total: int, indented: bool = False) -> 'ThoughtBuilder':
        """Add standardized 'summary: N of M pass' line."""
        text = f"summary: {pass_count} of {total} pass"
        self._reason_lines.append(f"  {text}" if indented else text)
        return self

    def verdict(self, v: str, prefix: str = "answer") -> 'ThoughtBuilder':
        """Set verdict value and store answer line (emitted by build())."""
        self._verdict_value = v
        self._verdict_line = f"{prefix}: {v}"
        return self

    def build(self) -> Tuple[str, str]:
        """Assemble: data + === + reasoning + answer → (_assemble_thought(), verdict).

        Skips the ``===`` separator when there are no reasoning lines
        (pure data-only blocks and lookup types).  The answer line is
        always appended last, after the ``===`` decision.
        """
        lines = list(self._data_lines)
        if self._reason_lines:
            lines.append("===")
            lines.extend(self._reason_lines)
        if self._verdict_line:
            lines.append(self._verdict_line)
        return _assemble_thought(lines), self._verdict_value


def check_trend_match(
    target_attr: Dict,
    candidate_attr: Dict,
    mode: str = 'trend',
    threshold: int = 0,
) -> bool:
    """Lightweight pre-check: would trend/anti_trend verification PASS?

    Replicates the match logic of ``generate_structural_verification`` for
    trend/anti_trend modes without any string formatting.  Use this to decide
    whether to skip expensive thought construction.
    """
    target_segments = target_attr.get('trend_list', [])
    candidate_segments = candidate_attr.get('trend_list', [])
    if len(target_segments) != len(candidate_segments):
        return False
    for seg_a, seg_b in zip(target_segments, candidate_segments):
        if not (isinstance(seg_a, (list, tuple)) and len(seg_a) >= 3):
            return False
        if not (isinstance(seg_b, (list, tuple)) and len(seg_b) >= 3):
            return False
        type_a, start_a, end_a = seg_a[0], seg_a[1], seg_a[2]
        type_b, start_b, end_b = seg_b[0], seg_b[1], seg_b[2]
        if mode == 'trend':
            if type_a != type_b:
                return False
        else:  # anti_trend
            if type_b != _anti_trend_type(type_a):
                return False
        if abs(start_a - start_b) > threshold:
            return False
        if abs(end_a - end_b) > threshold:
            return False
    return True


def generate_structural_verification(
    target_attr: Dict,
    candidate_attr: Dict,
    mode: str = 'trend',
    threshold: Optional[int] = None,
    anchor_point: Optional[int] = None,
    target_event_type: Optional[str] = None,
) -> str:
    """
    Generates gate-based, early-exit reasoning with explicit math.

    For 'local' mode: checks each candidate event against anchor_point step by step.
        When target_event_type is None: PASS as soon as one event is within threshold.
        When target_event_type is set: both timing AND event type must match to PASS.
        FAIL only after all events exhausted.
    For 'trend' mode: compares segment structure between target and candidate.
    """
    lines = []
    if mode == 'local':
        assert anchor_point is not None, "anchor_point required for local verification"
        candidate_events = candidate_attr.get('local', [])
        if not candidate_events:
            lines.append(f"no events. FAIL.")
            return " ".join(lines)
        checks = []
        for event in candidate_events:
            candidate_pos = event['position_start']
            dist_text, is_near = format_abs_diff_cmp(candidate_pos, anchor_point, threshold)
            if is_near:
                if target_event_type is None:
                    checks.append(dist_text)
                    lines.append(f"{'; '.join(checks)}. PASS.")
                    return " ".join(lines)
                else:
                    cand_type = event['type']
                    if isinstance(cand_type, list):
                        cand_type = cand_type[0] if cand_type else ''
                    type_text, type_ok = format_str_eq(cand_type, target_event_type)
                    checks.append(f"{dist_text} → type: {type_text}")
                    if type_ok:
                        lines.append(f"{'; '.join(checks)}. PASS.")
                        return " ".join(lines)
            else:
                checks.append(dist_text)
        lines.append(f"{'; '.join(checks)}. FAIL.")

    elif mode == 'local_end':
        # anchor_point is position_end (exclusive endpoint); displayed as-is.
        assert anchor_point is not None, "anchor_point required for local_end verification"
        candidate_events = candidate_attr.get('local', [])
        if not candidate_events:
            lines.append(f"no events. FAIL.")
            return " ".join(lines)
        checks = []
        for event in candidate_events:
            candidate_end = event.get('position_end')
            if candidate_end is None:
                continue
            dist_text, is_near = format_abs_diff_cmp(candidate_end, anchor_point, threshold)
            if is_near:
                checks.append(dist_text)
                lines.append(f"{'; '.join(checks)}. PASS.")
                return " ".join(lines)
            else:
                checks.append(dist_text)
        msg = '; '.join(checks) if checks else 'no events with end position'
        lines.append(f"{msg}. FAIL.")

    elif mode in ('trend', 'anti_trend'):
        target_segments = target_attr.get('trend_list', [])
        candidate_segments = candidate_attr.get('trend_list', [])
        if len(target_segments) != len(candidate_segments):
            lines.append(f"segments count: {len(target_segments)} vs {len(candidate_segments)}. FAIL.")
            return " ".join(lines)
        lines.append(f"segments count: {len(target_segments)} vs {len(candidate_segments)}.")
        for i, (seg_a, seg_b) in enumerate(zip(target_segments, candidate_segments)):
            type_a, start_a, end_a = seg_a[0], seg_a[1], seg_a[2]
            type_b, start_b, end_b = seg_b[0], seg_b[1], seg_b[2]
            if mode == 'trend':
                eq_text, matched = format_str_eq(type_a, type_b)
                if not matched:
                    lines.append(f"seg {i}: type {eq_text}. FAIL.")
                    return " ".join(lines)
                lines.append(f"seg {i}: {eq_text}.")
            else:  # anti_trend
                expected = _anti_trend_type(type_a)
                eq_text, matched = format_str_eq(type_b, expected)
                if not matched:
                    lines.append(f"seg {i}: '{type_a}' -> expected opposite '{expected}', got '{type_b}' ({eq_text}). FAIL.")
                    return " ".join(lines)
                lines.append(f"seg {i}: '{type_a}' vs '{type_b}' ({eq_text}, opposite OK).")
            start_text, start_ok = format_abs_diff_cmp(start_a, start_b, threshold)
            if not start_ok:
                lines.append(f"start diff: {start_text}. FAIL.")
                return " ".join(lines)
            end_text, end_ok = format_abs_diff_cmp(end_a, end_b, threshold)
            if not end_ok:
                lines.append(f"end diff: {end_text}. FAIL.")
                return " ".join(lines)
            lines.append(f"start diff: {start_text}.")
            lines.append(f"end diff: {end_text}.")
        lines.append("PASS.")

    return " ".join(lines)

def build_local_verification_thought(
    attributes: Union[Dict, List[Dict]],
    metric_names: Union[str, List[str]],
    include_attributes: List[str],
    anchor_point: int,
    mode: str = 'local',
    threshold: int = 15,
    threshold_provided: bool = False,
    show_verdict: bool = False,
    show_cluster: bool = False,
    target_event_type: Optional[str] = None,
) -> Tuple[str, List[str]]:
    """Local/local_end verification thought block.

    Format (consistent for BOTH pass and fail):
        Metric: <anchor>          <- always anchor data first
        local=[...]
        ===
        threshold=X. anchor=Y.   <- always === right after anchor
        <anchor>: <check>. PASS/FAIL.
        ---                      <- other metrics follow ONLY if anchor passed
        Metric: <other1>
        ...
        vs <other1>: <check>. PASS/FAIL.
        cluster=(...)

    Returns (thought_block, matching_metrics).
    """
    if isinstance(attributes, dict):
        attributes = [attributes]
    if isinstance(metric_names, str):
        metric_names = [metric_names]

    target_name = metric_names[0]
    target_attr = attributes[0] or {}
    metric_attr_pairs = list(zip(metric_names, attributes))

    tb = ThoughtBuilder()
    matching_metrics = []
    pre_check_failed = False

    # Data section: anchor metric only
    tb.metric(target_name, target_attr, include_attributes)

    # Reasoning section: threshold + anchor note
    note = get_threshold_note(threshold_provided, threshold)
    if mode == 'local':
        assert anchor_point is not None, "anchor_point required for local verification"
        note += f" anchor={anchor_point}."
    else:
        assert anchor_point is not None, "anchor_point required for local_end verification"
        note += f" anchor={anchor_point}."
    tb.line(note)

    # Anchor check
    target_events = target_attr.get('local', [])
    if mode == 'local':
        if not target_events:
            tb.line(f"{target_name}: no events. FAIL.")
            pre_check_failed = True
        else:
            checks = []
            found_pass = False
            for event in target_events:
                target_pos = event['position_start']
                dist_text, is_near = format_abs_diff_cmp(target_pos, anchor_point, threshold)
                checks.append(dist_text)
                if is_near:
                    found_pass = True
                    break
            if found_pass:
                type_note = f" target event type: {target_event_type}." if target_event_type is not None else ""
                tb.line(f"{target_name}: {'; '.join(checks)}. PASS.{type_note}")
                matching_metrics.append(target_name)
            else:
                tb.line(f"{target_name}: {'; '.join(checks)}. FAIL.")
                pre_check_failed = True
    else:  # local_end
        if not target_events:
            tb.line(f"{target_name}: no events. FAIL.")
            pre_check_failed = True
        else:
            checks = []
            found_pass = False
            for event in target_events:
                target_end = event.get('position_end')
                if target_end is None:
                    continue
                dist_text, is_near = format_abs_diff_cmp(target_end, anchor_point, threshold)
                checks.append(dist_text)
                if is_near:
                    found_pass = True
                    break
            if found_pass:
                tb.line(f"{target_name}: {'; '.join(checks)}. PASS.")
                matching_metrics.append(target_name)
            else:
                msg = '; '.join(checks) if checks else 'no events with end position'
                tb.line(f"{target_name}: {msg}. FAIL.")
                pre_check_failed = True

    # Other metrics data + pairwise (only if anchor passed)
    if not pre_check_failed and len(metric_attr_pairs) > 1:
        for name, attr in metric_attr_pairs[1:]:
            tb.metric_separator()
            for line in _render_metric_attrs(name, attr or {}, include_attributes):
                tb.data(line)

        tb.line("---")
        for other_name, other_attr in metric_attr_pairs[1:]:
            verify_text = generate_structural_verification(
                target_attr, other_attr or {},
                mode=mode, threshold=threshold, anchor_point=anchor_point,
                target_event_type=target_event_type,
            )
            tb.line(f"vs {other_name}: {verify_text}")
            if "PASS" in verify_text and "FAIL" not in verify_text:
                matching_metrics.append(other_name)

    # Verdict and cluster
    verdict = "similar" if len(matching_metrics) > 1 else "different"
    if show_verdict:
        tb.line(f"answer: {verdict}")
    if show_cluster:
        if matching_metrics:
            tb.line(f"answer: ({', '.join(matching_metrics)})")
        else:
            tb.line("answer: ()")

    return tb.build()[0], matching_metrics


def build_trend_verification_thought(
    attributes: Union[Dict, List[Dict]],
    metric_names: Union[str, List[str]],
    include_attributes: List[str],
    mode: str = 'trend',
    threshold: int = 15,
    threshold_provided: bool = False,
    show_verdict: bool = False,
    show_cluster: bool = False,
) -> Tuple[str, List[str], str]:
    """Trend/anti_trend verification thought block.

    Format (all data before ===):
        Metric: <anchor>
        trend_segments=[...]
        ---
        Metric: <other1>
        trend_segments=[...]
        ===
        Comparison with <anchor>
        vs <other1>: <check>. PASS/FAIL.
        cluster=(...)

    Returns (thought_block, matching_metrics, verdict_string).
    verdict_string: 'similar'/'different' for trend, 'opposite'/'not_opposite' for anti_trend.
    """
    if isinstance(attributes, dict):
        attributes = [attributes]
    if isinstance(metric_names, str):
        metric_names = [metric_names]

    target_name = metric_names[0]
    target_attr = attributes[0] or {}
    metric_attr_pairs = list(zip(metric_names, attributes))

    tb = ThoughtBuilder()
    matching_metrics = []

    # Data section: all metrics
    for i, (name, attr) in enumerate(metric_attr_pairs):
        if not attr:
            attr = {}
        if i > 0:
            tb.metric_separator()
        tb.metric(name, attr, include_attributes)

    # Reasoning section
    tb.line(get_threshold_note(threshold_provided, threshold))
    tb.line(f"comparison with {target_name}")
    matching_metrics.append(target_name)

    for other_name, other_attr in metric_attr_pairs[1:]:
        verify_text = generate_structural_verification(
            target_attr, other_attr or {},
            mode=mode, threshold=threshold,
        )
        tb.line(f"vs {other_name}: {verify_text}")
        if "PASS" in verify_text and "FAIL" not in verify_text:
            matching_metrics.append(other_name)

    if mode == 'anti_trend':
        verdict = "opposite" if len(matching_metrics) > 1 else "not_opposite"
    else:
        verdict = "similar" if len(matching_metrics) > 1 else "different"

    if show_verdict:
        tb.line(f"answer: {verdict}")
    if show_cluster:
        if matching_metrics:
            tb.line(f"answer: ({', '.join(matching_metrics)})")
        else:
            tb.line("answer: ()")

    return tb.build()[0], matching_metrics, verdict
