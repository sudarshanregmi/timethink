import math
import re
from typing import Dict, List, Any, Set, Optional

from loguru import logger


def parse_list_verdict(v: str) -> Optional[List[float]]:
    """Parse '[1.2, 3.4, ...]' into list of floats. Returns None if any value is NaN/Inf."""
    m = re.search(r'\[([^\]]*)\]', v)
    if not m:
        return None
    try:
        vals = [float(x.strip()) for x in m.group(1).split(',') if x.strip()]
        if any(not math.isfinite(f) for f in vals):
            return None
        return vals
    except ValueError:
        return None


def _safe_float(s: str) -> Optional[float]:
    """Convert string to float, returning None for malformed values (trailing dots, double dots, NaN, inf)."""
    try:
        f = float(s)
        if not math.isfinite(f):
            return None
        return f
    except (ValueError, TypeError):
        return None


def extract_think_content(text: str) -> Optional[str]:
    m = re.search(r'<think>(.*?)</think>', text, re.DOTALL)
    return m.group(1).strip() if m else None


# ---------------------------------------------------------------------------
# Molecular helpers: reasoning section extraction
# ---------------------------------------------------------------------------

def get_reasoning(think: str) -> str:
    """Return the reasoning section (below ===) of a think block.
    For data-only types (no ===), returns the full think content."""
    parts = think.split('===', 1)
    return parts[1] if len(parts) > 1 else think


def get_data_block(think: str) -> str:
    """Return the data section (above ===) of a think block."""
    return think.split('===', 1)[0]


def extract_answer_line(think: str) -> Optional[str]:
    """Extract raw answer text from the LAST 'answer:' or 'verdict:' line
    anywhere in the think block. No === dependence — answer can appear anywhere,
    last occurrence wins."""
    matches = re.findall(r'^(?:answer|verdict):\s*(.+)', think, re.MULTILINE | re.IGNORECASE)
    return matches[-1].strip() if matches else None


# ---------------------------------------------------------------------------
# Specialized verdict extractors (thin wrappers over extract_answer_line)
# ---------------------------------------------------------------------------

def extract_answer_scalar(think: str) -> Optional[str]:
    """Extract scalar answer from 'answer: value' line below === (or full block for data-only types)."""
    section = get_reasoning(think)
    m = re.search(r'^answer:\s*(\S+)', section, re.MULTILINE | re.IGNORECASE)  # single-word: callers use one word verdicts only
    return m.group(1).lower() if m else None


def extract_answer_set(think: str) -> Optional[Set[str]]:
    """Extract set from 'answer: (A, B, ...)' line below === (or full block for data-only types)."""
    section = get_reasoning(think)
    m = re.search(r'^answer:\s*\(([^)]*)\)', section, re.MULTILINE | re.IGNORECASE)
    if m is None:
        return None
    inner = m.group(1).strip()
    return {x.strip().lower() for x in inner.split(',') if x.strip()} if inner else set()



def extract_yes_no_result(think: str) -> Optional[bool]:
    """
    Scans the === verification section for the target metric's PASS/FAIL.
    Skips 'vs …' lines (those are other-metric comparisons).
    Returns True (PASS), False (FAIL), None (not determinable).
    """
    if '===' not in think:
        return None
    reasoning = get_reasoning(think)
    for line in reasoning.split('\n'):
        line = line.strip()
        if not line:
            continue
        upper = line.upper()
        if upper.startswith('VS ') or upper.startswith('THRESHOLD') or upper.startswith('COMPARISON'):
            continue
        if 'PASS.' in upper and 'FAIL.' not in upper:
            return True
        if 'FAIL.' in upper:
            return False
    return None


def extract_verification_per_metric(think: str) -> Dict[str, bool]:
    """
    Extracts per-metric PASS/FAIL from 'vs {metric}: ...' lines in the
    verification section (after ===).  Returns {metric_name: True/False}.
    Returns {} when there are no vs-lines (description QA, pre-check failed, etc.)
    """
    if '===' not in think:
        return {}
    reasoning = get_reasoning(think)
    result: Dict[str, bool] = {}
    for line in reasoning.split('\n'):
        line = line.strip()
        if not line.startswith('vs '):
            continue
        m = re.match(r'^vs (.+?):', line)
        if not m:
            continue
        metric = m.group(1).strip()
        upper = line.upper()
        if 'PASS.' in upper and 'FAIL.' not in upper:
            result[metric] = True
        elif 'FAIL.' in upper:
            result[metric] = False
    return result


def _extract_outcome(block: str) -> Optional[str]:
    """Extract condition outcome from a text block (expects lowercased input).

    Returns 'not met', 'met', 'pass', or 'fail'.
    Checks negated forms first so 'not met' isn't misread as 'met'.
    """
    if re.search(r'\bnot\s+met\b', block):
        return 'not met'
    m = re.search(r'\b(met|pass|fail)\b', block)
    return m.group(1) if m else None


def _extract_metadata_content(ev_str: str) -> Optional[str]:
    """
    Extract the content inside metadata=[...] from an event string,
    handling nested brackets (e.g. sub_amplitudes=[0.5, 0.3]).
    Returns None if no metadata=[...] is present.
    """
    m = re.search(r'metadata=\[', ev_str)
    if not m:
        return None
    start = m.end()
    depth = 1
    i = start
    while i < len(ev_str) and depth > 0:
        if ev_str[i] == '[':
            depth += 1
        elif ev_str[i] == ']':
            depth -= 1
        i += 1
    return ev_str[start:i - 1]


def _extract_key_points(ev_str: str) -> List[Dict]:
    """
    Parses label@(idx, val) or label@idx key-point patterns from:
    - New format: inside metadata=[...] block
    - Old format: inside points=[...] block
    """
    meta = _extract_metadata_content(ev_str)
    if meta is not None:
        search_str = meta
    else:
        m = re.search(r'points=\[([^\]]*)\]', ev_str)
        search_str = m.group(1) if m else ev_str  # fall back to full inline string

    kps = []
    for hit in re.finditer(
        r'(\w+)@\((\d+),\s*([\d.eE+-]+)\)|(\w+)@(\d+)',
        search_str
    ):
        if hit.group(1):   # label@(idx, val)
            val = _safe_float(hit.group(3))
            if val is not None:
                kps.append({'label': hit.group(1), 'index': int(hit.group(2)), 'value': val})
            else:
                kps.append({'label': hit.group(1), 'index': int(hit.group(2))})
        else:              # label@idx
            kps.append({'label': hit.group(4), 'index': int(hit.group(5))})
    return kps


def _extract_params(ev_str: str) -> Dict[str, Any]:
    """
    Parses structural param key=val entries from:
    - New format: key=val pairs inside metadata=[...] (excluding amplitude= and label@ patterns)
    - Old format: key=val pairs inside params={...}
    Values are typed: list-of-floats, float, or string.
    """
    meta = _extract_metadata_content(ev_str)
    if meta is not None:
        # Strip amplitude= and label@(...)/label@N patterns before extracting params.
        search_str = re.sub(r'\w+@\([^)]+\)', '', meta)
        search_str = re.sub(r'\w+@\d+', '', search_str)
        search_str = re.sub(r'\bamplitude=[\d.eE+-]+', '', search_str)
    else:
        m = re.search(r'params=\{([^}]*)\}', ev_str)
        if not m:
            return {}
        search_str = m.group(1)

    params: Dict[str, Any] = {}
    for kv in re.finditer(r'(\w+)=([\w.eE+-]+|\[[^\]]*\])', search_str):
        key = kv.group(1)
        val_str = kv.group(2).strip()
        if not val_str:
            continue
        if val_str.startswith('['):
            inner = val_str[1:-1]
            parsed = [_safe_float(x.strip()) for x in inner.split(',') if x.strip()]
            if all(v is not None for v in parsed) and parsed:
                params[key] = parsed
            else:
                params[key] = [x.strip() for x in inner.split(',') if x.strip()]
        else:
            sf = _safe_float(val_str)
            if sf is not None:
                params[key] = sf
            else:
                params[key] = val_str
    return params


def _extract_local_events(line: str) -> List[Dict]:
    """
    Extracts local events from a 'local=[...]' line.
    Handles nested brackets inside metadata=[...] (new format) and
    points=[...] / params={...} (old format).
    Captures: type, amplitude, key_points, params.
    Derives position_start/position_end from start@/end@ key_points.
    """
    m = re.search(r'local=\[', line)
    if not m:
        return []

    start = m.end()
    depth = 1
    i = start
    while i < len(line) and depth > 0:
        if line[i] in '[{':
            depth += 1
        elif line[i] in ']}':
            depth -= 1
        i += 1

    inner = line[start:i - 1].strip()
    if not inner:
        return []

    events = []
    for ev_str in inner.split(';'):
        ev_str = ev_str.strip()
        if not ev_str:
            continue
        ev: Dict[str, Any] = {}

        # Type: leading text before first key=val or label@( (absent in sparse format)
        type_m = re.match(r'^([^,=@]+?)(?=,\s*\w+[@=]|\s*$)', ev_str)
        if type_m:
            candidate = type_m.group(1).strip()
            if '=' not in candidate:
                ev['type'] = candidate

        # Amplitude: 'amp=X' or 'amplitude=X' inside metadata=[...]
        amp_m = re.search(r'\bamp=([\d.eE+-]+)', ev_str)
        if amp_m:
            amp_val = _safe_float(amp_m.group(1))
            if amp_val is not None:
                ev['amplitude'] = amp_val
        else:
            amp2_m = re.search(r'\bamplitude=([\d.eE+-]+)', ev_str)
            if amp2_m:
                amp_val = _safe_float(amp2_m.group(1))
                if amp_val is not None:
                    ev['amplitude'] = amp_val

        kps = _extract_key_points(ev_str)
        if kps:
            ev['key_points'] = kps

        # Derive position_start / position_end from key_points (start@/end@ format)
        for kp in kps:
            if kp['label'] == 'start':
                ev['position_start'] = float(kp['index'])
                break
        for kp in reversed(kps):
            if kp['label'] == 'end':
                ev['position_end'] = float(kp['index'])
                break

        params = _extract_params(ev_str)
        if params:
            ev['params'] = params

        if ev:
            events.append(ev)
    return events


def _extract_trend_segments(line: str) -> List[Dict]:
    """Extracts segments from 'trend_segments=[(type, start, end), ...]'."""
    segs = re.findall(r'\(([^,)]+),\s*(\d+),\s*(\d+)\)', line)
    return [{'type': s[0].strip(), 'start': int(s[1]), 'end': int(s[2])} for s in segs]


def extract_metric_blocks(think: str) -> Dict[str, Dict]:
    """
    Parses per-metric attribute sections from a think block.

    Works with both formats:
    - Old (trend/anti_trend): all metrics before ===
    - New (local/local_end): anchor before ===, other metrics after === but before
      pairwise vs-lines (consistent structure regardless of anchor pass/fail)

    Splits on --- across the full think content; unrecognized lines (===, threshold=,
    vs ..., cluster=, etc.) are silently ignored by the per-line parser.

    Returns: {metric_name: {seasonal, trend, trend_segments, local, noise, length}}
    """
    raw_blocks = re.split(r'\n---\n', think)

    result: Dict[str, Dict] = {}
    for block in raw_blocks:
        metric_name: Optional[str] = None
        attrs: Dict[str, Any] = {
            'seasonal': None,
            'trend': None,
            'trend_segments': None,
            'local': None,
            'noise': None,
            'length': None,
            'stats': None,
        }
        for line in block.strip().split('\n'):
            line = line.strip()
            if not line:
                continue

            if line.startswith('metric:'):
                metric_name = line.split(':', 1)[1].strip()

            elif line.startswith('season='):
                m = re.match(
                    r'season=([^,]+),\s*period=([\d.eE+-]+),\s*amp=([\d.eE+-]+)', line
                )
                if m:
                    period = _safe_float(m.group(2))
                    amplitude = _safe_float(m.group(3))
                    if period is not None and amplitude is not None:
                        attrs['seasonal'] = {
                            'type': m.group(1).strip(),
                            'period': period,
                            'amplitude': amplitude,
                        }

            elif 'overall trend=' in line and (line.startswith('overall trend=') or line.startswith('start=')):
                t_m   = re.search(r'overall trend=([^,\n]+)', line)
                st_m  = re.search(r'\bstart=([\d.eE+-]+)', line)
                en_m  = re.search(r'\bend=([\d.eE+-]+)', line)
                amp_m = re.search(r'\bamp=([\d.eE+-]+)', line)
                if t_m and st_m and en_m and amp_m:
                    start_v = _safe_float(st_m.group(1))
                    end_v = _safe_float(en_m.group(1))
                    amp_v = _safe_float(amp_m.group(1))
                    if start_v is not None and end_v is not None and amp_v is not None:
                        attrs['trend'] = {
                            'type': t_m.group(1).strip(),
                            'start': start_v,
                            'end': end_v,
                            'amplitude': amp_v,
                        }

            elif line.startswith('trend_segments='):
                attrs['trend_segments'] = _extract_trend_segments(line)

            elif line.startswith('local='):
                attrs['local'] = _extract_local_events(line)

            elif line.startswith('noise='):
                m = re.match(r'noise=([^,]+),\s*strength=([\d.eE+-]+)', line)
                if m:
                    strength = _safe_float(m.group(2))
                    if strength is not None:
                        attrs['noise'] = {
                            'type': m.group(1).strip(),
                            'strength': strength,
                        }

            elif line.startswith('stats:'):
                stat_pairs = re.findall(r'(\w+)=([\d.eE+-]+)', line)
                if stat_pairs:
                    if attrs['stats'] is None:
                        attrs['stats'] = {}
                    for k, v in stat_pairs:
                        parsed = _safe_float(v)
                        if parsed is not None:
                            attrs['stats'][k] = parsed

            elif line.startswith('len='):
                try:
                    attrs['length'] = int(line.split('=', 1)[1].strip())
                except (ValueError, IndexError) as e:
                    logger.warning(f"Failed to parse len= line: {line!r}: {e}")

        if metric_name:
            result[metric_name] = attrs
    return result


def extract_aggregate_durations(reasoning: str) -> Dict[str, float]:
    """Extract {trend_type: total_duration} from an 'aggregate durations:' section.

    Handles both simple and summation formats:
      - increase: 144
      - increase: 69 + 15 = 84
    Returns empty dict if the section is missing or unparseable.
    """
    reasoning_lower = reasoning.lower()
    agg_match = re.search(r'aggregate durations:', reasoning_lower)
    if not agg_match:
        return {}

    agg_section = reasoning_lower[agg_match.end():]
    end_match = re.search(r'dominant trend|answer:', agg_section)
    if end_match:
        agg_section = agg_section[:end_match.start()]

    durations: Dict[str, float] = {}
    for line in agg_section.split('\n'):
        line = line.strip()
        m = re.match(r'[-•*]\s*(increase|decrease|keep steady|steady)\s*:', line)
        if not m:
            continue
        trend_type = m.group(1)
        rest = line[m.end():]
        eq_pos = rest.rfind('=')
        if eq_pos != -1:
            val = _safe_float(rest[eq_pos + 1:].strip())
        else:
            num_match = re.search(r'([\d.]+)', rest)
            val = _safe_float(num_match.group(1)) if num_match else None
        if val is not None:
            durations[trend_type] = val
    return durations


def extract_dominant_winner(reasoning: str) -> Optional[str]:
    """Extract the winning trend type from 'dominant trend comparison:' line.

    Returns the first (dominant) type, or None if not found.
    Handles: 'dominant trend comparison: increase (84) > decrease (60)'
    """
    m = re.search(
        r'dominant trend comparison:\s*(increase|decrease|keep steady|steady)',
        reasoning, re.IGNORECASE,
    )
    return m.group(1).lower() if m else None
