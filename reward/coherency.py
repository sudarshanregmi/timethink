"""
Coherency scoring — internal reasoning consistency checks.

Each function operates PURELY on pred_think (never looks at GT).
Returns float in [0.0, 1.0]. Returns 0.5 when patterns can't be parsed
(neutral — neither rewards nor penalizes unfamiliar formats).
Thread-safe (stateless).
"""

import re
from typing import Dict, Optional, Set

from reward.parsing import _safe_float, _extract_outcome, get_reasoning


# ---------------------------------------------------------------------------
# 1. Condition → Summary → Verdict chain
#    Used by: compound_judgment, anti_judgment
# ---------------------------------------------------------------------------

def coherency_condition_verdict(pred_think: str) -> float:
    """Check logical consistency of condition outcomes → all conditions → answer.

    Handles:
      Format A: 'condition N:' blocks with met/not met → 'all conditions:' → 'answer:'
      Format B: 'event N:' blocks with pass/fail → 'summary:' count (count only, no verdict mapping)
      Format C: direct '-> met/not met' or '-> pass/fail' → 'answer:'
    """
    reasoning_lower = get_reasoning(pred_think).lower()

    # Extract answer (yes/no)
    answer_match = re.search(r'(?:answer|verdict):\s*(yes|no)', reasoning_lower)
    answer = answer_match.group(1) if answer_match else None

    # --- Format A: condition N: ---
    n_conds = len(re.findall(r'\bcondition \d+:', reasoning_lower))
    if n_conds > 0:
        # Parse individual condition outcomes (only from "pure" conditions
        # that don't contain embedded events — hybrid blocks with event N:
        # or summary: have unreliable outcomes from _extract_outcome).
        pure_outcomes = []
        has_hybrid = False
        for i in range(n_conds):
            block = _get_condition_block(reasoning_lower, i + 1, n_conds)
            if block is None:
                continue
            if re.search(r'\bevent \d+:', block) or 'summary:' in block:
                has_hybrid = True
                continue  # skip hybrid blocks
            outcome = _extract_outcome(block)
            if outcome is not None:
                pure_outcomes.append(outcome)

        checks = []

        # Parse all conditions summary once (reused by sub-checks 1 and 2)
        all_cond_match = re.search(
            r'all conditions:\s*(.*?)(?:\n|$)', reasoning_lower
        )
        summary_says_all = None
        if all_cond_match:
            summary_text = all_cond_match.group(1)
            has_not_all = bool(re.search(r'\bnot all met\b', summary_text))
            has_all_met = bool(re.search(r'\ball met\b', summary_text))
            if has_all_met or has_not_all:
                summary_says_all = has_all_met and not has_not_all

        # Sub-check 1: pure conditions → all conditions (only when no hybrid blocks)
        if pure_outcomes and not has_hybrid and n_conds >= 2 and summary_says_all is not None:
            all_met_from_conds = all(o in ('met', 'pass') for o in pure_outcomes)
            checks.append(1.0 if summary_says_all == all_met_from_conds else 0.0)

        # Sub-check 2: all conditions → answer (always reliable)
        if answer is not None and n_conds >= 2 and summary_says_all is not None:
            expected = 'yes' if summary_says_all else 'no'
            checks.append(1.0 if answer == expected else 0.0)

        # Sub-check 3: single condition → answer (no summary needed)
        if answer is not None and n_conds == 1 and pure_outcomes:
            all_met = all(o in ('met', 'pass') for o in pure_outcomes)
            expected = 'yes' if all_met else 'no'
            checks.append(1.0 if answer == expected else 0.0)

        # Sub-check 4: single hybrid condition → use embedded summary to derive outcome
        if answer is not None and n_conds == 1 and has_hybrid and not pure_outcomes:
            block = _get_condition_block(reasoning_lower, 1, n_conds)
            if block:
                summary_match = re.search(r'summary:\s*(\d+)\s+of\s+(\d+)\s+pass', block)
                if summary_match:
                    pass_count = int(summary_match.group(1))
                    condition_met = pass_count > 0
                    expected = 'yes' if condition_met else 'no'
                    checks.append(1.0 if answer == expected else 0.0)

        return sum(checks) / len(checks) if checks else 0.5

    # --- Format B: event N: ---
    # stat_threshold uses OR-logic (answer=yes if ANY event passes), not AND.
    # Since we can't determine the logic from pred_think alone, we only
    # check the summary count consistency (no verdict mapping).
    n_events = len(re.findall(r'\bevent \d+:', reasoning_lower))
    if n_events > 0:
        pass_count = 0
        total_parsed = 0
        for i in range(n_events):
            label = f'event {i + 1}:'
            start = reasoning_lower.find(label)
            if start == -1:
                continue
            end = len(reasoning_lower)
            for boundary in [f'event {i + 2}:', 'summary:', 'answer:']:
                pos = reasoning_lower.find(boundary, start + len(label))
                if pos != -1 and pos < end:
                    end = pos
            block = reasoning_lower[start:end]
            outcome = _extract_outcome(block)
            if outcome is not None:
                total_parsed += 1
                if outcome in ('pass', 'met'):
                    pass_count += 1

        if total_parsed == 0:
            return 0.5

        # Only check: summary count matches actual pass count
        summary_match = re.search(r'summary:\s*(\d+)\s+of\s+(\d+)\s+pass', reasoning_lower)
        if summary_match:
            claimed_pass = int(summary_match.group(1))
            return 1.0 if claimed_pass == pass_count else 0.0

        return 0.5  # no summary to check

    # --- Format C: direct -> met/not met or -> pass/fail or (pass)/(fail) ---
    # May have multiple outcomes on one line; use AND-logic (any fail → no).
    if answer is not None:
        all_outcomes = _extract_all_outcomes(reasoning_lower)
        if all_outcomes:
            any_fail = any(o in ('not met', 'fail') for o in all_outcomes)
            expected = 'no' if any_fail else 'yes'
            return 1.0 if answer == expected else 0.0

    return 0.5  # unparseable → neutral default


def _get_condition_block(reasoning_lower: str, cond_num: int, n_conds: int) -> Optional[str]:
    """Extract the text for a specific condition N block."""
    label = f'condition {cond_num}:'
    start = reasoning_lower.find(label)
    if start == -1:
        return None
    end = len(reasoning_lower)
    for boundary in [f'condition {cond_num + 1}:', 'all conditions:', 'answer:']:
        pos = reasoning_lower.find(boundary, start + len(label))
        if pos != -1 and pos < end:
            end = pos
    return reasoning_lower[start:end]


def _extract_all_outcomes(text: str) -> list:
    """Extract ALL met/not met/pass/fail outcomes from text (for multi-outcome Format C)."""
    outcomes = []
    # Find all 'not met' first (to avoid splitting 'not met' into 'met')
    for m in re.finditer(r'\bnot\s+met\b', text):
        outcomes.append(('not met', m.start()))
    for m in re.finditer(r'\b(met|pass|fail)\b', text):
        word = m.group(1)
        pos = m.start()
        # Skip if this 'met' is part of an already-matched 'not met'
        if word == 'met' and any(abs(pos - nm_pos) < 8 for _, nm_pos in
                                 [(o, p) for o, p in outcomes if o == 'not met']):
            continue
        outcomes.append((word, pos))
    # Sort by position, return just the outcome strings
    outcomes.sort(key=lambda x: x[1])
    return [o for o, _ in outcomes]


# ---------------------------------------------------------------------------
# 2. Aggregate Durations → Dominant Trend → Verdict chain
#    Used by: trend_dominance (cross_trend_query, segment_trend_dominance)
# ---------------------------------------------------------------------------

def coherency_duration_dominance(pred_think: str) -> float:
    """Check that the trend type with max aggregate duration matches the answer."""
    reasoning_lower = get_reasoning(pred_think).lower()

    # Parse aggregate durations
    agg_match = re.search(r'aggregate durations:', reasoning_lower)
    if not agg_match:
        return 0.5

    # Bound the search region
    agg_section = reasoning_lower[agg_match.end():]
    end_match = re.search(r'dominant trend|answer:', agg_section)
    if end_match:
        agg_section = agg_section[:end_match.start()]

    # Parse per-line durations. Handles both:
    #   - increase: 144            (simple)
    #   - increase: 69 + 15 = 84   (summation — take the value after '=')
    durations = {}
    for line in agg_section.split('\n'):
        line = line.strip()
        m = re.match(r'[-•*]\s*(increase|decrease|keep steady|steady)\s*:', line)
        if not m:
            continue
        trend_type = m.group(1)
        rest = line[m.end():]
        # If there's a '=', take the last number after it (the sum result)
        eq_pos = rest.rfind('=')
        if eq_pos != -1:
            val = _safe_float(rest[eq_pos + 1:].strip())
        else:
            # Simple format: just a single number
            num_match = re.search(r'([\d.]+)', rest)
            val = _safe_float(num_match.group(1)) if num_match else None
        if val is not None:
            durations[trend_type] = val

    if not durations:
        return 0.5

    # Parse answer — must handle multi-word types like "keep steady"
    answer_match = re.search(r'(?:answer|verdict):\s*(.+)', reasoning_lower)
    if not answer_match:
        return 0.5
    answer = re.sub(r'</?\w+>', '', answer_match.group(1)).strip()

    # Max duration type(s) — ties accept any tied type or "equal"
    max_dur = max(durations.values())
    max_types = [t for t, d in durations.items() if d == max_dur]

    if len(max_types) > 1 and answer == 'equal':
        return 1.0  # tie → "equal" is coherent
    return 1.0 if answer in max_types else 0.0


# ---------------------------------------------------------------------------
# 3. Stats Half-Comparison → Categorical Verdict chain
#    Used by: stat_numerical (categorical sub-types only)
# ---------------------------------------------------------------------------

def coherency_stats_categorical(pred_think: str) -> float:
    """Check that first/second half stat values are consistent with categorical answer."""
    parts = pred_think.split('===', 1)
    if len(parts) < 2:
        return 0.5
    data_block = parts[0].lower()
    reasoning = parts[1].lower()

    # Parse answer
    answer_match = re.search(r'(?:answer|verdict):\s*(first half|second half|equal)', reasoning)
    if not answer_match:
        return 0.5
    answer = answer_match.group(1)

    # Try to find first_half and second_half stat values from data block
    first_val = _try_parse_half_stat(data_block, 'first')
    second_val = _try_parse_half_stat(data_block, 'second')

    if first_val is None or second_val is None:
        return 0.5  # can't parse → neutral default

    # Determine expected answer
    max_ref = max(abs(first_val), abs(second_val), 1e-6)
    if abs(first_val - second_val) / max_ref < 0.05:
        return 1.0  # approximately equal — any answer is acceptable
    elif first_val > second_val:
        expected = 'first half'
    else:
        expected = 'second half'

    return 1.0 if answer == expected else 0.0


# ---------------------------------------------------------------------------
# 4. vs-line PASS/FAIL → scalar verdict chain
#    Used by: mts_verdict (correlation, anticorrelation)
# ---------------------------------------------------------------------------

# Answer vocabulary auto-detection for correlation vs anticorrelation
_CORR_PASS = frozenset({'similar'})
_CORR_FAIL = frozenset({'different'})
_ANTI_PASS = frozenset({'opposite'})
_ANTI_FAIL = frozenset({'not_opposite'})


def coherency_mts_verdict(pred_think: str) -> float:
    """Check vs-line PASS/FAIL → scalar answer consistency for correlation/anticorrelation.

    Auto-detects correlation (similar/different) vs anticorrelation (opposite/not_opposite)
    from the answer vocabulary.
    """
    if '===' not in pred_think:
        return 0.5
    reasoning = get_reasoning(pred_think)
    reasoning_lower = reasoning.lower()

    # Parse answer
    answer_match = re.search(r'^answer:\s*(\S+)', reasoning_lower, re.MULTILINE)  # single-word: similar/different/opposite/not_opposite
    if not answer_match:
        return 0.5
    answer = answer_match.group(1).strip()

    # Parse the single vs-line PASS/FAIL (with or without "vs " prefix for early exits)
    # Case-insensitive: model may output "pass"/"Pass"/"PASS"
    vs_match = re.search(r'^vs\s+.+?(PASS|FAIL)\.', reasoning, re.MULTILINE | re.IGNORECASE)
    if not vs_match:
        # Early-exit pattern: "MetricName: no events. FAIL." (no "vs" prefix)
        vs_match = re.search(r'^.+?(PASS|FAIL)\.', reasoning, re.MULTILINE | re.IGNORECASE)
    if not vs_match:
        return 0.5
    vs_passed = vs_match.group(1).upper() == 'PASS'

    # Auto-detect type from answer vocabulary and check consistency
    if answer in _CORR_PASS | _CORR_FAIL:
        expected = _CORR_PASS if vs_passed else _CORR_FAIL
        return 1.0 if answer in expected else 0.0
    elif answer in _ANTI_PASS | _ANTI_FAIL:
        expected = _ANTI_PASS if vs_passed else _ANTI_FAIL
        return 1.0 if answer in expected else 0.0

    return 0.5  # unknown answer vocabulary → neutral default


# ---------------------------------------------------------------------------
# 5. vs-line PASS/FAIL → cluster membership → answer set chain
#    Used by: mts_set (clustering, anticlustering)
# ---------------------------------------------------------------------------

def coherency_mts_cluster(pred_think: str) -> float:
    """Check vs-line PASS/FAIL → answer set membership consistency.

    For both clustering and anticlustering: answer set should contain
    the anchor + PASS metrics, and should NOT contain FAIL metrics.
    """
    if '===' not in pred_think:
        return 0.5
    reasoning = get_reasoning(pred_think)

    # Parse vs-lines: metric name → PASS/FAIL
    # Case-insensitive: model may output "pass"/"Pass"/"PASS"
    vs_verdicts: Dict[str, bool] = {}
    for m in re.finditer(r'^vs\s+(.+?):\s*.*?(PASS|FAIL)\.', reasoning, re.MULTILINE | re.IGNORECASE):
        metric = m.group(1).strip().lower()
        vs_verdicts[metric] = (m.group(2).upper() == 'PASS')

    # Fallback: early-exit patterns without "vs " prefix
    # e.g., "MetricName: no events. FAIL."
    if not vs_verdicts:
        for m in re.finditer(r'^(.+?):\s*.*?(PASS|FAIL)\.', reasoning, re.MULTILINE | re.IGNORECASE):
            line_text = m.group(0)
            # Skip answer: and threshold: lines
            if line_text.strip().lower().startswith(('answer:', 'threshold')):
                continue
            metric = m.group(1).strip().lower()
            vs_verdicts[metric] = (m.group(2).upper() == 'PASS')

    if not vs_verdicts:
        return 0.5

    # Parse answer set: answer: (A, B, C) — handles nested parens like (CTR)
    answer_set = _extract_answer_set_balanced(reasoning)
    if answer_set is None:
        return 0.5

    # Empty answer set is valid when all metrics FAIL
    if not answer_set:
        all_fail = all(not v for v in vs_verdicts.values())
        return 1.0 if all_fail else 0.0

    # Check: PASS metrics should be in answer set, FAIL should not
    checks = []
    for metric, passed in vs_verdicts.items():
        in_answer = metric in answer_set
        if passed:
            checks.append(1.0 if in_answer else 0.0)
        else:
            checks.append(1.0 if not in_answer else 0.0)

    return sum(checks) / len(checks) if checks else 0.5


def _extract_answer_set_balanced(text: str) -> Optional[Set[str]]:
    """Parse answer: (A, B, C (D, E)) with balanced parentheses and paren-aware comma split."""
    m = re.search(r'^answer:\s*\(', text, re.MULTILINE | re.IGNORECASE)
    if not m:
        return None
    start = m.end()
    depth = 1
    i = start
    while i < len(text) and depth > 0:
        if text[i] == '(':
            depth += 1
        elif text[i] == ')':
            depth -= 1
        i += 1
    if depth != 0:
        return None
    inner = text[start:i - 1].strip()
    if not inner:
        return set()
    # Split by commas, but NOT commas inside parentheses
    parts = []
    current = []
    paren_depth = 0
    for ch in inner:
        if ch == '(':
            paren_depth += 1
            current.append(ch)
        elif ch == ')':
            paren_depth -= 1
            current.append(ch)
        elif ch == ',' and paren_depth == 0:
            parts.append(''.join(current).strip().lower())
            current = []
        else:
            current.append(ch)
    if current:
        parts.append(''.join(current).strip().lower())
    return {p for p in parts if p}


# ---------------------------------------------------------------------------
# 6. PASS/FAIL → answer: yes/no chain
#    Used by: yes_no (local event verification)
# ---------------------------------------------------------------------------

def coherency_yes_no(pred_think: str) -> float:
    """Check PASS/FAIL → answer: yes/no consistency in yes_no think blocks."""
    if '===' not in pred_think:
        return 0.5
    reasoning = get_reasoning(pred_think)

    # Parse PASS/FAIL from non-vs lines (same logic as extract_yes_no_result)
    passed = None
    for line in reasoning.split('\n'):
        line = line.strip()
        if not line:
            continue
        upper = line.upper()
        if upper.startswith('VS ') or upper.startswith('THRESHOLD') or upper.startswith('COMPARISON'):
            continue
        if upper.startswith('ANSWER:'):
            continue  # skip the answer line itself
        if 'PASS.' in upper and 'FAIL.' not in upper:
            passed = True
            break
        if 'FAIL.' in upper:
            passed = False
            break

    if passed is None:
        return 0.5

    # Parse answer: yes/no
    answer_match = re.search(r'^answer:\s*(yes|no)\s*$', reasoning, re.MULTILINE | re.IGNORECASE)
    if not answer_match:
        return 0.5

    answer = answer_match.group(1).lower()
    expected = 'yes' if passed else 'no'
    return 1.0 if answer == expected else 0.0


def _try_parse_half_stat(data_block: str, half: str) -> Optional[float]:
    """Try to parse a first_half or second_half stat value from the data block."""
    # Format 1: first_half_mean=X or first_half_std=X in a stats: line
    for stat_name in ('mean', 'std'):
        m = re.search(rf'{half}_half_{stat_name}\s*=\s*([\d.eE+-]+)', data_block)
        if m:
            return _safe_float(m.group(1))

    # Format 2: first_half [...]: mean=X or std=X
    m = re.search(rf'{half}_half\s*\[.*?\]:\s*(?:mean|std)\s*=\s*([\d.eE+-]+)', data_block)
    if m:
        return _safe_float(m.group(1))

    return None
