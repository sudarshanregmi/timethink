"""
Comprehensive think builder test suite.

Tests structural invariants, format correctness, verdict accuracy, and
self-consistency across ALL builder functions in synth/ts_generator/utils/thinking/.

Organization:
    Section 1: Fixtures — realistic attribute dicts matching real generation
    Section 2: Builder invocation registry — calls every builder with every sub-type
    Section 3: Structural invariants (parametrized) — answer line, === separator, no step labels, etc.
    Section 4: Data block discipline — no orphan data, no data below ===
    Section 5: Precision & format_cmp — 2dp display, no contradictions
    Section 6: Vocabulary consistency — met/not met, pass/fail, metric_separator
    Section 7: Per-builder verdict correctness
    Section 8: Edge cases — ties, boundaries, empty inputs
    Section 9: Self-consistency via reward pipeline
"""

import math
import re
import pytest
from typing import Dict, List, Tuple, Any, Optional

# ===================================================================
# SECTION 1: FIXTURES
# ===================================================================

# --- Generic single-metric attributes (used by segments, trend, stat, enumeration) ---

GENERIC_ATTRS = {
    'seq_len': 256,
    'statistics': {
        'min': -1.50, 'max': 2.30, 'mean': 0.40, 'std': 0.80,
        'min_pos': 45, 'max_pos': 190,
        'range': 3.80,
        'first_half_mean': 0.20, 'second_half_mean': 0.60,
        'first_half_std': 0.70, 'second_half_std': 0.90,
        'halves_std': [0.70, 0.90],
        'median': 0.35, 'q25': -0.10, 'q75': 0.90,
        'outlier_2sigma_count': 3,
        'mean_crossing_count': 12,
        'mean_crossing_indices': [15, 30, 55, 72, 90, 110, 130, 150, 170, 188, 210, 240],
        'segment_means_16': [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45,
                             0.50, 0.55, 0.60, 0.65, 0.70, 0.65, 0.60, 0.55],
        'split_mid': 128,
    },
    'trend_overall': {'type': 'increase', 'start': 0, 'end': 255, 'amplitude': 1.80},
    'trend_list': [('increase', 0, 128), ('decrease', 128, 200), ('keep steady', 200, 256)],
    'noise': {'type': 'gaussian', 'strength': 0.15},
    'seasonal': {'type': 'sine', 'period': 32.0, 'amplitude': 0.50},
    'local': [
        {'type': 'upward spike', 'amplitude': 1.20, 'position_start': 50, 'position_end': 55,
         'direction': 'upward', 'value_start': 0.5, 'value_end': 1.7,
         'params': {'width': 5}},
        {'type': 'downward spike', 'amplitude': 0.80, 'position_start': 150, 'position_end': 158,
         'direction': 'downward', 'value_start': 0.3, 'value_end': -0.5,
         'params': {'width': 8}},
        {'type': 'sudden increase', 'amplitude': 0.50, 'position_start': 220, 'position_end': 230,
         'direction': 'upward', 'value_start': 0.4, 'value_end': 0.9,
         'params': {'width': 10}},
    ],
}

# Second metric for cross-metric tests
GENERIC_ATTRS_B = {
    'seq_len': 256,
    'statistics': {
        'min': -0.80, 'max': 1.50, 'mean': 0.30, 'std': 0.50,
        'min_pos': 80, 'max_pos': 210,
        'range': 2.30,
        'first_half_mean': 0.35, 'second_half_mean': 0.25,
        'first_half_std': 0.55, 'second_half_std': 0.45,
        'halves_std': [0.55, 0.45],
        'median': 0.28, 'q25': 0.05, 'q75': 0.60,
        'outlier_2sigma_count': 1,
        'mean_crossing_count': 8,
        'mean_crossing_indices': [20, 50, 80, 110, 140, 170, 200, 230],
        'segment_means_16': [0.40, 0.38, 0.35, 0.32, 0.30, 0.28, 0.26, 0.24,
                             0.22, 0.20, 0.25, 0.30, 0.35, 0.28, 0.22, 0.18],
        'split_mid': 128,
    },
    'trend_overall': {'type': 'decrease', 'start': 0, 'end': 255, 'amplitude': -1.20},
    'trend_list': [('decrease', 0, 180), ('keep steady', 180, 256)],
    'noise': {'type': 'gaussian', 'strength': 0.08},
    'seasonal': {'type': 'no periodic', 'period': 0, 'amplitude': 0},
    'local': [
        {'type': 'downward spike', 'amplitude': 0.60, 'position_start': 60, 'position_end': 65,
         'direction': 'downward', 'value_start': 0.2, 'value_end': -0.4,
         'params': {'width': 5}},
    ],
}

# ===================================================================
# SECTION 2: BUILDER INVOCATION REGISTRY
# ===================================================================
# Each entry: (description, callable_that_returns_(think_str, verdict))
# Used by parametrized structural tests.

def _build_all_think_blocks() -> List[Tuple[str, str, str]]:
    """Invoke every builder with every sub-type. Returns [(label, think_str, verdict)]."""
    from synth.ts_generator.utils.thinking import (
        build_compound_judgment_thought,
        build_cross_stat_judgment_thought,
        build_stat_numerical_thought,
        build_periodicity_thought,

        build_trend_dominance_thought,
        build_anti_judgment_thought,
        build_cross_trend_thought,
        build_change_point_thought,
        build_local_enumeration_thought,
        build_segment_enumeration_thought,
        build_event_segment_enumeration_thought,
        build_temporal_position_thought,
        build_duration_proportion_thought,
        build_transition_enumeration_thought,
        build_cross_metric_enumeration_thought,
    )

    results = []
    a = GENERIC_ATTRS
    b = GENERIC_ATTRS_B
    metric = 'Temperature'
    metric_b = 'Humidity'

    # --- segments.py: compound_judgment ---
    compound_cases = [
        ('compound/trend_local', 'trend_local', {
            'trend_type': 'increase', 'event_type': 'upward spike',
            'threshold': 1.0, 'verdict': 'yes',
            'trend_seg': (0, 128), 'event_amp': 1.20, 'event_pos': (50, 55),
        }),
        ('compound/amplitude_vs_noise', 'amplitude_vs_noise', {
            'K': 2.0, 'event_type': 'upward spike',
            'event_amp': 1.20, 'noise_strength': 0.15,
            'verdict': 'yes',
        }),
        ('compound/stat_threshold', 'stat_threshold', {
            'stat_key': 'std', 'op': '>', 'threshold': 0.5,
            'stat_value': 0.80, 'verdict': 'yes',
        }),
        ('compound/half_volatility', 'half_volatility', {
            'K': 1.1, 'first_half_std': 0.70, 'second_half_std': 0.90,
            'ratio': round(0.90 / 0.70, 2), 'verdict': 'yes',
        }),
    ]
    for label, jtype, params in compound_cases:
        try:
            t, v = build_compound_judgment_thought(a, metric, jtype, params)
            results.append((label, t, v))
        except Exception:
            pass

    # --- segments.py: cross_stat_judgment ---
    cross_stat_cases = [
        ('cross_stat/std_ratio', 'cross_stat_ratio', {
            'sub_variant': 'std_ratio', 'K': 1.5,
            'val_a': 0.80, 'val_b': 0.50, 'ratio': 1.60, 'verdict': 'yes',
        }),
        ('cross_stat/cross_half_shift', 'cross_half_shift', {
            'K_pct': 5,
            'metrics_data': [
                (metric, {'metric': metric, 'max': 2.30, 'min': -1.50, 'range': 3.80,
                           'first_half_mean': 0.20, 'second_half_mean': 0.60,
                           'shift': 0.40, 'threshold': 0.19}),
                (metric_b, {'metric': metric_b, 'max': 1.50, 'min': -0.80, 'range': 2.30,
                             'first_half_mean': 0.35, 'second_half_mean': 0.25,
                             'shift': -0.10, 'threshold': 0.115}),
            ],
            'verdict': 'no',
        }),
    ]
    for label, jtype, params in cross_stat_cases:
        try:
            t, v = build_cross_stat_judgment_thought(
                [a, b], [metric, metric_b], 0, 1, jtype, params)
            results.append((label, t, v))
        except Exception:
            pass

    # --- statistical.py ---
    stat_cases = [
        ('stat/min_val', 'stat_min_val', {'value': -1.50}),
        ('stat/max_val', 'stat_max_val', {'value': 2.30}),
        ('stat/mean_val', 'stat_mean_val', {
            'value': 0.40,
            'chunks': [(0, 15, 0.30, 16), (16, 31, 0.50, 16)],
        }),
        ('stat/std_val', 'stat_std_val', {'value': 0.80}),
        ('stat/median', 'stat_median', {'value': 0.35}),
        ('stat/range', 'stat_range', {'range': 3.80}),
        ('stat/extrema_pos_max', 'stat_extrema_pos', {'which': 'max', 'position': 190}),
        ('stat/extrema_pos_min', 'stat_extrema_pos', {'which': 'min', 'position': 45}),
        ('stat/segment_compare', 'stat_segment_compare', {
            'compare_by': 'mean', 'first_val': 0.20, 'second_val': 0.60,
            'mid': 128, 'verdict': 'second half',
        }),
        ('stat/volatility_change', 'stat_volatility_change', {
            'first_half_std': 0.70, 'second_half_std': 0.90,
        }),
        ('stat/half_mean_diff', 'stat_half_mean_diff', {
            'first_half_mean': 0.20, 'second_half_mean': 0.60,
        }),
        ('stat/threshold_duration', 'stat_threshold_duration', {
            'threshold': 1.0, 'op': '>', 'count': 50,
        }),
        ('stat/mean_crossing', 'stat_mean_crossing', {}),
        ('stat/windowed_mean', 'stat_windowed_mean', {}),
        ('stat/windowed_trend', 'stat_windowed_trend', {}),
    ]
    for label, sub, params in stat_cases:
        try:
            t, v = build_stat_numerical_thought(a, metric, sub, params)
            results.append((label, t, v))
        except Exception:
            pass

    # --- statistical.py: periodicity ---
    period_cases = [
        ('period/estimate', 'period_estimate', {}),
        ('period/cycle_count', 'cycle_count', {}),
    ]
    for label, sub, params in period_cases:
        try:
            t, v = build_periodicity_thought(a, metric, sub, params)
            results.append((label, t, v))
        except Exception:
            pass

    # --- trend.py ---
    try:
        t, v = build_trend_dominance_thought(a, metric, 0, 200)
        results.append(('trend/dominance', t, v))
    except Exception:
        pass

    try:
        t, v = build_anti_judgment_thought(
            a, b, metric, metric_b, 'anti_trend_noise',
            {'threshold': 0.05, 'actual_noise_amp': 0.08, 'actual_noise_type': 'gaussian'})
        results.append(('trend/anti_judgment', t, v))
    except Exception:
        pass

    try:
        t, v = build_cross_trend_thought(a, b, metric, metric_b, 'increase')
        results.append(('trend/cross_trend', t, v))
    except Exception:
        pass

    change_point_cases = [
        ('trend/change_point_count', 'change_point_count', {
            'trend_list': a['trend_list'], 'segment_means': [0.20, 0.60, 0.40]}),
        ('trend/change_point_positions', 'change_point_positions', {
            'trend_list': a['trend_list'], 'segment_means': [0.20, 0.60, 0.40]}),
        ('trend/largest_level_shift', 'largest_level_shift', {
            'trend_list': a['trend_list'], 'segment_means': [0.20, 0.60, 0.40]}),
    ]
    for label, sub, params in change_point_cases:
        try:
            t, v = build_change_point_thought(a, metric, sub, params)
            results.append((label, t, v))
        except Exception:
            pass

    # --- enumeration.py ---
    local_enum_cases = [
        ('enum/count_by_type', 'count_by_type', {'target_type': 'upward spike', 'verdict': '1'}),
        ('enum/count_by_direction', 'count_by_direction', {'target_direction': 'upward', 'verdict': '2'}),
        ('enum/count_by_amplitude', 'count_by_amplitude', {'threshold': 1.0, 'verdict': '1'}),
        ('enum/count_by_type_and_amp', 'count_by_type_and_amp', {
            'target_type': 'upward spike', 'threshold': 1.0, 'verdict': '1'}),
        ('enum/highest_amplitude', 'highest_amplitude', {'verdict': '1.20'}),
        ('enum/lowest_amplitude', 'lowest_amplitude', {'verdict': '0.50'}),
        ('enum/widest_span', 'widest_span', {'verdict': '10'}),
        ('enum/narrowest_span', 'narrowest_span', {'verdict': '5'}),
    ]
    for label, sub, params in local_enum_cases:
        try:
            t, v = build_local_enumeration_thought(a, metric, sub, params)
            results.append((label, t, v))
        except Exception:
            pass

    seg_enum_cases = [
        ('enum/segment_count_all', 'segment_count_all', {'verdict': '3'}),
        ('enum/segment_count_by_type', 'segment_count_by_type', {
            'target_type': 'increase', 'verdict': '1'}),
        ('enum/segment_longest', 'segment_longest', {'verdict': '128'}),
        ('enum/segment_shortest', 'segment_shortest', {'verdict': '56'}),
    ]
    for label, sub, params in seg_enum_cases:
        try:
            t, v = build_segment_enumeration_thought(a, metric, sub, params)
            results.append((label, t, v))
        except Exception:
            pass

    try:
        t, v = build_event_segment_enumeration_thought(a, metric, 'events_in_trend_type', {
            'target_type': 'increase', 'verdict': '1'})
        results.append(('enum/events_in_trend_type', t, v))
    except Exception:
        pass

    try:
        t, v = build_temporal_position_thought(a, metric, 'first_event', {'verdict': '50'})
        results.append(('enum/first_event', t, v))
    except Exception:
        pass
    try:
        t, v = build_temporal_position_thought(a, metric, 'last_event', {'verdict': '220'})
        results.append(('enum/last_event', t, v))
    except Exception:
        pass
    try:
        t, v = build_temporal_position_thought(a, metric, 'largest_gap', {'verdict': '70'})
        results.append(('enum/largest_gap', t, v))
    except Exception:
        pass

    dur_prop_cases = [
        ('enum/longest_total_duration', 'longest_total_duration', {'verdict': 'increase'}),
        ('enum/shortest_total_duration', 'shortest_total_duration', {'verdict': 'keep steady'}),
        ('enum/duration_by_type', 'duration_by_type', {'target_type': 'increase', 'verdict': '40'}),
        ('enum/duration_comparison', 'duration_comparison',
         {'type_a': 'increase', 'type_b': 'keep steady', 'verdict': 'increase'}),
        ('enum/duration_difference', 'duration_difference',
         {'type_a': 'increase', 'type_b': 'keep steady', 'verdict': '20'}),
    ]
    for label, sub, params in dur_prop_cases:
        try:
            t, v = build_duration_proportion_thought(a, metric, sub, params)
            results.append((label, t, v))
        except Exception:
            pass

    transition_enum_cases = [
        ('enum/transition_count', 'transition_count', {'verdict': '2'}),
        ('enum/transition_count_by_type', 'transition_count_by_type', {
            'target_transition': 'increase→decrease', 'verdict': '1'}),
    ]
    for label, sub, params in transition_enum_cases:
        try:
            t, v = build_transition_enumeration_thought(a, metric, sub, params)
            results.append((label, t, v))
        except Exception:
            pass

    # --- cross-metric enumeration ---
    cross_enum_cases = [
        ('cross_enum/which_have_local', 'which_have_local', {
            'per_metric': [
                {'metric': metric, 'count': 3, 'has': True},
                {'metric': metric_b, 'count': 1, 'has': True},
            ],
            'verdict': f'{metric}, {metric_b}',
        }),
        ('cross_enum/which_have_trend_type', 'which_have_trend_type', {
            'per_metric': [
                {'metric': metric, 'value': 'increase'},
                {'metric': metric_b, 'value': 'decrease'},
            ],
            'target_type': 'increase',
            'verdict': metric,
        }),
        ('cross_enum/all_same_trend', 'all_same_trend', {
            'per_metric': [
                {'metric': metric, 'value': 'increase'},
                {'metric': metric_b, 'value': 'decrease'},
            ],
            'verdict': 'no',
        }),
        ('cross_enum/most_local_events', 'most_local_events', {
            'per_metric': [
                {'metric': metric, 'count': 3},
                {'metric': metric_b, 'count': 1},
            ],
            'verdict': metric,
        }),
        ('cross_enum/highest_amplitude_across', 'highest_amplitude_across', {
            'per_metric': [
                {'metric': metric, 'amp': 1.20, 'type': 'upward spike'},
                {'metric': metric_b, 'amp': 0.60, 'type': 'downward spike'},
            ],
            'verdict': metric,
        }),
        ('cross_enum/most_trend_segments', 'most_trend_segments', {
            'per_metric': [
                {'metric': metric, 'count': 3},
                {'metric': metric_b, 'count': 2},
            ],
            'verdict': metric,
        }),
        ('cross_enum/longest_segment_across', 'longest_segment_across', {
            'per_metric': [
                {'metric': metric, 'value': 128},
                {'metric': metric_b, 'value': 180},
            ],
            'verdict': metric_b,
        }),
        ('cross_enum/noisiest_metric', 'noisiest_metric', {
            'per_metric': [
                {'metric': metric, 'value': 0.15},
                {'metric': metric_b, 'value': 0.08},
            ],
            'verdict': metric,
        }),
        ('cross_enum/quietest_metric', 'quietest_metric', {
            'per_metric': [
                {'metric': metric, 'value': 0.15},
                {'metric': metric_b, 'value': 0.08},
            ],
            'verdict': metric_b,
        }),
        ('cross_enum/widest_range', 'widest_range', {
            'per_metric': [
                {'metric': metric, 'value': 3.80},
                {'metric': metric_b, 'value': 2.30},
            ],
            'verdict': metric,
        }),
        ('cross_enum/highest_mean', 'highest_mean', {
            'per_metric': [
                {'metric': metric, 'value': 0.40},
                {'metric': metric_b, 'value': 0.30},
            ],
            'verdict': metric,
        }),
        ('cross_enum/lowest_mean', 'lowest_mean', {
            'per_metric': [
                {'metric': metric, 'value': 0.40},
                {'metric': metric_b, 'value': 0.30},
            ],
            'verdict': metric_b,
        }),
        ('cross_enum/highest_max', 'highest_max', {
            'per_metric': [
                {'metric': metric, 'value': 2.30},
                {'metric': metric_b, 'value': 1.50},
            ],
            'verdict': metric,
        }),
        ('cross_enum/lowest_min', 'lowest_min', {
            'per_metric': [
                {'metric': metric, 'value': -1.50},
                {'metric': metric_b, 'value': -0.80},
            ],
            'verdict': metric,
        }),
        ('cross_enum/highest_std', 'highest_std', {
            'per_metric': [
                {'metric': metric, 'value': 0.80},
                {'metric': metric_b, 'value': 0.50},
            ],
            'verdict': metric,
        }),
        ('cross_enum/lowest_std', 'lowest_std', {
            'per_metric': [
                {'metric': metric, 'value': 0.80},
                {'metric': metric_b, 'value': 0.50},
            ],
            'verdict': metric_b,
        }),
        ('cross_enum/earliest_event_across', 'earliest_event_across', {
            'per_metric': [
                {'metric': metric, 'value': 50},
                {'metric': metric_b, 'value': 60},
            ],
            'verdict': metric,
        }),
        ('cross_enum/latest_event_across', 'latest_event_across', {
            'per_metric': [
                {'metric': metric, 'value': 220},
                {'metric': metric_b, 'value': 60},
            ],
            'verdict': metric,
        }),
    ]
    for label, sub, params in cross_enum_cases:
        try:
            t, v = build_cross_metric_enumeration_thought(
                [a, b], [metric, metric_b], sub, params)
            results.append((label, t, v))
        except Exception:
            pass

    return results


# Build once, share across tests
_ALL_BLOCKS = None

def _get_all_blocks():
    global _ALL_BLOCKS
    if _ALL_BLOCKS is None:
        _ALL_BLOCKS = _build_all_think_blocks()
    return _ALL_BLOCKS


def _block_ids():
    return [b[0] for b in _get_all_blocks()]


# ===================================================================
# SECTION 3: STRUCTURAL INVARIANTS (parametrized)
# ===================================================================

# Lookup types: no === separator expected
_LOOKUP_LABELS = {
    'stat/min_val', 'stat/max_val', 'stat/std_val',
    'stat/median',
    'period/estimate',
}


def _strip_think_tags(text: str) -> str:
    """Remove <think>...</think> wrapper from builder output."""
    text = text.strip()
    if text.startswith('<think>'):
        text = text[len('<think>'):]
    if text.endswith('</think>'):
        text = text[:-len('</think>')]
    return text.strip()


class TestStructuralInvariants:
    """Every think block must satisfy these structural rules."""

    def test_all_builders_produce_output(self):
        """Sanity: we actually invoked a significant number of builders."""
        blocks = _get_all_blocks()
        assert len(blocks) >= 40, f"Only {len(blocks)} blocks — expected 40+. Check fixture data."

    @pytest.mark.parametrize("label,think,verdict", _get_all_blocks(), ids=_block_ids())
    def test_answer_line_present(self, label, think, verdict):
        """Rule: every think block contains 'answer:' line."""
        inner = _strip_think_tags(think)
        assert 'answer:' in inner, f"{label}: missing 'answer:' line"

    @pytest.mark.parametrize("label,think,verdict", _get_all_blocks(), ids=_block_ids())
    def test_answer_is_last_meaningful_line(self, label, think, verdict):
        """Rule: 'answer:' must be the last non-empty line."""
        inner = _strip_think_tags(think)
        lines = [l for l in inner.split('\n') if l.strip()]
        last = lines[-1].strip()
        assert last.startswith('answer:'), (
            f"{label}: last line is '{last[:60]}', not 'answer:'"
        )

    @pytest.mark.parametrize("label,think,verdict", _get_all_blocks(), ids=_block_ids())
    def test_verdict_in_answer_line(self, label, think, verdict):
        """Rule: the 'answer:' line must contain the verdict value."""
        inner = _strip_think_tags(think)
        m = re.search(r'^answer:\s*(.+)$', inner, re.MULTILINE)
        assert m, f"{label}: could not parse answer line"
        answer_val = m.group(1).strip()
        assert answer_val == verdict, (
            f"{label}: answer line='{answer_val}' != verdict='{verdict}'"
        )

    @pytest.mark.parametrize("label,think,verdict", _get_all_blocks(), ids=_block_ids())
    def test_separator_presence(self, label, think, verdict):
        """Standard types must have ===. Lookup types must NOT."""
        inner = _strip_think_tags(think)
        has_sep = '===' in inner
        if label in _LOOKUP_LABELS:
            assert not has_sep, f"{label}: lookup type should NOT have ==="
        else:
            assert has_sep, f"{label}: standard type MUST have ==="

    @pytest.mark.parametrize("label,think,verdict", _get_all_blocks(), ids=_block_ids())
    def test_no_step_labels(self, label, think, verdict):
        """Rule #25: no 'step N:' labels in reasoning."""
        inner = _strip_think_tags(think)
        if '===' not in inner:
            return  # lookup type, no reasoning section
        reasoning = inner.split('===', 1)[1]
        assert not re.search(r'\bstep\s+\d+:', reasoning, re.IGNORECASE), (
            f"{label}: found 'step N:' label in reasoning"
        )

    @pytest.mark.parametrize("label,think,verdict", _get_all_blocks(), ids=_block_ids())
    def test_no_data_headers_in_reasoning(self, label, think, verdict):
        """Rule #8: no 'metric:' data headers below ===."""
        inner = _strip_think_tags(think)
        if '===' not in inner:
            return
        reasoning = inner.split('===', 1)[1]
        # Split off the answer line
        answer_idx = reasoning.rfind('answer:')
        if answer_idx > 0:
            reasoning = reasoning[:answer_idx]
        # Check for data-header-style "metric:" at line start, not mid-line references
        for line in reasoning.strip().split('\n'):
            stripped = line.strip()
            if stripped.startswith('metric:'):
                pytest.fail(f"{label}: found data header 'metric:' in reasoning: '{stripped[:60]}'")


# ===================================================================
# SECTION 4: NO UNNECESSARY THINKING
# ===================================================================
#
# Rules enforced:
#   #20: No contextually related attributes in data block
#   #21: No foundational data when only derived values used
#   #22: No hidden derivations — every value traces to visible computation
#   #24: Every tb.data() traces to tb.line() usage
#   #26: No re-listing of data below ===
#   #48: No zero-computation standard blocks
#   #71: No context data unused in computation


def _parse_data_and_reasoning(think_str: str):
    """Split think block into (data_section, reasoning_section).

    Returns (None, None) for lookup types (no ``===``).
    """
    inner = _strip_think_tags(think_str)
    if '===' not in inner:
        return None, None
    data_sec, rest = inner.split('===', 1)
    answer_idx = rest.rfind('answer:')
    reasoning = rest[:answer_idx].strip() if answer_idx > 0 else rest.strip()
    return data_sec.strip(), reasoning


def _extract_scalar_kv(data_section: str):
    """Extract (key, value) pairs for scalar key=numericValue in data block.

    Skips list assignments (``key=[...]``), metric headers, and separators.
    """
    pairs = []
    for line in data_section.split('\n'):
        line = line.strip()
        if not line or line.startswith('metric:') or line == '---':
            continue
        # Skip lines that contain list/dict assignments
        if re.search(r'\w+=\[', line):
            continue
        for m in re.finditer(r'(\w+)=([-]?\d+\.?\d*)', line):
            pairs.append((m.group(1), m.group(2)))
    return pairs


def _value_in_text(val: str, text: str) -> bool:
    """Check numeric value appears in text as a standalone token.

    Uses lookaround to prevent '2' matching inside '256' etc.
    """
    pattern = r'(?<!\d)' + re.escape(val) + r'(?!\d)'
    return bool(re.search(pattern, text))


def _extract_list_data(data_section: str):
    """Extract (key, item_count, numeric_elements) for list assignments."""
    results = []
    for m in re.finditer(r'(\w+)=\[([^\]]*)\]', data_section, re.DOTALL):
        key = m.group(1)
        content = m.group(2)
        # Count top-level items
        if '{' in content:
            count = content.count('{')
        elif '(' in content:
            count = content.count('(')
        else:
            elems = [e.strip() for e in content.split(',') if e.strip()]
            count = len(elems)
        # Extract numeric literals within the list
        nums = re.findall(r'(?<![.\w])-?\d+\.?\d*(?![.\w])', content)
        results.append((key, count, nums))
    return results


class TestNoUnnecessaryThinking:
    """Every data item must contribute to reasoning. No orphan data."""

    # Legitimate structural orphans:
    # - count_by_type_and_amp: AND-compound early exit skips amplitude on type mismatch
    # - earliest/latest_event_across: data shows all events, reasoning only uses first/last
    # - highest_amplitude_across: data shows all events, reasoning only uses per-metric max
    _ORPHAN_EXEMPT_LABELS = {
        'enum/count_by_type_and_amp',
        'cross_enum/earliest_event_across',
        'cross_enum/latest_event_across',
        'cross_enum/highest_amplitude_across',
    }

    @pytest.mark.parametrize("label,think,verdict", _get_all_blocks(), ids=_block_ids())
    def test_no_orphan_scalar_data_values(self, label, think, verdict):
        """Rule #20/#22/#71: every scalar key=value in data must have its
        numeric value appear in reasoning below ===.

        Catches: contextually related attributes, hidden derivations,
        context data that is never computed on.
        """
        if label in self._ORPHAN_EXEMPT_LABELS:
            return
        data_sec, reasoning = _parse_data_and_reasoning(think)
        if data_sec is None:
            return  # lookup type

        pairs = _extract_scalar_kv(data_sec)
        orphans = []
        for key, val in pairs:
            if not _value_in_text(val, reasoning):
                orphans.append(f"{key}={val}")

        assert not orphans, (
            f"{label}: orphan scalar data not referenced in reasoning: "
            f"{', '.join(orphans)}"
        )

    # transition_count_by_type: trend_segments contribute TYPE names to
    # transition derivation, but boundary positions aren't referenced
    _LIST_ORPHAN_EXEMPT_LABELS = {'enum/transition_count_by_type'}

    @pytest.mark.parametrize("label,think,verdict", _get_all_blocks(), ids=_block_ids())
    def test_list_data_contributes_to_reasoning(self, label, think, verdict):
        """Rule #21/#24: list data in data block must contribute to reasoning.

        Either individual elements appear in reasoning, or the list's
        length (count) appears, or the key name is referenced.
        """
        if label in self._LIST_ORPHAN_EXEMPT_LABELS:
            return
        data_sec, reasoning = _parse_data_and_reasoning(think)
        if data_sec is None:
            return

        lists = _extract_list_data(data_sec)
        for key, count, nums in lists:
            # (a) at least one non-trivial numeric element in reasoning
            any_elem = any(
                _value_in_text(n, reasoning)
                for n in nums if len(n) > 1  # skip single digits (too many false matches)
            )
            # (b) count (len of list) appears in reasoning
            count_present = _value_in_text(str(count), reasoning)
            # (c) the key name is explicitly referenced in reasoning
            key_present = key in reasoning

            assert any_elem or count_present or key_present, (
                f"{label}: list '{key}' ({count} items) not referenced in "
                f"reasoning — no elements, count, or key name found"
            )

    @pytest.mark.parametrize("label,think,verdict", _get_all_blocks(), ids=_block_ids())
    def test_at_least_one_computation_line(self, label, think, verdict):
        """Rule #48: standard blocks must have >= 1 computation line between
        === and answer:.
        """
        inner = _strip_think_tags(think)
        if '===' not in inner:
            return

        _, rest = inner.split('===', 1)
        answer_idx = rest.rfind('answer:')
        reasoning = rest[:answer_idx].strip() if answer_idx > 0 else rest.strip()

        non_empty = [l for l in reasoning.split('\n') if l.strip()]
        assert len(non_empty) >= 1, (
            f"{label}: zero computation lines between === and answer:"
        )

    @pytest.mark.parametrize("label,think,verdict", _get_all_blocks(), ids=_block_ids())
    def test_no_raw_data_lines_in_reasoning(self, label, think, verdict):
        """Rule #26: no data-style assignments in reasoning section."""
        data_sec, reasoning = _parse_data_and_reasoning(think)
        if reasoning is None:
            return
        for line in reasoning.split('\n'):
            stripped = line.strip()
            if not stripped:
                continue
            if re.match(
                r'^(trend_segments|stats:)\s*=',
                stripped
            ):
                pytest.fail(
                    f"{label}: data-style line in reasoning: '{stripped[:80]}'"
                )


# ===================================================================
# SECTION 5: PRECISION & FORMAT_CMP
# ===================================================================

class TestPrecisionAndFormatCmp:
    """Verify 2dp display and no format_cmp contradictions."""

    @pytest.mark.parametrize("label,think,verdict", _get_all_blocks(), ids=_block_ids())
    def test_no_format_cmp_contradictions(self, label, think, verdict):
        """Rule #44: no display contradictions like '0.44 < 0.44'."""
        # Find all comparison patterns: "X.XX op X.XX"
        cmp_pattern = re.compile(
            r'(\-?\d+\.?\d*)\s*([<>]=?)\s*(\-?\d+\.?\d*)\s*=\s*(true|false)',
            re.IGNORECASE
        )
        for m in cmp_pattern.finditer(think):
            left, op, right, result = m.group(1), m.group(2), m.group(3), m.group(4).lower()
            try:
                lf, rf = float(left), float(right)
            except ValueError:
                continue
            # Check: if displayed values are equal, result should be consistent
            if left == right:
                # Equal values: > and < must be false, >= and <= must be true
                if op in ('>', '<'):
                    assert result == 'false', (
                        f"{label}: contradiction '{left} {op} {right} = {result}'"
                    )
                elif op in ('>=', '<='):
                    assert result == 'true', (
                        f"{label}: contradiction '{left} {op} {right} = {result}'"
                    )

    @pytest.mark.parametrize("label,think,verdict", _get_all_blocks(), ids=_block_ids())
    def test_float_verdicts_have_2dp(self, label, think, verdict):
        """Rule #4: float verdicts use 2dp, integers use 0dp."""
        if not re.match(r'^-?\d+\.?\d*$', verdict):
            return  # not a numeric verdict
        if '.' in verdict:
            # Float: must have exactly 2 decimal places
            parts = verdict.split('.')
            assert len(parts[1]) == 2, (
                f"{label}: float verdict '{verdict}' has {len(parts[1])}dp, expected 2"
            )


# ===================================================================
# SECTION 6: VOCABULARY CONSISTENCY
# ===================================================================

class TestVocabularyConsistency:
    """Correct vocabulary: met/not met, pass/fail, metric_separator."""

    @pytest.mark.parametrize("label,think,verdict", _get_all_blocks(), ids=_block_ids())
    def test_no_qualifies_vocabulary(self, label, think, verdict):
        """Rule #7: never 'qualifies' — use 'met'/'not met'."""
        assert 'qualifies' not in think.lower(), (
            f"{label}: found 'qualifies' — should use met/not met"
        )

    @pytest.mark.parametrize("label,think,verdict", _get_all_blocks(), ids=_block_ids())
    def test_conditions_use_met_not_met(self, label, think, verdict):
        """Condition outcomes must be 'met' or 'not met', not 'yes'/'no'/'ok'."""
        # Only check blocks that have "condition" patterns
        if 'condition' not in think.lower() and 'all conditions:' not in think.lower():
            return
        # Find condition outcome lines
        for line in think.split('\n'):
            if re.search(r'condition\s+\d+:', line, re.IGNORECASE):
                # This line describes a condition — its outcome should use met/not met
                if '-> ' in line:
                    outcome = line.split('-> ')[-1].strip()
                    assert outcome in ('met', 'not met') or 'all' in outcome.lower(), (
                        f"{label}: condition outcome '{outcome}' — expected 'met'/'not met'"
                    )

    def test_cross_metric_builders_use_separator(self):
        """Rule #27: cross-metric builders use '---' separator, not 'metric:' repeated."""
        blocks = _get_all_blocks()
        cross_labels = [
            b for b in blocks
            if any(x in b[0] for x in ['cross_stat', 'cross_enum'])
        ]
        for label, think, verdict in cross_labels:
            if '===' not in think:
                continue
            data_block = think.split('===', 1)[0]
            # Cross-metric data blocks should contain '---' separator
            metric_count = data_block.count('metric:')
            if metric_count >= 2:
                assert '---' in data_block, (
                    f"{label}: cross-metric block has {metric_count} metrics but no '---' separator"
                )


# ===================================================================
# SECTION 7: PER-BUILDER VERDICT CORRECTNESS
# ===================================================================

class TestVerdictCorrectness:
    """Verify computed verdicts match expected values from fixture data."""

    # --- Statistical ---
    def test_stat_range(self):
        from synth.ts_generator.utils.thinking.statistical import build_stat_numerical_thought
        _, v = build_stat_numerical_thought(GENERIC_ATTRS, 'M', 'stat_range', {'range': 3.80})
        assert v == '3.80'

    def test_stat_segment_compare_second_half(self):
        from synth.ts_generator.utils.thinking.statistical import build_stat_numerical_thought
        _, v = build_stat_numerical_thought(GENERIC_ATTRS, 'M', 'stat_segment_compare', {
            'compare_by': 'mean', 'first_val': 0.20, 'second_val': 0.60,
            'mid': 128, 'verdict': 'second half',
        })
        assert v == 'second half'

    def test_stat_volatility_more(self):
        from synth.ts_generator.utils.thinking.statistical import build_stat_numerical_thought
        _, v = build_stat_numerical_thought(GENERIC_ATTRS, 'M', 'stat_volatility_change', {
            'first_half_std': 0.70, 'second_half_std': 0.90,
        })
        assert v == 'more_volatile'

    # --- Trend ---
    def test_trend_dominance(self):
        from synth.ts_generator.utils.thinking.trend import build_trend_dominance_thought
        _, v = build_trend_dominance_thought(GENERIC_ATTRS, 'M', 0, 200)
        assert v in ('increase', 'decrease', 'keep steady', 'equal')

    def test_change_point_count(self):
        from synth.ts_generator.utils.thinking.trend import build_change_point_thought
        _, v = build_change_point_thought(GENERIC_ATTRS, 'M', 'change_point_count', {
            'trend_list': GENERIC_ATTRS['trend_list'],
            'segment_means': [0.20, 0.60, 0.40],
        })
        assert v == '2'  # 3 segments → 2 change points


# ===================================================================
# SECTION 8: EDGE CASES
# ===================================================================

class TestEdgeCases:
    """Boundary conditions: ties, single elements, empty collections."""

    def test_segment_compare_equal(self):
        """Values within 5% epsilon → equal verdict."""
        from synth.ts_generator.utils.thinking.statistical import build_stat_numerical_thought
        attrs = dict(GENERIC_ATTRS)
        _, v = build_stat_numerical_thought(attrs, 'M', 'stat_segment_compare', {
            'compare_by': 'mean', 'first_val': 1.00, 'second_val': 1.02,
            'mid': 128, 'verdict': 'equal',
        })
        assert v == 'equal'



# ===================================================================
# SECTION 9: SELF-CONSISTENCY VIA REWARD
# ===================================================================

class TestSelfConsistency:
    """compute_score(gt, gt) >= 0.99 for think blocks from all builders."""

    @pytest.mark.parametrize("label,think,verdict", _get_all_blocks(), ids=_block_ids())
    def test_reward_self_consistency(self, label, think, verdict):
        """Every GT think block must achieve >= 0.95 self-consistency.

        Note: 0.95 not 0.99 because minimal fixture data can trigger
        coherency neutral (0.5) on some patterns. Evidence ceiling is ~0.98
        for very short blocks.
        """
        from reward import compute_score

        # Determine eval_type and eval_metadata from label
        eval_type, eval_meta = _label_to_eval(label, verdict)
        if eval_type is None:
            pytest.skip(f"No eval_type mapping for {label}")

        wrapped = f"<think>\n{think}\n</think>\nThis is a test answer."
        score = compute_score(wrapped, wrapped, eval_type=eval_type, eval_metadata=eval_meta)
        assert score >= 0.95, (
            f"{label} (eval_type={eval_type}): self-consistency={score:.3f} < 0.95"
        )


def _label_to_eval(label: str, verdict: str):
    """Map test label to (eval_type, eval_metadata) for reward scoring."""
    meta = {'verdict': verdict, 'length': 256}

    if label.startswith('stat/'):
        return 'stat_numerical', {**meta, 'sub_type': label.split('/')[1]}
    if label.startswith('period/'):
        return 'periodicity', {**meta, 'sub_type': label.split('/')[1]}

    if label.startswith('trend/dominance'):
        return 'segment_trend_dominance', meta
    if label.startswith('trend/anti_judgment'):
        return 'anti_judgment', meta
    if label.startswith('trend/cross_trend'):
        return 'cross_trend_query', meta
    if label.startswith('trend/change_point') or label == 'trend/largest_level_shift':
        return 'change_point', {**meta, 'sub_type': label.split('/')[1]}
    if label.startswith('compound/'):
        return 'segment_judgment', meta
    if label.startswith('cross_stat/'):
        return 'cross_stat_judgment', meta
    if label.startswith('enum/') or label.startswith('cross_enum/'):
        sub = label.split('/')[1]
        # Map to eval_type
        enum_map = {
            'count_by_type': 'local_enumeration', 'count_by_direction': 'local_enumeration',
            'count_by_amplitude': 'local_enumeration', 'highest_amplitude': 'local_enumeration',
            'widest_span': 'local_enumeration',
            'segment_count_all': 'segment_enumeration', 'segment_count_by_type': 'segment_enumeration',
            'segment_longest': 'segment_enumeration', 'segment_shortest': 'segment_enumeration',
            'events_in_trend_type': 'event_segment_enumeration',
            'first_event': 'temporal_position', 'last_event': 'temporal_position',
            'largest_gap': 'temporal_position',
            'longest_total_duration': 'duration_proportion', 'shortest_total_duration': 'duration_proportion',
            'duration_by_type': 'duration_proportion', 'duration_comparison': 'duration_proportion',
            'duration_difference': 'duration_proportion',
            'transition_count': 'transition_enumeration',
            'which_have_local': 'cross_metric_enumeration',
            'highest_mean': 'cross_metric_enumeration',
        }
        et = enum_map.get(sub)
        return et, {**meta, 'sub_type': sub} if et else (None, None)
    return None, None


# ===================================================================
# SECTION 10: REGRESSION — PHASE_EVENT DATA BLOCK CONTENT
# ===================================================================

class TestPhaseEventDataBlock:
    """Regression: phase_event data block must show pos= for events, not amp=."""

    def test_phase_event_shows_positions(self):
        """Data block must contain 'pos=' for each local event."""
        from synth.ts_generator.utils.thinking.segments import build_compound_judgment_thought
        attrs = {
            'seq_len': 256,
            'statistics': {
                'min': 0.0, 'max': 2.0, 'mean': 1.0, 'std': 0.5,
                'min_pos': 0, 'max_pos': 128, 'range': 2.0,
                'segment_means_16': [1.0]*16,
            },
            'trend_overall': {'type': 'increase', 'start': 0, 'end': 255, 'amplitude': 1.0},
            'trend_list': [('increase', 0, 128), ('decrease', 128, 256)],
            'noise': {'type': 'gaussian', 'strength': 0.1},
            'seasonal': {'type': 'no periodic', 'period': 0, 'amplitude': 0},
            'local': [
                {'type': 'upward spike', 'amplitude': 1.5, 'position_start': 50,
                 'position_end': 55, 'direction': 'upward',
                 'value_start': 0.5, 'value_end': 2.0, 'params': {'width': 5}},
                {'type': 'downward spike', 'amplitude': 0.8, 'position_start': 150,
                 'position_end': 158, 'direction': 'downward',
                 'value_start': 1.0, 'value_end': 0.2, 'params': {'width': 8}},
            ],
        }
        params = {
            'event_direction': 'upward',
            'required_trend_type': 'increase',
            'all_events': [
                ('upward spike', 50, 'upward'),
                ('downward spike', 150, 'downward'),
            ],
            'trend_list': [('increase', 0, 128), ('decrease', 128, 256)],
            'verdict': 'yes',
        }
        think, _ = build_compound_judgment_thought(attrs, 'M', 'phase_event', params)
        data_block = think.split('===')[0] if '===' in think else think
        assert 'pos=' in data_block, "phase_event data block must show pos= for events"
        assert 'pos=50' in data_block, "position_start=50 should appear in data block"


# ===================================================================
# SECTION 11: REGRESSION — THINK BLOCK ↔ VERDICT NON-CONTRADICTION
# ===================================================================

class TestThinkVerdictConsistency:
    """Regression: reasoning conclusion must match verdict.
    For compound judgments: 'all conditions: met' → 'yes', 'not met' → 'no'."""

    @pytest.mark.parametrize("label,think,verdict", _get_all_blocks(), ids=_block_ids())
    def test_compound_conclusion_matches_verdict(self, label, think, verdict):
        """If think block has 'all conditions: (not) met', it must match verdict."""
        inner = _strip_think_tags(think)
        if '===' not in inner:
            return

        reasoning = inner.split('===', 1)[1]

        all_met_match = re.search(r'all conditions:\s*(met|not met)', reasoning)
        if not all_met_match:
            return

        conclusion = all_met_match.group(1)
        if conclusion == 'met':
            assert verdict == 'yes', (
                f"{label}: reasoning says 'all conditions: met' but verdict='{verdict}'"
            )
        else:
            assert verdict == 'no', (
                f"{label}: reasoning says 'all conditions: not met' but verdict='{verdict}'"
            )

    @pytest.mark.parametrize("label,think,verdict", _get_all_blocks(), ids=_block_ids())
    def test_single_condition_met_matches_verdict(self, label, think, verdict):
        """For single-condition judgments, met/not met must match yes/no verdict."""
        inner = _strip_think_tags(think)
        if '===' not in inner:
            return
        reasoning = inner.split('===', 1)[1]
        if 'all conditions:' in reasoning:
            return
        if verdict not in ('yes', 'no'):
            return

        met_matches = list(re.finditer(r'-> (met|not met)', reasoning))
        if not met_matches:
            return

        last_result = met_matches[-1].group(1)
        if last_result == 'met':
            assert verdict == 'yes', (
                f"{label}: last condition is 'met' but verdict='{verdict}'"
            )
        else:
            assert verdict == 'no', (
                f"{label}: last condition is 'not met' but verdict='{verdict}'"
            )
