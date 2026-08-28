"""
synth.ts_generator.utils.thinking
===================================
Chain-of-Thought (CoT) block builders for all QA families.

Sub-modules:
  core        — ThoughtBuilder, build_local/trend_verification_thought, helpers
  segments    — Family 7: compound judgment
  trend       — Family 8 + MTS: trend dominance, anti-judgment, cross-trend
  statistical — Family 9: stat numerical
  enumeration — Families 11E: local event + segment + transition + duration proportion + cross-metric enumeration
"""

from synth.ts_generator.utils.thinking.core import (
    _anti_trend_type,
    _format_trend_segs,
    _assemble_thought,
    format_float,
    get_threshold_note,
    met_str,
    pass_str,
    format_seg_display,
    format_window_calc,
    format_cmp,
    format_abs_diff_cmp,
    format_str_eq,
    format_add_sub_calc,
    format_div_calc,
    format_mul_calc,
    format_tie,
    check_trend_match,
    ThoughtBuilder,
    generate_structural_verification,
    build_local_verification_thought,
    build_trend_verification_thought,
)
from synth.ts_generator.utils.thinking.segments import (
    _UPWARD_EVENT_TYPES,
    _DOWNWARD_EVENT_TYPES,
    build_compound_judgment_thought,
    build_cross_stat_judgment_thought,
)
from synth.ts_generator.utils.thinking.trend import (
    build_trend_dominance_thought,
    build_anti_judgment_thought,
    build_cross_trend_thought,
    build_change_point_thought,
)
from synth.ts_generator.utils.thinking.statistical import (
    build_stat_numerical_thought,
    build_periodicity_thought,

)
from synth.ts_generator.utils.thinking.enumeration import (
    build_local_enumeration_thought,
    build_segment_enumeration_thought,
    build_event_segment_enumeration_thought,
    build_temporal_position_thought,
    build_duration_proportion_thought,
    build_transition_enumeration_thought,
    build_cross_metric_enumeration_thought,
)

__all__ = [
    # core — atomic primitives
    "format_float",
    "met_str",
    "pass_str",
    "format_seg_display",
    # core — molecular helpers (3 single-source primitives + comparisons)
    "format_add_sub_calc",
    "format_div_calc",
    "format_mul_calc",
    "format_window_calc",
    "format_cmp",
    "format_abs_diff_cmp",
    "format_str_eq",
    "format_tie",
    # core — internals
    "_anti_trend_type",
    "_format_trend_segs",
    "_assemble_thought",
    "get_threshold_note",
    "check_trend_match",
    "ThoughtBuilder",
    "generate_structural_verification",
    "build_local_verification_thought",
    "build_trend_verification_thought",
    # segments
    "_UPWARD_EVENT_TYPES",
    "_DOWNWARD_EVENT_TYPES",
    "build_compound_judgment_thought",
    "build_cross_stat_judgment_thought",
    # trend
    "build_trend_dominance_thought",
    "build_anti_judgment_thought",
    "build_cross_trend_thought",
    "build_change_point_thought",
    # statistical
    "build_stat_numerical_thought",
    "build_periodicity_thought",

    # enumeration
    "build_local_enumeration_thought",
    "build_segment_enumeration_thought",
    "build_event_segment_enumeration_thought",
    "build_temporal_position_thought",
    "build_duration_proportion_thought",
    "build_transition_enumeration_thought",
    "build_cross_metric_enumeration_thought",
]
