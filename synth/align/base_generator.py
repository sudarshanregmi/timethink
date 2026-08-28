"""Base class for QA generators with shared functionality."""

import random
from abc import ABC, abstractmethod
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

from loguru import logger

import numpy as np

from synth.ts_generator.generate import attribute_to_text
from synth.align.templates.prompts import (
    _fix_article,
    pluralize,
    COMPOUND_DESCRIPTION_MIXED_TEMPLATES,
    COMPOUND_DESCRIPTION_UNIFORM_TEMPLATES,
)
from synth.ts_generator.utils.common_utils import (
    article,
    format_float,
    format_list_natural_language,
    has_local_event_near,
    get_local_event_positions,
    find_point_far_from_all_events,
    find_point_at_distance_range,
    sanitize_attributes_for_sync,
    DEFAULT_THRESHOLD,
)
from synth.ts_generator.utils.thinking import (
    build_local_verification_thought,
    ThoughtBuilder,
    build_compound_judgment_thought,
    build_cross_stat_judgment_thought,
    build_trend_dominance_thought,
    build_stat_numerical_thought,
    build_periodicity_thought,

    build_local_enumeration_thought,
    build_segment_enumeration_thought,
    build_event_segment_enumeration_thought,
    build_temporal_position_thought,
    build_duration_proportion_thought,
    build_transition_enumeration_thought,
    build_cross_metric_enumeration_thought,
    build_change_point_thought,
    _UPWARD_EVENT_TYPES,
    _DOWNWARD_EVENT_TYPES,
)
from synth.ts_generator.utils.probability_utils import (
    weighted_random_choice,
    QAType,
)


def _classify_taxonomy_qa(r: Dict[str, Any]) -> QAType:
    """Dispatch a taxonomy QA to its semantic bucket.

    Rule order matters: bridges must be checked BEFORE the `rl_*` prefix,
    because bridge samples share eval_types with their RL counterparts
    (e.g., a bridge demonstrating rl_half_mean_compare has
    eval_type='rl_half_mean_compare' but eval_metadata['bridge']=True).
    """
    eval_meta = r.get('eval_metadata') or {}
    if eval_meta.get('bridge') is True:
        return QAType.SFT_BRIDGE
    eval_type = r.get('eval_type', '')
    if eval_type.startswith('atomic_'):
        return QAType.SFT_ATOMIC
    if eval_type.startswith('rl_'):
        return QAType.RL_COMPOSITION
    # Fallback — shouldn't happen for current taxonomy generators, but if a
    # new generator family appears without updating this classifier, bucket
    # it as RL_COMPOSITION (the largest existing bucket) rather than
    # silently dropping. Startup validation will still flag the missing
    # prefix during generation.
    return QAType.RL_COMPOSITION
from synth.align.templates.uts_questions import (
    _TREND_DOMINANCE_QUESTIONS,
    _JUDGMENT_TREND_LOCAL_QUESTIONS,
    _JUDGMENT_MULTI_PHASE_QUESTIONS,
    _JUDGMENT_NOISE_TREND_QUESTIONS,
    _JUDGMENT_STAT_QUESTIONS,
    _JUDGMENT_AMPLITUDE_VS_NOISE_QUESTIONS,
    _JUDGMENT_AMPLITUDE_VS_STD_QUESTIONS,
    _JUDGMENT_RANGE_VS_STD_QUESTIONS,
    _JUDGMENT_HALF_VOLATILITY_QUESTIONS,
    _JUDGMENT_HALF_MEAN_SHIFT_QUESTIONS,
    _JUDGMENT_WINDOWED_MONOTONICITY_QUESTIONS,
    _JUDGMENT_PHASE_EVENT_QUESTIONS,
    _JUDGMENT_SEQUENTIAL_EVENTS_QUESTIONS,
    _STAT_MIN_VAL_QUESTIONS,
    _STAT_MAX_VAL_QUESTIONS,
    _STAT_MEAN_VAL_QUESTIONS,
    _STAT_STD_VAL_QUESTIONS,
    _STAT_PEAK_QUESTIONS,
    _STAT_TROUGH_QUESTIONS,
    _STAT_WINDOWED_MEAN_FULL_QUESTIONS,
    _STAT_WINDOWED_MEAN_PARTIAL_QUESTIONS,
    _STAT_THRESHOLD_ABOVE_MEAN_QUESTIONS,
    _STAT_THRESHOLD_ABOVE_STD_QUESTIONS,
    _STAT_THRESHOLD_BELOW_STD_QUESTIONS,
    _STAT_MEAN_CROSSING_QUESTIONS,
    _STAT_RANGE_QUESTIONS,
    _STAT_SEGMENT_COMPARE_MEAN_QUESTIONS,
    _STAT_SEGMENT_COMPARE_STD_QUESTIONS,
    _LOCAL_COUNT_BY_TYPE_QUESTIONS,
    _LOCAL_COUNT_BY_DIRECTION_QUESTIONS,
    _LOCAL_COUNT_BY_AMPLITUDE_QUESTIONS,
    _LOCAL_COUNT_BY_TYPE_AND_AMP_QUESTIONS,
    _LOCAL_HIGHEST_AMPLITUDE_QUESTIONS,
    _LOCAL_LOWEST_AMPLITUDE_QUESTIONS,
    _LOCAL_WIDEST_SPAN_QUESTIONS,
    _LOCAL_NARROWEST_SPAN_QUESTIONS,
    _SEGMENT_COUNT_ALL_QUESTIONS,
    _SEGMENT_COUNT_BY_TYPE_QUESTIONS,
    _SEGMENT_LONGEST_QUESTIONS,
    _SEGMENT_SHORTEST_QUESTIONS,
    _TRANSITION_COUNT_QUESTIONS,
    _TRANSITION_COUNT_BY_TYPE_QUESTIONS,
    _TRANSITION_MOST_COMMON_QUESTIONS,
    _EVENT_SEG_COUNT_IN_TREND_TYPE_QUESTIONS,
    _EVENT_SEG_MOST_EVENTS_QUESTIONS,
    _EVENT_SEG_TREND_TYPE_MOST_EVENTS_QUESTIONS,
    _FIRST_EVENT_QUESTIONS,
    _LAST_EVENT_QUESTIONS,
    _LARGEST_GAP_QUESTIONS,
    _SMALLEST_GAP_QUESTIONS,
    _DURATION_BY_TYPE_QUESTIONS,
    _LONGEST_DURATION_QUESTIONS,
    _SHORTEST_DURATION_QUESTIONS,
    _DURATION_COMPARISON_QUESTIONS,
    _DURATION_DIFFERENCE_QUESTIONS,
    _PERIODICITY_PERIOD_QUESTIONS,
    _PERIODICITY_CYCLE_COUNT_QUESTIONS,
    _CHANGE_POINT_COUNT_QUESTIONS,
    _CHANGE_POINT_POSITIONS_QUESTIONS,
    _CHANGE_POINT_LARGEST_SHIFT_QUESTIONS,
    _STAT_VOLATILITY_CHANGE_QUESTIONS,
    _STAT_HALF_MEAN_DIFF_QUESTIONS,
    _STAT_MEDIAN_QUESTIONS,
    _STAT_WINDOWED_TREND_QUESTIONS,
)
from synth.align.templates.mts_questions import (
    _CROSS_STAT_STD_RATIO_QUESTIONS,
    _CROSS_STAT_RANGE_RATIO_QUESTIONS,
    _CROSS_HALF_SHIFT_QUESTIONS,
    _CROSS_HALF_VOLATILITY_QUESTIONS,
    _CROSS_WHICH_HAVE_LOCAL_QUESTIONS,
    _CROSS_WHICH_HAVE_TREND_TYPE_QUESTIONS,
    _CROSS_ALL_SAME_TREND_QUESTIONS,
    _CROSS_MOST_LOCAL_EVENTS_QUESTIONS,
    _CROSS_HIGHEST_AMPLITUDE_QUESTIONS,
    _CROSS_MOST_TREND_SEGMENTS_QUESTIONS,
    _CROSS_LONGEST_SEGMENT_QUESTIONS,
    _CROSS_NOISIEST_METRIC_QUESTIONS,
    _CROSS_QUIETEST_METRIC_QUESTIONS,
    _CROSS_WIDEST_RANGE_QUESTIONS,
    _CROSS_HIGHEST_MEAN_QUESTIONS,
    _CROSS_LOWEST_MEAN_QUESTIONS,
    _CROSS_HIGHEST_MAX_QUESTIONS,
    _CROSS_LOWEST_MIN_QUESTIONS,
    _CROSS_HIGHEST_STD_QUESTIONS,
    _CROSS_LOWEST_STD_QUESTIONS,
    _CROSS_EARLIEST_EVENT_QUESTIONS,
    _CROSS_LATEST_EVENT_QUESTIONS,
    _CROSS_LARGEST_GAP_QUESTIONS,
)
from synth.align.config import Config, Difficulty, PromptIndexer
from synth.align.templates import PromptRegistry

_NOISE_DISPLAY_ALIASES = [
    "noise",
    "noise pattern",
    "noise evolution",
    "noise behavior",
    "noise characteristics",
    "noise pattern evolution",
    "noise dynamics",
    "noise level",
]

# Each variant specifies which local event fields to expose in the CoT thinking
# and how to phrase the hint and answer. Type is always fundamental and included
# by the rendering code whenever 'type' key is in the stripped event dict.
_LOCAL_DESC_VARIANTS = [
    {
        'key': 'type_pos_start',
        'keep_fields': ('type',),
        'keep_kp_labels': ('start_context',),
        'keep_params': None,
        'hint': ' When describing local fluctuations, state the type and approximate starting position.',
    },
    {
        'key': 'type_pos_end',
        'keep_fields': ('type',),
        'keep_kp_labels': ('end_context',),
        'keep_params': None,
        'hint': ' When describing local fluctuations, state the type and approximate ending position.',
    },
    {
        'key': 'type_duration',
        'keep_fields': ('type',),
        'keep_kp_labels': ('start_context', 'end_context'),
        'keep_params': None,
        'hint': ' When describing local fluctuations, state the type and duration in timesteps.',
    },
    {
        'key': 'type_pos_both',
        'keep_fields': ('type',),
        'keep_kp_labels': ('start_context', 'end_context'),
        'keep_params': None,
        'hint': ' When describing local fluctuations, state the type, starting point, and ending point.',
    },
    {
        'key': 'type_amplitude',
        'keep_fields': ('type', 'amplitude'),
        'keep_params': None,
        'hint': ' When describing local fluctuations, state the type and amplitude.',
    },
    {
        'key': 'type_n_spikes',
        'keep_fields': ('type',),
        'keep_params': ('n_periods',),
        'hint': ' When describing local fluctuations, state the type and number of oscillation cycles.',
    },
]


class BaseQAGenerator(ABC):
    """Abstract base class for QA generators.
    
    Provides shared functionality for generating description and yes/no QA pairs
    that is common across UTS and MTS generators.
    """

    def __init__(
        self,
        config: Config,
        indexer: PromptIndexer,
        threshold: int = DEFAULT_THRESHOLD,
        threshold_provided: bool = False,
        seq_len: int = 0,
        difficulty: Difficulty = Difficulty.EASY,
    ):
        self.config = config
        self.indexer = indexer
        self.threshold = threshold
        self.threshold_provided = threshold_provided
        self.seq_len = seq_len
        self.difficulty = difficulty
        
        # Accumulated QA data
        self.questions: List[str] = []
        self.answers: List[str] = []
        self.llm_prompts: List[List[str]] = []
        self.fields: List[Dict] = []
        self.qa_types: List[str] = []
        self.eval_tasks: List[str] = []
        self.eval_metadatas: List[Dict[str, Any]] = []

    @abstractmethod
    def generate_all_qa(self) -> "GenerationResult":
        """Generate all QA pairs. Must be implemented by subclasses."""
        pass

    def _generate_taxonomy_qa(
        self,
        timeseries: np.ndarray,
        attributes: Dict,
        metric: str,
        series_index=None,
    ) -> None:
        """Generate QA pairs from the new taxonomy generators (A-G + Bridge).

        Runs all applicable generators (atomic SFT, RL compositions, bridge)
        for a single metric/timeseries, appending results to internal lists.
        """
        from synth.align.generators.atomic_qa import AtomicMeanGenerator
        from synth.align.generators.atomic_stat_qa import AtomicStatGenerator
        from synth.align.generators.atomic_trend_qa import AtomicTrendGenerator
        from synth.align.generators.atomic_event_qa import AtomicEventGenerator
        from synth.align.generators.atomic_periodic_qa import AtomicPeriodicGenerator
        from synth.align.generators.compositional_qa import CompositionalMeanGenerator
        from synth.align.generators.compositional_stat_qa import CompositionalStatGenerator
        from synth.align.generators.compositional_trend_qa import CompositionalTrendGenerator
        from synth.align.generators.compositional_event_qa import CompositionalEventGenerator
        from synth.align.generators.compositional_periodic_qa import CompositionalPeriodicGenerator
        from synth.align.generators.decomposition_bridge_qa import DecompositionBridgeGenerator

        seq_len = len(timeseries)
        trend_list = attributes.get('trend_list', [])
        local_events = attributes.get('local', [])
        seasonal = attributes.get('seasonal', {})

        field_key = 'series_index' if series_index is not None else 'timeseries'
        field_idx = series_index if series_index is not None else 0

        generators = [
            # SFT atoms (A-G)
            AtomicMeanGenerator(timeseries, metric, seq_len),
            AtomicStatGenerator(timeseries, metric, seq_len, trend_list),
            AtomicTrendGenerator(timeseries, metric, seq_len, trend_list),
            AtomicEventGenerator(timeseries, metric, seq_len, local_events),
            AtomicPeriodicGenerator(timeseries, metric, seq_len, seasonal),
            # RL compositions
            CompositionalMeanGenerator(timeseries, metric, seq_len),
            CompositionalStatGenerator(timeseries, metric, seq_len),
            CompositionalTrendGenerator(timeseries, metric, seq_len, trend_list),
            CompositionalEventGenerator(timeseries, metric, seq_len, local_events, trend_list),
            CompositionalPeriodicGenerator(timeseries, metric, seq_len, seasonal),
            # Decomposition bridge (force-SFT)
            DecompositionBridgeGenerator(
                timeseries, metric, seq_len, trend_list, local_events, seasonal,
            ),
        ]

        for gen in generators:
            for r in gen.generate_all():
                self.questions.append(r['question'])
                self.answers.append(r['answer'])
                self.llm_prompts.append([])
                self.fields.append({field_key: [field_idx]})
                self.qa_types.append(_classify_taxonomy_qa(r))
                self.eval_tasks.append(r['eval_type'])
                self.eval_metadatas.append(r['eval_metadata'])

    def _generate_cross_metric_taxonomy_qa(
        self,
        timeseries_list: List[np.ndarray],
        metrics: List[str],
        attributes_list: List[Dict],
        seq_len: int,
    ) -> None:
        """Generate cross-metric taxonomy QAs (Category H).

        Called ONCE per MTS sample (after per-metric loop), not per-metric.
        Runs atomic SFT, RL compositions, and bridge generators across all metrics.
        """
        if len(metrics) < 2:
            return

        from synth.align.generators.atomic_cross_metric_qa import AtomicCrossMetricGenerator
        from synth.align.generators.compositional_cross_metric_qa import CompositionalCrossMetricGenerator
        from synth.align.generators.bridge_cross_metric_qa import BridgeCrossMetricGenerator

        generators = [
            AtomicCrossMetricGenerator(timeseries_list, metrics, attributes_list, seq_len),
            CompositionalCrossMetricGenerator(timeseries_list, metrics, attributes_list, seq_len),
            BridgeCrossMetricGenerator(timeseries_list, metrics, attributes_list, seq_len),
        ]

        for gen in generators:
            for r in gen.generate_all():
                self.questions.append(r['question'])
                self.answers.append(r['answer'])
                self.llm_prompts.append([])
                self.fields.append({'series_index': list(range(len(metrics)))})
                self.qa_types.append(_classify_taxonomy_qa(r))
                self.eval_tasks.append(r['eval_type'])
                self.eval_metadatas.append(r['eval_metadata'])

    def _select_random_perspectives(self, attributes: Dict) -> List[str]:
        """Select random perspectives for analysis based on available attributes.
        
        Args:
            attributes: Attribute dictionary for a single time series
            
        Returns:
            List of selected perspective names
        """
        available = ["periodicity", "trend", "noise"]
        if attributes.get("local"):
            available.append("local events")

        n = len(available)
        size_weights = {str(i): float(i) for i in range(1, n + 1)}
        subset_size = int(weighted_random_choice(size_weights))
        selected = random.sample(available, subset_size)
        
        return selected

    def _build_thought_includes(self, selected_perspectives: List[str]) -> List[str]:
        """Build deduplicated list of thought block includes from perspectives.
        
        Args:
            selected_perspectives: List of perspective names
            
        Returns:
            Deduplicated list of attribute names for thought block
        """
        _MAP = {
            "periodicity": "periodicity",
            "trend": "trend",
            "local events": "local",
            "noise": "noise",
        }
        return list(dict.fromkeys(
            _MAP[p] for p in selected_perspectives if p in _MAP
        ))

    def _build_description_question(
        self,
        metric: str,
        selected_perspectives: List[str],
        local_variant: Optional[Dict] = None,
    ) -> str:
        """Build description question text using template registry.

        Args:
            metric: Name of the metric
            selected_perspectives: List of perspectives to analyze
            local_variant: Local description variant dict (from _LOCAL_DESC_VARIANTS);
                           if provided, its hint is used instead of the default local hint.

        Returns:
            Formatted question string
        """
        display_perspectives = []
        for p in selected_perspectives:
            if p == "noise":
                display_perspectives.append(random.choice(_NOISE_DISPLAY_ALIASES))
            else:
                display_perspectives.append(p)
        perspectives_str = format_list_natural_language(display_perspectives)
        question = PromptRegistry.get_prompt(
            QAType.DESCRIPTION,
            context={"metric": metric, "perspectives": perspectives_str},
            augment=True,
        )

        if "local events" in selected_perspectives:
            if local_variant is not None:
                question += local_variant['hint']
            else:
                question += PromptRegistry.get_local_hint()

        return question

    def _select_local_description_variant(self, local_chars: List[Dict]) -> Dict:
        """Pick a local description variant applicable to all events.

        Returns a variant from _LOCAL_DESC_VARIANTS where every local event has
        the required fields. Falls back to 'type_pos_start' if none qualifies.
        """
        def is_applicable(v: Dict) -> bool:
            key = v['key']
            for ev in local_chars:
                if key == 'type_pos_end' and ev.get('position_end') is None:
                    return False
                if key in ('type_duration', 'type_pos_both') and ev.get('position_end') is None:
                    return False
                if key == 'type_amplitude' and ev.get('amplitude') is None:
                    return False
                if key == 'type_n_spikes' and not ev.get('params', {}).get('n_periods'):
                    return False
            return True

        applicable = [v for v in _LOCAL_DESC_VARIANTS if is_applicable(v)]
        return random.choice(applicable) if applicable else _LOCAL_DESC_VARIANTS[0]

    def _strip_event_for_variant(self, ev: Dict, variant: Dict) -> Dict:
        """Return a stripped copy of a local event keeping only variant-relevant fields.

        The result is passed to build_local_verification_thought so the CoT only exposes
        the fields that are actually asked about.
        """
        keep = set(variant['keep_fields'])
        result = {k: v for k, v in ev.items() if k in keep}

        # Keep relevant key_points for position display as start@(idx, val) / end@(idx, val)
        keep_kp_labels = variant.get('keep_kp_labels')
        if keep_kp_labels:
            kp_list = ev.get('key_points_resolved') or ev.get('key_points_intent')
            if kp_list:
                filtered_kp = [kp for kp in kp_list if kp.get('label') in keep_kp_labels]
                if filtered_kp:
                    kp_key = 'key_points_resolved' if ev.get('key_points_resolved') else 'key_points_intent'
                    result[kp_key] = filtered_kp

        keep_params = variant.get('keep_params')
        if keep_params and ev.get('params'):
            filtered = {k: v for k, v in ev['params'].items() if k in keep_params}
            if filtered:
                result['params'] = filtered
        return result

    def _format_event_for_variant(self, ev: Dict, variant: Dict) -> str:
        """Format a local event's answer text based on the chosen variant."""
        key = variant['key']
        t = ev.get('type', 'unknown')
        pos_start = ev.get('position_start', 0)
        pos_end = ev.get('position_end')
        if key == 'type_pos_start':
            return f"{t}, position around point {pos_start}"
        if key == 'type_pos_end':
            return f"{t}, ends around point {pos_end}"
        if key == 'type_duration':
            dur = (pos_end - pos_start) if pos_end is not None else 0
            return f"{t}, duration {dur} steps"
        if key == 'type_pos_both':
            return f"{t}, from point {pos_start} to point {pos_end}"
        if key == 'type_amplitude':
            return f"{t}, amplitude {format_float(ev['amplitude'], 2)}"
        if key == 'type_n_spikes':
            n = int(ev.get('params', {}).get('n_periods', 1))
            return f"{t}, {n} oscillation cycles"
        # fallback
        return f"{t}, position around point {pos_start}"

    def _build_description_answer_parts(
        self,
        timeseries: np.ndarray,
        attributes: Dict,
        metric: str,
        situation: str,
        selected_perspectives: List[str],
        series_index: Optional[int] = None,
        include_thought: bool = True,
        local_variant: Optional[Dict] = None,
    ) -> Tuple[str, List[str], Dict]:
        """Build answer parts, LLM prompts, and fields for description QA.

        Args:
            timeseries: Time series data
            attributes: Attribute dictionary
            metric: Metric name
            situation: Category/situation name
            selected_perspectives: List of perspectives to include
            series_index: Index of series in MTS (None for UTS)
            include_thought: Whether to prepend a <think> block
            local_variant: Local description variant from _LOCAL_DESC_VARIANTS;
                           strips local event fields in thinking to match what is asked.

        Returns:
            Tuple of (full_answer, llm_prompts, fields)
        """
        thought_block = ""
        if include_thought:
            thought_includes = self._build_thought_includes(selected_perspectives)
            # For local events, build a stripped copy of attrs so the CoT
            # only exposes the fields relevant to this question variant.
            if local_variant is not None and 'local events' in selected_perspectives:
                local_chars = attributes.get('local', [])
                stripped_attrs = dict(attributes)
                stripped_attrs['local'] = [
                    self._strip_event_for_variant(ev, local_variant) for ev in local_chars
                ]
            else:
                stripped_attrs = attributes
            tb = ThoughtBuilder()
            tb.metric(metric, stripped_attrs, thought_includes)
            thought_block = tb.build()[0]

        answer_parts = []
        llm_prompts = []
        fields = {}

        # Determine field index
        field_idx = series_index if series_index is not None else 0

        # Build answer parts in the same order as selected_perspectives
        for p in selected_perspectives:
            if p == "periodicity":
                attr_text = attribute_to_text(
                    timeseries, attributes,
                    generate_values=False,
                    include_attributes=["periodicity", "frequency"]
                )
                placeholder = self.indexer.placeholder()
                answer_parts.append(f"{attr_text} {placeholder}")
                fields["seasonal"] = [field_idx]
                llm_prompts.append(
                    f"There is a metric called {metric} collected from {situation} "
                    f"with length of {self.seq_len}. The periodicity of this metric is as follow: "
                    + attribute_to_text(
                        timeseries, attributes,
                        generate_values=False,
                        include_attributes=["periodicity"]
                    )
                    + " Please analyze the physical meaning of this kind of periodicity in one sentence."
                )

            elif p == "trend":
                attr_text = attribute_to_text(
                    timeseries, attributes,
                    generate_values=False,
                    include_attributes=["trend"]
                )
                placeholder = self.indexer.placeholder()
                answer_parts.append(f"{attr_text} {placeholder}")
                fields["trend"] = [field_idx]
                trend_type = attributes.get('trend', {}).get('type', 'unknown')
                llm_prompts.append(
                    f"There is a metric called {metric} collected from {situation} "
                    f"with length of {self.seq_len}. The trend of this metric is {trend_type}. "
                    f"Please analyze the physical meaning of this kind of trend in one sentence."
                )

            elif p == "noise":
                attr_text = attribute_to_text(
                    timeseries, attributes,
                    generate_values=False,
                    include_attributes=["noise"]
                )
                placeholder = self.indexer.placeholder()
                answer_parts.append(f"{attr_text} {placeholder}")
                fields["noise"] = [field_idx]
                noise_level = attributes.get('noise', {}).get('type', 'unknown')
                llm_prompts.append(
                    f"There is a metric called {metric} collected from {situation} "
                    f"with length of {self.seq_len}. The noise level of this metric is {noise_level}. "
                    f"Please analyze the physical meaning of this noise level in one sentence."
                )

            elif p == "local events":
                local_chars = attributes.get("local", [])
                if local_chars:
                    fields["local"] = [field_idx]
                    local_texts = []
                    for local_char in local_chars:
                        placeholder = self.indexer.placeholder()
                        if local_variant is not None:
                            answer_text = self._format_event_for_variant(local_char, local_variant)
                        else:
                            answer_text = (
                                f"{local_char['type']}, position around point "
                                f"{local_char['position_start']}, "
                                f"amplitude {format_float(local_char['amplitude'], 2)}"
                            )
                        local_texts.append(f"{answer_text}. {placeholder}")
                        llm_prompts.append(
                            f"There is a metric called {metric} collected from {situation} "
                            f"with length of {self.seq_len}. A local fluctuation found is {local_char['type']}. "
                            f"Please analyze the physical meaning of this fluctuation in one sentence."
                        )
                    answer_parts.append("; ".join(local_texts))

        full_answer = thought_block + " ".join(answer_parts)
        return full_answer, llm_prompts, fields

    def _generate_description_qa(
        self,
        timeseries: np.ndarray,
        attributes: Dict,
        metric: str,
        situation: str,
        series_index: Optional[int] = None
    ) -> None:
        """Generate a description QA for a single time series."""
        selected = self._select_random_perspectives(attributes)

        # Pick a local description variant when local events are in scope.
        local_variant = None
        if "local events" in selected and attributes.get("local"):
            local_variant = self._select_local_description_variant(attributes["local"])

        question = self._build_description_question(metric, selected, local_variant=local_variant)
        answer, llm_prompts, fields = self._build_description_answer_parts(
            timeseries, attributes, metric, situation, selected, series_index,
            local_variant=local_variant,
        )
        
        self.questions.append(_fix_article(question))
        self.answers.append(answer)
        self.llm_prompts.append(llm_prompts)
        self.fields.append(fields)
        self.qa_types.append(QAType.DESCRIPTION)
        self.eval_tasks.append(QAType.DESCRIPTION)
        self.eval_metadatas.append({"perspectives": selected, "length": self.seq_len})

    def _generate_compound_description_qa(
        self,
        timeseries_list: List[np.ndarray],
        attributes_list: List[Dict],
        metrics: List[str],
        situation: str,
        series_indices: List[int],
    ) -> None:
        """Generate a compound description QA spanning multiple metrics.

        Asks about different descriptive aspects of different metrics in one question,
        e.g. "Describe the periodicity of metric_A, and the noise and trend of metric_B."
        """
        n = len(metrics)
        if n < 2:
            return

        k = random.randint(2, min(n, 4))
        chosen_idx = random.sample(range(n), k)

        chosen_metrics = [metrics[i] for i in chosen_idx]
        chosen_ts = [timeseries_list[i] for i in chosen_idx]
        chosen_attrs = [attributes_list[i] for i in chosen_idx]
        chosen_series_indices = [series_indices[i] for i in chosen_idx]

        # Per-metric perspective selection
        perspectives_per_metric: List[List[str]] = []
        for attrs in chosen_attrs:
            perspectives_per_metric.append(self._select_random_perspectives(attrs))

        # Per-metric thought includes (for single <think> block)
        thought_includes_per_metric = [
            self._build_thought_includes(p) for p in perspectives_per_metric
        ]

        tb = ThoughtBuilder()
        for i, (name, attr) in enumerate(zip(chosen_metrics, chosen_attrs)):
            if i > 0:
                tb.metric_separator()
            tb.metric(name, attr or {}, thought_includes_per_metric[i])
        thought_block = tb.build()[0]

        # Build question
        all_same_perspectives = all(
            sorted(p) == sorted(perspectives_per_metric[0]) for p in perspectives_per_metric
        )
        if all_same_perspectives:
            p_str = format_list_natural_language(perspectives_per_metric[0])
            metrics_str = format_list_natural_language(chosen_metrics)
            template = random.choice(COMPOUND_DESCRIPTION_UNIFORM_TEMPLATES)
            question = template.format(perspectives=p_str, metrics=metrics_str)
        else:
            parts = []
            for metric, perspectives in zip(chosen_metrics, perspectives_per_metric):
                p_str = format_list_natural_language(perspectives)
                parts.append(f"the {p_str} of {metric}")
            parts_str = format_list_natural_language(parts)
            template = random.choice(COMPOUND_DESCRIPTION_MIXED_TEMPLATES)
            question = template.format(parts=parts_str)

        # Build per-metric answer sections (no individual thought blocks)
        all_answer_parts: List[str] = []
        all_llm_prompts: List[str] = []
        combined_fields: Dict = {}

        for ts, attrs, metric, si, perspectives in zip(
            chosen_ts, chosen_attrs, chosen_metrics, chosen_series_indices, perspectives_per_metric
        ):
            ans_text, llm_prompts, fields = self._build_description_answer_parts(
                ts, attrs, metric, situation, perspectives, si, include_thought=False
            )
            if ans_text.strip():
                all_answer_parts.append(f"{metric}: {ans_text}")
            all_llm_prompts.extend(llm_prompts)
            for key, val in fields.items():
                combined_fields.setdefault(key, []).extend(val)

        full_answer = thought_block + " ".join(all_answer_parts)
        eval_metadata = {
            "perspectives": {m: p for m, p in zip(chosen_metrics, perspectives_per_metric)},
            "length": self.seq_len,
        }

        self.questions.append(_fix_article(question))
        self.answers.append(full_answer)
        self.llm_prompts.append(all_llm_prompts)
        self.fields.append(combined_fields)
        self.qa_types.append(QAType.DESCRIPTION)
        self.eval_tasks.append(QAType.DESCRIPTION)
        self.eval_metadatas.append(eval_metadata)

    @staticmethod
    def _yes_offset_range(difficulty: Difficulty, threshold: int) -> Tuple[int, int]:
        """Return (min_offset, max_offset) for want_yes placement at given difficulty."""
        if difficulty == Difficulty.EASY:
            hi = max(threshold // 3, 1)
            return (0, hi)
        elif difficulty == Difficulty.MEDIUM:
            lo = max(threshold * 2 // 3, 1)
            return (lo, threshold)
        else:  # HARD
            lo = max(threshold * 3 // 4, 1)
            return (lo, threshold)

    @staticmethod
    def _no_distance_range(difficulty: Difficulty, threshold: int) -> Tuple[int, Optional[int]]:
        """Return (min_distance, max_distance) for want_no placement at given difficulty."""
        if difficulty == Difficulty.EASY:
            return (threshold * 3, None)
        elif difficulty == Difficulty.MEDIUM:
            return (threshold, threshold * 2)
        else:  # HARD
            return (threshold, threshold + max(3, threshold // 4))

    _FALLBACK_ORDER = {
        Difficulty.HARD: [Difficulty.MEDIUM, Difficulty.EASY],
        Difficulty.MEDIUM: [Difficulty.EASY],
        Difficulty.EASY: [],
    }

    def _select_yes_no_query_point(
        self,
        local_positions: List[int],
        has_events: bool,
        want_yes: bool,
    ) -> Optional[int]:
        """Select query point for yes/no question with difficulty-aware placement.

        Difficulty controls how close the query point is to the threshold boundary.
        Falls back through easier difficulties if the desired placement is impossible.

        Returns:
            Query point or None if the desired verdict is unachievable.
        """
        if want_yes:
            if not has_events:
                return None
            event_pos = random.choice(local_positions)
            # Try difficulty levels from requested down to EASY
            for diff in [self.difficulty] + self._FALLBACK_ORDER[self.difficulty]:
                lo, hi = self._yes_offset_range(diff, self.threshold)
                if lo > hi:
                    continue
                offset = random.randint(lo, hi) * random.choice([-1, 1])
                point = max(0, min(event_pos + offset, self.seq_len - 1))
                return point
            # Absolute fallback (shouldn't reach here)
            return max(0, min(
                event_pos + random.randint(-self.threshold, self.threshold),
                self.seq_len - 1,
            ))
        else:
            if not has_events:
                return random.randint(0, self.seq_len - 1)
            # Try difficulty levels from requested down to EASY
            for diff in [self.difficulty] + self._FALLBACK_ORDER[self.difficulty]:
                min_dist, max_dist = self._no_distance_range(diff, self.threshold)
                point = find_point_at_distance_range(
                    local_positions, self.seq_len, min_dist, max_dist
                )
                if point is not None:
                    return point
            # Final fallback: any point >= threshold away
            return find_point_far_from_all_events(
                local_positions, self.seq_len, self.threshold
            )

    def _build_yes_no_answer(
        self,
        attributes: Dict,
        metric: str,
        query_point: int,
        thought_block: str
    ) -> str:
        """Build answer for yes/no question about local events.
        
        Args:
            attributes: Attribute dictionary
            metric: Metric name
            query_point: Point being queried
            thought_block: Pre-built thought block
            
        Returns:
            Formatted answer string
        """
        event_type = has_local_event_near(attributes, query_point, self.threshold)

        verdict = "yes" if event_type else "no"
        thought_block = thought_block.replace("\n</think>", f"\nanswer: {verdict}\n</think>", 1)

        if event_type:
            answer = f"{thought_block}Yes, {metric} has a fluctuation around point {query_point}."
        else:
            answer = (
                f"{thought_block}I did not find any local event "
                f"starting around point {query_point} in {metric}."
            )

        return answer

    def _generate_single_yes_no_qa(
        self,
        attributes: Dict,
        metric: str,
        query_point: int,
        series_index: Optional[int] = None
    ) -> None:
        """Generate a single yes/no QA about a local event.
        
        Args:
            attributes: Attribute dictionary
            metric: Metric name
            query_point: Point to query about
            series_index: Index of series in MTS (None for UTS)
        """
        tolerance_text = (
            f" within the tolerance of {self.threshold} timestep{'s' if self.threshold != 1 else ''}"
            if self.threshold_provided else ""
        )
        
        question = PromptRegistry.get_prompt(
            QAType.YES_NO,
            context={
                "point": query_point,
                "metric": metric,
                "tolerance": tolerance_text,
            },
            augment=True,
        )
        
        sparse_attr = sanitize_attributes_for_sync(attributes)
        # Strip position_end — start-pos questions only need the starting position.
        for ev in sparse_attr.get('local', []):
            ev.pop('position_end', None)
        thought, _ = build_local_verification_thought(
            [sparse_attr],
            [metric],
            include_attributes=['local'],
            anchor_point=query_point,
            mode='local',
            threshold=self.threshold,
            threshold_provided=self.threshold_provided,
        )

        answer = self._build_yes_no_answer(attributes, metric, query_point, thought)

        # Determine verdict for metadata (same logic as _build_yes_no_answer)
        event_type = has_local_event_near(attributes, query_point, self.threshold)
        yn_verdict = "yes" if event_type else "no"

        field_idx = series_index if series_index is not None else 0

        self.questions.append(_fix_article(question))
        self.answers.append(answer)
        self.llm_prompts.append([])
        self.fields.append({"local": [field_idx]})
        self.qa_types.append(QAType.YES_NO)
        self.eval_tasks.append(QAType.YES_NO)
        self.eval_metadatas.append({
            "target_point": query_point,
            "metric": metric,
            "threshold": self.threshold,
            "variant": "generic",
            "verdict": yn_verdict,
            "length": self.seq_len,
        })

    # ------------------------------------------------------------------ #
    # Helpers for compound judgment (Family 7)                             #
    # ------------------------------------------------------------------ #

    def _generate_compound_judgment_qa(
        self,
        attributes: Dict,
        metric: str,
        series_index,
    ) -> None:
        """Family 7: Compound judgment combining 2+ attribute conditions."""
        import math as _math

        field_key = 'series_index' if series_index is not None else 'timeseries'
        field_idx = series_index if series_index is not None else 0

        local_events = attributes.get('local', [])
        trend        = attributes.get('trend', {})
        trend_list   = attributes.get('trend_list', [])
        noise        = attributes.get('noise', {})
        statistics   = attributes.get('statistics') or {}

        actual_trend_type = trend.get('type', 'keep steady')
        all_trend_types   = ['increase', 'decrease', 'keep steady']

        # ---- gather candidates ----
        candidates = []
        if noise.get('strength', 0) > 0:
            candidates.append('noise_trend')
        if local_events:
            candidates.append('trend_local')
        if len(trend_list) >= 2:
            candidates.append('multi_phase')
        if statistics.get('min') is not None and statistics.get('max') is not None:
            candidates.append('stat_threshold')
        # ---- cross-attribute candidates (A1–D1) ----
        noise_strength = noise.get('strength', 0.0)
        if local_events and noise_strength > 0:
            candidates.append('amplitude_vs_noise')
        if local_events and statistics.get('std', 0) > 0:
            candidates.append('amplitude_vs_std')
        if statistics.get('std', 0) > 0:
            candidates.append('range_vs_std')
        if statistics.get('first_half_std', 0) > 0:
            candidates.append('half_volatility')
        if statistics.get('first_half_mean') is not None and statistics.get('max') != statistics.get('min'):
            candidates.append('half_mean_shift')
        win16 = statistics.get('segment_means_16', [])
        if len(win16) >= 3:
            candidates.append('windowed_monotonicity')
        if local_events and len(trend_list) >= 2:
            candidates.append('phase_event')
        up_evts = [e for e in local_events if e.get('type') in _UPWARD_EVENT_TYPES]
        down_evts = [e for e in local_events if e.get('type') in _DOWNWARD_EVENT_TYPES]
        if len(local_events) >= 2 and up_evts and down_evts:
            candidates.append('sequential_events')

        if self.config.debug:
            chosen = list(candidates)
        else:
            chosen = random.sample(candidates, min(2, len(candidates)))

        _EVENT_LABELS_UP   = ['surge event', 'peak anomaly', 'upward spike event', 'load spike']
        _EVENT_LABELS_DOWN = ['crash event', 'dip anomaly', 'downward drop event', 'critical drop']
        _PATTERN_LABELS    = ['growth cycle', 'recovery pattern', 'load cycle', 'demand cycle']
        _NOISE_LABELS      = ['unstable state', 'high-noise regime', 'signal degradation', 'noisy period']
        _AMP_NOISE_LABELS  = ['significant anomaly', 'noise-relative spike', 'detectable disturbance', 'signal breakthrough']
        _AMP_STD_LABELS    = ['outlier event', 'statistical anomaly', 'deviation spike', 'extreme fluctuation']
        _RANGE_STD_LABELS  = ['critical range event', 'range anomaly', 'spread alert', 'high-dispersion signal']
        _HALF_VOL_LABELS   = ['volatility escalation', 'instability growth', 'variance surge', 'increasing turbulence']
        _HALF_SHIFT_LABELS = ['significant upward shift', 'upward level shift', 'mean elevation', 'progressive increase']
        _MONO_UP_LABELS    = ['sustained upward drift', 'consistent ramp-up', 'monotonic ascent', 'progressive buildup']
        _MONO_DOWN_LABELS  = ['sustained downward drift', 'consistent decline', 'monotonic descent', 'progressive drawdown']
        _PHASE_EVT_LABELS  = ['recovery signal', 'counter-trend event', 'phase anomaly', 'opposing fluctuation']
        _SEQ_EVT_LABELS    = ['fault pattern', 'paired anomaly', 'sequential disturbance', 'cascade event']

        for jtype in chosen:

            # ============================================================
            if jtype == 'trend_local':
                direction = random.choice(['upward', 'downward'])
                dir_set   = _UPWARD_EVENT_TYPES if direction == 'upward' else _DOWNWARD_EVENT_TYPES
                dir_evts  = [(e['type'], e['position_start'], e.get('amplitude',0.0))
                             for e in local_events if e.get('type') in dir_set]

                # Single coin flip for desired verdict
                desired_yes = random.random() < 0.5

                if desired_yes:
                    if not dir_evts:
                        continue  # Can't produce "yes" without events
                    max_amp = max(a for _, _, a in dir_evts)
                    threshold = round(max_amp * random.uniform(0.3, 0.85), 2)
                    req_trend = actual_trend_type
                else:
                    # "No" via either path: wrong trend OR high threshold
                    if random.random() < 0.5:
                        req_trend = random.choice([t for t in all_trend_types if t != actual_trend_type])
                        if dir_evts:
                            max_amp = max(a for _, _, a in dir_evts)
                            threshold = round(max_amp * random.uniform(0.3, 0.85), 2)
                        else:
                            threshold = round(statistics.get('range', 1.0) * random.uniform(0.1, 0.5) + 0.01, 2)
                    else:
                        req_trend = actual_trend_type
                        if dir_evts:
                            max_amp = max(a for _, _, a in dir_evts)
                            threshold = round(max_amp * random.uniform(1.1, 2.5), 2)
                        else:
                            threshold = round(statistics.get('range', 1.0) * random.uniform(0.1, 0.5) + 0.01, 2)

                qual_evts = [(t, p, a) for t, p, a in dir_evts if round(a, 2) > threshold]
                verdict = 'yes' if (req_trend == actual_trend_type and len(qual_evts) > 0) else 'no'
                label   = random.choice(_EVENT_LABELS_UP if direction == 'upward' else _EVENT_LABELS_DOWN)

                all_evts = [
                    (e['type'], e.get('amplitude', 0.0),
                     'upward' if e.get('type') in _UPWARD_EVENT_TYPES else
                     ('downward' if e.get('type') in _DOWNWARD_EVENT_TYPES else 'none'))
                    for e in local_events
                ]
                params = dict(
                    event_direction='upward' if direction == 'upward' else 'downward',
                    event_label=label,
                    threshold=threshold,
                    required_trend=req_trend,
                    actual_trend=actual_trend_type,
                    qualifying_events=qual_evts,
                    all_direction_events=dir_evts,
                    all_events=all_evts,
                    verdict=verdict,
                )
                question = random.choice(_JUDGMENT_TREND_LOCAL_QUESTIONS).format(
                    metric=metric, label=label, direction=direction,
                    threshold=threshold, req_trend=req_trend,
                )

            # ============================================================
            elif jtype == 'multi_phase':
                actual_seq = [t for t, _, _ in trend_list]
                n_phases   = random.randint(2, min(3, len(actual_seq)))
                label      = random.choice(_PATTERN_LABELS)

                if random.random() < 0.50:
                    target_seq = actual_seq[:n_phases]      # -> yes
                else:
                    # generate wrong sequence (swap two types)
                    target_seq = list(actual_seq[:n_phases])
                    wrong_type = random.choice([t for t in all_trend_types if t != target_seq[0]])
                    target_seq[0] = wrong_type              # -> no (unless coincidence)

                match = actual_seq[:n_phases] == target_seq
                verdict = 'yes' if (len(actual_seq) >= n_phases and match) else 'no'
                phase_seq_str = ' -> '.join(target_seq)

                params = dict(
                    pattern_label=label,
                    target_sequence=target_seq,
                    actual_sequence=actual_seq,
                    trend_list=trend_list,
                    verdict=verdict,
                )
                question = random.choice(_JUDGMENT_MULTI_PHASE_QUESTIONS).format(
                    metric=metric, label=label, phase_seq=phase_seq_str,
                )

            # ============================================================
            elif jtype == 'noise_trend':
                act_n_amp  = noise['strength']
                act_n_type = noise['type']
                noise_cat  = 'noisy' if 'noisy' in act_n_type.lower() else 'smooth'
                label      = random.choice(_NOISE_LABELS)

                # Single coin flip for desired verdict
                desired_yes = random.random() < 0.5

                if desired_yes:
                    if act_n_amp == 0.0:
                        continue  # Can't produce "yes" with zero noise
                    threshold = round(act_n_amp * random.uniform(0.3, 0.85), 2)
                    req_trend = actual_trend_type
                else:
                    if random.random() < 0.5:
                        req_trend = random.choice([t for t in all_trend_types if t != actual_trend_type])
                        threshold = round(act_n_amp * random.uniform(0.3, 0.85), 2) if act_n_amp > 0 else 0.05
                    else:
                        req_trend = actual_trend_type
                        threshold = round(act_n_amp * random.uniform(1.1, 2.5) + 1e-8, 2)

                c2 = round(act_n_amp, 2) > threshold
                verdict = 'yes' if (req_trend == actual_trend_type and c2) else 'no'

                params = dict(
                    label=label,
                    noise_category=noise_cat,
                    threshold=threshold,
                    required_trend=req_trend,
                    actual_trend=actual_trend_type,
                    actual_noise_type=act_n_type,
                    actual_noise_amp=act_n_amp,
                    verdict=verdict,
                )
                question = random.choice(_JUDGMENT_NOISE_TREND_QUESTIONS).format(
                    metric=metric, label=label, noise_cat=noise_cat,
                    threshold=threshold, req_trend=req_trend,
                )

            # ============================================================
            elif jtype == 'stat_threshold':
                act_min = statistics['min']
                act_max = statistics['max']
                mean    = statistics['mean']
                std     = statistics['std']
                if std == 0.0:
                    continue  # constant signal — skip stat_threshold

                direction = random.choice(['below', 'above'])
                if direction == 'below':
                    # threshold around min ± spread; 50/50 yes/no
                    threshold = round(act_min + random.uniform(-0.5, 1.5) * std, 2)
                    verdict   = 'yes' if round(act_min, 2) < threshold else 'no'
                else:
                    threshold = round(act_max + random.uniform(-1.5, 0.5) * std, 2)
                    verdict   = 'yes' if round(act_max, 2) > threshold else 'no'

                params = dict(
                    direction=direction,
                    threshold=threshold,
                    actual_min=act_min,
                    actual_max=act_max,
                    verdict=verdict,
                )
                question = random.choice(_JUDGMENT_STAT_QUESTIONS).format(
                    metric=metric, direction=direction, threshold=threshold,
                )

            # ============================================================
            elif jtype in ('amplitude_vs_noise', 'amplitude_vs_std'):
                if jtype == 'amplitude_vs_noise':
                    ref_val = noise['strength']
                    K = random.choice([2, 3, 4, 5])
                else:
                    ref_val = statistics['std']
                    K = random.choice([1.5, 2, 2.5, 3])

                threshold_val = ref_val * K
                amps = [(e['type'], e.get('amplitude', 0.0),
                         'upward' if e.get('type') in _UPWARD_EVENT_TYPES else
                         ('downward' if e.get('type') in _DOWNWARD_EVENT_TYPES else 'none'))
                        for e in local_events]
                max_amp = max(a for _, a, _ in amps) if amps else 0.0

                # yes/no control: 50/50
                if random.random() < 0.50 and max_amp > 0:
                    # pick K so threshold < max_amp → yes
                    K = random.choice([k for k in ([2, 3, 4, 5] if jtype == 'amplitude_vs_noise' else [1.5, 2, 2.5, 3])
                                       if ref_val * k < max_amp] or [K])
                    threshold_val = ref_val * K
                else:
                    # pick K so threshold >= max_amp → no
                    K = random.choice([k for k in ([2, 3, 4, 5] if jtype == 'amplitude_vs_noise' else [1.5, 2, 2.5, 3])
                                       if ref_val * k >= max_amp] or [K])
                    threshold_val = ref_val * K

                qual_count = sum(1 for _, a, _ in amps if round(a, 2) > round(threshold_val, 2))
                verdict = 'yes' if qual_count > 0 else 'no'
                label = random.choice(_AMP_NOISE_LABELS if jtype == 'amplitude_vs_noise' else _AMP_STD_LABELS)

                params = dict(
                    K=K,
                    ref_val=ref_val,
                    threshold=threshold_val,
                    all_events=amps,
                    qualifying_count=qual_count,
                    verdict=verdict,
                )
                if jtype == 'amplitude_vs_noise':
                    question = random.choice(_JUDGMENT_AMPLITUDE_VS_NOISE_QUESTIONS).format(
                        metric=metric, label=label, K=K,
                        noise_strength=format_float(ref_val, 2),
                    )
                else:
                    question = random.choice(_JUDGMENT_AMPLITUDE_VS_STD_QUESTIONS).format(
                        metric=metric, label=label, K=K,
                        std=format_float(ref_val, 2),
                    )

            # ============================================================
            elif jtype == 'range_vs_std':
                act_max = statistics['max']
                act_min = statistics['min']
                act_std = statistics['std']
                if act_std == 0.0:
                    continue  # constant signal — skip range_vs_std
                rng = act_max - act_min
                actual_ratio = rng / act_std if act_std > 0 else 0.0

                K_options = [3, 4, 5, 6]
                if random.random() < 0.50:
                    K = random.choice([k for k in K_options if k < actual_ratio] or [K_options[0]])
                else:
                    K = random.choice([k for k in K_options if k >= actual_ratio] or [K_options[-1]])

                threshold_val = K * act_std
                verdict = 'yes' if round(rng, 2) > round(threshold_val, 2) else 'no'
                label = random.choice(_RANGE_STD_LABELS)

                params = dict(
                    K=K,
                    actual_max=act_max,
                    actual_min=act_min,
                    actual_std=act_std,
                    range=rng,
                    threshold=threshold_val,
                    verdict=verdict,
                )
                question = random.choice(_JUDGMENT_RANGE_VS_STD_QUESTIONS).format(
                    metric=metric, label=label, K=K,
                )

            # ============================================================
            elif jtype == 'half_volatility':
                fhs = statistics['first_half_std']
                shs = statistics['second_half_std']
                if fhs <= 0:
                    continue

                K_options = [1.3, 1.5, 2.0, 2.5]
                ratio_for_balance = shs / fhs
                if random.random() < 0.50:
                    K = random.choice([k for k in K_options if k <= ratio_for_balance] or [K_options[0]])
                else:
                    K = random.choice([k for k in K_options if k > ratio_for_balance] or [K_options[-1]])

                threshold = round(K * fhs, 2)
                verdict = 'yes' if round(shs, 2) >= threshold else 'no'
                label = random.choice(_HALF_VOL_LABELS)

                params = dict(
                    K=K,
                    first_half_std=fhs,
                    second_half_std=shs,
                    verdict=verdict,
                )
                question = random.choice(_JUDGMENT_HALF_VOLATILITY_QUESTIONS).format(
                    metric=metric, label=label, K=format_float(K, 2) if K != int(K) else K,
                )

            # ============================================================
            elif jtype == 'half_mean_shift':
                act_max = statistics['max']
                act_min = statistics['min']
                fhm = statistics['first_half_mean']
                shm = statistics['second_half_mean']
                rng = act_max - act_min
                shift = shm - fhm
                # Use absolute shift so negative shifts (second half < first half)
                # don't structurally prevent "yes" verdicts (mirrors cross_half_shift).
                abs_actual_pct = (abs(shift) / rng * 100) if rng > 0 else 0.0

                K_options = [5, 10, 15, 20, 25]
                if random.random() < 0.50:
                    K_pct = random.choice([k for k in K_options if k < abs_actual_pct] or [K_options[0]])
                else:
                    K_pct = random.choice([k for k in K_options if k >= abs_actual_pct] or [K_options[-1]])

                threshold_val = K_pct / 100.0 * rng
                # Match builder rounding: signed_shift=round(shm-fhm,2), abs=round(|signed|,2)
                shift_r = round(abs(round(shift, 2)), 2)
                verdict = 'yes' if shift_r > round(threshold_val, 2) else 'no'
                label = random.choice(_HALF_SHIFT_LABELS)

                params = dict(
                    K_pct=K_pct,
                    actual_max=act_max,
                    actual_min=act_min,
                    first_half_mean=fhm,
                    second_half_mean=shm,
                    range=rng,
                    shift=shift,
                    threshold=threshold_val,
                    verdict=verdict,
                )
                question = random.choice(_JUDGMENT_HALF_MEAN_SHIFT_QUESTIONS).format(
                    metric=metric, label=label, K=K_pct,
                )

            # ============================================================
            elif jtype == 'windowed_monotonicity':
                means = [round(m, 2) for m in win16]
                direction = random.choice(['higher', 'lower'])
                is_mono = True
                for i in range(1, len(means)):
                    if direction == 'higher' and means[i] <= means[i - 1]:
                        is_mono = False
                        break
                    if direction == 'lower' and means[i] >= means[i - 1]:
                        is_mono = False
                        break

                verdict = 'yes' if is_mono else 'no'
                label = random.choice(_MONO_UP_LABELS if direction == 'higher' else _MONO_DOWN_LABELS)

                params = dict(
                    means=means,
                    direction=direction,
                    seq_len=self.seq_len,
                    verdict=verdict,
                )
                question = random.choice(_JUDGMENT_WINDOWED_MONOTONICITY_QUESTIONS).format(
                    metric=metric, label=label, direction=direction,
                )

            # ============================================================
            elif jtype == 'phase_event':
                direction = random.choice(['upward', 'downward'])
                dir_set = _UPWARD_EVENT_TYPES if direction == 'upward' else _DOWNWARD_EVENT_TYPES
                seg_types = list({t for t, _, _ in trend_list})

                # all events with direction info and 0-indexed position
                all_evts = [
                    (e['type'], e['position_start'],
                     'upward' if e.get('type') in _UPWARD_EVENT_TYPES else
                     ('downward' if e.get('type') in _DOWNWARD_EVENT_TYPES else 'none'))
                    for e in local_events
                ]

                # 0-indexed trend segments (already 0-indexed from trend_utils)
                tl_1idx = list(trend_list)

                # find actual matches: events with correct direction that fall in a matching phase
                matches = []
                for e_type, e_pos, e_dir in all_evts:
                    if e_dir != direction:
                        continue
                    for seg_type, seg_s, seg_e in tl_1idx:
                        if seg_s <= e_pos <= seg_e:
                            matches.append((e_type, e_pos, seg_type))
                            break

                if random.random() < 0.50 and matches:
                    # pick a trend type that has a match → yes
                    match_types = list({st for _, _, st in matches})
                    req_trend_type = random.choice(match_types)
                    verdict = 'yes'
                else:
                    # pick a trend type with no match → no
                    match_types = {st for _, _, st in matches}
                    no_match_types = [t for t in seg_types if t not in match_types]
                    if no_match_types:
                        req_trend_type = random.choice(no_match_types)
                        verdict = 'no'
                    elif seg_types:
                        req_trend_type = random.choice(seg_types)
                        verdict = 'yes' if any(st == req_trend_type for _, _, st in matches) else 'no'
                    else:
                        continue

                label = random.choice(_PHASE_EVT_LABELS)
                params = dict(
                    event_direction=direction,
                    required_trend_type=req_trend_type,
                    all_events=all_evts,
                    trend_list=tl_1idx,
                    verdict=verdict,
                )
                question = random.choice(_JUDGMENT_PHASE_EVENT_QUESTIONS).format(
                    metric=metric, label=label, direction=direction,
                    trend_type=req_trend_type,
                )

            # ============================================================
            elif jtype == 'sequential_events':
                # pick event_a (upward family) and event_b (downward family) or vice versa
                if random.random() < 0.5:
                    pool_a, pool_b = up_evts, down_evts
                    dir_a, dir_b = 'upward', 'downward'
                else:
                    pool_a, pool_b = down_evts, up_evts
                    dir_a, dir_b = 'downward', 'upward'

                # build ordered pairs (a before b)
                pairs = []
                for ea in pool_a:
                    for eb in pool_b:
                        pos_a = ea['position_start']
                        pos_b = eb['position_start']
                        if pos_a < pos_b:
                            pairs.append((ea['type'], pos_a, eb['type'], pos_b))

                if not pairs:
                    continue

                gaps = [pos_b - pos_a for _, pos_a, _, pos_b in pairs]
                min_gap = min(gaps)

                gap_options = [10, 15, 20, 25, 30]
                if random.random() < 0.50:
                    gap_limit = random.choice([g for g in gap_options if g >= min_gap] or [gap_options[-1]])
                else:
                    gap_limit = random.choice([g for g in gap_options if g < min_gap] or [gap_options[0]])

                qualifying = [(ta, pa, tb, pb) for ta, pa, tb, pb in pairs if pb - pa <= gap_limit]
                verdict = 'yes' if qualifying else 'no'

                # pick representative type names for the question
                event_a_name = f"{dir_a} event"
                event_b_name = f"{dir_b} event"

                label = random.choice(_SEQ_EVT_LABELS)
                params = dict(
                    pairs=qualifying if qualifying else pairs[:3],
                    gap_limit=gap_limit,
                    event_a_type=event_a_name,
                    event_b_type=event_b_name,
                    verdict=verdict,
                )
                question = random.choice(_JUDGMENT_SEQUENTIAL_EVENTS_QUESTIONS).format(
                    metric=metric, label=label,
                    event_a=event_a_name, event_b=event_b_name,
                    gap=gap_limit,
                )

            else:
                continue

            try:
                think_str, verdict = build_compound_judgment_thought(
                    attributes, metric, jtype, params
                )
            except Exception:
                logger.warning("QA generation failed", exc_info=True)
                continue

            answer_text = self._compound_judgment_answer_text(jtype, verdict, params, metric)

            self.questions.append(_fix_article(question))
            self.answers.append(think_str + answer_text)
            self.llm_prompts.append([])
            self.fields.append({field_key: [field_idx]})
            self.qa_types.append(QAType.SEGMENT_MASK)
            self.eval_tasks.append("segment_judgment")
            self.eval_metadatas.append({
                "judgment_type": jtype,
                "verdict": verdict,
                "length": self.seq_len,
            })

    @staticmethod
    def _compound_judgment_answer_text(jtype: str, verdict: str, params: dict, metric: str) -> str:
        """Build grounded answer text for compound judgment QA, citing key computed values."""
        yes = (verdict == 'yes')

        if jtype == 'trend_local':
            trend = params['actual_trend']
            direction = params['event_direction']
            if yes:
                return f"Yes. {metric} has {article(trend)} {trend} overall trend with qualifying {direction} events."
            return f"No. {metric} does not jointly satisfy the trend and amplitude conditions."

        if jtype == 'multi_phase':
            seq = params['target_sequence']
            seq_str = ' → '.join(seq)
            if yes:
                return f"Yes. {metric}'s trend phases begin with {seq_str}."
            return f"No. {metric}'s trend phases do not match the required sequence {seq_str}."

        if jtype == 'noise_trend':
            trend = params['actual_trend']
            if yes:
                return f"Yes. {metric} has {article(trend)} {trend} trend with noise exceeding the threshold."
            return f"No. {metric} does not satisfy both the trend and noise conditions."

        if jtype == 'stat_threshold':
            direction = params['direction']
            if yes:
                return f"Yes. {metric} has values {direction} the threshold."
            return f"No. {metric} does not have values {direction} the threshold."

        if jtype in ('amplitude_vs_noise', 'amplitude_vs_std'):
            ref_name = 'noise strength' if jtype == 'amplitude_vs_noise' else 'standard deviation'
            if yes:
                return f"Yes. Some events in {metric} have amplitude exceeding the {ref_name}-based threshold."
            return f"No. No events in {metric} have amplitude exceeding the {ref_name}-based threshold."

        if jtype == 'range_vs_std':
            if yes:
                return f"Yes. {metric}'s range exceeds the standard deviation threshold."
            return f"No. {metric}'s range does not exceed the standard deviation threshold."

        if jtype == 'half_volatility':
            if yes:
                return f"Yes. {metric}'s second-half volatility is significantly higher than the first half."
            return f"No. {metric}'s second-half volatility is not significantly higher than the first half."

        if jtype == 'half_mean_shift':
            if yes:
                return f"Yes. {metric}'s mean shifts significantly between halves."
            return f"No. {metric}'s mean does not shift significantly between halves."

        if jtype == 'windowed_monotonicity':
            direction = params['direction']
            if yes:
                return f"Yes. {metric}'s windowed means show a consistently {direction} pattern."
            return f"No. {metric}'s windowed means do not show a consistently {direction} pattern."

        if jtype == 'phase_event':
            direction = params['event_direction']
            trend_type = params['required_trend_type']
            if yes:
                return f"Yes. {article(direction).capitalize()} {direction} event occurs during {article(trend_type)} {trend_type} phase in {metric}."
            return f"No. No {direction} event occurs during {article(trend_type)} {trend_type} phase in {metric}."

        if jtype == 'sequential_events':
            gap = params['gap_limit']
            ea = params['event_a_type']
            eb = params['event_b_type']
            if yes:
                return f"Yes. {article(ea).capitalize()} qualifying {ea}-then-{eb} pair exists within {gap} timestep{'s' if gap != 1 else ''} in {metric}."
            return f"No. No {ea}-then-{eb} pair occurs within {gap} timestep{'s' if gap != 1 else ''} in {metric}."

        # fallback (should not reach)
        if yes:
            return f"Yes. {metric} satisfies the condition."
        return f"No. {metric} does not satisfy the condition."

    # ------------------------------------------------------------------ #
    # Cross-metric statistical judgment (E1–E3)                           #
    # ------------------------------------------------------------------ #

    def _generate_cross_stat_judgment_qa(
        self,
        attributes_list: list,
        metrics: list,
    ) -> None:
        """Cross-metric statistical judgment: compare stats across 2 metrics."""
        if len(metrics) < 2:
            return

        _CROSS_STD_RATIO_LABELS = ['load imbalance', 'volatility dominance', 'spread disparity', 'instability gap']
        _CROSS_RANGE_RATIO_LABELS = ['range dominance', 'amplitude disparity', 'dynamic range imbalance', 'magnitude gap']
        _CROSS_SHIFT_LABELS = ['parallel mean shift', 'coordinated level change', 'joint half-shift', 'synchronized mean divergence']
        _CROSS_VOL_LABELS = ['asymmetric stress response', 'divergent volatility', 'unbalanced instability', 'volatility mismatch']

        # Build candidates for each pair
        pairs = []
        for i in range(len(metrics)):
            for j in range(len(metrics)):
                if i == j:
                    continue
                pairs.append((i, j))

        candidates = []
        for i, j in pairs:
            stats_i = (attributes_list[i].get('statistics') or {})
            stats_j = (attributes_list[j].get('statistics') or {})
            if (stats_i.get('std') or 0) > 0 and (stats_j.get('std') or 0) > 0:
                candidates.append(('cross_stat_ratio', 'std_ratio', i, j))
            ri = (stats_i.get('max', 0) or 0) - (stats_i.get('min', 0) or 0)
            rj = (stats_j.get('max', 0) or 0) - (stats_j.get('min', 0) or 0)
            if ri > 0 and rj > 0:
                candidates.append(('cross_stat_ratio', 'range_ratio', i, j))
            if stats_i.get('max') is not None and stats_i.get('max') != stats_i.get('min') and \
               stats_j.get('max') is not None and stats_j.get('max') != stats_j.get('min'):
                candidates.append(('cross_half_shift', None, i, j))
            if stats_i.get('first_half_std') is not None and stats_j.get('first_half_std') is not None:
                # Only add when exactly one metric increases volatility,
                # so "yes" verdict is achievable.  When both increase or
                # both decrease, "yes" is structurally impossible → skew.
                fhs_i = stats_i.get('first_half_std', 0.0) or 0.0
                shs_i = stats_i.get('second_half_std', 0.0) or 0.0
                fhs_j = stats_j.get('first_half_std', 0.0) or 0.0
                shs_j = stats_j.get('second_half_std', 0.0) or 0.0
                if (round(shs_i, 2) > round(fhs_i, 2)) != (round(shs_j, 2) > round(fhs_j, 2)):
                    candidates.append(('cross_half_volatility', None, i, j))

        if not candidates:
            return

        if self.config.debug:
            chosen = list(candidates)
        else:
            chosen = random.sample(candidates, min(2, len(candidates)))

        for jtype, sub_variant, idx_a, idx_b in chosen:
            m_a = metrics[idx_a]
            m_b = metrics[idx_b]
            stats_a = (attributes_list[idx_a].get('statistics') or {})
            stats_b = (attributes_list[idx_b].get('statistics') or {})

            if jtype == 'cross_stat_ratio':
                K_options = [0.2, 0.3, 0.5, 0.8, 1.0, 1.2, 1.5, 2.0, 3.0, 5.0, 8.0]
                if sub_variant == 'std_ratio':
                    val_a = stats_a.get('std', 0.0)
                    val_b = stats_b.get('std', 0.0)
                else:
                    val_a = stats_a.get('max', 0) - stats_a.get('min', 0)
                    val_b = stats_b.get('max', 0) - stats_b.get('min', 0)

                if val_b <= 0:
                    continue
                actual_ratio = val_a / val_b
                if random.random() < 0.50:
                    K = random.choice([k for k in K_options if k <= actual_ratio] or [K_options[0]])
                else:
                    K = random.choice([k for k in K_options if k > actual_ratio] or [K_options[-1]])

                threshold = round(K * val_b, 2)
                verdict = 'yes' if round(val_a, 2) >= threshold else 'no'
                label = random.choice(_CROSS_STD_RATIO_LABELS if sub_variant == 'std_ratio' else _CROSS_RANGE_RATIO_LABELS)

                params = dict(
                    sub_variant=sub_variant,
                    K=K,
                    val_a=val_a,
                    val_b=val_b,
                    verdict=verdict,
                )
                q_bank = _CROSS_STAT_STD_RATIO_QUESTIONS if sub_variant == 'std_ratio' else _CROSS_STAT_RANGE_RATIO_QUESTIONS
                question = random.choice(q_bank).format(
                    metric_a=m_a, metric_b=m_b, label=label, K=K,
                )

            elif jtype == 'cross_half_shift':
                K_options = [1, 2, 5, 8, 10, 15, 20]

                # Pre-compute actual absolute shift percentages for balancing.
                # Use absolute shift so negative shifts (second half < first half)
                # don't structurally prevent "yes" verdicts.
                pre_data = []
                for m, stats in [(m_a, stats_a), (m_b, stats_b)]:
                    act_max = stats.get('max', 0.0)
                    act_min = stats.get('min', 0.0)
                    fhm = stats.get('first_half_mean', 0.0) or 0.0
                    shm = stats.get('second_half_mean', 0.0) or 0.0
                    rng = act_max - act_min
                    shift = abs(shm - fhm)
                    actual_pct = (shift / rng * 100) if rng > 0 else 0.0
                    pre_data.append((m, act_max, act_min, fhm, shm, rng, shift, actual_pct))

                # Both must pass for "yes": min pct is the bottleneck
                min_actual_pct = min(d[7] for d in pre_data)
                if random.random() < 0.50:
                    K_pct = random.choice([k for k in K_options if k <= min_actual_pct] or [K_options[0]])
                else:
                    K_pct = random.choice([k for k in K_options if k > min_actual_pct] or [K_options[-1]])

                metrics_data = []
                all_met = True
                for m, act_max, act_min, fhm, shm, rng, shift, _ in pre_data:
                    threshold_val = K_pct / 100.0 * rng
                    # Match builder rounding: signed_shift=round(shm-fhm,2), abs=round(|signed|,2)
                    shift_r = round(abs(round(shm - fhm, 2)), 2)
                    thresh_r = round(threshold_val, 2)
                    if shift_r <= thresh_r:
                        all_met = False
                    metrics_data.append({
                        'metric': m, 'max': act_max, 'min': act_min,
                        'first_half_mean': fhm, 'second_half_mean': shm,
                        'range': rng, 'shift': shift, 'threshold': threshold_val,
                    })

                verdict = 'yes' if all_met else 'no'
                label = random.choice(_CROSS_SHIFT_LABELS)

                params = dict(
                    K_pct=K_pct,
                    metrics_data=[(m_a, metrics_data[0]), (m_b, metrics_data[1])],
                    verdict=verdict,
                )
                question = random.choice(_CROSS_HALF_SHIFT_QUESTIONS).format(
                    metric_a=m_a, metric_b=m_b, label=label, K=K_pct,
                )

            elif jtype == 'cross_half_volatility':
                # Guard ensures exactly one metric increases volatility.
                # Place the increasing metric as A for "yes", swap for "no".
                fhs_a = stats_a.get('first_half_std', 0.0) or 0.0
                shs_a = stats_a.get('second_half_std', 0.0) or 0.0
                fhs_b = stats_b.get('first_half_std', 0.0) or 0.0
                shs_b = stats_b.get('second_half_std', 0.0) or 0.0

                # Ensure A is the increasing metric
                if round(shs_b, 2) > round(fhs_b, 2):
                    idx_a, idx_b = idx_b, idx_a
                    m_a, m_b = m_b, m_a
                    stats_a, stats_b = stats_b, stats_a
                    fhs_a, fhs_b = fhs_b, fhs_a
                    shs_a, shs_b = shs_b, shs_a

                # 50/50 verdict: "yes" keeps A-increases ordering,
                # "no" swaps so the non-increasing metric is A
                if random.random() < 0.5:
                    verdict = 'yes'
                else:
                    idx_a, idx_b = idx_b, idx_a
                    m_a, m_b = m_b, m_a
                    stats_a, stats_b = stats_b, stats_a
                    fhs_a, fhs_b = fhs_b, fhs_a
                    shs_a, shs_b = shs_b, shs_a
                    verdict = 'no'
                label = random.choice(_CROSS_VOL_LABELS)

                params = dict(
                    fhs_a=fhs_a, shs_a=shs_a,
                    fhs_b=fhs_b, shs_b=shs_b,
                    verdict=verdict,
                )
                question = random.choice(_CROSS_HALF_VOLATILITY_QUESTIONS).format(
                    metric_a=m_a, metric_b=m_b, label=label,
                )

            else:
                continue

            try:
                think_str, verdict = build_cross_stat_judgment_thought(
                    attributes_list, metrics, idx_a, idx_b, jtype, params,
                )
            except Exception:
                logger.warning("Cross-stat QA generation failed", exc_info=True)
                continue

            answer_text = self._cross_stat_judgment_answer_text(
                jtype, verdict, params, m_a, m_b, sub_variant if jtype == 'cross_stat_ratio' else None,
            )

            self.questions.append(_fix_article(question))
            self.answers.append(think_str + answer_text)
            self.llm_prompts.append([])
            self.fields.append({'timeseries': [idx_a, idx_b]})
            self.qa_types.append(QAType.SEGMENT_MASK)
            self.eval_tasks.append("cross_stat_judgment")
            self.eval_metadatas.append({
                "judgment_type": jtype,
                "sub_variant": sub_variant if jtype == 'cross_stat_ratio' else None,
                "verdict": verdict,
                "length": self.seq_len,
            })

    @staticmethod
    def _cross_stat_judgment_answer_text(
        jtype: str, verdict: str, params: dict,
        m_a: str, m_b: str, sub_variant: str = None,
    ) -> str:
        """Build grounded answer text for cross-stat judgment QA."""
        yes = (verdict == 'yes')

        if jtype == 'cross_stat_ratio':
            stat_name = 'std' if sub_variant == 'std_ratio' else 'range'
            if yes:
                return f"Yes. {m_a}'s {stat_name} exceeds the required multiple of {m_b}'s."
            return f"No. {m_a}'s {stat_name} does not reach the required multiple of {m_b}'s."

        if jtype == 'cross_half_shift':
            if yes:
                return f"Yes. Both {m_a} and {m_b} show a significant mean shift between halves."
            return f"No. {m_a} and {m_b} do not both show a significant mean shift between halves."

        if jtype == 'cross_half_volatility':
            if yes:
                return (f"Yes. {m_a}'s volatility increases in the second half "
                        f"while {m_b}'s does not.")
            return (f"No. {m_a} and {m_b} do not show the required "
                    f"divergent volatility pattern.")

        # fallback
        if yes:
            return f"Yes. The condition is met for {m_a} and {m_b}."
        return f"No. The condition is not met for {m_a} and {m_b}."

    # ------------------------------------------------------------------ #
    # Family 8 — Trend dominance in range                                 #
    # ------------------------------------------------------------------ #

    def _generate_segment_trend_dominance_qa(
        self,
        attributes: Dict,
        metric: str,
        series_index: Optional[int] = None,
    ) -> None:
        """Family 8: 'In range [A,B], which trend type dominates?'"""
        trend_list = [s for s in attributes.get('trend_list', [])
                      if isinstance(s, (list, tuple)) and len(s) >= 3]
        if len(trend_list) < 2:
            return

        # Pick adjacent segment pair to anchor the query range
        seg_a_idx = random.randrange(len(trend_list) - 1)
        seg_a = trend_list[seg_a_idx]
        seg_b = trend_list[seg_a_idx + 1]

        # 0-indexed boundaries
        a_s1, a_e1 = seg_a[1], seg_a[2]
        b_s1, b_e1 = seg_b[1], seg_b[2]

        # query_a: first half of seg_a; query_b: second half of seg_b
        query_a = random.randint(a_s1, max(a_s1, (a_s1 + a_e1) // 2))
        query_b = random.randint(min(b_e1, (b_s1 + b_e1) // 2), b_e1)

        think_str, verdict = build_trend_dominance_thought(attributes, metric, query_a, query_b)

        question = random.choice(_TREND_DOMINANCE_QUESTIONS).format(
            metric=metric, a=query_a, b=query_b
        )

        direction_text = {
            'increase': 'an increasing (upward) trend',
            'decrease': 'a decreasing (downward) trend',
            'keep steady': 'a steady (flat) trend',
            'equal': 'equal coverage of multiple trend types',
        }.get(verdict, verdict)
        answer = think_str + f"In the range [{query_a}, {query_b}], {metric} is dominated by {direction_text}."

        field_idx = series_index if series_index is not None else 0
        self.questions.append(_fix_article(question))
        self.answers.append(answer)
        self.llm_prompts.append([])
        self.fields.append({'trend': [field_idx]})
        self.qa_types.append(QAType.SEGMENT_MASK)
        self.eval_tasks.append("segment_trend_dominance")
        self.eval_metadatas.append({'verdict': verdict, 'length': self.seq_len})

    # ------------------------------------------------------------------ #
    # Family 9 — Statistical numerical QA                                 #
    # ------------------------------------------------------------------ #

    def _generate_stat_numerical_qa(
        self,
        attributes: Dict,
        metric: str,
        series_index=None,
        timeseries=None,
    ) -> None:
        """Family 9: Deterministic numerical statistical queries."""
        statistics = attributes.get('statistics') or {}
        if not statistics:
            return
        seq_len = attributes['seq_len']
        if seq_len < 16:
            return

        field_key = 'series_index' if series_index is not None else 'timeseries'
        field_idx = series_index if series_index is not None else 0

        # Duration-weighted sub-type pool: longer-range QAs get higher weight
        # (windowed_mean over full series = highest signal)
        # Note: weights sum to 1.24 (not 1.0); weighted_random_choice normalises
        # them internally via normalize_probabilities(), so relative ratios matter.
        sub_type_weights = {
            'stat_windowed_mean':       0.20,
            'stat_extrema_pos':         0.15,
            'stat_threshold_duration':  0.15,
            'stat_segment_compare':     0.15,
            'stat_mean_crossing':       0.12,
            'stat_range':               0.10,
            'stat_min_val':             0.06,
            'stat_max_val':             0.06,
            'stat_mean_val':            0.06,
            'stat_std_val':             0.06,
            'stat_volatility_change':   0.08,
            'stat_half_mean_diff':      0.08,
            'stat_median':              0.06,
            'stat_windowed_trend':      0.08,
        }
        sub_types = list(sub_type_weights.keys())
        weights   = list(sub_type_weights.values())

        if self.config.debug:
            n_generate = len(sub_types)
        else:
            n_generate = random.randint(2, 4)
        chosen = []
        remaining = list(range(len(sub_types)))
        for _ in range(min(n_generate, len(sub_types))):
            if not remaining:
                break
            pool_dict = {sub_types[i]: weights[i] for i in remaining}
            pick = weighted_random_choice(pool_dict)
            idx = sub_types.index(pick)
            chosen.append(pick)
            remaining.remove(idx)

        for sub_type in chosen:
            try:
                params, question = self._build_stat_qa_params(
                    sub_type, statistics, seq_len, metric, timeseries
                )
                if params is None:
                    continue
                think_str, verdict = build_stat_numerical_thought(
                    attributes, metric, sub_type, params
                )
            except Exception:
                logger.warning("QA generation failed", exc_info=True)
                continue

            answer_text = self._stat_answer_text(sub_type, verdict, params, metric, seq_len)
            self.questions.append(_fix_article(question))
            self.answers.append(think_str + answer_text)
            self.llm_prompts.append([])
            self.fields.append({field_key: [field_idx]})
            self.qa_types.append(QAType.SEGMENT_MASK)
            self.eval_tasks.append("stat_numerical")
            self.eval_metadatas.append({
                "sub_type": sub_type,
                "verdict":  verdict,
                "length":   seq_len,
            })

    def _build_stat_qa_params(self, sub_type, statistics, seq_len, metric, timeseries=None):
        """Build params dict and question string for each stat sub-type."""

        if sub_type == 'stat_extrema_pos':
            direction = random.choice(['peak', 'trough'])
            if direction == 'peak':
                pos = statistics.get('max_pos', 0)
                val = statistics.get('max', 0.0)
                q   = random.choice(_STAT_PEAK_QUESTIONS).format(metric=metric)
            else:
                pos = statistics.get('min_pos', 0)
                val = statistics.get('min', 0.0)
                q   = random.choice(_STAT_TROUGH_QUESTIONS).format(metric=metric)
            return {'direction': direction, 'pos': pos, 'val': val}, q

        elif sub_type == 'stat_windowed_mean':
            # Duration-weighted: 60% full series, 40% partial range
            window_size = random.choice([16, 32])
            key         = f'segment_means_{window_size}'
            all_means   = statistics.get(key, [])
            if not all_means:
                return None, None

            n_wins = len(all_means)
            use_full = random.random() < 0.60
            if use_full or n_wins <= 2:
                start_win    = 0
                window_means = all_means
                win_start    = 0
                win_end      = seq_len - 1
                q = random.choice(_STAT_WINDOWED_MEAN_FULL_QUESTIONS).format(
                    metric=metric, window=window_size
                )
            else:
                # Weighted toward longer partial ranges
                min_wins   = max(2, n_wins // 3)
                win_range  = list(range(min_wins, n_wins))
                total_w    = sum(win_range)
                win_dict   = {str(v): v / total_w for v in win_range}
                n_selected = int(weighted_random_choice(win_dict))
                start_win  = random.randint(0, n_wins - n_selected)
                window_means = all_means[start_win: start_win + n_selected]
                win_start    = start_win * window_size
                win_end      = min((start_win + n_selected) * window_size - 1, seq_len - 1)
                q = random.choice(_STAT_WINDOWED_MEAN_PARTIAL_QUESTIONS).format(
                    metric=metric, window=window_size, start=win_start, end=win_end
                )
            return {'window_size': window_size, 'window_means': window_means,
                    'start_window': start_win, 'win_start': win_start,
                    'win_end': win_end}, q

        elif sub_type == 'stat_threshold_duration':
            mean_val = round(statistics['mean'], 2)
            std_val  = round(statistics['std'], 2)
            if std_val == 0.0:
                return None, None
            variant  = random.choice(['above_mean', 'above_std', 'below_std'])
            if variant == 'above_mean':
                threshold = round(mean_val, 2)
                count     = statistics['above_mean_count']
                q = random.choice(_STAT_THRESHOLD_ABOVE_MEAN_QUESTIONS).format(
                    metric=metric, threshold=format_float(threshold, 2), seq_len=seq_len
                )
                return {'direction': 'above', 'threshold': threshold, 'count': count,
                        'variant': variant, 'mean_val': mean_val, 'std_val': std_val}, q
            elif variant == 'above_std':
                threshold = round(mean_val + std_val, 2)
                count     = statistics['above_mean_std_count']
                q = random.choice(_STAT_THRESHOLD_ABOVE_STD_QUESTIONS).format(
                    metric=metric, threshold=format_float(threshold, 2)
                )
                return {'direction': 'above', 'threshold': threshold, 'count': count,
                        'variant': variant, 'mean_val': mean_val, 'std_val': std_val}, q
            else:
                threshold = round(mean_val - std_val, 2)
                count     = statistics['below_mean_std_count']
                q = random.choice(_STAT_THRESHOLD_BELOW_STD_QUESTIONS).format(
                    metric=metric, threshold=format_float(threshold, 2)
                )
                return {'direction': 'below', 'threshold': threshold, 'count': count,
                        'variant': variant, 'mean_val': mean_val, 'std_val': std_val}, q

        elif sub_type == 'stat_mean_crossing':
            crossings        = statistics.get('mean_crossings', 0)
            crossing_indices = statistics.get('mean_crossing_indices', [])
            mean_val         = round(statistics.get('mean', 0.0), 3)
            q = random.choice(_STAT_MEAN_CROSSING_QUESTIONS).format(
                metric=metric, mean=format_float(mean_val, 2)
            )
            return {'crossings': crossings, 'crossing_indices': crossing_indices}, q

        elif sub_type == 'stat_range':
            r = statistics.get('range', 0.0)
            q = random.choice(_STAT_RANGE_QUESTIONS).format(metric=metric)
            return {'range': r}, q

        elif sub_type == 'stat_segment_compare':
            mid = statistics.get('split_mid')
            if mid is None or mid <= 0:
                return None, None
            compare_by = random.choice(['mean', 'std'])
            if compare_by == 'mean':
                first_val  = statistics['first_half_mean']
                second_val = statistics['second_half_mean']
                q = random.choice(_STAT_SEGMENT_COMPARE_MEAN_QUESTIONS).format(
                    metric=metric, mid=mid - 1, mid1=mid, end=seq_len - 1
                )
            else:
                first_val  = statistics['first_half_std']
                second_val = statistics['second_half_std']
                q = random.choice(_STAT_SEGMENT_COMPARE_STD_QUESTIONS).format(metric=metric)
            first_val = round(first_val, 2)
            second_val = round(second_val, 2)
            eps = 0.05 * max(abs(first_val), abs(second_val), 1e-6)
            if first_val > second_val + eps:
                verdict = 'first half'
            elif second_val > first_val + eps:
                verdict = 'second half'
            else:
                verdict = 'equal'
            return {'compare_by': compare_by, 'first_val': first_val,
                    'second_val': second_val, 'mid': mid, 'verdict': verdict}, q

        elif sub_type == 'stat_min_val':
            val = statistics.get('min', 0.0)
            q   = random.choice(_STAT_MIN_VAL_QUESTIONS).format(metric=metric)
            return {'value': val}, q

        elif sub_type == 'stat_max_val':
            val = statistics.get('max', 0.0)
            q   = random.choice(_STAT_MAX_VAL_QUESTIONS).format(metric=metric)
            return {'value': val}, q

        elif sub_type == 'stat_mean_val':
            # Teach chunk-and-average decomposition. Works at any seq_len;
            # at long seq_len the think block enumerates more chunks
            # (n_chunks = seq_len / 16), intentional for length generalization.
            if timeseries is None:
                return None, None
            from synth.align.generators.atomic_qa import _compute_chunks
            chunks = _compute_chunks(np.asarray(timeseries, dtype=float), 0, seq_len - 1)
            chunk_means = [cm for _, _, cm, _ in chunks]
            mean_of_means = round(float(np.mean(chunk_means)), 2)
            q = random.choice(_STAT_MEAN_VAL_QUESTIONS).format(metric=metric)
            return {'value': mean_of_means, 'chunks': chunks}, q

        elif sub_type == 'stat_std_val':
            val = statistics.get('std', 0.0)
            q   = random.choice(_STAT_STD_VAL_QUESTIONS).format(metric=metric)
            return {'value': val}, q

        elif sub_type == 'stat_volatility_change':
            mid = statistics.get('split_mid')
            if mid is None or mid <= 0:
                return None, None
            first_half_std  = statistics['first_half_std']
            second_half_std = statistics['second_half_std']
            q = random.choice(_STAT_VOLATILITY_CHANGE_QUESTIONS).format(metric=metric)
            return {'first_half_std': first_half_std, 'second_half_std': second_half_std}, q

        elif sub_type == 'stat_half_mean_diff':
            mid = statistics.get('split_mid')
            if mid is None or mid <= 0:
                return None, None
            first_half_mean  = statistics['first_half_mean']
            second_half_mean = statistics['second_half_mean']
            q = random.choice(_STAT_HALF_MEAN_DIFF_QUESTIONS).format(metric=metric)
            return {'first_half_mean': first_half_mean, 'second_half_mean': second_half_mean}, q

        elif sub_type == 'stat_median':
            median_val = statistics.get('median')
            if median_val is None:
                return None, None
            q = random.choice(_STAT_MEDIAN_QUESTIONS).format(metric=metric)
            return {'median': median_val}, q

        elif sub_type == 'stat_windowed_trend':
            segment_means = statistics.get('segment_means_16', [])
            if len(segment_means) < 2:
                return None, None
            q = random.choice(_STAT_WINDOWED_TREND_QUESTIONS).format(metric=metric)
            return {'segment_means_16': segment_means}, q

        return None, None

    @staticmethod
    def _stat_answer_text(sub_type, verdict, params, metric, seq_len):
        """Build the human-readable answer sentence for a stat QA."""
        if sub_type == 'stat_extrema_pos':
            direction = params['direction']
            val       = params['val']
            pos       = params['pos']
            word      = 'peak' if direction == 'peak' else 'trough'
            return f"The {word} of {metric} occurs at step {pos} (value = {format_float(val, 2)})."

        elif sub_type == 'stat_windowed_mean':
            window_size  = params['window_size']
            window_means = params['window_means']
            start_win    = params.get('start_window', 0)
            win_start    = start_win * window_size
            win_end      = min((start_win + len(window_means)) * window_size - 1, seq_len - 1)
            vals_str     = ", ".join(format_float(v, 2) for v in window_means)
            return (f"The mean of {metric} for each {window_size}-step window "
                    f"(points {win_start}–{win_end}) is: [{vals_str}].")

        elif sub_type == 'stat_threshold_duration':
            direction = params['direction']
            threshold = params['threshold']
            count     = params['count']
            op        = 'above' if direction == 'above' else 'below'
            return (f"In {metric}, {count} out of {seq_len} steps have values "
                    f"{op} the threshold of {format_float(threshold, 2)}.")

        elif sub_type == 'stat_mean_crossing':
            crossings = params['crossings']
            unit = "time" if int(crossings) == 1 else "times"
            return f"{metric} crosses its mean value {crossings} {unit} throughout the series."

        elif sub_type == 'stat_range':
            r = params['range']
            return f"The value range of {metric} (max − min) is {format_float(r, 2)}."

        elif sub_type == 'stat_segment_compare':
            compare_by = params['compare_by']
            verdict    = params['verdict']
            if verdict == 'equal':
                return f"Both halves of {metric} have approximately equal {compare_by}."
            return f"The {verdict} of {metric} has a higher {compare_by}."

        elif sub_type == 'stat_min_val':
            return f"The minimum value of {metric} is {format_float(params['value'], 2)}."

        elif sub_type == 'stat_max_val':
            return f"The maximum value of {metric} is {format_float(params['value'], 2)}."

        elif sub_type == 'stat_mean_val':
            return f"The mean (average) value of {metric} is {format_float(params['value'], 2)}."

        elif sub_type == 'stat_std_val':
            return f"The standard deviation of {metric} is {format_float(params['value'], 2)}."

        elif sub_type == 'stat_volatility_change':
            return f"The volatility of {metric} is {verdict} in the second half compared to the first."

        elif sub_type == 'stat_half_mean_diff':
            return f"The absolute difference between the first-half and second-half means of {metric} is {verdict}."

        elif sub_type == 'stat_median':
            return f"The median value of {metric} is {verdict}."

        elif sub_type == 'stat_windowed_trend':
            return f"The windowed means of {metric} show a {verdict} trend."

        return verdict

    # ------------------------------------------------------------------ #
    # Periodicity QA family                                                #
    # ------------------------------------------------------------------ #

    def _generate_periodicity_qa(
        self,
        attributes: Dict,
        metric: str,
        series_index=None,
    ) -> None:
        """Periodicity QA: period_estimate, cycle_count."""
        seasonal = attributes.get('seasonal', {})
        if not seasonal or seasonal.get('period', 0) <= 0:
            return
        seq_len = attributes['seq_len']

        field_key = 'series_index' if series_index is not None else 'timeseries'
        field_idx = series_index if series_index is not None else 0

        sub_type_weights = {
            'period_estimate': 0.55,
            'cycle_count': 0.45,
        }

        if not sub_type_weights:
            return

        sub_types = list(sub_type_weights.keys())
        weights = list(sub_type_weights.values())

        if self.config.debug:
            n_generate = len(sub_types)
        else:
            n_generate = random.randint(1, 2)
        chosen = []
        remaining = list(range(len(sub_types)))
        for _ in range(min(n_generate, len(sub_types))):
            if not remaining:
                break
            pool_dict = {sub_types[i]: weights[i] for i in remaining}
            pick = weighted_random_choice(pool_dict)
            idx = sub_types.index(pick)
            chosen.append(pick)
            remaining.remove(idx)

        for sub_type in chosen:
            try:
                params, question = self._build_periodicity_params(
                    sub_type, attributes, metric
                )
                if params is None:
                    continue
                think_str, verdict = build_periodicity_thought(
                    attributes, metric, sub_type, params
                )
            except Exception:
                logger.warning(
                    f"Periodicity QA generation failed for sub_type={sub_type}",
                    exc_info=True,
                )
                continue

            answer_text = self._periodicity_answer_text(sub_type, verdict, params, metric)
            self.questions.append(_fix_article(question))
            self.answers.append(think_str + answer_text)
            self.llm_prompts.append([])
            self.fields.append({field_key: [field_idx]})
            self.qa_types.append(QAType.SEGMENT_MASK)
            self.eval_tasks.append("periodicity")
            self.eval_metadatas.append({
                "sub_type": sub_type,
                "verdict": verdict,
                "length": self.seq_len,
            })

    def _build_periodicity_params(self, sub_type, attributes, metric):
        """Build params dict and question string for each periodicity sub-type."""
        seasonal = attributes.get('seasonal', {})
        if not seasonal or seasonal.get('period', 0) <= 0:
            return None, None

        period = seasonal['period']

        if sub_type == 'period_estimate':
            q = random.choice(_PERIODICITY_PERIOD_QUESTIONS).format(metric=metric)
            return {'period': period}, q

        elif sub_type == 'cycle_count':
            seq_len = attributes['seq_len']
            q = random.choice(_PERIODICITY_CYCLE_COUNT_QUESTIONS).format(metric=metric)
            return {'period': period, 'seq_len': seq_len}, q

        return None, None

    @staticmethod
    def _periodicity_answer_text(sub_type, verdict, params, metric):
        """Build the human-readable answer sentence for a periodicity QA."""
        if sub_type == 'period_estimate':
            return (f"The estimated period of {metric} is "
                    f"{format_float(params['period'], 2)} time steps.")
        elif sub_type == 'cycle_count':
            return f"{metric} contains {verdict} complete cycles."
        return verdict

    # ------------------------------------------------------------------ #
    # Family 11E — Local event enumeration QA                              #
    # ------------------------------------------------------------------ #

    def _generate_local_enumeration_qa(
        self,
        attributes: Dict,
        metric: str,
        series_index=None,
    ) -> None:
        """Enumeration QA over local events: count/filter/extreme."""
        local_events = attributes.get('local', [])
        if len(local_events) < 2:
            return

        field_key = 'series_index' if series_index is not None else 'timeseries'
        field_idx = series_index if series_index is not None else 0

        # Build sub-type pool based on event properties
        sub_type_weights: Dict[str, float] = {'count_by_type': 0.15}

        has_direction = any(
            (ev.get('params') or {}).get('direction') for ev in local_events
        )
        if has_direction:
            sub_type_weights['count_by_direction'] = 0.15

        events_with_amp = [ev for ev in local_events if 'amplitude' in ev]
        if len(events_with_amp) >= 2:
            sub_type_weights['count_by_amplitude'] = 0.15
            sub_type_weights['count_by_type_and_amp'] = 0.10
            sub_type_weights['highest_amplitude'] = 0.12
            sub_type_weights['lowest_amplitude'] = 0.12

        events_with_end = [ev for ev in local_events if 'position_end' in ev]
        if len(events_with_end) >= 2:
            sub_type_weights['widest_span'] = 0.12
            sub_type_weights['narrowest_span'] = 0.12

        sub_types = list(sub_type_weights.keys())
        weights = list(sub_type_weights.values())

        if self.config.debug:
            n_generate = len(sub_types)
        else:
            n_generate = random.randint(1, min(3, len(sub_types)))
        chosen = []
        remaining = list(range(len(sub_types)))
        for _ in range(n_generate):
            if not remaining:
                break
            pool_dict = {sub_types[i]: weights[i] for i in remaining}
            pick = weighted_random_choice(pool_dict)
            idx = sub_types.index(pick)
            chosen.append(pick)
            remaining.remove(idx)

        for sub_type in chosen:
            try:
                params, question = self._build_local_enum_params(
                    sub_type, local_events, metric
                )
                if params is None:
                    continue
                think_str, verdict = build_local_enumeration_thought(
                    attributes, metric, sub_type, params
                )
            except Exception:
                logger.warning("QA generation failed", exc_info=True)
                continue

            answer_text = self._local_enum_answer_text(
                sub_type, verdict, params, metric
            )
            self.questions.append(_fix_article(question))
            self.answers.append(think_str + answer_text)
            self.llm_prompts.append([])
            self.fields.append({field_key: [field_idx]})
            self.qa_types.append(QAType.SEGMENT_MASK)
            self.eval_tasks.append("local_enumeration")
            self.eval_metadatas.append({
                "sub_type": sub_type,
                "verdict": verdict,
                "length": self.seq_len,
            })

    def _build_local_enum_params(self, sub_type, local_events, metric):
        """Build params dict and question string for local enumeration sub-type."""

        if sub_type == 'count_by_type':
            types = [ev.get('type', '?') for ev in local_events]
            target_type = random.choice(types)
            count = sum(1 for t in types if t == target_type)
            q = random.choice(_LOCAL_COUNT_BY_TYPE_QUESTIONS).format(
                metric=metric, target_type=target_type
            )
            return {'target_type': target_type, 'verdict': str(count)}, q

        elif sub_type == 'count_by_direction':
            dirs = [
                (ev.get('params') or {}).get('direction', 'none')
                for ev in local_events
            ]
            valid_dirs = [d for d in dirs if d and d != 'none']
            if not valid_dirs:
                return None, None
            target_dir = random.choice(valid_dirs)
            count = sum(1 for d in dirs if d == target_dir)
            q = random.choice(_LOCAL_COUNT_BY_DIRECTION_QUESTIONS).format(
                metric=metric, target_direction=target_dir
            )
            return {'target_direction': target_dir, 'verdict': str(count)}, q

        elif sub_type == 'count_by_amplitude':
            amps = [ev['amplitude'] for ev in local_events if 'amplitude' in ev]
            if len(amps) < 2:
                return None, None
            min_amp, max_amp = min(amps), max(amps)
            threshold = round(random.uniform(min_amp, max_amp), 2)
            count = sum(1 for a in amps if round(a, 2) > threshold)
            q = random.choice(_LOCAL_COUNT_BY_AMPLITUDE_QUESTIONS).format(
                metric=metric, threshold=format_float(threshold, 2)
            )
            return {'threshold': threshold, 'verdict': str(count)}, q

        elif sub_type == 'count_by_type_and_amp':
            events_with_amp = [ev for ev in local_events if 'amplitude' in ev]
            if len(events_with_amp) < 2:
                return None, None
            target_type = random.choice(
                [ev.get('type', '?') for ev in events_with_amp]
            )
            type_amps = [
                ev['amplitude'] for ev in events_with_amp
                if ev.get('type') == target_type
            ]
            if not type_amps:
                return None, None
            min_a, max_a = min(type_amps), max(type_amps)
            threshold = round(random.uniform(0.5 * min_a, 1.2 * max_a), 2)
            count = sum(
                1 for ev in local_events
                if ev.get('type') == target_type
                and 'amplitude' in ev
                and round(ev['amplitude'], 2) > threshold
            )
            q = random.choice(_LOCAL_COUNT_BY_TYPE_AND_AMP_QUESTIONS).format(
                metric=metric, target_type=target_type,
                threshold=format_float(threshold, 2),
            )
            return {
                'target_type': target_type, 'threshold': threshold,
                'verdict': str(count),
            }, q

        elif sub_type == 'highest_amplitude':
            events_with_amp = [
                (i, ev) for i, ev in enumerate(local_events)
                if 'amplitude' in ev
            ]
            if len(events_with_amp) < 2:
                return None, None
            best_amp = max(ev['amplitude'] for _, ev in events_with_amp)
            winner_types = [ev.get('type', '?') for _, ev in events_with_amp
                           if ev['amplitude'] == best_amp]
            q = random.choice(_LOCAL_HIGHEST_AMPLITUDE_QUESTIONS).format(
                metric=metric
            )
            return {
                'winner_types': winner_types,
                'winner_amp': best_amp,
                'verdict': format_float(best_amp, 2),
            }, q

        elif sub_type == 'lowest_amplitude':
            events_with_amp = [
                (i, ev) for i, ev in enumerate(local_events)
                if 'amplitude' in ev
            ]
            if len(events_with_amp) < 2:
                return None, None
            best_amp = min(ev['amplitude'] for _, ev in events_with_amp)
            winner_types = [ev.get('type', '?') for _, ev in events_with_amp
                           if ev['amplitude'] == best_amp]
            q = random.choice(_LOCAL_LOWEST_AMPLITUDE_QUESTIONS).format(
                metric=metric
            )
            return {
                'winner_types': winner_types,
                'winner_amp': best_amp,
                'verdict': format_float(best_amp, 2),
            }, q

        elif sub_type == 'widest_span':
            events_with_end = [
                (i, ev) for i, ev in enumerate(local_events)
                if 'position_end' in ev
            ]
            if len(events_with_end) < 2:
                return None, None
            def _span(ev):
                return ev['position_end'] - ev['position_start']
            best_span = max(_span(ev) for _, ev in events_with_end)
            winner_types = [ev.get('type', '?') for _, ev in events_with_end
                           if _span(ev) == best_span]
            q = random.choice(_LOCAL_WIDEST_SPAN_QUESTIONS).format(
                metric=metric
            )
            return {
                'winner_types': winner_types,
                'winner_span': best_span,
                'verdict': str(best_span),
            }, q

        elif sub_type == 'narrowest_span':
            events_with_end = [
                (i, ev) for i, ev in enumerate(local_events)
                if 'position_end' in ev
            ]
            if len(events_with_end) < 2:
                return None, None
            def _span(ev):
                return ev['position_end'] - ev['position_start']
            best_span = min(_span(ev) for _, ev in events_with_end)
            winner_types = [ev.get('type', '?') for _, ev in events_with_end
                           if _span(ev) == best_span]
            q = random.choice(_LOCAL_NARROWEST_SPAN_QUESTIONS).format(
                metric=metric
            )
            return {
                'winner_types': winner_types,
                'winner_span': best_span,
                'verdict': str(best_span),
            }, q

        return None, None

    @staticmethod
    def _local_enum_answer_text(sub_type, verdict, params, metric):
        """Build human-readable answer for local enumeration QA."""
        if sub_type == 'count_by_type':
            noun = f"{params['target_type']} event"
            return f"There {pluralize(verdict, noun)} in {metric}."

        elif sub_type == 'count_by_direction':
            return (f"There {pluralize(verdict, 'local event')} with direction "
                    f"{params['target_direction']} in {metric}.")

        elif sub_type == 'count_by_amplitude':
            return (f"There {pluralize(verdict, 'local event')} in {metric} with "
                    f"amplitude greater than {format_float(params['threshold'], 2)}.")

        elif sub_type == 'count_by_type_and_amp':
            noun = f"{params['target_type']} event"
            return (f"There {pluralize(verdict, noun)} "
                    f"in {metric} with amplitude greater than "
                    f"{format_float(params['threshold'], 2)}.")

        elif sub_type == 'highest_amplitude':
            winner_types = params.get('winner_types', [params.get('winner_type', '?')])
            if len(winner_types) > 1:
                return (f"The local events with the highest amplitude in {metric} "
                        f"are {' and '.join(winner_types)}, each with amplitude "
                        f"{format_float(params['winner_amp'], 2)}.")
            return (f"The local event with the highest amplitude in {metric} "
                    f"is {winner_types[0]} with amplitude "
                    f"{format_float(params['winner_amp'], 2)}.")

        elif sub_type == 'lowest_amplitude':
            winner_types = params.get('winner_types', [params.get('winner_type', '?')])
            if len(winner_types) > 1:
                return (f"The local events with the lowest amplitude in {metric} "
                        f"are {' and '.join(winner_types)}, each with amplitude "
                        f"{format_float(params['winner_amp'], 2)}.")
            return (f"The local event with the lowest amplitude in {metric} "
                    f"is {winner_types[0]} with amplitude "
                    f"{format_float(params['winner_amp'], 2)}.")

        elif sub_type == 'widest_span':
            winner_types = params.get('winner_types', [params.get('winner_type', '?')])
            if len(winner_types) > 1:
                return (f"The widest local events in {metric} are "
                        f"{' and '.join(winner_types)}, each spanning "
                        f"{params['winner_span']} timesteps.")
            return (f"The widest local event in {metric} is "
                    f"{winner_types[0]} spanning "
                    f"{params['winner_span']} timesteps.")

        elif sub_type == 'narrowest_span':
            winner_types = params.get('winner_types', [params.get('winner_type', '?')])
            if len(winner_types) > 1:
                return (f"The narrowest local events in {metric} are "
                        f"{' and '.join(winner_types)}, each spanning "
                        f"{params['winner_span']} timesteps.")
            return (f"The narrowest local event in {metric} is "
                    f"{winner_types[0]} spanning "
                    f"{params['winner_span']} timesteps.")

        return verdict

    # ------------------------------------------------------------------ #
    # Family 11E — Segment enumeration QA                                  #
    # ------------------------------------------------------------------ #

    def _generate_segment_enumeration_qa(
        self,
        attributes: Dict,
        metric: str,
        series_index=None,
    ) -> None:
        """Enumeration QA over trend segments: count/filter/extreme."""
        trend_list = [
            s for s in attributes.get('trend_list', [])
            if isinstance(s, (list, tuple)) and len(s) >= 3
        ]
        if len(trend_list) < 2:
            return

        field_key = 'series_index' if series_index is not None else 'timeseries'
        field_idx = series_index if series_index is not None else 0

        sub_type_weights = {
            'segment_count_all': 0.20,
            'segment_count_by_type': 0.30,
            'segment_longest': 0.25,
            'segment_shortest': 0.25,
        }
        sub_types = list(sub_type_weights.keys())
        weights = list(sub_type_weights.values())

        if self.config.debug:
            n_generate = len(sub_types)
        else:
            n_generate = random.randint(1, 2)
        chosen = []
        remaining = list(range(len(sub_types)))
        for _ in range(n_generate):
            if not remaining:
                break
            pool_dict = {sub_types[i]: weights[i] for i in remaining}
            pick = weighted_random_choice(pool_dict)
            idx = sub_types.index(pick)
            chosen.append(pick)
            remaining.remove(idx)

        for sub_type in chosen:
            try:
                params, question = self._build_segment_enum_params(
                    sub_type, trend_list, metric
                )
                if params is None:
                    continue
                think_str, verdict = build_segment_enumeration_thought(
                    attributes, metric, sub_type, params
                )
            except Exception:
                logger.warning("QA generation failed", exc_info=True)
                continue

            answer_text = self._segment_enum_answer_text(
                sub_type, verdict, params, metric
            )
            self.questions.append(_fix_article(question))
            self.answers.append(think_str + answer_text)
            self.llm_prompts.append([])
            self.fields.append({field_key: [field_idx]})
            self.qa_types.append(QAType.SEGMENT_MASK)
            self.eval_tasks.append("segment_enumeration")
            self.eval_metadatas.append({
                "sub_type": sub_type,
                "verdict": verdict,
                "length": self.seq_len,
            })

    def _build_segment_enum_params(self, sub_type, trend_list, metric):
        """Build params dict and question string for segment enumeration sub-type."""

        if sub_type == 'segment_count_all':
            count = len(trend_list)
            q = random.choice(_SEGMENT_COUNT_ALL_QUESTIONS).format(
                metric=metric
            )
            return {'verdict': str(count)}, q

        elif sub_type == 'segment_count_by_type':
            types = [s[0] for s in trend_list]
            target_type = random.choice(types)
            count = sum(1 for t in types if t == target_type)
            q = random.choice(_SEGMENT_COUNT_BY_TYPE_QUESTIONS).format(
                metric=metric, target_type=target_type
            )
            return {'target_type': target_type, 'verdict': str(count)}, q

        elif sub_type == 'segment_longest':
            durations = []
            for seg in trend_list:
                dur = seg[2] - seg[1]
                durations.append((seg, dur))
            best_dur = max(d for _, d in durations)
            winner_segs = [seg for seg, d in durations if d == best_dur]
            q = random.choice(_SEGMENT_LONGEST_QUESTIONS).format(
                metric=metric
            )
            return {
                'winner_segs': list(winner_segs),
                'winner_dur': best_dur,
                'verdict': str(best_dur),
            }, q

        elif sub_type == 'segment_shortest':
            durations = []
            for seg in trend_list:
                dur = seg[2] - seg[1]
                durations.append((seg, dur))
            best_dur = min(d for _, d in durations)
            winner_segs = [seg for seg, d in durations if d == best_dur]
            q = random.choice(_SEGMENT_SHORTEST_QUESTIONS).format(
                metric=metric
            )
            return {
                'winner_segs': list(winner_segs),
                'winner_dur': best_dur,
                'verdict': str(best_dur),
            }, q

        return None, None

    @staticmethod
    def _segment_enum_answer_text(sub_type, verdict, params, metric):
        """Build human-readable answer for segment enumeration QA."""
        if sub_type == 'segment_count_all':
            return f"There {pluralize(verdict, 'trend segment')} in {metric}."

        elif sub_type == 'segment_count_by_type':
            noun = f"{params['target_type']} trend segment"
            return f"There {pluralize(verdict, noun)} in {metric}."

        elif sub_type == 'segment_longest':
            winner_segs = params.get('winner_segs', [])
            if len(winner_segs) > 1:
                segs_str = " and ".join(
                    f"{t} ({s}, {e})" for t, s, e in winner_segs
                )
                return (f"The longest trend segments in {metric} are "
                        f"{segs_str}, each spanning {params['winner_dur']} "
                        f"timesteps.")
            t, s, e = winner_segs[0]
            return (f"The longest trend segment in {metric} is "
                    f"{t} ({s}, {e}) spanning {params['winner_dur']} "
                    f"timesteps.")

        elif sub_type == 'segment_shortest':
            winner_segs = params.get('winner_segs', [])
            if len(winner_segs) > 1:
                segs_str = " and ".join(
                    f"{t} ({s}, {e})" for t, s, e in winner_segs
                )
                return (f"The shortest trend segments in {metric} are "
                        f"{segs_str}, each spanning {params['winner_dur']} "
                        f"timesteps.")
            t, s, e = winner_segs[0]
            return (f"The shortest trend segment in {metric} is "
                    f"{t} ({s}, {e}) spanning {params['winner_dur']} "
                    f"timesteps.")

        return verdict

    # ------------------------------------------------------------------ #
    # Transition enumeration QA                                            #
    # ------------------------------------------------------------------ #

    def _generate_transition_enumeration_qa(
        self,
        attributes: Dict,
        metric: str,
        series_index=None,
    ) -> None:
        """Enumeration QA over adjacent trend segment transitions."""
        trend_list = [
            s for s in attributes.get('trend_list', [])
            if isinstance(s, (list, tuple)) and len(s) >= 3
        ]
        # Need at least 3 segments (2 transitions) for non-trivial questions
        if len(trend_list) < 3:
            return

        field_key = 'series_index' if series_index is not None else 'timeseries'
        field_idx = series_index if series_index is not None else 0

        # Build transitions for feasibility checks
        transitions = [
            f"{trend_list[i][0]}→{trend_list[i + 1][0]}"
            for i in range(len(trend_list) - 1)
        ]

        sub_type_weights = {
            'transition_count': 0.25,
            'transition_count_by_type': 0.40,
            'most_common_transition': 0.35,
        }

        sub_types = list(sub_type_weights.keys())
        weights = list(sub_type_weights.values())

        if self.config.debug:
            n_generate = len(sub_types)
        else:
            n_generate = random.randint(1, min(2, len(sub_types)))
        chosen = []
        remaining = list(range(len(sub_types)))
        for _ in range(n_generate):
            if not remaining:
                break
            pool_dict = {sub_types[i]: weights[i] for i in remaining}
            pick = weighted_random_choice(pool_dict)
            idx = sub_types.index(pick)
            chosen.append(pick)
            remaining.remove(idx)

        for sub_type in chosen:
            try:
                params, question = self._build_transition_enum_params(
                    sub_type, trend_list, transitions, metric
                )
                if params is None:
                    continue
                think_str, verdict = build_transition_enumeration_thought(
                    attributes, metric, sub_type, params
                )
            except Exception:
                logger.warning("QA generation failed", exc_info=True)
                continue

            answer_text = self._transition_enum_answer_text(
                sub_type, verdict, params, metric
            )
            self.questions.append(_fix_article(question))
            self.answers.append(think_str + answer_text)
            self.llm_prompts.append([])
            self.fields.append({field_key: [field_idx]})
            self.qa_types.append(QAType.SEGMENT_MASK)
            self.eval_tasks.append("transition_enumeration")
            self.eval_metadatas.append({
                "sub_type": sub_type,
                "verdict": verdict,
                "length": self.seq_len,
            })

    def _build_transition_enum_params(self, sub_type, trend_list, transitions, metric):
        """Build params dict and question string for transition enumeration."""

        if sub_type == 'transition_count':
            count = len(transitions)
            q = random.choice(_TRANSITION_COUNT_QUESTIONS).format(
                metric=metric
            )
            return {'verdict': str(count)}, q

        elif sub_type == 'transition_count_by_type':
            target = random.choice(transitions)
            count = sum(1 for tr in transitions if tr == target)
            type_a, type_b = target.split('→')
            q = random.choice(_TRANSITION_COUNT_BY_TYPE_QUESTIONS).format(
                metric=metric, type_a=type_a, type_b=type_b
            )
            return {
                'target_transition': target,
                'verdict': str(count),
            }, q

        elif sub_type == 'most_common_transition':
            counts = Counter(transitions)
            best_count = counts.most_common(1)[0][1]
            winners = [tr for tr, c in counts.items() if c == best_count]
            verdict = ", ".join(winners)
            q = random.choice(_TRANSITION_MOST_COMMON_QUESTIONS).format(
                metric=metric
            )
            return {
                'winners': winners,
                'winner_count': best_count,
                'verdict': verdict,
            }, q

        return None, None

    @staticmethod
    def _transition_enum_answer_text(sub_type, verdict, params, metric):
        """Build human-readable answer for transition enumeration QA."""
        if sub_type == 'transition_count':
            return f"There {pluralize(verdict, 'trend transition')} in {metric}."

        elif sub_type == 'transition_count_by_type':
            target = params['target_transition']
            return (f"There {pluralize(verdict, f'{target} transition')} in "
                    f"{metric}.")

        elif sub_type == 'most_common_transition':
            winners = params.get('winners', [verdict])
            count = params['winner_count']
            unit = "time" if int(count) == 1 else "times"
            if len(winners) > 1:
                return (f"The most common transitions in {metric} are "
                        f"{' and '.join(winners)}, each occurring "
                        f"{count} {unit}.")
            return (f"The most common transition in {metric} is "
                    f"{verdict}, occurring {count} {unit}.")

        return verdict

    # ------------------------------------------------------------------ #
    # Event-segment enumeration QA                                         #
    # ------------------------------------------------------------------ #

    def _generate_event_segment_enumeration_qa(
        self,
        attributes: Dict,
        metric: str,
        series_index=None,
    ) -> None:
        """Enumeration QA: local events cross-referenced against trend segments."""
        local_events = attributes['local']
        trend_list = [
            s for s in attributes['trend_list']
            if isinstance(s, (list, tuple)) and len(s) >= 3
        ]
        # Need events AND multiple segments for non-trivial questions
        if not local_events or len(trend_list) < 2:
            return

        field_key = 'series_index' if series_index is not None else 'timeseries'
        field_idx = series_index if series_index is not None else 0

        sub_type_weights = {
            'events_in_trend_type': 0.35,
            'segment_with_most_events': 0.35,
            'trend_type_with_most_events': 0.30,
        }

        sub_types = list(sub_type_weights.keys())
        weights = list(sub_type_weights.values())

        if self.config.debug:
            n_generate = len(sub_types)
        else:
            n_generate = random.randint(1, 2)
        chosen = []
        remaining = list(range(len(sub_types)))
        for _ in range(n_generate):
            if not remaining:
                break
            pool_dict = {sub_types[i]: weights[i] for i in remaining}
            pick = weighted_random_choice(pool_dict)
            idx = sub_types.index(pick)
            chosen.append(pick)
            remaining.remove(idx)

        for sub_type in chosen:
            try:
                params, question = self._build_event_segment_enum_params(
                    sub_type, attributes, trend_list, local_events, metric
                )
                if params is None:
                    continue
                think_str, verdict = build_event_segment_enumeration_thought(
                    attributes, metric, sub_type, params
                )
            except Exception:
                logger.warning("QA generation failed", exc_info=True)
                continue

            answer_text = self._event_segment_enum_answer_text(
                sub_type, verdict, params, metric
            )
            self.questions.append(_fix_article(question))
            self.answers.append(think_str + answer_text)
            self.llm_prompts.append([])
            self.fields.append({field_key: [field_idx]})
            self.qa_types.append(QAType.SEGMENT_MASK)
            self.eval_tasks.append("event_segment_enumeration")
            self.eval_metadatas.append({
                "sub_type": sub_type,
                "verdict": verdict,
                "length": self.seq_len,
            })

    def _build_event_segment_enum_params(
        self, sub_type, attributes, trend_list, local_events, metric
    ):
        """Build params dict and question string for event-segment enumeration."""

        def _seg_for_pos(pos):
            for seg in trend_list:
                if seg[1] <= pos <= seg[2]:
                    return seg[0], seg[1], seg[2]
            return None, None, None

        if sub_type == 'events_in_trend_type':
            # Pick a trend type that actually exists
            seg_types = list(dict.fromkeys(s[0] for s in trend_list))
            target_type = random.choice(seg_types)
            count = 0
            for ev in local_events:
                seg_type, _, _ = _seg_for_pos(ev['position_start'])
                if seg_type == target_type:
                    count += 1
            q = random.choice(_EVENT_SEG_COUNT_IN_TREND_TYPE_QUESTIONS).format(
                metric=metric, target_type=target_type
            )
            return {'target_type': target_type, 'verdict': str(count)}, q

        elif sub_type == 'segment_with_most_events':
            seg_counts = []
            for seg in trend_list:
                s_type, s_start, s_end = seg[0], seg[1], seg[2]
                count = sum(
                    1 for ev in local_events
                    if s_start <= ev['position_start'] <= s_end
                )
                seg_counts.append((s_type, s_start, s_end, count))
            best_count = max(c for _, _, _, c in seg_counts)
            winners = [(t, s, e) for t, s, e, c in seg_counts if c == best_count]
            # Use format_seg_display for canonical verdict
            verdict = ", ".join(
                f"({t}, {s}, {e})" for t, s, e in winners
            )
            q = random.choice(_EVENT_SEG_MOST_EVENTS_QUESTIONS).format(
                metric=metric
            )
            return {
                'winners': winners,
                'winner_count': best_count,
                'verdict': verdict,
            }, q

        elif sub_type == 'trend_type_with_most_events':
            type_counts: Dict[str, int] = {}
            for ev in local_events:
                seg_type, _, _ = _seg_for_pos(ev['position_start'])
                if seg_type is not None:
                    type_counts[seg_type] = type_counts.get(seg_type, 0) + 1
            all_types = list(dict.fromkeys(s[0] for s in trend_list))
            for t in all_types:
                if t not in type_counts:
                    type_counts[t] = 0
            best_count = max(type_counts.values())
            winners = [t for t in all_types if type_counts[t] == best_count]
            verdict = ", ".join(winners)
            q = random.choice(_EVENT_SEG_TREND_TYPE_MOST_EVENTS_QUESTIONS).format(
                metric=metric
            )
            return {
                'winners': winners,
                'winner_count': best_count,
                'verdict': verdict,
            }, q

        return None, None

    @staticmethod
    def _event_segment_enum_answer_text(sub_type, verdict, params, metric):
        """Build human-readable answer for event-segment enumeration QA."""
        if sub_type == 'events_in_trend_type':
            return (f"There {pluralize(verdict, 'local event')} during "
                    f"{params['target_type']} trend phases in {metric}.")

        elif sub_type == 'segment_with_most_events':
            winners = params['winners']
            count = params['winner_count']
            if len(winners) > 1:
                segs_str = " and ".join(
                    f"({t}, {s}, {e})" for t, s, e in winners
                )
                return (f"The segments {segs_str} in {metric} are tied for "
                        f"the most local events, each with {count} events.")
            t, s, e = winners[0]
            return (f"The segment ({t}, {s}, {e}) in {metric} contains "
                    f"the most local events with {count} events.")

        elif sub_type == 'trend_type_with_most_events':
            winners = params['winners']
            count = params['winner_count']
            if len(winners) > 1:
                return (f"The trend types {' and '.join(winners)} in {metric} "
                        f"are tied for the most local events, each with "
                        f"{count} events.")
            return (f"{verdict} trend phases in {metric} contain the "
                    f"most local events with {count} events total.")

        return verdict

    # ------------------------------------------------------------------ #
    # Temporal position QA                                                 #
    # ------------------------------------------------------------------ #

    def _generate_temporal_position_qa(
        self,
        attributes: Dict,
        metric: str,
        series_index=None,
    ) -> None:
        """First/last event position and inter-event gap QA."""
        local_events = attributes['local']
        if len(local_events) < 2:
            return

        field_key = 'series_index' if series_index is not None else 'timeseries'
        field_idx = series_index if series_index is not None else 0

        # Need 3+ unique positions for gap sub-types to be interesting
        unique_positions = sorted(set(ev['position_start'] for ev in local_events))
        gap_ok = len(unique_positions) >= 3

        sub_type_weights = {
            'first_event': 0.20,
            'last_event': 0.20,
        }
        if gap_ok:
            sub_type_weights['largest_gap'] = 0.30
            sub_type_weights['smallest_gap'] = 0.30

        sub_types = list(sub_type_weights.keys())
        weights = list(sub_type_weights.values())

        if self.config.debug:
            n_generate = len(sub_types)
        else:
            n_generate = random.randint(1, 2)
        chosen = []
        remaining = list(range(len(sub_types)))
        for _ in range(n_generate):
            if not remaining:
                break
            pool_dict = {sub_types[i]: weights[i] for i in remaining}
            pick = weighted_random_choice(pool_dict)
            idx = sub_types.index(pick)
            chosen.append(pick)
            remaining.remove(idx)

        for sub_type in chosen:
            try:
                params, question = self._build_temporal_position_params(
                    sub_type, local_events, metric
                )
                if params is None:
                    continue
                think_str, verdict = build_temporal_position_thought(
                    attributes, metric, sub_type, params
                )
            except Exception:
                logger.warning("QA generation failed", exc_info=True)
                continue

            answer_text = self._temporal_position_answer_text(
                sub_type, verdict, params, metric
            )
            self.questions.append(_fix_article(question))
            self.answers.append(think_str + answer_text)
            self.llm_prompts.append([])
            self.fields.append({field_key: [field_idx]})
            self.qa_types.append(QAType.SEGMENT_MASK)
            self.eval_tasks.append("temporal_position")
            self.eval_metadatas.append({
                "sub_type": sub_type,
                "verdict": verdict,
                "length": self.seq_len,
            })

    @staticmethod
    def _build_temporal_position_params(sub_type, local_events, metric):
        """Build params dict and question string for temporal position."""
        positions = sorted(set(ev['position_start'] for ev in local_events))

        if sub_type == 'first_event':
            q = random.choice(_FIRST_EVENT_QUESTIONS).format(metric=metric)
            return {'verdict': str(positions[0])}, q

        elif sub_type == 'last_event':
            q = random.choice(_LAST_EVENT_QUESTIONS).format(metric=metric)
            return {'verdict': str(positions[-1])}, q

        elif sub_type == 'largest_gap':
            gaps = [positions[j + 1] - positions[j] for j in range(len(positions) - 1)]
            q = random.choice(_LARGEST_GAP_QUESTIONS).format(metric=metric)
            return {'verdict': str(max(gaps))}, q

        elif sub_type == 'smallest_gap':
            gaps = [positions[j + 1] - positions[j] for j in range(len(positions) - 1)]
            q = random.choice(_SMALLEST_GAP_QUESTIONS).format(metric=metric)
            return {'verdict': str(min(gaps))}, q

        return None, None

    @staticmethod
    def _temporal_position_answer_text(sub_type, verdict, params, metric):
        """Build human-readable answer for temporal position QA."""
        if sub_type == 'first_event':
            return f"The first local event in {metric} occurs at position {verdict}."
        elif sub_type == 'last_event':
            return f"The last local event in {metric} occurs at position {verdict}."
        elif sub_type == 'largest_gap':
            return (f"The largest gap between consecutive local events "
                    f"in {metric} is {verdict} timesteps.")
        elif sub_type == 'smallest_gap':
            return (f"The smallest gap between consecutive local events "
                    f"in {metric} is {verdict} timesteps.")
        return verdict

    # ------------------------------------------------------------------ #
    # Duration proportion QA                                               #
    # ------------------------------------------------------------------ #

    def _generate_duration_proportion_qa(
        self,
        attributes: Dict,
        metric: str,
        series_index=None,
    ) -> None:
        """Proportion of series spent in each trend type."""
        trend_list = [
            s for s in attributes['trend_list']
            if isinstance(s, (list, tuple)) and len(s) >= 3
        ]
        if len(trend_list) < 2:
            return
        # Need at least 2 distinct trend types for meaningful proportions
        distinct_types = set(s[0] for s in trend_list)
        if len(distinct_types) < 2:
            return

        field_key = 'series_index' if series_index is not None else 'timeseries'
        field_idx = series_index if series_index is not None else 0

        sub_type_weights = {
            'duration_by_type': 0.25,
            'longest_total_duration': 0.20,
            'shortest_total_duration': 0.20,
            'duration_comparison': 0.20,
            'duration_difference': 0.15,
        }
        sub_types = list(sub_type_weights.keys())
        weights = list(sub_type_weights.values())

        if self.config.debug:
            n_generate = len(sub_types)
        else:
            n_generate = 1
        chosen = []
        remaining = list(range(len(sub_types)))
        for _ in range(n_generate):
            if not remaining:
                break
            pool_dict = {sub_types[i]: weights[i] for i in remaining}
            pick = weighted_random_choice(pool_dict)
            idx = sub_types.index(pick)
            chosen.append(pick)
            remaining.remove(idx)

        for sub_type in chosen:
            try:
                params, question = self._build_duration_proportion_params(
                    sub_type, trend_list, metric
                )
                if params is None:
                    continue
                think_str, verdict = build_duration_proportion_thought(
                    attributes, metric, sub_type, params
                )
            except Exception:
                logger.warning("QA generation failed", exc_info=True)
                continue

            answer_text = self._duration_proportion_answer_text(
                sub_type, verdict, params, metric
            )
            self.questions.append(_fix_article(question))
            self.answers.append(think_str + answer_text)
            self.llm_prompts.append([])
            self.fields.append({field_key: [field_idx]})
            self.qa_types.append(QAType.SEGMENT_MASK)
            self.eval_tasks.append("duration_proportion")
            self.eval_metadatas.append({
                "sub_type": sub_type,
                "verdict": verdict,
                "length": self.seq_len,
            })

    def _build_duration_proportion_params(self, sub_type, trend_list, metric):
        """Build params dict and question string for duration proportion."""
        # Compute type totals
        type_parts: Dict[str, int] = {}
        total = 0
        for seg in trend_list:
            dur = seg[2] - seg[1]
            total += dur
            type_parts[seg[0]] = type_parts.get(seg[0], 0) + dur
        all_types = list(dict.fromkeys(s[0] for s in trend_list))

        if sub_type == 'duration_by_type':
            target_type = random.choice(all_types)
            dur = type_parts[target_type]
            q = random.choice(_DURATION_BY_TYPE_QUESTIONS).format(
                metric=metric, target_type=target_type
            )
            return {
                'target_type': target_type,
                'verdict': str(dur),
            }, q

        elif sub_type == 'longest_total_duration':
            best = max(type_parts.values())
            winners = [t for t in all_types if type_parts[t] == best]
            verdict = ", ".join(winners)
            q = random.choice(_LONGEST_DURATION_QUESTIONS).format(metric=metric)
            return {
                'winners': winners,
                'winner_duration': best,
                'verdict': verdict,
            }, q

        elif sub_type == 'shortest_total_duration':
            worst = min(type_parts.values())
            winners = [t for t in all_types if type_parts[t] == worst]
            verdict = ", ".join(winners)
            q = random.choice(_SHORTEST_DURATION_QUESTIONS).format(metric=metric)
            return {
                'winners': winners,
                'winner_duration': worst,
                'verdict': verdict,
            }, q

        elif sub_type == 'duration_comparison':
            if len(all_types) < 2:
                return None, None
            type_a, type_b = random.sample(all_types, 2)
            da, db = type_parts[type_a], type_parts[type_b]
            if da > db:
                verdict = type_a
            elif db > da:
                verdict = type_b
            else:
                verdict = "equal"
            q = random.choice(_DURATION_COMPARISON_QUESTIONS).format(
                metric=metric, type_a=type_a, type_b=type_b
            )
            return {
                'type_a': type_a,
                'type_b': type_b,
                'verdict': verdict,
            }, q

        elif sub_type == 'duration_difference':
            if len(all_types) < 2:
                return None, None
            type_a, type_b = random.sample(all_types, 2)
            diff = abs(type_parts[type_a] - type_parts[type_b])
            q = random.choice(_DURATION_DIFFERENCE_QUESTIONS).format(
                metric=metric, type_a=type_a, type_b=type_b
            )
            return {
                'type_a': type_a,
                'type_b': type_b,
                'verdict': str(diff),
            }, q

        return None, None

    @staticmethod
    def _duration_proportion_answer_text(sub_type, verdict, params, metric):
        """Build human-readable answer for duration QA (count-based, no ratio)."""
        if sub_type == 'duration_by_type':
            return (f"The total duration of {params['target_type']} trend "
                    f"segments in {metric} is {verdict} timesteps.")

        elif sub_type == 'longest_total_duration':
            winners = params['winners']
            dur = params['winner_duration']
            if len(winners) > 1:
                return (f"The trend types {' and '.join(winners)} in {metric} "
                        f"are tied for the longest total duration at {dur} timesteps each.")
            return (f"{winners[0]} has the longest total duration in "
                    f"{metric} at {dur} timesteps.")

        elif sub_type == 'shortest_total_duration':
            winners = params['winners']
            dur = params['winner_duration']
            if len(winners) > 1:
                return (f"The trend types {' and '.join(winners)} in {metric} "
                        f"are tied for the shortest total duration at {dur} timesteps each.")
            return (f"{winners[0]} has the shortest total duration in "
                    f"{metric} at {dur} timesteps.")

        elif sub_type == 'duration_comparison':
            a, b = params['type_a'], params['type_b']
            if verdict == "equal":
                return (f"{metric} spends equal time in {a} and "
                        f"{b} trend segments.")
            return (f"{metric} spends more time in {verdict} than in "
                    f"{a if verdict == b else b} trend segments.")

        elif sub_type == 'duration_difference':
            a, b = params['type_a'], params['type_b']
            return (f"The difference in total duration between {a} and "
                    f"{b} trend segments in {metric} is {verdict} timesteps.")

        return verdict

    # ------------------------------------------------------------------ #
    # Cross-metric enumeration QA                                          #
    # ------------------------------------------------------------------ #

    _CROSS_METRIC_QUESTION_TEMPLATES = {
        'which_have_local': _CROSS_WHICH_HAVE_LOCAL_QUESTIONS,
        'which_have_trend_type': _CROSS_WHICH_HAVE_TREND_TYPE_QUESTIONS,
        'all_same_trend': _CROSS_ALL_SAME_TREND_QUESTIONS,
        'most_local_events': _CROSS_MOST_LOCAL_EVENTS_QUESTIONS,
        'highest_amplitude_across': _CROSS_HIGHEST_AMPLITUDE_QUESTIONS,
        'most_trend_segments': _CROSS_MOST_TREND_SEGMENTS_QUESTIONS,
        'longest_segment_across': _CROSS_LONGEST_SEGMENT_QUESTIONS,
        'noisiest_metric': _CROSS_NOISIEST_METRIC_QUESTIONS,
        'quietest_metric': _CROSS_QUIETEST_METRIC_QUESTIONS,
        'widest_range': _CROSS_WIDEST_RANGE_QUESTIONS,
        'highest_mean': _CROSS_HIGHEST_MEAN_QUESTIONS,
        'lowest_mean': _CROSS_LOWEST_MEAN_QUESTIONS,
        'highest_max': _CROSS_HIGHEST_MAX_QUESTIONS,
        'lowest_min': _CROSS_LOWEST_MIN_QUESTIONS,
        'highest_std': _CROSS_HIGHEST_STD_QUESTIONS,
        'lowest_std': _CROSS_LOWEST_STD_QUESTIONS,
        'earliest_event_across': _CROSS_EARLIEST_EVENT_QUESTIONS,
        'latest_event_across': _CROSS_LATEST_EVENT_QUESTIONS,
        'largest_gap_across': _CROSS_LARGEST_GAP_QUESTIONS,
    }

    def _generate_cross_metric_enumeration_qa(
        self,
        attributes_list: List[Dict],
        metrics: List[str],
    ) -> None:
        """Cross-metric enumeration QA: filter/argmax/argmin across all metrics."""
        if len(metrics) < 2:
            return

        has_local = any(len(a.get('local', [])) > 0 for a in attributes_list)

        # Build sub-type pool — LOCAL-only sub-types gated on local event presence
        sub_type_weights: Dict[str, float] = {
            'which_have_trend_type': 0.08,
            'all_same_trend': 0.06,
            'most_trend_segments': 0.07,
            'longest_segment_across': 0.08,
            'noisiest_metric': 0.07,
            'quietest_metric': 0.06,
            'widest_range': 0.07,
            'highest_mean': 0.07,
            'lowest_mean': 0.06,
            'highest_max': 0.07,
            'lowest_min': 0.06,
            'highest_std': 0.07,
            'lowest_std': 0.06,
        }
        if has_local:
            sub_type_weights['which_have_local'] = 0.10
            sub_type_weights['most_local_events'] = 0.08
            sub_type_weights['earliest_event_across'] = 0.07
            sub_type_weights['latest_event_across'] = 0.07
            # highest_amplitude_across needs 2+ metrics with amplitude data
            metrics_with_amp = sum(
                1 for a in attributes_list
                if any('amplitude' in ev for ev in a.get('local', []))
            )
            if metrics_with_amp >= 2:
                sub_type_weights['highest_amplitude_across'] = 0.10
            # largest_gap_across needs 2+ metrics with ≥2 unique positions
            metrics_with_gaps = sum(
                1 for a in attributes_list
                if len(set(ev['position_start'] for ev in a['local'])) >= 2
            )
            if metrics_with_gaps >= 2:
                sub_type_weights['largest_gap_across'] = 0.08

        sub_types = list(sub_type_weights.keys())
        weights = list(sub_type_weights.values())

        if self.config.debug:
            n_generate = len(sub_types)
        else:
            n_generate = random.randint(1, min(3, len(sub_types)))
        chosen = []
        remaining = list(range(len(sub_types)))
        for _ in range(n_generate):
            if not remaining:
                break
            pool_dict = {sub_types[i]: weights[i] for i in remaining}
            pick = weighted_random_choice(pool_dict)
            idx = sub_types.index(pick)
            chosen.append(pick)
            remaining.remove(idx)

        for sub_type in chosen:
            try:
                params, question = self._build_cross_metric_enum_params(
                    sub_type, attributes_list, metrics
                )
                if params is None:
                    continue
                think_str, verdict = build_cross_metric_enumeration_thought(
                    attributes_list, metrics, sub_type, params
                )
            except Exception:
                logger.warning("QA generation failed", exc_info=True)
                continue

            answer_text = self._cross_metric_enum_answer_text(
                sub_type, verdict, params, metrics
            )
            self.questions.append(_fix_article(question))
            self.answers.append(think_str + answer_text)
            self.llm_prompts.append([])
            self.fields.append({'series_index': list(range(len(metrics)))})
            self.qa_types.append(QAType.SEGMENT_MASK)
            self.eval_tasks.append("cross_metric_enumeration")
            self.eval_metadatas.append({
                "sub_type": sub_type,
                "verdict": verdict,
                "length": self.seq_len,
            })

    def _build_cross_metric_enum_params(
        self,
        sub_type: str,
        attributes_list: List[Dict],
        metrics: List[str],
    ):
        """Build params dict and question string for cross-metric enumeration."""
        templates = self._CROSS_METRIC_QUESTION_TEMPLATES[sub_type]

        if sub_type == 'which_have_local':
            per_metric = [{'count': len(a.get('local', []))} for a in attributes_list]
            passing = [m for m, a in zip(metrics, attributes_list)
                       if len(a.get('local', [])) > 0]
            if not passing:
                return None, None
            verdict = ", ".join(passing)
            q = random.choice(templates)
            return {'verdict': verdict, 'per_metric': per_metric, 'passing': passing}, q

        elif sub_type == 'which_have_trend_type':
            trend_types = []
            for a in attributes_list:
                trend = a.get('trend', {})
                t = trend.get('type', '?') if isinstance(trend, dict) else '?'
                trend_types.append(t)
            valid_types = [t for t in trend_types if t != '?']
            if not valid_types:
                return None, None
            target_type = random.choice(valid_types)
            passing = [m for m, t in zip(metrics, trend_types) if t == target_type]
            if not passing:
                return None, None
            verdict = ", ".join(passing)
            q = random.choice(templates).format(target_type=target_type)
            return {'verdict': verdict, 'per_metric': trend_types,
                    'target_type': target_type, 'passing': passing}, q

        elif sub_type == 'all_same_trend':
            trend_types = []
            for a in attributes_list:
                trend = a.get('trend', {})
                t = trend.get('type', '?') if isinstance(trend, dict) else '?'
                trend_types.append(t)
            all_same = len(set(trend_types)) == 1
            verdict = "yes" if all_same else "no"
            q = random.choice(templates)
            return {'verdict': verdict, 'per_metric': trend_types,
                    'trend_types': trend_types}, q

        elif sub_type == 'most_local_events':
            counts = [len(a.get('local', [])) for a in attributes_list]
            if len(set(counts)) == 1:
                return None, None  # all identical — no discrimination
            best = max(counts)
            winners = [m for m, c in zip(metrics, counts) if c == best]
            verdict = ", ".join(winners)
            q = random.choice(templates)
            return {'verdict': verdict, 'per_metric': counts,
                    'winners': winners, 'winner_count': best}, q

        elif sub_type == 'highest_amplitude_across':
            per_metric = []
            for a in attributes_list:
                local = a.get('local', [])
                amps = [(ev.get('amplitude', 0.0), ev.get('type', '?'))
                        for ev in local if 'amplitude' in ev]
                if amps:
                    best = max(amps, key=lambda x: x[0])
                    per_metric.append({'amp': best[0], 'type': best[1]})
                else:
                    per_metric.append({'amp': 0.0, 'type': 'none'})
            amp_vals = [pm['amp'] for pm in per_metric]
            if len(set(amp_vals)) == 1:
                return None, None
            best_amp = max(amp_vals)
            winners = [m for m, v in zip(metrics, amp_vals) if v == best_amp]
            verdict = ", ".join(winners)
            q = random.choice(templates)
            return {'verdict': verdict, 'per_metric': per_metric,
                    'winners': winners, 'winner_amp': best_amp}, q

        elif sub_type == 'most_trend_segments':
            counts = []
            for a in attributes_list:
                tl = [s for s in a.get('trend_list', [])
                      if isinstance(s, (list, tuple)) and len(s) >= 3]
                counts.append(len(tl))
            if len(set(counts)) == 1:
                return None, None
            best = max(counts)
            winners = [m for m, c in zip(metrics, counts) if c == best]
            verdict = ", ".join(winners)
            q = random.choice(templates)
            return {'verdict': verdict, 'per_metric': counts,
                    'winners': winners, 'winner_count': best}, q

        elif sub_type == 'longest_segment_across':
            per_metric = []
            for a in attributes_list:
                tl = [s for s in a.get('trend_list', [])
                      if isinstance(s, (list, tuple)) and len(s) >= 3]
                if tl:
                    best = max(tl, key=lambda s: s[2] - s[1])
                    dur = best[2] - best[1]
                    per_metric.append({
                        'type': best[0],
                        'start': best[1],
                        'end': best[2],
                        'dur': dur,
                    })
                else:
                    per_metric.append({'type': '?', 'start': 0, 'end': 0, 'dur': 0})
            durs = [pm['dur'] for pm in per_metric]
            if len(set(durs)) == 1:
                return None, None
            best_dur = max(durs)
            winners = [m for m, d in zip(metrics, durs) if d == best_dur]
            verdict = ", ".join(winners)
            q = random.choice(templates)
            winner_segs = [pm for pm, d in zip(per_metric, durs) if d == best_dur]
            return {'verdict': verdict, 'per_metric': per_metric,
                    'winners': winners, 'winner_seg': winner_segs[0],
                    'winner_segs': winner_segs}, q

        elif sub_type in ('noisiest_metric', 'quietest_metric'):
            strengths = [a['noise']['strength'] for a in attributes_list]
            if len(set(strengths)) == 1:
                return None, None
            if sub_type == 'noisiest_metric':
                best = max(strengths)
            else:
                best = min(strengths)
            winners = [m for m, s in zip(metrics, strengths) if s == best]
            verdict = ", ".join(winners)
            q = random.choice(templates)
            return {'verdict': verdict, 'per_metric': strengths,
                    'winners': winners, 'winner_strength': best}, q

        elif sub_type == 'widest_range':
            ranges = [a['statistics']['range'] for a in attributes_list]
            if len(set(ranges)) == 1:
                return None, None
            best = max(ranges)
            winners = [m for m, r in zip(metrics, ranges) if r == best]
            verdict = ", ".join(winners)
            q = random.choice(templates)
            return {'verdict': verdict, 'per_metric': ranges,
                    'winners': winners, 'winner_range': best}, q

        elif sub_type in ('highest_mean', 'lowest_mean',
                          'highest_max', 'lowest_min',
                          'highest_std', 'lowest_std'):
            # Map sub_type to statistics key
            _STAT_KEY = {
                'highest_mean': 'mean', 'lowest_mean': 'mean',
                'highest_max': 'max', 'lowest_min': 'min',
                'highest_std': 'std', 'lowest_std': 'std',
            }
            stat_key = _STAT_KEY[sub_type]
            vals = [a['statistics'][stat_key] for a in attributes_list]
            if len(set(vals)) == 1:
                return None, None
            is_max = sub_type.startswith('highest')
            best = max(vals) if is_max else min(vals)
            winners = [m for m, v in zip(metrics, vals) if v == best]
            verdict = ", ".join(winners)
            q = random.choice(templates)
            return {'verdict': verdict, 'per_metric': vals,
                    'winners': winners, 'winner_val': best,
                    'stat_key': stat_key}, q

        elif sub_type in ('earliest_event_across', 'latest_event_across'):
            per_metric = []
            for a in attributes_list:
                local = a['local']
                if local:
                    positions = sorted(ev['position_start'] for ev in local)
                    per_metric.append(positions)
                else:
                    per_metric.append([])
            # Extract first or last position per metric
            if sub_type == 'earliest_event_across':
                vals = [(m, ps[0]) for m, ps in zip(metrics, per_metric) if ps]
                if len(vals) < 2:
                    return None, None
                best = min(v for _, v in vals)
            else:
                vals = [(m, ps[-1]) for m, ps in zip(metrics, per_metric) if ps]
                if len(vals) < 2:
                    return None, None
                best = max(v for _, v in vals)
            winners = [m for m, v in vals if v == best]
            verdict = ", ".join(winners)
            q = random.choice(templates)
            return {'verdict': verdict, 'winners': winners,
                    'winner_pos': best}, q

        elif sub_type == 'largest_gap_across':
            per_metric_gaps = []
            for a in attributes_list:
                local = a['local']
                positions = sorted(set(ev['position_start'] for ev in local))
                if len(positions) >= 2:
                    gaps = [positions[j + 1] - positions[j]
                            for j in range(len(positions) - 1)]
                    per_metric_gaps.append(max(gaps))
                else:
                    per_metric_gaps.append(None)
            vals = [(m, g) for m, g in zip(metrics, per_metric_gaps) if g is not None]
            if len(vals) < 2:
                return None, None
            best = max(v for _, v in vals)
            winners = [m for m, v in vals if v == best]
            verdict = ", ".join(winners)
            q = random.choice(templates)
            return {'verdict': verdict, 'winners': winners,
                    'winner_gap': best}, q

        return None, None

    @staticmethod
    def _cross_metric_enum_answer_text(sub_type, verdict, params, metrics):
        """Build human-readable answer for cross-metric enumeration QA."""
        if sub_type == 'which_have_local':
            return (f"The following metrics have local fluctuations: "
                    f"{verdict}.")

        elif sub_type == 'which_have_trend_type':
            return (f"The following metrics have an overall "
                    f"{params['target_type']} trend: {verdict}.")

        elif sub_type == 'all_same_trend':
            if verdict == 'yes':
                t = params['trend_types'][0]
                return (f"Yes, all metrics share the same overall trend: "
                        f"{t}.")
            else:
                # Only mention metrics that were actually scanned (early exit)
                types_seen = set()
                scanned = []
                for m, t in zip(metrics, params['trend_types']):
                    types_seen.add(t)
                    scanned.append((m, t))
                    if len(types_seen) > 1:
                        break
                parts = [f"{m} ({t})" for m, t in scanned]
                return (f"Not all metrics share the same trend. Observed: "
                        f"{', '.join(parts)}.")

        elif sub_type == 'most_local_events':
            winners = params.get('winners', [verdict])
            if len(winners) > 1:
                return (f"{' and '.join(winners)} are tied for the most local events, "
                        f"each with {params['winner_count']} events.")
            return (f"{verdict} has the most local events with "
                    f"{params['winner_count']} events.")

        elif sub_type == 'highest_amplitude_across':
            winners = params.get('winners', [verdict])
            if len(winners) > 1:
                return (f"{' and '.join(winners)} are tied for the highest "
                        f"amplitude of {format_float(params['winner_amp'], 2)}.")
            return (f"{verdict} has the local event with the highest "
                    f"amplitude of {format_float(params['winner_amp'], 2)}.")

        elif sub_type == 'most_trend_segments':
            winners = params.get('winners', [verdict])
            if len(winners) > 1:
                return (f"{' and '.join(winners)} are tied for the most trend segments, "
                        f"each with {params['winner_count']} segments.")
            return (f"{verdict} has the most trend segments with "
                    f"{params['winner_count']} segments.")

        elif sub_type == 'longest_segment_across':
            winners = params.get('winners', [verdict])
            seg = params['winner_seg']
            if len(winners) > 1:
                return (f"{' and '.join(winners)} are tied for the longest trend segment, "
                        f"each spanning {seg['dur']} timesteps.")
            return (f"{verdict} has the longest trend segment: "
                    f"{seg['type']} ({seg['start']}, {seg['end']}) "
                    f"spanning {seg['dur']} timesteps.")

        elif sub_type == 'noisiest_metric':
            winners = params.get('winners', [verdict])
            if len(winners) > 1:
                return (f"{' and '.join(winners)} are tied for the noisiest metric, "
                        f"each with noise strength {format_float(params['winner_strength'], 2)}.")
            return (f"{verdict} is the noisiest metric with noise "
                    f"strength {format_float(params['winner_strength'], 2)}.")

        elif sub_type == 'quietest_metric':
            winners = params.get('winners', [verdict])
            if len(winners) > 1:
                return (f"{' and '.join(winners)} are tied for the quietest metric, "
                        f"each with noise strength {format_float(params['winner_strength'], 2)}.")
            return (f"{verdict} is the quietest metric with noise "
                    f"strength {format_float(params['winner_strength'], 2)}.")

        elif sub_type == 'widest_range':
            winners = params.get('winners', [verdict])
            if len(winners) > 1:
                return (f"{' and '.join(winners)} are tied for the widest value range, "
                        f"each at {format_float(params['winner_range'], 2)}.")
            return (f"{verdict} has the widest value range at "
                    f"{format_float(params['winner_range'], 2)}.")

        elif sub_type in ('highest_mean', 'lowest_mean',
                          'highest_max', 'lowest_min',
                          'highest_std', 'lowest_std'):
            _LABELS = {
                'highest_mean': 'the highest mean',
                'lowest_mean': 'the lowest mean',
                'highest_max': 'the highest peak value',
                'lowest_min': 'the lowest trough value',
                'highest_std': 'the highest standard deviation',
                'lowest_std': 'the lowest standard deviation',
            }
            label = _LABELS[sub_type]
            winners = params.get('winners', [verdict])
            val_str = format_float(params['winner_val'], 2)
            if len(winners) > 1:
                return (f"{' and '.join(winners)} are tied for {label}, "
                        f"each at {val_str}.")
            return f"{verdict} has {label} at {val_str}."

        elif sub_type == 'earliest_event_across':
            winners = params['winners']
            pos = params['winner_pos']
            if len(winners) > 1:
                return (f"{' and '.join(winners)} are tied for the earliest event, "
                        f"each at position {pos}.")
            return f"{verdict} has the earliest local event at position {pos}."

        elif sub_type == 'latest_event_across':
            winners = params['winners']
            pos = params['winner_pos']
            if len(winners) > 1:
                return (f"{' and '.join(winners)} are tied for the latest event, "
                        f"each at position {pos}.")
            return f"{verdict} has the latest local event at position {pos}."

        elif sub_type == 'largest_gap_across':
            winners = params['winners']
            gap = params['winner_gap']
            if len(winners) > 1:
                return (f"{' and '.join(winners)} are tied for the largest inter-event gap, "
                        f"each with a gap of {gap} timesteps.")
            return (f"{verdict} has the largest inter-event gap at "
                    f"{gap} timesteps.")

        return verdict

    # ------------------------------------------------------------------ #
    # Change point detection QA                                            #
    # ------------------------------------------------------------------ #

    def _generate_change_point_qa(
        self,
        attributes: Dict,
        metric: str,
        series_index=None,
    ) -> None:
        """Change point detection QA: count, positions, largest level shift."""
        trend_list = attributes.get('trend_list', [])
        segs = [s for s in trend_list if isinstance(s, (list, tuple)) and len(s) >= 3]
        if len(segs) < 2:
            return

        field_key = 'series_index' if series_index is not None else 'timeseries'
        field_idx = series_index if series_index is not None else 0

        sub_type_weights = {
            'change_point_count': 0.35,
            'change_point_positions': 0.35,
            'largest_level_shift': 0.30,
        }
        # Need at least 3 segments (2 change points) for meaningful largest_level_shift comparison
        if len(segs) < 3:
            sub_type_weights.pop('largest_level_shift', None)

        if self.config.debug:
            n_generate = len(sub_type_weights)
        else:
            n_generate = random.randint(1, 2)

        sub_types = list(sub_type_weights.keys())
        weights = list(sub_type_weights.values())
        chosen = []
        remaining = list(range(len(sub_types)))
        for _ in range(min(n_generate, len(sub_types))):
            if not remaining:
                break
            pool_dict = {sub_types[i]: weights[i] for i in remaining}
            pick = weighted_random_choice(pool_dict)
            idx = sub_types.index(pick)
            chosen.append(pick)
            remaining.remove(idx)

        for sub_type in chosen:
            try:
                params, question = self._build_change_point_params(
                    sub_type, attributes, metric
                )
                if params is None:
                    continue
                think_str, verdict = build_change_point_thought(
                    attributes, metric, sub_type, params
                )
            except Exception:
                logger.warning("Change point QA generation failed", exc_info=True)
                continue

            answer_text = self._change_point_answer_text(
                sub_type, verdict, params, metric
            )
            self.questions.append(_fix_article(question))
            self.answers.append(think_str + answer_text)
            self.llm_prompts.append([])
            self.fields.append({field_key: [field_idx]})
            self.qa_types.append(QAType.SEGMENT_MASK)
            self.eval_tasks.append("change_point")
            self.eval_metadatas.append({
                "sub_type": sub_type,
                "verdict": verdict,
                "length": self.seq_len,
            })

    def _build_change_point_params(
        self,
        sub_type: str,
        attributes: Dict,
        metric: str,
    ):
        """Build params dict and question string for each change point sub-type."""
        trend_list = attributes.get('trend_list', [])
        segs = [s for s in trend_list if isinstance(s, (list, tuple)) and len(s) >= 3]

        if sub_type == 'change_point_count':
            q = random.choice(_CHANGE_POINT_COUNT_QUESTIONS).format(metric=metric)
            return {'trend_list': trend_list}, q

        elif sub_type == 'change_point_positions':
            q = random.choice(_CHANGE_POINT_POSITIONS_QUESTIONS).format(metric=metric)
            return {'trend_list': trend_list}, q

        elif sub_type == 'largest_level_shift':
            # Compute per-segment means from segment_means_16
            statistics = attributes.get('statistics') or {}
            win16 = statistics.get('segment_means_16', [])
            seq_len = attributes.get('seq_len', 0)
            if not win16 or seq_len <= 0:
                return None, None

            window_size = 16
            segment_means = []
            for seg in segs:
                seg_start = seg[1]
                seg_end = seg[2]
                # Map segment range to window indices
                win_start = seg_start // window_size
                win_end = min((seg_end - 1) // window_size, len(win16) - 1)
                if win_start > win_end or win_start >= len(win16):
                    # Fallback: use overall mean
                    segment_means.append(statistics.get('mean', 0.0))
                    continue
                chunk = win16[win_start:win_end + 1]
                segment_means.append(sum(chunk) / len(chunk) if chunk else 0.0)

            if len(segment_means) != len(segs):
                return None, None

            q = random.choice(_CHANGE_POINT_LARGEST_SHIFT_QUESTIONS).format(metric=metric)
            return {'trend_list': trend_list, 'segment_means': segment_means}, q

        return None, None

    @staticmethod
    def _change_point_answer_text(sub_type, verdict, params, metric):
        """Build the human-readable answer sentence for a change point QA."""
        if sub_type == 'change_point_count':
            return f"There {pluralize(verdict, 'behavioral change point')} in {metric}."

        elif sub_type == 'change_point_positions':
            return f"The behavioral changes in {metric} occur at positions {verdict}."

        elif sub_type == 'largest_level_shift':
            return f"The largest level shift in {metric} occurs at position {verdict}."

        return verdict
