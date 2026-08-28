"""Out-of-distribution (OOD) evaluation-only QA generator.

Generates questions that compose 2+ trained skills in novel combinations.
No think block builders — ground truth is computed deterministically from metadata.
These questions go ONLY to the test split, never to training.
"""
import copy
import random
from collections import Counter
from typing import Dict, List, Optional, Any

import numpy as np
from loguru import logger

from synth.align.config import Mode, SampleResult
from synth.align.generators.compositional_qa import _coin_flip_keep
from synth.align.templates.ood_questions import (
    OOD_CONDITIONAL_STAT_TEMPLATES,
    OOD_NESTED_EXTREMA_TEMPLATES,
    OOD_EVENT_DENSITY_TEMPLATES,
    OOD_CONDITIONAL_COUNT_TEMPLATES,
    OOD_TREND_REVERSAL,
    OOD_RANGE_NORMALIZED_AMPLITUDE,
    OOD_SEGMENT_STAT_COMPARE,
)


def _fmt(val: float, dp: int = 2) -> str:
    """Format numeric value with specified decimal places."""
    if dp == 0:
        return str(int(round(val)))
    return f"{round(val, dp):.{dp}f}"


def _build_ood_answer(verdict) -> str:
    """Build minimal OOD ground truth answer in expected format."""
    return f"<think>\nanswer: {verdict}\n</think>\nThe answer is {verdict}."


def _event_in_segment(event: Dict, seg_start: int, seg_end: int) -> bool:
    """Check if a local event falls within a segment [start, end) exclusive-right."""
    return seg_start <= event['position_start'] < seg_end


class OODQAGenerator:
    """Generates OOD evaluation questions from existing time series metadata.

    Each OOD type composes 2+ atomic skills in ways never seen during training.
    Ground truth is computed deterministically — no think block builders needed.
    """

    def __init__(
        self,
        parent_result: SampleResult,
        metric_idx: int = 0,
    ):
        self.parent = parent_result
        self.metric_idx = metric_idx
        self.attributes = parent_result.attributes[metric_idx]
        self.timeseries = np.asarray(parent_result.original_timeseries[metric_idx])
        self.metric_name = parent_result.metrics[metric_idx]
        self.seq_len = len(self.timeseries)
        self.trend_list = self.attributes.get('trend_list', [])
        self.local_events = self.attributes.get('local', [])
        self.statistics = self.attributes.get('statistics') or {}

    def _make_result(
        self,
        question: str,
        verdict,
        eval_type: str,
        sub_type: Optional[str] = None,
    ) -> SampleResult:
        """Create an OOD SampleResult by deep-copying parent and overriding QA fields."""
        r = copy.deepcopy(self.parent)
        r.questions = [question]
        r.answers = [_build_ood_answer(verdict)]
        r.eval_task = eval_type
        r.eval_tasks = [eval_type]
        r.qa_types = [eval_type]
        meta: Dict[str, Any] = {'length': self.seq_len, 'verdict': str(verdict)}
        if sub_type:
            meta['sub_type'] = sub_type
        r.eval_metadata = meta
        r.eval_metadatas = [meta]
        # Mark fields as empty for evol_labels (OOD has no placeholders)
        r.fields = [{}]
        r.llm_prompts = [[]]
        return r

    def generate(self) -> List[SampleResult]:
        """Try all OOD types, return those whose preconditions are met."""
        results = []
        generators = [
            self._gen_conditional_stat,
            self._gen_nested_extrema,
            self._gen_event_density,
            self._gen_conditional_count,
            self._gen_trend_reversal,
            self._gen_range_normalized_amplitude,
            self._gen_segment_stat_compare,
        ]
        for gen_fn in generators:
            try:
                result = gen_fn()
                if result is not None:
                    results.append(result)
            except Exception:
                logger.warning(f"OOD {gen_fn.__name__} failed for {self.metric_name}", exc_info=True)
                continue
        return results

    # ------------------------------------------------------------------
    # Tier 1: Strongest OOD evidence
    # ------------------------------------------------------------------

    def _gen_conditional_stat(self) -> Optional[SampleResult]:
        """Stats restricted to a specific trend type's segments."""
        if len(self.trend_list) < 2:
            return None

        # Find trend types with enough data
        type_counts: Dict[str, int] = Counter()
        for seg in self.trend_list:
            type_counts[seg[0]] += seg[2] - seg[1]

        eligible = [t for t, count in type_counts.items() if count >= 4]
        if len(eligible) < 1:
            return None
        # Need at least 2 distinct types to be meaningfully conditional
        distinct_types = set(seg[0] for seg in self.trend_list)
        if len(distinct_types) < 2:
            return None

        target_type = random.choice(eligible)
        sub_type = random.choice(list(OOD_CONDITIONAL_STAT_TEMPLATES.keys()))

        # Collect indices within target trend type segments
        indices = []
        for seg in self.trend_list:
            if seg[0] == target_type:
                indices.extend(range(seg[1], seg[2]))

        if len(indices) < 4:
            return None

        subset = self.timeseries[indices]

        if sub_type == 'conditional_mean':
            verdict = _fmt(float(np.mean(subset)))
        elif sub_type == 'conditional_std':
            verdict = _fmt(float(np.std(subset)))
        elif sub_type == 'conditional_range':
            verdict = _fmt(float(np.max(subset) - np.min(subset)))
        else:
            return None

        templates = OOD_CONDITIONAL_STAT_TEMPLATES[sub_type]
        question = random.choice(templates).format(
            metric=self.metric_name,
            trend_type=target_type,
            stat_type=sub_type.replace('conditional_', ''),
        )

        return self._make_result(
            question, verdict,
            'ood_conditional_stat', sub_type,
        )

    def _gen_nested_extrema(self) -> Optional[SampleResult]:
        """Find extrema event within extrema segment."""
        if len(self.trend_list) < 2 or len(self.local_events) < 2:
            return None

        sub_type = random.choice(list(OOD_NESTED_EXTREMA_TEMPLATES.keys()))

        if sub_type == 'max_amp_in_longest':
            # Find longest segment
            best_seg = max(self.trend_list, key=lambda s: s[2] - s[1])
            # Find events within that segment
            events_in_seg = [
                e for e in self.local_events
                if _event_in_segment(e, best_seg[1], best_seg[2])
            ]
            if not events_in_seg:
                return None
            best_event = max(events_in_seg, key=lambda e: e['amplitude'])
            verdict = _fmt(best_event['amplitude'])
        elif sub_type == 'min_amp_in_shortest':
            # Find shortest segment
            best_seg = min(self.trend_list, key=lambda s: s[2] - s[1])
            events_in_seg = [
                e for e in self.local_events
                if _event_in_segment(e, best_seg[1], best_seg[2])
            ]
            if not events_in_seg:
                return None
            best_event = min(events_in_seg, key=lambda e: e['amplitude'])
            verdict = _fmt(best_event['amplitude'])
        else:
            return None

        templates = OOD_NESTED_EXTREMA_TEMPLATES[sub_type]
        question = random.choice(templates).format(metric=self.metric_name)

        return self._make_result(
            question, verdict,
            'ood_nested_extrema', sub_type,
        )

    def _gen_event_density(self) -> Optional[SampleResult]:
        """Which trend type has the highest/lowest event density (events per unit time)."""
        if len(self.trend_list) < 2 or len(self.local_events) < 3:
            return None

        distinct_types = set(seg[0] for seg in self.trend_list)
        if len(distinct_types) < 2:
            return None

        # Compute density per trend type
        type_duration: Dict[str, int] = {}
        type_events: Dict[str, int] = {}
        for t in distinct_types:
            type_duration[t] = sum(s[2] - s[1] for s in self.trend_list if s[0] == t)
            type_events[t] = sum(
                1 for e in self.local_events
                if any(
                    _event_in_segment(e, s[1], s[2])
                    for s in self.trend_list if s[0] == t
                )
            )

        # Need events in at least 2 trend types
        types_with_events = [t for t in distinct_types if type_events[t] > 0]
        if len(types_with_events) < 2:
            return None

        densities: Dict[str, float] = {}
        for t in distinct_types:
            if type_duration[t] > 0:
                densities[t] = type_events[t] / type_duration[t]

        if not densities:
            return None

        sub_type = random.choice(list(OOD_EVENT_DENSITY_TEMPLATES.keys()))

        if sub_type == 'highest_density':
            max_density = max(densities.values())
            winners = [t for t, d in densities.items() if d == max_density]
            verdict = winners[0] if len(winners) == 1 else 'equal'
        elif sub_type == 'lowest_density':
            # Include types with 0 events (density=0) if they have duration
            for t in distinct_types:
                if t not in densities and type_duration.get(t, 0) > 0:
                    densities[t] = 0.0
            min_density = min(densities.values())
            winners = [t for t, d in densities.items() if d == min_density]
            verdict = winners[0] if len(winners) == 1 else 'equal'
        else:
            return None

        templates = OOD_EVENT_DENSITY_TEMPLATES[sub_type]
        question = random.choice(templates).format(metric=self.metric_name)

        return self._make_result(
            question, verdict,
            'ood_event_density', sub_type,
        )

    # ------------------------------------------------------------------
    # Tier 2: Good OOD evidence
    # ------------------------------------------------------------------

    def _gen_conditional_count(self) -> Optional[SampleResult]:
        """Count segments of a type whose mean exceeds/falls below overall mean."""
        if len(self.trend_list) < 2:
            return None

        overall_mean = round(float(self.statistics.get('mean', 0)), 2)

        # Pick a trend type that has >= 2 segments
        type_segments: Dict[str, list] = {}
        for seg in self.trend_list:
            type_segments.setdefault(seg[0], []).append(seg)

        eligible = [t for t, segs in type_segments.items() if len(segs) >= 2]
        if not eligible:
            return None

        target_type = random.choice(eligible)
        sub_type = random.choice(list(OOD_CONDITIONAL_COUNT_TEMPLATES.keys()))

        count = 0
        for seg in type_segments[target_type]:
            seg_values = self.timeseries[seg[1]:seg[2]]
            if len(seg_values) == 0:
                continue
            seg_mean = round(float(np.mean(seg_values)), 2)
            if sub_type == 'count_above_mean' and seg_mean > overall_mean:
                count += 1
            elif sub_type == 'count_below_mean' and seg_mean < overall_mean:
                count += 1

        verdict = _fmt(count, dp=0)

        templates = OOD_CONDITIONAL_COUNT_TEMPLATES[sub_type]
        question = random.choice(templates).format(
            metric=self.metric_name,
            trend_type=target_type,
        )

        return self._make_result(
            question, verdict,
            'ood_conditional_count', sub_type,
        )

    def _gen_trend_reversal(self) -> Optional[SampleResult]:
        """Does the dominant trend reverse after the largest change point?"""
        if len(self.trend_list) < 3:
            return None

        # Find largest change point (boundary with max |mean_after - mean_before|)
        max_shift = -1.0
        cp_idx = -1
        for i in range(len(self.trend_list) - 1):
            seg_a = self.trend_list[i]
            seg_b = self.trend_list[i + 1]
            vals_a = self.timeseries[seg_a[1]:seg_a[2]]
            vals_b = self.timeseries[seg_b[1]:seg_b[2]]
            if len(vals_a) == 0 or len(vals_b) == 0:
                continue
            shift = abs(round(float(np.mean(vals_b)), 2) - round(float(np.mean(vals_a)), 2))
            if shift > max_shift:
                max_shift = shift
                cp_idx = i

        if cp_idx < 0:
            return None

        # Change point position = end of segment at cp_idx
        cp_pos = self.trend_list[cp_idx][2]

        # Dominant trend before change point
        before_durations: Dict[str, int] = {}
        for seg in self.trend_list:
            if seg[2] <= cp_pos:
                before_durations[seg[0]] = before_durations.get(seg[0], 0) + (seg[2] - seg[1])
            elif seg[1] < cp_pos:
                # Partial overlap
                before_durations[seg[0]] = before_durations.get(seg[0], 0) + (cp_pos - seg[1])

        after_durations: Dict[str, int] = {}
        for seg in self.trend_list:
            if seg[1] >= cp_pos:
                after_durations[seg[0]] = after_durations.get(seg[0], 0) + (seg[2] - seg[1])
            elif seg[2] > cp_pos:
                after_durations[seg[0]] = after_durations.get(seg[0], 0) + (seg[2] - cp_pos)

        if not before_durations or not after_durations:
            return None

        dom_before = max(before_durations, key=before_durations.get)
        dom_after = max(after_durations, key=after_durations.get)

        verdict = "yes" if dom_before != dom_after else "no"

        question = random.choice(OOD_TREND_REVERSAL).format(metric=self.metric_name)
        return self._make_result(
            question, verdict,
            'ood_trend_reversal',
        )

    def _gen_range_normalized_amplitude(self) -> Optional[SampleResult]:
        """Is the largest event amplitude > half the total value range?"""
        if not self.local_events:
            return None

        value_range = self.statistics.get('range', 0)
        if value_range <= 0:
            return None

        max_amp = max(e['amplitude'] for e in self.local_events)
        threshold = round(value_range / 2, 2)
        verdict = "yes" if round(max_amp, 2) > threshold else "no"

        if not _coin_flip_keep(
            verdict, ['yes', 'no'], 'ood_range_normalized_amplitude',
        ):
            return None

        question = random.choice(OOD_RANGE_NORMALIZED_AMPLITUDE).format(
            metric=self.metric_name,
        )
        return self._make_result(
            question, verdict,
            'ood_range_normalized_amplitude',
        )

    # ------------------------------------------------------------------
    # Tier 3: Simpler but still OOD
    # ------------------------------------------------------------------

    def _gen_segment_stat_compare(self) -> Optional[SampleResult]:
        """Is the mean of the first trend segment > the mean of the last?"""
        if len(self.trend_list) < 2:
            return None

        first_seg = self.trend_list[0]
        last_seg = self.trend_list[-1]

        first_vals = self.timeseries[first_seg[1]:first_seg[2]]
        last_vals = self.timeseries[last_seg[1]:last_seg[2]]

        if len(first_vals) == 0 or len(last_vals) == 0:
            return None

        first_mean = round(float(np.mean(first_vals)), 2)
        last_mean = round(float(np.mean(last_vals)), 2)

        if abs(first_mean - last_mean) < 0.1:
            return None

        verdict = "yes" if first_mean > last_mean else "no"

        if not _coin_flip_keep(
            verdict, ['yes', 'no'], 'ood_segment_stat_compare',
        ):
            return None

        question = random.choice(OOD_SEGMENT_STAT_COMPARE).format(
            metric=self.metric_name,
        )
        return self._make_result(
            question, verdict,
            'ood_segment_stat_compare',
        )


# ------------------------------------------------------------------
# Public API: Generate OOD questions for a split
# ------------------------------------------------------------------

def _ood_dict_to_sample(
    parent: SampleResult,
    metric_idx: int,
    qa_dict: Dict[str, Any],
) -> SampleResult:
    """Convert a new-style OOD dict to a SampleResult (deep-copies parent)."""
    r = copy.deepcopy(parent)
    r.questions = [qa_dict['question']]
    r.answers = [qa_dict['answer']]
    r.eval_task = qa_dict['eval_type']
    r.eval_tasks = [qa_dict['eval_type']]
    r.qa_types = [qa_dict['eval_type']]
    r.eval_metadata = qa_dict['eval_metadata']
    r.eval_metadatas = [qa_dict['eval_metadata']]
    r.fields = [{}]
    r.llm_prompts = [[]]
    return r


def _run_cross_metric_ood_generators(
    parent: SampleResult,
) -> List[SampleResult]:
    """Run cross-metric OOD generators (once per MTS sample)."""
    from synth.align.generators.ood_cross_metric_qa import OODCrossMetricGenerator
    from synth.align.generators.ood_composition_qa import OODCompositionCrossGenerator

    ts_list = [np.asarray(ts) for ts in parent.original_timeseries]
    seq_len = len(ts_list[0]) if ts_list else 0

    results = []
    for gen in (
        OODCrossMetricGenerator(ts_list, parent.metrics, parent.attributes, seq_len),
        OODCompositionCrossGenerator(ts_list, parent.metrics, parent.attributes, seq_len),
    ):
        for qa_dict in gen.generate_all():
            # Use metric_idx=0 for deep-copy base (all metrics are in the parent)
            results.append(_ood_dict_to_sample(parent, 0, qa_dict))
    return results


def _run_new_ood_generators(
    parent: SampleResult,
    metric_idx: int,
) -> List[SampleResult]:
    """Run new taxonomy OOD generators for one metric."""
    from synth.align.generators.ood_mean_qa import OODMeanGenerator
    from synth.align.generators.ood_stat_qa import OODStatGenerator
    from synth.align.generators.ood_trend_qa import OODTrendGenerator
    from synth.align.generators.ood_event_qa import OODEventGenerator
    from synth.align.generators.ood_periodic_qa import OODPeriodicGenerator
    from synth.align.generators.ood_composition_qa import OODCompositionSingleGenerator

    attrs = parent.attributes[metric_idx]
    ts = np.asarray(parent.original_timeseries[metric_idx])
    metric = parent.metrics[metric_idx]
    seq_len = len(ts)
    trend_list = attrs.get('trend_list', [])
    local_events = attrs.get('local', [])
    seasonal = attrs.get('seasonal', {})

    generators = [
        OODMeanGenerator(ts, metric, seq_len),
        OODStatGenerator(ts, metric, seq_len),
        OODTrendGenerator(ts, metric, seq_len, trend_list),
        OODEventGenerator(ts, metric, seq_len, local_events, trend_list),
        OODPeriodicGenerator(ts, metric, seq_len, seasonal, trend_list),
        OODCompositionSingleGenerator(ts, metric, seq_len, trend_list),
    ]

    results = []
    for gen in generators:
        for qa_dict in gen.generate_all():
            results.append(_ood_dict_to_sample(parent, metric_idx, qa_dict))
    return results


def generate_ood_for_split(
    results: List[SampleResult],
    max_ood_samples: int = 0,
) -> List[SampleResult]:
    """Generate OOD questions from existing SampleResults.

    Runs both legacy OODQAGenerator and new taxonomy OOD generators
    for each eligible sample (UTS or MTS_LOCAL/MTS_SHAPE).

    If max_ood_samples > 0, caps output with stratified sampling
    (equal per OOD eval_type) to prevent combinatorial explosion.
    """
    ood_results: List[SampleResult] = []
    tried = 0
    for result in results:
        if result.mode not in (Mode.UTS, Mode.MTS_LOCAL, Mode.MTS_SHAPE):
            continue
        n_metrics = len(result.metrics)
        for idx in range(n_metrics):
            tried += 1
            # Legacy OOD generators
            if result.mode in (Mode.UTS, Mode.MTS_LOCAL):
                gen = OODQAGenerator(result, metric_idx=idx)
                ood_results.extend(gen.generate())
            # New taxonomy OOD generators (single-metric, same guard as legacy)
            if result.mode in (Mode.UTS, Mode.MTS_LOCAL):
                try:
                    ood_results.extend(_run_new_ood_generators(result, idx))
                except Exception:
                    logger.warning(
                        f"New OOD generators failed for {result.metrics[idx]}",
                        exc_info=True,
                    )

        # Cross-metric OOD generators (once per MTS sample, not per-metric)
        if n_metrics >= 2:
            try:
                ood_results.extend(_run_cross_metric_ood_generators(result))
            except Exception:
                logger.warning(
                    "Cross-metric OOD generators failed", exc_info=True,
                )

    raw_count = len(ood_results)

    # Stratified cap: equal representation per OOD eval_type
    if max_ood_samples > 0 and len(ood_results) > max_ood_samples:
        by_type: dict = {}
        for r in ood_results:
            et = r.eval_task or ''
            by_type.setdefault(et, []).append(r)

        n_types = len(by_type)
        per_type = max(1, max_ood_samples // n_types)
        capped = []
        for et in sorted(by_type):
            pool = by_type[et]
            random.shuffle(pool)
            capped.extend(pool[:per_type])

        # Fill remainder if under budget
        if len(capped) < max_ood_samples:
            used = set(id(r) for r in capped)
            leftover = [r for r in ood_results if id(r) not in used]
            random.shuffle(leftover)
            capped.extend(leftover[:max_ood_samples - len(capped)])

        ood_results = capped[:max_ood_samples]
        logger.info(f"OOD capped: {raw_count} -> {len(ood_results)} "
                     f"({per_type} per type, {n_types} types)")

    logger.info(
        f"OOD generation: {tried} metrics tried, "
        f"{raw_count} produced ({raw_count / max(tried, 1):.1%} hit rate)"
        + (f", capped to {len(ood_results)}" if max_ood_samples > 0 else "")
    )
    return ood_results
