"""OOD compositional probes (reverse-engineered from observed RL wins).

Four eval-only probes designed to compose atomic + rl primitives that
the model independently demonstrates it has learned. Each probe pairs
two named skills in a form that does not appear in any training
eval_type, so a positive result constitutes held-out composition
transfer. Detailed rationale per probe is inline.

Primitives and source evaluations (full-test gains vs SFT):
    atomic_max_position (preserved by SFT)
    atomic_interval_mean / atomic_global_mean (preserved)
    rl_segment_duration (+59pp)
    rl_dominant_trend_type (+50pp)
    rl_cluster_count / rl_cluster_dominant (+53 / +31 pp)
    rl_cross_event_sync (cross-metric, positive)

Probes:
    ood_peak_in_longest_segment   (single-metric, yes/no)
    ood_duration_weighted_mean    (single-metric, float)
    ood_dominant_trend_in_cluster (cross-metric, categorical)
    ood_cross_event_causality     (cross-metric, yes/no)
"""

import random
from itertools import combinations
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from loguru import logger

from synth.align.generators.atomic_qa import _fmt
from synth.align.generators.atomic_event_qa import _normalize_event
from synth.align.generators.compositional_qa import (
    _coin_flip_keep, _interval_mean, _make_rl_result,
)


# ============================================================
# Single-metric probes
# ============================================================

PEAK_IN_LONGEST_SEGMENT_QUESTIONS = [
    "Does the global maximum of {metric} lie within the longest trend segment?",
    "For {metric}, is the argmax inside the longest-duration trend segment?",
    "Is the peak of {metric} contained in the longest trend segment?",
    "Check whether the global maximum of {metric} falls inside the segment of the longest duration.",
    "In {metric}, does the largest value occur within the longest trend segment?",
]

PEAK_IN_LONGEST_SEGMENT_ANSWERS = {
    'yes': [
        "Yes, the global maximum of {metric} is inside the longest trend segment.",
        "The peak of {metric} falls within the longest segment.",
        "Confirmed, argmax lies inside the longest-duration segment of {metric}.",
    ],
    'no': [
        "No, the global maximum of {metric} is not in the longest trend segment.",
        "The peak of {metric} is outside the longest segment.",
        "The argmax lies in a segment other than the longest for {metric}.",
    ],
}

DURATION_WEIGHTED_MEAN_QUESTIONS = [
    "Compute the duration-weighted mean of {metric}: sum each trend segment's mean, weighted by its duration, divided by total duration. Report to 2 decimals.",
    "For {metric}, calculate the average of segment means weighted by segment durations, rounded to 2 decimals.",
    "What is the duration-weighted segment mean of {metric}? Weight each segment mean by its length and normalize by total length.",
    "Compute sum(mean_i * duration_i) / sum(duration_i) across trend segments of {metric}, to 2 decimals.",
]

DURATION_WEIGHTED_MEAN_ANSWERS = [
    "The duration-weighted mean of {metric} is {verdict}.",
    "Weighting segment means by their durations, the mean for {metric} is {verdict}.",
    "The length-weighted average of segment means for {metric} is {verdict}.",
]


class OODCompositionSingleGenerator:
    """Single-metric OOD compositional probes."""

    MIN_SEGMENTS = 2

    def __init__(
        self,
        timeseries: np.ndarray,
        metric: str,
        seq_len: int,
        trend_list: List[Tuple[str, int, int]],
    ):
        self.ts = np.asarray(timeseries, dtype=float)
        self.metric = metric
        self.seq_len = seq_len
        self.trend_list = trend_list

    # --- Peak-in-longest-segment ---
    # Composes: atomic_max_position (SFT-preserved, global argmax)
    #        ⊕ rl_segment_duration (+59pp, longest segment identification)

    def generate_peak_in_longest_segment(self) -> Optional[Dict[str, Any]]:
        if len(self.trend_list) < self.MIN_SEGMENTS:
            return None

        # Longest trend segment (inclusive end).
        longest = max(self.trend_list, key=lambda s: s[2] - s[1])
        _, l_s, l_e = longest
        max_pos = int(np.argmax(self.ts))
        contained = l_s <= max_pos <= l_e

        verdict = "yes" if contained else "no"
        if not _coin_flip_keep(
            verdict, ['yes', 'no'], 'ood_peak_in_longest_segment',
        ):
            return None

        answer_text = random.choice(
            PEAK_IN_LONGEST_SEGMENT_ANSWERS[verdict]
        ).format(metric=self.metric)

        return _make_rl_result(
            random.choice(PEAK_IN_LONGEST_SEGMENT_QUESTIONS).format(
                metric=self.metric,
            ),
            verdict, 'ood_peak_in_longest_segment', answer_text, self.seq_len,
        )

    # --- Duration-weighted mean ---
    # Composes: atomic_interval_mean (SFT, per-segment mean)
    #        ⊕ rl_segment_duration (+59pp, segment length)
    # Novel: weighting is not a primitive in any training eval_type.

    def generate_duration_weighted_mean(self) -> Optional[Dict[str, Any]]:
        if len(self.trend_list) < self.MIN_SEGMENTS:
            return None

        total_weight = 0.0
        weighted_sum = 0.0
        for _, s, e in self.trend_list:
            dur = max(1, e - s + 1)
            seg_mean = _interval_mean(self.ts, s, e)
            weighted_sum += seg_mean * dur
            total_weight += dur

        if total_weight <= 0:
            return None

        weighted = round(weighted_sum / total_weight, 2)
        verdict = _fmt(weighted)

        answer_text = random.choice(DURATION_WEIGHTED_MEAN_ANSWERS).format(
            metric=self.metric, verdict=verdict,
        )

        return _make_rl_result(
            random.choice(DURATION_WEIGHTED_MEAN_QUESTIONS).format(
                metric=self.metric,
            ),
            verdict, 'ood_duration_weighted_mean', answer_text, self.seq_len,
        )

    def generate_all(self) -> List[Dict[str, Any]]:
        results = []
        for fn in (self.generate_peak_in_longest_segment,
                   self.generate_duration_weighted_mean):
            try:
                r = fn()
                if r is not None:
                    results.append(r)
            except Exception:
                logger.warning(f"OOD {fn.__name__} failed", exc_info=True)
        return results


# ============================================================
# Cross-metric probes
# ============================================================

DOMINANT_TREND_IN_CLUSTER_QUESTIONS = [
    "Among the metrics whose mean values fall in the upper half of all metric means, which trend type is most common?",
    "Cluster the metrics by whether their mean value is above or below the median across metrics. In the upper (larger-mean) cluster, what is the dominant trend type?",
    "For metrics with mean above the cross-metric median, which trend direction appears most often?",
]

DOMINANT_TREND_IN_CLUSTER_ANSWERS = [
    "Among metrics in the upper-mean cluster, the dominant trend type is {verdict}.",
    "The most common trend type in the upper-mean cluster is {verdict}.",
    "In the larger-mean group of metrics, the prevailing trend is {verdict}.",
]

CROSS_EVENT_CAUSALITY_QUESTIONS = [
    "Does at least one local event in {a} precede a local event in {b} by less than {window} timesteps?",
    "For the pair ({a}, {b}), is there any event in {a} followed within {window} timesteps by an event in {b}?",
    "Check whether an event in {a} occurs less than {window} timesteps before some event in {b}.",
    "Do events in {a} ever precede events in {b} within a {window}-timestep window?",
]

CROSS_EVENT_CAUSALITY_ANSWERS = {
    'yes': [
        "Yes, at least one event in {a} is followed within {window} timesteps by an event in {b}.",
        "An event in {a} precedes an event in {b} within a {window}-timestep window.",
        "Confirmed, {a} has an event shortly before an event in {b} (within {window} timesteps).",
    ],
    'no': [
        "No, no event in {a} is followed within {window} timesteps by an event in {b}.",
        "There is no causal-ordered event pair from {a} to {b} within {window} timesteps.",
        "{a} does not have any event preceding an event in {b} within the {window}-timestep window.",
    ],
}


class OODCompositionCrossGenerator:
    """Cross-metric OOD compositional probes."""

    def __init__(
        self,
        timeseries_list: List[np.ndarray],
        metrics: List[str],
        attributes_list: List[Dict],
        seq_len: int,
    ):
        self.ts_list = [np.asarray(ts, dtype=float) for ts in timeseries_list]
        self.metrics = metrics
        self.attrs = attributes_list
        self.seq_len = seq_len
        self.n = len(metrics)

    def _trend_type(self, i: int) -> str:
        return self.attrs[i].get('trend', {}).get('type', 'unknown')

    def _events(self, i: int) -> List[Dict[str, Any]]:
        raw = self.attrs[i].get('local', []) or []
        normed = []
        for ev in raw:
            n = _normalize_event(ev)
            if n is not None:
                normed.append(n)
        normed.sort(key=lambda e: e['position'])
        return normed

    # --- Dominant trend in cluster ---
    # Composes: rl_cluster_count / rl_cluster_dominant (+31-53pp)
    #        ⊕ rl_dominant_trend_type (+50pp)
    # Novel: conditional trend-dominance given a cluster definition.

    def generate_dominant_trend_in_cluster(self) -> Optional[Dict[str, Any]]:
        if self.n < 4:
            return None
        means = [float(np.mean(self.ts_list[i])) for i in range(self.n)]
        median_mean = float(np.median(means))

        upper_idx = [i for i in range(self.n) if means[i] >= median_mean]
        if len(upper_idx) < 2:
            return None

        trends = [self._trend_type(i) for i in upper_idx]
        if 'unknown' in trends:
            return None
        from collections import Counter
        counts = Counter(trends).most_common()
        # Require a clear winner (avoid ambiguous ties at the top).
        if len(counts) >= 2 and counts[0][1] == counts[1][1]:
            return None
        verdict = counts[0][0]

        answer_text = random.choice(DOMINANT_TREND_IN_CLUSTER_ANSWERS).format(
            verdict=verdict,
        )

        return _make_rl_result(
            random.choice(DOMINANT_TREND_IN_CLUSTER_QUESTIONS),
            verdict, 'ood_dominant_trend_in_cluster', answer_text, self.seq_len,
        )

    # --- Cross-metric event causality ---
    # Composes: rl_cross_event_sync (simultaneous, trained)
    #        ⊕ temporal ordering (not in any training eval_type).
    # Novel: precedence window instead of proximity window.

    def generate_cross_event_causality(self) -> Optional[Dict[str, Any]]:
        if self.n < 2:
            return None
        # Scale the window by seq_len so the probe works across lengths.
        window = max(5, self.seq_len // 16)

        # Pick a metric pair with events on both sides.
        pairs = [(i, j) for i, j in combinations(range(self.n), 2)]
        random.shuffle(pairs)
        for i, j in pairs:
            ev_i = self._events(i)
            ev_j = self._events(j)
            if not ev_i or not ev_j:
                continue

            has_cause = False
            for pi in (e['position'] for e in ev_i):
                for pj in (e['position'] for e in ev_j):
                    if 0 < (pj - pi) < window:
                        has_cause = True
                        break
                if has_cause:
                    break

            verdict = "yes" if has_cause else "no"
            if not _coin_flip_keep(
                verdict, ['yes', 'no'], 'ood_cross_event_causality',
            ):
                continue

            a, b = self.metrics[i], self.metrics[j]
            answer_text = random.choice(
                CROSS_EVENT_CAUSALITY_ANSWERS[verdict]
            ).format(a=a, b=b, window=window)

            return _make_rl_result(
                random.choice(CROSS_EVENT_CAUSALITY_QUESTIONS).format(
                    a=a, b=b, window=window,
                ),
                verdict, 'ood_cross_event_causality', answer_text, self.seq_len,
            )
        return None

    def generate_all(self) -> List[Dict[str, Any]]:
        results = []
        for fn in (self.generate_dominant_trend_in_cluster,
                   self.generate_cross_event_causality):
            try:
                r = fn()
                if r is not None:
                    results.append(r)
            except Exception:
                logger.warning(f"OOD {fn.__name__} failed", exc_info=True)
        return results
