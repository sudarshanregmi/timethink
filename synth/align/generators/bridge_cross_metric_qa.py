"""Cross-metric bridge QA generators (Category H: Bridge).

Force-SFT examples teaching how to decompose cross-metric reasoning.

    HB1: stat ratio — "compute stat(A), compute stat(B), ratio, threshold"
    HB2: full ordering — "compute stat for each, sort, report"
    HB3: trend concordance — "check each pair, count matches, fraction"
    HB4: conditional filter — "per-metric cond-A → filter → cond-B on filtered"

Bridge think blocks show the decomposition plan + step results + combination.
eval_metadata['bridge'] = True ensures force-SFT routing.
"""

import random
from collections import Counter
from itertools import combinations
from typing import Any, Dict, List, Optional

import numpy as np
from loguru import logger

from synth.align.generators.atomic_qa import _fmt
from synth.align.generators.decomposition_bridge_qa import (
    _build_bridge_think,
)


_STAT_LABELS = {
    'mean': 'mean', 'std': 'standard deviation',
    'max': 'maximum', 'range': 'value range',
}


class BridgeCrossMetricGenerator:
    """Generates cross-metric bridge SFT examples (HB1-HB3)."""

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

    def _stats(self, i: int) -> Dict:
        return self.attrs[i].get('statistics', {})

    def _trend_type(self, i: int) -> str:
        return self.attrs[i].get('trend', {}).get('type', 'unknown')

    def _make_result(self, question, think, answer_text, eval_type, verdict,
                     **extra_meta) -> Dict[str, Any]:
        meta = {'length': self.seq_len, 'verdict': str(verdict), 'bridge': True}
        meta.update(extra_meta)
        return {
            'question': question,
            'answer': f"{think}\n{answer_text}",
            'eval_type': eval_type,
            'eval_metadata': meta,
        }

    # --- HB1: Stat ratio bridge ---

    def generate_stat_ratio_bridge(self) -> Optional[Dict[str, Any]]:
        """HB1: compute stat(A), compute stat(B), multiply threshold, compare."""
        if self.n < 2:
            return None

        stat_key = random.choice(['std', 'range'])
        stat_label = _STAT_LABELS.get(stat_key, stat_key)
        i, j = random.sample(range(self.n), 2)
        a, b = self.metrics[i], self.metrics[j]

        va = self._stats(i).get(stat_key)
        vb = self._stats(j).get(stat_key)
        if va is None or vb is None or float(vb) == 0:
            return None

        va_r, vb_r = round(float(va), 2), round(float(vb), 2)
        k = round(random.uniform(0.5, 3.0), 1)
        threshold = round(k * vb_r, 2)
        verdict = "yes" if va_r >= threshold else "no"

        think = _build_bridge_think(
            plan_steps=[
                f"compute {stat_label} of {a}",
                f"compute {stat_label} of {b}",
                f"compute threshold ({_fmt(k)} × {b}'s value) and compare",
            ],
            step_results=[
                f"{a}: {stat_label} = {_fmt(va_r)}",
                f"{b}: {stat_label} = {_fmt(vb_r)}",
                f"threshold = {_fmt(k)} × {_fmt(vb_r)} = {_fmt(threshold)}",
            ],
            combination=f"{_fmt(va_r)} {'>=' if va_r >= threshold else '<'} {_fmt(threshold)} → {verdict}",
            verdict=verdict,
        )

        question = f"Is {a}'s {stat_label} at least {_fmt(k)} times {b}'s {stat_label}?"
        answer_text = (
            f"{'Yes' if verdict == 'yes' else 'No'}, {a}'s {stat_label} "
            f"({_fmt(va_r)}) {'meets' if verdict == 'yes' else 'does not meet'} "
            f"the threshold of {_fmt(k)} × {b}'s ({_fmt(vb_r)}) = {_fmt(threshold)}."
        )
        return self._make_result(
            question, think, answer_text, 'rl_cross_stat_ratio', verdict,
            sub_type=stat_key,
        )

    # --- HB2: Full ordering bridge ---

    def generate_full_ordering_bridge(self) -> Optional[Dict[str, Any]]:
        """HB2: compute stat for each metric, sort, report ordering."""
        if self.n < 3:
            return None

        stat_key = random.choice(['mean', 'max', 'min', 'range'])
        stat_label = _STAT_LABELS.get(stat_key, stat_key)

        values = []
        step_results = []
        for i in range(self.n):
            v = self._stats(i).get(stat_key)
            if v is None:
                continue
            val = round(float(v), 2)
            values.append((self.metrics[i], val))
            if stat_key == 'range':
                mx = self._stats(i).get('max')
                mn = self._stats(i).get('min')
                if mx is not None and mn is not None:
                    step_results.append(
                        f"{self.metrics[i]}: {stat_label} = max - min = "
                        f"{_fmt(float(mx))} - {_fmt(float(mn))} = {_fmt(val)}"
                    )
                else:
                    step_results.append(f"{self.metrics[i]}: {stat_label} = {_fmt(val)}")
            else:
                step_results.append(f"{self.metrics[i]}: {stat_label} = {_fmt(val)}")
        if len(values) < 3:
            return None

        values.sort(key=lambda x: x[1], reverse=True)
        ordering = ", ".join(name for name, _ in values)

        think = _build_bridge_think(
            plan_steps=[
                f"compute {stat_label} for each metric",
                "sort from highest to lowest",
                "report the ordering",
            ],
            step_results=step_results,
            combination=f"Sorted (highest to lowest): {ordering}",
            verdict=ordering,
        )

        question = f"Rank the metrics by {stat_label} from highest to lowest."
        answer_text = f"Ranked by {stat_label} (highest to lowest): {ordering}."
        return self._make_result(
            question, think, answer_text, 'rl_cross_full_ordering', ordering,
            sub_type=stat_key,
        )

    # --- HB3: Dominant-trend share bridge ---

    def generate_trend_concordance_bridge(self) -> Optional[Dict[str, Any]]:
        """HB3: tally trends per type → find max → divide by N → fraction.

        Reframed (2026-04-17) from O(N^2) pair enumeration to O(N) tally + argmax
        + division. Matches the reframed `rl_cross_trend_concordance` semantic
        (mode_count / N). Plants the smart algorithm so RL rollout doesn't
        imitate the expensive pair-by-pair version.
        """
        if self.n < 3:
            return None

        trends = [self._trend_type(i) for i in range(self.n)]
        if 'unknown' in trends:
            return None

        from collections import Counter
        counts = Counter(trends)
        ranked = counts.most_common()
        if len(ranked) >= 2 and ranked[0][1] == ranked[1][1]:
            return None  # tie at top — ambiguous fraction target

        mode_trend, mode_count = ranked[0]
        fraction = round(mode_count / self.n, 2)
        verdict = _fmt(fraction)

        per_metric_lines = [
            f"{self.metrics[i]}: trend = {trends[i]}" for i in range(self.n)
        ]
        tally_lines = [
            f"  {trend}: {c}" for trend, c in ranked
        ]

        think = _build_bridge_think(
            plan_steps=[
                "identify each metric's overall trend direction",
                "tally how many metrics fall into each trend type",
                "pick the trend type with the highest count (the mode)",
                "divide the mode's count by the total metric count to get the fraction",
            ],
            step_results=(
                per_metric_lines
                + ["", "Tally by trend type:"]
                + tally_lines
                + ["", f"Most common trend: {mode_trend} ({mode_count} of {self.n} metrics)"]
            ),
            combination=f"Fraction in dominant trend: {mode_count} / {self.n} = {verdict}",
            verdict=verdict,
        )

        question = (
            f"What fraction of the {self.n} metrics show the dominant trend direction?"
        )
        answer_text = (
            f"The dominant trend direction is {mode_trend}, shared by "
            f"{mode_count} of {self.n} metrics."
        )
        return self._make_result(
            question, think, answer_text, 'rl_cross_trend_concordance', verdict,
        )

    # --- HB4: Conditional cross-metric filter ---

    def generate_conditional_filter_bridge(self) -> Optional[Dict[str, Any]]:
        """HB4: per-metric condition A → filter → condition B on filtered set.
        (rl_corr_conditional pattern — same-trend metrics with nearby events)

        Teaches the archetype: for each metric evaluate property A, filter
        to matching set, then apply property B to the filtered pairs.
        """
        if self.n < 3:
            return None

        trends = [self._trend_type(i) for i in range(self.n)]
        if 'unknown' in trends:
            return None

        trend_counts = Counter(trends)
        target_trend = trend_counts.most_common(1)[0][0]
        filtered_idx = [i for i in range(self.n) if trends[i] == target_trend]
        if len(filtered_idx) < 2:
            return None

        trend_lines = [
            f"{self.metrics[i]}: trend = {trends[i]}"
            for i in range(self.n)
        ]
        filtered_names = [self.metrics[i] for i in filtered_idx]
        filter_line = (
            f"Metrics with trend={target_trend}: {', '.join(filtered_names)} "
            f"({len(filtered_idx)} metrics)"
        )

        pos_by_idx: Dict[int, List[int]] = {}
        event_enum_lines = []
        for i in filtered_idx:
            events = self.attrs[i].get('local_changes', [])
            positions = sorted({
                e['position_start'] for e in events
                if e.get('position_start') is not None
            })
            pos_by_idx[i] = positions
            if positions:
                event_enum_lines.append(
                    f"{self.metrics[i]} event positions: {positions}"
                )
            else:
                event_enum_lines.append(
                    f"{self.metrics[i]} event positions: (none)"
                )

        threshold = 20
        pair_lines = []
        has_sync = False
        for i, j in combinations(filtered_idx, 2):
            pos_i = pos_by_idx[i]
            pos_j = pos_by_idx[j]
            if not pos_i or not pos_j:
                pair_lines.append(
                    f"{self.metrics[i]} vs {self.metrics[j]}: "
                    f"at least one has no events → skip"
                )
                continue
            min_gap, best_pi, best_pj = min(
                (abs(pi - pj), pi, pj)
                for pi in pos_i for pj in pos_j
            )
            pair_match = min_gap <= threshold
            pair_lines.append(
                f"{self.metrics[i]} vs {self.metrics[j]}: "
                f"closest pair |{best_pi} - {best_pj}| = {min_gap} "
                f"{'<=' if pair_match else '>'} {threshold} "
                f"→ {'nearby' if pair_match else 'far'}"
            )
            if pair_match:
                has_sync = True

        verdict = "yes" if has_sync else "no"

        think = _build_bridge_think(
            plan_steps=[
                "identify each metric's overall trend",
                f"filter to metrics whose trend is {target_trend}",
                "enumerate the event positions of each filtered metric",
                "for each pair in the filtered set, find the closest event-gap and check whether it is within proximity",
            ],
            step_results=(
                trend_lines
                + ["", filter_line, ""]
                + event_enum_lines
                + [""]
                + pair_lines
            ),
            combination=(
                f"Any {target_trend} pair with nearby events? "
                f"→ {verdict}"
            ),
            verdict=verdict,
        )

        question = (
            f"Among metrics with a {target_trend} trend, do any pair share "
            f"nearby local events?"
        )
        if has_sync:
            answer_text = (
                f"Yes, some {target_trend}-trend metric pairs share nearby events."
            )
        else:
            answer_text = (
                f"No, no {target_trend}-trend metric pairs share nearby events."
            )
        return self._make_result(
            question, think, answer_text, 'rl_corr_conditional', verdict,
        )

    # --- Generate all ---

    def generate_all(self) -> List[Dict[str, Any]]:
        results = []
        generators = [
            self.generate_stat_ratio_bridge,
            self.generate_full_ordering_bridge,
            self.generate_trend_concordance_bridge,
            self.generate_conditional_filter_bridge,
        ]
        for gen_fn in generators:
            try:
                r = gen_fn()
                if r is not None:
                    results.append(r)
            except Exception:
                logger.warning(
                    f"CrossMetric bridge {gen_fn.__name__} failed", exc_info=True,
                )
        return results
