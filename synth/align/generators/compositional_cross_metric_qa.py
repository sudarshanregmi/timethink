"""Compositional cross-metric QA generators (Category H: RL).

RL compositions that compose cross-metric atoms. Model discovers
reasoning via reward — minimal think block, rich NL answer.

    HR1: rl_cross_stat_ratio          — is stat(A)/stat(B) > K?
    HR2: rl_cross_full_ordering       — rank metrics by stat
    HR3: rl_cross_event_sync          — events at similar positions?
    HR4: rl_cross_period_compare      — period(A) shorter than period(B)?
    HR5: rl_cross_trend_concordance   — fraction of concordant pairs
    HR6: rl_cross_conditional_query   — among X, which has highest Y?
    HR7: rl_cross_attribute_corr      — noisiest == widest range?
    HR8: rl_cross_asymmetric_behavior — count of metrics whose mean shifted in a given direction
"""

import random
from itertools import combinations
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from loguru import logger

from synth.align.generators.atomic_qa import _fmt
from synth.align.generators.compositional_qa import _coin_flip_keep, _make_rl_result


# ============================================================
# Question / answer templates
# ============================================================

_STAT_LABELS = {
    'mean': 'mean', 'std': 'standard deviation',
    'max': 'maximum', 'min': 'minimum', 'range': 'value range',
}

# HR1
HR1_QUESTIONS = [
    "Is {a}'s {stat_label} at least {k} times that of {b}?",
    "Does {a} have a {stat_label} at least {k} times {b}'s?",
    "Is {a}'s {stat_label} at least {k}x {b}'s {stat_label}?",
]
HR1_ANSWERS = {
    'yes': [
        "Yes, {a}'s {stat_label} is at least {k} times {b}'s.",
        "Yes, {a}'s {stat_label} meets the {k}x threshold against {b}'s.",
    ],
    'no': [
        "No, {a}'s {stat_label} is less than {k} times {b}'s.",
        "No, {a}'s {stat_label} does not reach {k}x {b}'s.",
    ],
}

# HR2
HR2_QUESTIONS = [
    "Rank the metrics by {stat_label} from highest to lowest.",
    "Order all metrics by their {stat_label}, starting with the highest.",
    "List the metrics sorted by {stat_label} in descending order.",
]
HR2_ANSWERS = [
    "Ranked by {stat_label} (highest to lowest): {ordering}.",
    "The ordering by {stat_label} is: {ordering}.",
]

# HR3
HR3_QUESTIONS = [
    "Do {a} and {b} both have local events occurring within {t} positions of each other?",
    "Is there a pair of local events in {a} and {b} within {t} timesteps of each other?",
    "Do {a} and {b} exhibit temporally synchronized local events (within {t} positions)?",
]
HR3_ANSWERS = {
    'yes': [
        "Yes, {a} and {b} have temporally synchronized events.",
        "Events are synchronized between {a} and {b}.",
    ],
    'no': [
        "No, {a} and {b} do not have temporally synchronized events.",
        "The events in {a} and {b} are not synchronized.",
    ],
}

# HR4
HR4_QUESTIONS = [
    "Is the period of {a}'s periodic component shorter than that of {b}?",
    "Does {a} oscillate faster (shorter period) than {b}?",
    "Compare the periodicities: is {a}'s period less than {b}'s?",
]
HR4_ANSWERS = {
    'yes': [
        "Yes, {a}'s period is shorter than {b}'s.",
        "{a} oscillates faster than {b}.",
    ],
    'no': [
        "No, {a}'s period is not shorter than {b}'s.",
        "{a} does not oscillate faster than {b}.",
    ],
}

# HR5 — reframed (2026-04-17) from pair-counting fraction to mode-share fraction.
# Old: concordant_pairs / total_pairs (O(N^2) enumeration).
# New: count_in_dominant_trend / total_metrics (O(N) tally). Complements the
# reframed rl_corr_count which returns the dominant trend's NAME; this returns
# that trend's SHARE.
HR5_QUESTIONS = [
    "What fraction of the {n_metrics} metrics show the dominant trend direction?",
    "Of the {n_metrics} metrics, what proportion share the most common trend direction?",
    "How dominant is the most common trend? Report its share of the {n_metrics} metrics as a fraction.",
    "What share of the metrics are in the dominant trend direction?",
]
# Post-think answers use natural count phrasing ("4 of 7") rather than the
# raw fraction ("0.57 of the 7"). The fraction stays in the rewarded
# `answer:` line; the narrative just has to be consistent, not verbatim.
HR5_ANSWERS = [
    "{count} of the {n_metrics} metrics show the {trend} direction, the most common.",
    "The dominant trend direction is {trend}, shared by {count} of {n_metrics} metrics.",
    "Most metrics are in the {trend} direction — {count} of {n_metrics}.",
]

# HR6
HR6_QUESTIONS = [
    "Among the metrics with an {trend_type} trend, which has the highest {stat_label}?",
    "Of the metrics showing {trend_type} trend, which one has the greatest {stat_label}?",
    "Filter to metrics with {trend_type} trend, then find the one with the highest {stat_label}.",
]
HR6_ANSWERS = [
    "{winner} has the highest {stat_label} among the {trend_type}-trending metrics.",
    "Among {trend_type} metrics, {winner} has the highest {stat_label}.",
]

# HR7 — parametrized cross-attribute argmax coincidence (2026-04-17).
# Old: hardcoded "noisiest vs widest range" only. New: rotates through 6
# meaningful attribute pairs per sample, dramatically increasing variety
# while testing the same skill (argmax twice + equality). Adds class-balance
# via _coin_flip_keep since natural yes-rate is heavily skewed (~1/N).
#
# Attribute spec: (display_name, accessor_path).
#   accessor_path is (top_key, sub_key) — top_key='stats' for statistics dict,
#   'noise' for the noise dict.
HR7_ATTRIBUTES = {
    'noise':  ('noise strength',    ('noise', 'strength')),
    'range':  ('value range',       ('stats', 'range')),
    'mean':   ('mean',              ('stats', 'mean')),
    'std':    ('standard deviation',('stats', 'std')),
    'max':    ('maximum value',     ('stats', 'max')),
    'min':    ('minimum value',     ('stats', 'min')),
}

# Meaningful pairs: avoid mathematically tautological pairs like (max, range)
# or (min, range) where one is a near-deterministic function of the other.
HR7_PAIRS = [
    ('noise', 'range'),
    ('noise', 'std'),
    ('mean', 'std'),
    ('mean', 'max'),
    ('std', 'range'),
    ('mean', 'min'),
]

HR7_QUESTIONS = [
    "Is the metric with the highest {x_name} also the one with the highest {y_name}?",
    "Does the metric maximizing {x_name} also maximize {y_name}?",
    "Among the metrics, is the top-{x_name} winner the same as the top-{y_name} winner?",
]
HR7_ANSWERS = {
    'yes': [
        "Yes, {winner} has both the highest {x_name} and the highest {y_name}.",
        "They coincide: {winner} tops both {x_name} and {y_name}.",
    ],
    'no': [
        "No, the highest {x_name} is {x_winner} but the highest {y_name} is {y_winner}.",
        "They differ: top {x_name} is {x_winner}, top {y_name} is {y_winner}.",
    ],
}

# HR8 — reframed (2026-04-17) from random-pair XOR ("is exactly one shifting
# up?") to cross-metric count of directional shifts. Old framing used academic
# language ("asymmetric between", "diverge in half-mean shift") and discarded
# signal by picking only 2 of N metrics. New framing tests the same per-metric
# skill (half-mean compare from B1) but across all metrics with a natural
# phrasing and informative numeric verdict.
HR8_UP_QUESTIONS = [
    "How many metrics shifted their mean upward from the first half to the second?",
    "Count how many of the metrics have a higher mean in the second half than the first.",
    "Among the metrics, how many show an upward shift in mean between the halves?",
]
HR8_DOWN_QUESTIONS = [
    "How many metrics shifted their mean downward from the first half to the second?",
    "Count how many of the metrics have a lower mean in the second half than the first.",
    "Among the metrics, how many show a downward shift in mean between the halves?",
]
HR8_UP_ANSWERS = [
    "{count} of the {n_metrics} metrics shifted their mean upward.",
    "{count} of {n_metrics} metrics have a higher mean in the second half.",
    "A total of {count} metrics (out of {n_metrics}) shifted mean upward.",
]
HR8_DOWN_ANSWERS = [
    "{count} of the {n_metrics} metrics shifted their mean downward.",
    "{count} of {n_metrics} metrics have a lower mean in the second half.",
    "A total of {count} metrics (out of {n_metrics}) shifted mean downward.",
]


# HR9
HR9_QUESTIONS = [
    "How many distinct trend clusters exist among the metrics?",
    "Into how many groups do the metrics fall based on their overall trend?",
    "Count the number of unique trend types across all metrics.",
]
HR9_ANSWERS_SINGULAR = [
    "There is {n} distinct trend cluster among the {total} metrics.",
    "The metrics form {n} distinct trend group.",
]
HR9_ANSWERS_PLURAL = [
    "There are {n} distinct trend clusters among the {total} metrics.",
    "The metrics form {n} distinct trend groups.",
]

# HR10
HR10_QUESTIONS = [
    "Which trend type has the most metrics?",
    "What is the dominant trend direction across all metrics?",
    "Which overall trend type is most common among the metrics?",
]
HR10_ANSWERS_SINGULAR = [
    "The dominant trend is {trend} with {count} metric: {metrics}.",
    "{trend} is the most common trend ({count} metric): {metrics}.",
]
HR10_ANSWERS_PLURAL = [
    "The dominant trend is {trend} with {count} metrics: {metrics}.",
    "{trend} is the most common trend ({count} metrics): {metrics}.",
]

# HR11
HR11_QUESTIONS = [
    "Which trend direction (increase, decrease, or keep steady) is most common among the metrics?",
    "What is the dominant trend direction across all metrics?",
    "Which overall trend is shared by the most metrics?",
    "Among the metrics, which trend direction appears most often?",
]
HR11_ANSWERS = [
    "The most common trend direction is {trend}.",
    "Across the metrics, the dominant trend direction is {trend}.",
    "The trend direction shared by the most metrics is {trend}.",
    "Most metrics share a common trend: {trend}.",
]

# HR12
HR12_QUESTIONS = [
    "Among the {count} metrics with {trend} trend, do any have synchronized local events?",
    "Of the metrics showing {trend} trend, are any event-correlated (events within 20 positions)?",
    "Do any of the {count} {trend}-trending metrics share temporally close events?",
]
HR12_ANSWERS = {
    'yes': [
        "Yes, among the {trend}-trending metrics ({metrics}), some have events within 20 positions.",
        "Events are synchronized among {trend}-trending metrics: {metrics}.",
    ],
    'no': [
        "No, the {trend}-trending metrics ({metrics}) do not have temporally close events.",
        "No event synchronization found among {trend}-trending metrics.",
    ],
}


# ============================================================
# Generator
# ============================================================

class CompositionalCrossMetricGenerator:
    """Generates RL cross-metric compositions (HR1-HR8)."""

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

    # --- HR1: Stat ratio with threshold ---

    def generate_stat_ratio(self) -> Optional[Dict[str, Any]]:
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

        va_f, vb_f = float(va), float(vb)
        actual_ratio = va_f / vb_f
        # Balance verdict via coin flip — pick k so threshold is reachable
        want_yes = random.random() < 0.5
        if want_yes:
            k = round(random.uniform(0.5, max(0.5, actual_ratio - 0.1)), 2)
        else:
            k = round(random.uniform(actual_ratio + 0.1, actual_ratio + 3.0), 2)
        k = max(0.1, k)
        threshold = round(k * vb_f, 2)
        verdict = "yes" if round(va_f, 2) >= threshold else "no"

        answer_text = random.choice(HR1_ANSWERS[verdict]).format(
            a=a, b=b, stat_label=stat_label,
            va=_fmt(va_f, 2), vb=_fmt(vb_f, 2), k=_fmt(k),
        )
        return _make_rl_result(
            random.choice(HR1_QUESTIONS).format(a=a, b=b, stat_label=stat_label, k=_fmt(k)),
            verdict, 'rl_cross_stat_ratio', answer_text, self.seq_len,
            sub_type=stat_key,
        )

    # --- HR2: Full metric ordering ---

    def generate_full_ordering(self) -> Optional[Dict[str, Any]]:
        if self.n < 3:
            return None
        # Mean via chunk-and-average blows up for long series; cap at 256.
        choices = ['max', 'min', 'range']
        if self.seq_len <= 256:
            choices.append('mean')
        stat_key = random.choice(choices)
        stat_label = _STAT_LABELS[stat_key]

        values = []
        for i in range(self.n):
            v = self._stats(i).get(stat_key)
            if v is not None:
                values.append((self.metrics[i], round(float(v), 2)))
        if len(values) < 3:
            return None

        values.sort(key=lambda x: x[1], reverse=True)
        ordering = ", ".join(name for name, _ in values)

        answer_text = random.choice(HR2_ANSWERS).format(
            stat_label=stat_label, ordering=ordering,
        )
        return _make_rl_result(
            random.choice(HR2_QUESTIONS).format(stat_label=stat_label),
            ordering, 'rl_cross_full_ordering', answer_text, self.seq_len,
            sub_type=stat_key,
        )

    # --- HR3: Event synchronization ---

    def generate_event_sync(self) -> Optional[Dict[str, Any]]:
        if self.n < 2:
            return None
        # Find pairs with events
        with_events = [i for i in range(self.n) if self.attrs[i].get('local')]
        if len(with_events) < 2:
            return None

        i, j = random.sample(with_events, 2)
        a, b = self.metrics[i], self.metrics[j]
        t = random.choice([5, 10, 15, 20, 30])

        events_a = [ev.get('position_start', ev.get('position', 0))
                     for ev in self.attrs[i].get('local', [])]
        events_b = [ev.get('position_start', ev.get('position', 0))
                     for ev in self.attrs[j].get('local', [])]

        # Find closest pair
        best_dist, best_pa, best_pb = float('inf'), 0, 0
        for pa in events_a:
            for pb in events_b:
                d = abs(pa - pb)
                if d < best_dist:
                    best_dist, best_pa, best_pb = d, pa, pb

        verdict = "yes" if best_dist <= t else "no"

        if verdict == "yes":
            answer_text = random.choice(HR3_ANSWERS['yes']).format(
                a=a, b=b, pa=best_pa, pb=best_pb, dist=best_dist, t=t,
            )
        else:
            answer_text = random.choice(HR3_ANSWERS['no']).format(a=a, b=b, t=t)

        return _make_rl_result(
            random.choice(HR3_QUESTIONS).format(a=a, b=b, t=t),
            verdict, 'rl_cross_event_sync', answer_text, self.seq_len,
        )

    # --- HR4: Period comparison ---

    def generate_period_compare(self) -> Optional[Dict[str, Any]]:
        if self.n < 2:
            return None
        periodic = []
        for i in range(self.n):
            s = self.attrs[i].get('seasonal', {})
            p = s.get('period', 0)
            if p > 0 and s.get('frequency_type', 'no periodicity') != 'no periodicity':
                periodic.append((i, round(p, 1)))
        if len(periodic) < 2:
            return None

        (i, pa), (j, pb) = random.sample(periodic, 2)
        a, b = self.metrics[i], self.metrics[j]
        verdict = "yes" if pa < pb else "no"

        answer_text = random.choice(HR4_ANSWERS[verdict]).format(
            a=a, b=b, pa=pa, pb=pb,
        )
        return _make_rl_result(
            random.choice(HR4_QUESTIONS).format(a=a, b=b),
            verdict, 'rl_cross_period_compare', answer_text, self.seq_len,
        )

    # --- HR5: Dominant trend share (reframed from pair-concordance fraction) ---

    def generate_trend_concordance(self) -> Optional[Dict[str, Any]]:
        """Fraction of metrics in the dominant trend direction.

        Reframed (2026-04-17) from concordant_pairs / total_pairs (O(N^2)) to
        mode_count / N (O(N)). Natural phrasing, same underlying skill tested.
        Skips ambiguous ties at the top so the verdict is unambiguous.
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

        answer_text = random.choice(HR5_ANSWERS).format(
            trend=mode_trend, count=mode_count, n_metrics=self.n,
        )
        return _make_rl_result(
            random.choice(HR5_QUESTIONS).format(n_metrics=self.n),
            verdict, 'rl_cross_trend_concordance', answer_text, self.seq_len,
        )

    # --- HR6: Conditional cross query ---

    def generate_conditional_query(self) -> Optional[Dict[str, Any]]:
        if self.n < 3:
            return None
        trend_type = random.choice(['increase', 'decrease', 'keep steady'])
        choices = ['std', 'max', 'range']
        if self.seq_len <= 256:
            choices.append('mean')
        stat_key = random.choice(choices)
        stat_label = _STAT_LABELS.get(stat_key, stat_key)

        filtered = []
        for i in range(self.n):
            if self._trend_type(i) == trend_type:
                v = self._stats(i).get(stat_key)
                if v is not None:
                    filtered.append((i, round(float(v), 2)))

        if len(filtered) < 2:
            return None

        winner_idx, winner_val = max(filtered, key=lambda x: x[1])
        winner = self.metrics[winner_idx]

        answer_text = random.choice(HR6_ANSWERS).format(
            winner=winner, stat_label=stat_label,
            val=_fmt(winner_val), trend_type=trend_type,
        )
        return _make_rl_result(
            random.choice(HR6_QUESTIONS).format(
                trend_type=trend_type, stat_label=stat_label,
            ),
            winner, 'rl_cross_conditional_query', answer_text, self.seq_len,
            sub_type=f"{trend_type}_{stat_key}",
        )

    # --- HR7: Cross-attribute argmax coincidence (parametrized + balanced) ---

    def _read_attribute(self, i: int, attr_key: str) -> Optional[float]:
        """Look up an attribute by key for metric i, returning None if absent."""
        if attr_key not in HR7_ATTRIBUTES:
            return None
        _, (top_key, sub_key) = HR7_ATTRIBUTES[attr_key]
        if top_key == 'stats':
            container = self._stats(i)
        elif top_key == 'noise':
            container = self.attrs[i].get('noise', {})
        else:
            return None
        val = container.get(sub_key)
        if val is None:
            return None
        try:
            return round(float(val), 2)
        except (TypeError, ValueError):
            return None

    def generate_attribute_corr(self) -> Optional[Dict[str, Any]]:
        """Pick a random attribute pair, check whether the same metric tops both.

        Reframed (2026-04-17) from hardcoded "noisiest vs widest range" to
        parametrized cross-attribute argmax coincidence. 6 attribute pairs
        rotated per sample. Class-balanced via _coin_flip_keep on yes/no
        because natural yes-rate is ~1/N (heavily skewed).
        """
        if self.n < 3:
            return None

        attr_x, attr_y = random.choice(HR7_PAIRS)

        x_vals: List[Tuple[int, float]] = []
        y_vals: List[Tuple[int, float]] = []
        for i in range(self.n):
            xv = self._read_attribute(i, attr_x)
            yv = self._read_attribute(i, attr_y)
            if xv is not None and yv is not None:
                x_vals.append((i, xv))
                y_vals.append((i, yv))

        if len(x_vals) < 3:
            return None

        # Special-case noise: argmax over all-zero noise is meaningless. Skip
        # if fewer than 3 metrics have noise > 0 (low-signal sample).
        if attr_x == 'noise' and sum(1 for _, v in x_vals if v > 0) < 3:
            return None
        if attr_y == 'noise' and sum(1 for _, v in y_vals if v > 0) < 3:
            return None

        x_winner_idx, _ = max(x_vals, key=lambda t: t[1])
        y_winner_idx, _ = max(y_vals, key=lambda t: t[1])

        verdict = "yes" if x_winner_idx == y_winner_idx else "no"

        # Class balance — namespaced PER PAIR, not just per eval_type.
        # Different attribute pairs have different natural yes-rates (e.g.
        # (std, range) ~50%, (mean, min) ~15%). A single per-eval_type
        # counter would push each pair toward the GLOBAL mean, leaving each
        # pair internally skewed — and the skew is predictable from the
        # question text, which RL can exploit as a shortcut. Per-pair
        # namespacing makes each pair land at 50/50 internally so no
        # pair-name → verdict shortcut exists.
        if not _coin_flip_keep(
            verdict, ['yes', 'no'],
            f'rl_cross_attribute_corr/{attr_x}_{attr_y}',
        ):
            return None

        x_name = HR7_ATTRIBUTES[attr_x][0]
        y_name = HR7_ATTRIBUTES[attr_y][0]
        x_winner = self.metrics[x_winner_idx]
        y_winner = self.metrics[y_winner_idx]

        if verdict == "yes":
            answer_text = random.choice(HR7_ANSWERS['yes']).format(
                winner=x_winner, x_name=x_name, y_name=y_name,
            )
        else:
            answer_text = random.choice(HR7_ANSWERS['no']).format(
                x_winner=x_winner, x_name=x_name,
                y_winner=y_winner, y_name=y_name,
            )

        question = random.choice(HR7_QUESTIONS).format(
            x_name=x_name, y_name=y_name,
        )
        return _make_rl_result(
            question, verdict, 'rl_cross_attribute_corr',
            answer_text, self.seq_len,
        )

    # --- HR8: Count of metrics with directional mean shift ---

    def generate_asymmetric_behavior(self) -> Optional[Dict[str, Any]]:
        """Count metrics whose mean shifted upward (or downward) between halves.

        Reframed (2026-04-17) from random-pair XOR ("is exactly one shifting
        up?") to cross-metric count of directional shifts. Tests the same
        per-metric skill (half-mean compare from B1 / rl_half_mean_compare)
        applied to all metrics and aggregated as a count.
        """
        if self.n < 3:
            return None

        halves = []
        for i in range(self.n):
            fh = self._stats(i).get('first_half_mean')
            sh = self._stats(i).get('second_half_mean')
            if fh is None or sh is None:
                return None
            halves.append((float(fh), float(sh)))

        direction = random.choice(['up', 'down'])
        if direction == 'up':
            count = sum(1 for fh, sh in halves if sh > fh)
            question = random.choice(HR8_UP_QUESTIONS)
            answer_text = random.choice(HR8_UP_ANSWERS).format(
                count=count, n_metrics=self.n,
            )
        else:
            count = sum(1 for fh, sh in halves if sh < fh)
            question = random.choice(HR8_DOWN_QUESTIONS)
            answer_text = random.choice(HR8_DOWN_ANSWERS).format(
                count=count, n_metrics=self.n,
            )

        verdict = str(count)
        return _make_rl_result(
            question, verdict, 'rl_cross_asymmetric_behavior',
            answer_text, self.seq_len,
        )

    # --- HR9: Cluster count ---

    def generate_cluster_count(self) -> Optional[Dict[str, Any]]:
        """How many distinct trend clusters exist among the metrics?"""
        if self.n < 3:
            return None
        trends = [self._trend_type(i) for i in range(self.n)]
        if 'unknown' in trends:
            return None
        distinct = len(set(trends))
        verdict = str(distinct)
        templates = HR9_ANSWERS_SINGULAR if distinct == 1 else HR9_ANSWERS_PLURAL
        answer_text = random.choice(templates).format(
            n=distinct, total=self.n,
        )
        return _make_rl_result(
            random.choice(HR9_QUESTIONS),
            verdict, 'rl_cluster_count', answer_text, self.seq_len,
        )

    # --- HR10: Dominant cluster ---

    def generate_cluster_dominant(self) -> Optional[Dict[str, Any]]:
        """Which trend type has the most metrics?"""
        if self.n < 3:
            return None
        trends = [self._trend_type(i) for i in range(self.n)]
        if 'unknown' in trends:
            return None
        from collections import Counter
        counts = Counter(trends)
        dominant_type, dominant_count = counts.most_common(1)[0]
        # Find metrics in the dominant cluster
        dominant_metrics = [self.metrics[i] for i in range(self.n)
                           if trends[i] == dominant_type]
        verdict = dominant_type
        templates = HR10_ANSWERS_SINGULAR if dominant_count == 1 else HR10_ANSWERS_PLURAL
        answer_text = random.choice(templates).format(
            trend=dominant_type, count=dominant_count,
            metrics=", ".join(dominant_metrics),
        )
        return _make_rl_result(
            random.choice(HR10_QUESTIONS),
            verdict, 'rl_cluster_dominant', answer_text, self.seq_len,
        )

    # --- HR11: Most common trend direction (reframed from pair-count) ---

    def generate_corr_count(self) -> Optional[Dict[str, Any]]:
        """Which trend direction is most common among the metrics?

        Reframed (2026-04-17) from "count of concordant pairs" to
        "argmax over trend-type tally". Same underlying information but
        O(N) instead of O(N^2) and more naturally phrasable. Categorical
        verdict (trend type name) instead of numeric pair count.
        """
        if self.n < 3:
            return None
        trends = [self._trend_type(i) for i in range(self.n)]
        if 'unknown' in trends:
            return None

        from collections import Counter
        counts = Counter(trends)
        ranked = counts.most_common()

        # Skip ambiguous ties at the top — verdict must be unambiguous.
        if len(ranked) >= 2 and ranked[0][1] == ranked[1][1]:
            return None

        verdict = ranked[0][0]  # may be multi-word ("keep steady")
        answer_text = random.choice(HR11_ANSWERS).format(trend=verdict)
        return _make_rl_result(
            random.choice(HR11_QUESTIONS),
            verdict, 'rl_corr_count', answer_text, self.seq_len,
        )

    # --- HR12: Conditional correlation ---

    def generate_corr_conditional(self) -> Optional[Dict[str, Any]]:
        """Among metrics with a specific trend, are any local-event correlated?

        Verdict balance notes:
          - Original tolerance `<= 20` at any seq_len made "yes" essentially
            unreachable (observed 100% "no" in 2K+ samples pre-fix). Tolerance
            now scales with seq_len (`max(20, seq_len // 10)`): at seq_len=256
            → 25, at seq_len=768 → 76, at seq_len=4096 → 409.
          - `_coin_flip_keep` enforces 50/50 at the source so any residual
            skew downsamples the majority stream.
        """
        if self.n < 3:
            return None
        trends = [self._trend_type(i) for i in range(self.n)]
        if 'unknown' in trends:
            return None

        # Pick the most common trend as filter
        from collections import Counter
        trend_counts = Counter(trends)
        target_trend = trend_counts.most_common(1)[0][0]
        filtered_idx = [i for i in range(self.n) if trends[i] == target_trend]
        if len(filtered_idx) < 2:
            return None

        # Sync tolerance scaled by seq_len so "yes" is reachable at all lengths.
        tolerance = max(20, self.seq_len // 10)

        has_sync = False
        for i, j in combinations(filtered_idx, 2):
            events_i = self.attrs[i].get('local_changes', [])
            events_j = self.attrs[j].get('local_changes', [])
            pos_i = {e.get('position_start', -999) for e in events_i}
            pos_j = {e.get('position_start', -999) for e in events_j}
            for pi in pos_i:
                for pj in pos_j:
                    if abs(pi - pj) <= tolerance:
                        has_sync = True
                        break
                if has_sync:
                    break
            if has_sync:
                break

        verdict = "yes" if has_sync else "no"

        # Stateful 50/50 balance — drops majority-verdict candidates when
        # the stream skews. Required because event positions are randomly
        # placed by the TS generator, and "yes" remains rarer than "no"
        # in steady state.
        if not _coin_flip_keep(verdict, ['yes', 'no'], 'rl_corr_conditional'):
            return None

        filtered_names = [self.metrics[i] for i in filtered_idx]
        answer_text = random.choice(HR12_ANSWERS[verdict]).format(
            trend=target_trend,
            metrics=", ".join(filtered_names),
        )
        return _make_rl_result(
            random.choice(HR12_QUESTIONS).format(
                trend=target_trend, count=len(filtered_idx),
            ),
            verdict, 'rl_corr_conditional', answer_text, self.seq_len,
        )

    # --- Generate all ---

    def generate_all(self) -> List[Dict[str, Any]]:
        results = []
        generators = [
            self.generate_stat_ratio,
            self.generate_full_ordering,
            self.generate_event_sync,
            self.generate_period_compare,
            self.generate_trend_concordance,
            self.generate_conditional_query,
            self.generate_attribute_corr,
            self.generate_asymmetric_behavior,
            self.generate_cluster_count,
            self.generate_cluster_dominant,
            self.generate_corr_count,
            self.generate_corr_conditional,
        ]
        for gen_fn in generators:
            try:
                r = gen_fn()
                if r is not None:
                    results.append(r)
            except Exception:
                logger.warning(
                    f"CrossMetric RL {gen_fn.__name__} failed", exc_info=True,
                )
        return results
