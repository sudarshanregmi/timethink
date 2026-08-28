"""Atomic cross-metric QA generators (Category H: SFT).

Teaches the model basic cross-metric reasoning operations:
    H1: atomic_cross_stat_compare  — pairwise stat comparison (yes/no)
    H2: atomic_cross_ranking       — which metric has highest/lowest stat
    H3: atomic_cross_filtering     — which metrics have property X
    H4: atomic_cross_counting      — how many metrics have property X
    H5: atomic_cross_trend_align   — do A and B trend the same way (yes/no)

Each produces a full think block showing step-by-step cross-metric reasoning.
"""

import random
from typing import Any, Dict, List, Optional

import numpy as np
from loguru import logger

from synth.align.generators.atomic_qa import _fmt


# ============================================================
# Think block helpers
# ============================================================

_OPENERS = [
    "Let me compare the metrics.",
    "I'll check each metric systematically.",
    "Let me examine the metrics one by one.",
    "I need to look at the metrics to answer this.",
    "Let me work through this for each metric.",
    "I'll analyze the metrics step by step.",
]


def _build_cross_think(lines: List[str], verdict) -> str:
    """Build a cross-metric SFT think block."""
    opener = random.choice(_OPENERS)
    content = "\n".join([opener, ""] + lines + ["", f"answer: {verdict}"])
    return f"<think>\n{content}\n</think>"


# ============================================================
# Question / answer templates
# ============================================================

_STAT_LABELS = {
    'mean': 'mean', 'std': 'standard deviation', 'min': 'minimum value',
    'max': 'maximum value', 'range': 'value range',
}

# H1: Pairwise stat compare
H1_QUESTIONS = [
    "Is the {stat_label} of {a} greater than the {stat_label} of {b}?",
    "Does {a} have a higher {stat_label} than {b}?",
    "Compare the {stat_label} of {a} and {b}: is {a}'s larger?",
    "For {stat_label}, does {a} exceed {b}?",
    "Is {a}'s {stat_label} higher than that of {b}?",
]

H1_ANSWERS_YES = [
    "Yes, {a}'s {stat_label} is greater than {b}'s.",
    "The {stat_label} of {a} exceeds that of {b}.",
]
H1_ANSWERS_NO = [
    "No, {a}'s {stat_label} is not greater than {b}'s.",
    "The {stat_label} of {a} does not exceed {b}'s.",
]

# H2: Metric ranking
H2_QUESTIONS_HIGHEST = [
    "Which metric has the highest {stat_label}?",
    "Among all the metrics, which one has the greatest {stat_label}?",
    "Which time series shows the highest {stat_label}?",
    "Identify the metric with the largest {stat_label}.",
]
H2_QUESTIONS_LOWEST = [
    "Which metric has the lowest {stat_label}?",
    "Among all the metrics, which one has the smallest {stat_label}?",
    "Which time series shows the lowest {stat_label}?",
    "Identify the metric with the least {stat_label}.",
]
H2_ANSWERS = [
    "{winner} has the {direction} {stat_label} at {val}.",
    "The metric with the {direction} {stat_label} is {winner} ({val}).",
]

# H3: Metric filtering
H3_TREND_QUESTIONS = [
    "Which metrics show an overall {trend_type} trend?",
    "Identify the metrics with an {trend_type} trend.",
    "Which of the time series have an {trend_type} trend direction?",
]
H3_EVENT_QUESTIONS = [
    "Which metrics have local events?",
    "Identify the metrics that contain local fluctuations or anomalies.",
    "Which time series exhibit local events?",
]
H3_PERIODIC_QUESTIONS = [
    "Which metrics exhibit periodic behavior?",
    "Identify the metrics with a periodic or seasonal component.",
    "Which time series show periodicity?",
]
H3_TREND_ANSWERS = [
    "The metrics with {trend_type} trends are {names}.",
    "The following metrics show a {trend_type} trend: {names}.",
    "{names} show a {trend_type} trend.",
]
H3_EVENT_ANSWERS = [
    "The metrics containing local {event_word} are {names}.",
    "The following metrics have local {event_word}: {names}.",
    "{names} contain local {event_word}.",
]
H3_PERIODIC_ANSWERS = [
    "The periodic metrics are {names}.",
    "The following metrics show periodic behavior: {names}.",
    "{names} exhibit periodicity.",
]
H3_NONE_ANSWERS = [
    "No metric satisfies the condition.",
    "None of the metrics match.",
]

# H4: Cross-metric counting
H4_TREND_QUESTIONS = [
    "How many metrics show an overall {trend_type} trend?",
    "Count the metrics with an {trend_type} trend direction.",
]
H4_EVENT_QUESTIONS = [
    "How many metrics have local events?",
    "Count the metrics that contain local fluctuations.",
]
H4_PERIODIC_QUESTIONS = [
    "How many metrics exhibit periodic behavior?",
    "Count the metrics with a periodic component.",
]
H4_TREND_ANSWERS_SINGULAR = [
    "{count} metric shows a {trend_type} trend: {names}.",
    "The number of metrics with a {trend_type} trend is {count} ({names}).",
    "There is {count} metric with a {trend_type} trend: {names}.",
]
H4_TREND_ANSWERS_PLURAL = [
    "{count} metrics show a {trend_type} trend: {names}.",
    "The number of metrics with a {trend_type} trend is {count} ({names}).",
    "There are {count} metrics with a {trend_type} trend: {names}.",
]
H4_EVENT_ANSWERS_SINGULAR = [
    "{count} metric contains local {event_word}: {names}.",
    "The number of metrics with local {event_word} is {count} ({names}).",
    "There is {count} metric containing local {event_word}: {names}.",
]
H4_EVENT_ANSWERS_PLURAL = [
    "{count} metrics contain local {event_word}: {names}.",
    "The number of metrics with local {event_word} is {count} ({names}).",
    "There are {count} metrics containing local {event_word}: {names}.",
]
H4_PERIODIC_ANSWERS_SINGULAR = [
    "{count} metric is periodic: {names}.",
    "The number of periodic metrics is {count} ({names}).",
    "There is {count} periodic metric: {names}.",
]
H4_PERIODIC_ANSWERS_PLURAL = [
    "{count} metrics are periodic: {names}.",
    "The number of periodic metrics is {count} ({names}).",
    "There are {count} periodic metrics: {names}.",
]
H4_NONE_ANSWERS = [
    "No metric satisfies the condition.",
    "None of the metrics match.",
]

# H5: Pairwise trend alignment
H5_QUESTIONS = [
    "Do {a} and {b} have the same overall trend direction?",
    "Is the overall trend of {a} the same as that of {b}?",
    "Do {a} and {b} trend in the same direction?",
    "Compare the trend directions of {a} and {b}: are they the same?",
]
H5_ANSWERS_YES = [
    "Yes, both {a} and {b} show an {trend} trend.",
    "They share the same trend direction: {trend}.",
]
H5_ANSWERS_NO = [
    "No, {a} shows {trend_a} while {b} shows {trend_b}.",
    "Their trends differ: {a} is {trend_a}, {b} is {trend_b}.",
]


# ============================================================
# Generator
# ============================================================

class AtomicCrossMetricGenerator:
    """Generates atomic cross-metric SFT examples (H1-H5)."""

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

    def _has_events(self, i: int) -> bool:
        return len(self.attrs[i].get('local', [])) > 0

    def _is_periodic(self, i: int) -> bool:
        s = self.attrs[i].get('seasonal', {})
        return (
            s.get('period', 0) > 0
            and s.get('frequency_type', 'no periodicity') != 'no periodicity'
        )

    def _make_result(self, question, think, answer_text, eval_type, verdict,
                     **extra_meta) -> Dict[str, Any]:
        meta = {'length': self.seq_len, 'verdict': str(verdict)}
        meta.update(extra_meta)
        return {
            'question': question,
            'answer': f"{think}\n{answer_text}",
            'eval_type': eval_type,
            'eval_metadata': meta,
        }

    # --- H1: Pairwise stat compare ---

    def generate_pairwise_stat_compare(self) -> Optional[Dict[str, Any]]:
        """H1: Is stat(A) > stat(B)?"""
        if self.n < 2:
            return None

        stat_key = random.choice(list(_STAT_LABELS.keys()))
        stat_label = _STAT_LABELS[stat_key]
        i, j = random.sample(range(self.n), 2)
        a, b = self.metrics[i], self.metrics[j]

        va = self._stats(i).get(stat_key)
        vb = self._stats(j).get(stat_key)
        if va is None or vb is None:
            return None

        va_r, vb_r = round(float(va), 2), round(float(vb), 2)
        verdict = "yes" if va_r > vb_r else "no"

        think_lines = [
            f"{a}: {stat_label} = {_fmt(va_r)}",
            f"{b}: {stat_label} = {_fmt(vb_r)}",
            "",
            f"{_fmt(va_r)} {'>' if va_r > vb_r else '<='} {_fmt(vb_r)} → {verdict}",
        ]
        think = _build_cross_think(think_lines, verdict)

        templates = H1_ANSWERS_YES if verdict == "yes" else H1_ANSWERS_NO
        answer_text = random.choice(templates).format(
            a=a, b=b, stat_label=stat_label,
            va=_fmt(va_r), vb=_fmt(vb_r),
        )
        return self._make_result(
            random.choice(H1_QUESTIONS).format(a=a, b=b, stat_label=stat_label),
            think, answer_text, 'atomic_cross_stat_compare', verdict,
            sub_type=stat_key,
        )

    # --- H2: Metric ranking ---

    def generate_metric_ranking(self) -> Optional[Dict[str, Any]]:
        """H2: Which metric has the highest/lowest stat?"""
        if self.n < 2:
            return None

        stat_key = random.choice(list(_STAT_LABELS.keys()))
        stat_label = _STAT_LABELS[stat_key]
        direction = random.choice(['highest', 'lowest'])

        values = []
        for i in range(self.n):
            v = self._stats(i).get(stat_key)
            if v is not None:
                values.append((i, round(float(v), 2)))
        if len(values) < 2:
            return None

        if direction == 'highest':
            winner_idx, winner_val = max(values, key=lambda x: x[1])
        else:
            winner_idx, winner_val = min(values, key=lambda x: x[1])
        winner = self.metrics[winner_idx]

        think_lines = []
        for idx, val in values:
            think_lines.append(f"{self.metrics[idx]}: {stat_label} = {_fmt(val)}")
        think_lines.append("")
        think_lines.append(f"{direction.capitalize()}: {winner} at {_fmt(winner_val)}")
        think = _build_cross_think(think_lines, winner)

        templates = H2_QUESTIONS_HIGHEST if direction == 'highest' else H2_QUESTIONS_LOWEST
        answer_text = random.choice(H2_ANSWERS).format(
            winner=winner, direction=direction,
            stat_label=stat_label, val=_fmt(winner_val),
        )
        return self._make_result(
            random.choice(templates).format(stat_label=stat_label),
            think, answer_text, 'atomic_cross_ranking', winner,
            sub_type=f"{direction}_{stat_key}",
        )

    # --- H3: Metric filtering ---

    def generate_metric_filtering(self) -> Optional[Dict[str, Any]]:
        """H3: Which metrics have property X?"""
        if self.n < 2:
            return None

        filter_type = random.choice(['trend', 'events', 'periodic'])
        answer_templates = None
        answer_kwargs = {}

        if filter_type == 'trend':
            trend_type = random.choice(['increase', 'decrease', 'keep steady'])
            matches = [
                self.metrics[i] for i in range(self.n)
                if self._trend_type(i) == trend_type
            ]
            question = random.choice(H3_TREND_QUESTIONS).format(trend_type=trend_type)
            think_lines = []
            for i in range(self.n):
                t = self._trend_type(i)
                mark = "✓" if t == trend_type else "✗"
                think_lines.append(f"{self.metrics[i]}: trend = {t} {mark}")
            answer_templates = H3_TREND_ANSWERS
            answer_kwargs = {'trend_type': trend_type}

        elif filter_type == 'events':
            matches = [
                self.metrics[i] for i in range(self.n)
                if self._has_events(i)
            ]
            question = random.choice(H3_EVENT_QUESTIONS)
            think_lines = []
            for i in range(self.n):
                n_ev = len(self.attrs[i].get('local', []))
                mark = "✓" if n_ev > 0 else "✗"
                think_lines.append(f"{self.metrics[i]}: {n_ev} events {mark}")
            answer_templates = H3_EVENT_ANSWERS
            answer_kwargs = {'event_word': random.choice(['fluctuations', 'anomalies', 'events'])}

        else:  # periodic
            matches = [
                self.metrics[i] for i in range(self.n)
                if self._is_periodic(i)
            ]
            question = random.choice(H3_PERIODIC_QUESTIONS)
            think_lines = []
            for i in range(self.n):
                s = self.attrs[i].get('seasonal', {})
                period = s.get('period', 0)
                freq = s.get('frequency_type', 'no periodicity')
                is_p = self._is_periodic(i)
                mark = "✓" if is_p else "✗"
                think_lines.append(f"{self.metrics[i]}: period={round(period, 1)}, {freq} {mark}")
            answer_templates = H3_PERIODIC_ANSWERS

        if not matches:
            verdict = "none"
        else:
            verdict = ", ".join(matches)

        think_lines.append("")
        think_lines.append(f"Matching: {verdict}")
        think = _build_cross_think(think_lines, verdict)

        if not matches:
            answer_text = random.choice(H3_NONE_ANSWERS)
        else:
            answer_text = random.choice(answer_templates).format(names=verdict, **answer_kwargs)
        return self._make_result(
            question, think, answer_text, 'atomic_cross_filtering', verdict,
            sub_type=filter_type,
        )

    # --- H4: Cross-metric counting ---

    def generate_cross_counting(self) -> Optional[Dict[str, Any]]:
        """H4: How many metrics have property X?"""
        if self.n < 2:
            return None

        filter_type = random.choice(['trend', 'events', 'periodic'])
        answer_templates = None
        answer_kwargs = {}
        matched_names = []

        if filter_type == 'trend':
            trend_type = random.choice(['increase', 'decrease', 'keep steady'])
            matched_names = [self.metrics[i] for i in range(self.n) if self._trend_type(i) == trend_type]
            count = len(matched_names)
            question = random.choice(H4_TREND_QUESTIONS).format(trend_type=trend_type)
            think_lines = []
            for i in range(self.n):
                t = self._trend_type(i)
                mark = "✓" if t == trend_type else "✗"
                think_lines.append(f"{self.metrics[i]}: {t} {mark}")
            _is_single = len(matched_names) == 1
            answer_templates = H4_TREND_ANSWERS_SINGULAR if _is_single else H4_TREND_ANSWERS_PLURAL
            answer_kwargs = {'trend_type': trend_type}

        elif filter_type == 'events':
            matched_names = [self.metrics[i] for i in range(self.n) if self._has_events(i)]
            count = len(matched_names)
            question = random.choice(H4_EVENT_QUESTIONS)
            think_lines = []
            for i in range(self.n):
                n_ev = len(self.attrs[i].get('local', []))
                mark = "✓" if n_ev > 0 else "✗"
                think_lines.append(f"{self.metrics[i]}: {n_ev} events {mark}")
            _is_single = count == 1
            answer_templates = H4_EVENT_ANSWERS_SINGULAR if _is_single else H4_EVENT_ANSWERS_PLURAL
            answer_kwargs = {'event_word': random.choice(['fluctuations', 'anomalies', 'events'])}

        else:  # periodic
            matched_names = [self.metrics[i] for i in range(self.n) if self._is_periodic(i)]
            count = len(matched_names)
            question = random.choice(H4_PERIODIC_QUESTIONS)
            think_lines = []
            for i in range(self.n):
                is_p = self._is_periodic(i)
                mark = "✓" if is_p else "✗"
                think_lines.append(f"{self.metrics[i]}: {'periodic' if is_p else 'not periodic'} {mark}")
            _is_single = count == 1
            answer_templates = H4_PERIODIC_ANSWERS_SINGULAR if _is_single else H4_PERIODIC_ANSWERS_PLURAL

        verdict = str(count)
        think_lines.append("")
        think_lines.append(f"Count: {count}")
        think = _build_cross_think(think_lines, verdict)

        if count == 0:
            answer_text = random.choice(H4_NONE_ANSWERS)
        else:
            names_str = ", ".join(matched_names)
            answer_text = random.choice(answer_templates).format(
                count=count, names=names_str, **answer_kwargs,
            )
        return self._make_result(
            question, think, answer_text, 'atomic_cross_counting', verdict,
            sub_type=filter_type,
        )

    # --- H5: Pairwise trend alignment ---

    def generate_pairwise_trend_align(self) -> Optional[Dict[str, Any]]:
        """H5: Do A and B have the same overall trend?"""
        if self.n < 2:
            return None

        i, j = random.sample(range(self.n), 2)
        a, b = self.metrics[i], self.metrics[j]
        ta, tb = self._trend_type(i), self._trend_type(j)

        if ta == 'unknown' or tb == 'unknown':
            return None

        verdict = "yes" if ta == tb else "no"

        think_lines = [
            f"{a}: overall trend = {ta}",
            f"{b}: overall trend = {tb}",
            "",
            f"{ta} {'==' if ta == tb else '!='} {tb} → {verdict}",
        ]
        think = _build_cross_think(think_lines, verdict)

        if verdict == "yes":
            answer_text = random.choice(H5_ANSWERS_YES).format(
                a=a, b=b, trend=ta,
            )
        else:
            answer_text = random.choice(H5_ANSWERS_NO).format(
                a=a, b=b, trend_a=ta, trend_b=tb,
            )
        return self._make_result(
            random.choice(H5_QUESTIONS).format(a=a, b=b),
            think, answer_text, 'atomic_cross_trend_align', verdict,
        )

    # --- Generate all ---

    def generate_all(self) -> List[Dict[str, Any]]:
        results = []
        generators = [
            self.generate_pairwise_stat_compare,
            self.generate_metric_ranking,
            self.generate_metric_filtering,
            self.generate_cross_counting,
            self.generate_pairwise_trend_align,
        ]
        for gen_fn in generators:
            try:
                r = gen_fn()
                if r is not None:
                    results.append(r)
            except Exception:
                logger.warning(
                    f"CrossMetric atomic {gen_fn.__name__} failed", exc_info=True,
                )
        return results
