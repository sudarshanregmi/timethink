"""OOD cross-metric QA generators (Category H: eval-only).

Novel cross-metric compositions never seen during training.
Tests generalization of cross-metric reasoning.

    HO1: ood_cross_corr_count        — count of concordant trend pairs
    HO2: ood_cross_trend_convergence — is second half more/less aligned? (yes/no)
    HO3: ood_cross_extrema_alignment — max positions within proximity?
    HO5: ood_cross_range_overlap     — do value ranges overlap?
    HO6: ood_cross_concordant_shift  — how many metrics shift mean up?
"""

import random
from itertools import combinations
from typing import Any, Dict, List, Optional

import numpy as np
from loguru import logger

from synth.align.generators.atomic_qa import _fmt
from synth.align.generators.compositional_qa import (
    _coin_flip_keep, _make_rl_result, _pick_balanced_param,
)


# ============================================================
# Question / answer templates
# ============================================================

# HO1
HO1_QUESTIONS = [
    "How many pairs of metrics share the same overall trend direction?",
    "Count the number of metric pairs with matching trend types.",
    "Of all metric pairs, how many are trend-concordant?",
]
HO1_ANSWERS_SINGULAR = [
    "{count} out of {total} metric pairs shares the same trend.",
    "There is {count} concordant pair (of {total} total).",
]
HO1_ANSWERS_PLURAL = [
    "{count} out of {total} metric pairs share the same trend.",
    "There are {count} concordant pairs (of {total} total).",
]

# HO2 — reframed from 3-class (converging/diverging/stable) to paired yes/no.
# Two directions are asked interchangeably so "yes" and "no" each cover a
# meaningful subset: "more aligned?" yes=converging, no=(stable or diverging);
# "less aligned?" yes=diverging, no=(stable or converging). Stable ends up as
# "no" under both framings, which is fine for binary scoring. Avoids the
# vocabulary gap (model was never trained on converging/diverging/stable words).
HO2_MORE_QUESTIONS = [
    "Is the trend agreement among metrics higher in the second half than the first?",
    "Do the metrics become more aligned in their trends over time?",
    "Is pairwise trend concordance stronger in the second half of the series?",
]
HO2_LESS_QUESTIONS = [
    "Is the trend agreement among metrics lower in the second half than the first?",
    "Do the metrics become less aligned in their trends over time?",
    "Is pairwise trend concordance weaker in the second half of the series?",
]
HO2_ANSWERS = {
    ('more', 'yes'): [
        "Yes, trend agreement is stronger in the second half.",
        "Yes, the metrics become more aligned over time.",
    ],
    ('more', 'no'): [
        "No, trend agreement is not higher in the second half.",
        "No, the metrics do not become more aligned over time.",
    ],
    ('less', 'yes'): [
        "Yes, trend agreement is weaker in the second half.",
        "Yes, the metrics become less aligned over time.",
    ],
    ('less', 'no'): [
        "No, trend agreement is not lower in the second half.",
        "No, the metrics do not become less aligned over time.",
    ],
}

# HO3
HO3_QUESTIONS = [
    "Do {a} and {b} have their maximum value at positions within {t} of each other?",
    "Are the peak positions of {a} and {b} within {t} timesteps?",
    "For {a} and {b}, are the maxima temporally aligned (within {t} positions)?",
]
HO3_ANSWERS = {
    'yes': [
        "Yes, {a} (peak at {pa}) and {b} (peak at {pb}) have aligned maxima within {t} positions.",
        "The peaks of {a} and {b} are only {dist} positions apart, within the {t}-position threshold.",
    ],
    'no': [
        "No, {a} (peak at {pa}) and {b} (peak at {pb}) have peaks {dist} positions apart, exceeding the {t}-position threshold.",
        "The maxima of {a} and {b} are not aligned; they are {dist} positions apart.",
    ],
}

# HO5
HO5_QUESTIONS = [
    "Do the value ranges of {a} and {b} overlap?",
    "Is there any overlap between the value range of {a} and {b}?",
    "Do {a} and {b} share overlapping value ranges?",
]
HO5_ANSWERS = {
    'yes': [
        "Yes, {a} and {b} have overlapping value ranges.",
        "The ranges of {a} and {b} overlap.",
    ],
    'no': [
        "No, {a} and {b} do not have overlapping value ranges.",
        "The ranges of {a} and {b} are disjoint.",
    ],
}

# HO6
HO6_QUESTIONS = [
    "How many metrics have a higher mean in the second half than the first half?",
    "Count the metrics whose second-half mean exceeds the first-half mean.",
    "How many metrics show an upward mean shift from first to second half?",
]
HO6_ANSWERS = [
    "{count} out of {total} metrics have a higher second-half mean.",
    "{count} metrics show an upward mean shift.",
]

# HO7
HO7_QUESTIONS = [
    "Is any metric in its own unique trend cluster (not sharing its trend with any other)?",
    "Are there any singleton metrics whose trend type is unique among all metrics?",
    "Does any metric have a trend that no other metric shares?",
]
HO7_ANSWERS = {
    'yes_singular': [
        "Yes, {count} metric has a unique trend: {metrics}.",
        "Singleton metric: {metrics} ({count} with a unique trend type).",
    ],
    'yes_plural': [
        "Yes, {count} metrics have unique trends: {metrics}.",
        "Singleton metrics: {metrics} ({count} with unique trend types).",
    ],
    'no': [
        "No, every metric shares its trend type with at least one other metric.",
        "No singletons — all metrics cluster with at least one partner.",
    ],
}

# HO8
HO8_QUESTIONS = [
    "Given {a} and {b} trends, and {b} and {c} trends — does {a} share a trend with {c}?",
    "If {a}'s trend matches {b}'s, and {b}'s matches {c}'s — do {a} and {c} match?",
    "Is the trend relationship transitive: {a} ~ {b} and {b} ~ {c} implies {a} ~ {c}?",
]
HO8_ANSWERS = {
    'yes': [
        "Yes, transitivity holds among the three metrics.",
        "Transitivity holds: all three share the same trend.",
    ],
    'no': [
        "No, transitivity does not hold.",
        "Transitivity breaks among the three metrics.",
    ],
}

# HO9
HO9_QUESTIONS = [
    "How many metric pairs have opposite trends (one increasing, one decreasing)?",
    "Count the anti-correlated pairs (increase vs decrease) among all metrics.",
    "How many pairs of metrics show opposing trend directions?",
]
HO9_ANSWERS_SINGULAR = [
    "{count} of {total} metric pairs has opposite trends.",
    "There is {count} anti-correlated pair out of {total} total.",
]
HO9_ANSWERS_PLURAL = [
    "{count} of {total} metric pairs have opposite trends.",
    "There are {count} anti-correlated pairs out of {total} total.",
]


# ============================================================
# Generator
# ============================================================

class OODCrossMetricGenerator:
    """Generates OOD cross-metric eval-only examples (HO1-HO9)."""

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

    # --- HO1: Pairwise correlation counting ---

    def generate_corr_count(self) -> Optional[Dict[str, Any]]:
        if self.n < 3:
            return None
        trends = [self._trend_type(i) for i in range(self.n)]
        if 'unknown' in trends:
            return None

        total = 0
        concordant = 0
        for i, j in combinations(range(self.n), 2):
            total += 1
            if trends[i] == trends[j]:
                concordant += 1

        verdict = str(concordant)
        templates = HO1_ANSWERS_SINGULAR if concordant == 1 else HO1_ANSWERS_PLURAL
        answer_text = random.choice(templates).format(count=concordant, total=total)
        return _make_rl_result(
            random.choice(HO1_QUESTIONS),
            verdict, 'ood_cross_corr_count', answer_text, self.seq_len,
        )

    # --- HO2: Trend convergence ---

    def generate_trend_convergence(self) -> Optional[Dict[str, Any]]:
        if self.n < 3:
            return None
        # Compare trend agreement in first half of trend_list vs second half
        # For each metric, split trend_list at temporal midpoint
        mid = self.seq_len // 2
        first_half_trends = []
        second_half_trends = []
        for i in range(self.n):
            tl = self.attrs[i].get('trend_list', [])
            if not tl:
                return None
            # Dominant trend in first half
            fh = [(t, s, min(e, mid)) for t, s, e in tl if s < mid]
            sh = [(t, max(s, mid), e) for t, s, e in tl if e > mid]
            if fh:
                fh_dom = max(fh, key=lambda x: x[2] - x[1])[0]
            else:
                fh_dom = tl[0][0]
            if sh:
                sh_dom = max(sh, key=lambda x: x[2] - x[1])[0]
            else:
                sh_dom = tl[-1][0]
            first_half_trends.append(fh_dom)
            second_half_trends.append(sh_dom)

        # Count concordant pairs in each half
        fh_conc = sum(1 for i, j in combinations(range(self.n), 2)
                      if first_half_trends[i] == first_half_trends[j])
        sh_conc = sum(1 for i, j in combinations(range(self.n), 2)
                      if second_half_trends[i] == second_half_trends[j])

        # Randomly pick which direction to ask about. Balances the verdict
        # distribution implicitly: each direction produces its own yes/no split.
        direction = random.choice(['more', 'less'])
        if direction == 'more':
            verdict = "yes" if sh_conc > fh_conc + 1 else "no"
            question = random.choice(HO2_MORE_QUESTIONS)
        else:
            verdict = "yes" if fh_conc > sh_conc + 1 else "no"
            question = random.choice(HO2_LESS_QUESTIONS)

        # Coin-flip over {yes, no} to keep the binary verdict balanced.
        if not _coin_flip_keep(
            verdict, ['yes', 'no'],
            'ood_cross_trend_convergence',
        ):
            return None

        answer_text = random.choice(HO2_ANSWERS[(direction, verdict)])
        return _make_rl_result(
            question,
            verdict, 'ood_cross_trend_convergence', answer_text, self.seq_len,
        )

    # --- HO3: Extrema alignment ---

    def generate_extrema_alignment(self) -> Optional[Dict[str, Any]]:
        # Pick two random metrics, then use _pick_balanced_param on the
        # threshold t ∈ {5,10,15,20} to flip verdicts 50/50 by construction.
        # If the computed distance is outside the tunable band (always ≤5
        # or always >20), skip the sample.
        if self.n < 2:
            return None

        max_positions = []
        for i in range(self.n):
            pos = self._stats(i).get('max_pos')
            if pos is not None:
                max_positions.append((i, int(pos)))
        if len(max_positions) < 2:
            return None

        (ai, pa), (bi, pb) = random.sample(max_positions, 2)
        dist = abs(pa - pb)

        def verdict_for(t):
            return "yes" if dist <= t else "no"

        picked = _pick_balanced_param(verdict_for, [5, 10, 15, 20])
        if picked is None:
            return None
        t, verdict = picked
        a, b = self.metrics[ai], self.metrics[bi]

        if verdict == "yes":
            answer_text = random.choice(HO3_ANSWERS['yes']).format(
                a=a, b=b, pa=pa, pb=pb, dist=dist, t=t,
            )
        else:
            answer_text = random.choice(HO3_ANSWERS['no']).format(
                a=a, b=b, pa=pa, pb=pb, dist=dist, t=t,
            )

        return _make_rl_result(
            random.choice(HO3_QUESTIONS).format(t=t, a=a, b=b),
            verdict, 'ood_cross_extrema_alignment', answer_text, self.seq_len,
        )

    # --- HO5: Range overlap ---

    def generate_range_overlap(self) -> Optional[Dict[str, Any]]:
        if self.n < 2:
            return None
        i, j = random.sample(range(self.n), 2)
        a, b = self.metrics[i], self.metrics[j]

        min_a = self._stats(i).get('min')
        max_a = self._stats(i).get('max')
        min_b = self._stats(j).get('min')
        max_b = self._stats(j).get('max')
        if any(v is None for v in [min_a, max_a, min_b, max_b]):
            return None

        min_a, max_a = round(float(min_a), 2), round(float(max_a), 2)
        min_b, max_b = round(float(min_b), 2), round(float(max_b), 2)

        # Ranges overlap iff max(min_a, min_b) <= min(max_a, max_b)
        overlap = max(min_a, min_b) <= min(max_a, max_b)
        verdict = "yes" if overlap else "no"

        answer_text = random.choice(HO5_ANSWERS[verdict]).format(
            a=a, b=b,
            min_a=_fmt(min_a), max_a=_fmt(max_a),
            min_b=_fmt(min_b), max_b=_fmt(max_b),
        )
        return _make_rl_result(
            random.choice(HO5_QUESTIONS).format(a=a, b=b),
            verdict, 'ood_cross_range_overlap', answer_text, self.seq_len,
        )

    # --- HO6: Concordant half-shift count ---

    def generate_concordant_shift(self) -> Optional[Dict[str, Any]]:
        if self.n < 2:
            return None
        count = 0
        valid = 0
        for i in range(self.n):
            fh = self._stats(i).get('first_half_mean')
            sh = self._stats(i).get('second_half_mean')
            if fh is not None and sh is not None:
                valid += 1
                if float(sh) > float(fh):
                    count += 1

        if valid < 2:
            return None

        verdict = str(count)
        answer_text = random.choice(HO6_ANSWERS).format(count=count, total=valid)
        return _make_rl_result(
            random.choice(HO6_QUESTIONS),
            verdict, 'ood_cross_concordant_shift', answer_text, self.seq_len,
        )

    # --- HO7: Cluster singleton ---

    def generate_cluster_singleton(self) -> Optional[Dict[str, Any]]:
        """Is any metric in its own cluster (unique trend type)?"""
        if self.n < 3:
            return None
        trends = [self._trend_type(i) for i in range(self.n)]
        if 'unknown' in trends:
            return None
        from collections import Counter
        counts = Counter(trends)
        singletons = [self.metrics[i] for i in range(self.n)
                       if counts[trends[i]] == 1]
        verdict = "yes" if singletons else "no"
        # Natural rate is skewed (one direction dominant depending on trend
        # diversity); coin-flip over {yes, no} forces 50/50 emission.
        if not _coin_flip_keep(verdict, ['yes', 'no'], 'ood_cluster_singleton'):
            return None
        if singletons:
            key = 'yes_singular' if len(singletons) == 1 else 'yes_plural'
            answer_text = random.choice(HO7_ANSWERS[key]).format(
                metrics=", ".join(singletons), count=len(singletons),
            )
        else:
            answer_text = random.choice(HO7_ANSWERS['no'])
        return _make_rl_result(
            random.choice(HO7_QUESTIONS),
            verdict, 'ood_cluster_singleton', answer_text, self.seq_len,
        )

    # --- HO8: Correlation transitivity ---

    def generate_corr_transitivity(self) -> Optional[Dict[str, Any]]:
        """If A and B share a trend, and B and C share a trend, does A share with C?"""
        if self.n < 3:
            return None
        trends = [self._trend_type(i) for i in range(self.n)]
        if 'unknown' in trends:
            return None

        # Pick 3 distinct metrics
        indices = random.sample(range(self.n), 3)
        a_i, b_i, c_i = indices
        a, b, c = self.metrics[a_i], self.metrics[b_i], self.metrics[c_i]
        t_a, t_b, t_c = trends[a_i], trends[b_i], trends[c_i]

        ab_same = t_a == t_b
        bc_same = t_b == t_c
        ac_same = t_a == t_c

        # Only interesting if AB and BC match (transitivity test)
        if not (ab_same and bc_same):
            # Try to find a triplet where AB and BC match
            found = False
            for combo in combinations(range(self.n), 3):
                i, j, k = combo
                if trends[i] == trends[j] and trends[j] == trends[k]:
                    a_i, b_i, c_i = i, j, k
                    a, b, c = self.metrics[i], self.metrics[j], self.metrics[k]
                    ab_same = bc_same = ac_same = True
                    found = True
                    break
                if trends[i] == trends[j] and trends[j] != trends[k]:
                    a_i, b_i, c_i = i, j, k
                    a, b, c = self.metrics[i], self.metrics[j], self.metrics[k]
                    ab_same = True; bc_same = False; ac_same = trends[i] == trends[k]
                    found = True
                    break
            if not found:
                return None

        # Verdict: does transitivity hold? (A~B and B~C → A~C?)
        if ab_same and bc_same:
            verdict = "yes"  # trivially true by transitivity of equality
        else:
            verdict = "yes" if ac_same else "no"

        answer_text = random.choice(HO8_ANSWERS[verdict]).format(
            a=a, b=b, c=c,
            t_a=trends[a_i], t_b=trends[b_i], t_c=trends[c_i],
        )
        return _make_rl_result(
            random.choice(HO8_QUESTIONS).format(a=a, b=b, c=c),
            verdict, 'ood_corr_transitivity', answer_text, self.seq_len,
        )

    # --- HO9: Mixed correlation/anti-correlation count ---

    def generate_mixed_corr_anti(self) -> Optional[Dict[str, Any]]:
        """How many metric pairs have opposite trends (anti-correlated)?"""
        if self.n < 3:
            return None
        trends = [self._trend_type(i) for i in range(self.n)]
        if 'unknown' in trends:
            return None

        # Define opposites
        opposites = {
            ('increase', 'decrease'), ('decrease', 'increase'),
        }
        count = 0
        total = 0
        for i, j in combinations(range(self.n), 2):
            total += 1
            if (trends[i], trends[j]) in opposites:
                count += 1

        verdict = str(count)
        templates = HO9_ANSWERS_SINGULAR if count == 1 else HO9_ANSWERS_PLURAL
        answer_text = random.choice(templates).format(count=count, total=total)
        return _make_rl_result(
            random.choice(HO9_QUESTIONS),
            verdict, 'ood_mixed_corr_anti', answer_text, self.seq_len,
        )

    # --- Generate all ---

    def generate_all(self) -> List[Dict[str, Any]]:
        results = []
        generators = [
            self.generate_corr_count,
            self.generate_trend_convergence,
            self.generate_extrema_alignment,
            self.generate_range_overlap,
            self.generate_concordant_shift,
            self.generate_cluster_singleton,
            self.generate_corr_transitivity,
            self.generate_mixed_corr_anti,
        ]
        for gen_fn in generators:
            try:
                r = gen_fn()
                if r is not None:
                    results.append(r)
            except Exception:
                logger.warning(
                    f"CrossMetric OOD {gen_fn.__name__} failed", exc_info=True,
                )
        return results
