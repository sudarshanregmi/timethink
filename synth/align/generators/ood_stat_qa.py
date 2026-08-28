"""Cross-category OOD evaluation QA generators.

Novel compositions that combine atoms from different categories (A-D)
in ways RL never explicitly rewards. Eval-only — never seen during training.

Why each is OOD:
    - max_in_highest_mean_quarter: RL checks max position (R10), RL checks
      quarter means (O4), but never combines "which quarter has highest mean"
      with "is the max in that quarter"
    - max_before_min: RL checks same-half (R13), but never asks relative
      ordering (which comes first)
    - std_exceeds_half_range: RL knows std (B1) and range (R7) separately,
      but never compares them
"""

import random
from typing import Any, Dict, List, Optional

import numpy as np
from loguru import logger

from synth.align.generators.atomic_qa import _compute_chunks, _fmt
from synth.align.generators.compositional_qa import (
    _coin_flip_keep, _interval_mean, _make_rl_result, _pick_balanced_param,
)

# ============================================================
# Question templates
# ============================================================

MAX_IN_HIGHEST_MEAN_QUARTER_QUESTIONS = [
    "Is the global maximum of {metric} located in the quarter with the highest mean?",
    "For {metric}, does the global maximum occur in the quarter that has the largest average?",
    "Check whether the peak of {metric} falls within the quarter whose mean is the highest.",
    "Does the quarter containing the global maximum of {metric} also have the highest mean among all four quarters?",
    "For the {metric} timeseries, is the global max in the quarter with the greatest mean value?",
]

MAX_BEFORE_MIN_QUESTIONS = [
    "Does the global maximum of {metric} occur before the global minimum? That is, is argmax < argmin?",
    "For {metric}, does the maximum come before the minimum in the timeseries?",
    "Check whether the position of the maximum in {metric} is earlier (smaller index) than the position of the minimum.",
    "In the {metric} data, does the peak precede the trough?",
    "Is the index of the global max of {metric} less than the index of the global min?",
]

STD_EXCEEDS_FRACTION_RANGE_QUESTIONS = [
    "For {metric}, is the standard deviation greater than 1/{k} of the range? That is, is std > (max - min) / {k}?",
    "Check whether the standard deviation of {metric} exceeds one-{kth} of its range.",
    "Does std({metric}) > range({metric}) / {k}? Compare the standard deviation to 1/{k} of the max-minus-min spread.",
    "For the {metric} timeseries, is the standard deviation more than one-{kth} of the difference between the maximum and minimum?",
    "Is the {metric} data's std larger than (max - min) / {k}?",
]

# ============================================================
# Answer templates
# ============================================================

MAX_IN_HIGHEST_MEAN_QUARTER_ANSWERS = {
    'yes': [
        "Yes, the global maximum of {metric} is in the quarter with the highest mean.",
        "The peak of {metric} falls in the highest-mean quarter.",
        "Confirmed, the global max and highest quarter mean are co-located for {metric}.",
    ],
    'no': [
        "No, the global maximum of {metric} is not in the quarter with the highest mean.",
        "The peak of {metric} is not in the highest-mean quarter.",
        "The global maximum and highest-mean quarter are not co-located for {metric}.",
    ],
}

MAX_BEFORE_MIN_ANSWERS = {
    'yes': [
        "Yes, the maximum of {metric} occurs before the minimum.",
        "The peak precedes the trough in {metric}.",
        "Confirmed, the maximum comes first in {metric}.",
    ],
    'no': [
        "No, the maximum of {metric} does not occur before the minimum.",
        "The peak does not precede the trough in {metric}.",
        "The maximum does not come first in {metric}.",
    ],
}

STD_EXCEEDS_FRACTION_RANGE_ANSWERS = {
    'yes': [
        "Yes, the standard deviation of {metric} exceeds 1/{k} of the range.",
        "For {metric}, the std is greater than one-{kth} of the range.",
        "The std of {metric} exceeds (max - min)/{k}.",
    ],
    'no': [
        "No, the standard deviation of {metric} does not exceed 1/{k} of the range.",
        "For {metric}, the std is not greater than one-{kth} of the range.",
        "The std of {metric} does not exceed (max - min)/{k}.",
    ],
}


# ============================================================
# Generator
# ============================================================

class OODStatGenerator:
    """Cross-category OOD compositions — eval only.

    Each task combines atoms from different categories in novel ways.
    """

    MIN_SEQ_LEN = 64

    def __init__(self, timeseries: np.ndarray, metric: str, seq_len: int):
        self.ts = np.asarray(timeseries, dtype=float)
        self.metric = metric
        self.seq_len = seq_len

    def _quarter_boundaries(self):
        q = self.seq_len // 4
        return [
            (0, q - 1),
            (q, 2 * q - 1),
            (2 * q, 3 * q - 1),
            (3 * q, self.seq_len - 1),
        ]

    # --- O5: Max in highest-mean quarter ---

    def generate_max_in_highest_mean_quarter(self) -> Optional[Dict[str, Any]]:
        if self.seq_len < self.MIN_SEQ_LEN:
            return None

        quarters = self._quarter_boundaries()
        q_means = [_interval_mean(self.ts, s, e) for s, e in quarters]
        best_q = int(np.argmax(q_means))

        max_pos = int(np.argmax(self.ts))
        max_val = round(float(self.ts[max_pos]), 2)

        # Which quarter is the max in?
        max_q = None
        for i, (s, e) in enumerate(quarters):
            if s <= max_pos <= e:
                max_q = i
                break

        same = max_q == best_q
        verdict = "yes" if same else "no"
        if not _coin_flip_keep(
            verdict, ['yes', 'no'], 'ood_max_in_highest_mean_quarter',
        ):
            return None

        all_qm = "[" + ", ".join(_fmt(m) for m in q_means) + "]"
        qs, qe = quarters[best_q]

        if verdict == "yes":
            answer_text = random.choice(
                MAX_IN_HIGHEST_MEAN_QUARTER_ANSWERS['yes']
            ).format(
                metric=self.metric, max_val=_fmt(max_val), max_pos=max_pos,
                max_q=max_q, qs=qs, qe=qe, qm=_fmt(q_means[best_q]),
                all_qm=all_qm,
            )
        else:
            answer_text = random.choice(
                MAX_IN_HIGHEST_MEAN_QUARTER_ANSWERS['no']
            ).format(
                metric=self.metric, max_val=_fmt(max_val), max_pos=max_pos,
                max_q=max_q, max_q_mean=_fmt(q_means[max_q]),
                best_q=best_q, qm=_fmt(q_means[best_q]), all_qm=all_qm,
            )

        return _make_rl_result(
            random.choice(MAX_IN_HIGHEST_MEAN_QUARTER_QUESTIONS).format(
                metric=self.metric,
            ),
            verdict, 'ood_max_in_highest_mean_quarter', answer_text, self.seq_len,
        )

    # --- O8: Max before min ---

    def generate_max_before_min(self) -> Dict[str, Any]:
        max_pos = int(np.argmax(self.ts))
        min_pos = int(np.argmin(self.ts))
        max_val = round(float(self.ts[max_pos]), 2)
        min_val = round(float(self.ts[min_pos]), 2)

        verdict = "yes" if max_pos < min_pos else "no"

        answer_text = random.choice(MAX_BEFORE_MIN_ANSWERS[verdict]).format(
            metric=self.metric,
            max_val=_fmt(max_val), max_pos=max_pos,
            min_val=_fmt(min_val), min_pos=min_pos,
        )

        return _make_rl_result(
            random.choice(MAX_BEFORE_MIN_QUESTIONS).format(metric=self.metric),
            verdict, 'ood_max_before_min', answer_text, self.seq_len,
        )

    # --- O9: Std exceeds fraction of range ---

    def generate_std_exceeds_half_range(self) -> Optional[Dict[str, Any]]:
        # Fixed "half-range" always collapses to "no" (Popoviciu: std ≤ range/2
        # only reaches equality at binary distributions). Pick the divisor k
        # via _pick_balanced_param to guarantee 50/50 verdict distribution:
        # if both "yes" and "no" are reachable across k ∈ {2,3,4,5,6} we
        # flip a coin and choose a k matching it; if not, return None.
        std_val = round(float(np.std(self.ts)), 2)
        max_val = round(float(np.max(self.ts)), 2)
        min_val = round(float(np.min(self.ts)), 2)
        range_val = round(max_val - min_val, 2)
        if range_val <= 0:
            return None

        def verdict_for(k):
            return "yes" if std_val > round(range_val / k, 2) else "no"

        picked = _pick_balanced_param(verdict_for, [2, 3, 4, 5, 6])
        if picked is None:
            return None
        k, verdict = picked
        threshold = round(range_val / k, 2)
        kth = {2: 'half', 3: 'third', 4: 'fourth', 5: 'fifth', 6: 'sixth'}[k]

        answer_text = random.choice(STD_EXCEEDS_FRACTION_RANGE_ANSWERS[verdict]).format(
            metric=self.metric,
            std=_fmt(std_val), max_val=_fmt(max_val), min_val=_fmt(min_val),
            range_val=_fmt(range_val), threshold=_fmt(threshold),
            k=k, kth=kth,
        )

        return _make_rl_result(
            random.choice(STD_EXCEEDS_FRACTION_RANGE_QUESTIONS).format(
                metric=self.metric, k=k, kth=kth,
            ),
            verdict, 'ood_std_exceeds_half_range', answer_text, self.seq_len,
        )

    # --- Generate all ---

    def generate_all(self) -> List[Dict[str, Any]]:
        results = []
        generators = [
            self.generate_max_in_highest_mean_quarter,
            self.generate_max_before_min,
            self.generate_std_exceeds_half_range,
        ]
        for gen_fn in generators:
            try:
                r = gen_fn()
                if r is not None:
                    results.append(r)
            except Exception:
                logger.warning(f"OOD {gen_fn.__name__} failed", exc_info=True)
        return results
