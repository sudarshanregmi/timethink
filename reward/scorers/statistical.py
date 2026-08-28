import math
import re
from typing import List, Optional

from reward.config import RewardConfig, score_numeric_proximity
from reward.coherency import coherency_stats_categorical
from reward.evidence import score_evidence_quality
from reward.parsing import extract_answer_line


def _parse_float_verdict(v: str) -> Optional[float]:
    try:
        f = float(v)
        if not math.isfinite(f):
            return None
        return f
    except (ValueError, TypeError):
        return None


from reward.parsing import parse_list_verdict as _parse_list_verdict


# Categorical verdicts WITH coherency chain (stat_segment_compare: half-comparison pattern)
_CATEGORICAL_VERDICTS = frozenset({
    'first half', 'second half', 'equal',
})

# Simple categorical verdicts: exact match, NO coherency chain
_SIMPLE_CATEGORICAL_VERDICTS = frozenset({
    'more_volatile', 'less_volatile', 'stable',
    'upward', 'downward', 'flat',
})

_BINARY_VERDICTS = frozenset({'yes', 'no'})


def score_stat_numerical(pred_think: str, gt_think: str) -> float:
    """
    Score Family 9 statistical numerical QA.

    Weights: 80% verdict + 20% evidence.

    Scoring strategy is inferred from the GT verdict FORMAT:
      list [...]           → element-wise relative_accuracy average
      categorical string   → exact match (first half / second half / equal)
      integer (count)      → exact match with ±10% tolerance
      float (value)        → relative_accuracy
    """
    gt_raw   = extract_answer_line(gt_think)
    pred_raw = extract_answer_line(pred_think)

    if gt_raw is None:
        verdict_score = 1.0
    elif pred_raw is None:
        # No verdict found — rebalance weights so evidence differentiates quality.
        # Without this, all no-verdict outputs cluster at 0.80*0.20+0.20*ev ≈ 0.36
        evidence = score_evidence_quality(pred_think, gt_think)
        return 0.50 * RewardConfig.PARTIAL_NO_VERDICT + 0.50 * evidence
    else:
        gt_v = gt_raw.strip()
        pv   = pred_raw.strip()

        # List verdict: element-wise comparison (stat_windowed_mean)
        gt_list = _parse_list_verdict(gt_v)
        if gt_list is not None:
            pred_list = _parse_list_verdict(pv)
            if pred_list:
                paired = zip(gt_list, pred_list[:len(gt_list)])
                scores = []
                for g, p in paired:
                    scale = max(abs(g), 1.0)
                    scores.append(score_numeric_proximity(p, g, scale, RewardConfig.K_VALUE))
                len_penalty = min(len(gt_list), len(pred_list)) / max(len(gt_list), len(pred_list))
                verdict_score = (sum(scores) / len(gt_list)) * len_penalty
            else:
                verdict_score = 0.0

        # Binary verdict: exact match (yes/no sub-types)
        elif gt_v.lower() in _BINARY_VERDICTS:
            verdict_score = 1.0 if gt_v.lower() == pv.lower() else RewardConfig.VERDICT_WRONG_FLOOR

        # Simple categorical: exact match, no coherency chain
        elif gt_v.lower() in _SIMPLE_CATEGORICAL_VERDICTS:
            verdict_score = 1.0 if gt_v.lower() == pv.lower() else RewardConfig.VERDICT_WRONG_FLOOR

        # Categorical verdict with coherency chain (stat_segment_compare)
        elif gt_v.lower() in _CATEGORICAL_VERDICTS:
            verdict_score = 1.0 if gt_v.lower() == pv.lower() else RewardConfig.VERDICT_WRONG_FLOOR
            # Categorical has a coherency chain: half means → comparison → verdict
            coh = coherency_stats_categorical(pred_think)
            evidence = score_evidence_quality(pred_think, gt_think)
            return 0.75 * verdict_score + 0.05 * coh + 0.20 * evidence

        # Numeric verdict
        else:
            gt_f  = _parse_float_verdict(gt_v)
            pred_f = _parse_float_verdict(pv)
            if gt_f is not None and pred_f is not None:
                scale = max(abs(gt_f), 1.0)
                verdict_score = score_numeric_proximity(pred_f, gt_f, scale, RewardConfig.K_VALUE)
            else:
                verdict_score = 0.0

    return (0.80 * verdict_score
            + 0.20 * score_evidence_quality(pred_think, gt_think))
