"""Verdict-only scorers used for all RL training types.

The reward signal is intentionally minimal:
  - <think>...</think> presence is enforced by extract_think_content (caller).
  - Each scorer reads the LAST `answer:` (or `verdict:`) line anywhere in
    the think block and returns a verdict score in [0, 1].
  - No evidence/structural rewards — the model is free to discover any
    reasoning chain that produces the right answer (good for OOD generalization).
"""
import re

from reward.config import RewardConfig, score_numeric_proximity
from reward.parsing import (
    extract_answer_line,
    extract_answer_set,
    _safe_float,
    parse_list_verdict,
)


def _extract_yes_no(think: str):
    raw = extract_answer_line(think)
    if raw is None:
        return None
    m = re.match(r'(yes|no)', raw, re.IGNORECASE)
    return m.group(1).lower() if m else None


def score_verdict_numeric(pred_think: str, gt_think: str) -> float:
    """Numeric verdict: smooth proximity decay, 0 if unparseable or missing."""
    gt_raw = extract_answer_line(gt_think)
    pred_raw = extract_answer_line(pred_think)
    if gt_raw is None:
        return 1.0
    if pred_raw is None:
        return 0.0
    gt_f = _safe_float(gt_raw.strip())
    pred_f = _safe_float(pred_raw.strip())
    if gt_f is None or pred_f is None:
        return 0.0
    scale = max(abs(gt_f), 1.0)
    return score_numeric_proximity(pred_f, gt_f, scale, RewardConfig.K_VALUE)


def score_verdict_binary(pred_think: str, gt_think: str) -> float:
    """Binary yes/no verdict: 1.0 match, VERDICT_WRONG_FLOOR mismatch, 0 missing."""
    gt_v = _extract_yes_no(gt_think)
    pred_v = _extract_yes_no(pred_think)
    if gt_v is None:
        return 1.0
    if pred_v is None:
        return 0.0
    if pred_v == gt_v:
        return 1.0
    return RewardConfig.VERDICT_WRONG_FLOOR


def score_verdict_categorical(pred_think: str, gt_think: str) -> float:
    """Categorical exact-match verdict: 1.0 match, VERDICT_WRONG_FLOOR mismatch, 0 missing."""
    gt_raw = extract_answer_line(gt_think)
    pred_raw = extract_answer_line(pred_think)
    if gt_raw is None:
        return 1.0
    if pred_raw is None:
        return 0.0
    gt_v = re.sub(r'</?\w+>', '', gt_raw).strip().lower()
    pred_v = re.sub(r'</?\w+>', '', pred_raw).strip().lower()
    # Normalize "steady" → "keep steady" for vocabulary compatibility
    if pred_v == 'steady':
        pred_v = 'keep steady'
    if gt_v == pred_v:
        return 1.0
    return RewardConfig.VERDICT_WRONG_FLOOR


def score_verdict_polymorphic(pred_think: str, gt_think: str) -> float:
    """Polymorphic verdict: inspects GT answer FORMAT and routes.

    - List `[a, b, c]`      → element-wise numeric proximity (length-penalized)
    - Yes/no / categorical  → exact match (1.0 or VERDICT_WRONG_FLOOR)
    - Numeric (int/float)   → numeric proximity decay
    - Unparseable GT        → 1.0 (can't penalize what we can't parse)

    Used for heterogeneous-verdict eval_types like stat_numerical where
    the answer may be any of the above depending on the sub_type.
    """
    gt_raw = extract_answer_line(gt_think)
    pred_raw = extract_answer_line(pred_think)
    if gt_raw is None:
        return 1.0
    if pred_raw is None:
        return 0.0
    gt_v = gt_raw.strip()
    pv = pred_raw.strip()

    # List format first (must precede float parse).
    gt_list = parse_list_verdict(gt_v)
    if gt_list is not None:
        pred_list = parse_list_verdict(pv)
        if pred_list is None:
            return 0.0
        paired = zip(gt_list, pred_list[:len(gt_list)])
        scores = []
        for g, p in paired:
            scale = max(abs(g), 1.0)
            scores.append(score_numeric_proximity(p, g, scale, RewardConfig.K_VALUE))
        len_penalty = min(len(gt_list), len(pred_list)) / max(len(gt_list), len(pred_list))
        return (sum(scores) / len(gt_list)) * len_penalty if scores else 0.0

    # Numeric (int or float).
    gt_f = _safe_float(gt_v)
    pred_f = _safe_float(pv)
    if gt_f is not None:
        if pred_f is None:
            return 0.0
        scale = max(abs(gt_f), 1.0)
        return score_numeric_proximity(pred_f, gt_f, scale, RewardConfig.K_VALUE)

    # Categorical / binary — exact string match.
    if gt_v.lower() == pv.lower():
        return 1.0
    return RewardConfig.VERDICT_WRONG_FLOOR


def score_verdict_set(pred_think: str, gt_think: str) -> float:
    """Set verdict (clustering / anticlustering): answer is `(A, B, C)`.

    Score = set-F1 between pred and gt. Unparseable → 0.0; empty sets on
    both sides → 1.0. Matches clustering/anticlustering answer format
    parsed by `extract_answer_set`.
    """
    gt_set = extract_answer_set(gt_think)
    pred_set = extract_answer_set(pred_think)
    if gt_set is None:
        return 1.0
    if pred_set is None:
        return 0.0
    if not gt_set and not pred_set:
        return 1.0
    if not gt_set or not pred_set:
        return 0.0
    tp = len(gt_set & pred_set)
    if tp == 0:
        return 0.0
    precision = tp / len(pred_set)
    recall = tp / len(gt_set)
    return 2 * precision * recall / (precision + recall)


def score_verdict_enumeration(pred_think: str, gt_think: str) -> float:
    """Enumeration verdict: exact match, set-F1 for comma lists, numeric tolerance."""
    gt_raw = extract_answer_line(gt_think)
    pred_raw = extract_answer_line(pred_think)
    if gt_raw is None:
        return 1.0
    if pred_raw is None:
        return 0.0
    gt_v = gt_raw.strip().lower()
    pred_v = pred_raw.strip().lower()
    if gt_v == pred_v:
        return 1.0
    # Numeric tolerance: "3" == "3.0", "5.46" == "5.460"
    gt_f = _safe_float(gt_v)
    pred_f = _safe_float(pred_v)
    if gt_f is not None and pred_f is not None and abs(gt_f - pred_f) < 0.01:
        return 1.0
    # Set-based F1 for comma-separated lists (order-insensitive)
    if ',' in gt_v or ',' in pred_v:
        gt_set = {x.strip() for x in gt_v.split(',') if x.strip()}
        pred_set = {x.strip() for x in pred_v.split(',') if x.strip()}
        if not gt_set and not pred_set:
            return 1.0
        if not gt_set or not pred_set:
            return 0.0
        tp = len(gt_set & pred_set)
        if tp == 0:
            return 0.0
        precision = tp / len(pred_set)
        recall = tp / len(gt_set)
        return 2 * precision * recall / (precision + recall)
    return 0.0
