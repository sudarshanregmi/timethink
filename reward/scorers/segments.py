import re
from typing import Optional

from reward.config import RewardConfig, score_verdict_simple
from reward.coherency import coherency_condition_verdict
from reward.evidence import score_evidence_quality
from reward.parsing import get_reasoning, _extract_outcome
from reward.scorers.core import _extract_yes_no_verdict_simple


def _score_judgment_conditions(pred_think: str, gt_think: str = None) -> float:
    """
    Score condition/event lines in compound judgment think block.

    Handles three formats:
      A) 'condition N:' + 'all conditions:' — multi-condition compound judgments
      B) 'event N:' + 'summary:' — stat_threshold amplitude checks
      C) Direct computation with '-> met/not met' or '-> pass/fail'

    Each item: present + outcome → 1.0, present no outcome → 0.5, absent → 0.0.
    """
    # Search reasoning section only (below ===) to avoid false matches in data blocks
    ref_full = gt_think if gt_think is not None else pred_think
    ref = get_reasoning(ref_full).lower()
    pred_lower = get_reasoning(pred_think).lower()

    # --- Format A: condition N: labels ---
    n_conds = len(re.findall(r'\bcondition \d+:', ref))
    if n_conds > 0:
        score = 0.0
        # all-conditions summary only expected with 2+ conditions (per design)
        total = n_conds + (1 if n_conds >= 2 else 0)

        for i in range(n_conds):
            label = f'condition {i+1}:'
            if label not in pred_lower:
                continue
            cond_start = pred_lower.find(label)
            next_label = f'condition {i+2}:' if i + 1 < n_conds else 'all conditions:'
            next_pos = pred_lower.find(next_label, cond_start)
            block = pred_lower[cond_start: next_pos if next_pos != -1 else len(pred_lower)]
            pred_outcome = _extract_outcome(block)
            if pred_outcome is None:
                score += 0.5   # present but no outcome
            elif gt_think is not None:
                gt_cond_start = ref.find(label)
                gt_next_pos = ref.find(next_label, gt_cond_start) if gt_cond_start != -1 else -1
                gt_block = ref[gt_cond_start: gt_next_pos if gt_next_pos != -1 else len(ref)] if gt_cond_start != -1 else ''
                gt_outcome = _extract_outcome(gt_block)
                score += 1.0 if (gt_outcome is None or pred_outcome == gt_outcome) else 0.25
            else:
                score += 1.0

        if n_conds >= 2 and 'all conditions:' in pred_lower:
            all_start = pred_lower.find('all conditions:')
            all_block = pred_lower[all_start: all_start + 80]
            score += 1.0 if (re.search(r'\ball met\b', all_block) or re.search(r'\bnot all met\b', all_block)) else 0.5

        return score / total

    # --- Format B: event N: labels with pass/fail + summary: ---
    n_events = len(re.findall(r'\bevent \d+:', ref))
    if n_events > 0:
        score = 0.0
        total = n_events + 1  # events + summary

        for i in range(n_events):
            label = f'event {i+1}:'
            if label not in pred_lower:
                continue
            cond_start = pred_lower.find(label)
            next_label = f'event {i+2}:' if i + 1 < n_events else 'summary:'
            next_pos = pred_lower.find(next_label, cond_start)
            block = pred_lower[cond_start: next_pos if next_pos != -1 else len(pred_lower)]
            pred_outcome = _extract_outcome(block)
            if pred_outcome is None:
                score += 0.5   # present but no outcome
            elif gt_think is not None:
                gt_cond_start = ref.find(label)
                gt_next_pos = ref.find(next_label, gt_cond_start) if gt_cond_start != -1 else -1
                gt_block = ref[gt_cond_start: gt_next_pos if gt_next_pos != -1 else len(ref)] if gt_cond_start != -1 else ''
                gt_outcome = _extract_outcome(gt_block)
                score += 1.0 if (gt_outcome is None or pred_outcome == gt_outcome) else 0.25
            else:
                score += 1.0

        if 'summary:' in pred_lower:
            sum_start = pred_lower.find('summary:')
            sum_block = pred_lower[sum_start: sum_start + 80]
            has_count = bool(re.search(r'\d+ of \d+ pass', sum_block))
            score += 1.0 if has_count else 0.5

        return score / total

    # --- Format C: direct computation with -> met/not met, -> pass/fail, or (pass)/(fail) ---
    has_outcome = (
        bool(re.search(r'→\s*(?:not )?met|->\s*(?:not )?met', pred_lower))
        or bool(re.search(r'->\s*\b(?:pass|fail)\b', pred_lower))
        or bool(re.search(r'\(pass\)|\(fail\)', pred_lower))
    )
    if not has_outcome:
        return 0.0
    # Check correctness against GT if available
    if gt_think is not None:
        pred_outcome = _extract_outcome(pred_lower)
        gt_outcome = _extract_outcome(ref)
        if pred_outcome is not None and gt_outcome is not None and pred_outcome != gt_outcome:
            return 0.25
    return 1.0


def score_compound_judgment(pred_think: str, gt_think: str) -> float:
    """
    Weights: 65% verdict (yes/no), 20% condition steps, 15% evidence.

    When the verdict is wrong, total score is capped at WRONG_VERDICT_CAP to
    prevent the model from exploiting majority-class guessing.
    """
    gt_v   = _extract_yes_no_verdict_simple(gt_think)
    pred_v = _extract_yes_no_verdict_simple(pred_think)
    verdict_score = score_verdict_simple(pred_v, gt_v)

    cond_score = _score_judgment_conditions(pred_think, gt_think)
    coh = coherency_condition_verdict(pred_think)

    raw = (0.60 * verdict_score
           + 0.05 * coh
           + 0.20 * cond_score
           + 0.15 * score_evidence_quality(pred_think, gt_think))
    if verdict_score < 1.0:
        raw = min(raw, RewardConfig.WRONG_VERDICT_CAP)
    return raw
