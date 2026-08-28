import re
from typing import Optional

from reward.config import RewardConfig, score_verdict_simple
from reward.parsing import _safe_float
from reward.coherency import (
    coherency_condition_verdict,
    coherency_duration_dominance,
    coherency_mts_verdict,
    coherency_mts_cluster,
    coherency_yes_no,
)
from reward.evidence import score_evidence_quality, score_verification_intermediate
from reward.parsing import (
    get_reasoning,
    extract_answer_line,
    extract_answer_scalar,
    extract_answer_set,
    extract_yes_no_result,
    extract_aggregate_durations,
    extract_dominant_winner,
)


def _score_verdict_match(pred_think: str, gt_think: str, extract_fn) -> float:
    """Helper for correlation and anticorrelation scoring.

    When the verdict is wrong, total score is capped at WRONG_VERDICT_CAP to
    prevent the model from exploiting majority-class guessing.
    """
    gt_verdict   = extract_fn(gt_think)
    pred_verdict = extract_fn(pred_think)
    verdict_score = score_verdict_simple(pred_verdict, gt_verdict)

    coh      = coherency_mts_verdict(pred_think)
    verif    = score_verification_intermediate(pred_think, gt_think)
    evidence = score_evidence_quality(pred_think, gt_think)
    raw = (0.55 * verdict_score
           + RewardConfig.W_COHERENCY * coh
           + RewardConfig.W_VERIFICATION * verif
           + RewardConfig.W_EVIDENCE * evidence)
    if verdict_score < 1.0:
        raw = min(raw, RewardConfig.WRONG_VERDICT_CAP)
    return raw


def _score_set_match(pred_think: str, gt_think: str, extract_fn) -> float:
    """Helper for clustering and anticlustering F1 set scoring."""
    gt_cluster   = extract_fn(gt_think) or set()
    pred_cluster = extract_fn(pred_think)

    if pred_cluster is None:
        cluster_score = RewardConfig.PARTIAL_NO_CLUSTER_LINE
    elif not gt_cluster and not pred_cluster:
        cluster_score = 1.0
    elif not gt_cluster:
        cluster_score = 0.0   # hallucinated cluster members
    elif not pred_cluster:
        cluster_score = 0.0   # missed all GT members
    else:
        inter = gt_cluster & pred_cluster
        p = len(inter) / len(pred_cluster)
        r = len(inter) / len(gt_cluster)
        cluster_score = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        # Penalize over-prediction (spray-and-pray): predicting everything
        # guarantees recall=1.0 and an F1 floor of ~0.40-0.50.
        n_extra = len(pred_cluster) - len(gt_cluster)
        if n_extra > 0:
            cluster_score *= max(0.5, 1.0 - 0.1 * n_extra)

    coh      = coherency_mts_cluster(pred_think)
    verif    = score_verification_intermediate(pred_think, gt_think)
    evidence = score_evidence_quality(pred_think, gt_think)
    return (0.55 * cluster_score
            + RewardConfig.W_COHERENCY * coh
            + RewardConfig.W_VERIFICATION * verif
            + RewardConfig.W_EVIDENCE * evidence)



def _extract_yes_no_verdict_simple(think: str) -> Optional[str]:
    """Extracts 'yes' or 'no' from 'answer: yes/no' line below === (or full block for data-only types)."""
    raw = extract_answer_line(think)
    if raw is None:
        return None
    m = re.match(r'(yes|no)', raw, re.IGNORECASE)
    return m.group(1).lower() if m else None


def _extract_trend_type_verdict(think: str) -> Optional[str]:
    """Extracts trend type verdict from 'answer: X' line below === (or full block for data-only types)."""
    raw = extract_answer_line(think)
    if raw is None:
        return None
    answer = re.sub(r'</?\w+>', '', raw).strip().lower()
    return answer if answer else None


def _score_condition_steps(pred_think: str, gt_think: str) -> float:
    """
    Score multi-condition judgment steps common to anti_judgment and segment_judgment.
    Checks for 'condition N:' lines and 'all conditions:' summary.
    Mirrors _score_judgment_conditions logic but simpler (presence-based).
    Only searches reasoning section (below ===) to avoid false matches in data blocks.
    """
    gt_reasoning = get_reasoning(gt_think)
    pred_reasoning = get_reasoning(pred_think)

    gt_conditions = re.findall(r'condition \d+:', gt_reasoning, re.IGNORECASE)
    if not gt_conditions:
        return 1.0
    pred_conditions = re.findall(r'condition \d+:', pred_reasoning, re.IGNORECASE)
    cond_score = min(len(pred_conditions), len(gt_conditions)) / len(gt_conditions)
    # Only expect 'all conditions:' when GT has 2+ conditions (per design)
    if len(gt_conditions) >= 2:
        has_summary = 1.0 if re.search(r'all conditions:', pred_reasoning, re.IGNORECASE) else 0.0
        return 0.7 * cond_score + 0.3 * has_summary
    return cond_score


def score_yes_no(pred_think: str, gt_think: str) -> float:
    """Rewards correct PASS/FAIL (target metric) + intermediate per-metric reasoning + attribute evidence.

    When the verdict is wrong, total score is capped at WRONG_VERDICT_CAP to
    prevent the model from exploiting majority-class guessing.
    """
    gt_result   = extract_yes_no_result(gt_think)
    pred_result = extract_yes_no_result(pred_think)
    pass_fail_score = score_verdict_simple(
        pred_result, gt_result,
        missing_floor=RewardConfig.PARTIAL_NO_PASS_FAIL,
    )

    coh      = coherency_yes_no(pred_think)
    verif    = score_verification_intermediate(pred_think, gt_think)
    evidence = score_evidence_quality(pred_think, gt_think)
    raw = (0.55 * pass_fail_score
           + RewardConfig.W_COHERENCY * coh
           + RewardConfig.W_VERIFICATION * verif
           + RewardConfig.W_EVIDENCE * evidence)
    if pass_fail_score < 1.0:
        raw = min(raw, RewardConfig.WRONG_VERDICT_CAP)
    return raw


def score_mts_verdict(pred_think: str, gt_think: str) -> float:
    """Unified scorer for correlation + anticorrelation (scalar answer match)."""
    return _score_verdict_match(pred_think, gt_think, extract_answer_scalar)


def score_mts_set(pred_think: str, gt_think: str) -> float:
    """Unified scorer for clustering + anticlustering (F1 set match)."""
    return _score_set_match(pred_think, gt_think, extract_answer_set)


def _extract_generic_answer(think: str) -> Optional[str]:
    """Extract answer value from 'answer: ...' line below === (or full block for data-only types)."""
    raw = extract_answer_line(think)
    return raw.lower() if raw else None


def _set_f1(gt_answer: str, pred_answer: str) -> float:
    """Set-based F1 for comma-separated list verdicts (unordered, whitespace-tolerant)."""
    gt_set = {x.strip() for x in gt_answer.split(',') if x.strip()}
    pred_set = {x.strip() for x in pred_answer.split(',') if x.strip()}
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


def score_enumeration(pred_think: str, gt_think: str) -> float:
    """Generic scorer for enumeration families — answer match + evidence.

    Comma-separated verdicts are treated as sets (order-insensitive,
    whitespace-tolerant, partial credit via F1). Scalar verdicts fall through
    to exact-string or numeric-tolerance match.
    """
    gt_answer = _extract_generic_answer(gt_think)
    pred_answer = _extract_generic_answer(pred_think)
    if gt_answer is None:
        verdict_score = 1.0
    elif pred_answer is None:
        verdict_score = RewardConfig.PARTIAL_NO_VERDICT
    elif pred_answer == gt_answer:
        verdict_score = 1.0
    else:
        # Numeric tolerance: "3" == "3.0", "5.46" == "5.460"
        gt_f = _safe_float(gt_answer)
        pred_f = _safe_float(pred_answer)
        if gt_f is not None and pred_f is not None and abs(gt_f - pred_f) < 0.01:
            verdict_score = 1.0
        elif ',' in gt_answer or ',' in pred_answer:
            # Set-based F1 for comma-separated list verdicts
            verdict_score = _set_f1(gt_answer, pred_answer)
        else:
            verdict_score = 0.0
    evidence = score_evidence_quality(pred_think, gt_think)
    return 0.65 * verdict_score + 0.35 * evidence


def score_description(pred_think: str, gt_think: str) -> float:
    """Pure attribute quality score (covers UTS yes/no and description QA)."""
    return score_evidence_quality(pred_think, gt_think)


def score_trend_dominance(pred_think: str, gt_think: str) -> float:
    """
    Score Family 8 trend-dominance QA.

    Weights:
      60%  verdict correctness (increase/decrease/steady/equal)
      15%  structural markers (∩ lines, aggregate durations, dominant trend comparison)
      25%  data evidence
    """
    gt_verdict  = _extract_trend_type_verdict(gt_think)
    pred_verdict = _extract_trend_type_verdict(pred_think)
    verdict_score = score_verdict_simple(pred_verdict, gt_verdict)

    reasoning = get_reasoning(pred_think)
    gt_reasoning = get_reasoning(gt_think)

    # Sub-1: ∩ intersection lines — presence + numeric content (format proxy)
    idx = reasoning.find('∩')
    if idx >= 0:
        after = reasoning[idx + 1: idx + 41]
        sub1 = 1.0 if re.search(r'\d', after) else 0.3
    else:
        sub1 = 0.0

    # Sub-2: aggregate durations — compare type→duration mappings against GT
    gt_durs = extract_aggregate_durations(gt_reasoning)
    pred_durs = extract_aggregate_durations(reasoning)
    if not gt_durs:
        sub2 = 1.0  # GT has no durations section — can't check
    elif not pred_durs:
        sub2 = 0.0  # GT has durations but pred doesn't
    else:
        matches = 0
        for t, gt_d in gt_durs.items():
            pred_d = pred_durs.get(t)
            if pred_d is not None and (gt_d == 0 or abs(pred_d - gt_d) / max(abs(gt_d), 1) <= 0.20):
                matches += 1
        sub2 = matches / len(gt_durs)

    # Sub-3: dominant trend comparison — check if winner type matches GT
    # When GT verdict is "equal" (tie), any winner order is acceptable.
    if gt_verdict == 'equal':
        sub3 = 1.0
    else:
        gt_winner = extract_dominant_winner(gt_reasoning)
        pred_winner = extract_dominant_winner(reasoning)
        if gt_winner is None:
            sub3 = 1.0  # GT has no comparison line
        elif pred_winner is None:
            sub3 = 0.0
        else:
            sub3 = 1.0 if pred_winner == gt_winner else 0.0

    step_score = (sub1 + sub2 + sub3) / 3

    coh = coherency_duration_dominance(pred_think)
    evidence = score_evidence_quality(pred_think, gt_think)
    return 0.55 * verdict_score + 0.05 * coh + 0.15 * step_score + 0.25 * evidence


def score_anti_judgment(pred_think: str, gt_think: str) -> float:
    """
    Score multi-metric anti-trend compound judgment.

    Weights:
      65%  verdict (yes/no)
      20%  condition steps
      15%  evidence

    When the verdict is wrong, total score is capped at 0.30 to prevent
    the model from learning that good structure with wrong answer is OK.
    """
    gt_verdict   = _extract_yes_no_verdict_simple(gt_think)
    pred_verdict = _extract_yes_no_verdict_simple(pred_think)
    verdict_score = score_verdict_simple(pred_verdict, gt_verdict)

    cond_score = _score_condition_steps(pred_think, gt_think)
    coh = coherency_condition_verdict(pred_think)
    evidence   = score_evidence_quality(pred_think, gt_think)
    raw = 0.60 * verdict_score + 0.05 * coh + 0.20 * cond_score + 0.15 * evidence
    # Cap total score when verdict is wrong — good structure with wrong
    # answer must not earn enough reward to survive selection.
    if verdict_score < 1.0:
        raw = min(raw, RewardConfig.WRONG_VERDICT_CAP)
    return raw


_MCQ_LETTER_RE = re.compile(r'\b([ABCDabcd])\b')


def _extract_mcq_letter(think: str) -> Optional[str]:
    """Extract uppercase A/B/C/D from the 'answer:' line of a think block."""
    raw = extract_answer_line(think)
    if raw is None:
        return None
    m = _MCQ_LETTER_RE.search(raw)
    return m.group(1).upper() if m else None


def score_mcq_letter(pred_think: str, gt_think: str) -> float:
    """Pure verdict-only MCQ scoring for SenTSR-synthetic post-RL training.

    Verifiable GRPO contract: extract answer letter (A-D) from the
    'answer:' line; exact match against ground-truth letter; no coherency
    or evidence weighting. Wrong/missing maps to the standard floors via
    score_verdict_simple.
    """
    return score_verdict_simple(_extract_mcq_letter(pred_think),
                                _extract_mcq_letter(gt_think))


# score_cross_trend_query removed — identical to score_trend_dominance.
# Legacy 'cross_trend_query' eval_type now routes to score_trend_dominance.
