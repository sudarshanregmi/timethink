import threading

import numpy as np
from scipy.optimize import linear_sum_assignment
from typing import Dict, List

from reward.config import RewardConfig, get_type_similarity, score_numeric_proximity
from reward.parsing import (
    extract_metric_blocks,
    extract_verification_per_metric,
)


def _score_key_points(pred_kps: List[Dict], gt_kps: List[Dict], seq_len: int) -> float:
    """
    Match key_points by label, then score (index proximity) + (value proximity if both have it).
    Missing labels in pred are penalised as 0.
    """
    if not gt_kps:
        return 1.0
    if not pred_kps:
        return 0.0

    pred_by_label = {kp['label']: kp for kp in pred_kps}
    scores = []
    for gt_kp in gt_kps:
        label = gt_kp['label']
        pred_kp = pred_by_label.get(label)
        if pred_kp is None or 'index' not in pred_kp:
            scores.append(0.0)
            continue

        idx_s = score_numeric_proximity(
            pred_kp['index'], gt_kp['index'], seq_len, RewardConfig.K_POSITION
        )
        if 'value' in gt_kp and 'value' in pred_kp:
            val_scale = max(abs(gt_kp['value']), 1.0)
            val_s = score_numeric_proximity(
                pred_kp['value'], gt_kp['value'], val_scale, RewardConfig.K_VALUE
            )
            scores.append(0.5 * idx_s + 0.5 * val_s)
        else:
            scores.append(idx_s)

    return sum(scores) / len(scores)


def _score_params(pred_params: Dict, gt_params: Dict) -> float:
    """
    Score structural params dict against GT. Only GT-present keys contribute.
    - List-of-floats: element-wise proximity (length mismatch → 0)
    - Float: numeric proximity
    - String: case-insensitive exact match
    """
    if not gt_params:
        return 1.0
    if not pred_params:
        return 0.0
    scores = []
    for key, gt_val in gt_params.items():
        pred_val = pred_params.get(key)
        if pred_val is None:
            scores.append(0.0)
            continue
        if isinstance(gt_val, list):
            if not isinstance(pred_val, list) or len(pred_val) != len(gt_val):
                scores.append(0.0)
            else:
                elem_scores = [
                    score_numeric_proximity(pv, gv, max(abs(gv), 1e-6), RewardConfig.K_AMPLITUDE)
                    for pv, gv in zip(pred_val, gt_val)
                ]
                scores.append(sum(elem_scores) / len(elem_scores) if elem_scores else 1.0)
        elif isinstance(gt_val, float):
            pv = pred_val if isinstance(pred_val, (int, float)) else 0.0
            scores.append(score_numeric_proximity(float(pv), gt_val, max(abs(gt_val), 1e-6), RewardConfig.K_AMPLITUDE))
        else:
            scores.append(1.0 if str(pred_val).lower().strip() == str(gt_val).lower().strip() else 0.0)
    return sum(scores) / len(scores) if scores else 1.0


def _score_event_pair(pred_ev: Dict, gt_ev: Dict, seq_len: int) -> float:
    """
    Score a single matched (pred, gt) local event pair.
    Uses only the fields present in GT — weights are computed adaptively
    so the score is always in [0, 1] and self-consistent.
    """
    components: Dict[str, float] = {}

    # type — always attempted (defaults to '' when absent = sparse format)
    components['type'] = get_type_similarity(
        pred_ev.get('type', ''), gt_ev.get('type', '')
    )

    if 'position_start' in gt_ev:
        if 'position_start' in pred_ev:
            components['pos_start'] = score_numeric_proximity(
                pred_ev['position_start'], gt_ev['position_start'],
                seq_len, RewardConfig.K_POSITION
            )
        else:
            components['pos_start'] = 0.0

    if 'position_end' in gt_ev:
        if 'position_end' in pred_ev:
            components['pos_end'] = score_numeric_proximity(
                pred_ev['position_end'], gt_ev['position_end'],
                seq_len, RewardConfig.K_POSITION
            )
        else:
            components['pos_end'] = 0.0

    if 'amplitude' in gt_ev:
        if 'amplitude' in pred_ev:
            components['amplitude'] = score_numeric_proximity(
                pred_ev['amplitude'], gt_ev['amplitude'],
                max(abs(gt_ev['amplitude']), 1e-6), RewardConfig.K_AMPLITUDE
            )
        else:
            components['amplitude'] = 0.0

    if gt_ev.get('key_points'):
        components['key_points'] = _score_key_points(
            pred_ev.get('key_points', []), gt_ev['key_points'], seq_len
        )

    if gt_ev.get('params'):
        components['params'] = _score_params(
            pred_ev.get('params', {}), gt_ev['params']
        )

    # Position/amplitude accuracy is only meaningful if the type is at least partially right.
    # When type is completely wrong, non-type components contribute at half value.
    if 'type' in components and len(components) > 1:
        type_score = components['type']
        scale = 0.5 + 0.5 * type_score   # [0.5, 1.0]
        for key in components:
            if key != 'type':
                components[key] *= scale
    return sum(components.values()) / len(components) if components else 1.0


def _score_local_events(pred_events: List[Dict], gt_events: List[Dict], seq_len: int) -> float:
    """
    Hungarian-matched scoring across all local events.

    - GT empty, pred empty  → 1.0
    - GT empty, pred has N  → hallucination penalty  (0.15 per extra event)
    - GT has events, pred empty → 0.0
    - Otherwise: F1-style with unmatched-pred penalty
    """
    n_gt, n_pred = len(gt_events), len(pred_events)

    if n_gt == 0 and n_pred == 0:
        return 1.0
    if n_gt == 0:
        # Pred hallucinated events that don't exist in GT
        return max(0.0, 1.0 - RewardConfig.LOCAL_HALLUCINATION_PER_EVENT * n_pred)
    if n_pred == 0:
        return 0.0

    cost = np.zeros((n_gt, n_pred))
    for i, gt in enumerate(gt_events):
        for j, pred in enumerate(pred_events):
            cost[i, j] = 1.0 - _score_event_pair(pred, gt, seq_len)

    row_ind, col_ind = linear_sum_assignment(cost)
    matched = sum(1.0 - cost[r, c] for r, c in zip(row_ind, col_ind))
    n_extra = n_pred - len(col_ind)
    denominator = n_gt + n_extra
    content_score = matched / denominator if denominator > 0 else 0.0
    detection_floor = RewardConfig.LOCAL_DETECTION_FLOOR
    count_bonus = RewardConfig.LOCAL_COUNT_BONUS if n_pred == n_gt else 0.0
    return max(content_score, detection_floor + count_bonus)


def _score_trend_segments(pred_segs: List[Dict], gt_segs: List[Dict], seq_len: int) -> float:
    """Segment-by-segment: count must match, then type + boundary proximity."""
    if not gt_segs and not pred_segs:
        return 1.0
    if not gt_segs or not pred_segs:
        return 0.0
    count_penalty = max(0.5, 1.0 - 0.2 * abs(len(pred_segs) - len(gt_segs)))
    n_common = min(len(pred_segs), len(gt_segs))

    scores = []
    for ps, gs in zip(pred_segs[:n_common], gt_segs[:n_common]):
        t_s = get_type_similarity(ps.get('type', ''), gs['type'])
        start_s = score_numeric_proximity(
            ps.get('start', 0), gs['start'], seq_len, RewardConfig.K_POSITION
        )
        end_s = score_numeric_proximity(
            ps.get('end', 0), gs['end'], seq_len, RewardConfig.K_POSITION
        )
        boundary_scale = 0.5 + 0.5 * t_s
        scores.append(0.5 * t_s + 0.25 * (start_s * boundary_scale) + 0.25 * (end_s * boundary_scale))

    content_score = sum(scores) / len(gt_segs) if gt_segs else 1.0  # missing segs → 0 contrib
    raw = content_score * count_penalty
    detection_floor = RewardConfig.SEGMENT_DETECTION_FLOOR
    count_bonus = RewardConfig.SEGMENT_COUNT_BONUS if len(pred_segs) == len(gt_segs) else 0.0
    return max(raw, detection_floor + count_bonus)


def _score_single_metric(pred_attrs: Dict, gt_attrs: Dict, seq_len: int) -> float:
    """
    Score one metric's full attribute profile.
    Only GT-present fields are included in the average, so
    the score degrades gracefully for richer vs sparser think blocks.
    """
    components: Dict[str, float] = {}

    if gt_attrs.get('seasonal'):
        gs = gt_attrs['seasonal']
        ps = pred_attrs.get('seasonal') or {}
        t_s   = get_type_similarity(ps.get('type', ''), gs['type'])
        per_s = score_numeric_proximity(
            ps.get('period', 0), gs['period'],
            max(gs['period'], 1.0), RewardConfig.K_PERIOD
        ) if gs['period'] > 0 else 1.0
        amp_s = score_numeric_proximity(
            ps.get('amplitude', 0), gs['amplitude'],
            max(abs(gs['amplitude']), 1.0), RewardConfig.K_AMPLITUDE
        ) if gs['amplitude'] != 0 else 1.0
        components['seasonal'] = (t_s + per_s + amp_s) / 3.0

    if gt_attrs.get('trend'):
        gt_t = gt_attrs['trend']
        pt   = pred_attrs.get('trend') or {}
        scale = max(abs(gt_t['amplitude']), 1.0)
        t_s   = get_type_similarity(pt.get('type', ''), gt_t['type'])
        amp_s = score_numeric_proximity(
            pt.get('amplitude', 0), gt_t['amplitude'], scale, RewardConfig.K_AMPLITUDE
        )
        # start and end are meaningful values — score them against amplitude scale
        start_s = score_numeric_proximity(
            pt.get('start', 0), gt_t['start'], scale, RewardConfig.K_AMPLITUDE
        )
        end_s = score_numeric_proximity(
            pt.get('end', 0), gt_t['end'], scale, RewardConfig.K_AMPLITUDE
        )
        components['trend'] = (t_s + amp_s + start_s + end_s) / 4.0

    if gt_attrs.get('trend_segments') is not None:
        components['trend_segments'] = _score_trend_segments(
            pred_attrs.get('trend_segments') or [], gt_attrs['trend_segments'], seq_len
        )

    if gt_attrs.get('noise'):
        gn = gt_attrs['noise']
        pn = pred_attrs.get('noise') or {}
        t_s   = get_type_similarity(pn.get('type', ''), gn['type'])
        strength_s = score_numeric_proximity(
            pn.get('strength', 0), gn['strength'],
            max(abs(gn['strength']), 1e-6), RewardConfig.K_AMPLITUDE
        )
        components['noise'] = (t_s + strength_s) / 2.0

    # Score whenever GT explicitly carries a local list (even if empty),
    # so hallucinated events in pred are always penalised.
    if gt_attrs.get('local') is not None:
        components['local'] = _score_local_events(
            pred_attrs.get('local') or [], gt_attrs['local'], seq_len
        )

    if gt_attrs.get('length'):
        pred_len = pred_attrs.get('length')
        components['length'] = (
            score_numeric_proximity(
                pred_len, gt_attrs['length'], gt_attrs['length'], RewardConfig.K_LENGTH
            ) if pred_len is not None else 0.0
        )

    if gt_attrs.get('stats'):
        gs = gt_attrs['stats']
        ps = pred_attrs.get('stats') or {}
        # Normalize aliases: model may use mean_first_half ↔ first_half_mean etc.
        _STAT_ALIASES = {
            'mean_first_half': 'first_half_mean',
            'mean_second_half': 'second_half_mean',
            'std_first_half': 'first_half_std',
            'std_second_half': 'second_half_std',
            'first_mean': 'first_half_mean',
            'second_mean': 'second_half_mean',
            'first_std': 'first_half_std',
            'second_std': 'second_half_std',
        }
        ps_norm = {_STAT_ALIASES.get(k, k): v for k, v in ps.items()}
        stat_scores = []
        for key in ('min', 'max', 'mean', 'std',
                     'first_half_mean', 'second_half_mean',
                     'first_half_std', 'second_half_std'):
            if key in gs:
                scale = max(abs(gs[key]), 1.0)
                pred_val = ps_norm.get(key)
                if pred_val is not None and isinstance(pred_val, (int, float)):
                    stat_scores.append(
                        score_numeric_proximity(float(pred_val), gs[key], scale, RewardConfig.K_VALUE)
                    )
                else:
                    stat_scores.append(0.0)
        if stat_scores:
            components['stats'] = sum(stat_scores) / len(stat_scores)

    base = sum(components.values()) / len(components) if components else 1.0

    # Penalize unnecessary local events when GT has no local field at all
    if gt_attrs.get('local') is None:
        pred_local = pred_attrs.get('local')
        if pred_local:
            n_unnecessary = len(pred_local)
            base *= max(0.85, 1.0 - RewardConfig.LOCAL_UNNECESSARY_PER_EVENT * n_unnecessary)

    return base * _consistency_multiplier(pred_attrs)


def _consistency_multiplier(pred_attrs: Dict) -> float:
    """
    Check internal consistency of predicted attributes.
    Each violation costs 0.1; result clamped to [0.7, 1.0].
    Only checks fields present in pred — missing fields already penalized elsewhere.
    """
    violations = 0

    # Trend: type must be consistent with start/end direction.
    # The amplitude field may be signed (end-start) or unsigned (magnitude),
    # so we check consistency against start/end values, not amplitude sign.
    trend = pred_attrs.get('trend')
    if trend and isinstance(trend, dict):
        t_type = str(trend.get('type', '')).lower()
        start = trend.get('start')
        end = trend.get('end')
        if start is not None and end is not None:
            diff = end - start
            ref = max(abs(start), abs(end), 1.0)
            sign_eps = 0.01 * ref
            steady_eps = 0.10 * ref
            if 'increase' in t_type and diff < -sign_eps:
                violations += 1
            elif 'decrease' in t_type and diff > sign_eps:
                violations += 1
            elif 'steady' in t_type and abs(diff) > steady_eps:
                violations += 1

    # Seasonal: "no periodic" must have period=0, amp≈0
    seasonal = pred_attrs.get('seasonal')
    if seasonal and isinstance(seasonal, dict):
        s_type = str(seasonal.get('type', '')).lower()
        if 'no periodic' in s_type:
            if seasonal.get('period', 0) > 0:
                violations += 1
            if seasonal.get('amplitude', 0) > 0.01:
                violations += 1

    # Noise: smooth↔strength=0, noisy↔strength>0
    noise = pred_attrs.get('noise')
    if noise and isinstance(noise, dict):
        n_type = str(noise.get('type', '')).lower()
        strength = noise.get('strength', 0)
        if n_type == 'smooth' and strength > 0.005:
            violations += 1
        elif n_type == 'noisy' and strength < 0.005:
            violations += 1

    # Stats: min ≤ mean ≤ max, std ≥ 0
    stats = pred_attrs.get('stats')
    if stats and isinstance(stats, dict):
        s_min, s_max = stats.get('min'), stats.get('max')
        s_mean, s_std = stats.get('mean'), stats.get('std')
        if s_min is not None and s_max is not None and s_min > s_max:
            violations += 1
        if s_mean is not None:
            if s_min is not None and s_mean < s_min - 1e-6:
                violations += 1
            if s_max is not None and s_mean > s_max + 1e-6:
                violations += 1
        if s_std is not None and s_std < -1e-6:
            violations += 1

    return max(0.7, 1.0 - 0.1 * violations)


# Thread-safe context set by compute_score() before calling any scorer.
# Avoids threading seq_len through 20+ scorer function signatures.
_thread_local = threading.local()


def set_context_seq_len(seq_len: int) -> None:
    """Set the seq_len context for evidence scoring. Called from compute_score()."""
    _thread_local.seq_len = seq_len


def score_evidence_quality(pred_think: str, gt_think: str) -> float:
    """
    Compares the attribute data in pred's thinking against GT's thinking.
    Handles single-metric and multi-metric (compound) think blocks.
    Applies a small penalty for each metric hallucinated in pred beyond GT.
    """
    gt_metrics   = extract_metric_blocks(gt_think)
    pred_metrics = extract_metric_blocks(pred_think)

    if not gt_metrics:
        return 1.0

    # seq_len must come from eval_metadata (set by compute_score via set_context_seq_len)
    seq_len = getattr(_thread_local, 'seq_len', 0)
    if not seq_len:
        raise ValueError(
            "seq_len not set in context. eval_metadata must contain 'length'."
        )

    # Per-GT-metric scores
    scores = [
        _score_single_metric(pred_metrics.get(name, {}), gt_attrs, seq_len)
        for name, gt_attrs in gt_metrics.items()
    ]
    base = sum(scores) / len(scores)

    # Penalty for metrics pred invented that are not in GT
    n_extra = max(0, len(pred_metrics) - len(gt_metrics))
    penalty  = n_extra * RewardConfig.EXTRA_METRIC_PENALTY

    return max(0.0, base - penalty)


def score_verification_intermediate(pred_think: str, gt_think: str) -> float:
    """
    Scores per-metric vs-line PASS/FAIL agreement in the verification section.

    For each 'vs {metric}: ...' line in GT:
      - correct verdict in pred  → 1.0
      - wrong verdict or absent  → 0.0

    Returns 1.0 when GT has no vs-lines (description QA, or pre-check failed).
    """
    gt_verdicts = extract_verification_per_metric(gt_think)
    if not gt_verdicts:
        return 1.0
    pred_verdicts = extract_verification_per_metric(pred_think)
    scores = [
        (1.0 if pred_verdicts.get(m) == gt_pass else 0.0)
        for m, gt_pass in gt_verdicts.items()
    ]
    return sum(scores) / len(scores)
