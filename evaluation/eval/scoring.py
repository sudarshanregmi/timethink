import math
import re

def relative_accuracy(pred, gt):
    if gt is None or pred is None:
        return None
    if gt == 0:
        return 1.0 if pred == 0 else 0.0
    return max(0.0, min(1.0, 1.0 - abs(pred - gt) / abs(gt)))


def position_accuracy(pred, gt, total_len=None):
    if gt is None or pred is None:
        return None
    if total_len is None:
        raise ValueError("total_len must be provided for position accuracy")
    if total_len <= 0:
        raise ValueError("total_len must be positive")
    return max(0.0, min(1.0, 1.0 - abs(pred - gt) / total_len))


def detect_description_perspectives(question_text: str) -> list:
    q_lower = question_text.lower()
    perspectives = []
    if 'trend' in q_lower:
        perspectives.append('trend')
    if 'periodicity' in q_lower or 'seasonal' in q_lower:
        perspectives.append('seasonal')
    if 'noise' in q_lower:
        perspectives.append('noise')
    if 'local' in q_lower:
        perspectives.append('local')
    return perspectives


def _normalize_trend_type(trend_str: str) -> str:
    if not trend_str:
        return ''
    t = trend_str.lower().strip()
    if 'increas' in t:
        return 'increase'
    if 'decreas' in t:
        return 'decrease'
    if 'steady' in t or 'stable' in t or 'flat' in t:
        return 'steady'
    return t


def _score_trend_segments(gt_segments, pred_segments, total_len=None):
    if not gt_segments and not pred_segments:
        return 1.0, 1.0, 1.0, []
    if not gt_segments or not pred_segments:
        n_unmatched = max(len(gt_segments), len(pred_segments))
        return 0.0, 0.0, 0.0, [0.0] * (n_unmatched * 2)

    matched = 0
    segment_num_scores = []
    for gt_seg, pred_seg in zip(gt_segments, pred_segments):
        if _normalize_trend_type(gt_seg.segment_type) == _normalize_trend_type(pred_seg.segment_type):
            matched += 1
            segment_num_scores.append(position_accuracy(pred_seg.start_point, gt_seg.start_point, total_len))
            segment_num_scores.append(position_accuracy(pred_seg.end_point, gt_seg.end_point, total_len))
        else:
            segment_num_scores.extend([0.0, 0.0])

    n_extra = abs(len(gt_segments) - len(pred_segments))
    segment_num_scores.extend([0.0] * (n_extra * 2))

    precision = matched / len(pred_segments)
    recall = matched / len(gt_segments)
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1, segment_num_scores


def score_description_perspective(perspective: str, gt_extraction, pred_extraction, total_len=None) -> dict:
    scores = {'cat': None, 'num': [], 'num_by_type': {}}

    if perspective == 'trend':
        gt_trend = gt_extraction.trend if gt_extraction else None
        pred_trend = pred_extraction.trend if pred_extraction else None
        if gt_trend is None:
            return scores

        if pred_trend is not None:
            gt_type = _normalize_trend_type(gt_trend.trend_type) if gt_trend.trend_type else ''
            pred_type = _normalize_trend_type(pred_trend.trend_type) if pred_trend.trend_type else ''
            overall_cat = 1.0 if gt_type == pred_type else 0.0
            scores['overall_cat'] = overall_cat

            gt_segments = gt_trend.segments or []
            pred_segments = pred_trend.segments or []

            if gt_segments:
                seg_p, seg_r, seg_f1, seg_num = _score_trend_segments(gt_segments, pred_segments, total_len)
                scores['segment_precision'] = seg_p
                scores['segment_recall'] = seg_r
                scores['segment_f1'] = seg_f1
                scores['segment_num'] = seg_num
            else:
                scores['segment_precision'] = None
                scores['segment_recall'] = None
                scores['segment_f1'] = None
                scores['segment_num'] = []

            scores['cat'] = overall_cat

            if pred_trend.start_value is not None and gt_trend.start_value is not None:
                val = relative_accuracy(pred_trend.start_value, gt_trend.start_value)
                scores['num'].append(val)
                scores['num_by_type'].setdefault('position', []).append(val)
            if pred_trend.end_value is not None and gt_trend.end_value is not None:
                val = relative_accuracy(pred_trend.end_value, gt_trend.end_value)
                scores['num'].append(val)
                scores['num_by_type'].setdefault('position', []).append(val)
            if pred_trend.amplitude is not None and gt_trend.amplitude is not None:
                val = relative_accuracy(pred_trend.amplitude, gt_trend.amplitude)
                scores['num'].append(val)
                scores['num_by_type'].setdefault('amplitude', []).append(val)
        else:
            scores['cat'] = 0.0
            scores['overall_cat'] = 0.0
            scores['segment_precision'] = 0.0 if (gt_trend and gt_trend.segments) else None
            scores['segment_recall'] = 0.0 if (gt_trend and gt_trend.segments) else None
            scores['segment_f1'] = 0.0 if (gt_trend and gt_trend.segments) else None

    elif perspective == 'seasonal':
        gt_season = gt_extraction.seasonal if gt_extraction else None
        pred_season = pred_extraction.seasonal if pred_extraction else None
        if gt_season is None:
            return scores

        if pred_season is not None:
            scores['cat'] = 1.0 if pred_season.has_periodicity == gt_season.has_periodicity else 0.0

            if gt_season.has_periodicity:
                if gt_season.period is not None:
                    pred_period = pred_season.period if pred_season.has_periodicity else None
                    val = relative_accuracy(pred_period, gt_season.period) if pred_period is not None else 0.0
                    scores['num'].append(val)
                    scores['num_by_type'].setdefault('period', []).append(val)
                if gt_season.amplitude is not None:
                    pred_amp = pred_season.amplitude if pred_season.has_periodicity else None
                    val = relative_accuracy(pred_amp, gt_season.amplitude) if pred_amp is not None else 0.0
                    scores['num'].append(val)
                    scores['num_by_type'].setdefault('amplitude', []).append(val)
        else:
            scores['cat'] = 0.0
            if gt_season.has_periodicity:
                if gt_season.period is not None:
                    scores['num'].append(0.0)
                    scores['num_by_type'].setdefault('period', []).append(0.0)
                if gt_season.amplitude is not None:
                    scores['num'].append(0.0)
                    scores['num_by_type'].setdefault('amplitude', []).append(0.0)

    elif perspective == 'noise':
        gt_noise = gt_extraction.noise if gt_extraction else None
        pred_noise = pred_extraction.noise if pred_extraction else None
        if gt_noise is None:
            return scores

        if pred_noise is not None:
            gt_level = gt_noise.noise_level.lower() if gt_noise.noise_level else ''
            pred_level = pred_noise.noise_level.lower() if pred_noise.noise_level else ''

            def noise_bucket(level):
                if 'noisy' in level:
                    return 'noisy'
                return 'smooth'

            scores['cat'] = 1.0 if noise_bucket(gt_level) == noise_bucket(pred_level) else 0.0

            if pred_noise.noise_strength is not None and gt_noise.noise_strength is not None:
                val = relative_accuracy(pred_noise.noise_strength, gt_noise.noise_strength)
                scores['num'].append(val)
                scores['num_by_type'].setdefault('strength', []).append(val)
        else:
            scores['cat'] = 0.0

    elif perspective == 'local':
        gt_local = gt_extraction.local if gt_extraction else None
        pred_local = pred_extraction.local if pred_extraction else None
        if gt_local is None or not gt_local.events:
            return scores

        if pred_local is not None and pred_local.events:
            potential_matches = []

            gt_events = gt_local.events
            pred_events = pred_local.events

            for i, gt_evt in enumerate(gt_events):
                gt_type = gt_evt.event_type.lower() if gt_evt.event_type else ''

                for j, pred_evt in enumerate(pred_events):
                    pred_type = pred_evt.event_type.lower() if pred_evt.event_type else ''

                    if gt_type in pred_type or pred_type in gt_type:
                        cost = float('inf')
                        if gt_evt.position is not None and pred_evt.position is not None:
                            cost = abs(gt_evt.position - pred_evt.position)
                        elif gt_evt.position is None and pred_evt.position is None:
                            cost = 0.0

                        potential_matches.append((cost, i, j))

            potential_matches.sort(key=lambda x: x[0])

            used_gt = set()
            used_pred = set()
            matched_pairs = []

            for cost, i, j in potential_matches:
                if i not in used_gt and j not in used_pred:
                    used_gt.add(i)
                    used_pred.add(j)
                    matched_pairs.append((gt_events[i], pred_events[j]))

            scores['cat'] = len(matched_pairs) / len(gt_events) if gt_events else 0.0

            num_scores = []
            for gt_evt, pred_evt in matched_pairs:
                if pred_evt.position is not None and gt_evt.position is not None:
                    val = position_accuracy(pred_evt.position, gt_evt.position, total_len)
                    num_scores.append(val)
                    scores['num_by_type'].setdefault('position', []).append(val)
                if pred_evt.amplitude is not None and gt_evt.amplitude is not None:
                    val = relative_accuracy(pred_evt.amplitude, gt_evt.amplitude)
                    num_scores.append(val)
                    scores['num_by_type'].setdefault('amplitude', []).append(val)

            scores['num'] = num_scores
        else:
            scores['cat'] = 0.0

    return scores


