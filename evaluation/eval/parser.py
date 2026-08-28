import json
import re
from loguru import logger
from pydantic import ValidationError

from evaluation.ragas.metric import MTSExtraction, DescriptionExtraction
from evaluation.eval.utils import clean_json_text, extract_subject_from_question
from evaluation.eval.config import SEGMENT_FAMILY_TYPES, resolve_task_type
from evaluation.eval.scoring import (
    detect_description_perspectives,
    score_description_perspective,
)

# Eval types where the verdict is numeric and should use relative accuracy
# instead of binary exact-match from the LLM judge.
# Small-integer counts (e.g. rl_event_count, ood_conditional_count) are
# deliberately excluded — exact match is appropriate there.
NUMERIC_VERDICT_TYPES = frozenset({
    # Positional (scaled by seq_len)
    'temporal_position', 'atomic_max_position', 'atomic_min_position',
    'change_point',
    # Polymorphic (value-based; string GT falls through to verdict_match)
    'stat_numerical', 'duration_proportion', 'atomic_periodic_description',
    'ood_max_amp_in_longest_segment',
    # Single-metric atomic floats
    'atomic_global_mean', 'atomic_global_std',
    'atomic_interval_mean', 'atomic_interval_std',
    'atomic_max_value', 'atomic_min_value', 'atomic_percentile',
    'atomic_chunked_means',  # always list-shaped
    'local_enumeration', 'periodicity',
    # RL float-valued
    'rl_half_mean_diff', 'rl_normalized_range', 'rl_period_estimate',
    'rl_range', 'rl_cycle_count', 'rl_type_duration_fraction',
    # OOD float-valued
    'ood_chunk_above_proportion', 'ood_conditional_mean_by_type',
    'ood_event_density_by_trend', 'ood_longest_type_fraction',
    # OOD compositional (existing)
    'ood_conditional_stat', 'ood_nested_extrema',
    # OOD compositional probes (reverse-engineered)
    'ood_duration_weighted_mean',
    # RL cross-metric float fractions (post-think prose typically says
    # "X of Y metrics" rather than "0.43" — judge prompt computes X/Y).
    'rl_cross_trend_concordance',
    # Count-valued types previously stuck on binary verdict_match.
    # Audit on 2026-04-24: these all have 100% numeric GT but were
    # routed through verdict_match → string-exact match → ~0 credit
    # for any prose deviation. Proximity scoring restores partial
    # credit for near-correct counts.
    'ood_cross_concordant_shift', 'ood_cross_corr_count',
    'ood_mixed_corr_anti', 'ood_conditional_count',
    'rl_event_count', 'rl_event_count_by_type', 'rl_segment_count',
    'rl_cluster_count', 'rl_cross_asymmetric_behavior',
    'atomic_cross_counting', 'atomic_trend_enumeration',
    'atomic_event_enumeration', 'segment_enumeration',
    # Positional / duration types (also added to POSITIONAL_VERDICT_TYPES
    # so seq_len-scaled accuracy is used).
    'ood_max_mean_chunk_pos', 'rl_segment_duration', 'rl_longest_segment',
})

# Positional types: relative accuracy scaled by sequence length, not by |gt|.
POSITIONAL_VERDICT_TYPES = frozenset({
    'temporal_position', 'atomic_max_position', 'atomic_min_position',
    'change_point',
    # Durations / segment positions — values up to seq_len, so scaling
    # by seq_len gives a meaningful "off-by-N positions" accuracy.
    'rl_segment_duration', 'rl_longest_segment',
})


# List / ordering / set / multi-element verdicts where binary string-exact
# match systematically undercredits partial-correct answers (e.g., one swap
# in a 6-element ordering → 0). The judge's `correctness_score` field gives
# a graded similarity score (Kendall's tau intuition for ordered, Jaccard
# intuition for sets), focused on answer correctness only (NOT explanation
# quality — that's `score`/reasoning_score, which conflates depth+correctness).
LIST_VERDICT_TYPES = frozenset({
    # Ordered-list verdicts (rank metrics by some criterion)
    'rl_cross_full_ordering',
    'atomic_cross_ranking',
    # Set / filter verdicts (which metrics meet the condition)
    'atomic_cross_filtering',
    'clustering', 'anticlustering',
    # Enumeration verdicts where verdict is a sequence/list (order matters)
    # Note: segment_enumeration / local_enumeration despite their names emit
    # scalar verdicts (single count/position) — those stay in NUMERIC.
    'cross_metric_enumeration',
    'transition_enumeration',
    'event_segment_enumeration',
    # Sequence pattern verdicts
    'ood_symmetric_trend_sequence',
})


# Length buckets for generalization reporting. Boundaries match the
# data-generation split (SFT ≤256, RL trained on 32-768, OOD near/far/extreme).
# See memory/pipeline_length_generalization_2026_04_21.md.
def length_bucket(seq_len):
    """Map a sequence length to a stratification bucket.

    Buckets:
      in_sft   : seq_len ≤ 256        (both SFT and RL trained here)
      rl_only  : 256 < seq_len ≤ 768  (only RL trained — "in-distribution-RL")
      ood_near : 768 < seq_len ≤ 1536 (neither trained — near OOD)
      ood_far  : seq_len > 1536       (extreme OOD)
      unknown  : seq_len is None/invalid
    """
    if seq_len is None:
        return 'unknown'
    try:
        L = int(seq_len)
    except (TypeError, ValueError):
        return 'unknown'
    if L <= 256:
        return 'in_sft'
    if L <= 768:
        return 'rl_only'
    if L <= 1536:
        return 'ood_near'
    return 'ood_far'


def _extract_first_number(text: str):
    """Extract the first number (int or float, possibly negative) from text."""
    if text is None:
        return None
    m = re.search(r'-?\d+\.?\d*', str(text))
    if m:
        try:
            return float(m.group())
        except ValueError:
            return None
    return None


def _extract_number_list(text: str):
    """Extract an ordered list of numbers from a '[a, b, c]'-style string.

    Returns None if the text isn't bracketed or can't be parsed as numbers.
    Falls back gracefully for mixed content.
    """
    if text is None:
        return None
    s = str(text).strip()
    # Find the first [...] block
    m = re.search(r'\[([^\[\]]*)\]', s)
    if not m:
        return None
    inner = m.group(1).strip()
    if not inner:
        return []
    parts = [p.strip() for p in inner.split(',')]
    nums = []
    for p in parts:
        n = _extract_first_number(p)
        if n is None:
            return None
        nums.append(n)
    return nums


def _numeric_relative_accuracy(pred_val, gt_val, seq_len=None):
    """Compute relative accuracy for a single numeric verdict.

    For positional verdicts (temporal_position), scale by sequence length.
    For other numeric verdicts, scale by |gt| (with a floor to avoid div-by-zero).
    Returns a float in [0, 1].
    """
    if pred_val is None or gt_val is None:
        return 0.0
    diff = abs(pred_val - gt_val)
    if seq_len and seq_len > 0:
        return max(0.0, 1.0 - diff / seq_len)
    scale = max(abs(gt_val), 1.0)
    return max(0.0, min(1.0, 1.0 - diff / scale))


def _list_relative_accuracy(pred_list, gt_list, seq_len=None):
    """Element-wise relative accuracy for ordered numeric lists.

    Length mismatch penalizes: divide the summed per-element accuracy by
    max(len(pred), len(gt)), so missing or extra elements score 0.
    """
    if gt_list is None:
        return 0.0
    n_gt = len(gt_list)
    n_pred = len(pred_list) if pred_list is not None else 0
    n = max(n_gt, n_pred)
    if n == 0:
        return 1.0  # both empty → perfect
    total = 0.0
    for i in range(n):
        if i >= n_gt or i >= n_pred:
            continue  # missing/extra element contributes 0
        total += _numeric_relative_accuracy(pred_list[i], gt_list[i], seq_len)
    return total / n


def parse_and_score(raw_results):
    logger.info(f"PHASE 3: Parsing {len(raw_results)} JSONs and calculating metrics (CPU)...")

    organized = {}
    for task, response_text in raw_results:
        idx = task['idx']
        t_type = task['type']

        if idx not in organized:
            organized[idx] = {'question': task['question']}

        if 'eval_type' in task and task['eval_type'] is not None:
            organized[idx]['eval_type'] = task['eval_type']
        if 'eval_metadata' in task and task['eval_metadata'] is not None:
            organized[idx]['eval_metadata'] = task['eval_metadata']

        if t_type in ('desc_gt', 'desc_pred'):
            organized[idx]['task_type'] = 'description'
            organized[idx]['gt_text'] = task.get('gt_text', '')
            organized[idx]['response_text'] = task.get('response_text', '')

        if t_type in ('gt', 'pred'):
            organized[idx]['gt_answer_text'] = task.get('gt_answer_text', '')
            organized[idx]['pred_answer_text'] = task.get('pred_answer_text', '')

        if t_type == 'reasoning_judge' and task.get('pred_answer_text') is not None:
            organized[idx].setdefault('pred_answer_text', task['pred_answer_text'])

        if response_text is None:
            continue

        organized[idx][t_type] = response_text

    final_rows = []

    for idx, data in organized.items():
        reasoning_score = None
        verdict_match = None
        correctness_score = None
        judge_json = data.get('reasoning_judge')
        if judge_json:
            clean_judge = clean_json_text(judge_json)
            if clean_judge:
                try:
                    score_obj = json.loads(clean_judge)
                    reasoning_score = float(score_obj["score"])
                    # Segment-family judge also outputs verdict_match
                    vm = score_obj.get("verdict_match")
                    if vm is not None:
                        verdict_match = float(vm)
                    # Numeric eval_types: judge extracts the final numeric
                    # verdict from prose (robust against explanatory
                    # numbers that would confuse a regex extractor).
                    extracted = score_obj.get("extracted_number")
                    if extracted is not None:
                        data['extracted_number'] = extracted
                    # List / ordering / set verdicts: judge gives a
                    # graded correctness score (Kendall's-tau / Jaccard
                    # intuition) focused on answer correctness only.
                    cs = score_obj.get("correctness_score")
                    if cs is not None:
                        correctness_score = float(cs)
                    # MCQ judge: the picked letter is informative for
                    # diagnostics; stash it on the row so LaTeX tables
                    # and confusion matrices can use it later.
                    picked_letter = score_obj.get("picked_letter")
                    if picked_letter is not None:
                        data['picked_letter'] = picked_letter
                except (json.JSONDecodeError, KeyError, ValueError, TypeError) as e:
                    logger.warning(f"Failed to parse judge JSON for idx {idx}: {e}")
                    reasoning_score = 0.0

        question = data.get('question', "")
        q_lower = question.lower()

        eval_type = data.get('eval_type')
        eval_metadata = data.get('eval_metadata', {})
        task_type = data.get('task_type')

        if task_type is None:
            task_type = resolve_task_type(eval_type, question)

        if task_type == 'tsevol':
            # Evolved QA: judge-only scoring, no extraction.
            # Track evol_strategy for per-strategy performance breakdown.
            evol_strategy = (eval_metadata or {}).get('evol_strategy', 'unknown')
            final_rows.append({
                "idx": idx,
                "task_type": task_type,
                "reasoning_score": reasoning_score,
                "evol_strategy": evol_strategy,
                "binary_accuracy": None,
                "gt_verdict": None,
                "pred_verdict": None,
                "precision": None,
                "recall": None,
                "f1": None,
                "gt_len": None,
                "gt_entities": None,
                "pred_entities": None,
            })
            continue

        if task_type in SEGMENT_FAMILY_TYPES:
            # binary_accuracy and reasoning_score both come from the LLM judge.
            # The segment judge prompt asks for both verdict_match and score.
            # No think-block parsing — evaluation must be identical for
            # baseline (no think) and think-block models.
            gt_verdict_raw = (eval_metadata or {}).get('verdict')
            accuracy = verdict_match

            # For list / ordering / set verdicts, the judge's
            # `correctness_score` is the right partial-credit metric
            # (Kendall's-tau / Jaccard intuition). Binary verdict_match
            # punishes any swap → systematic undercredit on ordering tasks.
            # correctness_score is FOCUSED on answer correctness, not
            # explanation quality, so it doesn't import the conflation
            # that reasoning_score has.
            if task_type in LIST_VERDICT_TYPES and correctness_score is not None:
                accuracy = correctness_score

            # For numeric verdict types, override the LLM judge's binary
            # verdict_match with relative accuracy.  Exact-match is too
            # strict for positions/counts/proportions — a prediction of
            # "43" for GT "42" in a 256-length series is 99.6% correct,
            # not 0%.
            if task_type in NUMERIC_VERDICT_TYPES and gt_verdict_raw is not None:
                pred_text = data.get('pred_answer_text', '')
                seq_len = (eval_metadata or {}).get('length')
                use_seq_len = (
                    seq_len if task_type in POSITIONAL_VERDICT_TYPES else None
                )
                # Prefer the judge-extracted numeric value when present —
                # the judge's semantic extraction handles prose like
                # "accounts for 0.52 of the total (out of 255)" without
                # confusing the verdict (0.52) with explanatory numbers.
                # See SEGMENT_JUDGE_PROMPT_TEMPLATE: `extracted_number`
                # field is populated for numeric eval_types.
                judge_extracted = data.get('extracted_number')
                # Try list-shaped GT first (e.g. '[106, 151]',
                # '[-14.48, -7.75, 3.19]') — element-wise scoring avoids
                # giving full credit for matching only the first element.
                gt_list = _extract_number_list(gt_verdict_raw)
                if gt_list is not None:
                    pred_list = _extract_number_list(pred_text)
                    if pred_list is not None:
                        accuracy = _list_relative_accuracy(
                            pred_list, gt_list, use_seq_len
                        )
                    # else: no parseable list in prediction → keep LLM
                    # judge's verdict_match rather than silently grading
                    # the pred's first number against the full GT list.
                else:
                    gt_num = _extract_first_number(gt_verdict_raw)
                    pred_num = None
                    if judge_extracted is not None:
                        try:
                            pred_num = float(judge_extracted)
                        except (TypeError, ValueError):
                            pred_num = None
                    if pred_num is None:
                        pred_num = _extract_first_number(pred_text)
                    if gt_num is not None and pred_num is not None:
                        accuracy = _numeric_relative_accuracy(
                            pred_num, gt_num, use_seq_len
                        )

            final_rows.append({
                "idx": idx,
                "task_type": task_type,
                "binary_accuracy": accuracy,
                "reasoning_score": reasoning_score,
                "gt_verdict": str(gt_verdict_raw) if gt_verdict_raw is not None else None,
                "pred_verdict": None,
                "precision": None,
                "recall": None,
                "f1": None,
                "gt_len": None,
                "gt_entities": None,
                "pred_entities": None,
            })
            continue

        if task_type == "description":
            raw_perspectives = (
                eval_metadata.get('perspectives')
                if eval_metadata and eval_metadata.get('perspectives')
                else None
            )
            if raw_perspectives is None:
                perspectives = detect_description_perspectives(question)
            elif isinstance(raw_perspectives, dict):
                # Compound description: {metric: [perspective_list]} → flatten
                perspectives = list(dict.fromkeys(
                    p for plist in raw_perspectives.values() for p in plist
                ))
            else:
                perspectives = list(raw_perspectives)
            # Normalize generation names → eval scoring names
            _PERSPECTIVE_MAP = {
                'periodicity': 'seasonal',
                'local events': 'local',
            }
            perspectives = [_PERSPECTIVE_MAP.get(p, p) for p in perspectives]

            def parse_desc(raw_text):
                clean = clean_json_text(raw_text)
                if not clean: return None
                try:
                    return DescriptionExtraction.model_validate_json(clean)
                except Exception as e:
                    logger.warning(f"Desc Parse Error: {e} | Text: {clean[:50]}...")
                    return None

            gt_extraction = parse_desc(data.get('desc_gt', ''))
            pred_extraction = parse_desc(data.get('desc_pred', ''))

            if gt_extraction is None or pred_extraction is None:
                final_rows.append({
                    "idx": idx,
                    "task_type": "description",
                    "reasoning_score": None,
                    "description_overall": None
                })
                continue

            series_len = eval_metadata.get('length') if eval_metadata else None

            row = {
                "idx": idx,
                "task_type": "description",
                "gt_len": None,
                "precision": None,
                "recall": None,
                "f1": None,
                "binary_accuracy": None,
                "gt_verdict": None,
                "pred_verdict": None,
                "gt_entities": None,
                "pred_entities": None,
            }

            all_cat_scores = []
            all_num_scores = []

            for p in perspectives:
                scores = score_description_perspective(p, gt_extraction, pred_extraction, total_len=series_len)
                cat_val = scores['cat']
                num_vals = [v for v in scores['num'] if v is not None]
                num_by_type = scores.get('num_by_type', {})

                row[f'{p}_cat'] = cat_val
                row[f'{p}_num'] = float(sum(num_vals) / len(num_vals)) if num_vals else None

                for ntype, nvals in num_by_type.items():
                    valid = [v for v in nvals if v is not None]
                    row[f'{p}_num_{ntype}'] = float(sum(valid) / len(valid)) if valid else None

                if p == 'trend':
                    row['trend_overall_cat'] = scores.get('overall_cat')
                    row['trend_segment_precision'] = scores.get('segment_precision')
                    row['trend_segment_recall'] = scores.get('segment_recall')
                    row['trend_segment_f1'] = scores.get('segment_f1')
                    seg_num = [v for v in scores.get('segment_num', []) if v is not None]
                    row['trend_segment_num'] = float(sum(seg_num) / len(seg_num)) if seg_num else None

                if cat_val is not None:
                    all_cat_scores.append(cat_val)
                if num_vals:
                    all_num_scores.extend(num_vals)

            row['reasoning_score'] = reasoning_score
            row['local_reasoning_score'] = None

            combined = all_cat_scores + all_num_scores
            if reasoning_score is not None:
                combined.append(reasoning_score)
            row['description_overall'] = float(sum(combined) / len(combined)) if combined else None

            final_rows.append(row)
            continue

        def parse_mts(raw_text):
            clean = clean_json_text(raw_text)
            if not clean: return None
            try:
                return MTSExtraction.model_validate_json(clean)
            except ValidationError as e:
                logger.warning(f"MTS Parse Error: {e.json()}")
                return None

        gt_obj = parse_mts(data.get('gt', ''))
        pred_obj = parse_mts(data.get('pred', ''))

        if gt_obj is None or pred_obj is None:
            final_rows.append({
                "idx": idx,
                "task_type": task_type,
                "gt_len": None,
                "precision": None,
                "recall": None,
                "f1": None,
                "binary_accuracy": None,
                "reasoning_score": None,
                "gt_verdict": None,
                "pred_verdict": None,
                "gt_entities": None,
                "pred_entities": None
            })
            continue

        gt_entities = set(gt_obj.related_entities) if gt_obj else set()
        pred_entities = set(pred_obj.related_entities) if pred_obj else set()

        subject_to_remove = extract_subject_from_question(q_lower)
        gt_entities = {e for e in gt_entities if e.lower() != subject_to_remove.lower()}
        pred_entities = {e for e in pred_entities if e.lower() != subject_to_remove.lower()}

        gt_verdict = gt_obj.verdict if gt_obj else False
        pred_verdict = pred_obj.verdict if pred_obj else False

        gt_len = len(gt_entities)

        f1 = None
        precision = None
        recall = None
        binary_accuracy = None

        # Verdict-based correctness for tasks with a single right answer.
        # `correlation`/`anticorrelation`/`yes_no` are synthetic binary-judgment types.
        if task_type in ["correlation", "anticorrelation", "yes_no"]:
            if pred_obj is None:
                binary_accuracy = 0.0
            else:
                binary_accuracy = 1.0 if gt_verdict == pred_verdict else 0.0

        elif task_type in ("clustering", "anticlustering"):
            if not gt_entities and not pred_entities:
                p, r, f1_calc = 1.0, 1.0, 1.0
            elif not gt_entities or not pred_entities:
                p, r, f1_calc = 0.0, 0.0, 0.0
            else:
                tp = len(gt_entities.intersection(pred_entities))
                p = tp / len(pred_entities)
                r = tp / len(gt_entities)
                f1_calc = 2 * (p * r) / (p + r) if (p + r) > 0 else 0

            precision = p
            recall = r
            f1 = f1_calc

        if task_type in ("correlation", "anticorrelation", "yes_no"):
            if pred_obj is None or gt_verdict != pred_verdict:
                reasoning_score = 0.0

        # Consistent with above: if the model got zero entities right, the
        # reasoning led to a wrong conclusion.  The general judge template
        # does NOT see the expected entities, so it can't factor accuracy
        # into reasoning_score on its own — we must zero it explicitly.
        if task_type in ("clustering", "anticlustering"):
            if f1 is not None and f1 == 0.0:
                reasoning_score = 0.0

        final_rows.append({
            "idx": idx,
            "task_type": task_type,
            "gt_len": gt_len,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "binary_accuracy": binary_accuracy,
            "reasoning_score": reasoning_score,
            "gt_verdict": gt_verdict,
            "pred_verdict": pred_verdict,
            "gt_entities": list(gt_entities),
            "pred_entities": list(pred_entities)
        })

    # Inject length_bucket / length onto every row from eval_metadata.length,
    # so aggregation can stratify by training-envelope bucket (in_sft / rl_only
    # / ood_near / ood_far). See length_bucket() and the data-generation
    # design in memory/pipeline_length_generalization_2026_04_21.md.
    for row in final_rows:
        idx = row.get('idx')
        md = organized.get(idx, {}).get('eval_metadata') or {}
        row_len = md.get('length')
        row['length'] = row_len
        row['length_bucket'] = length_bucket(row_len)

    return final_rows
