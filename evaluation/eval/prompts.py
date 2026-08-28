import json
from loguru import logger

from evaluation.ragas.metric import (
    MTSExtractionInput, MTSExtractionPrompt,
    DescriptionExtractionInput, DescriptionExtractionPrompt,
)
from evaluation.eval.config import (
    JUDGE_PROMPT_TEMPLATE, SEGMENT_JUDGE_PROMPT_TEMPLATE, MCQ_JUDGE_PROMPT_TEMPLATE,
    RAGAS_QUESTION_BY_TASK,
    SEGMENT_FAMILY_TYPES, resolve_judge_question,
)
from evaluation.eval.utils import extract_subject_from_question


_MCQ_LETTERS = ("A", "B", "C", "D")


def _build_mcq_options_block(options):
    """Render options as 'A) <opt>' lines. Returns '' if options missing/malformed."""
    if not options or not isinstance(options, (list, tuple)):
        return ""
    lines = []
    for letter, opt in zip(_MCQ_LETTERS, options):
        lines.append(f"{letter}) {str(opt).strip()}")
    return "\n".join(lines)


def prepare_prompts(data_list):
    logger.info("PHASE 1: Pre-generating prompts for all samples...")

    extractor = MTSExtractionPrompt()
    output_schema = json.dumps(extractor.output_model.model_json_schema(), indent=2)
    system_content = (
        f"{extractor.instruction}\n\n"
        f"You must output a valid JSON object matching this schema:\n"
        f"{output_schema}"
    )

    desc_extractor = DescriptionExtractionPrompt()
    desc_output_schema = json.dumps(desc_extractor.output_model.model_json_schema(), indent=2)
    desc_system_content = (
        f"{desc_extractor.instruction}\n\n"
        f"You must output a valid JSON object matching this schema:\n"
        f"{desc_output_schema}"
    )

    prepared_tasks = []

    for sample in data_list:
        gt_text = sample.get("ground_truth", sample.get("output", ""))
        model_response = sample["response"] if "response" in sample else sample.get("output", "")

        q_text = sample.get("question_text", sample.get("input", ""))
        q_lower = q_text.lower()

        eval_type = sample.get("eval_type")
        eval_metadata = sample.get("eval_metadata", {})

        # Pass FULL content (think + response) to the judge on both GT and
        # pred. See memory/feedback_no_strip_think_for_judge.md — the original
        # strip design was for fairness vs a no-think baseline, which doesn't
        # apply to SFT-with-think vs SFT+RL-with-think comparisons. Strip
        # also amplifies verdict-only-RL terseness into fake regression on
        # free-form (real_*) eval types.
        gt_answer_text = (gt_text or "").strip()
        pred_answer_text = (model_response or "").strip()

        if gt_answer_text.strip() and pred_answer_text.strip():
            judge_q = resolve_judge_question(eval_type, q_text)

            # MCQ-evolved samples take precedence over the source eval_type's
            # default routing — the source eval_type might be in SEGMENT_FAMILY_TYPES
            # (since MCQ data is built on top of all the source taxonomy types),
            # but the actual answer is a letter A/B/C/D, not a free-form verdict.
            if (eval_metadata or {}).get('mcq_evolved'):
                md = eval_metadata or {}
                options = md.get('options', [])
                correct_letter = md.get('answer_letter', '')
                correct_option = ''
                answer_idx = md.get('answer_index')
                if isinstance(answer_idx, int) and 0 <= answer_idx < len(options):
                    correct_option = str(options[answer_idx]).strip()
                options_block = _build_mcq_options_block(options)
                user_content = MCQ_JUDGE_PROMPT_TEMPLATE.format(
                    question=judge_q,
                    options_block=options_block,
                    correct_letter=correct_letter,
                    correct_option=correct_option,
                    pred=pred_answer_text,
                )
            elif eval_type in SEGMENT_FAMILY_TYPES:
                # Segment-family: judge sees full response (not first-sentence-stripped)
                # and the expected verdict, so it can evaluate both correctness and quality.
                expected_verdict = (eval_metadata or {}).get('verdict', '')
                user_content = SEGMENT_JUDGE_PROMPT_TEMPLATE.format(
                    question=judge_q,
                    expected_verdict=expected_verdict,
                    gt=gt_answer_text,
                    pred=pred_answer_text,
                )
            else:
                user_content = JUDGE_PROMPT_TEMPLATE.format(
                    question=judge_q,
                    gt=gt_answer_text,
                    pred=pred_answer_text,
                )

            judge_task = {
                "idx": sample.get("idx"),
                "question": q_text,
                "type": "reasoning_judge",
                "eval_type": eval_type,
                "eval_metadata": eval_metadata,
                "pred_answer_text": pred_answer_text,
                "payload": [
                    {"role": "system", "content": "You are a helpful evaluator. Output JSON only."},
                    {"role": "user", "content": user_content}
                ]
            }
            prepared_tasks.append(judge_task)

        # TSEvol: evolved QA — judge already scheduled above, no extraction needed.
        if eval_type == "tsevol":
            continue

        is_description = (
            eval_type == "description"
            or (eval_type is None and "analyze the characteristics" in q_lower and "perspective" in q_lower)
        )

        if is_description:
            gt_input_json = DescriptionExtractionInput(text=gt_answer_text).model_dump_json()
            pred_input_json = DescriptionExtractionInput(text=pred_answer_text).model_dump_json()

            prepared_tasks.append({
                "idx": sample.get("idx"),
                "question": q_text,
                "type": "desc_gt",
                "task_type": "description",
                "eval_type": eval_type,
                "eval_metadata": eval_metadata,
                "gt_text": gt_answer_text,
                "response_text": pred_answer_text,
                "payload": [
                    {"role": "system", "content": desc_system_content},
                    {"role": "user", "content": gt_input_json}
                ]
            })

            prepared_tasks.append({
                "idx": sample.get("idx"),
                "question": q_text,
                "type": "desc_pred",
                "task_type": "description",
                "eval_type": eval_type,
                "eval_metadata": eval_metadata,
                "gt_text": gt_answer_text,
                "response_text": pred_answer_text,
                "payload": [
                    {"role": "system", "content": desc_system_content},
                    {"role": "user", "content": pred_input_json}
                ]
            })
            continue

        # Segment family types: verdict comes from eval_metadata directly; no LLM extraction needed.
        if eval_type in SEGMENT_FAMILY_TYPES:
            continue

        subject = extract_subject_from_question(q_lower)
        gt_input_json = MTSExtractionInput(text=gt_answer_text, subject_to_exclude=subject).model_dump_json()
        pred_input_json = MTSExtractionInput(text=pred_answer_text, subject_to_exclude=subject).model_dump_json()

        prepared_tasks.append({
            "idx": sample.get("idx"),
            "question": q_text,
            "type": "gt",
            "eval_type": eval_type,
            "eval_metadata": eval_metadata,
            "gt_answer_text": gt_answer_text,
            "pred_answer_text": pred_answer_text,
            "payload": [
                {"role": "system", "content": system_content},
                {"role": "user", "content": gt_input_json}
            ]
        })

        prepared_tasks.append({
            "idx": sample.get("idx"),
            "question": q_text,
            "type": "pred",
            "eval_type": eval_type,
            "eval_metadata": eval_metadata,
            "gt_answer_text": gt_answer_text,
            "pred_answer_text": pred_answer_text,
            "payload": [
                {"role": "system", "content": system_content},
                {"role": "user", "content": pred_input_json}
            ]
        })

    return prepared_tasks
