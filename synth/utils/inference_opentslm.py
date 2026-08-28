"""OpenTSLM inference runner — separate from the vLLM driver.

Why a separate runner: OpenTSLM is a custom multi-modal architecture
(time-series encoder + Llama/Gemma + Soft-Prompt or Flamingo projector)
loaded via `OpenTSLM.load_pretrained()`. It is NOT vLLM-compatible:
generation goes through their own batch-dict schema and collator. We
write the same `generated_answer.json` that the vLLM runner does, so
the downstream judge pipeline (`evaluation.eval.main`) is unchanged.

Schema mapping (our test JSONL → OpenTSLM batch dict):
- `pre_prompt`  ← original `input` text with each `<ts>` replaced by
                  a positional marker like "[Time Series 1 below]" so
                  the prompt still describes which series is which.
- `time_series` ← list of 1-D arrays, z-score normalized to match the
                  TSQA training distribution (TSQADataset normalizes
                  inside `_get_text_time_series_prompt_list`).
- `time_series_text` ← per-series descriptor in the TSQA-recommended
                       format: "This is time series N, it has mean
                       <m> and std <s>."
- `post_prompt` ← "Answer:".

OpenTSLM-TSQA was trained on single-TS multi-choice questions; multi-TS
samples (mcq2 has 2, sentsr has 3, our default eval has up to 14) are
out-of-distribution. That's an honest limitation, not a bug.

Output dir: `{exp_root}/{name}/{dataset_stem}/generated_answer.json`,
matching the vLLM runner so `evaluation.eval.main` can read either.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
from typing import List

import numpy as np
from loguru import logger
from tqdm import tqdm


# Registered names → HF repo id. Mirrors the role of MODEL_CONFIGS in
# inference_tsmllm_vllm.py. Keep names lower-kebab to match exp/ dir
# convention (timeomni-1-7b, time-r1-s1p1, etc.).
MODEL_CONFIGS = {
    # Largest Llama-3.2 base (3B) + TSQA head + Soft-Prompt projector.
    # README's recommended architecture; most general OpenTSLM variant.
    "opentslm-tsqa-3b": "OpenTSLM/llama-3.2-3b-tsqa-sp",
}


def _build_sample(input_text: str, ts_arrays: List[np.ndarray]) -> dict:
    """Convert one of our test samples into an OpenTSLM batch dict."""
    # Lazy import — opentslm is a heavyweight dep (Llama + custom encoder)
    # and shouldn't be required for users who only run the vLLM models.
    from opentslm.prompt.text_prompt import TextPrompt
    from opentslm.prompt.text_time_series_prompt import TextTimeSeriesPrompt
    from opentslm.prompt.prompt_with_answer import PromptWithAnswer

    pre = input_text
    for i in range(len(ts_arrays)):
        pre = pre.replace("<ts>", f"[Time Series {i + 1} below]", 1)

    ts_prompts = []
    for i, ts in enumerate(ts_arrays):
        arr = np.asarray(ts, dtype=np.float32).reshape(-1)
        m = float(arr.mean())
        s = float(arr.std())
        s_safe = s if s > 1e-8 else 1.0
        norm = (arr - m) / s_safe
        ts_prompts.append(
            TextTimeSeriesPrompt(
                f"This is time series {i + 1}, it has mean {m:.4f} and std {s:.4f}.",
                norm.tolist(),
            )
        )

    return PromptWithAnswer(
        TextPrompt(pre.strip()),
        ts_prompts,
        TextPrompt("Answer:"),
        answer="",  # ignored at inference time
    ).to_dict()


def _load_jsonl(path: str) -> list:
    with open(path) as f:
        try:
            return json.load(f)
        except json.JSONDecodeError:
            f.seek(0)
            return [json.loads(line) for line in f if line.strip()]


def _run_dataset(model, samples, batch_size: int, max_new_tokens: int, desc: str) -> list:
    """Generate responses for each sample. Returns list aligned with `samples`."""
    from opentslm.time_series_datasets.util import (
        extend_time_series_to_match_patch_size_and_aggregate,
    )
    from opentslm.model_config import PATCH_SIZE

    answers = [""] * len(samples)
    records = []  # (orig_idx, batch_dict)
    for idx, s in enumerate(samples):
        try:
            ts_arrays = [np.asarray(t) for t in s.get("timeseries", [])]
            if not ts_arrays:
                logger.warning(f"sample {idx}: no timeseries, returning empty response")
                continue
            records.append((idx, _build_sample(s["input"], ts_arrays)))
        except Exception as e:
            logger.warning(f"sample {idx}: build failed ({e}); empty response")

    for start in tqdm(range(0, len(records), batch_size), desc=desc):
        chunk = records[start:start + batch_size]
        idxs = [r[0] for r in chunk]
        dicts = [dict(r[1]) for r in chunk]  # collator mutates in-place
        try:
            collated = extend_time_series_to_match_patch_size_and_aggregate(
                dicts, patch_size=PATCH_SIZE
            )
            preds = model.generate(collated, max_new_tokens=max_new_tokens)
        except Exception as e:
            logger.warning(
                f"batch starting at {start} failed ({type(e).__name__}: {e}); "
                f"isolating per-sample"
            )
            preds = []
            for one in dicts:
                try:
                    one_collated = extend_time_series_to_match_patch_size_and_aggregate(
                        [dict(one)], patch_size=PATCH_SIZE
                    )
                    p = model.generate(one_collated, max_new_tokens=max_new_tokens)
                    preds.append(p[0] if p else "")
                except Exception as e2:
                    logger.warning(f"  single-sample also failed: {e2}")
                    preds.append("")
        for idx, pred in zip(idxs, preds):
            answers[idx] = pred if isinstance(pred, str) else str(pred)
    return answers


def _write_output(samples: list, answers: list, output_file: Path) -> None:
    """Mirror the vLLM runner's generated_answer.json schema."""
    generated = []
    for idx, sample in enumerate(samples):
        eval_md = sample.get("eval_metadata") or {}
        if "length" not in eval_md:
            ts_data = sample.get("timeseries", [])
            if ts_data:
                eval_md = dict(eval_md)
                eval_md["length"] = len(ts_data[0])
        generated.append({
            "idx": idx,
            "question_text": sample.get("input", ""),
            "response": answers[idx],
            "ground_truth": sample.get("output", ""),
            "eval_type": sample.get("eval_type"),
            "eval_metadata": eval_md,
        })
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "wt") as f:
        json.dump(generated, f, ensure_ascii=False, indent=4)
    logger.info(f"Wrote {output_file} ({len(generated)} rows)")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Run OpenTSLM inference across one-or-more datasets and one-or-more "
            "registered OpenTSLM checkpoints. Output schema matches the vLLM "
            "runner so `evaluation.eval.main` can score either path."
        ),
    )
    parser.add_argument("--models", nargs="+", choices=list(MODEL_CONFIGS.keys()),
                        default=list(MODEL_CONFIGS.keys()),
                        help="Which OpenTSLM checkpoints to run (default: all)")
    parser.add_argument("--datasets", nargs="+", required=True,
                        help="One or more dataset paths (.jsonl).")
    parser.add_argument("--exp-root", default="exp",
                        help="Root output dir. Results -> {exp-root}/{model}/{stem}/.")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--force", action="store_true",
                        help="Re-run (model, dataset) pairs whose generated_answer.json exists.")
    args = parser.parse_args()

    import torch
    from opentslm.model.llm.OpenTSLM import OpenTSLM as OpenTSLMFactory

    device = "cuda" if torch.cuda.is_available() else "cpu"

    dataset_paths = [Path(p) for p in args.datasets]
    for p in dataset_paths:
        if not p.exists():
            raise SystemExit(f"Dataset not found: {p}")
    logger.info(f"Datasets: {[str(p) for p in dataset_paths]}")

    # Outer loop = model (loaded once), inner loop = dataset.
    for model_name in args.models:
        repo_id = MODEL_CONFIGS[model_name]
        # Per OpenTSLM's HAR demo: SP-suffix repos load with enable_lora=True.
        enable_lora = repo_id.endswith("-sp")

        # Plan: skip pairs whose output already exists (matches vLLM runner).
        pending = []
        for ds_path in dataset_paths:
            stem = ds_path.stem
            output_file = Path(args.exp_root) / model_name / stem / "generated_answer.json"
            if output_file.exists() and not args.force:
                logger.info(f"[{model_name}/{stem}] SKIP — {output_file} exists (use --force)")
                continue
            pending.append((ds_path, output_file))

        if not pending:
            logger.info(f"[{model_name}] nothing to do")
            continue

        logger.info(f"=== {model_name} ({repo_id}, lora={enable_lora}, device={device}) ===")
        model = OpenTSLMFactory.load_pretrained(
            repo_id, enable_lora=enable_lora, device=device,
        )
        try:
            for ds_path, output_file in pending:
                stem = ds_path.stem
                samples = _load_jsonl(str(ds_path))
                logger.info(f"[{model_name}/{stem}] {len(samples)} samples")
                answers = _run_dataset(
                    model, samples,
                    batch_size=args.batch_size,
                    max_new_tokens=args.max_new_tokens,
                    desc=f"{model_name}/{stem}",
                )
                _write_output(samples, answers, output_file)
        finally:
            # Free GPU before loading the next checkpoint (if any).
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
