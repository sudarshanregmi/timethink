import ts_vllm
from vllm import LLM, SamplingParams
from synth.utils.llm_utils import LLMClient
from synth.utils.baseline_text_rendering import inline_timeseries_as_text
import torch
import os
import json
import argparse
from pathlib import Path
from loguru import logger
import numpy as np

DATASET = './data/test_lite.jsonl'
WORKDIR = os.path.abspath('./')
NUM_GPUS = 8
NUM_GPUS_PER_PROCESS = 1
MAX_MM_PER_PROMPT = 50
CHUNK_SIZE = 5000

MODEL_CONFIGS = {
    "sft":           os.path.abspath("sft_ckpt"),
    "sft_nothink":   os.path.abspath("sft_nothink_ckpt"),
    "sft_huge":      os.path.abspath("sft_huge_ckpt"),
    "rl":            os.path.abspath("rl_ckpt"),
    "rl_balanced":   os.path.abspath("rl_balanced_ckpt"),
    "rl_10k":        os.path.abspath("rl_ckpt_10k"),
    # Third-party text-only baselines; <ts> placeholders rendered as
    # `[v1, v2, ...]` text instead of multi-modal embeddings. See
    # MODEL_OVERRIDES below for per-model engine / sampling / prompt
    # behavior. timeomni and time-r1 are Qwen-Instruct fine-tunes (use
    # chat template). time-mqa is a base-Qwen LoRA continual-pretraining
    # fine-tune (NO chat template; completion-style with Q/A wrap).
    "timeomni-1-7b":      os.path.abspath("timeomni-1-7b-ckpt"),
    "time-r1-s1p1":       os.path.abspath("time-r1-s1p1-ckpt"),
    "time-mqa-qwen25-7b": os.path.abspath("time-mqa-qwen25-7b-ckpt"),
    # Stock Instruct baselines — no domain TS training. Zero-shot reference
    # columns. Chat template ON (default), default system prompt.
    "qwen25-instruct-7b":      os.path.abspath("qwen25-instruct-7b-ckpt"),
    "llama3-instruct-8b":      os.path.abspath("llama3-instruct-8b-ckpt"),
    "mistral-instruct-7b-v03": os.path.abspath("mistral-instruct-7b-v03-ckpt"),
}

# Default sampling for our ChatTS-style models.
sampling_params = SamplingParams(
    max_tokens=4096,
    temperature=0.5
)

# TimeOmni-1 official inference recipe (eval/inference.py in their repo +
# the per-question example in inference/inference.py): low-temperature,
# tight top-p, mild repetition penalty, 4096 output cap.
_TIMEOMNI_SAMPLING = SamplingParams(
    max_tokens=4096,
    temperature=0.1,
    top_p=0.001,
    repetition_penalty=1.05,
)

_TIMEOMNI_SYSTEM_PROMPT = (
    "Output Format:\n"
    "<think>Your step-by-step reasoning process that justifies your answer</think>\n"
    "<answer>Your final answer(Note: Only output a single uppercase letter of "
    "the correct option)</answer>"
)

# Time-R1 (Qwen2.5-3B-Instruct fine-tune) — its preliminary/test_original_ability.py
# inference example uses the standard Qwen system prompt and `model.generate`'s
# default sampling. We mirror that with Qwen2.5-Instruct's recommended decoding
# defaults (temperature=0.7, top_p=0.8, top_k=20, repetition_penalty=1.05) and
# a 4096-token cap, matching our other models' output ceiling.
_TIME_R1_SAMPLING = SamplingParams(
    max_tokens=4096,
    temperature=0.7,
    top_p=0.8,
    top_k=20,
    repetition_penalty=1.05,
)

# Time-MQA (arXiv:2503.01875) is base-Qwen2.5-7B continual-pretraining
# with LoRA on TSQA. The paper says "we formatted all QA pairs so that
# the question and answer were clearly labeled, and then we tokenized
# the text" (§4.1) — completion-style, no chat template. Their tokenizer
# config has `chat_template: None`. The wrapper below produces the
# Q/A-labeled prompt the model was trained to complete.
_TIME_MQA_SAMPLING = SamplingParams(
    max_tokens=4096,
    temperature=0.1,
    top_p=0.9,
    repetition_penalty=1.05,
    stop=["\nQuestion:", "<|endoftext|>"],
)


def _time_mqa_wrap(question: str) -> str:
    return f"Question: {question}\nAnswer:"


# Stock Instruct baselines — sampling per each family's recommended defaults.
# stop_token_ids unset → vLLM falls back to each model's tokenizer EOS, which is
# correct (Qwen → <|endoftext|>/<|im_end|>, Llama-3 → <|eot_id|>,
# Mistral-v0.3 → </s>).
_QWEN25_INSTRUCT_SAMPLING = SamplingParams(
    max_tokens=4096,
    temperature=0.7,
    top_p=0.8,
    top_k=20,
    repetition_penalty=1.05,
)
_LLAMA3_INSTRUCT_SAMPLING = SamplingParams(
    max_tokens=4096,
    temperature=0.6,
    top_p=0.9,
)
_MISTRAL_INSTRUCT_SAMPLING = SamplingParams(
    max_tokens=4096,
    temperature=0.7,
    top_p=1.0,
)


# Per-model overrides. Models absent from this table use the multi-modal
# `vllm-ts` engine + default sampling + "You are a helpful assistant."
# system prompt (LLMClient default) + chat template ON.
#
# Override knobs:
#   engine:            'vllm-ts' (default; multi-modal) or 'vllm' (text-only)
#   text_only:         if True, inline <ts> as text + skip multi-modal packing
#   system_prompt:     overrides LLMClient default
#   sampling_params:   per-model SamplingParams
#   use_chat_template: default True; set False for base-LM fine-tunes
#                      (LLMClient passes the prompt as-is to vLLM)
#   prompt_wrapper:    callable str->str, applied after <ts> rendering
#                      (used when use_chat_template=False to add Q/A labels)
MODEL_OVERRIDES = {
    "timeomni-1-7b": {
        "engine": "vllm",
        "text_only": True,
        "system_prompt": _TIMEOMNI_SYSTEM_PROMPT,
        "sampling_params": _TIMEOMNI_SAMPLING,
    },
    "time-r1-s1p1": {
        "engine": "vllm",
        "text_only": True,
        # LLMClient default ("You are a helpful assistant.") matches their
        # preliminary inference script — no system_prompt override needed.
        "sampling_params": _TIME_R1_SAMPLING,
    },
    "time-mqa-qwen25-7b": {
        "engine": "vllm",
        "text_only": True,
        "use_chat_template": False,    # base Qwen2.5-7B has no chat template
        "prompt_wrapper": _time_mqa_wrap,
        "sampling_params": _TIME_MQA_SAMPLING,
    },
    # Stock Instruct baselines — chat template ON (default), default LLMClient
    # system prompt ("You are a helpful assistant."). No TS-specific training;
    # they read text-rendered timeseries and answer zero-shot.
    "qwen25-instruct-7b": {
        "engine": "vllm",
        "text_only": True,
        "sampling_params": _QWEN25_INSTRUCT_SAMPLING,
    },
    "llama3-instruct-8b": {
        "engine": "vllm",
        "text_only": True,
        "sampling_params": _LLAMA3_INSTRUCT_SAMPLING,
    },
    "mistral-instruct-7b-v03": {
        "engine": "vllm",
        "text_only": True,
        "sampling_params": _MISTRAL_INSTRUCT_SAMPLING,
    },
}


def _override(model_name: str, key: str, default):
    return MODEL_OVERRIDES.get(model_name, {}).get(key, default)

RESTART_EVERY = 30000  # restart LLMClient every N samples to flush vLLM multimodal cache


def _new_client(model_path, engine='vllm-ts', system_prompt='You are a helpful assistant.',
                ctx_length=None, rope_scaling=None):
    kwargs = dict(model_path=model_path, engine=engine,
                  num_gpus=NUM_GPUS, gpus_per_model=NUM_GPUS_PER_PROCESS,
                  system_prompt=system_prompt)
    if ctx_length is not None:
        kwargs["ctx_length"] = ctx_length
    if rope_scaling is not None:
        kwargs["rope_scaling"] = rope_scaling
    c = LLMClient(**kwargs)
    c.wait_for_ready()
    return c


def _render_for_text_only(questions, ts_list):
    """Inline `<ts>` placeholders with text arrays for non-multimodal models."""
    rendered = []
    for q, ts in zip(questions, ts_list):
        rendered.append(inline_timeseries_as_text(q, ts))
    return rendered


def run_model_over_datasets(model_name, model_path, datasets, ctx_length=None,
                             rope_scaling=None):
    """Run one model across multiple datasets using a single vLLM load.

    datasets: list of dicts with keys {stem, questions, ts_list, output_file}.
    Clients are restarted only when cumulative sample count crosses RESTART_EVERY
    — NOT between datasets. That's the whole point: avoid unload/reload per dataset.

    Per-model overrides (engine, system prompt, sampling, text-only rendering)
    are read from MODEL_OVERRIDES; absent entries fall back to the multi-modal
    ChatTS defaults.

    ctx_length: optional override for vLLM max_model_len. If None, uses
    LLMClient default (8192). Larger values trade KV cache memory for more
    headroom on long-prompt datasets (e.g. text-rendered timeseries baselines).
    """
    engine = _override(model_name, 'engine', 'vllm-ts')
    text_only = _override(model_name, 'text_only', False)
    system_prompt = _override(model_name, 'system_prompt', 'You are a helpful assistant.')
    model_sampling = _override(model_name, 'sampling_params', sampling_params)
    use_chat_template = _override(model_name, 'use_chat_template', True)
    prompt_wrapper = _override(model_name, 'prompt_wrapper', None)

    logger.info(
        f"[{model_name}] Loading vLLM once for {len(datasets)} dataset(s) "
        f"(engine={engine}, text_only={text_only}, "
        f"chat_template={use_chat_template}, wrapper={prompt_wrapper is not None}, "
        f"ctx_length={ctx_length})"
    )
    llm_client = _new_client(model_path, engine=engine, system_prompt=system_prompt,
                              ctx_length=ctx_length, rope_scaling=rope_scaling)
    processed_since_restart = 0

    try:
        for ds in datasets:
            stem = ds['stem']
            questions = ds['questions']
            ts_list = ds['ts_list']
            output_file = ds['output_file']
            total = len(questions)
            logger.info(f"[{model_name}/{stem}] {total} samples -> {output_file}")

            answer_dict = {}
            for start in range(0, total, CHUNK_SIZE):
                if processed_since_restart >= RESTART_EVERY:
                    logger.info(
                        f"[{model_name}] Restarting client after "
                        f"{processed_since_restart} samples (cache flush)"
                    )
                    llm_client.kill()
                    llm_client = _new_client(model_path, engine=engine,
                                             system_prompt=system_prompt,
                                             ctx_length=ctx_length,
                                             rope_scaling=rope_scaling)
                    processed_since_restart = 0

                end = min(start + CHUNK_SIZE, total)
                logger.info(f"[{model_name}/{stem}] chunk {start}-{end} / {total}")

                if text_only:
                    # Text-only baseline: bake the timeseries into the prompt
                    # text and disable multi-modal packing entirely.
                    chunk_questions = _render_for_text_only(
                        questions[start:end], ts_list[start:end]
                    )
                    if prompt_wrapper is not None:
                        chunk_questions = [prompt_wrapper(q) for q in chunk_questions]
                    chunk_ts = None
                else:
                    chunk_questions = questions[start:end]
                    chunk_ts = ts_list[start:end]

                chunk_answers = llm_client.llm_batch_generate(
                    chunk_questions, chunk_ts,
                    sampling_params=model_sampling,
                    use_chat_template=use_chat_template,
                    desc=f"{model_name}/{stem} [{start}:{end}]",
                )
                for idx, ans in enumerate(chunk_answers):
                    answer_dict[start + idx] = {"response": ans}
                processed_since_restart += (end - start)

            # Write this dataset's output before moving to the next.
            # NOTE: we deliberately keep the ORIGINAL questions[] (with
            # <ts> placeholders) in `question_text` of the on-disk JSON,
            # not the text-rendered version. The judge resolves a fixed
            # per-eval_type question and never displays the raw timeseries,
            # so storing 256-3274 floats per sample would just bloat the
            # judge's prompt and risk overflowing the 8K context.
            ds['answer_dict'] = answer_dict
            _write_output(ds)
    finally:
        llm_client.kill()


def _write_output(ds):
    """Write generated_answer.json for one dataset using the shared schema.

    Both `response` (model output) and `ground_truth` are stored as full
    content (with `<think>` block intact). The eval LLM-as-judge consumes
    them directly; no client-side stripping. See
    `memory/feedback_no_strip_think_for_judge.md` for rationale.
    """
    answer = ds['answer_dict']
    questions = ds['questions']
    gt_texts = ds['gt_texts']
    eval_types = ds['eval_types']
    eval_mds = ds['eval_mds']
    output_file = ds['output_file']

    generated = []
    for idx, ans in answer.items():
        generated.append({
            'idx': idx,
            'question_text': questions[idx],
            'response': ans['response'],
            'ground_truth': gt_texts[idx],
            'eval_type': eval_types[idx],
            'eval_metadata': eval_mds[idx],
        })

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "wt") as f:
        json.dump(generated, f, ensure_ascii=False, indent=4)
    logger.info(f"Wrote {output_file} ({len(generated)} rows)")


def _load_and_prepare(dataset_path):
    """Load a dataset and extract inference inputs + gt bookkeeping."""
    with open(dataset_path) as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError:
            f.seek(0)
            data = [json.loads(line) for line in f]

    questions, ts_list = [], []
    gt_texts = []
    eval_types, eval_mds = [], []
    for idx, sample in enumerate(data):
        questions.append(sample['input'])
        gt_texts.append(sample['output'])  # full GT (with <think>)

        eval_type = sample.get('eval_type')
        eval_md = sample.get('eval_metadata')
        if eval_type is None:
            raise ValueError(f"{dataset_path}: sample {idx} missing 'eval_type'")
        if eval_md is None:
            raise ValueError(f"{dataset_path}: sample {idx} missing 'eval_metadata'")
        if 'length' not in eval_md:
            ts_data = sample.get('timeseries', [])
            if ts_data:
                eval_md = dict(eval_md)
                eval_md['length'] = len(ts_data[0])
        eval_types.append(eval_type)
        eval_mds.append(eval_md)
        ts_list.append([np.array(item) for item in sample['timeseries']])

    return {
        'questions': questions,
        'ts_list': ts_list,
        'gt_texts': gt_texts,
        'eval_types': eval_types,
        'eval_mds': eval_mds,
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description=(
            "Run vLLM inference across one-or-more datasets and one-or-more models. "
            "Loop order is model-outer, dataset-inner — each model is loaded ONCE and "
            "reused across all datasets, avoiding unload/reload between datasets."
        ),
    )
    parser.add_argument("--models", nargs="+", choices=list(MODEL_CONFIGS.keys()),
                        default=list(MODEL_CONFIGS.keys()),
                        help="Which models to run (default: all)")
    parser.add_argument("--datasets", nargs="+", default=None,
                        help="One or more dataset paths. Default: [%(default)s]" % {"default": DATASET})
    parser.add_argument("--dataset", default=None,
                        help="(Deprecated) single-dataset alias for --datasets.")
    parser.add_argument("--chunk-size", type=int, default=CHUNK_SIZE)
    parser.add_argument("--exp-root", default="exp",
                        help="Root output dir. Results go to {exp-root}/{model}/{dataset_stem}/. Default: exp")
    parser.add_argument("--force", action="store_true",
                        help="Re-run (model, dataset) pairs whose generated_answer.json already exists.")
    parser.add_argument("--max-tokens", type=int, default=None,
                        help="Override default sampling max_tokens for our ChatTS-style models. "
                             "Per-baseline overrides (TimeOmni, Time-R1, etc.) ignore this. "
                             "Useful for re-running truncated samples at higher cap.")
    parser.add_argument("--ctx-length", type=int, default=None,
                        help="Override vLLM max_model_len for this run. Default uses LLMClient "
                             "default (8192). Use larger values for text-rendered baselines on "
                             "long-prompt datasets (e.g. the long length-OOD buckets). KV cache scales "
                             "linearly so concurrency drops; smaller is better when not needed.")
    parser.add_argument("--yarn-factor", type=float, default=None,
                        help="If set, configure YaRN rope_scaling on vLLM workers. Combine with "
                             "--ctx-length for >native context (e.g. --yarn-factor 2.0 "
                             "--ctx-length 65536 to extend a 32K-native model to 64K).")
    parser.add_argument("--yarn-original-max", type=int, default=32768,
                        help="original_max_position_embeddings for YaRN scaling (default 32768 — "
                             "Qwen2.5/Mistral-v0.3 native).")
    args = parser.parse_args()
    CHUNK_SIZE = args.chunk_size

    if args.max_tokens is not None:
        # Override the default ChatTS sampling. Don't touch baseline-specific
        # SamplingParams in MODEL_OVERRIDES — those baselines have their own
        # tested recipes.
        import synth.utils.inference_tsmllm_vllm as _self
        _self.sampling_params = SamplingParams(
            max_tokens=args.max_tokens,
            temperature=0.5,
        )
        logger.info(f"Default sampling overridden: max_tokens={args.max_tokens}")

    # Resolve datasets (accept --datasets OR legacy --dataset).
    if args.datasets:
        dataset_paths = [Path(p) for p in args.datasets]
    elif args.dataset:
        dataset_paths = [Path(args.dataset)]
    else:
        dataset_paths = [Path(DATASET)]
    for p in dataset_paths:
        if not p.exists():
            raise SystemExit(f"Dataset not found: {p}")
    logger.info(f"Datasets: {[str(p) for p in dataset_paths]}")

    # Load all datasets upfront (TS + questions + eval metadata).
    prepared: list[dict] = []
    for ds_path in dataset_paths:
        logger.info(f"Preparing {ds_path}")
        data = _load_and_prepare(ds_path)
        data['path'] = ds_path
        data['stem'] = ds_path.stem
        prepared.append(data)

    # Outer loop = model (vLLM loaded once), inner loop = dataset.
    for model_name in args.models:
        model_path = MODEL_CONFIGS[model_name]

        # Attach per-(model, dataset) output files and filter skips.
        datasets_to_run = []
        for d in prepared:
            exp_dir = os.path.join(WORKDIR, args.exp_root, model_name, d['stem'])
            output_file = os.path.join(exp_dir, "generated_answer.json")
            if os.path.exists(output_file) and not args.force:
                logger.info(
                    f"[{model_name}/{d['stem']}] SKIP — {output_file} exists (use --force)"
                )
                continue
            datasets_to_run.append({
                **d,
                'output_file': output_file,
            })

        if not datasets_to_run:
            logger.info(f"[{model_name}] nothing to do — all outputs already exist")
            continue

        logger.info(
            f"=== {model_name} ({model_path}) over "
            f"{[d['stem'] for d in datasets_to_run]} ==="
        )
        rope_scaling = None
        if args.yarn_factor is not None:
            rope_scaling = {
                "rope_type": "yarn",
                "factor": args.yarn_factor,
                "original_max_position_embeddings": args.yarn_original_max,
            }
        run_model_over_datasets(model_name, model_path, datasets_to_run,
                                ctx_length=args.ctx_length,
                                rope_scaling=rope_scaling)
