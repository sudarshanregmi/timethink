"""Merge Time-MQA's LoRA adapter into Qwen2.5-7B and save as a regular HF model.

Time-MQA (arXiv:2503.01875) is a continual-pretraining LoRA fine-tune of
Qwen2.5-7B (base, NOT Instruct) on the TSQA dataset (~200K time-series QA
pairs across 12 domains, 5 task types). The HF repo `Time-MQA/Qwen-2.5-7B`
ships ONLY the LoRA adapter (~50 MB), not full weights — so we merge.

Pipeline:
1. Download `Qwen/Qwen2.5-7B` (the unquantized base, ~15 GB).
2. Download `Time-MQA/Qwen-2.5-7B` LoRA adapter (~50 MB).
3. Apply + merge via PEFT.
4. Save merged model + Time-MQA's tokenizer to `time-mqa-qwen25-7b-ckpt/`.

After this runs once, the merged model is a regular HF checkpoint that
vLLM can load natively, just like our other text-only baselines.

Fidelity note: the adapter's `base_model_name_or_path` is
`unsloth/qwen2.5-7b-unsloth-bnb-4bit` (4-bit quantized base). We merge
into the unquantized `Qwen/Qwen2.5-7B` instead. The unsloth bnb-4bit was
a training-time memory optimization; the underlying base weights are
identical to `Qwen/Qwen2.5-7B`. The merged model is semantically
near-identical to applying the adapter on the quantized base at
inference, modulo tiny precision drift from quant-aware vs full-precision
LoRA application.
"""
from __future__ import annotations
import argparse
import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


DEFAULT_BASE = "Qwen/Qwen2.5-7B"
DEFAULT_ADAPTER = "Time-MQA/Qwen-2.5-7B"
DEFAULT_OUT = os.path.abspath("time-mqa-qwen25-7b-ckpt")


def main():
    p = argparse.ArgumentParser(
        description="Merge Time-MQA LoRA into Qwen2.5-7B (one-time)"
    )
    p.add_argument("--base", default=DEFAULT_BASE,
                   help=f"Base model HF id (default: {DEFAULT_BASE})")
    p.add_argument("--adapter", default=DEFAULT_ADAPTER,
                   help=f"LoRA adapter HF id (default: {DEFAULT_ADAPTER})")
    p.add_argument("--out_dir", default=DEFAULT_OUT,
                   help=f"Output dir for merged model (default: {DEFAULT_OUT})")
    p.add_argument("--dtype", default="bfloat16",
                   choices=["float16", "bfloat16", "float32"],
                   help="Save dtype (default: bfloat16)")
    args = p.parse_args()

    dtype = getattr(torch, args.dtype)

    print(f"[1/4] Loading base model {args.base} (dtype={args.dtype}, device=cpu)...")
    base = AutoModelForCausalLM.from_pretrained(
        args.base,
        torch_dtype=dtype,
        device_map="cpu",
        low_cpu_mem_usage=True,
    )

    print(f"[2/4] Loading LoRA adapter {args.adapter}...")
    peft_model = PeftModel.from_pretrained(base, args.adapter)

    print("[3/4] Merging adapter into base weights...")
    merged = peft_model.merge_and_unload()

    print(f"[4/4] Saving merged model to {args.out_dir}...")
    os.makedirs(args.out_dir, exist_ok=True)
    merged.save_pretrained(args.out_dir, safe_serialization=True)

    # Use Time-MQA's tokenizer (preserves their EOS / pad / no-chat-template
    # config exactly). Falls back to base if the adapter repo's tokenizer
    # is unloadable.
    print(f"  saving tokenizer (from {args.adapter})...")
    try:
        tok = AutoTokenizer.from_pretrained(args.adapter)
    except Exception as e:
        print(f"  WARN: tokenizer load from adapter failed ({e}); using base tokenizer")
        tok = AutoTokenizer.from_pretrained(args.base)
    tok.save_pretrained(args.out_dir)

    print("done")


if __name__ == "__main__":
    main()
