"""Download an OpenTSLM checkpoint into the local HF cache.

Default = `OpenTSLM/llama-3.2-3b-tsqa-sp`: largest Llama-3.2 (3B) base + TSQA
(Time Series Question Answering) head + Soft-Prompt projector. TSQA is the
most general OpenTSLM variant — multi-choice TS questions over varied
domains, rather than the medical-specialized HAR / Sleep / ECG variants.

OpenTSLM models load via `OpenTSLM.load_pretrained(repo_id)`, which uses the
HF cache directly. There's no need for a top-level `*-ckpt/` symlink dir
like the vLLM models. This script is a thin wrapper that pre-warms the
cache so the first run of `inference_opentslm.py` doesn't pay the download
inside the eval loop.
"""
from __future__ import annotations
import argparse
import os
from huggingface_hub import snapshot_download


DEFAULT_REPO = "OpenTSLM/llama-3.2-3b-tsqa-sp"


def main():
    p = argparse.ArgumentParser(description="Pre-warm HF cache for an OpenTSLM model")
    p.add_argument("--repo", default=DEFAULT_REPO,
                   help=f"HF repo id (default: {DEFAULT_REPO})")
    p.add_argument("--cache_dir", default=None,
                   help="Override HF cache dir (default: ~/.cache/huggingface)")
    args = p.parse_args()

    print(f"Pre-warming HF cache for {args.repo}")
    snapshot_download(repo_id=args.repo, cache_dir=args.cache_dir)
    print("done")


if __name__ == "__main__":
    main()
