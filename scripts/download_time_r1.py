"""Download the oldest Time-R1 checkpoint (`ulab-ai/Time-R1-S1P1`).

Layout mirrors `scripts/download_timeomni.py`: a top-level `time-r1-s1p1-ckpt/`
holding the Hugging Face snapshot. Time-R1-S1P1 is the Stage-1 / Phase-1
checkpoint (the earliest released artifact in the Time-R1 collection),
fine-tuned from Qwen2.5-3B-Instruct.
"""
from __future__ import annotations
import argparse
import os
from huggingface_hub import snapshot_download


DEFAULT_REPO = "ulab-ai/Time-R1-S1P1"
DEFAULT_OUT = os.path.abspath("time-r1-s1p1-ckpt")


def main():
    p = argparse.ArgumentParser(description="Download Time-R1 baseline checkpoint")
    p.add_argument("--repo", default=DEFAULT_REPO,
                   help=f"HF repo id (default: {DEFAULT_REPO})")
    p.add_argument("--out_dir", default=DEFAULT_OUT,
                   help=f"Local destination (default: {DEFAULT_OUT})")
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    print(f"Downloading {args.repo} -> {args.out_dir}")
    snapshot_download(
        repo_id=args.repo,
        local_dir=args.out_dir,
        local_dir_use_symlinks=False,
    )
    print("done")


if __name__ == "__main__":
    main()
