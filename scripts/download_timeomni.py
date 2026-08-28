"""Download the oldest TimeOmni-1 checkpoint (`anton-hugging/TimeOmni-1-7B`).

Layout mirrors the other model dirs in this repo: a top-level `timeomni-1-7b-ckpt/`
holding the Hugging Face snapshot. Default is the original Feb-2026 release (Qwen2.5
-Instruct based) — NOT the Apr-2026 Qwen3.5 4B/9B versions.
"""
from __future__ import annotations
import argparse
import os
from huggingface_hub import snapshot_download


DEFAULT_REPO = "anton-hugging/TimeOmni-1-7B"
DEFAULT_OUT = os.path.abspath("timeomni-1-7b-ckpt")


def main():
    p = argparse.ArgumentParser(description="Download TimeOmni-1 baseline checkpoint")
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
