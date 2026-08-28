"""Download stock Instruct-tuned baseline models.

These are off-the-shelf Instruct fine-tunes (no domain-specific TS training)
used as zero-shot reference points alongside our fine-tuned models and the
TS-trained third-party baselines (TimeOmni, Time-R1, Time-MQA, OpenTSLM).

Layout matches the rest of our baselines: snapshot to top-level `<key>-ckpt/`
so vLLM can load the local dir directly.

Default downloads ALL three. Use `--models` to pick a subset.

Note on gating:
- `meta-llama/Meta-Llama-3-8B-Instruct` and `mistralai/Mistral-7B-Instruct-v0.3`
  require Hugging Face access approval. If `snapshot_download` returns 403,
  visit the model page on HF, request access, then re-run after approval.
- `huggingface-cli login` (read token) is required for gated repos.
"""
from __future__ import annotations
import argparse
import os
from huggingface_hub import snapshot_download


# (key, hf_repo, output_dir_name)
BASELINES = {
    "qwen25-instruct-7b":      "Qwen/Qwen2.5-7B-Instruct",
    "llama3-instruct-8b":      "meta-llama/Meta-Llama-3-8B-Instruct",
    "mistral-instruct-7b-v03": "mistralai/Mistral-7B-Instruct-v0.3",
}


def main():
    p = argparse.ArgumentParser(description="Download stock Instruct baselines")
    p.add_argument("--models", nargs="+", choices=list(BASELINES.keys()),
                   default=list(BASELINES.keys()),
                   help="Which baselines to download (default: all)")
    p.add_argument("--out_root", default=os.path.abspath("."),
                   help="Parent dir for {key}-ckpt/ snapshots (default: cwd)")
    args = p.parse_args()

    for key in args.models:
        repo = BASELINES[key]
        out = os.path.join(args.out_root, f"{key}-ckpt")
        os.makedirs(out, exist_ok=True)
        print(f"=== {key} :: {repo} -> {out} ===")
        try:
            snapshot_download(
                repo_id=repo,
                local_dir=out,
                local_dir_use_symlinks=False,
            )
            print(f"    done")
        except Exception as e:
            print(f"    FAILED: {type(e).__name__}: {e}")
            print(
                f"    hint: {repo} may require HF approval — visit "
                f"https://huggingface.co/{repo} and request access, then "
                f"`huggingface-cli login` and re-run."
            )

    print("all done")


if __name__ == "__main__":
    main()
