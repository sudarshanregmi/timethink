"""Build the TimeThink base checkpoint: a fresh Qwen3-8B backbone plus the
time-series encoder ported from ChatTS-8B.

The ready-made result of this script is published at
    https://huggingface.co/sudarshanregmi/timethink-base
so most users can skip this and simply point training at that checkpoint. This
script documents how that base was constructed.

Recipe
------
TimeThink's model is a ``Qwen3TSForCausalLM`` = a Qwen3-8B language-model
backbone plus a lightweight time-series encoder (``ts_encoder``: an MLP
patch-embedder + a position embedding). The backbone is a *fresh* base Qwen3-8B;
only the ``ts_encoder`` weights are taken from ChatTS-8B
(``bytedance-research/ChatTS-8B``), which shares the exact same ``qwen3ts``
architecture and encoder config (hidden_size 4096), so the encoder transfers by
shape. The LLM backbone is NOT taken from ChatTS.

Steps
-----
1. Prepare the TARGET: instantiate ``Qwen3TSForCausalLM`` from the TimeThink
   config (see ``ickpt/config.json``) and load base Qwen3-8B weights into its
   language-model backbone; the ``ts_encoder`` is left freshly initialized.
   (Any Qwen3-8B checkpoint wrapped in this architecture works as the target.)
2. Run this script to copy ChatTS-8B's ``ts_encoder.*`` weights into the target,
   overwriting the fresh encoder. Shapes are verified before every copy.
3. The output is the untrained base that SFT trains from.

Usage
-----
    python scripts/build_base_model.py \
        --source bytedance-research/ChatTS-8B \
        --target ./qwen3_8b_ts \
        --output ./ickpt
"""
import argparse
import gc

from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--source", default="bytedance-research/ChatTS-8B",
                    help="Model providing the ts_encoder weights (HF id or path).")
    ap.add_argument("--target", required=True,
                    help="Qwen3TS model with a fresh Qwen3-8B backbone (the destination).")
    ap.add_argument("--output", required=True, help="Where to write the base checkpoint.")
    ap.add_argument("--key", default="ts_encoder",
                    help="Parameter-name substring identifying the encoder weights.")
    args = ap.parse_args()

    print(f"[1/4] loading source (encoder donor): {args.source}")
    src = AutoModelForCausalLM.from_pretrained(
        args.source, device_map="cpu", torch_dtype="auto", trust_remote_code=True)
    enc = {k: v.clone() for k, v in src.state_dict().items() if args.key in k}
    print(f"      extracted {len(enc)} '{args.key}' tensors")
    del src
    gc.collect()
    if not enc:
        raise SystemExit(f"no '{args.key}' tensors found in source")

    print(f"[2/4] loading target (fresh Qwen3-8B backbone): {args.target}")
    dst = AutoModelForCausalLM.from_pretrained(
        args.target, device_map="cpu", torch_dtype="auto", trust_remote_code=True)
    sd = dst.state_dict()

    print("[3/4] transplanting encoder (shape-checked)")
    n = 0
    for k, w in enc.items():
        if k not in sd:
            raise SystemExit(f"key {k} absent from target architecture")
        if sd[k].shape != w.shape:
            raise SystemExit(
                f"shape mismatch {k}: source {tuple(w.shape)} vs target {tuple(sd[k].shape)}")
        sd[k].copy_(w)
        n += 1
    print(f"      transferred {n}/{len(enc)} tensors")

    print(f"[4/4] saving base -> {args.output}")
    dst.save_pretrained(args.output, max_shard_size="5GB")
    AutoTokenizer.from_pretrained(args.target, trust_remote_code=True).save_pretrained(args.output)
    print("done")


if __name__ == "__main__":
    main()
