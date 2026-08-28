"""
Measure actual tokenized prompt and response lengths using the real processor/tokenizer.
This gives exact token counts (not estimates) including chat template overhead.

Usage: python myscripts/measure_token_lengths.py [--max_samples 500]
"""
import argparse
import json
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoProcessor

MODEL_PATH = "/scratch/sudarshan/grpo/outputs/sft/global_step_153/huggingface"


def load_processor_and_tokenizer():
    print(f"Loading processor from {MODEL_PATH}...")
    processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    return processor, tokenizer


def measure_prompt_length(row, processor):
    """Measure actual tokenized prompt length including chat template + timeseries."""
    messages = row["prompt"]
    if isinstance(messages, np.ndarray):
        messages = messages.tolist()

    # timeseries is an object-dtype ndarray of 1D arrays
    ts_raw = row["timeseries"]
    ts_data = []
    if ts_raw is not None:
        if isinstance(ts_raw, np.ndarray) and ts_raw.dtype == object:
            ts_data = [np.array(ts, dtype=np.float32) for ts in ts_raw]
        elif isinstance(ts_raw, np.ndarray) and ts_raw.ndim == 1:
            ts_data = [ts_raw.astype(np.float32)]
        elif isinstance(ts_raw, list):
            ts_data = [np.array(ts, dtype=np.float32) for ts in ts_raw]

    raw_prompt = processor.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False
    )
    inputs = processor(text=[raw_prompt], timeseries=ts_data, return_tensors="pt")
    return inputs["input_ids"].shape[-1]


def measure_response_length(row, tokenizer):
    """Measure response token length."""
    response = None
    if "response" in row and row["response"]:
        response = row["response"]
    elif "reward_model" in row and row["reward_model"]:
        rm = row["reward_model"]
        if isinstance(rm, str):
            rm = json.loads(rm)
        response = rm.get("ground_truth", "")

    if not response:
        return 0
    return len(tokenizer.encode(response, add_special_tokens=False))


def analyze_lengths(lengths, name):
    arr = np.array(lengths)
    print(f"\n{name}:")
    print(f"  count: {len(arr)}")
    print(f"  mean:  {arr.mean():.0f}")
    print(f"  std:   {arr.std():.0f}")
    print(f"  min:   {arr.min():.0f}")
    print(f"  max:   {arr.max():.0f}")
    for p in [25, 50, 75, 90, 95, 99, 100]:
        print(f"  p{p:>3}:  {np.percentile(arr, p):.0f}")
    return arr


def suggest_max_length(prompt_lens, response_lens):
    print("\n" + "=" * 60)
    print("  Recommendations")
    print("=" * 60)
    for max_prompt in [512, 1024, 2048, 4096, 8192]:
        surviving_mask = np.array(prompt_lens) <= max_prompt
        n_surviving = surviving_mask.sum()
        pct = n_surviving / len(prompt_lens) * 100
        for max_resp in [1024, 2048, 4096]:
            total = max_prompt + max_resp
            print(f"  max_prompt={max_prompt:>5}, max_resp={max_resp:>5} "
                  f"(total={total:>5}): {n_surviving}/{len(prompt_lens)} "
                  f"({pct:.1f}%) prompts kept")
        print()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_samples", type=int, default=500,
                        help="Max samples to measure (random subset). Use -1 for all.")
    parser.add_argument("--data", default="data/train_rl.parquet")
    args = parser.parse_args()

    processor, tokenizer = load_processor_and_tokenizer()

    print(f"\nLoading data from {args.data}...")
    df = pd.read_parquet(args.data)
    print(f"Total samples: {len(df)}")

    if args.max_samples > 0 and args.max_samples < len(df):
        df = df.sample(n=args.max_samples, random_state=42).reset_index(drop=True)
        print(f"Sampled {args.max_samples} for measurement")

    prompt_lens = []
    response_lens = []
    errors = 0

    for i in range(len(df)):
        try:
            row = df.iloc[i]
            pl = measure_prompt_length(row, processor)
            prompt_lens.append(pl)

            rl = measure_response_length(row, tokenizer)
            response_lens.append(rl)

            if (i + 1) % 100 == 0:
                print(f"  Measured {i + 1}/{len(df)}...")
        except Exception as e:
            errors += 1
            if errors <= 3:
                print(f"  Error on sample {i}: {e}")

    if errors:
        print(f"\n{errors} samples had errors")

    prompt_arr = analyze_lengths(prompt_lens, "Prompt token lengths (with chat template + timeseries)")
    response_arr = analyze_lengths(response_lens, "Response token lengths")

    total_arr = prompt_arr + response_arr[:len(prompt_arr)]
    analyze_lengths(total_arr.tolist(), "Total (prompt + response) token lengths")

    suggest_max_length(prompt_lens, response_lens)


if __name__ == "__main__":
    main()
