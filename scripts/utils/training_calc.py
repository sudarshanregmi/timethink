"""
Quick calculator for training step counts and config sanity checks.
Usage: python myscripts/training_calc.py
"""
import pandas as pd


def main():
    # ── Data counts ──
    files = {
        "train_sft": "data/train_sft.parquet",
        "train_rl": "data/train_rl.parquet",
        "val_sft": "data/val_sft.parquet",
        "val_rl": "data/val_rl.parquet",
    }
    counts = {}
    for name, path in files.items():
        try:
            df = pd.read_parquet(path, columns=["data_source"])  # minimal read
            counts[name] = len(df)
            print(f"{name}: {len(df)} samples")
        except Exception as e:
            print(f"{name}: ERROR - {e}")

    print()

    # ── SFT config ──
    sft_batch = 192
    sft_micro = 4
    sft_gpus = 4
    sft_epochs = 3
    sft_samples = counts.get("train_sft", 0)
    sft_steps_per_epoch = sft_samples // sft_batch if sft_batch else 0
    sft_total_steps = sft_steps_per_epoch * sft_epochs

    print("=== SFT Training ===")
    print(f"  samples: {sft_samples}")
    print(f"  train_batch_size: {sft_batch}")
    print(f"  micro_batch_size_per_gpu: {sft_micro}")
    print(f"  gpus: {sft_gpus}")
    print(f"  grad_accum_steps: {sft_batch // (sft_micro * sft_gpus)}")
    print(f"  epochs: {sft_epochs}")
    print(f"  steps/epoch: {sft_steps_per_epoch}")
    print(f"  total steps: {sft_total_steps}")
    print()

    # ── RL (GRPO) config ──
    rl_batch = 256
    rl_n = 16  # rollouts per prompt
    rl_epochs = 1
    rl_gpus = 4
    rl_micro = 64
    rl_samples = counts.get("train_rl", 0)
    rl_unique_prompts_per_batch = rl_batch // rl_n
    rl_steps_per_epoch = rl_samples // rl_batch if rl_batch else 0
    rl_total_steps = rl_steps_per_epoch * rl_epochs

    print("=== RL (GRPO) Training ===")
    print(f"  samples: {rl_samples}")
    print(f"  train_batch_size: {rl_batch}")
    print(f"  rollout n: {rl_n}")
    print(f"  unique prompts/batch: {rl_unique_prompts_per_batch}")
    print(f"  ppo_micro_batch_size_per_gpu: {rl_micro}")
    print(f"  gpus: {rl_gpus}")
    print(f"  epochs: {rl_epochs}")
    print(f"  steps/epoch: {rl_steps_per_epoch}")
    print(f"  total steps: {rl_total_steps}")
    print(f"  max_prompt_length: 512")
    print(f"  max_response_length: 2048")
    print(f"  val samples: {counts.get('test', 0)} (all)")
    print(f"  test_freq: 10")
    print(f"  save_freq: 100")
    print(f"  validations during training: {rl_total_steps // 10}")
    print(f"  checkpoints saved: {rl_total_steps // 100}")


if __name__ == "__main__":
    main()
