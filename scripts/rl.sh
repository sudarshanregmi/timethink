#!/bin/bash
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
set -x

# Resolve SFT checkpoint. Override precedence: SFT_CKPT > (SFT_DIR + SFT_STEP) > (SFT_DIR + latest_checkpointed_iteration.txt)
SFT_DIR="${SFT_DIR:-$REPO/outputs/sft_4gpu}"
if [[ -z "${SFT_CKPT:-}" ]]; then
    SFT_STEP="${SFT_STEP:-$(cat "$SFT_DIR/latest_checkpointed_iteration.txt" 2>/dev/null)}"
    [[ -n "$SFT_STEP" ]] || { echo "error: no SFT_CKPT set and no $SFT_DIR/latest_checkpointed_iteration.txt" >&2; exit 1; }
    SFT_CKPT="$SFT_DIR/global_step_${SFT_STEP}/huggingface"
fi
[[ -d "$SFT_CKPT" ]] || { echo "error: SFT checkpoint dir not found: $SFT_CKPT" >&2; exit 1; }
echo "Using SFT checkpoint: $SFT_CKPT"

PER_GPU=8
SAVE_PATH="${SAVE_PATH:-$REPO/outputs/rl_4gpu}"

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=$REPO/data/train_rl.parquet \
    data.val_files=$REPO/data/val_rl.parquet \
    data.val_max_samples=1000 \
    data.filter_overlong_prompts=True \
    data.train_batch_size=256 \
    data.max_prompt_length=8192 \
    data.max_response_length=2048 \
    data.truncation='error' \
    data.dataloader_num_workers=4 \
    data.trust_remote_code=True \
    data.custom_cls.path="$REPO/ts_data.py" \
    data.custom_cls.name='TSRLHFDataset' \
    actor_rollout_ref.model.path=${SFT_CKPT} \
    actor_rollout_ref.model.trust_remote_code=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=256 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${PER_GPU} \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=${PER_GPU} \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.n=16 \
    actor_rollout_ref.rollout.dtype=bfloat16 \
    actor_rollout_ref.rollout.stop='["</think>"]' \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=${PER_GPU} \
    actor_rollout_ref.ref.fsdp_config.param_offload=False \
    algorithm.use_kl_in_reward=False \
    reward_model.enable=False \
    custom_reward_function.path="$REPO/reward/__init__.py" \
    custom_reward_function.name='compute_score' \
    trainer.max_actor_ckpt_to_keep=1 \
    trainer.max_critic_ckpt_to_keep=1 \
    trainer.critic_warmup=0 \
    trainer.project_name=timethink \
    trainer.experiment_name=rl_4gpu \
    trainer.logger='["console","wandb"]' \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    trainer.save_freq=50 \
    trainer.test_freq=50 \
    trainer.log_val_generations=10 \
    trainer.default_local_dir=$SAVE_PATH \
    trainer.validation_data_dir=$SAVE_PATH/val_dir \
    trainer.total_epochs=1 $@
