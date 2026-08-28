#!/bin/bash
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
set -x

# OOM smoke test for sft_nothink_fair.
#
# Trains for one epoch on the top-256 worst-case (longest-context) rows from
# train_sft_nothink_fair.parquet, at a candidate MICRO_BATCH_SIZE_PER_GPU.
# Surviving a handful of forward+backward passes on max_length=8192 sequences
# without OOM means that micro_batch is safe for the real ~46h run.
#
# Build the subset first:
#   python scripts/utils/build_oom_smoke_subset.py
#
# Usage:
#   bash scripts/sft_nothink_fair_smoke.sh 16    # try micro_batch=16
#   bash scripts/sft_nothink_fair_smoke.sh 32    # try micro_batch=32
#   bash scripts/sft_nothink_fair_smoke.sh 8     # safety check at current+1
#
# train_batch_size is set to micro_batch * N_GPUS so each step = exactly one
# micro batch per GPU (no gradient accumulation hiding peak memory).
# Peak GPU memory is determined by a single micro batch; if it OOMs at step 1
# or 2, the candidate is too aggressive.

MICRO_BATCH=${1:-4}

N_GPUS=4
SAVE_PATH="outputs/sft_nothink_fair_smoke_mb${MICRO_BATCH}"
TRAIN_FILE=$REPO/data/train_oom_smoke.parquet
TEST_FILE=$REPO/data/train_oom_smoke.parquet
MODEL_PATH="$REPO/ickpt"
TRAIN_BATCH_SIZE=$((MICRO_BATCH * N_GPUS))
# val_max_samples must be >= TRAIN_BATCH_SIZE so the val dataloader yields at
# least one full global batch (drop_last=True). Earlier value of 8 caused
# `RuntimeError: stack expects a non-empty TensorList` when MICRO_BATCH * N_GPUS > 8.
VAL_MAX_SAMPLES=$((TRAIN_BATCH_SIZE * 2))
MAX_LENGTH=8192

mkdir -p "$SAVE_PATH"

torchrun --standalone --nnodes=1 --nproc_per_node=$N_GPUS \
    -m verl.trainer.fsdp_sft_trainer \
    data.val_max_samples=$VAL_MAX_SAMPLES \
    data.train_files=$TRAIN_FILE \
    data.val_files=$TEST_FILE \
    data.dataloader_num_workers=0 \
    data.prompt_key=prompt \
    data.response_key=response \
    data.micro_batch_size_per_gpu=$MICRO_BATCH \
    data.train_batch_size=$TRAIN_BATCH_SIZE \
    data.max_length=$MAX_LENGTH \
    data.truncation='right' \
    data.chat_template=null \
    data.custom_cls.path="$REPO/ts_sft_data.py" \
    data.custom_cls.name=TSSFTDataset \
    model.partial_pretrain=$MODEL_PATH \
    model.trust_remote_code=True \
    trainer.default_local_dir="$SAVE_PATH" \
    trainer.project_name=timethink \
    trainer.experiment_name="sft_nothink_fair_smoke_mb${MICRO_BATCH}" \
    trainer.logger='["console"]' \
    trainer.total_epochs=1 \
    trainer.save_freq=10000 \
    trainer.test_freq=10000 \
    trainer.max_ckpt_to_keep=1 \
    optim.lr=1e-5 \
    optim.weight_decay=0.01 \
    ulysses_sequence_parallel_size=1 \
    use_remove_padding=true
