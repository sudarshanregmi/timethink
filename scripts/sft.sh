#!/bin/bash
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
set -x

N_GPUS=4
SAVE_PATH="outputs/sft_4gpu"
TRAIN_FILE=$REPO/data/train_sft.parquet
TEST_FILE=$REPO/data/val_sft.parquet
MODEL_PATH="$REPO/ickpt"
MICRO_BATCH_SIZE_PER_GPU=4
TRAIN_BATCH_SIZE=192
MAX_LENGTH=8192

torchrun --standalone --nnodes=1 --nproc_per_node=$N_GPUS \
    -m verl.trainer.fsdp_sft_trainer \
    data.val_max_samples=500 \
    data.train_files=$TRAIN_FILE \
    data.val_files=$TEST_FILE \
    data.dataloader_num_workers=4 \
    data.prompt_key=prompt \
    data.response_key=response \
    data.micro_batch_size_per_gpu=$MICRO_BATCH_SIZE_PER_GPU \
    data.train_batch_size=$TRAIN_BATCH_SIZE \
    data.max_length=$MAX_LENGTH \
    data.truncation='error' \
    data.chat_template=null \
    data.custom_cls.path="$REPO/ts_sft_data.py" \
    data.custom_cls.name=TSSFTDataset \
    model.partial_pretrain=$MODEL_PATH \
    model.trust_remote_code=True \
    trainer.default_local_dir=$SAVE_PATH \
    trainer.project_name=timethink \
    trainer.experiment_name=sft_4gpu \
    trainer.logger='["console","wandb"]' \
    trainer.total_epochs=3 \
    trainer.save_freq=200 \
    trainer.test_freq=200 \
    trainer.max_ckpt_to_keep=1 \
    optim.lr=1e-5 \
    optim.weight_decay=0.01 \
    ulysses_sequence_parallel_size=1 \
    use_remove_padding=true
