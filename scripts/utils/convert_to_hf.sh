# Run from the root of the verl repo
python -m verl.model_merger merge --trust-remote-code \
    --backend fsdp \
    --local_dir $1 \
    --target_dir $2
