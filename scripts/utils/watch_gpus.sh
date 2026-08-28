#!/bin/bash
# Watch GPU 4-7 utilization, memory, and processes every 2 seconds
# Usage: bash myscripts/watch_gpus.sh [interval_seconds]

INTERVAL=${1:-2}

while true; do
    clear
    echo "=== GPU Monitor (GPU 4-7) === $(date '+%H:%M:%S')"
    echo ""
    nvidia-smi --query-gpu=index,utilization.gpu,utilization.memory,memory.used,memory.total,temperature.gpu \
        --format=csv,noheader --id=4,5,6,7
    echo ""
    echo "--- Processes on GPU 4-7 ---"
    nvidia-smi --query-compute-apps=pid,gpu_uuid,used_memory,name \
        --format=csv,noheader 2>/dev/null | while read line; do
        # Filter to only GPU 4-7
        for gpu_id in 4 5 6 7; do
            gpu_uuid=$(nvidia-smi --query-gpu=uuid --format=csv,noheader --id=$gpu_id 2>/dev/null)
            if echo "$line" | grep -q "$gpu_uuid"; then
                pid=$(echo "$line" | cut -d',' -f1 | tr -d ' ')
                mem=$(echo "$line" | cut -d',' -f3)
                echo "  GPU $gpu_id | PID $pid | Mem $mem"
            fi
        done
    done
    sleep $INTERVAL
done
