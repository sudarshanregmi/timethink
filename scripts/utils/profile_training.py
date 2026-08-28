"""
Profile GPU utilization during verl GRPO training on GPU 4-7.
Logs GPU stats every N seconds, identifies training phases (rollout, actor update, ref forward),
and produces a summary of time and utilization per phase.

Usage:
    python myscripts/profile_training.py --duration 600 --interval 1
    python myscripts/profile_training.py --duration 0  # run until ctrl+c

While this runs, run rl.sh in another terminal.
PIDs for GPU 4-7 are auto-detected.
"""
import argparse
import csv
import os
import re
import subprocess
import signal
import sys
import time
from collections import defaultdict
from datetime import datetime


GPU_IDS = [4, 5, 6, 7]
LOG_FILE = "myscripts/gpu_profile_log.csv"


def get_gpu_stats():
    """Get utilization and memory for GPU 4-7."""
    result = subprocess.run(
        ["nvidia-smi",
         "--query-gpu=index,utilization.gpu,memory.used,memory.total,memory.free",
         "--format=csv,noheader,nounits",
         f"--id={','.join(map(str, GPU_IDS))}"],
        capture_output=True, text=True
    )
    stats = []
    for line in result.stdout.strip().split("\n"):
        if not line.strip():
            continue
        parts = [p.strip() for p in line.split(",")]
        stats.append({
            "gpu_id": int(parts[0]),
            "util_pct": int(parts[1]),
            "mem_used_mb": int(parts[2]),
            "mem_total_mb": int(parts[3]),
            "mem_free_mb": int(parts[4]),
        })
    return stats


def get_verl_pids():
    """Find PIDs running on GPU 4-7."""
    result = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid,gpu_bus_id,used_memory",
         "--format=csv,noheader,nounits"],
        capture_output=True, text=True
    )
    # Map GPU bus IDs to our GPU indices
    bus_result = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,gpu_bus_id",
         "--format=csv,noheader",
         f"--id={','.join(map(str, GPU_IDS))}"],
        capture_output=True, text=True
    )
    bus_to_gpu = {}
    for line in bus_result.stdout.strip().split("\n"):
        if not line.strip():
            continue
        parts = [p.strip() for p in line.split(",")]
        bus_to_gpu[parts[1]] = int(parts[0])

    pids = {}
    for line in result.stdout.strip().split("\n"):
        if not line.strip():
            continue
        parts = [p.strip() for p in line.split(",")]
        pid = int(parts[0])
        bus_id = parts[1]
        if bus_id in bus_to_gpu:
            gpu_id = bus_to_gpu[bus_id]
            mem = int(parts[2])
            pids[pid] = {"gpu_id": gpu_id, "mem_mb": mem}
    return pids


def detect_phase_from_logs(pids):
    """Try to detect current training phase from process CPU usage patterns."""
    # Heuristic: check /proc/PID/stat for CPU time changes
    # High GPU util on all 4 = actor update or ref forward
    # High GPU util on individual GPUs = vLLM rollout (TP=1, 4 independent workers)
    pass


def classify_phase(gpu_stats):
    """
    Classify training phase based on GPU utilization patterns.

    Heuristics:
    - All GPUs high util (>50%): actor update (FSDP backward) or ref forward
    - Mixed util (some high, some low): vLLM rollout (independent workers)
    - All GPUs low util (<10%): idle / data loading / reward computation
    - High memory, low util: vLLM prefill waiting or memory allocation
    """
    utils = [s["util_pct"] for s in gpu_stats]
    avg_util = sum(utils) / len(utils)
    max_util = max(utils)
    min_util = min(utils)
    spread = max_util - min_util

    mems = [s["mem_used_mb"] for s in gpu_stats]
    avg_mem = sum(mems) / len(mems)

    if avg_util < 5:
        return "idle/cpu-work"
    elif avg_util > 50 and spread < 30:
        return "actor-update/ref-fwd"  # all GPUs busy uniformly = FSDP
    elif max_util > 30 and spread > 40:
        return "vllm-rollout"  # uneven = independent vLLM workers
    elif avg_util > 20:
        return "mixed-compute"
    else:
        return "low-activity"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--duration", type=int, default=0,
                        help="Seconds to profile (0 = until ctrl+c)")
    parser.add_argument("--interval", type=float, default=1.0,
                        help="Sampling interval in seconds")
    parser.add_argument("--output", default=LOG_FILE,
                        help="CSV output file")
    args = parser.parse_args()

    print(f"Profiling GPU {GPU_IDS} every {args.interval}s")
    print(f"Logging to {args.output}")
    print("Press Ctrl+C to stop and see summary\n")

    records = []
    phase_times = defaultdict(float)
    phase_util_sum = defaultdict(lambda: defaultdict(float))
    phase_count = defaultdict(int)

    stop = [False]
    def handler(sig, frame):
        stop[0] = True
    signal.signal(signal.SIGINT, handler)

    with open(args.output, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "timestamp", "elapsed_s",
            "gpu4_util", "gpu5_util", "gpu6_util", "gpu7_util",
            "gpu4_mem_gb", "gpu5_mem_gb", "gpu6_mem_gb", "gpu7_mem_gb",
            "avg_util", "phase"
        ])

        start = time.time()
        sample_count = 0

        while not stop[0]:
            elapsed = time.time() - start
            if args.duration > 0 and elapsed > args.duration:
                break

            try:
                stats = get_gpu_stats()
            except Exception as e:
                print(f"Error: {e}")
                time.sleep(args.interval)
                continue

            if len(stats) != 4:
                time.sleep(args.interval)
                continue

            phase = classify_phase(stats)
            utils = [s["util_pct"] for s in stats]
            mems = [s["mem_used_mb"] / 1024 for s in stats]
            avg_util = sum(utils) / len(utils)

            ts = datetime.now().strftime("%H:%M:%S")
            writer.writerow([
                ts, f"{elapsed:.1f}",
                *utils, *[f"{m:.1f}" for m in mems],
                f"{avg_util:.1f}", phase
            ])
            f.flush()

            # Accumulate phase stats
            phase_times[phase] += args.interval
            phase_count[phase] += 1
            for i, gpu_id in enumerate(GPU_IDS):
                phase_util_sum[phase][gpu_id] += utils[i]

            # Print live
            bar = "".join(f"G{GPU_IDS[i]}:{utils[i]:>3}%" for i in range(4))
            mem_bar = "".join(f" {mems[i]:.0f}G" for i in range(4))
            print(f"\r[{ts}] {bar} |{mem_bar} | {phase:<25}", end="", flush=True)

            sample_count += 1
            time.sleep(args.interval)

    # Summary
    print("\n\n" + "=" * 70)
    print("  PROFILING SUMMARY")
    print("=" * 70)
    total_time = sum(phase_times.values())
    if total_time == 0:
        print("No data collected.")
        return

    print(f"\nTotal profiled: {total_time:.0f}s ({sample_count} samples)\n")
    print(f"{'Phase':<25} {'Time(s)':>8} {'%Time':>7} {'AvgUtil':>8} {'G4':>5} {'G5':>5} {'G6':>5} {'G7':>5}")
    print("-" * 70)

    for phase in sorted(phase_times, key=phase_times.get, reverse=True):
        t = phase_times[phase]
        pct = t / total_time * 100
        n = phase_count[phase]
        gpu_avgs = {
            gid: phase_util_sum[phase][gid] / n if n > 0 else 0
            for gid in GPU_IDS
        }
        overall_avg = sum(gpu_avgs.values()) / len(GPU_IDS)
        print(f"{phase:<25} {t:>7.0f}s {pct:>6.1f}% {overall_avg:>7.1f}% "
              f"{gpu_avgs[4]:>4.0f}% {gpu_avgs[5]:>4.0f}% {gpu_avgs[6]:>4.0f}% {gpu_avgs[7]:>4.0f}%")

    print("\n" + "=" * 70)
    print(f"Log saved to: {args.output}")
    print("Analyze with: python -c \"import pandas as pd; df=pd.read_csv('{args.output}'); print(df.groupby('phase')['avg_util'].describe())\"")


if __name__ == "__main__":
    main()
