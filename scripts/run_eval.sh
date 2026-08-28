#!/usr/bin/env bash
# End-to-end evaluation wrapper.
#
# For every (model, dataset) in the matrix:
#   Phase 1 (inference): run the 7B TS model in vLLM  -> exp/{model}/{dataset}/generated_answer.json
#   Phase 2 (judge):     launch 72B judge on :8000, then score each pair
#                        -> exp/{model}/{dataset}/evaluation_report_{optimized.csv, summary.txt}
#   Phase 3 (hack-check): compare SFT vs RL with scripts/utils/check_reward_hacking.py
#
# Loop order inside Phase 1: model-outer / dataset-inner (vLLM loaded ONCE per model).
# Phase 2 detects NaN-contaminated eval dirs (previous judge-down runs) and re-scores
# them using the existing generated_answer.json, without re-running inference.
#
# Usage:
#   bash scripts/run_eval.sh                       # run everything
#   bash scripts/run_eval.sh test test_len_near      # only named datasets
#   SKIP_INFERENCE=1 bash scripts/run_eval.sh      # re-score only (inputs must exist)
#   SKIP_JUDGE=1 bash scripts/run_eval.sh          # inference only
#   FORCE_EVAL=1 bash scripts/run_eval.sh          # re-score everything (ignore validity check)
#   MODELS="rl" bash scripts/run_eval.sh           # override model list
#   MODELS="timeomni-1-7b" bash scripts/run_eval.sh   # text-only baseline (Qwen2.5-Instruct
#                                                  # fine-tune); needs `python scripts/download_timeomni.py`
#                                                  # first to populate timeomni-1-7b-ckpt/
#   MODELS="time-r1-s1p1" bash scripts/run_eval.sh    # text-only baseline (Qwen2.5-3B-Instruct
#                                                  # fine-tune, oldest checkpoint); needs
#                                                  # `python scripts/download_time_r1.py` first.
#   MODELS="opentslm-tsqa-3b" bash scripts/run_eval.sh   # multi-modal baseline (OpenTSLM:
#                                                  # Llama-3.2-3B + TS encoder + Soft-Prompt;
#                                                  # TSQA head); needs opentslm pip pkg + HF
#                                                  # cache pre-warm (`python scripts/download_opentslm.py`).
#                                                  # Dispatched to a separate HF-based runner
#                                                  # (`synth.utils.inference_opentslm`); not vLLM.
#   MODELS="time-mqa-qwen25-7b" bash scripts/run_eval.sh # text-only baseline (Time-MQA LoRA on
#                                                  # base Qwen2.5-7B, TSQA continual pretraining,
#                                                  # arXiv:2503.01875). Needs `python
#                                                  # scripts/download_time_mqa.py` first to merge
#                                                  # the LoRA into the base and write the
#                                                  # time-mqa-qwen25-7b-ckpt/ snapshot.
#                                                  # NO chat template; completion-style with
#                                                  # `Question: ... \nAnswer:` wrapping.
#   MODELS="qwen25-instruct-7b llama3-instruct-8b mistral-instruct-7b-v03" \
#       bash scripts/run_eval.sh                   # stock Instruct zero-shot baselines (no TS
#                                                  # training). Need `python
#                                                  # scripts/download_instruct_baselines.py`
#                                                  # first. Llama-3 + Mistral are HF-gated.

set -euo pipefail

cd "$(dirname "$0")/.."

# ── Configuration ─────────────────────────────────────────────────────────
ALL_DATASETS=(test test_len_near test_len_far test_len_extreme)
SYNTHETIC_DATASETS=(test test_len_near test_len_far test_len_extreme)

MODELS="${MODELS:-sft rl}"
DATA_DIR="${DATA_DIR:-data}"
EXP_ROOT="${EXP_ROOT:-exp}"
SKIP_INFERENCE="${SKIP_INFERENCE:-0}"
SKIP_JUDGE="${SKIP_JUDGE:-0}"
SKIP_HACK_CHECK="${SKIP_HACK_CHECK:-0}"
FORCE_EVAL="${FORCE_EVAL:-0}"

# Judge server (72B LLM on port 8000).
JUDGE_MODEL="${JUDGE_MODEL:-Qwen/Qwen2.5-72B-Instruct-GPTQ-Int4}"
JUDGE_PORT="${JUDGE_PORT:-8000}"
JUDGE_TP_SIZE="${JUDGE_TP_SIZE:-1}"
# Data parallel: number of full model replicas. Each replica uses TP_SIZE GPUs;
# total GPUs needed = TP_SIZE × DP_SIZE. For our 72B GPTQ-Int4 (~36GB) on H200
# (144GB), DP=8 + TP=1 gives 8 independent replicas → much higher throughput
# vs TP=8 (no inter-GPU comm during inference).
JUDGE_DP_SIZE="${JUDGE_DP_SIZE:-8}"
JUDGE_MAX_LEN="${JUDGE_MAX_LEN:-8192}"
JUDGE_GPU_UTIL="${JUDGE_GPU_UTIL:-0.95}"
JUDGE_READY_TIMEOUT="${JUDGE_READY_TIMEOUT:-1200}"  # seconds to wait for server up
JUDGE_LOG_DIR="${EXP_ROOT}/logs"

# State for cleanup trap.
JUDGE_STARTED_BY_US=0
JUDGE_PID=""

# ── Helpers ───────────────────────────────────────────────────────────────
have_cmd() { command -v "$1" >/dev/null 2>&1; }

is_port_up() {
    # Returns 0 if the judge server is responsive on JUDGE_PORT.
    curl -sf -o /dev/null "http://localhost:${JUDGE_PORT}/v1/models" 2>/dev/null
}

start_judge() {
    if is_port_up; then
        echo "== Judge already responding on :${JUDGE_PORT} (not started by this script)"
        return 0
    fi
    if ! have_cmd vllm; then
        echo "ERROR: 'vllm' command not found — cannot start judge server." >&2
        exit 1
    fi
    mkdir -p "${JUDGE_LOG_DIR}"
    local log="${JUDGE_LOG_DIR}/judge_$(date +%Y%m%d_%H%M%S).log"
    echo "== Launching judge: ${JUDGE_MODEL} on :${JUDGE_PORT} (tp=${JUDGE_TP_SIZE}, dp=${JUDGE_DP_SIZE}, max_len=${JUDGE_MAX_LEN})"
    echo "== Log: ${log}"
    vllm serve "${JUDGE_MODEL}" \
        --port "${JUDGE_PORT}" \
        --tensor-parallel-size "${JUDGE_TP_SIZE}" \
        --data-parallel-size "${JUDGE_DP_SIZE}" \
        --max-model-len "${JUDGE_MAX_LEN}" \
        --trust-remote-code \
        --gpu-memory-utilization "${JUDGE_GPU_UTIL}" \
        > "${log}" 2>&1 &
    JUDGE_PID=$!
    JUDGE_STARTED_BY_US=1
    echo "== Judge PID: ${JUDGE_PID}  — waiting for readiness (timeout=${JUDGE_READY_TIMEOUT}s)"

    local waited=0
    while (( waited < JUDGE_READY_TIMEOUT )); do
        if ! kill -0 "${JUDGE_PID}" 2>/dev/null; then
            echo "ERROR: judge process exited before becoming ready. Tail of log:" >&2
            tail -50 "${log}" >&2
            exit 1
        fi
        if is_port_up; then
            echo "== Judge ready after ${waited}s"
            return 0
        fi
        sleep 5
        waited=$((waited + 5))
        if (( waited % 60 == 0 )); then
            echo "   ... still waiting (${waited}s)"
        fi
    done
    echo "ERROR: judge did not become ready within ${JUDGE_READY_TIMEOUT}s" >&2
    tail -50 "${log}" >&2
    exit 1
}

stop_judge() {
    if [[ "${JUDGE_STARTED_BY_US}" == "1" ]] && [[ -n "${JUDGE_PID}" ]]; then
        echo "== Stopping judge (PID ${JUDGE_PID})"
        kill "${JUDGE_PID}" 2>/dev/null || true
        # Give it a few seconds, then SIGKILL if still alive.
        local waited=0
        while kill -0 "${JUDGE_PID}" 2>/dev/null && (( waited < 30 )); do
            sleep 1
            waited=$((waited + 1))
        done
        kill -9 "${JUDGE_PID}" 2>/dev/null || true
        JUDGE_STARTED_BY_US=0
        JUDGE_PID=""
    fi
}
trap stop_judge EXIT INT TERM

is_eval_valid() {
    # Returns 0 if existing eval results are valid; 1 if broken/missing (needs re-run).
    python3 scripts/utils/check_eval_valid.py "$1" >/dev/null 2>&1
}

# ── Dataset selection ─────────────────────────────────────────────────────
if [[ $# -gt 0 ]]; then
    REQUESTED=("$@")
else
    REQUESTED=("${ALL_DATASETS[@]}")
fi
DATASETS=()
for d in "${REQUESTED[@]}"; do
    if [[ -f "${DATA_DIR}/${d}.jsonl" ]]; then
        DATASETS+=("$d")
    else
        echo "WARN: ${DATA_DIR}/${d}.jsonl not found, skipping" >&2
    fi
done

echo "== Models:   ${MODELS}"
echo "== Datasets: ${DATASETS[*]}"
echo "== Output:   ${EXP_ROOT}/{model}/{dataset}/"
echo

# ── Phase 1: inference ────────────────────────────────────────────────────
# Two runners with disjoint model registries:
#   - inference_tsmllm_vllm: vLLM-based (sft/rl/sft_nothink, plus the text-only
#     baselines timeomni-1-7b, time-r1-s1p1 that piggyback on vLLM via <ts>-text
#     rendering).
#   - inference_opentslm: HF-based custom architecture (Llama+TS-encoder).
# Models with `opentslm-` prefix are dispatched to the OpenTSLM runner; everything
# else goes to the vLLM runner. Each runner does its own model-outer/dataset-inner
# loop so each checkpoint is loaded once across all datasets.
VLLM_MODELS=""
OPENTSLM_MODELS=""
for m in ${MODELS}; do
    case "$m" in
        opentslm-*) OPENTSLM_MODELS+=" $m" ;;
        *)          VLLM_MODELS+=" $m" ;;
    esac
done

if [[ "${SKIP_INFERENCE}" != "1" ]]; then
    ds_paths=()
    for d in "${DATASETS[@]}"; do
        ds_paths+=("${DATA_DIR}/${d}.jsonl")
    done

    if [[ -n "${VLLM_MODELS// /}" ]]; then
        echo "---- Phase 1a: vLLM inference [models=${VLLM_MODELS}] over ${#ds_paths[@]} dataset(s) ----"
        python -m synth.utils.inference_tsmllm_vllm \
            --datasets "${ds_paths[@]}" \
            --models ${VLLM_MODELS} \
            --exp-root "${EXP_ROOT}"
    fi

    if [[ -n "${OPENTSLM_MODELS// /}" ]]; then
        echo "---- Phase 1b: OpenTSLM inference [models=${OPENTSLM_MODELS}] over ${#ds_paths[@]} dataset(s) ----"
        python -m synth.utils.inference_opentslm \
            --datasets "${ds_paths[@]}" \
            --models ${OPENTSLM_MODELS} \
            --exp-root "${EXP_ROOT}"
    fi
else
    echo "SKIP_INFERENCE=1  — skipping inference"
fi

# ── Phase 2: LLM-as-judge evaluation ──────────────────────────────────────
if [[ "${SKIP_JUDGE}" != "1" ]]; then
    # Plan: which (model, dataset) pairs actually need eval.
    declare -a pairs_to_eval=()
    for model in ${MODELS}; do
        for d in "${DATASETS[@]}"; do
            exp_dir="${EXP_ROOT}/${model}/${d}"
            if [[ ! -f "${exp_dir}/generated_answer.json" ]]; then
                echo "WARN: ${exp_dir}/generated_answer.json missing, skipping judge for this pair" >&2
                continue
            fi
            if [[ "${FORCE_EVAL}" == "1" ]]; then
                pairs_to_eval+=("${model}|${d}")
                continue
            fi
            if is_eval_valid "${exp_dir}"; then
                reason=$(python3 scripts/utils/check_eval_valid.py "${exp_dir}" 2>&1 || true)
                echo "==  ${model}/${d}  already scored, skipping (${reason})"
            else
                reason=$(python3 scripts/utils/check_eval_valid.py "${exp_dir}" 2>&1 || true)
                echo "==  ${model}/${d}  needs re-score (${reason})"
                pairs_to_eval+=("${model}|${d}")
            fi
        done
    done

    if [[ ${#pairs_to_eval[@]} -eq 0 ]]; then
        echo "== Nothing to eval — all results valid. Skipping judge server."
    else
        echo "== ${#pairs_to_eval[@]} pair(s) to score — starting judge"
        start_judge
        for pair in "${pairs_to_eval[@]}"; do
            model="${pair%%|*}"
            d="${pair##*|}"
            exp_dir="${EXP_ROOT}/${model}/${d}"
            echo "---- judge ${model}/${d} ----"
            python -m evaluation.eval.main --exp-dir "${exp_dir}"
        done
        stop_judge
    fi
else
    echo "SKIP_JUDGE=1  — skipping judge phase"
fi

# ── Phase 3: reward-hacking check (synthetic only, needs both SFT and RL) ─
if [[ "${SKIP_HACK_CHECK}" != "1" ]] && [[ " ${MODELS} " == *" sft "* ]] && [[ " ${MODELS} " == *" rl "* ]]; then
    for d in "${DATASETS[@]}"; do
        is_synth=0
        for s in "${SYNTHETIC_DATASETS[@]}"; do
            [[ "$s" == "$d" ]] && is_synth=1 && break
        done
        [[ "$is_synth" != "1" ]] && continue
        sft_dir="${EXP_ROOT}/sft/${d}"
        rl_dir="${EXP_ROOT}/rl/${d}"
        if [[ -f "${sft_dir}/evaluation_report_optimized.csv" && -f "${rl_dir}/evaluation_report_optimized.csv" ]]; then
            echo "---- reward-hacking check ${d} ----"
            python scripts/utils/check_reward_hacking.py "${sft_dir}" "${rl_dir}" \
                --names "SFT-${d}" "RL-${d}" \
                > "${EXP_ROOT}/hacking_check_${d}.txt" 2>&1 || true
            echo "    report: ${EXP_ROOT}/hacking_check_${d}.txt"
        fi
    done
else
    echo "SKIP_HACK_CHECK or single-model — skipping hacking check"
fi

echo
echo "Done. To build the LaTeX report:"
echo "  python scripts/utils/results_to_latex.py --combine ${EXP_ROOT}/sft ${EXP_ROOT}/rl --names SFT 'SFT+RL' -o figures/results.tex"
