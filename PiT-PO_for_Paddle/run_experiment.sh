#!/usr/bin/env bash
# Start one Paddle PiT-PO search; extra launch_grpo.py options override defaults.
set -euo pipefail
WORK_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -f "${WORK_DIR}/scripts/runtime_env.sh" ]]; then
    source "${WORK_DIR}/scripts/runtime_env.sh"
fi
if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    cat <<'EOF'
Usage: bash run_experiment.sh [problem] [physical_gpu] [launch_grpo.py options]
Example: bash run_experiment.sh oscillator1 2 --max_samples 8 --warmup_samples 0 --required_train_samples 2 --grpo_train_every 2
Environment: MODEL_PATH, PYTHON_BIN, PITPO_OUTPUT_ROOT, MAX_SAMPLES,
             SAMPLES_PER_PROMPT, MAX_NEW_TOKENS, GRPO_LR, GRPO_BATCH_SIZE,
             BUFFER_SIZE, GRPO_TRAIN_EVERY
For a foreground run: python launch_grpo.py --help
EOF
    exit 0
fi
PROBLEM="${1:-oscillator1}"
if (( $# > 0 )); then shift; fi
GPU="${1:-0}"
if (( $# > 0 )); then shift; fi
PYTHON_BIN="${PYTHON_BIN:-python}"
PITPO_OUTPUT_ROOT="${PITPO_OUTPUT_ROOT:-${WORK_DIR}/artifacts}"
RUN_DIR="${PITPO_OUTPUT_ROOT}/paddle_${PROBLEM}_$(date +%Y%m%d_%H%M%S)_$$"
mkdir -p "${RUN_DIR}"
COMMAND=("${PYTHON_BIN}" -u "${WORK_DIR}/launch_grpo.py"
    --problem "${PROBLEM}" --gpu "${GPU}"
    --model_path "${MODEL_PATH:-${PITPO_MODEL_PATH:-Qwen/Qwen2.5-0.5B-Instruct}}"
    --max_samples "${MAX_SAMPLES:-3000}"
    --samples_per_prompt "${SAMPLES_PER_PROMPT:-4}"
    --max_new_tokens "${MAX_NEW_TOKENS:-512}"
    --grpo_lr "${GRPO_LR:-1e-6}"
    --grpo_batch_size "${GRPO_BATCH_SIZE:-2}"
    --buffer_size "${BUFFER_SIZE:-500}"
    --grpo_train_every "${GRPO_TRAIN_EVERY:-64}"
    --output_dir "${RUN_DIR}"
    "$@")
printf '%q ' "${COMMAND[@]}" > "${RUN_DIR}/command.txt"
printf '\n' >> "${RUN_DIR}/command.txt"
cd "${WORK_DIR}"
nohup "${COMMAND[@]}" > "${RUN_DIR}/train.out" 2>&1 &
TRAIN_PID=$!
printf 'PiT-PO-Paddle %s\n' "${TRAIN_PID}" > "${RUN_DIR}/pids.txt"
printf '%s\n' "${RUN_DIR}" > "${WORK_DIR}/last_run.txt"
echo "Started Paddle PiT-PO PID ${TRAIN_PID}"
echo "Run: ${RUN_DIR}"
echo "Log: ${RUN_DIR}/train.out"
echo "Stop: bash ${WORK_DIR}/stop_experiment.sh ${RUN_DIR}"
