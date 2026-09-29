#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

PYTHON_BIN="${PYTHON_BIN:-python}"
PROFILE="${PROFILE:-small_0_6b}"
CONFIG="${CONFIG:-config.yaml}"
DATA_ROOT="${DATA_ROOT:-data/grit1}"
SAFETY_TASK_DATA_DIR="${SAFETY_TASK_DATA_DIR:-${DATA_ROOT}/safety_task}"
PRESERVATION_DATA_DIR="${PRESERVATION_DATA_DIR:-${DATA_ROOT}/preservation}"
TASK_DATA="${TASK_DATA:-${SAFETY_TASK_DATA_DIR}/task_train.parquet}"
PRESERVATION_DATA="${PRESERVATION_DATA:-${PRESERVATION_DATA_DIR}/preserve_1000.parquet}"
PROJECTORS="${PROJECTORS:-artifacts/${PROFILE}_projectors.pt}"
OUTPUT_DIR="${OUTPUT_DIR:-checkpoints/${PROFILE}}"
EPOCHS="${EPOCHS:-1}"
STEPS_PER_EPOCH="${STEPS_PER_EPOCH:-10}"
CENTRAL_FD_RADIUS="${CENTRAL_FD_RADIUS:-0.05}"
BATCH_SIZE="${BATCH_SIZE:-1}"
PRESERVATION_BATCH_SIZE="${PRESERVATION_BATCH_SIZE:-1}"
AUTO_PREPARE="${AUTO_PREPARE:-1}"

# GPU 0 hosts both small vLLM services by default. GPU 1 and GPU 2 run two
# replicated training workers. Override these values for the target machine.
INFERENCE_GPU="${INFERENCE_GPU:-0}"
TRAIN_GPUS="${TRAIN_GPUS:-1,2}"
NUM_TRAIN_GPUS="${NUM_TRAIN_GPUS:-2}"
POLICY_PORT="${POLICY_PORT:-8000}"
SAFETY_PORT="${SAFETY_PORT:-8001}"
POLICY_URL="${POLICY_URL:-http://127.0.0.1:${POLICY_PORT}}"
SAFETY_URL="${SAFETY_URL:-http://127.0.0.1:${SAFETY_PORT}}"
START_POLICY_SERVER="${START_POLICY_SERVER:-1}"
START_SAFETY_SERVER="${START_SAFETY_SERVER:-1}"
VLLM_POLICY_MEMORY="${VLLM_POLICY_MEMORY:-0.42}"
VLLM_SAFETY_MEMORY="${VLLM_SAFETY_MEMORY:-0.35}"

MODEL_PATH="${MODEL_PATH:-$(${PYTHON_BIN} -c "from grit1.config import load_project_config; print(load_project_config('${CONFIG}', '${PROFILE}').profile.policy_model)")}" 
BASE_MODEL_PATH="${BASE_MODEL_PATH:-${MODEL_PATH}}"
SAFETY_MODEL_PATH="${SAFETY_MODEL_PATH:-$(${PYTHON_BIN} -c "from grit1.config import load_project_config; print(load_project_config('${CONFIG}', '${PROFILE}').profile.safety_model)")}" 
PRM_MODEL_PATH="${PRM_MODEL_PATH:-$(${PYTHON_BIN} -c "from grit1.config import load_project_config; print(load_project_config('${CONFIG}', '${PROFILE}').profile.prm_model or '')")}" 

mkdir -p "${SAFETY_TASK_DATA_DIR}" "${PRESERVATION_DATA_DIR}" \
  "$(dirname "${PROJECTORS}")" "${OUTPUT_DIR}" logs

if [[ "${AUTO_PREPARE}" == "1" && ! -f "${TASK_DATA}" ]]; then
  "${PYTHON_BIN}" scripts/prepare_grit_data.py \
    --task-only \
    --task-output-dir "${SAFETY_TASK_DATA_DIR}" \
    --preservation-output-dir "${PRESERVATION_DATA_DIR}" \
    --task-max-samples "${TASK_MAX_SAMPLES:-10000}" \
    --val-size "${VAL_SIZE:-1000}"
fi

if [[ "${AUTO_PREPARE}" == "1" && ! -f "${PRESERVATION_DATA}" ]]; then
  "${PYTHON_BIN}" scripts/prepare_grit_data.py \
    --preservation-only \
    --task-output-dir "${SAFETY_TASK_DATA_DIR}" \
    --preservation-output-dir "${PRESERVATION_DATA_DIR}" \
    --preserve-max-samples "${PRESERVE_MAX_SAMPLES:-1000}"
fi

if [[ ! -f "${TASK_DATA}" || ! -f "${PRESERVATION_DATA}" ]]; then
  echo "Missing task or preservation data. Set TASK_DATA/PRESERVATION_DATA or AUTO_PREPARE=1." >&2
  exit 2
fi

if [[ ! -f "${PROJECTORS}" ]]; then
  CUDA_VISIBLE_DEVICES="${INFERENCE_GPU}" "${PYTHON_BIN}" scripts/build_projectors.py \
    --model-path "${BASE_MODEL_PATH}" \
    --dataset-path "${PRESERVATION_DATA}" \
    --text-column text \
    --output-path "${PROJECTORS}" \
    --max-samples "${PROJECTOR_MAX_SAMPLES:-1000}" \
    --batch-size "${PROJECTOR_BATCH_SIZE:-1}" \
    --max-length "${PROJECTOR_MAX_LENGTH:-1024}" \
    --dtype "${PROJECTOR_DTYPE:-float16}" \
    --device cuda \
    --trust-remote-code
fi

safety_pid=""
policy_pid=""
cleanup() {
  if [[ -n "${policy_pid}" ]] && kill -0 "${policy_pid}" 2>/dev/null; then
    kill "${policy_pid}" 2>/dev/null || true
    wait "${policy_pid}" 2>/dev/null || true
  fi
  if [[ -n "${safety_pid}" ]] && kill -0 "${safety_pid}" 2>/dev/null; then
    kill "${safety_pid}" 2>/dev/null || true
    wait "${safety_pid}" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

if [[ "${START_SAFETY_SERVER}" == "1" ]]; then
  CUDA_VISIBLE_DEVICES="${INFERENCE_GPU}" vllm serve "${SAFETY_MODEL_PATH}" \
    --served-model-name guard \
    --host 127.0.0.1 \
    --port "${SAFETY_PORT}" \
    --dtype auto \
    --max-model-len "${SAFETY_MAX_MODEL_LEN:-4096}" \
    --gpu-memory-utilization "${VLLM_SAFETY_MEMORY}" \
    >logs/safety-vllm.log 2>&1 &
  safety_pid=$!
  "${PYTHON_BIN}" scripts/wait_for_server.py "${SAFETY_URL}/v1/models"
fi

current_model="${MODEL_PATH}"
for ((epoch=0; epoch<EPOCHS; epoch++)); do
  if [[ "${START_POLICY_SERVER}" == "1" ]]; then
    CUDA_VISIBLE_DEVICES="${INFERENCE_GPU}" vllm serve "${current_model}" \
      --served-model-name policy \
      --host 127.0.0.1 \
      --port "${POLICY_PORT}" \
      --dtype auto \
      --max-model-len "${POLICY_MAX_MODEL_LEN:-2048}" \
      --gpu-memory-utilization "${VLLM_POLICY_MEMORY}" \
      >logs/policy-vllm.log 2>&1 &
    policy_pid=$!
    "${PYTHON_BIN}" scripts/wait_for_server.py "${POLICY_URL}/v1/models"
  fi

  trainer_args=(
    --config "${CONFIG}"
    --profile "${PROFILE}"
    --model-path "${current_model}"
    --base-model-path "${BASE_MODEL_PATH}"
    --task-data "${TASK_DATA}"
    --preservation-data "${PRESERVATION_DATA}"
    --projectors "${PROJECTORS}"
    --output-dir "${OUTPUT_DIR}"
    --rollout-url "${POLICY_URL}"
    --rollout-model policy
    --safety-url "${SAFETY_URL}"
    --safety-model guard
    --batch-size "${BATCH_SIZE}"
    --preservation-batch-size "${PRESERVATION_BATCH_SIZE}"
    --max-steps "${STEPS_PER_EPOCH}"
    --epoch "${epoch}"
    --use-prm
    --soft-projection
    --trust-region
    --competence-gating
    --use-curvature
    --central-fd-radius "${CENTRAL_FD_RADIUS}"
    --central-fd-normalize-direction
    --hvp-last-linear-layers "${HVP_LAST_LINEAR_LAYERS:-1}"
    --gradient-checkpointing
  )
  if [[ -n "${PRM_MODEL_PATH}" ]]; then
    trainer_args+=(--prm-model "${PRM_MODEL_PATH}")
  fi
  if [[ -f "${OUTPUT_DIR}/optimizer.pt" ]]; then
    trainer_args+=(--resume-optimizer "${OUTPUT_DIR}/optimizer.pt")
  fi

  CUDA_VISIBLE_DEVICES="${TRAIN_GPUS}" torchrun \
    --standalone \
    --nproc_per_node "${NUM_TRAIN_GPUS}" \
    -m grit1.train "${trainer_args[@]}"

  if [[ -n "${policy_pid}" ]] && kill -0 "${policy_pid}" 2>/dev/null; then
    kill "${policy_pid}" 2>/dev/null || true
    wait "${policy_pid}" 2>/dev/null || true
  fi
  policy_pid=""
  current_model="${OUTPUT_DIR}"
done

echo "GRIT_1 training complete: ${OUTPUT_DIR}"
