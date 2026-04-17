
#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
CACHE_ROOT_DEFAULT="${REPO_ROOT}/.cache/huggingface"

export WANDB_BASE_URL=${WANDB_BASE_URL:-https://api.bandw.top}
export WANDB_API_KEY=${WANDB_API_KEY:-}
export HF_HOME=${HF_HOME:-"${CACHE_ROOT_DEFAULT}"}
export HUGGINGFACE_HUB_CACHE=${HUGGINGFACE_HUB_CACHE:-"${HF_HOME}/hub"}
export HF_MODULES_CACHE=${HF_MODULES_CACHE:-"${HF_HOME}/modules"}
export HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}

ROOT_DIR=${ROOT_DIR:-"${REPO_ROOT}"}

MODEL=${1:-"gpt-oss-120b"}
DATA_ROOT=${DATA_ROOT:-"${ROOT_DIR}/data/train-origin"}
TRAIN_DATA=${2:-"${ROOT_DIR}/data/train/harmony_prefix4k_margin512_training.jsonl"}
VAL_DATA=${3:-"${ROOT_DIR}/data/test/harmony_prefix4k_margin512_training.jsonl"}
RANK=${4:-256}
BATCH_SIZE=${5:-4}
NUM_ASSESS_TOKENS=${6:-4}
EPOCHS=${7:-10}
MAX_LENGTH=${8:-5120}
# Hard-label mode is parked for now to avoid mixing training targets.
# USE_HARD_LABELS=${9:-"False"}
USE_HARD_LABELS="False"
NUM_PROCESSES=${10:-2}
LEARNING_RATE=${11:-1e-4}
GRAD_ACCUM=${12:-8}
DTYPE=${13:-"bf16"}
GRADIENT_CHECKPOINTING=${14:-"True"}
LORA_ALPHA=${15:-$((RANK * 2))}

MODEL_ROOT=${MODEL_ROOT:-"${ROOT_DIR}/artifacts/models"}
MODEL_PATH="${MODEL_ROOT}/${MODEL}"
TRAIN_TAG=$(basename "${TRAIN_DATA}")
TRAIN_TAG=${TRAIN_TAG%.jsonl}
VAL_TAG=""
if [[ -n "${VAL_DATA}" ]]; then
    VAL_TAG=$(basename "${VAL_DATA}")
    VAL_TAG=${VAL_TAG%.jsonl}
fi
DATA_TAG=${DATA_TAG:-"${TRAIN_TAG}"}
if [[ -n "${VAL_TAG}" && "${VAL_TAG}" != "${TRAIN_TAG}" ]]; then
    DATA_TAG="${DATA_TAG}__val_${VAL_TAG}"
fi
WANDB_PROJECT=${WANDB_PROJECT:-"harmony_prefix_classifier"}
LABEL_TAG="soft"
# if [[ "${USE_HARD_LABELS}" == "True" || "${USE_HARD_LABELS}" == "true" ]]; then
#     LABEL_TAG="hard"
# fi
WANDB_RUN_NAME=${WANDB_RUN_NAME:-"${MODEL}_${DATA_TAG}_harmony_${LABEL_TAG}_r${RANK}_a${NUM_ASSESS_TOKENS}_e${EPOCHS}_bs${BATCH_SIZE}_ml${MAX_LENGTH}"}
OUTPUT_ROOT=${OUTPUT_ROOT:-"${ROOT_DIR}/output/harmony_multi_gpu"}
OUTPUT_DIR="${OUTPUT_ROOT}/${WANDB_RUN_NAME}"
LOG_ROOT=${LOG_ROOT:-"${ROOT_DIR}/output"}
LOG_TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="${LOG_ROOT}/${WANDB_RUN_NAME}_${LOG_TIMESTAMP}.log"
ACCELERATE_CONFIG=${ACCELERATE_CONFIG:-"${SCRIPT_DIR}/accelerate_harmony_multi_gpu.yaml"}
TRAIN_SCRIPT="${ROOT_DIR}/src/finetuning_harmony.py"

mkdir -p "${OUTPUT_ROOT}"
mkdir -p "${LOG_ROOT}"
exec > >(tee -a "${LOG_FILE}") 2>&1

echo "Logging to ${LOG_FILE}"

if [[ ! -d "${MODEL_PATH}" ]]; then
    echo "Model path does not exist: ${MODEL_PATH}" >&2
    exit 1
fi

if [[ ! -f "${TRAIN_SCRIPT}" ]]; then
    echo "Training script does not exist: ${TRAIN_SCRIPT}" >&2
    exit 1
fi

if [[ ! -f "${ACCELERATE_CONFIG}" ]]; then
    echo "Accelerate config does not exist: ${ACCELERATE_CONFIG}" >&2
    echo "Set ACCELERATE_CONFIG to a valid file before running." >&2
    exit 1
fi

if ! command -v accelerate >/dev/null 2>&1; then
    echo "accelerate is not available in PATH." >&2
    exit 1
fi

CMD=(
    accelerate launch
    --config_file "${ACCELERATE_CONFIG}"
    --num_processes "${NUM_PROCESSES}"
    "${TRAIN_SCRIPT}"
    --model_path "${MODEL_PATH}"
    --data_path "${TRAIN_DATA}"
    --output_dir "${OUTPUT_DIR}"
    --max_length "${MAX_LENGTH}"
    --epochs "${EPOCHS}"
    --learning_rate "${LEARNING_RATE}"
    --lora_r "${RANK}"
    --lora_alpha "${LORA_ALPHA}"
    --num_assess_tokens "${NUM_ASSESS_TOKENS}"
    --batch_size "${BATCH_SIZE}"
    --gradient_accumulation_steps "${GRAD_ACCUM}"
    --wandb_project "${WANDB_PROJECT}"
    --wandb_run_name "${WANDB_RUN_NAME}"
    --use_hard_labels "${USE_HARD_LABELS}"
    --dtype "${DTYPE}"
    --gradient_checkpointing "${GRADIENT_CHECKPOINTING}"
    --local_files_only True
)

if [[ -n "${VAL_DATA}" ]]; then
    CMD+=(--val_data_path "${VAL_DATA}")
fi

printf 'Running command:\n%s\n' "${CMD[*]}"
"${CMD[@]}"
