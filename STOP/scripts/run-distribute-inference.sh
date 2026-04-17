#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
WORK_DIR="${REPO_ROOT}/Prefix-Generation."

cd "${WORK_DIR}"

if [[ -n "${CONDA_SH_PATH:-}" && -f "${CONDA_SH_PATH}" ]]; then
    # Optional conda bootstrap for local environments.
    source "${CONDA_SH_PATH}"
elif command -v conda >/dev/null 2>&1; then
    CONDA_BASE="$(conda info --base 2>/dev/null || true)"
    if [[ -n "${CONDA_BASE}" && -f "${CONDA_BASE}/etc/profile.d/conda.sh" ]]; then
        source "${CONDA_BASE}/etc/profile.d/conda.sh"
    fi
fi

if [[ -n "${STOP_CONDA_ENV:-}" ]] && command -v conda >/dev/null 2>&1; then
    conda activate "${STOP_CONDA_ENV}"
fi

MODEL_PATH="${1:-${MODEL_PATH:-${REPO_ROOT}/artifacts/models/gpt-oss-120b}}"
INPUT="${2:-${LOCAL_REFERENCE_FILE:-${REPO_ROOT}/artifacts/data/eval_large.csv}}"
GPUS="${3:-1}"
OUTPUT_DIR="${4:-${OUTPUT_DIR:-${REPO_ROOT}/output}}"
NOTEBOOK_FILENAME="${5:-${NOTEBOOK_FILENAME:-${WORK_DIR}/saving-trace-fix.ipynb}}"


export TOTAL_GPUS="${GPUS}"
export MODEL_PATH="${MODEL_PATH}"
export LOCAL_REFERENCE_FILE="${INPUT}"
export OUTPUT_DIR="${OUTPUT_DIR}"
export NOTEBOOK_FILENAME="${NOTEBOOK_FILENAME}"

python distribute_inference.py
