#!/usr/bin/env bash
# ==============================================================================
# Zalo AI Challenge 2026 - DataMate Track
# Offline Inference Runner Script for Docker Container Submission
# ==============================================================================

set -euo pipefail

# Default Configuration Paths
DATA_DIR="${DATA_DIR:-/app/data}"
QUESTIONS_FILE="${QUESTIONS_FILE:-${DATA_DIR}/questions.jsonl}"
OUTPUT_FILE="${OUTPUT_FILE:-/app/submission.jsonl}"
MODEL_PATH="${MODEL_PATH:-/app/model}"
TIMEOUT="${TIMEOUT:-15}"
MAX_RETRIES="${MAX_RETRIES:-1}"

# Parse optional command line arguments
while [[ $# -gt 0 ]]; do
  case "$1" in
    --data_dir|-d)
      DATA_DIR="$2"
      shift 2
      ;;
    --questions_file|-q)
      QUESTIONS_FILE="$2"
      shift 2
      ;;
    --output_file|-o)
      OUTPUT_FILE="$2"
      shift 2
      ;;
    --model_path|-m)
      MODEL_PATH="$2"
      shift 2
      ;;
    --timeout|-t)
      TIMEOUT="$2"
      shift 2
      ;;
    --max_retries|-r)
      MAX_RETRIES="$2"
      shift 2
      ;;
    --help|-h)
      echo "Usage: $0 [options]"
      echo "Options:"
      echo "  --data_dir, -d        Directory containing tabular data & README.md (default: ${DATA_DIR})"
      echo "  --questions_file, -q  Path to questions.jsonl (default: ${QUESTIONS_FILE})"
      echo "  --output_file, -o     Path to save submission.jsonl (default: ${OUTPUT_FILE})"
      echo "  --model_path, -m      Path to local vLLM weights (default: ${MODEL_PATH})"
      echo "  --timeout, -t         Execution timeout per question in seconds (default: ${TIMEOUT})"
      echo "  --max_retries, -r     Maximum self-correction retries per question (default: ${MAX_RETRIES})"
      exit 0
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

echo "======================================================================"
echo " Starting Zalo DataMate 2026 Inference Pipeline"
echo " Time: $(date -u +"%Y-%m-%dT%H:%M:%SZ")"
echo " Data Directory:    ${DATA_DIR}"
echo " Questions File:    ${QUESTIONS_FILE}"
echo " Output File:       ${OUTPUT_FILE}"
echo " Model Weights:     ${MODEL_PATH}"
echo " Timeout / Retries: ${TIMEOUT}s / ${MAX_RETRIES}"
echo "======================================================================"

# Ensure output directory exists
mkdir -p "$(dirname "${OUTPUT_FILE}")"

# Export system & vLLM optimizations for 100% offline Docker container
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export OMP_NUM_THREADS=4
export PYTHONDONTWRITEBYTECODE=1

# Check CUDA status if nvidia-smi is available
if command -v nvidia-smi &> /dev/null; then
  echo ">> GPU detected:"
  nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader || true
else
  echo ">> No nvidia-smi detected, running in CPU / standard mode."
fi

# Execute Python orchestrator
python -m src.main \
  --data_dir "${DATA_DIR}" \
  --questions_file "${QUESTIONS_FILE}" \
  --output_file "${OUTPUT_FILE}" \
  --model_path "${MODEL_PATH}" \
  --timeout "${TIMEOUT}" \
  --max_retries "${MAX_RETRIES}"

EXIT_CODE=$?

if [[ ${EXIT_CODE} -eq 0 ]]; then
  echo "======================================================================"
  echo " Inference finished successfully!"
  echo " Generated: ${OUTPUT_FILE}"
  echo " Total rows: $(wc -l < "${OUTPUT_FILE}" 2>/dev/null || echo "N/A")"
  echo "======================================================================"
else
  echo "[ERROR] Inference failed with exit code ${EXIT_CODE}"
fi

exit ${EXIT_CODE}
