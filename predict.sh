#!/usr/bin/env bash
# ==============================================================================
# Zalo AI Challenge 2026 - DataMate Track
# predict.sh: Container entrypoint for offline evaluation
# Supports invocation with NO required arguments: bash /code/predict.sh
# Optional flags: --data-dir (default /data), --output-dir (default /result), --limit
# ==============================================================================

set -euo pipefail

# 1. Default paths per DataMate Docker specification
DATA_DIR="${DATA_DIR:-/data}"
OUTPUT_DIR="${OUTPUT_DIR:-/result}"
LIMIT="${LIMIT:-}"
MODEL_PATH="${MODEL_PATH:-/code/models/Qwen3-14B}"
TIMEOUT="${TIMEOUT:-15}"
MAX_RETRIES="${MAX_RETRIES:-3}"
BATCH_SIZE="${BATCH_SIZE:-32}"
EXTRA_ARGS=()

# Local environment fallback if running outside container where /data doesn't exist
if [[ ! -d "${DATA_DIR}" ]] && [[ -d "./data/sample_dataset" ]]; then
  DATA_DIR="./data/sample_dataset"
elif [[ ! -d "${DATA_DIR}" ]] && [[ -d "./data" ]]; then
  DATA_DIR="./data"
fi

if [[ ! -d "${MODEL_PATH}" ]] && [[ -d "./models/Qwen3-14B" ]]; then
  MODEL_PATH="./models/Qwen3-14B"
fi

# 2. Parse command line arguments
while [[ $# -gt 0 ]]; do
  case "$1" in
    --data-dir|--data_dir|-d)
      DATA_DIR="$2"
      shift 2
      ;;
    --output-dir|--output_dir|-o)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    --limit|-n|-l)
      LIMIT="$2"
      shift 2
      ;;
    --model-path|--model_path|-m)
      MODEL_PATH="$2"
      shift 2
      ;;
    --timeout|-t)
      TIMEOUT="$2"
      shift 2
      ;;
    --max-retries|--max_retries|-r)
      MAX_RETRIES="$2"
      shift 2
      ;;
    --batch-size|--batch_size|-b)
      BATCH_SIZE="$2"
      shift 2
      ;;
    --questions-file|--questions_file|-q)
      EXTRA_ARGS+=(--questions_file "$2")
      shift 2
      ;;
    --output-file|--output_file)
      EXTRA_ARGS+=(--output_file "$2")
      shift 2
      ;;
    --dry-run|--dry_run)
      EXTRA_ARGS+=(--dry_run)
      shift 1
      ;;
    --help|-h)
      echo "Usage: $0 [--data-dir /data] [--output-dir /result] [--limit N] [--model-path PATH]"
      exit 0
      ;;
    *)
      EXTRA_ARGS+=("$1")
      shift 1
      ;;
  esac
done

# 3. Create /result and /result/logs if missing
mkdir -p "${OUTPUT_DIR}" "${OUTPUT_DIR}/logs" 2>/dev/null || true

# 4. Strict offline environment variables
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export OMP_NUM_THREADS=4
export PYTHONDONTWRITEBYTECODE=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TMPDIR="${TMPDIR:-/tmp}"
mkdir -p "${TMPDIR}" 2>/dev/null || true

echo "======================================================================"
echo " Zalo AI Challenge 2026 - DataMate Prediction Entrypoint"
echo " Time:        $(date -u +"%Y-%m-%dT%H:%M:%SZ" 2>/dev/null || date)"
echo " Data Dir:    ${DATA_DIR}"
echo " Output Dir:  ${OUTPUT_DIR}"
echo " Model Path:  ${MODEL_PATH}"
echo " Limit:       ${LIMIT:-None (process all)}"
echo "======================================================================"

# 5. Build Python execution command
CMD=(python -m src.main
  --data_dir "${DATA_DIR}"
  --output_dir "${OUTPUT_DIR}"
  --model_path "${MODEL_PATH}"
  --timeout "${TIMEOUT}"
  --max_retries "${MAX_RETRIES}"
  --batch_size "${BATCH_SIZE}"
)

if [[ -n "${LIMIT}" ]]; then
  CMD+=(--limit "${LIMIT}")
fi

if [[ ${#EXTRA_ARGS[@]} -gt 0 ]]; then
  CMD+=("${EXTRA_ARGS[@]}")
fi

exec "${CMD[@]}"