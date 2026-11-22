#!/usr/bin/env bash
set -euo pipefail

DATA_DIR="${DATA_DIR:-/data}"
OUTPUT_DIR="${OUTPUT_DIR:-/result}"
LIMIT="${LIMIT:-}"
MODEL_PATH="${MODEL_PATH:-/code/models/Qwen3-14B}"
TIMEOUT="${TIMEOUT:-15}"
SOFT_TIMEOUT="${SOFT_TIMEOUT:-100}"
TIME_BUDGET_SECONDS="${TIME_BUDGET_SECONDS:-20700}"
MAX_RETRIES="${MAX_RETRIES:-3}"
BATCH_SIZE="${BATCH_SIZE:-32}"
EXTRA_ARGS=()

if [[ ! -d "${DATA_DIR}" ]] && [[ -d "./data/sample_dataset" ]]; then
  DATA_DIR="./data/sample_dataset"
elif [[ ! -d "${DATA_DIR}" ]] && [[ -d "./data" ]]; then
  DATA_DIR="./data"
fi

if [[ ! -d "${MODEL_PATH}" ]] && [[ -d "./models/Qwen3-14B" ]]; then
  MODEL_PATH="./models/Qwen3-14B"
fi

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
    --soft-timeout|--soft_timeout)
      SOFT_TIMEOUT="$2"
      shift 2
      ;;
    --time-budget-seconds|--time_budget_seconds)
      TIME_BUDGET_SECONDS="$2"
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

mkdir -p "${OUTPUT_DIR}" "${OUTPUT_DIR}/logs" 2>/dev/null || true

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export OMP_NUM_THREADS=4
export PYTHONDONTWRITEBYTECODE=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export NCCL_P2P_DISABLE=1
export PYTHONHASHSEED=42
export VLLM_SEED=42
export VLLM_MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-8}"
export TMPDIR=/tmp
export TEMP=/tmp
export TMP=/tmp
mkdir -p /tmp

START_TIME=$(date +%s)
echo "======================================================================"
echo " Zalo AI Challenge 2026 - DataMate Prediction Entrypoint"
echo " Time:        $(date -u +"%Y-%m-%dT%H:%M:%SZ" 2>/dev/null || date)"
echo " Data Dir:    ${DATA_DIR}"
echo " Output Dir:  ${OUTPUT_DIR}"
echo " Model Path:  ${MODEL_PATH}"
echo " Limit:       ${LIMIT:-None (process all)}"
echo "======================================================================"

CMD=(python -m src.main
  --data_dir "${DATA_DIR}"
  --output_dir "${OUTPUT_DIR}"
  --model_path "${MODEL_PATH}"
  --timeout "${TIMEOUT}"
  --soft_timeout "${SOFT_TIMEOUT}"
  --time_budget_seconds "${TIME_BUDGET_SECONDS}"
  --max_retries "${MAX_RETRIES}"
  --batch_size "${BATCH_SIZE}"
)

if [[ -n "${LIMIT}" ]]; then
  CMD+=(--limit "${LIMIT}")
fi

if [[ ${#EXTRA_ARGS[@]} -gt 0 ]]; then
  CMD+=("${EXTRA_ARGS[@]}")
fi

"${CMD[@]}"

END_TIME=$(date +%s)
TOTAL_SEC=$((END_TIME - START_TIME))
TOTAL_MS=$((TOTAL_SEC * 1000))

echo "======================================================================"
echo " Inference Summary"
echo " Total Elapsed: ${TOTAL_SEC}s (${TOTAL_MS} ms)"
if [[ -f "${OUTPUT_DIR}/logs/runtime_metrics.json" ]]; then
  python -c '
import json
with open("'"${OUTPUT_DIR}"'/logs/runtime_metrics.json", "r", encoding="utf-8") as f:
    metrics = json.load(f)
print(f" load_model_ms={metrics.get('"'"'load_model_ms'"'"', 0)}")
print(f" total_ms={metrics.get('"'"'total_ms'"'"', 0)}")
'
fi
if [[ -f "${OUTPUT_DIR}/logs/predictions.jsonl" ]]; then
  echo " Route Breakdown:"
  python -c '
import json, sys
from collections import Counter
counts = Counter()
try:
    with open("'"${OUTPUT_DIR}"'/logs/predictions.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            rec = json.loads(line)
            rt = rec.get("route_trace") or {}
            init = rt.get("route_initial", "UNKNOWN")
            counts[init] += 1
    for k, v in counts.items():
        print(f"   - {k}: {v} questions")
except Exception as e:
    pass
'
fi
echo "======================================================================"
