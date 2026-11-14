#!/usr/bin/env bash
# ==============================================================================
# Zalo AI Challenge 2026 - DataMate Track
# Backward-compatible thin wrapper delegating to predict.sh
# Supports CLI parameters:
#   --data_dir, --questions_file, --output_file, --model_path, --timeout, --max_retries
# ==============================================================================

set -euo pipefail

# Export offline environment variables
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONUNBUFFERED=1

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PREDICT_SCRIPT="${SCRIPT_DIR}/predict.sh"

if [[ ! -f "${PREDICT_SCRIPT}" ]]; then
  PREDICT_SCRIPT="/code/predict.sh"
fi

exec bash "${PREDICT_SCRIPT}" "$@"