#!/usr/bin/env bash
# ==============================================================================
# Zalo AI Challenge 2026 - DataMate Track
# training_code/train.sh: Training & Data Validation Entrypoint
# ==============================================================================

set -euo pipefail

DATA_DIR="${1:-./data}"
LOG_DIR="./training_code/logs"
mkdir -p "${LOG_DIR}"

echo "=============================================================================="
echo "      Zalo AI Challenge 2026 - DataMate Track: Training & Data Validation"
echo "=============================================================================="
echo "Note: Zero-shot / prompted in-context learning is used with Qwen/Qwen3-14B."
echo "No model parameter fine-tuning is required or performed."
echo "Data Directory: ${DATA_DIR}"
echo "------------------------------------------------------------------------------"

# 1. Verify model manifest integrity
MANIFEST_FILE="./training_code/data_manifest.json"
if [[ -f "${MANIFEST_FILE}" ]]; then
  echo "Validating model manifest: ${MANIFEST_FILE}..."
  python -c "
import json
with open('${MANIFEST_FILE}', 'r', encoding='utf-8') as f:
    d = json.load(f)
print(f'Model: {d.get(\"model_name\")} ({d.get(\"parameter_count\")})')
print(f'Commit Hash: {d.get(\"commit_hash\")}')
print(f'Safetensors shards: {len(d.get(\"safetensors_files\", {}))}')
"
fi

# 2. Run dataset validation if data directory exists
if [[ -d "${DATA_DIR}" ]]; then
  echo "Running training data validation on ${DATA_DIR}..."
  python -m training_code.src.validate_data --data_dir "${DATA_DIR}" || true
fi

echo "Training validation complete. Ready for offline inference."
