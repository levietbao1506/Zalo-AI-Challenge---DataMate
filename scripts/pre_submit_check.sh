#!/usr/bin/env bash
set -euo pipefail

IMAGE="${IMAGE:-datamate:latest}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="${DATA_DIR:-${1:-$(pwd)/data/sample_dataset}}"
WORK_ROOT="${WORK_ROOT:-$(mktemp -d)}"
RUN1="${WORK_ROOT}/run1"
RUN2="${WORK_ROOT}/run2"
MAX_IMAGE_BYTES=$((100 * 1024 * 1024 * 1024))

DATA_DIR="$(cd "${DATA_DIR}" && pwd)"
mkdir -p "${RUN1}" "${RUN2}"

docker run --rm --platform linux/amd64 --gpus '"device=0,1"' --network none --shm-size=16g --entrypoint bash -v "${DATA_DIR}:/data:ro" -v "${RUN1}:/result" "${IMAGE}" /code/predict.sh
docker run --rm --platform linux/amd64 --gpus '"device=0,1"' --network none --shm-size=16g --entrypoint bash -v "${DATA_DIR}:/data:ro" -v "${RUN2}:/result" "${IMAGE}" /code/predict.sh

ANSWERS_FILE="${DATA_DIR}/answers.jsonl"
if [[ ! -f "${ANSWERS_FILE}" ]]; then
  ANSWERS_FILE="${DATA_DIR}/ground_truth.jsonl"
fi
if [[ ! -f "${ANSWERS_FILE}" ]]; then
  echo "answers.jsonl or ground_truth.jsonl is required for PA reproducibility checks" >&2
  exit 1
fi

python "${SCRIPT_DIR}/verify_submission_runs.py" --questions "${DATA_DIR}/questions.jsonl" --answers "${ANSWERS_FILE}" --run1 "${RUN1}" --run2 "${RUN2}" --max-pa-delta 0.03

IMAGE_ARCH="$(docker image inspect --format '{{.Architecture}}' "${IMAGE}")"
IMAGE_BYTES="$(docker image inspect --format '{{.Size}}' "${IMAGE}")"
if [[ "${IMAGE_ARCH}" != "amd64" ]]; then
  echo "Docker image architecture must be amd64, got ${IMAGE_ARCH}" >&2
  exit 1
fi
if (( IMAGE_BYTES > MAX_IMAGE_BYTES )); then
  echo "Docker image uncompressed size exceeds 100 GB: ${IMAGE_BYTES} bytes" >&2
  exit 1
fi

EXPORT_FILE="${EXPORT_FILE:-datamate_image.tar.gz}"
echo "Image architecture: ${IMAGE_ARCH}"
echo "Image uncompressed size: ${IMAGE_BYTES} bytes"
echo "Safe export commands:"
echo "set -o pipefail"
printf 'docker save %q | pigz -p 8 > %q\n' "${IMAGE}" "${EXPORT_FILE}"
printf 'pigz -dc %q | wc -c\n' "${EXPORT_FILE}"
printf 'md5sum %q > %q\n' "${EXPORT_FILE}" "${EXPORT_FILE}.md5"
