# ==============================================================================
# Zalo AI Challenge 2026 - DataMate Track
# Production Dockerfile for Offline Evaluation (<15B Model, vLLM)
# Base: vllm/vllm-openai:v0.10.1.1
# ==============================================================================

FROM vllm/vllm-openai:v0.10.1.1

LABEL maintainer="Zalo AI Challenge - DataMate 2026"
LABEL description="Offline tabular QA inference container running vLLM, Pandas, DuckDB"

# Override default vLLM entrypoint to allow custom script execution
ENTRYPOINT []

# Strict offline and runtime environment flags
ENV HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    HF_DATASETS_OFFLINE=1 \
    PIP_NO_CACHE_DIR=1 \
    NCCL_P2P_DISABLE=1 \
    PYTHONHASHSEED=42 \
    VLLM_SEED=42 \
    VLLM_MAX_NUM_SEQS=8 \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    TOKENIZERS_PARALLELISM=false \
    VLLM_WORKER_MULTIPROC_METHOD=spawn \
    OMP_NUM_THREADS=4

WORKDIR /code

# Install competition Python dependencies and pin nvidia-nccl-cu12
COPY requirements.txt /code/requirements.txt
RUN pip install --no-cache-dir --break-system-packages -r /code/requirements.txt && \
    pip install --no-cache-dir --break-system-packages --no-deps nvidia-nccl-cu12==2.27.7

# Copy models, training code, predict runner, documentation, and source code
COPY models/ /code/models/
COPY training_code/ /code/training_code/
COPY predict.sh /code/predict.sh
COPY README.md /code/README.md
COPY src/ /code/src/

# Fail the build if the local weights are absent or the declared model exceeds 15B.
RUN python -c 'import json,pathlib; manifest=json.loads(pathlib.Path("/code/training_code/data_manifest.json").read_text()); assert manifest["parameters_total"] <= 15_000_000_000; assert pathlib.Path("/code/models/Qwen3-14B/config.json").is_file()'

# Ensure predict script is executable and has Unix LF line endings
RUN sed -i 's/\r$//' /code/predict.sh && \
    chmod +x /code/predict.sh

# Default execution command
CMD ["bash", "/code/predict.sh"]
