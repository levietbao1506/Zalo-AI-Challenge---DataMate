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
    PIP_NO_CACHE_DIR=1 \
    NCCL_P2P_DISABLE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    TOKENIZERS_PARALLELISM=false \
    VLLM_WORKER_MULTIPROC_METHOD=spawn \
    OMP_NUM_THREADS=4

WORKDIR /code

# Install competition Python dependencies and pin nvidia-nccl-cu12
COPY requirements.txt /code/requirements.txt
RUN pip install --no-cache-dir -r /code/requirements.txt && \
    pip install --no-cache-dir --no-deps nvidia-nccl-cu12==2.27.7

# Copy models, training code, predict runner, documentation, and source code
COPY models/ /code/models/
COPY training_code/ /code/training_code/
COPY predict.sh /code/predict.sh
COPY README.md /code/README.md
COPY src/ /code/src/

# Ensure predict script is executable and has Unix LF line endings
RUN sed -i 's/\r$//' /code/predict.sh && \
    chmod +x /code/predict.sh

# Default execution command
CMD ["bash", "/code/predict.sh"]
