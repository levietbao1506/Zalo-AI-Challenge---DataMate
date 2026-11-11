# ==============================================================================
# Zalo AI Challenge 2026 - DataMate Track
# Official Production-Grade Dockerfile for Offline Evaluation (<15B Model, vLLM)
# Base: NVIDIA CUDA 12.8 on Ubuntu 22.04 per Competition Rules
# ==============================================================================

FROM nvidia/cuda:12.8.0-runtime-ubuntu22.04

LABEL maintainer="Zalo AI Challenge - DataMate 2026"
LABEL description="Offline tabular QA inference container running vLLM, Pandas, DuckDB"

# 1. System environment configuration
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    TOKENIZERS_PARALLELISM=false \
    VLLM_WORKER_MULTIPROC_METHOD=spawn \
    OMP_NUM_THREADS=4 \
    PATH="/usr/local/bin:$PATH"

# 2. Install Python 3.10 and necessary runtime libraries
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-dev \
    git \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Set python3 as default python
RUN ln -sf /usr/bin/python3 /usr/bin/python

# 3. Create competition directories
WORKDIR /app
RUN mkdir -p /app/data /app/model /app/result

# 4. Install Python dependencies
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r /app/requirements.txt

# 5. Copy source code and scripts
COPY . /app

# 6. Ensure shell script has executable permissions and Unix LF line endings
RUN sed -i 's/\r$//' /app/run_inference.sh && \
    chmod +x /app/run_inference.sh

# 7. Default Entrypoint for BTC Evaluation Harness
ENTRYPOINT ["/app/run_inference.sh"]
