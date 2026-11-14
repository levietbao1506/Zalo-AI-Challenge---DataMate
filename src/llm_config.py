"""vLLM engine configuration and offline loader for Qwen/Qwen3-14B.

Designed for the Zalo AI Challenge 2026 (DataMate Track) offline evaluation container
running vllm/vllm-openai:v0.10.1.1 on dual RTX 5090 Blackwell GPUs (CUDA 12.8, PyTorch 2.7.1).
"""

from __future__ import annotations

from dataclasses import dataclass, field
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger(__name__)

# Default offline model path in Docker container
DEFAULT_OFFLINE_MODEL_PATH = "/code/models/Qwen3-14B"
FALLBACK_LOCAL_MODEL_PATH = "models/Qwen3-14B"

# Thinking mode sampling parameters
THINKING_TEMPERATURE = 0.6
THINKING_TOP_P = 0.95
THINKING_TOP_K = 20

# Non-thinking (deterministic) sampling parameters
NON_THINKING_TEMPERATURE = 0.0

DEFAULT_SEED = 42
DEFAULT_MAX_MODEL_LEN = 32768
DEFAULT_GPU_MEMORY_UTILIZATION = 0.85
DEFAULT_TENSOR_PARALLEL_SIZE = 2
DEFAULT_MAX_TOKENS = 8192

DEFAULT_STOP_TOKENS = [
    "<|im_end|>",
    "<|endoftext|>",
    "<tool_call>",
]

try:
    from vllm import LLM, SamplingParams
    _HAS_VLLM = True
except ImportError:
    _HAS_VLLM = False
    LLM = Any
    SamplingParams = Any


@dataclass
class VLLMConfig:
    """Configuration dataclass for offline vLLM engine initialization."""

    model_path: str = DEFAULT_OFFLINE_MODEL_PATH
    tensor_parallel_size: int = DEFAULT_TENSOR_PARALLEL_SIZE
    gpu_memory_utilization: float = DEFAULT_GPU_MEMORY_UTILIZATION
    max_model_len: int = DEFAULT_MAX_MODEL_LEN
    seed: int = DEFAULT_SEED
    local_files_only: bool = True
    trust_remote_code: bool = True
    enable_thinking: bool = False
    max_tokens: int = DEFAULT_MAX_TOKENS
    stop_tokens: List[str] = field(default_factory=lambda: list(DEFAULT_STOP_TOKENS))

    def resolve_model_path(self) -> str:
        """Resolve model path between container path and host repo path."""
        if Path(self.model_path).exists():
            return self.model_path
        if Path(FALLBACK_LOCAL_MODEL_PATH).exists():
            return str(Path(FALLBACK_LOCAL_MODEL_PATH).resolve())
        return self.model_path

    def to_engine_kwargs(self) -> Dict[str, Any]:
        """Convert config to arguments required by vLLM LLM constructor."""
        resolved_path = self.resolve_model_path()
        kwargs: Dict[str, Any] = {
            "model": resolved_path,
            "tensor_parallel_size": self.tensor_parallel_size,
            "gpu_memory_utilization": self.gpu_memory_utilization,
            "max_model_len": self.max_model_len,
            "seed": self.seed,
            "trust_remote_code": self.trust_remote_code,
        }
        if self.local_files_only:
            kwargs["download_dir"] = resolved_path
            # Set offline environment variables for Hugging Face Hub & Transformers
            os.environ["HF_HUB_OFFLINE"] = "1"
            os.environ["TRANSFORMERS_OFFLINE"] = "1"
            os.environ["HF_DATASETS_OFFLINE"] = "1"
        return kwargs


def get_sampling_params(
    enable_thinking: bool = False,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    stop_tokens: Optional[List[str]] = None,
    seed: int = DEFAULT_SEED,
    **kwargs: Any,
) -> Any:
    """Build SamplingParams for vLLM supporting thinking mode vs deterministic mode.

    Args:
        enable_thinking: If True, uses exploratory sampling (temp=0.6, top_p=0.95, top_k=20).
                         If False, uses greedy deterministic decoding (temp=0.0).
        max_tokens: Maximum tokens to generate.
        stop_tokens: Optional list of stop string tokens.
        seed: Random seed for deterministic generation.
        **kwargs: Additional overrides for SamplingParams.

    Returns:
        vllm.SamplingParams instance (or dictionary fallback if vLLM is not installed).
    """
    stop = stop_tokens if stop_tokens is not None else list(DEFAULT_STOP_TOKENS)

    if enable_thinking:
        params: Dict[str, Any] = {
            "temperature": THINKING_TEMPERATURE,
            "top_p": THINKING_TOP_P,
            "top_k": THINKING_TOP_K,
            "max_tokens": max_tokens,
            "stop": stop,
            "seed": seed,
        }
    else:
        params = {
            "temperature": NON_THINKING_TEMPERATURE,
            "max_tokens": max_tokens,
            "stop": stop,
            "seed": seed,
        }

    params.update(kwargs)

    if _HAS_VLLM and SamplingParams is not None and not isinstance(SamplingParams, type(Any)):
        return SamplingParams(**params)
    return params


class Qwen3OfflineEngine:
    """Wrapper around vLLM for offline Qwen3-14B inference with dual-mode thinking support."""

    def __init__(self, config: Optional[VLLMConfig] = None, dry_run: bool = False):
        self.config = config or VLLMConfig()
        self.dry_run = dry_run or not _HAS_VLLM
        self.llm = None

        if self.config.local_files_only:
            os.environ["HF_HUB_OFFLINE"] = "1"
            os.environ["TRANSFORMERS_OFFLINE"] = "1"

        if not self.dry_run and _HAS_VLLM:
            self._init_engine()
        else:
            logger.info("Initializing Qwen3OfflineEngine in DRY-RUN / MOCK mode.")

    def _init_engine(self) -> None:
        """Instantiate offline vLLM engine."""
        engine_kwargs = self.config.to_engine_kwargs()
        logger.info(
            f"Loading offline Qwen3-14B vLLM engine with TP={self.config.tensor_parallel_size}, "
            f"mem_util={self.config.gpu_memory_utilization}, max_len={self.config.max_model_len}..."
        )
        self.llm = LLM(**engine_kwargs)
        logger.info("Offline Qwen3-14B vLLM engine loaded successfully.")

    def get_sampling_params(self, enable_thinking: Optional[bool] = None) -> Any:
        """Get sampling parameters according to thinking toggle."""
        thinking = self.config.enable_thinking if enable_thinking is None else enable_thinking
        return get_sampling_params(
            enable_thinking=thinking,
            max_tokens=self.config.max_tokens,
            stop_tokens=self.config.stop_tokens,
            seed=self.config.seed,
        )

    def generate(
        self,
        prompts: Union[str, List[str]],
        enable_thinking: Optional[bool] = None,
        sampling_params: Optional[Any] = None,
    ) -> List[str]:
        """Generate completions for one or more prompts."""
        if isinstance(prompts, str):
            prompts = [prompts]

        if self.dry_run or self.llm is None:
            return [
                "```python\nimport pandas as pd\n# Mock response\nfinal_result = 'abstain'\n```"
                for _ in prompts
            ]

        sp = sampling_params or self.get_sampling_params(enable_thinking=enable_thinking)
        outputs = self.llm.generate(prompts, sp)
        return [output.outputs[0].text for output in outputs]
