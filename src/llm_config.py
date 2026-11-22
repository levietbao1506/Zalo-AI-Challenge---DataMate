"""vLLM engine configuration and offline loader for Qwen/Qwen3-14B.

Designed for the Zalo AI Challenge 2026 (DataMate Track) offline evaluation container
running vllm/vllm-openai:v0.10.1.1 on dual RTX 5090 Blackwell GPUs (CUDA 12.8, PyTorch 2.7.1).
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)

DEFAULT_OFFLINE_MODEL_PATH = "/code/models/Qwen3-14B"
FALLBACK_LOCAL_MODEL_PATH = "models/Qwen3-14B"

THINKING_TEMPERATURE = 0.0
THINKING_TOP_P = 1.0
THINKING_TOP_K = -1

NON_THINKING_TEMPERATURE = 0.0

DEFAULT_SEED = 42
DEFAULT_MAX_MODEL_LEN = 32768
DEFAULT_GPU_MEMORY_UTILIZATION = 0.85
DEFAULT_TENSOR_PARALLEL_SIZE = int(os.environ.get("TENSOR_PARALLEL_SIZE", 2))
DEFAULT_MAX_TOKENS = 8192
DEFAULT_MAX_NUM_SEQS = 8

DEFAULT_STOP_TOKENS = [
    "<|im_end|>",
    "<|endoftext|>",
    "<tool_call>",
]

try:
    import torch
    _HAS_TORCH = True
except ImportError:
    torch = None
    _HAS_TORCH = False

try:
    from vllm import LLM, SamplingParams
    _HAS_VLLM = True
except ImportError:
    _HAS_VLLM = False
    LLM = Any
    SamplingParams = Any


def estimate_model_weight_gb(model_dir: Union[str, Path]) -> float:
    path = Path(model_dir)
    if not path.exists():
        return 0.0
    total_bytes = 0
    weight_extensions = {".safetensors", ".bin", ".pt"}
    for f in path.rglob("*"):
        if f.is_file() and f.suffix.lower() in weight_extensions:
            try:
                total_bytes += f.stat().st_size
            except OSError:
                pass
    return total_bytes / (1024 ** 3)


def detect_optimal_gpu_mode(
    model_dir: Union[str, Path],
    override_tp: Optional[int] = None,
) -> Tuple[str, int, int]:
    if override_tp is not None:
        return ("CUSTOM", override_tp, 1)

    gpu_count = 0
    if _HAS_TORCH and torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()

    if gpu_count < 2:
        return ("SINGLE_GPU", 1, 1)

    weight_gb = estimate_model_weight_gb(model_dir)
    is_quantized = False
    config_file = Path(model_dir) / "config.json"
    if config_file.is_file():
        try:
            cfg = json.loads(config_file.read_text(encoding="utf-8"))
            quantization_values = [
                cfg.get("quantization_config"),
                cfg.get("quantization"),
                cfg.get("quant_method"),
                cfg.get("torch_dtype"),
            ]
            quantization_text = json.dumps(quantization_values, ensure_ascii=True).lower()
            is_quantized = any(marker in quantization_text for marker in ("fp8", "float8", "awq", "gptq"))
        except Exception:
            pass

    if not is_quantized:
        model_name = Path(model_dir).name.lower()
        is_quantized = any(marker in model_name for marker in ("fp8", "float8", "awq", "gptq"))

    if is_quantized and 0.0 < weight_gb < 22.0:
        return ("DP", 1, 2)

    return ("TP", 2, 1)


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
    max_num_seqs: int = DEFAULT_MAX_NUM_SEQS
    enable_prefix_caching: bool = True
    stop_tokens: List[str] = field(default_factory=lambda: list(DEFAULT_STOP_TOKENS))

    def resolve_model_path(self) -> str:
        if Path(self.model_path).exists():
            return self.model_path
        if Path(FALLBACK_LOCAL_MODEL_PATH).exists():
            return str(Path(FALLBACK_LOCAL_MODEL_PATH).resolve())
        return self.model_path

    def to_engine_kwargs(self) -> Dict[str, Any]:
        resolved_path = self.resolve_model_path()
        kwargs: Dict[str, Any] = {
            "model": resolved_path,
            "tensor_parallel_size": self.tensor_parallel_size,
            "gpu_memory_utilization": self.gpu_memory_utilization,
            "max_model_len": self.max_model_len,
            "seed": self.seed,
            "max_num_seqs": self.max_num_seqs,
            "enable_prefix_caching": self.enable_prefix_caching,
            "trust_remote_code": self.trust_remote_code,
        }
        if self.local_files_only:
            kwargs["download_dir"] = resolved_path
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
            "top_p": 1.0,
            "max_tokens": max_tokens,
            "stop": stop,
            "seed": seed,
        }

    params.update(kwargs)

    if _HAS_VLLM and SamplingParams is not None and not isinstance(SamplingParams, type(Any)):
        return SamplingParams(**params)
    return params


def extract_reasoning_and_content(vllm_output_item: Any) -> Tuple[Optional[str], str]:
    if isinstance(vllm_output_item, str):
        full_text = vllm_output_item
        reasoning = None
        if "<think>" in full_text:
            after = full_text.split("<think>", 1)[1]
            if "</think>" in after:
                reasoning = after.split("</think>", 1)[0]
            else:
                reasoning = after
        return reasoning, full_text

    text = ""
    reasoning = None
    if hasattr(vllm_output_item, "outputs") and len(vllm_output_item.outputs) > 0:
        out_seq = vllm_output_item.outputs[0]
        text = getattr(out_seq, "text", "")
        if hasattr(out_seq, "reasoning_content") and out_seq.reasoning_content:
            reasoning = out_seq.reasoning_content
    elif hasattr(vllm_output_item, "text"):
        text = getattr(vllm_output_item, "text", "")
        if hasattr(vllm_output_item, "reasoning_content") and vllm_output_item.reasoning_content:
            reasoning = vllm_output_item.reasoning_content

    if reasoning is None and "<think>" in text:
        after = text.split("<think>", 1)[1]
        if "</think>" in after:
            reasoning = after.split("</think>", 1)[0]
        else:
            reasoning = after

    return reasoning, text


class Qwen3OfflineEngine:
    def __init__(self, config: Optional[VLLMConfig] = None, dry_run: bool = False):
        self.config = config or VLLMConfig()
        self.dry_run = dry_run or not _HAS_VLLM
        self.llm = None
        self.last_reasoning: List[Optional[str]] = []

        if self.config.local_files_only:
            os.environ["HF_HUB_OFFLINE"] = "1"
            os.environ["TRANSFORMERS_OFFLINE"] = "1"

        if not self.dry_run and _HAS_VLLM:
            self._init_engine()
        else:
            logger.info("Initializing Qwen3OfflineEngine in DRY-RUN / MOCK mode.")

    def _init_engine(self) -> None:
        engine_kwargs = self.config.to_engine_kwargs()
        logger.info(
            f"Loading offline Qwen3-14B vLLM engine with TP={self.config.tensor_parallel_size}, "
            f"mem_util={self.config.gpu_memory_utilization}, max_len={self.config.max_model_len}..."
        )
        self.llm = LLM(**engine_kwargs)
        logger.info("Offline Qwen3-14B vLLM engine loaded successfully.")

    def get_sampling_params(self, enable_thinking: Optional[bool] = None) -> Any:
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
        if isinstance(prompts, str):
            prompts = [prompts]

        if self.dry_run or self.llm is None:
            self.last_reasoning = [None for _ in prompts]
            return [
                "```python\nimport pandas as pd\nfinal_result = 'abstain'\n```"
                for _ in prompts
            ]

        sp = sampling_params or self.get_sampling_params(enable_thinking=enable_thinking)
        outputs = self.llm.generate(prompts, sp)
        res_texts: List[str] = []
        self.last_reasoning = []
        for out in outputs:
            r_content, t_content = extract_reasoning_and_content(out)
            self.last_reasoning.append(r_content)
            res_texts.append(t_content)
        return res_texts
