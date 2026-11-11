"""Agent managing local vLLM inference engine for offline code generation (Bước 3).

Configured with Qwen2.5-14B-Instruct and temperature=0.0 for deterministic, greedy code generation.
Includes code extraction, regex cleaning, and seamless dry-run/mock fallback
for running tests and developing without dedicated GPUs.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import re
import sys
from typing import Any, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Try importing vLLM
try:
    from vllm import LLM, SamplingParams
    _HAS_VLLM = True
except ImportError:
    _HAS_VLLM = False
    LLM = None
    SamplingParams = None

DEFAULT_MODEL_NAME = "Qwen/Qwen2.5-14B-Instruct"
DEFAULT_LOCAL_MODEL_PATH = "/app/model"
DEFAULT_TEMPERATURE = 0.0
DEFAULT_STOP_TOKENS = ["<|im_end|>", "</s>", "<|endoftext|>"]


def load_vllm_engine(
    model_path: str = DEFAULT_LOCAL_MODEL_PATH,
    temperature: float = DEFAULT_TEMPERATURE,
    max_tokens: int = 1024,
    gpu_memory_utilization: float = 0.90,
    max_model_len: int = 4096,
    tensor_parallel_size: int = 1,
    trust_remote_code: bool = True,
) -> Tuple[Any, Any]:
    """Load Qwen2.5-14B-Instruct (or local weights) using vLLM with temperature=0.0.

    Args:
        model_path: Path to model directory or HuggingFace repo ID (e.g. Qwen/Qwen2.5-14B-Instruct).
        temperature: Decoding temperature (default: 0.0 for greedy, deterministic generation).
        max_tokens: Maximum completion tokens (default: 1024).
        gpu_memory_utilization: Fraction of GPU VRAM allocated to vLLM (default: 0.90).
        max_model_len: Maximum model sequence length (default: 4096).
        tensor_parallel_size: Tensor parallel degree (default: 1 for <15B models).
        trust_remote_code: Trust remote code flag (default: True).

    Returns:
        Tuple of (LLM instance, SamplingParams instance).

    Raises:
        RuntimeError: If vLLM library is not installed or model loading fails.
    """
    if not _HAS_VLLM:
        raise RuntimeError(
            "Thư viện vLLM chưa được cài đặt. Hãy cài đặt qua 'pip install vllm' "
            "hoặc chạy trong môi trường Docker GPU của BTC."
        )

    logger.info(f"Initializing vLLM engine with model: '{model_path}'...")
    logger.info(f"Parameters: temperature={temperature}, max_model_len={max_model_len}, gpu_mem={gpu_memory_utilization}")

    llm = LLM(
        model=model_path,
        tensor_parallel_size=tensor_parallel_size,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=max_model_len,
        trust_remote_code=trust_remote_code,
        enforce_eager=False,
    )

    # Greedy decoding for reproducible, deterministic code generation
    sampling_params = SamplingParams(
        temperature=temperature,
        max_tokens=max_tokens,
        stop=DEFAULT_STOP_TOKENS,
    )

    logger.info("vLLM engine successfully loaded with temperature=0.0.")
    return llm, sampling_params


class VLLMAgent:
    """Manages local vLLM instance for high-throughput, offline Python code generation."""

    def __init__(
        self,
        model_path: str = DEFAULT_LOCAL_MODEL_PATH,
        temperature: float = DEFAULT_TEMPERATURE,
        gpu_memory_utilization: float = 0.90,
        max_model_len: int = 4096,
        tensor_parallel_size: int = 1,
        dry_run: bool = False,
    ):
        """Initialize vLLM engine or mock engine for local testing.

        Args:
            model_path: Local path to model weights directory or HuggingFace repo ID.
            temperature: Sampling temperature (default: 0.0 for deterministic greedy decoding).
            gpu_memory_utilization: Fraction of GPU memory allocated to vLLM (default: 0.90).
            max_model_len: Context window limit (default: 4096).
            tensor_parallel_size: Tensor parallel degree (default: 1 for <15B models).
            dry_run: If True, forces dry-run mock mode without loading weights.
        """
        self.model_path = model_path
        self.temperature = temperature
        self.dry_run = dry_run or not _HAS_VLLM
        self.llm = None
        self.sampling_params = None

        if not self.dry_run:
            self._init_vllm(gpu_memory_utilization, max_model_len, tensor_parallel_size)
        else:
            logger.info("Initializing Agent in DRY-RUN / MOCK mode (vLLM not loaded).")

    def _init_vllm(
        self,
        gpu_memory_utilization: float,
        max_model_len: int,
        tensor_parallel_size: int,
    ) -> None:
        """Initialize the real vLLM engine."""
        path_obj = Path(self.model_path)
        # Check if local path exists or if it's a HuggingFace hub model ID
        if not path_obj.exists() and not ("/" in self.model_path and not self.model_path.startswith("/")):
            logger.warning(
                f"Model path '{self.model_path}' not found on local disk. Falling back to Mock mode for testing."
            )
            self.dry_run = True
            return

        try:
            self.llm, self.sampling_params = load_vllm_engine(
                model_path=self.model_path,
                temperature=self.temperature,
                gpu_memory_utilization=gpu_memory_utilization,
                max_model_len=max_model_len,
                tensor_parallel_size=tensor_parallel_size,
            )
        except Exception as e:
            logger.error(f"Failed to initialize vLLM engine: {e}. Falling back to dry-run mock mode.")
            self.dry_run = True

    def generate_code(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """Generate Python code to solve the tabular QA question.

        Args:
            prompt: User prompt containing schema context, reference date, and question.
            system_prompt: System prompt containing role instructions and few-shots.

        Returns:
            Extracted Python code string ready for sandboxed execution.
        """
        if self.dry_run or self.llm is None:
            return self._mock_generate_code(prompt)

        # Format chat prompt according to ChatML / Qwen instruct template
        if system_prompt:
            full_text = (
                f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
                f"<|im_start|>user\n{prompt}<|im_end|>\n"
                f"<|im_start|>assistant\n```python\n"
            )
        else:
            full_text = (
                f"<|im_start|>user\n{prompt}<|im_end|>\n"
                f"<|im_start|>assistant\n```python\n"
            )

        outputs = self.llm.generate([full_text], self.sampling_params)
        raw_output = outputs[0].outputs[0].text

        # Prepend ```python\n if omitted because we guided the prefix
        if not raw_output.startswith("```"):
            raw_output = f"```python\n{raw_output}"

        return self.extract_python_code(raw_output)

    def extract_python_code(self, text: str) -> str:
        """Extract clean executable code from LLM completion text.

        Extracts blocks bounded by ```python ... ``` or ``` ... ```.

        Args:
            text: Raw LLM output containing code blocks or markdown.

        Returns:
            Extracted clean Python code string.
        """
        if not text:
            return "result = None"

        # Match ```python ... ``` or ```py ... ```
        m_py = re.search(r"```(?:python|py)\s*\n(.*?)```", text, re.DOTALL | re.IGNORECASE)
        if m_py:
            return m_py.group(1).strip()

        # Match generic ``` ... ```
        m_generic = re.search(r"```\s*\n(.*?)```", text, re.DOTALL)
        if m_generic:
            return m_generic.group(1).strip()

        # If LLM opened ```python but didn't close it with ```
        if "```" in text:
            parts = text.split("```")
            if len(parts) >= 2:
                code_part = parts[1]
                # Strip language tag like 'python\n'
                if code_part.lower().startswith("python"):
                    code_part = code_part[6:].lstrip("\r\n")
                elif code_part.lower().startswith("py"):
                    code_part = code_part[2:].lstrip("\r\n")
                if code_part.strip():
                    return code_part.strip()

        # Fallback: scan lines
        lines = text.splitlines()
        code_lines: List[str] = []
        in_block = False
        for line in lines:
            if line.strip().startswith("```"):
                in_block = not in_block
                continue
            if in_block:
                code_lines.append(line)

        if code_lines:
            return "\n".join(code_lines).strip()

        return text.strip()

    def _mock_generate_code(self, prompt: str) -> str:
        """Generate a basic, safe fallback script when running in dry-run mode."""
        logger.debug("Generating mock code for dry-run testing.")
        return (
            "# Auto-generated Mock Code for Offline Testing\n"
            "try:\n"
            "    # By default safely abstain to avoid penalized accuracy penalty\n"
            "    result = None\n"
            "except Exception:\n"
            "    result = None\n"
        )


def main() -> None:
    """CLI test script to load vLLM model and verify greedy generation."""
    parser = argparse.ArgumentParser(description="Load Qwen2.5-14B-Instruct via vLLM with temperature=0.0")
    parser.add_argument("--model_path", type=str, default=DEFAULT_LOCAL_MODEL_PATH, help="Model weights path")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature")
    parser.add_argument("--dry_run", action="store_true", help="Force dry-run mock mode")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    agent = VLLMAgent(model_path=args.model_path, temperature=args.temperature, dry_run=args.dry_run)

    test_prompt = "Hãy viết code pandas tính tổng doanh thu từ sales.csv"
    code = agent.generate_code(test_prompt)
    print("\n--- GENERATED CODE ---\n")
    print(code)


if __name__ == "__main__":
    main()
