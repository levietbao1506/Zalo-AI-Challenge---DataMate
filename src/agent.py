"""Agent managing local vLLM inference engine for offline code generation.

Configured with temperature=0.0 for deterministic, greedy code generation.
Includes code extraction, regex cleaning, and seamless dry-run/mock fallback
for running tests and developing without dedicated GPUs.
"""

from __future__ import annotations

import logging
from pathlib import Path
import re
from typing import List, Optional

logger = logging.getLogger(__name__)

# Try importing vLLM
try:
    from vllm import LLM, SamplingParams
    _HAS_VLLM = True
except ImportError:
    _HAS_VLLM = False
    LLM = None
    SamplingParams = None


class VLLMAgent:
    """Manages local vLLM instance for high-throughput, offline Python code generation."""

    def __init__(
        self,
        model_path: str = "/app/model",
        gpu_memory_utilization: float = 0.90,
        max_model_len: int = 4096,
        tensor_parallel_size: int = 1,
        dry_run: bool = False,
    ):
        """Initialize vLLM engine or mock engine for local testing.

        Args:
            model_path: Local path to model weights directory.
            gpu_memory_utilization: Fraction of GPU memory allocated to vLLM (default: 0.90).
            max_model_len: Context window limit (default: 4096).
            tensor_parallel_size: Tensor parallel degree (default: 1 for <15B models).
            dry_run: If True, forces dry-run mock mode without loading weights.
        """
        self.model_path = model_path
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
        if not path_obj.exists():
            logger.warning(
                f"Model path {self.model_path} not found. Falling back to Mock mode for testing."
            )
            self.dry_run = True
            return

        logger.info(f"Loading local vLLM engine from '{self.model_path}'...")
        try:
            self.llm = LLM(
                model=str(path_obj),
                tensor_parallel_size=tensor_parallel_size,
                gpu_memory_utilization=gpu_memory_utilization,
                max_model_len=max_model_len,
                trust_remote_code=True,
                enforce_eager=False,
            )
            # Greedy decoding for reproducible, deterministic code generation
            self.sampling_params = SamplingParams(
                temperature=0.0,
                max_tokens=1024,
                stop=["<|im_end|>", "</s>", "<|endoftext|>"],
            )
            logger.info("vLLM engine initialized successfully with temperature=0.0.")
        except Exception as e:
            logger.error(f"Failed to initialize vLLM engine: {e}. Falling back to dry-run mock mode.")
            self.dry_run = True

    def generate_code(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """Generate Python code to solve the tabular QA question.

        Args:
            prompt: User prompt containing schema context and question.
            system_prompt: System prompt containing role instructions and few-shots.

        Returns:
            Extracted Python code string ready for sandboxed execution.
        """
        if self.dry_run or self.llm is None:
            return self._mock_generate_code(prompt)

        # Format chat prompt according to common instruct templates (e.g. ChatML / Qwen)
        full_text = prompt
        if system_prompt:
            full_text = f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n```python\n"
        else:
            full_text = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n```python\n"

        outputs = self.llm.generate([full_text], self.sampling_params)
        raw_output = outputs[0].outputs[0].text
        # Prepend ```python\n since we guided the prefix
        if not raw_output.startswith("```"):
            raw_output = f"```python\n{raw_output}"

        return self.extract_python_code(raw_output)

    def extract_python_code(self, text: str) -> str:
        """Extract clean executable code from LLM completion text.

        Args:
            text: Raw LLM output containing code blocks or markdown.

        Returns:
            Extracted clean Python code string.
        """
        if not text:
            return "result = None"

        # Match ```python ... ```
        m_py = re.search(r"```(?:python|py)\s*\n(.*?)```", text, re.DOTALL | re.IGNORECASE)
        if m_py:
            return m_py.group(1).strip()

        # Match any ``` ... ```
        m_generic = re.search(r"```\s*\n(.*?)```", text, re.DOTALL)
        if m_generic:
            return m_generic.group(1).strip()

        # If LLM didn't close backticks or gave raw code
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
        # If question contains simple arithmetic or mentions column, create basic fallback
        return (
            "# Auto-generated Mock Code for Offline Testing\n"
            "try:\n"
            "    # By default safely abstain to avoid penalized accuracy penalty\n"
            "    result = None\n"
            "except Exception:\n"
            "    result = None\n"
        )
