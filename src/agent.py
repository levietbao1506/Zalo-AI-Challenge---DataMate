"""Agent managing local vLLM, llama-cpp-python (GGUF), or Ollama API inference (Bước 3).

Supports local vLLM models (e.g. Qwen2.5-Coder), GGUF models via ``llama-cpp-python``
(when ``model_path`` ends with ``.gguf``), and local Ollama API models prefixed with
``ollama:`` with temperature=0.0 for deterministic code generation.
"""

from __future__ import annotations

import argparse
from datetime import date
import logging
from pathlib import Path
import re
import sys
from typing import Any, Dict, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)

# Try importing vLLM
try:
    from vllm import LLM, SamplingParams
    _HAS_VLLM = True
except ImportError:
    _HAS_VLLM = False
    LLM = None
    SamplingParams = None

# Try importing llama-cpp-python for GGUF inference
try:
    from llama_cpp import Llama
except ImportError:
    Llama = None  # type: ignore[assignment]

# Try importing requests for Ollama API support
try:
    import requests
except ImportError:
    requests = None  # type: ignore[assignment]

DEFAULT_MODEL_NAME = "Qwen/Qwen2.5-Coder-3B-Instruct"
DEFAULT_LOCAL_MODEL_PATH = "./models/Qwen2.5-Coder-3B-Instruct"
DEFAULT_OLLAMA_URL = "http://localhost:11434/api/generate"
DEFAULT_TEMPERATURE = 0.0
DEFAULT_STOP_TOKENS = ["<|im_end|>", "</s>", "<|endoftext|>"]


def load_vllm_engine(
    model_path: str = DEFAULT_LOCAL_MODEL_PATH,
    temperature: float = DEFAULT_TEMPERATURE,
    max_tokens: int = 1024,
    gpu_memory_utilization: float = 0.85,
    max_model_len: int = 4096,
    tensor_parallel_size: int = 1,
    trust_remote_code: bool = True,
) -> Tuple[Any, Any]:
    """Load Qwen2.5-14B-Instruct (or local weights) using vLLM with temperature=0.0.

    Args:
        model_path: Path to model directory or HuggingFace repo ID (e.g. Qwen/Qwen2.5-14B-Instruct).
        temperature: Decoding temperature (default: 0.0 for greedy, deterministic generation).
        max_tokens: Maximum completion tokens (default: 1024).
        gpu_memory_utilization: Fraction of GPU VRAM allocated to vLLM (default: 0.85).
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


class CodeAgent:
    """Manages local vLLM, llama.cpp (GGUF), or Ollama API backend for Python code generation."""

    def __init__(
        self,
        model_path: str = DEFAULT_LOCAL_MODEL_PATH,
        temperature: float = DEFAULT_TEMPERATURE,
        gpu_memory_utilization: float = 0.85,
        max_model_len: int = 4096,
        tensor_parallel_size: int = 1,
        trust_remote_code: bool = True,
        dry_run: bool = False,
    ):
        """Initialize llama.cpp, vLLM engine, Ollama HTTP client, or mock engine.

        Args:
            model_path: Path to ``.gguf`` file, vLLM directory/repo ID, or ``ollama:<model>``.
            temperature: Sampling temperature (default: 0.0 for deterministic greedy decoding).
            gpu_memory_utilization: Fraction of GPU memory allocated to vLLM (default: 0.85).
            max_model_len: Context window limit (default: 4096).
            tensor_parallel_size: Tensor parallel degree (default: 1 for <15B models).
            trust_remote_code: Whether to pass ``trust_remote_code=True`` to vLLM (default: True).
            dry_run: If True, forces dry-run mock mode without calling a model backend.
        """
        self.model_path = model_path
        self.temperature = temperature
        self.trust_remote_code = trust_remote_code
        self.ollama_url = DEFAULT_OLLAMA_URL
        self.llm = None
        self.sampling_params = None

        if isinstance(model_path, str) and model_path.startswith("ollama:"):
            self.backend = "ollama"
            self.model_name = model_path[len("ollama:"):].strip()
            if not self.model_name:
                raise ValueError("Ollama model name cannot be empty after 'ollama:' prefix.")
            self.dry_run = dry_run
            if self.dry_run:
                logger.info("Initializing Agent in DRY-RUN / MOCK mode (Ollama disabled).")
            else:
                logger.info(
                    f"Initializing Ollama backend with model: '{self.model_name}' "
                    f"(endpoint={self.ollama_url}, temperature={self.temperature})."
                )
        elif isinstance(model_path, str) and model_path.lower().endswith(".gguf"):
            self.backend = "llama_cpp"
            self.model_name = model_path
            self.dry_run = False
            if not dry_run:
                self._init_llama_cpp(n_ctx=8192)
            else:
                self.dry_run = True
                logger.info("Initializing Agent in DRY-RUN / MOCK mode (llama.cpp disabled).")
        else:
            self.backend = "vllm"
            self.model_name = model_path
            self.dry_run = dry_run or not _HAS_VLLM

            if not self.dry_run:
                self._init_vllm(
                    gpu_memory_utilization,
                    max_model_len,
                    tensor_parallel_size,
                    trust_remote_code=trust_remote_code,
                )
            else:
                logger.info("Initializing Agent in DRY-RUN / MOCK mode (vLLM not loaded).")

    def _init_llama_cpp(self, n_ctx: int = 8192) -> None:
        """Initialize local GGUF model via llama-cpp-python."""
        llama_cls = Llama
        if llama_cls is None:
            from llama_cpp import Llama as llama_cls

        logger.info(f"Initializing llama_cpp backend with GGUF model: '{self.model_path}' (n_ctx={n_ctx})...")
        self.llm = llama_cls(model_path=self.model_path, n_ctx=n_ctx, verbose=False)
        logger.info("llama_cpp GGUF model loaded successfully.")

    def _init_vllm(
        self,
        gpu_memory_utilization: float,
        max_model_len: int,
        tensor_parallel_size: int,
        trust_remote_code: bool = True,
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
                trust_remote_code=trust_remote_code,
            )
        except Exception as e:
            logger.error(f"Failed to initialize vLLM engine: {e}. Falling back to dry-run mock mode.")
            self.dry_run = True

    def _generate_ollama(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """Send a single completion request to the local Ollama HTTP API."""
        if requests is None:
            raise RuntimeError(
                "Thư viện 'requests' chưa được cài đặt. Hãy cài đặt 'requests' để sử dụng Ollama backend."
            )

        payload: dict[str, Any] = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": self.temperature},
        }
        if system_prompt is not None:
            payload["system"] = system_prompt

        try:
            response = requests.post(self.ollama_url, json=payload, timeout=None)
            response.raise_for_status()
            return response.json()["response"]
        except requests.RequestException as e:
            logger.error(f"Ollama API request failed for model '{self.model_name}': {e}")
            raise

    def generate_code(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """Generate Python code to solve the tabular QA question.

        Args:
            prompt: User prompt containing schema context, reference date, and question.
            system_prompt: System prompt containing role instructions and few-shots.

        Returns:
            Generated response text (for Ollama/llama_cpp) or extracted Python code (for vLLM).
        """
        if self.dry_run:
            return self._mock_generate_code(prompt)

        if self.backend == "ollama":
            return self._generate_ollama(prompt, system_prompt=system_prompt)

        if self.backend == "llama_cpp":
            try:
                system_content = (
                    system_prompt
                    if system_prompt
                    else "You are a Python Data Analyst. Output ONLY valid executable pandas code inside a ```python block. Do not provide any explanations."
                )
                messages = [
                    {"role": "system", "content": system_content},
                    {"role": "user", "content": prompt},
                ]
                response = self.llm.create_chat_completion(
                    messages=messages,
                    max_tokens=1024,
                    temperature=0.0,
                )
                return response["choices"][0]["message"]["content"]
            except Exception as e:
                print(f"❌ Llama CPP Generation Error: {e}")
                return ""

        if self.llm is None:
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

    def generate_code_batch(
        self,
        prompts: List[str],
        system_prompt: Optional[str] = None,
    ) -> List[str]:
        """Generate Python completions for a batch of prompts.

        Uses single-pass vLLM batch inference when ``backend == "vllm"`` or loops
        sequentially via :meth:`generate_code` when ``backend`` is ``"ollama"``
        or ``"llama_cpp"``.

        Args:
            prompts: Ordered list of prompts to feed into inference.
            system_prompt: Optional system prompt to forward/wrap with prompts.

        Returns:
            List of text completions in the exact same order as ``prompts``.
        """
        if not prompts:
            return []

        if self.dry_run:
            return [self._mock_generate_code(prompt) for prompt in prompts]

        if self.backend == "ollama":
            if system_prompt is None:
                return [self.generate_code(prompt) for prompt in prompts]
            return [
                self.generate_code(prompt, system_prompt=system_prompt)
                for prompt in prompts
            ]

        if self.backend == "llama_cpp":
            results = []
            for i, p in enumerate(prompts):
                print(f"  ⏳ Llama-CPP đang xử lý câu {i+1}/{len(prompts)}...")
                results.append(self.generate_code(p))
            return results

        if self.llm is None:
            return [self._mock_generate_code(prompt) for prompt in prompts]

        if system_prompt is not None:
            batch_inputs = [
                (
                    f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
                    f"<|im_start|>user\n{prompt}<|im_end|>\n"
                    f"<|im_start|>assistant\n"
                )
                for prompt in prompts
            ]
        else:
            batch_inputs = prompts

        outputs = self.llm.generate(batch_inputs, self.sampling_params)
        return [
            output.outputs[0].text if getattr(output, "outputs", None) else ""
            for output in outputs
        ]

    def generate_with_retry(
        self,
        initial_prompt: str,
        data_dir: str,
        max_retries: int = 3,
        system_prompt: Optional[str] = None,
        file_map: Optional[Dict[str, str]] = None,
        as_of_date: Optional[Union[str, date]] = None,
        timeout_seconds: Optional[float] = None,
        base_prompt: Optional[str] = None,
    ) -> tuple[bool, Any, str]:
        """Generate code and execute with self-correction feedback loop up to max_retries.

        If execution fails, feeds the exact error message and failed code back
        to the model.
        """
        from functools import partial
        from src import executor

        single_fn = partial(self.generate_code, system_prompt=system_prompt)
        return executor.execute_with_retry(
            initial_prompt=initial_prompt,
            generate_fn=single_fn,
            data_dir=data_dir,
            max_retries=max_retries,
            file_map=file_map,
            as_of_date=as_of_date,
            timeout_seconds=timeout_seconds,
            base_prompt=base_prompt or initial_prompt,
        )

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


# Backwards-compatible alias
VLLMAgent = CodeAgent


def main() -> None:
    """CLI test script to load vLLM model and verify greedy generation."""
    parser = argparse.ArgumentParser(description="Load Qwen2.5-14B-Instruct via vLLM with temperature=0.0")
    parser.add_argument("--model_path", type=str, default=DEFAULT_LOCAL_MODEL_PATH, help="Model weights path")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature")
    parser.add_argument("--dry_run", action="store_true", help="Force dry-run mock mode")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    agent = CodeAgent(model_path=args.model_path, temperature=args.temperature, dry_run=args.dry_run)

    test_prompt = "Hãy viết code pandas tính tổng doanh thu từ sales.csv"
    code = agent.generate_code(test_prompt)
    print("\n--- GENERATED CODE ---\n")
    print(code)


if __name__ == "__main__":
    main()
