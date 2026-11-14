"""Agent managing local offline vLLM inference for Qwen/Qwen3-14B (Bước 3).

Supports local offline vLLM inference with temperature=0.0 for deterministic, reproducible code generation.
"""

from __future__ import annotations

import argparse
import os
from datetime import date
import logging
import math
from pathlib import Path
import re
import sys
from typing import Any, Dict, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)

DEFAULT_ABSTAIN_THRESHOLD: float = 0.6
ABSTAIN_THRESHOLD: float = DEFAULT_ABSTAIN_THRESHOLD


def extract_thinking(text: str) -> Optional[str]:
    """Extract raw reasoning content from model output (<think>...</think>).

    Captures content between <think> and </think> (or from <think> onwards if unclosed),
    preserving the exact raw reasoning without summarization or truncation.
    """
    if not text or not isinstance(text, str):
        return None
    if "<think>" in text:
        after_think = text.split("<think>", 1)[1]
        if "</think>" in after_think:
            thinking_body = after_think.split("</think>", 1)[0]
        else:
            thinking_body = after_think
        stripped = thinking_body.strip()
        return stripped if stripped else None
    return None

# Try importing vLLM
try:
    from vllm import LLM, SamplingParams
    _HAS_VLLM = True
except ImportError:
    _HAS_VLLM = False
    LLM = None
    SamplingParams = None

# Try importing llama-cpp-python for GGUF inference (optional fallback)
try:
    from llama_cpp import Llama
except ImportError:
    Llama = None  # type: ignore[assignment]


DEFAULT_MODEL_NAME = "Qwen/Qwen3-14B"
DEFAULT_LOCAL_MODEL_PATH = "/code/models/Qwen3-14B" if Path("/code/models/Qwen3-14B").exists() else "./models/Qwen3-14B"
DEFAULT_TEMPERATURE = 0.0
DEFAULT_STOP_TOKENS = ["<|im_end|>", "</s>", "<|endoftext|>"]


def load_vllm_engine(
    model_path: str = DEFAULT_LOCAL_MODEL_PATH,
    temperature: float = DEFAULT_TEMPERATURE,
    max_tokens: int = 1024,
    gpu_memory_utilization: float = 0.85,
    max_model_len: int = 4096,
    tensor_parallel_size: Optional[int] = None,
    trust_remote_code: bool = True,
) -> Tuple[Any, Any]:
    """Load Qwen3-14B (or local weights) using vLLM with temperature=0.0.

    Args:
        model_path: Path to model directory or HuggingFace repo ID (e.g. Qwen/Qwen3-14B).
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

    # Resolve tensor parallel degree
    if tensor_parallel_size is None:
        tp_env = os.environ.get("TENSOR_PARALLEL_SIZE")
        if tp_env:
            tensor_parallel_size = int(tp_env)
        elif Path("/code").exists() or (isinstance(model_path, str) and model_path.startswith("/code")):
            tensor_parallel_size = 2
        else:
            tensor_parallel_size = 1

    # Auto-adjust tensor_parallel_size based on available CUDA devices
    try:
        import torch
        if torch.cuda.is_available():
            available_gpus = torch.cuda.device_count()
            if available_gpus < tensor_parallel_size:
                logger.warning(
                    f"Requested tensor_parallel_size={tensor_parallel_size} but only {available_gpus} GPU(s) available. "
                    f"Adjusting tensor_parallel_size to {available_gpus}."
                )
                tensor_parallel_size = max(1, available_gpus)
    except Exception:
        pass

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
    """Manages local offline vLLM (or optional llama.cpp GGUF) backend for Python code generation."""

    def __init__(
        self,
        model_path: str = DEFAULT_LOCAL_MODEL_PATH,
        temperature: float = DEFAULT_TEMPERATURE,
        gpu_memory_utilization: float = 0.85,
        max_model_len: int = 4096,
        tensor_parallel_size: Optional[int] = None,
        trust_remote_code: bool = True,
        dry_run: bool = False,
        abstain_threshold: float = DEFAULT_ABSTAIN_THRESHOLD,
    ):
        """Initialize vLLM engine, llama.cpp GGUF engine, or mock engine.

        Args:
            model_path: Path to vLLM directory or .gguf file (default: Qwen/Qwen3-14B).
            temperature: Sampling temperature (default: 0.0 for deterministic greedy decoding).
            gpu_memory_utilization: Fraction of GPU memory allocated to vLLM (default: 0.85).
            max_model_len: Context window limit (default: 4096).
            tensor_parallel_size: Tensor parallel degree (default: 2 or from TENSOR_PARALLEL_SIZE).
            trust_remote_code: Whether to pass ``trust_remote_code=True`` to vLLM (default: True).
            dry_run: If True, forces dry-run mock mode without calling a model backend.
            abstain_threshold: Confidence threshold below which answers are downgraded to abstain (default: 0.6).
        """
        self.model_path = model_path
        self.temperature = temperature
        self.trust_remote_code = trust_remote_code
        self.abstain_threshold = abstain_threshold
        self.llm = None
        self.sampling_params = None
        self.last_thinking: Optional[str] = None
        self.last_batch_thinking: List[Optional[str]] = []

        # Resolve tensor parallel degree (default: 2 in Docker / TENSOR_PARALLEL_SIZE, 1 otherwise)
        if tensor_parallel_size is None:
            tp_env = os.environ.get("TENSOR_PARALLEL_SIZE")
            if tp_env:
                tensor_parallel_size = int(tp_env)
            elif Path("/code").exists() or (isinstance(model_path, str) and model_path.startswith("/code")):
                tensor_parallel_size = 2
            else:
                tensor_parallel_size = 1

        if isinstance(model_path, str) and model_path.lower().endswith(".gguf"):
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

    def generate_code(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """Generate Python code to solve the tabular QA question.

        Args:
            prompt: User prompt containing schema context, reference date, and question.
            system_prompt: System prompt containing role instructions and few-shots.

        Returns:
            Extracted Python code.
        """
        if self.dry_run:
            mock_res = self._mock_generate_code(prompt)
            self.last_thinking = extract_thinking(mock_res)
            return mock_res

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
                msg = response["choices"][0]["message"]
                content = msg.get("content", "")
                self.last_thinking = msg.get("reasoning_content") or extract_thinking(content)
                return content
            except Exception as e:
                print(f"❌ Llama CPP Generation Error: {e}")
                self.last_thinking = None
                return ""

        if self.llm is None:
            mock_res = self._mock_generate_code(prompt)
            self.last_thinking = extract_thinking(mock_res)
            return mock_res

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
        out_obj = outputs[0].outputs[0]
        raw_output = out_obj.text
        self.last_thinking = getattr(out_obj, "reasoning_content", None) or extract_thinking(raw_output)

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
        sequentially via :meth:`generate_code` when ``backend == "llama_cpp"``.

        Args:
            prompts: Ordered list of prompts to feed into inference.
            system_prompt: Optional system prompt to forward/wrap with prompts.

        Returns:
            List of text completions in the exact same order as ``prompts``.
        """
        if not prompts:
            self.last_batch_thinking = []
            return []

        if self.dry_run:
            results = [self._mock_generate_code(prompt) for prompt in prompts]
            self.last_batch_thinking = [extract_thinking(r) for r in results]
            return results

        if self.backend == "llama_cpp":
            results = []
            thinkings = []
            for i, p in enumerate(prompts):
                print(f"  ⏳ Llama-CPP đang xử lý câu {i+1}/{len(prompts)}...")
                if system_prompt is not None:
                    code = self.generate_code(p, system_prompt=system_prompt)
                else:
                    code = self.generate_code(p)
                results.append(code)
                thinkings.append(self.last_thinking)
            self.last_batch_thinking = thinkings
            return results

        if self.llm is None:
            results = [self._mock_generate_code(prompt) for prompt in prompts]
            self.last_batch_thinking = [extract_thinking(r) for r in results]
            return results

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
        results = []
        thinkings = []
        for output in outputs:
            if getattr(output, "outputs", None):
                out_obj = output.outputs[0]
                text = out_obj.text
                reasoning = getattr(out_obj, "reasoning_content", None) or extract_thinking(text)
                results.append(text)
                thinkings.append(reasoning)
            else:
                results.append("")
                thinkings.append(None)
        self.last_batch_thinking = thinkings
        return results

    def generate_code_with_thinking(
        self, prompt: str, system_prompt: Optional[str] = None
    ) -> tuple[str, Optional[str]]:
        """Generate code and return tuple of (code, thinking)."""
        code = self.generate_code(prompt, system_prompt=system_prompt)
        return code, self.last_thinking

    def generate_code_batch_with_thinking(
        self, prompts: List[str], system_prompt: Optional[str] = None
    ) -> tuple[List[str], List[Optional[str]]]:
        """Generate batch completions and return tuple of (codes, thinkings)."""
        codes = self.generate_code_batch(prompts, system_prompt=system_prompt)
        return codes, list(self.last_batch_thinking)

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
        use_subprocess: bool = False,
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
            use_subprocess=use_subprocess,
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

    def verify_prediction(
        self,
        execution_success: bool,
        raw_result: Any = None,
        result: Any = None,
        q_type: Optional[str] = None,
        code_str: Optional[str] = None,
        retry_results: Optional[List[Any]] = None,
        error_msg: Optional[str] = None,
        threshold: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Apply verification gate using configured abstain threshold.

        Guards against negative PA score (-1 vs 0) by downgrading to {"type": "abstain"}
        when execution fails, result is empty/NaN, schema constraints are violated,
        or retries yielded conflicting outputs.
        """
        eff_threshold = threshold if threshold is not None else self.abstain_threshold
        val = raw_result if raw_result is not None else result
        return verification_gate(
            execution_success=execution_success,
            result=val,
            q_type=q_type,
            code_str=code_str,
            retry_results=retry_results,
            error_msg=error_msg,
            threshold=eff_threshold,
        )

    # Alias
    verification_gate = verify_prediction


def verification_gate(
    execution_success: bool,
    result: Any = None,
    raw_result: Any = None,
    q_type: Optional[str] = None,
    code_str: Optional[str] = None,
    retry_results: Optional[List[Any]] = None,
    error_msg: Optional[str] = None,
    threshold: Optional[float] = None,
) -> Dict[str, Any]:
    """Verification gate to guard against negative Penalized Accuracy scores (-1 vs 0).

    a) If code execution still raises an exception after retries, or returns
       NaN/None/empty results on an aggregation query, emits {"type": "abstain", "answer": "abstain"}.
    b) Runs lightweight verification on output schema/type constraints and
       filtered row counts; if uncertain or conflicting across retries, downgrades
       to {"type": "abstain", "answer": "abstain"} based on ABSTAIN_THRESHOLD.

    Args:
        execution_success: Whether sandbox execution succeeded.
        result: Raw or intermediate result from code execution.
        q_type: Expected question type (e.g., 'number', 'percent', 'string', 'date', 'boolean', 'list', 'table').
        code_str: Executed Python code string.
        retry_results: List of result values obtained across attempts.
        error_msg: Traceback or error message on failure.
        threshold: Confidence cutoff threshold (default: ABSTAIN_THRESHOLD).

    Returns:
        Dict with "type", "answer", "confidence", "passed", and "reason".
    """
    from src import executor, formatter, evaluator

    eff_threshold = float(threshold) if isinstance(threshold, (int, float)) else ABSTAIN_THRESHOLD
    result = result if result is not None else raw_result

    # a.1 Hard execution failure check
    if not execution_success:
        return {
            "type": "abstain",
            "answer": "abstain",
            "confidence": 0.0,
            "passed": False,
            "reason": f"Execution failed after retries: {error_msg or 'runtime exception'}",
        }

    # a.2 None check
    if result is None:
        return {
            "type": "abstain",
            "answer": "abstain",
            "confidence": 0.0,
            "passed": False,
            "reason": "Execution returned None",
        }

    # a.3 Float NaN or Inf check
    if isinstance(result, float) and (math.isnan(result) or math.isinf(result)):
        return {
            "type": "abstain",
            "answer": "abstain",
            "confidence": 0.0,
            "passed": False,
            "reason": "Execution returned NaN or infinite float",
        }

    # a.4 Logical empty check (DataFrame, Series, list, dict, empty string)
    if executor.is_empty_result(result):
        return {
            "type": "abstain",
            "answer": "abstain",
            "confidence": 0.0,
            "passed": False,
            "reason": "Execution returned logically empty container or string",
        }

    # a.5 Explicit abstain / unanswerable strings
    if isinstance(result, str):
        s_norm = result.strip().lower()
        if s_norm in ("abstain", "null", "none", "nan", "không có", "không xác định", "n/a", "na", ""):
            return {
                "type": "abstain",
                "answer": "abstain",
                "confidence": 0.0,
                "passed": False,
                "reason": f"Result explicitly indicates abstain: {result}",
            }
        if s_norm == "unanswerable" or q_type == "unanswerable":
            return {
                "type": "unanswerable",
                "answer": "unanswerable",
                "confidence": 1.0,
                "passed": True,
                "reason": "Question is unanswerable",
            }

    confidence = 1.0
    reason_notes: List[str] = []

    # b.1 Output schema & type constraints check
    formatted_answer: Any = result
    if q_type == "number":
        try:
            formatted_answer = formatter.format_number(result)
        except Exception as e:
            return {
                "type": "abstain",
                "answer": "abstain",
                "confidence": 0.0,
                "passed": False,
                "reason": f"Schema constraint violated for number: {e}",
            }
    elif q_type == "percent":
        try:
            formatted_answer = formatter.format_percent(result)
        except Exception as e:
            return {
                "type": "abstain",
                "answer": "abstain",
                "confidence": 0.0,
                "passed": False,
                "reason": f"Schema constraint violated for percent: {e}",
            }
    elif q_type == "date":
        try:
            formatted_answer = formatter.format_date(result)
            if not (re.match(r"^\d{4}-\d{2}-\d{2}$", str(formatted_answer)) or re.match(r"^\d{4}-\d{2}$", str(formatted_answer))):
                confidence -= 0.5
                reason_notes.append("Date string does not match standard ISO format")
        except Exception as e:
            return {
                "type": "abstain",
                "answer": "abstain",
                "confidence": 0.0,
                "passed": False,
                "reason": f"Schema constraint violated for date: {e}",
            }
    elif q_type == "boolean":
        try:
            formatted_answer = formatter.format_boolean(result)
        except Exception as e:
            return {
                "type": "abstain",
                "answer": "abstain",
                "confidence": 0.0,
                "passed": False,
                "reason": f"Schema constraint violated for boolean: {e}",
            }
    elif q_type == "list":
        try:
            formatted_answer = formatter.format_list(result)
            if not formatted_answer:
                return {
                    "type": "abstain",
                    "answer": "abstain",
                    "confidence": 0.0,
                    "passed": False,
                    "reason": "List is empty",
                }
        except Exception as e:
            return {
                "type": "abstain",
                "answer": "abstain",
                "confidence": 0.0,
                "passed": False,
                "reason": f"Schema constraint violated for list: {e}",
            }
    elif q_type == "table":
        try:
            formatted_answer = formatter.format_table(result)
            if not formatted_answer.get("columns") or not formatted_answer.get("rows"):
                return {
                    "type": "abstain",
                    "answer": "abstain",
                    "confidence": 0.0,
                    "passed": False,
                    "reason": "Table columns or rows are empty",
                }
        except Exception as e:
            return {
                "type": "abstain",
                "answer": "abstain",
                "confidence": 0.0,
                "passed": False,
                "reason": f"Schema constraint violated for table: {e}",
            }
    elif q_type == "string":
        try:
            formatted_answer = formatter.format_string(result)
            if not formatted_answer:
                return {
                    "type": "abstain",
                    "answer": "abstain",
                    "confidence": 0.0,
                    "passed": False,
                    "reason": "String answer is empty",
                }
        except Exception as e:
            return {
                "type": "abstain",
                "answer": "abstain",
                "confidence": 0.0,
                "passed": False,
                "reason": f"Schema constraint violated for string: {e}",
            }
    else:
        # q_type is None or unspecified: infer using format_scalar
        try:
            formatted_answer = formatter.format_scalar(result, q_type=None)
            if formatted_answer in ("abstain", None):
                return {
                    "type": "abstain",
                    "answer": "abstain",
                    "confidence": 0.0,
                    "passed": False,
                    "reason": "Scalar formatting yielded abstain",
                }
        except Exception as e:
            return {
                "type": "abstain",
                "answer": "abstain",
                "confidence": 0.0,
                "passed": False,
                "reason": f"Formatting error on un-typed value: {e}",
            }

    # b.2 Filtered row counts and zero aggregation check
    if code_str:
        has_filter = any(pat in code_str for pat in (".query(", "[df[", ".loc[", "== ", "isin("))
        has_agg = any(pat in code_str for pat in (".sum()", ".mean()", ".count()", ".agg(", "len("))
        if has_filter and has_agg:
            if formatted_answer == 0 and ("empty" in code_str or "len(" in code_str):
                confidence -= 0.1
                reason_notes.append("Filtered aggregation returned 0")

    # b.3 Retry consistency / conflict check
    if retry_results and len(retry_results) > 1:
        valid_retries = []
        for r in retry_results:
            if r is not None and not executor.is_empty_result(r):
                r_str = str(r).strip().lower()
                if r_str not in ("abstain", "unanswerable", "none", "null"):
                    valid_retries.append(r)

        if len(valid_retries) > 1:
            first_val = valid_retries[0]
            conflicts = False
            for other_val in valid_retries[1:]:
                if not evaluator.is_answer_match(first_val, other_val, float_tol=0.01, q_type=q_type):
                    conflicts = True
                    break

            if conflicts:
                confidence -= 0.5
                reason_notes.append(f"Conflicting results across retries: {valid_retries}")

    # Final gate decision against threshold
    if confidence < eff_threshold:
        reason_msg = "; ".join(reason_notes) if reason_notes else "Confidence below threshold"
        return {
            "type": "abstain",
            "answer": "abstain",
            "confidence": round(confidence, 2),
            "passed": False,
            "reason": f"{reason_msg} (confidence={confidence:.2f} < threshold={eff_threshold:.2f})",
        }

    resolved_type = q_type or evaluator._infer_type_from_value(formatted_answer)
    return {
        "type": resolved_type,
        "answer": formatted_answer,
        "confidence": round(confidence, 2),
        "passed": True,
        "reason": "Passed verification gate",
    }


# Backwards-compatible alias
VLLMAgent = CodeAgent


def main() -> None:
    """CLI test script to load vLLM model and verify greedy generation."""
    parser = argparse.ArgumentParser(description="Load Qwen/Qwen3-14B via vLLM with temperature=0.0")
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
