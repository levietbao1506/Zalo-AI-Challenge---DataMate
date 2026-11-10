"""Sandboxed Python Execution Environment & Self-Correction Engine (Bước 4).

Executes generated Pandas, DuckDB, and SQLite scripts safely with:
- Code extraction from LLM completions (bounded by backticks).
- Execution via exec() in an isolated namespace containing data files, calendar tools, and libraries.
- Try...except safety trapping for runtime exceptions (KeyError, ZeroDivisionError, etc.).
- Resource timeouts via worker threads to prevent infinite loops.
- Self-Correction loop: feedback error message and re-prompt LLM up to 3 times.
"""

from __future__ import annotations

from dataclasses import dataclass
import datetime
from datetime import date, datetime as dt, timedelta
import io
import logging
import math
from pathlib import Path
import re
import sys
import threading
import time
import traceback
from typing import Any, Dict, List, Optional, Set, Union
import unicodedata

try:
    import pandas as pd
    _HAS_PANDAS = True
except ImportError:
    _HAS_PANDAS = False

try:
    import numpy as np
    _HAS_NUMPY = True
except ImportError:
    _HAS_NUMPY = False

try:
    import duckdb
    _HAS_DUCKDB = True
except ImportError:
    _HAS_DUCKDB = False

import sqlite3
from src import calendar_utils

logger = logging.getLogger(__name__)

# Disallowed builtins and dangerous modules
BLOCKED_MODULES: Set[str] = {
    "os", "subprocess", "shutil", "socket", "urllib", "requests", "http", "ftplib",
    "smtplib", "telnetlib", "posix", "nt", "pty", "commands",
}


def _safe_import(name: str, *args: Any, **kwargs: Any) -> Any:
    """Restricted import hook that blocks network and OS alteration modules."""
    root_module = name.split(".")[0]
    if root_module in BLOCKED_MODULES:
        raise ImportError(f"Import of '{name}' is restricted in sandboxed execution.")
    return __import__(name, *args, **kwargs)


def extract_python_code(text: str) -> str:
    """Extract clean executable Python code from LLM completion text (bounded by backticks).

    Handles:
    - Standard markdown ```python ... ``` or ```py ... ```
    - Generic ``` ... ```
    - Unclosed ```python blocks
    - Raw code fallback

    Args:
        text: Raw text response from LLM.

    Returns:
        Clean Python code string.
    """
    if not text:
        return "result = None"

    # 1. Match ```python ... ``` or ```py ... ```
    m_py = re.search(r"```(?:python|py)\s*\n(.*?)```", text, re.DOTALL | re.IGNORECASE)
    if m_py:
        return m_py.group(1).strip()

    # 2. Match generic ``` ... ```
    m_generic = re.search(r"```\s*\n(.*?)```", text, re.DOTALL)
    if m_generic:
        return m_generic.group(1).strip()

    # 3. Handle unclosed ```python block
    if "```" in text:
        parts = text.split("```")
        if len(parts) >= 2:
            code_part = parts[1]
            if code_part.lower().startswith("python"):
                code_part = code_part[6:].lstrip("\r\n")
            elif code_part.lower().startswith("py"):
                code_part = code_part[2:].lstrip("\r\n")
            if code_part.strip():
                return code_part.strip()

    # 4. Fallback: filter lines
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


@dataclass
class ExecutionResult:
    """Encapsulates outcome of sandboxed code execution."""
    success: bool
    result: Any = None
    error: Optional[str] = None
    error_type: Optional[str] = None
    error_msg: Optional[str] = None
    stdout: Optional[str] = None
    duration_seconds: float = 0.0
    attempts_used: int = 1


class SandboxedExecutor:
    """Executes generated Python scripts in an isolated namespace with resource timeouts."""

    def __init__(self, default_timeout_seconds: float = 15.0):
        self.default_timeout_seconds = default_timeout_seconds

    def _build_safe_globals(
        self,
        data_dir: Optional[str] = None,
        file_map: Optional[Dict[str, str]] = None,
        as_of_date: Optional[Union[str, date]] = None,
    ) -> Dict[str, Any]:
        """Construct isolated globals dictionary containing approved tools and data paths."""
        safe_builtins = dict(__builtins__) if isinstance(__builtins__, dict) else dict(__builtins__.__dict__)
        safe_builtins["__import__"] = _safe_import

        resolved_as_of = calendar_utils.resolve_as_of_date(as_of_date)

        env_globals: Dict[str, Any] = {
            "__builtins__": safe_builtins,
            # Data manipulation
            "pd": pd if _HAS_PANDAS else None,
            "np": np if _HAS_NUMPY else None,
            "duckdb": duckdb if _HAS_DUCKDB else None,
            "sqlite3": sqlite3,
            # Standard math & date tools
            "math": math,
            "re": re,
            "unicodedata": unicodedata,
            "datetime": datetime,
            "date": date,
            "dt": dt,
            "timedelta": timedelta,
            # Calendar & domain helpers
            "calendar_utils": calendar_utils,
            "resolve_as_of_date": calendar_utils.resolve_as_of_date,
            "parse_vietnamese_date": calendar_utils.parse_vietnamese_date,
            "resolve_vietnamese_relative_date": calendar_utils.resolve_vietnamese_relative_date,
            "get_month_date_range": calendar_utils.get_month_date_range,
            "get_quarter_date_range": calendar_utils.get_quarter_date_range,
            "solar_to_lunar": calendar_utils.solar_to_lunar,
            "get_tet_date": calendar_utils.get_tet_date,
            # Data context
            "DATA_DIR": data_dir,
            "FILE_MAP": file_map or {},
            "as_of_date": resolved_as_of,
            "AS_OF_DATE": str(resolved_as_of),
            # Default placeholder for result
            "result": None,
        }

        # Convenience aliases for file paths directly in globals (e.g. orders_csv = '/path/to/orders.csv')
        if file_map:
            for fname, fpath in file_map.items():
                clean_var = re.sub(r"[^a-zA-Z0-9_]", "_", fname)
                env_globals[clean_var] = fpath

        return env_globals

    def execute(
        self,
        code: str,
        data_dir: Optional[str] = None,
        file_map: Optional[Dict[str, str]] = None,
        as_of_date: Optional[Union[str, date]] = None,
        timeout_seconds: Optional[float] = None,
    ) -> ExecutionResult:
        """Run code in an isolated sandbox with timeout and exception trapping.

        Args:
            code: Python code string.
            data_dir: Directory containing tabular data.
            file_map: Map of filename to absolute path.
            as_of_date: Reference date for relative date questions.
            timeout_seconds: Maximum run time in seconds.

        Returns:
            ExecutionResult containing execution status, result, error, and stdout.
        """
        timeout = timeout_seconds if timeout_seconds is not None else self.default_timeout_seconds
        clean_code = extract_python_code(code)
        start_time = time.perf_counter()

        def _run_worker() -> ExecutionResult:
            local_scope: Dict[str, Any] = {}
            global_scope = self._build_safe_globals(
                data_dir=data_dir,
                file_map=file_map,
                as_of_date=as_of_date,
            )

            # Redirect stdout to capture print outputs
            old_stdout = sys.stdout
            captured_stdout = io.StringIO()
            sys.stdout = captured_stdout

            try:
                # 1. Compile check
                compiled = compile(clean_code, "<agent_script>", "exec")

                # 2. Execute within isolated namespace
                exec(compiled, global_scope, local_scope)

                # 3. Extract result: check local_scope first, then global_scope
                extracted_result = None
                if "result" in local_scope and local_scope["result"] is not None:
                    extracted_result = local_scope["result"]
                elif "result" in global_scope and global_scope["result"] is not None:
                    extracted_result = global_scope["result"]
                else:
                    # If result was not explicitly assigned, check captured stdout for output
                    stdout_content = captured_stdout.getvalue().strip()
                    if stdout_content:
                        lines = stdout_content.splitlines()
                        extracted_result = lines[-1].strip()

                duration = time.perf_counter() - start_time
                return ExecutionResult(
                    success=True,
                    result=extracted_result,
                    stdout=captured_stdout.getvalue(),
                    duration_seconds=duration,
                )

            except Exception as exc:
                duration = time.perf_counter() - start_time
                tb = traceback.format_exc()
                err_type = type(exc).__name__
                err_msg = str(exc) or "Lỗi không xác định"
                logger.debug(f"Code execution encountered exception ({err_type}): {err_msg}")

                return ExecutionResult(
                    success=False,
                    result=None,
                    error=f"{err_type}: {err_msg}\n{tb}",
                    error_type=err_type,
                    error_msg=f"{err_type}: {err_msg}",
                    stdout=captured_stdout.getvalue(),
                    duration_seconds=duration,
                )
            finally:
                sys.stdout = old_stdout

        worker_res: Dict[str, Any] = {"result_obj": None, "exception": None}

        def _worker_target() -> None:
            try:
                worker_res["result_obj"] = _run_worker()
            except BaseException as e:
                worker_res["exception"] = e

        thread = threading.Thread(target=_worker_target, daemon=True)
        thread.start()
        thread.join(timeout=timeout)

        if thread.is_alive():
            duration = time.perf_counter() - start_time
            logger.warning(f"Script execution timed out after {timeout} seconds.")
            return ExecutionResult(
                success=False,
                result=None,
                error=f"TimeoutError: Execution exceeded time limit of {timeout:.1f}s",
                error_type="TimeoutError",
                error_msg=f"Quá thời gian thực thi ({timeout:.1f}s)",
                duration_seconds=duration,
            )

        if worker_res["result_obj"] is not None:
            return worker_res["result_obj"]

        if worker_res["exception"] is not None:
            duration = time.perf_counter() - start_time
            exc = worker_res["exception"]
            return ExecutionResult(
                success=False,
                result=None,
                error=str(exc),
                error_type=type(exc).__name__,
                error_msg=str(exc),
                duration_seconds=duration,
            )

        return ExecutionResult(
            success=False,
            result=None,
            error="Unknown execution error: worker thread did not return a result.",
            error_type="WorkerError",
            error_msg="Lỗi tiến trình thực thi",
            duration_seconds=time.perf_counter() - start_time,
        )

    def execute_with_self_correction(
        self,
        agent: Any,
        initial_code: str,
        question: str,
        prompt_builder: Any,
        system_prompt: Optional[str] = None,
        data_dir: Optional[str] = None,
        file_map: Optional[Dict[str, str]] = None,
        as_of_date: Optional[Union[str, date]] = None,
        max_retries: int = 3,
        timeout_seconds: Optional[float] = None,
    ) -> ExecutionResult:
        """Execute code with self-correction feedback loop up to max_retries (max 3 times).

        If exec() encounters an exception (e.g. KeyError), captures the error message,
        appends 'Đoạn code trên bị lỗi: {error_msg}. Hãy viết lại code sửa lỗi',
        and prompts LLM to regenerate code.

        Args:
            agent: Agent instance with generate_code() method.
            initial_code: Initial code generated by LLM.
            question: Vietnamese natural language question.
            prompt_builder: PromptBuilder instance with build_retry_prompt().
            system_prompt: System prompt for LLM instructions.
            data_dir: Directory containing tabular data.
            file_map: Map of filenames to paths.
            as_of_date: Reference date.
            max_retries: Maximum self-correction retry attempts (default: 3).
            timeout_seconds: Execution timeout per run.

        Returns:
            ExecutionResult from the final or successful attempt.
        """
        current_code = initial_code
        attempts_left = max_retries
        attempts_used = 1

        exec_res = self.execute(
            code=current_code,
            data_dir=data_dir,
            file_map=file_map,
            as_of_date=as_of_date,
            timeout_seconds=timeout_seconds,
        )

        while not exec_res.success and attempts_left > 0:
            error_msg = exec_res.error_msg or (exec_res.error.splitlines()[0] if exec_res.error else "Unknown error")
            logger.info(
                f"Self-correction: Code failed with [{error_msg}]. "
                f"Retrying with error feedback ({attempts_left} attempts left)..."
            )

            # Build retry prompt with the required Vietnamese error notification
            retry_prompt = prompt_builder.build_retry_prompt(
                question=question,
                previous_code=current_code,
                error_traceback=exec_res.error or error_msg,
            )

            # Generate corrected code
            current_code = agent.generate_code(retry_prompt, system_prompt=system_prompt)
            attempts_used += 1

            # Re-execute corrected code
            exec_res = self.execute(
                code=current_code,
                data_dir=data_dir,
                file_map=file_map,
                as_of_date=as_of_date,
                timeout_seconds=timeout_seconds,
            )
            attempts_left -= 1

        exec_res.attempts_used = attempts_used
        return exec_res


def execute_sandboxed_code(
    code: str,
    data_dir: Optional[str] = None,
    file_map: Optional[Dict[str, str]] = None,
    as_of_date: Optional[Union[str, date]] = None,
    timeout_seconds: float = 15.0,
) -> ExecutionResult:
    """Convenience standalone function to execute code in sandbox."""
    executor = SandboxedExecutor(default_timeout_seconds=timeout_seconds)
    return executor.execute(
        code=code,
        data_dir=data_dir,
        file_map=file_map,
        as_of_date=as_of_date,
        timeout_seconds=timeout_seconds,
    )
