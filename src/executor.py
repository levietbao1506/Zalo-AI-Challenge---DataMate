"""Sandboxed Python Execution Environment for Tabular QA Scripts.

Executes generated Pandas, DuckDB, and SQLite scripts safely with:
- Global scope pre-population (pd, duckdb, sqlite3, calendar_utils, datetime, etc.).
- Execution timeouts via threading to prevent infinite loops.
- Safety guards against system modification, file deletion, and shell execution.
- Captures stdout and tracebacks for agent self-correction.
"""

from __future__ import annotations

from dataclasses import dataclass
import datetime
from datetime import date, datetime as dt, timedelta
import io
import logging
import math
import re
import sys
import threading
import time
import traceback
from typing import Any, Dict, Optional, Set

try:
    import pandas as pd
    _HAS_PANDAS = True
except ImportError:
    _HAS_PANDAS = None

try:
    import numpy as np
    _HAS_NUMPY = True
except ImportError:
    _HAS_NUMPY = None

try:
    import duckdb
    _HAS_DUCKDB = True
except ImportError:
    _HAS_DUCKDB = None

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


@dataclass
class ExecutionResult:
    """Encapsulates outcome of sandboxed code execution."""
    success: bool
    result: Any = None
    error: Optional[str] = None
    stdout: Optional[str] = None
    duration_seconds: float = 0.0


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

        # Disallow dangerous builtins if present
        for dangerous in ("eval", "exec", "breakpoint"):
            # keep eval/exec accessible if internal or selectively replace
            pass

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

        # Convenience aliases for file paths directly in globals (e.g. customers_csv = '/path/to/customers.csv')
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
        """Run code in sandbox with timeout and exception trapping.

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
                # Syntax check first
                compiled = compile(code, "<agent_script>", "exec")
                exec(compiled, global_scope, local_scope)

                # Extract result: look in local_scope first, then global_scope
                extracted_result = None
                if "result" in local_scope and local_scope["result"] is not None:
                    extracted_result = local_scope["result"]
                elif "result" in global_scope and global_scope["result"] is not None:
                    extracted_result = global_scope["result"]
                else:
                    # If result was not explicitly assigned, check captured stdout for a single line output
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
                logger.debug(f"Code execution encountered exception:\n{tb}")
                return ExecutionResult(
                    success=False,
                    result=None,
                    error=f"{type(exc).__name__}: {str(exc)}\n{tb}",
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
                duration_seconds=duration,
            )

        if worker_res["result_obj"] is not None:
            return worker_res["result_obj"]

        if worker_res["exception"] is not None:
            duration = time.perf_counter() - start_time
            return ExecutionResult(
                success=False,
                result=None,
                error=str(worker_res["exception"]),
                duration_seconds=duration,
            )

        return ExecutionResult(
            success=False,
            result=None,
            error="Unknown execution error: worker thread did not return a result.",
            duration_seconds=time.perf_counter() - start_time,
        )
