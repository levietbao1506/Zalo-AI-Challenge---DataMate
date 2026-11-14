"""Sandboxed Python Execution Environment & Self-Correction Engine (Bước 4).

Executes generated Pandas, DuckDB, and SQLite scripts safely with:
- Code extraction from LLM completions (bounded by backticks).
- Execution via exec() in an isolated namespace containing data files, calendar tools, and libraries.
- Try...except safety trapping for runtime exceptions (KeyError, ZeroDivisionError, etc.).
- Resource timeouts via worker threads to prevent infinite loops.
- Self-Correction loop: feedback error message and re-prompt LLM up to 3 times.
"""

from __future__ import annotations

import base64
from dataclasses import dataclass
import datetime
from datetime import date, datetime as dt, timedelta
import io
import json
import logging
import math
import os
from pathlib import Path
import pickle
import re
import subprocess
import sys
import tempfile
import threading
import time
import traceback
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
import unicodedata
import uuid

import pandas as pd

_HAS_PANDAS = True

EMPTY_RESULT_FEEDBACK = (
    "Đoạn code trên chạy thành công nhưng kết quả trả về bị rỗng. "
    "Hãy kiểm tra lại điều kiện lọc (filter) ngày tháng hoặc định dạng chuỗi."
)

RETRY_STRICT_INSTRUCTION = (
    "The previous code failed with the error above. "
    "Write a new, corrected Python code block to solve the problem and fix the error. "
    "DO NOT repeat the exact same code."
)

try:
    import numpy as np
    _HAS_NUMPY = True
except ImportError:
    _HAS_NUMPY = False

try:
    import openpyxl
    _HAS_OPENPYXL = True
except ImportError:
    _HAS_OPENPYXL = False
    openpyxl = None  # type: ignore[assignment]

try:
    import duckdb
    _HAS_DUCKDB = True
except ImportError:
    _HAS_DUCKDB = False

import sqlite3
from src import calendar_utils

if "calendar_utils" not in sys.modules:
    sys.modules["calendar_utils"] = calendar_utils

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


def extract_code(llm_response: str) -> str:
    """Parse and extract the Python code block from an LLM response.

    Specifically extracts text between ````python` and ```` (or generic ```` fences).
    Strips leading and trailing whitespace. If no markdown code fences are found,
    assumes the entire response is Python code and returns it stripped.

    Args:
        llm_response: Raw completion string returned by the LLM.

    Returns:
        Clean Python code string ready for execution.
    """
    if not llm_response:
        return ""

    # 1. Match ```python ... ``` (case-insensitive)
    m_py = re.search(r"```python\s*\n?(.*?)```", llm_response, re.DOTALL | re.IGNORECASE)
    if m_py:
        return m_py.group(1).strip()

    # 2. Match generic ``` ... ```
    m_generic = re.search(r"```\s*\n?(.*?)```", llm_response, re.DOTALL)
    if m_generic:
        return m_generic.group(1).strip()

    # 3. Handle unclosed opening fence (e.g. truncated at stop token)
    m_unclosed = re.match(r"^\s*```(?:python|py)?\s*\n?(.*)$", llm_response, re.DOTALL | re.IGNORECASE)
    if m_unclosed:
        return m_unclosed.group(1).strip()

    # 4. No markdown formatting found: return entire response stripped
    return llm_response.strip()


def is_empty_result(result: Any) -> bool:
    """Check whether an extracted execution result is logically empty.

    Returns ``True`` for ``None``, empty Pandas DataFrames/Series, empty NumPy arrays,
    scalar NaN values, empty strings ``""`` (including whitespace-only), empty lists ``[]``,
    empty dictionaries ``{}``, empty tuples/sets.

    Critically, explicit intentional abstain tokens ``"unanswerable"`` and ``"abstain"``
    are considered valid answers and return ``False``. Numeric ``0`` / ``0.0`` and boolean
    ``False`` also return ``False``.

    Args:
        result: Extracted ``final_result`` value returned by executed script.

    Returns:
        ``True`` if ``result`` represents logically empty data; ``False`` otherwise.
    """
    if result is None:
        return True

    if isinstance(result, str):
        normalized = result.strip().lower()
        if normalized in ("unanswerable", "abstain"):
            return False
        return normalized == ""

    if isinstance(result, pd.DataFrame):
        return bool(result.empty)

    if isinstance(result, pd.Series):
        return bool(result.empty)

    if isinstance(result, (list, dict, tuple, set)):
        return len(result) == 0

    if _HAS_NUMPY and isinstance(result, np.ndarray):
        return result.size == 0

    try:
        na_check = pd.isna(result)
        is_bool_scalar = isinstance(na_check, bool) or (
            _HAS_NUMPY and isinstance(na_check, np.bool_)
        )
        if is_bool_scalar and bool(na_check):
            return True
    except Exception:
        pass

    if isinstance(result, float) or (_HAS_NUMPY and isinstance(result, np.floating)):
        return math.isnan(result)

    return False


def _build_local_namespace(
    data_dir: str,
    file_map: Optional[Dict[str, str]] = None,
    as_of_date: Optional[Union[str, date]] = None,
) -> Dict[str, Any]:
    """Build isolated execution namespace containing data context and analytical libraries."""
    if file_map and "calendar.csv" in file_map:
        calendar_utils.set_calendar_csv_path(file_map["calendar.csv"])

    namespace: Dict[str, Any] = {
        "data_dir": data_dir,
        "DATA_DIR": data_dir,
        "FILE_MAP": file_map or {},
        "as_of_date": calendar_utils.resolve_as_of_date(as_of_date),
        "pd": pd,
        "np": np if _HAS_NUMPY else None,
        "duckdb": duckdb if _HAS_DUCKDB else None,
        "sqlite3": sqlite3,
        "math": math,
        "re": re,
        "unicodedata": unicodedata,
        "datetime": datetime,
        "date": date,
        "dt": dt,
        "timedelta": timedelta,
        "calendar_utils": calendar_utils,
        "resolve_as_of_date": calendar_utils.resolve_as_of_date,
        "parse_vietnamese_date": calendar_utils.parse_vietnamese_date,
        "resolve_vietnamese_relative_date": calendar_utils.resolve_vietnamese_relative_date,
        "get_month_date_range": calendar_utils.get_month_date_range,
        "get_quarter_date_range": calendar_utils.get_quarter_date_range,
        "get_week_date_range": calendar_utils.get_week_date_range,
        "get_same_period_last_year": calendar_utils.get_same_period_last_year,
        "get_tet_window": calendar_utils.get_tet_window,
        "get_tet_date": calendar_utils.get_tet_date,
        "get_holiday_range": calendar_utils.get_holiday_range,
        "solar_to_lunar": calendar_utils.solar_to_lunar,
        "lunar_to_solar": calendar_utils.lunar_to_solar,
    }
    if file_map:
        for fname, fpath in file_map.items():
            clean_var = re.sub(r"[^a-zA-Z0-9_]", "_", fname)
            namespace[clean_var] = fpath
    return namespace


class SubprocessResult(tuple):
    """Structured execution result (success, return_value, stdout, stderr)."""

    def __new__(cls, success: bool, return_value: Any, stdout: str, stderr: str):
        return super().__new__(cls, (success, return_value, stdout, stderr))

    @property
    def success(self) -> bool:
        return self[0]

    @property
    def return_value(self) -> Any:
        return self[1]

    @property
    def stdout(self) -> str:
        return self[2]

    @property
    def stderr(self) -> str:
        return self[3]


def _resolve_subprocess_working_dir(working_dir: Optional[Union[str, Path]] = None) -> Path:
    """Resolve working directory for subprocess execution (defaults to /tmp or scratch)."""
    if working_dir:
        p = Path(working_dir).resolve()
        p.mkdir(parents=True, exist_ok=True)
        return p
    tmp_dir = Path("/tmp")
    if tmp_dir.exists() and tmp_dir.is_dir():
        return tmp_dir
    scratch = Path(tempfile.gettempdir()) / "datamate_tmp"
    scratch.mkdir(parents=True, exist_ok=True)
    return scratch


def execute_subprocess(
    code_str: str,
    data_dir: str,
    file_map: Optional[Dict[str, str]] = None,
    as_of_date: Optional[Union[str, date]] = None,
    timeout_seconds: float = 25.0,
    working_dir: Optional[Union[str, Path]] = None,
) -> SubprocessResult:
    """Execute generated Python script in an isolated subprocess.

    Runs in a dedicated scratch working directory (default: /tmp on Linux or
    tempfile.gettempdir()/datamate_tmp on Windows). Pre-imports pandas, numpy,
    sqlite3, openpyxl, calendar_utils, and provides read-only paths to business data.
    Enforces a per-execution timeout (default: 25.0s) to handle datasets up to 300,000
    rows safely.

    Args:
        code_str: Python code string to execute.
        data_dir: Directory containing business data files.
        file_map: Optional mapping from filenames to absolute paths.
        as_of_date: Optional reference date for relative date resolution.
        timeout_seconds: Subprocess execution timeout in seconds (default: 25.0).
        working_dir: Optional working directory for subprocess (defaults to /tmp).

    Returns:
        SubprocessResult: A 4-tuple of (success, return_value, stdout, stderr).
    """
    clean_code = extract_python_code(code_str)
    scratch_dir = _resolve_subprocess_working_dir(working_dir)
    uid = uuid.uuid4().hex[:12]
    script_path = scratch_dir / f"runner_{uid}.py"
    pkl_path = scratch_dir / f"result_{uid}.pkl"

    resolved_as_of = str(calendar_utils.resolve_as_of_date(as_of_date))
    project_root = str(Path(__file__).resolve().parent.parent)

    code_b64 = base64.b64encode(clean_code.encode("utf-8")).decode("ascii")
    file_map_b64 = base64.b64encode(json.dumps(file_map or {}).encode("utf-8")).decode("ascii")
    data_dir_b64 = base64.b64encode(str(data_dir).encode("utf-8")).decode("ascii")
    pkl_b64 = base64.b64encode(str(pkl_path).encode("utf-8")).decode("ascii")
    root_b64 = base64.b64encode(project_root.encode("utf-8")).decode("ascii")

    runner_template = f'''# Auto-generated subprocess execution runner
import base64
import json
import math
import os
import pickle
import re
import sys
import unicodedata
import datetime
from datetime import date, datetime as dt, timedelta
import sqlite3

# Inject project root for module imports
project_root = base64.b64decode("{root_b64}").decode("utf-8")
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Pre-import core data analytics helpers
try:
    import pandas as pd
except ImportError:
    pd = None

try:
    import numpy as np
except ImportError:
    np = None

try:
    import openpyxl
except ImportError:
    openpyxl = None

try:
    import duckdb
except ImportError:
    duckdb = None

try:
    import calendar_utils
except ImportError:
    from src import calendar_utils

# Decode configuration
data_dir = base64.b64decode("{data_dir_b64}").decode("utf-8")
file_map = json.loads(base64.b64decode("{file_map_b64}").decode("utf-8"))
result_pkl_path = base64.b64decode("{pkl_b64}").decode("utf-8")
as_of_date_str = "{resolved_as_of}"
user_code = base64.b64decode("{code_b64}").decode("utf-8")

if file_map and "calendar.csv" in file_map:
    try:
        calendar_utils.set_calendar_csv_path(file_map["calendar.csv"])
    except Exception:
        pass

as_of_date = calendar_utils.resolve_as_of_date(as_of_date_str)

# Pre-populated execution namespace
exec_globals = {{
    "__builtins__": __builtins__,
    "pd": pd,
    "np": np,
    "sqlite3": sqlite3,
    "openpyxl": openpyxl,
    "duckdb": duckdb,
    "math": math,
    "re": re,
    "unicodedata": unicodedata,
    "datetime": datetime,
    "date": date,
    "dt": dt,
    "timedelta": timedelta,
    "calendar_utils": calendar_utils,
    "resolve_as_of_date": calendar_utils.resolve_as_of_date,
    "parse_vietnamese_date": calendar_utils.parse_vietnamese_date,
    "resolve_vietnamese_relative_date": calendar_utils.resolve_vietnamese_relative_date,
    "get_month_date_range": calendar_utils.get_month_date_range,
    "get_quarter_date_range": calendar_utils.get_quarter_date_range,
    "get_week_date_range": calendar_utils.get_week_date_range,
    "get_same_period_last_year": calendar_utils.get_same_period_last_year,
    "get_tet_window": calendar_utils.get_tet_window,
    "get_tet_date": calendar_utils.get_tet_date,
    "get_holiday_range": calendar_utils.get_holiday_range,
    "solar_to_lunar": calendar_utils.solar_to_lunar,
    "lunar_to_solar": calendar_utils.lunar_to_solar,
    "data_dir": data_dir,
    "DATA_DIR": data_dir,
    "FILE_MAP": file_map,
    "as_of_date": as_of_date,
    "AS_OF_DATE": str(as_of_date),
    "result": None,
    "final_result": None,
}}

# Inject direct aliases for business files (e.g. orders_csv = '/path/to/orders.csv')
for fname, fpath in file_map.items():
    clean_var = re.sub(r"[^a-zA-Z0-9_]", "_", fname)
    exec_globals[clean_var] = fpath

exec_locals = {{}}

try:
    compiled = compile(user_code, "<agent_script>", "exec")
    exec(compiled, exec_globals, exec_locals)

    extracted_result = None
    for var_name in ("final_result", "result"):
        if var_name in exec_locals and exec_locals[var_name] is not None:
            extracted_result = exec_locals[var_name]
            break
        if var_name in exec_globals and exec_globals[var_name] is not None:
            extracted_result = exec_globals[var_name]
            break

    try:
        with open(result_pkl_path, "wb") as f:
            pickle.dump(extracted_result, f, protocol=pickle.HIGHEST_PROTOCOL)
    except Exception:
        try:
            with open(result_pkl_path, "wb") as f:
                pickle.dump(str(extracted_result), f, protocol=pickle.HIGHEST_PROTOCOL)
        except Exception:
            pass
    sys.exit(0)
except Exception as e:
    import traceback
    traceback.print_exc(file=sys.stderr)
    sys.exit(1)
'''

    try:
        script_path.write_text(runner_template, encoding="utf-8")

        env = os.environ.copy()
        env["PYTHONPATH"] = project_root + os.pathsep + env.get("PYTHONPATH", "")

        proc = subprocess.run(
            [sys.executable, str(script_path)],
            cwd=str(scratch_dir),
            env=env,
            timeout=timeout_seconds,
            capture_output=True,
            text=True,
        )

        stdout = proc.stdout or ""
        stderr = proc.stderr or ""

        if proc.returncode == 0:
            extracted_result = None
            if pkl_path.exists() and pkl_path.stat().st_size > 0:
                try:
                    with open(pkl_path, "rb") as f:
                        extracted_result = pickle.load(f)
                except Exception as pkl_err:
                    logger.debug(f"Could not unpickle result: {pkl_err}")

            if extracted_result is None and stdout.strip():
                lines = stdout.strip().splitlines()
                extracted_result = lines[-1].strip()

            return SubprocessResult(True, extracted_result, stdout, stderr)
        else:
            return SubprocessResult(False, None, stdout, stderr)

    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout or ""
        stderr = f"TimeoutError: Execution exceeded time limit of {timeout_seconds:.1f}s"
        return SubprocessResult(False, None, stdout, stderr)
    except Exception as exc:
        tb = traceback.format_exc()
        return SubprocessResult(False, None, "", f"{type(exc).__name__}: {exc}\n{tb}".strip())
    finally:
        try:
            if script_path.exists():
                script_path.unlink()
        except Exception:
            pass
        try:
            if pkl_path.exists():
                pkl_path.unlink()
        except Exception:
            pass


def execute_local(
    code_str: str,
    data_dir: str,
    file_map: Optional[Dict[str, str]] = None,
    as_of_date: Optional[Union[str, date]] = None,
    timeout_seconds: Optional[float] = 25.0,
    use_subprocess: bool = False,
) -> tuple[bool, Any, str]:
    """Execute a Python code string in an isolated namespace and capture ``final_result``.

    Injects ``{"data_dir": data_dir, ...}`` (plus ``FILE_MAP``, ``as_of_date``,
    ``calendar_utils``, and Pandas/DuckDB/SQLite utilities) into the execution
    namespace so generated scripts can locate and query tabular datasets safely.

    If ``use_subprocess`` is True, runs inside an isolated subprocess with 25s timeout
    and pre-imported helpers.

    Args:
        code_str: Python code string to execute.
        data_dir: Path to the directory containing tabular data files.
        file_map: Optional mapping from filenames to absolute file paths.
        as_of_date: Optional reference date for relative-date resolution.
        timeout_seconds: Optional timeout in seconds (default: 25.0).
        use_subprocess: If True, execute in an isolated subprocess.

    Returns:
        Tuple of ``(success, final_result, error_message)`` where:
            - ``success`` is True if execution completed without exceptions.
            - ``final_result`` is the extracted ``final_result`` (or ``result`` fallback), else None.
            - ``error_message`` is ``""`` on success, or ``str(e)`` on failure.
    """
    timeout = timeout_seconds if timeout_seconds is not None else 25.0
    if use_subprocess:
        sub_res = execute_subprocess(
            code_str=code_str,
            data_dir=data_dir,
            file_map=file_map,
            as_of_date=as_of_date,
            timeout_seconds=timeout,
        )
        return (sub_res.success, sub_res.return_value, sub_res.stderr if not sub_res.success else "")

    namespace = _build_local_namespace(data_dir=data_dir, file_map=file_map, as_of_date=as_of_date)

    def _run() -> tuple[bool, Any, str]:
        try:
            exec(code_str, namespace)
            extracted = namespace["final_result"] if "final_result" in namespace else namespace.get("result")
            return (True, extracted, "")
        except Exception as e:
            tb = traceback.format_exc()
            err_type = type(e).__name__
            err_msg = str(e) or "Unknown error"
            full_error = f"{err_type}: {err_msg}\n{tb}".strip()
            return (False, None, full_error)

    if timeout_seconds is None:
        return _run()

    holder: Dict[str, tuple[bool, Any, str]] = {}

    def _target() -> None:
        holder["out"] = _run()

    thread = threading.Thread(target=_target, daemon=True)
    thread.start()
    thread.join(timeout=timeout)

    if thread.is_alive():
        return (False, None, f"TimeoutError: Execution exceeded time limit of {timeout:.1f}s")

    return holder.get("out", (False, None, "Worker execution failed without returning a result."))


def execute_with_retry(
    initial_prompt: str,
    generate_fn: Callable[[str], str],
    data_dir: str,
    max_retries: int = 3,
    file_map: Optional[Dict[str, str]] = None,
    as_of_date: Optional[Union[str, date]] = None,
    timeout_seconds: Optional[float] = None,
    base_prompt: Optional[str] = None,
    on_attempt: Optional[Callable[[str, bool, Any, str], None]] = None,
    use_subprocess: bool = False,
) -> tuple[bool, Any, str]:
    """Run LLM code generation and local sandbox execution with a self-correction retry loop.

    Calls ``generate_fn(current_prompt)``, extracts Python code via :func:`extract_code`,
    and runs it via :func:`execute_local` or :func:`execute_subprocess`. Retries up to
    ``max_retries`` times both on runtime/syntax exceptions and on logically empty results
    detected by :func:`is_empty_result`.

    Args:
        initial_prompt: Initial prompt passed to ``generate_fn``.
        generate_fn: Callable accepting a prompt string and returning the LLM text response.
        data_dir: Path to the directory containing tabular data files.
        max_retries: Maximum number of attempts before giving up. Defaults to 3.
        file_map: Optional mapping from filenames to absolute file paths.
        as_of_date: Optional reference date for relative-date resolution.
        timeout_seconds: Optional timeout in seconds (default: 25.0).
        base_prompt: Optional original prompt containing question and schema.
        on_attempt: Optional callback invoked after each attempt with ``(code_str, success, result, error_msg)``.
        use_subprocess: If True, executes code in an isolated subprocess.

    Returns:
        Tuple of ``(success, final_result, error_message)`` from the first valid
        non-empty execution, or ``(False, None, error_message)`` if retries are exhausted.
    """
    current_prompt = initial_prompt
    base_schema_question = base_prompt or initial_prompt
    error_msg = ""
    timeout = timeout_seconds if timeout_seconds is not None else 25.0

    for _ in range(max_retries):
        llm_response = generate_fn(current_prompt)
        code_str = extract_code(llm_response)

        if use_subprocess:
            sub_res = execute_subprocess(
                code_str=code_str,
                data_dir=data_dir,
                file_map=file_map,
                as_of_date=as_of_date,
                timeout_seconds=timeout,
            )
            success = sub_res.success
            result = sub_res.return_value
            error_msg = sub_res.stderr if not success else ""
        else:
            success, result, error_msg = execute_local(
                code_str,
                data_dir,
                file_map=file_map,
                as_of_date=as_of_date,
                timeout_seconds=timeout,
            )

        if on_attempt is not None:
            on_attempt(code_str, success, result, error_msg)

        if success and not is_empty_result(result):
            return (True, result, "")

        if success:
            error_msg = EMPTY_RESULT_FEEDBACK
            current_prompt += f"\n{EMPTY_RESULT_FEEDBACK}"
        else:
            failed_block = f"```python\n{code_str}\n```" if code_str else "```python\n# (No code extracted)\n```"
            feedback_line = f"\nĐoạn code trên bị lỗi: {error_msg}. Hãy viết lại code sửa lỗi."
            strict_line = f"\n{RETRY_STRICT_INSTRUCTION}"
            current_prompt = (
                f"{base_schema_question}\n\n"
                f"**FAILED CODE:**\n{failed_block}\n\n"
                f"**EXACT ERROR:**\n```\n{error_msg}\n```"
                f"{feedback_line}"
                f"{strict_line}"
            )

    return (False, None, error_msg)


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
    stderr: Optional[str] = None
    duration_seconds: float = 0.0
    attempts_used: int = 1

    @property
    def return_value(self) -> Any:
        """Alias matching structured (success, return_value, stdout, stderr) tuple."""
        return self.result


class SandboxedExecutor:
    """Executes generated Python scripts in an isolated namespace with resource timeouts."""

    def __init__(self, default_timeout_seconds: float = 25.0):
        self.default_timeout_seconds = default_timeout_seconds

    def execute_subprocess(
        self,
        code: str,
        data_dir: Optional[str] = None,
        file_map: Optional[Dict[str, str]] = None,
        as_of_date: Optional[Union[str, date]] = None,
        timeout_seconds: Optional[float] = None,
        working_dir: Optional[Union[str, Path]] = None,
    ) -> ExecutionResult:
        """Execute Python script in an isolated subprocess with 25s timeout and pre-imported helpers."""
        timeout = timeout_seconds if timeout_seconds is not None else self.default_timeout_seconds
        start_time = time.perf_counter()
        sub_res = execute_subprocess(
            code_str=code,
            data_dir=data_dir or ".",
            file_map=file_map,
            as_of_date=as_of_date,
            timeout_seconds=timeout,
            working_dir=working_dir,
        )
        duration = time.perf_counter() - start_time
        if sub_res.success:
            return ExecutionResult(
                success=True,
                result=sub_res.return_value,
                stdout=sub_res.stdout,
                stderr=sub_res.stderr or "",
                duration_seconds=duration,
            )
        else:
            err = sub_res.stderr or "Subprocess execution failed"
            lines = [l.strip() for l in err.splitlines() if l.strip()]
            err_msg = lines[-1] if lines else err
            err_type = err_msg.split(":")[0].strip() if ":" in err_msg else "ExecutionError"
            return ExecutionResult(
                success=False,
                result=None,
                error=err,
                error_type=err_type,
                error_msg=err_msg,
                stdout=sub_res.stdout,
                stderr=sub_res.stderr,
                duration_seconds=duration,
            )

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
            "openpyxl": openpyxl if _HAS_OPENPYXL else None,
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
            "get_week_date_range": calendar_utils.get_week_date_range,
            "get_same_period_last_year": calendar_utils.get_same_period_last_year,
            "get_tet_window": calendar_utils.get_tet_window,
            "get_tet_date": calendar_utils.get_tet_date,
            "get_holiday_range": calendar_utils.get_holiday_range,
            "solar_to_lunar": calendar_utils.solar_to_lunar,
            "lunar_to_solar": calendar_utils.lunar_to_solar,
            # Data context
            "DATA_DIR": data_dir,
            "data_dir": data_dir,
            "FILE_MAP": file_map or {},
            "as_of_date": resolved_as_of,
            "AS_OF_DATE": str(resolved_as_of),
            # Default placeholder for result
            "result": None,
            "final_result": None,
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
        use_subprocess: bool = False,
    ) -> ExecutionResult:
        """Run code in an isolated sandbox with timeout and exception trapping.

        Args:
            code: Python code string.
            data_dir: Directory containing tabular data.
            file_map: Map of filename to absolute path.
            as_of_date: Reference date for relative date questions.
            timeout_seconds: Maximum run time in seconds (default: 25.0).
            use_subprocess: If True, executes inside an isolated subprocess.

        Returns:
            ExecutionResult containing execution status, result, error, and stdout.
        """
        if use_subprocess:
            return self.execute_subprocess(
                code=code,
                data_dir=data_dir,
                file_map=file_map,
                as_of_date=as_of_date,
                timeout_seconds=timeout_seconds,
            )

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

                # 3. Extract result: look for final_result first, then result
                extracted_result = None
                for var_name in ("final_result", "result"):
                    if var_name in local_scope and local_scope[var_name] is not None:
                        extracted_result = local_scope[var_name]
                        break
                    if var_name in global_scope and global_scope[var_name] is not None:
                        extracted_result = global_scope[var_name]
                        break

                if extracted_result is None:
                    # If neither was explicitly assigned, check captured stdout for output
                    stdout_content = captured_stdout.getvalue().strip()
                    if stdout_content:
                        lines = stdout_content.splitlines()
                        extracted_result = lines[-1].strip()

                duration = time.perf_counter() - start_time
                return ExecutionResult(
                    success=True,
                    result=extracted_result,
                    stdout=captured_stdout.getvalue(),
                    stderr="",
                    duration_seconds=duration,
                )

            except Exception as exc:
                duration = time.perf_counter() - start_time
                tb = traceback.format_exc()
                err_type = type(exc).__name__
                err_msg = str(exc) or "Lỗi không xác định"
                full_error = f"{err_type}: {err_msg}\n{tb}".strip()
                logger.debug(f"Code execution encountered exception ({err_type}): {err_msg}")

                return ExecutionResult(
                    success=False,
                    result=None,
                    error=full_error,
                    error_type=err_type,
                    error_msg=f"{err_type}: {err_msg}",
                    stdout=captured_stdout.getvalue(),
                    stderr=full_error,
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
                stderr=f"TimeoutError: Execution exceeded time limit of {timeout:.1f}s",
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
                stderr=str(exc),
                duration_seconds=duration,
            )

        return ExecutionResult(
            success=False,
            result=None,
            error="Unknown execution error: worker thread did not return a result.",
            error_type="WorkerError",
            error_msg="Lỗi tiến trình thực thi",
            stderr="Unknown execution error: worker thread did not return a result.",
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
        max_retries: int = 2,
        timeout_seconds: Optional[float] = None,
        use_subprocess: bool = False,
    ) -> ExecutionResult:
        """Execute code with self-correction feedback loop up to max_retries (up to 2 retries).

        If execution encounters an exception (e.g. KeyError, SyntaxError, TypeError)
        or produces a logically empty result detected by :func:`is_empty_result`,
        captures traceback feedback and prompts the LLM to regenerate corrected code.

        Args:
            agent: Agent instance with generate_code() method.
            initial_code: Initial code generated by LLM.
            question: Vietnamese natural language question.
            prompt_builder: PromptBuilder instance with build_retry_prompt().
            system_prompt: System prompt for LLM instructions.
            data_dir: Directory containing tabular data.
            file_map: Map of filenames to paths.
            as_of_date: Reference date.
            max_retries: Maximum self-correction retry attempts (default: 2).
            timeout_seconds: Execution timeout per run (default: 25.0s).
            use_subprocess: If True, executes code in an isolated subprocess.

        Returns:
            ExecutionResult from the final or successful attempt.
        """
        current_code = initial_code
        attempts_left = max_retries
        attempts_used = 1

        exec_fn = self.execute_subprocess if use_subprocess else self.execute
        exec_res = exec_fn(
            code=current_code,
            data_dir=data_dir,
            file_map=file_map,
            as_of_date=as_of_date,
            timeout_seconds=timeout_seconds,
        )

        while (not exec_res.success or is_empty_result(exec_res.result)) and attempts_left > 0:
            if not exec_res.success:
                error_msg = exec_res.error_msg or (exec_res.error.splitlines()[0] if exec_res.error else "Unknown error")
                error_feedback = exec_res.stderr or exec_res.error or error_msg
                logger.info(
                    f"Self-correction: Code failed with [{error_msg}]. "
                    f"Retrying with traceback feedback ({attempts_left} attempts left)..."
                )
            else:
                error_feedback = EMPTY_RESULT_FEEDBACK
                logger.info(
                    "Self-correction: Code returned empty result. "
                    f"Retrying with logical-error feedback ({attempts_left} attempts left)..."
                )

            # Build retry prompt with Vietnamese error notification & traceback
            retry_prompt = prompt_builder.build_retry_prompt(
                question=question,
                previous_code=current_code,
                error_traceback=error_feedback,
            )

            # Generate corrected code
            current_code = agent.generate_code(retry_prompt, system_prompt=system_prompt)
            attempts_used += 1

            # Re-execute corrected code
            exec_res = exec_fn(
                code=current_code,
                data_dir=data_dir,
                file_map=file_map,
                as_of_date=as_of_date,
                timeout_seconds=timeout_seconds,
            )
            attempts_left -= 1

        if exec_res.success and is_empty_result(exec_res.result):
            exec_res.success = False
            exec_res.result = None
            exec_res.error_type = "EmptyResultError"
            exec_res.error_msg = EMPTY_RESULT_FEEDBACK
            exec_res.error = EMPTY_RESULT_FEEDBACK
            exec_res.stderr = EMPTY_RESULT_FEEDBACK

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
