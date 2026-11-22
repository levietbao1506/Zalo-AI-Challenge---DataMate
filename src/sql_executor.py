"""Safe read-only DuckDB SQL execution for generated queries."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from decimal import Decimal
import math
from pathlib import Path
import re
import threading
import time
import unicodedata
from typing import Any, Dict, List, Optional, Tuple, Union

import duckdb

try:
    import numpy as np
except ImportError:
    np = None


@dataclass
class SQLExecutionResult:
    success: bool
    sql: str
    columns: List[str] = field(default_factory=list)
    rows: List[List[Any]] = field(default_factory=list)
    row_count: int = 0
    truncated: bool = False
    elapsed_ms: int = 0
    error: Optional[str] = None
    anomalies: List[str] = field(default_factory=list)


def vn_norm(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    normalized = unicodedata.normalize("NFC", str(value)).lower()
    characters = [character if character.isalnum() or character.isspace() else " " for character in normalized]
    return " ".join("".join(characters).split())


def vn_unaccent(value: Optional[str]) -> Optional[str]:
    normalized = vn_norm(value)
    if normalized is None:
        return None
    decomposed = unicodedata.normalize("NFD", normalized.replace("đ", "d"))
    return "".join(character for character in decomposed if not unicodedata.combining(character))


class DuckDBSQLExecutor:
    forbidden_keywords = {
        "alter",
        "attach",
        "call",
        "checkpoint",
        "comment",
        "copy",
        "create",
        "delete",
        "detach",
        "drop",
        "export",
        "import",
        "insert",
        "install",
        "load",
        "merge",
        "pragma",
        "reset",
        "set",
        "truncate",
        "update",
        "use",
        "vacuum",
    }
    external_functions = {
        "arrow_scan",
        "csv_scan",
        "delta_scan",
        "excel_scan",
        "glob",
        "httpfs",
        "iceberg_scan",
        "parquet_scan",
        "postgres_query",
        "postgres_scan",
        "query",
        "query_table",
        "read_blob",
        "read_csv",
        "read_csv_auto",
        "read_json",
        "read_json_auto",
        "read_ndjson",
        "read_parquet",
        "read_text",
        "sqlite_scan",
    }

    def __init__(
        self,
        business_id: Union[str, Any],
        cache_root: Union[str, Path] = "/tmp/datamate_duckdb",
        memory_limit: str = "512MB",
        threads: int = 2,
        max_rows: int = 500,
        timeout_seconds: float = 10.0,
    ) -> None:
        if hasattr(business_id, "business_id"):
            b_id = getattr(business_id, "business_id")
            if hasattr(business_id, "cache_root"):
                cache_root = getattr(business_id, "cache_root")
        else:
            b_id = str(business_id)
        if not re.fullmatch(r"[A-Za-z0-9_\-]+", b_id):
            raise ValueError("business_id contains unsupported characters")
        if not re.fullmatch(r"[1-9]\d*(?:KB|MB|GB)", memory_limit.upper()):
            raise ValueError("memory_limit must use KB, MB, or GB units")
        if not isinstance(threads, int) or not 1 <= threads <= 64:
            raise ValueError("threads must be between 1 and 64")
        if not isinstance(max_rows, int) or max_rows < 1:
            raise ValueError("max_rows must be a positive integer")
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        self.business_id = b_id
        self.database_path = Path(cache_root) / f"{b_id}.duckdb"
        self.memory_limit = memory_limit.upper()
        self.threads = threads
        self.max_rows = max_rows
        self.timeout_seconds = float(timeout_seconds)
        self._execution_lock = threading.Lock()
        self._closed = False
        if not self.database_path.is_file():
            raise FileNotFoundError(f"DuckDB database does not exist: {self.database_path}")
        self._connection = duckdb.connect(str(self.database_path), read_only=True)
        self._connection.execute(f"SET memory_limit='{self.memory_limit}'")
        self._connection.execute(f"SET threads={self.threads}")
        self._connection.create_function(
            "vn_norm",
            vn_norm,
            ["VARCHAR"],
            "VARCHAR",
            null_handling="special",
        )
        self._connection.create_function(
            "vn_unaccent",
            vn_unaccent,
            ["VARCHAR"],
            "VARCHAR",
            null_handling="special",
        )

    def close(self) -> None:
        with self._execution_lock:
            if not self._closed:
                self._connection.close()
                self._closed = True

    def __enter__(self) -> "DuckDBSQLExecutor":
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        self.close()

    def execute(self, sql: str, timeout_seconds: Optional[float] = None) -> SQLExecutionResult:
        started = time.perf_counter()
        cleaned_sql = sql.strip() if isinstance(sql, str) else ""
        validation_error = self._validate_sql(cleaned_sql)
        if validation_error:
            return self._failure(cleaned_sql, started, validation_error)
        timeout = self.timeout_seconds if timeout_seconds is None else float(timeout_seconds)
        if timeout <= 0:
            return self._failure(cleaned_sql, started, "Timeout must be positive")
        with self._execution_lock:
            if self._closed:
                return self._failure(cleaned_sql, started, "Executor is closed")
            outcome: Dict[str, Any] = {}
            completed = threading.Event()

            def run_query() -> None:
                try:
                    cursor = self._connection.execute(cleaned_sql)
                    outcome["columns"] = [str(item[0]) for item in cursor.description or []]
                    outcome["rows"] = cursor.fetchmany(self.max_rows + 1)
                except BaseException as exc:
                    outcome["error"] = str(exc)
                finally:
                    completed.set()

            worker = threading.Thread(target=run_query, name="duckdb-query", daemon=True)
            worker.start()
            if not completed.wait(timeout):
                self._connection.interrupt()
                completed.wait(max(1.0, min(5.0, timeout)))
                elapsed = self._elapsed_ms(started)
                return SQLExecutionResult(
                    success=False,
                    sql=cleaned_sql,
                    elapsed_ms=elapsed,
                    error=f"TimeoutError: query exceeded {timeout:.3f} seconds",
                )
            if "error" in outcome:
                return self._failure(cleaned_sql, started, outcome["error"])
            raw_rows = outcome.get("rows", [])
            truncated = len(raw_rows) > self.max_rows
            visible_rows = raw_rows[:self.max_rows]
            columns = outcome.get("columns", [])
            anomalies = self._detect_anomalies(columns, visible_rows, truncated)
            converted_rows = [
                [self._to_json_value(value) for value in row]
                for row in visible_rows
            ]
            return SQLExecutionResult(
                success=True,
                sql=cleaned_sql,
                columns=columns,
                rows=converted_rows,
                row_count=len(converted_rows),
                truncated=truncated,
                elapsed_ms=self._elapsed_ms(started),
                anomalies=anomalies,
            )

    @classmethod
    def _validate_sql(cls, sql: str) -> Optional[str]:
        if not sql:
            return "UnsafeSQL: query is empty"
        try:
            tokens, statement_count, literal_sources = cls._lex_sql(sql)
        except ValueError as exc:
            return f"UnsafeSQL: {exc}"
        if statement_count != 1:
            return "UnsafeSQL: exactly one statement is required"
        if not tokens or tokens[0] not in {"select", "with"}:
            return "UnsafeSQL: only SELECT or WITH ... SELECT is allowed"
        if tokens[0] == "with" and "select" not in tokens:
            return "UnsafeSQL: WITH must lead to a SELECT query"
        forbidden = sorted(set(tokens) & cls.forbidden_keywords)
        if forbidden:
            return f"UnsafeSQL: forbidden keyword {forbidden[0]}"
        external = sorted(
            token
            for token in set(tokens)
            if token in cls.external_functions or token.startswith("read_") or token.endswith("_scan")
        )
        if external:
            return f"UnsafeSQL: external file function {external[0]}"
        if literal_sources:
            return "UnsafeSQL: external file source in FROM or JOIN"
        return None

    @staticmethod
    def _lex_sql(sql: str) -> Tuple[List[str], int, List[str]]:
        tokens: List[str] = []
        literal_sources: List[str] = []
        index = 0
        length = len(sql)
        terminated = False
        statements = 1
        while index < length:
            character = sql[index]
            if character.isspace():
                index += 1
                continue
            if sql.startswith("--", index):
                newline = sql.find("\n", index + 2)
                index = length if newline < 0 else newline + 1
                continue
            if sql.startswith("/*", index):
                end = sql.find("*/", index + 2)
                if end < 0:
                    raise ValueError("unterminated block comment")
                index = end + 2
                continue
            if character == "'":
                index += 1
                value: List[str] = []
                while index < length:
                    if sql[index] == "'":
                        if index + 1 < length and sql[index + 1] == "'":
                            value.append("'")
                            index += 2
                            continue
                        index += 1
                        break
                    value.append(sql[index])
                    index += 1
                else:
                    raise ValueError("unterminated string literal")
                if tokens and tokens[-1] in {"from", "join"}:
                    literal_sources.append("".join(value))
                continue
            if character == '"':
                index += 1
                value: List[str] = []
                while index < length:
                    if sql[index] == '"':
                        if index + 1 < length and sql[index + 1] == '"':
                            value.append('"')
                            index += 2
                            continue
                        index += 1
                        break
                    value.append(sql[index])
                    index += 1
                else:
                    raise ValueError("unterminated quoted identifier")
                identifier = "".join(value).lower()
                if re.fullmatch(r"[a-z_][a-z0-9_$]*", identifier):
                    if terminated:
                        statements += 1
                        terminated = False
                    tokens.append(identifier)
                continue
            if character == ";":
                if terminated:
                    statements += 1
                terminated = True
                index += 1
                continue
            if character.isalpha() or character == "_":
                end = index + 1
                while end < length and (sql[end].isalnum() or sql[end] in "_$"):
                    end += 1
                if terminated:
                    statements += 1
                    terminated = False
                tokens.append(sql[index:end].lower())
                index = end
                continue
            if terminated:
                statements += 1
                terminated = False
            index += 1
        return tokens, statements, literal_sources

    @classmethod
    def _detect_anomalies(
        cls,
        columns: List[str],
        rows: List[Tuple[Any, ...]],
        truncated: bool,
    ) -> List[str]:
        anomalies: List[str] = []
        if not rows:
            anomalies.append("empty_result")
        if len(rows) == 1 and len(rows[0]) == 1 and rows[0][0] is None:
            anomalies.append("all_null")
        if len(rows) == 1 and len(rows[0]) == 1:
            scalar = rows[0][0]
            if isinstance(scalar, (int, float, Decimal)) and not isinstance(scalar, bool) and scalar == 0:
                anomalies.append("zero_scalar")
        percentage_indexes = {
            index
            for index, column in enumerate(columns)
            if cls._is_percentage_column(column)
        }
        suspicious = False
        for row in rows:
            for index, value in enumerate(row):
                if cls._is_non_finite(value):
                    suspicious = True
                if index in percentage_indexes and cls._is_number(value):
                    numeric = float(value)
                    if math.isfinite(numeric) and not -100 <= numeric <= 1000:
                        suspicious = True
        if suspicious:
            anomalies.append("suspicious_extreme")
        if truncated:
            anomalies.append("truncated_rows")
        return anomalies

    @staticmethod
    def _is_percentage_column(column: str) -> bool:
        normalized = vn_unaccent(column) or ""
        normalized = normalized.replace(" ", "_")
        return bool(re.search(r"(^|_)(percent|percentage|pct|ty_le|ti_le)(_|$)", normalized)) or "%" in column

    @staticmethod
    def _is_number(value: Any) -> bool:
        return isinstance(value, (int, float, Decimal)) and not isinstance(value, bool)

    @staticmethod
    def _is_non_finite(value: Any) -> bool:
        if isinstance(value, Decimal):
            return not value.is_finite()
        if isinstance(value, float):
            return not math.isfinite(value)
        if np is not None and isinstance(value, np.floating):
            return not bool(np.isfinite(value))
        return False

    @classmethod
    def _to_json_value(cls, value: Any) -> Any:
        if value is None:
            return None
        if np is not None and isinstance(value, np.generic):
            return cls._to_json_value(value.item())
        if isinstance(value, bool):
            return value
        if isinstance(value, Decimal):
            if not value.is_finite():
                return None
            return int(value) if value == value.to_integral_value() else float(value)
        if isinstance(value, int):
            return value
        if isinstance(value, float):
            return value if math.isfinite(value) else None
        if isinstance(value, (date, datetime)):
            return value.isoformat()
        if isinstance(value, str):
            return value
        return str(value)

    @staticmethod
    def _elapsed_ms(started: float) -> int:
        return max(0, int(round((time.perf_counter() - started) * 1000)))

    @classmethod
    def _failure(cls, sql: str, started: float, error: str) -> SQLExecutionResult:
        return SQLExecutionResult(
            success=False,
            sql=sql,
            elapsed_ms=cls._elapsed_ms(started),
            error=error,
        )
