"""Automatic read-only business data ingestion into a temporary DuckDB store."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
import csv
import difflib
import math
from pathlib import Path
import re
import sqlite3
import unicodedata
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import duckdb
import pandas as pd


@dataclass
class TableMetadata:
    sql_name: str
    source_file: str
    source_table: Optional[str] = None
    source_sheet: Optional[str] = None
    original_columns: Dict[str, str] = field(default_factory=dict)
    coerced_columns: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    etl_warnings: List[str] = field(default_factory=list)
    row_count: int = 0


class BusinessDuckDBStore:
    encodings = ("utf-8-sig", "utf-8", "cp1258", "latin-1")
    delimiters = (",", ";", "\t")
    supported_suffixes = (".csv", ".xlsx", ".sqlite")

    def __init__(
        self,
        business_dir: Union[str, Path],
        calendar_csv_path: Optional[Union[str, Path]],
        cache_root: Union[str, Path] = "/tmp/datamate_duckdb",
    ) -> None:
        self.business_dir = Path(business_dir).resolve()
        self.calendar_csv_path = Path(calendar_csv_path).resolve() if calendar_csv_path else None
        self.cache_root = Path(cache_root)
        self.business_id = self._sql_identifier(self.business_dir.name) or "business"
        self.cache_root.mkdir(parents=True, exist_ok=True)
        self.database_path = self.cache_root / f"{self.business_id}.duckdb"
        self.table_metadata: Dict[str, TableMetadata] = {}
        self.table_name_map: Dict[str, str] = {}
        self.column_name_maps: Dict[str, Dict[str, str]] = {}
        self._used_table_names: set[str] = set()
        self.candidate_unique_keys: Dict[str, List[str]] = {}
        self.join_hints: Any = []
        self.schema_meta: Dict[str, Any] = {}
        if not self.business_dir.is_dir():
            raise FileNotFoundError(f"Business directory does not exist: {self.business_dir}")
        if self.database_path.exists():
            try:
                self.database_path.unlink()
            except PermissionError:
                pass
        self.connection = duckdb.connect(str(self.database_path))
        try:
            self._load_business_files()
            self._load_calendar()
            self.connection.close()
            self.connection = duckdb.connect(str(self.database_path), read_only=True)
            self.profile_keys_and_joins()
        except Exception:
            self.close_and_cleanup()
            raise

    def profile_keys_and_joins(self) -> Any:
        connection = getattr(self, "connection", None)
        if connection is None:
            return self.join_hints

        table_names = sorted(tbl for tbl in self.table_metadata if tbl != "_calendar")
        candidate_unique_keys: Dict[str, List[str]] = {}
        candidate_cols_by_table: Dict[str, List[str]] = {}
        col_types_by_table: Dict[str, Dict[str, str]] = {}
        key_tokens = ("id", "ma", "code", "so_hd", "sku", "khach", "sp")

        for tbl in table_names:
            candidate_unique_keys[tbl] = []
            candidate_cols_by_table[tbl] = []
            col_types_by_table[tbl] = {}
            try:
                described = connection.execute(f"DESCRIBE SELECT * FROM {self._quote(tbl)}").fetchall()
            except Exception:
                continue
            col_types = {str(row[0]): str(row[1]).upper() for row in described}
            col_types_by_table[tbl] = col_types
            try:
                total_rows = connection.execute(f"SELECT COUNT(*) FROM {self._quote(tbl)}").fetchone()[0]
            except Exception:
                total_rows = 0
            if total_rows == 0:
                continue

            unique_keys: List[str] = []
            cand_cols: List[str] = []
            for col, dtype in sorted(col_types.items()):
                col_lower = col.lower()
                matches_kw = any(token in col_lower for token in key_tokens)
                try:
                    non_null = connection.execute(
                        f"SELECT COUNT({self._quote(col)}) FROM {self._quote(tbl)}"
                    ).fetchone()[0]
                except Exception:
                    non_null = 0
                non_null_ratio = non_null / total_rows if total_rows > 0 else 0.0
                if matches_kw or non_null_ratio >= 0.95:
                    cand_cols.append(col)
                    try:
                        distinct_count = connection.execute(
                            f"SELECT COUNT(DISTINCT {self._quote(col)}) FROM {self._quote(tbl)}"
                        ).fetchone()[0]
                    except Exception:
                        distinct_count = 0
                    uniqueness_ratio = distinct_count / non_null if non_null > 0 else 0.0
                    if uniqueness_ratio >= 0.995:
                        unique_keys.append(col)

            candidate_unique_keys[tbl] = sorted(unique_keys)
            candidate_cols_by_table[tbl] = sorted(cand_cols)

        def is_type_compatible(t1: str, t2: str) -> bool:
            int_types = ("INT", "BIGINT", "SMALLINT", "TINYINT", "INTEGER", "HUGEINT", "UBIGINT", "UINTEGER")
            str_types = ("VARCHAR", "CHAR", "TEXT", "STRING")
            t1_is_int = any(k in t1 for k in int_types)
            t2_is_int = any(k in t2 for k in int_types)
            t1_is_str = any(k in t1 for k in str_types)
            t2_is_str = any(k in t2 for k in str_types)
            return (t1_is_int and t2_is_int) or (t1_is_str and t2_is_str)

        def are_names_similar(c1: str, c2: str) -> bool:
            n1 = re.sub(r"^(c_|col_|column_)", "", c1.lower()).replace("_", "")
            n2 = re.sub(r"^(c_|col_|column_)", "", c2.lower()).replace("_", "")
            if n1 == n2 or n1 in n2 or n2 in n1:
                return True
            return difflib.SequenceMatcher(None, n1, n2).ratio() >= 0.7

        joins: List[Dict[str, Any]] = []

        for left_table in table_names:
            for right_table in table_names:
                if left_table == right_table:
                    continue
                for left_col in candidate_cols_by_table[left_table]:
                    for right_col in candidate_cols_by_table[right_table]:
                        t1 = col_types_by_table[left_table].get(left_col, "")
                        t2 = col_types_by_table[right_table].get(right_col, "")
                        if not is_type_compatible(t1, t2):
                            continue

                        query = (
                            f"WITH src_sample AS ("
                            f"SELECT {self._quote(left_col)} AS val FROM {self._quote(left_table)} "
                            f"WHERE {self._quote(left_col)} IS NOT NULL LIMIT 2000"
                            f") "
                            f"SELECT COUNT(DISTINCT val) AS sample_distinct, "
                            f"COUNT(DISTINCT CASE WHEN tgt.{self._quote(right_col)} IS NOT NULL THEN val END) AS matched_distinct "
                            f"FROM src_sample src "
                            f"LEFT JOIN ("
                            f"SELECT DISTINCT {self._quote(right_col)} FROM {self._quote(right_table)} "
                            f"WHERE {self._quote(right_col)} IS NOT NULL"
                            f") tgt ON src.val = tgt.{self._quote(right_col)}"
                        )
                        try:
                            res = connection.execute(query).fetchone()
                            sample_distinct = res[0] or 0
                            matched_distinct = res[1] or 0
                        except Exception:
                            continue

                        if sample_distinct == 0:
                            continue

                        overlap_ratio = matched_distinct / sample_distinct
                        threshold = 0.3 if are_names_similar(left_col, right_col) else 0.5
                        if overlap_ratio >= threshold:
                            left_is_unique = left_col in candidate_unique_keys[left_table]
                            right_is_unique = right_col in candidate_unique_keys[right_table]
                            if right_is_unique:
                                relationship_type = "one_to_one" if left_is_unique else "many_to_one"
                            elif not left_is_unique:
                                relationship_type = "many_to_many"
                            else:
                                continue

                            joins.append({
                                "left_table": left_table,
                                "left_col": left_col,
                                "right_table": right_table,
                                "right_col": right_col,
                                "overlap_ratio": round(float(overlap_ratio), 4),
                                "relationship_type": relationship_type,
                            })

        joins.sort(key=lambda item: (item["left_table"], item["left_col"], item["right_table"], item["right_col"]))

        class JoinHints(list):
            def __init__(self, items: List[Dict[str, Any]], unique_keys: Dict[str, List[str]]):
                super().__init__(items)
                self.candidate_unique_keys = unique_keys
                self.joins = items

            def __getitem__(self, item: Any) -> Any:
                if isinstance(item, str):
                    if item == "joins":
                        return self.joins
                    if item in ("candidate_unique_keys", "unique_keys"):
                        return self.candidate_unique_keys
                    raise KeyError(item)
                return super().__getitem__(item)

            def get(self, key: str, default: Any = None) -> Any:
                if key == "joins":
                    return self.joins
                if key in ("candidate_unique_keys", "unique_keys"):
                    return self.candidate_unique_keys
                return default

        hints_obj = JoinHints(joins, candidate_unique_keys)
        self.candidate_unique_keys = candidate_unique_keys
        self.join_hints = hints_obj
        self.schema_meta["join_hints"] = hints_obj
        return self.join_hints

    @property
    def tables(self) -> Dict[str, TableMetadata]:
        return self.table_metadata

    def close_and_cleanup(self) -> None:
        connection = getattr(self, "connection", None)
        if connection is not None:
            try:
                connection.close()
            except Exception:
                pass
            del self.connection
            self.connection = None
        import gc
        gc.collect()
        for suffix in ("", ".wal"):
            path = Path(f"{self.database_path}{suffix}")
            if path.exists():
                try:
                    path.unlink()
                except (PermissionError, OSError):
                    pass

    def _load_business_files(self) -> None:
        files = sorted(
            path for path in self.business_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in self.supported_suffixes
        )
        calendar_resolved = self.calendar_csv_path.resolve() if self.calendar_csv_path else None
        for path in files:
            if calendar_resolved and path.resolve() == calendar_resolved:
                continue
            suffix = path.suffix.lower()
            if suffix == ".csv":
                self._load_csv(path)
            elif suffix == ".xlsx":
                self._load_excel(path)
            elif suffix == ".sqlite":
                self._load_sqlite(path)

    def _load_calendar(self) -> None:
        if not self.calendar_csv_path or not self.calendar_csv_path.is_file():
            return
        from src.calendar_utils import load_normalized_calendar_df
        norm_frame = load_normalized_calendar_df(self.calendar_csv_path)
        if norm_frame is None or (norm_frame.empty and not len(norm_frame.columns)):
            return
        db_cols = [c for c in norm_frame.columns if not str(c).startswith("_")]
        db_frame = norm_frame[db_cols].copy()
        self._store_frame(
            db_frame,
            "_calendar",
            self.calendar_csv_path.name,
            warnings=[],
            fixed_name=True,
        )

    def _load_csv(self, path: Path) -> None:
        frame, warnings = self._read_csv_frame(path)
        self._store_frame(frame, path.stem, path.name, warnings=warnings)

    def _read_csv_frame(self, path: Path) -> Tuple[pd.DataFrame, List[str]]:
        encoding = self._detect_encoding(path)
        delimiter = self._detect_delimiter(path, encoding)
        frame = pd.read_csv(
            path,
            encoding=encoding,
            sep=delimiter,
            header=None,
            dtype=object,
            keep_default_na=False,
            na_values=[],
            engine="python",
        )
        frame, header_warning = self._prepare_raw_frame(frame)
        warnings = [header_warning] if header_warning else []
        return frame, warnings

    def _load_excel(self, path: Path) -> None:
        merged_warnings = self._excel_merged_warnings(path)
        with pd.ExcelFile(path, engine="openpyxl") as workbook:
            for sheet_name in workbook.sheet_names:
                raw = pd.read_excel(
                    workbook,
                    sheet_name=sheet_name,
                    header=None,
                    dtype=object,
                    keep_default_na=False,
                )
                if raw.replace("", pd.NA).dropna(how="all").empty:
                    continue
                frame, header_warning = self._prepare_raw_frame(raw)
                warnings = list(merged_warnings.get(sheet_name, []))
                if header_warning:
                    warnings.append(header_warning)
                self._store_frame(
                    frame,
                    f"{path.stem}__{sheet_name}",
                    path.name,
                    source_sheet=sheet_name,
                    warnings=warnings,
                )

    def _load_sqlite(self, path: Path) -> None:
        uri = f"file:{path.as_posix()}?mode=ro"
        source = sqlite3.connect(uri, uri=True)
        try:
            names = [
                row[0]
                for row in source.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' "
                    "AND name NOT LIKE 'sqlite_%' ORDER BY name"
                ).fetchall()
            ]
            for source_table in names:
                quoted = source_table.replace('"', '""')
                frame = pd.read_sql_query(f'SELECT * FROM "{quoted}"', source)
                self._store_frame(
                    frame,
                    f"{path.stem}__{source_table}",
                    path.name,
                    source_table=source_table,
                )
        finally:
            source.close()

    def _store_frame(
        self,
        frame: pd.DataFrame,
        proposed_name: str,
        source_file: str,
        source_table: Optional[str] = None,
        source_sheet: Optional[str] = None,
        warnings: Optional[Iterable[str]] = None,
        fixed_name: bool = False,
    ) -> None:
        sql_name = proposed_name if fixed_name else self._unique_table_name(proposed_name)
        if fixed_name:
            self._used_table_names.add(sql_name)
        cleaned, original_columns, coercions, detected_warnings = self._clean_frame(frame)
        for column in cleaned.columns:
            if str(cleaned[column].dtype).lower() in ("str", "string"):
                cleaned[column] = cleaned[column].astype(object)
        all_warnings = sorted(set([*(warnings or []), *detected_warnings]))
        register_name = f"_incoming_{len(self.table_metadata)}"
        self.connection.register(register_name, cleaned)
        try:
            self.connection.execute(
                f"CREATE TABLE {self._quote(sql_name)} AS SELECT * FROM {self._quote(register_name)}"
            )
        finally:
            self.connection.unregister(register_name)
        source_key = source_file
        if source_sheet is not None:
            source_key = f"{source_file}::{source_sheet}"
        elif source_table is not None:
            source_key = f"{source_file}::{source_table}"
        self.table_name_map[source_key] = sql_name
        self.column_name_maps[sql_name] = dict(original_columns)
        self.table_metadata[sql_name] = TableMetadata(
            sql_name=sql_name,
            source_file=source_file,
            source_table=source_table,
            source_sheet=source_sheet,
            original_columns=original_columns,
            coerced_columns=coercions,
            etl_warnings=all_warnings,
            row_count=len(cleaned),
        )

    def _prepare_raw_frame(self, raw: pd.DataFrame) -> Tuple[pd.DataFrame, Optional[str]]:
        raw = raw.copy()
        raw = raw.replace({None: ""})
        raw = raw.loc[~raw.apply(self._row_is_empty, axis=1)]
        raw = raw.loc[:, ~raw.apply(self._column_is_empty, axis=0)]
        if raw.empty or not len(raw.columns):
            return pd.DataFrame(), None
        header_index = self._detect_header_row(raw)
        header = [self._normalize_text(value) or f"column_{index + 1}" for index, value in enumerate(raw.iloc[header_index])]
        frame = raw.iloc[header_index + 1:].copy()
        frame.columns = header
        frame = frame.reset_index(drop=True)
        warning = "shifted_header_detected" if header_index > 0 else None
        return frame, warning

    def _clean_frame(
        self,
        frame: pd.DataFrame,
    ) -> Tuple[pd.DataFrame, Dict[str, str], Dict[str, Dict[str, Any]], List[str]]:
        normalized_columns = [
            frame.iloc[:, index].map(self._normalize_cell)
            for index in range(frame.shape[1])
        ]
        cleaned = pd.concat(normalized_columns, axis=1) if normalized_columns else pd.DataFrame()
        cleaned.columns = list(frame.columns)
        cleaned = cleaned.loc[~cleaned.apply(self._row_is_empty, axis=1)].reset_index(drop=True)
        cleaned = self._remove_summary_rows(cleaned)
        sql_columns: List[str] = []
        original_columns: Dict[str, str] = {}
        used: set[str] = set()
        for index, original in enumerate(cleaned.columns):
            base = self._sql_identifier(str(original)) or f"column_{index + 1}"
            sql_column = self._deduplicate_name(base, used)
            used.add(sql_column)
            sql_columns.append(sql_column)
            original_columns[sql_column] = str(original)
        cleaned.columns = sql_columns
        coercions: Dict[str, Dict[str, Any]] = {}
        warnings: List[str] = []
        for column in list(cleaned.columns):
            result = self._coerce_series(cleaned[column], column)
            if result is None:
                parse_warning = self._type_parse_warning(cleaned[column], column)
                if parse_warning:
                    warnings.append(parse_warning)
                continue
            parsed, target_type, success_rate, raw_required = result
            raw_series = cleaned[column].copy()
            cleaned[column] = parsed
            if raw_required:
                raw_name = self._deduplicate_name(f"{column}__raw", set(cleaned.columns))
                cleaned[raw_name] = raw_series.replace("", pd.NA)
                original_columns[raw_name] = original_columns[column]
            coercions[column] = {
                "target_type": target_type,
                "success_rate": round(success_rate, 6),
                "raw_column": raw_name if raw_required else None,
            }
            if raw_required:
                warnings.append(f"partial_type_parse:{column}:{success_rate:.1%}")
        if len(cleaned.columns) >= 12:
            date_headers = sum(self._looks_like_date_text(column) for column in cleaned.columns)
            if date_headers >= 6 and date_headers >= len(cleaned.columns) / 2:
                warnings.append("horizontal_pivot_suspected")
        if len(cleaned.columns) > 30 and len(cleaned) < len(cleaned.columns):
            warnings.append("wide_table_suspected")
        return cleaned, original_columns, coercions, warnings

    def _remove_summary_rows(self, frame: pd.DataFrame) -> pd.DataFrame:
        if frame.empty or not len(frame.columns):
            return frame
        keywords = {"tong cong", "tong", "cong", "total", "subtotal"}
        identity_indexes = [
            index for index, column in enumerate(frame.columns[1:], start=1)
            if re.search(r"(^|_)(id|ma|code|date|ngay|time)(_|$)", self._sql_identifier(str(column)))
        ]

        def is_summary(row: pd.Series) -> bool:
            marker = self._ascii_text(row.iloc[0]).lower()
            if marker not in keywords:
                return False
            if identity_indexes:
                return all(self._is_empty(row.iloc[index]) for index in identity_indexes)
            filled = sum(not self._is_empty(value) for value in row)
            return filled <= max(2, math.ceil(len(row) / 2))

        return frame.loc[~frame.apply(is_summary, axis=1)].reset_index(drop=True)

    def _coerce_series(
        self,
        series: pd.Series,
        column_name: str,
    ) -> Optional[Tuple[pd.Series, str, float, bool]]:
        if pd.api.types.is_numeric_dtype(series) or pd.api.types.is_datetime64_any_dtype(series):
            return None
        nonempty_mask = ~series.map(self._is_empty)
        nonempty = series[nonempty_mask]
        if nonempty.empty:
            return None
        date_results = nonempty.map(lambda value: self._parse_date(value, column_name))
        date_rate = float(date_results.notna().mean())
        numeric_results = nonempty.map(self._parse_number)
        numeric_rate = float(numeric_results.notna().mean())
        date_hint = bool(re.search(r"(^|_)(date|day|ngay|time|created|updated)(_|$)", column_name))
        if date_rate >= 0.8 and (date_hint or date_rate > numeric_rate):
            parsed = pd.Series(pd.NaT, index=series.index, dtype="datetime64[ns]")
            parsed.loc[nonempty.index] = pd.to_datetime(date_results)
            raw_required = date_rate < 1.0
            return parsed.dt.date, "DATE", date_rate, raw_required
        if numeric_rate >= 0.8:
            parsed_numeric = pd.Series(pd.NA, index=series.index, dtype="Float64")
            parsed_numeric.loc[nonempty.index] = numeric_results.astype(float)
            valid = parsed_numeric.dropna()
            integral = bool(not valid.empty and valid.map(float.is_integer).all())
            raw_required = numeric_rate < 1.0
            if integral:
                return parsed_numeric.round().astype("Int64"), "BIGINT", numeric_rate, raw_required
            return parsed_numeric, "DOUBLE", numeric_rate, raw_required
        return None

    def _type_parse_warning(self, series: pd.Series, column_name: str) -> Optional[str]:
        if pd.api.types.is_numeric_dtype(series) or pd.api.types.is_datetime64_any_dtype(series):
            return None
        nonempty = series[~series.map(self._is_empty)]
        if len(nonempty) < 3:
            return None
        numeric_rate = float(nonempty.map(self._parse_number).notna().mean())
        date_rate = float(nonempty.map(lambda value: self._parse_date(value, column_name)).notna().mean())
        rate = max(numeric_rate, date_rate)
        if 0.5 <= rate < 0.8:
            return f"high_type_parse_errors:{column_name}:{rate:.1%}"
        return None

    @classmethod
    def _parse_number(cls, value: Any) -> Optional[float]:
        if isinstance(value, bool) or value is None:
            return None
        if isinstance(value, (int, float)) and not pd.isna(value):
            return float(value)
        text = cls._normalize_text(value).lower()
        if not text:
            return None
        negative = text.startswith("(") and text.endswith(")")
        percent = "%" in text
        multiplier = 1000.0 if re.search(r"k\s*$", text) else 1.0
        text = re.sub(r"(?i)(vnd|vnđ|₫|đ|k|%)", "", text)
        text = text.replace(" ", "").strip("()")
        if not re.fullmatch(r"[-+]?\d[\d.,]*", text):
            return None
        if "." in text and "," in text:
            if text.rfind(",") > text.rfind("."):
                text = text.replace(".", "").replace(",", ".")
            else:
                text = text.replace(",", "")
        elif text.count(".") > 1 or ("." in text and len(text.rsplit(".", 1)[1]) == 3):
            text = text.replace(".", "")
        elif text.count(",") > 1 or ("," in text and len(text.rsplit(",", 1)[1]) == 3):
            text = text.replace(",", "")
        else:
            text = text.replace(",", ".")
        try:
            number = float(text) * multiplier
        except ValueError:
            return None
        if negative:
            number = -abs(number)
        if percent:
            number /= 100.0
        return number

    @classmethod
    def _parse_date(cls, value: Any, column_name: str) -> Optional[date]:
        if isinstance(value, datetime):
            return value.date()
        if isinstance(value, date):
            return value
        if isinstance(value, (int, float)) and not isinstance(value, bool) and not pd.isna(value):
            if 20000 <= float(value) <= 80000 and re.search(r"date|day|ngay|time", column_name):
                return date(1899, 12, 30) + timedelta(days=int(float(value)))
            return None
        text = cls._normalize_text(value)
        if not cls._looks_like_date_text(text):
            if re.fullmatch(r"\d+(?:\.0+)?", text) and re.search(r"date|day|ngay|time", column_name):
                serial = float(text)
                if 20000 <= serial <= 80000:
                    return date(1899, 12, 30) + timedelta(days=int(serial))
            return None
        day_first = bool(re.fullmatch(r"\d{1,2}/\d{1,2}/\d{4}", text))
        parsed = pd.to_datetime(text, errors="coerce", dayfirst=day_first)
        return None if pd.isna(parsed) else parsed.date()

    @staticmethod
    def _looks_like_date_text(value: Any) -> bool:
        text = str(value).strip()
        return bool(
            re.fullmatch(r"\d{1,2}/\d{1,2}/\d{4}(?:\s+.*)?", text)
            or re.fullmatch(r"\d{4}-\d{1,2}-\d{1,2}(?:[ T].*)?", text)
        )

    @classmethod
    def _detect_encoding(cls, path: Path) -> str:
        for encoding in cls.encodings:
            try:
                path.read_text(encoding=encoding)
                return encoding
            except UnicodeDecodeError:
                continue
        return "latin-1"

    @classmethod
    def _detect_delimiter(cls, path: Path, encoding: str) -> str:
        sample = path.read_text(encoding=encoding)[:65536]
        try:
            detected = csv.Sniffer().sniff(sample, delimiters=";,\t").delimiter
            if detected in cls.delimiters:
                return detected
        except csv.Error:
            pass
        first_lines = sample.splitlines()[:10]
        return max(cls.delimiters, key=lambda delimiter: sum(line.count(delimiter) for line in first_lines))

    @classmethod
    def _detect_header_row(cls, frame: pd.DataFrame) -> int:
        limit = min(10, len(frame))
        for index in range(limit):
            values = [cls._normalize_text(value) for value in frame.iloc[index]]
            nonempty = [value for value in values if value]
            ratio = len(nonempty) / max(1, len(values))
            textual = sum(bool(re.search(r"[^\W\d_]", value, re.UNICODE)) for value in nonempty)
            if ratio >= 0.6 and textual >= max(1, math.ceil(len(nonempty) / 2)):
                return index
        return 0

    @staticmethod
    def _excel_merged_warnings(path: Path) -> Dict[str, List[str]]:
        from openpyxl import load_workbook

        workbook = load_workbook(path, read_only=False, data_only=True)
        result: Dict[str, List[str]] = {}
        try:
            for worksheet in workbook.worksheets:
                ranges = list(worksheet.merged_cells.ranges)
                complex_ranges = [
                    merged for merged in ranges
                    if merged.max_row > 2 or (merged.max_row - merged.min_row) > 0
                ]
                if len(ranges) > 1 or complex_ranges:
                    result[worksheet.title] = ["complex_merged_cells"]
        finally:
            workbook.close()
        return result

    @classmethod
    def _normalize_cell(cls, value: Any) -> Any:
        if isinstance(value, str):
            return cls._normalize_text(value)
        return value

    @staticmethod
    def _normalize_text(value: Any) -> str:
        if value is None or (isinstance(value, float) and pd.isna(value)):
            return ""
        return re.sub(r"\s+", " ", unicodedata.normalize("NFC", str(value))).strip()

    @staticmethod
    def _ascii_text(value: Any) -> str:
        normalized = unicodedata.normalize("NFKD", str(value))
        return "".join(character for character in normalized if not unicodedata.combining(character)).replace("đ", "d").replace("Đ", "D")

    @classmethod
    def _sql_identifier(cls, value: str) -> str:
        ascii_value = cls._ascii_text(cls._normalize_text(value)).lower()
        identifier = re.sub(r"[^a-z0-9_]+", "_", ascii_value).strip("_")
        if identifier and identifier[0].isdigit():
            identifier = f"c_{identifier}"
        return identifier

    def _unique_table_name(self, proposed: str) -> str:
        base = self._sql_identifier(proposed) or "table"
        name = self._deduplicate_name(base, self._used_table_names)
        self._used_table_names.add(name)
        return name

    @staticmethod
    def _deduplicate_name(base: str, used: set[str]) -> str:
        if base not in used:
            return base
        index = 2
        while f"{base}_{index}" in used:
            index += 1
        return f"{base}_{index}"

    @classmethod
    def _row_is_empty(cls, row: pd.Series) -> bool:
        return all(cls._is_empty(value) for value in row)

    @classmethod
    def _column_is_empty(cls, column: pd.Series) -> bool:
        return all(cls._is_empty(value) for value in column)

    @staticmethod
    def _is_empty(value: Any) -> bool:
        return value is None or (isinstance(value, str) and not value.strip()) or bool(pd.isna(value))

    @staticmethod
    def _quote(identifier: str) -> str:
        return '"' + identifier.replace('"', '""') + '"'
