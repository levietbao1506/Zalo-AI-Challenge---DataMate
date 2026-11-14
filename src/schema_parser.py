"""Dataset Schema & Metadata Extractor for Zalo AI Challenge - DataMate 2026.

Extracts schema information, column data types, business documentation from README.md,
and sample rows (head 5) from CSV, XLSX, and SQLite files within the evaluation directory.
Ensures token-efficient prompt construction without overflowing LLM context.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from functools import lru_cache
import json
import logging
import math
from pathlib import Path
import re
import sqlite3
from typing import Any, Dict, List, Optional, Union

try:
    import pandas as pd
    _HAS_PANDAS = True
except ImportError:
    _HAS_PANDAS = False

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------

@dataclass
class TableSchema:
    """Represents schema, column types, categorical examples, and sample preview of a single tabular resource."""
    name: str
    source_type: str  # 'csv', 'xlsx', 'sqlite'
    file_path: str
    columns: List[str] = field(default_factory=list)
    column_types: Dict[str, str] = field(default_factory=dict)
    raw_column_types: Dict[str, str] = field(default_factory=dict)
    column_examples: Dict[str, List[str]] = field(default_factory=dict)
    date_ranges: Dict[str, str] = field(default_factory=dict)
    low_cardinality_values: Dict[str, List[Any]] = field(default_factory=dict)
    sample_rows: List[Dict[str, Any]] = field(default_factory=list)
    total_rows: Optional[int] = None
    sheet_name: Optional[str] = None
    extra_notes: Optional[str] = None

    def format_enriched_schema(self) -> str:
        """Render clean enriched schema string with top unique examples for categorical/string columns.

        Format:
            Table: orders.csv
            - order_id (int64)
            - order_date (datetime) [Date range: 2024-01-01 to 2024-12-31]
            - product_name (object) [Examples: 'Trà sữa trân châu', 'Cà phê sữa', 'Bạc xỉu']
            - status (object) [Examples: 'Hoàn thành', 'Đang giao', 'Đã hủy']
            - revenue (float64)
        """
        lines = [f"Table: {self.name}"]
        for col in self.columns:
            dtype_str = self.raw_column_types.get(col) or self.column_types.get(col, "object")
            parts = [f"- {col} ({dtype_str})"]
            if col in self.date_ranges:
                parts.append(f"[Date range: {self.date_ranges[col]}]")
            examples = self.column_examples.get(col)
            if examples:
                formatted_vals = ", ".join(repr(str(v)) for v in examples)
                parts.append(f"[Examples: {formatted_vals}]")
            elif col in self.low_cardinality_values and self.low_cardinality_values[col]:
                formatted_vals = ", ".join(repr(str(v)) for v in self.low_cardinality_values[col])
                parts.append(f"[Examples: {formatted_vals}]")
            lines.append(" ".join(parts))
        return "\n".join(lines)

    def to_markdown(self) -> str:
        """Format table schema, categorical examples, and sample preview as clean Markdown."""
        lines = [f"### Table: `{self.name}` ({self.source_type.upper()})"]
        lines.append(f"- **File Path**: `{self.file_path}`")
        if self.sheet_name:
            lines.append(f"- **Sheet Name**: `{self.sheet_name}`")
        if self.total_rows is not None:
            lines.append(f"- **Approx Rows**: {self.total_rows}")
        if self.date_ranges:
            date_range_str = ", ".join(f"`{col}`: {rng}" for col, rng in self.date_ranges.items())
            lines.append(f"- **Date Ranges**: {date_range_str}")

        # Columns & Types inline summary
        col_type_str = ", ".join(f"`{col}` ({self.column_types.get(col, 'string')})" for col in self.columns)
        lines.append(f"- **Columns & Types**: {col_type_str}")

        # Enriched Schema block with categorical examples
        if self.columns:
            lines.append("\n**Enriched Schema & Unique Value Examples:**")
            lines.append(self.format_enriched_schema())

        # Sample rows table (head 3-5)
        if self.sample_rows and self.columns:
            preview_count = min(len(self.sample_rows), 5)
            lines.append(f"\n**Sample Records (head {preview_count}):**")
            header = "| " + " | ".join(self.columns) + " |"
            sep = "| " + " | ".join("---" for _ in self.columns) + " |"
            lines.append(header)
            lines.append(sep)

            for row in self.sample_rows[:preview_count]:
                row_vals = []
                for col in self.columns:
                    val = row.get(col, "")
                    if val is None or (isinstance(val, float) and math.isnan(val)):
                        val_str = ""
                    else:
                        val_str = str(val).replace("\n", " ").replace("|", "\\|").strip()
                        if len(val_str) > 80:
                            val_str = val_str[:77] + "..."
                    row_vals.append(val_str)
                lines.append("| " + " | ".join(row_vals) + " |")

        return "\n".join(lines)

    def to_compact_string(self) -> str:
        """Render minimal one-line summary for tight context limits."""
        col_str = ", ".join(f"{col}:{self.column_types.get(col, 'string')}" for col in self.columns)
        return f"[{self.source_type.upper()}] {self.name} -> Columns: ({col_str})"


@dataclass
class DatasetContext:
    """Aggregates schema and documentation context for a specific business or dataset."""
    data_dir: str
    business_id: Optional[str] = None
    readme_content: Optional[str] = None
    as_of_date: Optional[str] = None
    tables: List[TableSchema] = field(default_factory=list)
    file_map: Dict[str, str] = field(default_factory=dict)

    def to_prompt_context(self, max_readme_chars: int = 1500) -> str:
        """Render complete dataset context for inclusion in LLM prompt.

        Formats business context from README.md and head(5) sample rows with
        column data types into a structured, readable string.
        """
        sections: List[str] = []

        # 1. Header & Business Identification
        header_lines = ["=== THÔNG TIN NGỮ CẢNH KINH DOANH (BUSINESS CONTEXT) ==="]
        if self.business_id:
            header_lines.append(f"- **Business ID**: `{self.business_id}`")
        if self.as_of_date:
            header_lines.append(f"- **Ngày tham chiếu (as_of_date)**: `{self.as_of_date}`")
        sections.append("\n".join(header_lines))

        # 2. Business Documentation (README.md)
        if self.readme_content:
            readme_snippet = self.readme_content.strip()
            if len(readme_snippet) > max_readme_chars:
                readme_snippet = readme_snippet[:max_readme_chars] + "\n... [Cắt bớt để tránh tràn ngữ cảnh]"
            sections.append(f"**Nội dung tài liệu kinh doanh (README.md):**\n```markdown\n{readme_snippet}\n```")
        else:
            sections.append("*(Không có file README.md cho doanh nghiệp này)*")

        # 3. Tabular Data Schemas & Sample Rows (head 5)
        sections.append("=== CẤU TRÚC BẢNG DỮ LIỆU & DỮ LIỆU MẪU (SCHEMAS & SAMPLE HEAD) ===")
        if not self.tables:
            sections.append("*(Không tìm thấy bảng dữ liệu CSV/XLSX/SQLite)*")
        else:
            for table in self.tables:
                sections.append(table.to_markdown())
                sections.append("-" * 40)

        return "\n\n".join(sections)

    def to_schema_string(self, max_readme_chars: int = 1500) -> str:
        """Alias for to_prompt_context()."""
        return self.to_prompt_context(max_readme_chars=max_readme_chars)

    def to_enriched_schema_string(self) -> str:
        """Return only the clean enriched schema representation across all tables."""
        return "\n\n".join(table.format_enriched_schema() for table in self.tables)


# ---------------------------------------------------------------------------
# Core Schema Parsing & Enrichment Functions
# ---------------------------------------------------------------------------

def strip_dataframe_whitespace(df: "pd.DataFrame") -> "pd.DataFrame":
    """Apply `.str.strip()` to all string/object columns in a Pandas DataFrame.

    Ensures hidden leading or trailing spaces in CSV/Excel cells do not break
    exact string matching or skew categorical unique value counts.

    Args:
        df: Input Pandas DataFrame.

    Returns:
        DataFrame with whitespace stripped from all string/object columns.
    """
    cleaned = df.copy()
    for col in cleaned.columns:
        series = cleaned[col]
        if pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series):
            non_null = series.dropna()
            if non_null.empty or non_null.map(lambda v: isinstance(v, str)).all():
                cleaned[col] = series.str.strip()
            else:
                cleaned[col] = series.apply(lambda v: v.strip() if isinstance(v, str) else v)
    return cleaned


def sample_unique_values(
    df: "pd.DataFrame",
    top_n: int = 3,
    max_val_len: int = 80,
) -> Dict[str, List[str]]:
    """Calculate top-N most frequent unique values for every object/string column.

    Drops NaNs/nulls and empty strings prior to ranking by frequency.

    Args:
        df: Cleaned Pandas DataFrame.
        top_n: Number of top frequent unique examples to extract (default: 3).
        max_val_len: Maximum character length per example value.

    Returns:
        Dictionary mapping string/object column names to lists of top unique values.
    """
    examples: Dict[str, List[str]] = {}
    for col in df.columns:
        series = df[col]
        if not (pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series)):
            continue

        non_null = series.dropna()
        if non_null.empty:
            continue

        # Convert to stripped string and ignore blank entries
        str_series = non_null.astype(str).str.strip()
        str_series = str_series[str_series != ""]
        if str_series.empty:
            continue

        top_values = str_series.value_counts().head(top_n).index.tolist()
        formatted_vals: List[str] = []
        for val in top_values:
            val_str = str(val).replace("\n", " ").strip()
            if len(val_str) > max_val_len:
                val_str = val_str[: max_val_len - 3] + "..."
            formatted_vals.append(val_str)

        if formatted_vals:
            examples[str(col)] = formatted_vals

    return examples


def extract_date_ranges(df: "pd.DataFrame") -> Dict[str, str]:
    """Detect date/datetime columns and extract min/max date ranges.

    Args:
        df: Pandas DataFrame.

    Returns:
        Dictionary mapping date column name to 'YYYY-MM-DD to YYYY-MM-DD'.
    """
    ranges: Dict[str, str] = {}
    for col in df.columns:
        series = df[col].dropna()
        if series.empty:
            continue
        col_str = str(col)
        col_lower = col_str.lower()
        is_date = False
        parsed_dates = None

        if pd.api.types.is_datetime64_any_dtype(df[col]):
            is_date = True
            parsed_dates = series
        elif re.search(r"(date|time|ngay|created|updated|thoi_gian)", col_lower):
            try:
                sample_s = series.head(100)
                converted = pd.to_datetime(sample_s, errors="coerce")
                if converted.notna().sum() >= max(1, int(len(sample_s) * 0.5)):
                    is_date = True
                    parsed_dates = pd.to_datetime(series, errors="coerce").dropna()
            except Exception:
                pass

        if is_date and parsed_dates is not None and not parsed_dates.empty:
            try:
                min_val = parsed_dates.min()
                max_val = parsed_dates.max()
                min_str = min_val.strftime("%Y-%m-%d") if hasattr(min_val, "strftime") else str(min_val)[:10]
                max_str = max_val.strftime("%Y-%m-%d") if hasattr(max_val, "strftime") else str(max_val)[:10]
                ranges[col_str] = f"{min_str} to {max_str}"
            except Exception:
                pass
    return ranges


def extract_low_cardinality_values(
    df: "pd.DataFrame",
    max_unique: int = 12,
    max_val_len: int = 60,
) -> Dict[str, List[str]]:
    """Extract all distinct values for status, category, or low-cardinality string/category columns.

    Args:
        df: Pandas DataFrame.
        max_unique: Maximum distinct values to include (default: 12).
        max_val_len: Maximum length per value.

    Returns:
        Dictionary mapping column names to sorted list of distinct values.
    """
    distinct_vals: Dict[str, List[str]] = {}
    for col in df.columns:
        series = df[col].dropna()
        if series.empty:
            continue
        col_str = str(col)
        col_lower = col_str.lower()
        is_cat_keyword = bool(
            re.search(r"(status|trang_thai|state|type|loai|category|danh_muc|flag|return|tra_hang|payment|gender|gioi_tinh)", col_lower)
        )
        # Skip purely numeric columns unless explicitly a status/flag/category keyword
        is_str_or_cat = (
            pd.api.types.is_object_dtype(series)
            or pd.api.types.is_string_dtype(series)
            or pd.api.types.is_bool_dtype(series)
            or isinstance(series.dtype, pd.CategoricalDtype)
        )
        if not is_str_or_cat and not is_cat_keyword:
            continue
        if pd.api.types.is_numeric_dtype(series) and not is_cat_keyword:
            continue

        str_series = series.astype(str).str.strip()
        str_series = str_series[str_series != ""]
        if str_series.empty:
            continue
        unique_vals = str_series.unique()
        if len(unique_vals) <= max_unique or is_cat_keyword:
            if len(unique_vals) <= max_unique:
                formatted: List[str] = []
                for v in sorted(unique_vals, key=lambda x: str(x)):
                    v_str = str(v).replace("\n", " ").strip()
                    if len(v_str) > max_val_len:
                        v_str = v_str[: max_val_len - 3] + "..."
                    formatted.append(v_str)
                if formatted:
                    distinct_vals[col_str] = formatted
    return distinct_vals


def format_table_schema(table: TableSchema) -> str:
    """Format a TableSchema object into a clean enriched schema text block."""
    return table.format_enriched_schema()


def extract_columns_and_types(df: "pd.DataFrame") -> Dict[str, str]:
    """Extract column names and normalize their data types from a Pandas DataFrame.

    Maps Pandas internal dtypes to clean, readable types (int64, float64, string, datetime, bool).

    Args:
        df: Pandas DataFrame (typically first 5 rows).

    Returns:
        Dictionary mapping column names to readable data type strings.
    """
    col_types: Dict[str, str] = {}
    for col, dtype in df.dtypes.items():
        dtype_str = str(dtype).lower()
        if "int" in dtype_str:
            clean_type = "int64" if "64" in dtype_str else "int"
        elif "float" in dtype_str or "double" in dtype_str or "decimal" in dtype_str:
            clean_type = "float64" if "64" in dtype_str else "float"
        elif "datetime" in dtype_str or "timestamp" in dtype_str:
            clean_type = "datetime"
        elif "bool" in dtype_str:
            clean_type = "bool"
        elif "category" in dtype_str:
            clean_type = "category"
        else:
            clean_type = "string"
        col_types[str(col)] = clean_type
    return col_types


def clean_sample_records(df: "pd.DataFrame", n_rows: int = 5) -> List[Dict[str, Any]]:
    """Extract and sanitize head(n_rows) sample records from DataFrame for prompt safety.

    Args:
        df: Pandas DataFrame.
        n_rows: Number of top rows to extract (default: 5).

    Returns:
        List of row dictionaries with NaN/NaT cleaned and strings formatted.
    """
    sample_df = df.head(n_rows).copy()
    records = sample_df.to_dict(orient="records")

    cleaned_records: List[Dict[str, Any]] = []
    for row in records:
        cleaned_row: Dict[str, Any] = {}
        for k, v in row.items():
            key = str(k)
            if v is None:
                cleaned_row[key] = None
            elif isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
                cleaned_row[key] = None
            elif hasattr(v, "isoformat"):
                cleaned_row[key] = v.isoformat()
            else:
                cleaned_row[key] = v
        cleaned_records.append(cleaned_row)

    return cleaned_records


def read_business_readme(
    business_id: Optional[str] = None,
    data_dir: Union[str, Path] = "./data",
) -> str:
    """Read README.md of a business_id to extract business domain context.

    Searches in:
    1. `<data_dir>/<business_id>/README.md` (or readme.md, README.txt)
    2. Direct path `<business_id>/README.md` if business_id is a path
    3. Fallback: `<data_dir>/README.md`

    Args:
        business_id: Business identifier (e.g., 'retail_01', 'doanh_nghiep_A') or None.
        data_dir: Base directory containing business subdirectories or root data.

    Returns:
        Text content of README.md, or empty string if not found.
    """
    base_dir = Path(data_dir).resolve()
    candidate_names = ["README.md", "readme.md", "README.txt", "readme.txt"]

    # Candidate search paths
    search_dirs: List[Path] = []
    if business_id:
        # Check subfolder <data_dir>/data/<business_id> (per DataMate Docker spec)
        search_dirs.append(base_dir / "data" / business_id)
        # Check subfolder <data_dir>/<business_id>
        search_dirs.append(base_dir / business_id)
        # Check if business_id itself is a valid path
        direct_p = Path(business_id)
        if direct_p.is_dir():
            search_dirs.append(direct_p.resolve())

    # Fallback to base_dir
    search_dirs.append(base_dir)

    for s_dir in search_dirs:
        if not s_dir.exists():
            continue
        for name in candidate_names:
            readme_path = s_dir / name
            if readme_path.is_file():
                try:
                    for enc in ("utf-8-sig", "utf-8", "latin-1"):
                        try:
                            content = readme_path.read_text(encoding=enc)
                            logger.info(f"Loaded business README from: {readme_path}")
                            return content.strip()
                        except UnicodeDecodeError:
                            continue
                except Exception as e:
                    logger.warning(f"Error reading {readme_path}: {e}")

    logger.debug(f"No README found for business_id '{business_id}' in {base_dir}")
    return ""


def extract_as_of_date(text: str) -> Optional[str]:
    """Extract reference as_of_date from text (README or metadata).

    Supports formats:
    - as_of_date: 2024-06-15
    - as_of = "2024-06-15"
    - Ngày tham chiếu: 2024-06-15
    """
    if not text:
        return None
    patterns = [
        r"as_of_date\s*[:=]\s*['\"]?(\d{4}-\d{2}-\d{2})['\"]?",
        r"as_of\s*[:=]\s*['\"]?(\d{4}-\d{2}-\d{2})['\"]?",
        r"ngày\s+tham\s+chiếu\s*[:=]\s*['\"]?(\d{4}-\d{2}-\d{2})['\"]?",
        r"thời\s+điểm\s+neo\s*[:=]\s*['\"]?(\d{4}-\d{2}-\d{2})['\"]?",
    ]
    for pattern in patterns:
        m = re.search(pattern, text, re.IGNORECASE)
        if m:
            return m.group(1)
    return None


def parse_csv_head(
    file_path: Union[str, Path],
    n_rows: int = 5,
    stats_rows: int = 5000,
    top_n_examples: int = 3,
) -> TableSchema:
    """Read schema, top unique categorical examples, and head(5) rows of a CSV file.

    Applies whitespace stripping across all string/object columns prior to sampling
    the top 3 most frequent non-null values and top sample records.

    Args:
        file_path: Path to the .csv file.
        n_rows: Number of top rows to keep in sample_rows (default: 5).
        stats_rows: Maximum rows to inspect for categorical frequency enrichment (default: 5000).
        top_n_examples: Top N frequent unique values to capture per string/object column (default: 3).

    Returns:
        Enriched TableSchema containing columns, types, categorical examples, and sample rows.
    """
    path = Path(file_path).resolve()
    if not path.exists():
        logger.warning(f"CSV file not found: {path}")
        return TableSchema(name=path.name, source_type="csv", file_path=str(path))

    # Fast row count estimation without loading full dataframe
    try:
        with open(path, "rb") as f:
            total_rows = sum(1 for _ in f) - 1
            if total_rows < 0:
                total_rows = 0
    except Exception:
        total_rows = None

    read_limit = max(n_rows, stats_rows) if stats_rows else n_rows

    if _HAS_PANDAS:
        for enc in ("utf-8-sig", "utf-8", "latin-1", "cp1252"):
            try:
                # 1. Read sample window with standard comma
                df = pd.read_csv(path, nrows=read_limit, encoding=enc)

                # Check if file uses semicolon or tab delimiter instead of comma
                if len(df.columns) == 1 and total_rows and total_rows > 0:
                    first_col = str(df.columns[0])
                    if ";" in first_col:
                        df = pd.read_csv(path, nrows=read_limit, encoding=enc, sep=";")
                    elif "\t" in first_col:
                        df = pd.read_csv(path, nrows=read_limit, encoding=enc, sep="\t")

                # 2. Clean leading/trailing whitespace on all string/object columns
                df = strip_dataframe_whitespace(df)

                columns = [str(c) for c in df.columns]
                col_types = extract_columns_and_types(df)
                raw_col_types = {str(col): str(dtype) for col, dtype in df.dtypes.items()}
                date_ranges = extract_date_ranges(df)
                low_card = extract_low_cardinality_values(df)
                col_examples = sample_unique_values(df, top_n=top_n_examples)
                for col, vals in low_card.items():
                    if col not in col_examples:
                        col_examples[col] = vals
                sample_rows = clean_sample_records(df, n_rows=n_rows)

                return TableSchema(
                    name=path.name,
                    source_type="csv",
                    file_path=str(path),
                    columns=columns,
                    column_types=col_types,
                    raw_column_types=raw_col_types,
                    column_examples=col_examples,
                    date_ranges=date_ranges,
                    low_cardinality_values=low_card,
                    sample_rows=sample_rows,
                    total_rows=total_rows,
                )
            except Exception:
                continue

    # Pure Python fallback if Pandas is unavailable or failed
    try:
        from collections import Counter
        with open(path, mode="r", encoding="utf-8-sig", errors="replace") as f:
            reader = csv.DictReader(f)
            columns = [str(c) for c in (reader.fieldnames or [])]
            sample_rows = []
            col_counters: Dict[str, Counter] = {c: Counter() for c in columns}
            for idx, row in enumerate(reader):
                if idx >= read_limit:
                    break
                cleaned_row = {
                    str(k): (v.strip() if isinstance(v, str) else v)
                    for k, v in row.items()
                }
                if idx < n_rows:
                    sample_rows.append(cleaned_row)
                for c in columns:
                    val = cleaned_row.get(c)
                    if val is not None and str(val).strip() != "":
                        col_counters[c][str(val).strip()] += 1

        col_examples = {
            c: [val for val, _ in col_counters[c].most_common(top_n_examples)]
            for c in columns
            if col_counters[c]
        }
        return TableSchema(
            name=path.name,
            source_type="csv",
            file_path=str(path),
            columns=columns,
            column_types={c: "string" for c in columns},
            raw_column_types={c: "object" for c in columns},
            column_examples=col_examples,
            sample_rows=sample_rows,
            total_rows=total_rows,
        )
    except Exception as e:
        logger.error(f"Error parsing CSV {path}: {e}")
        return TableSchema(name=path.name, source_type="csv", file_path=str(path))


def parse_excel_head(
    file_path: Union[str, Path],
    n_rows: int = 5,
    stats_rows: int = 5000,
    top_n_examples: int = 3,
) -> List[TableSchema]:
    """Read schema, categorical examples, and head(5) rows of each sheet in an Excel workbook.

    Applies whitespace stripping across all string/object columns before calculating
    top unique values and sample records.

    Args:
        file_path: Path to .xlsx or .xls file.
        n_rows: Number of top rows to keep in sample_rows (default: 5).
        stats_rows: Maximum rows to read per sheet for frequency sampling (default: 5000).
        top_n_examples: Top N frequent unique values per string/object column (default: 3).

    Returns:
        List of enriched TableSchema (one per sheet in workbook).
    """
    path = Path(file_path).resolve()
    if not path.exists():
        logger.warning(f"Excel file not found: {path}")
        return [TableSchema(name=path.name, source_type="xlsx", file_path=str(path))]

    schemas: List[TableSchema] = []
    if not _HAS_PANDAS:
        logger.warning("Pandas is not installed. Cannot parse Excel file.")
        return [TableSchema(name=path.name, source_type="xlsx", file_path=str(path))]

    read_limit = max(n_rows, stats_rows) if stats_rows else n_rows

    try:
        with pd.ExcelFile(path) as excel_file:
            for sheet_name in excel_file.sheet_names:
                try:
                    df = pd.read_excel(excel_file, sheet_name=sheet_name, nrows=read_limit)
                    df = strip_dataframe_whitespace(df)

                    columns = [str(c) for c in df.columns]
                    col_types = extract_columns_and_types(df)
                    raw_col_types = {str(col): str(dtype) for col, dtype in df.dtypes.items()}
                    date_ranges = extract_date_ranges(df)
                    low_card = extract_low_cardinality_values(df)
                    col_examples = sample_unique_values(df, top_n=top_n_examples)
                    for col, vals in low_card.items():
                        if col not in col_examples:
                            col_examples[col] = vals
                    sample_rows = clean_sample_records(df, n_rows=n_rows)

                    schemas.append(
                        TableSchema(
                            name=f"{path.name} [{sheet_name}]",
                            source_type="xlsx",
                            file_path=str(path),
                            columns=columns,
                            column_types=col_types,
                            raw_column_types=raw_col_types,
                            column_examples=col_examples,
                            date_ranges=date_ranges,
                            low_cardinality_values=low_card,
                            sample_rows=sample_rows,
                            sheet_name=sheet_name,
                            total_rows=len(df),
                        )
                    )
                except Exception as sheet_err:
                    logger.warning(f"Error parsing sheet '{sheet_name}' in {path}: {sheet_err}")
                    schemas.append(
                        TableSchema(
                            name=f"{path.name} [{sheet_name}]",
                            source_type="xlsx",
                            file_path=str(path),
                            sheet_name=sheet_name,
                        )
                    )
    except Exception as e:
        logger.error(f"Error reading Excel file {path}: {e}")
        schemas.append(TableSchema(name=path.name, source_type="xlsx", file_path=str(path)))

    return schemas


def parse_sqlite_head(
    file_path: Union[str, Path],
    n_rows: int = 5,
    top_n_examples: int = 3,
) -> List[TableSchema]:
    """Read schema, top unique string examples, and head(5) rows of each table in SQLite.

    Args:
        file_path: Path to .sqlite / .db / .sqlite3 file.
        n_rows: Number of top rows to read (default: 5).
        top_n_examples: Top N frequent unique values per text column (default: 3).

    Returns:
        List of enriched TableSchema (one per user table in database).
    """
    path = Path(file_path).resolve()
    if not path.exists():
        return [TableSchema(name=path.name, source_type="sqlite", file_path=str(path))]

    schemas: List[TableSchema] = []
    conn = None
    try:
        conn = sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True)
        cursor = conn.cursor()

        # Retrieve all user tables
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%';")
        table_names = [row[0] for row in cursor.fetchall()]

        for t_name in table_names:
            try:
                # Column definitions
                cursor.execute(f"PRAGMA table_info('{t_name}')")
                cols_info = cursor.fetchall()
                columns = [str(c[1]) for c in cols_info]
                col_types = {str(c[1]): (c[2] or "TEXT").lower() for c in cols_info}
                raw_col_types = {
                    str(c[1]): ("object" if "char" in (c[2] or "text").lower() or "text" in (c[2] or "text").lower() or "clob" in (c[2] or "text").lower() else (c[2] or "object").lower())
                    for c in cols_info
                }

                # Count rows
                cursor.execute(f"SELECT COUNT(*) FROM '{t_name}'")
                total_rows = cursor.fetchone()[0]

                # Head n_rows sample (stripping string whitespace)
                cursor.execute(f"SELECT * FROM '{t_name}' LIMIT {n_rows}")
                raw_rows = cursor.fetchall()
                sample_rows = [
                    {
                        col: (val.strip() if isinstance(val, str) else val)
                        for col, val in zip(columns, row)
                    }
                    for row in raw_rows
                ]

                # Sample top unique values for text/object columns and date ranges
                col_examples: Dict[str, List[str]] = {}
                date_ranges: Dict[str, str] = {}
                for col in columns:
                    c_lower = col.lower()
                    if "date" in c_lower or "time" in c_lower or "ngay" in c_lower:
                        try:
                            safe_col = col.replace('"', '""')
                            safe_tbl = t_name.replace('"', '""')
                            cursor.execute(
                                f'SELECT MIN("{safe_col}"), MAX("{safe_col}") '
                                f'FROM "{safe_tbl}" '
                                f'WHERE "{safe_col}" IS NOT NULL AND TRIM("{safe_col}") != ""'
                            )
                            min_val, max_val = cursor.fetchone()
                            if min_val is not None and max_val is not None:
                                date_ranges[col] = f"{str(min_val)[:10]} to {str(max_val)[:10]}"
                        except Exception:
                            pass

                    if raw_col_types.get(col) == "object" or "text" in col_types.get(col, ""):
                        try:
                            safe_col = col.replace('"', '""')
                            safe_tbl = t_name.replace('"', '""')
                            cursor.execute(
                                f'SELECT TRIM("{safe_col}") AS v, COUNT(*) AS cnt '
                                f'FROM "{safe_tbl}" '
                                f'WHERE "{safe_col}" IS NOT NULL AND TRIM("{safe_col}") != "" '
                                f'GROUP BY TRIM("{safe_col}") '
                                f'ORDER BY cnt DESC LIMIT {top_n_examples}'
                            )
                            vals = [str(r[0]) for r in cursor.fetchall() if r[0] is not None]
                            if vals:
                                col_examples[col] = vals
                        except Exception:
                            pass

                schemas.append(
                    TableSchema(
                        name=f"{path.name} -> {t_name}",
                        source_type="sqlite",
                        file_path=str(path),
                        columns=columns,
                        column_types=col_types,
                        raw_column_types=raw_col_types,
                        column_examples=col_examples,
                        date_ranges=date_ranges,
                        sample_rows=sample_rows,
                        total_rows=total_rows,
                        extra_notes=f"SQLite Table: {t_name}",
                    )
                )
            except Exception as t_err:
                logger.warning(f"Error reading SQLite table '{t_name}' in {path}: {t_err}")

    except Exception as e:
        logger.error(f"Error opening SQLite database {path}: {e}")
    finally:
        if conn is not None:
            conn.close()

    return schemas


def parse_table_head(
    file_path: Union[str, Path],
    n_rows: int = 5,
    top_n_examples: int = 3,
) -> List[TableSchema]:
    """Generic dispatcher: reads schema, categorical examples, and head rows of CSV, Excel, or SQLite.

    Args:
        file_path: File path to parse.
        n_rows: Number of top rows to read (default: 5).
        top_n_examples: Top N frequent unique values per string/object column (default: 3).

    Returns:
        List of TableSchema objects.
    """
    p = Path(file_path).resolve()
    ext = p.suffix.lower()

    if ext == ".csv":
        return [parse_csv_head(p, n_rows=n_rows, top_n_examples=top_n_examples)]
    elif ext in (".xlsx", ".xls"):
        return parse_excel_head(p, n_rows=n_rows, top_n_examples=top_n_examples)
    elif ext in (".sqlite", ".db", ".sqlite3"):
        return parse_sqlite_head(p, n_rows=n_rows, top_n_examples=top_n_examples)
    else:
        logger.debug(f"Unsupported file format for table parsing: {p.name}")
        return []


def _scan_directory_context(
    target_dir: Path,
    business_id: Optional[str] = None,
    readme_base_dir: Optional[Path] = None,
    n_rows: int = 5,
) -> DatasetContext:
    """Scan a resolved dataset directory for README, metadata, and tabular schemas."""
    if not target_dir.exists():
        logger.warning(f"Target directory {target_dir} does not exist.")
        return DatasetContext(data_dir=str(target_dir), business_id=business_id)

    readme_root = readme_base_dir if readme_base_dir is not None else target_dir
    readme_content = read_business_readme(business_id=business_id, data_dir=readme_root)
    # Also check directly inside target_dir for README if not found
    if not readme_content and target_dir.is_dir():
        for name in ("README.md", "readme.md", "README.txt", "readme.txt"):
            r_p = target_dir / name
            if r_p.is_file():
                try:
                    for enc in ("utf-8-sig", "utf-8", "latin-1"):
                        try:
                            txt = r_p.read_text(encoding=enc).strip()
                            if txt:
                                readme_content = txt
                                break
                        except UnicodeDecodeError:
                            continue
                    if readme_content:
                        break
                except Exception:
                    pass

    as_of_date = extract_as_of_date(readme_content)

    for meta_name in ("metadata.json", "meta.json"):
        meta_path = target_dir / meta_name
        if meta_path.is_file():
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                if not as_of_date and "as_of_date" in meta:
                    as_of_date = str(meta["as_of_date"])
            except Exception as e:
                logger.debug(f"Error reading {meta_path}: {e}")

    tables: List[TableSchema] = []
    file_map: Dict[str, str] = {}

    candidate_files: List[Path] = []
    for item in sorted(target_dir.iterdir()):
        if item.is_file() and not item.name.startswith("."):
            ext = item.suffix.lower()
            if ext in (".csv", ".xlsx", ".xls", ".sqlite", ".db", ".sqlite3"):
                candidate_files.append(item)
        elif item.is_dir() and not item.name.startswith(".") and item.name not in ("__pycache__", ".venv", ".git", ".cache", "sqlite_copies"):
            for sub in sorted(item.iterdir()):
                if sub.is_file() and not sub.name.startswith("."):
                    ext = sub.suffix.lower()
                    if ext in (".csv", ".xlsx", ".xls", ".sqlite", ".db", ".sqlite3"):
                        candidate_files.append(sub)

    for file in candidate_files:
        file_map[file.name] = str(file.resolve())
        if file.parent != target_dir:
            file_map[f"{file.parent.name}/{file.name}"] = str(file.resolve())
        tables.extend(parse_table_head(file, n_rows=n_rows))

    return DatasetContext(
        data_dir=str(target_dir),
        business_id=business_id,
        readme_content=readme_content if readme_content else None,
        as_of_date=as_of_date,
        tables=tables,
        file_map=file_map,
    )


@lru_cache(maxsize=256)
def get_business_schema(
    business_dir: str,
    max_readme_chars: int = 1500,
    n_rows: int = 5,
) -> str:
    """Read tabular files and README inside ``business_dir`` and return a cached schema string.

    Decorated with ``@lru_cache(maxsize=256)`` so repeated calls for the same
    ``business_dir`` immediately return the cached string without hitting disk.

    Args:
        business_dir: Path to the directory containing README.md and CSV/XLSX/SQLite files.
        max_readme_chars: Maximum characters of README to include in prompt (default: 1500).
        n_rows: Number of sample rows to extract per table (default: 5).

    Returns:
        Formatted schema string ready to inject into ``PromptBuilder``.
    """
    target_dir = Path(business_dir).resolve()
    context = _scan_directory_context(target_dir=target_dir, business_id=None, n_rows=n_rows)
    return context.to_prompt_context(max_readme_chars=max_readme_chars)


def build_schema_string(
    business_id: Optional[str] = None,
    data_dir: Union[str, Path] = "./data",
    max_readme_chars: int = 1500,
    n_rows: int = 5,
) -> str:
    """Extract README context and head(5) schema of all tables, returning clean prompt string.

    Convenience all-in-one function for Bước 2 (Schema Parser).

    Args:
        business_id: ID of the business (e.g. 'retail_01') or None.
        data_dir: Directory containing the business datasets.
        max_readme_chars: Maximum characters of README to include in prompt.
        n_rows: Number of sample rows to preview (default: 5).

    Returns:
        Neat schema string ready to be injected into PromptBuilder.
    """
    if not business_id:
        return get_business_schema(
            str(Path(data_dir).resolve()),
            max_readme_chars=max_readme_chars,
            n_rows=n_rows,
        )

    parser = SchemaParser(data_dir=data_dir)
    context = parser.parse(business_id=business_id, n_rows=n_rows)
    return context.to_prompt_context(max_readme_chars=max_readme_chars)


# ---------------------------------------------------------------------------
# SchemaParser Class
# ---------------------------------------------------------------------------

class SchemaParser:
    """Parses schema, head(5) sample rows, and business documentation from datasets.

    Supports single-business and multi-business directories. Caches parsed
    contexts to avoid redundant I/O during test execution.
    """

    def __init__(self, data_dir: Union[str, Path] = "./data"):
        self.data_dir = Path(data_dir).resolve()
        self._cached_contexts: Dict[str, DatasetContext] = {}

    def get_business_dir(self, business_id: Optional[str] = None) -> Path:
        """Resolve directory path for a specific business_id."""
        if not business_id:
            data_sub = self.data_dir / "data"
            if data_sub.is_dir():
                tabular_in_data = any(
                    f.is_file() and f.suffix.lower() in (".csv", ".xlsx", ".xls", ".sqlite", ".db", ".sqlite3")
                    for f in data_sub.iterdir()
                )
                if tabular_in_data:
                    return data_sub
            return self.data_dir

        # 1. Check <data_dir>/data/<business_id> (per DataMate Docker spec)
        data_sub_dir = self.data_dir / "data" / business_id
        if data_sub_dir.is_dir():
            return data_sub_dir

        # 2. Check <data_dir>/<business_id>
        sub_dir = self.data_dir / business_id
        if sub_dir.is_dir():
            return sub_dir

        # 3. Check direct path
        direct_p = Path(business_id)
        if direct_p.is_dir():
            return direct_p.resolve()

        return self.data_dir

    def read_readme(self, business_id: Optional[str] = None) -> str:
        """Read README.md of the given business_id."""
        return read_business_readme(business_id=business_id, data_dir=self.data_dir)

    def parse(
        self,
        business_id: Optional[str] = None,
        force_refresh: bool = False,
        n_rows: int = 5,
    ) -> DatasetContext:
        """Parse all tabular resources and business README for a given business_id.

        Results are cached per business_id.

        Args:
            business_id: Identifier of the business, or None for root data directory.
            force_refresh: If True, re-parses files ignoring cached result.
            n_rows: Number of top rows to read per table (default: 5).

        Returns:
            DatasetContext aggregating all table schemas, README context, and file mapping.
        """
        cache_key = business_id or "__root__"
        if not force_refresh and cache_key in self._cached_contexts:
            return self._cached_contexts[cache_key]

        target_dir = self.get_business_dir(business_id)
        context = _scan_directory_context(
            target_dir=target_dir,
            business_id=business_id,
            readme_base_dir=self.data_dir,
            n_rows=n_rows,
        )

        self._cached_contexts[cache_key] = context
        return context

    def parse_business(
        self,
        business_id: str,
        force_refresh: bool = False,
        n_rows: int = 5,
    ) -> DatasetContext:
        """Alias for parse(business_id=business_id)."""
        return self.parse(business_id=business_id, force_refresh=force_refresh, n_rows=n_rows)

    def get_schema_string(
        self,
        business_id: Optional[str] = None,
        max_readme_chars: int = 1500,
    ) -> str:
        """Extract and format schema string for prompt injection."""
        ctx = self.parse(business_id=business_id)
        return ctx.to_prompt_context(max_readme_chars=max_readme_chars)

    def list_businesses(self) -> List[str]:
        """List all discovered business subdirectories in data_dir."""
        if not self.data_dir.exists():
            return []
        businesses = []
        for p in sorted(self.data_dir.iterdir()):
            if p.is_dir() and not p.name.startswith("."):
                businesses.append(p.name)
        return businesses
