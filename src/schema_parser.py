"""Dataset Schema & Metadata Extractor for Zalo AI Challenge - DataMate 2026.

Extracts schema information, column data types, business documentation from README.md,
and sample rows (head 5) from CSV, XLSX, and SQLite files within the evaluation directory.
Ensures token-efficient prompt construction without overflowing LLM context.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
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
    """Represents schema, column types, and sample preview of a single tabular resource."""
    name: str
    source_type: str  # 'csv', 'xlsx', 'sqlite'
    file_path: str
    columns: List[str] = field(default_factory=list)
    column_types: Dict[str, str] = field(default_factory=dict)
    sample_rows: List[Dict[str, Any]] = field(default_factory=list)
    total_rows: Optional[int] = None
    sheet_name: Optional[str] = None
    extra_notes: Optional[str] = None

    def to_markdown(self) -> str:
        """Format table schema and sample preview as clean, token-efficient Markdown."""
        lines = [f"### Table: `{self.name}` ({self.source_type.upper()})"]
        lines.append(f"- **File Path**: `{self.file_path}`")
        if self.sheet_name:
            lines.append(f"- **Sheet Name**: `{self.sheet_name}`")
        if self.total_rows is not None:
            lines.append(f"- **Approx Rows**: {self.total_rows}")

        # Columns & Types
        col_type_str = ", ".join(f"`{col}` ({self.column_types.get(col, 'string')})" for col in self.columns)
        lines.append(f"- **Columns & Types**: {col_type_str}")

        # Sample rows table (head 5)
        if self.sample_rows and self.columns:
            lines.append(f"\n**Sample Records (head {len(self.sample_rows)}):**")
            header = "| " + " | ".join(self.columns) + " |"
            sep = "| " + " | ".join("---" for _ in self.columns) + " |"
            lines.append(header)
            lines.append(sep)

            for row in self.sample_rows[:5]:
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


# ---------------------------------------------------------------------------
# Core Schema Parsing Functions
# ---------------------------------------------------------------------------

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


def parse_csv_head(file_path: Union[str, Path], n_rows: int = 5) -> TableSchema:
    """Read top 5 rows (head) of a CSV file using Pandas without full file loading.

    Extracts column names, data types, and top 5 sample rows.
    Supports encoding fallback (utf-8-sig, utf-8, latin-1, cp1252) and auto delimiter detection.

    Args:
        file_path: Path to the .csv file.
        n_rows: Number of top rows to read (default: 5).

    Returns:
        TableSchema containing columns, types, and sample rows.
    """
    path = Path(file_path).resolve()
    if not path.exists():
        logger.warning(f"CSV file not found: {path}")
        return TableSchema(name=path.name, source_type="csv", file_path=str(path))

    # Fast row count estimation without loading dataframe
    try:
        with open(path, "rb") as f:
            total_rows = sum(1 for _ in f) - 1
            if total_rows < 0:
                total_rows = 0
    except Exception:
        total_rows = None

    if _HAS_PANDAS:
        for enc in ("utf-8-sig", "utf-8", "latin-1", "cp1252"):
            try:
                # 1. Read first n_rows with standard comma
                df = pd.read_csv(path, nrows=n_rows, encoding=enc)

                # Check if file uses semicolon or tab delimiter instead of comma
                if len(df.columns) == 1 and total_rows and total_rows > 0:
                    first_col = str(df.columns[0])
                    if ";" in first_col:
                        df = pd.read_csv(path, nrows=n_rows, encoding=enc, sep=";")
                    elif "\t" in first_col:
                        df = pd.read_csv(path, nrows=n_rows, encoding=enc, sep="\t")

                columns = [str(c) for c in df.columns]
                col_types = extract_columns_and_types(df)
                sample_rows = clean_sample_records(df, n_rows=n_rows)

                return TableSchema(
                    name=path.name,
                    source_type="csv",
                    file_path=str(path),
                    columns=columns,
                    column_types=col_types,
                    sample_rows=sample_rows,
                    total_rows=total_rows,
                )
            except Exception:
                continue

    # Pure Python fallback if Pandas is unavailable or failed
    try:
        with open(path, mode="r", encoding="utf-8-sig", errors="replace") as f:
            reader = csv.DictReader(f)
            columns = [str(c) for c in (reader.fieldnames or [])]
            sample_rows = []
            for idx, row in enumerate(reader):
                if idx >= n_rows:
                    break
                sample_rows.append(row)
        return TableSchema(
            name=path.name,
            source_type="csv",
            file_path=str(path),
            columns=columns,
            column_types={c: "string" for c in columns},
            sample_rows=sample_rows,
            total_rows=total_rows,
        )
    except Exception as e:
        logger.error(f"Error parsing CSV {path}: {e}")
        return TableSchema(name=path.name, source_type="csv", file_path=str(path))


def parse_excel_head(file_path: Union[str, Path], n_rows: int = 5) -> List[TableSchema]:
    """Read top 5 rows (head) of each sheet in an Excel (.xlsx, .xls) file using Pandas.

    Extracts sheet names, column names, data types, and top 5 sample rows.

    Args:
        file_path: Path to .xlsx or .xls file.
        n_rows: Number of top rows to read (default: 5).

    Returns:
        List of TableSchema (one per sheet in workbook).
    """
    path = Path(file_path).resolve()
    if not path.exists():
        logger.warning(f"Excel file not found: {path}")
        return [TableSchema(name=path.name, source_type="xlsx", file_path=str(path))]

    schemas: List[TableSchema] = []
    if not _HAS_PANDAS:
        logger.warning("Pandas is not installed. Cannot parse Excel file.")
        return [TableSchema(name=path.name, source_type="xlsx", file_path=str(path))]

    try:
        with pd.ExcelFile(path) as excel_file:
            for sheet_name in excel_file.sheet_names:
                try:
                    df = pd.read_excel(excel_file, sheet_name=sheet_name, nrows=n_rows)
                    columns = [str(c) for c in df.columns]
                    col_types = extract_columns_and_types(df)
                    sample_rows = clean_sample_records(df, n_rows=n_rows)

                    schemas.append(
                        TableSchema(
                            name=f"{path.name} [{sheet_name}]",
                            source_type="xlsx",
                            file_path=str(path),
                            columns=columns,
                            column_types=col_types,
                            sample_rows=sample_rows,
                            sheet_name=sheet_name,
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


def parse_sqlite_head(file_path: Union[str, Path], n_rows: int = 5) -> List[TableSchema]:
    """Read top 5 rows (head) of each table in a SQLite database file.

    Args:
        file_path: Path to .sqlite / .db / .sqlite3 file.
        n_rows: Number of top rows to read (default: 5).

    Returns:
        List of TableSchema (one per user table in database).
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

                # Count rows
                cursor.execute(f"SELECT COUNT(*) FROM '{t_name}'")
                total_rows = cursor.fetchone()[0]

                # Head n_rows sample
                cursor.execute(f"SELECT * FROM '{t_name}' LIMIT {n_rows}")
                raw_rows = cursor.fetchall()
                sample_rows = [dict(zip(columns, row)) for row in raw_rows]

                schemas.append(
                    TableSchema(
                        name=f"{path.name} -> {t_name}",
                        source_type="sqlite",
                        file_path=str(path),
                        columns=columns,
                        column_types=col_types,
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


def parse_table_head(file_path: Union[str, Path], n_rows: int = 5) -> List[TableSchema]:
    """Generic dispatcher: reads 5 rows head and schema of CSV, Excel, or SQLite file.

    Args:
        file_path: File path to parse.
        n_rows: Number of top rows to read (default: 5).

    Returns:
        List of TableSchema objects.
    """
    p = Path(file_path).resolve()
    ext = p.suffix.lower()

    if ext == ".csv":
        return [parse_csv_head(p, n_rows=n_rows)]
    elif ext in (".xlsx", ".xls"):
        return parse_excel_head(p, n_rows=n_rows)
    elif ext in (".sqlite", ".db", ".sqlite3"):
        return parse_sqlite_head(p, n_rows=n_rows)
    else:
        logger.debug(f"Unsupported file format for table parsing: {p.name}")
        return []


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
    parser = SchemaParser(data_dir=data_dir)
    context = parser.parse(business_id=business_id)
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
            return self.data_dir

        # Check <data_dir>/<business_id>
        sub_dir = self.data_dir / business_id
        if sub_dir.is_dir():
            return sub_dir

        # Check direct path
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
        if not target_dir.exists():
            logger.warning(f"Target directory {target_dir} does not exist.")
            ctx = DatasetContext(data_dir=str(target_dir), business_id=business_id)
            self._cached_contexts[cache_key] = ctx
            return ctx

        # 1. Read README.md for business context
        readme_content = read_business_readme(business_id=business_id, data_dir=self.data_dir)
        as_of_date = extract_as_of_date(readme_content)

        # Also check metadata.json if exists
        for meta_name in ("metadata.json", "meta.json"):
            meta_path = target_dir / meta_name
            if meta_path.is_file():
                try:
                    meta = json.loads(meta_path.read_text(encoding="utf-8"))
                    if not as_of_date and "as_of_date" in meta:
                        as_of_date = str(meta["as_of_date"])
                except Exception as e:
                    logger.debug(f"Error reading {meta_path}: {e}")

        # 2. Inspect and parse all tabular files in target directory
        tables: List[TableSchema] = []
        file_map: Dict[str, str] = {}

        # Scan files (non-recursive in target_dir to avoid indexing sub-businesses)
        for file in sorted(target_dir.iterdir()):
            if file.is_dir() or file.name.startswith("."):
                continue

            ext = file.suffix.lower()
            if ext in (".csv", ".xlsx", ".xls", ".sqlite", ".db", ".sqlite3"):
                file_map[file.name] = str(file.resolve())
                parsed_schemas = parse_table_head(file, n_rows=n_rows)
                tables.extend(parsed_schemas)

        context = DatasetContext(
            data_dir=str(target_dir),
            business_id=business_id,
            readme_content=readme_content if readme_content else None,
            as_of_date=as_of_date,
            tables=tables,
            file_map=file_map,
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
