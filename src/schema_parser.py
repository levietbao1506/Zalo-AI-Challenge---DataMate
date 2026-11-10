"""Dataset Schema & Metadata Extractor for DataMate 2026.

Extracts schema information, column data types, business documentation from README.md,
and sample rows (head 5) from CSV, XLSX, and SQLite files within the evaluation directory.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
import json
import logging
from pathlib import Path
import sqlite3
from typing import Any, Dict, List, Optional, Union

try:
    import pandas as pd
    _HAS_PANDAS = True
except ImportError:
    _HAS_PANDAS = False

logger = logging.getLogger(__name__)


@dataclass
class TableSchema:
    """Represents schema and sample preview of a single tabular resource."""
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
        """Format table schema and sample preview as compact Markdown."""
        lines = [f"### Table / Resource: `{self.name}` ({self.source_type.upper()})"]
        lines.append(f"- **File Path**: `{self.file_path}`")
        if self.sheet_name:
            lines.append(f"- **Sheet Name**: `{self.sheet_name}`")
        if self.total_rows is not None:
            lines.append(f"- **Approx Rows**: {self.total_rows}")

        # Columns & Types
        col_type_str = ", ".join(f"`{col}` ({self.column_types.get(col, 'UNKNOWN')})" for col in self.columns)
        lines.append(f"- **Columns**: {col_type_str}")

        # Sample rows table
        if self.sample_rows and self.columns:
            lines.append("\n**Sample Rows (head 5):**")
            # Header
            header = "| " + " | ".join(self.columns) + " |"
            sep = "| " + " | ".join("---" for _ in self.columns) + " |"
            lines.append(header)
            lines.append(sep)

            for row in self.sample_rows[:5]:
                row_vals = [str(row.get(col, "")).replace("\n", " ").strip() for col in self.columns]
                lines.append("| " + " | ".join(row_vals) + " |")

        return "\n".join(lines)


@dataclass
class DatasetContext:
    """Aggregates all schema and documentation context for the dataset."""
    data_dir: str
    readme_content: Optional[str] = None
    as_of_date: Optional[str] = None
    tables: List[TableSchema] = field(default_factory=list)
    file_map: Dict[str, str] = field(default_factory=dict)

    def to_prompt_context(self, max_readme_chars: int = 1500) -> str:
        """Render complete dataset context for inclusion in LLM prompt."""
        sections = []

        # Reference as_of_date if extracted from README or metadata
        if self.as_of_date:
            sections.append(f"**Reference as_of_date**: `{self.as_of_date}`\n")

        # Documentation
        if self.readme_content:
            readme_snippet = self.readme_content.strip()
            if len(readme_snippet) > max_readme_chars:
                readme_snippet = readme_snippet[:max_readme_chars] + "\n... [truncated]"
            sections.append(f"**Dataset Documentation (README.md):**\n```\n{readme_snippet}\n```\n")

        # Tables
        sections.append("**Tabular Data Schemas & Sample Rows:**\n")
        for table in self.tables:
            sections.append(table.to_markdown())
            sections.append("\n" + "-" * 40 + "\n")

        return "\n".join(sections)


class SchemaParser:
    """Parses schema, head(5) sample rows, and metadata from tabular datasets."""

    def __init__(self, data_dir: Union[str, Path]):
        self.data_dir = Path(data_dir).resolve()
        self._cached_context: Optional[DatasetContext] = None

    def parse(self, force_refresh: bool = False) -> DatasetContext:
        """Parse all data files in the directory. Results are cached."""
        if self._cached_context is not None and not force_refresh:
            return self._cached_context

        if not self.data_dir.exists():
            logger.warning(f"Data directory {self.data_dir} does not exist.")
            return DatasetContext(data_dir=str(self.data_dir))

        tables: List[TableSchema] = []
        file_map: Dict[str, str] = {}
        readme_content: Optional[str] = None
        as_of_date: Optional[str] = None

        # 1. Parse README.md / metadata.json
        readme_path = self._find_file(["README.md", "readme.md", "README.txt"])
        if readme_path and readme_path.exists():
            try:
                readme_content = readme_path.read_text(encoding="utf-8", errors="replace")
                as_of_date = self._extract_as_of_date(readme_content)
            except Exception as e:
                logger.error(f"Failed to read README: {e}")

        metadata_path = self._find_file(["metadata.json", "meta.json"])
        if metadata_path and metadata_path.exists():
            try:
                meta = json.loads(metadata_path.read_text(encoding="utf-8"))
                if not as_of_date and "as_of_date" in meta:
                    as_of_date = str(meta["as_of_date"])
            except Exception as e:
                logger.error(f"Failed to parse metadata.json: {e}")

        # 2. Inspect all files in data_dir
        for file in sorted(self.data_dir.iterdir()):
            if file.is_dir() or file.name.startswith("."):
                continue

            file_map[file.name] = str(file)
            ext = file.suffix.lower()

            if ext == ".csv":
                tables.append(self._parse_csv(file))
            elif ext in (".xlsx", ".xls"):
                tables.extend(self._parse_excel(file))
            elif ext in (".sqlite", ".db", ".sqlite3"):
                tables.extend(self._parse_sqlite(file))

        context = DatasetContext(
            data_dir=str(self.data_dir),
            readme_content=readme_content,
            as_of_date=as_of_date,
            tables=tables,
            file_map=file_map,
        )
        self._cached_context = context
        return context

    def _find_file(self, candidates: List[str]) -> Optional[Path]:
        for name in candidates:
            p = self.data_dir / name
            if p.exists():
                return p
        return None

    def _extract_as_of_date(self, text: str) -> Optional[str]:
        import re
        m = re.search(r"as_of_date\s*[:=]\s*['\"]?(\d{4}-\d{2}-\d{2})['\"]?", text, re.IGNORECASE)
        if m:
            return m.group(1)
        m_vn = re.search(r"ngày\s+tham\s+chiếu\s*[:=]\s*['\"]?(\d{4}-\d{2}-\d{2})['\"]?", text, re.IGNORECASE)
        if m_vn:
            return m_vn.group(1)
        return None

    def _parse_csv(self, file_path: Path) -> TableSchema:
        """Extract schema and head(5) from a CSV file."""
        if _HAS_PANDAS:
            for enc in ("utf-8-sig", "utf-8", "latin-1"):
                try:
                    df = pd.read_csv(file_path, nrows=5, encoding=enc)
                    # Quick estimate of total rows
                    try:
                        with open(file_path, "rb") as f:
                            num_rows = sum(1 for _ in f) - 1
                    except Exception:
                        num_rows = None

                    col_types = {col: str(dtype) for col, dtype in df.dtypes.items()}
                    sample_records = df.head(5).to_dict(orient="records")
                    return TableSchema(
                        name=file_path.name,
                        source_type="csv",
                        file_path=str(file_path),
                        columns=list(df.columns),
                        column_types=col_types,
                        sample_rows=sample_records,
                        total_rows=num_rows,
                    )
                except Exception:
                    continue

        # Fallback pure python csv reader
        try:
            with open(file_path, mode="r", encoding="utf-8-sig", errors="replace") as f:
                reader = csv.DictReader(f)
                columns = reader.fieldnames or []
                sample_rows = []
                for idx, row in enumerate(reader):
                    if idx >= 5:
                        break
                    sample_rows.append(row)
            return TableSchema(
                name=file_path.name,
                source_type="csv",
                file_path=str(file_path),
                columns=list(columns),
                column_types={c: "string" for c in columns},
                sample_rows=sample_rows,
            )
        except Exception as e:
            logger.error(f"Error parsing CSV {file_path}: {e}")
            return TableSchema(name=file_path.name, source_type="csv", file_path=str(file_path))

    def _parse_excel(self, file_path: Path) -> List[TableSchema]:
        """Extract schema and head(5) from all sheets in an Excel workbook."""
        schemas: List[TableSchema] = []
        if not _HAS_PANDAS:
            return [TableSchema(name=file_path.name, source_type="xlsx", file_path=str(file_path))]

        try:
            excel_file = pd.ExcelFile(file_path)
            for sheet in excel_file.sheet_names:
                try:
                    df = pd.read_excel(file_path, sheet_name=sheet, nrows=5)
                    col_types = {col: str(dtype) for col, dtype in df.dtypes.items()}
                    sample_records = df.head(5).to_dict(orient="records")
                    schemas.append(
                        TableSchema(
                            name=f"{file_path.name} [{sheet}]",
                            source_type="xlsx",
                            file_path=str(file_path),
                            columns=list(df.columns),
                            column_types=col_types,
                            sample_rows=sample_records,
                            sheet_name=sheet,
                        )
                    )
                except Exception as sheet_err:
                    logger.warning(f"Error parsing sheet {sheet} in {file_path}: {sheet_err}")
        except Exception as e:
            logger.error(f"Error reading Excel file {file_path}: {e}")
            schemas.append(TableSchema(name=file_path.name, source_type="xlsx", file_path=str(file_path)))

        return schemas

    def _parse_sqlite(self, file_path: Path) -> List[TableSchema]:
        """Extract schema and head(5) from all tables in a SQLite database."""
        schemas: List[TableSchema] = []
        try:
            conn = sqlite3.connect(f"file:{file_path}?mode=ro", uri=True)
            cursor = conn.cursor()

            # Retrieve all tables
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%';")
            table_names = [row[0] for row in cursor.fetchall()]

            for t_name in table_names:
                try:
                    # Column definitions
                    cursor.execute(f"PRAGMA table_info('{t_name}')")
                    cols_info = cursor.fetchall()
                    columns = [c[1] for c in cols_info]
                    col_types = {c[1]: c[2] or "TEXT" for c in cols_info}

                    # Total rows count
                    cursor.execute(f"SELECT COUNT(*) FROM '{t_name}'")
                    total_rows = cursor.fetchone()[0]

                    # Sample 5 rows
                    cursor.execute(f"SELECT * FROM '{t_name}' LIMIT 5")
                    raw_rows = cursor.fetchall()
                    sample_rows = [dict(zip(columns, row)) for row in raw_rows]

                    schemas.append(
                        TableSchema(
                            name=f"{file_path.name} -> {t_name}",
                            source_type="sqlite",
                            file_path=str(file_path),
                            columns=columns,
                            column_types=col_types,
                            sample_rows=sample_rows,
                            total_rows=total_rows,
                            extra_notes=f"SQLite Table: {t_name}",
                        )
                    )
                except Exception as t_err:
                    logger.warning(f"Error reading table {t_name} from {file_path}: {t_err}")

            conn.close()
        except Exception as e:
            logger.error(f"Error opening SQLite database {file_path}: {e}")

        return schemas
