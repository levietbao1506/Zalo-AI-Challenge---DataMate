"""Dataset Inspection Script for Zalo AI Challenge 2026 - DataMate Track.

Inspects table schemas, row counts, and date ranges without loading entire tables into memory.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sqlite3
from typing import Any, Dict

import pandas as pd


def inspect_csv(path: Path) -> Dict[str, Any]:
    """Inspect CSV columns and approximate row count."""
    try:
        df_head = pd.read_csv(path, nrows=5)
        with open(path, "rb") as f:
            line_count = sum(1 for _ in f) - 1
        return {
            "type": "CSV",
            "rows": max(0, line_count),
            "columns": list(df_head.columns),
            "dtypes": {c: str(t) for c, t in df_head.dtypes.items()},
        }
    except Exception as e:
        return {"type": "CSV", "error": str(e)}


def inspect_business(biz_dir: Path) -> Dict[str, Any]:
    """Inspect all data files in a business folder."""
    res = {}
    for f in biz_dir.iterdir():
        if f.suffix.lower() == ".csv":
            res[f.name] = inspect_csv(f)
        elif f.suffix.lower() in (".sqlite", ".db"):
            try:
                conn = sqlite3.connect(f)
                cur = conn.cursor()
                cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
                tables = [r[0] for r in cur.fetchall()]
                conn.close()
                res[f.name] = {"type": "SQLite", "tables": tables}
            except Exception as e:
                res[f.name] = {"type": "SQLite", "error": str(e)}
    return res


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect DataMate business tables")
    parser.add_argument("--data_dir", type=str, default="./data", help="Root data directory")
    args = parser.parse_args()

    data_path = Path(args.data_dir).resolve()
    biz_base = data_path / "data" if (data_path / "data").is_dir() else data_path

    print(f"\n--- DATASET INSPECTION: {data_path} ---")
    for b in biz_base.iterdir():
        if b.is_dir() and not b.name.startswith("."):
            info = inspect_business(b)
            print(f"\nBusiness: {b.name}")
            for fname, meta in info.items():
                print(f"  [{meta.get('type')}] {fname}: {meta.get('rows', '?')} rows, cols={meta.get('columns', meta.get('tables', []))}")


if __name__ == "__main__":
    main()
