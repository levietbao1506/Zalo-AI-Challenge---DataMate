"""Dataset Validation Script for Zalo AI Challenge 2026 - DataMate Track.

Validates the integrity of questions.jsonl, rulebook.md, calendar.csv, and multi-file business folders.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import sqlite3
import sys
from typing import Any, Dict, List

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("validate_data")


def validate_questions_file(questions_path: Path) -> int:
    """Validate questions.jsonl integrity."""
    if not questions_path.exists():
        logger.warning(f"Questions file not found: {questions_path}")
        return 0

    valid_count = 0
    with open(questions_path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f, 1):
            line_str = line.strip()
            if not line_str:
                continue
            try:
                rec = json.loads(line_str)
                assert "id" in rec or "question_id" in rec, "Missing 'id'"
                assert "question" in rec, "Missing 'question'"
                valid_count += 1
            except Exception as e:
                logger.error(f"Invalid record at line {idx}: {e}")

    logger.info(f"Validated {valid_count} questions in {questions_path}")
    return valid_count


def validate_business_folder(biz_dir: Path) -> Dict[str, Any]:
    """Validate files inside a business directory."""
    stats = {"csv": 0, "xlsx": 0, "sqlite": 0, "readme": False}
    if not biz_dir.is_dir():
        return stats

    for f in biz_dir.iterdir():
        if f.suffix.lower() == ".csv":
            stats["csv"] += 1
        elif f.suffix.lower() in (".xlsx", ".xls"):
            stats["xlsx"] += 1
        elif f.suffix.lower() in (".sqlite", ".db", ".sqlite3"):
            try:
                conn = sqlite3.connect(f)
                cursor = conn.cursor()
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
                tables = cursor.fetchall()
                conn.close()
                stats["sqlite"] += 1
            except Exception as e:
                logger.warning(f"Failed to inspect SQLite database {f}: {e}")
        elif f.name.lower() in ("readme.md", "readme.txt"):
            stats["readme"] = True

    return stats


def validate_dataset(data_dir: Path) -> Dict[str, Any]:
    """Validate full dataset layout under data_dir."""
    logger.info(f"Validating dataset at: {data_dir}")
    summary = {
        "questions_count": 0,
        "has_rulebook": False,
        "has_calendar": False,
        "businesses": {},
    }

    # Check questions.jsonl
    q_file = data_dir / "questions.jsonl"
    summary["questions_count"] = validate_questions_file(q_file)

    # Check rulebook.md
    rb_file = data_dir / "rulebook.md"
    summary["has_rulebook"] = rb_file.is_file()

    # Check calendar.csv
    cal_file = data_dir / "calendar.csv"
    summary["has_calendar"] = cal_file.is_file()

    # Check business directories under data_dir/data or data_dir
    biz_base = data_dir / "data" if (data_dir / "data").is_dir() else data_dir
    for item in biz_base.iterdir():
        if item.is_dir() and item.name not in ("__pycache__", ".git"):
            biz_stats = validate_business_folder(item)
            summary["businesses"][item.name] = biz_stats

    logger.info(f"Dataset summary: {summary}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate DataMate dataset structure")
    parser.add_argument("--data_dir", type=str, default="./data", help="Root data directory")
    args = parser.parse_args()

    data_path = Path(args.data_dir).resolve()
    if not data_path.exists():
        logger.error(f"Data directory does not exist: {data_path}")
        sys.exit(1)

    summary = validate_dataset(data_path)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
