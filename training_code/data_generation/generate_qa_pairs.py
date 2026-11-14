"""Synthetic QA Pair Generator for DataMate Validation.

Generates representative question-pandas pairs to validate prompt formatting,
unanswerable detection, and self-correction before evaluation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


SAMPLE_TEMPLATES = [
    {
        "type": "number",
        "question_vi": "Tổng doanh thu của tất cả các đơn hàng thành công là bao nhiêu?",
        "pandas_code": "final_result = df[df['status'] == 'completed']['revenue'].sum()",
    },
    {
        "type": "number",
        "question_vi": "Có bao nhiêu đơn hàng bị hủy trong tháng vừa qua?",
        "pandas_code": "final_result = len(df[df['status'] == 'cancelled'])",
    },
    {
        "type": "string",
        "question_vi": "Tên chi nhánh có doanh số cao nhất là gì?",
        "pandas_code": "final_result = df.groupby('branch')['revenue'].sum().idxmax()",
    },
    {
        "type": "unanswerable",
        "question_vi": "Nhiệt độ trung bình trong kho bảo quản là bao nhiêu?",
        "pandas_code": "final_result = 'unanswerable'",
    },
]


def generate_synthetic_samples(count: int = 10) -> List[Dict[str, Any]]:
    """Generate synthetic QA pairs for validation testing."""
    records = []
    for i in range(count):
        tmpl = SAMPLE_TEMPLATES[i % len(SAMPLE_TEMPLATES)]
        records.append({
            "id": f"syn_q_{i+1:03d}",
            "type": tmpl["type"],
            "question": tmpl["question_vi"],
            "reference_code": tmpl["pandas_code"],
        })
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic QA pairs")
    parser.add_argument("--output", type=str, default="./training_code/data/synthetic_qa.jsonl")
    parser.add_argument("--count", type=int, default=10)
    args = parser.parse_args()

    samples = generate_synthetic_samples(args.count)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    print(f"Generated {len(samples)} synthetic QA pairs -> {out_path}")


if __name__ == "__main__":
    main()
