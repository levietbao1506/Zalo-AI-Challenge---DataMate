"""Evaluation & Benchmarking Tool for Zalo AI Challenge 2026 - DataMate Track (Bước 6).

Compares generated submission.jsonl against ground-truth answers:
- Computes Penalized Accuracy (correct: +1, abstain: 0, wrong: -penalty).
- Measures average latency (response time) per question.
- Normalizes comparisons (float tolerance <= 0.01, NFC string comparison, case-insensitive).
"""

from __future__ import annotations

import argparse
import json
import logging
import math
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Tuple, Union
import unicodedata

from src.formatter import clean_string, normalize_unicode_nfc

logger = logging.getLogger(__name__)


def is_answer_match(pred: Any, expected: Any, float_tol: float = 0.01) -> bool:
    """Compare predicted answer with ground truth answer.

    Handles:
    - Floats & Ints: absolute difference <= float_tol.
    - Strings: NFC normalized and case-insensitive comparison.
    - Abstain: matches 'abstain' or None.
    - Booleans: exact match.
    - Lists / Tuples: element-wise match.
    """
    # 1. Abstain check
    pred_is_abstain = pred in ("abstain", None, "null", "")
    exp_is_abstain = expected in ("abstain", None, "null", "")
    if exp_is_abstain:
        return pred_is_abstain
    if pred_is_abstain:
        return False

    # 2. Boolean check (must precede numeric check)
    if isinstance(expected, bool) or isinstance(pred, bool):
        return bool(pred) is bool(expected)

    # 3. Numeric check (float / int)
    try:
        p_num = float(pred)
        e_num = float(expected)
        return abs(p_num - e_num) <= float_tol
    except (ValueError, TypeError):
        pass

    # 4. String comparison
    p_str = clean_string(str(pred))
    e_str = clean_string(str(expected))
    return p_str == e_str


def evaluate_submission(
    submission_file: Union[str, Path],
    ground_truth_file: Union[str, Path],
    penalty_weight: float = 1.0,
    total_duration_seconds: Optional[float] = None,
) -> Dict[str, Any]:
    """Evaluate submission.jsonl against ground_truth.jsonl.

    Args:
        submission_file: Path to generated submission.jsonl.
        ground_truth_file: Path to ground truth questions.jsonl (containing expected_answer).
        penalty_weight: Penalty deducted for an incorrect answer (default: 1.0).
        total_duration_seconds: Optional measured elapsed time of inference.

    Returns:
        Evaluation metrics summary dictionary.
    """
    sub_path = Path(submission_file).resolve()
    gt_path = Path(ground_truth_file).resolve()

    if not sub_path.exists():
        raise FileNotFoundError(f"Submission file not found: {sub_path}")
    if not gt_path.exists():
        raise FileNotFoundError(f"Ground truth file not found: {gt_path}")

    # Read ground truth
    gt_map: Dict[str, Dict[str, Any]] = {}
    with open(gt_path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f, 1):
            line_str = line.strip()
            if not line_str:
                continue
            item = json.loads(line_str)
            q_id = str(item.get("id") or item.get("question_id") or f"q_{idx}")
            gt_map[q_id] = item

    # Read predictions
    sub_map: Dict[str, Any] = {}
    with open(sub_path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f, 1):
            line_str = line.strip()
            if not line_str:
                continue
            item = json.loads(line_str)
            q_id = str(item.get("id") or f"q_{idx}")
            sub_map[q_id] = item.get("answer")

    total_questions = len(gt_map)
    correct_count = 0
    abstain_count = 0
    wrong_count = 0

    detailed_results = []

    for q_id, gt_item in gt_map.items():
        expected_ans = gt_item.get("expected_answer")
        if expected_ans is None:
            expected_ans = gt_item.get("answer")

        predicted_ans = sub_map.get(q_id, "abstain")
        matched = is_answer_match(predicted_ans, expected_ans)

        is_abstain = predicted_ans in ("abstain", None, "null")

        if matched:
            status = "CORRECT"
            correct_count += 1
            score = 1.0
        elif is_abstain:
            status = "ABSTAIN"
            abstain_count += 1
            score = 0.0
        else:
            status = "WRONG"
            wrong_count += 1
            score = -penalty_weight

        detailed_results.append({
            "id": q_id,
            "question": gt_item.get("question", ""),
            "expected": expected_ans,
            "predicted": predicted_ans,
            "status": status,
            "score": score,
        })

    # Metric calculations
    penalized_score = (correct_count * 1.0) - (wrong_count * penalty_weight)
    raw_accuracy = (correct_count / total_questions) if total_questions > 0 else 0.0
    penalized_accuracy = (penalized_score / total_questions) if total_questions > 0 else 0.0

    avg_latency = (
        (total_duration_seconds / total_questions)
        if total_duration_seconds is not None and total_questions > 0
        else 0.0
    )

    summary = {
        "total_questions": total_questions,
        "correct": correct_count,
        "abstained": abstain_count,
        "wrong": wrong_count,
        "raw_accuracy": round(raw_accuracy, 4),
        "penalized_score": round(penalized_score, 2),
        "penalized_accuracy": round(penalized_accuracy, 4),
        "total_duration_seconds": round(total_duration_seconds, 2) if total_duration_seconds else None,
        "average_latency_seconds": round(avg_latency, 3),
        "details": detailed_results,
    }

    return summary


def print_evaluation_report(summary: Dict[str, Any]) -> None:
    """Print readable benchmark report to console."""
    print("\n" + "=" * 65)
    print("      ZALO AI CHALLENGE 2026 - DATAMATE BENCHMARK REPORT")
    print("=" * 65)
    print(f" Tổng số câu hỏi:        {summary['total_questions']}")
    print(f" Trả lời đúng (Correct): {summary['correct']} (+1.0 điểm/câu)")
    print(f" Không trả lời (Abstain): {summary['abstained']} (0.0 điểm)")
    print(f" Trả lời sai (Wrong):     {summary['wrong']} (-1.0 điểm phạt)")
    print("-" * 65)
    print(f" Độ chính xác thô (Raw Accuracy):        {summary['raw_accuracy'] * 100:.2f}%")
    print(f" Điểm số Penalized Score:               {summary['penalized_score']}")
    print(f" Penalized Accuracy:                    {summary['penalized_accuracy'] * 100:.2f}%")
    if summary.get("average_latency_seconds"):
        print(f" Thời gian phản hồi trung bình (Latency): {summary['average_latency_seconds']} giây/câu")
    print("=" * 65 + "\n")
