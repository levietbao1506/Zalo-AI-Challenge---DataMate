"""Evaluation & Benchmarking Tool for Zalo AI Challenge 2026 - DataMate Track (Bước 6).

Compares generated submission.jsonl against ground-truth answers per official rules:
- Computes Penalized Accuracy: PA = (1 / N) * sum(s_i)
  * s_i = +1: if type matches AND value matches reference (including any entry in alias after normalization).
  * s_i =  0: if type == 'abstain' or missing from submission.
  * s_i = -1: otherwise (including any non-unanswerable answer given to an unanswerable question).
- Breakdown report by answer type (number, percent, string, date, boolean, list, table, unanswerable, abstain).
- Measures average latency (response time) per question.
"""

from __future__ import annotations

import argparse
from datetime import date, datetime
import json
import logging
import math
from pathlib import Path
import re
import time
from typing import Any, Dict, List, Optional, Tuple, Union
import unicodedata

from src.formatter import (
    clean_string,
    format_boolean,
    format_date,
    format_number,
    format_percent,
    format_table,
    normalize_for_comparison,
    normalize_unicode_nfc,
)

logger = logging.getLogger(__name__)

STANDARD_ANSWER_TYPES: List[str] = [
    "number",
    "percent",
    "string",
    "date",
    "boolean",
    "list",
    "table",
    "unanswerable",
    "abstain",
]


def is_answer_match(
    pred: Any,
    expected: Any,
    float_tol: float = 0.01,
    q_type: Optional[str] = None,
) -> bool:
    """Compare predicted answer with ground truth candidate answer.

    Handles:
    - Abstain: matches 'abstain' or None.
    - Unanswerable: matches 'unanswerable'.
    - Booleans: exact match.
    - Floats & Ints: absolute difference <= float_tol.
    - Dates: ISO date comparison ('YYYY-MM-DD' or 'YYYY-MM').
    - Strings: NFC normalized and clean_string comparison.
    - Lists / Tuples: element-wise match.
    - Tables: column names and cell matching.
    """
    # 1. Abstain check
    pred_is_abstain = pred in ("abstain", None, "null", "")
    exp_is_abstain = expected in ("abstain", None, "null", "")
    if exp_is_abstain:
        return pred_is_abstain
    if pred_is_abstain:
        return False

    # 2. Unanswerable check
    pred_is_unans = isinstance(pred, str) and pred.strip().lower() == "unanswerable"
    exp_is_unans = isinstance(expected, str) and expected.strip().lower() == "unanswerable"
    if exp_is_unans:
        return pred_is_unans
    if pred_is_unans:
        return False

    # 3. Boolean check (must precede numeric check)
    if isinstance(expected, bool) or isinstance(pred, bool):
        try:
            b_pred = format_boolean(pred)
            b_exp = format_boolean(expected)
            return b_pred is b_exp
        except Exception:
            return bool(pred) is bool(expected)

    # 4. Numeric check (float / int / percent)
    try:
        p_num = float(pred)
        e_num = float(expected)
        if not (math.isnan(p_num) or math.isnan(e_num) or math.isinf(p_num) or math.isinf(e_num)):
            return abs(p_num - e_num) <= float_tol
    except (ValueError, TypeError):
        pass

    # 5. Date check
    if q_type == "date" or isinstance(expected, (date, datetime)) or isinstance(pred, (date, datetime)):
        try:
            d_p = format_date(pred)
            d_e = format_date(expected)
            return d_p == d_e
        except Exception:
            pass

    # 6. List / Tuple check
    if isinstance(expected, (list, tuple)) and isinstance(pred, (list, tuple)):
        if len(pred) != len(expected):
            return False
        return all(is_answer_match(p, e, float_tol=float_tol) for p, e in zip(pred, expected))

    # 7. Table check
    if isinstance(expected, dict) and isinstance(pred, dict):
        if "columns" in expected and "rows" in expected and "columns" in pred and "rows" in pred:
            p_cols = [clean_string(str(c)) for c in pred["columns"]]
            e_cols = [clean_string(str(c)) for c in expected["columns"]]
            if p_cols != e_cols:
                return False
            p_rows = pred["rows"]
            e_rows = expected["rows"]
            if len(p_rows) != len(e_rows):
                return False
            for rp, re_row in zip(p_rows, e_rows):
                if len(rp) != len(re_row):
                    return False
                if not all(is_answer_match(cp, ce, float_tol=float_tol) for cp, ce in zip(rp, re_row)):
                    return False
            return True

    # 8. String comparison (fallback)
    p_str = clean_string(str(pred))
    e_str = clean_string(str(expected))
    return p_str == e_str


def _infer_type_from_value(val: Any) -> str:
    """Infer answer type from Python value when type field is omitted."""
    if val in ("abstain", None):
        return "abstain"
    if isinstance(val, str) and val.strip().lower() == "unanswerable":
        return "unanswerable"
    if isinstance(val, bool):
        return "boolean"
    if isinstance(val, (int, float)):
        return "number"
    if isinstance(val, (list, tuple)):
        return "list"
    if isinstance(val, dict) and "columns" in val and "rows" in val:
        return "table"
    return "string"


def evaluate_submission(
    submission_file: Union[str, Path],
    ground_truth_file: Union[str, Path],
    penalty_weight: float = 1.0,
    total_duration_seconds: Optional[float] = None,
) -> Dict[str, Any]:
    """Evaluate submission.jsonl against ground_truth.jsonl using exact Penalized Accuracy (PA).

    PA = (1 / N) * sum(s_i)
    where:
      * s_i = +1.0: if type matches AND value matches reference (including any alias after normalization).
      * s_i =  0.0: if type == 'abstain' or question is missing from submission.
      * s_i = -penalty_weight: otherwise (including any non-unanswerable answer given to an unanswerable question).

    Args:
        submission_file: Path to generated submission.jsonl.
        ground_truth_file: Path to ground truth questions.jsonl.
        penalty_weight: Penalty deducted for an incorrect answer (default: 1.0).
        total_duration_seconds: Optional measured elapsed time of inference.

    Returns:
        Evaluation metrics summary dictionary including type_breakdown.
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
    sub_map: Dict[str, Dict[str, Any]] = {}
    with open(sub_path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f, 1):
            line_str = line.strip()
            if not line_str:
                continue
            item = json.loads(line_str)
            q_id = str(item.get("id") or f"q_{idx}")
            sub_map[q_id] = item

    total_questions = len(gt_map)
    correct_count = 0
    abstain_count = 0
    wrong_count = 0

    detailed_results = []
    type_stats: Dict[str, Dict[str, int]] = {}

    for q_id, gt_item in gt_map.items():
        exp_type = gt_item.get("type")
        expected_ans = gt_item.get("expected_answer")
        if expected_ans is None:
            expected_ans = gt_item.get("answer")

        # Collect reference candidates: primary answer + any aliases
        candidates: List[Any] = []
        if expected_ans is not None:
            candidates.append(expected_ans)

        for a_key in ("alias", "aliases"):
            raw_alias = gt_item.get(a_key)
            if raw_alias is not None:
                if isinstance(raw_alias, (list, tuple, set)):
                    candidates.extend(list(raw_alias))
                else:
                    candidates.append(raw_alias)

        # Lookup submission prediction
        sub_item = sub_map.get(q_id)
        if sub_item is None:
            # Missing from submission -> treated as abstain
            predicted_type = "abstain"
            predicted_ans = "abstain"
        else:
            predicted_type = sub_item.get("type")
            predicted_ans = sub_item.get("answer")
            if predicted_type is None:
                predicted_type = _infer_type_from_value(predicted_ans)

        is_abstain = (predicted_type == "abstain" or predicted_ans in ("abstain", None, "null", ""))

        # Check candidate value match
        value_matched = False
        for cand in candidates:
            if is_answer_match(predicted_ans, cand, float_tol=0.01, q_type=exp_type):
                value_matched = True
                break

        # Check type match
        type_matched = True
        if exp_type is not None:
            if exp_type == "unanswerable":
                type_matched = (predicted_type == "unanswerable" or str(predicted_ans).strip().lower() == "unanswerable")
            elif exp_type == "abstain":
                type_matched = is_abstain
            else:
                type_matched = (predicted_type == exp_type)

        # Determine scoring s_i
        if is_abstain:
            status = "ABSTAIN"
            score = 0.0
            abstain_count += 1
        elif exp_type == "unanswerable":
            # If the question was unanswerable:
            # Must predict unanswerable to score +1. Any non-unanswerable answer scores -1.
            if type_matched and (predicted_type == "unanswerable" or str(predicted_ans).strip().lower() == "unanswerable"):
                status = "CORRECT"
                score = 1.0
                correct_count += 1
            else:
                status = "WRONG"
                score = -penalty_weight
                wrong_count += 1
        elif type_matched and value_matched:
            status = "CORRECT"
            score = 1.0
            correct_count += 1
        else:
            status = "WRONG"
            score = -penalty_weight
            wrong_count += 1

        # Track type-level breakdown stats
        breakdown_key = exp_type or "unknown"
        if breakdown_key not in type_stats:
            type_stats[breakdown_key] = {"total": 0, "correct": 0, "abstained": 0, "wrong": 0}
        type_stats[breakdown_key]["total"] += 1
        if status == "CORRECT":
            type_stats[breakdown_key]["correct"] += 1
        elif status == "ABSTAIN":
            type_stats[breakdown_key]["abstained"] += 1
        else:
            type_stats[breakdown_key]["wrong"] += 1

        detailed_results.append({
            "id": q_id,
            "type": exp_type,
            "question": gt_item.get("question", ""),
            "expected": expected_ans,
            "predicted_type": predicted_type,
            "predicted": predicted_ans,
            "status": status,
            "score": score,
        })

    # Overall metric calculations
    penalized_score = (correct_count * 1.0) - (wrong_count * penalty_weight)
    raw_accuracy = (correct_count / total_questions) if total_questions > 0 else 0.0
    penalized_accuracy = (penalized_score / total_questions) if total_questions > 0 else 0.0

    avg_latency = (
        (total_duration_seconds / total_questions)
        if total_duration_seconds is not None and total_questions > 0
        else 0.0
    )

    # Build structured type_breakdown covering all standard types
    all_types = list(STANDARD_ANSWER_TYPES)
    for t in type_stats.keys():
        if t not in all_types:
            all_types.append(t)

    type_breakdown: Dict[str, Dict[str, Any]] = {}
    for t in all_types:
        stats = type_stats.get(t, {"total": 0, "correct": 0, "abstained": 0, "wrong": 0})
        t_tot = stats["total"]
        t_corr = stats["correct"]
        t_abs = stats["abstained"]
        t_wrg = stats["wrong"]
        t_net = (t_corr * 1.0) - (t_wrg * penalty_weight)
        t_pa = (t_net / t_tot) if t_tot > 0 else 0.0
        type_breakdown[t] = {
            "total": t_tot,
            "correct": t_corr,
            "abstained": t_abs,
            "wrong": t_wrg,
            "net_score": round(t_net, 2),
            "penalized_accuracy": round(t_pa, 4),
        }

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
        "type_breakdown": type_breakdown,
        "details": detailed_results,
    }

    return summary


def print_evaluation_report(summary: Dict[str, Any]) -> None:
    """Print readable benchmark report with type breakdown to console."""
    print("\n" + "=" * 80)
    print("      ZALO AI CHALLENGE 2026 - DATAMATE PENALIZED ACCURACY (PA) REPORT")
    print("=" * 80)
    print(f" Tổng số câu hỏi (N):               {summary['total_questions']}")
    print(f" Trả lời đúng (Correct +1):         {summary['correct']}")
    print(f" Không trả lời / Thiếu (Abstain 0): {summary['abstained']}")
    print(f" Trả lời sai (Wrong -1):            {summary['wrong']}")
    print("-" * 80)
    print(f" Độ chính xác thô (Raw Accuracy):   {summary['raw_accuracy'] * 100:.2f}%")
    print(f" Net Penalized Score:               {summary['penalized_score']:+.2f}")
    print(f" Penalized Accuracy (PA):           {summary['penalized_accuracy'] * 100:.2f}%")
    if summary.get("average_latency_seconds"):
        print(f" Thời gian phản hồi TB (Latency):   {summary['average_latency_seconds']:.3f} s/câu")
    print("=" * 80)
    print("                     CHI TIẾT THEO LOẠI CÂU HỎI (TYPE BREAKDOWN)")
    print("-" * 80)
    print(f" {'Loại (Type)':<14} {'Tổng':>6} {'Đúng (+1)':>10} {'Bỏ (0)':>9} {'Sai (-1)':>9} {'Điểm Net':>11} {'Net PA':>10}")
    print("-" * 80)

    breakdown = summary.get("type_breakdown", {})
    for t_name, stats in breakdown.items():
        if stats["total"] > 0:
            pa_pct = f"{stats['penalized_accuracy'] * 100:+.2f}%"
            net_str = f"{stats['net_score']:+.1f}"
            print(
                f" {t_name:<14} {stats['total']:>6} {stats['correct']:>10} {stats['abstained']:>9} "
                f"{stats['wrong']:>9} {net_str:>11} {pa_pct:>10}"
            )

    print("-" * 80)
    tot_pa_pct = f"{summary['penalized_accuracy'] * 100:+.2f}%"
    tot_net_str = f"{summary['penalized_score']:+.1f}"
    print(
        f" {'TỔNG CỘNG':<14} {summary['total_questions']:>6} {summary['correct']:>10} {summary['abstained']:>9} "
        f"{summary['wrong']:>9} {tot_net_str:>11} {tot_pa_pct:>10}"
    )
    print("=" * 80 + "\n")


def main() -> None:
    """CLI runner for evaluating submission.jsonl."""
    parser = argparse.ArgumentParser(description="Evaluate submission.jsonl against ground truth")
    parser.add_argument("--submission", "-s", type=str, required=True, help="Path to submission.jsonl")
    parser.add_argument("--ground-truth", "--gt", type=str, required=True, help="Path to questions.jsonl ground truth")
    parser.add_argument("--penalty", type=float, default=1.0, help="Penalty weight (default: 1.0)")
    args = parser.parse_args()

    summary = evaluate_submission(
        submission_file=args.submission,
        ground_truth_file=args.ground_truth,
        penalty_weight=args.penalty,
    )
    print_evaluation_report(summary)


if __name__ == "__main__":
    main()
