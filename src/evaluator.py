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
from datetime import date
from decimal import Decimal, InvalidOperation, ROUND_HALF_EVEN
import json
import logging
from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Union

from src.formatter import normalize_for_comparison

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
    """Match a value using the official rule for its declared answer type.

    ``float_tol`` remains in the public signature for compatibility; PA uses
    exact values after two-decimal rounding, never a numeric tolerance.
    """
    if q_type is None:
        q_type = _infer_type_from_value(expected)
    if q_type == "abstain":
        return _is_abstain_value(pred)
    if _is_abstain_value(pred):
        return False
    if q_type == "unanswerable":
        return isinstance(pred, str) and pred.strip().lower() == "unanswerable"
    if q_type in ("number", "percent"):
        try:
            p_num = _rounded_number(pred, q_type)
            e_num = _rounded_number(expected, q_type)
            return p_num == e_num and (q_type != "percent" or p_num != 0 or p_num.is_signed() == e_num.is_signed())
        except (ValueError, InvalidOperation):
            return False
    if q_type == "boolean":
        return _boolean_value(pred) is not None and _boolean_value(pred) == _boolean_value(expected)
    if q_type == "date":
        return _date_value(pred) is not None and _date_value(pred) == _date_value(expected)
    if q_type == "list":
        return (isinstance(pred, (list, tuple)) and isinstance(expected, (list, tuple))
                and len(pred) == len(expected)
                and all(is_answer_match(p, e) for p, e in zip(pred, expected)))
    if q_type == "table":
        return _table_match(pred, expected)
    return normalize_for_comparison(pred) == normalize_for_comparison(expected)


def _is_abstain_value(value: Any) -> bool:
    return value is None or (isinstance(value, str) and value.strip().lower() == "abstain")


def _rounded_number(value: Any, q_type: str) -> Decimal:
    if isinstance(value, bool) or value is None:
        raise ValueError("not a numeric answer")
    raw = str(value).strip()
    if q_type == "percent" and raw.endswith("%"):
        raw = raw[:-1].strip()
    number = Decimal(raw)
    if not number.is_finite():
        raise ValueError("non-finite answer")
    return number.quantize(Decimal("0.01"), rounding=ROUND_HALF_EVEN)


def _boolean_value(value: Any) -> Optional[bool]:
    if isinstance(value, bool):
        return value
    if isinstance(value, str) and value.strip().lower() in ("true", "false"):
        return value.strip().lower() == "true"
    return None


def _date_value(value: Any) -> Optional[str]:
    if not isinstance(value, str) or not re.fullmatch(r"\d{4}-\d{2}(?:-\d{2})?", value):
        return None
    try:
        if len(value) == 7:
            date.fromisoformat(value + "-01")
        else:
            date.fromisoformat(value)
    except ValueError:
        return None
    return value


def _table_match(pred: Any, expected: Any) -> bool:
    if not (isinstance(pred, dict) and isinstance(expected, dict)):
        return False
    p_cols, e_cols = pred.get("columns"), expected.get("columns")
    p_rows, e_rows = pred.get("rows"), expected.get("rows")
    if not all(isinstance(v, (list, tuple)) for v in (p_cols, e_cols, p_rows, e_rows)):
        return False
    if len(p_cols) != len(e_cols) or len(p_rows) != len(e_rows):
        return False
    p_names = [normalize_for_comparison(c) for c in p_cols]
    e_names = [normalize_for_comparison(c) for c in e_cols]
    if p_names == e_names:
        order = list(range(len(e_cols)))
    elif len(set(p_names)) == len(p_names) and set(p_names) == set(e_names):
        order = [p_names.index(name) for name in e_names]
    else:
        order = list(range(len(e_cols)))  # Official positional fallback.
    unused = list(p_rows)
    for e_row in e_rows:
        if not isinstance(e_row, (list, tuple)) or len(e_row) != len(e_cols):
            return False
        match_index = next((i for i, p_row in enumerate(unused)
                            if isinstance(p_row, (list, tuple)) and len(p_row) == len(p_cols)
                            and all(is_answer_match(p_row[order[j]], e_cell)
                                    for j, e_cell in enumerate(e_row))), None)
        if match_index is None:
            return False
        unused.pop(match_index)
    return True


def _infer_type_from_value(val: Any) -> str:
    """Infer answer type from Python value when type field is omitted."""
    if _is_abstain_value(val):
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


def _has_execution_error(sub_item: Optional[Dict[str, Any]]) -> bool:
    """Only classify execution failures when the submitted record supplies evidence."""
    if not sub_item:
        return False
    if sub_item.get("execution_success") is False or sub_item.get("code_execution_error"):
        return True
    return bool(sub_item.get("error") or sub_item.get("execution_error"))


def evaluate_submission(
    submission_file: Union[str, Path],
    ground_truth_file: Union[str, Path],
    penalty_weight: float = 1.0,
    total_duration_seconds: Optional[float] = None,
) -> Dict[str, Any]:
    """Evaluate submission.jsonl against ground_truth.jsonl using Penalized Accuracy.

    PA = (1 / N) * sum(s_i)
    where:
      * s_i = +1.0: if type matches AND value matches reference (including any alias after normalization).
      * s_i =  0.0: if type == 'abstain' or question is missing from submission.
      * s_i = -1.0: otherwise (including a concrete answer to an unanswerable question).

    Args:
        submission_file: Path to generated submission.jsonl.
        ground_truth_file: Path to ground truth questions.jsonl.
        penalty_weight: Deprecated compatibility argument; only 1.0 is valid.
        total_duration_seconds: Optional measured elapsed time of inference.

    Returns:
        Evaluation metrics summary dictionary including type_breakdown.
    """
    if penalty_weight != 1.0:
        raise ValueError("Penalized Accuracy requires penalty_weight=1.0")
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
    error_names = (
        "wrong_type", "false_answer_on_unanswerable", "missed_unanswerable",
        "scale_or_rounding_error", "wrong_date_or_filter", "code_execution_error",
        "abstain_rate",
    )
    error_counts = {name: 0 for name in error_names}

    for q_id, gt_item in gt_map.items():
        exp_type = gt_item.get("type")
        expected_ans = gt_item.get("expected_answer")
        if expected_ans is None:
            expected_ans = gt_item.get("answer")

        # Collect reference candidates: primary answer + any aliases
        candidates: List[Any] = []
        if expected_ans is not None:
            candidates.append(expected_ans)

        for a_key in ("alias", "aliases") if exp_type == "string" else ():
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
            # Submission records must declare one of the nine answer types.

        is_abstain = predicted_type == "abstain" or _is_abstain_value(predicted_ans)

        # Check candidate value match
        value_matched = False
        for cand in candidates:
            if is_answer_match(predicted_ans, cand, q_type=exp_type):
                value_matched = True
                break

        # Check type match
        type_matched = True
        if exp_type is not None:
            type_matched = predicted_type == exp_type

        # Determine scoring s_i
        if is_abstain:
            status = "ABSTAIN"
            score = 0.0
            abstain_count += 1
        elif exp_type == "unanswerable":
            # If the question was unanswerable:
            # Must predict unanswerable to score +1. Any non-unanswerable answer scores -1.
            if type_matched and value_matched:
                status = "CORRECT"
                score = 1.0
                correct_count += 1
            else:
                status = "WRONG"
                score = -1.0
                wrong_count += 1
        elif type_matched and value_matched:
            status = "CORRECT"
            score = 1.0
            correct_count += 1
        else:
            status = "WRONG"
            score = -1.0
            wrong_count += 1

        error_category = None
        if status == "ABSTAIN":
            error_counts["abstain_rate"] += 1
            if _has_execution_error(sub_item):
                error_category = "code_execution_error"
                error_counts[error_category] += 1
        elif status == "WRONG":
            if exp_type == "unanswerable":
                error_category = "false_answer_on_unanswerable"
            elif predicted_type == "unanswerable":
                error_category = "missed_unanswerable"
            elif predicted_type != exp_type and exp_type is not None:
                error_category = "wrong_type"
            elif _has_execution_error(sub_item):
                error_category = "code_execution_error"
            elif exp_type in ("number", "percent"):
                error_category = "scale_or_rounding_error"
            elif exp_type == "date":
                error_category = "wrong_date_or_filter"
            if error_category:
                error_counts[error_category] += 1

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
            "error_category": error_category,
        })

    # Overall metric calculations
    penalized_score = float(correct_count - wrong_count)
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
        t_net = float(t_corr - t_wrg)
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
        "error_breakdown": {
            name: {"count": count, "rate": round(count / total_questions, 4) if total_questions else 0.0}
            for name, count in error_counts.items()
        },
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
    print("ERROR BREAKDOWN (rate = count / total questions)")
    for name, stats in summary.get("error_breakdown", {}).items():
        print(f" {name:<34} {stats['count']:>6} {stats['rate'] * 100:>7.2f}%")


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
