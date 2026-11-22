from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import sys
from typing import Any, Dict, List, Optional, Set

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluator import analyze_hybrid_results


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                item = json.loads(line)
                if isinstance(item, dict):
                    records.append(item)
    return records


def stratified_ids(
    answers_path: Path,
    questions_path: Optional[Path],
    limit: Optional[int],
    seed: int,
) -> Set[str]:
    answers = read_jsonl(answers_path)
    allowed: Optional[Set[str]] = None
    if questions_path is not None and questions_path.is_file():
        allowed = {
            str(item.get("id") or item.get("question_id"))
            for item in read_jsonl(questions_path)
            if item.get("id") is not None or item.get("question_id") is not None
        }
    groups: Dict[str, List[str]] = {}
    for item in answers:
        record_id = item.get("id") or item.get("question_id")
        if record_id is None or (allowed is not None and str(record_id) not in allowed):
            continue
        groups.setdefault(str(item.get("type") or "unknown"), []).append(str(record_id))
    rng = random.Random(seed)
    for values in groups.values():
        rng.shuffle(values)
    ordered: List[str] = []
    while any(groups.values()):
        for answer_type in sorted(groups):
            if groups[answer_type]:
                ordered.append(groups[answer_type].pop())
    if limit is not None and limit >= 0:
        ordered = ordered[:limit]
    return set(ordered)


def print_report(report: Dict[str, Any], title: str) -> None:
    overview = report["overview"]
    print(f"\n{title}")
    print("=" * len(title))
    print(
        f"N={overview['total_questions']}  PA={overview['penalized_accuracy']:.4f}  "
        f"Correct={overview['correct']}  Wrong={overview['wrong']}  Abstain={overview['abstain']}"
    )
    print(
        f"Avg={overview['avg_elapsed_ms']:.1f} ms  P95={overview['p95_elapsed_ms']:.1f} ms  "
        f"Estimate-500={overview['estimated_private_500_hours']:.3f} h"
    )
    headers = ("Route", "N", "Correct", "Wrong", "Abstain", "PA", "Avg ms")
    print(f"{headers[0]:<28}{headers[1]:>6}{headers[2]:>10}{headers[3]:>8}{headers[4]:>10}{headers[5]:>10}{headers[6]:>12}")
    for route, stats in report["route_breakdown"].items():
        if stats["count"] == 0:
            continue
        print(
            f"{route:<28}{stats['count']:>6}{stats['correct']:>10}{stats['wrong']:>8}"
            f"{stats['abstain']:>10}{stats['penalized_accuracy']:>10.4f}{stats['avg_elapsed_ms']:>12.1f}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark Adaptive Hybrid routing and Penalized Accuracy")
    parser.add_argument("--submission", default="result/submission.jsonl")
    parser.add_argument("--answers", default="data/sample_dataset/ground_truth.jsonl")
    parser.add_argument("--logs-dir", default="result/logs")
    parser.add_argument("--questions", default="data/sample_dataset/questions.jsonl")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--baseline-submission", default=None)
    parser.add_argument("--baseline-logs-dir", default=None)
    parser.add_argument("--json-output", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    answers = Path(args.answers).resolve()
    questions = Path(args.questions).resolve() if args.questions else None
    selected = stratified_ids(answers, questions, args.limit, args.seed)
    report = analyze_hybrid_results(
        submission_file=Path(args.submission),
        answers_file=answers,
        logs_path=Path(args.logs_dir),
        selected_ids=selected,
    )
    print_report(report, "ADAPTIVE HYBRID")

    baseline_report = None
    if args.baseline_submission:
        baseline_logs = Path(args.baseline_logs_dir) if args.baseline_logs_dir else Path("__missing_baseline_logs__")
        baseline_report = analyze_hybrid_results(
            submission_file=Path(args.baseline_submission),
            answers_file=answers,
            logs_path=baseline_logs,
            selected_ids=selected,
        )
        print_report(baseline_report, "BASELINE")
        delta = report["overview"]["penalized_accuracy"] - baseline_report["overview"]["penalized_accuracy"]
        print(f"\nPA delta Adaptive Hybrid - Baseline: {delta:+.4f}")

    sql_wrong_rate = report["error_analysis"]["sql_fast_clean"]["wrong_rate"]
    if sql_wrong_rate > 0.05:
        print(
            f"\nROUTER WARNING: SQL_FAST clean wrong rate is {sql_wrong_rate * 100:.2f}% (>5%). "
            "Review wrong_cases and tighten router.py escalation rules."
        )
    else:
        print(f"\nRouter signal: SQL_FAST clean wrong rate is {sql_wrong_rate * 100:.2f}%.")

    if args.json_output:
        payload = {"adaptive_hybrid": report, "baseline": baseline_report}
        Path(args.json_output).write_text(
            json.dumps(payload, ensure_ascii=False, allow_nan=False, sort_keys=True, indent=2),
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
