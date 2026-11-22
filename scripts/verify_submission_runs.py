from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import tempfile
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluator import evaluate_submission


REQUIRED_LOG_FIELDS = {"id", "answer", "elapsed_ms", "thinking", "code", "output"}
TRUNCATION_MARKERS = ("[truncated]", "<truncated>", "...truncated...", "output truncated")


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}: {exc}") from exc
            if not isinstance(item, dict):
                raise ValueError(f"Record at {path}:{line_number} must be an object")
            records.append(item)
    return records


def index_unique(records: List[Dict[str, Any]], path: Path) -> Dict[str, Dict[str, Any]]:
    indexed: Dict[str, Dict[str, Any]] = {}
    for item in records:
        record_id = item.get("id") or item.get("question_id")
        if record_id is None:
            raise ValueError(f"Record without id in {path}")
        key = str(record_id)
        if key in indexed:
            raise ValueError(f"Duplicate id {key} in {path}")
        indexed[key] = item
    return indexed


def validate_thinking(value: Any, record_id: str) -> None:
    if value is None:
        return
    values = value if isinstance(value, list) else [value]
    if not all(isinstance(item, str) for item in values):
        raise ValueError(f"thinking for {record_id} must be null, string, or list of strings")
    lowered = "\n".join(values).lower()
    if any(marker in lowered for marker in TRUNCATION_MARKERS):
        raise ValueError(f"thinking for {record_id} contains a truncation marker")


def load_logs(logs_dir: Path) -> Dict[str, Dict[str, Any]]:
    for name in ("predictions.jsonl", "logs.jsonl"):
        candidate = logs_dir / name
        if candidate.is_file():
            return index_unique(read_jsonl(candidate), candidate)
    records: List[Dict[str, Any]] = []
    for path in sorted(logs_dir.glob("*.json")):
        if path.name == "runtime_metrics.json":
            continue
        item = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(item, dict) and item.get("id") is not None:
            records.append(item)
    return index_unique(records, logs_dir)


def validate_run(run_dir: Path, question_ids: set[str]) -> Dict[str, Dict[str, Any]]:
    submission_path = run_dir / "submission.jsonl"
    logs_dir = run_dir / "logs"
    submissions = index_unique(read_jsonl(submission_path), submission_path)
    if set(submissions) != question_ids:
        missing = sorted(question_ids - set(submissions))
        extra = sorted(set(submissions) - question_ids)
        raise ValueError(f"Submission id mismatch in {run_dir}: missing={missing}, extra={extra}")
    for record_id, item in submissions.items():
        if item.get("type") not in {
            "number", "percent", "string", "date", "boolean", "list", "table", "unanswerable", "abstain"
        }:
            raise ValueError(f"Invalid submission type for {record_id}")

    logs = load_logs(logs_dir)
    if set(logs) != question_ids:
        missing = sorted(question_ids - set(logs))
        extra = sorted(set(logs) - question_ids)
        raise ValueError(f"Log id mismatch in {run_dir}: missing={missing}, extra={extra}")
    for record_id, item in logs.items():
        missing_fields = REQUIRED_LOG_FIELDS - set(item)
        if missing_fields:
            raise ValueError(f"Log {record_id} missing fields: {sorted(missing_fields)}")
        elapsed = item["elapsed_ms"]
        if isinstance(elapsed, bool) or not isinstance(elapsed, (int, float)) or elapsed < 0:
            raise ValueError(f"Log {record_id} has invalid elapsed_ms")
        validate_thinking(item["thinking"], record_id)
    return submissions


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate two offline DataMate submission runs")
    parser.add_argument("--questions", required=True)
    parser.add_argument("--answers", required=True)
    parser.add_argument("--run1", required=True)
    parser.add_argument("--run2", required=True)
    parser.add_argument("--max-pa-delta", type=float, default=0.03)
    args = parser.parse_args()

    questions_path = Path(args.questions).resolve()
    answers_path = Path(args.answers).resolve()
    run1 = Path(args.run1).resolve()
    run2 = Path(args.run2).resolve()
    question_ids = set(index_unique(read_jsonl(questions_path), questions_path))
    submission1 = validate_run(run1, question_ids)
    submission2 = validate_run(run2, question_ids)
    answers = index_unique(read_jsonl(answers_path), answers_path)
    missing_answers = sorted(question_ids - set(answers))
    if missing_answers:
        raise ValueError(f"Answers missing question ids: {missing_answers}")
    with tempfile.TemporaryDirectory() as temp_dir:
        filtered_answers = Path(temp_dir) / "answers.jsonl"
        with open(filtered_answers, "w", encoding="utf-8") as handle:
            for record_id, record in answers.items():
                if record_id in question_ids:
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        evaluation1 = evaluate_submission(run1 / "submission.jsonl", filtered_answers)
        evaluation2 = evaluate_submission(run2 / "submission.jsonl", filtered_answers)
    pa_delta = abs(evaluation1["penalized_accuracy"] - evaluation2["penalized_accuracy"])
    exact_matches = sum(submission1[record_id] == submission2[record_id] for record_id in question_ids)
    exact_match_rate = exact_matches / len(question_ids) if question_ids else 1.0
    if pa_delta > args.max_pa_delta:
        raise ValueError(f"PA delta {pa_delta:.6f} exceeds {args.max_pa_delta:.6f}")
    if exact_match_rate != 1.0:
        raise ValueError(f"Submission exact match rate is {exact_match_rate:.6f}, expected 1.0")
    result = {
        "questions": len(question_ids),
        "pa_1": evaluation1["penalized_accuracy"],
        "pa_2": evaluation2["penalized_accuracy"],
        "pa_delta": pa_delta,
        "exact_match_rate": exact_match_rate,
    }
    print(json.dumps(result, ensure_ascii=False, allow_nan=False, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
