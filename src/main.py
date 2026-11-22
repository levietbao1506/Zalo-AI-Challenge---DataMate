"""Main Orchestrator for Zalo AI Challenge 2026 - DataMate Track.

Adaptive Hybrid Architecture on dual NVIDIA RTX 5090 GPUs.
Groups questions by business_id, initializes temporary DuckDB in /tmp,
evaluates questions using Adaptive Hybrid State Machine,
and immediately flushes streaming results with file locking.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import json
import logging
import math
import multiprocessing as mp
import os
from pathlib import Path
import shutil
import sys
import tempfile
import time
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple

from src import calendar_utils, evaluator, executor, formatter
from src.agent import (
    CodeAgent,
    VLLMAgent,
    execute_adaptive_hybrid,
    extract_thinking,
    verification_gate,
)
from src.duckdb_store import BusinessDuckDBStore
from src.executor import SandboxedExecutor
from src.formatter import format_output, format_submission_record
from src.llm_config import detect_optimal_gpu_mode
from src.prompt_builder import PromptBuilder, build_business_prefix, read_rulebook
from src.router import classify_question_route
from src.schema_parser import DatasetContext, SchemaParser, build_sql_schema_card, get_business_schema
from src.sql_executor import DuckDBSQLExecutor

BATCH_SIZE = 32
DEFAULT_MODEL_PATH = "/code/models/Qwen3-14B" if Path("/code/models/Qwen3-14B").exists() else "./models/Qwen3-14B"
DEFAULT_DATA_DIR = "./data/sample_dataset"
DEFAULT_OUTPUT_FILE = "result/submission.jsonl"
TOTAL_TIME_BUDGET_SEC = 20700.0

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    stream=sys.stdout,
)
logger = logging.getLogger("datamate_main")


def get_scratch_dir() -> Path:
    tmp_path = Path("/tmp")
    try:
        tmp_path.mkdir(parents=True, exist_ok=True)
        test_file = tmp_path / ".perm_check"
        test_file.touch()
        test_file.unlink()
        return tmp_path
    except (PermissionError, OSError):
        base_temp = Path(tempfile.gettempdir())
        fallback = base_temp if base_temp.name == "datamate_tmp" else base_temp / "datamate_tmp"
        fallback.mkdir(parents=True, exist_ok=True)
        return fallback


def ensure_output_directories(output_file: Path) -> None:
    for default_p in (Path("/result"), Path("/result/logs")):
        try:
            default_p.mkdir(parents=True, exist_ok=True)
        except (PermissionError, OSError):
            pass

    try:
        output_file.parent.mkdir(parents=True, exist_ok=True)
        logs_dir = output_file.parent / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)
    except (PermissionError, OSError):
        pass


def make_readonly_safe_file_map(file_map: Dict[str, str], scratch_dir: Path) -> Dict[str, str]:
    safe_map = dict(file_map)
    sqlite_dir = scratch_dir / "sqlite_copies"
    sqlite_dir.mkdir(parents=True, exist_ok=True)

    for fname, fpath_str in file_map.items():
        fpath = Path(fpath_str)
        if fpath.suffix.lower() in (".sqlite", ".sqlite3", ".db"):
            dest_copy = sqlite_dir / f"{fpath.parent.name}_{fpath.name}"
            try:
                if not dest_copy.exists() or (fpath.exists() and dest_copy.stat().st_mtime < fpath.stat().st_mtime):
                    shutil.copy2(fpath, dest_copy)
                safe_map[fname] = str(dest_copy)
            except Exception as e:
                logger.warning(f"Could not copy SQLite database {fpath} to {dest_copy}: {e}")
    return safe_map


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Zalo AI Challenge 2026 - DataMate Offline Inference Orchestrator"
    )
    parser.add_argument(
        "--data_dir",
        "--data-dir",
        type=str,
        default=DEFAULT_DATA_DIR,
        help=f"Directory containing tabular data files and README.md (default: {DEFAULT_DATA_DIR})",
    )
    parser.add_argument(
        "--questions_file",
        "--questions-file",
        type=str,
        default=None,
        help="Path to questions.jsonl (defaults to <data_dir>/questions.jsonl)",
    )
    parser.add_argument(
        "--output_file",
        "--output-file",
        type=str,
        default=DEFAULT_OUTPUT_FILE,
        help=f"Destination path for submission.jsonl (default: {DEFAULT_OUTPUT_FILE})",
    )
    parser.add_argument(
        "--output_dir",
        "--output-dir",
        type=str,
        default=None,
        help="Destination directory for submission.jsonl and logs",
    )
    parser.add_argument(
        "--limit",
        "-n",
        type=int,
        default=None,
        help="Optional limit on number of questions to process",
    )
    parser.add_argument(
        "--model_path",
        "--model-path",
        type=str,
        default=DEFAULT_MODEL_PATH,
        help=f"Model directory path for offline vLLM (default: {DEFAULT_MODEL_PATH})",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=12.0,
        help="Execution timeout per attempt in seconds (default: 12.0)",
    )
    parser.add_argument(
        "--soft_timeout",
        type=float,
        default=100.0,
        help="Maximum wall time per question (default: 100.0s)",
    )
    parser.add_argument(
        "--time_budget_seconds",
        type=float,
        default=TOTAL_TIME_BUDGET_SEC,
        help=f"Total inference wall-time budget (default: {TOTAL_TIME_BUDGET_SEC}s)",
    )
    parser.add_argument(
        "--max_retries",
        type=int,
        default=3,
        help="Maximum retry attempts (default: 3)",
    )
    parser.add_argument(
        "--batch_size",
        "--batch-size",
        type=int,
        default=BATCH_SIZE,
        help=f"Batch size (default: {BATCH_SIZE})",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Run in mock/dry-run mode without loading full vLLM weights",
    )
    parser.add_argument(
        "--verify_with_model",
        action="store_true",
        help="Use extra model verification",
    )
    parser.add_argument(
        "--tensor_parallel_size",
        "--tensor-parallel-size",
        type=int,
        default=None,
        help="Tensor parallel degree for vLLM",
    )
    return parser.parse_args()


def load_questions(questions_path: Path) -> List[Dict[str, Any]]:
    if not questions_path.exists():
        logger.error(f"Questions file not found: {questions_path}")
        return []

    questions = []
    with open(questions_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line_str = line.strip()
            if not line_str:
                continue
            try:
                item = json.loads(line_str)
                questions.append(item)
            except json.JSONDecodeError as e:
                logger.warning(f"Skipping malformed line {line_num} in {questions_path}: {e}")

    logger.info(f"Loaded {len(questions)} questions from {questions_path}")
    return questions


def sort_questions_by_business(
    questions: List[Dict[str, Any]],
    default_business_id: Optional[str] = None,
) -> List[Dict[str, Any]]:
    return [
        item for _, item in sorted(
            enumerate(questions),
            key=lambda pair: (
                str(pair[1].get("business_id") or pair[1].get("db_id") or default_business_id or ""),
                pair[0],
            ),
        )
    ]


def group_questions_by_business(
    questions: List[Dict[str, Any]],
    default_business_id: Optional[str] = "business",
) -> Dict[str, List[Dict[str, Any]]]:
    groups: Dict[str, List[Dict[str, Any]]] = {}
    for q in questions:
        b_id = str(q.get("business_id") or q.get("db_id") or default_business_id or "business")
        if b_id not in groups:
            groups[b_id] = []
        groups[b_id].append(q)
    return groups


def partition_businesses_lpt(
    business_groups: Dict[str, List[Dict[str, Any]]],
    num_workers: int = 2,
) -> List[List[Tuple[str, List[Dict[str, Any]]]]]:
    if num_workers <= 1:
        return [list(business_groups.items())]

    sorted_b = sorted(business_groups.items(), key=lambda kv: len(kv[1]), reverse=True)
    buckets: List[List[Tuple[str, List[Dict[str, Any]]]]] = [[] for _ in range(num_workers)]
    bucket_counts: List[int] = [0 for _ in range(num_workers)]

    for item in sorted_b:
        min_idx = bucket_counts.index(min(bucket_counts))
        buckets[min_idx].append(item)
        bucket_counts[min_idx] += len(item[1])

    return buckets


def append_submission_line_with_lock(
    file_path: Path,
    record: Dict[str, Any],
) -> None:
    line_bytes = (json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, "a+b") as f:
        if os.name == "nt":
            import msvcrt
            f.seek(0, os.SEEK_END)
            msvcrt.locking(f.fileno(), msvcrt.LK_LOCK, max(1, len(line_bytes)))
            try:
                f.write(line_bytes)
                f.flush()
                try:
                    os.fsync(f.fileno())
                except OSError:
                    pass
            finally:
                f.seek(0, os.SEEK_END)
                try:
                    msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, max(1, len(line_bytes)))
                except OSError:
                    pass
        else:
            import fcntl
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            try:
                f.seek(0, os.SEEK_END)
                f.write(line_bytes)
                f.flush()
                try:
                    os.fsync(f.fileno())
                except OSError:
                    pass
            finally:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)


def append_log_record_with_lock(
    logs_dir: Path,
    q_id: str,
    log_rec: Dict[str, Any],
) -> None:
    logs_dir.mkdir(parents=True, exist_ok=True)
    q_file = logs_dir / f"{q_id}.json"
    q_bytes = json.dumps(
        make_json_serializable(log_rec),
        ensure_ascii=False,
        allow_nan=False,
        indent=2,
    ).encode("utf-8")
    with open(q_file, "wb") as f:
        f.write(q_bytes)
        f.flush()
        try:
            os.fsync(f.fileno())
        except OSError:
            pass

    summary_log = logs_dir / "predictions.jsonl"
    append_submission_line_with_lock(summary_log, log_rec)


def read_business_readme(business_dir: Path) -> str:
    for name in ("README.md", "readme.md", "README.txt", "readme.txt"):
        path = business_dir / name
        if not path.is_file():
            continue
        for encoding in ("utf-8-sig", "utf-8", "cp1258", "latin-1"):
            try:
                return path.read_text(encoding=encoding)
            except UnicodeError:
                continue
    return ""


def calculate_time_guard(
    pipeline_start: float,
    deadline: float,
    processed_count: int,
    remaining_count: int,
    now: Optional[float] = None,
) -> Dict[str, Any]:
    current = time.perf_counter() if now is None else now
    elapsed = max(0.0, current - pipeline_start)
    remaining_seconds = max(0.0, deadline - current)
    remaining = max(1, remaining_count)
    average_seconds = elapsed / processed_count if processed_count > 0 else 0.0
    budget_per_question = remaining_seconds / remaining
    return {
        "time_pressure": budget_per_question < 25.0,
        "remaining_seconds": remaining_seconds,
        "remaining_questions": remaining_count,
        "budget_per_question": budget_per_question,
        "average_seconds_per_processed_question": average_seconds,
    }


def build_store_schema_meta(store: BusinessDuckDBStore) -> Dict[str, Any]:
    tables: Dict[str, Any] = {}
    for table_name in sorted(store.tables):
        metadata = store.tables[table_name]
        quoted = '"' + table_name.replace('"', '""') + '"'
        described = store.connection.execute(f"DESCRIBE SELECT * FROM {quoted}").fetchall()
        columns = {str(row[0]): str(row[1]) for row in described}
        tables[table_name] = {
            "row_count": metadata.row_count,
            "columns": columns,
            "original_columns": dict(metadata.original_columns),
            "coerced_columns": dict(metadata.coerced_columns),
            "source_file": metadata.source_file,
            "source_sheet": metadata.source_sheet,
            "source_table": metadata.source_table,
            "etl_warnings": list(metadata.etl_warnings),
        }
    meta = {
        "tables": tables,
        "etl_warnings": sorted({warning for table in store.tables.values() for warning in table.etl_warnings}),
    }
    if hasattr(store, "join_hints"):
        meta["join_hints"] = store.join_hints
    store.schema_meta.update(meta)
    return meta


def make_json_serializable(val: Any) -> Any:
    if isinstance(val, float) and not math.isfinite(val):
        return str(val)
    if val is None or isinstance(val, (int, float, str, bool)):
        return val
    try:
        import numpy as np
        if isinstance(val, np.generic):
            return make_json_serializable(val.item())
        if isinstance(val, np.ndarray):
            return make_json_serializable(val.tolist())
    except ImportError:
        pass

    try:
        import pandas as pd
        if isinstance(val, pd.DataFrame):
            if val.empty:
                return None
            return make_json_serializable(val.to_dict(orient="records"))
        if isinstance(val, pd.Series):
            if val.empty:
                return None
            return {str(k): make_json_serializable(v) for k, v in val.to_dict().items()}
        if isinstance(val, pd.Timestamp):
            return val.isoformat()
    except ImportError:
        pass

    if isinstance(val, dict):
        return {str(k): make_json_serializable(v) for k, v in val.items()}
    if isinstance(val, (list, tuple, set)):
        return [make_json_serializable(item) for item in val]
    return str(val)


def process_business_questions(
    business_id: str,
    business_questions: List[Dict[str, Any]],
    data_dir: Path,
    output_file: Path,
    logs_dir: Path,
    agent: Any,
    prompt_builder: PromptBuilder,
    rulebook_text: str,
    calendar_csv_path: Optional[str],
    deadline_wall_time: float,
    total_remaining_questions: List[int],
    timeout: float = 12.0,
    soft_timeout: float = 100.0,
    pipeline_start_perf: Optional[float] = None,
    entity_index: Optional[Any] = None,
    batch_entered_at: Optional[float] = None,
) -> Tuple[int, int, int]:
    biz_entered_at = batch_entered_at if batch_entered_at is not None else time.perf_counter()
    biz_dir = data_dir / business_id if (data_dir / business_id).is_dir() else data_dir
    duckdb_cache_root = get_scratch_dir() / "datamate_duckdb"
    duckdb_cache_root.mkdir(parents=True, exist_ok=True)

    store = BusinessDuckDBStore(
        business_dir=biz_dir,
        calendar_csv_path=calendar_csv_path,
        cache_root=duckdb_cache_root,
    )
    sql_executor = DuckDBSQLExecutor(store)
    readme_text = read_business_readme(biz_dir)

    schema_card = build_sql_schema_card(store, readme_text)
    business_prefix = build_business_prefix(rulebook_text, readme_text, schema_card)
    schema_meta = build_store_schema_meta(store)

    if entity_index is None:
        try:
            from src.entity_probe import BusinessEntityIndex
            entity_index = BusinessEntityIndex(store)
        except Exception:
            entity_index = None

    answered_count = 0
    abstained_count = 0
    processed_count = 0
    guard_start = pipeline_start_perf if pipeline_start_perf is not None else time.perf_counter()

    try:
        for q_record in business_questions:
            q_id = str(q_record.get("id") or q_record.get("question_id") or "q_unknown")
            time_guard = calculate_time_guard(
                pipeline_start=guard_start,
                deadline=deadline_wall_time,
                processed_count=processed_count,
                remaining_count=total_remaining_questions[0],
            )
            rem_budget = time_guard["remaining_seconds"]
            budget_per_q = time_guard["budget_per_question"]
            time_pressure = time_guard["time_pressure"]

            if rem_budget <= 0.0:
                q_elapsed = max(0, int(round((time.perf_counter() - biz_entered_at) * 1000)))
                sub_rec = format_submission_record(q_id, None, q_type="abstain")
                log_rec = {
                    "id": q_id,
                    "answer": {"type": "abstain", "answer": None},
                    "elapsed_ms": q_elapsed,
                    "thinking": None,
                    "code": None,
                    "output": None,
                    "route_trace": {"route_initial": "TIME_BUDGET_EXCEEDED", "final_stage": "ABSTAIN"},
                    "time_guard": time_guard,
                }
                append_submission_line_with_lock(output_file, sub_rec)
                append_log_record_with_lock(logs_dir, q_id, log_rec)
                abstained_count += 1
                processed_count += 1
                total_remaining_questions[0] = max(0, total_remaining_questions[0] - 1)
                continue

            hybrid_result = execute_adaptive_hybrid(
                question_record=q_record,
                duckdb_store=store,
                sql_executor=sql_executor,
                schema_card=schema_card,
                schema_meta=schema_meta,
                rulebook_text=rulebook_text,
                time_pressure=time_pressure,
                agent=agent,
                prompt_builder=prompt_builder,
                business_prefix=business_prefix,
                timeout=min(timeout, max(2.0, budget_per_q)),
                entity_index=entity_index,
                readme_text=readme_text,
                batch_entered_at=biz_entered_at,
            )

            sub_record = hybrid_result["submission_record"]
            log_record = hybrid_result["log_record"]
            log_record["time_guard"] = time_guard

            append_submission_line_with_lock(output_file, sub_record)
            append_log_record_with_lock(logs_dir, q_id, log_record)

            if sub_record.get("type") in (None, "abstain"):
                abstained_count += 1
            else:
                answered_count += 1
            processed_count += 1
            total_remaining_questions[0] = max(0, total_remaining_questions[0] - 1)

    finally:
        sql_executor.close()
        store.close_and_cleanup()

    return processed_count, answered_count, abstained_count


def run_worker_partition(
    worker_id: int,
    cuda_device_id: int,
    business_partition: List[Tuple[str, List[Dict[str, Any]]]],
    data_dir: Path,
    output_file: Path,
    logs_dir: Path,
    model_path: str,
    rulebook_text: str,
    calendar_csv_path: Optional[str],
    deadline_wall_time: float,
    total_q_count: int,
    dry_run: bool,
    timeout: float,
    soft_timeout: float,
) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(cuda_device_id)
    os.environ["NCCL_P2P_DISABLE"] = "1"
    os.environ["PYTHONHASHSEED"] = "42"

    load_started = time.perf_counter()
    agent = CodeAgent(model_path=model_path, dry_run=dry_run, tensor_parallel_size=1)
    load_model_ms = int((time.perf_counter() - load_started) * 1000)
    logger.info(f"load_model_ms={load_model_ms} worker_id={worker_id}")
    append_submission_line_with_lock(
        logs_dir / "runtime_events.jsonl",
        {"event": "load_model", "worker_id": worker_id, "load_model_ms": load_model_ms},
    )
    prompt_builder = PromptBuilder(rulebook_content=rulebook_text)

    remaining_holder = [total_q_count]
    for b_id, b_questions in business_partition:
        b_entered_at = time.perf_counter()
        process_business_questions(
            business_id=b_id,
            business_questions=b_questions,
            data_dir=data_dir,
            output_file=output_file,
            logs_dir=logs_dir,
            agent=agent,
            prompt_builder=prompt_builder,
            rulebook_text=rulebook_text,
            calendar_csv_path=calendar_csv_path,
            deadline_wall_time=deadline_wall_time,
            total_remaining_questions=remaining_holder,
            timeout=timeout,
            soft_timeout=soft_timeout,
            pipeline_start_perf=load_started,
            batch_entered_at=b_entered_at,
        )


def collect_runtime_summary(logs_dir: Path, total_ms: int, fallback_load_model_ms: int = 0) -> Dict[str, Any]:
    route_counts: Dict[str, int] = {}
    predictions_path = logs_dir / "predictions.jsonl"
    if predictions_path.is_file():
        for line in predictions_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            route = str((record.get("route_trace") or {}).get("route_initial") or "UNKNOWN")
            route_counts[route] = route_counts.get(route, 0) + 1

    load_values = [fallback_load_model_ms]
    events_path = logs_dir / "runtime_events.jsonl"
    if events_path.is_file():
        for line in events_path.read_text(encoding="utf-8").splitlines():
            try:
                event = json.loads(line)
                if event.get("event") == "load_model":
                    load_values.append(int(event.get("load_model_ms", 0)))
            except (json.JSONDecodeError, TypeError, ValueError):
                continue
    return {
        "load_model_ms": max(load_values),
        "route_counts": dict(sorted(route_counts.items())),
        "total_ms": total_ms,
    }


def main() -> None:
    start_time = time.time()
    perf_start = time.perf_counter()
    args = parse_arguments()
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("NCCL_P2P_DISABLE", "1")
    os.environ.setdefault("PYTHONHASHSEED", "42")

    data_dir = Path(args.data_dir).resolve()
    questions_file = Path(args.questions_file).resolve() if args.questions_file else data_dir / "questions.jsonl"

    if args.output_dir:
        output_file = (Path(args.output_dir) / "submission.jsonl").resolve()
        logs_dir = (Path(args.output_dir) / "logs").resolve()
    else:
        output_file = Path(args.output_file).resolve()
        logs_dir = (output_file.parent / "logs").resolve()

    ensure_output_directories(output_file)

    scratch_dir = get_scratch_dir()
    os.environ["TMPDIR"] = str(scratch_dir)
    os.environ["TEMP"] = str(scratch_dir)
    os.environ["TMP"] = str(scratch_dir)
    tempfile.tempdir = str(scratch_dir)

    rulebook_content = read_rulebook(data_dir=data_dir)
    calendar_csv_path = None
    for c_cand in (data_dir / "calendar.csv", Path("/data/calendar.csv")):
        if c_cand.exists():
            calendar_csv_path = str(c_cand.resolve())
            calendar_utils.set_calendar_csv_path(calendar_csv_path)
            break

    questions = load_questions(questions_file)
    if args.limit is not None and args.limit > 0:
        questions = questions[:args.limit]

    total_q = len(questions)
    if total_q == 0:
        ensure_output_directories(output_file)
        output_file.write_text("", encoding="utf-8")
        (logs_dir / "predictions.jsonl").write_text("", encoding="utf-8")
        return

    output_file.write_text("", encoding="utf-8")
    (logs_dir / "predictions.jsonl").write_text("", encoding="utf-8")
    (logs_dir / "runtime_events.jsonl").write_text("", encoding="utf-8")

    deadline_wall_time = perf_start + max(0.0, float(args.time_budget_seconds))

    biz_groups = group_questions_by_business(questions)
    gpu_mode, opt_tp, num_workers = detect_optimal_gpu_mode(
        args.model_path,
        override_tp=args.tensor_parallel_size,
    )
    logger.info(f"Optimal GPU Execution Mode: {gpu_mode} (TP={opt_tp}, workers={num_workers})")

    load_model_ms = 0
    if gpu_mode == "DP" and num_workers >= 2 and not args.dry_run:
        partitions = partition_businesses_lpt(biz_groups, num_workers=2)
        processes = []
        for worker_idx in range(2):
            p = mp.Process(
                target=run_worker_partition,
                args=(
                    worker_idx,
                    worker_idx,
                    partitions[worker_idx],
                    data_dir,
                    output_file,
                    logs_dir,
                    args.model_path,
                    rulebook_content,
                    calendar_csv_path,
                    deadline_wall_time,
                    sum(len(item[1]) for item in partitions[worker_idx]),
                    args.dry_run,
                    args.timeout,
                    args.soft_timeout,
                ),
            )
            p.start()
            processes.append(p)
        for p in processes:
            p.join()
    else:
        load_started = time.perf_counter()
        agent = CodeAgent(
            model_path=args.model_path,
            dry_run=args.dry_run,
            tensor_parallel_size=opt_tp,
        )
        load_model_ms = int((time.perf_counter() - load_started) * 1000)
        logger.info(f"load_model_ms={load_model_ms}")
        prompt_builder = PromptBuilder(rulebook_content=rulebook_content)
        rem_counter = [total_q]
        for b_id, b_questions in biz_groups.items():
            b_entered_at = time.perf_counter()
            process_business_questions(
                business_id=b_id,
                business_questions=b_questions,
                data_dir=data_dir,
                output_file=output_file,
                logs_dir=logs_dir,
                agent=agent,
                prompt_builder=prompt_builder,
                rulebook_text=rulebook_content,
                calendar_csv_path=calendar_csv_path,
                deadline_wall_time=deadline_wall_time,
                total_remaining_questions=rem_counter,
                timeout=args.timeout,
                soft_timeout=args.soft_timeout,
                pipeline_start_perf=perf_start,
                batch_entered_at=b_entered_at,
            )

    elapsed = time.time() - start_time
    total_ms = int(elapsed * 1000)
    runtime_summary = collect_runtime_summary(logs_dir, total_ms, load_model_ms)
    runtime_summary["gpu_mode"] = gpu_mode
    runtime_summary["worker_count"] = 2 if gpu_mode == "DP" and num_workers >= 2 and not args.dry_run else 1
    (logs_dir / "runtime_metrics.json").write_text(
        json.dumps(runtime_summary, ensure_ascii=False, allow_nan=False, sort_keys=True, indent=2),
        encoding="utf-8",
    )
    logger.info(f"load_model_ms={runtime_summary['load_model_ms']}")
    logger.info(f"route_counts={json.dumps(runtime_summary['route_counts'], sort_keys=True)}")
    logger.info(f"total_ms={runtime_summary['total_ms']}")
    logger.info("=" * 60)
    logger.info("Inference Complete!")
    logger.info(f"Total Questions: {total_q}")
    logger.info(f"Total Elapsed:   {elapsed:.2f}s")
    logger.info(f"Submission:      {output_file}")
    logger.info("=" * 60)


def process_single_question(
    q_data: Dict[str, Any],
    dataset_context: DatasetContext,
    schema_prompt_str: str,
    agent: VLLMAgent,
    executor: SandboxedExecutor,
    prompt_builder: PromptBuilder,
    max_retries: int,
    timeout: float,
    parser: Optional[SchemaParser] = None,
    scratch_dir: Optional[Path] = None,
    calendar_csv_path: Optional[str] = None,
    batch_entered_at: Optional[float] = None,
) -> Any:
    q_text = q_data.get("question") or q_data.get("text") or ""
    q_as_of = q_data.get("as_of_date") or dataset_context.as_of_date
    u_prompt = prompt_builder.build_user_prompt(question=q_text, as_of_date=q_as_of)
    s_prompt = prompt_builder.build_system_prompt(
        readme_content=dataset_context.readme_content or "",
        schema_context=dataset_context.to_enriched_schema_string(),
        business_id=dataset_context.business_id,
    )
    code = agent.generate_code(u_prompt, system_prompt=s_prompt)
    exec_res = executor.execute_with_self_correction(
        agent=agent,
        initial_code=code,
        question=q_text,
        prompt_builder=prompt_builder,
        system_prompt=s_prompt,
        data_dir=dataset_context.data_dir,
        file_map=dataset_context.file_map,
        as_of_date=q_as_of,
        max_retries=max_retries,
        timeout_seconds=timeout,
    )
    raw_res = exec_res.result if exec_res.success else None
    return format_output(
        val=raw_res,
        success=exec_res.success,
        round_decimals=2,
        abstain_token="abstain",
        clean_text=False,
    )


def process_batch_chunk(
    chunk: List[Dict[str, Any]],
    dataset_context: DatasetContext,
    schema_prompt_str: str,
    agent: Any,
    prompt_builder: PromptBuilder,
    system_prompt: str,
    max_retries: int,
    timeout: float,
    parser: Optional[SchemaParser] = None,
    scratch_dir: Optional[Path] = None,
    calendar_csv_path: Optional[str] = None,
    on_log: Optional[Callable[[Dict[str, Any]], None]] = None,
    soft_timeout: float = 100.0,
    batch_entered_at: Optional[float] = None,
) -> List[Dict[str, Any]]:
    chunk_entered_at = batch_entered_at if batch_entered_at is not None else time.perf_counter()
    batch_prompts = []
    meta_list = []
    for q in chunk:
        q_id = str(q.get("id") or q.get("question_id") or "q")
        q_text = q.get("question") or q.get("text") or ""
        q_as_of = q.get("as_of_date") or dataset_context.as_of_date
        u_p = prompt_builder.build_user_prompt(question=q_text, as_of_date=q_as_of)
        batch_prompts.append(u_p)
        meta_list.append({"id": q_id, "text": q_text, "as_of": q_as_of, "type": q.get("type")})

    if hasattr(agent, "generate_code_batch"):
        initial_codes = agent.generate_code_batch(batch_prompts)
    else:
        initial_codes = [agent.generate_code(p, system_prompt=system_prompt) for p in batch_prompts]

    records = []
    for idx, (m, code) in enumerate(zip(meta_list, initial_codes)):
        sandboxed = SandboxedExecutor(default_timeout_seconds=timeout)
        effective_retries = 0 if soft_timeout <= 0 else max_retries
        exec_res = sandboxed.execute_with_self_correction(
            agent=agent,
            initial_code=code,
            question=m["text"],
            prompt_builder=prompt_builder,
            system_prompt=system_prompt,
            data_dir=dataset_context.data_dir,
            file_map=dataset_context.file_map,
            as_of_date=m["as_of"],
            max_retries=effective_retries,
            timeout_seconds=timeout,
        )
        raw_val = exec_res.result if exec_res.success else None
        formatted = format_output(
            val=raw_val,
            success=exec_res.success,
            round_decimals=2,
            abstain_token="abstain",
            clean_text=False,
        )
        sub_rec = format_submission_record(
            q_id=m["id"],
            answer=formatted,
            q_type=m["type"] if formatted != "abstain" else "abstain",
        )
        records.append(sub_rec)
        elapsed_q = max(0, int(round((time.perf_counter() - chunk_entered_at) * 1000)))
        if on_log:
            on_log({
                "id": m["id"],
                "answer": sub_rec,
                "elapsed_ms": elapsed_q,
                "thinking": None,
                "code": exec_res.executed_code,
                "output": make_json_serializable(raw_val),
            })
    return records


def _format_chatml_prompt(system_prompt: str, user_prompt: str) -> str:
    return (
        f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
        f"<|im_start|>user\n{user_prompt}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )


if __name__ == "__main__":
    main()
