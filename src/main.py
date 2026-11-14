"""Main Orchestrator for Zalo AI Challenge 2026 - DataMate Track.

Reads questions.jsonl, extracts schemas from CSV/XLSX/SQLite, invokes local vLLM agent,
safely runs Python scripts in sandboxed executor with self-correction retry,
normalizes answers per competition rules (NFC Unicode, 2 decimal places, abstain fallback),
and writes streaming results to submission.jsonl.
"""

from __future__ import annotations

import argparse
from datetime import datetime
from functools import partial
import json
import logging
import os
from pathlib import Path
import shutil
import sys
import tempfile
import time
from typing import Any, Callable, Dict, Iterator, List, Optional

from src import calendar_utils, executor
from src.agent import CodeAgent, VLLMAgent, extract_thinking, verification_gate
from src.executor import EMPTY_RESULT_FEEDBACK, SandboxedExecutor
from src.formatter import format_output, format_submission_record
from src.prompt_builder import PromptBuilder, read_rulebook
from src.schema_parser import DatasetContext, SchemaParser, get_business_schema

BATCH_SIZE = 32
DEFAULT_MODEL_PATH = "./models/Qwen3-14B"
DEFAULT_DATA_DIR = "./data/sample_dataset"
DEFAULT_OUTPUT_FILE = "result/submission.jsonl"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    stream=sys.stdout,
)
logger = logging.getLogger("datamate_main")


def get_scratch_dir() -> Path:
    """Return /tmp if writable, otherwise user temp dir."""
    tmp_path = Path("/tmp")
    try:
        tmp_path.mkdir(parents=True, exist_ok=True)
        test_file = tmp_path / ".perm_check"
        test_file.touch()
        test_file.unlink()
        return tmp_path
    except (PermissionError, OSError):
        fallback = Path(tempfile.gettempdir()) / "datamate_tmp"
        fallback.mkdir(parents=True, exist_ok=True)
        return fallback


def ensure_output_directories(output_file: Path) -> None:
    """Create /result and /result/logs if missing, as well as the destination directory."""
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
    """Copy SQLite database files into scratch_dir so queries run without writing to /data."""
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
    """Parse command line arguments."""
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
        help="Destination directory for submission.jsonl and logs (overrides output_file parent)",
    )
    parser.add_argument(
        "--limit",
        "-n",
        type=int,
        default=None,
        help="Optional limit on number of questions to process (for smoke tests)",
    )
    parser.add_argument(
        "--model_path",
        "--model-path",
        type=str,
        default=DEFAULT_MODEL_PATH,
        help=f"Model path (.gguf for llama.cpp, directory for vLLM, or 'ollama:<model>') (default: {DEFAULT_MODEL_PATH})",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=25.0,
        help="Execution timeout per question attempt in seconds (default: 25.0)",
    )
    parser.add_argument(
        "--max_retries",
        type=int,
        default=3,
        help="Maximum self-correction attempts including initial pass (default: 3)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=BATCH_SIZE,
        help=f"Number of questions per vLLM batch inference chunk (default: {BATCH_SIZE})",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Run in mock/dry-run mode without loading full vLLM weights",
    )

    return parser.parse_args()


def load_questions(questions_path: Path) -> List[Dict[str, Any]]:
    """Load evaluation questions from JSONL file."""
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
) -> Any:
    """Process a single question through agent generation, execution, and self-correction."""
    question_text = q_data.get("question") or q_data.get("text") or ""
    active_context, active_schema_str, q_as_of, safe_file_map = _resolve_question_context(
        q_data=q_data,
        dataset_context=dataset_context,
        schema_prompt_str=schema_prompt_str,
        parser=parser,
        scratch_dir=scratch_dir,
        calendar_csv_path=calendar_csv_path,
    )

    # 1. Compose user prompt
    user_prompt = prompt_builder.build_user_prompt(
        question=question_text,
        schema_context=active_schema_str,
        as_of_date=q_as_of,
        file_map=safe_file_map,
    )
    system_prompt = prompt_builder.build_system_prompt()

    # 2. Agent generates code
    code = agent.generate_code(user_prompt, system_prompt=system_prompt)

    # 3. Sandboxed execution with self-correction retry loop (up to max_retries = 3)
    exec_result = executor.execute_with_self_correction(
        agent=agent,
        initial_code=code,
        question=question_text,
        prompt_builder=prompt_builder,
        system_prompt=system_prompt,
        data_dir=active_context.data_dir,
        file_map=safe_file_map,
        as_of_date=q_as_of,
        max_retries=max_retries,
        timeout_seconds=timeout,
    )

    # 5. Normalize output per competition rules (Step 5 Formatter & Abstain)
    # If execution failed after retries, format_output will fall back to "abstain"
    raw_result = exec_result.result if exec_result.success else None
    formatted_answer = format_output(
        val=raw_result,
        success=exec_result.success,
        round_decimals=2,
        abstain_token="abstain",
        clean_text=False,
    )

    return formatted_answer


def _format_chatml_prompt(system_prompt: str, user_prompt: str) -> str:
    """Wrap system and user prompts into Qwen ChatML format for batch inference."""
    return (
        f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
        f"<|im_start|>user\n{user_prompt}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )


def _resolve_question_context(
    q_data: Dict[str, Any],
    dataset_context: DatasetContext,
    schema_prompt_str: str,
    parser: Optional[SchemaParser] = None,
    scratch_dir: Optional[Path] = None,
    calendar_csv_path: Optional[str] = None,
) -> tuple[DatasetContext, str, Optional[str], Dict[str, str]]:
    """Resolve active DatasetContext, cached schema string, as_of_date, and safe file map."""
    biz_id = q_data.get("business_id") or q_data.get("db_id")
    if biz_id and parser:
        active_context = parser.parse(business_id=biz_id)
        active_schema_str = get_business_schema(active_context.data_dir)
    else:
        active_context = dataset_context
        active_schema_str = schema_prompt_str

    q_as_of = q_data.get("as_of_date") or active_context.as_of_date

    safe_file_map = dict(active_context.file_map)
    if calendar_csv_path:
        safe_file_map.setdefault("calendar.csv", calendar_csv_path)
        safe_file_map.setdefault("calendar", calendar_csv_path)

    if scratch_dir is not None:
        safe_file_map = make_readonly_safe_file_map(safe_file_map, scratch_dir)

    return active_context, active_schema_str, q_as_of, safe_file_map


def make_json_serializable(val: Any) -> Any:
    """Ensure raw output values or containers are JSON-serializable."""
    if val is None or isinstance(val, (int, float, str, bool)):
        return val
    try:
        import numpy as np
        if isinstance(val, np.generic):
            return val.item()
        if isinstance(val, np.ndarray):
            return val.tolist()
    except ImportError:
        pass

    try:
        import pandas as pd
        if isinstance(val, pd.DataFrame):
            if val.empty:
                return None
            return val.to_dict(orient="records")
        if isinstance(val, pd.Series):
            if val.empty:
                return None
            return {str(k): make_json_serializable(v) for k, v in val.to_dict().items()}
        if isinstance(val, pd.Timestamp):
            return val.isoformat()
    except ImportError:
        pass

    from datetime import date, datetime
    if isinstance(val, (datetime, date)):
        return val.isoformat()
    if isinstance(val, (list, tuple, set)):
        return [make_json_serializable(x) for x in val]
    if isinstance(val, dict):
        return {str(k): make_json_serializable(v) for k, v in val.items()}

    return str(val)


def iter_process_batch_chunk(
    chunk: List[Dict[str, Any]],
    dataset_context: DatasetContext,
    schema_prompt_str: str,
    agent: CodeAgent,
    prompt_builder: PromptBuilder,
    system_prompt: str,
    max_retries: int,
    timeout: float,
    parser: Optional[SchemaParser] = None,
    scratch_dir: Optional[Path] = None,
    calendar_csv_path: Optional[str] = None,
    on_log: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> Iterator[Dict[str, Any]]:
    """Process one batch chunk of questions: build prompts, run vLLM batch, execute & retry sequentially.

    Step A: Build initial prompts for every question in ``chunk`` via ``prompt_builder``.
    Step B: Call ``agent.generate_code_batch(batch_prompts)`` in one vLLM batch pass.
    Step C: For each generated completion, extract code and execute locally via
        :func:`execute_local`. If execution succeeds and :func:`is_empty_result` is
        ``False``, format and yield immediately; otherwise fall back to sequential
        :func:`execute_with_retry` powered by ``agent.generate_code`` for subsequent retries.
    """
    batch_prompts: List[str] = []
    batch_meta: List[Dict[str, Any]] = []

    # Step A: Build initial prompts for the chunk
    for idx, q_data in enumerate(chunk, 1):
        q_id = q_data.get("id") or q_data.get("question_id") or f"unknown_{idx}"
        question_text = q_data.get("question") or q_data.get("text") or ""
        active_context, active_schema_str, q_as_of, safe_file_map = _resolve_question_context(
            q_data=q_data,
            dataset_context=dataset_context,
            schema_prompt_str=schema_prompt_str,
            parser=parser,
            scratch_dir=scratch_dir,
            calendar_csv_path=calendar_csv_path,
        )
        user_prompt = prompt_builder.build_user_prompt(
            question=question_text,
            schema_context=active_schema_str,
            as_of_date=q_as_of,
            file_map=safe_file_map,
        )
        batch_prompts.append(_format_chatml_prompt(system_prompt, user_prompt))
        batch_meta.append({
            "q_id": q_id,
            "q_type": q_data.get("type"),
            "data_dir": active_context.data_dir,
            "file_map": safe_file_map,
            "as_of_date": q_as_of,
            "user_prompt": user_prompt,
            "q_start": time.perf_counter(),
        })

    # Step B: Simultaneous vLLM batch inference for initial code generation
    try:
        batch_responses = agent.generate_code_batch(batch_prompts)
    except Exception as e:
        logger.error(f"Batch code generation failed: {e}. Falling back to empty batch responses.")
        batch_responses = [""] * len(batch_meta)

    if len(batch_responses) < len(batch_meta):
        batch_responses = list(batch_responses) + [""] * (len(batch_meta) - len(batch_responses))

    single_generate_fn = partial(agent.generate_code, system_prompt=system_prompt)
    retry_budget = max(0, max_retries - 1)

    # Step C: Execute each initial completion, falling back to sequential retry only on failure
    for idx, (meta, raw_response) in enumerate(zip(batch_meta, batch_responses)):
        q_id = meta["q_id"]
        q_start = meta["q_start"]

        thinkings: List[str] = []
        init_th = None
        if hasattr(agent, "last_batch_thinking") and len(agent.last_batch_thinking) > idx:
            init_th = agent.last_batch_thinking[idx]
        if not init_th:
            init_th = extract_thinking(raw_response)
        if init_th:
            thinkings.append(init_th)

        llm_call_count = 1
        code_str = executor.extract_code(raw_response)
        executed_code = code_str if code_str else None
        last_output = None

        retry_outputs: List[Any] = []
        try:
            success, raw_result, error_msg = executor.execute_local(
                code_str,
                meta["data_dir"],
                file_map=meta["file_map"],
                as_of_date=meta["as_of_date"],
                timeout_seconds=timeout,
            )
            if code_str:
                last_output = raw_result if success and raw_result is not None else (error_msg if error_msg else None)
            if success and raw_result is not None:
                retry_outputs.append(raw_result)

            if success and not executor.is_empty_result(raw_result):
                final_success, final_result = True, raw_result
            elif retry_budget > 0:
                if not success:
                    failed_block = f"```python\n{code_str}\n```" if code_str else "```python\n# (No code extracted)\n```"
                    retry_initial_prompt = (
                        f"{meta['user_prompt']}\n\n"
                        f"**FAILED CODE:**\n{failed_block}\n\n"
                        f"**EXACT ERROR:**\n```\n{error_msg}\n```\n"
                        f"Đoạn code trên bị lỗi: {error_msg}. Hãy viết lại code sửa lỗi.\n"
                        f"{executor.RETRY_STRICT_INSTRUCTION}"
                    )
                else:
                    failed_code = f"```python\n{code_str}\n```\n" if code_str else ""
                    retry_initial_prompt = f"{meta['user_prompt']}\n{failed_code}{EMPTY_RESULT_FEEDBACK}"

                def retry_gen_wrapper(p: str) -> str:
                    nonlocal llm_call_count
                    llm_call_count += 1
                    resp = single_generate_fn(p)
                    th = getattr(agent, "last_thinking", None) or extract_thinking(resp)
                    if th:
                        thinkings.append(th)
                    return resp

                def handle_retry_attempt(c_code: str, c_succ: bool, c_res: Any, c_err: str) -> None:
                    nonlocal executed_code, last_output
                    if c_code:
                        executed_code = c_code
                    if c_succ and c_res is not None:
                        retry_outputs.append(c_res)
                    last_output = c_res if c_succ and c_res is not None else (c_err if c_err else None)

                final_success, final_result, _ = executor.execute_with_retry(
                    initial_prompt=retry_initial_prompt,
                    generate_fn=retry_gen_wrapper,
                    data_dir=meta["data_dir"],
                    max_retries=retry_budget,
                    file_map=meta["file_map"],
                    as_of_date=meta["as_of_date"],
                    timeout_seconds=timeout,
                    base_prompt=meta["user_prompt"],
                    on_attempt=handle_retry_attempt,
                )
            else:
                final_success, final_result = False, None

            # Verification gate: validate schema constraints & retry consistency before emitting
            gate_res = None
            if hasattr(agent, "verify_prediction") and callable(getattr(agent, "verify_prediction", None)):
                try:
                    gate_res = agent.verify_prediction(
                        execution_success=final_success,
                        raw_result=final_result,
                        q_type=meta.get("q_type"),
                        code_str=executed_code,
                        retry_results=retry_outputs,
                        error_msg=last_output if not final_success else None,
                    )
                except Exception:
                    gate_res = None

            # Fallback to standalone verification_gate if missing, errored, or mocked
            if (
                not isinstance(gate_res, dict)
                or "type" not in gate_res
                or "answer" not in gate_res
                or type(gate_res.get("type")).__name__ == "MagicMock"
            ):
                gate_res = verification_gate(
                    execution_success=final_success,
                    result=final_result,
                    q_type=meta.get("q_type"),
                    code_str=executed_code,
                    retry_results=retry_outputs,
                    error_msg=last_output if not final_success else None,
                    threshold=(
                        float(getattr(agent, "abstain_threshold", None))
                        if isinstance(getattr(agent, "abstain_threshold", None), (int, float))
                        else None
                    ),
                )

            if gate_res.get("type") == "abstain":
                formatted_answer = "abstain"
                resolved_q_type = "abstain"
            else:
                formatted_answer = gate_res["answer"]
                resolved_q_type = gate_res["type"]
        except Exception as e:
            logger.error(f"Unexpected exception processing {q_id}: {e}")
            formatted_answer = "abstain"
            resolved_q_type = "abstain"

        elapsed_ms = max(0, int((time.perf_counter() - q_start) * 1000))

        if llm_call_count > 1:
            thinking_field = thinkings if len(thinkings) > 0 else None
        else:
            thinking_field = thinkings[0] if len(thinkings) > 0 else None

        log_answer = None if formatted_answer in (None, "abstain", "unanswerable") else formatted_answer
        log_output = make_json_serializable(last_output) if executed_code is not None else None

        log_record = {
            "id": str(q_id),
            "answer": log_answer,
            "elapsed_ms": elapsed_ms,
            "thinking": thinking_field,
            "code": executed_code,
            "output": log_output,
        }

        if on_log is not None:
            on_log(log_record)

        yield format_submission_record(
            q_id=q_id,
            answer=formatted_answer,
            q_type=resolved_q_type,
            abstain_token="abstain",
        )


def process_batch_chunk(
    chunk: List[Dict[str, Any]],
    dataset_context: DatasetContext,
    schema_prompt_str: str,
    agent: CodeAgent,
    prompt_builder: PromptBuilder,
    system_prompt: str,
    max_retries: int,
    timeout: float,
    parser: Optional[SchemaParser] = None,
    scratch_dir: Optional[Path] = None,
    calendar_csv_path: Optional[str] = None,
    on_log: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> List[Dict[str, Any]]:
    """Process a batch chunk and return formatted submission records in order."""
    return list(
        iter_process_batch_chunk(
            chunk=chunk,
            dataset_context=dataset_context,
            schema_prompt_str=schema_prompt_str,
            agent=agent,
            prompt_builder=prompt_builder,
            system_prompt=system_prompt,
            max_retries=max_retries,
            timeout=timeout,
            parser=parser,
            scratch_dir=scratch_dir,
            calendar_csv_path=calendar_csv_path,
            on_log=on_log,
        )
    )


def main() -> None:
    """Main pipeline execution entry point."""
    model_path = DEFAULT_MODEL_PATH
    start_time = time.time()
    args = parse_arguments()
    if args.model_path:
        model_path = args.model_path

    data_dir = Path(args.data_dir).resolve()
    questions_file = Path(args.questions_file).resolve() if args.questions_file else data_dir / "questions.jsonl"
    
    if args.output_dir:
        output_file = (Path(args.output_dir) / "submission.jsonl").resolve()
    else:
        output_file = Path(args.output_file).resolve()
        
    batch_size = max(1, int(args.batch_size))

    # Strict directory rule: create /result and /result/logs if missing
    ensure_output_directories(output_file)

    # Strict directory rule: redirect temporary files, SQLite copies, and scratch scripts to /tmp
    scratch_dir = get_scratch_dir()
    os.environ["TMPDIR"] = str(scratch_dir)
    os.environ["TEMP"] = str(scratch_dir)
    os.environ["TMP"] = str(scratch_dir)
    tempfile.tempdir = str(scratch_dir)

    # Read /data/rulebook.md if present
    rulebook_content = read_rulebook(data_dir=data_dir)
    if rulebook_content:
        logger.info(f"Loaded live rulebook ({len(rulebook_content)} chars)")

    # Read /data/calendar.csv if present
    calendar_csv_path = None
    for c_candidate in (data_dir / "calendar.csv", Path("/data/calendar.csv")):
        if c_candidate.exists():
            calendar_csv_path = str(c_candidate.resolve())
            calendar_utils.set_calendar_csv_path(calendar_csv_path)
            logger.info(f"Found global calendar.csv at {calendar_csv_path}")
            break

    logger.info("=" * 60)
    logger.info("Starting Zalo DataMate 2026 Pipeline Orchestrator (Batch Mode)")
    logger.info(f"Data Directory:    {data_dir}")
    logger.info(f"Questions File:    {questions_file}")
    logger.info(f"Output File:       {output_file}")
    logger.info(f"Scratch Directory: {scratch_dir}")
    logger.info(f"Model Path:        {model_path}")
    logger.info(f"Batch Size:        {batch_size}")
    logger.info(f"Execution Timeout: {args.timeout}s")
    logger.info(f"Max Attempts:      {args.max_retries}")
    if args.limit:
        logger.info(f"Question Limit:    {args.limit}")
    logger.info("=" * 60)

    # Step 1: Parse and cache dataset schema & documentation
    parser = SchemaParser(data_dir=data_dir)
    context = parser.parse()
    schema_prompt_str = get_business_schema(context.data_dir)
    logger.info(f"Discovered {len(context.tables)} table resources in dataset.")
    if context.as_of_date:
        logger.info(f"Dataset as_of_date detected: {context.as_of_date}")

    # Step 2: Initialize Agent and Prompt Builder
    prompt_builder = PromptBuilder(rulebook_content=rulebook_content)
    system_prompt = prompt_builder.build_system_prompt()

    agent = CodeAgent(model_path, dry_run=args.dry_run)

    # Step 3: Load questions
    questions = load_questions(questions_file)
    if args.limit is not None and args.limit > 0:
        questions = questions[:args.limit]
        logger.info(f"Applied --limit: processing {len(questions)} questions.")

    # Determine log destinations (competition requires /result/logs/predictions.jsonl and output logs)
    logs_files: List[Path] = []
    primary_log = (output_file.parent / "logs" / "predictions.jsonl").resolve()
    logs_files.append(primary_log)
    docker_log = Path("/result/logs/predictions.jsonl").resolve()
    try:
        if docker_log != primary_log:
            docker_log.parent.mkdir(parents=True, exist_ok=True)
            test_f = docker_log.parent / ".perm_check"
            test_f.touch()
            test_f.unlink()
            logs_files.append(docker_log)
    except (PermissionError, OSError):
        pass

    if not questions:
        logger.warning(f"No questions found in {questions_file}. Generating empty submission.")
        ensure_output_directories(output_file)
        output_file.write_text("", encoding="utf-8")
        for lf in logs_files:
            try:
                lf.parent.mkdir(parents=True, exist_ok=True)
                lf.write_text("", encoding="utf-8")
            except (PermissionError, OSError):
                pass
        return

    # Step 4: Process questions in BATCH_SIZE chunks and stream to output JSONL
    ensure_output_directories(output_file)
    total_q = len(questions)
    answered_count = 0
    abstained_count = 0
    processed_count = 0

    log_file_handles = []
    for lf in logs_files:
        try:
            lf.parent.mkdir(parents=True, exist_ok=True)
            log_file_handles.append(open(lf, "w", encoding="utf-8"))
        except (PermissionError, OSError) as e:
            logger.warning(f"Could not open log file {lf}: {e}")

    def on_question_log(log_rec: Dict[str, Any]) -> None:
        line_str = json.dumps(log_rec, ensure_ascii=False) + "\n"
        for handle in log_file_handles:
            try:
                handle.write(line_str)
                handle.flush()
                try:
                    os.fsync(handle.fileno())
                except OSError:
                    pass
            except Exception as e:
                logger.error(f"Failed writing to log file {handle.name}: {e}")

    try:
        with open(output_file, "w", encoding="utf-8") as out_f:
            for batch_start in range(0, total_q, batch_size):
                chunk = questions[batch_start : batch_start + batch_size]
                logger.info(
                    f"Processing batch {batch_start // batch_size + 1}: "
                    f"questions {batch_start + 1}-{min(batch_start + len(chunk), total_q)}/{total_q}"
                )

                for record in iter_process_batch_chunk(
                    chunk=chunk,
                    dataset_context=context,
                    schema_prompt_str=schema_prompt_str,
                    agent=agent,
                    prompt_builder=prompt_builder,
                    system_prompt=system_prompt,
                    max_retries=args.max_retries,
                    timeout=args.timeout,
                    parser=parser,
                    scratch_dir=scratch_dir,
                    calendar_csv_path=calendar_csv_path,
                    on_log=on_question_log,
                ):
                    processed_count += 1
                    if record.get("answer") in (None, "abstain"):
                        abstained_count += 1
                    else:
                        answered_count += 1

                    # Strictly omit alias field
                    record.pop("alias", None)
                    out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                    out_f.flush()
                    try:
                        os.fsync(out_f.fileno())
                    except OSError:
                        pass
    finally:
        for handle in log_file_handles:
            try:
                handle.close()
            except Exception:
                pass

    elapsed = time.time() - start_time
    logger.info("=" * 60)
    logger.info("Inference Complete!")
    logger.info(f"Total Questions:   {total_q} (Processed: {processed_count})")
    logger.info(f"Answered:          {answered_count}")
    logger.info(f"Abstained (Safe):  {abstained_count}")
    logger.info(f"Total Time:        {elapsed:.2f}s ({elapsed/max(1, total_q):.2f}s/q)")
    logger.info(f"Saved To:          {output_file}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
