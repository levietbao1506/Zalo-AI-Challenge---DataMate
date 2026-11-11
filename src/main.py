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
from pathlib import Path
import sys
import time
from typing import Any, Dict, Iterator, List, Optional

from src.agent import CodeAgent, VLLMAgent
from src.executor import (
    EMPTY_RESULT_FEEDBACK,
    SandboxedExecutor,
    execute_local,
    execute_with_retry,
    extract_code,
    is_empty_result,
)
from src.formatter import format_output, format_submission_record
from src.prompt_builder import PromptBuilder
from src.schema_parser import DatasetContext, SchemaParser, get_business_schema

BATCH_SIZE = 32

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    stream=sys.stdout,
)
logger = logging.getLogger("datamate_main")


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Zalo AI Challenge 2026 - DataMate Offline Inference Orchestrator"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="./data",
        help="Directory containing tabular data files and README.md (default: ./data)",
    )
    parser.add_argument(
        "--questions_file",
        type=str,
        default=None,
        help="Path to questions.jsonl (defaults to <data_dir>/questions.jsonl)",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="./submission.jsonl",
        help="Destination path for submission.jsonl (default: ./submission.jsonl)",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="/app/model",
        help="Path to local model weights for vLLM (default: /app/model)",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=15.0,
        help="Execution timeout per question attempt in seconds (default: 15.0)",
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
) -> Any:
    """Process a single question through agent generation, execution, and self-correction."""
    question_text = q_data.get("question") or q_data.get("text") or ""
    active_context, active_schema_str, q_as_of = _resolve_question_context(
        q_data=q_data,
        dataset_context=dataset_context,
        schema_prompt_str=schema_prompt_str,
        parser=parser,
    )

    # 1. Compose user prompt
    user_prompt = prompt_builder.build_user_prompt(
        question=question_text,
        schema_context=active_schema_str,
        as_of_date=q_as_of,
        file_map=active_context.file_map,
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
        file_map=active_context.file_map,
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
) -> tuple[DatasetContext, str, Optional[str]]:
    """Resolve active DatasetContext, cached schema string, and as_of_date for a question."""
    biz_id = q_data.get("business_id") or q_data.get("db_id")
    if biz_id and parser:
        active_context = parser.parse(business_id=biz_id)
        active_schema_str = get_business_schema(active_context.data_dir)
    else:
        active_context = dataset_context
        active_schema_str = schema_prompt_str

    q_as_of = q_data.get("as_of_date") or active_context.as_of_date
    return active_context, active_schema_str, q_as_of


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
        active_context, active_schema_str, q_as_of = _resolve_question_context(
            q_data=q_data,
            dataset_context=dataset_context,
            schema_prompt_str=schema_prompt_str,
            parser=parser,
        )
        user_prompt = prompt_builder.build_user_prompt(
            question=question_text,
            schema_context=active_schema_str,
            as_of_date=q_as_of,
            file_map=active_context.file_map,
        )
        batch_prompts.append(_format_chatml_prompt(system_prompt, user_prompt))
        batch_meta.append({
            "q_id": q_id,
            "q_type": q_data.get("type"),
            "data_dir": active_context.data_dir,
            "file_map": active_context.file_map,
            "as_of_date": q_as_of,
            "user_prompt": user_prompt,
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
    for meta, raw_response in zip(batch_meta, batch_responses):
        q_id = meta["q_id"]
        try:
            code_str = extract_code(raw_response)
            success, raw_result, error_msg = execute_local(
                code_str,
                meta["data_dir"],
                file_map=meta["file_map"],
                as_of_date=meta["as_of_date"],
                timeout_seconds=timeout,
            )

            if success and not is_empty_result(raw_result):
                final_success, final_result = True, raw_result
            elif retry_budget > 0:
                failed_code = f"```python\n{code_str}\n```\n" if code_str else ""
                feedback_line = (
                    f"Đoạn code trên bị lỗi: {error_msg}. Hãy viết lại code sửa lỗi."
                    if not success
                    else EMPTY_RESULT_FEEDBACK
                )
                retry_initial_prompt = f"{meta['user_prompt']}\n{failed_code}{feedback_line}"
                final_success, final_result, _ = execute_with_retry(
                    initial_prompt=retry_initial_prompt,
                    generate_fn=single_generate_fn,
                    data_dir=meta["data_dir"],
                    max_retries=retry_budget,
                    file_map=meta["file_map"],
                    as_of_date=meta["as_of_date"],
                    timeout_seconds=timeout,
                )
            else:
                final_success, final_result = False, None

            formatted_answer = format_output(
                val=final_result if final_success else None,
                success=final_success,
                round_decimals=2,
                abstain_token="abstain",
                clean_text=False,
            )
        except Exception as e:
            logger.error(f"Unexpected exception processing {q_id}: {e}")
            formatted_answer = "abstain"

        yield format_submission_record(
            q_id=q_id,
            answer=formatted_answer,
            q_type=meta["q_type"],
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
        )
    )


def main() -> None:
    """Main pipeline execution entry point."""
    start_time = time.time()
    args = parse_arguments()

    data_dir = Path(args.data_dir).resolve()
    questions_file = Path(args.questions_file).resolve() if args.questions_file else data_dir / "questions.jsonl"
    output_file = Path(args.output_file).resolve()
    batch_size = max(1, int(args.batch_size))

    logger.info("=" * 60)
    logger.info("Starting Zalo DataMate 2026 Pipeline Orchestrator (Batch Mode)")
    logger.info(f"Data Directory:    {data_dir}")
    logger.info(f"Questions File:    {questions_file}")
    logger.info(f"Output File:       {output_file}")
    logger.info(f"Model Path:        {args.model_path}")
    logger.info(f"Batch Size:        {batch_size}")
    logger.info(f"Execution Timeout: {args.timeout}s")
    logger.info(f"Max Attempts:      {args.max_retries}")
    logger.info("=" * 60)

    # Step 1: Parse and cache dataset schema & documentation
    parser = SchemaParser(data_dir=data_dir)
    context = parser.parse()
    schema_prompt_str = get_business_schema(context.data_dir)
    logger.info(f"Discovered {len(context.tables)} table resources in dataset.")
    if context.as_of_date:
        logger.info(f"Dataset as_of_date detected: {context.as_of_date}")

    # Step 2: Initialize Agent and Prompt Builder
    prompt_builder = PromptBuilder()
    system_prompt = prompt_builder.build_system_prompt()
    agent = CodeAgent(
        model_path=args.model_path,
        gpu_memory_utilization=0.90,
        max_model_len=4096,
        tensor_parallel_size=1,
        dry_run=args.dry_run,
    )

    # Step 3: Load questions
    questions = load_questions(questions_file)
    if not questions:
        logger.warning(f"No questions found in {questions_file}. Generating empty submission.")
        output_file.parent.mkdir(parents=True, exist_ok=True)
        output_file.write_text("", encoding="utf-8")
        return

    # Step 4: Process questions in BATCH_SIZE chunks and stream to output JSONL
    output_file.parent.mkdir(parents=True, exist_ok=True)
    total_q = len(questions)
    answered_count = 0
    abstained_count = 0
    processed_count = 0

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
            ):
                processed_count += 1
                if record.get("answer") in (None, "abstain"):
                    abstained_count += 1
                else:
                    answered_count += 1

                out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                out_f.flush()

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
