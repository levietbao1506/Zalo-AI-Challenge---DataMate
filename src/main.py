"""Main Orchestrator for Zalo AI Challenge 2026 - DataMate Track.

Reads questions.jsonl, extracts schemas from CSV/XLSX/SQLite, invokes local vLLM agent,
safely runs Python scripts in sandboxed executor with self-correction retry,
normalizes answers per competition rules (NFC Unicode, 2 decimal places, abstain fallback),
and writes streaming results to submission.jsonl.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import json
import logging
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional

from src.agent import VLLMAgent
from src.executor import SandboxedExecutor
from src.formatter import format_scalar
from src.prompt_builder import PromptBuilder
from src.schema_parser import DatasetContext, SchemaParser

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
        help="Maximum self-correction retries for code execution errors (default: 3)",
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
    q_id = q_data.get("id") or q_data.get("question_id") or "unknown"
    question_text = q_data.get("question") or q_data.get("text") or ""

    # Dynamically resolve business_id context if present
    biz_id = q_data.get("business_id") or q_data.get("db_id")
    if biz_id and parser:
        active_context = parser.parse(business_id=biz_id)
        active_schema_str = active_context.to_prompt_context()
    else:
        active_context = dataset_context
        active_schema_str = schema_prompt_str

    # Extract reference as_of_date if specified at question level, otherwise use dataset default
    q_as_of = q_data.get("as_of_date") or active_context.as_of_date

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

    # 5. Normalize output per competition rules
    # If execution failed after retries, format_scalar will fall back to abstain (None/null)
    raw_result = exec_result.result if exec_result.success else None
    formatted_answer = format_scalar(raw_result, round_decimals=2, abstain_fallback=None)

    return formatted_answer


def main() -> None:
    """Main pipeline execution entry point."""
    start_time = time.time()
    args = parse_arguments()

    data_dir = Path(args.data_dir).resolve()
    questions_file = Path(args.questions_file).resolve() if args.questions_file else data_dir / "questions.jsonl"
    output_file = Path(args.output_file).resolve()

    logger.info("=" * 60)
    logger.info("Starting Zalo DataMate 2026 Pipeline Orchestrator")
    logger.info(f"Data Directory:    {data_dir}")
    logger.info(f"Questions File:    {questions_file}")
    logger.info(f"Output File:       {output_file}")
    logger.info(f"Model Path:        {args.model_path}")
    logger.info(f"Execution Timeout: {args.timeout}s")
    logger.info(f"Max Retries:       {args.max_retries}")
    logger.info("=" * 60)

    # Step 1: Parse and cache dataset schema & documentation
    parser = SchemaParser(data_dir=data_dir)
    context = parser.parse()
    schema_prompt_str = context.to_prompt_context()
    logger.info(f"Discovered {len(context.tables)} table resources in dataset.")
    if context.as_of_date:
        logger.info(f"Dataset as_of_date detected: {context.as_of_date}")

    # Step 2: Initialize Agent, Sandbox, and Prompt Builder
    prompt_builder = PromptBuilder()
    executor = SandboxedExecutor(default_timeout_seconds=args.timeout)
    agent = VLLMAgent(
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

    # Step 4: Process questions and stream to submission.jsonl
    output_file.parent.mkdir(parents=True, exist_ok=True)
    total_q = len(questions)
    answered_count = 0
    abstained_count = 0

    with open(output_file, "w", encoding="utf-8") as out_f:
        for idx, q_data in enumerate(questions, 1):
            q_id = q_data.get("id") or q_data.get("question_id") or f"q_{idx}"
            logger.info(f"[{idx}/{total_q}] Processing Question ID: {q_id}")

            try:
                answer = process_single_question(
                    q_data=q_data,
                    dataset_context=context,
                    schema_prompt_str=schema_prompt_str,
                    agent=agent,
                    executor=executor,
                    prompt_builder=prompt_builder,
                    max_retries=args.max_retries,
                    timeout=args.timeout,
                    parser=parser,
                )
            except Exception as e:
                logger.error(f"Unexpected exception processing {q_id}: {e}")
                # Safe abstain on catastrophic failure to avoid penalized accuracy penalty
                answer = None

            if answer is None:
                abstained_count += 1
            else:
                answered_count += 1

            record = {"id": q_id, "answer": answer}
            out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
            out_f.flush()

    elapsed = time.time() - start_time
    logger.info("=" * 60)
    logger.info("Inference Complete!")
    logger.info(f"Total Questions:   {total_q}")
    logger.info(f"Answered:          {answered_count}")
    logger.info(f"Abstained (Safe):  {abstained_count}")
    logger.info(f"Total Time:        {elapsed:.2f}s ({elapsed/max(1, total_q):.2f}s/q)")
    logger.info(f"Saved To:          {output_file}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
