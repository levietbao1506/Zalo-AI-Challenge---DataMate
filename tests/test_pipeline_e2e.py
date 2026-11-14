"""End-to-End Pipeline & Integration Unit Tests (Bước 6: Tích hợp Pipeline & Kiểm thử)."""

import json
from pathlib import Path
import tempfile
import time
import unittest

from src.evaluator import (
    evaluate_submission,
    is_answer_match,
)
from src.main import (
    BATCH_SIZE,
    DEFAULT_DATA_DIR,
    DEFAULT_MODEL_PATH,
    DEFAULT_OUTPUT_FILE,
    load_questions,
    main,
    parse_arguments,
    process_batch_chunk,
    process_single_question,
)
from src.agent import VLLMAgent
from src.executor import SandboxedExecutor
from src.prompt_builder import PromptBuilder
from src.schema_parser import SchemaParser


class TestPipelineE2E(unittest.TestCase):
    """Test suite for end-to-end integration, batch processing, and evaluation."""

    def setUp(self):
        self.sample_data_dir = Path("./data/sample_dataset").resolve()
        self.questions_file = self.sample_data_dir / "questions.jsonl"
        self.ground_truth_file = self.sample_data_dir / "ground_truth.jsonl"

    def test_load_questions(self):
        """Ensure load_questions reads all valid records and ignores blank/corrupt lines."""
        with tempfile.NamedTemporaryFile("w+", encoding="utf-8", delete=False) as f:
            f.write('{"id": "q1", "question": "Hôm nay là ngày mấy?"}\n')
            f.write("\n")  # Blank line
            f.write('{"id": "q2", "question": "Doanh số chi nhánh HN?"}\n')
            f.write("malformed json line\n")  # Corrupted line
            f.flush()
            temp_path = Path(f.name)

        try:
            loaded = load_questions(temp_path)
            self.assertEqual(len(loaded), 2)
            self.assertEqual(loaded[0]["id"], "q1")
            self.assertEqual(loaded[1]["id"], "q2")
        finally:
            temp_path.unlink(missing_ok=True)

    def test_process_single_question_full_loop(self):
        """Test single question execution through Prompt -> Agent -> Sandbox -> Formatter."""
        parser = SchemaParser(self.sample_data_dir)
        context = parser.parse()
        schema_str = context.to_prompt_context()

        agent = VLLMAgent(dry_run=True)
        executor = SandboxedExecutor(default_timeout_seconds=5.0)
        prompt_builder = PromptBuilder()

        q_data = {
            "id": "test_q",
            "as_of_date": "2024-06-15",
            "question": "Có bao nhiêu đơn hàng thành công?",
        }

        answer = process_single_question(
            q_data=q_data,
            dataset_context=context,
            schema_prompt_str=schema_str,
            agent=agent,
            executor=executor,
            prompt_builder=prompt_builder,
            max_retries=3,
            timeout=5.0,
            parser=parser,
        )

        # In dry_run mock mode, it safely falls back to abstain
        self.assertIn(answer, ("abstain", None))

    def test_evaluator_is_answer_match(self):
        """Test match logic for numbers, strings, and abstain fallback."""
        # 1. Float with tolerance <= 0.01
        self.assertTrue(is_answer_match(123.456, 123.46))
        self.assertTrue(is_answer_match(100.0, 100))
        self.assertFalse(is_answer_match(123.45, 125.00))

        # 2. String with NFC & case-insensitive matching
        self.assertTrue(is_answer_match("Hà Nội", "hà nội"))
        self.assertTrue(is_answer_match("  Đà Nẵng!  ", "đà nẵng"))
        self.assertFalse(is_answer_match("Hà Nội", "Sài Gòn"))

        # 3. Abstain matching
        self.assertTrue(is_answer_match("abstain", "abstain"))
        self.assertTrue(is_answer_match(None, "abstain"))
        self.assertTrue(is_answer_match("abstain", None))
        self.assertFalse(is_answer_match("hà nội", "abstain"))

        # 4. Boolean matching
        self.assertTrue(is_answer_match(True, True))
        self.assertFalse(is_answer_match(True, False))

    def test_end_to_end_batch_inference_and_latency(self):
        """Run full 15 questions batch pipeline, check submission output and latency."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            out_file = Path(tmp_dir) / "submission.jsonl"

            start_t = time.time()
            # Run main program via sys.argv mock
            import sys
            test_args = [
                "main.py",
                "--data_dir", str(self.sample_data_dir),
                "--questions_file", str(self.questions_file),
                "--output_file", str(out_file),
                "--dry_run",
            ]
            old_argv = sys.argv
            sys.argv = test_args
            try:
                main()
            finally:
                sys.argv = old_argv

            duration = time.time() - start_t

            # Verify submission.jsonl exists and has 2 lines
            self.assertTrue(out_file.exists())
            lines = out_file.read_text(encoding="utf-8").strip().splitlines()
            self.assertEqual(len(lines), 2)

            # Check JSON schema of each line
            for line in lines:
                rec = json.loads(line)
                self.assertIn("id", rec)
                self.assertIn("type", rec)
                self.assertIn("answer", rec)

            # Evaluate submission against ground truth
            eval_summary = evaluate_submission(
                submission_file=out_file,
                ground_truth_file=self.ground_truth_file,
                total_duration_seconds=duration,
            )

            self.assertEqual(eval_summary["total_questions"], 15)
            self.assertGreater(eval_summary["average_latency_seconds"], 0.0)
            self.assertLess(eval_summary["average_latency_seconds"], 5.0)  # fast batch latency
            self.assertGreaterEqual(eval_summary["penalized_score"], 0.0)

    def test_batch_chunk_pipeline_fast_path_and_sequential_retry_fallback(self):
        """Batch chunk runs generate_code_batch once and falls back to generate_code only for failures/empty."""
        self.assertEqual(BATCH_SIZE, 32)
        parser = SchemaParser(self.sample_data_dir)
        context = parser.parse()
        schema_str = context.to_prompt_context()
        prompt_builder = PromptBuilder()
        system_prompt = prompt_builder.build_system_prompt()

        batch_calls = []
        retry_calls = []

        class MockBatchAndRetryAgent:
            def generate_code_batch(self, prompts):
                batch_calls.append(prompts)
                return [
                    "```python\nfinal_result = 100\n```",
                    "```python\nimport pandas as pd\nfinal_result = pd.DataFrame()\n```",
                    "```python\nfinal_result = missing_var\n```",
                ]

            def generate_code(self, prompt, system_prompt=None):
                retry_calls.append(prompt)
                if "missing_var" in prompt:
                    return 'final_result = "Hà Nội"'
                return "final_result = 250.5"

        chunk = [
            {"id": "q1", "type": "number", "question": "Q1?"},
            {"id": "q2", "type": "number", "question": "Q2?"},
            {"id": "q3", "type": "string", "question": "Q3?"},
        ]

        records = process_batch_chunk(
            chunk=chunk,
            dataset_context=context,
            schema_prompt_str=schema_str,
            agent=MockBatchAndRetryAgent(),
            prompt_builder=prompt_builder,
            system_prompt=system_prompt,
            max_retries=3,
            timeout=5.0,
            parser=parser,
        )

        self.assertEqual(len(batch_calls), 1)
        self.assertEqual(len(batch_calls[0]), 3)
        # Only Q2 (empty DataFrame) and Q3 (NameError) needed sequential retries; Q1 succeeded on batch
        self.assertEqual(len(retry_calls), 2)
        self.assertEqual(records[0]["answer"], 100)
        self.assertEqual(records[1]["answer"], 250.5)
        self.assertEqual(records[2]["answer"], "Hà Nội")

    def test_main_defaults_match_qwen3_14b_config(self):
        """Ensure CLI defaults target local Qwen3-14B, sample_dataset, and result/submission.jsonl."""
        import sys

        old_argv = sys.argv
        sys.argv = ["main.py"]
        try:
            args = parse_arguments()
        finally:
            sys.argv = old_argv

        expected_model = "./models/Qwen3-14B"
        self.assertEqual(args.model_path, expected_model)
        self.assertEqual(DEFAULT_MODEL_PATH, expected_model)
        self.assertEqual(DEFAULT_DATA_DIR, "./data/sample_dataset")
        self.assertEqual(DEFAULT_OUTPUT_FILE, "result/submission.jsonl")
        self.assertEqual(args.batch_size, 32)

    def test_read_only_data_directory_enforcement(self):
        """Ensure pipeline treats data_dir as strictly read-only and writes only to output_dir and scratch."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            out_dir = Path(tmp_dir) / "test_out"
            data_files_before = set(self.sample_data_dir.rglob("*"))

            import sys
            test_args = [
                "main.py",
                "--data_dir", str(self.sample_data_dir),
                "--questions_file", str(self.questions_file),
                "--output_dir", str(out_dir),
                "--dry_run",
            ]
            old_argv = sys.argv
            sys.argv = test_args
            try:
                main()
            finally:
                sys.argv = old_argv

            # Verify no new files were created in data_dir
            data_files_after = set(self.sample_data_dir.rglob("*"))
            self.assertEqual(data_files_before, data_files_after, "data_dir must not have any new or modified files")

            # Verify outputs are written strictly in output_dir
            self.assertTrue((out_dir / "submission.jsonl").exists())
            self.assertTrue((out_dir / "logs" / "predictions.jsonl").exists())

    def test_end_to_end_predictions_log_generation_and_pa_report(self):
        """Verify pipeline writes competition-compliant predictions.jsonl with 6 fields and computes PA."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            out_dir = Path(tmp_dir) / "eval_out"
            out_file = out_dir / "submission.jsonl"
            log_file = out_dir / "logs" / "predictions.jsonl"

            import sys
            test_args = [
                "main.py",
                "--data_dir", str(self.sample_data_dir),
                "--questions_file", str(self.questions_file),
                "--output_dir", str(out_dir),
                "--dry_run",
            ]
            old_argv = sys.argv
            sys.argv = test_args
            try:
                main()
            finally:
                sys.argv = old_argv

            self.assertTrue(log_file.exists())
            log_lines = log_file.read_text(encoding="utf-8").strip().splitlines()
            self.assertEqual(len(log_lines), 2)

            required_fields = {"id", "answer", "elapsed_ms", "thinking", "code", "output"}
            for line in log_lines:
                rec = json.loads(line)
                self.assertEqual(set(rec.keys()), required_fields)
                self.assertIsInstance(rec["elapsed_ms"], int)
                self.assertGreaterEqual(rec["elapsed_ms"], 0)

            # Check Penalized Accuracy report
            summary = evaluate_submission(out_file, self.ground_truth_file)
            self.assertIn("penalized_accuracy", summary)
            self.assertIn("type_breakdown", summary)
            self.assertIsInstance(summary["type_breakdown"], dict)


if __name__ == "__main__":
    unittest.main()
