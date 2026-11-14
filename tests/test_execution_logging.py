"""Unit tests for competition-compliant per-question execution logging.

Verifies that /result/logs/predictions.jsonl (and <output_dir>/logs/predictions.jsonl)
is written immediately line-by-line as each question completes with the exact 6 fields:
- id: question ID string
- answer: final answer matching submission.jsonl (or null for abstain/unanswerable)
- elapsed_ms: non-negative integer wall-clock milliseconds
- thinking: exact raw reasoning string or list of strings in call order, or null
- code: exact Python/SQL code string executed, or null
- output: raw return value or stdout captured from code execution, or null
"""

import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd

from src.agent import CodeAgent, extract_thinking
from src.main import main, make_json_serializable


class TestExecutionLogging(unittest.TestCase):
    """Test suite for per-question execution logging in main.py and agent.py."""

    def setUp(self):
        self.sample_data_dir = Path("./data/sample_dataset").resolve()
        self.questions_file = self.sample_data_dir / "questions.jsonl"

    def test_make_json_serializable(self):
        """Ensure diverse pandas/numpy/python data structures are cleanly serialized."""
        df = pd.DataFrame({"col1": [1, 2], "col2": ["a", "b"]})
        serialized_df = make_json_serializable(df)
        self.assertIsInstance(serialized_df, list)
        self.assertEqual(len(serialized_df), 2)
        json.dumps(serialized_df)  # Must not raise TypeError

        series = pd.Series([10, 20], index=["x", "y"])
        serialized_series = make_json_serializable(series)
        self.assertIsInstance(serialized_series, dict)
        json.dumps(serialized_series)

        ts = pd.Timestamp("2026-05-01 12:00:00")
        self.assertEqual(make_json_serializable(ts), "2026-05-01T12:00:00")

        arr = np.array([1, 2, 3])
        self.assertEqual(make_json_serializable(arr), [1, 2, 3])

        np_int = np.int64(42)
        self.assertEqual(make_json_serializable(np_int), 42)
        self.assertIsInstance(make_json_serializable(np_int), int)

        np_float = np.float64(3.14)
        self.assertAlmostEqual(make_json_serializable(np_float), 3.14)
        self.assertIsInstance(make_json_serializable(np_float), float)

    def test_extract_thinking_helper(self):
        """Ensure extract_thinking extracts raw CoT reasoning without truncation."""
        text_closed = "<think>Here is some internal thinking step.</think>```python\nprint(1)\n```"
        self.assertEqual(extract_thinking(text_closed), "Here is some internal thinking step.")

        text_unclosed = "<think>Unclosed reasoning at the end"
        self.assertEqual(extract_thinking(text_unclosed), "Unclosed reasoning at the end")

        text_none = "```python\nprint(1)\n```"
        self.assertIsNone(extract_thinking(text_none))

    def test_predictions_jsonl_created_with_six_required_fields(self):
        """Verify predictions.jsonl is written to <output_dir>/logs/ with all 6 required fields."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            out_dir = Path(tmp_dir) / "test_out"
            q_file = Path(tmp_dir) / "test_q.jsonl"
            q_file.write_text(
                json.dumps({"id": "q_log_1", "question": "Hôm nay là ngày mấy?"}) + "\n"
                + json.dumps({"id": "q_log_2", "question": "Tổng doanh số?"}) + "\n",
                encoding="utf-8",
            )

            test_args = [
                "main.py",
                "--data_dir", str(self.sample_data_dir),
                "--questions_file", str(q_file),
                "--output_dir", str(out_dir),
                "--dry_run",
            ]

            old_argv = sys.argv
            sys.argv = test_args
            try:
                main()
            finally:
                sys.argv = old_argv

            sub_file = out_dir / "submission.jsonl"
            log_file = out_dir / "logs" / "predictions.jsonl"

            self.assertTrue(sub_file.exists(), "submission.jsonl must exist")
            self.assertTrue(log_file.exists(), "logs/predictions.jsonl must exist")

            sub_lines = [json.loads(l) for l in sub_file.read_text(encoding="utf-8").strip().splitlines()]
            log_lines = [json.loads(l) for l in log_file.read_text(encoding="utf-8").strip().splitlines()]

            self.assertEqual(len(log_lines), len(sub_lines))
            self.assertEqual(len(log_lines), 2)

            required_fields = {"id", "answer", "elapsed_ms", "thinking", "code", "output"}

            for sub_rec, log_rec in zip(sub_lines, log_lines):
                self.assertEqual(set(log_rec.keys()), required_fields)
                self.assertEqual(str(log_rec["id"]), str(sub_rec["id"]))

                # elapsed_ms must be non-negative integer
                self.assertIsInstance(log_rec["elapsed_ms"], int)
                self.assertGreaterEqual(log_rec["elapsed_ms"], 0)

                # answer handling
                if sub_rec.get("type") == "abstain" or sub_rec.get("answer") in ("abstain", None):
                    self.assertIsNone(log_rec["answer"])
                else:
                    self.assertEqual(log_rec["answer"], sub_rec["answer"])

                # code must be str or None
                if log_rec["code"] is not None:
                    self.assertIsInstance(log_rec["code"], str)

    def test_predictions_logging_with_thinking_and_retry(self):
        """Verify thinking field preserves raw text and captures multiple calls in order on retry."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            out_dir = Path(tmp_dir) / "test_out_retry"
            q_file = Path(tmp_dir) / "test_retry_q.jsonl"
            q_file.write_text(
                json.dumps({"id": "q_retry_1", "question": "Tính tổng lợi nhuận"}) + "\n",
                encoding="utf-8",
            )

            test_args = [
                "main.py",
                "--data_dir", str(self.sample_data_dir),
                "--questions_file", str(q_file),
                "--output_dir", str(out_dir),
                "--dry_run",
            ]

            # Mock CodeAgent to simulate a 1st attempt failure with thinking, then successful retry with thinking
            with patch("src.main.CodeAgent") as MockAgentCls:
                mock_agent = MagicMock()
                mock_agent.dry_run = False
                mock_agent.backend = "mock"
                # Batch returns first attempt that raises an error
                mock_agent.generate_code_batch.return_value = [
                    "<think>Batch attempt reasoning.</think>```python\nraise ValueError('Initial fail')\n```"
                ]
                mock_agent.last_batch_thinking = ["Batch attempt reasoning."]
                
                # Single generate for retry returns success
                def mock_single_gen(p, system_prompt=None):
                    mock_agent.last_thinking = "Retry attempt reasoning."
                    return "<think>Retry attempt reasoning.</think>```python\nresult = 12345\n```"
                
                mock_agent.generate_code.side_effect = mock_single_gen
                MockAgentCls.return_value = mock_agent

                old_argv = sys.argv
                sys.argv = test_args
                try:
                    main()
                finally:
                    sys.argv = old_argv

            log_file = out_dir / "logs" / "predictions.jsonl"
            self.assertTrue(log_file.exists())
            records = [json.loads(l) for l in log_file.read_text(encoding="utf-8").strip().splitlines()]
            self.assertEqual(len(records), 1)
            rec = records[0]

            self.assertEqual(rec["id"], "q_retry_1")
            self.assertEqual(rec["answer"], 12345)
            # Two calls were made: verify list of strings in call order
            self.assertIsInstance(rec["thinking"], list)
            self.assertEqual(rec["thinking"], ["Batch attempt reasoning.", "Retry attempt reasoning."])
            self.assertIn("result = 12345", rec["code"])
            self.assertEqual(rec["output"], 12345)

    def test_empty_questions_creates_empty_predictions_file(self):
        """Verify empty questions.jsonl creates an empty predictions.jsonl log file."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            out_dir = Path(tmp_dir) / "empty_out"
            q_file = Path(tmp_dir) / "empty_q.jsonl"
            q_file.write_text("", encoding="utf-8")

            test_args = [
                "main.py",
                "--data_dir", str(self.sample_data_dir),
                "--questions_file", str(q_file),
                "--output_dir", str(out_dir),
                "--dry_run",
            ]

            old_argv = sys.argv
            sys.argv = test_args
            try:
                main()
            finally:
                sys.argv = old_argv

            sub_file = out_dir / "submission.jsonl"
            log_file = out_dir / "logs" / "predictions.jsonl"
            self.assertTrue(sub_file.exists())
            self.assertTrue(log_file.exists())
            self.assertEqual(sub_file.read_text(encoding="utf-8"), "")
            self.assertEqual(log_file.read_text(encoding="utf-8"), "")


if __name__ == "__main__":
    unittest.main()
