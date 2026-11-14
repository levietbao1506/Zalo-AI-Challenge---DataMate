"""Tests for Penalized Accuracy (PA) evaluator and confidence verification gate."""

from __future__ import annotations

import io
import json
import math
from pathlib import Path
import tempfile
from typing import Any, Dict, List
import unittest
from unittest.mock import patch

import pandas as pd

from src.agent import CodeAgent, DEFAULT_ABSTAIN_THRESHOLD, verification_gate
from src.evaluator import (
    evaluate_submission,
    is_answer_match,
    print_evaluation_report,
    STANDARD_ANSWER_TYPES,
)


class TestIsAnswerMatch(unittest.TestCase):
    """Test individual answer matching rules in src/evaluator.py."""

    def test_numeric_matching(self):
        self.assertTrue(is_answer_match(100.0, 100.0))
        self.assertTrue(is_answer_match("100.005", 100.0, float_tol=0.01))
        self.assertFalse(is_answer_match(100.05, 100.0, float_tol=0.01))
        self.assertFalse(is_answer_match(float("nan"), 100.0))

    def test_boolean_matching(self):
        self.assertTrue(is_answer_match(True, True))
        self.assertTrue(is_answer_match("True", True))
        self.assertTrue(is_answer_match("Có", True))
        self.assertTrue(is_answer_match(False, False))
        self.assertTrue(is_answer_match("Không", False))
        self.assertFalse(is_answer_match(True, False))
        self.assertFalse(is_answer_match("Có", False))

    def test_string_and_unicode_matching(self):
        self.assertTrue(is_answer_match("Hà Nội", "Hà Nội"))
        self.assertTrue(is_answer_match(" Ha Noi ", "ha noi"))
        self.assertTrue(is_answer_match('"Sản phẩm A"', "sản phẩm a"))
        self.assertFalse(is_answer_match("Sản phẩm A", "Sản phẩm B"))

    def test_date_matching(self):
        self.assertTrue(is_answer_match("2024-05-01", "2024-05-01", q_type="date"))
        self.assertTrue(is_answer_match("2024/05/01", "2024-05-01", q_type="date"))
        self.assertTrue(is_answer_match("2024-05", "2024-05", q_type="date"))
        self.assertFalse(is_answer_match("2024-05-02", "2024-05-01", q_type="date"))

    def test_list_matching(self):
        self.assertTrue(is_answer_match(["A", "B"], ["A", "B"]))
        self.assertTrue(is_answer_match([10.0, 20.0], [10, 20]))
        self.assertFalse(is_answer_match(["A", "B"], ["A", "C"]))
        self.assertFalse(is_answer_match(["A"], ["A", "B"]))

    def test_table_matching(self):
        t1 = {"columns": ["Ten", "Tuoi"], "rows": [["An", 20], ["Binh", 25]]}
        t2 = {"columns": ["ten", "tuoi"], "rows": [["an", 20.0], ["Binh", 25]]}
        t3 = {"columns": ["Ten", "Tuoi"], "rows": [["An", 21], ["Binh", 25]]}
        self.assertTrue(is_answer_match(t1, t2))
        self.assertFalse(is_answer_match(t1, t3))

    def test_abstain_matching(self):
        self.assertTrue(is_answer_match("abstain", "abstain"))
        self.assertTrue(is_answer_match(None, "abstain"))
        self.assertFalse(is_answer_match(123, "abstain"))
        self.assertFalse(is_answer_match("abstain", 123))

    def test_unanswerable_matching(self):
        self.assertTrue(is_answer_match("unanswerable", "unanswerable"))
        self.assertTrue(is_answer_match("Unanswerable", "unanswerable"))
        self.assertFalse(is_answer_match("abstain", "unanswerable"))
        self.assertFalse(is_answer_match("123", "unanswerable"))


class TestEvaluateSubmission(unittest.TestCase):
    """Test Penalized Accuracy scoring and type breakdown reporting."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = Path(self.temp_dir.name)

    def tearDown(self):
        self.temp_dir.cleanup()

    def _write_jsonl(self, path: Path, records: List[Dict[str, Any]]):
        with open(path, "w", encoding="utf-8") as f:
            for rec in records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    def test_penalized_accuracy_calculation(self):
        """Test calculation of PA = (1 / N) * sum(s_i) with +1, 0, -1."""
        gt_records = [
            {"id": "q1", "type": "number", "expected_answer": 100},
            {"id": "q2", "type": "string", "expected_answer": "Hà Nội", "alias": ["Thủ đô Hà Nội", "HN"]},
            {"id": "q3", "type": "percent", "expected_answer": 15.5},
            {"id": "q4", "type": "unanswerable", "expected_answer": "unanswerable"},
            {"id": "q5", "type": "unanswerable", "expected_answer": "unanswerable"},
            {"id": "q6", "type": "date", "expected_answer": "2024-01-01"},
            {"id": "q7", "type": "boolean", "expected_answer": True},
        ]
        gt_file = self.temp_path / "ground_truth.jsonl"
        self._write_jsonl(gt_file, gt_records)

        # q1: correct (+1)
        # q2: matched via alias (+1)
        # q3: wrong value (-1)
        # q4: predicted unanswerable (+1)
        # q5: predicted concrete number on unanswerable (-1)
        # q6: abstained (0)
        # q7: missing from submission (0)
        sub_records = [
            {"id": "q1", "type": "number", "answer": 100.0},
            {"id": "q2", "type": "string", "answer": "Thủ đô Hà Nội"},
            {"id": "q3", "type": "percent", "answer": 99.9},
            {"id": "q4", "type": "unanswerable", "answer": "unanswerable"},
            {"id": "q5", "type": "number", "answer": 500000},
            {"id": "q6", "type": "abstain", "answer": "abstain"},
        ]
        sub_file = self.temp_path / "submission.jsonl"
        self._write_jsonl(sub_file, sub_records)

        summary = evaluate_submission(sub_file, gt_file, penalty_weight=1.0)

        self.assertEqual(summary["total_questions"], 7)
        self.assertEqual(summary["correct"], 3)  # q1, q2, q4
        self.assertEqual(summary["abstained"], 2)  # q6 (explicit abstain), q7 (missing)
        self.assertEqual(summary["wrong"], 2)  # q3, q5

        # Net penalized score: 3*(+1) + 2*(0) + 2*(-1) = 3 - 2 = 1.0
        self.assertEqual(summary["penalized_score"], 1.0)
        # PA = 1.0 / 7 = 0.1429
        self.assertAlmostEqual(summary["penalized_accuracy"], 1.0 / 7.0, places=4)
        # Raw accuracy = 3 / 7 = 0.4286
        self.assertAlmostEqual(summary["raw_accuracy"], 3.0 / 7.0, places=4)

    def test_unanswerable_scoring_cases(self):
        """Test unanswerable questions: unanswerable (+1), abstain (0), concrete answer (-1)."""
        gt_records = [
            {"id": "u1", "type": "unanswerable", "expected_answer": "unanswerable"},
            {"id": "u2", "type": "unanswerable", "expected_answer": "unanswerable"},
            {"id": "u3", "type": "unanswerable", "expected_answer": "unanswerable"},
        ]
        gt_file = self.temp_path / "gt_u.jsonl"
        self._write_jsonl(gt_file, gt_records)

        sub_records = [
            {"id": "u1", "type": "unanswerable", "answer": "unanswerable"},  # +1
            {"id": "u2", "type": "abstain", "answer": "abstain"},  # 0
            {"id": "u3", "type": "number", "answer": 42},  # -1
        ]
        sub_file = self.temp_path / "sub_u.jsonl"
        self._write_jsonl(sub_file, sub_records)

        summary = evaluate_submission(sub_file, gt_file, penalty_weight=1.0)
        self.assertEqual(summary["correct"], 1)
        self.assertEqual(summary["abstained"], 1)
        self.assertEqual(summary["wrong"], 1)
        self.assertEqual(summary["penalized_score"], 0.0)
        self.assertEqual(summary["penalized_accuracy"], 0.0)

    def test_type_breakdown_and_report_printing(self):
        """Ensure type breakdown dictionary is populated for all standard types and report prints cleanly."""
        gt_records = [
            {"id": "q1", "type": "number", "expected_answer": 10},
            {"id": "q2", "type": "string", "expected_answer": "test"},
            {"id": "q3", "type": "boolean", "expected_answer": False},
        ]
        gt_file = self.temp_path / "gt_types.jsonl"
        self._write_jsonl(gt_file, gt_records)

        sub_records = [
            {"id": "q1", "type": "number", "answer": 10},
            {"id": "q2", "type": "string", "answer": "wrong"},
            {"id": "q3", "type": "boolean", "answer": False},
        ]
        sub_file = self.temp_path / "sub_types.jsonl"
        self._write_jsonl(sub_file, sub_records)

        summary = evaluate_submission(sub_file, gt_file, total_duration_seconds=1.5)
        breakdown = summary["type_breakdown"]

        # All standard types should be in breakdown
        for std_type in STANDARD_ANSWER_TYPES:
            self.assertIn(std_type, breakdown)

        self.assertEqual(breakdown["number"]["total"], 1)
        self.assertEqual(breakdown["number"]["correct"], 1)
        self.assertEqual(breakdown["string"]["total"], 1)
        self.assertEqual(breakdown["string"]["wrong"], 1)
        self.assertEqual(breakdown["boolean"]["total"], 1)
        self.assertEqual(breakdown["boolean"]["correct"], 1)

        # Ensure print_evaluation_report runs without error
        out_buf = io.StringIO()
        with patch("sys.stdout", out_buf):
            print_evaluation_report(summary)
        output_text = out_buf.getvalue()
        self.assertIn("ZALO AI CHALLENGE 2026 - DATAMATE PENALIZED ACCURACY", output_text)
        self.assertIn("CHI TIẾT THEO LOẠI CÂU HỎI", output_text)


class TestVerificationGate(unittest.TestCase):
    """Test confidence verification gate in src/agent.py."""

    def test_hard_execution_failure_abstains(self):
        res = verification_gate(
            execution_success=False,
            result=None,
            q_type="number",
            error_msg="ZeroDivisionError: division by zero",
        )
        self.assertEqual(res["type"], "abstain")
        self.assertEqual(res["answer"], "abstain")
        self.assertEqual(res["confidence"], 0.0)
        self.assertFalse(res["passed"])
        self.assertIn("ZeroDivisionError", res["reason"])

    def test_none_or_nan_or_inf_abstains(self):
        res_none = verification_gate(execution_success=True, result=None, q_type="number")
        self.assertEqual(res_none["type"], "abstain")

        res_nan = verification_gate(execution_success=True, result=float("nan"), q_type="number")
        self.assertEqual(res_nan["type"], "abstain")

        res_inf = verification_gate(execution_success=True, result=float("inf"), q_type="number")
        self.assertEqual(res_inf["type"], "abstain")

    def test_empty_results_abstains(self):
        res_empty_df = verification_gate(execution_success=True, result=pd.DataFrame(), q_type="table")
        self.assertEqual(res_empty_df["type"], "abstain")

        res_empty_list = verification_gate(execution_success=True, result=[], q_type="list")
        self.assertEqual(res_empty_list["type"], "abstain")

        res_empty_str = verification_gate(execution_success=True, result="   ", q_type="string")
        self.assertEqual(res_empty_str["type"], "abstain")

    def test_explicit_abstain_and_unanswerable_tokens(self):
        res_abs = verification_gate(execution_success=True, result="không xác định", q_type="string")
        self.assertEqual(res_abs["type"], "abstain")

        res_unans = verification_gate(execution_success=True, result="unanswerable", q_type="unanswerable")
        self.assertEqual(res_unans["type"], "unanswerable")
        self.assertEqual(res_unans["answer"], "unanswerable")
        self.assertTrue(res_unans["passed"])

    def test_schema_constraint_violations(self):
        # Number expected, but cannot parse to number
        res_num_err = verification_gate(execution_success=True, result="not_a_number", q_type="number")
        self.assertEqual(res_num_err["type"], "abstain")

        # Percent expected, but cannot parse to float
        res_pct_err = verification_gate(execution_success=True, result="xyz%", q_type="percent")
        self.assertEqual(res_pct_err["type"], "abstain")

        # Date expected, but invalid date string
        res_date_err = verification_gate(execution_success=True, result="someday", q_type="date")
        self.assertEqual(res_date_err["type"], "abstain")

    def test_conflicting_retries_penalized_and_abstains(self):
        # Two retry attempts with conflicting outputs: 100 vs 250
        # Default threshold is 0.6. Initial confidence 1.0 - 0.5 conflict penalty = 0.5 < 0.6 -> abstain!
        res_conflict = verification_gate(
            execution_success=True,
            result=250,
            q_type="number",
            retry_results=[100, 250],
            threshold=DEFAULT_ABSTAIN_THRESHOLD,
        )
        self.assertEqual(res_conflict["type"], "abstain")
        self.assertEqual(res_conflict["answer"], "abstain")
        self.assertFalse(res_conflict["passed"])
        self.assertIn("Conflicting results", res_conflict["reason"])

    def test_consistent_retries_passes(self):
        # Two retry attempts with matching outputs: 100 and 100.0
        res_consistent = verification_gate(
            execution_success=True,
            result=100.0,
            q_type="number",
            retry_results=[100, 100.0],
            threshold=DEFAULT_ABSTAIN_THRESHOLD,
        )
        self.assertEqual(res_consistent["type"], "number")
        self.assertEqual(res_consistent["answer"], 100.0)
        self.assertTrue(res_consistent["passed"])
        self.assertEqual(res_consistent["confidence"], 1.0)

    def test_agent_verify_prediction_method(self):
        agent = CodeAgent(dry_run=True, abstain_threshold=0.6)
        res = agent.verify_prediction(
            execution_success=True,
            result=45.5,
            q_type="number",
        )
        self.assertEqual(res["type"], "number")
        self.assertEqual(res["answer"], 45.5)
        self.assertTrue(res["passed"])


if __name__ == "__main__":
    unittest.main()
