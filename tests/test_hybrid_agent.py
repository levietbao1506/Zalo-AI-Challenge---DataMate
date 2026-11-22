"""Unit tests for the Adaptive Hybrid state machine in src/agent.py."""

import unittest
from unittest.mock import MagicMock

from src.agent import execute_adaptive_hybrid
from src.sql_executor import SQLExecutionResult


class MockAgent:
    def __init__(self, responses: list[str]):
        self.responses = list(responses)
        self.call_count = 0
        self.last_thinking = None

    def generate_code(self, prompt: str, system_prompt: str = None) -> str:
        if self.call_count < len(self.responses):
            resp = self.responses[self.call_count]
        else:
            resp = self.responses[-1]
        self.call_count += 1
        return resp


class TestHybridAgent(unittest.TestCase):
    def setUp(self):
        self.mock_store = MagicMock()
        self.mock_store.business_id = "biz_test"
        self.mock_store.business_dir = "/tmp"

        self.mock_sql_executor = MagicMock()
        self.schema_card = "TABLE sales (id INTEGER, amount DOUBLE, order_date DATE)"
        self.schema_meta = {"tables": {"sales": {"columns": ["id", "amount", "order_date"]}}}
        self.rulebook_text = "Đơn hoàn thành có trạng thái COMPLETED."

    def test_sql_fast_closes_immediately(self):
        q = {
            "id": "q1",
            "question": "Tổng doanh thu là bao nhiêu?",
            "as_of_date": "2024-05-01",
            "type": "number",
        }
        sql_resp = (
            "<think>Reasoning cleanly for SQL Fast</think>\n"
            "```sql\nSELECT 1500000 AS amount;\n```\n"
            "```json\n"
            '{"expected_type":"number","post_process":"none","is_unanswerable":false,"reason":null}\n'
            "```"
        )
        agent = MockAgent([sql_resp])

        self.mock_sql_executor.execute.return_value = SQLExecutionResult(
            success=True,
            sql="SELECT 1500000 AS amount;",
            columns=["amount"],
            rows=[[1500000]],
            row_count=1,
            anomalies=[],
        )

        res = execute_adaptive_hybrid(
            question_record=q,
            duckdb_store=self.mock_store,
            sql_executor=self.mock_sql_executor,
            schema_card=self.schema_card,
            schema_meta=self.schema_meta,
            rulebook_text=self.rulebook_text,
            agent=agent,
        )

        self.assertEqual(res["submission_record"]["type"], "number")
        self.assertEqual(res["submission_record"]["answer"], 1500000)
        self.assertEqual(res["route_trace"]["route_initial"], "SQL_FAST")
        self.assertFalse(res["route_trace"]["escalated"])
        self.assertEqual(res["route_trace"]["llm_calls"], 1)
        self.assertIn("clean_schema_sql_candidate", res["route_trace"]["reasons"])
        self.assertEqual(res["route_trace"]["escalation_reasons"], [])
        self.assertEqual(res["log_record"]["thinking"], ["Reasoning cleanly for SQL Fast"])

    def test_malformed_sql_response_repairs_without_executing_select_one(self):
        question = {
            "id": "q_parse_repair",
            "question": "Count all rows",
            "as_of_date": "2024-05-01",
            "type": "number",
        }
        repaired = (
            "```sql\nSELECT 7 AS answer\n```\n"
            "```json\n"
            '{"expected_type":"number","post_process":"none","is_unanswerable":false,"reason":null}\n'
            "```"
        )
        agent = MockAgent(["not structured sql", repaired])
        self.mock_sql_executor.execute.return_value = SQLExecutionResult(
            success=True,
            sql="SELECT 7 AS answer",
            columns=["answer"],
            rows=[[7]],
            row_count=1,
            anomalies=[],
        )
        result = execute_adaptive_hybrid(
            question_record=question,
            duckdb_store=self.mock_store,
            sql_executor=self.mock_sql_executor,
            schema_card=self.schema_card,
            schema_meta=self.schema_meta,
            rulebook_text=self.rulebook_text,
            agent=agent,
        )
        self.assertEqual(result["submission_record"]["answer"], 7)
        self.mock_sql_executor.execute.assert_called_once_with("SELECT 7 AS answer", timeout_seconds=25.0)
        self.assertNotIn("SELECT 1", result["log_record"]["code"])

    def test_malformed_sql_and_repair_abstains_without_execution(self):
        question = {
            "id": "q_parse_fail",
            "question": "Count all rows",
            "as_of_date": "2024-05-01",
            "type": "number",
        }
        agent = MockAgent(["invalid first response", "invalid repair response"])
        result = execute_adaptive_hybrid(
            question_record=question,
            duckdb_store=self.mock_store,
            sql_executor=self.mock_sql_executor,
            schema_card=self.schema_card,
            schema_meta=self.schema_meta,
            rulebook_text=self.rulebook_text,
            agent=agent,
        )
        self.assertEqual(result["submission_record"]["type"], "abstain")
        self.mock_sql_executor.execute.assert_not_called()

    def test_sql_fast_zero_scalar_escalates_to_python(self):
        q = {
            "id": "q2",
            "question": "Tổng doanh thu là bao nhiêu?",
            "as_of_date": "2024-05-01",
            "type": "number",
        }
        sql_resp = (
            "<think>Calculating revenue with SQL</think>\n"
            "```sql\nSELECT 0 AS amount;\n```\n"
            "```json\n"
            '{"expected_type":"number","post_process":"none","is_unanswerable":false,"reason":null}\n'
            "```"
        )
        py_resp = (
            "<think>Python verification running</think>\n"
            "```python\n"
            'FINAL_ANSWER = {"type": "number", "answer": 250000}\n'
            "```"
        )
        agent = MockAgent([sql_resp, py_resp])

        self.mock_sql_executor.execute.return_value = SQLExecutionResult(
            success=True,
            sql="SELECT 0 AS amount;",
            columns=["amount"],
            rows=[[0]],
            row_count=1,
            anomalies=["zero_scalar"],
        )

        res = execute_adaptive_hybrid(
            question_record=q,
            duckdb_store=self.mock_store,
            sql_executor=self.mock_sql_executor,
            schema_card=self.schema_card,
            schema_meta=self.schema_meta,
            rulebook_text=self.rulebook_text,
            agent=agent,
        )

        self.assertTrue(res["route_trace"]["escalated"])
        self.assertEqual(res["submission_record"]["type"], "number")
        self.assertEqual(res["submission_record"]["answer"], 250000)
        self.assertEqual(
            res["log_record"]["thinking"],
            ["Calculating revenue with SQL", "Python verification running"],
        )

    def test_hybrid_dual_matches(self):
        q = {
            "id": "q3",
            "question": "Lợi nhuận gộp là bao nhiêu?",
            "as_of_date": "2024-05-01",
            "type": "number",
        }
        sql_resp = (
            "<think>SQL branch reasoning</think>\n"
            "```sql\nSELECT 50000 AS profit;\n```\n"
            "```json\n"
            '{"expected_type":"number","post_process":"none","is_unanswerable":false,"reason":null}\n'
            "```"
        )
        py_resp = (
            "<think>Python branch reasoning</think>\n"
            "```python\n"
            'FINAL_ANSWER = {"type": "number", "answer": 50000.0}\n'
            "```"
        )
        agent = MockAgent([sql_resp, py_resp])

        self.mock_sql_executor.execute.return_value = SQLExecutionResult(
            success=True,
            sql="SELECT 50000 AS profit;",
            columns=["profit"],
            rows=[[50000]],
            row_count=1,
            anomalies=[],
        )

        res = execute_adaptive_hybrid(
            question_record=q,
            duckdb_store=self.mock_store,
            sql_executor=self.mock_sql_executor,
            schema_card=self.schema_card,
            schema_meta=self.schema_meta,
            rulebook_text=self.rulebook_text,
            agent=agent,
        )

        self.assertEqual(res["route_trace"]["route_initial"], "HYBRID_DUAL")
        self.assertEqual(res["submission_record"]["type"], "number")
        self.assertEqual(res["submission_record"]["answer"], 50000)
        self.assertEqual(
            res["log_record"]["thinking"],
            ["SQL branch reasoning", "Python branch reasoning"],
        )

    def test_hybrid_dual_conflict_resolved_by_adjudicator(self):
        q = {
            "id": "q4",
            "question": "Lợi nhuận gộp là bao nhiêu?",
            "as_of_date": "2024-05-01",
            "type": "number",
        }
        sql_resp = (
            "<think>SQL Branch</think>\n"
            "```sql\nSELECT 50000 AS profit;\n```\n"
            "```json\n"
            '{"expected_type":"number","post_process":"none","is_unanswerable":false,"reason":null}\n'
            "```"
        )
        py_resp = (
            "<think>Python Branch</think>\n"
            "```python\n"
            'FINAL_ANSWER = {"type": "number", "answer": 60000}\n'
            "```"
        )
        adj_resp = (
            "<think>Adjudicating: Python omitted VAT deduction while SQL followed rulebook correctly</think>\n"
            "```json\n"
            '{"decision":"verified_answer","type":"number","answer":50000,"reason":"SQL correctly deducted VAT"}\n'
            "```"
        )
        agent = MockAgent([sql_resp, py_resp, adj_resp])

        self.mock_sql_executor.execute.return_value = SQLExecutionResult(
            success=True,
            sql="SELECT 50000 AS profit;",
            columns=["profit"],
            rows=[[50000]],
            row_count=1,
            anomalies=[],
        )

        res = execute_adaptive_hybrid(
            question_record=q,
            duckdb_store=self.mock_store,
            sql_executor=self.mock_sql_executor,
            schema_card=self.schema_card,
            schema_meta=self.schema_meta,
            rulebook_text=self.rulebook_text,
            agent=agent,
        )

        self.assertEqual(res["route_trace"]["final_stage"], "ADJUDICATOR")
        self.assertEqual(res["submission_record"]["type"], "number")
        self.assertEqual(res["submission_record"]["answer"], 50000)
        self.assertEqual(
            res["log_record"]["thinking"],
            [
                "SQL Branch",
                "Python Branch",
                "Adjudicating: Python omitted VAT deduction while SQL followed rulebook correctly",
            ],
        )

    def test_hybrid_dual_conflict_abstains_when_unresolved(self):
        q = {
            "id": "q5",
            "question": "Lợi nhuận gộp là bao nhiêu?",
            "as_of_date": "2024-05-01",
            "type": "number",
        }
        sql_resp = (
            "<think>SQL Branch</think>\n"
            "```sql\nSELECT 50000 AS profit;\n```\n"
            "```json\n"
            '{"expected_type":"number","post_process":"none","is_unanswerable":false,"reason":null}\n'
            "```"
        )
        py_resp = (
            "<think>Python Branch</think>\n"
            "```python\n"
            'FINAL_ANSWER = {"type": "number", "answer": 60000}\n'
            "```"
        )
        adj_resp = (
            "<think>Both branches have conflicting interpretations of missing cost data</think>\n"
            "```json\n"
            '{"decision":"abstain","type":"abstain","reason":"Data ambiguity"}\n'
            "```"
        )
        agent = MockAgent([sql_resp, py_resp, adj_resp])

        self.mock_sql_executor.execute.return_value = SQLExecutionResult(
            success=True,
            sql="SELECT 50000 AS profit;",
            columns=["profit"],
            rows=[[50000]],
            row_count=1,
            anomalies=[],
        )

        res = execute_adaptive_hybrid(
            question_record=q,
            duckdb_store=self.mock_store,
            sql_executor=self.mock_sql_executor,
            schema_card=self.schema_card,
            schema_meta=self.schema_meta,
            rulebook_text=self.rulebook_text,
            agent=agent,
        )

        self.assertEqual(res["route_trace"]["final_stage"], "ADJUDICATOR")
        self.assertEqual(res["submission_record"]["type"], "abstain")
        self.assertNotIn("answer", res["submission_record"])

    def test_unanswerable_with_evidence(self):
        q = {
            "id": "q6",
            "question": "Tính chi phí kho bãi theo từng lô nhập?",
            "as_of_date": "2024-05-01",
            "type": "unanswerable",
        }
        py_resp = (
            "<think>Inspecting schema for warehouse storage cost columns</think>\n"
            "```python\n"
            'FINAL_ANSWER = {"type": "unanswerable", "reason": "Thiếu dữ liệu chi phí kho bãi"}\n'
            "```"
        )
        agent = MockAgent([py_resp])

        res = execute_adaptive_hybrid(
            question_record=q,
            duckdb_store=self.mock_store,
            sql_executor=self.mock_sql_executor,
            schema_card=self.schema_card,
            schema_meta=self.schema_meta,
            rulebook_text=self.rulebook_text,
            time_pressure=True,
            agent=agent,
        )

        self.assertEqual(res["submission_record"]["type"], "unanswerable")
        self.assertEqual(res["submission_record"]["reason"], "Thiếu dữ liệu chi phí kho bãi")
        self.assertEqual(
            res["log_record"]["thinking"],
            ["Inspecting schema for warehouse storage cost columns"],
        )
