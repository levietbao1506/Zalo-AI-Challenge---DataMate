from __future__ import annotations

import unittest
from unittest.mock import MagicMock

from src import formatter, prompt_builder as pb, router
from src.agent import _is_safe_readme_direct, execute_adaptive_hybrid
from src.sql_executor import SQLExecutionResult


class MockAgent:
    def __init__(self, responses: list[str]) -> None:
        self.responses = list(responses)
        self.calls: list[str] = []
        self.last_thinking: str = "Test thinking trace"

    def generate_response(self, prompt: str, system_prompt: str | None = None) -> str:
        self.calls.append(prompt)
        if self.responses:
            return self.responses.pop(0)
        return (
            "```sql\nSELECT 1 AS answer\n```\n"
            "```json\n"
            '{"expected_type":"number","post_process":"none","answer_source":"sql","direct_answer":null,"is_unanswerable":false,"reason":null}\n'
            "```"
        )

    def generate_code(self, prompt: str, system_prompt: str | None = None) -> str:
        return self.generate_response(prompt, system_prompt)


class TestUnanswerableReadmeAndEdgeCases(unittest.TestCase):
    def test_parse_sql_response_default_answer_source_backwards_compatible(self) -> None:
        raw_llm_response = (
            "```sql\nSELECT count(*) AS total FROM orders\n```\n"
            "```json\n"
            '{"expected_type":"number","post_process":"none","is_unanswerable":false,"reason":null}\n'
            "```"
        )
        sql, meta = pb.parse_sql_response(raw_llm_response)
        self.assertEqual(sql, "SELECT count(*) AS total FROM orders")
        self.assertEqual(meta["answer_source"], "sql")
        self.assertIsNone(meta["direct_answer"])
        self.assertFalse(meta["is_unanswerable"])

    def test_parse_sql_response_readme_source_without_sql_block(self) -> None:
        raw_llm_response = (
            "<think>Direct descriptive lookup from README</think>\n"
            "```json\n"
            '{"expected_type":"string","post_process":"none","answer_source":"readme","direct_answer":"08:00 - 22:00","is_unanswerable":false,"reason":null}\n'
            "```"
        )
        sql, meta = pb.parse_sql_response(raw_llm_response)
        self.assertEqual(sql, "")
        self.assertEqual(meta["answer_source"], "readme")
        self.assertEqual(meta["direct_answer"], "08:00 - 22:00")
        self.assertEqual(meta["expected_type"], "string")

    def test_is_safe_readme_direct_guard(self) -> None:
        long_readme = "Cửa hàng trà sữa DataMate mở cửa từ 08:00 đến 22:00 hàng ngày tại 123 Đường Láng, Hà Nội."

        self.assertTrue(_is_safe_readme_direct("Giờ mở cửa của quán là mấy giờ?", "string", long_readme))
        self.assertTrue(_is_safe_readme_direct("Địa chỉ cửa hàng ở đâu?", "string", long_readme))

        self.assertFalse(_is_safe_readme_direct("Giờ mở cửa là mấy giờ?", "string", ""))
        self.assertFalse(_is_safe_readme_direct("Giờ mở cửa là mấy giờ?", "string", "Ngắn quá"))

        forbidden_questions = [
            "Doanh thu tháng này là bao nhiêu?",
            "Tổng số lượng sản phẩm đã bán?",
            "Lợi nhuận gộp trong quý 1 là bao nhiêu?",
            "Top 5 mặt hàng bán chạy nhất?",
            "Mặt hàng nào có số lượng ít nhất?",
            "Tỷ lệ tăng trưởng so với cùng kỳ?",
            "Doanh thu trung bình theo tuần rồi?",
            "Số lượng tồn kho hiện tại?",
            "Công nợ còn lại của khách hàng?",
            "Có bao nhiêu hóa đơn được lập?",
        ]
        for fq in forbidden_questions:
            self.assertFalse(_is_safe_readme_direct(fq, "string", long_readme), f"Failed for {fq}")

        self.assertFalse(_is_safe_readme_direct("Danh sách địa chỉ các chi nhánh?", "table", long_readme))
        self.assertFalse(_is_safe_readme_direct("Các chi nhánh của quán?", "list", long_readme))

    def test_readme_direct_execution_in_agent(self) -> None:
        question = {
            "id": "q_readme_01",
            "question": "Giờ mở cửa của cửa hàng là khi nào?",
            "as_of_date": "2024-05-01",
            "type": "string",
        }
        readme_text = "Cửa hàng mở cửa từ 08:00 đến 22:00 tất cả các ngày trong tuần."
        llm_resp = (
            "<think>Thông tin giờ mở cửa có sẵn trong README.md</think>\n"
            "```json\n"
            '{"expected_type":"string","post_process":"none","answer_source":"readme","direct_answer":"08:00 - 22:00","is_unanswerable":false,"reason":null}\n'
            "```"
        )
        agent = MockAgent([llm_resp])
        mock_store = MagicMock()
        mock_store.business_id = "biz_test"
        mock_store.business_dir = "."
        mock_executor = MagicMock()

        result = execute_adaptive_hybrid(
            question_record=question,
            duckdb_store=mock_store,
            sql_executor=mock_executor,
            schema_card="## Schema",
            schema_meta={},
            rulebook_text="Rulebook content",
            agent=agent,
            readme_text=readme_text,
        )

        sub_rec = result["submission_record"]
        log_rec = result["log_record"]
        route_tr = result["route_trace"]

        self.assertEqual(sub_rec["id"], "q_readme_01")
        self.assertEqual(sub_rec["type"], "string")
        self.assertEqual(sub_rec["answer"], "08:00 - 22:00")

        self.assertEqual(route_tr["final_stage"], "readme_direct")
        self.assertIsNone(log_rec["code"])
        self.assertIsNone(log_rec["output"])
        self.assertIsNotNone(log_rec["thinking"])

    def test_unsafe_readme_direct_rejected_and_escalated(self) -> None:
        question = {
            "id": "q_unsafe_readme",
            "question": "Doanh thu tháng này là bao nhiêu?",
            "as_of_date": "2024-05-01",
            "type": "number",
        }
        readme_text = "Doanh nghiệp thương mại DataMate chuyên cung cấp vật liệu xây dựng."
        fake_readme_resp = (
            "```json\n"
            '{"expected_type":"number","post_process":"none","answer_source":"readme","direct_answer":1000000,"is_unanswerable":false,"reason":null}\n'
            "```"
        )
        fallback_sql_resp = (
            "```sql\nSELECT 5000000 AS answer\n```\n"
            "```json\n"
            '{"expected_type":"number","post_process":"none","answer_source":"sql","direct_answer":null,"is_unanswerable":false,"reason":null}\n'
            "```"
        )
        agent = MockAgent([fake_readme_resp, fallback_sql_resp])
        mock_store = MagicMock()
        mock_store.business_id = "biz_test"
        mock_store.business_dir = "."
        mock_executor = MagicMock()
        mock_executor.execute.return_value = SQLExecutionResult(
            success=True,
            sql="SELECT 5000000 AS answer",
            columns=["answer"],
            rows=[[5000000]],
            row_count=1,
            anomalies=[],
        )

        result = execute_adaptive_hybrid(
            question_record=question,
            duckdb_store=mock_store,
            sql_executor=mock_executor,
            schema_card="## Schema",
            schema_meta={},
            rulebook_text="Rulebook content",
            agent=agent,
            readme_text=readme_text,
        )

        self.assertIn("unsafe_readme_direct_rejected", result["route_trace"]["escalation_reasons"])
        self.assertNotEqual(result["route_trace"]["final_stage"], "readme_direct")

    def test_normalize_weekday_answer_all_variants(self) -> None:
        self.assertEqual(formatter.normalize_weekday_answer(1), "Thứ 2")
        self.assertEqual(formatter.normalize_weekday_answer(6), "Thứ 7")
        self.assertEqual(formatter.normalize_weekday_answer(0), "Chủ nhật")
        self.assertEqual(formatter.normalize_weekday_answer(7), "Chủ nhật")

        self.assertEqual(formatter.normalize_weekday_answer("Monday"), "Thứ 2")
        self.assertEqual(formatter.normalize_weekday_answer("Sunday"), "Chủ nhật")
        self.assertEqual(formatter.normalize_weekday_answer("thu hai"), "Thứ 2")
        self.assertEqual(formatter.normalize_weekday_answer("t3"), "Thứ 3")
        self.assertEqual(formatter.normalize_weekday_answer("cn"), "Chủ nhật")

        rulebook_words = "Các ngày trong tuần tính từ Thứ Hai đến Chủ Nhật."
        self.assertEqual(formatter.normalize_weekday_answer(1, rulebook_words), "Thứ Hai")
        self.assertEqual(formatter.normalize_weekday_answer("t4", rulebook_words), "Thứ Tư")
        self.assertEqual(formatter.normalize_weekday_answer(0, rulebook_words), "Chủ Nhật")

    def test_format_submission_record_with_question_context(self) -> None:
        rec_month = formatter.format_submission_record(
            q_id="m1",
            answer="2025-09-15",
            question="Doanh số lớn nhất rơi vào tháng nào?",
        )
        self.assertEqual(rec_month["type"], "date")
        self.assertEqual(rec_month["answer"], "2025-09")

        rec_slash = formatter.format_submission_record(
            q_id="m2",
            answer="09/2025",
            question="Doanh thu cao nhất tháng mấy?",
        )
        self.assertEqual(rec_slash["type"], "date")
        self.assertEqual(rec_slash["answer"], "2025-09")

        rec_weekday = formatter.format_submission_record(
            q_id="w1",
            answer=1,
            question="Thứ mấy trong tuần bán chạy nhất?",
            rulebook_text="Thứ 2 đến Chủ nhật",
        )
        self.assertEqual(rec_weekday["type"], "string")
        self.assertEqual(rec_weekday["answer"], "Thứ 2")

    def test_router_flags_check_date_coverage_in_python(self) -> None:
        sql_outcome_unans = {
            "success": True,
            "sql": "SELECT 1",
            "columns": ["col"],
            "rows": [],
            "anomalies": ["empty_result"],
            "is_unanswerable": True,
        }
        should_esc, reasons = router.should_escalate_sql_result(
            sql_outcome=sql_outcome_unans,
            question="Doanh thu trong tháng 9 năm ngoái là bao nhiêu?",
        )
        self.assertTrue(should_esc)
        self.assertIn("check_date_coverage_in_python", reasons)

    def test_sort_ranking_items_less_than_k(self) -> None:
        candidates = [("SP002", 500), ("SP001", 1000)]
        sorted_items = formatter.sort_ranking_items(candidates)
        self.assertEqual(sorted_items, [("SP001", 1000), ("SP002", 500)])

        ties = [("SP003", 500), ("SP002", 500)]
        sorted_ties = formatter.sort_ranking_items(ties)
        self.assertEqual(sorted_ties, [("SP002", 500), ("SP003", 500)])


if __name__ == "__main__":
    unittest.main()
