"""Unit tests for deterministic Adaptive Hybrid routing."""

import time
import unittest

from src.router import RouteDecision, classify_question_route, should_escalate_sql_result


class TestQuestionRouting(unittest.TestCase):
    def setUp(self) -> None:
        self.clean_schema = {
            "tables": {
                "orders": {
                    "columns": [
                        "order_id",
                        "created_at",
                        "customer_id",
                        "branch",
                        "amount",
                        "status",
                        "product_id",
                        "quantity",
                        "unit_cost",
                    ],
                    "etl_warnings": [],
                }
            }
        }
        self.rulebook = "Doanh thu theo đơn hợp lệ. Lợi nhuận bằng doanh thu trừ giá vốn."

    def classify(self, question: str, schema_meta: dict = None, time_pressure: bool = False):
        return classify_question_route(
            question,
            "2024-06-15",
            self.clean_schema if schema_meta is None else schema_meta,
            self.rulebook,
            time_pressure=time_pressure,
        )

    def test_enum_values_are_stable(self) -> None:
        self.assertEqual(RouteDecision.SQL_FAST.value, "SQL_FAST")
        self.assertEqual(RouteDecision.HYBRID_DUAL.value, "HYBRID_DUAL")
        self.assertEqual(RouteDecision.PYTHON_FIRST.value, "PYTHON_FIRST")

    def test_clean_simple_aggregations_and_top_k_use_sql_fast(self) -> None:
        questions = [
            "Có bao nhiêu đơn hàng thành công?",
            "Tổng doanh thu tháng 5 là bao nhiêu?",
            "Tim min va max so luong ban.",
            "Top 5 sản phẩm theo doanh thu",
            "SL đơn hàng của chi nhánh Hà Nội",
        ]
        for question in questions:
            with self.subTest(question=question):
                decision, reasons = self.classify(question)
                self.assertEqual(decision, RouteDecision.SQL_FAST)
                self.assertEqual(reasons, ["clean_schema_sql_candidate"])

    def test_sequential_business_logic_uses_python_first(self) -> None:
        questions = [
            "Tính giá vốn bình quân gia quyền theo từng lô nhập xuất",
            "Tinh ton kho luy ke theo ngay",
            "Xuất kho theo FIFO cho từng sản phẩm",
            "Phân bổ chi phí vận chuyển phức tạp cho từng lô",
            "Calculate running inventory balance",
        ]
        for question in questions:
            with self.subTest(question=question):
                decision, reasons = self.classify(question)
                self.assertEqual(decision, RouteDecision.PYTHON_FIRST)
                self.assertTrue(reasons[0].startswith("sequential_business_logic:"))

    def test_severe_etl_warnings_use_python_first(self) -> None:
        warning_groups = [
            ["complex_merged_cells"],
            ["horizontal_pivot_suspected"],
            ["high_type_parse_errors:amount:55.0%"],
            ["partial_type_parse:amount:80.0%", "partial_type_parse:date:82.0%"],
        ]
        for warnings in warning_groups:
            schema = {"tables": [{"columns": ["amount"], "etl_warnings": warnings}]}
            with self.subTest(warnings=warnings):
                decision, reasons = self.classify("Tổng doanh thu", schema)
                self.assertEqual(decision, RouteDecision.PYTHON_FIRST)
                self.assertTrue(all(reason.startswith("severe_etl_warning:") for reason in reasons))

    def test_business_calendar_and_complex_outputs_use_hybrid(self) -> None:
        questions = [
            "Lợi nhuận và biên lợi nhuận của từng chi nhánh",
            "Tinh GVHB va lai gop thang nay",
            "LN theo từng chi nhánh",
            "Tỷ lệ hoàn trả so với tháng trước là bao nhiêu phần trăm?",
            "Công nợ quá hạn của khách hàng",
            "Doanh thu dịp Tết âm lịch và cùng kỳ năm ngoái",
            "Doanh thu từ mùng 1 đến Trung thu",
            "Lập bảng kết quả doanh thu theo chi nhánh",
            "Trả về bảng doanh thu theo tháng",
            "Top 5 sản phẩm trong số đơn hoàn thành, đồng thời chỉ những đơn tại Hà Nội",
            "Top 5 sản phẩm theo doanh thu và số lượng",
        ]
        for question in questions:
            with self.subTest(question=question):
                decision, reasons = self.classify(question)
                self.assertEqual(decision, RouteDecision.HYBRID_DUAL)
                self.assertTrue(reasons)

    def test_missing_schema_concepts_force_hybrid(self) -> None:
        schema = {"tables": {"orders": {"columns": ["order_id", "amount"]}}}
        cases = {
            "Khách hàng nào mua nhiều nhất?": "schema_missing:customer",
            "Nhân viên nào bán tốt nhất?": "schema_missing:employee",
            "Chi nhánh nào có doanh thu cao nhất?": "schema_missing:branch",
            "Lợi nhuận là bao nhiêu?": "schema_missing:profit_inputs",
            "Giá vốn là bao nhiêu?": "schema_missing:cost",
        }
        for question, expected_reason in cases.items():
            with self.subTest(question=question):
                decision, reasons = self.classify(question, schema)
                self.assertEqual(decision, RouteDecision.HYBRID_DUAL)
                self.assertIn(expected_reason, reasons)

    def test_id_suffixed_columns_are_recognized(self) -> None:
        decision, reasons = self.classify("Khách hàng nào mua nhiều nhất?")
        self.assertNotIn("schema_missing:customer", reasons)
        self.assertEqual(decision, RouteDecision.SQL_FAST)

    def test_time_pressure_downgrades_only_noncritical_hybrid_rules(self) -> None:
        decision, reasons = self.classify("Doanh thu dịp Trung thu so với tháng trước", time_pressure=True)
        self.assertEqual(decision, RouteDecision.SQL_FAST)
        self.assertIn("time_pressure_downgrade_to_sql_fast", reasons)
        decision, reasons = self.classify("Lợi nhuận tháng này", time_pressure=True)
        self.assertEqual(decision, RouteDecision.HYBRID_DUAL)
        self.assertIn("time_pressure_retained_high_penalty_risk", reasons)
        missing_schema = {"tables": {"orders": {"columns": ["order_id", "amount"]}}}
        decision, reasons = self.classify("Khách hàng nào mua nhiều nhất?", missing_schema, True)
        self.assertEqual(decision, RouteDecision.HYBRID_DUAL)
        self.assertIn("schema_missing:customer", reasons)
        mild_warning = {"tables": {"orders": {"columns": ["amount"], "etl_warnings": ["shifted_header_detected"]}}}
        decision, reasons = self.classify("Tổng doanh thu", mild_warning, True)
        self.assertEqual(decision, RouteDecision.HYBRID_DUAL)
        self.assertIn("etl_warning_present", reasons)

    def test_missing_as_of_date_escalates_relative_time(self) -> None:
        decision, reasons = classify_question_route(
            "Doanh thu tháng này",
            "",
            self.clean_schema,
            self.rulebook,
        )
        self.assertEqual(decision, RouteDecision.HYBRID_DUAL)
        self.assertIn("missing_as_of_date", reasons)

    def test_router_is_deterministic_and_fast(self) -> None:
        arguments = (
            "Top 5 sản phẩm theo doanh thu",
            "2024-06-15",
            self.clean_schema,
            self.rulebook,
        )
        expected = classify_question_route(*arguments)
        started = time.perf_counter()
        results = [classify_question_route(*arguments) for _ in range(1000)]
        elapsed = time.perf_counter() - started
        self.assertTrue(all(result == expected for result in results))
        self.assertLess(elapsed, 2.0)


class TestSQLResultEscalation(unittest.TestCase):
    def test_sql_failure_escalates(self) -> None:
        escalated, reasons = should_escalate_sql_result(
            {"success": False, "error": "Binder Error", "repair_attempted": True},
            "Tổng doanh thu",
        )
        self.assertTrue(escalated)
        self.assertEqual(reasons, ["sql_execution_failed_after_repair"])

    def test_each_sql_anomaly_escalates(self) -> None:
        for anomaly in ("empty_result", "all_null", "zero_scalar", "suspicious_extreme", "truncated_rows"):
            with self.subTest(anomaly=anomaly):
                escalated, reasons = should_escalate_sql_result(
                    {"success": True, "anomalies": [anomaly]},
                    "Tổng doanh thu",
                )
                self.assertTrue(escalated)
                self.assertIn(f"sql_anomaly:{anomaly}", reasons)

    def test_zero_scalar_boolean_exception(self) -> None:
        escalated, reasons = should_escalate_sql_result(
            {"success": True, "anomalies": ["zero_scalar"], "question_type": "boolean"},
            "Có đơn hàng bị hủy hay không?",
        )
        self.assertFalse(escalated)
        self.assertEqual(reasons, [])
        escalated, reasons = should_escalate_sql_result(
            {"success": True, "anomalies": ["zero_scalar"]},
            "Co don hang bi huy khong?",
        )
        self.assertFalse(escalated)

    def test_abstain_and_unanswerable_always_escalate(self) -> None:
        outcomes = [
            {"success": True, "type": "unanswerable"},
            {"success": True, "answer": "abstain"},
            {"success": True, "model_output": {"status": "unanswerable"}},
        ]
        for outcome in outcomes:
            with self.subTest(outcome=outcome):
                escalated, reasons = should_escalate_sql_result(outcome, "Lợi nhuận là bao nhiêu?")
                self.assertTrue(escalated)
                self.assertIn("sql_branch_abstain_or_unanswerable", reasons)

    def test_percent_fraction_escalates_but_normal_percent_does_not(self) -> None:
        escalated, reasons = should_escalate_sql_result(
            {"success": True, "columns": ["growth_percent"], "rows": [[0.15]]},
            "Tăng trưởng phần trăm là bao nhiêu?",
        )
        self.assertTrue(escalated)
        self.assertIn("percent_scale_suspected", reasons)
        escalated, reasons = should_escalate_sql_result(
            {"success": True, "columns": ["growth_percent"], "rows": [[15.0]]},
            "Tang truong % la bao nhieu?",
        )
        self.assertFalse(escalated)

    def test_ranking_requires_order_and_id_tiebreaker(self) -> None:
        missing_order = {
            "success": True,
            "sql": "SELECT product_id, SUM(amount) AS total FROM orders GROUP BY product_id LIMIT 3",
            "columns": ["product_id", "total"],
            "rows": [["A", 30], ["B", 20], ["C", 10]],
        }
        escalated, reasons = should_escalate_sql_result(missing_order, "Top 3 sản phẩm")
        self.assertTrue(escalated)
        self.assertIn("ranking_missing_order_by", reasons)
        missing_tiebreak = dict(missing_order)
        missing_tiebreak["sql"] = "SELECT product_id, SUM(amount) AS total FROM orders GROUP BY product_id ORDER BY total DESC LIMIT 3"
        escalated, reasons = should_escalate_sql_result(missing_tiebreak, "Top 3 sản phẩm")
        self.assertTrue(escalated)
        self.assertIn("ranking_missing_id_tiebreaker", reasons)
        clean = dict(missing_order)
        clean["sql"] = "SELECT product_id, SUM(amount) AS total FROM orders GROUP BY product_id ORDER BY total DESC, product_id ASC LIMIT 3"
        escalated, reasons = should_escalate_sql_result(clean, "Top 3 sản phẩm")
        self.assertFalse(escalated)
        self.assertEqual(reasons, [])

    def test_ranking_boundary_tie_escalates(self) -> None:
        outcome = {
            "success": True,
            "sql": "SELECT product_id, total FROM ranked ORDER BY total DESC, product_id ASC LIMIT 3",
            "columns": ["product_id", "total"],
            "rows": [["A", 30], ["B", 10], ["C", 10]],
        }
        escalated, reasons = should_escalate_sql_result(outcome, "Top 3 sản phẩm")
        self.assertTrue(escalated)
        self.assertIn("ranking_boundary_tie_suspected", reasons)


if __name__ == "__main__":
    unittest.main()
