"""Unit tests for Adaptive Hybrid prompts and structured response parsing."""

import unittest

from src.executor import execute_local
from src.prompt_builder import (
    PromptBuilder,
    build_adjudicator_prompt,
    build_business_prefix,
    build_python_prompt,
    build_sql_prompt,
    build_sql_repair_prompt,
    parse_json_response,
    parse_python_response,
    parse_sql_response,
)


class TestSharedBusinessPrefix(unittest.TestCase):
    def test_two_questions_share_byte_identical_prefix(self) -> None:
        rulebook = "Đơn hợp lệ là COMPLETED.\n" + "Quy tắc đầy đủ " * 3000
        readme = "Doanh nghiệp bán lẻ tiếng Việt"
        schema = "TABLE orders rows=10\nCOL amount DOUBLE"
        builder = PromptBuilder(rulebook_content=rulebook)
        prefix_for_first = builder.build_system_prompt(
            business_id="retail_01",
            readme_content=readme,
            schema_context=schema,
        )
        first_user = builder.build_sql_prompt("Tổng doanh thu?", "2024-06-15")
        prefix_for_second = builder.build_system_prompt(
            business_id="retail_01",
            readme_content=readme,
            schema_context=schema,
        )
        second_user = builder.build_python_prompt(
            "Lợi nhuận là bao nhiêu?",
            "2024-06-15",
            "retail_01",
            dual_path=True,
        )
        self.assertEqual(prefix_for_first.encode("utf-8"), prefix_for_second.encode("utf-8"))
        self.assertIs(prefix_for_first, prefix_for_second)
        self.assertNotEqual(first_user, second_user)
        self.assertNotIn("Tổng doanh thu?", prefix_for_first)
        self.assertNotIn("2024-06-15", prefix_for_first)
        self.assertIn(rulebook, prefix_for_first)
        self.assertEqual(prefix_for_first, build_business_prefix(rulebook, readme, schema))

    def test_prefix_section_order_and_full_content(self) -> None:
        rulebook = "RULEBOOK-FULL\nline-2\n"
        prefix = build_business_prefix(rulebook, "README-FULL", "SCHEMA-CARD")
        self.assertLess(prefix.index("OUTPUT CONTRACT"), prefix.index("RULEBOOK.MD (ACTIVE SPLIT)"))
        self.assertLess(prefix.index("RULEBOOK.MD (ACTIVE SPLIT)"), prefix.index("BUSINESS README.MD"))
        self.assertLess(prefix.index("BUSINESS README.MD"), prefix.index("SQL SCHEMA CARD"))
        self.assertIn(rulebook, prefix)
        self.assertTrue(prefix.endswith("SCHEMA-CARD"))


class TestHybridModePrompts(unittest.TestCase):
    def test_sql_mode_contains_required_contract(self) -> None:
        prompt = build_sql_prompt("Top 5 sản phẩm dịp Tết", "2024-06-15")
        for value in (
            "MODE: SQL_FAST",
            "_calendar",
            "vn_unaccent",
            "vn_norm",
            "DISTINCT",
            "ORDER BY metric DESC, code ASC",
            "nhân 100",
            "is_unanswerable",
            "```sql```",
            "```json```",
        ):
            self.assertIn(value, prompt)
        self.assertLess(prompt.index("_calendar"), prompt.index("QUESTION:"))

    def test_sql_repair_is_short_and_contains_failure(self) -> None:
        prompt = build_sql_repair_prompt("SELECT missing FROM orders", "Binder Error: missing")
        self.assertIn("MODE: SQL_REPAIR_ONCE", prompt)
        self.assertIn("SELECT missing FROM orders", prompt)
        self.assertIn("Binder Error: missing", prompt)
        self.assertIn("một block ```sql```", prompt)
        self.assertLess(len(prompt), 2400)

    def test_python_dual_path_is_independent(self) -> None:
        prompt = build_python_prompt(
            "Tính tồn kho lũy kế",
            "2024-06-15",
            "warehouse_01",
            dual_path=True,
            execution_mode="HYBRID_DUAL",
        )
        for value in (
            "MODE: HYBRID_DUAL",
            "DUCKDB_CONN",
            "/tmp/datamate_duckdb/warehouse_01.duckdb",
            "RAW_DATA_DIR",
            "assert",
            "FINAL_ANSWER",
            "Không được xem",
        ):
            self.assertIn(value, prompt)
        self.assertNotIn("SQL_BRANCH_QUERY", prompt)
        self.assertNotIn("SQL_BRANCH_RESULT", prompt)
        self.assertLess(prompt.index("DUCKDB_CONN"), prompt.index("QUESTION:"))

    def test_python_final_answer_contract_is_executable(self) -> None:
        success, result, error = execute_local(
            'FINAL_ANSWER = {"type": "number", "answer": 42}',
            data_dir=".",
            use_subprocess=False,
        )
        self.assertTrue(success, error)
        self.assertEqual(result, {"type": "number", "answer": 42})

    def test_adjudicator_contains_both_branches_and_safe_conclusions(self) -> None:
        prompt = build_adjudicator_prompt(
            "Lợi nhuận tháng 5?",
            "2024-06-15",
            "SELECT 100 AS profit",
            {"columns": ["profit"], "rows": [[100]]},
            "FINAL_ANSWER={'type':'number','answer':90}",
            "checked 10 rows",
            {"type": "number", "answer": 90},
        )
        for value in (
            "ADJUDICATOR_CONFLICT_ONLY",
            "SELECT 100 AS profit",
            "checked 10 rows",
            "VAT",
            "voucher",
            "verified_answer",
            "unanswerable",
            "abstain",
            "phạt -1",
        ):
            self.assertIn(value, prompt)
        self.assertLess(prompt.index("VAT"), prompt.index("SQL_BRANCH_QUERY"))


class TestStructuredResponseParsers(unittest.TestCase):
    def test_parse_sql_and_metadata_after_think(self) -> None:
        response = """<think>Phân tích nội bộ với ```sql\nDROP TABLE x\n```.</think>
```sql
SELECT ROUND(SUM(amount), 2) AS answer FROM orders
```
```json
{"expected_type":"number","post_process":"none","is_unanswerable":false,"reason":null}
```"""
        sql, metadata = parse_sql_response(response)
        self.assertEqual(sql, "SELECT ROUND(SUM(amount), 2) AS answer FROM orders")
        self.assertEqual(metadata["expected_type"], "number")
        self.assertFalse(metadata["is_unanswerable"])

    def test_parse_python_after_think(self) -> None:
        response = """<think>Không tiết lộ suy luận.</think>
```python
assert len(df) > 0
FINAL_ANSWER = {"type": "number", "answer": 42}
```"""
        code = parse_python_response(response)
        self.assertIn("assert len(df) > 0", code)
        self.assertIn("FINAL_ANSWER", code)

    def test_parse_adjudicator_json_after_optional_python(self) -> None:
        response = """<think>Đối chiếu.</think>
```python
check = 1
```
```json
{"decision":"abstain","type":"abstain","reason":"Hai nhánh còn xung đột"}
```"""
        parsed = parse_json_response(response)
        self.assertEqual(parsed["decision"], "abstain")

    def test_parsers_reject_ambiguous_or_invalid_blocks(self) -> None:
        with self.assertRaises(ValueError):
            parse_python_response("```python\nx=1\n```\n```python\ny=2\n```")
        with self.assertRaises(ValueError):
            parse_sql_response(
                "```sql\nSELECT 1\n```\n"
                "```json\n{\"expected_type\":\"bad\",\"post_process\":\"none\",\"is_unanswerable\":false,\"reason\":null}\n```"
            )
        with self.assertRaises(ValueError):
            parse_sql_response(
                "```sql\nSELECT 1\n```\n"
                "```json\n{\"expected_type\":\"unanswerable\",\"post_process\":\"none\",\"is_unanswerable\":true,\"reason\":null}\n```"
            )
        with self.assertRaises(ValueError):
            parse_json_response(
                "```json\n{\"decision\":\"guess\",\"type\":\"number\",\"answer\":1}\n```"
            )


if __name__ == "__main__":
    unittest.main()
