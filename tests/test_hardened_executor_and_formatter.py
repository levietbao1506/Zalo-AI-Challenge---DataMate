"""Unit tests for hardened executor and strict competition evaluation formatter (Task 10)."""

from datetime import date, datetime
import json
from pathlib import Path
import sqlite3
import tempfile
import unicodedata
import unittest

import pandas as pd

from src.executor import (
    EMPTY_RESULT_FEEDBACK,
    ExecutionResult,
    SandboxedExecutor,
    SubprocessResult,
    execute_local,
    execute_subprocess,
    execute_with_retry,
    fuzzy_match_name,
    normalize_vietnamese,
    parse_mixed_date,
    parse_vnd_amount,
    read_csv_safe,
    read_excel_safe,
    read_sqlite_safe,
)
from src.formatter import (
    DEFAULT_ABSTAIN_TOKEN,
    clean_string,
    format_boolean,
    format_date,
    format_list,
    format_number,
    format_output,
    format_percent,
    format_scalar,
    format_string,
    format_submission_json,
    format_submission_record,
    format_table,
    normalize_for_comparison,
    normalize_unicode_nfc,
    sort_ranking_items,
)
from src.prompt_builder import PromptBuilder


class TestHardenedExecutor(unittest.TestCase):
    """Test suite for isolated subprocess execution and self-correction retry feedback."""

    def setUp(self):
        self.scratch_dir = tempfile.TemporaryDirectory()
        self.data_dir = self.scratch_dir.name

    def tearDown(self):
        self.scratch_dir.cleanup()

    def test_packaged_vietnamese_helpers(self):
        self.assertEqual(normalize_vietnamese("Cà Phê Đá"), "ca phe da")
        self.assertEqual(fuzzy_match_name("ca phe da", ["Cà Phê Đá", "Trà Sữa"]), "Cà Phê Đá")
        self.assertEqual(parse_vnd_amount("1.500.000 đ"), 1500000)
        self.assertEqual(parse_vnd_amount("1,500,000 VND"), 1500000)
        self.assertEqual(parse_vnd_amount("500k"), 500000)
        self.assertEqual(str(parse_mixed_date("31/12/2024")), "2024-12-31")
        self.assertEqual(str(parse_mixed_date(45292)), "2024-01-01")

    def test_packaged_readers(self):
        csv_path = Path(self.data_dir) / "orders.csv"
        csv_path.write_text(" Mã ; Số tiền \nA;1.500.000\n", encoding="utf-8-sig")
        self.assertEqual(read_csv_safe(csv_path).columns.tolist(), ["Mã", "Số tiền"])
        sqlite_path = Path(self.data_dir) / "sales.sqlite"
        connection = sqlite3.connect(sqlite_path)
        try:
            connection.execute('CREATE TABLE sales (" Mã " TEXT)')
            connection.execute('INSERT INTO sales VALUES ("A")')
            connection.commit()
        finally:
            connection.close()
        self.assertEqual(read_sqlite_safe(sqlite_path, "sales").columns.tolist(), ["Mã"])
        excel_path = Path(self.data_dir) / "sales.xlsx"
        with pd.ExcelWriter(excel_path) as writer:
            pd.DataFrame({"Mã": ["A"], "Giá": [100]}).to_excel(writer, sheet_name="Dữ liệu", index=False)
        self.assertEqual(read_excel_safe(excel_path, "Dữ liệu").columns.tolist(), ["Mã", "Giá"])

    def test_subprocess_restricts_writes_and_bounds_stdout(self):
        denied = execute_subprocess('open("../escape.txt", "w").write("x")', data_dir=self.data_dir)
        self.assertFalse(denied.success)
        self.assertIn("PermissionError", denied.stderr)
        denied_read = execute_subprocess('open("../outside.txt").read()', data_dir=self.data_dir)
        self.assertFalse(denied_read.success)
        self.assertIn("PermissionError", denied_read.stderr)
        bounded = execute_subprocess('print("A" * 12000)\nfinal_result = 5', data_dir=self.data_dir)
        self.assertTrue(bounded.success, bounded.stderr)
        self.assertEqual(bounded.return_value, 5)
        self.assertLess(len(bounded.stdout), 9000)
        self.assertIn("stdout truncated", bounded.stdout)

    def test_unanswerable_preserves_reason(self):
        code = 'print(FILE_MAP)\nreason = "Thiếu dữ liệu giá vốn"\nfinal_result = "unanswerable"'
        result = execute_subprocess(code, data_dir=self.data_dir)
        self.assertTrue(result.success, result.stderr)
        self.assertEqual(result.return_value, {"type": "unanswerable", "reason": "Thiếu dữ liệu giá vốn"})

    def test_time_context_uses_split_calendar_above_business_dir(self):
        split_root = Path(self.data_dir)
        business_dir = split_root / "data" / "shop"
        business_dir.mkdir(parents=True)
        calendar_path = split_root / "calendar.csv"
        calendar_path.write_text("date,holiday\n2024-02-10,Tet Nguyen Dan\n", encoding="utf-8")
        (split_root / "rulebook.md").write_text("Tuần bắt đầu Chủ nhật.", encoding="utf-8")
        result = execute_subprocess(
            'final_result = time_context(as_of_date, DATA_DIR)["week_start"]',
            data_dir=str(business_dir),
            file_map={"calendar.csv": str(calendar_path)},
            as_of_date=date(2024, 2, 12),
        )
        self.assertTrue(result.success, result.stderr)
        self.assertEqual(result.return_value, "sunday")

    def test_execute_subprocess_basic_success(self):
        """Subprocess execution should run clean code and return structured 4-tuple."""
        code = """
import pandas as pd
df = pd.DataFrame({"a": [10, 20, 30]})
final_result = int(df["a"].sum())
"""
        res = execute_subprocess(code, data_dir=self.data_dir, timeout_seconds=15.0)
        self.assertIsInstance(res, SubprocessResult)
        success, return_val, stdout, stderr = res
        self.assertTrue(success)
        self.assertEqual(return_val, 60)
        self.assertEqual(stderr, "")
        self.assertEqual(res.return_value, 60)

    def test_execute_subprocess_preimported_helpers(self):
        """Pre-imported pandas, numpy, sqlite3, openpyxl, calendar_utils should be immediately available."""
        code = """
# sqlite3, math, pd, np, calendar_utils, resolve_as_of_date are all pre-imported
conn = sqlite3.connect(":memory:")
cur = conn.cursor()
cur.execute("CREATE TABLE t (val INT);")
cur.execute("INSERT INTO t VALUES (42);")
cur.execute("SELECT val FROM t;")
row = cur.fetchone()[0]
conn.close()

# Calendar helper
ref = resolve_as_of_date(as_of_date)
final_result = {"row": row, "ref_year": ref.year}
"""
        res = execute_subprocess(
            code,
            data_dir=self.data_dir,
            as_of_date=date(2024, 8, 15),
            timeout_seconds=15.0,
        )
        self.assertTrue(res.success)
        self.assertEqual(res.return_value, {"row": 42, "ref_year": 2024})

    def test_execute_subprocess_file_map_aliases(self):
        """File aliases in FILE_MAP should be injected directly as variables in subprocess."""
        fake_csv = Path(self.data_dir) / "orders.csv"
        fake_csv.write_text("order_id,amount\n1,100\n2,200\n", encoding="utf-8")
        file_map = {"orders.csv": str(fake_csv)}

        code = """
import pandas as pd
df = pd.read_csv(orders_csv)
final_result = int(df["amount"].sum())
"""
        res = execute_subprocess(
            code,
            data_dir=self.data_dir,
            file_map=file_map,
            timeout_seconds=15.0,
        )
        self.assertTrue(res.success)
        self.assertEqual(res.return_value, 300)

    def test_execute_subprocess_uppercase_final_result(self):
        """FINAL_RESULT, RESULT, and final_answer variable names must be properly extracted."""
        code_caps = """
import pandas as pd
df = pd.DataFrame({"rev": [100, 200, 300]})
FINAL_RESULT = int(df["rev"].sum())
"""
        res = execute_subprocess(code_caps, data_dir=self.data_dir, timeout_seconds=10.0)
        self.assertTrue(res.success)
        self.assertEqual(res.return_value, 600)

        # Local execution check as well
        succ, local_val, err = execute_local(code_caps, data_dir=self.data_dir)
        self.assertTrue(succ)
        self.assertEqual(local_val, 600)

    def test_execute_subprocess_traceback_on_error(self):
        """Syntax, KeyError, and TypeError exceptions should capture exact traceback on stderr."""
        # KeyError
        code_key = "d = {'x': 1}\nfinal_result = d['missing_key']"
        res_key = execute_subprocess(code_key, data_dir=self.data_dir, timeout_seconds=10.0)
        self.assertFalse(res_key.success)
        self.assertIsNone(res_key.return_value)
        self.assertIn("KeyError", res_key.stderr)
        self.assertIn("missing_key", res_key.stderr)
        self.assertIn("Traceback (most recent call last):", res_key.stderr)

        # SyntaxError
        code_syn = "def broken(:"
        res_syn = execute_subprocess(code_syn, data_dir=self.data_dir, timeout_seconds=10.0)
        self.assertFalse(res_syn.success)
        self.assertIn("SyntaxError", res_syn.stderr)

    def test_execute_subprocess_timeout(self):
        """Infinite loop or long execution should terminate cleanly within timeout."""
        code_inf = """
import time
while True:
    time.sleep(0.05)
"""
        res = execute_subprocess(code_inf, data_dir=self.data_dir, timeout_seconds=0.6)
        self.assertFalse(res.success)
        self.assertIsNone(res.return_value)
        self.assertIn("TimeoutError", res.stderr)

    def test_sandboxed_executor_subprocess_method(self):
        """SandboxedExecutor.execute_subprocess should wrap execute_subprocess returning ExecutionResult."""
        executor = SandboxedExecutor(default_timeout_seconds=25.0)
        self.assertEqual(executor.default_timeout_seconds, 25.0)

        code = "final_result = 99 * 2"
        res = executor.execute_subprocess(code, data_dir=self.data_dir)
        self.assertIsInstance(res, ExecutionResult)
        self.assertTrue(res.success)
        self.assertEqual(res.result, 198)
        self.assertEqual(res.return_value, 198)

    def test_self_correction_retries_with_traceback(self):
        """Self-correction should pass traceback feedback and succeed on retry attempt."""
        prompts = []
        codes = [
            "```python\nfinal_result = undefined_variable\n```",
            "```python\nfinal_result = 888\n```",
        ]

        def mock_gen(p: str) -> str:
            prompts.append(p)
            return codes[len(prompts) - 1]

        success, result, err = execute_with_retry(
            initial_prompt="Tính tổng số lượng đơn hàng",
            generate_fn=mock_gen,
            data_dir=self.data_dir,
            max_retries=2,
            use_subprocess=True,
        )
        self.assertTrue(success)
        self.assertEqual(result, 888)
        self.assertEqual(len(prompts), 2)
        # Verify traceback feedback was passed in the retry prompt
        self.assertIn("undefined_variable", prompts[1])
        self.assertIn("NameError", prompts[1])
        self.assertIn("Đoạn code trên bị lỗi:", prompts[1])


class TestHardenedFormatter(unittest.TestCase):
    """Test suite for strict competition type rules in src/formatter.py."""

    def test_number_formatting_and_vnd_cleaning(self):
        """Number should be clean int or float rounded to 2 decimals, with VND symbols and commas stripped."""
        # Standard numbers
        self.assertEqual(format_number(42), 42)
        self.assertEqual(format_number(123.456), 123.46)
        self.assertEqual(format_number(123.400), 123.4)
        self.assertEqual(format_number(100.0), 100)

        # VND amounts with commas and currency symbols
        self.assertEqual(format_number("15,000,000 VND"), 15000000)
        self.assertEqual(format_number("15,000,000 VNĐ"), 15000000)
        self.assertEqual(format_number("15,000,000vnđ"), 15000000)
        self.assertEqual(format_number("15.000.000 đ"), 15000000)
        self.assertEqual(format_number("15 000 000"), 15000000)
        self.assertEqual(format_number("$1500"), 1500)
        self.assertEqual(format_number("15,000.50 VNĐ"), 15000.5)
        self.assertEqual(format_number("15.000,50 đ"), 15000.5)

    def test_percent_formatting(self):
        """Percent should be numeric value e.g. -12.45, NOT '-12.45%' and NOT -0.1245."""
        # Percentage string
        self.assertEqual(format_percent("-12.45%"), -12.45)
        self.assertEqual(format_percent("15.5%"), 15.5)
        self.assertEqual(format_percent("20 %"), 20)
        self.assertEqual(format_percent("+8.25%"), 8.25)

        # Ratio floats e.g. -0.1245 -> -12.45
        self.assertEqual(format_percent(-0.1245), -0.12)
        self.assertEqual(format_percent(0.0825), 0.08)
        self.assertEqual(format_percent("-0.1245"), -0.12)

        # Pre-multiplied percentage numbers
        self.assertEqual(format_percent(-12.45), -12.45)
        self.assertEqual(format_percent(15.0), 15)
        self.assertEqual(format_percent(0), 0)

    def test_string_canonical_vs_comparison_normalization(self):
        """Validation helper should normalize strictly, while format_string preserves canonical business text."""
        raw_product = '  "Cà Phê Sữa Đá (Đặc Biệt) - Size L"  '

        # Canonical output (for submission.jsonl): preserves casing, accents, removes wrapper quotes
        canonical = format_string(raw_product, clean_text=False)
        self.assertEqual(canonical, '"Cà Phê Sữa Đá (Đặc Biệt) - Size L"')
        self.assertEqual(canonical, unicodedata.normalize("NFC", canonical))

        # Comparison helper (for evaluation matching): lowercase, replaces punctuation with spaces, collapses runs
        comp = normalize_for_comparison(raw_product)
        self.assertEqual(comp, "cà phê sữa đá đặc biệt size l")
        self.assertEqual(clean_string(raw_product), "cà phê sữa đá đặc biệt size l")

    def test_date_formatting(self):
        """Date should be strict ISO 'YYYY-MM-DD' or 'YYYY-MM'."""
        d = date(2024, 6, 1)
        self.assertEqual(format_date(d), "2024-06-01")
        self.assertEqual(format_date(d, as_month=True), "2024-06")

        # Datetime
        dt_full = datetime(2024, 6, 1, 15, 30, 45)
        self.assertEqual(format_date(dt_full), "2024-06-01")
        self.assertEqual(format_date(dt_full, as_month=True), "2024-06")

        # String dates
        self.assertEqual(format_date("2024-06-01 00:00:00"), "2024-06-01")
        self.assertEqual(format_date("2024-06"), "2024-06")
        with self.assertRaises(ValueError):
            format_date("01/06/2024")

    def test_boolean_formatting(self):
        """Boolean should be strict Python bool True/False (JSON true/false)."""
        self.assertIs(format_boolean(True), True)
        self.assertIs(format_boolean(False), False)
        self.assertIs(format_boolean("True"), True)
        self.assertIs(format_boolean("FALSE"), False)
        self.assertIs(format_boolean("đúng"), True)
        self.assertIs(format_boolean("sai"), False)
        self.assertIs(format_boolean(1), True)
        self.assertIs(format_boolean(0), False)

    def test_list_and_ranking_tie_breaking(self):
        """List formatting and tie-breaking by descending criterion, ascending item code."""
        items = [
            ("SP02", 500),
            ("SP01", 500),  # Same criterion (500) -> tie-break by code: SP01 before SP02
            ("SP03", 800),  # Highest criterion (800) -> first
            ("SP04", 100),  # Lowest criterion (100) -> last
        ]
        sorted_items = sort_ranking_items(
            items,
            criterion_fn=lambda x: x[1],
            code_fn=lambda x: x[0],
            descending=True,
        )
        ranked_codes = [x[0] for x in sorted_items]
        self.assertEqual(ranked_codes, ["SP03", "SP01", "SP02", "SP04"])

        # format_list
        res_list = format_list(ranked_codes)
        self.assertEqual(res_list, ["SP03", "SP01", "SP02", "SP04"])

    def test_table_formatting(self):
        """Table should be formatted as {"columns": [...], "rows": [[...], ...]} with matching types."""
        # DataFrame input
        df = pd.DataFrame({
            "Mã SP": ["SP01", "SP02"],
            "Doanh Thu": [15000000.5, 20000000],
        })
        table_dict = format_table(df)
        self.assertEqual(table_dict["columns"], ["Mã SP", "Doanh Thu"])
        self.assertEqual(table_dict["rows"], [
            ["SP01", 15000000.5],
            ["SP02", 20000000],
        ])

        # List of dicts (records format)
        records = [
            {"branch": "Hà Nội", "revenue": 100.555},
            {"branch": "Đà Nẵng", "revenue": 200},
        ]
        res_records = format_table(records)
        self.assertEqual(res_records["columns"], ["branch", "revenue"])
        self.assertEqual(res_records["rows"], [
            ["Hà Nội", 100.56],
            ["Đà Nẵng", 200],
        ])

    def test_format_output_with_q_type_dispatch(self):
        """format_output should strictly dispatch based on q_type."""
        # percent
        self.assertEqual(format_output("-12.45%", success=True, q_type="percent"), -12.45)
        self.assertEqual(format_output(-0.1245, success=True, q_type="percent"), -0.12)

        # number with VND
        self.assertEqual(format_output("15,000,000 VND", success=True, q_type="number"), 15000000)

        # boolean
        self.assertIs(format_output("đúng", success=True, q_type="boolean"), True)

        # date
        self.assertEqual(format_output("2024-08-15", success=True, q_type="date"), "2024-08-15")

        # table
        df = pd.DataFrame({"col": [1, 2]})
        tbl = format_output(df, success=True, q_type="table")
        self.assertEqual(tbl, {"columns": ["col"], "rows": [[1], [2]]})

        # abstain on failure or empty
        self.assertEqual(format_output(None, success=False, q_type="number"), "abstain")
        self.assertEqual(format_output(pd.DataFrame(), success=True, q_type="table"), "abstain")

    def test_submission_record_proper_typing_and_unanswerable(self):
        """Submission records must strictly omit alias, type correctly, and never output placeholder strings."""
        # Unanswerable
        rec_unans = format_submission_record(
            "q_100",
            answer="unanswerable",
            q_type="unanswerable",
            reason="Thiếu thông tin chi nhánh trong câu hỏi",
        )
        self.assertEqual(rec_unans["id"], "q_100")
        self.assertEqual(rec_unans["type"], "unanswerable")
        self.assertNotIn("answer", rec_unans)
        self.assertEqual(rec_unans["reason"], "Thiếu thông tin chi nhánh trong câu hỏi")
        self.assertNotIn("alias", rec_unans)

        # Abstain (Section 3.2: strictly id and type, no answer key)
        rec_abs = format_submission_record("q_101", answer=None)
        self.assertEqual(rec_abs, {"id": "q_101", "type": "abstain"})
        self.assertNotIn("answer", rec_abs)
        self.assertNotIn("alias", rec_abs)

        # Table
        df = pd.DataFrame({"item": ["A"], "qty": [5]})
        rec_tbl = format_submission_record("q_102", answer=df, q_type="table")
        self.assertEqual(rec_tbl["type"], "table")
        self.assertEqual(rec_tbl["answer"], {"columns": ["item", "qty"], "rows": [["A", 5]]})
        self.assertNotIn("alias", rec_tbl)

        # JSON line formatting
        json_line = format_submission_json("q_103", answer=True, q_type="boolean")
        parsed = json.loads(json_line)
        self.assertEqual(parsed["id"], "q_103")
        self.assertEqual(parsed["type"], "boolean")
        self.assertIs(parsed["answer"], True)

    def test_user_prompt_precomputed_intervals(self):
        """PromptBuilder should inject pre-computed intervals when as_of_date is provided."""
        pb = PromptBuilder()
        prompt = pb.build_user_prompt(
            question="Doanh thu tuần trước là bao nhiêu?",
            schema_context="Table: orders.csv",
            as_of_date="2024-06-19",
        )
        self.assertIn("PRE-COMPUTED TIME INTERVALS", prompt)
        self.assertIn("Tuần trước / Tuần rồi", prompt)
        self.assertIn("Tháng này (Month-to-Date", prompt)
        self.assertIn("2024-06-19", prompt)


if __name__ == "__main__":
    unittest.main()
