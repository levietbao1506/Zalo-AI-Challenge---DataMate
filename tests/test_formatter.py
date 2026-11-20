"""Unit tests for output normalization and formatting (Bước 5: Formatter & Abstain)."""

from datetime import date, datetime
from decimal import Decimal
import json
from pathlib import Path
import tempfile
import unicodedata
import unittest

from src.formatter import (
    DEFAULT_ABSTAIN_TOKEN,
    DEFAULT_ABSTAIN_VALUE,
    append_submission_line,
    clean_string,
    format_boolean,
    format_date,
    format_list,
    format_number,
    format_output,
    format_percent,
    format_scalar,
    format_submission_json,
    format_submission_record,
    format_table,
    is_abstain_signal,
    normalize_for_comparison,
    normalize_unicode_nfc,
)

try:
    import numpy as np
    _HAS_NUMPY = True
except ImportError:
    _HAS_NUMPY = False

try:
    import pandas as pd
    _HAS_PANDAS = True
except ImportError:
    _HAS_PANDAS = False


class TestFormatter(unittest.TestCase):
    """Test suite for competition output formatting and normalization."""

    def test_number_decimal_money_and_nonfinite(self):
        self.assertEqual(format_number("-1.500.000 đ"), -1500000)
        self.assertEqual(format_number("1,500,000 VND"), 1500000)
        self.assertEqual(format_number("1.234,50 đ"), 1234.5)
        self.assertEqual(format_number(12.340000000000002), 12.34)
        self.assertEqual(format_number(Decimal("2.675")), 2.68)
        self.assertEqual(format_number(-0.0), 0)
        self.assertIs(type(format_number(-0.0)), int)
        for bad in (float("nan"), float("inf"), float("-inf"), "NaN", "Infinity", True, "12%"):
            self.assertEqual(format_submission_record("n", bad, "number"), {"id": "n", "type": "abstain"})

    def test_percent_numeric_and_signed(self):
        self.assertEqual(format_submission_record("p", "-12.455%", "percent")["answer"], -12.46)
        self.assertEqual(format_submission_record("p", "+0.25%", "percent")["answer"], 0.25)
        self.assertEqual(format_submission_record("p", -0.0, "percent")["answer"], 0)
        self.assertEqual(format_submission_record("p", float("inf"), "percent")["type"], "abstain")

    def test_string_nfc_and_identifier_preservation(self):
        source = "  Đà Nẵng - Mã 001  "
        nfd = unicodedata.normalize("NFD", source)
        record = format_submission_record("s", nfd, "string")
        self.assertEqual(record["answer"], "Đà Nẵng - Mã 001")
        self.assertEqual(record["answer"], unicodedata.normalize("NFC", record["answer"]))
        self.assertEqual(format_submission_record("s", "001", "string")["answer"], "001")
        self.assertEqual(format_submission_record("s", 123, "string")["type"], "abstain")

    def test_date_iso_validation_and_timestamp(self):
        self.assertEqual(format_submission_record("d", datetime(2024, 2, 29, 9, 30), "date")["answer"], "2024-02-29")
        self.assertEqual(format_submission_record("d", "2024-02-29 00:00:00", "date")["answer"], "2024-02-29")
        self.assertEqual(format_submission_record("d", "2024-02", "date")["answer"], "2024-02")
        if _HAS_PANDAS:
            self.assertEqual(format_submission_record("d", pd.Timestamp("2024-02-29"), "date")["answer"], "2024-02-29")
        for bad in ("2023-02-29", "2024-13", "2024-02-29 12:00:00", "29/02/2024", "tomorrow"):
            self.assertEqual(format_submission_record("d", bad, "date")["type"], "abstain")

    def test_boolean_is_json_boolean(self):
        for raw, expected in ((True, True), (False, False), ("true", True), ("Có", True), (1, True), (0, False)):
            record = format_submission_record("b", raw, "boolean")
            self.assertIs(record["answer"], expected)
            self.assertIn('"answer": true' if expected else '"answer": false', json.dumps(record))
        for bad in (2, -1, float("nan"), "maybe"):
            self.assertEqual(format_submission_record("b", bad, "boolean")["type"], "abstain")

    def test_list_and_table_reject_malformed_cells(self):
        self.assertEqual(format_submission_record("l", ["001", " Đà Nẵng ", "1.500.000", 2.675], "list")["answer"], ["001", "Đà Nẵng", 1500000, 2.68])
        for bad in ([1, None], [float("nan")], [float("inf")], [], {"a", "b"}):
            self.assertEqual(format_submission_record("l", bad, "list")["type"], "abstain")
        table = {"columns": [" Mã ", "Doanh thu"], "rows": [["001", "1.500.000 đ"], ["002", 2.675]]}
        self.assertEqual(format_submission_record("t", table, "table")["answer"], {"columns": ["Mã", "Doanh thu"], "rows": [["001", 1500000], ["002", 2.68]]})
        if _HAS_PANDAS:
            date_table = pd.DataFrame({"ngày": [pd.Timestamp("2024-05-01")], "số tiền": ["15.000 đ"]})
            self.assertEqual(format_submission_record("t", date_table, "table")["answer"]["rows"], [["2024-05-01", 15000]])
        bad_tables = [
            {"columns": ["a", "b"], "rows": [[1]]},
            {"columns": ["a"], "rows": [[None]]},
            {"columns": ["a"], "rows": [[float("inf")]]},
            {"columns": ["a"], "rows": []},
            {"columns": ["a"], "rows": [[1]], "extra": 2},
        ]
        for bad in bad_tables:
            self.assertEqual(format_submission_record("t", bad, "table")["type"], "abstain")

    def test_submission_schema_and_append_fallback(self):
        self.assertEqual(format_submission_record("u", "unanswerable", "unanswerable", reason=" Thiếu cột giá vốn "), {"id": "u", "type": "unanswerable", "reason": "Thiếu cột giá vốn"})
        self.assertEqual(format_submission_record("u", {"type": "unanswerable", "reason": "Thiếu cột giá vốn"}), {"id": "u", "type": "unanswerable", "reason": "Thiếu cột giá vốn"})
        self.assertEqual(format_submission_record("u", "unanswerable", "unanswerable"), {"id": "u", "type": "abstain"})
        self.assertEqual(format_submission_record("a", None, "abstain"), {"id": "a", "type": "abstain"})
        self.assertEqual(format_submission_record("x", 1, "unknown"), {"id": "x", "type": "abstain"})
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "submission.jsonl"
            append_submission_line(path, {"id": "x", "type": "number", "answer": float("nan")})
            self.assertEqual(json.loads(path.read_text(encoding="utf-8")), {"id": "x", "type": "abstain"})

    def test_unicode_nfc_normalization(self):
        """Ensure Vietnamese decomposed Unicode (NFD) is normalized to NFC."""
        nfd_text = unicodedata.normalize("NFD", "Tiếng Việt, Thành phố Hồ Chí Minh, Đà Nẵng")
        self.assertNotEqual(nfd_text, unicodedata.normalize("NFC", nfd_text))

        normalized = normalize_unicode_nfc(nfd_text)
        expected_nfc = unicodedata.normalize("NFC", "Tiếng Việt, Thành phố Hồ Chí Minh, Đà Nẵng")
        self.assertEqual(normalized, expected_nfc)

    def test_clean_string_rule(self):
        """Chuẩn hóa String: NFC, chữ thường, thay ký tự đặc biệt bằng khoảng trắng, xóa khoảng trắng thừa."""
        raw_text = "  Tiếng Việt, TP. Hồ Chí Minh! (Đà Nẵng #1) - Chi_Nhánh  "
        cleaned = clean_string(raw_text)
        # Should be lowercase, punctuation and underscore replaced with space, collapsed spaces
        self.assertEqual(cleaned, "tiếng việt tp hồ chí minh đà nẵng 1 chi nhánh")

        # Test Unicode NFC guarantee
        self.assertEqual(cleaned, unicodedata.normalize("NFC", cleaned))

    def test_float_rounding(self):
        """Floats should be rounded to exactly 2 decimal places."""
        self.assertEqual(format_scalar(123.456), 123.46)
        self.assertEqual(format_scalar(123.454), 123.45)
        self.assertEqual(format_scalar(0.001), 0)

    def test_format_number_and_percent(self):
        """Chuẩn hóa Number/Percent: ép kiểu float, làm tròn 2 chữ số thập phân, hỗ trợ chuỗi %."""
        self.assertEqual(format_number(123.456), 123.46)
        self.assertEqual(format_number(42.0), 42)
        self.assertEqual(format_percent("15.75%"), 15.75)
        self.assertEqual(format_percent(" 20 % "), 20)
        self.assertEqual(format_percent("33.333%"), 33.33)
        self.assertEqual(format_percent("-5.25%"), -5.25)

    def test_float_integer_simplification(self):
        """Float representations of integers should be simplified to int."""
        self.assertEqual(format_scalar(42.0), 42)
        self.assertIsInstance(format_scalar(42.0), int)
        self.assertEqual(format_scalar(100.000), 100)
        self.assertIsInstance(format_scalar(100.000), int)

    def test_integer_preservation(self):
        """Integers should be preserved as ints."""
        self.assertEqual(format_scalar(1000), 1000)
        self.assertIsInstance(format_scalar(1000), int)

    def test_date_and_datetime(self):
        """Dates and datetimes should be formatted as clean ISO strings."""
        d = date(2024, 5, 15)
        self.assertEqual(format_scalar(d), "2024-05-15")

        dt = datetime(2024, 5, 15, 14, 30, 0)
        self.assertEqual(format_scalar(dt), "2024-05-15")

    def test_boolean(self):
        """Booleans must stay boolean and not convert to int 1 or 0."""
        self.assertIs(format_scalar(True), True)
        self.assertIs(format_scalar(False), False)

    def test_abstain_signals(self):
        """Null, NaN, empty strings, and explicit abstain tokens must yield abstain fallback."""
        self.assertEqual(format_scalar(None), DEFAULT_ABSTAIN_VALUE)
        self.assertEqual(format_scalar("ABSTAIN"), DEFAULT_ABSTAIN_VALUE)
        self.assertEqual(format_scalar("abstain"), DEFAULT_ABSTAIN_VALUE)
        self.assertEqual(format_scalar("unanswerable"), DEFAULT_ABSTAIN_VALUE)
        self.assertEqual(format_scalar("UNANSWERABLE"), DEFAULT_ABSTAIN_VALUE)
        self.assertEqual(format_scalar("null"), DEFAULT_ABSTAIN_VALUE)
        self.assertEqual(format_scalar("None"), DEFAULT_ABSTAIN_VALUE)
        self.assertEqual(format_scalar(""), DEFAULT_ABSTAIN_VALUE)
        self.assertEqual(format_scalar(float("nan")), DEFAULT_ABSTAIN_VALUE)
        self.assertEqual(format_scalar(float("inf")), DEFAULT_ABSTAIN_VALUE)

    def test_abstain_decision_after_failure_and_empty_dataframe(self):
        """Quyết định Abstain: lỗi sau retries hoặc DataFrame rỗng -> answer = 'abstain'."""
        if not _HAS_PANDAS:
            self.skipTest("Pandas is not installed.")

        # 1. Execution failure (success=False)
        self.assertEqual(format_output(None, success=False), "abstain")
        self.assertEqual(format_output("result_val", success=False), "abstain")

        # 2. Empty DataFrame
        empty_df = pd.DataFrame()
        self.assertEqual(format_output(empty_df, success=True), "abstain")

        # 3. Empty Series
        empty_s = pd.Series([], dtype="object")
        self.assertEqual(format_output(empty_s, success=True), "abstain")

        # 4. Valid result passes through
        self.assertEqual(format_output(500.555, success=True), 500.56)

    def test_string_quote_cleanup(self):
        """Strings wrapped in redundant quotes should be stripped cleanly."""
        self.assertEqual(format_scalar('"Hà Nội"'), '"Hà Nội"')
        self.assertEqual(format_scalar("'Sài Gòn'"), "'Sài Gòn'")

    def test_numeric_string_parsing(self):
        """Clean numeric strings should be parsed to numeric values."""
        self.assertEqual(format_scalar("42"), 42)
        self.assertEqual(format_scalar("123.456"), 123.46)

    def test_numpy_unwrapping(self):
        """NumPy generic scalar types should be cast to native Python types."""
        if not _HAS_NUMPY:
            self.skipTest("NumPy is not installed in the test environment.")
        np_int = np.int64(42)
        formatted = format_scalar(np_int)
        self.assertEqual(formatted, 42)
        self.assertIsInstance(formatted, int)

        np_float = np.float64(3.14159)
        formatted_float = format_scalar(np_float)
        self.assertEqual(formatted_float, 3.14)
        self.assertIsInstance(formatted_float, float)

    def test_pandas_unwrapping(self):
        """Pandas 1x1 DataFrame and 1-element Series should be unwrapped to scalars."""
        if not _HAS_PANDAS:
            self.skipTest("Pandas is not installed in the test environment.")
        # 1x1 DataFrame
        df = pd.DataFrame({"total": [150.758]})
        self.assertEqual(format_scalar(df), 150.76)

        # 1-element Series
        s = pd.Series([99])
        self.assertEqual(format_scalar(s), 99)

        # Empty DataFrame
        empty_df = pd.DataFrame()
        self.assertEqual(format_scalar(empty_df), DEFAULT_ABSTAIN_VALUE)

    def test_format_submission_record_and_json(self):
        """Format dictionary cuối cùng thành chuỗi JSON chứa id, type, answer."""
        # Number record
        rec_num = format_submission_record("q_001", 15000.5)
        self.assertEqual(rec_num, {"id": "q_001", "type": "number", "answer": 15000.5})

        # String record
        rec_str = format_submission_record("q_002", "hà nội")
        self.assertEqual(rec_str, {"id": "q_002", "type": "string", "answer": "hà nội"})

        # Abstain record (Section 3.2: strictly id and type, no answer key)
        rec_abs = format_submission_record("q_003", "abstain")
        self.assertEqual(rec_abs, {"id": "q_003", "type": "abstain"})
        self.assertNotIn("answer", rec_abs)

        # JSON line formatting
        json_line = format_submission_json("q_004", "đà nẵng", q_type="string")
        parsed = json.loads(json_line)
        self.assertEqual(parsed["id"], "q_004")
        self.assertEqual(parsed["answer"], "đà nẵng")
        self.assertEqual(parsed["type"], "string")

    def test_append_submission_line(self):
        """Ghi append vào file submission.jsonl."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            sub_file = Path(tmp_dir) / "submission.jsonl"

            rec1 = format_submission_record("q_1", 100)
            rec2 = format_submission_record("q_2", "abstain")
            append_submission_line(sub_file, rec1)
            append_submission_line(sub_file, rec2)

            lines = sub_file.read_text(encoding="utf-8").strip().splitlines()
            self.assertEqual(len(lines), 2)
            parsed_1 = json.loads(lines[0])
            parsed_2 = json.loads(lines[1])
            self.assertEqual(parsed_1["id"], "q_1")
            self.assertEqual(parsed_1["answer"], 100)
            self.assertEqual(parsed_2["id"], "q_2")
            self.assertEqual(parsed_2["type"], "abstain")
            self.assertNotIn("answer", parsed_2)

    def test_exact_string_normalization_and_comparison(self):
        """Test competition string normalization: NFC, lowercase, punctuation removal, whitespace collapse."""
        raw = "  Đại Học Bách Khoa - Hà Nội (Cơ sở 1)!  "
        normalized = normalize_for_comparison(raw)
        self.assertEqual(normalized, "đại học bách khoa hà nội cơ sở 1")

        # Must be NFC
        self.assertEqual(normalized, unicodedata.normalize("NFC", normalized))

    def test_number_and_currency_stripping(self):
        """Ensure number formatting strips VND/currency signs and commas, rounding to 2 decimals."""
        self.assertEqual(format_number("15,000,000 VNĐ"), 15000000)
        self.assertEqual(format_number("1,234.567 đồng"), 1234.57)
        self.assertEqual(format_number("$ 999.99"), 999.99)
        self.assertEqual(format_number("350.000"), 350)

    def test_percent_formatting_rules(self):
        """Percent must be a numeric float without percent signs, converting ratios when needed."""
        # Negative ratio converted to percentage number
        self.assertEqual(format_percent(-0.1245), -0.12)
        self.assertIsInstance(format_percent(-0.1245), float)

        # Percent string parsed to clean numeric float
        self.assertEqual(format_percent("-12.45%"), -12.45)
        self.assertEqual(format_percent("25%"), 25)

    def test_date_iso_formatting_comprehensive(self):
        """Ensure dates are formatted strictly as YYYY-MM-DD or YYYY-MM."""
        self.assertEqual(format_date("2024-05-01"), "2024-05-01")
        self.assertEqual(format_date("2024-05-01 00:00:00"), "2024-05-01")
        self.assertEqual(format_date("2024-05"), "2024-05")
        with self.assertRaises(ValueError):
            format_date("2024/05/01")

    def test_table_and_ranking_list_formatting(self):
        """Test formatting of table dictionaries and ranking lists."""
        table_in = {"columns": ["Ten", "Diem"], "rows": [["An", 9.556], ["Binh", 8.0]]}
        formatted_tbl = format_table(table_in)
        self.assertEqual(formatted_tbl["columns"], ["Ten", "Diem"])
        self.assertEqual(formatted_tbl["rows"], [["An", 9.56], ["Binh", 8]])

        # Ranking tie breaking
        ranked_items = ["Cửa hàng C", "Cửa hàng A", "Cửa hàng B"]
        formatted_list = format_list(ranked_items)
        self.assertEqual(formatted_list, ["Cửa hàng C", "Cửa hàng A", "Cửa hàng B"])


if __name__ == "__main__":
    unittest.main()
