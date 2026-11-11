"""Unit tests for output normalization and formatting (Bước 5: Formatter & Abstain)."""

from datetime import date, datetime
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
    format_number,
    format_output,
    format_scalar,
    format_submission_json,
    format_submission_record,
    is_abstain_signal,
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
        self.assertEqual(format_number("15.75%"), 15.75)
        self.assertEqual(format_number(" 20 % "), 20)
        self.assertEqual(format_number("33.333%"), 33.33)
        self.assertEqual(format_number("-5.25%"), -5.25)

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
        self.assertEqual(format_scalar(dt), "2024-05-15 14:30:00")

    def test_boolean(self):
        """Booleans must stay boolean and not convert to int 1 or 0."""
        self.assertIs(format_scalar(True), True)
        self.assertIs(format_scalar(False), False)

    def test_abstain_signals(self):
        """Null, NaN, empty strings, and explicit abstain tokens must yield abstain fallback."""
        self.assertEqual(format_scalar(None), DEFAULT_ABSTAIN_VALUE)
        self.assertEqual(format_scalar("ABSTAIN"), DEFAULT_ABSTAIN_VALUE)
        self.assertEqual(format_scalar("abstain"), DEFAULT_ABSTAIN_VALUE)
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
        self.assertEqual(format_scalar('"Hà Nội"'), "Hà Nội")
        self.assertEqual(format_scalar("'Sài Gòn'"), "Sài Gòn")

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

        # Abstain record
        rec_abs = format_submission_record("q_003", "abstain")
        self.assertEqual(rec_abs, {"id": "q_003", "type": "abstain", "answer": "abstain"})

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
            self.assertEqual(parsed_2["answer"], "abstain")


if __name__ == "__main__":
    unittest.main()
