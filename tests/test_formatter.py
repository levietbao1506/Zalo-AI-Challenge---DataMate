"""Unit tests for output normalization and formatting (Penalized Accuracy rules)."""

from datetime import date, datetime
import unicodedata
import unittest

from src.formatter import (
    DEFAULT_ABSTAIN_VALUE,
    format_scalar,
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
        # 'Tiếng Việt' decomposed into base letters + combining marks
        nfd_text = unicodedata.normalize("NFD", "Tiếng Việt, Thành phố Hồ Chí Minh, Đà Nẵng")
        # Ensure it was decomposed
        self.assertNotEqual(nfd_text, unicodedata.normalize("NFC", nfd_text))

        normalized = normalize_unicode_nfc(nfd_text)
        expected_nfc = unicodedata.normalize("NFC", "Tiếng Việt, Thành phố Hồ Chí Minh, Đà Nẵng")
        self.assertEqual(normalized, expected_nfc)

    def test_float_rounding(self):
        """Floats should be rounded to exactly 2 decimal places."""
        self.assertEqual(format_scalar(123.456), 123.46)
        self.assertEqual(format_scalar(123.454), 123.45)
        self.assertEqual(format_scalar(0.001), 0)

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


if __name__ == "__main__":
    unittest.main()
