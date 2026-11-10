"""Unit tests for SandboxedExecutor environment."""

from datetime import date
import unittest

from src.executor import SandboxedExecutor


class TestSandboxedExecutor(unittest.TestCase):
    """Test suite for sandboxed code execution, error trapping, and timeouts."""

    def setUp(self):
        self.executor = SandboxedExecutor(default_timeout_seconds=5.0)

    def test_basic_execution(self):
        """Simple code assigning result should succeed."""
        code = "result = 10 + 25"
        exec_res = self.executor.execute(code)
        self.assertTrue(exec_res.success)
        self.assertEqual(exec_res.result, 35)
        self.assertIsNone(exec_res.error)

    def test_calendar_utils_injection(self):
        """calendar_utils functions and as_of_date should be directly accessible."""
        code = """
start_date, end_date = calendar_utils.resolve_vietnamese_relative_date("hôm qua", as_of=as_of_date)
result = str(start_date)
"""
        exec_res = self.executor.execute(code, as_of_date=date(2024, 6, 15))
        self.assertTrue(exec_res.success)
        self.assertEqual(exec_res.result, "2024-06-14")

    def test_sqlite_execution(self):
        """SQLite in-memory queries should run smoothly."""
        code = """
conn = sqlite3.connect(":memory:")
cur = conn.cursor()
cur.execute("CREATE TABLE sales (branch TEXT, revenue REAL);")
cur.execute("INSERT INTO sales VALUES ('Hà Nội', 500.5), ('Hà Nội', 300.5), ('Đà Nẵng', 100.0);")
cur.execute("SELECT SUM(revenue) FROM sales WHERE branch = 'Hà Nội';")
result = cur.fetchone()[0]
conn.close()
"""
        exec_res = self.executor.execute(code)
        self.assertTrue(exec_res.success)
        self.assertEqual(exec_res.result, 801.0)

    def test_syntax_error_trapping(self):
        """Syntax errors should be safely caught without crashing host process."""
        code = "def invalid_syntax(:"
        exec_res = self.executor.execute(code)
        self.assertFalse(exec_res.success)
        self.assertIsNone(exec_res.result)
        self.assertIn("SyntaxError", exec_res.error)

    def test_runtime_error_trapping(self):
        """ZeroDivisionError and runtime exceptions should be safely caught."""
        code = "result = 10 / 0"
        exec_res = self.executor.execute(code)
        self.assertFalse(exec_res.success)
        self.assertIsNone(exec_res.result)
        self.assertIn("ZeroDivisionError", exec_res.error)

    def test_timeout_trapping(self):
        """Infinite loop should terminate gracefully within specified timeout."""
        code = """
import time
while True:
    time.sleep(0.05)
"""
        exec_res = self.executor.execute(code, timeout_seconds=0.5)
        self.assertFalse(exec_res.success)
        self.assertIsNone(exec_res.result)
        self.assertIn("TimeoutError", exec_res.error)

    def test_blocked_module_import(self):
        """Attempting to import restricted system modules should raise ImportError."""
        code = """
import os
result = os.listdir('.')
"""
        exec_res = self.executor.execute(code)
        self.assertFalse(exec_res.success)
        self.assertIn("ImportError", exec_res.error)
        self.assertIn("restricted", exec_res.error)

    def test_stdout_capture(self):
        """Prints should be captured and fallback to stdout if result is not assigned."""
        code = """
print("Computed output: 999")
"""
        exec_res = self.executor.execute(code)
        self.assertTrue(exec_res.success)
        self.assertIn("Computed output: 999", exec_res.stdout)
        self.assertEqual(exec_res.result, "Computed output: 999")


if __name__ == "__main__":
    unittest.main()
