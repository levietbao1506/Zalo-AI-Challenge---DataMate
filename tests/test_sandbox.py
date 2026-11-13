"""Unit tests for SandboxedExecutor environment & Self-Correction Engine (Bước 4)."""

from datetime import date
from typing import Optional
import unittest

import pandas as pd

from src.executor import (
    EMPTY_RESULT_FEEDBACK,
    ExecutionResult,
    SandboxedExecutor,
    execute_local,
    execute_sandboxed_code,
    execute_with_retry,
    extract_code,
    extract_python_code,
    is_empty_result,
)
from src.prompt_builder import PromptBuilder


class MockAgentForCorrection:
    """Mock agent designed to simulate self-correction code generation."""

    def __init__(self, code_sequence: list[str]):
        self.code_sequence = code_sequence
        self.call_count = 0
        self.captured_prompts: list[str] = []

    def generate_code(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        self.captured_prompts.append(prompt)
        idx = min(self.call_count, len(self.code_sequence) - 1)
        code = self.code_sequence[idx]
        self.call_count += 1
        return code


class TestSandboxedExecutor(unittest.TestCase):
    """Test suite for sandboxed code execution, error trapping, and timeouts."""

    def setUp(self):
        self.executor = SandboxedExecutor(default_timeout_seconds=5.0)
        self.prompt_builder = PromptBuilder()

    def test_extract_python_code(self):
        """Test extraction of python code from markdown backticks."""
        raw_llm = (
            "Chào bạn, đây là đoạn code bạn cần:\n"
            "```python\n"
            "import pandas as pd\n"
            "result = 123.45\n"
            "```\n"
            "Hy vọng giúp ích!"
        )
        code = extract_python_code(raw_llm)
        self.assertEqual(code, "import pandas as pd\nresult = 123.45")

        # Generic markdown
        self.assertEqual(extract_python_code("```\nresult = 'ABC'\n```"), "result = 'ABC'")

        # Unclosed backtick
        unclosed = "```python\nresult = 999"
        self.assertEqual(extract_python_code(unclosed), "result = 999")

        # Empty fallback
        self.assertEqual(extract_python_code(""), "result = None")

    def test_basic_execution(self):
        """Simple code assigning result should succeed."""
        code = "result = 10 + 25"
        exec_res = self.executor.execute(code)
        self.assertTrue(exec_res.success)
        self.assertEqual(exec_res.result, 35)
        self.assertIsNone(exec_res.error)

    def test_file_map_and_variable_alias_injection(self):
        """FILE_MAP and clean filename aliases should be available in execution globals."""
        code = """
import pandas as pd
df = pd.DataFrame({'val': [10, 20, 30]})
# Access via orders_csv alias
result = orders_csv
"""
        file_map = {"orders.csv": "/app/data/retail/orders.csv"}
        exec_res = self.executor.execute(code, file_map=file_map)
        self.assertTrue(exec_res.success)
        self.assertEqual(exec_res.result, "/app/data/retail/orders.csv")

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
        self.assertEqual(exec_res.error_type, "SyntaxError")

    def test_runtime_error_trapping(self):
        """KeyError and ZeroDivisionError exceptions should be safely caught."""
        # Zero division
        exec_div = self.executor.execute("result = 10 / 0")
        self.assertFalse(exec_div.success)
        self.assertEqual(exec_div.error_type, "ZeroDivisionError")

        # KeyError
        code_key = """
data = {'a': 1}
result = data['non_existent_key']
"""
        exec_key = self.executor.execute(code_key)
        self.assertFalse(exec_key.success)
        self.assertEqual(exec_key.error_type, "KeyError")
        self.assertIn("KeyError", exec_key.error_msg)

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
        self.assertEqual(exec_res.error_type, "TimeoutError")

    def test_blocked_module_import(self):
        """Attempting to import restricted system modules should raise ImportError."""
        code = """
import os
result = os.listdir('.')
"""
        exec_res = self.executor.execute(code)
        self.assertFalse(exec_res.success)
        self.assertEqual(exec_res.error_type, "ImportError")
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

    def test_self_correction_success_on_retry(self):
        """Test self-correction: turn 1 fails with KeyError, turn 2 corrects code and succeeds."""
        # Initial code has a bug (KeyError)
        bad_code = "data = {'total_rev': 500}\nresult = data['wrong_key']"
        # Corrected code fixes the key
        good_code = "data = {'total_rev': 500}\nresult = data['total_rev']"

        mock_agent = MockAgentForCorrection([good_code])

        exec_res = self.executor.execute_with_self_correction(
            agent=mock_agent,
            initial_code=bad_code,
            question="Tính tổng doanh thu?",
            prompt_builder=self.prompt_builder,
            max_retries=3,
        )

        self.assertTrue(exec_res.success)
        self.assertEqual(exec_res.result, 500)
        self.assertEqual(exec_res.attempts_used, 2)
        # Verify the feedback message contained the required text
        self.assertEqual(len(mock_agent.captured_prompts), 1)
        retry_prompt = mock_agent.captured_prompts[0]
        self.assertIn("Đoạn code trên bị lỗi: KeyError", retry_prompt)
        self.assertIn("Hãy viết lại code sửa lỗi", retry_prompt)

    def test_self_correction_max_3_retries_limit(self):
        """Test self-correction terminates after exactly max 3 retries if code keeps failing."""
        always_bad_code = "result = 1 / 0"
        mock_agent = MockAgentForCorrection([always_bad_code, always_bad_code, always_bad_code, always_bad_code])

        exec_res = self.executor.execute_with_self_correction(
            agent=mock_agent,
            initial_code=always_bad_code,
            question="Tính chia cho 0?",
            prompt_builder=self.prompt_builder,
            max_retries=3,
        )

        self.assertFalse(exec_res.success)
        self.assertIsNone(exec_res.result)
        # 1 initial + 3 retries = 4 attempts total
        self.assertEqual(exec_res.attempts_used, 4)
        self.assertEqual(mock_agent.call_count, 3)

    def test_standalone_execute_sandboxed_code(self):
        """Test convenience helper function execute_sandboxed_code."""
        res = execute_sandboxed_code("result = 7 * 8")
        self.assertTrue(res.success)
        self.assertEqual(res.result, 56)

    def test_extract_code_function(self):
        """Test extract_code handles python markdown fences and raw code."""
        llm_output = "Đây là mã Python:\n```python\n  final_result = 42  \n```\nChúc may mắn."
        self.assertEqual(extract_code(llm_output), "final_result = 42")

        # Raw code without markdown formatting
        raw_code = "   x = 10\nfinal_result = x * 2   "
        self.assertEqual(extract_code(raw_code), "x = 10\nfinal_result = x * 2")

    def test_execute_local_success_and_data_dir_injection(self):
        """Test execute_local injects data_dir and extracts final_result."""
        code = "import os\nfinal_result = os.path.join(data_dir, 'orders.csv')"
        success, result, err = execute_local(code, "/app/data")
        self.assertTrue(success)
        self.assertTrue(result.endswith("orders.csv"))
        self.assertEqual(err, "")

    def test_execute_local_exception_catching(self):
        """Test execute_local catches exceptions safely and returns (False, None, error_details)."""
        success, result, err = execute_local("x = 1 / 0\nfinal_result = x", "/app/data")
        self.assertFalse(success)
        self.assertIsNone(result)
        self.assertIn("ZeroDivisionError", err)
        self.assertIn("division by zero", err)
        self.assertIn("Traceback (most recent call last):", err)

        # KeyError test
        success_k, _, err_k = execute_local("d = {}\nfinal_result = d['col_missing']", "/app/data")
        self.assertFalse(success_k)
        self.assertIn("KeyError", err_k)
        self.assertIn("Traceback (most recent call last):", err_k)

    def test_execute_with_retry_self_correction(self):
        """Test execute_with_retry appends exact Vietnamese error prompt and succeeds on retry."""
        prompts_seen = []
        responses = [
            "```python\nfinal_result = missing_var\n```",
            "```python\nfinal_result = 123\n```",
        ]

        def mock_generate(prompt: str) -> str:
            prompts_seen.append(prompt)
            return responses[len(prompts_seen) - 1]

        success, result, err = execute_with_retry(
            initial_prompt="Tính tổng doanh thu",
            generate_fn=mock_generate,
            data_dir="./data",
            max_retries=3,
        )
        self.assertTrue(success)
        self.assertEqual(result, 123)
        self.assertEqual(err, "")
        self.assertEqual(len(prompts_seen), 2)
        # 1. Original question and schema
        self.assertIn("Tính tổng doanh thu", prompts_seen[1])
        # 2. Failed Python code
        self.assertIn("missing_var", prompts_seen[1])
        # 3. Exact error message
        self.assertIn("NameError", prompts_seen[1])
        # 4. Strict instruction
        self.assertIn("The previous code failed with the error above. Write a new, corrected Python code block to solve the problem and fix the error. DO NOT repeat the exact same code.", prompts_seen[1])
        self.assertIn("\nĐoạn code trên bị lỗi: ", prompts_seen[1])
        self.assertIn(". Hãy viết lại code sửa lỗi.", prompts_seen[1])

    def test_execute_with_retry_exhausted(self):
        """Test execute_with_retry returns (False, None, error_msg) when max_retries is exhausted."""
        calls = 0

        def always_fail(prompt: str) -> str:
            nonlocal calls
            calls += 1
            return "```python\nfinal_result = 1 / 0\n```"

        success, result, err = execute_with_retry(
            initial_prompt="Initial",
            generate_fn=always_fail,
            data_dir="./data",
            max_retries=3,
        )
        self.assertFalse(success)
        self.assertIsNone(result)
        self.assertIn("division by zero", err)
        self.assertEqual(calls, 3)

    def test_is_empty_result_detection_and_valid_exceptions(self):
        """Test is_empty_result catches empty structures/NA values while allowing abstain tokens and zeros."""
        self.assertTrue(is_empty_result(None))
        self.assertTrue(is_empty_result(""))
        self.assertTrue(is_empty_result("   "))
        self.assertTrue(is_empty_result([]))
        self.assertTrue(is_empty_result({}))
        self.assertTrue(is_empty_result(pd.DataFrame()))
        self.assertTrue(is_empty_result(pd.Series(dtype=object)))
        self.assertTrue(is_empty_result(float("nan")))
        self.assertTrue(is_empty_result(pd.NA))
        self.assertTrue(is_empty_result(pd.NaT))

        # Critical exception: intentional unanswerable/abstain tokens are valid
        self.assertFalse(is_empty_result("unanswerable"))
        self.assertFalse(is_empty_result("Unanswerable"))
        self.assertFalse(is_empty_result("abstain"))
        self.assertFalse(is_empty_result("ABSTAIN"))

        # Valid non-empty scalars & collections (including numeric zero & False)
        self.assertFalse(is_empty_result(0))
        self.assertFalse(is_empty_result(0.0))
        self.assertFalse(is_empty_result(False))
        self.assertFalse(is_empty_result("Hà Nội"))
        self.assertFalse(is_empty_result([1, 2]))
        self.assertFalse(is_empty_result({"a": 1}))
        self.assertFalse(is_empty_result(pd.DataFrame({"val": [10]})))

    def test_execute_with_retry_empty_result_triggers_retry(self):
        """Empty filtered results must trigger retry with the exact Vietnamese logical-error prompt."""
        prompts_seen = []
        responses = [
            "```python\nimport pandas as pd\nfinal_result = pd.DataFrame()\n```",
            "```python\nfinal_result = 450.5\n```",
        ]

        def mock_generate(prompt: str) -> str:
            prompts_seen.append(prompt)
            return responses[len(prompts_seen) - 1]

        success, result, err = execute_with_retry(
            initial_prompt="Doanh thu tháng này?",
            generate_fn=mock_generate,
            data_dir="./data",
            max_retries=3,
        )
        self.assertTrue(success)
        self.assertEqual(result, 450.5)
        self.assertEqual(err, "")
        self.assertEqual(len(prompts_seen), 2)
        self.assertTrue(
            prompts_seen[1].endswith(
                "\nĐoạn code trên chạy thành công nhưng kết quả trả về bị rỗng. "
                "Hãy kiểm tra lại điều kiện lọc (filter) ngày tháng hoặc định dạng chuỗi."
            )
        )

    def test_execute_with_retry_allows_unanswerable_and_abstain_without_retry(self):
        """Intentional 'unanswerable' and 'abstain' strings must immediately succeed without retrying."""
        for token in ("unanswerable", "abstain"):
            calls = 0

            def mock_generate(_prompt: str, tok: str = token) -> str:
                nonlocal calls
                calls += 1
                return f'```python\nfinal_result = "{tok}"\n```'

            success, result, err = execute_with_retry(
                initial_prompt="Question?",
                generate_fn=mock_generate,
                data_dir="./data",
                max_retries=3,
            )
            self.assertTrue(success)
            self.assertEqual(result, token)
            self.assertEqual(err, "")
            self.assertEqual(calls, 1)

    def test_execute_with_retry_empty_result_exhausted(self):
        """Exhausting retries on empty results returns (False, None, feedback_msg)."""
        calls = 0

        def always_empty(_prompt: str) -> str:
            nonlocal calls
            calls += 1
            return "```python\nfinal_result = []\n```"

        success, result, err = execute_with_retry(
            initial_prompt="Initial",
            generate_fn=always_empty,
            data_dir="./data",
            max_retries=3,
        )
        self.assertFalse(success)
        self.assertIsNone(result)
        self.assertEqual(err, EMPTY_RESULT_FEEDBACK)
        self.assertEqual(calls, 3)

    def test_sandboxed_self_correction_retries_on_empty_result(self):
        """SandboxedExecutor.execute_with_self_correction should also recover from empty DataFrame output."""
        bad_empty_code = "import pandas as pd\nfinal_result = pd.DataFrame()"
        fixed_code = "final_result = 'Cà phê sữa'"
        mock_agent = MockAgentForCorrection([fixed_code])

        exec_res = self.executor.execute_with_self_correction(
            agent=mock_agent,
            initial_code=bad_empty_code,
            question="Sản phẩm bán chạy?",
            prompt_builder=self.prompt_builder,
            max_retries=3,
        )
        self.assertTrue(exec_res.success)
        self.assertEqual(exec_res.result, "Cà phê sữa")
        self.assertEqual(exec_res.attempts_used, 2)
        self.assertEqual(len(mock_agent.captured_prompts), 1)
        self.assertIn(EMPTY_RESULT_FEEDBACK, mock_agent.captured_prompts[0])


if __name__ == "__main__":
    unittest.main()
