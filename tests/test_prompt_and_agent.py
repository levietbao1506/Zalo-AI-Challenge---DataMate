"""Unit tests for Step 3: LLM Engine (vLLM Agent) and PromptBuilder."""

import unittest

from src.agent import (
    DEFAULT_TEMPERATURE,
    VLLMAgent,
)
from src.executor import SandboxedExecutor
from src.prompt_builder import (
    FEW_SHOT_EXAMPLES,
    SYSTEM_PROMPT,
    PromptBuilder,
)


class TestPromptBuilder(unittest.TestCase):
    """Test suite for System Prompt, Few-Shot examples, and User Prompt composition."""

    def setUp(self):
        self.builder = PromptBuilder()

    def test_system_prompt_role_declaration(self):
        """Ensure System Prompt contains the exact required role declaration."""
        expected_role = (
            "Bạn là một Data Analyst chuyên nghiệp. Hãy viết mã Python "
            "(dùng thư viện pandas) để trả lời câu hỏi sau. Chỉ in ra code, không giải thích."
        )
        self.assertIn(expected_role, SYSTEM_PROMPT)
        system_full = self.builder.build_system_prompt()
        self.assertIn(expected_role, system_full)

    def test_user_prompt_concatenation(self):
        """Ensure User Prompt connects Schema String + as_of_date + Vietnamese Question."""
        schema_sample = "### Table: `sales.csv` (CSV)\n- Columns: `order_id`, `revenue`"
        as_of_date = "2024-06-15"
        question = "Tính tổng doanh số chi nhánh Đà Nẵng hôm qua?"
        file_map = {"sales.csv": "/app/data/sales.csv"}

        user_prompt = self.builder.build_user_prompt(
            question=question,
            schema_context=schema_sample,
            as_of_date=as_of_date,
            file_map=file_map,
        )

        # 1. Contains Schema String
        self.assertIn("THÔNG TIN CẤU TRÚC BẢNG VÀ DỮ LIỆU (SCHEMA STRING)", user_prompt)
        self.assertIn(schema_sample, user_prompt)

        # 2. Contains as_of_date
        self.assertIn("Ngày tham chiếu (as_of_date): `2024-06-15`", user_prompt)

        # 3. Contains File Map
        self.assertIn("`sales.csv`", user_prompt)

        # 4. Contains Question
        self.assertIn(f"\"{question}\"", user_prompt)

        # 5. Core instruction
        self.assertIn("Chỉ in ra code trong thẻ ```python ... ```, không giải thích.", user_prompt)

    def test_user_prompt_defaults_without_as_of_date(self):
        """Ensure User Prompt handles missing as_of_date and empty file_map gracefully."""
        user_prompt = self.builder.build_user_prompt(
            question="Số lượng khách hàng mới là bao nhiêu?",
            schema_context="Table: users.csv",
            as_of_date=None,
        )
        self.assertIn("Chưa chỉ định", user_prompt)
        self.assertIn("Số lượng khách hàng mới là bao nhiêu?", user_prompt)

    def test_few_shot_examples_content(self):
        """Verify Few-Shot examples cover date filtering, unaccented search, Excel, and abstain."""
        # Ví dụ 1: Filter ngày tháng với as_of_date
        self.assertIn("Lọc dữ liệu thời gian tương đối dựa vào as_of_date", FEW_SHOT_EXAMPLES)
        self.assertIn("calendar_utils.resolve_vietnamese_relative_date", FEW_SHOT_EXAMPLES)

        # Ví dụ 2: Query từ khóa tiếng Việt không dấu
        self.assertIn("Tìm kiếm từ khóa tiếng Việt không dấu", FEW_SHOT_EXAMPLES)
        self.assertIn("remove_accents", FEW_SHOT_EXAMPLES)
        self.assertIn("da nang", FEW_SHOT_EXAMPLES)

        # Ví dụ 3: Excel với sheet
        self.assertIn("Đọc file Excel nhiều sheets", FEW_SHOT_EXAMPLES)
        self.assertIn("sheet_name=\"Orders\"", FEW_SHOT_EXAMPLES)

        # Ví dụ 4: Penalized Accuracy safe abstain
        self.assertIn("Trường hợp dữ liệu không tồn tại -> Abstain để tránh bị trừ điểm", FEW_SHOT_EXAMPLES)
        self.assertIn("result = None", FEW_SHOT_EXAMPLES)

    def test_retry_prompt_formatting(self):
        """Ensure retry prompt formats code and error traceback for self-correction."""
        retry_prompt = self.builder.build_retry_prompt(
            question="Đếm đơn hàng?",
            previous_code="result = df['missing_col'].sum()",
            error_traceback="KeyError: 'missing_col'",
        )
        self.assertIn("KeyError: 'missing_col'", retry_prompt)
        self.assertIn("result = df['missing_col'].sum()", retry_prompt)
        self.assertIn("Đếm đơn hàng?", retry_prompt)
        self.assertIn("result = None", retry_prompt)


class TestVLLMAgent(unittest.TestCase):
    """Test suite for VLLMAgent initialization, temperature=0.0, and code extraction."""

    def test_agent_initialization_temperature_zero(self):
        """Verify default temperature is 0.0 for deterministic code generation."""
        self.assertEqual(DEFAULT_TEMPERATURE, 0.0)
        agent = VLLMAgent(dry_run=True)
        self.assertEqual(agent.temperature, 0.0)
        self.assertTrue(agent.dry_run)

    def test_extract_python_code_standard_block(self):
        """Extract code from standard ```python ... ``` block."""
        agent = VLLMAgent(dry_run=True)
        llm_output = (
            "Dưới đây là mã bạn cần:\n"
            "```python\n"
            "import pandas as pd\n"
            "df = pd.read_csv('sales.csv')\n"
            "result = int(df['total'].sum())\n"
            "```\n"
            "Hy vọng giúp ích cho bạn!"
        )
        code = agent.extract_python_code(llm_output)
        self.assertEqual(
            code,
            "import pandas as pd\ndf = pd.read_csv('sales.csv')\nresult = int(df['total'].sum())",
        )

    def test_extract_python_code_generic_block(self):
        """Extract code from generic ``` ... ``` block."""
        agent = VLLMAgent(dry_run=True)
        llm_output = "```\nresult = 42\n```"
        code = agent.extract_python_code(llm_output)
        self.assertEqual(code, "result = 42")

    def test_extract_python_code_unclosed_block(self):
        """Extract code when LLM opens ```python but forgets closing backticks."""
        agent = VLLMAgent(dry_run=True)
        llm_output = "```python\nimport pandas as pd\nresult = 100"
        code = agent.extract_python_code(llm_output)
        self.assertIn("result = 100", code)

    def test_extract_python_code_empty_fallback(self):
        """Empty completion should safely return 'result = None'."""
        agent = VLLMAgent(dry_run=True)
        self.assertEqual(agent.extract_python_code(""), "result = None")
        self.assertEqual(agent.extract_python_code(None), "result = None")

    def test_mock_generation_executable_in_sandbox(self):
        """Ensure dry-run mock generated code runs cleanly in SandboxedExecutor."""
        agent = VLLMAgent(dry_run=True)
        code = agent.generate_code("Tính tổng doanh số")
        self.assertIn("result", code)

        executor = SandboxedExecutor()
        res = executor.execute(code)
        self.assertTrue(res.success)
        self.assertIsNone(res.result)  # Default abstain


if __name__ == "__main__":
    unittest.main()
