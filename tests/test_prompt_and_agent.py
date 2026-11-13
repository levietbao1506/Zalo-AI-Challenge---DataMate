"""Unit tests for Step 3: LLM Engine (vLLM Agent) and PromptBuilder."""

from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

from src.agent import (
    DEFAULT_OLLAMA_URL,
    DEFAULT_TEMPERATURE,
    CodeAgent,
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

        # Ví dụ 4: Excel với sheet
        self.assertIn("Đọc file Excel nhiều sheets", FEW_SHOT_EXAMPLES)
        self.assertIn("sheet_name=\"Orders\"", FEW_SHOT_EXAMPLES)

        # Ví dụ 5: Penalized Accuracy safe abstain
        self.assertIn("Trường hợp dữ liệu không tồn tại -> Abstain để tránh bị trừ điểm", FEW_SHOT_EXAMPLES)
        self.assertIn("result = None", FEW_SHOT_EXAMPLES)

    def test_chain_of_thought_and_unanswerable_prompts(self):
        """Ensure SYSTEM_PROMPT and all FEW_SHOT_EXAMPLES enforce CoT and Unanswerable rules."""
        import re

        # 1. System Prompt CoT and Unanswerable requirements
        self.assertIn(
            "Trước khi viết code, bạn BẮT BUỘC phải viết 1-2 dòng comment bắt đầu bằng `# Phân tích: ` để giải thích ngắn gọn cách bạn sẽ lọc dữ liệu và tính toán.",
            SYSTEM_PROMPT,
        )
        self.assertIn(
            "Nếu lược đồ dữ liệu (schema) hoàn toàn không có bảng hoặc cột thông tin nào liên quan để trả lời câu hỏi, không được đoán mò. Hãy viết comment phân tích và gán `final_result = \"unanswerable\"`.",
            SYSTEM_PROMPT,
        )

        # 2. Example 3 (Unanswerable) presence in FEW_SHOT_EXAMPLES
        self.assertIn("cho biết tên nhân viên bán được nhiều hàng nhất", FEW_SHOT_EXAMPLES)
        self.assertIn('final_result = "unanswerable"', FEW_SHOT_EXAMPLES)

        # 3. Every python block in FEW_SHOT_EXAMPLES must start with "# Phân tích: "
        code_blocks = re.findall(r"```python\n(.*?)```", FEW_SHOT_EXAMPLES, re.DOTALL)
        self.assertGreaterEqual(len(code_blocks), 5)
        for block in code_blocks:
            self.assertTrue(
                block.lstrip().startswith("# Phân tích: "),
                f"Code block does not start with '# Phân tích: ':\n{block[:80]}",
            )

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

    def test_critical_rules_and_error_prevention_prompts(self):
        """Ensure SYSTEM_PROMPT contains CRITICAL RULES, strict output format, and error prevention."""
        self.assertIn("CRITICAL RULES", SYSTEM_PROMPT)
        self.assertIn("VI PHẠM BẤT KỲ QUY TẮC NÀO DƯỚI ĐÂY SẼ LÀM HỆ THỐNG CRASH", SYSTEM_PROMPT)
        self.assertIn("TUYỆT ĐỐI CẤM", SYSTEM_PROMPT)
        self.assertIn("TUYỆT ĐỐI KHÔNG tự bịa", SYSTEM_PROMPT)
        self.assertIn(".fillna()", SYSTEM_PROMPT)
        self.assertIn("df.empty", SYSTEM_PROMPT)
        self.assertIn("final_result", SYSTEM_PROMPT)

        user_prompt = self.builder.build_user_prompt(
            question="Doanh thu?",
            schema_context="Table: orders.csv",
        )
        self.assertIn("CRITICAL RULES", user_prompt)
        self.assertIn("TRÁNH CRASH HỆ THỐNG", user_prompt)


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

    def test_code_agent_batch_generation_preserves_order(self):
        """Verify CodeAgent.generate_code_batch passes prompt list to llm.generate and preserves order."""
        agent = CodeAgent(dry_run=True)
        self.assertIs(VLLMAgent, CodeAgent)
        self.assertEqual(agent.generate_code_batch([]), [])

        # Simulate active vLLM engine via mock
        agent.dry_run = False
        agent.sampling_params = object()
        agent.llm = MagicMock()
        agent.llm.generate.return_value = [
            SimpleNamespace(outputs=[SimpleNamespace(text="```python\nfinal_result = 1\n```")]),
            SimpleNamespace(outputs=[SimpleNamespace(text="```python\nfinal_result = 2\n```")]),
            SimpleNamespace(outputs=[SimpleNamespace(text="```python\nfinal_result = 3\n```")]),
        ]

        prompts = ["prompt_0", "prompt_1", "prompt_2"]
        results = agent.generate_code_batch(prompts)

        agent.llm.generate.assert_called_once_with(prompts, agent.sampling_params)
        self.assertEqual(
            results,
            [
                "```python\nfinal_result = 1\n```",
                "```python\nfinal_result = 2\n```",
                "```python\nfinal_result = 3\n```",
            ],
        )

    def test_ollama_backend_initialization(self):
        """Verify CodeAgent detects 'ollama:' prefix, extracts model name, and disables vLLM fallback."""
        agent = CodeAgent("ollama:llama3.2:1b")
        self.assertEqual(agent.backend, "ollama")
        self.assertEqual(agent.model_name, "llama3.2:1b")
        self.assertFalse(agent.dry_run)
        self.assertIsNone(agent.llm)

        vllm_agent = CodeAgent("./models/Qwen2.5-Coder-0.5B-Instruct", dry_run=True)
        self.assertEqual(vllm_agent.backend, "vllm")

        with self.assertRaises(ValueError):
            CodeAgent("ollama:")

    def test_ollama_generate_code_posts_expected_payload(self):
        """Verify Ollama backend sends expected POST JSON payload and returns response['response']."""
        agent = CodeAgent("ollama:llama3.2:1b")
        mock_response = MagicMock()
        mock_response.json.return_value = {"response": "```python\nfinal_result = 42\n```"}

        with patch("src.agent.requests.post", return_value=mock_response) as mock_post:
            output = agent.generate_code("Tính tổng doanh thu")

        self.assertEqual(output, "```python\nfinal_result = 42\n```")
        mock_post.assert_called_once_with(
            DEFAULT_OLLAMA_URL,
            json={
                "model": "llama3.2:1b",
                "prompt": "Tính tổng doanh thu",
                "stream": False,
                "options": {"temperature": 0.0},
            },
            timeout=None,
        )
        mock_response.raise_for_status.assert_called_once()

    def test_ollama_generate_code_with_system_prompt(self):
        """Verify Ollama backend forwards optional system_prompt in payload."""
        agent = CodeAgent("ollama:llama3.2:1b")
        mock_response = MagicMock()
        mock_response.json.return_value = {"response": "final_result = 1"}

        with patch("src.agent.requests.post", return_value=mock_response) as mock_post:
            output = agent.generate_code("Q?", system_prompt="SYS")

        self.assertEqual(output, "final_result = 1")
        mock_post.assert_called_once_with(
            DEFAULT_OLLAMA_URL,
            json={
                "model": "llama3.2:1b",
                "prompt": "Q?",
                "stream": False,
                "options": {"temperature": 0.0},
                "system": "SYS",
            },
            timeout=None,
        )

    def test_ollama_generate_code_batch_sequential(self):
        """Verify generate_code_batch calls generate_code sequentially for Ollama backend."""
        agent = CodeAgent("ollama:llama3.2:1b")
        self.assertEqual(agent.generate_code_batch([]), [])

        with patch.object(agent, "generate_code", side_effect=["resp_0", "resp_1", "resp_2"]) as mock_gen:
            results = agent.generate_code_batch(["prompt_0", "prompt_1", "prompt_2"])

        self.assertEqual(results, ["resp_0", "resp_1", "resp_2"])
        self.assertEqual(mock_gen.call_count, 3)
        mock_gen.assert_any_call("prompt_0")
        mock_gen.assert_any_call("prompt_1")
        mock_gen.assert_any_call("prompt_2")

    def test_vllm_directory_initialization_flags(self):
        """Verify standard local directory defaults to vLLM with trust_remote_code=True and tensor_parallel_size=1."""
        fake_llm_instance = MagicMock()
        fake_sampling = MagicMock()

        with (
            patch("src.agent._HAS_VLLM", True),
            patch("src.agent.LLM", return_value=fake_llm_instance) as mock_llm_cls,
            patch("src.agent.SamplingParams", return_value=fake_sampling) as mock_sp_cls,
        ):
            agent = CodeAgent("./models/Qwen2.5-Coder-3B-Instruct")

        self.assertEqual(agent.backend, "vllm")
        self.assertFalse(agent.dry_run)
        self.assertIs(agent.llm, fake_llm_instance)
        self.assertIs(agent.sampling_params, fake_sampling)
        mock_llm_cls.assert_called_once()
        call_kwargs = mock_llm_cls.call_args.kwargs
        self.assertEqual(call_kwargs["model"], "./models/Qwen2.5-Coder-3B-Instruct")
        self.assertTrue(call_kwargs["trust_remote_code"])
        self.assertEqual(call_kwargs["tensor_parallel_size"], 1)
        self.assertEqual(call_kwargs["max_model_len"], 4096)
        self.assertEqual(call_kwargs["gpu_memory_utilization"], 0.85)
        mock_sp_cls.assert_called_once()

    def test_llama_cpp_gguf_initialization_and_generation(self):
        """Verify .gguf model_path initializes llama_cpp backend, generates code via chat completion, and batches sequentially."""
        gguf_path = "./models/Qwen-3B-GGUF/qwen2.5-coder-3b-instruct-q4_k_m.gguf"
        mock_llama_instance = MagicMock()
        mock_llama_instance.create_chat_completion.return_value = {
            "choices": [{"message": {"content": "```python\nfinal_result = 7\n```"}}]
        }

        with patch("src.agent.Llama", return_value=mock_llama_instance) as mock_llama_cls:
            agent = CodeAgent(gguf_path)

        self.assertEqual(agent.backend, "llama_cpp")
        self.assertFalse(agent.dry_run)
        self.assertIs(agent.llm, mock_llama_instance)
        mock_llama_cls.assert_called_once_with(
            model_path=gguf_path,
            n_ctx=8192,
            verbose=False,
        )

        # Single generation without system_prompt
        out = agent.generate_code("Tính tổng?")
        self.assertEqual(out, "```python\nfinal_result = 7\n```")
        mock_llama_instance.create_chat_completion.assert_called_once_with(
            messages=[
                {
                    "role": "system",
                    "content": "You are a Python Data Analyst. Output ONLY valid executable pandas code inside a ```python block. Do not provide any explanations.",
                },
                {"role": "user", "content": "Tính tổng?"},
            ],
            max_tokens=1024,
            temperature=0.0,
        )

        # Sequential batch generation
        self.assertEqual(agent.generate_code_batch([]), [])
        with patch.object(agent, "generate_code", side_effect=["r0", "r1"]) as mock_gen:
            batch_out = agent.generate_code_batch(["p0", "p1"])
        self.assertEqual(batch_out, ["r0", "r1"])
        self.assertEqual(mock_gen.call_count, 2)
        mock_gen.assert_any_call("p0")
        mock_gen.assert_any_call("p1")

        # Error handling returns ""
        mock_llama_instance.create_chat_completion.side_effect = Exception("CUDA error")
        err_out = agent.generate_code("Error test")
        self.assertEqual(err_out, "")


if __name__ == "__main__":
    unittest.main()
