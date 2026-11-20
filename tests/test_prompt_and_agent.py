"""Unit tests for Step 3: LLM Engine (vLLM Agent) and PromptBuilder."""

from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

from src.agent import (
    DEFAULT_TEMPERATURE,
    CodeAgent,
    VLLMAgent,
    extract_thinking,
    verification_gate,
)
from src.executor import SandboxedExecutor
from src import executor as executor_module
from src.main import _format_chatml_prompt
from src.prompt_builder import (
    FEW_SHOT_EXAMPLES,
    SYSTEM_PROMPT,
    PromptBuilder,
)


class TestPromptArchitecture(unittest.TestCase):
    """Prompt architecture checks; no local model is loaded."""

    def setUp(self):
        self.builder = PromptBuilder(rulebook_content="Doanh thu chỉ gồm đơn COMPLETED.")

    def test_static_prefix_is_stable_and_ordered(self):
        kw = dict(business_id="biz_1", readme_content="README của biz_1",
                  schema_context="Table: orders.csv\n- status (string)")
        first = self.builder.build_system_prompt(**kw)
        self.assertIs(first, self.builder.build_system_prompt(**kw))
        self.assertLess(first.index("RULEBOOK.MD (ACTIVE SPLIT)"), first.index("BUSINESS README.MD"))
        self.assertLess(first.index("BUSINESS README.MD"), first.index("BUSINESS SCHEMA PROFILE"))
        self.assertIn("Doanh thu chỉ gồm đơn COMPLETED", first)
        self.assertIn("README của biz_1", first)
        self.assertIn("Table: orders.csv", first)
        self.assertNotIn("2025-08-04", first)
        other = self.builder.build_system_prompt(**{**kw, "business_id": "biz_2", "readme_content": "README khác"})
        self.assertNotEqual(first, other)

    def test_dynamic_query_suffix_and_time_context(self):
        one = self.builder.build_user_prompt("Doanh thu hôm qua?", schema_context="Table: orders.csv",
                                             as_of_date="2024-06-19")
        two = self.builder.build_user_prompt("Số đơn hôm qua?", as_of_date="2024-06-20")
        self.assertNotEqual(one, two)
        self.assertNotIn("Table: orders.csv", one)
        self.assertNotIn("Doanh thu chỉ gồm đơn COMPLETED", one)
        self.assertIn('"yesterday":"2024-06-18"', one)
        self.assertIn("CÂU HỎI: Doanh thu hôm qua?", one)
        self.assertLess(one.index("as_of_date"), one.index("TIME_CONTEXT_ISO"))

    def test_batch_chatml_keeps_common_prefix(self):
        prefix = self.builder.build_system_prompt(
            business_id="biz_1", readme_content="README", schema_context="Table: sales.csv")
        queries = [self.builder.build_user_prompt(q, as_of_date="2024-06-19")
                   for q in ("Doanh thu?", "Số đơn?")]
        prompts = [_format_chatml_prompt(prefix, query) for query in queries]
        common = f"<|im_start|>system\n{prefix}<|im_end|>\n<|im_start|>user\n"
        self.assertTrue(all(prompt.startswith(common) for prompt in prompts))
        self.assertNotEqual(prompts[0], prompts[1])

    def test_rules_and_output_contract(self):
        for phrase in ("Code-first", "Rulebook fidelity", "Pha Inspect", "Pha Compute & Verify",
                       "unanswerable", "abstain", "top-k", "tie-break", "-12.45", "FILE_MAP",
                       "kh=khách hàng", "number, percent, string, date, boolean, list, table,"):
            self.assertIn(phrase, SYSTEM_PROMPT)
        self.assertIn('final_result = "unanswerable"', SYSTEM_PROMPT)
        self.assertIn('final_result = "abstain"', SYSTEM_PROMPT)
        self.assertIn('"columns":["Tên"]', SYSTEM_PROMPT)
        self.assertEqual(FEW_SHOT_EXAMPLES, "")

    def test_retry_and_optional_verification(self):
        retry = self.builder.build_retry_prompt("Đếm đơn?", "df['missing']", "KeyError: missing",
                                                previous_user_prompt="TIME_CONTEXT_ISO: {}")
        self.assertIn("KeyError: missing", retry)
        self.assertIn("df['missing']", retry)
        self.assertIn("TIME_CONTEXT_ISO: {}", retry)
        self.assertIn("không lặp lại cùng code", retry)
        self.assertIn("rỗng/NaN", self.builder.build_retry_prompt("Đếm đơn?", "final_result = None", ""))
        self.assertIsNone(self.builder.build_verification_prompt("Đếm đơn?", 4))
        enabled = PromptBuilder(enable_verification=True)
        verify = enabled.build_verification_prompt("Đếm đơn?", 4, "2024-06-19")
        self.assertIn("COMPUTED ANSWER: 4", verify)
        self.assertIn("RULEBOOK.MD", verify)
        self.assertIn("2024-06-19", verify)

    def test_retry_builder_is_used_after_each_failure(self):
        generated_prompts = []
        def generate(prompt):
            generated_prompts.append(prompt)
            return "```python\nfinal_result = 3\n```"
        feedback = lambda base, code, error: self.builder.build_retry_prompt(
            "Đếm đơn?", code, error, previous_user_prompt=base)
        with patch.object(executor_module, "execute_local", side_effect=[
            (False, None, "KeyError: status"), (False, None, "ValueError: date"), (True, 3, "")
        ]):
            result = executor_module.execute_with_retry(
                "CÂU HỎI: Đếm đơn?", generate, ".", max_retries=3,
                retry_prompt_fn=feedback,
            )
        self.assertTrue(result[0])
        self.assertIn("KeyError: status", generated_prompts[1])
        self.assertIn("ValueError: date", generated_prompts[2])


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

    def test_vllm_directory_initialization_flags(self):
        """Verify standard local directory defaults to vLLM with trust_remote_code=True and tensor_parallel_size=1."""
        fake_llm_instance = MagicMock()
        fake_sampling = MagicMock()

        with (
            patch.dict("os.environ", {"TENSOR_PARALLEL_SIZE": ""}),
            patch("src.agent._HAS_VLLM", True),
            patch("src.agent.LLM", return_value=fake_llm_instance) as mock_llm_cls,
            patch("src.agent.SamplingParams", return_value=fake_sampling) as mock_sp_cls,
        ):
            agent = CodeAgent("./models/Qwen3-14B", tensor_parallel_size=1)

        self.assertEqual(agent.backend, "vllm")
        self.assertFalse(agent.dry_run)
        self.assertIs(agent.llm, fake_llm_instance)
        self.assertIs(agent.sampling_params, fake_sampling)
        mock_llm_cls.assert_called_once()
        call_kwargs = mock_llm_cls.call_args.kwargs
        self.assertEqual(call_kwargs["model"], "./models/Qwen3-14B")
        self.assertTrue(call_kwargs["trust_remote_code"])
        self.assertEqual(call_kwargs["tensor_parallel_size"], 1)
        self.assertEqual(call_kwargs["max_model_len"], 4096)
        self.assertEqual(call_kwargs["gpu_memory_utilization"], 0.85)
        mock_sp_cls.assert_called_once()

    def test_llama_cpp_gguf_initialization_and_generation(self):
        """Verify .gguf model_path initializes llama_cpp backend, generates code via chat completion, and batches sequentially."""
        gguf_path = "./models/mock-model.gguf"
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
        with patch.object(agent, "generate_code", side_effect=["r0", "r1"]) as mock_gen, patch("builtins.print"):
            batch_out = agent.generate_code_batch(["p0", "p1"])
        self.assertEqual(batch_out, ["r0", "r1"])
        self.assertEqual(mock_gen.call_count, 2)
        mock_gen.assert_any_call("p0")
        mock_gen.assert_any_call("p1")

        # Error handling returns ""
        mock_llama_instance.create_chat_completion.side_effect = Exception("CUDA error")
        with patch("builtins.print"):
            err_out = agent.generate_code("Error test")
        self.assertEqual(err_out, "")

    def test_extract_thinking_helper(self):
        """Test extraction of chain-of-thought reasoning inside <think>...</think> tags."""
        raw_output = "<think>Phân tích câu hỏi: cần tính doanh thu tháng 5.</think>```python\nresult = 100\n```"
        thinking = extract_thinking(raw_output)
        self.assertEqual(thinking, "Phân tích câu hỏi: cần tính doanh thu tháng 5.")

        # No thinking tags
        self.assertIsNone(extract_thinking("```python\nresult = 100\n```"))
        self.assertEqual(extract_thinking("<think>  raw\nline  </think>"), "  raw\nline  ")

    def test_agent_verification_gate_pa_guard(self):
        """Verify agent's verification_gate guards against negative PA scores."""
        agent = CodeAgent(dry_run=True, abstain_threshold=0.6)

        # 1. Successful execution with valid answer
        res_ok = agent.verify_prediction(execution_success=True, raw_result=150.0, q_type="number", code_str="final_result = 150.0")
        self.assertTrue(res_ok["passed"])
        self.assertEqual(res_ok["type"], "number")
        self.assertEqual(res_ok["answer"], 150.0)

        # 2. Conflicting retries drop confidence below threshold -> abstain
        res_conflict = agent.verify_prediction(
            execution_success=True,
            raw_result=200,
            q_type="number",
            retry_results=[100, 200],
            code_str="final_result = 200",
        )
        self.assertEqual(res_conflict["type"], "abstain")
        self.assertEqual(res_conflict["answer"], "abstain")
        self.assertFalse(res_conflict["passed"])

        # 3. Execution failure after retries -> abstain
        res_err = agent.verify_prediction(
            execution_success=False,
            raw_result=None,
            q_type="number",
            error_msg="KeyError: 'revenue'",
        )
        self.assertEqual(res_err["type"], "abstain")
        self.assertEqual(res_err["answer"], "abstain")


if __name__ == "__main__":
    unittest.main()
