"""Unit tests for src/llm_config.py and training_code/data_manifest.json."""

import json
import os
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

from src.llm_config import (
    DEFAULT_MAX_MODEL_LEN,
    DEFAULT_OFFLINE_MODEL_PATH,
    DEFAULT_SEED,
    DEFAULT_TENSOR_PARALLEL_SIZE,
    NON_THINKING_TEMPERATURE,
    THINKING_TEMPERATURE,
    THINKING_TOP_K,
    THINKING_TOP_P,
    Qwen3OfflineEngine,
    VLLMConfig,
    detect_optimal_gpu_mode,
    extract_reasoning_and_content,
    get_sampling_params,
)


class TestLLMConfigAndManifest(unittest.TestCase):
    """Test suite for Qwen3-14B vLLM configuration, thinking toggle, and data manifest."""

    def setUp(self):
        self.root_dir = Path(__file__).resolve().parent.parent
        self.manifest_path = self.root_dir / "training_code" / "data_manifest.json"
        self.dockerignore_path = self.root_dir / ".dockerignore"
        self.gitignore_path = self.root_dir / ".gitignore"
        self.readme_path = self.root_dir / "README.md"

    def test_vllm_config_defaults(self):
        """Verify VLLMConfig defaults conform to competition specification."""
        cfg = VLLMConfig()
        self.assertEqual(cfg.model_path, "/code/models/Qwen3-14B")
        self.assertEqual(cfg.tensor_parallel_size, 2)
        self.assertEqual(cfg.gpu_memory_utilization, 0.85)
        self.assertEqual(cfg.max_model_len, 32768)
        self.assertEqual(cfg.seed, 42)
        self.assertTrue(cfg.local_files_only)
        self.assertTrue(cfg.trust_remote_code)
        self.assertFalse(cfg.enable_thinking)
        self.assertEqual(cfg.max_num_seqs, 8)
        self.assertTrue(cfg.enable_prefix_caching)
        self.assertTrue(cfg.to_engine_kwargs()["enable_prefix_caching"])

    def test_sampling_params_thinking_vs_non_thinking(self):
        """Verify sampling parameters for enable_thinking=True vs enable_thinking=False."""
        # Non-thinking mode: greedy deterministic decoding
        non_thinking_params = get_sampling_params(enable_thinking=False)
        temp = (
            getattr(non_thinking_params, "temperature", None)
            if hasattr(non_thinking_params, "temperature")
            else non_thinking_params["temperature"]
        )
        self.assertEqual(temp, 0.0)

        # Thinking mode remains greedy for reproducibility.
        thinking_params = get_sampling_params(enable_thinking=True)
        if hasattr(thinking_params, "temperature"):
            self.assertEqual(thinking_params.temperature, 0.0)
            self.assertEqual(thinking_params.top_p, 1.0)
            self.assertEqual(thinking_params.top_k, -1)
        else:
            self.assertEqual(thinking_params["temperature"], 0.0)
            self.assertEqual(thinking_params["top_p"], 1.0)
            self.assertEqual(thinking_params["top_k"], -1)

    def test_qwen3_offline_engine_dry_run_and_env(self):
        """Verify engine initialization sets offline env and supports dry-run inference."""
        engine = Qwen3OfflineEngine(dry_run=True)
        self.assertEqual(os.environ.get("HF_HUB_OFFLINE"), "1")
        self.assertEqual(os.environ.get("TRANSFORMERS_OFFLINE"), "1")

        outputs = engine.generate(["Write code to count rows"])
        self.assertEqual(len(outputs), 1)
        self.assertIn("```python", outputs[0])

    def test_gpu_mode_requires_quantization_and_under_22gb_for_dp(self):
        fake_torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: True, device_count=lambda: 2))
        with tempfile.TemporaryDirectory() as tmp_dir:
            model_dir = Path(tmp_dir)
            (model_dir / "config.json").write_text(
                json.dumps({"quantization_config": {"quant_method": "awq"}}),
                encoding="utf-8",
            )
            with patch("src.llm_config.torch", fake_torch), patch(
                "src.llm_config._HAS_TORCH", True
            ), patch("src.llm_config.estimate_model_weight_gb", return_value=18.0):
                self.assertEqual(detect_optimal_gpu_mode(model_dir), ("DP", 1, 2))
            with patch("src.llm_config.torch", fake_torch), patch(
                "src.llm_config._HAS_TORCH", True
            ), patch("src.llm_config.estimate_model_weight_gb", return_value=24.0):
                self.assertEqual(detect_optimal_gpu_mode(model_dir), ("TP", 2, 1))
            (model_dir / "config.json").write_text(
                json.dumps({"torch_dtype": "bfloat16"}),
                encoding="utf-8",
            )
            with patch("src.llm_config.torch", fake_torch), patch(
                "src.llm_config._HAS_TORCH", True
            ), patch("src.llm_config.estimate_model_weight_gb", return_value=18.0):
                self.assertEqual(detect_optimal_gpu_mode(model_dir), ("TP", 2, 1))

    def test_gpu_mode_falls_back_to_single_gpu(self):
        fake_torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: True, device_count=lambda: 1))
        with patch("src.llm_config.torch", fake_torch), patch("src.llm_config._HAS_TORCH", True):
            self.assertEqual(detect_optimal_gpu_mode("missing"), ("SINGLE_GPU", 1, 1))

    def test_reasoning_content_is_preserved_without_truncation(self):
        reasoning = "lập luận tiếng Việt " * 2000
        output = SimpleNamespace(
            outputs=[SimpleNamespace(text="```sql\nSELECT 1\n```", reasoning_content=reasoning)]
        )
        extracted, content = extract_reasoning_and_content(output)
        self.assertEqual(extracted, reasoning)
        self.assertEqual(content, "```sql\nSELECT 1\n```")
        tagged = f"<think>{reasoning}</think>answer"
        extracted_tagged, full = extract_reasoning_and_content(tagged)
        self.assertEqual(extracted_tagged, reasoning)
        self.assertEqual(full, tagged)

    def test_data_manifest_specification(self):
        """Verify training_code/data_manifest.json contains required Qwen3-14B metadata and checksums."""
        self.assertTrue(self.manifest_path.exists(), "training_code/data_manifest.json must exist")
        data = json.loads(self.manifest_path.read_text(encoding="utf-8"))

        self.assertEqual(data["model_name"], "Qwen/Qwen3-14B")
        self.assertEqual(data["commit_hash"], "40c069824f4251a91eefaf281ebe4c544efd3e18")
        self.assertEqual(data["release_date"], "May 2025")
        self.assertEqual(data["license"], "Apache-2.0")
        self.assertEqual(data["parameter_count"], "14.8B")

        # 8 safetensors shards
        shards = data["safetensors_files"]
        self.assertEqual(len(shards), 8)
        for i in range(1, 9):
            shard_name = f"model-0000{i}-of-00008.safetensors"
            self.assertIn(shard_name, shards)
            self.assertIn("sha256", shards[shard_name])
            self.assertEqual(len(shards[shard_name]["sha256"]), 64)

    def test_dockerignore_and_gitignore(self):
        """Verify .dockerignore does NOT exclude models/ or training_code/ and .gitignore includes models/."""
        self.assertTrue(self.gitignore_path.exists())
        gitignore_content = self.gitignore_path.read_text(encoding="utf-8")
        self.assertIn("models/", gitignore_content)

        self.assertTrue(self.dockerignore_path.exists())
        dockerignore_content = self.dockerignore_path.read_text(encoding="utf-8")
        # Ensure !models/ or models is not excluded
        self.assertIn("!models/", dockerignore_content)
        self.assertIn("!training_code/", dockerignore_content)

    def test_readme_contains_model_spec(self):
        """Verify README.md documents the Qwen3-14B commit, license, and SHA-256 table."""
        readme = self.readme_path.read_text(encoding="utf-8")
        self.assertIn("Qwen/Qwen3-14B", readme)
        self.assertIn("40c069824f4251a91eefaf281ebe4c544efd3e18", readme)
        self.assertIn("May 2025", readme)
        self.assertIn("Apache-2.0", readme)
        self.assertIn("14.8B", readme)
        self.assertIn("model-00001-of-00008.safetensors", readme)
        self.assertIn("e942bdbdf08857d16a8fef7d1dae9fceabeb4e84def6043485fe2f6f085dab0e", readme)


if __name__ == "__main__":
    unittest.main()
