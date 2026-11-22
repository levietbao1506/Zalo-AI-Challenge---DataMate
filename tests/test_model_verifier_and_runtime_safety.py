from __future__ import annotations

import json
from pathlib import Path
import tempfile
import time
import unittest

from scripts.prepare_and_verify_model import (
    count_parameters_from_config,
    detect_quantization_format,
    estimate_weight_disk_gb,
    get_model_parameters,
    inspect_model,
)
from src.agent import CodeAgent, execute_adaptive_hybrid
from src.duckdb_store import BusinessDuckDBStore
from src.sql_executor import DuckDBSQLExecutor


class TestModelVerifierAndRuntimeSafety(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.test_dir = Path(self.temp_dir.name)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_dense_model_parameter_calculation_under_15b(self) -> None:
        qwen_14b_config = {
            "hidden_size": 5120,
            "num_hidden_layers": 48,
            "vocab_size": 152064,
            "intermediate_size": 13824,
            "num_attention_heads": 40,
            "num_key_value_heads": 8,
            "head_dim": 128,
            "tie_word_embeddings": False,
        }
        total_params = count_parameters_from_config(qwen_14b_config)
        self.assertLessEqual(total_params, 15_000_000_000)
        self.assertGreater(total_params, 14_000_000_000)

    def test_moe_model_parameter_calculation_exceeding_15b(self) -> None:
        moe_large_config = {
            "hidden_size": 4096,
            "num_hidden_layers": 32,
            "vocab_size": 128000,
            "intermediate_size": 11008,
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
            "head_dim": 128,
            "num_local_experts": 8,
            "moe_intermediate_size": 11008,
            "tie_word_embeddings": False,
        }
        total_params = count_parameters_from_config(moe_large_config)
        self.assertGreater(total_params, 15_000_000_000)

    def test_small_moe_model_under_15b(self) -> None:
        moe_small_config = {
            "hidden_size": 2048,
            "num_hidden_layers": 24,
            "vocab_size": 32000,
            "intermediate_size": 5632,
            "num_attention_heads": 16,
            "num_key_value_heads": 4,
            "head_dim": 128,
            "num_local_experts": 4,
            "moe_intermediate_size": 2816,
            "tie_word_embeddings": False,
        }
        total_params = count_parameters_from_config(moe_small_config)
        self.assertLessEqual(total_params, 15_000_000_000)

    def test_inspect_model_missing_files_error(self) -> None:
        empty_dir = self.test_dir / "empty_model"
        empty_dir.mkdir(parents=True)
        with self.assertRaises(FileNotFoundError) as ctx:
            inspect_model(empty_dir)
        self.assertIn("config.json", str(ctx.exception))

        (empty_dir / "config.json").write_text("{}", encoding="utf-8")
        with self.assertRaises(FileNotFoundError) as ctx:
            inspect_model(empty_dir)
        self.assertIn("tokenizer_config.json", str(ctx.exception))

        (empty_dir / "tokenizer_config.json").write_text("{}", encoding="utf-8")
        with self.assertRaises(FileNotFoundError) as ctx:
            inspect_model(empty_dir)
        self.assertIn("weight files", str(ctx.exception))

    def test_inspect_model_valid_and_over_limit(self) -> None:
        model_dir = self.test_dir / "mock_model"
        model_dir.mkdir(parents=True)

        config_data = {
            "hidden_size": 5120,
            "num_hidden_layers": 48,
            "vocab_size": 152064,
            "intermediate_size": 13824,
            "num_attention_heads": 40,
            "num_key_value_heads": 8,
            "torch_dtype": "bfloat16",
        }
        (model_dir / "config.json").write_text(json.dumps(config_data), encoding="utf-8")
        (model_dir / "tokenizer_config.json").write_text("{}", encoding="utf-8")
        (model_dir / "model.safetensors").write_bytes(b"\x00" * 1024)

        report = inspect_model(model_dir)
        self.assertTrue(report["is_under_15b"])
        self.assertEqual(report["forecast_mode"], "TP=2")
        self.assertEqual(report["worker_count"], 1)
        self.assertEqual(report["quant_format"], "bf16")
        self.assertGreaterEqual(report["kv_headroom_gb"], 0.0)

        over_config = dict(config_data)
        over_config["num_local_experts"] = 16
        (model_dir / "config.json").write_text(json.dumps(over_config), encoding="utf-8")

        with self.assertRaises(ValueError) as ctx:
            inspect_model(model_dir)
        self.assertIn("exceed competition limit", str(ctx.exception))

    def test_elapsed_ms_with_batch_entered_at(self) -> None:
        biz_dir = self.test_dir / "biz_elapsed"
        biz_dir.mkdir(parents=True)
        (biz_dir / "data.csv").write_text("id,val\n1,10\n2,20\n", encoding="utf-8")

        store = BusinessDuckDBStore(business_dir=biz_dir, calendar_csv_path=None)
        sql_executor = DuckDBSQLExecutor(store)

        q_record = {
            "id": "q_elapsed_test",
            "question": "Tính tổng val",
            "type": "number",
        }

        agent = CodeAgent(dry_run=True)
        batch_entered_at = time.perf_counter() - 0.250

        res = execute_adaptive_hybrid(
            question_record=q_record,
            duckdb_store=store,
            sql_executor=sql_executor,
            schema_card="CREATE TABLE data (id BIGINT, val BIGINT);",
            schema_meta={"tables": {"data": {"columns": ["id", "val"]}}},
            rulebook_text="quy định chuẩn",
            agent=agent,
            batch_entered_at=batch_entered_at,
        )

        sql_executor.close()
        store.close_and_cleanup()

        log_rec = res["log_record"]
        elapsed = log_rec["elapsed_ms"]
        self.assertIsInstance(elapsed, int)
        self.assertGreaterEqual(elapsed, 200)

    def test_elapsed_ms_default_without_batch_entered_at(self) -> None:
        biz_dir = self.test_dir / "biz_elapsed_def"
        biz_dir.mkdir(parents=True)
        (biz_dir / "data.csv").write_text("id,val\n1,10\n2,20\n", encoding="utf-8")

        store = BusinessDuckDBStore(business_dir=biz_dir, calendar_csv_path=None)
        sql_executor = DuckDBSQLExecutor(store)

        q_record = {
            "id": "q_elapsed_def",
            "question": "Tính tổng val",
            "type": "number",
        }

        agent = CodeAgent(dry_run=True)
        res = execute_adaptive_hybrid(
            question_record=q_record,
            duckdb_store=store,
            sql_executor=sql_executor,
            schema_card="CREATE TABLE data (id BIGINT, val BIGINT);",
            schema_meta={"tables": {"data": {"columns": ["id", "val"]}}},
            rulebook_text="quy định chuẩn",
            agent=agent,
        )

        sql_executor.close()
        store.close_and_cleanup()

        log_rec = res["log_record"]
        elapsed = log_rec["elapsed_ms"]
        self.assertIsInstance(elapsed, int)
        self.assertGreaterEqual(elapsed, 0)

    def test_predict_sh_contains_required_cache_and_offline_flags(self) -> None:
        predict_sh_path = Path("predict.sh")
        self.assertTrue(predict_sh_path.is_file())
        content = predict_sh_path.read_text(encoding="utf-8")

        required_strings = [
            "TMPDIR=/tmp",
            "XDG_CACHE_HOME=/tmp/.cache",
            "VLLM_CACHE_ROOT=/tmp/vllm_cache",
            "TRITON_CACHE_DIR=/tmp/triton_cache",
            "TORCHINDUCTOR_CACHE_DIR=/tmp/torchinductor_cache",
            "HF_HOME=/tmp/hf_cache",
            "TOKENIZERS_PARALLELISM=false",
            "HF_HUB_OFFLINE=1",
            "TRANSFORMERS_OFFLINE=1",
            "NCCL_P2P_DISABLE=1",
            "/dev/shm",
        ]

        for s in required_strings:
            self.assertIn(s, content, f"Missing required configuration '{s}' in predict.sh")

    def test_dockerfile_contains_required_cache_paths(self) -> None:
        dockerfile_path = Path("Dockerfile")
        self.assertTrue(dockerfile_path.is_file())
        content = dockerfile_path.read_text(encoding="utf-8")

        required_entries = [
            "TMPDIR=/tmp",
            "VLLM_CACHE_ROOT=/tmp/vllm_cache",
            "TRITON_CACHE_DIR=/tmp/triton_cache",
            "TORCHINDUCTOR_CACHE_DIR=/tmp/torchinductor_cache",
            "HF_HOME=/tmp/hf_cache",
            "mkdir -p /tmp/vllm_cache",
        ]

        for entry in required_entries:
            self.assertIn(entry, content, f"Missing required entry '{entry}' in Dockerfile")


if __name__ == "__main__":
    unittest.main()
