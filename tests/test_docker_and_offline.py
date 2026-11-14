"""Unit tests for Bước 7: Đóng gói Docker, Cấu hình run_inference.sh & Offline Testing."""

import json
from pathlib import Path
import socket
import sys
import tempfile
import unittest

from src.main import main


class TestDockerAndOffline(unittest.TestCase):
    """Test suite for Dockerfile compliance, runner script syntax, and offline resilience."""

    def setUp(self):
        self.root_dir = Path(__file__).resolve().parent.parent
        self.dockerfile_path = self.root_dir / "Dockerfile"
        self.runner_path = self.root_dir / "run_inference.sh"
        self.predict_path = self.root_dir / "predict.sh"
        self.requirements_path = self.root_dir / "requirements.txt"
        self.sample_data_dir = self.root_dir / "data" / "sample_dataset"

    def test_dockerfile_configuration(self):
        """Ensure Dockerfile complies with BTC rules: vLLM base, /code workdir, offline env, and predict entrypoint."""
        self.assertTrue(self.dockerfile_path.exists(), "Dockerfile must exist in workspace root")

        content = self.dockerfile_path.read_text(encoding="utf-8")

        # 1. Base image vLLM v0.10.1.1
        self.assertIn("vllm/vllm-openai:v0.10.1.1", content, "Dockerfile must use vllm/vllm-openai:v0.10.1.1 base image per BTC rules")

        # 2. Reset entrypoint
        self.assertIn("ENTRYPOINT []", content, "Must reset ENTRYPOINT [] to override base image API server")

        # 3. Offline environment variables
        self.assertIn("HF_HUB_OFFLINE=1", content, "Must declare HF_HUB_OFFLINE=1 for offline rule")
        self.assertIn("TRANSFORMERS_OFFLINE=1", content, "Must declare TRANSFORMERS_OFFLINE=1")
        self.assertIn("PIP_NO_CACHE_DIR=1", content, "Must declare PIP_NO_CACHE_DIR=1")
        self.assertIn("NCCL_P2P_DISABLE=1", content, "Must declare NCCL_P2P_DISABLE=1")
        self.assertIn("PYTHONUNBUFFERED=1", content)

        # 4. Workdir /code and required directory copies
        self.assertIn("WORKDIR /code", content)
        self.assertIn("COPY models/ /code/models/", content)
        self.assertIn("COPY training_code/ /code/training_code/", content)
        self.assertIn("COPY predict.sh /code/predict.sh", content)
        self.assertIn("COPY README.md /code/README.md", content)
        self.assertIn("COPY src/ /code/src/", content)

        # 5. NCCL pin and permissions
        self.assertIn("nvidia-nccl-cu12==2.27.7", content)
        self.assertIn("chmod +x /code/predict.sh", content)

        # 6. Default execution CMD
        self.assertIn('CMD ["bash", "/code/predict.sh"]', content)

    def test_run_inference_script_format_and_offline_exports(self):
        """Ensure run_inference.sh has Unix LF line endings, strict bash flags, and offline exports."""
        self.assertTrue(self.runner_path.exists(), "run_inference.sh must exist in workspace root")

        # Check binary contents for CRLF
        raw_bytes = self.runner_path.read_bytes()
        self.assertNotIn(b"\r\n", raw_bytes, "run_inference.sh must have Unix LF line endings (no Windows CRLF)")

        content = raw_bytes.decode("utf-8")

        # 1. Shebang & strict error trapping
        self.assertTrue(content.startswith("#!/usr/bin/env bash") or content.startswith("#!/bin/bash"))
        self.assertIn("set -euo pipefail", content)

        # 2. Offline environment exports
        self.assertIn("export HF_HUB_OFFLINE=1", content)
        self.assertIn("export TRANSFORMERS_OFFLINE=1", content)
        self.assertIn("export PYTHONUNBUFFERED=1", content)

        # 3. Argument handling
        self.assertIn("--data_dir", content)
        self.assertIn("--questions_file", content)
        self.assertIn("--output_file", content)
        self.assertIn("--model_path", content)
        self.assertIn("--timeout", content)
        self.assertIn("--max_retries", content)

    def test_requirements_completeness(self):
        """Ensure all required tabular and competition libraries are declared in requirements.txt."""
        self.assertTrue(self.requirements_path.exists())
        reqs = self.requirements_path.read_text(encoding="utf-8").lower()
        required_pkgs = [
            "vllm",
            "llama-cpp-python",
            "pandas",
            "duckdb",
            "lunarcalendar",
            "openpyxl",
            "pydantic",
            "requests",
        ]
        for pkg in required_pkgs:
            self.assertIn(pkg, reqs, f"Package '{pkg}' missing from requirements.txt")

    def test_offline_network_none_simulation(self):
        """Simulate strict --network none by intercepting sockets, ensuring pipeline runs offline."""
        orig_connect = socket.socket.connect

        def blocked_connect(self_sock, *args, **kwargs):
            raise OSError("Network is unreachable (--network none simulated)")

        # Block all outbound socket connections
        socket.socket.connect = blocked_connect

        with tempfile.TemporaryDirectory() as tmp_dir:
            out_file = Path(tmp_dir) / "offline_submission.jsonl"
            q_file = self.sample_data_dir / "questions.jsonl"

            test_args = [
                "main.py",
                "--data_dir", str(self.sample_data_dir),
                "--questions_file", str(q_file),
                "--output_file", str(out_file),
                "--dry_run",
            ]

            old_argv = sys.argv
            sys.argv = test_args
            try:
                main()
            finally:
                sys.argv = old_argv
                # Restore original socket connection
                socket.socket.connect = orig_connect

            # Verify submission was produced without network access
            self.assertTrue(out_file.exists())
            lines = out_file.read_text(encoding="utf-8").strip().splitlines()
            self.assertEqual(len(lines), 2)

            for line in lines:
                rec = json.loads(line)
                self.assertIn("id", rec)
                self.assertIn("type", rec)
                if rec["type"] == "abstain":
                    self.assertNotIn("answer", rec)
                else:
                    self.assertIn("answer", rec)

    def test_predict_script_specification(self):
        """Ensure predict.sh exists with Unix LF line endings and supports DataMate container arguments."""
        self.assertTrue(self.predict_path.exists(), "predict.sh must exist in workspace root")

        raw_bytes = self.predict_path.read_bytes()
        self.assertNotIn(b"\r\n", raw_bytes, "predict.sh must have Unix LF line endings (no Windows CRLF)")

        content = raw_bytes.decode("utf-8")
        self.assertTrue(content.startswith("#!/usr/bin/env bash") or content.startswith("#!/bin/bash"))
        self.assertIn("set -euo pipefail", content)
        self.assertIn("/data", content)
        self.assertIn("/result", content)
        self.assertIn("--data-dir", content)
        self.assertIn("--output-dir", content)
        self.assertIn("--limit", content)
        self.assertIn("export HF_HUB_OFFLINE=1", content)
        self.assertIn("export TRANSFORMERS_OFFLINE=1", content)

        # Check that run_inference.sh delegates to predict.sh
        runner_content = self.runner_path.read_text(encoding="utf-8")
        self.assertIn("predict.sh", runner_content)

    def test_strict_directory_rules_and_alias_omission(self):
        """Ensure /data is treated read-only, logs directory is created, alias is omitted, and --limit works."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            out_dir = Path(tmp_dir) / "custom_result"
            q_file = self.sample_data_dir / "questions.jsonl"

            # Create a mock question that artificially contains an 'alias' field
            mock_questions = Path(tmp_dir) / "questions_with_alias.jsonl"
            mock_questions.write_text(
                json.dumps({"id": "q_test", "question": "Có bao nhiêu dòng?", "alias": "should_be_omitted"}) + "\n"
                + json.dumps({"id": "q_test2", "question": "Có bao nhiêu sản phẩm?", "alias": "omit_me_too"}) + "\n",
                encoding="utf-8",
            )

            test_args = [
                "main.py",
                "--data_dir", str(self.sample_data_dir),
                "--questions_file", str(mock_questions),
                "--output_dir", str(out_dir),
                "--limit", "1",
                "--dry_run",
            ]

            old_argv = sys.argv
            sys.argv = test_args
            try:
                main()
            finally:
                sys.argv = old_argv

            # 1. Output file and logs directory created
            out_file = out_dir / "submission.jsonl"
            self.assertTrue(out_file.exists(), "submission.jsonl must be created inside output_dir")
            self.assertTrue((out_dir / "logs").is_dir(), "logs/ directory must be created inside output_dir")

            # 2. Limit verified (only 1 question processed)
            lines = out_file.read_text(encoding="utf-8").strip().splitlines()
            self.assertEqual(len(lines), 1, "Only 1 line should be produced due to --limit 1")

            # 3. Alias strictly omitted from output
            record = json.loads(lines[0])
            self.assertEqual(record["id"], "q_test")
            self.assertNotIn("alias", record, "Output record must omit 'alias' field per BTC rules")
            self.assertIn("type", record)
            if record["type"] == "abstain":
                self.assertNotIn("answer", record)
            else:
                self.assertIn("answer", record)

    def test_training_code_package_structure(self):
        """Ensure /code/training_code/ has all required directories, scripts, and zero-shot documentation."""
        tc_dir = self.root_dir / "training_code"
        self.assertTrue(tc_dir.is_dir(), "training_code/ directory must exist")

        required_files = [
            tc_dir / "README.md",
            tc_dir / "requirements.txt",
            tc_dir / "train.sh",
            tc_dir / "data_manifest.json",
            tc_dir / "configs" / "training_config.yaml",
            tc_dir / "configs" / "model_config.json",
            tc_dir / "src" / "__init__.py",
            tc_dir / "src" / "validate_data.py",
            tc_dir / "src" / "inspect_dataset.py",
            tc_dir / "data_generation" / "__init__.py",
            tc_dir / "data_generation" / "generate_qa_pairs.py",
            tc_dir / "data" / ".gitkeep",
            tc_dir / "logs" / ".gitkeep",
            tc_dir / "checkpoints" / "README.md",
        ]
        for f in required_files:
            self.assertTrue(f.exists(), f"Required file '{f.relative_to(self.root_dir)}' is missing")

        # Verify zero-shot documentation in root README and training_code README
        root_readme = (self.root_dir / "README.md").read_text(encoding="utf-8")
        tc_readme = (tc_dir / "README.md").read_text(encoding="utf-8")
        self.assertIn("Qwen/Qwen3-14B", root_readme)
        self.assertIn("Qwen/Qwen3-14B", tc_readme)
        self.assertIn("zero-shot", root_readme.lower())
        self.assertIn("zero-shot", tc_readme.lower())

        # Verify data_manifest.json integrity
        with open(tc_dir / "data_manifest.json", "r", encoding="utf-8") as f:
            manifest = json.load(f)
        self.assertEqual(manifest["model_name"], "Qwen/Qwen3-14B")
        self.assertEqual(manifest["commit_hash"], "40c069824f4251a91eefaf281ebe4c544efd3e18")
        self.assertEqual(len(manifest["safetensors_files"]), 8)


if __name__ == "__main__":
    unittest.main()
