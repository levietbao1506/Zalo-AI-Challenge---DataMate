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
        self.requirements_path = self.root_dir / "requirements.txt"
        self.sample_data_dir = self.root_dir / "data" / "sample_dataset"

    def test_dockerfile_configuration(self):
        """Ensure Dockerfile complies with BTC rules: CUDA 12.8, /app workdir, offline env."""
        self.assertTrue(self.dockerfile_path.exists(), "Dockerfile must exist in workspace root")

        content = self.dockerfile_path.read_text(encoding="utf-8")

        # 1. Base image CUDA 12.8
        self.assertIn("nvidia/cuda:12.8", content, "Dockerfile must use CUDA 12.8 base image per BTC rules")

        # 2. Offline environment variables
        self.assertIn("HF_HUB_OFFLINE=1", content, "Must declare HF_HUB_OFFLINE=1 for offline rule")
        self.assertIn("TRANSFORMERS_OFFLINE=1", content, "Must declare TRANSFORMERS_OFFLINE=1")
        self.assertIn("PYTHONUNBUFFERED=1", content)
        self.assertIn("VLLM_WORKER_MULTIPROC_METHOD=spawn", content)
        self.assertIn("TOKENIZERS_PARALLELISM=false", content)

        # 3. Workdir /app and directory layout
        self.assertIn("WORKDIR /app", content)
        self.assertIn("/app/data", content)
        self.assertIn("/app/model", content)

        # 4. Entrypoint to runner script
        self.assertIn('ENTRYPOINT ["/app/run_inference.sh"]', content)

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
                self.assertIn("answer", rec)


if __name__ == "__main__":
    unittest.main()
