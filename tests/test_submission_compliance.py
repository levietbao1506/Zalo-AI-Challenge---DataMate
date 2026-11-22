from __future__ import annotations

import json
from pathlib import Path
import shutil
import tempfile
import unittest

from scripts.audit_submission_compliance import (
    audit_submission_compliance,
    check_data_manifest,
    check_dockerfile,
    check_dockerignore,
    check_mandatory_files,
    check_static_code_integrity,
)
from training_code.src.generate_manifest import (
    compute_sha256,
    determine_category,
    determine_split,
    generate_manifest,
    write_manifest_file,
)


class TestSubmissionCompliance(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.work_root = Path(self.temp_dir.name)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_generate_manifest_structure_and_limits(self) -> None:
        manifest = generate_manifest(Path("."))
        self.assertIn("parameters_total", manifest)
        self.assertLessEqual(manifest["parameters_total"], 15_000_000_000)
        self.assertFalse(manifest["training_performed"])
        self.assertTrue(manifest["data_generation_performed"])
        self.assertIn("statement", manifest)
        self.assertTrue(len(manifest["statement"]) > 20)

        self.assertIn("models", manifest)
        self.assertTrue(len(manifest["models"]) >= 1)
        m0 = manifest["models"][0]
        self.assertEqual(m0["name"], "Qwen/Qwen3-14B")
        self.assertLess(m0["original_release_date"], "2026-10-26")
        self.assertLessEqual(m0["parameter_count"], 15_000_000_000)

        self.assertIn("local_files", manifest)
        self.assertTrue(len(manifest["local_files"]) > 5)

    def test_determine_category_and_split(self) -> None:
        self.assertEqual(determine_category("training_code/src/validate.py", ".py"), "script")
        self.assertEqual(determine_category("training_code/configs/cfg.yaml", ".yaml"), "config")
        self.assertEqual(determine_category("training_code/data/set.jsonl", ".jsonl"), "data")
        self.assertEqual(determine_category("training_code/checkpoints/m.bin", ".bin"), "checkpoint")
        self.assertEqual(determine_category("training_code/logs/run.log", ".log"), "log")

        self.assertEqual(determine_split("train_data.jsonl"), "train")
        self.assertEqual(determine_split("val_data.csv"), "val")
        self.assertEqual(determine_split("synthetic_qa.jsonl"), "synthetic")

    def test_manifest_write_and_reload(self) -> None:
        manifest = generate_manifest(Path("."))
        target_path = self.work_root / "manifest.json"
        write_manifest_file(manifest, target_path)
        self.assertTrue(target_path.is_file())

        reloaded = json.loads(target_path.read_text(encoding="utf-8"))
        self.assertEqual(reloaded["parameters_total"], manifest["parameters_total"])
        self.assertEqual(len(reloaded["local_files"]), len(manifest["local_files"]))

    def test_audit_submission_compliance_on_current_repo_passes(self) -> None:
        success, passed, failed = audit_submission_compliance(Path("."))
        self.assertTrue(success, f"Audit failed with errors: {failed}")
        self.assertEqual(len(failed), 0)
        self.assertEqual(len(passed), 5)

    def test_audit_detects_missing_public_test_in_dockerignore(self) -> None:
        fake_repo = self.work_root / "fake_repo1"
        fake_repo.mkdir()
        (fake_repo / ".dockerignore").write_text("*.tmp\nbuild/\n!training_code/\n", encoding="utf-8")

        ok, msg = check_dockerignore(fake_repo)
        self.assertFalse(ok)
        self.assertIn("public_test.zip", msg)

    def test_audit_detects_manifest_parameters_exceeding_15b(self) -> None:
        fake_repo = self.work_root / "fake_repo2"
        fake_tc = fake_repo / "training_code"
        fake_tc.mkdir(parents=True)

        bad_manifest = {
            "parameters_total": 20_000_000_000,
            "models": [{"name": "TooBig", "parameter_count": 20_000_000_000, "original_release_date": "2025-01-01"}],
            "local_files": [],
        }
        (fake_tc / "data_manifest.json").write_text(json.dumps(bad_manifest), encoding="utf-8")

        ok, msg = check_data_manifest(fake_repo)
        self.assertFalse(ok)
        self.assertIn("exceeds maximum allowed 15,000,000,000", msg)

    def test_audit_detects_checksum_mismatch_in_manifest(self) -> None:
        fake_repo = self.work_root / "fake_repo3"
        fake_tc = fake_repo / "training_code"
        fake_tc.mkdir(parents=True)

        sample_file = fake_tc / "sample.py"
        sample_file.write_text("print('hello')", encoding="utf-8")

        bad_manifest = {
            "parameters_total": 14_000_000_000,
            "models": [{"name": "M", "parameter_count": 14_000_000_000, "original_release_date": "2025-01-01"}],
            "local_files": [
                {
                    "local_path": "training_code/sample.py",
                    "sha256": "0000000000000000000000000000000000000000000000000000000000000000",
                }
            ],
        }
        (fake_tc / "data_manifest.json").write_text(json.dumps(bad_manifest), encoding="utf-8")

        ok, msg = check_data_manifest(fake_repo)
        self.assertFalse(ok)
        self.assertIn("Checksum verification failed", msg)

    def test_audit_detects_hardcoded_biz_id(self) -> None:
        fake_repo = self.work_root / "fake_repo4"
        fake_src = fake_repo / "src"
        fake_src.mkdir(parents=True)

        bad_file = fake_src / "hardcode_test.py"
        bad_file.write_text("def solve():\n    biz_id = 'biz_017'\n    return biz_id\n", encoding="utf-8")

        ok, msg = check_static_code_integrity(fake_repo)
        self.assertFalse(ok)
        self.assertIn("Potential hardcoded ID", msg)
        self.assertIn("biz_017", msg)

    def test_audit_detects_forbidden_network_call(self) -> None:
        fake_repo = self.work_root / "fake_repo5"
        fake_src = fake_repo / "src"
        fake_src.mkdir(parents=True)

        bad_file = fake_src / "network_test.py"
        bad_file.write_text("import os\nurl = 'https://api.openai.com/v1'\n", encoding="utf-8")

        ok, msg = check_static_code_integrity(fake_repo)
        self.assertFalse(ok)
        self.assertIn("Forbidden network call", msg)


if __name__ == "__main__":
    unittest.main()
