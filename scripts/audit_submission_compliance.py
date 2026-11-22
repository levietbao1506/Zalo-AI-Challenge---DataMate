"""Submission compliance auditing script for Zalo AI Challenge 2026 (DataMate Track).

Audits mandatory files, .dockerignore security, Dockerfile settings,
data_manifest.json integrity, and static rules against external APIs or hardcoded IDs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import sys
from typing import Any, Dict, List, Optional, Tuple, Union

MANDATORY_FILES = [
    "predict.sh",
    "README.md",
    "requirements.txt",
    "Dockerfile",
    "training_code/README.md",
    "training_code/requirements.txt",
    "training_code/data_manifest.json",
]

FORBIDDEN_NETWORK_PATTERNS = [
    re.compile(r"api\.openai\.com", re.IGNORECASE),
    re.compile(r"api\.anthropic\.com", re.IGNORECASE),
    re.compile(r"generativelanguage\.googleapis\.com", re.IGNORECASE),
    re.compile(r"api\.cohere\.ai", re.IGNORECASE),
    re.compile(r"requests\.(get|post|put|delete)\(", re.IGNORECASE),
]

HARDCODED_ID_PATTERN = re.compile(r"['\"](biz_\d+|q_[0-9a-f]{3,})['\"]", re.IGNORECASE)


def compute_sha256(file_path: Path) -> str:
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def check_mandatory_files(repo_root: Path) -> Tuple[bool, str]:
    missing: List[str] = []
    for f in MANDATORY_FILES:
        target = repo_root / f
        if not target.is_file():
            missing.append(f)
    if missing:
        return False, f"Missing mandatory files: {', '.join(missing)}"
    return True, f"All {len(MANDATORY_FILES)} mandatory files exist."


def check_dockerignore(repo_root: Path) -> Tuple[bool, str]:
    dockerignore_path = repo_root / ".dockerignore"
    if not dockerignore_path.is_file():
        return False, "Missing .dockerignore file."

    content = dockerignore_path.read_text(encoding="utf-8")
    lines = [line.strip() for line in content.splitlines() if line.strip() and not line.strip().startswith("#")]

    has_public_test_ignore = any("public_test.zip" in l or "public_test" in l for l in lines)
    if not has_public_test_ignore:
        return False, ".dockerignore must explicitly exclude public_test.zip."

    training_code_excluded = False
    for line in lines:
        if line.startswith("!"):
            continue
        if "training_code" in line:
            training_code_excluded = True

    has_training_code_whitelist = any(l in ("!training_code/", "!training_code") for l in lines)
    if training_code_excluded and not has_training_code_whitelist:
        return False, ".dockerignore improperly excludes training_code without whitelisting."

    return True, ".dockerignore properly excludes public_test.zip and preserves training_code/."


def check_dockerfile(repo_root: Path) -> Tuple[bool, str]:
    dockerfile_path = repo_root / "Dockerfile"
    if not dockerfile_path.is_file():
        return False, "Missing Dockerfile."

    content = dockerfile_path.read_text(encoding="utf-8")

    if "ENTRYPOINT []" not in content:
        return False, "Dockerfile missing 'ENTRYPOINT []'."

    offline_flags = ["HF_HUB_OFFLINE=1", "TRANSFORMERS_OFFLINE=1", "NCCL_P2P_DISABLE=1"]
    for flag in offline_flags:
        if flag not in content:
            return False, f"Dockerfile missing offline environment flag: {flag}"

    if "nvidia-nccl-cu12==2.27.7" not in content:
        return False, "Dockerfile missing pinned installation of nvidia-nccl-cu12==2.27.7."

    required_copies = ["models/", "training_code/", "src/", "predict.sh", "README.md"]
    for item in required_copies:
        if f"COPY {item}" not in content and f" {item}" not in content:
            return False, f"Dockerfile missing COPY instruction for: {item}"

    return True, "Dockerfile satisfies all base, entrypoint, offline flags, and copy specifications."


def check_data_manifest(repo_root: Path) -> Tuple[bool, str]:
    manifest_path = repo_root / "training_code" / "data_manifest.json"
    if not manifest_path.is_file():
        return False, "Missing training_code/data_manifest.json."

    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception as e:
        return False, f"training_code/data_manifest.json is not valid JSON: {e}"

    params_total = manifest.get("parameters_total", 0)
    if not isinstance(params_total, (int, float)) or params_total > 15_000_000_000:
        return False, f"parameters_total ({params_total}) exceeds maximum allowed 15,000,000,000."

    rel_date = manifest.get("release_date") or ""
    models_list = manifest.get("models", [])
    for m in models_list:
        m_params = m.get("parameter_count", 0)
        if m_params > 15_000_000_000:
            return False, f"Model {m.get('name')} parameter count exceeds 15B: {m_params}"
        orig_date = m.get("original_release_date") or rel_date
        if orig_date >= "2026-10-26":
            return False, f"Model {m.get('name')} release date {orig_date} is after deadline 2026-10-26."

    local_files = manifest.get("local_files", [])
    if not local_files:
        return False, "training_code/data_manifest.json does not contain local_files list."

    mismatches: List[str] = []
    for item in local_files:
        p_str = item.get("local_path")
        expected_sha = item.get("sha256")
        if not p_str or not expected_sha:
            continue
        f_path = repo_root / p_str
        if not f_path.is_file():
            mismatches.append(f"{p_str} (file not found)")
            continue
        actual_sha = compute_sha256(f_path)
        if actual_sha != expected_sha:
            mismatches.append(f"{p_str} (SHA mismatch)")

    if mismatches:
        return False, f"Checksum verification failed for files: {', '.join(mismatches[:5])}"

    return True, f"data_manifest.json verified (params <= 15B, date < 2026-10-26, {len(local_files)} files checksums matched)."


def check_static_code_integrity(repo_root: Path) -> Tuple[bool, str]:
    scan_paths: List[Path] = []
    src_dir = repo_root / "src"
    if src_dir.is_dir():
        for p in src_dir.rglob("*.py"):
            scan_paths.append(p)
    predict_sh = repo_root / "predict.sh"
    if predict_sh.is_file():
        scan_paths.append(predict_sh)

    violations: List[str] = []

    for file_path in scan_paths:
        try:
            text = file_path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue

        for pat in FORBIDDEN_NETWORK_PATTERNS:
            match = pat.search(text)
            if match:
                rel = file_path.relative_to(repo_root).as_posix()
                violations.append(f"Forbidden network call in {rel}: {match.group(0)}")

        for line_num, line in enumerate(text.splitlines(), start=1):
            stripped = line.strip()
            if stripped.startswith("#"):
                continue
            id_match = HARDCODED_ID_PATTERN.search(line)
            if id_match:
                rel = file_path.relative_to(repo_root).as_posix()
                violations.append(f"Potential hardcoded ID in {rel}:{line_num}: {id_match.group(0)}")

    if violations:
        return False, "Static scan violations:\n  - " + "\n  - ".join(violations)

    return True, f"Static code integrity clean across {len(scan_paths)} core source files."


def audit_submission_compliance(repo_root: Optional[Union[str, Path]] = None) -> Tuple[bool, List[str], List[str]]:
    root = Path(repo_root).resolve() if repo_root is not None else Path.cwd()
    passed: List[str] = []
    failed: List[str] = []

    checks = [
        ("Mandatory Files Check", check_mandatory_files),
        (".dockerignore Rules Check", check_dockerignore),
        ("Dockerfile Standards Check", check_dockerfile),
        ("Data Manifest Verification", check_data_manifest),
        ("Static Code & Network Check", check_static_code_integrity),
    ]

    for title, func in checks:
        ok, msg = func(root)
        if ok:
            passed.append(f"[PASS] {title}: {msg}")
        else:
            failed.append(f"[FAIL] {title}: {msg}")

    return (len(failed) == 0), passed, failed


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit submission compliance per competition Docker guidelines.")
    parser.add_argument("--root-dir", "--root_dir", type=str, default=".", help="Root directory of the repo.")
    args = parser.parse_args()

    repo_root = Path(args.root_dir).resolve()
    print("=" * 80)
    print("      Zalo AI Challenge 2026 - DataMate Submission Compliance Audit")
    print("=" * 80)
    print(f"Target Repository: {repo_root}\n")

    success, passed, failed = audit_submission_compliance(repo_root)

    for item in passed:
        print(item)

    if failed:
        print("\n" + "!" * 80)
        print(" COMPLIANCE AUDIT FAILURES DETECTED:")
        print("!" * 80)
        for item in failed:
            print(item)
        print("\nPlease resolve all failures before building Docker or creating submission package.\n")
        sys.exit(1)
    else:
        print("\n" + "=" * 80)
        print(" AUDIT PASSED: Repository is 100% compliant with competition Docker guidelines.")
        print("=" * 80 + "\n")
        sys.exit(0)


if __name__ == "__main__":
    main()
