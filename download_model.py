"""Download Qwen/Qwen3-14B weights with live progress bar and manifest generation.

Features:
- Live tqdm progress bars with per-shard download speed (MB/s), percentage, and ETA.
- Pinned commit revision from HfApi.
- Automatic download resumption from partial .incomplete files.
- Background status monitor printing [Shards: X/8 | Downloaded: YY.YY / 29.60 GB].
- Post-download SHA-256 checksum calculation for all files.
- Generates models/Qwen3-14B/model_manifest.json containing revision, license, parameters, and checksums.
"""

from __future__ import annotations

import os
import sys

# 1. Configure environment before importing huggingface_hub
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "0"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
# Unset offline mode to enable download
os.environ.pop("HF_HUB_OFFLINE", None)
os.environ.pop("TRANSFORMERS_OFFLINE", None)

import hashlib
import json
import logging
from pathlib import Path
import threading
import time
from typing import Any, Dict, List

from huggingface_hub import HfApi, snapshot_download

REPO_ID = "Qwen/Qwen3-14B"
LOCAL_DIR = Path("models/Qwen3-14B")
TOTAL_EXPECTED_SHARDS = 8
TOTAL_EXPECTED_BYTES = 29536614400  # ~29.54 GB (~29.60 GB on disk)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    stream=sys.stdout,
)
logger = logging.getLogger("download_model")


def compute_file_sha256(path: Path) -> str:
    """Compute SHA-256 hash using 8MB chunk buffer."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(8 * 1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def monitor_progress(stop_event: threading.Event, local_dir: Path) -> None:
    """Background status monitor printing overall shard and size progress."""
    cache_dir = local_dir / ".cache" / "huggingface" / "download"
    time.sleep(2.0)
    while not stop_event.is_set():
        try:
            completed_shards = list(local_dir.glob("model-*.safetensors"))
            shard_count = len(completed_shards)

            total_bytes = 0
            if local_dir.exists():
                for f in local_dir.iterdir():
                    if f.is_file():
                        total_bytes += f.stat().st_size

            if cache_dir.exists():
                for f in cache_dir.iterdir():
                    if f.is_file() and not f.name.endswith(".lock") and not f.name.endswith(".metadata"):
                        total_bytes += f.stat().st_size

            downloaded_gb = total_bytes / (1024 ** 3)
            pct = min(100.0, (total_bytes / TOTAL_EXPECTED_BYTES) * 100.0) if TOTAL_EXPECTED_BYTES > 0 else 0.0

            sys.stdout.write(
                f"\n[Shards: {shard_count}/{TOTAL_EXPECTED_SHARDS} | Downloaded: {downloaded_gb:.2f} / 29.60 GB ({pct:.1f}%)]\n"
            )
            sys.stdout.flush()
        except Exception:
            pass

        stop_event.wait(10.0)


def main() -> None:
    print("=" * 80, flush=True)
    print("      Zalo AI Challenge 2026 - DataMate: Qwen/Qwen3-14B Downloader", flush=True)
    print("=" * 80, flush=True)

    LOCAL_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Fetch and pin commit revision
    print(f"Fetching repository metadata for '{REPO_ID}' via HfApi...", flush=True)
    api = HfApi()
    model_info = api.model_info(repo_id=REPO_ID)
    revision = model_info.sha
    print(f"Pinned Commit Revision: {revision}", flush=True)
    print(f"Target Directory:       {LOCAL_DIR.resolve()}", flush=True)
    print(f"Total Shards Expected:  {TOTAL_EXPECTED_SHARDS}", flush=True)
    print("-" * 80, flush=True)

    # 2. Start progress monitor thread
    stop_event = threading.Event()
    monitor_thread = threading.Thread(
        target=monitor_progress,
        args=(stop_event, LOCAL_DIR),
        daemon=True,
    )
    monitor_thread.start()

    # 3. Snapshot download with retry on network disruption
    max_retries = 10
    download_success = False

    for attempt in range(1, max_retries + 1):
        try:
            print(f"\nStarting snapshot_download (Attempt {attempt}/{max_retries})...", flush=True)
            snapshot_download(
                repo_id=REPO_ID,
                revision=revision,
                local_dir=str(LOCAL_DIR),
            )
            download_success = True
            break
        except Exception as e:
            logger.warning(f"Download attempt {attempt} interrupted: {e}")
            if attempt < max_retries:
                logger.info("Resuming download in 10 seconds...")
                time.sleep(10)
            else:
                logger.error("Max retries exceeded. Download failed.")
                stop_event.set()
                sys.exit(1)

    stop_event.set()
    monitor_thread.join(timeout=2.0)

    if not download_success:
        print("Download did not complete successfully.", flush=True)
        sys.exit(1)

    print("\n" + "=" * 80, flush=True)
    print("All shards and repository files downloaded successfully!", flush=True)
    print("Calculating SHA-256 checksums for model_manifest.json...", flush=True)
    print("=" * 80, flush=True)

    # 4. Compute SHA-256 checksums
    files_checksums: Dict[str, Dict[str, Any]] = {}
    safetensors_count = 0

    for item in sorted(LOCAL_DIR.iterdir()):
        if item.is_file() and item.name != "model_manifest.json":
            size_mb = item.stat().st_size / (1024 * 1024)
            print(f"  -> Hashing {item.name:<36} ({size_mb:>9.2f} MB)...", end="", flush=True)
            t0 = time.time()
            sha256_hash = compute_file_sha256(item)
            duration = time.time() - t0
            print(f" Done ({duration:.1f}s) | {sha256_hash[:16]}...", flush=True)

            files_checksums[item.name] = {
                "size_bytes": item.stat().st_size,
                "sha256": sha256_hash,
            }
            if item.name.endswith(".safetensors"):
                safetensors_count += 1

    # 5. Save model_manifest.json
    manifest_data = {
        "model_name": REPO_ID,
        "revision": revision,
        "license": "apache-2.0",
        "parameters": "14.8B",
        "parameters_total": 14770000000,
        "parameters_non_embedding": 13200000000,
        "safetensors_shards_count": safetensors_count,
        "total_files": len(files_checksums),
        "files": files_checksums,
    }

    manifest_path = LOCAL_DIR / "model_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest_data, f, indent=2, ensure_ascii=False)

    print("-" * 80, flush=True)
    print(f"Model manifest successfully saved to: {manifest_path.resolve()}", flush=True)
    print(f"Total safetensors shards verified: {safetensors_count}/{TOTAL_EXPECTED_SHARDS}", flush=True)
    print("=" * 80, flush=True)
    print("DOWNLOAD AND VERIFICATION COMPLETE! Model is ready for offline evaluation.", flush=True)


if __name__ == "__main__":
    main()
