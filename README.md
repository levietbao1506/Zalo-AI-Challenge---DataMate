# Zalo AI Challenge 2026 - DataMate Track

Production-grade MVP scaffolding for **Vietnamese Natural-Language Question Answering over Tabular Data** (CSV, Excel XLSX, SQLite), designed to run **100% offline inside Docker containers** under strict competition constraints (<15B parameters, evaluated by **Penalized Accuracy**).

---

## 📁 Project Architecture

```text
zalo-datamate-2026/
├── pyproject.toml           # Project metadata, dependencies & pytest configuration
├── run_inference.sh         # Docker entrypoint runner with offline environment variables
├── src/
│   ├── __init__.py
│   ├── main.py              # Orchestrator: reads questions.jsonl, invokes agent, streams submission.jsonl
│   ├── agent.py             # Manages local vLLM engine, temperature=0.0, extracts clean Python code
│   ├── prompt_builder.py    # System prompt & few-shot instructions for Pandas/DuckDB script generation
│   ├── schema_parser.py     # Parses README.md, extracts head(5) and schema from CSV/XLSX/SQLite
│   ├── executor.py          # Sandboxed exec() environment with try-except, resource timeouts, and security traps
│   ├── calendar_utils.py    # Utilities for resolving as_of_date and Vietnamese relative/lunar dates
│   └── formatter.py         # Normalizes output per competition rules (NFC Unicode, 2 decimals, abstain fallback)
└── tests/
    ├── __init__.py
    ├── test_formatter.py    # Unit tests for NFC Unicode, float rounding, container unwrapping, and abstain
    └── test_sandbox.py      # Unit tests for sandboxed execution, timeouts, SQLite, and error trapping
```

---

## ⚡ Key Highlights & Competition Strategies

### 1. Penalized Accuracy Optimization
In competitive benchmarks evaluated with **Penalized Accuracy**, answering incorrectly carries a penalty, while abstaining (`null` / safe empty) yields 0.
- **Fail-Safe Abstain**: If a query is unanswerable, entities are missing from the data, or code execution fails even after self-correction, `src/formatter.py` cleanly outputs `null` to avoid penalties.

### 2. Tabular Support (CSV, XLSX, SQLite)
- **DuckDB + Pandas + SQLite3**: Fast SQL queries on tabular files and native DataFrame manipulation.
- **`src/schema_parser.py`**: Extracts `head(5)` sample records, data types, and `README.md` business context once and caches it to fit efficiently within the LLM context window.

### 3. Vietnamese Calendar & `as_of_date` Reasoning
- Resolves relative Vietnamese temporal expressions (`hôm nay`, `hôm qua`, `tuần trước`, `tháng này`, `quý 2 năm ngoái`, `năm nay`) relative to the anchor `as_of_date`.
- Supports Vietnamese Lunar Calendar conversions (`solar_to_lunar`, `get_tet_date`).

### 4. Deterministic Offline Inference (vLLM)
- Configured for `<15B` models (specifically `Qwen/Qwen3-14B`).
- Runs offline via vLLM with `tensor_parallel_size=2`, `gpu_memory_utilization=0.85`, and `max_model_len=32768`.
- Supports dual-mode sampling: greedy deterministic execution (`enable_thinking=False`, `temperature=0.0`) and chain-of-thought reasoning (`enable_thinking=True`, `temperature=0.6, top_p=0.95, top_k=20`).
- Includes automatic mock/dry-run fallback for local development without a GPU.

### 5. Training Code & Zero-Shot In-Context Learning Disclosure
- **Zero-Shot / Prompted In-Context Learning**: No model parameter fine-tuning was performed or required for this submission. The pipeline employs zero-shot and few-shot in-context learning with foundation weights from `Qwen/Qwen3-14B`.
- The `/code/training_code/` package structure provides complete dataset inspection, validation, and synthetic generation scripts (`training_code/src/validate_data.py`, `training_code/src/inspect_dataset.py`, `training_code/data_generation/generate_qa_pairs.py`), with model provenance and SHA-256 checksums documented in `training_code/data_manifest.json`.
- A training entrypoint is provided via `bash training_code/train.sh`.

---

## 🤖 Official Model Specification: Qwen/Qwen3-14B

The pipeline is standardized on **`Qwen/Qwen3-14B`** for offline evaluation in the Zalo AI Challenge 2026 (DataMate Track). Complete verification metadata is stored in `training_code/data_manifest.json`.

| Property | Value |
| :--- | :--- |
| **Model Repository** | [`Qwen/Qwen3-14B`](https://huggingface.co/Qwen/Qwen3-14B) |
| **Commit Hash** | `40c069824f4251a91eefaf281ebe4c544efd3e18` |
| **Release Date** | May 2025 |
| **License** | Apache-2.0 |
| **Total Parameter Count** | 14.8B (~14,770,000,000 parameters; 13.2B non-embedding) |
| **Architecture** | `Qwen3ForCausalLM` (`qwen3`) |
| **Context Length** | 32,768 native tokens (extendable to 131,072) |
| **Total Safetensors Size** | 29,536,614,400 bytes (~29.54 GB) |

### SHA-256 Checksums (`.safetensors` shards)

| File Name | Size (Bytes) | SHA-256 Checksum |
| :--- | :--- | :--- |
| `model-00001-of-00008.safetensors` | 3,841,788,544 | `e942bdbdf08857d16a8fef7d1dae9fceabeb4e84def6043485fe2f6f085dab0e` |
| `model-00002-of-00008.safetensors` | 3,963,750,816 | `f7c9c6eee628f5ad831d2d1d292e120505e5fcadeb38f88b4d3c4cb86306ccf9` |
| `model-00003-of-00008.safetensors` | 3,963,750,880 | `dfb8c5df9404b41ad6ae74e8b6b367135f017b4467b884cf71b17c71954f18a9` |
| `model-00004-of-00008.safetensors` | 3,963,750,880 | `eab286fec759e3e59ab228621aefa0fef14ed56039e06f959e67257d5af7604d` |
| `model-00005-of-00008.safetensors` | 3,963,750,880 | `97f0dc2992e59da95c466eff6f4fd0c8335843bbc36ed5c913a6f5150748c0e6` |
| `model-00006-of-00008.safetensors` | 3,963,750,880 | `9e8e76a013cd5e253865b792991e0b410f869b136b3c500079b531b09198e99e` |
| `model-00007-of-00008.safetensors` | 3,963,750,880 | `0aee70ee6e91dc00d818804fb47f124d13ee4ad5b4a64553e09dbf9391cd5750` |
| `model-00008-of-00008.safetensors` | 1,912,371,880 | `0d6b92296e326d39bbbaeb32c3ec454ac606da843d4c8ffa8edf010b62b8c9e0` |


---

## 🚀 Quickstart

### 1. Running Unit Tests
```bash
python -m unittest discover -s tests -v
# or with pytest
pytest tests/ -v
```

### 2. Local Dry-Run Test
```bash
python -m src.main \
  --data_dir ./data \
  --questions_file ./data/questions.jsonl \
  --output_file ./submission.jsonl \
  --dry_run
```

### 3. Running in Docker Container
```bash
chmod +x run_inference.sh
./run_inference.sh \
  --data_dir /app/data \
  --questions_file /app/data/questions.jsonl \
  --output_file /app/submission.jsonl \
  --model_path /app/model
```
