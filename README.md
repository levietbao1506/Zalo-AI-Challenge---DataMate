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
- Configured for `<15B` models (e.g., `Qwen2.5-Coder-14B-Instruct`, `Qwen2.5-Coder-7B-Instruct`, `DeepSeek-R1-Distill-Qwen-14B`).
- Runs at `temperature=0.0` (greedy decoding) for reproducibility.
- Includes automatic mock/dry-run fallback for local development without a GPU.

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
