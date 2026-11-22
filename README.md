# Zalo AI Challenge 2026 - DataMate Track

Production-grade, offline-first tabular question answering system designed for the **Zalo AI Challenge 2026 (DataMate Track)**. The system operates 100% offline inside Docker containers on dual NVIDIA RTX 5090 Blackwell GPUs (64 GB VRAM total), running under strict competition constraints (<15B parameters, evaluated by **Penalized Accuracy**).

---

## 1. Approach: Adaptive Hybrid Architecture (Method 3)

The solution implements an **Adaptive Hybrid (SQL + Python + Adjudicator)** architecture, dynamically routing questions to the most reliable, efficient, and risk-mitigated execution path while enforcing strict Penalized Accuracy gating:

```text
[Input Question + as_of_date]
              │
              ▼
   [BusinessDuckDBStore] (ETL to /tmp/datamate_duckdb, calendar.csv, schema cards)
              │
              ▼
   [Entity Probe & Join-Key Profiler] (<5ms token inverted index + key profiling)
              │
              ▼
   [Adaptive Risk Router] ── (Time Pressure & Complexity Analysis)
              │
    ┌─────────┼────────────────────────┐
    ▼         ▼                        ▼
[SQL_FAST]  [HYBRID_DUAL]         [PYTHON_FIRST]
 (DuckDB)   (SQL + Python)          (Sandbox)
    │         │                        │
    │         ▼                        │
    │     [answers_equivalent?]        │
    │     ├── MATCH ──> Final Result   │
    │     └── CONFLICT ──> [Adjudicator]
    │                                  │
    └─────────────────┬────────────────┘
                      ▼
        [Verification & Abstain Gate]
    (4 Unanswerable Classes vs Safe Abstain)
                      │
                      ▼
         [Standardized Submission Record]
```

### Key Technical Pillars:

1. **High-Performance DuckDB ETL (`src/duckdb_store.py`)**:
   - Automatically converts and stages multi-format tabular data (CSV, Excel XLSX, SQLite) and `calendar.csv` into local DuckDB databases stored in `/tmp/datamate_duckdb/<biz_id>.duckdb`.
   - Protects against read-only host mounts (`/data:ro`) and achieves sub-second analytical SQL query execution.

2. **Join-Key & Uniqueness Profiler (`src/schema_parser.py`, `src/duckdb_store.py`)**:
   - Detects primary key candidates (`uniqueness_ratio >= 0.995`) and cross-table foreign key links (`overlap_ratio >= 0.5`).
   - Flags dangerous `many-to-many` relationships with explicit fan-out warnings to force aggregation before joining.

3. **In-Memory Entity Probing (`src/entity_probe.py`)**:
   - Builds a lightweight inverted index over key business entities (products, categories, customer names, branch locations) using unaccented Vietnamese normalization.
   - Probes entity matches in <5 ms per question, providing precise schema hints without corrupting static system prompts.

4. **vLLM Automatic Prefix Caching (`src/prompt_builder.py`, `src/llm_config.py`)**:
   - System prompts are split into a fixed, business-level prefix (`build_business_prefix`: rulebook + schema card + sample records) and a dynamic per-question suffix.
   - Enables 100% KV prefix cache hits across questions within the same business, slashing time-to-first-token (TTFT) by over 80%.

5. **Cross-Verification & Adjudicator (`src/agent.py`)**:
   - For high-risk or complex aggregations (`HYBRID_DUAL`), the pipeline executes both DuckDB SQL and sandboxed Python in parallel.
   - If outputs match according to `answers_equivalent`, the answer is accepted with high confidence; on disagreement, an LLM Adjudicator resolves discrepancies by evaluating both SQL and Python traces.

6. **Penalized Accuracy Optimization & 4 Unanswerable Classes**:
   - In Penalized Accuracy, incorrect answers incur a penalty (-1.0), whereas abstaining yields 0.
   - Explicitly formalizes 4 distinct unanswerable scenarios:
     1. Missing core metric/column (e.g. asking for profit when cost data is absent).
     2. Question time window falls entirely outside available data boundaries.
     3. Insufficient granularity or lack of join keys between tables.
     4. Missing mandatory attributes required by `rulebook.md`.
   - If confidence is insufficient or execution fails, the system safely outputs `abstain` (`null`) instead of guessing.

7. **Dynamic Lunar-Solar Calendar Synchronization (`src/calendar_utils.py`)**:
   - Eliminates hardcoded Tet windows and fixed-year assumptions.
   - Derives all calendar mappings dynamically from `/data/calendar.csv` and active `/data/rulebook.md` rules as a Single Source of Truth shared between DuckDB table `_calendar` and Python sandbox functions.

8. **Isolated Sandbox & Runtime Safety (`src/executor.py`, `predict.sh`)**:
   - Sandboxed Python execution environment preventing destructive system modifications.
   - All runtime caches (`VLLM_CACHE_ROOT`, `TRITON_CACHE_DIR`, `TORCHINDUCTOR_CACHE_DIR`, `HF_HOME`, `TMPDIR`) are strictly locked into `/tmp`.

---

## 2. Models Declaration

The pipeline standardizes on **`Qwen/Qwen3-14B`** for official offline inference. Complete cryptographic hashes are recorded in `training_code/data_manifest.json`.

| Property | Specification |
| :--- | :--- |
| **Model Name** | `Qwen/Qwen3-14B` |
| **Public Repository** | [https://huggingface.co/Qwen/Qwen3-14B](https://huggingface.co/Qwen/Qwen3-14B) |
| **Fixed Commit Revision** | `40c069824f4251a91eefaf281ebe4c544efd3e18` |
| **Original Release Date** | `May 2025` (2025-05-15, confirmed prior to `2026-10-26` cutoff) |
| **License** | `Apache-2.0` |
| **Total Parameters** | `14,770,000,000` (~14.8B; strictly `<= 15,000,000,000` per competition rules) |
| **Non-Embedding Parameters** | `13,200,000,000` (~13.2B) |
| **Architecture** | `Qwen3ForCausalLM` |
| **Precision / Quantization** | BF16 native (TP=2) / FP8 supported (DP=2) |
| **Local Container Path** | `/code/models/Qwen3-14B` |

### Key Model Weights Checksums (SHA-256)

| Shard Name | Size (Bytes) | SHA-256 Checksum |
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

## 3. Data & Training Declaration

> **Official Declaration**:
> In accordance with competition guidelines, **no model weight fine-tuning was performed**.
> - **Inference Paradigm**: The pipeline employs zero-shot and few-shot in-context learning with the foundation model `Qwen/Qwen3-14B`.
> - **Data Provenance**: No public test or private test datasets were used for training, prompt crafting, or validation.
> - **Synthetic Testing Data**: Only synthetic question-code pairs (`training_code/data/synthetic_qa.jsonl`) generated from sample templates were used for regression verification.
> - **External APIs**: Zero external API calls are made during inference. All processing is 100% offline.

---

## 4. Folder Layout & Paths

```text
/code/
├── Dockerfile                  # Container definition based on vllm/vllm-openai:v0.10.1.1
├── README.md                   # System documentation and methodology
├── requirements.txt            # Python dependencies (pinned nvidia-nccl-cu12==2.27.7)
├── predict.sh                  # Mandatory container entrypoint
├── models/
│   └── Qwen3-14B/              # Offline weights, tokenizer, and config.json
├── training_code/
│   ├── README.md               # Training disclosure and execution guide
│   ├── requirements.txt        # Training and inspection dependencies
│   ├── train.sh                # Training and validation entrypoint
│   ├── data_manifest.json      # Checksums, parameters, and provenance manifest
│   ├── configs/                # Model architecture & evaluation configs
│   ├── data/                   # Synthetic QA test dataset
│   ├── data_generation/        # QA generation scripts
│   └── src/                    # Dataset validation and manifest generator scripts
├── scripts/
│   ├── audit_submission_compliance.py # Automated pre-submit rule checker
│   ├── prepare_and_verify_model.py    # Offline weight inspector and quant comparator
│   ├── pre_submit_check.sh            # Dual-run Docker reproducibility verification
│   └── verify_submission_runs.py      # Statistical reproducibility verifier
├── src/
│   ├── main.py                 # Batch orchestrator, time guard, output streaming
│   ├── agent.py                # Dual-path hybrid executor, Adjudicator, verification gate
│   ├── prompt_builder.py       # Prefix caching prompt builder & unanswerable rules
│   ├── router.py               # Question classification (SQL_FAST / HYBRID / PYTHON)
│   ├── duckdb_store.py         # DuckDB table loader, profiling, and calendar synchronization
│   ├── schema_parser.py        # Schema enrichment, sample rows, and join hints
│   ├── sql_executor.py         # Safe DuckDB SQL runner with timeouts
│   ├── executor.py             # Sandboxed Python code executor with AST traps
│   ├── entity_probe.py         # In-memory entity index & fast probe (<5ms)
│   ├── calendar_utils.py       # Single Source of Truth for Vietnamese calendar reasoning
│   ├── formatter.py            # NFC normalization, scalar unwrap, unanswerable formatting
│   └── llm_config.py           # vLLM engine settings and GPU mode auto-detection
└── tests/                      # 250+ unit tests verifying 100% system integrity
```

### Fixed Paths per Competition Specification:

* Container Entrypoint: `/code/predict.sh`
* Input Dataset (Read-Only): `/data`
* Output Submission: `/result/submission.jsonl`
* Runtime Logs & Metrics: `/result/logs/` (`predictions.jsonl`, `runtime_metrics.json`)
* Scratch / Temporary Cache: `/tmp`

---

## 5. Docker Build & Evaluation Instructions

### 1. Build the Docker Image

```bash
docker build --platform linux/amd64 -t datamate:latest .
```

### 2. Run Local Compliance Audit

Before running container tests, verify 100% rule compliance:

```bash
python scripts/audit_submission_compliance.py
```

### 3. Run Inference per Competition Specification

```bash
docker run --rm \
  --platform linux/amd64 \
  --gpus '"device=0,1"' \
  --network none \
  --shm-size=16g \
  --entrypoint bash \
  -v /path/to/test_data:/data:ro \
  -v /path/to/result:/result \
  datamate:latest /code/predict.sh
```

### 4. Verify Dual-Run Reproducibility

To confirm zero non-determinism across consecutive runs:

```bash
bash scripts/pre_submit_check.sh /path/to/dataset
```
