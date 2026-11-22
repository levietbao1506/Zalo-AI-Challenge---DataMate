# Training & Data Preparation Package (`/code/training_code/`)

This package contains the training environment specification, dataset validation tools, synthetic data generation scripts, and weight verification manifests for the **Zalo AI Challenge 2026 - DataMate Track**.

---

## 1. Official Model Training & Fine-Tuning Disclosure

> **Official Declaration**:
> In accordance with the competition rules, **no model weight fine-tuning was performed**.
>
> The pipeline strictly runs zero-shot and few-shot in-context learning using the official open-weight foundation checkpoint **`Qwen/Qwen3-14B`** without extra fine-tuning or external API calls.
>
> High Penalized Accuracy (PA) is achieved via:
> 1. Multi-file business schema parsing (`schema_parser.py`) including data types, row counts, categorical sample values, and join key hints.
> 2. Structured prompt engineering with Prefix Caching (`prompt_builder.py`) enforcing executable Python/DuckDB code and clear unanswerable boundaries.
> 3. Dynamic Lunar and Solar temporal anchor reasoning (`calendar_utils.py`) acting as a Single Source of Truth.
> 4. Sandboxed execution with self-correction retry feedback (`executor.py`).
> 5. A robust confidence verification gate and cross-verification adjudicator (`agent.py`) to prevent incorrect answers (-1 penalty) by safely downgrading uncertain results to abstain (0 points).

---

## 2. Package Structure

```text
training_code/
├── README.md               # Training and methodology documentation
├── requirements.txt        # Training and data processing dependencies
├── train.sh                # Training and data validation entrypoint
├── data_manifest.json      # Checksums, parameters, and provenance manifest
├── configs/
│   ├── model_config.json   # Qwen3-14B architecture and inference parameters
│   └── training_config.yaml# In-context learning and evaluation configuration
├── src/
│   ├── __init__.py
│   ├── generate_manifest.py# Automated manifest generator and file hasher
│   ├── validate_data.py    # Validates question schemas, tables, and ground truth integrity
│   └── inspect_dataset.py  # Inspects multi-file business datasets and prints statistics
├── data_generation/
│   ├── __init__.py
│   └── generate_qa_pairs.py# Generates synthetic question-code pairs for validation
├── data/
│   └── synthetic_qa.jsonl  # Synthetic question-answer validation pairs
├── logs/                   # Placeholder for validation and run logs
└── checkpoints/
    └── README.md           # Documentation confirming base model checkpoint usage
```

---

## 3. Environment Setup & Hardware Configuration

Dependencies can be installed from `training_code/requirements.txt`:

```bash
pip install -r training_code/requirements.txt
```

### Hardware & Runtime Specification:
* **Target GPU**: 2× NVIDIA RTX 5090 Blackwell (64 GB VRAM total)
* **Host RAM**: 512 GB RAM
* **CUDA / PyTorch**: CUDA 12.8, PyTorch 2.7.1, vLLM 0.10.1.1
* **Deterministic Seeds**: All random generators fixed to `seed = 42` (`PYTHONHASHSEED=42`, `VLLM_SEED=42`).

---

## 4. Execution & Validation Utilities

### 1. Generate / Update Data Manifest
To re-hash all local training code files and refresh `training_code/data_manifest.json`:

```bash
python training_code/src/generate_manifest.py --write
```

### 2. Validate Dataset Schema & Formats
To inspect and validate sample or competition datasets:

```bash
python -m training_code.src.validate_data --data_dir /data
```

### 3. Inspect Business Datasets & Summary Statistics
To print table schemas, row counts, and date distributions:

```bash
python -m training_code.src.inspect_dataset --data_dir /data
```

### 4. Generate Synthetic QA Pairs
To produce validation question pairs for pipeline testing:

```bash
python -m training_code.data_generation.generate_qa_pairs \
  --data_dir data/sample_dataset \
  --output training_code/data/synthetic_qa.jsonl
```

### 5. Run Training Entrypoint
```bash
bash training_code/train.sh /data
```
This executes the manifest verification and dataset validation routines to confirm readiness for offline inference.
