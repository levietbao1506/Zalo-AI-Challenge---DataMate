# Training & Data Preparation Package (`/code/training_code/`)

This directory contains the training environment specification, dataset validation scripts, and data generation utilities for the **Zalo AI Challenge 2026 - DataMate Track**.

---

## 📌 Model Training & Fine-Tuning Disclosure

> **Official Declaration**:
> In accordance with the competition rules, **no model weight fine-tuning was performed** for this submission. The pipeline employs **zero-shot and few-shot in-context learning** with the foundation model **`Qwen/Qwen3-14B`**.
>
> High Penalized Accuracy (PA) is achieved through:
> 1. Rich multi-file business schema parsing (`schema_parser.py`) including data types, row counts, and sample values.
> 2. Strict system prompt engineering (`prompt_builder.py`) enforcing executable Python/Pandas code output and unanswerable criteria.
> 3. Dynamic Lunar and Solar temporal anchor reasoning (`calendar_utils.py`).
> 4. Isolated subprocess execution with self-correction retry feedback (`executor.py`).
> 5. A robust confidence and schema verification gate (`agent.py` and `evaluator.py`) to prevent incorrect answers (-1 penalty) by safely downgrading to abstain (0 points).

---

## 🗂️ Package Structure

```text
training_code/
├── README.md               # Training and methodology documentation
├── requirements.txt        # Training and data processing dependencies
├── train.sh                # Training pipeline entrypoint (validation & checks)
├── data_manifest.json      # Complete checksums and provenance of foundation weights
├── configs/
│   ├── model_config.json   # Qwen3-14B architecture and inference parameters
│   └── training_config.yaml# In-context learning and evaluation configuration
├── src/
│   ├── __init__.py
│   ├── validate_data.py    # Validates question schemas, tables, and ground truth integrity
│   └── inspect_dataset.py  # Inspects multi-file business datasets and prints statistics
├── data_generation/
│   ├── __init__.py
│   └── generate_qa_pairs.py# Generates synthetic question-code pairs for validation
├── data/
│   └── .gitkeep            # Placeholder for training/validation datasets
├── logs/
│   └── .gitkeep            # Placeholder for data validation and run logs
└── checkpoints/
    └── README.md           # Documentation confirming base model checkpoint usage
```

---

## 🤖 Model Provenance (`data_manifest.json`)

The foundation weights are downloaded from Hugging Face and verified against cryptographic SHA-256 checksums:

- **Model Name**: `Qwen/Qwen3-14B`
- **Repository**: [https://huggingface.co/Qwen/Qwen3-14B](https://huggingface.co/Qwen/Qwen3-14B)
- **Commit Hash**: `40c069824f4251a91eefaf281ebe4c544efd3e18`
- **Release Date**: May 2025
- **License**: Apache-2.0
- **Total Parameters**: 14.8B (~14,770,000,000 parameters; 13.2B non-embedding)
- **Architecture**: `Qwen3ForCausalLM`
- **Total Weights Size**: 29,536,614,400 bytes (~29.54 GB across 8 `.safetensors` shards)

Detailed per-shard SHA-256 hashes are recorded in `data_manifest.json`.

---

## 🛠️ Usage

### 1. Validate Datasets
To inspect and validate sample or competition datasets:
```bash
python -m training_code.src.validate_data --data_dir /data
```

### 2. Run Data Inspection
To print table schemas, row counts, and date distributions:
```bash
python -m training_code.src.inspect_dataset --data_dir /data
```

### 3. Run Training Entrypoint
```bash
bash training_code/train.sh
```
This runs the dataset validator and confirms offline readiness.
