# Model Checkpoints (`/code/training_code/checkpoints/`)

## Base Model Usage

In accordance with the competition specification:
- The base foundation model **`Qwen/Qwen3-14B`** is utilized directly for offline in-context inference.
- Model weights reside in `models/Qwen3-14B/` (or `/code/models/Qwen3-14B/` in the Docker container).
- No adapter weights, LoRA deltas, or fine-tuned parameter checkpoints were produced or are required.
- All safetensors shard SHA-256 hashes and metadata are tracked in `training_code/data_manifest.json`.
