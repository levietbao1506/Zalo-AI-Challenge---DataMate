"""Model preparation, inspection, verification, and quantization comparison script.

Designed for Zalo AI Challenge 2026 (DataMate Track) on 2x NVIDIA RTX 5090 GPUs.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import struct
import sys
import time
from typing import Any, Dict, List, Optional, Tuple, Union

MAX_ALLOWED_PARAMETERS = 15_000_000_000
GPU_VRAM_PER_CARD_GB = 32.0
GPU_TOTAL_VRAM_GB = 64.0
DEFAULT_GPU_UTILIZATION = 0.85
CUDA_RUNTIME_OVERHEAD_GB = 1.5


def count_parameters_from_safetensors(model_dir: Union[str, Path]) -> Optional[int]:
    path = Path(model_dir)
    safetensors_files = list(path.glob("*.safetensors"))
    if not safetensors_files:
        return None
    total_params = 0
    for sf in safetensors_files:
        try:
            with open(sf, "rb") as f:
                header_len_bytes = f.read(8)
                if len(header_len_bytes) < 8:
                    continue
                header_len = struct.unpack("<Q", header_len_bytes)[0]
                header_json = f.read(header_len).decode("utf-8")
                header = json.loads(header_json)
                for k, v in header.items():
                    if k == "__metadata__":
                        continue
                    if isinstance(v, dict) and "shape" in v:
                        prod = 1
                        for dim in v["shape"]:
                            prod *= dim
                        total_params += prod
        except Exception:
            return None
    return total_params if total_params > 0 else None


def count_parameters_from_config(config: Dict[str, Any]) -> int:
    if config.get("num_parameters") and isinstance(config["num_parameters"], int):
        return int(config["num_parameters"])
    if config.get("total_params") and isinstance(config["total_params"], int):
        return int(config["total_params"])

    hidden_size = int(config.get("hidden_size") or 0)
    num_layers = int(config.get("num_hidden_layers") or 0)
    vocab_size = int(config.get("vocab_size") or 0)
    intermediate_size = int(config.get("intermediate_size") or 0)
    num_heads = int(config.get("num_attention_heads") or 0)
    num_kv_heads = int(config.get("num_key_value_heads") or num_heads)
    head_dim = int(config.get("head_dim") or (hidden_size // num_heads if num_heads else 128))
    tie_word_embeddings = bool(config.get("tie_word_embeddings", False))

    num_experts = int(
        config.get("num_local_experts")
        or config.get("n_routed_experts")
        or config.get("num_experts")
        or 1
    )
    moe_intermediate = int(config.get("moe_intermediate_size") or intermediate_size)
    shared_intermediate = int(config.get("shared_expert_intermediate_size") or 0)

    if hidden_size == 0 or num_layers == 0:
        return 0

    attn = (
        hidden_size * (num_heads * head_dim)
        + hidden_size * (num_kv_heads * head_dim)
        + hidden_size * (num_kv_heads * head_dim)
        + (num_heads * head_dim) * hidden_size
    )

    if num_experts > 1:
        routed_mlp = num_experts * (3 * hidden_size * moe_intermediate)
        router_gate = hidden_size * num_experts
        shared_mlp = 3 * hidden_size * shared_intermediate if shared_intermediate > 0 else 0
        mlp = routed_mlp + router_gate + shared_mlp
    else:
        mlp = 3 * hidden_size * intermediate_size

    layer_norm = 2 * hidden_size
    layer_total = attn + mlp + layer_norm

    embed = vocab_size * hidden_size
    lm_head = 0 if tie_word_embeddings else (vocab_size * hidden_size)
    final_norm = hidden_size

    return embed + num_layers * layer_total + lm_head + final_norm


def get_model_parameters(model_dir: Union[str, Path], config: Dict[str, Any]) -> int:
    sf_params = count_parameters_from_safetensors(model_dir)
    if sf_params is not None and sf_params > 0:
        return sf_params
    return count_parameters_from_config(config)


def detect_quantization_format(model_dir: Union[str, Path], config: Dict[str, Any]) -> str:
    quant_cfg = config.get("quantization_config")
    if isinstance(quant_cfg, dict):
        method = str(quant_cfg.get("quant_method") or "").lower()
        if any(marker in method for marker in ("fp8", "float8")):
            return "fp8"
        if "awq" in method:
            return "awq"
        if "gptq" in method:
            return "gptq"

    dtype_str = str(config.get("torch_dtype") or config.get("dtype") or "").lower()
    if any(marker in dtype_str for marker in ("fp8", "float8")):
        return "fp8"
    if any(marker in dtype_str for marker in ("bfloat16", "bf16")):
        return "bf16"
    if any(marker in dtype_str for marker in ("float16", "fp16")):
        return "fp16"

    path_str = str(model_dir).lower()
    if "fp8" in path_str:
        return "fp8"
    if "awq" in path_str:
        return "awq"
    if "gptq" in path_str:
        return "gptq"
    if "bf16" in path_str or "bfloat16" in path_str:
        return "bf16"
    if "fp16" in path_str:
        return "fp16"

    return "bf16"


def estimate_weight_disk_gb(model_dir: Union[str, Path]) -> float:
    path = Path(model_dir)
    if not path.exists():
        return 0.0
    total_bytes = 0
    weight_extensions = {".safetensors", ".bin", ".pt"}
    for f in path.rglob("*"):
        if f.is_file() and f.suffix.lower() in weight_extensions:
            try:
                total_bytes += f.stat().st_size
            except OSError:
                pass
    return total_bytes / (1024 ** 3)


def inspect_model(model_dir: Union[str, Path]) -> Dict[str, Any]:
    path = Path(model_dir).resolve()
    if not path.is_dir():
        raise FileNotFoundError(f"Model directory does not exist: {path}")

    config_path = path / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"Missing config.json in {path}")

    tokenizer_config_path = path / "tokenizer_config.json"
    if not tokenizer_config_path.is_file():
        raise FileNotFoundError(f"Missing tokenizer_config.json in {path}")

    weight_files = list(path.glob("*.safetensors")) + list(path.glob("*.bin"))
    if not weight_files:
        raise FileNotFoundError(f"No weight files (.safetensors or .bin) found in {path}")

    config_content = json.loads(config_path.read_text(encoding="utf-8"))
    weight_gb = estimate_weight_disk_gb(path)
    quant_format = detect_quantization_format(path, config_content)
    total_params = get_model_parameters(path, config_content)

    is_quantized = quant_format in ("fp8", "awq", "gptq")
    if is_quantized and 0.0 < weight_gb < 22.0:
        forecast_mode = "DP=2"
        worker_count = 2
        weight_per_gpu = weight_gb
    else:
        forecast_mode = "TP=2"
        worker_count = 1
        weight_per_gpu = weight_gb / 2.0

    usable_vram_limit = GPU_VRAM_PER_CARD_GB * DEFAULT_GPU_UTILIZATION
    kv_headroom_gb = max(0.0, usable_vram_limit - weight_per_gpu)
    absolute_headroom_gb = max(0.0, GPU_VRAM_PER_CARD_GB - weight_per_gpu - CUDA_RUNTIME_OVERHEAD_GB)

    if total_params > MAX_ALLOWED_PARAMETERS:
        raise ValueError(
            f"Model total parameters ({total_params:,}) exceed competition limit of {MAX_ALLOWED_PARAMETERS:,}!"
        )

    return {
        "model_dir": str(path),
        "total_params": total_params,
        "is_under_15b": total_params <= MAX_ALLOWED_PARAMETERS,
        "weight_gb": round(weight_gb, 2),
        "quant_format": quant_format,
        "is_quantized": is_quantized,
        "forecast_mode": forecast_mode,
        "worker_count": worker_count,
        "weight_per_gpu_gb": round(weight_per_gpu, 2),
        "kv_headroom_gb": round(kv_headroom_gb, 2),
        "absolute_headroom_gb": round(absolute_headroom_gb, 2),
    }


def compare_quantization(
    bf16_dir: Union[str, Path],
    fp8_dir: Union[str, Path],
    dataset_dir: Union[str, Path],
    questions_file: Optional[Union[str, Path]] = None,
    limit: int = 50,
    dry_run: bool = False,
) -> Dict[str, Any]:
    dataset_path = Path(dataset_dir).resolve()
    q_file = Path(questions_file).resolve() if questions_file else dataset_path / "questions.jsonl"
    if not q_file.is_file():
        raise FileNotFoundError(f"Questions file not found: {q_file}")

    from src.main import load_questions, read_rulebook

    questions = load_questions(q_file)[:limit]
    rulebook_text = read_rulebook(dataset_path)

    def evaluate_model(model_path: str, opt_tp: int) -> Tuple[float, float, int]:
        from src.agent import CodeAgent, execute_adaptive_hybrid
        from src.duckdb_store import BusinessDuckDBStore
        from src.prompt_builder import PromptBuilder, build_business_prefix
        from src.schema_parser import build_sql_schema_card, build_store_schema_meta, read_business_readme
        from src.sql_executor import DuckDBSQLExecutor

        agent = CodeAgent(model_path=model_path, dry_run=dry_run, tensor_parallel_size=opt_tp)
        prompt_builder = PromptBuilder(rulebook_content=rulebook_text)

        elapsed_records: List[int] = []
        valid_answers = 0

        calendar_path = None
        cand_cal = dataset_path / "calendar.csv"
        if cand_cal.is_file():
            calendar_path = str(cand_cal)

        for q in questions:
            biz_id = str(q.get("business_id") or "biz")
            biz_dir = dataset_path / biz_id if (dataset_path / biz_id).is_dir() else dataset_path
            readme_text = read_business_readme(biz_dir)

            store = BusinessDuckDBStore(business_dir=biz_dir, calendar_csv_path=calendar_path)
            sql_executor = DuckDBSQLExecutor(store)
            schema_card = build_sql_schema_card(store, readme_text)
            business_prefix = build_business_prefix(rulebook_text, readme_text, schema_card)
            schema_meta = build_store_schema_meta(store)

            try:
                res = execute_adaptive_hybrid(
                    question_record=q,
                    duckdb_store=store,
                    sql_executor=sql_executor,
                    schema_card=schema_card,
                    schema_meta=schema_meta,
                    rulebook_text=rulebook_text,
                    agent=agent,
                    prompt_builder=prompt_builder,
                    business_prefix=business_prefix,
                    readme_text=readme_text,
                )
                log_rec = res["log_record"]
                sub_rec = res["submission_record"]
                elapsed_records.append(int(log_rec.get("elapsed_ms", 0)))
                ans_type = sub_rec.get("type")
                if ans_type not in (None, "abstain"):
                    valid_answers += 1
            finally:
                sql_executor.close()
                store.close_and_cleanup()

        count = len(questions)
        pa = valid_answers / count if count > 0 else 0.0
        avg_ms = sum(elapsed_records) / len(elapsed_records) if elapsed_records else 0.0
        return pa, avg_ms, count

    pa_bf16, avg_ms_bf16, total_q = evaluate_model(str(bf16_dir), opt_tp=2)
    pa_fp8, avg_ms_fp8, _ = evaluate_model(str(fp8_dir), opt_tp=1)

    delta_pa = pa_bf16 - pa_fp8
    speedup_ratio = avg_ms_bf16 / max(1.0, avg_ms_fp8)

    if delta_pa <= 0.015 and speedup_ratio > 1.05:
        recommendation = "RECOMMENDED: FP8 + DP=2 (Delta PA <= 1.5% with higher throughput on dual RTX 5090)"
        chosen = "fp8_dp2"
    else:
        recommendation = "RECOMMENDED: BF16 + TP=2 (Higher fidelity with acceptable latency)"
        chosen = "bf16_tp2"

    return {
        "questions_count": total_q,
        "pa_bf16": round(pa_bf16, 4),
        "pa_fp8": round(pa_fp8, 4),
        "delta_pa": round(delta_pa, 4),
        "avg_ms_bf16": round(avg_ms_bf16, 2),
        "avg_ms_fp8": round(avg_ms_fp8, 2),
        "speedup_ratio": round(speedup_ratio, 2),
        "recommendation": recommendation,
        "chosen": chosen,
    }


def print_quantize_fp8_guidance(model_dir: Optional[str] = None) -> None:
    print("=" * 80)
    print("      Offline FP8 Quantization Preparation Guide for Zalo AI Challenge")
    print("=" * 80)
    print("CRITICAL NOTE: The competition evaluation environment runs with --network none")
    print("and /data is mounted read-only. All model checkpoints must be prepared, verified,")
    print("and baked into the Docker image BEFORE final submission packaging.\n")

    print("Option 1: Using Neural Magic llm-compressor (recommended for vLLM FP8):")
    print("  pip install llmcompressor")
    print("  python -c '")
    print("  from llmcompressor.transformers import SparseAutoModelForCausalLM")
    print("  from llmcompressor.transformers import oneshot")
    print("  # Run FP8 static/dynamic quantization on calibration dataset")
    print("  '\n")

    print("Option 2: Using AutoFP8:")
    print("  pip install auto-fp8")
    print("  python -c '")
    print("  from auto_fp8 import AutoFP8ForCausalLM, BaseQuantizeConfig")
    print("  quant_config = BaseQuantizeConfig(quant_method=\"fp8\", activation_scheme=\"static\")")
    print("  model = AutoFP8ForCausalLM.from_pretrained(\"/path/to/bf16\", quant_config=quant_config)")
    print("  model.quantize(examples)")
    print("  model.save_quantized(\"/path/to/fp8_model\")")
    print("  '\n")

    print("Option 3: Download verified public FP8 release published before Oct 26, 2026:")
    print("  Ensure model_manifest.json and config.json explicitly declare FP8 format.")
    print("=" * 80)

    if model_dir and Path(model_dir).is_dir():
        print(f"\nVerifying existing model checkpoint at: {model_dir}")
        try:
            info = inspect_model(model_dir)
            print(f"  Format:           {info['quant_format']}")
            print(f"  Parameters:       {info['total_params']:,} (Under 15B: {info['is_under_15b']})")
            print(f"  Weight size:      {info['weight_gb']} GB")
            print(f"  Forecast Mode:    {info['forecast_mode']} (Workers: {info['worker_count']})")
            print(f"  KV Headroom/GPU:  {info['kv_headroom_gb']} GB")
        except Exception as e:
            print(f"  Verification failed: {e}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Model preparation, verification, and quantization comparison.")
    parser.add_argument("--inspect", action="store_true", help="Inspect local model checkpoint offline.")
    parser.add_argument("--model-dir", "--model_dir", type=str, default="/code/models/Qwen3-14B", help="Path to model directory.")
    parser.add_argument("--compare-quant", "--compare_quant", action="store_true", help="Compare BF16 and FP8 checkpoints.")
    parser.add_argument("--bf16-dir", "--bf16_dir", type=str, default="models/Qwen3-14B", help="Path to BF16 model.")
    parser.add_argument("--fp8-dir", "--fp8_dir", type=str, default="models/Qwen3-14B-FP8", help="Path to FP8 model.")
    parser.add_argument("--dataset-dir", "--dataset_dir", type=str, default="data/sample_dataset", help="Path to dataset directory.")
    parser.add_argument("--questions-file", "--questions_file", type=str, default=None, help="Path to questions.jsonl.")
    parser.add_argument("--limit", type=int, default=50, help="Question limit for comparison.")
    parser.add_argument("--quantize-fp8", "--quantize_fp8", action="store_true", help="Show FP8 quantization guidance.")
    parser.add_argument("--dry-run", "--dry_run", action="store_true", help="Run without live GPU inference.")

    args = parser.parse_args()

    if args.inspect:
        print(f"Inspecting model checkpoint at: {args.model_dir} ...")
        try:
            info = inspect_model(args.model_dir)
            print("\n" + "=" * 60)
            print(" MODEL INSPECTION REPORT (2x RTX 5090 Offline Evaluation)")
            print("=" * 60)
            print(f" Model Path:          {info['model_dir']}")
            print(f" Total Parameters:    {info['total_params']:,}")
            print(f" Parameter Gate:      {'PASSED (<= 15B)' if info['is_under_15b'] else 'FAILED (> 15B)'}")
            print(f" Weight Size on Disk: {info['weight_gb']} GB")
            print(f" Quantization Format: {info['quant_format'].upper()}")
            print(f" Forecast GPU Mode:   {info['forecast_mode']} ({info['worker_count']} worker(s))")
            print(f" Weight per GPU:      {info['weight_per_gpu_gb']} GB")
            print(f" Usable KV Headroom:  {info['kv_headroom_gb']} GB (at 0.85 memory utilization)")
            print(f" Absolute Headroom:   {info['absolute_headroom_gb']} GB (after runtime context)")
            print("=" * 60 + "\n")
            sys.exit(0)
        except Exception as e:
            print(f"\nERROR: Model inspection failed: {e}\n", file=sys.stderr)
            sys.exit(1)

    elif args.compare_quant:
        print("Running comparative evaluation between BF16 and FP8 checkpoints...")
        try:
            res = compare_quantization(
                bf16_dir=args.bf16_dir,
                fp8_dir=args.fp8_dir,
                dataset_dir=args.dataset_dir,
                questions_file=args.questions_file,
                limit=args.limit,
                dry_run=args.dry_run,
            )
            print("\n" + "=" * 70)
            print(" QUANTIZATION COMPARISON SUMMARY")
            print("=" * 70)
            print(f" Questions Evaluated: {res['questions_count']}")
            print(f" BF16 PA:             {res['pa_bf16']:.4f} (avg {res['avg_ms_bf16']} ms/q)")
            print(f" FP8 PA:              {res['pa_fp8']:.4f} (avg {res['avg_ms_fp8']} ms/q)")
            print(f" Delta PA (BF16-FP8): {res['delta_pa']:+.4f}")
            print(f" Throughput Speedup:  {res['speedup_ratio']:.2f}x")
            print(f" Verdict:             {res['recommendation']}")
            print("=" * 70 + "\n")
            sys.exit(0)
        except Exception as e:
            print(f"\nERROR: Quantization comparison failed: {e}\n", file=sys.stderr)
            sys.exit(1)

    elif args.quantize_fp8:
        print_quantize_fp8_guidance(model_dir=args.model_dir)
        sys.exit(0)

    else:
        parser.print_help()
        sys.exit(0)


if __name__ == "__main__":
    main()
