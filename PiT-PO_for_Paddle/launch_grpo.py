#!/usr/bin/env python3
"""Run Paddle PiT-PO with one model shared by generation and GRPO training."""
from __future__ import annotations

import argparse
from datetime import datetime
import json
import logging
import os
from pathlib import Path
import re
import signal

os.environ["TOKENIZERS_PARALLELISM"] = "false"
logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--problem", default="oscillator1")
    parser.add_argument("--model_path", default=os.environ.get("PITPO_MODEL_PATH", "Qwen/Qwen2.5-0.5B-Instruct"),
                        help="PaddleFormers model identifier or local model directory")
    parser.add_argument("--device", choices=["gpu", "cpu"], default="gpu")
    parser.add_argument("--gpu", default="0", help="Physical GPU index; ignored on CPU")
    parser.add_argument("--dtype", choices=["float32", "bfloat16"], default="float32")
    parser.add_argument("--prompt_format", choices=["auto", "raw", "chat"], default="auto",
                        help="Use the tokenizer chat template for instruction models, or raw code completion")
    parser.add_argument("--max_samples", type=int, default=3000)
    parser.add_argument("--samples_per_prompt", type=int, default=4)
    parser.add_argument("--max_seq_length", type=int, default=4096)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--grpo_lr", type=float, default=1e-6)
    parser.add_argument("--grpo_batch_size", type=int, default=2,
                        help="Optimizer microbatch size; prompt group size is samples_per_prompt")
    parser.add_argument("--buffer_size", type=int, default=500)
    parser.add_argument("--reward_scaling", type=float, default=10.0)
    parser.add_argument("--training_strategy", default="continuous",
                        choices=["conservative", "adaptive", "aggressive", "continuous"])
    parser.add_argument("--grpo_train_every", type=int, default=64)
    parser.add_argument("--warmup_samples", type=int, default=200)
    parser.add_argument("--required_train_samples", type=int, default=16)
    parser.add_argument("--grpo_clip_epsilon", type=float, default=0.2)
    parser.add_argument("--grpo_kl_beta", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=3408)
    parser.add_argument("--checkpoint_every", type=int, default=100)
    parser.add_argument("--adapter_path", help="Load a saved Paddle LoRA adapter")
    parser.add_argument("--export_merged_model", action="store_true")
    parser.add_argument("--disable_grpo", action="store_true")
    parser.add_argument("--output_dir", help="Run directory; defaults to PITPO_OUTPUT_ROOT/<timestamp>")
    parser.add_argument("--evaluate_timeout_seconds", type=int, default=30)
    parser.add_argument("--log_level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser


def main(argv: list[str] | None = None):
    parser = build_parser()
    args = parser.parse_args(argv)
    for name in ("max_samples", "samples_per_prompt", "max_new_tokens", "grpo_batch_size",
                 "buffer_size", "grpo_train_every", "required_train_samples", "evaluate_timeout_seconds"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name} must be positive")
    if args.required_train_samples < 2:
        parser.error("--required_train_samples must be at least 2 for relative policy optimization")
    if args.warmup_samples < 0:
        parser.error("--warmup_samples must be non-negative")
    if args.max_new_tokens >= args.max_seq_length:
        parser.error("--max_new_tokens must be smaller than --max_seq_length")
    if args.buffer_size < args.required_train_samples:
        parser.error("--buffer_size must be at least --required_train_samples")
    if args.device == "cpu" and args.dtype != "float32":
        parser.error("CPU runs require --dtype float32")

    def handle_termination(signum, _frame):
        raise KeyboardInterrupt(f"received signal {signum}")

    signal.signal(signal.SIGTERM, handle_termination)
    # Set visibility before importing Paddle or creating the model.
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu if args.device == "gpu" else ""
    base_dir = Path(__file__).resolve().parent
    output_root = Path(os.environ.get("PITPO_OUTPUT_ROOT", str(base_dir / "artifacts")))
    run_dir = Path(args.output_dir) if args.output_dir else output_root / (
        f"paddle_{args.problem}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
    )
    run_dir = run_dir.resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=getattr(logging, args.log_level),
                        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                        handlers=[logging.StreamHandler(), logging.FileHandler(run_dir / "training.log")],
                        force=True)
    with (run_dir / "config.json").open("w", encoding="utf-8") as handle:
        json.dump(vars(args), handle, indent=2)

    import numpy as np
    import paddle
    import pandas as pd
    from pitpo import config, evaluator, pipeline, sampler

    if args.device == "gpu" and not paddle.is_compiled_with_cuda():
        raise RuntimeError("This Paddle build has no CUDA support. Install paddlepaddle-gpu or use --device cpu.")
    device = "gpu:0" if args.device == "gpu" else "cpu"
    paddle.set_device(device)
    np.random.seed(args.seed)
    paddle.seed(args.seed)
    exact_spec = base_dir / "specs" / f"specification_{args.problem}_numpy.txt"
    base_problem = re.sub(r"\d+$", "", args.problem)
    spec_path = exact_spec if exact_spec.exists() else (
        base_dir / "specs" / f"specification_{base_problem}_numpy.txt"
    )
    data_path = base_dir / "data" / args.problem / "train.csv"
    if not spec_path.is_file():
        raise FileNotFoundError(f"Missing specification: {spec_path}")
    if not data_path.is_file():
        raise FileNotFoundError(f"Missing training data: {data_path}")
    data = pd.read_csv(data_path).to_numpy(dtype=np.float64)
    dataset = {"data": {"inputs": data[:, :-1], "outputs": data[:, -1]}}
    class_config = config.ClassConfig(llm_class=sampler.LocalLLM, sandbox_class=evaluator.LocalSandbox)
    config_obj = config.Config(use_api=False, samples_per_prompt=args.samples_per_prompt,
                               max_new_tokens=args.max_new_tokens,
                               evaluate_timeout_seconds=args.evaluate_timeout_seconds)
    logger.info("Starting Paddle PiT-PO: task=%s model=%s device=%s output=%s",
                args.problem, args.model_path, device, run_dir)
    try:
        result = pipeline.main(
            specification=spec_path.read_text(encoding="utf-8"),
            inputs=dataset,
            config=config_obj,
            max_sample_nums=args.max_samples,
            class_config=class_config,
            enable_grpo=not args.disable_grpo,
            grpo_model_name=args.model_path,
            grpo_learning_rate=args.grpo_lr,
            grpo_batch_size=args.grpo_batch_size,
            grpo_reward_scaling=args.reward_scaling,
            grpo_buffer_size=args.buffer_size,
            grpo_train_every_n_valid=args.grpo_train_every,
            grpo_warmup_samples=args.warmup_samples,
            grpo_required_train_samples=args.required_train_samples,
            grpo_clip_epsilon=args.grpo_clip_epsilon,
            grpo_kl_beta=args.grpo_kl_beta,
            grpo_seed=args.seed,
            grpo_max_seq_length=args.max_seq_length,
            grpo_device=device,
            grpo_dtype=args.dtype,
            grpo_prompt_format=args.prompt_format,
            grpo_adapter_path=args.adapter_path,
            training_strategy=args.training_strategy,
            checkpoint_every=args.checkpoint_every,
            export_merged_model=args.export_merged_model,
            log_dir=str(run_dir),
            reward_curves_dir=str(run_dir / "reward_curves"),
        )
    except KeyboardInterrupt:
        logger.info("Search interrupted; any initialized trainer attempted an emergency adapter checkpoint.")
        raise SystemExit(130)
    logger.info("Search completed: %s", result)
    return result


if __name__ == "__main__":
    main()
