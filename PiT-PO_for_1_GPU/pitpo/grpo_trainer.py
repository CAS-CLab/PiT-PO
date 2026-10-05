from __future__ import annotations

import os
# Avoid the tokenizers parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
import numpy as np
import logging
import time
from typing import List, Dict, Any, Optional, Tuple
import math
import re
import csv
import json
import tempfile
import shutil
from contextlib import nullcontext

# Equation-analysis reward
from .equation_functions import (
    extract_equation_functions_from_text,
    analyze_equation_function,
)
from .coef_penalty import compute_coefficient_penalty

logger = logging.getLogger(__name__)

# EvoTune-aligned: matches `percentile: 70` in `configs/config.yaml`; threshold =
#   np.percentile(scores_since_tuning, 100 - percentile)
# Intentionally hard-coded; not exposed as a configurable option.
_DPO_STYLE_PERCENTILE = 70


def _numeric_gradient_hook(name: str):
    """Build a diagnostic hook without changing the gradient it observes."""
    def report(gradient: torch.Tensor) -> torch.Tensor:
        finite = torch.isfinite(gradient)
        finite_count = int(finite.sum().item())
        total_count = gradient.numel()
        abs_max = (
            float(gradient[finite].abs().max().item())
            if finite_count
            else float("nan")
        )
        print(
            f"[GRPO_NUMERICS] {name}: dtype={gradient.dtype} "
            f"finite={finite_count}/{total_count} abs_max={abs_max:.6e}",
            flush=True,
        )
        return gradient

    return report


class GRPOTrainer:
    """
    Group Relative Policy Optimization trainer for fine-tuning language models.
    Uses MSE as reward signal where lower MSE results in higher reward.
    """
    
    def __init__(
        self,
        model_name: str,
        learning_rate: float = 1e-5,
        batch_size: int = 4,
        max_length: Optional[int] = None,
        reward_scaling: float = 10.0,
        min_mse_threshold: float = 1e-15,
        buffer_size: int = 1000,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        required_train_samples: int = 16,
        # LoRA-related configuration (enabled by default; only LoRA params are trained)
        use_lora: bool = True,
        lora_r: int = 16,
        lora_alpha: int = 32,
        # CALM/Unsloth require zero dropout for the fused LoRA kernels.  A
        # non-zero value silently falls back to generic PEFT matmuls for every
        # adapted projection, which is both slower and numerically less stable
        # with 4-bit training.
        lora_dropout: float = 0.0,
        lora_target_modules: Optional[List[str]] = None,
        # Tunable training hyperparameters: per-device batch and gradient accumulation
        per_device_train_batch_size: int = 4,
        gradient_accumulation_steps: int = 4,
        # Legacy generation-based warmup; new runs use warmup_samples below.
        warmup_generations: Optional[int] = None,
        samples_per_prompt: Optional[int] = None,
        warmup_min_global_samples: Optional[int] = None,
        # PiT-PO Dual-Constraint configs (Paper Section 3.1 & 3.2)
        # -- AST complexity penalty P_cplx (Eq. 8)
        ast_complexity_weight: float = 0.05,
        # -- Gated physical penalty P_phy (Eq. 9)
        enable_physical_penalty: bool = True,
        phy_penalty_dim: float = 2.0,
        phy_penalty_diff: float = 1.5,
        phy_gate_mse_threshold: float = 1.0,
        phy_units: Optional[Dict[str, str]] = None,
        # -- Coefficient redundancy penalty via Support Exclusion Theorem (Eq. 5-6)
        enable_coef_penalty: bool = True,
        coef_ratio_threshold: float = 0.05,
        coef_penalty_weight: float = 1.0,
        # CALM-style colocated model runtime.
        max_seq_length: int = 4096,
        load_in_4bit: bool = True,
        fast_inference: bool = True,
        vllm_gpu_memory_utilization: float = 0.6,
        gradient_checkpointing: str = "unsloth",
        # BF16 has a wide exponent range, but the quantized model's internal
        # matmul backward can still overflow before global-norm clipping when
        # token penalties amplify a policy gradient.  Downscale the loss for
        # backward, restore LoRA gradients in FP32, then clip as usual.
        backward_stability_scale: float = 128.0,
        checkpoint_dir: Optional[str] = None,
        checkpoint_every: int = 100,
        grpo_clip_epsilon: float = 0.2,
        grpo_kl_beta: float = 0.01,
        warmup_samples: int = 200,
        seed: int = 3408,
    ):
        """
        Initialize GRPO trainer.
        
        Args:
            model_name: Hugging Face model name or path
            learning_rate: Learning rate for optimization
            batch_size: Batch size for training
            max_length: Training context length; defaults to max_seq_length
            reward_scaling: Scaling factor for log reward transformation
            min_mse_threshold: Minimum MSE threshold to prevent log(0)
            buffer_size: Maximum size of training buffer
            device: Device to run on (cuda/cpu)
            required_train_samples: Number of valid experiences required per fine-tuning step
            use_lora: Whether to use LoRA and train only LoRA adapters
            lora_r/lora_alpha/lora_dropout: LoRA hyperparameters
            lora_target_modules: Target modules for LoRA injection; if None will be resolved for common LLaMA blocks
            per_device_train_batch_size: training batch size per device
            gradient_accumulation_steps: gradient-accumulation steps (effective batch = per_device * accum)
            warmup_generations: Optional legacy warmup expressed in generation rounds
            samples_per_prompt: Number of candidate equations per generation round
            warmup_min_global_samples: Optional legacy absolute sample-index threshold
            ast_complexity_weight: lambda_len for P_cplx = lambda_len * AST_node_count (Paper Eq. 8)
            enable_physical_penalty: enable gated physical penalty P_phy (Paper Eq. 9)
            phy_penalty_dim: penalty for dimensional inconsistency
            phy_penalty_diff: penalty for non-differentiability
            phy_gate_mse_threshold: delta_gate; physical penalties only activate when MSE < this
            phy_units: unit mapping for dimensional analysis
            enable_coef_penalty: enable Support Exclusion Theorem penalty (Paper Eq. 5-6)
            coef_ratio_threshold: rho threshold on tau_i = |b_i|/(sum|b_j|+eps)
            coef_penalty_weight: scaling coefficient p in P_tok = p * max(0, -ln(|b_i|+eps))
            grpo_clip_epsilon: PPO/GRPO probability-ratio clipping range (default: 0.2)
            grpo_kl_beta: KL weight against the frozen base model (default: 0.01)
            warmup_samples: Number of candidate equations without training (default: 200)
        """
        self.model_name = model_name
        self.seed = int(seed)
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.max_length = int(max_seq_length if max_length is None else max_length)
        if not 2 <= self.max_length <= int(max_seq_length):
            raise ValueError("max_length must be between 2 and max_seq_length")
        self.reward_scaling = reward_scaling
        self.min_mse_threshold = min_mse_threshold
        self.buffer_size = buffer_size
        self.device = device
        self.required_train_samples = required_train_samples
        self.use_lora = use_lora
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.lora_target_modules = lora_target_modules
        # Save tunable training hyperparameters
        self.per_device_train_batch_size = per_device_train_batch_size
        self.gradient_accumulation_steps = gradient_accumulation_steps
        if self.gradient_accumulation_steps < 1:
            raise ValueError("gradient_accumulation_steps must be positive")
        
        # Warmup-related configuration
        self.samples_per_prompt = samples_per_prompt
        self.warmup_generations = warmup_generations
        self.warmup_samples = int(warmup_samples)
        if warmup_generations is not None:
            if int(warmup_generations) < 0:
                raise ValueError("warmup_generations must be non-negative")
            self.warmup_samples = int(warmup_generations) * int(samples_per_prompt or 1)
        if self.warmup_samples < 0:
            raise ValueError("warmup_samples must be non-negative")
        # Keep progress independently of buffer retention. Candidate indices
        # start at one; the default gate first permits candidate 201.
        self.candidate_count = 0
        self.warmup_min_global_samples = (
            int(warmup_min_global_samples)
            if warmup_min_global_samples is not None else None
        )
        self.grpo_clip_epsilon = float(grpo_clip_epsilon)
        self.grpo_kl_beta = float(grpo_kl_beta)
        if not math.isfinite(self.grpo_clip_epsilon) or not 0.0 < self.grpo_clip_epsilon < 1.0:
            raise ValueError("grpo_clip_epsilon must be finite and between 0 and 1")
        if not math.isfinite(self.grpo_kl_beta) or self.grpo_kl_beta < 0.0:
            raise ValueError("grpo_kl_beta must be finite and non-negative")
        
        # AST complexity penalty (Paper Eq. 8)
        self.ast_complexity_weight = float(ast_complexity_weight)

        # Gated physical penalty (Paper Eq. 9)
        self.enable_physical_penalty = bool(enable_physical_penalty)
        self.phy_penalty_dim = float(phy_penalty_dim)
        self.phy_penalty_diff = float(phy_penalty_diff)
        self.phy_gate_mse_threshold = float(phy_gate_mse_threshold)
        self.phy_units = dict(phy_units) if phy_units else None

        # Coefficient redundancy penalty (Paper Eq. 5-6)
        self.enable_coef_penalty = bool(enable_coef_penalty)
        self.coef_ratio_threshold = float(coef_ratio_threshold)
        self.coef_penalty_weight = float(coef_penalty_weight)

        # Shared training/inference runtime.  As in CALM, one process owns one
        # model object and alternates it between training and inference modes.
        self.max_seq_length = int(max_seq_length)
        self.load_in_4bit = bool(load_in_4bit)
        self.fast_inference = bool(fast_inference)
        self.vllm_gpu_memory_utilization = float(vllm_gpu_memory_utilization)
        checkpointing_modes = {
            "unsloth": "unsloth",
            "standard": True,
            "off": False,
        }
        if gradient_checkpointing not in checkpointing_modes:
            raise ValueError(
                "gradient_checkpointing must be one of: unsloth, standard, off"
            )
        self.gradient_checkpointing = gradient_checkpointing
        peft_gradient_checkpointing = checkpointing_modes[gradient_checkpointing]
        self.backward_stability_scale = float(backward_stability_scale)
        if (
            not math.isfinite(self.backward_stability_scale)
            or self.backward_stability_scale < 1.0
        ):
            raise ValueError("backward_stability_scale must be finite and >= 1")
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_every = max(0, int(checkpoint_every))
        self.training_updates = 0
        self._last_adapter_checkpoint_dir: Optional[str] = None
        self._runtime_mode = "initializing"
        self._last_training_completed_at: Optional[float] = None
        self._last_train_to_generation_seconds: Optional[float] = None
        if checkpoint_dir:
            runtime_root = os.path.dirname(checkpoint_dir)
            self._runtime_lora_dir = os.path.join(runtime_root, "runtime_lora_config")
        else:
            self._runtime_lora_dir = tempfile.mkdtemp(prefix="pitpo_runtime_lora_")

        try:
            from unsloth import FastLanguageModel, is_bfloat16_supported
        except Exception as exc:
            raise RuntimeError(
                "The shared PiT-PO runtime requires Unsloth. Install the "
                "versions listed in environment.yml before launching an experiment."
            ) from exc

        self._fast_language_model = FastLanguageModel
        self._bf16_supported = bool(is_bfloat16_supported()) if device == "cuda" else False
        target_modules = self.lora_target_modules or [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ]

        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_name,
            max_seq_length=self.max_seq_length,
            load_in_4bit=self.load_in_4bit,
            fast_inference=self.fast_inference,
            max_lora_rank=self.lora_r,
            gpu_memory_utilization=self.vllm_gpu_memory_utilization,
            random_state=self.seed,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Apply LoRA (only LoRA parameters are trained)
        if self.use_lora:
            self.model = FastLanguageModel.get_peft_model(
                self.model,
                r=self.lora_r,
                lora_alpha=self.lora_alpha,
                lora_dropout=self.lora_dropout,
                target_modules=target_modules,
                use_gradient_checkpointing=peft_gradient_checkpointing,
                random_state=self.seed,
            )
            try:
                self.model.print_trainable_parameters()
            except Exception:
                pass
        
        # Training data buffer (long-running, analogous to EvoTune's train_dataset)
        self.training_buffer = []
        # Reward of each new experience since the last successful train_step (EvoTune's scores_since_finetune)
        self.scores_since_finetune: list[float] = []

        self.enter_inference_mode()
        logger.info(
            "Initialized shared GRPO runtime with model %s | LoRA=%s | 4bit=%s "
            "| fast_inference=%s | warmup_samples=%s | clip_epsilon=%s | kl_beta=%s",
            model_name,
            self.use_lora,
            self.load_in_4bit,
            self.fast_inference,
            self.warmup_samples,
            self.grpo_clip_epsilon,
            self.grpo_kl_beta,
        )

    def enter_training_mode(self) -> None:
        """Switch the shared model to training without reloading any weights."""
        for_training = getattr(self._fast_language_model, "for_training", None)
        if callable(for_training):
            for_training(self.model)
        self.model.train()
        if hasattr(self.model, "config"):
            self.model.config.use_cache = False
        self._runtime_mode = "training"

    def enter_inference_mode(self) -> None:
        """Switch the same PEFT model to inference; updated LoRA stays live."""
        self._fast_language_model.for_inference(self.model)
        self.model.eval()
        if hasattr(self.model, "config"):
            self.model.config.use_cache = True
        self._runtime_mode = "inference"

    def generate(
        self,
        prompt: str,
        num_return_sequences: int,
        max_new_tokens: int = 1024,
        temperature: float = 0.6,
        top_p: float = 0.95,
        top_k: int = 50,
        repetition_penalty: float = 1.1,
    ) -> List[str]:
        """Generate directly from the colocated model using its current LoRA."""
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("A non-empty prompt is required for generation")
        if num_return_sequences < 1:
            raise ValueError("num_return_sequences must be positive")

        self.enter_inference_mode()
        max_prompt_tokens = max(1, self.max_seq_length - int(max_new_tokens))
        prompt_ids = self.tokenizer(
            prompt.strip(),
            truncation=True,
            max_length=max_prompt_tokens,
            add_special_tokens=False,
        )["input_ids"]
        truncated_prompt = self.tokenizer.decode(
            prompt_ids, skip_special_tokens=False
        )

        if self.fast_inference:
            if not hasattr(self.model, "fast_generate"):
                raise RuntimeError("Unsloth fast inference was requested but is unavailable")
            from vllm import SamplingParams

            sampling_params = SamplingParams(
                n=int(num_return_sequences),
                temperature=float(temperature),
                top_p=float(top_p),
                top_k=int(top_k),
                repetition_penalty=float(repetition_penalty),
                max_tokens=int(max_new_tokens),
                skip_special_tokens=True,
            )
            lora_request = None
            if self.use_lora:
                if not hasattr(self.model, "load_lora"):
                    raise RuntimeError(
                        "Unsloth did not expose in-memory LoRA loading for vLLM"
                    )
                # Unsloth creates only adapter_config.json on the first call;
                # all trainable tensors are transferred directly GPU-to-GPU.
                lora_request = self.model.load_lora(
                    self._runtime_lora_dir, load_tensors=True
                )
            request_outputs = self.model.fast_generate(
                [truncated_prompt],
                sampling_params=sampling_params,
                lora_request=lora_request,
                use_tqdm=False,
            )
            completions = [item.text for item in request_outputs[0].outputs]
        else:
            encoded = self.tokenizer(
                truncated_prompt,
                return_tensors="pt",
                add_special_tokens=False,
            )
            model_device = next(self.model.parameters()).device
            encoded = {name: value.to(model_device) for name, value in encoded.items()}
            prompt_length = encoded["input_ids"].shape[1]
            pad_token_id = self.tokenizer.pad_token_id
            if pad_token_id is None:
                pad_token_id = self.tokenizer.eos_token_id
            with torch.inference_mode():
                outputs = self.model.generate(
                    **encoded,
                    do_sample=True,
                    temperature=float(temperature),
                    top_p=float(top_p),
                    top_k=int(top_k),
                    repetition_penalty=float(repetition_penalty),
                    max_new_tokens=int(max_new_tokens),
                    num_return_sequences=int(num_return_sequences),
                    pad_token_id=pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    use_cache=True,
                )
            completions = self.tokenizer.batch_decode(
                outputs[:, prompt_length:], skip_special_tokens=True
            )

        if len(completions) != num_return_sequences:
            raise RuntimeError(
                "Shared runtime returned an unexpected number of completions: "
                f"expected {num_return_sequences}, got {len(completions)}"
            )
        if self._last_training_completed_at is not None:
            self._last_train_to_generation_seconds = (
                time.monotonic() - self._last_training_completed_at
            )
            self._last_training_completed_at = None
        return completions

    def consume_train_to_generation_seconds(self) -> Optional[float]:
        """Return the most recent train-to-next-generation delay exactly once."""
        elapsed = self._last_train_to_generation_seconds
        self._last_train_to_generation_seconds = None
        return elapsed

    @staticmethod
    def _json_default(value):
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, torch.Tensor):
            return value.detach().cpu().tolist()
        return str(value)

    def save_adapter_checkpoint(self, save_dir: str, reason: str) -> str:
        """Publish a complete adapter, then delete superseded recovery checkpoints.

        One completed checkpoint is retained per run, across periodic, final
        and emergency saves. A failed save leaves the previous checkpoint intact.
        """
        if not self.use_lora:
            logger.info("Skipping adapter checkpoint because GRPO/LoRA is disabled")
            return save_dir
        destination = os.path.abspath(save_dir)
        parent = os.path.dirname(destination)
        name = os.path.basename(destination)
        if not name:
            raise ValueError("A checkpoint must have its own directory")
        os.makedirs(parent, exist_ok=True)
        staging = tempfile.mkdtemp(prefix=f".{name}.pending-", dir=parent)
        state = {
            "reason": reason,
            "model_name": self.model_name,
            "seed": getattr(self, 'seed', 3408),
            "training_updates": self.training_updates,
            "candidate_count": self.candidate_count,
            "warmup_samples": self.warmup_samples,
            "grpo_clip_epsilon": self.grpo_clip_epsilon,
            "grpo_kl_beta": self.grpo_kl_beta,
            "checkpoint_retention_limit": 1,
            "scores_since_finetune": self.scores_since_finetune,
            "training_buffer": self.training_buffer,
        }
        backup = None
        try:
            self.model.save_pretrained(staging, safe_serialization=True)
            self.tokenizer.save_pretrained(staging)
            with open(os.path.join(staging, "pitpo_training_state.json"), "w", encoding="utf-8") as handle:
                json.dump(state, handle, ensure_ascii=False, indent=2, default=self._json_default)
                handle.flush()
                os.fsync(handle.fileno())
            # Validate the actual safetensors header and payload size before
            # making this the replacement for a usable recovery checkpoint.
            from safetensors import safe_open
            with safe_open(os.path.join(staging, "adapter_model.safetensors"), framework="pt") as weights:
                if not weights.keys():
                    raise RuntimeError("The saved LoRA checkpoint has no tensors")
            with open(os.path.join(staging, "adapter_config.json"), encoding="utf-8") as handle:
                json.load(handle)

            if os.path.lexists(destination):
                if not os.path.isdir(destination) or os.path.islink(destination):
                    raise ValueError("The checkpoint destination must be a regular directory")
                existing_state = os.path.join(destination, "pitpo_training_state.json")
                if os.listdir(destination) and not os.path.isfile(existing_state):
                    raise ValueError("Refusing to replace a directory without checkpoint state")
                backup = tempfile.mkdtemp(prefix=f".{name}.previous-", dir=parent)
                os.rmdir(backup)
                os.replace(destination, backup)
            try:
                os.replace(staging, destination)
            except BaseException:
                if backup is not None:
                    os.replace(backup, destination)
                    backup = None
                raise
        finally:
            if os.path.isdir(staging):
                shutil.rmtree(staging)

        if backup is not None:
            shutil.rmtree(backup)
        self._prune_adapter_checkpoints(destination)
        self._last_adapter_checkpoint_dir = destination
        logger.info("Saved %s adapter checkpoint to %s; retaining this checkpoint only", reason, save_dir)
        return save_dir

    def _prune_adapter_checkpoints(self, latest_dir: str) -> None:
        """Remove this run's older adapters only after the new one is published."""
        candidates = set()
        checkpoint_dir = getattr(self, "checkpoint_dir", None)
        if checkpoint_dir:
            checkpoint_dir = os.path.abspath(checkpoint_dir)
            if os.path.isdir(checkpoint_dir):
                with os.scandir(checkpoint_dir) as entries:
                    candidates.update(entry.path for entry in entries
                                      if entry.is_dir(follow_symlinks=False)
                                      and re.fullmatch(r"checkpoint-\d+", entry.name))
            run_root = os.path.dirname(checkpoint_dir)
            candidates.update(os.path.join(run_root, name)
                              for name in ("final_adapter", "emergency_adapter"))
        previous = getattr(self, "_last_adapter_checkpoint_dir", None)
        if previous:
            candidates.add(previous)
        for candidate in sorted(candidates):
            if candidate == latest_dir or not os.path.isdir(candidate) or os.path.islink(candidate):
                continue
            state_path = os.path.join(candidate, "pitpo_training_state.json")
            try:
                with open(state_path, encoding="utf-8") as handle:
                    previous_state = json.load(handle)
            except (OSError, ValueError):
                continue
            if previous_state.get("model_name") != self.model_name:
                continue
            shutil.rmtree(candidate)
            logger.info("Deleted superseded adapter checkpoint: %s", candidate)

    def _save_periodic_checkpoint_if_needed(self) -> None:
        if not self.checkpoint_dir or not self.checkpoint_every:
            return
        if self.training_updates % self.checkpoint_every != 0:
            return
        save_dir = os.path.join(
            self.checkpoint_dir, f"checkpoint-{self.training_updates:06d}"
        )
        self.save_adapter_checkpoint(save_dir, reason="periodic")

    def _is_valid_experience(self, exp: Dict[str, Any]) -> bool:
        """Check if an experience has a valid (non-None) score/MSE.
        Additional rule: if metadata contains a score field that is positive (usually meaning MSE was negated), treat the entry as invalid.
        """
        try:
            mse = exp.get("mse", None)
            if mse is None:
                return False
            # MSE must be a finite positive number
            if not (isinstance(mse, (int, float)) and np.isfinite(mse) and mse > 0):
                return False
            # If a positive score is provided (meaning -MSE > 0, i.e. MSE < 0), treat as invalid
            meta = exp.get("metadata", {}) or {}
            score_val = None
            for key in ("score", "raw_score", "eval_score"):
                val = meta.get(key, None)
                if val is None:
                    continue
                try:
                    score_val = float(val)
                    break
                except Exception:
                    continue
            if score_val is not None and score_val > 0:
                return False
            return True
        except Exception:
            return False

    def _evotune_reward_threshold(self) -> float:
        """Matches EvoTune `calculate_dataset_statistics`: take the (100 - percentile) quantile of scores_since_finetune."""
        if not self.scores_since_finetune:
            return float("-inf")
        arr = np.asarray(self.scores_since_finetune, dtype=np.float64)
        return float(np.percentile(arr, 100 - _DPO_STYLE_PERCENTILE))

    def mse_to_reward(self, mse: float) -> float:
        """
        Convert MSE to reward using enhanced log transformation.
        Lower MSE results in higher reward, very high MSE becomes penalty.
        
        Args:
            mse: Mean squared error value
            
        Returns:
            Reward value (positive for good performance, negative for poor performance)
        """
        # Ensure MSE is at least min_threshold to prevent log(0)
        mse = max(mse, self.min_mse_threshold)
        
        # Enhanced reward function based on log transformation
        # reward = -log(MSE) * scaling, but with penalty for very poor performance
        base_reward = -np.log(mse) * self.reward_scaling
        
        # Add penalty for very poor performance (MSE > 10)
        # This creates a stronger negative signal for bad solutions
        if mse > 10.0:
            penalty_factor = np.log(mse / 10.0) * self.reward_scaling * 0.5
            base_reward -= penalty_factor
        
        # Bonus for excellent performance (MSE < 0.01)
        elif mse < 0.01:
            bonus_factor = np.log(0.01 / mse) * self.reward_scaling * 0.3
            base_reward += bonus_factor
            
        return base_reward

    def add_experience(
        self,
        prompt: str,
        response: str,
        mse: float,
        **metadata
    ):
        """
        Add training experience to buffer with enhanced management strategy.
        
        Args:
            prompt: Input prompt
            response: Model response
            mse: Mean squared error for this response
            **metadata: Additional metadata
        """
        sample_index = metadata.get("global_sample_nums")
        if sample_index is not None:
            self.candidate_count = max(self.candidate_count, int(sample_index))

        # First, filter for input validity and positive-score entries (score>0 => drop)
        if mse is None or not np.isfinite(mse) or mse <= 0:
            logger.debug(f"Discarding experience due to invalid MSE: {mse}")
            return
        meta_score = None
        for key in ("score", "raw_score", "eval_score"):
            if key in metadata and metadata[key] is not None:
                try:
                    meta_score = float(metadata[key])
                    break
                except Exception:
                    continue
        if meta_score is not None and meta_score > 0:
            logger.debug(f"Discarding experience due to positive score (implies negative MSE): score={meta_score}, MSE={mse}")
            return
        
        # === R_fit: Fitting accuracy reward (Paper Eq. 7) ===
        r_fit = self.mse_to_reward(mse)

        # === P_cplx: AST complexity penalty (Paper Eq. 8) ===
        p_cplx = 0.0
        ast_count = 0
        try:
            fns = extract_equation_functions_from_text(response)
            if fns:
                analysis = analyze_equation_function(fns[0].source, symbol_units=self.phy_units or {})
                ast_count = int(analysis.get("ast_node_count", 0))
                p_cplx = self.ast_complexity_weight * ast_count
        except Exception:
            pass

        # === P_phy: Gated physical penalty (Paper Eq. 9) ===
        p_phy = 0.0
        dim_ok = True
        diff_ok = True
        if self.enable_physical_penalty and mse < self.phy_gate_mse_threshold:
            p_phy = self._compute_physical_penalty(response, mse)

        # === R_global = R_fit - P_cplx - P_phy (Paper Eq. 10) ===
        reward = r_fit - p_cplx - p_phy

        # === Coefficient redundancy penalty for token-level (Paper Eq. 5-6) ===
        coef_penalty_val = 0.0
        coef_details = {}
        if self.enable_coef_penalty:
            try:
                optimized_params_by_test = metadata.get("optimized_params_by_test", None)
            except Exception:
                optimized_params_by_test = None
            if optimized_params_by_test:
                coef_penalty_val, coef_details = compute_coefficient_penalty(
                    response_text=response,
                    optimized_params_by_test=optimized_params_by_test,
                    ratio_threshold=self.coef_ratio_threshold,
                    penalty_weight=self.coef_penalty_weight,
                )
                if coef_penalty_val > 0:
                    reward -= coef_penalty_val
                    logger.debug(f"Applied coefficient penalty: {coef_penalty_val:.4f}; reward now {reward:.2f}")
        
        experience = {
            "prompt": prompt,
            "response": response,
            "mse": mse,
            "reward": reward,
            "metadata": metadata,
            "timestamp": time.time()  # Add timestamp for experience aging
        }
        
        # Record reward decomposition for debugging
        try:
            experience["metadata"]["reward_decomposition"] = {
                "r_fit": float(r_fit),
                "p_cplx": float(p_cplx),
                "ast_node_count": int(ast_count),
                "p_phy": float(p_phy),
                "coef_penalty": float(coef_penalty_val),
                "total_reward": float(reward),
            }
            if coef_details:
                experience["metadata"]["coef_penalty_details"] = coef_details
        except Exception:
            pass

        self.training_buffer.append(experience)
        
        # Enhanced buffer management
        if len(self.training_buffer) > self.buffer_size:
            # Strategy: Keep best experiences + some diversity
            self.training_buffer.sort(key=lambda x: x["reward"], reverse=True)
            
            # Keep top 70% best experiences
            top_keep = int(self.buffer_size * 0.7)
            top_experiences = self.training_buffer[:top_keep]
            
            # From remaining, keep 30% for diversity (varied MSE)
            remaining = self.training_buffer[top_keep:]
            remaining_keep = self.buffer_size - top_keep
            
            if len(remaining) > remaining_keep:
                # Sort remaining by MSE diversity (keep spread of MSE values)
                mse_values = [exp["mse"] for exp in remaining]
                mse_sorted_indices = np.argsort(mse_values)
                
                # Take evenly spaced samples for diversity
                step = len(mse_sorted_indices) // remaining_keep if remaining_keep > 0 else len(mse_sorted_indices)
                diverse_indices = mse_sorted_indices[::step][:remaining_keep]
                diverse_experiences = [remaining[i] for i in diverse_indices]
            else:
                diverse_experiences = remaining
                
            self.training_buffer = top_experiences + diverse_experiences
            
        # EvoTune: every experience successfully written to the buffer has an associated score for the next pre-fine-tune quantile threshold
        self.scores_since_finetune.append(float(reward))
        
        logger.debug(f"Added experience: MSE={mse:.2e}, Reward={reward:.2f}, Buffer size={len(self.training_buffer)}")
        
        # Log buffer statistics periodically
        if len(self.training_buffer) % 100 == 0:
            mse_values = [exp["mse"] for exp in self.training_buffer]
            logger.info(f"Buffer stats - Size: {len(self.training_buffer)}, "
                       f"MSE range: [{np.min(mse_values):.2e}, {np.max(mse_values):.2e}], "
                       f"MSE mean: {np.mean(mse_values):.2e}")
    
    def prepare_training_data(self, experiences: Optional[List[Dict[str, Any]]] = None) -> Dataset:
        """
        Prepare training dataset from buffer experiences.
        If `experiences` is provided, only those are used; otherwise use the entire buffer.
        
        Returns:
            Hugging Face Dataset for training
        """
        from datasets import Dataset

        exps = experiences if experiences is not None else self.training_buffer
        if not exps:
            raise ValueError("No training data in buffer")
        
        # Prepare input texts and rewards
        texts: List[str] = []
        rewards: List[float] = []
        
        for exp in exps:
            # Combine prompt and response for training
            full_text = exp["prompt"] + exp["response"]
            texts.append(full_text)
            rewards.append(exp["reward"])
        
        # Tokenize texts
        encodings = self.tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        # Build per-token penalty weights (true token-level weighting)
        # 1) Compute offset mappings aligned with above tokenization
        try:
            enc_with_offsets = self.tokenizer(
                texts,
                truncation=True,
                padding=True,
                max_length=self.max_length,
                return_offsets_mapping=True,
            )
            offsets_batch = enc_with_offsets.get("offset_mapping", None)
        except Exception:
            offsets_batch = None

        token_penalty_weights: List[List[float]] = []
        completion_masks: List[List[int]] = []
        for exp_idx, exp in enumerate(exps):
            seq_len = int(encodings["input_ids"][exp_idx].shape[0]) if hasattr(encodings["input_ids"][exp_idx], 'shape') else len(encodings["input_ids"][exp_idx])
            weights_vec = [0.0] * seq_len

            # Recover prompt/response for this sample
            prompt_text = exp.get("prompt", "")
            response_text = exp.get("response", "")
            full_text = (prompt_text or "") + (response_text or "")
            prompt_char_len = len(prompt_text or "")

            # Only generated tokens participate in the policy objective and KL.
            # Offsets also handle a token spanning the prompt/response boundary.
            completion_mask = [0] * seq_len
            if offsets_batch is not None and exp_idx < len(offsets_batch):
                for t_idx, (tok_s, tok_e) in enumerate(offsets_batch[exp_idx]):
                    if tok_e > prompt_char_len and encodings["attention_mask"][exp_idx][t_idx]:
                        completion_mask[t_idx] = 1
            else:
                # Slow-tokenizer fallback: find the common token prefix, rather
                # than assuming separate tokenization keeps the boundary token.
                prompt_ids = self.tokenizer(
                    prompt_text, truncation=True, max_length=self.max_length,
                )["input_ids"]
                full_ids = encodings["input_ids"][exp_idx].tolist()
                prefix_length = 0
                for prompt_id, full_id in zip(prompt_ids, full_ids):
                    if prompt_id != full_id:
                        break
                    prefix_length += 1
                for t_idx in range(prefix_length, seq_len):
                    completion_mask[t_idx] = int(encodings["attention_mask"][exp_idx][t_idx])
            if not any(completion_mask[1:]):
                raise ValueError(
                    "Training sequence has no completion tokens; increase max_length "
                    "or shorten the search prompt"
                )
            completion_masks.append(completion_mask)

            # Small-index penalties per index based on optimized params
            penalty_by_index: Dict[int, float] = {}
            small_indices: List[int] = []
            try:
                from .coef_penalty import compute_coefficient_penalty
                opt_params = (exp.get("metadata", {}) or {}).get("optimized_params_by_test", None)
                if opt_params:
                    _penalty_total, details = compute_coefficient_penalty(
                        response_text=response_text,
                        optimized_params_by_test=opt_params,
                        ratio_threshold=getattr(self, 'coef_ratio_threshold', 0.05),
                        penalty_weight=getattr(self, 'coef_penalty_weight', 1.0),
                    )
                    penalty_by_index = details.get("penalty_by_index", {}) if isinstance(details, dict) else {}
                    small_indices = details.get("redundant_indices", details.get("small_indices", [])) if isinstance(details, dict) else []
            except Exception:
                penalty_by_index = {}
                small_indices = []

            # Build char spans for occurrences of params[i] in response
            spans: List[Tuple[int, int, float]] = []  # (global_start, global_end, weight)
            if response_text and small_indices:
                for i in small_indices:
                    if i in penalty_by_index and penalty_by_index[i] > 0:
                        pattern = rf"params\s*\[\s*{i}\s*\]"
                        for m in re.finditer(pattern, response_text):
                            s, e = m.span()
                            # map to full_text global char positions
                            spans.append((prompt_char_len + s, prompt_char_len + e, float(penalty_by_index[i])))

            # Project spans onto token offsets (if available)
            if offsets_batch is not None and isinstance(offsets_batch, list) and exp_idx < len(offsets_batch):
                offsets = offsets_batch[exp_idx]
                for t_idx, off in enumerate(offsets):
                    try:
                        tok_s, tok_e = int(off[0]), int(off[1])
                    except Exception:
                        continue
                    if tok_e <= tok_s:
                        continue
                    # overlap with any span
                    w_sum = 0.0
                    for (gs, ge, w) in spans:
                        if not (tok_e <= gs or tok_s >= ge):
                            w_sum += float(w)
                    if w_sum > 0:
                        weights_vec[t_idx] = w_sum
            # If no offsets, leave weights as zeros (fallback to sample-level penalty only)
            token_penalty_weights.append(weights_vec)

        # Create dataset
        dataset_dict = {
            "input_ids": encodings["input_ids"].tolist(),
            "attention_mask": encodings["attention_mask"].tolist(),
            "completion_mask": completion_masks,
            "rewards": rewards,
            "token_penalty_weights": token_penalty_weights,
        }
        
        dataset = Dataset.from_dict(dataset_dict)
        return dataset

    @staticmethod
    def _token_log_probs(
        logits: torch.Tensor,
        labels: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Gather causal token log probabilities, excluding masked targets first."""
        shift_logits = logits[..., :-1, :]
        shift_labels = labels[..., 1:].contiguous()
        valid_token_mask = attention_mask[..., 1:].bool() & shift_labels.ne(-100)
        if not bool(valid_token_mask.any()):
            raise FloatingPointError("GRPO batch contains no valid language-model tokens")

        valid_logits = shift_logits[valid_token_mask].float()
        if os.environ.get("PITPO_DEBUG_NUMERICS") == "1" and valid_logits.requires_grad:
            valid_logits.register_hook(_numeric_gradient_hook("valid_logits"))
        valid_log_probs = -torch.nn.functional.cross_entropy(
            valid_logits, shift_labels[valid_token_mask], reduction="none",
        )
        if not bool(torch.isfinite(valid_log_probs).all()):
            raise FloatingPointError("GRPO produced non-finite valid-token log probabilities")
        log_probs = torch.zeros_like(shift_labels, dtype=torch.float32).masked_scatter(
            valid_token_mask, valid_log_probs,
        )
        return log_probs, valid_token_mask

    def _training_autocast(self):
        if self.device != "cuda":
            return nullcontext()
        return torch.autocast(
            "cuda", dtype=torch.bfloat16 if self._bf16_supported else torch.float16,
        )

    def _cache_policy_log_probs(self, dataset):
        """Freeze old-policy and base-reference log probabilities before updates.

        Only small [sample, token] arrays are kept on CPU. Disabling the current
        LoRA uses the frozen base weights as pi_ref without a second GPU model.
        """
        if self.grpo_kl_beta > 0 and not callable(getattr(self.model, "disable_adapter", None)):
            raise RuntimeError("GRPO KL requires a LoRA model exposing disable_adapter()")
        model_device = next(self.model.parameters()).device
        was_training = self.model.training
        old_rows, ref_rows = [], []
        self.model.eval()
        try:
            with torch.no_grad():
                for start in range(0, len(dataset), self.per_device_train_batch_size):
                    rows = dataset[start:start + self.per_device_train_batch_size]
                    input_ids = torch.tensor(rows["input_ids"], dtype=torch.long, device=model_device)
                    policy_mask = torch.tensor(rows["attention_mask"], dtype=torch.long, device=model_device)
                    completion_mask = torch.tensor(rows["completion_mask"], dtype=torch.bool, device=model_device)
                    labels = input_ids.masked_fill(~(completion_mask & policy_mask.bool()), -100)
                    model_mask = self._causal_training_attention_mask(policy_mask)
                    with self._training_autocast():
                        outputs = self.model(input_ids=input_ids, attention_mask=model_mask)
                        old_log_probs, _ = self._token_log_probs(outputs.logits, labels, policy_mask)
                    old_rows.extend(old_log_probs.cpu().tolist())
                    del outputs, old_log_probs

                    if self.grpo_kl_beta > 0:
                        with self.model.disable_adapter(), self._training_autocast():
                            outputs = self.model(input_ids=input_ids, attention_mask=model_mask)
                            ref_log_probs, _ = self._token_log_probs(outputs.logits, labels, policy_mask)
                        ref_rows.extend(ref_log_probs.cpu().tolist())
                        del outputs, ref_log_probs
        finally:
            self.model.train(was_training)

        dataset = dataset.add_column("old_log_probs", old_rows)
        if self.grpo_kl_beta > 0:
            dataset = dataset.add_column("ref_log_probs", ref_rows)
        return dataset

    def compute_grpo_loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        rewards: torch.Tensor,
        attention_mask: torch.Tensor,
        token_penalty_weights: Optional[torch.Tensor] = None,
        old_log_probs: Optional[torch.Tensor] = None,
        ref_log_probs: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute the clipped GRPO objective with PiT-PO token advantages.

        Token-aware advantage:
            A_hat_{i,k} = (R_global(o_i) - mu_group) / sigma_group  -  P_{i,k}

        L = -mean_i mean_k[min(ratio * A, clip(ratio, 1-eps, 1+eps) * A)
                          - beta * KL(pi_theta || pi_ref)]
        old_log_probs/ref_log_probs have shape [batch, sequence_length - 1]
        and are held fixed throughout this training update.
        """
        # Shift logits and labels for language modeling.  Unsloth pads with
        # <|finetune_right_pad_id|>, whose logit can be -inf.  Computing that
        # token's log-probability and multiplying it by an attention mask of
        # zero produces NaN gradients (0 * -inf).  Select valid token rows
        # before cross entropy so padding never enters the loss graph.
        log_probs, valid_token_mask = self._token_log_probs(logits, labels, attention_mask)
        valid_log_probs = log_probs[valid_token_mask]

        # === Group normalization of rewards (Paper Eq. 4) ===
        rewards = rewards.float()
        if not bool(torch.isfinite(rewards).all()):
            raise FloatingPointError("GRPO rewards contain NaN or infinity")
        mu = rewards.mean()
        sigma = (
            rewards.std().clamp_min(1e-8)
            if rewards.numel() > 1
            else rewards.new_tensor(1.0)
        )
        normalized_advantages = (rewards - mu) / sigma  # [batch_size]

        # === Token-aware advantage (Paper Eq. 11) ===
        # Expand normalized advantage to per-token: [batch_size, seq_len-1]
        seq_advantages = normalized_advantages.unsqueeze(-1).expand_as(log_probs)

        if token_penalty_weights is not None:
            shift_penalty = token_penalty_weights[..., 1:].float()
            if not bool(torch.isfinite(shift_penalty[valid_token_mask]).all()):
                raise FloatingPointError("GRPO token penalties contain NaN or infinity")
            seq_advantages = seq_advantages - shift_penalty

        if old_log_probs is None or old_log_probs.shape != log_probs.shape:
            raise ValueError("Fixed old_log_probs must have shape [batch, sequence_length - 1]")
        valid_old = old_log_probs.detach().float()[valid_token_mask]
        if not bool(torch.isfinite(valid_old).all()):
            raise FloatingPointError("GRPO old-policy log probabilities contain NaN or infinity")
        ratios = torch.exp(valid_log_probs - valid_old)
        if not bool(torch.isfinite(ratios).all()):
            raise FloatingPointError("GRPO probability ratios contain NaN or infinity")
        advantages = seq_advantages[valid_token_mask]
        unclipped = ratios * advantages
        clipped = ratios.clamp(1.0 - self.grpo_clip_epsilon, 1.0 + self.grpo_clip_epsilon) * advantages
        surrogate = torch.minimum(unclipped, clipped)

        # Non-negative per-token KL estimator used by GRPO (TRL/DeepSeekMath).
        # expm1 is accurate when pi_ref and pi_theta are nearly identical.
        kl = torch.zeros_like(valid_log_probs)
        if self.grpo_kl_beta > 0:
            if ref_log_probs is None or ref_log_probs.shape != log_probs.shape:
                raise ValueError("Fixed ref_log_probs are required when grpo_kl_beta > 0")
            valid_ref = ref_log_probs.detach().float()[valid_token_mask]
            if not bool(torch.isfinite(valid_ref).all()):
                raise FloatingPointError("GRPO reference log probabilities contain NaN or infinity")
            ref_log_ratio = valid_ref - valid_log_probs
            kl = (torch.expm1(ref_log_ratio) - ref_log_ratio).clamp_min(0.0)

        def sequence_mean(values):
            token_values = torch.zeros_like(log_probs).masked_scatter(valid_token_mask, values)
            return (token_values.sum(dim=-1) / valid_token_mask.sum(dim=-1).clamp_min(1)).mean()

        grpo_loss = sequence_mean(-surrogate + self.grpo_kl_beta * kl)

        if not bool(torch.isfinite(grpo_loss)):
            raise FloatingPointError("GRPO loss is NaN or infinity")

        self._last_loss_metrics = {
            "policy_loss": float(sequence_mean(-surrogate).detach().item()),
            "kl": float(sequence_mean(kl).detach().item()),
            "clip_fraction": float((unclipped > clipped).float().mean().detach().item()),
        }

        return grpo_loss

    def _causal_training_attention_mask(
        self, policy_attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """Keep right-pad query activations finite without changing real tokens."""
        if getattr(self.tokenizer, "padding_side", "right") != "right":
            raise RuntimeError("PiT-PO training requires right-side padding")
        return torch.ones_like(policy_attention_mask)

    class GRPOTrainingDataset(torch.utils.data.Dataset):
        """Custom dataset for GRPO training with proper tensor handling."""
        def __init__(self, dataset):
            self.dataset = dataset
            
        def __len__(self):
            return len(self.dataset)
            
        def __getitem__(self, idx):
            item = self.dataset[idx]
            input_ids = torch.tensor(item["input_ids"], dtype=torch.long)
            attention_mask = torch.tensor(item["attention_mask"], dtype=torch.long)
            labels = input_ids.clone()
            labels.masked_fill_(attention_mask.eq(0), -100)
            if "completion_mask" in item:
                labels.masked_fill_(torch.tensor(item["completion_mask"]).eq(0), -100)
            result = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels,
                "grpo_rewards": item["rewards"],  # use a different name to avoid being dropped
                "token_penalty_weights": torch.tensor(item.get("token_penalty_weights", [0.0]*len(item["input_ids"])), dtype=torch.float32),
            }
            for name in ("old_log_probs", "ref_log_probs"):
                if name in item:
                    result[name] = torch.tensor(item[name], dtype=torch.float32)
            # Debug info
            if idx == 0:  # print debug info only for the first sample
                logger.info(f"Dataset __getitem__ returning keys: {list(result.keys())}")
                logger.info(f"Rewards value: {result['grpo_rewards']}")
            return result
    
    def train_step(self, num_epochs: int = 1) -> Dict[str, float]:
        """
        Perform one training step using experiences in buffer with enhanced strategy.
        
        Args:
            num_epochs: Number of training epochs
            
        Returns:
            Training statistics
        """
        # Use only samples with a valid score (MSE positive and finite)
        valid_exps = [exp for exp in self.training_buffer if self._is_valid_experience(exp)]
        if len(valid_exps) == 0:
            logger.warning("No valid data for training (all scores invalid). Skipping.")
            return {}
        # EvoTune-style: reward >= np.percentile(scores_since_finetune, 100 - _DPO_STYLE_PERCENTILE)
        reward_thr = self._evotune_reward_threshold()
        filtered = [e for e in valid_exps if e.get("reward", float("-inf")) >= reward_thr]
        if len(filtered) == 0:
            logger.warning(
                "EvoTune-style filter removed all valid samples; falling back to unfiltered valid_exps."
            )
            filtered = valid_exps
        take_count = min(len(filtered), self.required_train_samples)
        selected_exps = filtered[-take_count:]
        
        logger.info("Starting GRPO reinforcement-learning fine-tune...")
        logger.info(
            f"📚 EvoTune-style: magic percentile={_DPO_STYLE_PERCENTILE} → reward threshold={reward_thr:.6f} "
            f"| kept {len(filtered)}/{len(valid_exps)} valid in buffer | train batch {len(selected_exps)}"
        )
        logger.info(f"Training data: using {len(selected_exps)} valid samples")
        logger.info(f"Training epochs: {num_epochs}")
        
        # Prepare training data
        dataset = self.prepare_training_data(experiences=selected_exps)
        
        # Calculate buffer statistics for adaptive learning rate
        mse_values = [exp["mse"] for exp in selected_exps]
        avg_mse = np.mean(mse_values)
        mse_std = np.std(mse_values)
        
        # Adaptive learning rate based on MSE quality
        adaptive_lr = self.learning_rate
        if avg_mse < 0.1:  # Very good results - use higher LR to exploit
            adaptive_lr = self.learning_rate * 1.5
        elif avg_mse > 10.0:  # Poor results - use lower LR for stability
            adaptive_lr = self.learning_rate * 0.5
        
        # Adaptive batch size (at least 1, at most half the samples or batch_size)
        adaptive_batch_size = min(self.batch_size, max(1, len(selected_exps) // 2))
        adaptive_batch_size = max(adaptive_batch_size, 1)
        
        # The colocated Unsloth runtime owns placement for the entire run.
        # Moving a 4-bit model after construction is unsupported and would also
        # break the shared inference state.
        self.enter_training_mode()
        dataset = self._cache_policy_log_probs(dataset)
        train_dataset = self.GRPOTrainingDataset(dataset)

        # Define a custom data collator class
        class GRPODataCollator:
            def __init__(self, tokenizer):
                self.tokenizer = tokenizer
                
            def __call__(self, features):
                """Custom data collator that preserves rewards field."""
                # Debug info
                logger.info(f"Data collator received {len(features)} features")
                if features:
                    logger.info(f"First feature keys: {list(features[0].keys())}")
                
                # Extract rewards (using the new field name)
                rewards = [f.pop("grpo_rewards", 0.0) for f in features]  # pop to avoid conflicts
                
                # Let transformers handle other fields with its default behavior
                batch = {}
                
                # Manually handle each field
                if features:
                    batch["input_ids"] = torch.stack([f["input_ids"] for f in features])
                    batch["attention_mask"] = torch.stack([f["attention_mask"] for f in features])  
                    batch["labels"] = torch.stack([f["labels"] for f in features])
                    batch["rewards"] = torch.tensor(rewards, dtype=torch.float32)  # still called rewards downstream
                    # token-level penalty weights
                    if "token_penalty_weights" in features[0]:
                        batch["token_penalty_weights"] = torch.stack([f["token_penalty_weights"] for f in features])
                    for name in ("old_log_probs", "ref_log_probs"):
                        if name in features[0]:
                            batch[name] = torch.stack([f[name] for f in features])
                
                return batch

        # Initialize a plain PyTorch loader.  Hugging Face Trainer/Accelerate
        # must not wrap this already-patched shared Unsloth model: on the exact
        # same finite batch that direct autograd handles correctly, that
        # combination produced NaN LoRA gradients before the first optimizer
        # step.  CALM likewise keeps ownership of one model in one process.
        data_collator = GRPODataCollator(self.tokenizer)
        data_generator = torch.Generator()
        data_generator.manual_seed(getattr(self, 'seed', 3408) + self.training_updates)
        data_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self.per_device_train_batch_size,
            shuffle=True,
            drop_last=False,
            collate_fn=data_collator,
            generator=data_generator,
        )
        
        # Train model
        try:
            if int(num_epochs) < 1:
                raise ValueError("num_epochs must be positive")

            logger.info(f"Starting training; dataset size: {len(train_dataset)}")
            logger.info(f"Micro-batch: {self.per_device_train_batch_size}, gradient-accum: {self.gradient_accumulation_steps} (effective batch ~= {self.per_device_train_batch_size * self.gradient_accumulation_steps})")
            logger.info(f"Adaptive learning rate: {adaptive_lr:.2e}")

            trainable_parameters = [
                parameter for parameter in self.model.parameters()
                if parameter.requires_grad
            ]
            if not trainable_parameters:
                raise RuntimeError("GRPO has no trainable LoRA parameters")
            optimizer = torch.optim.AdamW(
                trainable_parameters,
                lr=adaptive_lr,
                betas=(0.9, 0.99),
                weight_decay=0.1,
            )
            use_fp16_scaler = self.device == "cuda" and not self._bf16_supported
            scaler = torch.amp.GradScaler("cuda", enabled=use_fp16_scaler)
            model_device = next(self.model.parameters()).device
            debug_numerics = os.environ.get("PITPO_DEBUG_NUMERICS") == "1"
            trace_training = os.environ.get("PITPO_TRACE_TRAIN") == "1"
            accumulated_losses: List[float] = []
            loss_metrics: List[Dict[str, float]] = []
            optimizer_steps = 0
            optimizer.zero_grad(set_to_none=True)

            for epoch_index in range(int(num_epochs)):
                batches_in_epoch = len(data_loader)
                for batch_index, batch in enumerate(data_loader):
                    batch = {
                        name: value.to(model_device)
                        for name, value in batch.items()
                    }
                    rewards = batch.pop("rewards")
                    labels = batch.pop("labels")
                    token_penalty_weights = batch.pop(
                        "token_penalty_weights", None
                    )
                    old_log_probs = batch.pop("old_log_probs")
                    ref_log_probs = batch.pop("ref_log_probs", None)
                    logger.info(
                        "GRPO epoch %d/%d batch %d/%d rewards=%s",
                        epoch_index + 1,
                        int(num_epochs),
                        batch_index + 1,
                        batches_in_epoch,
                        rewards.detach().cpu().tolist(),
                    )
                    if trace_training:
                        print(
                            f"[GRPO_TRACE] epoch={epoch_index + 1} "
                            f"batch={batch_index + 1}/{batches_in_epoch} "
                            f"rewards={rewards.detach().cpu().tolist()}",
                            flush=True,
                        )

                    window_start = (
                        batch_index // self.gradient_accumulation_steps
                    ) * self.gradient_accumulation_steps
                    window_size = min(
                        self.gradient_accumulation_steps,
                        batches_in_epoch - window_start,
                    )
                    with self._training_autocast():
                        # Right-padded query rows can be fully masked inside
                        # the attention kernel and produce NaN activations.
                        # Even though their LM loss is masked, LoRA weight
                        # gradients then encounter 0 * NaN.  Let pad queries
                        # attend causally during the model forward; real-token
                        # states are unchanged because every pad is to their
                        # right.  The original mask below still excludes all
                        # pad targets from the policy loss.
                        policy_attention_mask = batch["attention_mask"]
                        model_inputs = dict(batch)
                        model_inputs["attention_mask"] = self._causal_training_attention_mask(
                            policy_attention_mask
                        )
                        outputs = self.model(
                            **model_inputs,
                            output_hidden_states=debug_numerics,
                        )
                        logits = outputs.logits
                        if debug_numerics:
                            logits.register_hook(
                                _numeric_gradient_hook("logits")
                            )
                        raw_loss = self.compute_grpo_loss(
                            logits=logits,
                            labels=labels,
                            rewards=rewards,
                            attention_mask=policy_attention_mask,
                            token_penalty_weights=token_penalty_weights,
                            old_log_probs=old_log_probs,
                            ref_log_probs=ref_log_probs,
                        )
                        scaled_loss = raw_loss / (
                            float(window_size) * self.backward_stability_scale
                        )

                    accumulated_losses.append(float(raw_loss.detach().item()))
                    loss_metrics.append(dict(self._last_loss_metrics))
                    if scaler.is_enabled():
                        scaler.scale(scaled_loss).backward()
                    else:
                        scaled_loss.backward()

                    bad_gradients = []
                    for name, parameter in self.model.named_parameters():
                        gradient = parameter.grad
                        if (
                            parameter.requires_grad
                            and gradient is not None
                            and not bool(torch.isfinite(gradient).all())
                        ):
                            bad_gradients.append(name)
                            if len(bad_gradients) >= 5:
                                break
                    if bad_gradients:
                        raise FloatingPointError(
                            "Non-finite GRPO gradients detected before optimizer step: "
                            + ", ".join(bad_gradients)
                        )
                    if trace_training:
                        print(
                            f"[GRPO_TRACE] backward_finite epoch={epoch_index + 1} "
                            f"batch={batch_index + 1}",
                            flush=True,
                        )

                    end_of_window = (
                        (batch_index + 1) % self.gradient_accumulation_steps == 0
                        or batch_index + 1 == batches_in_epoch
                    )
                    if end_of_window:
                        if scaler.is_enabled():
                            scaler.unscale_(optimizer)
                        # Undo only PiT-PO's deliberate down-scaling.  This is
                        # done after the numerically sensitive model backward
                        # and before clipping, so the optimizer sees the same
                        # policy gradient as without the stabilization.
                        if self.backward_stability_scale != 1.0:
                            for parameter in trainable_parameters:
                                if parameter.grad is not None:
                                    parameter.grad.mul_(
                                        self.backward_stability_scale
                                    )
                        gradient_norm = torch.nn.utils.clip_grad_norm_(
                            trainable_parameters, max_norm=0.1
                        )
                        if not bool(torch.isfinite(gradient_norm)):
                            raise FloatingPointError(
                                "Non-finite GRPO gradient norm before optimizer step"
                            )
                        if trace_training:
                            print(
                                f"[GRPO_TRACE] optimizer_step={optimizer_steps + 1} "
                                f"gradient_norm={float(gradient_norm.item()):.6e}",
                                flush=True,
                            )
                        if scaler.is_enabled():
                            scaler.step(optimizer)
                            scaler.update()
                        else:
                            optimizer.step()
                        optimizer.zero_grad(set_to_none=True)
                        optimizer_steps += 1

            if not accumulated_losses or optimizer_steps == 0:
                raise RuntimeError("GRPO training produced no optimizer steps")
            training_loss = float(np.mean(accumulated_losses))

            bad_parameters = []
            for name, parameter in self.model.named_parameters():
                if parameter.requires_grad and not bool(torch.isfinite(parameter).all()):
                    bad_parameters.append(name)
                    if len(bad_parameters) >= 5:
                        break
            if bad_parameters:
                raise FloatingPointError(
                    "Non-finite LoRA parameters detected after optimizer step: "
                    + ", ".join(bad_parameters)
                )
            
            logger.info(f"Training finished successfully; final loss: {training_loss:.4f}")
            
        except Exception as e:
            logger.error(f"Training failed: {e}")
            logger.error(f"Error type: {type(e).__name__}")
            logger.error(f"Dataset size: {len(train_dataset) if train_dataset else 'None'}")
            logger.error(f"Model device: {next(self.model.parameters()).device if hasattr(self.model, 'parameters') else 'Unknown'}")
            
            import traceback
            logger.error(f"Full traceback:\n{traceback.format_exc()}")
            raise RuntimeError("PiT-PO GRPO training failed") from e

        # No checkpoint merge or service restart is needed: inference reuses
        # this exact PEFT object, whose LoRA tensors were just updated.
        self.enter_inference_mode()
        self.training_updates += 1
        self._last_training_completed_at = time.monotonic()
        
        # Compute comprehensive statistics
        avg_reward = np.mean([exp["reward"] for exp in selected_exps])
        min_mse = np.min(mse_values)
        max_mse = np.max(mse_values)
        
        stats = {
            "train_loss": training_loss,
            "avg_reward": avg_reward,
            "avg_mse": avg_mse,
            "min_mse": min_mse,
            "max_mse": max_mse,
            "mse_std": mse_std,
            "buffer_size": len(self.training_buffer),
            "used_samples": len(selected_exps),
            "adaptive_lr": adaptive_lr,
            "adaptive_batch_size": adaptive_batch_size,
            "num_epochs": num_epochs,
            "evotune_magic_percentile": _DPO_STYLE_PERCENTILE,
            "evotune_reward_threshold": reward_thr,
            "evotune_filtered_valid": len(filtered),
            "model_version": self.training_updates,
            "grpo_clip_epsilon": self.grpo_clip_epsilon,
            "grpo_kl_beta": self.grpo_kl_beta,
            "candidate_count": self.candidate_count,
        }
        for name in ("policy_loss", "kl", "clip_fraction"):
            stats[name] = float(np.mean([metrics[name] for metrics in loss_metrics]))
        # Curve logging
        try:
            self._append_curve_record(stats)
        except Exception:
            pass
        # EvoTune: scores_since_finetune is cleared after prepare_dpo_chats; here we clear after a successful training step
        self.scores_since_finetune = []
        logger.info(
            "EvoTune-style: cleared scores_since_finetune after successful train_step "
            f"(next round accumulates from scratch; training_buffer size={len(self.training_buffer)})"
        )
        self._save_periodic_checkpoint_if_needed()
        return stats
    
    def _compute_physical_penalty(self, text: str, mse: float) -> float:
        """Compute gated physical penalty P_phy (Paper Eq. 9).

        Penalties are applied only when MSE < phy_gate_mse_threshold (gating).
        - Dimensional inconsistency -> +phy_penalty_dim
        - Non-differentiability     -> +phy_penalty_diff
        Satisfying a constraint incurs zero penalty for that constraint.
        """
        if not self.enable_physical_penalty or not text:
            return 0.0
        if mse >= self.phy_gate_mse_threshold:
            return 0.0
        try:
            fns = extract_equation_functions_from_text(text)
            if not fns:
                return 0.0
            analysis = analyze_equation_function(fns[0].source, symbol_units=self.phy_units or {})
            penalty = 0.0
            if not analysis.get("dimensionally_consistent", True):
                penalty += self.phy_penalty_dim
            if not analysis.get("differentiable", True):
                penalty += self.phy_penalty_diff
            return float(penalty)
        except Exception:
            return 0.0

    def should_train(self) -> bool:
        """
        Allow online training only after the candidate warmup and with valid data.
        The default skips candidates 1..200 and first permits candidate 201.
        """
        # Require at least 1 valid experience
        valid_count = sum(1 for exp in self.training_buffer if self._is_valid_experience(exp))
        if valid_count < 1:
            logger.debug("should_train=False (no valid experiences)")
            return False
        
        threshold = self.warmup_min_global_samples
        ok = (
            self.candidate_count >= threshold if threshold is not None
            else self.candidate_count > self.warmup_samples
        )
        logger.debug(
            "should_train=%s | valid_count=%s | candidate_count=%s | warmup_samples=%s",
            ok, valid_count, self.candidate_count, self.warmup_samples,
        )
        return ok
    
    def save_merged_model(self, save_dir: str) -> bool:
        """Optionally export merged weights after the complete search.

        The current self.model (with LoRA layers) is NOT modified so training
        could continue afterwards. This method is never called between search
        iterations.

        Returns True on success.
        """
        try:
            from peft import PeftModel
            from transformers import AutoModelForCausalLM
        except ImportError:
            logger.warning("peft not installed – cannot save merged model")
            return False

        if not isinstance(self.model, PeftModel):
            logger.info("Model has no LoRA adapters; saving base model directly")
            os.makedirs(save_dir, exist_ok=True)
            self.model.save_pretrained(save_dir)
            self.tokenizer.save_pretrained(save_dir)
            return True

        adapter_dir = os.path.join(save_dir, "_lora_adapter")
        os.makedirs(adapter_dir, exist_ok=True)
        os.makedirs(save_dir, exist_ok=True)

        try:
            logger.info(f"Saving LoRA adapter to {adapter_dir} ...")
            self.model.save_pretrained(adapter_dir)
            self.tokenizer.save_pretrained(adapter_dir)

            logger.info("Loading fresh base model on CPU for merge ...")
            base = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16,
                device_map={"": "cpu"},
            )
            merged = PeftModel.from_pretrained(base, adapter_dir)
            merged = merged.merge_and_unload()

            logger.info(f"Saving merged model to {save_dir} ...")
            merged.save_pretrained(save_dir)
            self.tokenizer.save_pretrained(save_dir)

            del merged, base
            torch.cuda.empty_cache()
            logger.info("Merged model saved successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to save merged model: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False

    # ================== Curve & metrics logging ==================
    def set_reward_curves_dir(self, path: str):
        """Configure the output directory for training metrics and curves.
        Creates the directory and initializes the CSV header; also prepares a JSONL for flexible later use.
        Degradation strategy: mark logging disabled if the directory is unwritable."""
        try:
            self.reward_curves_dir = path
            os.makedirs(self.reward_curves_dir, exist_ok=True)
        except Exception as e:
            logger.warning(f"Failed to create reward-curves directory {path}: {e}; curve logging will be disabled")
            self.reward_curves_dir = None
            return

        # CSV file path and JSON Lines path
        self.curve_csv_path = os.path.join(self.reward_curves_dir, "training_curve.csv")
        self.curve_jsonl_path = os.path.join(self.reward_curves_dir, "training_curve.jsonl")
        self.curve_png_path = os.path.join(self.reward_curves_dir, "curves.png")
        # Counters
        self._curve_step = 0

        if not os.path.exists(self.curve_csv_path):
            try:
                with open(self.curve_csv_path, "w", encoding="utf-8", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        "step","timestamp","train_loss","avg_reward","avg_mse","min_mse","max_mse","mse_std",
                        "buffer_size","used_samples","adaptive_lr","adaptive_batch_size","num_epochs",
                        "policy_loss","kl","clip_fraction"
                    ])
            except Exception as e:
                logger.warning(f"Failed to initialize curve CSV file: {e}")
        logger.info(f"Reward curves and metrics will be written to: {self.reward_curves_dir}")

    def _append_curve_record(self, stats: dict):
        """Append a training-statistics record and try to update the curve plots.
        Fields in `stats` come from the train_step summary.
        If matplotlib/pandas are unavailable, only CSV/JSONL are written."""
        if not hasattr(self, 'reward_curves_dir') or not self.reward_curves_dir:
            return  # output directory not configured
        self._curve_step = getattr(self, '_curve_step', 0) + 1
        now_ts = int(time.time())

        row = [
            self._curve_step,
            now_ts,
            stats.get('train_loss',''),
            stats.get('avg_reward',''),
            stats.get('avg_mse',''),
            stats.get('min_mse',''),
            stats.get('max_mse',''),
            stats.get('mse_std',''),
            stats.get('buffer_size',''),
            stats.get('used_samples',''),
            stats.get('adaptive_lr',''),
            stats.get('adaptive_batch_size',''),
            stats.get('num_epochs',''),
            stats.get('policy_loss',''),
            stats.get('kl',''),
            stats.get('clip_fraction',''),
        ]
        # Write CSV
        try:
            with open(self.curve_csv_path, 'a', encoding='utf-8', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(row)
        except Exception as e:
            logger.warning(f"Failed to write training-curve CSV: {e}")
        # Write JSONL (easier to post-process later)
        try:
            json_obj = {
                'step': self._curve_step,
                'timestamp': now_ts,
                **stats
            }
            with open(self.curve_jsonl_path, 'a', encoding='utf-8') as f:
                f.write(json.dumps(json_obj, ensure_ascii=False) + '\n')
        except Exception as e:
            logger.warning(f"Failed to write training-curve JSONL: {e}")

        # Attempt to plot
        self._maybe_plot_curves()

    def _maybe_plot_curves(self):
        """Try to plot reward and MSE curves with pandas+matplotlib.
        Skip if dependencies are missing or there is not enough data."""
        try:
            import pandas as pd
            import matplotlib.pyplot as plt
        except Exception:
            return
        # Read the CSV
        try:
            df = pd.read_csv(self.curve_csv_path)
        except Exception:
            return
        if df.empty:
            return
        # Need at least two records to be meaningful
        if len(df) < 2:
            return
        try:
            fig, axes = plt.subplots(1, 2, figsize=(10, 4))
            # Reward curve
            axes[0].plot(df['step'], df['avg_reward'], label='avg_reward', color='tab:blue')
            axes[0].set_xlabel('step')
            axes[0].set_ylabel('avg_reward')
            axes[0].set_title('Reward Curve')
            axes[0].grid(alpha=0.3)
            # MSE curve (log scale)
            axes[1].plot(df['step'], df['avg_mse'], label='avg_mse', color='tab:red')
            axes[1].set_xlabel('step')
            axes[1].set_ylabel('avg_mse (log)')
            try:
                axes[1].set_yscale('log')
            except Exception:
                pass
            axes[1].set_title('MSE Curve')
            axes[1].grid(alpha=0.3)
            fig.tight_layout()
            fig.savefig(self.curve_png_path)
            plt.close(fig)
        except Exception as e:
            logger.debug(f"Plotting curve failed (ignored): {e}")
