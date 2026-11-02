"""Paddle-native shared generation and online PiT-PO GRPO runtime.

PaddleFormers owns the causal LM and LoRA layers. Policy updates, fixed old
policy probabilities, frozen base-reference probabilities and token penalties
are evaluated by Paddle autograd. Generation uses those same live parameters.
"""
from __future__ import annotations

import ast
import csv
import json
import logging
import math
import os
import re
import shutil
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import paddle
import paddle.nn.functional as F

from .equation_functions import extract_equation_functions_from_text, analyze_equation_function
from .coef_penalty import compute_coefficient_penalty
from .model_loading import prepare_model_checkpoint

logger = logging.getLogger(__name__)
_DPO_STYLE_PERCENTILE = 70


class GRPOTrainer:
    """One Paddle causal LM shared by search generation and LoRA training."""

    def __init__(
        self, model_name: str, learning_rate: float = 1e-5,
        batch_size: int = 4, max_length: Optional[int] = None,
        reward_scaling: float = 10.0, min_mse_threshold: float = 1e-15,
        buffer_size: int = 1000, device: str = "cpu",
        required_train_samples: int = 16, use_lora: bool = True,
        lora_r: int = 16, lora_alpha: int = 32, lora_dropout: float = 0.0,
        lora_target_modules: Optional[List[str]] = None,
        per_device_train_batch_size: int = 2, gradient_accumulation_steps: int = 1,
        warmup_generations: Optional[int] = None, samples_per_prompt: Optional[int] = None,
        warmup_min_global_samples: Optional[int] = None,
        ast_complexity_weight: float = 0.05, enable_physical_penalty: bool = True,
        phy_penalty_dim: float = 2.0, phy_penalty_diff: float = 1.5,
        phy_gate_mse_threshold: float = 1.0, phy_units: Optional[Dict[str, str]] = None,
        enable_coef_penalty: bool = True, coef_ratio_threshold: float = 0.05,
        coef_penalty_weight: float = 1.0, max_seq_length: int = 4096,
        checkpoint_dir: Optional[str] = None, checkpoint_every: int = 100,
        grpo_clip_epsilon: float = 0.2, grpo_kl_beta: float = 0.01,
        warmup_samples: int = 200, seed: int = 3408, dtype: str = "float32",
        adapter_path: Optional[str] = None, model=None, tokenizer=None,
        prompt_format: str = "auto",
    ):
        from paddleformers.peft import LoRAConfig, LoRAModel
        from paddleformers.transformers import AutoModelForCausalLM, AutoTokenizer

        if device == "cuda":
            device = "gpu:0"
        paddle.set_device(device)
        paddle.seed(seed)
        np.random.seed(seed)
        if dtype not in ("float32", "bfloat16"):
            raise ValueError("dtype must be float32 or bfloat16")
        if prompt_format not in ("auto", "chat", "raw"):
            raise ValueError("prompt_format must be auto, chat, or raw")
        if device == "cpu" and dtype != "float32":
            raise ValueError("CPU execution requires float32")
        if not 0 < grpo_clip_epsilon < 1 or grpo_kl_beta < 0:
            raise ValueError("Invalid GRPO clipping range or KL weight")
        if buffer_size < 2 or required_train_samples < 2:
            raise ValueError("GRPO requires buffer_size and required_train_samples >= 2")
        if per_device_train_batch_size < 1 or gradient_accumulation_steps < 1:
            raise ValueError("Batch size and gradient accumulation must be positive")
        self.model_name, self.device, self.dtype = model_name, device, dtype
        self.learning_rate, self.batch_size = learning_rate, batch_size
        self.max_seq_length = int(max_seq_length)
        self.max_length = int(max_length or max_seq_length)
        if not 2 <= self.max_length <= self.max_seq_length:
            raise ValueError("max_length must be between 2 and max_seq_length")
        self.reward_scaling, self.min_mse_threshold = reward_scaling, min_mse_threshold
        self.buffer_size, self.required_train_samples = buffer_size, required_train_samples
        self.use_lora, self.lora_r, self.lora_alpha = use_lora, lora_r, lora_alpha
        self.lora_dropout, self.lora_target_modules = lora_dropout, lora_target_modules
        self.per_device_train_batch_size = per_device_train_batch_size
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.samples_per_prompt = samples_per_prompt
        self.warmup_samples = int(warmup_samples if warmup_generations is None else warmup_generations * (samples_per_prompt or 1))
        self.warmup_min_global_samples = warmup_min_global_samples
        self.candidate_count = 0
        self.ast_complexity_weight = ast_complexity_weight
        self.enable_physical_penalty = enable_physical_penalty
        self.phy_penalty_dim, self.phy_penalty_diff = phy_penalty_dim, phy_penalty_diff
        self.phy_gate_mse_threshold, self.phy_units = phy_gate_mse_threshold, phy_units
        self.enable_coef_penalty = enable_coef_penalty
        self.coef_ratio_threshold, self.coef_penalty_weight = coef_ratio_threshold, coef_penalty_weight
        self.grpo_clip_epsilon, self.grpo_kl_beta = grpo_clip_epsilon, grpo_kl_beta
        self.checkpoint_dir, self.checkpoint_every = checkpoint_dir, int(checkpoint_every)
        self.seed, self.training_updates = seed, 0
        self.training_buffer, self.scores_since_finetune = [], []
        self._last_adapter_checkpoint_dir = None
        self._last_training_completed_at = None
        self._last_train_to_generation_seconds = None
        self._runtime_mode = "initializing"
        self._optimizer = None
        self._last_loss_metrics = {}
        self.tokenizer = tokenizer if tokenizer is not None else AutoTokenizer.from_pretrained(model_name)
        self.prompt_format = prompt_format
        has_chat_template = bool(getattr(self.tokenizer, "chat_template", None))
        if prompt_format == "chat" and not has_chat_template:
            raise ValueError("The selected tokenizer does not provide a chat template")
        self._chat_template_enabled = prompt_format == "chat" or (prompt_format == "auto" and has_chat_template)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"
        resolved_model_name = prepare_model_checkpoint(model_name) if model is None else model_name
        local_model = Path(resolved_model_name)
        native_weights = local_model.is_dir() and (
            any(local_model.glob("*.pdparams")) or (local_model / "model_state.pdparams.index.json").exists()
            or (local_model / "pitpo_hf_conversion.json").exists()
        )
        base_model = model if model is not None else AutoModelForCausalLM.from_pretrained(
            resolved_model_name, dtype=dtype, convert_from_hf=not native_weights,
            max_position_embeddings=self.max_seq_length,
        )
        self.loaded_model_path = str(local_model)
        self.model_weight_verification = (
            self._verify_converted_checkpoint(base_model, local_model)
            if model is None and (local_model / "pitpo_hf_conversion.json").exists() else None
        )
        if self.use_lora:
            if adapter_path:
                self.model = LoRAModel.from_pretrained(base_model, adapter_path)
            else:
                names = lora_target_modules or ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
                targets = [rf".*\.{re.escape(name)}$" for name in names]
                self.model = LoRAModel(base_model, LoRAConfig(
                    r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
                    target_modules=targets, dtype=dtype, merge_weights=False,
                    tensor_model_parallel_size=1, use_quick_lora=False,
                ))
            self.model.mark_only_lora_as_trainable()
            if not any(not p.stop_gradient for p in self.model.parameters()):
                raise RuntimeError("No trainable LoRA parameters matched the selected model")
        else:
            self.model = base_model
            for parameter in self.model.parameters():
                parameter.stop_gradient = True
        if adapter_path:
            state_path = Path(adapter_path) / "pitpo_training_state.json"
            if state_path.exists():
                state = json.loads(state_path.read_text())
                self.training_updates = int(state.get("training_updates", 0))
                self.candidate_count = int(state.get("candidate_count", 0))
                self.training_buffer = state.get("training_buffer", [])
                self.scores_since_finetune = state.get("scores_since_finetune", [])
                self._pending_optimizer_state = Path(adapter_path) / "optimizer.pdopt"
                self._checkpoint_parameter_names = state.get("optimizer_parameter_names", {})
        self.enter_inference_mode()
        logger.info("Paddle shared model ready: %s, device=%s, LoRA=%s", model_name, device, use_lora)

    @staticmethod
    def _verify_converted_checkpoint(model, directory):
        """Fail before search if the native model did not receive its converted weights."""
        from paddleformers.utils.safetensors import fast_safe_open

        receipt = json.loads((directory / "pitpo_hf_conversion.json").read_text())
        state = model.state_dict()
        shapes = {item["target"]: item["target_shape"] for item in receipt["tensors"]}
        missing = set(state) - set(shapes)
        tied = bool(getattr(model.config, "tie_word_embeddings", False))
        if tied:
            missing.discard("lm_head.weight")
            if model.get_input_embeddings().weight is not model.get_output_embeddings().weight:
                raise RuntimeError("Converted Llama output head is not tied to its loaded embedding")
        unexpected = set(shapes) - set(state)
        if missing or unexpected:
            raise RuntimeError(f"Converted checkpoint coverage mismatch: missing={sorted(missing)}, unexpected={sorted(unexpected)}")
        for name, shape in shapes.items():
            if list(state[name].shape) != shape:
                raise RuntimeError(f"Converted parameter shape mismatch: {name}")
        index = json.loads((directory / "model.safetensors.index.json").read_text())["weight_map"]
        samples = ["llama.embed_tokens.weight", "llama.layers.0.self_attn.k_proj.weight",
                   "llama.layers.0.mlp.down_proj.weight"]
        for name in samples:
            with fast_safe_open(str(directory / index[name]), framework="np") as checkpoint:
                expected = checkpoint.get_slice(name)[:2, :8].astype("float32")
            expected = paddle.to_tensor(expected, dtype=state[name].dtype).astype("float32").cpu().numpy()
            actual = state[name][:2, :8].astype("float32").cpu().numpy()
            if not np.array_equal(actual, expected):
                raise RuntimeError(f"Pretrained parameter was not correctly loaded: {name}")
        logger.info("Verified %d converted pretrained parameter shapes and %d loaded parameter slices; tied head=%s",
                    len(shapes), len(samples), tied)
        return {"source": receipt["source_signature"]["source"], "native_cache": str(directory),
                "mapped_tensors": len(shapes), "all_parameter_shapes_match": True,
                "sampled_parameters_match": samples, "tied_output_head": tied}

    def enter_training_mode(self):
        # Disable dropout for a deterministic ratio against the fixed old policy;
        # eval mode still builds a full Paddle autograd graph.
        self.model.eval()
        self.model.config.use_cache = False
        self._runtime_mode = "training"

    def enter_inference_mode(self):
        self.model.eval()
        self.model.config.use_cache = True
        self._runtime_mode = "inference"

    @contextmanager
    def _reference_policy(self):
        if not self.use_lora:
            raise RuntimeError("Reference-policy evaluation requires frozen base weights and LoRA")
        self.model.disable_lora()
        try:
            yield
        finally:
            self.model.enable_lora()

    def _format_generation_prompt(self, prompt: str) -> str:
        """Adapt code-completion search prompts to an Instruct model's chat format."""
        if not self._chat_template_enabled:
            return prompt
        signature = "the last unfinished equation function"
        variables = []
        try:
            functions = [node for node in ast.parse(prompt).body if isinstance(node, ast.FunctionDef)]
            if functions:
                function = functions[-1]
                signature = f"{function.name}({ast.unparse(function.args)})"
                variables = [arg.arg for arg in function.args.posonlyargs + function.args.args + function.args.kwonlyargs]
        except SyntaxError:
            matches = re.findall(r"(?m)^def\s+([^\n]+):", prompt)
            if matches:
                signature = matches[-1]
        bounds = re.findall(r"\bMAX_NPARAMS\s*=\s*(\d+)\b", prompt)
        if bounds and int(bounds[-1]) > 0:
            parameter_rule = f"Use only coefficient indices params[0] through params[{int(bounds[-1]) - 1}]."
        else:
            used = [int(index) for index in re.findall(r"\bparams\s*\[\s*(\d+)\s*\]", prompt)]
            parameter_rule = (f"Keep coefficient indices within the demonstrated range params[0] through params[{max(used)}]."
                              if used else "Keep the coefficient array and its demonstrated indexing convention unchanged.")
        variable_rule = (f"The only input variables are {', '.join(variables)}; np is available. "
                         "Any other local name must be assigned before use.") if variables else "Use exactly the input variables in that function signature."
        request = (
            "Propose one mathematical equation candidate using this search context:\n\n"
            + prompt
            + f"\n\nComplete only the body of {signature}.\n"
            + variable_rule + "\n" + parameter_rule + "\n"
            + "Write the mathematical expression directly using those inputs, params and NumPy. "
              "Never call equation, equation_v0, or any earlier equation_vN function; those names are unavailable in the candidate.\n"
            + "Output indented Python body statements ending with one return statement. "
              "Do not output a def line, another function, imports, markdown fences, explanations, or example usage. "
              "Stop immediately after the return statement."
        )
        messages = [
            {"role": "system", "content": "You write NumPy mathematical equation candidates for symbolic regression. Follow the requested Python interface exactly."},
            {"role": "user", "content": request},
        ]
        formatted = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False,
        )
        if not isinstance(formatted, str):
            raise TypeError("Tokenizer chat template must produce a string")
        return formatted

    def generate(self, prompt: str, num_return_sequences: int = 1,
                 max_new_tokens: int = 1024, temperature: float = 0.6,
                 top_p: float = 0.95, top_k: int = 50,
                 repetition_penalty: float = 1.1) -> List[str]:
        if max_new_tokens < 1 or max_new_tokens >= self.max_seq_length:
            raise ValueError("max_new_tokens must be positive and smaller than max_seq_length")
        self.enter_inference_mode()
        formatted_prompt = self._format_generation_prompt(prompt)
        ids = self.tokenizer(formatted_prompt, add_special_tokens=not self._chat_template_enabled)["input_ids"]
        prompt_budget = self.max_seq_length - max_new_tokens
        was_truncated = len(ids) > prompt_budget
        ids = ids[-prompt_budget:]
        self.last_generation_prompt = self.tokenizer.decode(ids, skip_special_tokens=False) if was_truncated else formatted_prompt
        self.last_generation_prompt_ids = list(ids)
        input_ids = paddle.to_tensor([ids], dtype="int64")
        with paddle.no_grad():
            generated = self.model.generate(
                input_ids=input_ids, attention_mask=paddle.ones_like(input_ids),
                max_new_tokens=max_new_tokens, decode_strategy="sampling" if temperature > 0 else "greedy_search",
                do_sample=temperature > 0,
                temperature=max(temperature, 1e-6), top_p=top_p, top_k=top_k,
                repetition_penalty=repetition_penalty, num_return_sequences=num_return_sequences,
                pad_token_id=self.tokenizer.pad_token_id, eos_token_id=self.tokenizer.eos_token_id,
                use_cache=True, trunc_input=True,
            )
        tokens = generated[0] if isinstance(generated, tuple) else generated
        result = self.tokenizer.batch_decode(tokens.tolist(), skip_special_tokens=True)
        if self._last_training_completed_at is not None:
            self._last_train_to_generation_seconds = time.perf_counter() - self._last_training_completed_at
            self._last_training_completed_at = None
        return result

    def consume_train_to_generation_seconds(self):
        value = self._last_train_to_generation_seconds
        self._last_train_to_generation_seconds = None
        return value

    @staticmethod
    def _json_default(value):
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, paddle.Tensor):
            return value.tolist()
        raise TypeError(f"Cannot serialize {type(value).__name__}")

    def save_adapter_checkpoint(self, save_dir: str, reason: str = "manual") -> str:
        if not self.use_lora:
            return save_dir
        destination = Path(save_dir).absolute()
        destination.parent.mkdir(parents=True, exist_ok=True)
        staging = Path(tempfile.mkdtemp(prefix=f".{destination.name}.pending-", dir=destination.parent))
        try:
            self.model.save_pretrained(str(staging))
            self.tokenizer.save_pretrained(str(staging))
            if self._optimizer is not None:
                paddle.save(self._optimizer.state_dict(), str(staging / "optimizer.pdopt"))
            state = {"reason": reason, "model_name": self.model_name, "seed": self.seed,
                     "training_updates": self.training_updates, "candidate_count": self.candidate_count,
                     "training_buffer": self.training_buffer, "scores_since_finetune": self.scores_since_finetune,
                     "optimizer_parameter_names": {name: parameter.name for name, parameter in self.model.named_parameters()
                                                   if not parameter.stop_gradient},
                     "warmup_samples": self.warmup_samples, "grpo_clip_epsilon": self.grpo_clip_epsilon,
                     "grpo_kl_beta": self.grpo_kl_beta}
            (staging / "pitpo_training_state.json").write_text(json.dumps(state, indent=2, default=self._json_default))
            if destination.exists():
                raise FileExistsError(f"Checkpoint destination already exists: {destination}")
            os.replace(staging, destination)
        finally:
            if staging.exists():
                shutil.rmtree(staging)
        self._last_adapter_checkpoint_dir = str(destination)
        return str(destination)

    def save_merged_model(self, save_dir: str) -> bool:
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        if not self.use_lora:
            self.model.save_pretrained(save_dir)
        else:
            self.model.merge()
            try:
                # Retain the base model's canonical names; remove adapter tensors.
                merged = {name: value for name, value in self.model.model.state_dict().items()
                          if "lora_" not in name}
                self.model.model.save_pretrained(save_dir, state_dict=merged)
            finally:
                self.model.unmerge()
        self.tokenizer.save_pretrained(save_dir)
        return True

    def should_train(self) -> bool:
        group_sizes = {}
        for exp in self.training_buffer:
            if self._is_valid_experience(exp):
                group_sizes[exp["prompt"]] = group_sizes.get(exp["prompt"], 0) + 1
        valid = sum(size for size in group_sizes.values() if size >= 2)
        gate = (self.candidate_count >= self.warmup_min_global_samples
                if self.warmup_min_global_samples is not None else self.candidate_count > self.warmup_samples)
        return self.use_lora and valid >= self.required_train_samples and gate

    def prepare_training_data(self, experiences=None):
        exps = self.training_buffer if experiences is None else experiences
        rows = []
        for exp in exps:
            prompt, response = exp["prompt"], exp["response"]
            text = prompt + response
            try:
                encoded = self.tokenizer(text, add_special_tokens=not self._chat_template_enabled, return_offsets_mapping=True)
                offsets = encoded.get("offset_mapping")
            except (TypeError, NotImplementedError, ValueError):
                encoded = self.tokenizer(text, add_special_tokens=not self._chat_template_enabled)
                offsets = None
            ids = list(encoded["input_ids"])
            if offsets is not None:
                completion = [int(end > len(prompt)) for start, end in offsets]
            else:
                prompt_ids = self.tokenizer(prompt, add_special_tokens=not self._chat_template_enabled)["input_ids"]
                prefix = 0
                for left, right in zip(prompt_ids, ids):
                    if left != right:
                        break
                    prefix += 1
                completion = [int(i >= prefix) for i in range(len(ids))]
            penalties = [0.0] * len(ids)
            params = (exp.get("metadata") or {}).get("optimized_params_by_test")
            if self.enable_coef_penalty and params:
                _, details = compute_coefficient_penalty(response, params, ratio_threshold=self.coef_ratio_threshold,
                                                          penalty_weight=self.coef_penalty_weight)
                spans = []
                for index, weight in details.get("penalty_by_index", {}).items():
                    for match in re.finditer(rf"params\s*\[\s*{index}\s*\]", response):
                        spans.append((len(prompt) + match.start(), len(prompt) + match.end(), weight))
                if offsets is not None:
                    penalties = [sum(w for a, b, w in spans if end > a and start < b and end > start)
                                 for start, end in offsets]
                elif spans:
                    raise RuntimeError("Token redundancy regularization requires a tokenizer with offset mappings")
            # Retain the response when a growing evolutionary prompt exceeds the
            # context window. The left-truncated sequence still uses causal loss.
            ids, completion, penalties = ids[-self.max_length:], completion[-self.max_length:], penalties[-self.max_length:]
            if not any(completion[1:]):
                raise ValueError("Training example has no completion tokens after causal shifting")
            rows.append({"input_ids": ids, "completion_mask": completion,
                         "token_penalty_weights": penalties, "rewards": float(exp["reward"]),
                         "prompt": prompt})
        if not rows:
            raise ValueError("No training experiences")
        width = max(len(row["input_ids"]) for row in rows)
        for row in rows:
            n = len(row["input_ids"])
            row["attention_mask"] = [1] * n + [0] * (width-n)
            row["input_ids"] += [self.tokenizer.pad_token_id] * (width-n)
            row["completion_mask"] += [0] * (width-n)
            row["token_penalty_weights"] += [0.0] * (width-n)
        # Normalize before microbatching, within each actual search prompt group.
        # A singleton or tied group has zero sequence advantage; token penalties
        # still supply the PiT-specific signal where fitted terms are redundant.
        groups = {}
        for i, row in enumerate(rows):
            groups.setdefault(row["prompt"], []).append(i)
        for indices in groups.values():
            values = np.array([rows[i]["rewards"] for i in indices])
            sigma = max(float(values.std(ddof=1)) if len(indices) > 1 else 1.0, 1e-8)
            for i in indices:
                rows[i]["advantages"] = float((rows[i]["rewards"] - values.mean()) / sigma)
        return rows

    @staticmethod
    def _token_log_probs(logits, labels, attention_mask):
        shifted_labels = labels[:, 1:]
        valid = (attention_mask[:, 1:] != 0) & (shifted_labels != -100)
        indices = paddle.nonzero(valid)
        if indices.shape[0] == 0:
            raise FloatingPointError("GRPO batch has no completion tokens")
        valid_logits = paddle.gather_nd(logits[:, :-1, :], indices).astype("float32")
        targets = paddle.gather_nd(shifted_labels, indices).unsqueeze(-1)
        valid_log_probs = -F.cross_entropy(valid_logits, targets, reduction="none").squeeze(-1)
        if not bool(paddle.isfinite(valid_log_probs).all()):
            raise FloatingPointError("Non-finite completion token log probabilities")
        log_probs = paddle.scatter_nd(indices, valid_log_probs, valid.shape)
        return log_probs, valid

    def _batch(self, rows):
        batch = {name: paddle.to_tensor([row[name] for row in rows], dtype="int64")
                 for name in ("input_ids", "attention_mask", "completion_mask")}
        batch["labels"] = paddle.where(batch["completion_mask"].astype("bool"), batch["input_ids"],
                                       paddle.full_like(batch["input_ids"], -100))
        for name in ("rewards", "advantages", "token_penalty_weights", "old_log_probs", "ref_log_probs"):
            if name in rows[0]:
                batch[name] = paddle.to_tensor([row[name] for row in rows], dtype="float32")
        return batch

    def _forward_logits(self, batch):
        # All positions attend causally. Right-pad query activations remain finite;
        # prompt/padding exclusion is enforced separately in the objective mask.
        output = self.model(input_ids=batch["input_ids"], attention_mask=paddle.ones_like(batch["attention_mask"]),
                            use_cache=False, return_dict=True)
        return output.logits if hasattr(output, "logits") else output[0]

    def _cache_policy_log_probs(self, rows):
        self.model.eval()
        with paddle.no_grad():
            for start in range(0, len(rows), self.per_device_train_batch_size):
                subset = rows[start:start+self.per_device_train_batch_size]
                batch = self._batch(subset)
                old, _ = self._token_log_probs(self._forward_logits(batch), batch["labels"], batch["attention_mask"])
                with self._reference_policy():
                    ref, _ = self._token_log_probs(self._forward_logits(batch), batch["labels"], batch["attention_mask"])
                for i, row in enumerate(subset):
                    row["old_log_probs"], row["ref_log_probs"] = old[i].tolist(), ref[i].tolist()
        return rows

    def compute_grpo_loss(self, logits, labels, rewards, attention_mask,
                          token_penalty_weights=None, old_log_probs=None, ref_log_probs=None,
                          advantages=None):
        log_probs, valid = self._token_log_probs(logits, labels, attention_mask)
        if not bool(paddle.isfinite(rewards).all()):
            raise FloatingPointError("GRPO rewards contain NaN or infinity")
        if advantages is None:
            sigma = paddle.clip(paddle.std(rewards), min=1e-8) if rewards.numel() > 1 else paddle.ones([])
            advantages = (rewards - rewards.mean()) / sigma
        per_token_advantage = advantages.unsqueeze(-1).expand(log_probs.shape)
        if token_penalty_weights is not None:
            per_token_advantage = per_token_advantage - token_penalty_weights[:, 1:]
        indices = paddle.nonzero(valid)
        current = paddle.gather_nd(log_probs, indices)
        adv = paddle.gather_nd(per_token_advantage, indices)
        if not bool(paddle.isfinite(adv).all()):
            raise FloatingPointError("GRPO advantages or token penalties are non-finite")
        if old_log_probs is None or old_log_probs.shape != log_probs.shape:
            raise ValueError("Fixed old_log_probs must have shape [batch, sequence_length - 1]")
        old = paddle.gather_nd(old_log_probs.detach(), indices)
        ratio = paddle.exp(current - old)
        unclipped = ratio * adv
        clipped = paddle.clip(ratio, min=1-self.grpo_clip_epsilon, max=1+self.grpo_clip_epsilon) * adv
        surrogate = paddle.minimum(unclipped, clipped)
        kl = paddle.zeros_like(current)
        if self.grpo_kl_beta > 0:
            if ref_log_probs is None or ref_log_probs.shape != log_probs.shape:
                raise ValueError("Fixed ref_log_probs required when KL beta > 0")
            delta = paddle.gather_nd(ref_log_probs.detach(), indices) - current
            kl = paddle.clip(paddle.expm1(delta) - delta, min=0)
        def sequence_mean(values):
            dense = paddle.scatter_nd(indices, values, valid.shape)
            return (dense.sum(axis=-1) / paddle.clip(valid.astype("float32").sum(axis=-1), min=1)).mean()
        loss = sequence_mean(-surrogate + self.grpo_kl_beta * kl)
        if not bool(paddle.isfinite(loss)):
            raise FloatingPointError("GRPO loss is non-finite")
        self._last_loss_metrics = {"policy_loss": float(sequence_mean(-surrogate)), "kl": float(sequence_mean(kl)),
                                   "clip_fraction": float((unclipped > clipped).astype("float32").mean())}
        return loss

    def train_step(self, num_epochs=1):
        if not self.use_lora:
            raise RuntimeError("GRPO is disabled")
        if num_epochs < 1:
            raise ValueError("num_epochs must be positive")
        valid = [exp for exp in self.training_buffer if self._is_valid_experience(exp)]
        if len(valid) < 2:
            return {"skipped": True, "reason": "insufficient_samples", "training_updates": self.training_updates}
        groups = {}
        for exp in valid:
            groups.setdefault(exp["prompt"], []).append(exp)
        threshold = self._evotune_reward_threshold()
        selected = []
        for group in reversed(list(groups.values())):
            if len(group) < 2:
                continue
            remaining = self.required_train_samples - len(selected)
            if remaining < 2:
                break
            # Retain comparative outcomes for a selected prompt. Filtering
            # individual responses can leave only tied winners or singletons.
            if max(exp["reward"] for exp in group) < threshold:
                continue
            selected = group[-remaining:] + selected
        if len(selected) < 2:
            return {"skipped": True, "reason": "no_comparable_groups", "training_updates": self.training_updates}
        rows = self.prepare_training_data(selected)
        if not any(abs(row["advantages"]) > 1e-8 or any(row["token_penalty_weights"]) for row in rows):
            logger.info("Skipping tied-reward groups with no token penalty: no policy learning signal")
            return {"skipped": True, "reason": "no_policy_signal", "training_updates": self.training_updates}
        self.enter_training_mode()
        rows = self._cache_policy_log_probs(rows)
        parameters = [parameter for parameter in self.model.parameters() if not parameter.stop_gradient]
        avg_mse = float(np.mean([exp["mse"] for exp in selected]))
        rate = self.learning_rate * (1.5 if avg_mse < 0.1 else 0.5 if avg_mse > 10 else 1.0)
        if self._optimizer is None:
            self._optimizer = paddle.optimizer.AdamW(learning_rate=rate, parameters=parameters,
                                                     grad_clip=paddle.nn.ClipGradByGlobalNorm(1.0),
                                                     multi_precision=self.dtype == "bfloat16")
            pending = getattr(self, "_pending_optimizer_state", None)
            if pending is not None and pending.exists():
                state = paddle.load(str(pending))
                old_names = getattr(self, "_checkpoint_parameter_names", {})
                current_names = {name: parameter.name for name, parameter in self.model.named_parameters()
                                 if not parameter.stop_gradient}
                replacements = sorted(((old, current_names[name]) for name, old in old_names.items()
                                       if name in current_names), key=lambda item: -len(item[0]))
                def remap(value):
                    if not isinstance(value, dict):
                        return value
                    result = {}
                    for key, item in value.items():
                        renamed = key
                        for old, new in replacements:
                            if key == old or key.startswith(old + "_"):
                                renamed = new + key[len(old):]
                                break
                        result[renamed] = remap(item)
                    return result
                self._optimizer.set_state_dict(remap(state))
                self._pending_optimizer_state = None
        self._optimizer.set_lr(rate)
        losses = []
        metric_values = []
        micro = self.per_device_train_batch_size
        generator = np.random.default_rng(self.seed + self.training_updates)
        try:
            for epoch in range(int(num_epochs)):
                order = generator.permutation(len(rows)).tolist()
                batches = [order[start:start+micro] for start in range(0, len(order), micro)]
                for block_start in range(0, len(batches), self.gradient_accumulation_steps):
                    block = batches[block_start:block_start+self.gradient_accumulation_steps]
                    self._optimizer.clear_grad()
                    total = sum(len(indices) for indices in block)
                    for indices in block:
                        batch = self._batch([rows[i] for i in indices])
                        loss = self.compute_grpo_loss(
                            self._forward_logits(batch), batch["labels"], batch["rewards"], batch["attention_mask"],
                            batch["token_penalty_weights"], batch["old_log_probs"], batch["ref_log_probs"], batch["advantages"])
                        (loss * (len(indices) / total)).backward()
                        losses.append(float(loss))
                        metric_values.append(dict(self._last_loss_metrics))
                    for parameter in parameters:
                        if parameter.grad is not None and not bool(paddle.isfinite(parameter.grad).all()):
                            raise FloatingPointError("Non-finite LoRA gradient before optimizer step")
                    self._optimizer.step()
                    self.training_updates += 1
            for parameter in parameters:
                if not bool(paddle.isfinite(parameter).all()):
                    raise FloatingPointError("Non-finite LoRA parameter after update")
        finally:
            self._optimizer.clear_grad()
            self.enter_inference_mode()
        self._last_training_completed_at = time.perf_counter()
        self.scores_since_finetune.clear()
        mse = [exp["mse"] for exp in selected]
        stats = {"train_loss": float(np.mean(losses)), "avg_reward": float(np.mean([exp["reward"] for exp in selected])),
                 "avg_mse": avg_mse, "min_mse": min(mse), "max_mse": max(mse), "mse_std": float(np.std(mse)),
                 "used_samples": len(selected), "buffer_size": len(self.training_buffer), "num_epochs": num_epochs,
                 "adaptive_lr": rate, "adaptive_batch_size": micro, "training_updates": self.training_updates,
                 **{key: float(np.mean([item[key] for item in metric_values])) for key in metric_values[0]}}
        self._append_curve_record(stats)
        if self.checkpoint_dir and self.checkpoint_every and self.training_updates % self.checkpoint_every == 0:
            self.save_adapter_checkpoint(os.path.join(self.checkpoint_dir, f"checkpoint-{self.training_updates:06d}"), "periodic")
        return stats

    def _is_valid_experience(self, exp: Dict[str, Any]) -> bool:
        """Check if an experience has a valid (non-None) score/MSE.
        Additional rule: if metadata contains a score field that is positive (usually meaning MSE was negated), treat the entry as invalid.
        """
        try:
            mse = exp.get("mse", None)
            if mse is None:
                return False
            # A perfect fit is valid; mse_to_reward clamps its logarithm.
            if not (isinstance(mse, (int, float)) and np.isfinite(mse) and mse >= 0):
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

    @staticmethod
    def _analysis_source(prompt: str, response: str, metadata: dict) -> str:
        """Recover a complete function when the sampler returns its body only."""
        if metadata.get("equation_source"):
            return metadata["equation_source"]
        combined = response if re.search(r"(?m)^def equation\s*\(", response) else prompt + response
        combined = re.sub(r"(?m)^(def\s+)equation_v\d+(\s*\()", r"\1equation\2", combined)
        try:
            tree = ast.parse(combined)
            functions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "equation"]
            if functions:
                return ast.get_source_segment(combined, functions[-1]) or response
        except SyntaxError:
            pass
        return response

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
        if mse is None or not np.isfinite(mse) or mse < 0:
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

        analysis_source = self._analysis_source(prompt, response, metadata)

        # === R_fit: Fitting accuracy reward (Paper Eq. 7) ===
        r_fit = self.mse_to_reward(mse)

        # === P_cplx: AST complexity penalty (Paper Eq. 8) ===
        p_cplx = 0.0
        ast_count = 0
        try:
            fns = extract_equation_functions_from_text(analysis_source)
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
            p_phy = self._compute_physical_penalty(analysis_source, mse)

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
                    # Preserve the upstream implementation's sequence-level
                    # penalty together with its token-local regularization.
                    reward -= coef_penalty_val
                    logger.debug("Applied coefficient penalty: %.4f; reward now %.2f", coef_penalty_val, reward)

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
