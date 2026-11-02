#!/usr/bin/env python3
"""Offline smoke test with a real tiny PaddleFormers Llama and native LoRA.

The byte tokenizer and random tiny model validate execution only; generated
strings do not establish scientific equation discovery quality.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import paddle

from pitpo.grpo_trainer import GRPOTrainer


class TinyCharTokenizer:
    pad_token_id = 0
    bos_token_id = 1
    eos_token_id = 2
    pad_token = "<pad>"
    eos_token = "<eos>"
    padding_side = "right"

    def __call__(self, text, add_special_tokens=True, return_offsets_mapping=False, **kwargs):
        ids = [3 + (ord(char) % 256) for char in text]
        offsets = [(i, i + 1) for i in range(len(text))]
        if add_special_tokens:
            ids = [self.bos_token_id] + ids
            offsets = [(0, 0)] + offsets
        result = {"input_ids": ids, "attention_mask": [1] * len(ids)}
        if return_offsets_mapping:
            result["offset_mapping"] = offsets
        return result

    def decode(self, ids, skip_special_tokens=True):
        return "".join(chr(int(index) - 3) for index in ids if int(index) >= 3)

    def batch_decode(self, batch, skip_special_tokens=True):
        return [self.decode(row, skip_special_tokens) for row in batch]

    def save_pretrained(self, directory):
        Path(directory).mkdir(parents=True, exist_ok=True)
        (Path(directory) / "tiny_tokenizer.json").write_text(json.dumps({"type": "offline_test_byte_tokenizer"}))


def make_tiny_model(seed=3408, device="cpu"):
    from paddleformers.transformers import LlamaConfig, LlamaForCausalLM
    paddle.set_device(device)
    paddle.seed(seed)
    config = LlamaConfig(
        vocab_size=259, hidden_size=32, intermediate_size=64,
        num_hidden_layers=1, num_attention_heads=4, num_key_value_heads=4,
        max_position_embeddings=512, bos_token_id=1, eos_token_id=2,
        pad_token_id=0, use_flash_attention=False, use_fused_rms_norm=False,
        fuse_attention_qkv=False, fuse_attention_ffn=False,
        tensor_model_parallel_size=1, dtype="float32",
    )
    return LlamaForCausalLM(config)


def make_runtime(**kwargs):
    options = dict(model_name="offline-tiny-llama", model=make_tiny_model(device=kwargs.get("device", "cpu")),
                   tokenizer=TinyCharTokenizer(), max_seq_length=512,
                   lora_r=4, lora_alpha=8, warmup_samples=0, required_train_samples=4,
                   per_device_train_batch_size=2, learning_rate=1e-3)
    options.update(kwargs)
    return GRPOTrainer(**options)


def add_test_experiences(runtime):
    prompt = "def equation(x, params):\n"
    for index, mse in enumerate([0.2, 0.5, 1.0, 2.0], 1):
        runtime.add_experience(
            prompt, f"    return params[0] * x + params[1] * {index}\n", mse,
            optimized_params_by_test={"data": [0.001, 1.0]}, global_sample_nums=index,
        )


def run_smoke(output_dir=None):
    runtime = make_runtime()
    add_test_experiences(runtime)
    assert runtime.should_train()
    probe_batch = runtime._batch(runtime.prepare_training_data()[:2])
    with paddle.no_grad():
        logits_before = runtime._forward_logits(probe_batch).numpy().copy()
        with runtime._reference_policy():
            reference_before = runtime._forward_logits(probe_batch).numpy().copy()
    weights_before = {name: parameter.numpy().copy() for name, parameter in runtime.model.named_parameters()}
    generation_before = runtime.generate("def equation(x, params):\n", max_new_tokens=8, temperature=0)
    stats = runtime.train_step(num_epochs=2)
    assert stats["training_updates"] > 0 and np.isfinite(stats["train_loss"])
    changed, frozen = [], []
    for name, parameter in runtime.model.named_parameters():
        equal = np.array_equal(weights_before[name], parameter.numpy())
        if parameter.stop_gradient:
            assert equal, f"Frozen base parameter changed: {name}"
            frozen.append(name)
        elif not equal:
            changed.append(name)
    assert changed, "No LoRA weights changed"
    generation_after = runtime.generate("def equation(x, params):\n", max_new_tokens=8, temperature=0)
    assert len(generation_before) == len(generation_after) == 1
    with paddle.no_grad():
        logits_after = runtime._forward_logits(probe_batch).numpy().copy()
        with runtime._reference_policy():
            reference_after = runtime._forward_logits(probe_batch).numpy().copy()
    policy_logits_max_change = float(np.max(np.abs(logits_after - logits_before)))
    assert policy_logits_max_change > 0
    np.testing.assert_array_equal(reference_before, reference_after)
    if output_dir is None:
        artifact_root = Path(os.environ.get("PITPO_ARTIFACT_ROOT", str(ROOT / "artifacts")))
        artifact_root.mkdir(parents=True, exist_ok=True)
        output_dir = tempfile.mkdtemp(prefix="offline-smoke-", dir=artifact_root)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = output / "adapter"
    runtime.save_adapter_checkpoint(str(checkpoint), reason="smoke")
    restored = make_runtime(adapter_path=str(checkpoint))
    for name, parameter in runtime.model.named_parameters():
        np.testing.assert_array_equal(parameter.numpy(), dict(restored.model.named_parameters())[name].numpy())
    assert restored.training_updates == runtime.training_updates
    # JSON stores mapping keys as strings, including diagnostic coefficient IDs.
    assert restored.training_buffer == json.loads(json.dumps(runtime.training_buffer, default=runtime._json_default))
    reloaded_generation = restored.generate("def equation(x, params):\n", max_new_tokens=8, temperature=0)
    assert reloaded_generation == generation_after
    resumed_stats = restored.train_step(num_epochs=1)
    assert resumed_stats["training_updates"] > runtime.training_updates
    merged_dir = output / "merged"
    assert runtime.save_merged_model(str(merged_dir))
    merged_runtime = GRPOTrainer(model_name=str(merged_dir), tokenizer=TinyCharTokenizer(),
                                 use_lora=False, max_seq_length=512)
    with paddle.no_grad():
        merged_logits = merged_runtime._forward_logits(probe_batch).numpy()
    np.testing.assert_allclose(merged_logits, logits_after, rtol=1e-5, atol=1e-6)
    assert merged_runtime.generate("def equation(x, params):\n", max_new_tokens=8, temperature=0) == generation_after
    result = {"paddle_version": paddle.__version__, "device": paddle.get_device(),
              "model": "offline random tiny PaddleFormers Llama", "stats": stats,
              "changed_lora_tensors": len(changed), "unchanged_frozen_tensors": len(frozen),
              "generation_before_repr": repr(generation_before), "generation_after_repr": repr(generation_after),
              "policy_logits_max_change": policy_logits_max_change,
              "frozen_reference_logits_equal": True, "merged_native_reload_equal": True,
              "optimizer_resume_updates": resumed_stats["training_updates"],
              "checkpoint_reload_equal": True, "checkpoint": str(checkpoint)}
    (output / "result.json").write_text(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir")
    args = parser.parse_args()
    print(json.dumps(run_smoke(args.output_dir), indent=2))
