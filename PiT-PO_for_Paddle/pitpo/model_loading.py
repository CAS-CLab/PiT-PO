"""Strict local HF Llama checkpoint conversion for PaddleFormers 0.4.1.

The upstream safetensors loader transposes linear weights but does not apply
Llama's HF ``model.`` to Paddle ``llama.`` name mapping. Convert both names and
layouts with PaddleFormers' official StateDictNameMapping before loading.
"""
from __future__ import annotations

import fcntl
import hashlib
import json
import logging
import os
from pathlib import Path
import shutil
import tempfile

import numpy as np

logger = logging.getLogger(__name__)
_RECEIPT = "pitpo_hf_conversion.json"
_CONVERSION_VERSION = 1


def _source_checkpoint(root: Path):
    index = root / "model.safetensors.index.json"
    if index.exists():
        data = json.loads(index.read_text())
        return data["weight_map"]
    single = root / "model.safetensors"
    if single.exists():
        from paddleformers.utils.safetensors import fast_safe_open
        with fast_safe_open(str(single), framework="np") as checkpoint:
            return {key: single.name for key in checkpoint.keys()}
    return None


def prepare_model_checkpoint(model_name: str) -> str:
    """Return a verified native cache for a local HF Llama, otherwise input.

    The source directory is read only. The completed cache is published
    atomically, and a file lock prevents duplicate conversion by two runs.
    A malformed or incomplete Llama checkpoint raises before any model starts.
    """
    source = Path(model_name)
    if not source.is_dir() or (source / _RECEIPT).exists():
        return str(model_name)
    config_path = source / "config.json"
    if not config_path.exists():
        return str(model_name)
    config_data = json.loads(config_path.read_text())
    if config_data.get("model_type") != "llama":
        return str(model_name)
    if any(source.glob("*.pdparams")):
        return str(model_name)
    weight_map = _source_checkpoint(source)
    if not weight_map or not any(key.startswith("model.") for key in weight_map):
        return str(model_name)

    source = source.resolve()
    files = sorted(set(weight_map.values()))
    signature = {"version": _CONVERSION_VERSION, "source": str(source),
                 "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
                 "shards": []}
    for filename in files:
        path = source / filename
        if not path.is_file():
            raise FileNotFoundError(f"Missing HF checkpoint shard: {path}")
        stat = path.stat()
        signature["shards"].append({"name": filename, "bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns})
    digest = hashlib.sha256(json.dumps(signature, sort_keys=True).encode()).hexdigest()[:16]
    default_cache = Path(__file__).resolve().parents[1] / ".runtime" / "paddleformers"
    cache_root = Path(os.environ.get("PF_HOME", str(default_cache))) / "pitpo_native"
    cache_root.mkdir(parents=True, exist_ok=True)
    destination = cache_root / f"{source.name}-{digest}"
    with (cache_root / f".{destination.name}.lock").open("a") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        receipt_path = destination / _RECEIPT
        if receipt_path.exists():
            receipt = json.loads(receipt_path.read_text())
            if receipt.get("source_signature") != signature or not receipt.get("all_tensors_verified"):
                raise RuntimeError(f"Invalid converted checkpoint receipt: {receipt_path}")
            return str(destination)
        staging = Path(tempfile.mkdtemp(prefix=f".{destination.name}.pending-", dir=cache_root))
        try:
            _convert_llama(source, staging, config_data, weight_map, signature)
            os.replace(staging, destination)
        finally:
            if staging.exists():
                shutil.rmtree(staging)
    logger.info("Verified HF Llama checkpoint converted to native Paddle layout: %s", destination)
    return str(destination)


def _convert_llama(source, output, config_data, weight_map, signature):
    from paddleformers.transformers import LlamaConfig, LlamaForCausalLM
    from paddleformers.utils.safetensors import fast_safe_open
    from safetensors.numpy import save_file

    config = LlamaConfig.from_dict(config_data)
    mappings = LlamaForCausalLM._get_name_mappings(config)
    mapping_by_source = {mapping.source_name: mapping for mapping in mappings}
    required = {mapping.source_name for mapping in mappings
                if not mapping.source_name.endswith("rotary_emb.inv_freq")}
    missing = required - set(weight_map)
    unexpected = set(weight_map) - set(mapping_by_source)
    if missing or unexpected:
        raise ValueError(f"Incomplete or unsupported HF Llama checkpoint: missing={sorted(missing)}, unexpected={sorted(unexpected)}")
    native_map, details = {}, []
    total_size = 0
    for filename in sorted(set(weight_map.values())):
        converted = {}
        with fast_safe_open(str(source / filename), framework="np") as checkpoint:
            expected_keys = {key for key, shard in weight_map.items() if shard == filename}
            if set(checkpoint.keys()) != expected_keys:
                raise ValueError(f"Checkpoint index disagrees with shard {filename}")
            for key in sorted(expected_keys):
                mapping = mapping_by_source[key]
                tensor = checkpoint.get_tensor(key)
                mapped = np.ascontiguousarray(mapping.run({key: tensor}, key))
                if mapping.target_name in native_map:
                    raise ValueError(f"Duplicate converted parameter: {mapping.target_name}")
                converted[mapping.target_name] = mapped
                native_map[mapping.target_name] = filename
                total_size += mapped.nbytes
                details.append({"source": key, "target": mapping.target_name, "action": mapping.action,
                                "source_shape": list(tensor.shape), "target_shape": list(mapped.shape),
                                "dtype": str(mapped.dtype),
                                "sha256": hashlib.sha256(mapped.tobytes()).hexdigest()})
        logger.info("Writing native Llama checkpoint shard %s (%d tensors)", filename, len(converted))
        save_file(converted, str(output / filename), metadata={"format": "np"})
        # Re-read every serialized tensor and compare values, shapes, and dtype.
        with fast_safe_open(str(output / filename), framework="np") as check:
            if set(check.keys()) != set(converted):
                raise RuntimeError(f"Native shard lost parameters: {filename}")
            for key, expected in converted.items():
                actual = check.get_tensor(key)
                if actual.shape != expected.shape or actual.dtype != expected.dtype or not np.array_equal(actual, expected):
                    raise RuntimeError(f"Native parameter verification failed: {key}")
        del converted
    for path in source.iterdir():
        if path.name in set(weight_map.values()) or path.name == "model.safetensors.index.json":
            continue
        if path.name.startswith("."):
            continue
        if path.name == "config.json":
            (output / path.name).write_text(json.dumps(config_data, indent=2) + "\n")
        else:
            (output / path.name).symlink_to(path, target_is_directory=path.is_dir())
    index = {"metadata": {"total_size": total_size}, "weight_map": native_map}
    (output / "model.safetensors.index.json").write_text(json.dumps(index, indent=2) + "\n")
    receipt = {"conversion_version": _CONVERSION_VERSION, "source_signature": signature,
               "mapping": "PaddleFormers LlamaForCausalLM._get_name_mappings / StateDictNameMapping.run",
               "convert_from_hf": False, "mapped_tensors": len(native_map),
               "expected_native_parameter_keys": sorted(mapping_by_source[key].target_name for key in required),
               "allowed_missing_native_keys": ["lm_head.weight"] if config.tie_word_embeddings else [],
               "all_tensors_verified": True, "tie_word_embeddings": bool(config.tie_word_embeddings),
               "shared_output_head": "lm_head.weight aliases llama.embed_tokens.weight" if config.tie_word_embeddings else None,
               "tensors": details}
    (output / _RECEIPT).write_text(json.dumps(receipt, indent=2) + "\n")
