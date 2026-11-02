"""HF Llama shards must populate native parameters instead of random weights."""
import json
from pathlib import Path

import ml_dtypes
import numpy as np
import paddle
import pytest
from safetensors.numpy import save_file

from pitpo.model_loading import prepare_model_checkpoint


def make_hf_shards(root, bf16=False):
    from paddleformers.transformers import LlamaConfig, LlamaForCausalLM
    paddle.set_device("cpu")
    paddle.seed(123)
    config = LlamaConfig(
        vocab_size=67, hidden_size=32, intermediate_size=48, num_hidden_layers=1,
        num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=128,
        tie_word_embeddings=True, architectures=["LlamaForCausalLM"],
        use_flash_attention=False, use_fused_rms_norm=False, dtype="float32",
    )
    model = LlamaForCausalLM(config)
    model.eval()
    state = model.state_dict()
    hf = {}
    for mapping in LlamaForCausalLM._get_name_mappings(config):
        if mapping.target_name not in state:
            continue
        value = state[mapping.target_name].numpy()
        if mapping.action == "transpose":
            value = value.T
        if bf16:
            value = value.astype(ml_dtypes.bfloat16)
        hf[mapping.source_name] = np.ascontiguousarray(value)
    root.mkdir()
    (root / "config.json").write_text(json.dumps(config.to_dict()))
    keys = sorted(hf)
    shards = [keys[::2], keys[1::2]]
    index = {}
    for i, group in enumerate(shards):
        name = f"model-{i+1:05d}-of-00002.safetensors"
        save_file({key: hf[key] for key in group}, str(root / name), metadata={"format": "pt"})
        index.update({key: name for key in group})
    (root / "model.safetensors.index.json").write_text(json.dumps({"weight_map": index}))
    return model, hf


def test_sharded_hf_llama_loads_all_native_parameters_and_matching_logits(tmp_path, monkeypatch):
    from paddleformers.transformers import AutoModelForCausalLM
    monkeypatch.setenv("PF_HOME", str(tmp_path / "cache"))
    source = tmp_path / "hf"
    original, hf = make_hf_shards(source)
    resolved = prepare_model_checkpoint(str(source))
    loaded = AutoModelForCausalLM.from_pretrained(resolved, convert_from_hf=False, dtype="float32")
    loaded.eval()
    expected = original.state_dict()
    for name, value in loaded.state_dict().items():
        np.testing.assert_array_equal(value.numpy(), expected[name].numpy())
    assert loaded.get_input_embeddings().weight is loaded.get_output_embeddings().weight
    tokens = paddle.to_tensor([[1, 5, 6, 7]], dtype="int64")
    with paddle.no_grad():
        np.testing.assert_array_equal(loaded(tokens, return_dict=True).logits.numpy(),
                                      original(tokens, return_dict=True).logits.numpy())
    receipt = json.loads((Path(resolved) / "pitpo_hf_conversion.json").read_text())
    assert receipt["mapped_tensors"] == len(hf) and receipt["all_tensors_verified"]
    # A second caller reuses the fully verified atomic cache.
    assert prepare_model_checkpoint(str(source)) == resolved


def test_bf16_is_preserved_and_rectangular_projection_transposed(tmp_path, monkeypatch):
    from paddleformers.utils.safetensors import fast_safe_open
    monkeypatch.setenv("PF_HOME", str(tmp_path / "cache"))
    source = tmp_path / "hf"
    _, hf = make_hf_shards(source, bf16=True)
    resolved = Path(prepare_model_checkpoint(str(source)))
    index = json.loads((resolved / "model.safetensors.index.json").read_text())["weight_map"]
    target = "llama.layers.0.self_attn.k_proj.weight"
    with fast_safe_open(str(resolved / index[target]), framework="np") as checkpoint:
        actual = checkpoint.get_tensor(target)
    assert str(actual.dtype) == "bfloat16"
    np.testing.assert_array_equal(actual, hf["model.layers.0.self_attn.k_proj.weight"].T)
    assert actual.shape == (32, 16)


def test_missing_checkpoint_parameter_is_rejected(tmp_path, monkeypatch):
    monkeypatch.setenv("PF_HOME", str(tmp_path / "cache"))
    source = tmp_path / "hf"
    make_hf_shards(source)
    index_path = source / "model.safetensors.index.json"
    index = json.loads(index_path.read_text())
    del index["weight_map"]["model.layers.0.self_attn.k_proj.weight"]
    index_path.write_text(json.dumps(index))
    with pytest.raises(ValueError, match="Incomplete or unsupported"):
        prepare_model_checkpoint(str(source))
