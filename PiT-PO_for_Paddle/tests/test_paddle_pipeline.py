"""Exercise the real search/evaluation/training loop using deterministic candidates.

The candidate generator is a fixture. It proves that evaluated equation feedback
updates Paddle LoRA and reaches the next search round; it does not establish that
an untrained tiny language model can discover equations.
"""
from __future__ import annotations

import json

import numpy as np
import paddle
import pytest

from pitpo import config, evaluator, grpo_trainer, pipeline, sampler

from smoke_paddle_runtime import TinyCharTokenizer, make_tiny_model


def test_real_pipeline_fits_candidates_updates_lora_and_saves_adapter(tmp_path, monkeypatch):
    paddle.set_device("cpu")
    paddle.seed(123)
    original_trainer = grpo_trainer.GRPOTrainer
    runtimes = []
    initial_trainable_weights = {}
    generations_at_updates = []

    def factory(**kwargs):
        runtime = original_trainer(**kwargs, model=make_tiny_model(seed=123), tokenizer=TinyCharTokenizer(),
                                   lora_r=2, lora_alpha=4, enable_physical_penalty=False)
        runtimes.append(runtime)
        initial_trainable_weights.update(
            (name, parameter.numpy().copy())
            for name, parameter in runtime.model.named_parameters() if not parameter.stop_gradient
        )

        def deterministic_candidates(**generation_kwargs):
            generations_at_updates.append(runtime.training_updates)
            # Parameter fitting and all feedback/training remain real.
            bodies = [
                "    return params[0] * x + params[1]\n",
                "    return params[0] * x + params[1] + params[2] * x**2\n",
            ]
            return bodies[:generation_kwargs["num_return_sequences"]]

        runtime.generate = deterministic_candidates
        return runtime

    monkeypatch.setattr(pipeline.grpo_trainer, "GRPOTrainer", factory)
    # Keep the upstream least-squares fitter and spawned sandbox in this test.
    x = np.linspace(-2, 2, 64)
    inputs = {"data": {"inputs": x[:, None], "outputs": 2.5 * x - 0.75 + 0.02 * np.cos(3 * x)}}
    specification = '''import numpy as np
@equation.evolve
def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    return params[0] * x + params[1]
'''
    result = pipeline.main(
        specification=specification, inputs=inputs,
        config=config.Config(samples_per_prompt=2, max_new_tokens=32, evaluate_timeout_seconds=30),
        max_sample_nums=4,
        class_config=config.ClassConfig(llm_class=sampler.LocalLLM, sandbox_class=evaluator.LocalSandbox),
        enable_grpo=True, grpo_model_name="offline-tiny-llama-fixture", grpo_device="cpu",
        grpo_max_seq_length=256, grpo_learning_rate=1e-3,
        grpo_warmup_samples=0, grpo_required_train_samples=2,
        grpo_train_every_n_valid=2, grpo_buffer_size=16,
        log_dir=str(tmp_path), training_strategy="continuous",
    )
    assert result["completed"]
    assert result["generated_candidates"] == 4
    assert result["training_updates"] > 0
    assert result["best_score"] is not None and -1e-3 < result["best_score"] <= 0
    assert generations_at_updates[0] == 0
    assert generations_at_updates[-1] > 0
    assert list((tmp_path / "final_adapter").rglob("*.pdparams"))
    assert (tmp_path / "metrics.jsonl").is_file()
    assert json.loads((tmp_path / "run_summary.json").read_text())["training_updates"] > 0
    assert runtimes[0].training_updates == result["training_updates"]
    assert any(not np.array_equal(initial_trainable_weights[name], parameter.numpy())
               for name, parameter in runtimes[0].model.named_parameters()
               if not parameter.stop_gradient), "Pipeline training left every LoRA tensor unchanged"
    assert any(exp["metadata"]["reward_decomposition"]["ast_node_count"] > 0
               for exp in runtimes[0].training_buffer)


def test_evaluator_propagates_training_failure():
    class FailingTrainer:
        def __init__(self):
            self.training_buffer = []
            self.scores_since_finetune = []

        def add_experience(self, **kwargs):
            self.training_buffer.append({"mse": kwargs["mse"]})
            self.scores_since_finetune.append(1.0)

        def _is_valid_experience(self, experience):
            return True

        def should_train(self):
            return True

        def train_step(self, num_epochs):
            raise RuntimeError("fixture training failure")

    instance = evaluator.Evaluator.__new__(evaluator.Evaluator)
    instance._grpo_trainer = FailingTrainer()
    instance._training_counter = 0
    instance._train_every_n_valid = 1
    instance._strategy_config = None
    with pytest.raises(RuntimeError, match="fixture training failure"):
        instance._handle_grpo_training("    return x\n", {"data": -1.0}, original_prompt="prompt")
