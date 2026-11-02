"""Numerical and masking checks for Paddle GRPO, without downloading a model."""
import unittest
import tempfile

import numpy as np
import paddle

from pitpo.grpo_trainer import GRPOTrainer
from smoke_paddle_runtime import make_runtime, add_test_experiences, make_tiny_model, TinyCharTokenizer


class PaddleGRPOLossTests(unittest.TestCase):
    def setUp(self):
        paddle.set_device("cpu")
        paddle.seed(3408)
        self.runtime = GRPOTrainer.__new__(GRPOTrainer)
        self.runtime.grpo_clip_epsilon = 0.2
        self.runtime.grpo_kl_beta = 0.01

    def batch(self):
        logits = paddle.randn([2, 5, 11])
        logits.stop_gradient = False
        labels = paddle.to_tensor([[-100, -100, 2, 3, 4], [-100, -100, 3, 4, -100]])
        mask = paddle.to_tensor([[1, 1, 1, 1, 1], [1, 1, 1, 1, 0]])
        old, _ = self.runtime._token_log_probs(logits.detach(), labels, mask)
        return logits, labels, mask, old

    def test_masked_prompt_and_padding_have_zero_gradient(self):
        logits, labels, mask, old = self.batch()
        loss = self.runtime.compute_grpo_loss(logits, labels, paddle.to_tensor([1.0, 2.0]), mask,
                                               old_log_probs=old, ref_log_probs=old)
        loss.backward()
        self.assertTrue(bool(paddle.isfinite(logits.grad).all()))
        np.testing.assert_array_equal(logits.grad[:, 0, :].numpy(), 0)
        np.testing.assert_array_equal(logits.grad[1, 3:, :].numpy(), 0)
        self.assertGreater(float(paddle.abs(logits.grad).sum()), 0)

    def test_fixed_old_policy_controls_probability_clipping(self):
        logits, labels, mask, old = self.batch()
        self.runtime.grpo_kl_beta = 0
        advantages = paddle.to_tensor([1.0, 1.0])
        loss = self.runtime.compute_grpo_loss(logits, labels, paddle.to_tensor([1.0, 2.0]), mask,
                                               old_log_probs=old - float(np.log(2)), advantages=advantages)
        self.assertAlmostEqual(float(loss), -1.2, places=5)
        self.assertAlmostEqual(self.runtime._last_loss_metrics["clip_fraction"], 1.0)

    def test_nonfinite_rewards_raise(self):
        logits, labels, mask, old = self.batch()
        with self.assertRaisesRegex(FloatingPointError, "rewards"):
            self.runtime.compute_grpo_loss(logits, labels, paddle.to_tensor([1.0, float("nan")]), mask,
                                           old_log_probs=old, ref_log_probs=old)

    def test_missing_old_policy_is_rejected(self):
        logits, labels, mask, old = self.batch()
        with self.assertRaisesRegex(ValueError, "old_log_probs"):
            self.runtime.compute_grpo_loss(logits, labels, paddle.to_tensor([1.0, 2.0]), mask)

    def test_completion_mask_and_coefficient_penalty_map_to_response(self):
        runtime = make_runtime()
        add_test_experiences(runtime)
        rows = runtime.prepare_training_data()
        for row in rows:
            self.assertEqual(row["completion_mask"][0], 0)
            self.assertGreater(sum(row["token_penalty_weights"]), 0)
            for completion, penalty in zip(row["completion_mask"], row["token_penalty_weights"]):
                if penalty:
                    self.assertEqual(completion, 1)
        # A one-example microbatch retains the globally prepared group advantage.
        self.assertGreater(np.std([row["advantages"] for row in rows]), 0)

    def test_single_example_microbatches_train_from_group_advantages(self):
        runtime = make_runtime(enable_coef_penalty=False, per_device_train_batch_size=1)
        add_test_experiences(runtime)
        before = {name: value.numpy().copy() for name, value in runtime.model.named_parameters()
                  if not value.stop_gradient}
        stats = runtime.train_step()
        self.assertEqual(stats["training_updates"], 4)
        self.assertTrue(any(not np.array_equal(before[name], value.numpy())
                            for name, value in runtime.model.named_parameters() if not value.stop_gradient))

    def test_singleton_prompt_groups_do_not_trigger_training(self):
        runtime = make_runtime()
        for index in range(4):
            runtime.add_experience(f"prompt {index}", "    return params[0] * x\n", 1.0,
                                   global_sample_nums=index + 1)
        self.assertFalse(runtime.should_train())

    def test_equation_body_recovers_ast_analysis(self):
        runtime = make_runtime()
        runtime.add_experience("def equation_v3(x, params):\n", "    return params[0] * x\n", 1.0,
                               global_sample_nums=1)
        decomposition = runtime.training_buffer[0]["metadata"]["reward_decomposition"]
        self.assertGreater(decomposition["ast_node_count"], 0)
        self.assertGreater(decomposition["p_cplx"], 0)

    def test_chat_prompt_uses_last_signature_and_task_parameter_bound(self):
        class ChatTokenizer:
            def apply_chat_template(self, messages, **kwargs):
                self.messages, self.options = messages, kwargs
                return "<chat>" + messages[-1]["content"] + "<assistant>"
        runtime = GRPOTrainer.__new__(GRPOTrainer)
        runtime._chat_template_enabled = True
        runtime.tokenizer = ChatTokenizer()
        formatted = runtime._format_generation_prompt(
            "import numpy as np\nMAX_NPARAMS = 23\n"
            "def equation_v0(x, z, params):\n    return params[0] * x\n"
            "def equation_v1(x, z, params):\n    \"\"\"Improved equation.\"\"\"\n"
        )
        self.assertIn("equation_v1(x, z, params)", formatted)
        self.assertIn("params[22]", formatted)
        self.assertIn("input variables are x, z, params", formatted)
        self.assertFalse(runtime.tokenizer.options["enable_thinking"])

    def test_loaded_rotary_cache_matches_runtime_window_without_changing_logits(self):
        model = make_tiny_model()
        model.eval()
        tokens = paddle.to_tensor([[1, 10, 11, 12, 13, 14]], dtype="int64")
        with paddle.no_grad():
            expected = model(input_ids=tokens, return_dict=True).logits.numpy()
        with tempfile.TemporaryDirectory() as checkpoint:
            model.save_pretrained(checkpoint)
            runtime = GRPOTrainer(
                model_name=checkpoint, tokenizer=TinyCharTokenizer(),
                max_seq_length=128, lora_r=4, lora_alpha=8,
            )
            self.assertEqual(runtime.model.config.max_position_embeddings, 128)
            rotary = runtime.model.model.llama.layers[0].self_attn.rotary_emb
            self.assertEqual(rotary.cos_cached.shape[1], 128)
            with paddle.no_grad():
                actual = runtime.model(input_ids=tokens, return_dict=True).logits.numpy()
            np.testing.assert_array_equal(actual, expected)


if __name__ == "__main__":
    unittest.main()
