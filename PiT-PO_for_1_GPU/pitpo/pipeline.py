from __future__ import annotations

import os
# Avoid the tokenizers parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# from collections.abc import Sequence
from typing import Any, Tuple, Sequence
import time
import numpy as np

from pitpo import code_manipulation
from pitpo import config as config_lib
from pitpo import evaluator
from pitpo import buffer
from pitpo import sampler
from pitpo import profile
from pitpo import grpo_trainer


def _extract_function_names(specification: str) -> Tuple[str, str]:
    """ Return the name of the function to evolve and of the function to run.

    The so-called specification refers to the boilerplate code template for a task.
    The template MUST have two important functions decorated with '@evaluate.run', '@equation.evolve' respectively.
    The function labeled with '@evaluate.run' is going to evaluate the generated code (like data-diven fitness evaluation).
    The function labeled with '@equation.evolve' is the function to be searched (like 'equation' structure).
    """
    run_functions = list(code_manipulation.yield_decorated(specification, 'evaluate', 'run'))
    evolve_functions = list(code_manipulation.yield_decorated(specification, 'equation', 'evolve'))
    
    if len(evolve_functions) != 1:
        raise ValueError('Expected 1 function decorated with `@equation.evolve`.')
    
    # If the spec does not define evaluate.run, return the placeholder name 'evaluate'; main() will inject a default impl
    function_to_run = run_functions[0] if len(run_functions) == 1 else 'evaluate'
    return evolve_functions[0], function_to_run



def main(
        specification: str,
        inputs: Sequence[Any],
        config: config_lib.Config,
        max_sample_nums: int | None,
        class_config: config_lib.ClassConfig,
        enable_grpo: bool = False,
        grpo_model_name: str = "/data/home/zdhs0037/zdhs0037_data/meta-llama/Llama-3.1-8B-Instruct/LLM-Research/Meta-Llama-3___1-8B-Instruct",
        grpo_learning_rate: float = 1e-5,
        grpo_batch_size: int = 2,
        grpo_reward_scaling: float = 10.0,
        grpo_min_mse_threshold: float = 1e-15,
        grpo_buffer_size: int = 1000,
        training_strategy: str = 'adaptive',
        grpo_train_every_n_valid: int | None = None,
        grpo_warmup_samples: int = 200,
        grpo_clip_epsilon: float = 0.2,
        grpo_kl_beta: float = 0.01,
        grpo_seed: int = 3408,
        grpo_max_seq_length: int = 4096,
        grpo_load_in_4bit: bool = True,
        grpo_fast_inference: bool = True,
        grpo_vllm_gpu_memory_utilization: float = 0.6,
        checkpoint_every: int = 100,
        export_merged_model: bool = False,
        **kwargs
):
    """Launch a single-GPU PiT-PO experiment.
    Args:
        specification: the boilerplate code for the problem.
        inputs       : the data instances for the problem.
        config       : config file.
        max_sample_nums: the maximum samples nums from LLM. 'None' refers to no stop.
        class_config: Configuration for LLM and sandbox classes.
        enable_grpo  : Whether to enable GRPO fine-tuning.
        grpo_model_name: Local path to the model for GRPO training.
        grpo_learning_rate: Learning rate for GRPO training.
        grpo_batch_size: Batch size for GRPO training.
        grpo_reward_scaling: Reward scaling factor for log transformation.
        grpo_min_mse_threshold: Minimum MSE threshold to prevent log(0).

        Training and generation are deliberately sequential and share one
        colocated model runtime. Updated LoRA tensors are handed to vLLM in
        memory; this entry point never starts or restarts a model server.
    """
    # If the specification lacks @evaluate.run, inject a default implementation
    run_functions_check = list(code_manipulation.yield_decorated(specification, 'evaluate', 'run'))
    if len(run_functions_check) != 1:
        default_eval = r'''
@evaluate.run
def evaluate(data: dict) -> float:
    """Evaluate the equation on data observations."""
    import numpy as _np
    from scipy.optimize import minimize as _minimize

    inputs, outputs = data['inputs'], data['outputs']
    X = inputs

    def _loss(params):
        y_pred = equation(*X.T, params)
        return _np.mean((y_pred - outputs) ** 2)

    result = _minimize(_loss, [1.0]*10, method='BFGS')
    optimized_params = getattr(result, 'x', None)
    loss_val = float(getattr(result, 'fun', _np.inf))

    if not _np.isfinite(loss_val):
        return None
    return -loss_val
'''
        specification = specification + "\n\n" + default_eval
    
    function_to_evolve, function_to_run = _extract_function_names(specification)
    template = code_manipulation.text_to_program(specification)
    database = buffer.ExperienceBuffer(config.experience_buffer, template, function_to_evolve)

    # get log_dir and create profiler
    log_dir = kwargs.get('log_dir', None)
    if log_dir is None:
        profiler = None
    else:
        profiler = profile.Profiler(log_dir)
    # Reward-curve directory
    reward_curves_dir = kwargs.get('reward_curves_dir', None)

    if not config.use_api and (config.num_samplers != 1 or config.num_evaluators != 1):
        raise ValueError(
            "The colocated PiT-PO runtime requires num_samplers=1 and "
            "num_evaluators=1 so training and generation cannot overlap."
        )
    if config.use_api and enable_grpo:
        raise ValueError(
            "GRPO cannot update a hosted API model. Use the colocated local runtime."
        )

    # Local generation and GRPO share this one model object.  Even the
    # --disable_grpo baseline uses it, but without a trainable LoRA adapter.
    grpo_trainer_instance = None
    if not config.use_api:
        print(f"Initializing shared model runtime: {grpo_model_name}")
        checkpoint_dir = os.path.join(log_dir, "checkpoints") if log_dir else None
        grpo_trainer_instance = grpo_trainer.GRPOTrainer(
            model_name=grpo_model_name,
            learning_rate=grpo_learning_rate,
            batch_size=grpo_batch_size,
            reward_scaling=grpo_reward_scaling,
            min_mse_threshold=grpo_min_mse_threshold,
            buffer_size=grpo_buffer_size,
            use_lora=enable_grpo,
            max_seq_length=grpo_max_seq_length,
            load_in_4bit=grpo_load_in_4bit,
            fast_inference=grpo_fast_inference,
            vllm_gpu_memory_utilization=grpo_vllm_gpu_memory_utilization,
            samples_per_prompt=config.samples_per_prompt,
            warmup_samples=grpo_warmup_samples,
            grpo_clip_epsilon=grpo_clip_epsilon,
            grpo_kl_beta=grpo_kl_beta,
            seed=grpo_seed,
            checkpoint_dir=checkpoint_dir,
            checkpoint_every=checkpoint_every,
        )
        # Pass the reward-curve directory
        if reward_curves_dir is not None and hasattr(grpo_trainer_instance, 'set_reward_curves_dir'):
            try:
                grpo_trainer_instance.set_reward_curves_dir(reward_curves_dir)
            except Exception:
                pass
        print("Shared model runtime initialized successfully")

    evaluators = []
    for _ in range(config.num_evaluators):
        evaluators.append(evaluator.Evaluator(
            database,
            template,
            function_to_evolve,
            function_to_run,
            inputs,
            timeout_seconds=config.evaluate_timeout_seconds,
            sandbox_class=class_config.sandbox_class,
            grpo_trainer=grpo_trainer_instance,
            enable_grpo=enable_grpo,
            training_strategy=training_strategy
        ))


    # Configure per-evaluator training cadence
    try:
        if enable_grpo and grpo_trainer_instance is not None and grpo_train_every_n_valid:
            for ev in evaluators:
                if hasattr(ev, 'set_train_every_n_valid'):
                    ev.set_train_every_n_valid(int(grpo_train_every_n_valid))
    except Exception:
        pass

    initial = template.get_function(function_to_evolve).body
    evaluators[0].analyse(initial, island_id=None, version_generated=None, profiler=profiler)

    llm_class = class_config.llm_class if config.use_api else sampler.GRPOLocalLLM
    print(f"Using {'hosted API' if config.use_api else 'colocated shared'} LLM: {llm_class.__name__}")

    # Build samplers
    samplers_list = []
    for _ in range(config.num_samplers):
        sampler_instance = sampler.Sampler(
            database, evaluators,
            config.samples_per_prompt,
            max_sample_nums=max_sample_nums,
            llm_class=llm_class,
            config=config
        )
        if grpo_trainer_instance is not None:
            sampler_instance._llm.set_grpo_trainer(grpo_trainer_instance)
        samplers_list.append(sampler_instance)

    completed = False
    try:
        for s in samplers_list:
            s.sample(profiler=profiler)
        completed = True
    except BaseException:
        if enable_grpo and grpo_trainer_instance is not None and log_dir:
            emergency_dir = os.path.join(log_dir, "emergency_adapter")
            try:
                grpo_trainer_instance.save_adapter_checkpoint(
                    emergency_dir, reason="emergency"
                )
            except Exception as checkpoint_error:
                print(f"[GRPO] Emergency checkpoint failed: {checkpoint_error}")
        raise
    finally:
        if profiler:
            try:
                profiler.write_time_summary()
            except Exception:
                pass

    if completed and enable_grpo and grpo_trainer_instance is not None and log_dir:
        final_adapter_dir = os.path.join(log_dir, "final_adapter")
        grpo_trainer_instance.save_adapter_checkpoint(
            final_adapter_dir, reason="final"
        )
        if export_merged_model:
            merged_dir = os.path.join(log_dir, "final_merged_model")
            if not grpo_trainer_instance.save_merged_model(merged_dir):
                raise RuntimeError("Failed to export the final merged model")
