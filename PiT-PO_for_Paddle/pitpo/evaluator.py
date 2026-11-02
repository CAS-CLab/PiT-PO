from __future__ import annotations

import os
# Avoid the tokenizers parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from abc import abstractmethod, ABC
import ast
import time
import logging
from collections.abc import Sequence
import copy
from typing import Any, Type
import profile
import multiprocessing
import numbers
import math

from pitpo import code_manipulation
from pitpo import buffer
from pitpo import evaluator_accelerate
from pitpo import grpo_trainer
from pitpo import evaluate_on_problems

logger = logging.getLogger(__name__)


class _FunctionLineVisitor(ast.NodeVisitor):
    """ Visitor that finds the last line number of a function with a given name."""

    def __init__(self, target_function_name: str) -> None:
        self._target_function_name: str = target_function_name
        self._function_end_line: int | None = None

    def visit_FunctionDef(self, node: Any) -> None: 
        """ Collect the end line number of the target function."""
        if node.name == self._target_function_name:
            self._function_end_line = node.end_lineno
        self.generic_visit(node)

    @property
    def function_end_line(self) -> int:
        """ Line number of the final line of function `target_function_name`."""
        assert self._function_end_line is not None 
        return self._function_end_line


def _trim_function_body(generated_code: str) -> str:
    """ Extract the body of the generated function, trimming anything after it.
    Please note that the indentation is REQUIRED !!!
    """
    if not generated_code:
        return ''

    code = f'def fake_function_header():\n{generated_code}'

    tree = None
    while tree is None:
        try:
            tree = ast.parse(code)
        
        except SyntaxError as e:
            if e.lineno is None: # Nothing could be saved when syntaxError
                return ''
            code = '\n'.join(code.splitlines()[:e.lineno - 1])

    if not code:
        return ''

    visitor = _FunctionLineVisitor('fake_function_header')
    visitor.visit(tree)
    body_lines = code.splitlines()[1:visitor.function_end_line]
    return '\n'.join(body_lines) + '\n\n'


def _sample_to_program(
        generated_code: str,
        version_generated: int | None,
        template: code_manipulation.Program,
        function_to_evolve: str,
) -> tuple[code_manipulation.Function, str]:
    """ 
    Return the compiled generated function and the full runnable program.
    This function removes the content after the generated function body.
    """
    body = _trim_function_body(generated_code)
    if version_generated is not None:
        body = code_manipulation.rename_function_calls(
            code=body,
            source_name=f'{function_to_evolve}_v{version_generated}',
            target_name=function_to_evolve
        )

    program = copy.deepcopy(template)
    evolved_function = program.get_function(function_to_evolve)
    evolved_function.body = body
    
    return evolved_function, str(program)


class Sandbox(ABC):
    """ Sandbox for executing generated code. """

    @abstractmethod
    def run(
            self,
            program: str,
            function_to_run: str,
            function_to_evolve: str,
            inputs: Any,  
            test_input: str, 
            timeout_seconds: int,
            **kwargs
    ) -> tuple[Any, bool]:
        """ Return `function_to_run(test_input)` and whether execution succeeded. """
        raise NotImplementedError(
            'Must provide a sandbox for executing untrusted code.')


class LocalSandbox(Sandbox):
    """
    Secure environment for executing and evaluating LLM generated programs.
    Prevents harmful operations, limits resource usage, and enforces timeouts.
    Returns a 'score' for the executed program.
    """

    def __init__(self, verbose=False, numba_accelerate=False):
        """
        Initialize Sandbox.
        
        Args:
        verbose (bool): Enable detailed output.
        numba_accelerate (bool): Use Numba for acceleration of evaluation (limited compatibility). 
        """
        self._verbose = verbose
        self._numba_accelerate = numba_accelerate


    def run(self, program: str, function_to_run: str, function_to_evolve: str, 
        inputs: Any, test_input: str, timeout_seconds: int, **kwargs) -> tuple[Any, bool]:
        """
        Execute the given program sample and return its score and success status.
        
        Note: This sandbox is specific to the equation program skeleton discovery problem.
        """

        dataset = inputs[test_input]
        # Spawn avoids inheriting an initialized Paddle CUDA runtime.
        context = multiprocessing.get_context("spawn")
        result_queue = context.Queue()
        
        try:
            process = context.Process(
                target=self._compile_and_run_function_with_params,
                args=(program, function_to_run, function_to_evolve, dataset, self._numba_accelerate, result_queue)
            )
            process.start()
            process.join(timeout=timeout_seconds)
            if process.is_alive():
                process.terminate()
                process.join()
                results = None, False
            else:
                results = self._get_results(result_queue)
        finally:
            try:
                import gc
                gc.collect()
                result_queue.close()
            except Exception:
                pass

        if self._verbose:
            self._print_evaluation_details(program, results, **kwargs)

        return results


    def _get_results(self, queue):
        for _ in range(5):
            if not queue.empty():
                return queue.get_nowait()
            time.sleep(0.1)
        return None, False


    def _print_evaluation_details(self, program, results, **kwargs):
        print('================= Evaluated Program =================')
        function = code_manipulation.text_to_program(program).get_function(kwargs.get('func_to_evolve', 'equation'))
        print(f'{str(function).strip()}\n-----------------------------------------------------')
        print(f'Score: {results}\n=====================================================\n\n')



    def _compile_and_run_function(self, program, function_to_run, function_to_evolve, 
                                  dataset, numba_accelerate, result_queue):
        try:
            # optimize the code (decorate function_to_run with @numba.jit())
            if numba_accelerate:
                program = evaluator_accelerate.add_numba_decorator(
                    program=program,
                    function_to_evolve=function_to_evolve
                )
            
            # execute the program, map func/var/class to global namespace
            all_globals_namespace = {}
            exec(program, all_globals_namespace)
            function_to_run = all_globals_namespace[function_to_run]
            evolved_function = all_globals_namespace[function_to_evolve]
            results = evaluate_on_problems.evaluate(dataset, evolved_function)
            print('results:', results)
            try:
            # (translated)
                score = float(results)
            except Exception:
                try:
                    result_queue.put((None, False))
                    # (translated)
                    try:
                        result_queue.close()
                        result_queue.join_thread()
                    except Exception:
                        pass
                    os._exit(1)
                except Exception:
                    os._exit(1)
                return
            
            # (translated)
            try:
                result_queue.put((score, True))
                try:
                    result_queue.close()
                    result_queue.join_thread()
                except Exception:
                    pass
                os._exit(0)
            except Exception:
                os._exit(1)
            
        # if raise any exception, execution is failed
        except Exception as e:
            print(f"Execution Error: {e}")
            try:
                result_queue.put((None, False))
                try:
                    result_queue.close()
                    result_queue.join_thread()
                except Exception:
                    pass
                os._exit(1)
            except Exception:
                os._exit(1)


    def _compile_and_run_function_with_params(self, program, function_to_run, function_to_evolve,
                                              dataset, numba_accelerate, result_queue):
        """Variant that returns score and optimized parameters (if available)."""
        try:
            # Optional acceleration
            if numba_accelerate:
                program = evaluator_accelerate.add_numba_decorator(
                    program=program,
                    function_to_evolve=function_to_evolve
                )

            # Compile
            all_globals_namespace = {}
            exec(program, all_globals_namespace)
            function_to_run = all_globals_namespace[function_to_run]
            evolved_function = all_globals_namespace[function_to_evolve]

            # Prefer evaluate_with_params if available
            eval_with_params = getattr(evaluate_on_problems, 'evaluate_with_params', None)
            if callable(eval_with_params):
                results = eval_with_params(dataset, evolved_function)
            else:
                results = evaluate_on_problems.evaluate(dataset, evolved_function)

            # Normalize payload shape for parent
            payload = None
            if isinstance(results, dict) and ('score' in results):
                try:
                    results['score'] = float(results['score'])
                except Exception:
                    results['score'] = None
                payload = results
            else:
                try:
                    payload = float(results)
                except Exception:
                    payload = None

            # Emit to queue and exit child
            try:
                result_queue.put((payload, payload is not None))
                try:
                    result_queue.close()
                    result_queue.join_thread()
                except Exception:
                    pass
                os._exit(0)
            except Exception:
                os._exit(1)
        except Exception:
            try:
                result_queue.put((None, False))
                try:
                    result_queue.close()
                    result_queue.join_thread()
                except Exception:
                    pass
                os._exit(1)
            except Exception:
                os._exit(1)

def _calls_ancestor(program: str, function_to_evolve: str) -> bool:
    """ Return whether the generated function is calling an earlier version. """
    for name in code_manipulation.get_functions_called(program):
        if name.startswith(f'{function_to_evolve}_v'):
            return True
    return False



class Evaluator:
    """ Class that analyses functions generated by LLMs. """

    def __init__(
            self,
            database: buffer.ExperienceBuffer,
            template: code_manipulation.Program,
            function_to_evolve: str, 
            function_to_run: str, 
            inputs: Sequence[Any], 
            timeout_seconds: int = 30,
            sandbox_class: Type[Sandbox] = Sandbox,
            grpo_trainer: grpo_trainer.GRPOTrainer = None,
            enable_grpo: bool = False,
            training_strategy: str = 'adaptive'
    ):
        self._database = database
        self._template = template
        self._function_to_evolve = function_to_evolve
        self._function_to_run = function_to_run
        self._inputs = inputs
        self._timeout_seconds = timeout_seconds
        self._sandbox = sandbox_class()
        self._grpo_trainer = grpo_trainer
        self._enable_grpo = enable_grpo
        self._training_counter = 0
        self._training_strategy = training_strategy
        self._notify_model_update_callback = None
        self._model_updated = False
        self._train_every_n_valid = None
        # Load the GRPO training-strategy preset
        if enable_grpo:
            try:
                import grpo_config
                self._strategy_config = grpo_config.get_strategy_config(training_strategy)
                logger.info(f"Using GRPO training strategy: {training_strategy}")
                logger.info(f"Strategy description: {self._strategy_config['training']['description']}")
            except ImportError:
                logger.warning('grpo_config not found; GRPO strategy config unavailable')
                self._strategy_config = None
        else:
            self._strategy_config = None

    def analyse(
            self,
            sample: str,
            island_id: int | None,
            version_generated: int | None,
            **kwargs 
    ) -> None:
        """ Compile the hypothesis sample into a program and executes it on test inputs. """
        new_function, program = _sample_to_program(
            sample, version_generated, self._template, self._function_to_evolve)
        scores_per_test = {}
        optimized_params_by_test = {}

        time_reset = time.time()
        
        for current_input in self._inputs:
            # Run the user-defined evaluate function in the sandbox for the current input
            eval_start = time.time()
            test_output, runs_ok = self._sandbox.run(
                program, self._function_to_run, self._function_to_evolve, self._inputs, current_input,
                self._timeout_seconds
            )
            single_eval_time = time.time() - eval_start
            # Accumulate the per-input evaluation time so the reported total reflects all sandbox calls
            if '___eval_time_acc' not in locals():
                ___eval_time_acc = 0.0
            ___eval_time_acc += single_eval_time

            if runs_ok and not _calls_ancestor(program, self._function_to_evolve) and test_output is not None:
                # Accept either plain float or dict with score/optimized_params
                if isinstance(test_output, dict) and ('score' in test_output):
                    try:
                        score_val = float(test_output.get('score', None))
                    except Exception:
                        score_val = None
                    if score_val is not None:
                        scores_per_test[current_input] = score_val
                        if 'optimized_params' in test_output and test_output['optimized_params'] is not None:
                            optimized_params_by_test[current_input] = test_output['optimized_params']
                else:
                    if not isinstance(test_output, (int, float)):
                        print(f'Error: test_output is {test_output}')
                        raise ValueError('@function.run did not return an int/float score.')
                    scores_per_test[current_input] = test_output

        evaluate_time = time.time() - time_reset
        # Prefer the accumulator if it has data; otherwise the wall-clock value remains
        try:
            if ___eval_time_acc > 0:
                evaluate_time = ___eval_time_acc
        except Exception:
            pass

        if scores_per_test:
            # Register program in database as before
            self._database.register_program(
                new_function,
                island_id,
                scores_per_test,
                **kwargs,
                evaluate_time=evaluate_time
            )
            
            # GRPO Training Integration
            # The seed equation populates islands but has no model-generated prompt group.
            if self._enable_grpo and self._grpo_trainer is not None and kwargs.get("original_prompt"):
                # Attach optimized params for token-level penalty
                extra_kwargs = dict(kwargs)
                extra_kwargs["equation_source"] = str(new_function)
                if optimized_params_by_test:
                    extra_kwargs['optimized_params_by_test'] = optimized_params_by_test
                self._handle_grpo_training(sample, scores_per_test, evaluate_time=evaluate_time, **extra_kwargs)
        
        else:
            profiler: profile.Profiler = kwargs.get('profiler', None)
            if profiler:
                global_sample_nums = kwargs.get('global_sample_nums', None)
                sample_time = kwargs.get('sample_time', None)
                new_function.global_sample_nums = global_sample_nums
                new_function.score = None
                new_function.sample_time = sample_time
                new_function.evaluate_time = evaluate_time
                profiler.register_function(new_function)

    def set_train_every_n_valid(self, n: int):
        try:
            n_val = int(n)
            if n_val > 0:
                self._train_every_n_valid = n_val
                logger.info(f"GRPO will trigger one fine-tune step every {n_val} valid samples")
        except Exception:
            pass

    def _handle_grpo_training(self, sample: str, scores_per_test: dict, **kwargs):
        """
        Handle GRPO training integration with configurable training strategies.
        
        Args:
            sample: The generated code sample
            scores_per_test: Dictionary of scores per test case
            **kwargs: Additional metadata
        """
        # Extract the prompt used to generate this sample
        prompt = kwargs.get('original_prompt', '')
        
        # Calculate average MSE from scores
        # Note: In the spec, the score is -loss, so we need to convert back to MSE
        losses = [-score for score in scores_per_test.values()]
        avg_mse = sum(losses) / len(losses) if losses else float('inf')
        
        # Skip if MSE is invalid
        if not (math.isfinite(avg_mse) and avg_mse >= 0):
            return
            
        # Add experience to GRPO trainer
        try:
            self._grpo_trainer.add_experience(
                prompt=prompt,
                response=sample,
                mse=avg_mse,
                scores_per_test=scores_per_test,
                equation_source=kwargs.get('equation_source'),
                optimized_params_by_test=kwargs.get('optimized_params_by_test'),
                island_id=kwargs.get('island_id'),
                version_generated=kwargs.get('version_generated'),
                global_sample_nums=kwargs.get('global_sample_nums'),
                evaluate_time=kwargs.get('evaluate_time')
            )
            
            # Count the number of currently buffered valid samples
            valid_count = 0
            try:
                valid_count = sum(1 for exp in self._grpo_trainer.training_buffer if self._grpo_trainer._is_valid_experience(exp))
            except Exception:
                valid_count = len(self._grpo_trainer.training_buffer)

            self._training_counter += 1

            # EvoTune-aligned: train once the count since the last successful step in scores_since_finetune reaches N
            trigger_by_n = False
            if self._train_every_n_valid:
                n_since = len(getattr(self._grpo_trainer, "scores_since_finetune", []) or [])
                if n_since >= self._train_every_n_valid:
                    trigger_by_n = True

            # Keep the legacy should_train guard (e.g. warmup); train only when both conditions are met
            should_train_now = trigger_by_n and self._grpo_trainer.should_train()
            
            if should_train_now:
                profiler: profile.Profiler = kwargs.get('profiler', None)
                _grpo_t0 = time.time()
                num_epochs = self._get_adaptive_epochs(avg_mse)
                stats = self._grpo_trainer.train_step(num_epochs=num_epochs)
                grpo_elapsed = time.time() - _grpo_t0

                if stats and stats.get("skipped"):
                    logger.info("GRPO update skipped: %s", stats.get("reason", stats["skipped"]))
                    return
                if not stats:
                    raise RuntimeError(
                        "GRPO was triggered but did not produce a model update"
                    )

                if profiler:
                    try:
                        profiler.add_grpo_time(grpo_elapsed)
                    except Exception:
                        pass

                self._model_updated = True
                try:
                    curve_dir = getattr(self._grpo_trainer, '_reward_curves_dir', None)
                    if curve_dir:
                        print(f"[GRPO] Reward curves updated at: {curve_dir}")
                except Exception:
                    pass
                print("=" * 60)
                print(f"GRPO training done | elapsed: {grpo_elapsed:.3f}s | stats: {stats}")
                print("=" * 60 + "\n")

                if hasattr(self, '_notify_model_update_callback') and self._notify_model_update_callback:
                    self._notify_model_update_callback()
        except Exception as e:
            print(f"Error in GRPO training: {e}")
            import traceback
            traceback.print_exc()
            # Scientific runs must never continue silently with stale weights.
            raise

    def _should_trigger_training(self, current_mse: float) -> bool:
        """
        Determine if training should be triggered based on current MSE and strategy.
        Modified to trigger training every time (continuous learning).
        
        Args:
            current_mse: Current sample's MSE
            
        Returns:
            True if training should be triggered (always True for continuous learning)
        """
        # Always trigger training check - let the GRPO trainer's should_train() method
        # decide based on buffer state
        return True
            
    def _get_adaptive_epochs(self, current_mse: float) -> int:
        """
        Get adaptive number of training epochs based on MSE quality.
        
        Args:
            current_mse: Current sample's MSE
            
        Returns:
            Number of epochs to train
        """
        if not self._strategy_config:
            # Optionally trigger training based on the configured strategy
            return 2 if current_mse < 0.1 else 1
            
        adaptive_params = self._strategy_config['training']['adaptive_params']
        
        if current_mse < 0.05:  # Excellent
            return 1 + adaptive_params['epoch_boost_excellent']
        elif current_mse < 0.5:  # Good
            return 1 + adaptive_params['epoch_boost_good']
        else:  # Regular
            return 1
