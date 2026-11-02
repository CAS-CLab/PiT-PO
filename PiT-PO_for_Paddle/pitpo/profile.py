
from __future__ import annotations

import os.path
from typing import List, Dict
import logging
import json
from pitpo import code_manipulation


class Profiler:
    def __init__(
            self,
            log_dir: str | None = None,
            pkl_dir: str | None = None,
            max_log_nums: int | None = None,
    ):
        """
        Args:
            log_dir     : folder path for JSON samples, metrics, and timing logs.
            pkl_dir     : save the results to a pkl file.
            max_log_nums: stop logging if exceeding max_log_nums.
        """
        logging.getLogger().setLevel(logging.INFO)
        self._log_dir = log_dir
        self._json_dir = os.path.join(log_dir, 'samples') if log_dir else None
        if self._json_dir:
            os.makedirs(self._json_dir, exist_ok=True)
        self._max_log_nums = max_log_nums
        self._num_samples = 0
        self._cur_best_program_sample_order = None
        self._cur_best_program_score = -99999999
        self._cur_best_program_str = None
        self._evaluate_success_program_num = 0
        self._evaluate_failed_program_num = 0
        self._tot_sample_time = 0
        self._tot_evaluate_time = 0
        # Cumulative GRPO training time
        self._tot_grpo_time = 0
        self._train_to_generation_times = []
        self._all_sampled_functions: Dict[int, code_manipulation.Function] = {}

        self._metrics_path = os.path.join(log_dir, "metrics.jsonl") if log_dir else None

        self._each_sample_best_program_score = []
        self._each_sample_evaluate_success_program_num = []
        self._each_sample_evaluate_failed_program_num = []
        self._each_sample_tot_sample_time = []
        self._each_sample_tot_evaluate_time = []
        # Per-step GRPO training time
        self._each_sample_tot_grpo_time = []

    def _write_metrics(self):
        """Append portable metrics without importing another tensor framework."""
        if self._metrics_path is None:
            return
        record = {
            "step": self._num_samples,
            "best_score": self._cur_best_program_score,
            "legal_functions": self._evaluate_success_program_num,
            "illegal_functions": self._evaluate_failed_program_num,
            "sample_seconds": self._tot_sample_time,
            "evaluate_seconds": self._tot_evaluate_time,
            "train_seconds": self._tot_grpo_time,
        }
        if self._each_sample_tot_grpo_time:
            record["last_train_seconds"] = self._each_sample_tot_grpo_time[-1]
        if self._train_to_generation_times:
            record["train_to_generation_seconds"] = self._train_to_generation_times[-1]
        with open(self._metrics_path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    def _write_json(self, programs: code_manipulation.Function):
        if not self._json_dir:
            return
        sample_order = programs.global_sample_nums
        sample_order = sample_order if sample_order is not None else 0
        function_str = str(programs)
        score = programs.score
        content = {
            'sample_order': sample_order,
            'function': function_str,
            'score': score
        }
        path = os.path.join(self._json_dir, f'samples_{sample_order}.json')
        with open(path, 'w') as json_file:
            json.dump(content, json_file)

    def register_function(self, programs: code_manipulation.Function):
        if self._max_log_nums is not None and self._num_samples >= self._max_log_nums:
            return

        sample_orders: int = programs.global_sample_nums
        if sample_orders not in self._all_sampled_functions:
            self._num_samples += 1
            self._all_sampled_functions[sample_orders] = programs
            self._record_and_verbose(sample_orders)
            self._write_metrics()
            self._write_json(programs)

    def _record_and_verbose(self, sample_orders: int):
        function = self._all_sampled_functions[sample_orders]
        function_str = str(function).strip('\n')
        sample_time = function.sample_time
        evaluate_time = function.evaluate_time
        score = function.score
        # log attributes of the function
        print(f'================= Evaluated Function =================')
        print(f'{function_str}')
        print(f'------------------------------------------------------')
        print(f'Score        : {str(score)}')
        print(f'Sample time  : {str(sample_time)}')
        print(f'Evaluate time: {str(evaluate_time)}')
        print(f'Sample orders: {str(sample_orders)}')
        print(f'======================================================\n\n')

        # update best function in curve
        if function.score is not None and score > self._cur_best_program_score:
            self._cur_best_program_score = score
            self._cur_best_program_sample_order = sample_orders
            self._cur_best_program_str = function_str

        # update statistics about function
        if score is not None:
            self._evaluate_success_program_num += 1
        else:
            self._evaluate_failed_program_num += 1

        if sample_time:
            self._tot_sample_time += sample_time
        if evaluate_time:
            self._tot_evaluate_time += evaluate_time

    # Record one GRPO training duration
    def add_grpo_time(self, t: float):
        if t is None:
            return
        try:
            t = float(t)
        except Exception:
            return
        self._tot_grpo_time += t
        self._each_sample_tot_grpo_time.append(t)
        # Write the completed training duration immediately
        self._write_metrics()

    def add_train_to_generation_time(self, seconds: float):
        """Record the delay between a completed update and the next generation."""
        try:
            seconds = float(seconds)
        except (TypeError, ValueError):
            return
        if seconds < 0:
            return
        self._train_to_generation_times.append(seconds)
        self._write_metrics()

    # Summarize timings to a file and print to console
    def write_time_summary(self):
        summary = {
            'total_sample_time': self._tot_sample_time,
            'total_evaluate_time': self._tot_evaluate_time,
            'total_grpo_time': self._tot_grpo_time,
            'num_samples_logged': self._num_samples,
            'num_grpo_trainings': len(self._each_sample_tot_grpo_time),
            'num_train_to_generation_transitions': len(self._train_to_generation_times),
            'avg_sample_time': (self._tot_sample_time / self._num_samples) if self._num_samples else 0.0,
            'avg_evaluate_time': (self._tot_evaluate_time / self._num_samples) if self._num_samples else 0.0,
            'avg_grpo_time': (self._tot_grpo_time / len(self._each_sample_tot_grpo_time)) if self._each_sample_tot_grpo_time else 0.0,
            'avg_train_to_generation_time': (
                sum(self._train_to_generation_times) / len(self._train_to_generation_times)
                if self._train_to_generation_times else 0.0
            ),
            'max_train_to_generation_time': (
                max(self._train_to_generation_times)
                if self._train_to_generation_times else 0.0
            ),
        }
        # Console output
        print('================= Timing Summary =================')
        print(f"Total sample time   : {summary['total_sample_time']:.6f} s")
        print(f"Total evaluate time : {summary['total_evaluate_time']:.6f} s")
        print(f"Total GRPO time     : {summary['total_grpo_time']:.6f} s")
        print(f"#Samples logged     : {summary['num_samples_logged']}")
        print(f"#GRPO trainings     : {summary['num_grpo_trainings']}")
        print(f"Avg sample time     : {summary['avg_sample_time']:.6f} s")
        print(f"Avg evaluate time   : {summary['avg_evaluate_time']:.6f} s")
        print(f"Avg GRPO time       : {summary['avg_grpo_time']:.6f} s")
        print(f"Avg train->generate : {summary['avg_train_to_generation_time']:.6f} s")
        print(f"Max train->generate : {summary['max_train_to_generation_time']:.6f} s")
        print('===================================================')
        # File output
        if self._log_dir:
            out_path = os.path.join(self._log_dir, 'time_summary.json')
            try:
                with open(out_path, 'w') as f:
                    json.dump(summary, f, indent=2)
                print(f"[Profiler] Timing summary saved to: {out_path}")
            except Exception as e:
                print(f"[Profiler] Failed to write timing summary: {e}")
