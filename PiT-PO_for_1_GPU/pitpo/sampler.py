from __future__ import annotations

from abc import ABC, abstractmethod
import ast
import codecs
import http.client
import json
import os
import time
from typing import Collection, Sequence, Type

import numpy as np

from pitpo import buffer
from pitpo import config as config_lib
from pitpo import evaluator
from pitpo import grpo_trainer


class LLM(ABC):
    def __init__(self, samples_per_prompt: int) -> None:
        self._samples_per_prompt = samples_per_prompt

    def _draw_sample(self, prompt: str) -> str:
        raise NotImplementedError("Must provide a language model.")

    @abstractmethod
    def draw_samples(self, prompt: str, config: config_lib.Config) -> Collection[str]:
        return [self._draw_sample(prompt) for _ in range(self._samples_per_prompt)]


class Sampler:
    """Samples programs and evaluates them sequentially against one shared model."""

    _global_samples_nums: int = 0

    def __init__(
        self,
        database: buffer.ExperienceBuffer,
        evaluators: Sequence[evaluator.Evaluator],
        samples_per_prompt: int,
        config: config_lib.Config,
        max_sample_nums: int | None = None,
        llm_class: Type[LLM] = LLM,
    ):
        self._samples_per_prompt = samples_per_prompt
        self._database = database
        self._evaluators = evaluators
        self._llm = llm_class(samples_per_prompt)
        self._max_sample_nums = max_sample_nums
        self.config = config

    def sample(self, **kwargs):
        """Continuously get prompts, generate programs, and evaluate them."""
        while True:
            if self._max_sample_nums and self.__class__._global_samples_nums >= self._max_sample_nums:
                break

            prompt = self._database.get_prompt()
            previous_sample_count = self._get_global_sample_nums()
            reset_time = time.time()
            samples = self._llm.draw_samples(prompt.code, self.config)
            sample_time = (time.time() - reset_time) / self._samples_per_prompt
            sample_indices = getattr(self._llm, "last_sample_indices", range(1, len(samples) + 1))
            drawn_count = getattr(self._llm, "last_draw_sample_count", len(samples))

            transition_seconds = None
            consume_transition = getattr(
                self._llm, "consume_train_to_generation_seconds", None
            )
            if callable(consume_transition):
                transition_seconds = consume_transition()
            profiler = kwargs.get("profiler")
            if transition_seconds is not None and profiler is not None:
                profiler.add_train_to_generation_time(transition_seconds)

            for relative_index, sample in zip(sample_indices, samples):
                cur_global_sample_nums = previous_sample_count + relative_index
                if self._max_sample_nums and cur_global_sample_nums > self._max_sample_nums:
                    break
                self.set_global_sample_nums(cur_global_sample_nums)
                chosen_evaluator: evaluator.Evaluator = np.random.choice(self._evaluators)
                chosen_evaluator.analyse(
                    sample,
                    prompt.island_id,
                    prompt.version_generated,
                    **kwargs,
                    global_sample_nums=cur_global_sample_nums,
                    sample_time=sample_time,
                    original_prompt=prompt.code,
                )
            # Count every generated candidate, including bodies dropped during
            # parsing, so the warmup boundary follows candidate 200 exactly.
            drawn_total = previous_sample_count + drawn_count
            if self._max_sample_nums:
                drawn_total = min(drawn_total, self._max_sample_nums)
            self.set_global_sample_nums(drawn_total)

    def _get_global_sample_nums(self) -> int:
        return self.__class__._global_samples_nums

    def set_global_sample_nums(self, num):
        self.__class__._global_samples_nums = num

    def _global_samples_nums_plus_one(self):
        self.__class__._global_samples_nums += 1


def _extract_body(sample: str, config: config_lib.Config) -> str:
    """Extract and clean the last generated function body."""
    lines = sample.splitlines()

    def _postprocess_body_lines(body_lines: list[str]) -> list[str]:
        cleaned: list[str] = []
        in_triple = False
        for line in body_lines:
            raw = line
            stripped = raw.strip()
            if stripped.startswith("```"):
                continue
            if '\"\"\"' in raw or "'''" in raw:
                count = raw.count('\"\"\"') + raw.count("'''")
                if count % 2 == 1:
                    in_triple = not in_triple
                continue
            if in_triple:
                continue
            lowered = stripped.lower()
            if (
                stripped.startswith("#")
                or lowered.startswith("note:")
                or lowered.startswith("assistant")
                or stripped.startswith("import ")
            ):
                continue
            if stripped.startswith("return") and "assistant" in lowered:
                stripped = stripped[: lowered.find("assistant")].rstrip()
            if stripped and not raw.startswith("    "):
                raw = "    " + stripped
            elif not stripped:
                raw = ""
            cleaned.append(raw)
        while cleaned and not cleaned[0].strip():
            cleaned.pop(0)
        while cleaned and not cleaned[-1].strip():
            cleaned.pop()
        return cleaned

    last_return = None
    for index, line in enumerate(lines):
        stripped = line.strip()
        if "return" in stripped and not stripped.startswith("#"):
            last_return = index
    if last_return is not None:
        for index in range(last_return, -1, -1):
            if lines[index].lstrip().startswith("def "):
                body = _postprocess_body_lines(lines[index + 1:last_return + 1])
                if body:
                    return "\n".join(body) + "\n"
                break

    for index, line in enumerate(lines):
        if line.lstrip().startswith("def "):
            body = _postprocess_body_lines(lines[index + 1:])
            if body:
                return "\n".join(body) + "\n"
            break
    return sample


def _is_valid_generated_body(body: str) -> bool:
    if not isinstance(body, str) or not body.strip() or "return" not in body:
        return False
    lines = [
        line if line.startswith("    ") or not line.strip() else "    " + line
        for line in body.splitlines()
    ]
    try:
        tree = ast.parse("def __tmp__():\n" + "\n".join(lines))
    except SyntaxError:
        return False
    return any(isinstance(node, ast.Return) for node in ast.walk(tree))


def _unescape_raw(text: str) -> str:
    if not isinstance(text, str):
        return text
    try:
        return codecs.decode(text, "unicode_escape")
    except Exception:
        try:
            return text.encode("utf-8").decode("unicode_escape")
        except Exception:
            return text


def _log_raw_response(raw: str, source: str = "LLM") -> None:
    try:
        unescaped = _unescape_raw(raw)
    except Exception:
        unescaped = raw
    print(f"[RAW OUTPUT - {source}]\n{unescaped}\n" + "-" * 80, flush=True)


class LocalLLM(LLM):
    """Adapter for the colocated model runtime and the optional hosted API."""

    def __init__(self, samples_per_prompt: int, trim: bool = True) -> None:
        super().__init__(samples_per_prompt)
        self._trim = trim
        self._grpo_trainer: grpo_trainer.GRPOTrainer | None = None

    def set_grpo_trainer(self, trainer: grpo_trainer.GRPOTrainer) -> None:
        self._grpo_trainer = trainer
        print("Shared GRPO model runtime attached to LocalLLM")

    def draw_samples(self, prompt: str, config: config_lib.Config) -> Collection[str]:
        if config.use_api:
            raw_samples = self._draw_samples_api(prompt, config)
            source = "API"
        else:
            if self._grpo_trainer is None:
                raise RuntimeError("Shared model runtime was not attached to LocalLLM")
            raw_samples = self._grpo_trainer.generate(
                prompt=prompt,
                num_return_sequences=self._samples_per_prompt,
                max_new_tokens=config.max_new_tokens,
                temperature=0.6,
                top_p=0.95,
                top_k=50,
                repetition_penalty=1.1,
            )
            source = "SHARED_LLM"

        for index, raw in enumerate(raw_samples, start=1):
            _log_raw_response(raw, f"{source}_{index}")
        samples = (
            [_extract_body(sample, config) for sample in raw_samples]
            if self._trim
            else list(raw_samples)
        )
        self.last_draw_sample_count = len(samples)
        self.last_sample_indices = [
            index for index, sample in enumerate(samples, start=1)
            if _is_valid_generated_body(sample)
        ]
        valid = [samples[index - 1] for index in self.last_sample_indices]
        dropped = len(samples) - len(valid)
        if dropped:
            print(f"[WARN] Dropped {dropped} invalid generated function bodies")
        return valid

    def consume_train_to_generation_seconds(self) -> float | None:
        if self._grpo_trainer is None:
            return None
        return self._grpo_trainer.consume_train_to_generation_seconds()

    def _draw_samples_api(self, prompt: str, config: config_lib.Config) -> list[str]:
        """Keep hosted API experiments without retaining a local HTTP server."""
        samples: list[str] = []
        for _ in range(self._samples_per_prompt):
            connection = http.client.HTTPSConnection("api.openai.com", timeout=30)
            payload = json.dumps({
                "max_tokens": config.max_new_tokens,
                "temperature": 1.0,
                "top_p": 0.9,
                "frequency_penalty": 0.2,
                "presence_penalty": 0.1,
                "stop": ["Okay", "I need", "Let me", "Looking"],
                "model": config.api_model,
                "messages": [{"role": "user", "content": prompt}],
            })
            headers = {
                "Authorization": f"Bearer {os.environ['API_KEY']}",
                "Content-Type": "application/json",
            }
            connection.request("POST", "/v1/chat/completions", payload, headers)
            response = connection.getresponse()
            raw_text = response.read().decode("utf-8", errors="ignore")
            if response.status != 200:
                raise RuntimeError(f"API HTTP {response.status}: {raw_text[:200]}")
            choices = json.loads(raw_text).get("choices") or []
            if not choices:
                raise RuntimeError("API response did not contain a completion")
            samples.append(choices[0].get("message", {}).get("content", ""))
        return samples


GRPOLocalLLM = LocalLLM
