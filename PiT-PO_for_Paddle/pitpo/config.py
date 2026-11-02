from __future__ import annotations

import dataclasses
from typing import Type

from pitpo import sampler
from pitpo import evaluator


@dataclasses.dataclass(frozen=True)
class ExperienceBufferConfig:
    """Configures Experience Buffer parameters.
    
    Args:
        functions_per_prompt (int): Number of previous hypotheses to include in prompts
        num_islands (int): Number of islands in experience buffer for diversity
        reset_period (int): Seconds between weakest island resets
        cluster_sampling_temperature_init (float): Initial cluster softmax sampling temperature
        cluster_sampling_temperature_period (int): Period for temperature decay
    """
    functions_per_prompt: int = 2 
    num_islands: int = 4 
    reset_period: int = 4 * 60 * 60
    cluster_sampling_temperature_init: float = 0.1
    cluster_sampling_temperature_period: int = 30_000


@dataclasses.dataclass(frozen=True)
class Config:
    """Configuration for PiT-PO experiments.
   
   Args:
       experience_buffer: Evolution multi-population settings
       num_samplers (int): Number of parallel samplers
       num_evaluators (int): Number of parallel evaluators
       samples_per_prompt (int): Number of hypotheses generated per prompt
       max_new_tokens (int): Maximum completion length for each hypothesis
       evaluate_timeout_seconds (int): Hypothesis evaluation timeout
       use_api (bool): API usage flag
   """
    experience_buffer: ExperienceBufferConfig = dataclasses.field(default_factory=ExperienceBufferConfig)
    num_samplers: int = 1 
    num_evaluators: int = 1
    samples_per_prompt: int = 4
    max_new_tokens: int = 1024
    evaluate_timeout_seconds: int = 30  
    use_api: bool = False
    api_model: str = "gpt-3.5-turbo"


@dataclasses.dataclass()
class ClassConfig:
    llm_class: Type[sampler.LLM]
    sandbox_class: Type[evaluator.Sandbox]
