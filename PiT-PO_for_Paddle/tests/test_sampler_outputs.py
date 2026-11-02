"""Instruction models can wrap a valid function body in Markdown fences."""
from pitpo import config, sampler


def test_fenced_body_can_be_evaluated_as_the_requested_equation():
    raw = "```python\n    result = params[0] * x + params[1]\n    return result\n```"
    body = sampler._extract_body(raw, config.Config())
    assert sampler._is_valid_generated_body(body)
    namespace = {}
    exec("def equation(x, params):\n" + body, namespace)
    assert namespace["equation"](2, [3, 4]) == 10
