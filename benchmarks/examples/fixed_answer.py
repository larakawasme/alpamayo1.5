"""Minimal `Model` subclass: no weights, no GPU, returns a fixed string.

Shows the smallest interface a custom model needs. Useful for checking that a
plugin is picked up and that the prediction/score/result files are written.

    PYTHONPATH=examples uv run python -m RunBenchmark.evals.math500 \
        --model-class fixed_answer:FixedAnswerModel --model fixed-answer --limit 4
"""

from RunBenchmark.models import Model


class FixedAnswerModel(Model):
    def __init__(self, name, answer):
        self.name = name
        self.answer = answer

    @classmethod
    def add_args(cls, parser):
        parser.add_argument(
            "--answer",
            default="The answer is (A). \\boxed{0}",
            help="Text returned for every prompt.",
        )

    @classmethod
    def from_args(cls, args):
        return cls(args.model, args.answer)

    # Tokens are whitespace-separated words here; a real model must count
    # exactly as its own encoder does.
    def num_tokens(self, text):
        return len(text.split())

    def truncate_middle(self, text, max_tokens):
        words = text.split(" ")
        if len(words) <= max_tokens:
            return text
        half = max_tokens // 2
        return " ".join(words[:half] + words[-half:])

    def generate(self, prompts, params):
        return [self.answer for _ in prompts]
