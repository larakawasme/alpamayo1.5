#!/usr/bin/env python3
"""MATH-500 (Lightman et al., 2023, arXiv:2305.20050).

The 500-problem MATH test subset from "Let's Verify Step by Step"
(HuggingFaceH4/MATH-500). The benchmark defines problems and reference answers
only, not a prompt or a grader. Here: zero-shot, the commonly used instruction
"Please reason step by step, and put your final answer within \\boxed{}."
appended to the problem, chat template on, greedy decoding, graded by
math-verify (symbolic equivalence of the extracted answer with the reference).
"""

import sys
from collections import defaultdict

from datasets import load_dataset
from math_verify import parse, verify

from RunBenchmark.evals.common import Job, pct, run_benchmark
from RunBenchmark.models import GenerationParams

NAME = "math500"
METRIC = "accuracy"
INSTRUCTION = "Please reason step by step, and put your final answer within \\boxed{}."


def add_args(parser):
    parser.add_argument("--max-new-tokens", type=int, default=4096)


def load_jobs(args):
    data = load_dataset("HuggingFaceH4/MATH-500", split="test")
    if args.limit is not None:
        data = data.select(range(min(args.limit, len(data))))
    examples = [
        {
            "id": row["unique_id"],
            "reference": row["answer"],
            "subject": row["subject"],
            "level": row["level"],
            "_problem": row["problem"],
        }
        for row in data
    ]
    return [Job(NAME, examples, GenerationParams(max_new_tokens=args.max_new_tokens), "predictions.jsonl")]


def build_prompt(model, args, job, example):
    return {"prompt": model.format_prompt(f"{example['_problem']}\n\n{INSTRUCTION}")}


def grade(output, answer):
    """Returns (extracted answer or None, correct)."""
    try:
        gold = parse(f"${answer}$")
        pred = parse(output)
        if not pred:
            return None, False
        return str(pred[-1]), bool(verify(gold, pred))
    except Exception:
        return None, False


def score(args, jobs, records):
    rows = []
    by_subject, by_level = defaultdict(list), defaultdict(list)
    for record in records[NAME]:
        extracted, ok = grade(record["output"], record["reference"])
        rows.append({"id": record["id"], "extracted": extracted, "score": float(ok)})
        by_subject[record["subject"]].append(ok)
        by_level[f"level_{record['level']}"].append(ok)
    breakdown = {
        "by_level": {k: pct(v) for k, v in sorted(by_level.items())},
        "by_subject": {k: pct(v) for k, v in sorted(by_subject.items())},
    }
    return rows, pct(r["score"] for r in rows), breakdown


if __name__ == "__main__":
    run_benchmark(sys.modules[__name__])
