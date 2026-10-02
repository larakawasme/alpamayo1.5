#!/usr/bin/env python3
"""MMLU-Pro (Wang et al., 2024, arXiv:2406.01574).

Follows TIGER-AI-Lab/MMLU-Pro evaluate_from_local.py: 5-shot CoT drawn from
the validation split of the same category, shots dropped until the prompt is
under max_model_length - max_new_tokens (4096 - 2048) tokens, greedy decoding,
stop at "Question:", three-stage regex answer extraction, and a seeded random
guess (random.seed(12345)) when no letter can be extracted.

Differences from upstream: generation uses the configured model backend
instead of vLLM, and --chat (wrap the prompt in the chat template) is an
optional extension; the official script sends the raw few-shot prompt.
"""

import random
import re
import sys
from collections import defaultdict

from datasets import load_dataset

from RunBenchmark.evals.common import Job, pct, run_benchmark
from RunBenchmark.models import GenerationParams

NAME = "mmlu_pro"
METRIC = "accuracy"
CHOICES = "ABCDEFGHIJKLMNOP"
# cot_prompt_lib/initial_prompt.txt, with "{$}" replaced by the category.
INITIAL_PROMPT = (
    "The following are multiple choice questions (with answers) about {}. "
    'Think step by step and then finish your answer with "the answer is (X)" '
    "where X is the correct letter choice.\n\n\n"
)
SEED = 12345


def add_args(parser):
    parser.add_argument("--ntrain", type=int, default=5)
    parser.add_argument("--max-model-length", type=int, default=4096)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument(
        "--subjects",
        default="all",
        help='Comma-separated categories, matched like upstream (substring, "_" = " "). Default: all 14.',
    )
    parser.add_argument(
        "--chat",
        action="store_true",
        help="Wrap the few-shot prompt in the chat template (not part of the official protocol).",
    )


def output_name(args):
    return f"{NAME}_chat" if args.chat else NAME


def preprocess(rows):
    rows = [dict(row) for row in rows]
    for row in rows:
        row["options"] = [opt for opt in row["options"] if opt != "N/A"]
    return rows


def select_subjects(all_subjects, spec):
    if spec == "all":
        return sorted(all_subjects)
    wanted = [s.replace(" ", "_") for s in spec.split(",")]
    selected = [s for s in all_subjects if any(w in s.replace(" ", "_") for w in wanted)]
    if not selected:
        raise SystemExit(f"No subject matches {spec!r}; choose from {sorted(all_subjects)}")
    return sorted(selected)


def load_jobs(args):
    dataset = load_dataset("TIGER-Lab/MMLU-Pro")
    test = preprocess(dataset["test"])
    val = preprocess(dataset["validation"])

    all_subjects = list(dict.fromkeys(row["category"] for row in test))
    shots_by_subject = defaultdict(list)
    for row in val:
        shots_by_subject[row["category"]].append(row)

    examples = []
    for subject in select_subjects(all_subjects, args.subjects):
        rows = [row for row in test if row["category"] == subject]
        if args.limit is not None:
            rows = rows[:args.limit]
        examples.extend(
            {
                "id": row["question_id"],
                "reference": row["answer"],
                "answer_index": row["answer_index"],
                "num_options": len(row["options"]),
                "subject": subject,
                "_row": row,
            }
            for row in rows
        )

    params = GenerationParams(max_new_tokens=args.max_new_tokens, stop_strings=["Question:"])
    return [Job(NAME, examples, params, "predictions.jsonl", {"shots": shots_by_subject})]


def format_example(example, including_answer):
    prompt = "Question:\n" + example["question"] + "\nOptions:\n"
    for letter, opt in zip(CHOICES, example["options"]):
        prompt += f"{letter}. {opt}\n"
    if including_answer:
        cot = example["cot_content"].replace(
            "A: Let's think step by step.", "Answer: Let's think step by step."
        )
        prompt += cot + "\n\n"
    else:
        prompt += "Answer: Let's think step by step."
    return prompt


def few_shot_prompt(shots, example, k):
    prompt = INITIAL_PROMPT.format(example["category"]) + "\n"
    for shot in shots[:k]:
        prompt += format_example(shot, including_answer=True)
    return prompt + format_example(example, including_answer=False)


def build_prompt(model, args, job, example):
    row = example["_row"]
    shots = job.context["shots"][row["category"]]
    budget = args.max_model_length - args.max_new_tokens
    k = args.ntrain
    while True:
        prompt = model.format_prompt(few_shot_prompt(shots, row, k), chat=args.chat)
        if model.num_tokens(prompt) < budget or k == 0:
            return {"prompt": prompt, "ntrain": k}
        k -= 1


def extract_answer(text):
    match = re.search(r"answer is \(?([A-J])\)?", text)
    if match:
        return match.group(1)
    match = re.search(r".*[aA]nswer:\s*([A-J])", text)
    if match:
        return match.group(1)
    match = re.search(r"\b[A-J]\b(?!.*\b[A-J]\b)", text, re.DOTALL)
    return match.group(0) if match else None


def score(args, jobs, records):
    by_subject_records = defaultdict(list)
    for record in records[NAME]:
        by_subject_records[record["subject"]].append(record)

    rng = random.Random(SEED)
    rows, by_subject, misses = [], {}, 0
    for subject in sorted(by_subject_records):
        subject_records = by_subject_records[subject]
        preds = [extract_answer(r["output"].split("Question:")[0]) for r in subject_records]
        # Upstream scores each subject twice (save_res is called twice in eval_cot),
        # drawing random guesses both times; the second pass is the reported one.
        for _ in range(2):
            correct = [
                rng.randint(0, r["num_options"] - 1) == r["answer_index"] if pred is None else pred == r["reference"]
                for r, pred in zip(subject_records, preds)
            ]
        by_subject[subject] = pct(correct)
        misses += sum(pred is None for pred in preds)
        rows.extend(
            {"id": r["id"], "subject": subject, "extracted": pred, "random_guess": pred is None, "score": float(ok)}
            for r, pred, ok in zip(subject_records, preds, correct)
        )

    breakdown = {"by_subject": by_subject, "extraction_failures": misses}
    return rows, pct(r["score"] for r in rows), breakdown


if __name__ == "__main__":
    run_benchmark(sys.modules[__name__])
