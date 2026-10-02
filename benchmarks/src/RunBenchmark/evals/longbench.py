#!/usr/bin/env python3
"""LongBench v1 (Bai et al., 2023, arXiv:2308.14508).

Follows THUDM/LongBench pred.py + eval.py: per-task prompts and generation
lengths from config/dataset2prompt.json and dataset2maxlen.json, middle
truncation of the formatted prompt to --max-length tokens, chat template on
all tasks except trec/triviaqa/samsum/lsht/lcc/repobench-p, greedy decoding,
samsum additionally stops at a newline, and the official per-task metrics.

Aggregates follow the README leaderboard: the English and Chinese averages
are each the mean of six category scores, a category score is the mean of its
tasks in that language, and the two code tasks count towards both.
LongBench-E (--e) reports per-task scores in the 0-4k / 4-8k / 8k+ buckets.

Data comes from the dataset repo's data.zip, which holds the same files the
official load_dataset("THUDM/LongBench", ...) script reads.
"""

import json
import sys
import zipfile
from pathlib import Path

from huggingface_hub import hf_hub_download

from RunBenchmark.evals.common import Job, pct, read_jsonl, run_benchmark
from RunBenchmark.evals.longbench_metrics import (
    DATASET2METRIC,
    code_sim_score,
    first_code_line,
    sample_score,
    scored_prediction,
)
from RunBenchmark.models import GenerationParams

NAME = "longbench"
METRIC = "longbench_avg"

HERE = Path(__file__).parent
DATASET2PROMPT = json.loads((HERE / "longbench_prompts.json").read_text(encoding="utf-8"))
DATASET2MAXLEN = json.loads((HERE / "longbench_maxlen.json").read_text(encoding="utf-8"))

ALL_TASKS = [
    "narrativeqa", "qasper", "multifieldqa_en", "multifieldqa_zh", "hotpotqa", "2wikimqa", "musique",
    "dureader", "gov_report", "qmsum", "multi_news", "vcsum", "trec", "triviaqa", "samsum", "lsht",
    "passage_count", "passage_retrieval_en", "passage_retrieval_zh", "lcc", "repobench-p",
]
E_TASKS = [
    "qasper", "multifieldqa_en", "hotpotqa", "2wikimqa", "gov_report", "multi_news",
    "trec", "triviaqa", "samsum", "passage_count", "passage_retrieval_en", "lcc", "repobench-p",
]
CATEGORIES = {
    "en": {
        "Single-Doc QA": ["narrativeqa", "qasper", "multifieldqa_en"],
        "Multi-Doc QA": ["hotpotqa", "2wikimqa", "musique"],
        "Summarization": ["gov_report", "qmsum", "multi_news"],
        "Few-shot Learning": ["trec", "triviaqa", "samsum"],
        "Synthetic Tasks": ["passage_count", "passage_retrieval_en"],
        "Code Completion": ["lcc", "repobench-p"],
    },
    "zh": {
        "Single-Doc QA": ["multifieldqa_zh"],
        "Multi-Doc QA": ["dureader"],
        "Summarization": ["vcsum"],
        "Few-shot Learning": ["lsht"],
        "Synthetic Tasks": ["passage_retrieval_zh"],
        "Code Completion": ["lcc", "repobench-p"],
    },
}
NO_CHAT_TASKS = {"trec", "triviaqa", "samsum", "lsht", "lcc", "repobench-p"}
BUCKETS = ["0-4k", "4-8k", "8k+"]


def add_args(parser):
    parser.add_argument(
        "--max-length",
        type=int,
        default=31500,
        help="Prompt token budget; longer prompts are truncated in the middle. "
        "31500 is upstream's model2maxlen value for 32k-context models.",
    )
    parser.add_argument(
        "--tasks",
        default="all",
        help='Comma-separated task names, e.g. "hotpotqa,lcc". Default: all 21 (13 with --e).',
    )
    parser.add_argument("--e", action="store_true", help="Evaluate LongBench-E instead.")


def output_name(args):
    return "longbench_e" if args.e else NAME


def data_dir():
    zip_path = Path(hf_hub_download("THUDM/LongBench", "data.zip", repo_type="dataset"))
    out = zip_path.parent / "longbench_data"
    if not out.exists():
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(out)
    return out


def selected_tasks(args):
    tasks = E_TASKS if args.e else ALL_TASKS
    if args.tasks == "all":
        return tasks
    wanted = [t.strip() for t in args.tasks.split(",")]
    unknown = set(wanted) - set(tasks)
    if unknown:
        raise SystemExit(f"Unknown tasks {sorted(unknown)}; choose from {tasks}")
    return wanted


def load_jobs(args):
    root = data_dir()
    jobs = []
    for task in selected_tasks(args):
        [path] = root.rglob(f"{task}_e.jsonl" if args.e else f"{task}.jsonl")
        rows = read_jsonl(path)
        if args.limit is not None:
            rows = rows[:args.limit]
        examples = [
            {
                "id": row["_id"],
                "reference": row["answers"],
                "all_classes": row["all_classes"],
                "length": row["length"],
                "_row": row,
            }
            for row in rows
        ]
        params = GenerationParams(max_new_tokens=DATASET2MAXLEN[task])
        if task == "samsum":
            # Upstream: stops the model endlessly repeating "\nDialogue".
            params.min_new_tokens = 1
            params.stop_at_newline = True
        jobs.append(Job(task, examples, params, f"predictions/{task}.jsonl"))
    return jobs


def build_prompt(model, args, job, example):
    prompt = model.truncate_middle(DATASET2PROMPT[job.name].format(**example["_row"]), args.max_length)
    return {"prompt": model.format_prompt(prompt, chat=job.name not in NO_CHAT_TASKS)}


def length_bucket(length):
    if length < 4000:
        return "0-4k"
    if length < 8000:
        return "4-8k"
    return "8k+"


def mean2(values):
    values = list(values)
    return round(sum(values) / len(values), 2) if values else None


def language_scores(tasks):
    """Per-language category scores and averages; a category needs all its tasks."""
    categories, averages = {}, {}
    for lang, groups in CATEGORIES.items():
        cats = {
            name: mean2(tasks[t] for t in members)
            for name, members in groups.items()
            if all(tasks.get(t) is not None for t in members)
        }
        categories[lang] = cats
        averages[lang] = mean2(cats.values()) if len(cats) == len(groups) else None
    return categories, averages


def score(args, jobs, records):
    rows, tasks = [], {}
    for job in jobs:
        task_rows = []
        for record in records[job.name]:
            output = record["output"]
            extracted = first_code_line(output) if DATASET2METRIC[job.name] is code_sim_score else (
                scored_prediction(job.name, output)
            )
            value = sample_score(job.name, output, record["reference"], record["all_classes"])
            task_rows.append({"id": record["id"], "task": job.name, "extracted": extracted, "score": value})
            if args.e:
                task_rows[-1]["length_bucket"] = length_bucket(record["length"])
        rows.extend(task_rows)
        if args.e:
            tasks[job.name] = {
                b: pct(r["score"] for r in task_rows if r["length_bucket"] == b) for b in BUCKETS
            }
        else:
            tasks[job.name] = pct(r["score"] for r in task_rows)

    if args.e:
        complete = all(t in tasks for t in E_TASKS)
        overall = {
            b: mean2(tasks[t][b] for t in E_TASKS if tasks[t][b] is not None) for b in BUCKETS
        } if complete else None
        return rows, overall, {"tasks": tasks}

    categories, averages = language_scores(tasks)
    score_value = averages if any(v is not None for v in averages.values()) else None
    return rows, score_value, {"tasks": tasks, "categories": categories}


if __name__ == "__main__":
    run_benchmark(sys.modules[__name__])
