"""Shared benchmark driver: CLI flags, resumable predictions, batching, unified result files.

A benchmark module defines:
    NAME, METRIC                      result.json "benchmark" / "metric"
    add_args(parser)                  benchmark flags
    output_name(args) -> str          optional; output sub-directory (default NAME)
    load_jobs(args) -> [Job]          examples to evaluate; no model needed
    build_prompt(model, args, job, example) -> {"prompt": str, ...extra record fields}
    score(args, jobs, records) -> (score_rows, score, breakdown)
and calls `run_benchmark(module)`.
"""

import argparse
import dataclasses
import json
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from tqdm import tqdm

from RunBenchmark.models import GenerationParams, add_model_args, load_model, model_class_from_argv

# Fraction of the probed maximum batch size actually used, as headroom for
# allocator fragmentation and prompts whose padding differs from the probe.
AUTO_BATCH_MARGIN = 0.9


@dataclass
class Job:
    """One generation pass: a benchmark, or one task of a multi-task benchmark."""

    name: str
    examples: list
    params: GenerationParams
    pred_file: str
    context: dict = field(default_factory=dict)


def batch_size_arg(value):
    if value == "auto":
        return value
    size = int(value)
    if size < 1:
        raise argparse.ArgumentTypeError("batch size must be >= 1 or 'auto'")
    return size


def add_run_args(parser):
    parser.add_argument(
        "--batch-size",
        type=batch_size_arg,
        default="auto",
        help="Generation batch size, or 'auto' to probe the largest one that fits in GPU memory.",
    )
    parser.add_argument("--max-batch-size", type=int, default=64, help="Upper bound for --batch-size auto.")
    parser.add_argument("--limit", type=int, default=None, help="Only evaluate the first N examples (per task/subject).")
    parser.add_argument("--output-root", type=Path, default=Path("outputs"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Default: <output-root>/<model>/<benchmark>.",
    )
    parser.add_argument("--score-only", action="store_true", help="Skip generation; rescore existing predictions.")
    return parser


def build_parser(bench, argv=None):
    parser = argparse.ArgumentParser(
        description=bench.__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    add_model_args(parser)
    model_class_from_argv(argv).add_args(parser)
    add_run_args(parser)
    bench.add_args(parser)
    return parser


def model_slug(model):
    name = model.rstrip("/")
    if Path(name).exists():
        return Path(name).name
    return name.replace("/", "__")


def output_dir(args, bench):
    if args.output_dir is not None:
        return args.output_dir
    name = bench.output_name(args) if hasattr(bench, "output_name") else bench.NAME
    return args.output_root / model_slug(args.model) / name


def read_jsonl(path):
    path = Path(path)
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def append_jsonl(path, records):
    with Path(path).open("a", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_jsonl(path, records):
    Path(path).write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records), encoding="utf-8"
    )


def write_json(path, obj):
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def pct(values):
    """Mean of 0..1 scores as a percentage rounded to 2 decimals; None if empty."""
    values = list(values)
    return round(100 * sum(values) / len(values), 2) if values else None


def auto_batch_size(model, prompt_len, max_new_tokens, n_items, max_batch_size):
    cap = max(1, min(max_batch_size, n_items))
    if cap == 1:
        return 1
    found = model.probe_batch_size(prompt_len, max_new_tokens, cap)
    if found is None:
        print(f"[batch] {type(model).__name__} has no probe_batch_size; using batch size 1", flush=True)
        return 1
    if found == 0:
        raise RuntimeError(f"Batch size 1 does not fit {prompt_len}+{max_new_tokens} tokens in GPU memory.")
    if found >= cap:
        return cap
    return max(1, int(found * AUTO_BATCH_MARGIN))


def prediction_record(item, output):
    record = {k: v for k, v in item.items() if k != "prompt" and not k.startswith("_")}
    record["output"] = output
    return record


def run_generation(model, items, path, params, args, desc):
    """Generate for items lacking a prediction in `path`, appending as it goes.

    Each item needs "id" and "prompt"; keys starting with "_" are not saved.
    Returns (records in item order, batching info).
    """
    path = Path(path)
    done = {record["id"]: record for record in read_jsonl(path)}
    todo = [item for item in items if item["id"] not in done]
    info = {"cached": len(items) - len(todo), "generated": len(todo)}
    print(f"[{desc}] {info['cached']} cached, {len(todo)} to generate", flush=True)

    if todo:
        lengths = {item["id"]: model.num_tokens(item["prompt"]) for item in todo}
        # Longest first: an OOM surfaces immediately, and padding stays small.
        todo.sort(key=lambda item: lengths[item["id"]], reverse=True)
        if args.batch_size == "auto":
            batch_size = auto_batch_size(
                model, lengths[todo[0]["id"]], params.max_new_tokens, len(todo), args.max_batch_size
            )
        else:
            batch_size = args.batch_size
        info["batch_size"] = batch_size
        print(f"[{desc}] batch size {batch_size} (longest prompt {lengths[todo[0]['id']]} tokens)", flush=True)

        start = time.time()
        i = 0
        with tqdm(total=len(todo), desc=desc) as bar:
            while i < len(todo):
                batch = todo[i:i + batch_size]
                oom = False
                try:
                    outputs = model.generate([item["prompt"] for item in batch], params)
                except Exception as exc:
                    if not model.is_oom(exc) or len(batch) == 1:
                        raise
                    oom = True
                if oom:
                    model.free_memory()
                    batch_size = max(1, len(batch) // 2)
                    print(f"\n[{desc}] OOM at batch size {len(batch)}; continuing with {batch_size}", flush=True)
                    continue
                records = [prediction_record(item, output) for item, output in zip(batch, outputs)]
                append_jsonl(path, records)
                done.update((record["id"], record) for record in records)
                i += len(batch)
                bar.update(len(batch))
        info["final_batch_size"] = batch_size
        info["seconds"] = round(time.time() - start, 1)

    return [done[item["id"]] for item in items], info


def jsonable(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {k: jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    return value


def print_result(result):
    print(f"\n=== {result['benchmark']} ({result['model']}) ===")
    print(f"{result['metric']}: {json.dumps(result['score'], ensure_ascii=False)}  (n={result['n']})")
    print(json.dumps(result["breakdown"], ensure_ascii=False, indent=2))


def run_benchmark(bench, argv=None):
    args = build_parser(bench, argv).parse_args(argv)
    out_dir = output_dir(args, bench)
    out_dir.mkdir(parents=True, exist_ok=True)
    jobs = bench.load_jobs(args)

    model = None if args.score_only else load_model(args)
    records, batching = {}, {}
    for job in jobs:
        path = out_dir / job.pred_file
        path.parent.mkdir(parents=True, exist_ok=True)
        if args.score_only:
            cached = {r["id"]: r for r in read_jsonl(path)}
            missing = sum(ex["id"] not in cached for ex in job.examples)
            if missing:
                print(f"[{job.name}] {missing} examples have no prediction; scoring the rest", flush=True)
            records[job.name] = [cached[ex["id"]] for ex in job.examples if ex["id"] in cached]
        else:
            items = [{**ex, **bench.build_prompt(model, args, job, ex)} for ex in job.examples]
            records[job.name], batching[job.name] = run_generation(
                model, items, path, job.params, args, desc=job.name
            )

    score_rows, score, breakdown = bench.score(args, jobs, records)
    write_jsonl(out_dir / "scores.jsonl", score_rows)

    result_path = out_dir / "result.json"
    if args.score_only and result_path.exists():
        batching = json.loads(result_path.read_text(encoding="utf-8")).get("config", {}).get("batching", {})
    result = {
        "benchmark": bench.NAME,
        "model": args.model,
        "metric": bench.METRIC,
        "score": score,
        "n": len(score_rows),
        "breakdown": breakdown,
        "config": {
            "args": jsonable(vars(args)),
            "generation": {job.name: dataclasses.asdict(job.params) for job in jobs},
            "batching": batching,
        },
        "timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
    }
    write_json(result_path, result)
    print_result(result)
    return result
