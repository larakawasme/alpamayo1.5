#!/usr/bin/env python3
"""Report the largest generation batch size that fits on this machine, per benchmark job.

Builds the benchmark's real prompts, then probes with its longest prompt plus
its max_new_tokens. No generation is run. Accepts the same flags as the benchmark.

    python -m RunBenchmark.probe --benchmark math500 --model google/gemma-3-12b-it
    python -m RunBenchmark.probe --benchmark longbench --tasks hotpotqa,lcc --model ...
"""

import argparse
import importlib
import sys

from RunBenchmark.evals.common import AUTO_BATCH_MARGIN, build_parser
from RunBenchmark.models import load_model

BENCHMARKS = ["math500", "mmlu_pro", "longbench"]


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    pre = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    pre.add_argument("--benchmark", choices=BENCHMARKS, required="-h" not in argv and "--help" not in argv)
    known, _ = pre.parse_known_args(argv)
    bench = importlib.import_module(f"RunBenchmark.evals.{known.benchmark or 'math500'}")

    parser = build_parser(bench, argv)
    parser.description = __doc__
    parser.add_argument("--benchmark", choices=BENCHMARKS, required=True)
    args = parser.parse_args(argv)

    model = load_model(args)
    rows = []
    for job in bench.load_jobs(args):
        prompts = [bench.build_prompt(model, args, job, ex)["prompt"] for ex in job.examples]
        longest = max(model.num_tokens(p) for p in prompts)
        cap = max(1, min(args.max_batch_size, len(prompts)))
        found = model.probe_batch_size(longest, job.params.max_new_tokens, cap)
        if found is None:
            chosen = 1
        elif found >= cap:
            chosen = cap
        else:
            chosen = max(1, int(found * AUTO_BATCH_MARGIN)) if found else 0
        rows.append((job.name, len(prompts), longest, job.params.max_new_tokens, cap, found, chosen))
        print(f"[{job.name}] max fitting batch {found} (cap {cap}) -> auto uses {chosen}", flush=True)

    print(f"\n{'job':<22}{'n':>6}{'longest':>9}{'new':>6}{'cap':>6}{'max_fit':>9}{'auto':>6}")
    for name, n, longest, new, cap, found, chosen in rows:
        print(f"{name:<22}{n:>6}{longest:>9}{new:>6}{cap:>6}{str(found):>9}{chosen:>6}")
    print("max_fit == cap means the cap was reached without running out of memory.")


if __name__ == "__main__":
    main()
