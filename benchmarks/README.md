# benchmarks

Runners for **MATH-500**, **MMLU-Pro** and **LongBench v1 / LongBench-E** that follow the
official evaluation code as closely as possible. They sit behind a small model interface, so any
model can be evaluated: a stock Hugging Face checkpoint or your own implementation.

```
src/RunBenchmark/
  models.py            Model interface + HFModel (transformers) + quick single-prompt CLI
  probe.py             largest batch size that fits on this machine, per benchmark job
  evals/
    common.py          shared driver: CLI, resumable predictions, batching, result files
    math500.py
    mmlu_pro.py
    longbench.py       + longbench_metrics.py, longbench_prompts.json, longbench_maxlen.json
scripts/run_evals.sh   convenience wrapper
examples/              custom model classes for --model-class (see "Plugging in a custom model")
  fixed_answer.py      minimal Model subclass, no GPU
  pruned_hf.py         HFModel subclass that magnitude-prunes MLP weights
```

## Setup

```bash
cd benchmarks
uv sync                       # Python 3.13, torch 2.13 (cu132), transformers
```

Datasets download from the Hugging Face Hub on first use (`HuggingFaceH4/MATH-500`,
`TIGER-Lab/MMLU-Pro`, `THUDM/LongBench`).

## Quick start

```bash
scripts/run_evals.sh smoke                    # a few examples per benchmark
scripts/run_evals.sh probe                    # report max batch sizes, no generation
scripts/run_evals.sh math500                  # full runs: math500 | mmlu_pro | longbench | all
MODEL=Qwen/Qwen3-8B scripts/run_evals.sh mmlu_pro --subjects math,physics

# or call a benchmark directly
uv run python -m RunBenchmark.evals.longbench --model google/gemma-3-12b-it --tasks hotpotqa,lcc
uv run python -m RunBenchmark.models --model google/gemma-3-12b-it --prompt "Hello"
```

Common flags (all benchmarks):

| flag | default | meaning |
| --- | --- | --- |
| `--model` | required | model name/path, interpreted by the model class |
| `--model-class` | HF transformers | `pkg.module:Class`, a `Model` subclass |
| `--batch-size` | `auto` | integer, or `auto` to probe the largest batch that fits |
| `--max-batch-size` | 64 | cap for `auto` |
| `--limit` | none | first N examples (per subject for MMLU-Pro, per task for LongBench) |
| `--output-root` / `--output-dir` | `outputs` | results go to `<root>/<model>/<benchmark>/` unless `--output-dir` is set |
| `--score-only` | off | rescore saved predictions without loading the model |

`HFModel` flags: `--dtype` (bfloat16), `--device-map` (auto), `--cpu-memory` (off; set it to
allow weight offload to CPU), `--attn-implementation` (sdpa), `--prefill-chunk-size` (4096; 0
disables), `--trust-remote-code`. GPU memory is not capped: accelerate places the weights using
each GPU's free memory.

`run_evals.sh` reads `MODEL`, `MODEL_CLASS`, `MODEL_ARGS` (extra model flags) and `OUTPUT_ROOT`
from the environment.

## Batch size

With `--batch-size auto`, before generating each job (a benchmark, or one LongBench task) the
runner probes the largest batch that fits in GPU memory:

1. It takes the longest prompt of the job, `L`, and the job's `max_new_tokens`, `N`.
2. For a candidate batch `B`, it builds a dummy `B x (L + N)` input and runs the forward
   passes of a worst-case generation: the prompt is prefilled as `generate` would (chunked if
   long), then the KV cache is grown to `L + N` in 256-token chunks. This reaches roughly the
   peak memory of a full-length generation without decoding token by token.
3. `B` is doubled until it runs out of memory, then binary-searched between the last success
   and the failure. If the cap (`--max-batch-size` or the number of examples) fits, the cap is
   used; otherwise 90% of the maximum that fits.

If real generation still runs out of memory, the batch is halved and the smaller size is kept
for the rest of the job. The chosen sizes are saved in `result.json` under `config.batching`.
`scripts/run_evals.sh probe` (or `python -m RunBenchmark.probe --benchmark ...`) prints the
probed sizes without generating. A custom model class that does not implement
`probe_batch_size` runs with batch size 1.

Batched generation uses left padding, so outputs can differ slightly from batch size 1 because of
numerics. Use `--batch-size 1` for the closest match with the official single-sample loops.

## Output format

Every benchmark writes the same three files to `<output-root>/<model>/<benchmark>/`
(`mmlu_pro_chat` with `--chat`, `longbench_e` with `--e`):

- `predictions.jsonl` (LongBench: `predictions/<task>.jsonl`): one generation per line,
  `{"id", "reference", ...metadata, "output"}`. Runs resume from these files.
- `scores.jsonl`: one line per example, `{"id", "extracted", "score", ...}`. `score` is 0/1 for
  accuracy benchmarks and the raw 0..1 metric for LongBench.
- `result.json`:

```json
{
  "benchmark": "mmlu_pro",
  "model": "google/gemma-3-12b-it",
  "metric": "accuracy",
  "score": 64.29,
  "n": 14,
  "breakdown": {"by_subject": {"biology": 0.0, "...": 0.0}, "extraction_failures": 0},
  "config": {"args": {}, "generation": {}, "batching": {}},
  "timestamp": "2026-10-01T16:30:00-07:00"
}
```

Scores are percentages with 2 decimals.

| benchmark | `score` | `breakdown` |
| --- | --- | --- |
| math500 | accuracy | `by_level`, `by_subject` |
| mmlu_pro | accuracy | `by_subject`, `extraction_failures` |
| longbench | `{"en": avg, "zh": avg}` | `tasks`, `categories.en`, `categories.zh` |
| longbench_e | `{"0-4k", "4-8k", "8k+"}`: mean over the 13 tasks | `tasks` with per-bucket scores |

## Benchmarks

### MATH-500

- Data: `HuggingFaceH4/MATH-500`, the 500-problem MATH test subset from Lightman et al. (2023),
  "Let's Verify Step by Step".
- The benchmark defines problems and reference answers but no prompt or grader. This runner uses
  a common zero-shot setup: the problem followed by "Please reason step by step, and put your
  final answer within \boxed{}.", the chat template, greedy decoding, and 4096 new tokens.
- Grading: [math-verify](https://github.com/huggingface/Math-Verify) checks the parsed model
  answer against the reference for symbolic equivalence.

### MMLU-Pro

Follows [TIGER-AI-Lab/MMLU-Pro `evaluate_from_local.py`](https://github.com/TIGER-AI-Lab/MMLU-Pro):

- 5-shot chain-of-thought from the validation split of the same category, with
  `cot_prompt_lib/initial_prompt.txt` as the header. Shots are dropped one at a time until the
  prompt is shorter than `4096 - 2048` tokens.
- Greedy decoding, 2048 new tokens, stop at `Question:`.
- Answer extraction: `answer is (X)`, then `Answer: X`, then the last standalone letter A-J.
- If nothing can be extracted, a random guess is drawn from `random.seed(12345)`. Upstream
  scores each subject twice and so draws the guesses twice per subject, in sorted subject order.
  This runner replicates that. vLLM may also reseed Python's global RNG while loading the model,
  so random hits are not guaranteed to match an upstream run exactly. The number of examples
  affected is reported as `extraction_failures`.
- `--subjects` matches upstream: substring match, `_` equals space.
- Differences: generation goes through the model interface instead of vLLM. `--chat` (wrap the
  few-shot prompt in the chat template) is an optional extension; the official script sends the
  raw prompt.

### LongBench v1 / LongBench-E

Follows [THUDM/LongBench `pred.py`, `eval.py`, `metrics.py`](https://github.com/THUDM/LongBench/tree/main/LongBench):

- The prompts and generation lengths (`longbench_prompts.json`, `longbench_maxlen.json`) are
  identical to upstream `config/dataset2prompt.json` and `dataset2maxlen.json`, including the
  upstream typo "asconcisely".
- Prompts longer than `--max-length` tokens are truncated in the middle. Tokens are counted with
  special tokens, as upstream does. The default 31500 is upstream's `model2maxlen` value for
  32k-context models; raise it for longer-context models if you want.
- Chat template on every task except trec, triviaqa, samsum, lsht, lcc and repobench-p. Greedy
  decoding.
- samsum generates at least one token and also stops at a newline. Upstream stops at
  `[tokenizer.eos_token_id, "\n"]`. Here it stops at the checkpoint's generation-config EOS
  tokens plus `"\n"`, which for chat models also includes the end-of-turn token.
- Metrics are upstream's. trec, triviaqa, samsum and lsht are scored on the first line of the
  output. Each sample takes the best score over its references.
- `code_sim_score`: upstream uses `fuzzywuzzy`, whose requirements leave out
  `python-Levenshtein`, so `fuzz.ratio` runs on its `difflib` fallback (integer-rounded). This
  runner reimplements that fallback with the standard library.
- Aggregation follows the README leaderboard, with separate English and Chinese tables. Each
  table's average is the mean of six category scores. A category score is the mean of that
  language's tasks in the category, and lcc and repobench-p count for both languages. Upstream
  `eval.py` itself writes only per-task scores, which are under `breakdown.tasks`.
- LongBench-E (`--e`): per-task scores in the 0-4k / 4-8k / 8k+ buckets, by the dataset's
  `length` field, as upstream `scorer_e`. The top-level `score` (mean over tasks per bucket) is a
  convenience summary, not an upstream output.
- Data comes from the dataset repo's `data.zip`, which holds the same files that the
  `load_dataset("THUDM/LongBench", ...)` script reads. Script-based datasets no longer load
  with `datasets>=4`.

## Plugging in a custom model

Pass `--model-class module:Class`, where `Class` subclasses `RunBenchmark.models.Model`. The
runner imports `module` before parsing the rest of the command line, so the class's own flags
(from its `add_args`) are accepted. The module must be importable from the benchmarks
environment. Two working samples are in [`examples/`](examples); put `examples` on
`PYTHONPATH` to use them.

### What a model class implements

| method | required | used for |
| --- | --- | --- |
| `from_args(args)` (classmethod) | yes | build the model from parsed flags; `args.model` is the `--model` value |
| `add_args(parser)` (classmethod) | no | register extra flags |
| `num_tokens(text)` | yes | prompt length: MMLU-Pro shot trimming, batch ordering, auto batch size |
| `generate(prompts, params)` | yes | one continuation per prompt, without the prompt text |
| `format_prompt(text, chat)` | no | chat template; default returns `text` unchanged |
| `truncate_middle(text, max_tokens)` | LongBench only | keep the first and last `max_tokens // 2` tokens |
| `probe_batch_size(prompt_len, max_new_tokens, max_batch_size)` | no | `--batch-size auto`; without it the runner uses batch size 1 |
| `is_oom(exc)`, `free_memory()` | no | halve the batch on out-of-memory errors (defaults handle `torch.OutOfMemoryError`) |

`params` is a `GenerationParams` holding `max_new_tokens`, `temperature` (0 means greedy),
`stop_strings` (MMLU-Pro stops at `Question:`), `stop_at_newline` (LongBench samsum) and
`min_new_tokens`. A class that ignores a field changes the benchmark protocol.

### Sample 1: a minimal `Model` ([`examples/fixed_answer.py`](examples/fixed_answer.py))

This class returns the same text for every prompt and needs no weights or GPU. Use it to check
that a plugin is picked up and that the output files are written:

```bash
PYTHONPATH=examples uv run python -m RunBenchmark.evals.math500 \
    --model-class fixed_answer:FixedAnswerModel --model fixed-answer --limit 4
```

### Sample 2: modify a Hugging Face model ([`examples/pruned_hf.py`](examples/pruned_hf.py))

For a model that starts from a transformers checkpoint, subclass `HFModel`. Call
`super().from_args(args)` and then change `self.model`. Tokenization, chat templates, truncation,
generation and batch probing are inherited. The sample zeroes the smallest-magnitude fraction of
each row in the MLP projections:

```python
class MagnitudePrunedHFModel(HFModel):
    @classmethod
    def add_args(cls, parser):
        super().add_args(parser)          # keep --dtype, --device-map, ...
        parser.add_argument("--sparsity", type=float, default=0.5)
        parser.add_argument("--prune-pattern", default=r"\.mlp\.(gate_proj|up_proj|down_proj)$")
        return parser

    @classmethod
    def from_args(cls, args):
        self = super().from_args(args)    # loads args.model as HFModel does
        ...                               # prune matching nn.Linear weights in self.model
        return self
```

```bash
PYTHONPATH=examples uv run python -m RunBenchmark.evals.math500 \
    --model-class pruned_hf:MagnitudePrunedHFModel \
    --model google/gemma-3-12b-it --sparsity 0.5 \
    --output-dir outputs/gemma-3-12b-it-pruned0.5/math500
```

The pruned weights are still stored densely, so this measures accuracy under sparsity, not
speed. A sparse-GEMV implementation would replace the pruned `nn.Linear` modules with its own
kernels at the same point.

With `scripts/run_evals.sh`, pass the class and its flags through the environment:

```bash
PYTHONPATH=examples MODEL_CLASS=pruned_hf:MagnitudePrunedHFModel MODEL_ARGS="--sparsity 0.5" \
    OUTPUT_ROOT=outputs/pruned0.5 scripts/run_evals.sh smoke
```

### Output directory

`--model` names the output directory (`<output-root>/<model>/<benchmark>/`) and the `model`
field of `result.json`, whatever the model class. Predictions resume from that directory. A
modified model run with the same `--model` as the original checkpoint would reuse the
original's cached predictions, so give each variant its own `--output-dir` or `--output-root`.
The class's flags (for example `sparsity`) are recorded in `result.json` under `config.args`.

### Code from another uv project

Each uv project has its own `.venv`, and `uv run` here uses the benchmarks environment. A class
from another project works only if its code and all its dependencies are installed in this
environment, built against the same torch. The usual way is to make that project an installable
package and add it as a path dependency in `pyproject.toml`:

```toml
[project]
dependencies = [..., "my-kernels"]

[tool.uv.sources]
my-kernels = { path = "../my_kernels", editable = true }
```

Its version requirements must be compatible with the ones here (Python >= 3.13, `torch==2.13.0`).
Adding another project's `src` to `PYTHONPATH` works only when its dependencies happen to be
installed already.
