#!/usr/bin/env bash
# Benchmark runner: MATH-500, MMLU-Pro, LongBench.
#
#   scripts/run_evals.sh smoke        # a few examples per benchmark
#   scripts/run_evals.sh probe        # largest batch size per benchmark on this machine
#   scripts/run_evals.sh math500      # full MATH-500 (500 problems)
#   scripts/run_evals.sh mmlu_pro     # full MMLU-Pro (12,032 questions)
#   scripts/run_evals.sh longbench    # full LongBench v1 (21 tasks, 4,750 samples)
#   scripts/run_evals.sh all          # the three full runs in sequence
#
# Extra args are forwarded, e.g.
#   scripts/run_evals.sh math500 --batch-size 16
#   scripts/run_evals.sh longbench --tasks hotpotqa,lcc --max-length 16000
#   scripts/run_evals.sh mmlu_pro --subjects math,physics --chat
#
# Environment:
#   MODEL        model name or path (default: google/gemma-3-12b-it)
#   MODEL_CLASS  optional pkg.module:Class implementing RunBenchmark.models.Model
#   MODEL_ARGS   extra model flags, e.g. "--device-map balanced --dtype float16"
#   OUTPUT_ROOT  results root (default: outputs; smoke: outputs/smoke)
#
# Results go to $OUTPUT_ROOT/<model>/<benchmark>/{predictions.jsonl,scores.jsonl,result.json}
# and logs to $OUTPUT_ROOT/<model>/logs/. Runs resume from existing predictions;
# rescore with --score-only.

set -euo pipefail
cd "$(dirname "$0")/.."

MODEL="${MODEL:-google/gemma-3-12b-it}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
read -r -a EXTRA_MODEL_ARGS <<< "${MODEL_ARGS:-}"

target="${1:-smoke}"
shift || true

if [[ "$target" == smoke ]]; then
    OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/smoke}"
else
    OUTPUT_ROOT="${OUTPUT_ROOT:-outputs}"
fi

if [[ -d "$MODEL" ]]; then
    slug="$(basename "${MODEL%/}")"
else
    slug="${MODEL//\//__}"
fi
LOG_DIR="$OUTPUT_ROOT/$slug/logs"
mkdir -p "$LOG_DIR"

MODEL_FLAGS=(--model "$MODEL" "${EXTRA_MODEL_ARGS[@]}")
if [[ -n "${MODEL_CLASS:-}" ]]; then
    MODEL_FLAGS+=(--model-class "$MODEL_CLASS")
fi

run() {
    local bench="$1"; shift
    local log="$LOG_DIR/${bench}.log"
    echo ">>> ${bench} $*  (log: ${log})"
    uv run python -m "RunBenchmark.evals.${bench}" "${MODEL_FLAGS[@]}" \
        --output-root "$OUTPUT_ROOT" "$@" 2>&1 | tee -a "$log"
}

probe() {
    local bench="$1"; shift
    local log="$LOG_DIR/probe_${bench}.log"
    echo ">>> probe ${bench} $*  (log: ${log})"
    uv run python -m RunBenchmark.probe --benchmark "$bench" "${MODEL_FLAGS[@]}" "$@" 2>&1 | tee -a "$log"
}

case "$target" in
    smoke)
        run math500   --limit 8 "$@"
        run mmlu_pro  --limit 1 "$@"
        run longbench --limit 1 "$@"
        ;;
    probe)
        probe math500 "$@"
        probe mmlu_pro "$@"
        probe longbench "$@"
        ;;
    math500|mmlu_pro|longbench)
        run "$target" "$@"
        ;;
    all)
        run math500 "$@"
        run mmlu_pro "$@"
        run longbench "$@"
        ;;
    *)
        echo "usage: $0 {smoke|probe|math500|mmlu_pro|longbench|all} [extra args]" >&2
        exit 2
        ;;
esac
