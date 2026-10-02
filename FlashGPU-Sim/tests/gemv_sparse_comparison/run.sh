#!/usr/bin/env bash
set -eo pipefail
TASK_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$TASK_DIR/../.." && pwd)"
export CUDA_INSTALL_PATH="${CUDA_INSTALL_PATH:-/usr/local/cuda-12.8}"
mkdir -p "$TASK_DIR/build" "$TASK_DIR/results"
set +e
source "$REPO_ROOT/setup_environment" > "$TASK_DIR/build/setup.log" 2>&1
SETUP_STATUS=$?
set -e
if [[ "$SETUP_STATUS" != 0 ]]; then
  cat "$TASK_DIR/build/setup.log" >&2
  exit "$SETUP_STATUS"
fi
set -u
SIM_LIB="$REPO_ROOT/lib/$GPGPUSIM_CONFIG"
if [[ ! -f "$SIM_LIB/libcudart.so" ]]; then
  echo 'Build the simulator first: source setup_environment && make -j8' >&2
  exit 1
fi
if [[ "${GEMV_SKIP_BUILD:-0}" != 1 ]]; then
  for PROGRAM in original compare; do
    SOURCE="$TASK_DIR/compare.cu"
    [[ "$PROGRAM" == original ]] && SOURCE="$TASK_DIR/original/run/gemv_standalone.cu"
    "$CUDA_INSTALL_PATH/bin/nvcc" -O2 -std=c++17 -cudart shared \
      -gencode arch=compute_120a,code=sm_120a -gencode arch=compute_120a,code=compute_120a \
      -I"$REPO_ROOT/src" -I"$REPO_ROOT/tests/src/include" \
      "$SOURCE" -L"$CUDA_INSTALL_PATH/lib64" -L"$CUDA_INSTALL_PATH/lib64/stubs" -lcudart -lcuda -o "$TASK_DIR/build/$PROGRAM"
  done
fi
MODE="${1:-compare}"
shift || true
[[ "$MODE" == original || "$MODE" == compare ]] || { echo 'Use original or compare' >&2; exit 1; }
CASE_TAG="${MODE}"
for ARG in "$@"; do CASE_TAG+="_${ARG}"; done
RUN_DIR="$TASK_DIR/results/$CASE_TAG"
mkdir -p "$RUN_DIR"
cp -a "$REPO_ROOT/configs/SM120_RTX5090/." "$RUN_DIR/"
ldd "$TASK_DIR/build/$MODE" > "$RUN_DIR/linkage.txt"
if ! grep -Fq "$SIM_LIB/libcudart" "$RUN_DIR/linkage.txt"; then
  echo 'Executable does not resolve to this simulator runtime' >&2; exit 1
fi
cd "$RUN_DIR"
# Always exercise the detailed simulator, including sparse TMA completion.
export PTX_SIM_MODE_FUNC=0
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
STATUS=0
"$TASK_DIR/build/$MODE" "$@" > simulation.log 2>&1 || STATUS=$?
rg '^(CASE |PASS|TRANSFER|RESULT|FAIL|Mismatch)' simulation.log || true
echo "Simulation log: $RUN_DIR/simulation.log"
exit "$STATUS"
