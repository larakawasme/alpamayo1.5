# Indexed sparse GEMV versus sparse TMA

`original/rough_topk_gemv_sparse_improved_sim_modified.cu` is a byte-identical
copy of the requested `gemv_sparse_ext_no_malloc.cu`, renamed in this workspace.
`original/run/gemv_standalone.cu` copies the requested standalone harness with
only its include path updated for that rename. Sources came from
`/home/lara/gpgpu-sim_distribution-funtional/tests/gemv_kernels/`.
The standalone copy runs its original all-ones smoke test.

`compare.cu` includes the original kernel source with `GEMV_STANDALONE`, so the
indexed baseline and rough-top-k kernels are the original implementations.
It generates signed random BF16 weights and activations on the CPU, runs the
original histogram/cutoff/collection kernels in the simulator, checks that the
selected indices are unique and their values match the input, and runs the
original indexed GEMV without modifying its selection order.

The harness converts that exact selection to a bitmap on the host and encodes a
256-byte sparse tensor descriptor for the unchanged `[K, M]` weight layout.
Dimension 0 is contiguous M; sparse dimension 1 is K. `sparse_tma_gemv` loads
128-by-16 BF16 tiles with `cp.async.bulk.tensor.2d` and an mbarrier expecting the
full 4096-byte tile. The simulator transfers selected rows and zero-fills skipped
rows. Threads multiply regular shared-memory positions by the original vector;
they do not gather weight addresses using an index list. Split-K partitions tile
ranges, and FP32 atomics combine the partial sums.

The descriptor does not compact rows: the TMA kernel still visits the full K
range and computes on zero-filled rows. This implementation validates sparse
transfer correctness; it is not a performance-optimized GEMV. Bitmap construction
and descriptor encoding are host-side setup, not an end-to-end GPU top-k-to-bitmap
pipeline. The sparse descriptor ABI is simulator-only.

A transfer audit additionally checks the exact BF16 bits for the complete first
weight column, including all zero-filled rows, and fails on any mismatch.

Every output is checked three ways: original versus TMA, original versus an
independent FP64 CPU sum over the actual selected BF16 inputs, and TMA versus
that CPU sum. Nonfinite results fail. The tolerance is
`0.001 + 0.0001 * abs(cpu_reference)` because the two FP32 reductions accumulate
in different orders. The program prints maximum absolute errors and returns
nonzero on any mismatch or CUDA error. This compares the sparse selected-vector
product, not the dense product using all vector entries.

## Build and run

From the repository root, build FlashGPU-Sim once:

```bash
export CUDA_INSTALL_PATH=/usr/local/cuda-12.8
source setup_environment
make -j8
```

Then:

```bash
./tests/gemv_sparse_comparison/run.sh original
./tests/gemv_sparse_comparison/run.sh compare 512 256 32 2 42
./tests/gemv_sparse_comparison/run.sh compare 4096 4096 512 16 20261002
```

Arguments after `compare` are `K M keep_count split_k seed`; the large case is
the default. M must be divisible by 8 for the BF16 tensor-map row-stride alignment.
The default matrix contains 16,777,216 BF16 elements (32 MiB), with 512 of 4096
activations selected. CUDA builds target SM120a, and runs use the repository's
SM120_RTX5090 configuration in detailed timing mode (`PTX_SIM_MODE_FUNC=0`).
The runner verifies dynamic linkage to this checkout's simulator and defaults
`OMP_NUM_THREADS` to 1. Both programs are compiled before each run unless `GEMV_SKIP_BUILD=1` is set
to reuse an existing build (useful for concurrent runs).

Run logs and a record of library linkage are under
`results/<program>_<arguments>/`; generated files are git-ignored.
See `VALIDATION.md` for recorded outcomes.
