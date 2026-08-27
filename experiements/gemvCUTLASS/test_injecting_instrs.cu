// harness.cu generated to test injecting ptx instrs
#include "cutlass/gemm/kernel/gemv_custom.h"                      // your edited kernel
#include "cutlass/gemm/device/gemv.h"                // launcher (unmodified)
#include "cutlass/epilogue/thread/linear_combination.h"
#include <cstdio>

using EpilogueOp = cutlass::epilogue::thread::LinearCombination<float, 1, float, float>;
using GemvKernel = cutlass::gemm::kernel::GemvCustom<
    float, cutlass::layout::RowMajor, float, float, float, EpilogueOp, 4>;
using Gemv = cutlass::gemm::device::Gemv<GemvKernel>;

int main() {
    int M = 3584, K = 3584, batch = 1;               // your model's shape
    float *A, *x, *y;
    cudaMalloc(&A, sizeof(float)*M*K);
    cudaMalloc(&x, sizeof(float)*batch*K);
    cudaMalloc(&y, sizeof(float)*batch*M);
    // fill A, x (cudaMemcpy from host arrays; include ~40% zeros in x
    // if your instruction's behavior is sparsity-dependent)

    typename EpilogueOp::Params ep(1.0f, 0.0f);
    typename Gemv::Arguments args({M, K}, batch, ep,
        {A, K}, x, y, y, 0, (int64_t)K, (int64_t)M, (int64_t)M);

    Gemv op;
    op.initialize(args);
    op.run();
    cudaDeviceSynchronize();

    float out[4]; cudaMemcpy(out, y, sizeof(out), cudaMemcpyDeviceToHost);
    printf("y[0..3] = %f %f %f %f\n", out[0], out[1], out[2], out[3]);
    return 0;
}   