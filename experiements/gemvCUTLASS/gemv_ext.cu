// gemv_ext.cu - pytroch cuda extrantion that wraps cutlass gemv kerne. uses kernel/gemv.h and device/gemv.h
// written with generative LLM , Opus 4.8

#include <torch/extension.h>

#include "cutlass/cutlass.h"
#include "cutlass/matrix_coord.h" // gives you MatrixCoord, which is just a pair of (rows, cols) to describe problem size
#include "cutlass/tensor_ref.h" //gives you TensorRef, which is how CUTLASS points at a matrix in GPU memory (pointer + stride)
#include "cutlass/gemm/kernel/gemv.h" //the GPU kernel that runs on the GPU threads
#include "cutlass/gemm/device/gemv.h" //host-side launcher that sets up and fires that kernel from CPU code
#include "cutlass/epilogue/thread/linear_combination.h"   //applies alpha * result + beta * C

// ─── Type definitions ─────────────────────────────────────────────────────────

using ElementInput       = cutlass::bfloat16_t; //dtype of matrix A and vector x coming in
using ElementOutput      = cutlass::bfloat16_t; //output dtype vector y
using ElementAccumulator = float; //dtype used for the running sum during the dot product. 
using LayoutA            = cutlass::layout::RowMajor; //describes how the matrix is stored in memory. RowMajor means row by row, left to right — which is the default in PyTorch (torch.randn(M, K) is row major)

// kElementsPerAccess: how many floats loaded per vectorized memory op
// must divide K evenly — 8 is good for b16
static constexpr int kElementsPerAccess = 8; //load instruction loads 4x32 bits = 128 bits

// EpilogueOutputOp: applies alpha * acc + beta * C
// second template param is elements per output access — use 1 here (scalar epilogue)
using EpilogueOp = cutlass::epilogue::thread::LinearCombination<
    ElementOutput,
    1,  //because gemv
    ElementAccumulator,
    ElementAccumulator
>;

using GemvKernel = cutlass::gemm::kernel::Gemv<
    ElementInput,         // ElementA (matrix)
    LayoutA,              // LayoutA  (RowMajor)
    ElementInput,         // ElementB (vector x)
    ElementOutput,        // ElementC (output y)
    ElementAccumulator,
    EpilogueOp,
    kElementsPerAccess
>;

using Gemv = cutlass::gemm::device::Gemv<GemvKernel>;

// ─── Launch function ──────────────────────────────────────────────────────────

// weight: [M, K]   nn.Linear weight matrix
// x:      [..., K] input vector (we handle batch dims)
// returns [..., M]
torch::Tensor cutlass_gemv(torch::Tensor weight, torch::Tensor x) {

    // ── checks ────────────────────────────────────────────────────────────
    TORCH_CHECK(weight.is_cuda() && x.is_cuda(), "tensors must be on CUDA");
    TORCH_CHECK(weight.dtype() == torch::kBFloat16, "weight must be b16");
    TORCH_CHECK(x.dtype()      == torch::kBFloat16, "x must be b16");
    TORCH_CHECK(weight.is_contiguous(), "weight must be contiguous");

    int M = weight.size(0);
    int K = weight.size(1);
    TORCH_CHECK(x.size(-1) == K, "x last dim must match K=", K);
    TORCH_CHECK(K % kElementsPerAccess == 0,
        "K=", K, " must be divisible by kElementsPerAccess=", kElementsPerAccess);

        // ── flatten batch dims ────────────────────────────────────────────────
    // kernel handles one (M, K) x (K,) at a time via batch_count
    // we flatten everything except last dim into batch
    auto x_contig = x.contiguous();
    auto batch_shape = x_contig.sizes().vec();  // save shape as vector for output reshape

    int batch = x_contig.numel() / K;           // total number of vectors
    auto x_flat = x_contig.reshape({batch, K}); // [batch, K]
    auto y_flat = torch::zeros({batch, M}, x_contig.options()); // [batch, M]

    // ── TensorRef for A (weight matrix, shared across all batches) ────────
    // TensorRef<Element, Layout>(pointer, stride)
    // RowMajor stride for [M x K] matrix = K (elements per row)
    cutlass::TensorRef<ElementInput, LayoutA> ref_A(
        reinterpret_cast<ElementInput*>(weight.data_ptr<at::BFloat16>()),
        LayoutA(K)   // stride = K
    );

    // ── Problem size: MatrixCoord(rows, cols) = (M, K) ───────────────────
    // from the kernel: problem_size.row() = M, problem_size.column() = K
    cutlass::MatrixCoord problem_size(M, K);

    // ── Arguments — matching the real RowMajor constructor exactly ────────
    // Arguments(problem_size, batch_count, output_op, ref_A,
    //           ptr_B, ptr_C, ptr_D,
    //           batch_stride_A, batch_stride_B, batch_stride_C, batch_stride_D)
    //
    // batch_stride_A = 0  (weight is shared across all batches, don't advance it)
    // batch_stride_B = K  (each input vector is K floats apart)
    // batch_stride_C = M  (each output vector is M floats apart)
    // batch_stride_D = M

    typename EpilogueOp::Params epilogue_params(
        ElementAccumulator(1.0f),  // alpha
        ElementAccumulator(0.0f)   // beta (no add from C)
    );

    typename Gemv::Arguments args(
    problem_size,
    batch,                              // batch_count
    epilogue_params,                    // {alpha, beta}
    ref_A,                              // matrix A (weight)
    reinterpret_cast<ElementInput*>(x_flat.data_ptr<at::BFloat16>()),   // ptr_B(input vectors)
    reinterpret_cast<ElementOutput*>(y_flat.data_ptr<at::BFloat16>()),  // ptr_C(source, unused when beta=0)
    reinterpret_cast<ElementOutput*>(y_flat.data_ptr<at::BFloat16>()),  // ptr_D(output)
    0,                                  // batch_stride_A (weight shared)
    (int64_t)K,                         // batch_stride_B
    (int64_t)M,                         // batch_stride_C
    (int64_t)M                          // batch_stride_D
);

    // ── Initialize and run ────────────────────────────────────────────────
    Gemv gemv_op;

    cutlass::Status status = gemv_op.can_implement(args);
    TORCH_CHECK(status == cutlass::Status::kSuccess,
        "CUTLASS GEMV can_implement failed: ", cutlassGetStatusString(status));

    status = gemv_op.initialize(args); // copies args into the op's internal params, allocates any workspace (GEMV needs none), computes the launch grid.
    TORCH_CHECK(status == cutlass::Status::kSuccess,
        "CUTLASS GEMV initialize failed: ", cutlassGetStatusString(status));

    status = gemv_op.run(); // launches the CUDA kernel. N
    TORCH_CHECK(status == cutlass::Status::kSuccess,
        "CUTLASS GEMV run failed: ", cutlassGetStatusString(status));

    // ── Restore batch dims ────────────────────────────────────────────────
    batch_shape.back() = M;
    return y_flat.reshape(batch_shape);
}

// ─── pybind11 module ──────────────────────────────────────────────────────────

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("gemv", &cutlass_gemv,
        "CUTLASS GEMV: y = x @ weight.T\n"
        "  weight: [M, K] bf16 CUDA contiguous\n"
        "  x:      [..., K] bf16 CUDA\n"
        "  return: [..., M] bf16 CUDA");
}