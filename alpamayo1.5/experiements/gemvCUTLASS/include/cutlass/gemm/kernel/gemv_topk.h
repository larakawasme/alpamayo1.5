#pragma once

#include "cutlass/cutlass.h"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/gemm/gemm.h"
#include "cutlass/layout/matrix.h"
#include "cutlass/tensor_ref.h"

namespace cutlass {
namespace gemm {
namespace kernel {

// Indexed top-k GEMV using CUTLASS's cooperative threads-per-output pattern.
// A is physically [K, M] row-major, where A[k, m] == original_weight[m, k].
template <
    typename ElementA_,
    typename ElementB_,
    typename ElementC_,
    typename ElementAccumulator_,
    typename EpilogueOutputOp_,
    int kThreadCount_ = 128,
    int kThreadsPerRow_ = 16>
struct GemvTopK {
  using ElementA = ElementA_;
  using LayoutA = layout::RowMajor;
  using TensorRefA = TensorRef<ElementA, LayoutA>;
  using ElementB = ElementB_;
  using ElementC = ElementC_;
  using ElementAccumulator = ElementAccumulator_;
  using EpilogueOutputOp = EpilogueOutputOp_;

  static ComplexTransform const kTransformA = ComplexTransform::kNone;
  static ComplexTransform const kTransformB = ComplexTransform::kNone;
  static int const kThreadCount = kThreadCount_;
  static int const kThreadsPerRow = kThreadsPerRow_;

  struct Arguments {
    MatrixCoord problem_size; // row=M, column=number_non_zeros
    int32_t batch_count;
    typename EpilogueOutputOp::Params output_op;
    TensorRefA ref_A;         // physical shape [K, M], stride M
    ElementB const* ptr_B;    // selected signed values [batch, nnz]
    int64_t const* ptr_indices; // selected K indices [batch, nnz]
    ElementC const* ptr_C;
    ElementC* ptr_D;
    int64_t physical_M;
    int64_t batch_stride_B;
    int64_t batch_stride_indices;
    int64_t batch_stride_C;
    int64_t batch_stride_D;

    Arguments() : batch_count(0) {}

    Arguments(
        MatrixCoord problem_size_, int32_t batch_count_,
        typename EpilogueOutputOp::Params output_op_, TensorRefA ref_A_,
        void const* ptr_B_, int64_t const* ptr_indices_, void const* ptr_C_,
        void* ptr_D_, int64_t physical_M_, int64_t batch_stride_B_,
        int64_t batch_stride_indices_, int64_t batch_stride_C_,
        int64_t batch_stride_D_)
        : problem_size(problem_size_), batch_count(batch_count_),
          output_op(output_op_), ref_A(ref_A_),
          ptr_B(static_cast<ElementB const*>(ptr_B_)),
          ptr_indices(ptr_indices_), ptr_C(static_cast<ElementC const*>(ptr_C_)),
          ptr_D(static_cast<ElementC*>(ptr_D_)), physical_M(physical_M_),
          batch_stride_B(batch_stride_B_),
          batch_stride_indices(batch_stride_indices_),
          batch_stride_C(batch_stride_C_), batch_stride_D(batch_stride_D_) {}

    Status update(Arguments const& args) {
      *this = args;
      return Status::kSuccess;
    }
  };

  using Params = Arguments;
  union SharedStorage {};

  static Status can_implement(Arguments const& args) {
    if (args.problem_size.row() <= 0 || args.problem_size.column() <= 0 ||
        kThreadsPerRow <= 0 || kThreadsPerRow > 32 ||
        (kThreadsPerRow & (kThreadsPerRow - 1)) != 0 ||
        kThreadCount % kThreadsPerRow != 0) {
      return Status::kErrorInvalidProblem;
    }
    return Status::kSuccess;
  }

  CUTLASS_DEVICE
  void operator()(Params const& params, SharedStorage&) {
    for (int batch_idx = blockIdx.z; batch_idx < params.batch_count;
         batch_idx += gridDim.z) {
      int const lane_k = threadIdx.x;
      int const m = blockIdx.x * blockDim.y + threadIdx.y;
      if (m >= params.problem_size.row()) return;

      ElementB const* values =
          params.ptr_B + batch_idx * params.batch_stride_B;
      int64_t const* indices =
          params.ptr_indices + batch_idx * params.batch_stride_indices;
      ElementC const* ptr_C =
          params.ptr_C + batch_idx * params.batch_stride_C + m;
      ElementC* ptr_D =
          params.ptr_D + batch_idx * params.batch_stride_D + m;

      ElementAccumulator accum = ElementAccumulator(0);
      for (int j = lane_k; j < params.problem_size.column();
           j += kThreadsPerRow) {
        int64_t const k = indices[j];
        ElementA const a = params.ref_A.data()[k * params.physical_M + m];
        accum += ElementAccumulator(a) * ElementAccumulator(values[j]);
      }

      for (int mask = kThreadsPerRow >> 1; mask > 0; mask >>= 1) {
        accum += __shfl_xor_sync(0xffffffff, accum, mask, 32);
      }

      if (lane_k == 0) {
        EpilogueOutputOp output_op(params.output_op);
        typename EpilogueOutputOp::FragmentAccumulator accumulator_fragment;
        typename EpilogueOutputOp::FragmentOutput source_fragment;
        typename EpilogueOutputOp::FragmentOutput output_fragment;
        accumulator_fragment[0] = accum;
        if (output_op.is_source_needed()) {
          source_fragment[0] = *ptr_C;
          output_fragment = output_op(accumulator_fragment, source_fragment);
        } else {
          output_fragment = output_op(accumulator_fragment);
        }
        *ptr_D = output_fragment[0];
      }
    }
  }
};

} // namespace kernel
} // namespace gemm
} // namespace cutlass
