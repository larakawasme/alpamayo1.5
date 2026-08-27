#pragma once
#include "cutlass/cutlass.h"
#include "cutlass/gemm/gemm.h"
#include "cutlass/layout/matrix.h"
#include "cutlass/tensor_ref.h"

namespace cutlass { namespace gemm { namespace kernel {
template <typename ElementA_, typename ElementB_, typename ElementC_,
          typename ElementAccumulator_, typename EpilogueOutputOp_,
          int kThreadCount_ = 256>
struct GemvColumnShared {
  using ElementA = ElementA_; using LayoutA = layout::ColumnMajor;
  using TensorRefA = TensorRef<ElementA, LayoutA>; using ElementB = ElementB_;
  using ElementC = ElementC_; using ElementAccumulator = ElementAccumulator_;
  using EpilogueOutputOp = EpilogueOutputOp_;
  static ComplexTransform const kTransformA = ComplexTransform::kNone;
  static ComplexTransform const kTransformB = ComplexTransform::kNone;
  static int const kThreadCount = kThreadCount_;
  static int const kThreadsPerRow = 1;

  struct Arguments {
    MatrixCoord problem_size; int32_t batch_count;
    typename EpilogueOutputOp::Params output_op; TensorRefA ref_A;
    ElementB const* ptr_B; ElementC const* ptr_C; ElementC* ptr_D;
    int64_t batch_stride_A, batch_stride_B, batch_stride_C, batch_stride_D;
    Arguments() : batch_count(0) {}
    Arguments(MatrixCoord p, int32_t b, typename EpilogueOutputOp::Params e,
              TensorRefA a, void const* pb, void const* pc, void* pd,
              int64_t sa, int64_t sb, int64_t sc, int64_t sd)
      : problem_size(p), batch_count(b), output_op(e), ref_A(a),
        ptr_B(static_cast<ElementB const*>(pb)), ptr_C(static_cast<ElementC const*>(pc)),
        ptr_D(static_cast<ElementC*>(pd)), batch_stride_A(sa), batch_stride_B(sb),
        batch_stride_C(sc), batch_stride_D(sd) {}
    Status update(Arguments const& a) { *this = a; return Status::kSuccess; }
  };
  using Params = Arguments;
  struct SharedStorage { ElementB activation[kThreadCount]; };
  static Status can_implement(Arguments const& a) {
    return a.problem_size.row() > 0 && a.problem_size.column() > 0
      ? Status::kSuccess : Status::kErrorInvalidProblem;
  }
  CUTLASS_DEVICE void operator()(Params const& p, SharedStorage& shared) {
    for (int batch = blockIdx.z; batch < p.batch_count; batch += gridDim.z) {
      int const m = blockIdx.x * kThreadCount + threadIdx.x;
      ElementAccumulator accum = ElementAccumulator(0);
      ElementB const* x = p.ptr_B + batch * p.batch_stride_B;
      ElementA const* w = p.ref_A.data() + batch * p.batch_stride_A;
      for (int base = 0; base < p.problem_size.column(); base += kThreadCount) {
        int const count = min(kThreadCount, p.problem_size.column() - base);
        if (threadIdx.x < count) shared.activation[threadIdx.x] = x[base + threadIdx.x];
        __syncthreads();
        if (m < p.problem_size.row()) {
          CUTLASS_PRAGMA_NO_UNROLL
          for (int j = 0; j < count; ++j) {
            int const k = base + j;
            accum += ElementAccumulator(w[m + int64_t(k) * p.ref_A.stride(0)]) *
                     ElementAccumulator(shared.activation[j]);
          }
        }
        __syncthreads();
      }
      if (m < p.problem_size.row()) {
        EpilogueOutputOp op(p.output_op);
        typename EpilogueOutputOp::FragmentAccumulator af; af[0] = accum;
        typename EpilogueOutputOp::FragmentOutput sf, of;
        ElementC const* c = p.ptr_C + batch * p.batch_stride_C + m;
        ElementC* d = p.ptr_D + batch * p.batch_stride_D + m;
        if (op.is_source_needed()) { sf[0] = *c; of = op(af, sf); } else { of = op(af); }
        *d = of[0];
      }
    }
  }
};
}}}
