#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include "cutlass/cutlass.h"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/gemm/device/gemv.h"
#include "cutlass/gemm/kernel/gemv_topk.h"

using Element = cutlass::bfloat16_t;
using Accumulator = float;
using Epilogue = cutlass::epilogue::thread::LinearCombination<
    Element, 1, Accumulator, Accumulator>;
using Kernel = cutlass::gemm::kernel::GemvTopK<
    Element, Element, Element, Accumulator, Epilogue, 128, 16>;
using DeviceGemv = cutlass::gemm::device::Gemv<Kernel>;

torch::Tensor cutlass_topk_gemv(
    torch::Tensor weight_col_major, torch::Tensor x, int64_t keep_count) {
  TORCH_CHECK(weight_col_major.is_cuda() && x.is_cuda(), "tensors must be CUDA");
  TORCH_CHECK(weight_col_major.device() == x.device(), "tensors must share a device");
  TORCH_CHECK(weight_col_major.scalar_type() == at::kBFloat16 &&
              x.scalar_type() == at::kBFloat16, "tensors must be bf16");
  TORCH_CHECK(weight_col_major.dim() == 2 && weight_col_major.is_contiguous(),
              "weight must be weight.T.contiguous() with shape [K,M]");
  TORCH_CHECK(x.dim() >= 1, "x must have at least one dimension");

  int64_t const K = weight_col_major.size(0);
  int64_t const M = weight_col_major.size(1);
  TORCH_CHECK(x.size(-1) == K, "x last dimension must equal K=", K);
  TORCH_CHECK(keep_count >= 1 && keep_count <= K, "invalid keep_count");
  TORCH_CHECK(M <= INT_MAX && keep_count <= INT_MAX, "dimensions exceed int32");
  c10::cuda::CUDAGuard guard(x.device());

  auto output_shape = x.sizes().vec();
  output_shape.back() = M;
  auto x_flat = x.contiguous().reshape({-1, K});
  int64_t const batch = x_flat.size(0);

  auto topk = at::topk(at::abs(x_flat), keep_count, 1, true, false);
  auto indices = std::get<1>(topk).contiguous();
  auto values = x_flat.gather(1, indices).contiguous();
  auto output = torch::empty({batch, M}, x.options());

  cutlass::TensorRef<Element, cutlass::layout::RowMajor> ref_A(
      reinterpret_cast<Element*>(weight_col_major.data_ptr<at::BFloat16>()),
      cutlass::layout::RowMajor(M));
  typename Epilogue::Params epilogue(Accumulator(1), Accumulator(0));
  typename DeviceGemv::Arguments args(
      cutlass::MatrixCoord(M, keep_count), static_cast<int32_t>(batch), epilogue,
      ref_A, values.data_ptr<at::BFloat16>(), indices.data_ptr<int64_t>(),
      output.data_ptr<at::BFloat16>(), output.data_ptr<at::BFloat16>(), M,
      keep_count, keep_count, M, M);

  auto status = DeviceGemv::can_implement(args);
  TORCH_CHECK(status == cutlass::Status::kSuccess, "CUTLASS top-k cannot implement");
  DeviceGemv op;
  status = op.initialize(args);
  TORCH_CHECK(status == cutlass::Status::kSuccess, "CUTLASS top-k initialize failed");
  status = op.run(at::cuda::getCurrentCUDAStream());
  TORCH_CHECK(status == cutlass::Status::kSuccess, "CUTLASS top-k launch failed");
  return output.reshape(output_shape);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def("gemv", &cutlass_topk_gemv, "CUTLASS-style indexed top-k GEMV");
}
