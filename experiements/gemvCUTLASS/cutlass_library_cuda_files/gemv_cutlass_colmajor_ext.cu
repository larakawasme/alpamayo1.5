#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include "cutlass/cutlass.h"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/gemm/device/gemv.h"
 #include "cutlass/gemm/kernel/gemv.h"

using Element = cutlass::bfloat16_t;
using Accumulator = float;
using LayoutA = cutlass::layout::ColumnMajor;
using Epilogue = cutlass::epilogue::thread::LinearCombination<Element, 1, Accumulator, Accumulator>;
using Kernel = cutlass::gemm::kernel::Gemv<
      Element,
      LayoutA,
      Element,
      Element,
      Accumulator,
      Epilogue,
      1,    // kElementsPerAccess
      256  // kThreadCount
  >;


using DeviceGemv = cutlass::gemm::device::Gemv<Kernel>;

torch::Tensor cutlass_column_major_gemv(torch::Tensor weight_col_major, torch::Tensor x) {
  TORCH_CHECK(weight_col_major.is_cuda() && x.is_cuda(), "tensors must be GPU");
  TORCH_CHECK(weight_col_major.device() == x.device(), "tensors must share a device");
  TORCH_CHECK(weight_col_major.dim() == 2 && weight_col_major.is_contiguous(), "weight must be weight.T.contiguous() [K,M]");


  int64_t const K = weight_col_major.size(0); 
  int64_t const M = weight_col_major.size(1);

  TORCH_CHECK(x.size(-1) == K, "x last dimension must equal K=", K);


  auto output_shape = x.sizes().vec(); 
  output_shape.back() = M;

  auto x_flat = x.contiguous().reshape({-1, K});

  int64_t const batch = x_flat.size(0); 
  auto output = torch::empty({batch, M}, x.options());

  // Row-major [K,M] bytes are identical to column-major [M,K] bytes.
  cutlass::TensorRef<Element, LayoutA> ref_A(
                            reinterpret_cast<Element*>(weight_col_major.data_ptr<at::BFloat16>()),
                            LayoutA(M));

                      
  typename Epilogue::Params epilogue(Accumulator(1), Accumulator(0));

  typename DeviceGemv::Arguments args(
      cutlass::MatrixCoord(M, K), 
      static_cast<int32_t>(batch), 
      epilogue, 
      ref_A,
      x_flat.data_ptr<at::BFloat16>(), 
      output.data_ptr<at::BFloat16>(), 
      output.data_ptr<at::BFloat16>(),
      0, K, M, M);

  auto status = DeviceGemv::can_implement(args);
  TORCH_CHECK(status == cutlass::Status::kSuccess, "cannot implement");
  DeviceGemv op;
  TORCH_CHECK(op.initialize(args) == cutlass::Status::kSuccess, "initialize failed");
  at::cuda::getCurrentCUDAStream();
  TORCH_CHECK(op.run() == cutlass::Status::kSuccess, "launch failed");
  return output.reshape(output_shape);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def("gemv", &cutlass_column_major_gemv, "CUTLASS dense column-major BF16 GEMV");
}
