
// made to use w gpgpusim
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdint>

#define GEMV_STANDALONE
#include "gemv_standalone_kernels.cu"
#undef GEMV_STANDALONE

torch::Tensor standalone_gemv(torch::Tensor weight_col_major, torch::Tensor x,
                              int64_t keep_count, int64_t split_k) {
  TORCH_CHECK(weight_col_major.is_cuda() && x.is_cuda(), "tensors must be CUDA tensors");
  TORCH_CHECK(weight_col_major.scalar_type() == at::kBFloat16 && x.scalar_type() == at::kBFloat16,
              "weight and x must be BF16");
  TORCH_CHECK(weight_col_major.dim() == 2 && weight_col_major.is_contiguous(),
              "weight must be contiguous [K, M]");
  TORCH_CHECK(x.dim() >= 1 && x.size(-1) == weight_col_major.size(0),
              "x last dimension must equal weight K");
  TORCH_CHECK(x.size(0) == 1, "only batch size 1 is supported");
  TORCH_CHECK(keep_count >= 1 && keep_count <= weight_col_major.size(0),
              "keep_count must be in [1, K]");
  TORCH_CHECK(split_k >= 1, "split_k must be positive");

  auto x_flat = x.contiguous().reshape({1, weight_col_major.size(0)});
  const int K = static_cast<int>(weight_col_major.size(0));
  const int M = static_cast<int>(weight_col_major.size(1));
  auto output_f32 = torch::empty({1, M}, x.options().dtype(torch::kFloat32));
  auto stream = at::cuda::getCurrentCUDAStream();
  __nv_bfloat16* values = nullptr;
  int64_t* indices = nullptr;
  int* workspace_ptr = nullptr;
  C10_CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&values), keep_count * sizeof(__nv_bfloat16)));
  C10_CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&indices), keep_count * sizeof(int64_t)));
  C10_CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&workspace_ptr), 260 * sizeof(int)));
  C10_CUDA_CHECK(cudaMemsetAsync(workspace_ptr, 0, 260 * sizeof(int), stream.stream()));

  rough_topk_create_histogram<<<(K + 255) / 256, 256, 0, stream.stream()>>>(
      reinterpret_cast<const __nv_bfloat16*>(x_flat.data_ptr<at::BFloat16>()),
      K, static_cast<int>(keep_count), workspace_ptr, output_f32.data_ptr<float>(), M);
  rough_topk_find_exponent_cutoff<<<1, 256, 0, stream.stream()>>>(
      workspace_ptr, static_cast<int>(keep_count), workspace_ptr + 256, workspace_ptr + 257);
  rough_topk_collect<<<(K + 255) / 256, 256, 0, stream.stream()>>>(
      reinterpret_cast<const __nv_bfloat16*>(x_flat.data_ptr<at::BFloat16>()),
      K, static_cast<int>(keep_count), workspace_ptr + 256, workspace_ptr + 257,
      workspace_ptr + 259, workspace_ptr + 258, indices, values);

  indexed_sparse_gemv_bf16<<<dim3(1, (M + kThreads - 1) / kThreads, split_k),
                              kThreads, 0, stream.stream()>>>(
      reinterpret_cast<const __nv_bfloat16*>(weight_col_major.data_ptr<at::BFloat16>()),
      values, indices, output_f32.data_ptr<float>(), M,
      static_cast<int>(keep_count), static_cast<int>(split_k));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  C10_CUDA_CHECK(cudaStreamSynchronize(stream.stream()));
  C10_CUDA_CHECK(cudaFree(workspace_ptr));
  C10_CUDA_CHECK(cudaFree(indices));
  C10_CUDA_CHECK(cudaFree(values));
  auto shape = x.sizes().vec();
  shape.back() = M;
  return output_f32.to(x.scalar_type()).reshape(shape);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def("gemv", &standalone_gemv, "Standalone sparse GEMV kernels with PyTorch tensors");
}
