// Dense column-oriented BF16 GEMV matching the sparse kernel's thread mapping.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_bf16.h>
#include <cstdint>

namespace {
constexpr int kThreads = 256;
constexpr int kActivationChunkSize = 256;

__global__ void dense_column_gemv_bf16(
    __nv_bfloat16 const* __restrict__ weight_col_major,
    __nv_bfloat16 const* __restrict__ x,
    __nv_bfloat16* __restrict__ output, int M, int K) {
  int const batch_idx = blockIdx.x;
  int const m = blockIdx.y * blockDim.x + threadIdx.x;
  __shared__ __nv_bfloat16 activation_chunk[kActivationChunkSize];
  float accumulator = 0.0f;

  for (int base = 0; base < K; base += kActivationChunkSize) {
    int const count = min(kActivationChunkSize, K - base);
    if (threadIdx.x < count) {
      activation_chunk[threadIdx.x] = x[batch_idx * int64_t(K) + base + threadIdx.x];
    }
    __syncthreads();
    if (m < M) {
      for (int j = 0; j < count; ++j) {
        int const k = base + j;
        float const activation = __bfloat162float(activation_chunk[j]);
        float const weight = __bfloat162float(weight_col_major[k * int64_t(M) + m]);
        accumulator = fmaf(weight, activation, accumulator);
      }
    }
    __syncthreads();
  }
  if (m < M) output[batch_idx * int64_t(M) + m] = __float2bfloat16_rn(accumulator);
}
}

torch::Tensor dense_column_gemv(torch::Tensor weight_col_major, torch::Tensor x) {
  TORCH_CHECK(weight_col_major.is_cuda() && x.is_cuda(), "tensors must be on GPU");
  TORCH_CHECK(weight_col_major.device() == x.device(), "tensors must be on the same GPU");
  TORCH_CHECK(weight_col_major.scalar_type() == at::kBFloat16 && x.scalar_type() == at::kBFloat16, "tensors must be bf16");
  TORCH_CHECK(weight_col_major.dim() == 2 && weight_col_major.is_contiguous(), "weight must be contiguous [K, M]");
  TORCH_CHECK(x.dim() >= 1, "x must have at least one dimension");
  int64_t const K = weight_col_major.size(0), M = weight_col_major.size(1);
  TORCH_CHECK(x.size(-1) == K, "x last dimension must equal K=", K);
  TORCH_CHECK(K <= INT_MAX && M <= INT_MAX, "dimensions exceed int32 limits");
  c10::cuda::CUDAGuard device_guard(x.device());
  auto output_shape = x.sizes().vec(); output_shape.back() = M;
  auto x_flat = x.contiguous().reshape({-1, K});
  int64_t const batch_count = x_flat.size(0);
  auto output = torch::empty({batch_count, M}, x.options());
  dim3 const block(kThreads);
  dim3 const grid(static_cast<unsigned>(batch_count), static_cast<unsigned>((M + kThreads - 1) / kThreads));
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  dense_column_gemv_bf16<<<grid, block, 0, stream>>>(
      reinterpret_cast<__nv_bfloat16 const*>(weight_col_major.data_ptr<at::BFloat16>()),
      reinterpret_cast<__nv_bfloat16 const*>(x_flat.data_ptr<at::BFloat16>()),
      reinterpret_cast<__nv_bfloat16*>(output.data_ptr<at::BFloat16>()),
      static_cast<int>(M), static_cast<int>(K));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output.reshape(output_shape);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def("gemv", &dense_column_gemv, "Dense column-oriented BF16 GEMV");
}
