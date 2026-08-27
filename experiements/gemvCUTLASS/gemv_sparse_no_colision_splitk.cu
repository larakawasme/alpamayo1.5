// Sparse, column-oriented GEMV PyTorch extension.
//
// weight_col_major is prepared once as weight.t().contiguous(), with shape [K, M].
// For each input vector, top-k is selected on the current CUDA stream, then the
// CUDA kernel reads only weight_col_major[k, :] for those selected k indices.

#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_bf16.h>

#include <algorithm>
#include <cstdint>
#include <tuple>

namespace {

constexpr int kThreads = 256;
constexpr int topKChunkSize = 256; //how may topk values to processat once

__global__ void indexed_sparse_gemv_bf16(
    __nv_bfloat16 const* __restrict__ weight_col_major, // [K, M]
    __nv_bfloat16 const* __restrict__ values,     // [batch, non zeros]
    int64_t const* __restrict__ indices,          // [batch, non zeroes]
    float* __restrict__ partials, // [batch, split_k, M]
    int M,
    int number_non_zeros,
    int split_k) {

  int const batch_idx = blockIdx.x; //current block idx in grid 
  int const m = blockIdx.y * blockDim.x + threadIdx.x; //which output element this thread handles. basically idx of the column

  int const split_idx = blockIdx.z;
  int const elements_per_split = (number_non_zeros + split_k -1) / split_k;
  int const curr_split_begin = split_idx * elements_per_split;
  int const curr_split_end = min(curr_split_begin + elements_per_split, number_non_zeros);

  __shared__ int64_t selected_idx[topKChunkSize];
  __shared__ __nv_bfloat16 selected_x[topKChunkSize];

  float accumulator = 0.0f;

  for (int base = curr_split_begin; base < curr_split_end; base += topKChunkSize) { //within each split-k, can load topkchunksize 
    int const count = min(topKChunkSize, curr_split_end - base); //count in this chunk. in case number_non_zeros not divisible by topKChunkSize

    // copy idx/value pair of activation vector from global to shared memory. 
    // batch should be 0 but handling anyways
    if (threadIdx.x < count) {
      int const offset = batch_idx * number_non_zeros + base + threadIdx.x;
      selected_idx[threadIdx.x] = indices[offset];
      selected_x[threadIdx.x] = values[offset];
    }
    __syncthreads();

    //threads process one selected weight column (original W) at a time. a thread calculates across all rows and maintains partial sum for outputo
    if (m < M) { //based on m, its what weight column element that thread is wokring on
      //#pragma unroll 4
      for (int j = 0; j < count; j++) {
        int64_t const activation_idx = selected_idx[j]; //get activation idx
        float const activation_value_float = __bfloat162float(selected_x[j]); //convert activation vlaue to float for accumulation
        float const w = __bfloat162float(weight_col_major[activation_idx * int64_t(M) + m]);
        accumulator = fmaf(w, activation_value_float, accumulator);
      }
    }
    __syncthreads();
  }


    // ----------------------------------------------------------
  // IMPORTANT:
  //
  // Instead of:
  // atomicAdd(&output[m], accumulator);
  // each split writes to its OWN location:
  // partials[batch][split][m]
  // Therefore there is NO write collision.
  // ----------------------------------------------------------

  if (m < M) {
    int64_t const partial_offset =(int64_t(batch_idx) * split_k + split_idx)* int64_t(M)+ m;
    partials[partial_offset] = accumulator;
  }


} 

// Kernel 2: reduce Split-K partial
// For every output m:
// output[m] =
//     partial[0][m]
//   + partial[1][m]
//   + ...
//   + partial[split_k-1][m]
//
// Then convert directly FP32 -> BF16.

__global__ void reduce_splitk_partials_to_bf16(
    float const* __restrict__ partials,
    __nv_bfloat16* __restrict__ output,
    int M,
    int split_k) {

  int const m =blockIdx.x * blockDim.x + threadIdx.x;

  int const batch_idx =blockIdx.y;
  if (m >= M) {
    return;
  }
  float sum = 0.0f;

  for (int s = 0; s < split_k; ++s) {

    int64_t const partial_offset =(int64_t(batch_idx) * split_k + s)* int64_t(M)+ m;

    sum +=partials[partial_offset];
  }

  // Direct FP32 -> BF16 store
  output[
      int64_t(batch_idx) * M + m
  ] = __float2bfloat16_rn(sum);
}


} // namespace





// returns:    [..., M]
torch::Tensor sparse_gemv(
    torch::Tensor weight_col_major, // weight_col_major: [K, M], created with weight.t().contiguous()
    torch::Tensor x,// x:[1,1,K]
    int64_t keep_count,
    int64_t split_k) {// keep_count: exact number of activations retained per vector

  TORCH_CHECK(weight_col_major.is_cuda() && x.is_cuda(), "tensors must be on GPU");
  TORCH_CHECK(weight_col_major.device() == x.device(), "weight_col_major and x must be on same GPU");
  TORCH_CHECK(weight_col_major.scalar_type() == at::kBFloat16, "weight_col_major must be bf16");
  TORCH_CHECK(x.scalar_type() == at::kBFloat16, "x must be bf16");
  TORCH_CHECK(weight_col_major.dim() == 2, "weight_col_major must have shape [K, M]");
  TORCH_CHECK(weight_col_major.is_contiguous(),"weight_col_major must be contiguous; use weight.t().contiguous()");
  TORCH_CHECK(x.dim() >= 1, "x must have at least one dimension");

  int64_t const K = weight_col_major.size(0);
  int64_t const M = weight_col_major.size(1);

  TORCH_CHECK(x.size(-1) == K, "x last dimension must equal K=", K);
  TORCH_CHECK(keep_count >= 1 && keep_count <= K,"keep_count must be between 1 and K=", K);
  
  auto output_shape = x.sizes().vec();
  output_shape.back() = M; //output shape is 1xM


  auto x_flat = x.contiguous().reshape({-1, K});
  int64_t const batch_count = x_flat.size(0);
  // std::cout << "x_flat: " << x_flat << std::endl;
  // std::cout << "x_flat shape: " << x_flat.sizes() << std::endl;

  // topk ranks by magnitude; gather recovers the original signed values.
  // aten::topk(Tensor self, SymInt k, int dim=-1, bool largest=True, bool sorted=True) -> (Tensor values, Tensor indices)
  auto topk_result = at::topk(at::abs(x_flat), keep_count, 1, true, false);
  auto indices = std::get<1>(topk_result).contiguous(); // get the indicies
  //indices = std::get<0>(at::sort(indices, 1, false)).contiguous(); //sort in ascending fashio NOTE** DOESNT MAKE A DIFFERENCE

  auto values = x_flat.gather(1, indices).contiguous();


  // partial buffer
  // Shape:[batch_count, split_k, M]
  // We can use torch::empty() instead of torch::zeros()
  // because every element gets overwritten by kernel 1.
  // ==========================================================

  auto partials =torch::empty({batch_count, split_k, M},x.options().dtype(torch::kFloat32));
  // Final BF16 output
  auto output =torch::empty({batch_count, M},x.options());


  dim3 const block(kThreads); // 1d block of kthreads

  //3d block where x is batch count and y is number of blocks for M outputs and z K splits
  dim3 const grid(static_cast<unsigned>(batch_count),static_cast<unsigned>((M + kThreads - 1) / kThreads), static_cast<unsigned>(split_k)); //kthreads=1 makes roud up

  cudaStream_t stream = at::cuda::getCurrentCUDAStream(); //ensure ordering

  //launch kernel
  indexed_sparse_gemv_bf16<<<grid, block, 0, stream>>>(
      reinterpret_cast<__nv_bfloat16 const*>(weight_col_major.data_ptr<at::BFloat16>()),
      reinterpret_cast<__nv_bfloat16 const*>(values.data_ptr<at::BFloat16>()),
      indices.data_ptr<int64_t>(),
      partials.data_ptr<float>(),
      static_cast<int>(M), //num of columns in weight
      static_cast<int>(keep_count),
      static_cast<int>(split_k));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  
  // Kernel 2 launchReduce:
  // [batch, split_k, M] into [batch, M]
  dim3 const reduce_block(kThreads);
  dim3 const reduce_grid(static_cast<unsigned>((M + kThreads - 1)/ kThreads),
                        static_cast<unsigned>(batch_count));


  reduce_splitk_partials_to_bf16<<<reduce_grid, reduce_block, 0, stream>>>(
    partials.data_ptr<float>(),
    reinterpret_cast<__nv_bfloat16*>(
    output.data_ptr<at::BFloat16>()),
    static_cast<int>(M),
    static_cast<int>(split_k));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output.reshape(output_shape);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def(
      "gemv",
      &sparse_gemv,
      "Top-k indexed sparse GEMV with column-oriented BF16 weights");
}
