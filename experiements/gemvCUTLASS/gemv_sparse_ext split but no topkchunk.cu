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
//constexpr int topKChunkSize = 256; //how may topk values to processat once

__global__ void indexed_sparse_gemv_bf16(
    __nv_bfloat16 const* __restrict__ weight_col_major, // [K, M]
    __nv_bfloat16 const* __restrict__ values,     // [batch, non zeros]
    int64_t const* __restrict__ indices,          // [batch, non zeroes]
    float* __restrict__ output,           // [batch, M]
    int M,
    int number_non_zeros,
    int split_k) {

  int const batch_idx = blockIdx.x; //current block idx in grid 
  int const m = blockIdx.y * blockDim.x + threadIdx.x; //which output element this thread handles. basically idx of the column

  int const split_idx = blockIdx.z;
  int const elements_per_split = (number_non_zeros + split_k -1) / split_k;
  int const curr_split_begin = split_idx * elements_per_split;
  int const curr_split_end = min(curr_split_begin + elements_per_split, number_non_zeros);
  int const split_size = curr_split_end - curr_split_begin;

  __shared__ int64_t selected_idx[kThreads];
  __shared__ __nv_bfloat16 selected_x[kThreads];

  float accumulator = 0.0f;

  //for (int base = curr_split_begin; base < curr_split_end; base += topKChunkSize) { //within each split-k, can load topkchunksize 
    //int const count = min(topKChunkSize, curr_split_end - base); //count in this chunk. in case number_non_zeros not divisible by topKChunkSize
    
    //copperatively load our split of activation vector
    // copy idx/value pair of activation vector from global to shared memory. 
    // batch should be 0 but handling anyways
    if (threadIdx.x < split_size) {
      int const offset = batch_idx * number_non_zeros + curr_split_begin + threadIdx.x;
      selected_idx[threadIdx.x] = indices[offset];
      selected_x[threadIdx.x] = values[offset];
    }
    __syncthreads();

    //threads process one selected weight column (original W) at a time. a thread calculates across all rows and maintains partial sum for outputo
    if (m < M) { //based on m, its what weight column element that thread is wokring on
      //#pragma unroll 4
      for (int j = 0; j < split_size; j++) {
        int64_t const activation_idx = selected_idx[j]; //get activation idx
        float const activation_value_float = __bfloat162float(selected_x[j]); //convert activation vlaue to float for accumulation
        float const w = __bfloat162float(weight_col_major[activation_idx * int64_t(M) + m]);
        accumulator = fmaf(w, activation_value_float, accumulator);
      }
    }
    __syncthreads();
  

  if (m < M) {
    atomicAdd(&output[batch_idx*M + m], accumulator); //atomic add since multiple blocks accesing this add
  }
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
  auto output_f32 = torch::zeros({batch_count, M}, x.options().dtype(torch::kFloat32)); //32 bit output
  

  int64_t const elements_per_split =
    (keep_count + split_k - 1) / split_k;

  TORCH_CHECK(
      elements_per_split <= kThreads,
      "split_k too small: each split must contain at most ",
      kThreads,
      " selected activations"
  );
  dim3 const block(kThreads); // 1d block of kthreads

  //3d block where x is batch count and y is number of blocks for M outputs and z K splits
  dim3 const grid(static_cast<unsigned>(batch_count),static_cast<unsigned>((M + kThreads - 1) / kThreads), static_cast<unsigned>(split_k)); //kthreads=1 makes roud up

  cudaStream_t stream = at::cuda::getCurrentCUDAStream(); //ensure ordering

  //launch kernel
  indexed_sparse_gemv_bf16<<<grid, block, 0, stream>>>(
      reinterpret_cast<__nv_bfloat16 const*>(weight_col_major.data_ptr<at::BFloat16>()),
      reinterpret_cast<__nv_bfloat16 const*>(values.data_ptr<at::BFloat16>()),
      indices.data_ptr<int64_t>(),
      output_f32.data_ptr<float>(),
      static_cast<int>(M), //num of columns in weight
      static_cast<int>(keep_count),
      static_cast<int>(split_k));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  
  auto output = output_f32.to(x.scalar_type());
  return output.reshape(output_shape);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def(
      "gemv",
      &sparse_gemv,
      "Top-k indexed sparse GEMV with column-oriented BF16 weights");
}
