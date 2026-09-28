//same as rough_topk_sparse_gemv_improved but with some mods to work with gpgpusim. 
//copied whole file just in case. will clean up soon

// Sparse, column-oriented GEMV PyTorch extension.
//
// weight_col_major is prepared once as weight.t().contiguous(), with shape [K, M].
// For each input vector, top-k is selected on the current CUDA stream, then the
// CUDA kernel reads only weight_col_major[k, :] for those selected k indices.

#ifndef GEMV_STANDALONE
#include <torch/torch.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_bf16.h>
#include <ATen/cuda/CUDAEvent.h>
#include <c10/cuda/CUDACachingAllocator.h>
#endif
#include <cuda_runtime.h>
#include <cuda_bf16.h>

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <exception>
#include <tuple>

// Give the standalone kernel external linkage so GPGPU-Sim loads its PTX body.
#ifndef GEMV_STANDALONE
namespace {
#endif

// CUDA 12.8 emits a packed mov with an immediate for __bfloat162float,
// which this simulator's PTX parser does not accept. Widening and shifting
// preserves the exact BF16 bit pattern, including signed zero and NaNs.
__device__ __forceinline__ float gemv_bf16_to_float(__nv_bfloat16 value) {
#ifdef GEMV_STANDALONE
  unsigned bits;
  asm("cvt.u32.u16 %0, %1;" : "=r"(bits) : "h"(__bfloat16_as_ushort(value)));
  asm("shl.b32 %0, %0, 16;" : "+r"(bits));
  return __uint_as_float(bits);
#else
  return __bfloat162float(value);
#endif
}

constexpr int kThreads = 256;
constexpr int topKChunkSize = 256; //how may topk values to processat once

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
        float const activation_value_float = gemv_bf16_to_float(selected_x[j]); //convert activation vlaue to float for accumulation
        float const w = gemv_bf16_to_float(weight_col_major[activation_idx * int64_t(M) + m]);
        accumulator = fmaf(w, activation_value_float, accumulator);
      }
    }
    __syncthreads();
  }

  if (m < M) {
    atomicAdd(&output[batch_idx*M + m], accumulator); //atomic add since multiple blocks accesing this add
  }
}

#ifndef GEMV_STANDALONE
} // namespace
#endif


//rough topk, based off values of exponent bits
//3 step process

//step 1: create histogram. launch with 1D grid of 1D blocks
__global__ void rough_topk_create_histogram(
  const __nv_bfloat16* __restrict__ activation_vector,
  int K, //large dimension of vector
  int target_k,
  int* __restrict__ global_histogram,
  float* __restrict__ output,
      int64_t output_numel
){
  //activation vector is [1,K]
  //each thread responsible for conevrting from bf16 form to just exponent form
  int tid = threadIdx.x;
  int idx_in_vector = blockIdx.x * blockDim.x + threadIdx.x; 
  __shared__ int local_hist[256];
  int grid_stride = blockDim.x * gridDim.x;
  //initi block local histogram to 0
  if (tid < 256){
    local_hist[tid]=0;
  }

  // Cooperatively zero the GEMV output.
    for (int64_t i = idx_in_vector; i < output_numel; i+=grid_stride) {
        output[i] = 0.0f;
    }

  __syncthreads();

  //each thread handles on activation

  if (idx_in_vector < K) {
    // extract exponent
    unsigned short av_bits = __bfloat16_as_ushort(activation_vector[idx_in_vector]);
    int exponent = (av_bits >>7) & 0xFF;
    //increment histogram
    atomicAdd(&local_hist[exponent], 1);
}

  __syncthreads();
  // merge this block's histogram into the global one
  if (tid < 256) {
      atomicAdd(&global_histogram[tid], local_hist[tid]);
  }
}

#include <cuda_bf16.h>
#include <cub/block/block_scan.cuh>

//https://nvidia.github.io/cccl/unstable/cub/api/classcub_1_1BlockScan.html#_CPPv4N3cub9BlockScan9BlockScanEv
__global__ void rough_topk_find_exponent_cutoff(
      const int* __restrict__ global_histogram,
    int target_k,
    int* __restrict__ cutoff_exp,
    int* __restrict__ needed_from_cutoff
){

  // block wide scan
  using BlockScan = cub::BlockScan<int, 256>;
  __shared__ typename BlockScan::TempStorage scan_storage;

  int tid = threadIdx.x;
  int exp = 255-tid;
  int count = global_histogram[exp];

  int exclusive_prefix;
  BlockScan(scan_storage).ExclusiveSum(count, exclusive_prefix);
  //each thread loads part of histogram but in reverse. i.e thread 0 loads 255, thread 1 loads 254..etc

  //thread corresponds to bin where u reach target k
  if(exclusive_prefix < target_k && exclusive_prefix + count >= target_k){
    *cutoff_exp = exp;
    *needed_from_cutoff = target_k - exclusive_prefix;
  }
}

// //step3, create mask
__global__ void rough_topk_build_mask(
    const __nv_bfloat16* __restrict__ activation_vector,
    int K,
    const int* __restrict__ cutoff_exp,
    const int* __restrict__ needed_from_cutoff,
    int* __restrict__ cutoff_counter,
    uint8_t* __restrict__ mask
) {
    int idx_in_vector = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx_in_vector >= K) {
        return;
    }
    unsigned short av_bits = __bfloat16_as_ushort(activation_vector[idx_in_vector]);
    int exponent = (av_bits >> 7) & 0xFF;
    int cutoff = *cutoff_exp; //read exp value
    int needed = *needed_from_cutoff;

    if (exponent > cutoff) {
        mask[idx_in_vector] = 1;
    }else if (exponent < cutoff) {
        mask[idx_in_vector] = 0;
    }
    else {
        // exponent == cutoff
        int position = atomicAdd(cutoff_counter, 1);
        mask[idx_in_vector] = (position < *needed_from_cutoff) ? 1 : 0;
    }
}

__global__ void rough_topk_collect(
      const __nv_bfloat16* __restrict__ activation_vector,
      int K,
      int target_k,
      const int* __restrict__ cutoff_exp,
      const int* __restrict__ needed_from_cutoff,
      int* __restrict__ above_counter,
      int* __restrict__ cutoff_counter,
      int64_t* __restrict__ output_indices,
      __nv_bfloat16* __restrict__ output_values)
  {
    int idx_in_vector = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx_in_vector >= K) {
        return;
    }
    unsigned short av_bits = __bfloat16_as_ushort(activation_vector[idx_in_vector]);
    int exponent = (av_bits >> 7) & 0xFF;
    int cutoff = *cutoff_exp; //read exp value

    int needed = *needed_from_cutoff;

    // Number of elements whose exponent is strictly above cutoff.
    int count_above = target_k - needed;

    int output_pos = -1;

    if (exponent > cutoff) {
        output_pos = atomicAdd(above_counter, 1); //strictle above the cutoff exponenet
    } else if (exponent == cutoff) {
        int cutoff_pos = atomicAdd(cutoff_counter, 1); //at cutoff exponene, so pickng randomly ish

        if (cutoff_pos < needed) {
            output_pos = count_above + cutoff_pos;
        }
    }

    if (output_pos >= 0) {
        output_indices[output_pos] = static_cast<int64_t>(idx_in_vector);
        output_values[output_pos] = activation_vector[idx_in_vector];
    }
}

#ifndef GEMV_STANDALONE
//wrapper for rough topk
void launch_rough_topk(
    const __nv_bfloat16* activation_vector,
    int K,
    int target_k,
    //uint8_t* d_mask,
    int64_t* output_indices, //new
    __nv_bfloat16* output_values, //new
    float* final_output_vector,
    int output_vector_len, 
    const torch::TensorOptions& options,
    c10::cuda::CUDAStream stream
) {
    int threads = 256;
    int blocks = (K + threads - 1) / threads;
    
    // auto histogram = torch::empty({256}, options);
    // auto cutoff_exp = torch::empty({1}, options);
    // auto needed_from_cutoff = torch::empty({1}, options);
    // auto cutoff_counter = torch::empty({1}, options);
    // auto above_count = torch::empty({1}, options);


    auto workspace = torch::empty({260}, options.dtype(torch::kInt32));
    int *workspace_ptr = workspace.data_ptr<int>();
    int* d_histogram = workspace_ptr;
    int *d_cutoff_exp = workspace_ptr + 256;
    int *d_needed_from_cutoff =workspace_ptr + 257;
    int *d_cutoff_counter = workspace_ptr + 258;
    int *d_above_counter = workspace_ptr + 259;

    C10_CUDA_CHECK(cudaMemsetAsync(
      d_histogram,
      0,
      256 * sizeof(int),
      stream.stream()));

    C10_CUDA_CHECK(cudaMemsetAsync(
        d_cutoff_counter,
        0,
        2 * sizeof(int),
        stream.stream()));

    rough_topk_create_histogram<<<blocks, threads, 0, stream>>>(
        activation_vector,
        K,
        target_k,
        d_histogram,
        final_output_vector,
        output_vector_len
    );

    
    rough_topk_find_exponent_cutoff<<<1, 256, 0, stream>>>(
        d_histogram,
        target_k,
        d_cutoff_exp,
        d_needed_from_cutoff
    );

    rough_topk_collect<<<blocks, threads, 0, stream>>>(
        activation_vector,
        K,
        target_k,//new
        d_cutoff_exp,
        d_needed_from_cutoff,
        d_above_counter, //new
        d_cutoff_counter,
        //d_mask
        output_indices, //new 
        output_values //new
    );

    c10::cuda::CUDACachingAllocator::recordStream(
      workspace.storage().data_ptr(),
      stream);

}

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
  TORCH_CHECK(x.size(0) ==1, "expected batch size of 1, got", x.size(0));
  TORCH_CHECK(keep_count >= 1 && keep_count <= K,"keep_count must be between 1 and K=", K);
  
  auto output_shape = x.sizes().vec();
  output_shape.back() = M; //output shape is 1xM


  auto x_flat = x.contiguous().reshape({-1, K});
  int64_t const batch_count = x_flat.size(0);
  // std::cout << "x_flat: " << x_flat << std::endl;
  // std::cout << "x_flat shape: " << x_flat.sizes() << std::endl;
  c10::cuda::CUDAStream  primary_stream = at::cuda::getCurrentCUDAStream(); //ensure ordering

  //alternaitve topk version 1
  // auto mask = torch::empty({K},x.options().dtype(torch::kUInt8));
  // launch_rough_topk(reinterpret_cast<__nv_bfloat16 const*>(x_flat.data_ptr<at::BFloat16>()),
  //   K,
  //   keep_count,
  //   mask.data_ptr<uint8_t>(),
  // stream);
  // auto indices = at::nonzero(mask).squeeze(1).to(torch::kInt64).contiguous();
  // auto values = x_flat.index_select(1, indices).contiguous();
//END OF ALTERNATIVE TOPK VERSION 1

//aternative topk version2
  auto indices = torch::empty({keep_count},x.options().dtype(torch::kInt64));
  auto values = torch::empty({keep_count},x.options().dtype(torch::kBFloat16));
  auto output_f32 = torch::empty({batch_count, M}, x.options().dtype(torch::kFloat32)); //32 bit output
  //auto output_f32 = torch::zeros({batch_count, M}, x.options().dtype(torch::kFloat32)); //32 bit output

  //int output_len = batch_count * M;

  launch_rough_topk(
    reinterpret_cast<const __nv_bfloat16*>(x_flat.data_ptr<at::BFloat16>()),
    static_cast<int>(K),
    static_cast<int>(keep_count),
    indices.data_ptr<int64_t>(),
    reinterpret_cast<__nv_bfloat16*>(values.data_ptr<at::BFloat16>()),
    output_f32.data_ptr<float>(),
    output_f32.numel(),
    x.options(),
    primary_stream);
    //END OF ALTERNAITVE TOPKVERSION2
  //START OF TOPK TRUE CODE
  // topk ranks by magnitude; gather recovers the original signed values.
  //aten::topk(Tensor self, SymInt k, int dim=-1, bool largest=True, bool sorted=True) -> (Tensor values, Tensor indices)
  
  
  // auto topk_result = at::topk(at::abs(x_flat), keep_count, 1, true, false);
  // auto indices = std::get<1>(topk_result).contiguous(); // get the indicies
  // //indices = std::get<0>(at::sort(indices, 1, false)).contiguous(); //sort in ascending fashio NOTE** DOESNT MAKE A DIFFERENCE

  // auto values = x_flat.gather(1, indices).contiguous();
  //END OF TOPK TRUE CODE

  


  dim3 const block(kThreads); // 1d block of kthreads

  //3d block where x is batch count and y is number of blocks for M outputs and z K splits
  dim3 const grid(static_cast<unsigned>(batch_count),static_cast<unsigned>((M + kThreads - 1) / kThreads), static_cast<unsigned>(split_k)); //kthreads=1 makes roud up

  //launch kernel
  indexed_sparse_gemv_bf16<<<grid, block, 0, primary_stream.stream()>>>(
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

//small test for gpgpusim
#ifndef GEMV_NO_MAIN
int main() {
  try {
    torch::NoGradGuard no_grad;
    c10::cuda::CUDAGuard device_guard(c10::Device(c10::kCUDA, 0));

    constexpr int64_t K = 512;
    constexpr int64_t M = 256;
    constexpr int64_t keep_count = 32;
    constexpr int64_t split_k = 2;

    const auto cpu_options = torch::TensorOptions()
                                 .device(torch::kCPU)
                                 .dtype(torch::kBFloat16);
    const torch::Device device(torch::kCUDA, 0);

    // Already in the layout expected by sparse_gemv: [K, M].
    auto weight_col_major = torch::ones({K, M}, cpu_options).to(device);
    auto x = torch::ones({1, 1, K}, cpu_options).to(device);

    auto output = sparse_gemv(weight_col_major, x, keep_count, split_k);
    C10_CUDA_CHECK(cudaDeviceSynchronize());
    auto output_cpu = output.to(torch::kCPU).to(torch::kFloat32);

    // Every selected value and weight is one, so any valid tie selection
    // must produce keep_count for each output element.
    const float expected = static_cast<float>(keep_count);
    const float* result = output_cpu.data_ptr<float>();
    for (int64_t m = 0; m < M; ++m) {
      if (result[m] != expected) {
        std::cerr << "FAIL at output " << m << ": expected " << expected
                  << ", got " << result[m] << '\n';
        return 1;
      }
    }
    std::cout << "PASS: output shape " << output_cpu.sizes()
              << ", all " << M << " elements equal " << expected << '\n';
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "GEMV test failed: " << error.what() << '\n';
    return 1;
  }
}

#endif // GEMV_NO_MAIN
#endif // GEMV_STANDALONE
