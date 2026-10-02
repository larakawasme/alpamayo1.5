// Compare the unchanged indexed kernel against simulator sparse TMA and a CPU oracle.
#define GEMV_STANDALONE
#include "original/rough_topk_gemv_sparse_improved_sim_modified.cu"
#include <cuda.h>
#include "gpgpu-sim/flash/tensormap.h"
#include "ptx/tma.cuh"
#include "ptx/mbarrier.cuh"
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <random>
#include <stdexcept>
#include <vector>
using namespace flashgpu::test::ptx;

// Axis 0 is M (contiguous outputs); axis 1 is K (bitmap-selected rows).
// Threads never gather weights by selected indices: TMA transfers/zero-fills
// rectangular tiles, and threads consume them with regular shared addresses.
constexpr int TileM = 128, TileK = 16;
__global__ void sparse_tma_gemv(const sparse_tensormap_descriptor_t* descriptor,
                              const __nv_bfloat16* x, float* output,
                              int M, int K, int splits, uint16_t* audit) {
  __shared__ __align__(128) __nv_bfloat16 tile[TileM * TileK];
  __shared__ uint64_t barrier;
  int m = blockIdx.x * TileM + threadIdx.x;
  int tiles = (K + TileK - 1) / TileK;
  int per_split = (tiles + splits - 1) / splits;
  int begin = blockIdx.y * per_split;
  int end = min(begin + per_split, tiles);
  float sum = 0;
  for (int t = begin; t < end; ++t) {
    if (threadIdx.x == 0) {
      mbarrier_init(&barrier, 1);
      mbarrier_arrive_expect_tx(&barrier, sizeof(tile));
      asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
      cp_async_bulk_tensor_2d_load(smem_u32_addr(tile),
          reinterpret_cast<uint64_t>(descriptor), blockIdx.x * TileM,
          t * TileK, smem_u32_addr(&barrier));
      mbarrier_wait_parity(&barrier, 0);
      mbarrier_inval(&barrier);
    }
    __syncthreads();
    if (m < M) {
      for (int j = 0; j < TileK && t * TileK + j < K; ++j) {
        // Audit one complete weight column, including every skipped row.
        if (m == 0) audit[t*TileK+j] = __bfloat16_as_ushort(tile[j*TileM]);
        sum = fmaf(gemv_bf16_to_float(tile[j * TileM + threadIdx.x]),
                   gemv_bf16_to_float(x[t * TileK + j]), sum);
      }
    }
    __syncthreads();
  }
  if (m < M) atomicAdd(output + m, sum);
}

#define CUDA_OK(expr) do { cudaError_t e = (expr); if (e != cudaSuccess) \
  throw std::runtime_error(std::string(#expr) + ": " + cudaGetErrorString(e)); } while (0)
struct Allocations {
  std::vector<void*> pointers;
  template<class T> T* get(size_t n) {
    T* p = nullptr; CUDA_OK(cudaMalloc(reinterpret_cast<void**>(&p), n * sizeof(T)));
    pointers.push_back(p); return p;
  }
  ~Allocations() { for (void* p : pointers) cudaFree(p); }
};
void sync_kernel() { CUDA_OK(cudaGetLastError()); CUDA_OK(cudaDeviceSynchronize()); }

int main(int argc, char** argv) {
  try {
    if (!std::getenv("GPGPUSIM_SETUP_ENVIRONMENT_WAS_RUN"))
      throw std::runtime_error("Sparse TMA requires FlashGPU-Sim; use run.sh");
    int K = argc > 1 ? std::stoi(argv[1]) : 4096;
    int M = argc > 2 ? std::stoi(argv[2]) : 4096;
    int keep = argc > 3 ? std::stoi(argv[3]) : 512;
    int splits = argc > 4 ? std::stoi(argv[4]) : 16;
    unsigned seed = argc > 5 ? std::stoul(argv[5]) : 20261002;
    if (K < 1 || M < 1 || M % 8 || keep < 1 || keep > K || splits < 1)
      throw std::runtime_error("Usage: compare [K M keep splits seed]; M must be divisible by 8");
    std::mt19937 rng(seed);
    // Signed, nonconstant BF16-rounded values, deterministic across runs.
    auto sample = [&]() { return __float2bfloat16(float(int(rng() % 65537) - 32768) / 16384.0f); };
    std::vector<__nv_bfloat16> weights(size_t(K) * M), x(K), values(keep);
    for (auto& v : weights) v = sample();
    for (auto& v : x) v = sample();
    std::cout << "CASE K=" << K << " M=" << M << " keep=" << keep
              << " splits=" << splits << " seed=" << seed
              << " matrix_bytes=" << weights.size() * 2 << std::endl;
    Allocations a;
    auto dw = a.get<__nv_bfloat16>(weights.size());
    auto dx = a.get<__nv_bfloat16>(K);
    auto dv = a.get<__nv_bfloat16>(keep);
    auto di = a.get<int64_t>(keep);
    auto original = a.get<float>(M);
    auto tma = a.get<float>(M);
    auto workspace = a.get<int>(260);
    CUDA_OK(cudaMemcpy(dw, weights.data(), weights.size()*2, cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemcpy(dx, x.data(), K*2, cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemset(workspace, 0, 260*sizeof(int)));
    CUDA_OK(cudaMemset(tma, 0, M*sizeof(float)));
    rough_topk_create_histogram<<<(K+255)/256,256>>>(dx,K,keep,workspace,original,M);
    sync_kernel();
    rough_topk_find_exponent_cutoff<<<1,256>>>(workspace,keep,workspace+256,workspace+257);
    sync_kernel();
    rough_topk_collect<<<(K+255)/256,256>>>(dx,K,keep,workspace+256,workspace+257,
                                          workspace+259,workspace+258,di,dv);
    sync_kernel();
    std::vector<int64_t> indices(keep);
    CUDA_OK(cudaMemcpy(indices.data(),di,keep*sizeof(int64_t),cudaMemcpyDeviceToHost));
    CUDA_OK(cudaMemcpy(values.data(),dv,keep*2,cudaMemcpyDeviceToHost));
    std::vector<uint8_t> bitmap((K+7)/8,0);
    for (int i=0;i<keep;++i) {
      auto k=indices[i];
      if(k<0 || k>=K || (bitmap[k/8] & (1u<<(k%8))) ||
         __bfloat162float(values[i]) != __bfloat162float(x[k]))
        throw std::runtime_error("Invalid, duplicate, or mismatched top-k selection");
      bitmap[k/8] |= 1u<<(k%8);
    }
    std::cout << "PASS selection: " << keep << " unique indices and matching values" << std::endl;
    indexed_sparse_gemv_bf16<<<dim3(1,(M+255)/256,splits),256>>>(dw,dv,di,original,M,keep,splits);
    sync_kernel();
    auto db = a.get<uint8_t>(bitmap.size());
    CUDA_OK(cudaMemcpy(db,bitmap.data(),bitmap.size(),cudaMemcpyHostToDevice));
    sparse_tensormap_descriptor_t descriptor{};
    uint64_t dims[]={uint64_t(M),uint64_t(K)}, strides[]={uint64_t(M)*2};
    uint32_t box[]={TileM,TileK}, steps[]={1,1};
    if (cuTensorMapEncodeTiled(reinterpret_cast<CUtensorMap*>(&descriptor.base),
        CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,2,dw,dims,strides,box,steps,
        CU_TENSOR_MAP_INTERLEAVE_NONE,CU_TENSOR_MAP_SWIZZLE_NONE,
        CU_TENSOR_MAP_L2_PROMOTION_NONE,CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE)!=CUDA_SUCCESS)
      throw std::runtime_error("cuTensorMapEncodeTiled failed");
    descriptor.enable_sparse(1,reinterpret_cast<uint64_t>(db));
    auto audit=a.get<uint16_t>(K);
    auto dd=a.get<sparse_tensormap_descriptor_t>(1);
    CUDA_OK(cudaMemcpy(dd,&descriptor,sizeof(descriptor),cudaMemcpyHostToDevice));
    sparse_tma_gemv<<<dim3((M+TileM-1)/TileM,splits),TileM>>>(dd,dx,tma,M,K,splits,audit);
    sync_kernel();
    std::vector<uint16_t> bits(K);
    CUDA_OK(cudaMemcpy(bits.data(),audit,K*2,cudaMemcpyDeviceToHost));
    int bit_errors=0;
    for(int k=0;k<K;++k) {
      uint16_t expected=(bitmap[k/8] & (1u<<(k%8))) ? __bfloat16_as_ushort(weights[size_t(k)*M]) : 0;
      if(bits[k]!=expected && bit_errors++<12) std::cerr<<"TRANSFER k="<<k<<" got="<<bits[k]<<" expected="<<expected<<"\n";
    }
    std::cout<<"TRANSFER errors="<<bit_errors<<std::endl;
    std::vector<float> got_original(M),got_tma(M);
    CUDA_OK(cudaMemcpy(got_original.data(),original,M*sizeof(float),cudaMemcpyDeviceToHost));
    CUDA_OK(cudaMemcpy(got_tma.data(),tma,M*sizeof(float),cudaMemcpyDeviceToHost));
    double max_pair=0,max_original=0,max_tma=0; int failures=bit_errors;
    for(int m=0;m<M;++m) {
      double ref=0;
      for(int i=0;i<keep;++i)
        ref+=double(__bfloat162float(weights[size_t(indices[i])*M+m]))*__bfloat162float(values[i]);
      double eo=std::abs(got_original[m]-ref), et=std::abs(got_tma[m]-ref);
      double ep=std::abs(double(got_original[m])-got_tma[m]);
      max_original=std::max(max_original,eo); max_tma=std::max(max_tma,et); max_pair=std::max(max_pair,ep);
      // FP32 FMA/atomic reduction orders differ; compare against FP64 oracle.
      double tolerance=1e-3+1e-4*std::abs(ref);
      if(!std::isfinite(got_original[m]) || !std::isfinite(got_tma[m]) ||
         eo>tolerance || et>tolerance || ep>tolerance) {
        if(failures++<8) std::cerr << "Mismatch m="<<m<<" cpu="<<ref<<" indexed="<<got_original[m]<<" tma="<<got_tma[m]<<'\n';
      }
    }
    std::cout << std::setprecision(10) << "RESULT outputs="<<M<<" failures="<<failures
              <<" max_abs_indexed_vs_tma="<<max_pair<<" max_abs_indexed_vs_cpu="<<max_original
              <<" max_abs_tma_vs_cpu="<<max_tma<<" atol=0.001 rtol=0.0001\n";
    if(failures) return 1;
    std::cout << "PASS: sparse TMA matches unchanged no_malloc indexed GEMV and CPU reference\n";
    return 0;
  } catch(const std::exception& e) { std::cerr<<"FAIL: "<<e.what()<<'\n'; return 1; }
}
