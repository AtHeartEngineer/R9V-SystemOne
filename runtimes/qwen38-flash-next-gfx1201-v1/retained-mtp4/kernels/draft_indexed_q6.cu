// SPDX-License-Identifier: Apache-2.0
// R9V gfx1201 specializations. GGUF quant primitives are supplied by the
// vLLM GGUF plugin; see the repository's THIRD_PARTY_NOTICES.md.

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

#include <hip/hip_bfloat16.h>
#include <hip/hip_runtime.h>

#include "hip_compat.h"
#include "gguf/ggml-common_hip.h"
#include "gguf/vecdotq_hip.cuh"

namespace {

using bf16 = __hip_bfloat16;

template <typename scalar_t>
__global__ void quantize_q8_1(const scalar_t* __restrict__ x,
                              block_q8_1* __restrict__ y, int cols,
                              int padded) {
  const int ix = blockIdx.x * blockDim.x + threadIdx.x;
  if (ix >= padded) return;
  const int vec = blockIdx.y;
  const int offset = vec * padded + ix;
  const int block = offset / QK8_1;
  const int iqs = offset % QK8_1;
  const float value = ix < cols ? static_cast<float>(x[vec * cols + ix]) : 0.0f;
  float amax = fabsf(value);
  float sum = value;
#pragma unroll
  for (int mask = 16; mask > 0; mask >>= 1) {
    amax = fmaxf(amax, __shfl_xor(amax, mask, 32));
    sum += __shfl_xor(sum, mask, 32);
  }
  const float scale = amax / 127.0f;
  y[block].qs[iqs] = amax == 0.0f ? 0 : static_cast<int8_t>(roundf(value / scale));
  if (iqs == 0) y[block].ds = __floats2half2_rn(scale, sum);
}


// Match actual pinned GGUF M1: one wave per row, ordered K-block accumulation.
// Coarse mode deliberately omits eight blocks, solely for draft ranking.
template<bool coarse, typename id_t>
__global__ void project(const block_q6_K* __restrict__ weight,
                        const block_q8_1* __restrict__ input,
                        const id_t* __restrict__ ids, bf16* __restrict__ output,
                        int rows, int outputs, int block0, int block1) {
  const int out_row=blockIdx.x;
  if(out_row>=outputs)return;
  const int64_t row=coarse ? static_cast<int64_t>(out_row) : static_cast<int64_t>(ids[out_row]);
  const int lane=threadIdx.x, tid=threadIdx.y*32+lane;
  if(row<0 || row>=rows){if(tid==0)output[out_row]=__float2bfloat16(-INFINITY);return;}
  constexpr int waves=1, blocks=10;
  constexpr int per_iter=VDR_Q6_K_Q8_1_MMVQ*waves*32/QI6_K;
  float sum=0.0f;
  for(int block=tid/(QI6_K/VDR_Q6_K_Q8_1_MMVQ);block<blocks;block+=per_iter){
    if constexpr(coarse){if(block!=block0 && block!=block1)continue;}
    const int iqs=VDR_Q6_K_Q8_1_MMVQ*(tid%(QI6_K/VDR_Q6_K_Q8_1_MMVQ));
    sum+=vec_dot_q6_K_q8_1(&weight[row*blocks+block],&input[block*(QK_K/QK8_1)],iqs);
  }
#pragma unroll
  for(int mask=16;mask>0;mask>>=1)sum+=__shfl_xor(sum,mask,32);
  if(lane==0)output[out_row]=__float2bfloat16(sum);
}
void check(torch::Tensor weight,torch::Tensor x){
 TORCH_CHECK(weight.is_cuda() && x.is_cuda() && weight.device()==x.device(),"weight and input must share a GPU");
 TORCH_CHECK(weight.scalar_type()==torch::kUInt8 && x.scalar_type()==torch::kBFloat16,"Q6_K uint8 weight and BF16 input required");
 TORCH_CHECK(weight.dim()==2 && weight.size(0)>0 && weight.size(0)<=2147483647 && weight.size(1)==2100,"expected nonempty Q6_K rows x2100bytes");
 TORCH_CHECK(x.dim()==2 && x.size(0)==1 && x.size(1)==2560,"input must be [1,2560]");
 TORCH_CHECK(weight.is_contiguous() && x.is_contiguous(),"contiguous tensors required");
}
torch::Tensor indexed(torch::Tensor weight,torch::Tensor x,torch::Tensor ids){
 check(weight,x);
 TORCH_CHECK(ids.is_cuda() && ids.device()==x.device() && ids.dim()==1 && ids.is_contiguous(),"IDs must be contiguous1D on input GPU");
 TORCH_CHECK(ids.scalar_type()==torch::kInt32 || ids.scalar_type()==torch::kInt64,"IDs must be int32/int64");
 TORCH_CHECK(ids.numel()<=2147483647,"too many IDs");
 c10::cuda::CUDAGuard guard(x.device());
 auto out=torch::empty({1,ids.numel()},x.options());if(ids.numel()==0)return out;
 auto q=torch::empty({1,80*9},x.options().dtype(torch::kInt32));
 const auto stream=at::cuda::getCurrentHIPStreamMasqueradingAsCUDA();
 quantize_q8_1<<<dim3(10,1,1),256,0,stream>>>(reinterpret_cast<const bf16*>(x.data_ptr()),reinterpret_cast<block_q8_1*>(q.data_ptr()),2560,2560);
 if(ids.scalar_type()==torch::kInt32)
  project<false,int32_t><<<dim3(ids.numel(),1,1),dim3(32,1,1),0,stream>>>(reinterpret_cast<const block_q6_K*>(weight.data_ptr()),reinterpret_cast<const block_q8_1*>(q.data_ptr()),ids.data_ptr<int32_t>(),reinterpret_cast<bf16*>(out.data_ptr()),weight.size(0),ids.numel(),0,0);
 else
  project<false,int64_t><<<dim3(ids.numel(),1,1),dim3(32,1,1),0,stream>>>(reinterpret_cast<const block_q6_K*>(weight.data_ptr()),reinterpret_cast<const block_q8_1*>(q.data_ptr()),ids.data_ptr<int64_t>(),reinterpret_cast<bf16*>(out.data_ptr()),weight.size(0),ids.numel(),0,0);
 AT_CUDA_CHECK(hipGetLastError());return out;
}
torch::Tensor coarse(torch::Tensor weight,torch::Tensor x,int64_t block0,int64_t block1){
 check(weight,x);
 TORCH_CHECK(block0>=0 && block0<10 && block1>=0 && block1<10 && block0!=block1,"two distinct Q6_K blocks in0..9 required");
 c10::cuda::CUDAGuard guard(x.device());
 auto out=torch::empty({1,weight.size(0)},x.options());
 auto q=torch::empty({1,80*9},x.options().dtype(torch::kInt32));
 const auto stream=at::cuda::getCurrentHIPStreamMasqueradingAsCUDA();
 quantize_q8_1<<<dim3(10,1,1),256,0,stream>>>(reinterpret_cast<const bf16*>(x.data_ptr()),reinterpret_cast<block_q8_1*>(q.data_ptr()),2560,2560);
 project<true,int32_t><<<dim3(weight.size(0),1,1),dim3(32,1,1),0,stream>>>(reinterpret_cast<const block_q6_K*>(weight.data_ptr()),reinterpret_cast<const block_q8_1*>(q.data_ptr()),nullptr,reinterpret_cast<bf16*>(out.data_ptr()),weight.size(0),weight.size(0),block0,block1);
 AT_CUDA_CHECK(hipGetLastError());return out;
}
} // namespace
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){m.def("indexed",&indexed);m.def("coarse",&coarse);}
