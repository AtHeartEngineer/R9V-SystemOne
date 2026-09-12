// SPDX-License-Identifier: Apache-2.0
// Exact small-M Q8_0 projection. GGUF primitives retain their notices.
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>
#include <hip/hip_runtime.h>
#include <hip/hip_bfloat16.h>
#include "hip_compat.h"
#include "gguf/ggml-common_hip.h"
#include "gguf/vecdotq_hip.cuh"
namespace {
using bf16=__hip_bfloat16;
__global__ void quantize_hc4(const bf16* x,block_q8_1* y) {
 const int col=blockIdx.x*256+threadIdx.x;
 const int token=blockIdx.y;
 const float value=col<320?float(x[token*320+col]):0.0f;
 float amax=fabsf(value),sum=value;
 #pragma unroll
 for(int mask=16;mask>0;mask>>=1) {
  amax=fmaxf(amax,__shfl_xor(amax,mask,32));sum+=__shfl_xor(sum,mask,32);
 }
 const float scale=amax/127.0f;
 auto* out=y+token*16+col/32;
 out->qs[col%32]=amax==0.0f?0:static_cast<int8_t>(roundf(value/scale));
 if(col%32==0)out->ds=__floats2half2_rn(scale,sum);
}
// Four lanes form the complete integer dot product for each 32-element block.
// Only then scale to FP32 and add blocks in ascending order, as GGUF MMQ does.
// Each packed weight load is reused for all four actual token rows.
__global__ __launch_bounds__(256) void hc_q8_mmq4_exact(
 const block_q8_0* __restrict__ w,const block_q8_1* __restrict__ x,
 bf16* __restrict__ y) {
 const int row=blockIdx.x*64+threadIdx.x/4;
 const int lane=threadIdx.x%4;
 float sums[4]={};
 #pragma unroll
 for(int b=0;b<10;++b) {
  const auto* wb=w+row*10+b;
  const int w0=get_int_from_int8(wb->qs,lane*2);
  const int w1=get_int_from_int8(wb->qs,lane*2+1);
  const float dw=__half2float(wb->d);
  #pragma unroll
  for(int t=0;t<4;++t) {
   const auto* xb=x+t*16+b;
   int acc=__dp4a(w0,get_int_from_int8_aligned(xb->qs,lane*2),0);
   acc=__dp4a(w1,get_int_from_int8_aligned(xb->qs,lane*2+1),acc);
   acc+=__shfl_xor(acc,2,4);acc+=__shfl_xor(acc,1,4);
   const float dx=__low2float(xb->ds);
   sums[t]+=dw*dx*acc;
  }
 }
 if(lane==0) {
  #pragma unroll
  for(int t=0;t<4;++t)y[t*10240+row]=__float2bfloat16(sums[t]);
 }
}
}
torch::Tensor hc_mmq4(torch::Tensor w,torch::Tensor x) {
 TORCH_CHECK(w.is_cuda()&&x.is_cuda()&&w.device()==x.device(),"matching GPU tensors required");
 TORCH_CHECK(w.scalar_type()==torch::kUInt8&&x.scalar_type()==torch::kBFloat16,"Q8 packed weight/BF16 input required");
 TORCH_CHECK(w.dim()==2&&w.size(0)==10240&&w.size(1)==340&&w.is_contiguous(),"expected contiguous Q8_0 [10240,340]");
 TORCH_CHECK(x.dim()==2&&x.size(0)==4&&x.size(1)==320&&x.is_contiguous(),"expected contiguous BF16 [4,320]");
 c10::cuda::CUDAGuard guard(x.device());
 auto q=torch::empty({4,16*36},torch::TensorOptions().dtype(torch::kUInt8).device(x.device()));
 auto y=torch::empty({4,10240},x.options());
 auto stream=at::cuda::getCurrentHIPStreamMasqueradingAsCUDA();
 quantize_hc4<<<dim3(2,4),256,0,stream>>>(reinterpret_cast<const bf16*>(x.data_ptr()),reinterpret_cast<block_q8_1*>(q.data_ptr()));
 hc_q8_mmq4_exact<<<160,256,0,stream>>>(reinterpret_cast<const block_q8_0*>(w.data_ptr()),reinterpret_cast<const block_q8_1*>(q.data_ptr()),reinterpret_cast<bf16*>(y.data_ptr()));
 AT_CUDA_CHECK(hipGetLastError());return y;
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){m.def("hc_mmq4",&hc_mmq4);}
