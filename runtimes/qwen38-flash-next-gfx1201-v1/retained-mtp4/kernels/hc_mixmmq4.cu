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
__global__ void quantize_hc4(const bf16* x,block_q8_1* y,int64_t stride) {
 const int col=blockIdx.x*256+threadIdx.x;
 const int token=blockIdx.y;
 float value=0.0f;
 if(col<320) {
  const float scaled=float(x[token*stride+col])*0.25f;
  const float sig=1.0f/(1.0f+exp2f(-scaled*1.4426950408889634f));
  value=float(__float2bfloat16(__fmul_rn(scaled,sig)));
 }
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
 bf16* __restrict__ y,const bf16* __restrict__ xn,int64_t stride_xn) {
 const int group=threadIdx.x/4;
 const int hcstream=group/16,inner=group%16;
 const int row=hcstream*2560+blockIdx.x*16+inner;
 __shared__ bf16 gates[4][4][16];
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
  for(int t=0;t<4;++t) {
   const bf16 value=__float2bfloat16(sums[t]);
   gates[t][hcstream][inner]=isfinite(float(value)) ? value : __float2bfloat16(0.0f);
  }
 }
 __syncthreads();
 if(threadIdx.x<64) {
  const int token=threadIdx.x/16,hidden=blockIdx.x*16+threadIdx.x%16;
  float mixed=0.0f;
  #pragma unroll
  for(int stream=0;stream<4;++stream) {
   const float gate=float(gates[token][stream][threadIdx.x%16]);
   const float sig=1.0f/(1.0f+exp2f(-gate*1.4426950408889634f));
   const float val=float(xn[token*stride_xn+stream*2560+hidden]);
   mixed=fmaf(sig,val,mixed);
  }
  y[token*2560+hidden]=__float2bfloat16(mixed*0.25f);
 }
}
}
torch::Tensor hc_mmq4(torch::Tensor w,torch::Tensor x,torch::Tensor xn) {
 TORCH_CHECK(w.is_cuda()&&x.is_cuda()&&w.device()==x.device(),"matching GPU tensors required");
 TORCH_CHECK(w.scalar_type()==torch::kUInt8&&x.scalar_type()==torch::kBFloat16,"Q8 packed weight/BF16 input required");
 TORCH_CHECK(w.dim()==2&&w.size(0)==10240&&w.size(1)==340&&w.is_contiguous(),"expected contiguous Q8_0 [10240,340]");
 TORCH_CHECK(x.dim()==2&&x.size(0)==4&&x.size(1)==320&&x.stride(1)==1,"expected contiguous BF16 [4,320]");
 TORCH_CHECK(xn.is_cuda()&&xn.device()==x.device()&&xn.scalar_type()==torch::kBFloat16&&xn.dim()==2&&xn.size(0)==4&&xn.size(1)==10240&&xn.stride(1)==1,"expected BF16 xn[4,10240] on matching device");
 c10::cuda::CUDAGuard guard(x.device());
 auto q=torch::empty({4,16*36},torch::TensorOptions().dtype(torch::kUInt8).device(x.device()));
 auto y=torch::empty({4,2560},x.options());
 auto stream=at::cuda::getCurrentHIPStreamMasqueradingAsCUDA();
 quantize_hc4<<<dim3(2,4),256,0,stream>>>(reinterpret_cast<const bf16*>(x.data_ptr()),reinterpret_cast<block_q8_1*>(q.data_ptr()),x.stride(0));
 hc_q8_mmq4_exact<<<160,256,0,stream>>>(reinterpret_cast<const block_q8_0*>(w.data_ptr()),reinterpret_cast<const block_q8_1*>(q.data_ptr()),reinterpret_cast<bf16*>(y.data_ptr()),reinterpret_cast<const bf16*>(xn.data_ptr()),xn.stride(0));
 AT_CUDA_CHECK(hipGetLastError());return y;
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){m.def("hc_mmq4",&hc_mmq4);}
