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

__device__ __forceinline__ void vec_dot_q8_0_q8_1_quint(
    const block_q8_0* __restrict__ weight,
    const block_q8_1* __restrict__ input0,
    const block_q8_1* __restrict__ input1,
    const block_q8_1* __restrict__ input2,
    const block_q8_1* __restrict__ input3,
    const block_q8_1* __restrict__ input4, int iqs,
    float& sum0, float& sum1, float& sum2, float& sum3, float& sum4) {
  int v[VDR_Q8_0_Q8_1_MMVQ];
  int u0[VDR_Q8_0_Q8_1_MMVQ];
  int u1[VDR_Q8_0_Q8_1_MMVQ];
  int u2[VDR_Q8_0_Q8_1_MMVQ];
  int u3[VDR_Q8_0_Q8_1_MMVQ];
  int u4[VDR_Q8_0_Q8_1_MMVQ];
#pragma unroll
  for (int i = 0; i < VDR_Q8_0_Q8_1_MMVQ; ++i) {
    v[i] = get_int_from_int8(weight->qs, iqs + i);
    u0[i] = get_int_from_int8_aligned(input0->qs, iqs + i);
    u1[i] = get_int_from_int8_aligned(input1->qs, iqs + i);
    u2[i] = get_int_from_int8_aligned(input2->qs, iqs + i);
    u3[i] = get_int_from_int8_aligned(input3->qs, iqs + i);
    u4[i] = get_int_from_int8_aligned(input4->qs, iqs + i);
  }
  const float dw = __half2float(weight->d);
  sum0 += vec_dot_q8_0_q8_1_impl<VDR_Q8_0_Q8_1_MMVQ>(
      v, u0, dw, __low2float(input0->ds));
  sum1 += vec_dot_q8_0_q8_1_impl<VDR_Q8_0_Q8_1_MMVQ>(
      v, u1, dw, __low2float(input1->ds));
  sum2 += vec_dot_q8_0_q8_1_impl<VDR_Q8_0_Q8_1_MMVQ>(
      v, u2, dw, __low2float(input2->ds));
  sum3 += vec_dot_q8_0_q8_1_impl<VDR_Q8_0_Q8_1_MMVQ>(
      v, u3, dw, __low2float(input3->ds));
  sum4 += vec_dot_q8_0_q8_1_impl<VDR_Q8_0_Q8_1_MMVQ>(
      v, u4, dw, __low2float(input4->ds));
}

template <int waves>
__global__ void dense_mmvq_q8_reuse5(
    const block_q8_0* __restrict__ weight,
    const block_q8_1* __restrict__ input,
    bf16* __restrict__ output, int cols, int rows) {
  const int row = blockIdx.x;
  if (row >= rows) return;
  const int lane = threadIdx.x;
  const int tid = threadIdx.y * 32 + lane;
  const int blocks_per_row = cols / QK8_0;
  const int blocks_per_iter = VDR_Q8_0_Q8_1_MMVQ * waves * 32 / QI8_0;
  const int padded = (cols + 511) / 512 * 512;
  const int input_stride = padded / QK8_1;
  float sum0 = 0.0f;
  float sum1 = 0.0f;
  float sum2 = 0.0f;
  float sum3 = 0.0f;
  float sum4 = 0.0f;
  for (int block = tid / (QI8_0 / VDR_Q8_0_Q8_1_MMVQ);
       block < blocks_per_row; block += blocks_per_iter) {
    const int iqs = VDR_Q8_0_Q8_1_MMVQ *
                    (tid % (QI8_0 / VDR_Q8_0_Q8_1_MMVQ));
    vec_dot_q8_0_q8_1_quint(
        &weight[row * blocks_per_row + block], &input[block],
        &input[input_stride + block], &input[2 * input_stride + block],
        &input[3 * input_stride + block],
        &input[4 * input_stride + block], iqs, sum0, sum1, sum2, sum3, sum4);
  }
  __shared__ float partials[5][waves > 1 ? waves - 1 : 1][32];
  if (threadIdx.y > 0) {
    partials[0][threadIdx.y - 1][lane] = sum0;
    partials[1][threadIdx.y - 1][lane] = sum1;
    partials[2][threadIdx.y - 1][lane] = sum2;
    partials[3][threadIdx.y - 1][lane] = sum3;
    partials[4][threadIdx.y - 1][lane] = sum4;
  }
  __syncthreads();
  if (threadIdx.y > 0) return;
#pragma unroll
  for (int wave = 0; wave < waves - 1; ++wave) {
    sum0 += partials[0][wave][lane];
    sum1 += partials[1][wave][lane];
    sum2 += partials[2][wave][lane];
    sum3 += partials[3][wave][lane];
    sum4 += partials[4][wave][lane];
  }
#pragma unroll
  for (int mask = 16; mask > 0; mask >>= 1) {
    sum0 += __shfl_xor(sum0, mask, 32);
    sum1 += __shfl_xor(sum1, mask, 32);
    sum2 += __shfl_xor(sum2, mask, 32);
    sum3 += __shfl_xor(sum3, mask, 32);
    sum4 += __shfl_xor(sum4, mask, 32);
  }
  if (lane == 0) {
    output[row] = __float2bfloat16(sum0);
    output[rows + row] = __float2bfloat16(sum1);
    output[2 * rows + row] = __float2bfloat16(sum2);
    output[3 * rows + row] = __float2bfloat16(sum3);
    output[4 * rows + row] = __float2bfloat16(sum4);
  }
}


}
torch::Tensor q8_mmvq5(torch::Tensor w,torch::Tensor x) {
 TORCH_CHECK(w.is_cuda()&&x.is_cuda()&&w.device()==x.device(),"matching GPU tensors required");
 TORCH_CHECK(w.scalar_type()==torch::kUInt8&&x.scalar_type()==torch::kBFloat16,"Q8 packed weight/BF16 input required");
 TORCH_CHECK(w.dim()==2&&x.dim()==2&&x.size(0)==5&&w.is_contiguous()&&x.is_contiguous(),"contiguous five-row inputs required");
 const int n=w.size(0),k=x.size(1),padded=(k+511)/512*512;
 TORCH_CHECK((n==640&&k==2560)||(n==2560&&k==320)||(n==2560&&k==3072),"only TP shared-expert shapes allowed");
 TORCH_CHECK(w.size(1)==k/32*34,"incorrect packed weight stride");
 c10::cuda::CUDAGuard guard(x.device());
 auto q=torch::empty({5,padded/32*36},torch::TensorOptions().dtype(torch::kUInt8).device(x.device()));
 auto y=torch::empty({5,n},x.options());
 auto stream=at::cuda::getCurrentHIPStreamMasqueradingAsCUDA();
 quantize_q8_1<<<dim3((padded+255)/256,5),256,0,stream>>>(reinterpret_cast<const bf16*>(x.data_ptr()),reinterpret_cast<block_q8_1*>(q.data_ptr()),k,padded);
 dense_mmvq_q8_reuse5<1><<<n,dim3(32,1),0,stream>>>(reinterpret_cast<const block_q8_0*>(w.data_ptr()),reinterpret_cast<const block_q8_1*>(q.data_ptr()),reinterpret_cast<bf16*>(y.data_ptr()),k,n);
 AT_CUDA_CHECK(hipGetLastError());return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){m.def("q8_mmvq5",&q8_mmvq5);}
