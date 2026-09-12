// SPDX-License-Identifier: Apache-2.0
// R9V draft-only group-128 W2 coarse projection. Target Q6_K is untouched.
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>
#include <hip/hip_bfloat16.h>
#include <hip/hip_runtime.h>

using bf16 = __hip_bfloat16;

template <int waves>
__global__ void project_w2(const uint8_t* __restrict__ q,
                           const at::Half* __restrict__ scales,
                           const bf16* __restrict__ x, bf16* __restrict__ y,
                           int n, int k, int m) {
  const int lane = threadIdx.x & 31;
  const int row = blockIdx.x * waves + threadIdx.x / 32;
  const int token = blockIdx.y;
  if (row >= n || token >= m) return;

  float sum = 0.0f;
  for (int start = lane * 8; start < k; start += 256) {
    const float scale =
        float(scales[row * (k / 128) + start / 128]);
    // Eight 2-bit codes occupy one 16-bit word; start is always lane*8.
    const uint16_t packed =
        *reinterpret_cast<const uint16_t*>(q + row * (k / 4) + start / 4);
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      const int code = (packed >> (2 * j)) & 3;
      const float weight = float(2 * code - 3) * scale;
      sum = fmaf(float(x[token * k + start + j]), weight, sum);
    }
  }
#pragma unroll
  for (int delta = 16; delta; delta /= 2) {
    sum += __shfl_down(sum, delta, 32);
  }
  if (lane == 0) y[token * n + row] = bf16(sum);
}

torch::Tensor project_w2_rank(torch::Tensor packed, torch::Tensor scales,
                              torch::Tensor x) {
  TORCH_CHECK(packed.is_cuda() && scales.is_cuda() && x.is_cuda(),
              "W2 packed weights, scales, and input must be GPU tensors");
  TORCH_CHECK(packed.device() == x.device() && scales.device() == x.device(),
              "W2 packed weights, scales, and input must share a device");
  TORCH_CHECK(packed.scalar_type() == torch::kUInt8 &&
                  scales.scalar_type() == torch::kFloat16 &&
                  x.scalar_type() == torch::kBFloat16,
              "W2 expects uint8 packed weights, FP16 scales, and BF16 input");
  TORCH_CHECK(packed.dim() == 2 && scales.dim() == 2 && x.dim() == 2,
              "W2 expects matrix tensors");
  TORCH_CHECK(packed.is_contiguous() && scales.is_contiguous() &&
                  x.is_contiguous(),
              "W2 tensors must be contiguous");
  TORCH_CHECK(packed.size(0) == 124160 && x.size(1) == 2560 && x.size(0) == 1,
              "W2 draft projection requires M1 shape [124160,2560]");
  const int n = 124160, k = 2560, m = 1;
  TORCH_CHECK(packed.size(1) == k / 4 && scales.size(0) == n &&
                  scales.size(1) == k / 128,
              "W2 packed shape does not match [N,K]");

  c10::cuda::CUDAGuard guard(x.device());
  auto output = torch::empty({m, n}, x.options());
  const auto stream = at::cuda::getCurrentHIPStreamMasqueradingAsCUDA();
  constexpr int waves = 4;
  project_w2<waves><<<dim3((n + waves - 1) / waves, m), dim3(32 * waves),
                       0, stream>>>(
      packed.data_ptr<uint8_t>(), scales.data_ptr<at::Half>(),
      reinterpret_cast<const bf16*>(x.data_ptr()),
      reinterpret_cast<bf16*>(output.data_ptr()), n, k, m);
  AT_CUDA_CHECK(hipGetLastError());
  return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("project", &project_w2_rank,
        "Draft-only W2 coarse projection (HIP)");
}
