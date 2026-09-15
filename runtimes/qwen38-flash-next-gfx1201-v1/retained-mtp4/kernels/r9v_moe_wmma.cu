// SPDX-License-Identifier: Apache-2.0
// Expert-grouped GGUF MoE prefill on gfx12 int8 WMMA (wave32).
//
// Same contract as tiered_iq_moe_prefill_grouped: packed GGUF expert rows live
// in VRAM (hot), the decode cache (cache) or pinned host memory (cold); routes
// are pre-grouped per expert by vLLM's alignment kernel.  One workgroup owns
// one group of 32 routes of one expert and a 128-row output tile.  Each wave
// owns 16 rows; every lane dequantizes eight int8 weights per 16-wide K step
// directly into its WMMA A operand, so packed weights are read once per group
// and never staged through LDS.  Activations are the Q8_1 blocks produced by
// the reference quantizer; their per-32 scales and the per-row sub-block
// scales are applied in fp32 after every 32-wide K step.
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

#include <hip/hip_bfloat16.h>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <cstdint>

#include "hip_compat.h"
#include "gguf/ggml-common_hip.h"
#include "gguf/vecdotq_hip.cuh"

namespace {

using bf16 = __hip_bfloat16;
typedef int v2i __attribute__((ext_vector_type(2)));
typedef int v8i __attribute__((ext_vector_type(8)));

constexpr int kWave = 32;
constexpr int kWaves = 4;
constexpr int kThreads = kWave * kWaves;
constexpr int kGroup = 32;             // routes per workgroup (two 16-wide N tiles)
constexpr int kRowsPerWave = 16;
constexpr int kRowsPerBlock = kWaves * kRowsPerWave;

template <typename scalar_t>
__global__ void quantize_q8_1_rows(const scalar_t* __restrict__ x,
                                   block_q8_1* __restrict__ y, int kx,
                                   int kx_padded) {
  // Same arithmetic as the reference quantizer; rows are in grid.x so the
  // row count is not limited to 65535.
  const int ix = blockDim.x * blockIdx.y + threadIdx.x;
  if (ix >= kx_padded) return;
  const int iy = blockIdx.x;
  const int i_padded = iy * kx_padded + ix;
  const int ib = i_padded / QK8_1;
  const int iqs = i_padded % QK8_1;
  const float value = ix < kx ? static_cast<float>(x[iy * kx + ix]) : 0.0f;
  float amax = fabsf(value);
  float sum = value;
#pragma unroll
  for (int mask = 16; mask > 0; mask >>= 1) {
    amax = fmaxf(amax, __shfl_xor(amax, mask, 32));
    sum += __shfl_xor(sum, mask, 32);
  }
  const float scale = amax / 127.0f;
  y[ib].qs[iqs] = amax == 0.0f ? 0 : static_cast<int8_t>(roundf(value / scale));
  if (iqs == 0) y[ib].ds = __floats2half2_rn(scale, sum);
}

__device__ __forceinline__ uint32_t load_u32_unaligned(const uint8_t* p) {
  uint32_t v;
  __builtin_memcpy(&v, p, 4);
  return v;
}

__device__ __forceinline__ uint16_t load_u16_unaligned(const uint8_t* p) {
  uint16_t v;
  __builtin_memcpy(&v, p, 2);
  return v;
}

// Weight dequantizers.  For K step `kstep` (32 columns) and lane half `h`
// (0/1) they return the eight int8 weights of columns [8h, 8h+8) as `lo`
// (WMMA sub-step 0) and of columns [16+8h, 16+8h+8) as `hi` (sub-step 1),
// plus the fp32 scale of this row's 32-column block.
struct DequantIQ4XS {
  static constexpr int kBlockBytes = 136;   // sizeof(block_iq4_xs)
  static constexpr int kBlockCols = 256;
  __device__ __forceinline__ static void load(const uint8_t* row, int kstep,
                                              int h, v2i& lo, v2i& hi,
                                              float& scale) {
    const uint8_t* block = row + (kstep >> 3) * kBlockBytes;
    const int sub = kstep & 7;
    const uint8_t* q = block + 8 + 16 * sub + 8 * h;
    const uint8_t* values = reinterpret_cast<const uint8_t*>(kvalues_iq4nl);
    int v1, v2;
    get_int_from_table_16(load_u32_unaligned(q), values, v1, v2);
    lo[0] = v1;
    hi[0] = v2;
    get_int_from_table_16(load_u32_unaligned(q + 4), values, v1, v2);
    lo[1] = v1;
    hi[1] = v2;
    const uint16_t scales_h = load_u16_unaligned(block + 2);
    const int ls = ((block[4 + (sub >> 1)] >> (4 * (sub & 1))) & 0xF) |
                   (((scales_h >> (2 * sub)) & 3) << 4);
    const half d = *reinterpret_cast<const half*>(block);
    scale = __half2float(d) * static_cast<float>(ls - 32);
  }
};

struct DequantIQ3S {
  static constexpr int kBlockBytes = 110;   // sizeof(block_iq3_s)
  static constexpr int kBlockCols = 256;
  __device__ __forceinline__ static int pair(const uint8_t* qs,
                                             const uint8_t qh,
                                             const uint8_t sign_nibble,
                                             int l, int which) {
    // which == 0: grid entry qs[2l] with qh bit (8-2l); which == 1: qs[2l+1] with bit (7-2l)
    const int index = which == 0 ? (qs[2 * l] | ((qh << (8 - 2 * l)) & 256))
                                 : (qs[2 * l + 1] | ((qh << (7 - 2 * l)) & 256));
    const uint32_t grid = iq3xs_grid[index];
    const uint32_t signs = __vcmpeq4(((sign_nibble & 0xF) * 0x01010101) & 0x08040201,
                                     0x08040201);
    return __vsub4(grid ^ signs, signs);
  }
  __device__ __forceinline__ static void load(const uint8_t* row, int kstep,
                                              int h, v2i& lo, v2i& hi,
                                              float& scale) {
    const uint8_t* block = row + (kstep >> 3) * kBlockBytes;
    const int sub = kstep & 7;
    const uint8_t* qs = block + 2 + 8 * sub;
    const uint8_t qh = block[66 + sub];
    const uint8_t* signs = block + 74 + 4 * sub;
    {
      const int l = h;                          // sub-step 0: values [8h, 8h+8)
      lo[0] = pair(qs, qh, signs[l] & 0xF, l, 0);
      lo[1] = pair(qs, qh, signs[l] >> 4, l, 1);
    }
    {
      const int l = 2 + h;                      // sub-step 1: values [16+8h, 16+8h+8)
      hi[0] = pair(qs, qh, signs[l] & 0xF, l, 0);
      hi[1] = pair(qs, qh, signs[l] >> 4, l, 1);
    }
    const int ls = (block[106 + (sub >> 1)] >> (4 * (sub & 1))) & 0xF;
    const half d = *reinterpret_cast<const half*>(block);
    scale = __half2float(d) * (0.5f + static_cast<float>(ls)) * 0.5f;
  }
};

struct DequantIQ4NL {
  static constexpr int kBlockBytes = 18;    // sizeof(block_iq4_nl)
  static constexpr int kBlockCols = 32;
  __device__ __forceinline__ static void load(const uint8_t* row, int kstep,
                                              int h, v2i& lo, v2i& hi,
                                              float& scale) {
    const uint8_t* block = row + kstep * kBlockBytes;
    const uint8_t* q = block + 2 + 8 * h;
    const uint8_t* values = reinterpret_cast<const uint8_t*>(kvalues_iq4nl);
    int v1, v2;
    get_int_from_table_16(load_u32_unaligned(q), values, v1, v2);
    lo[0] = v1;
    hi[0] = v2;
    get_int_from_table_16(load_u32_unaligned(q + 4), values, v1, v2);
    lo[1] = v1;
    hi[1] = v2;
    scale = __half2float(*reinterpret_cast<const half*>(block));
  }
};

struct DequantQ80 {
  static constexpr int kBlockBytes = 34;    // sizeof(block_q8_0)
  static constexpr int kBlockCols = 32;
  __device__ __forceinline__ static void load(const uint8_t* row, int kstep,
                                              int h, v2i& lo, v2i& hi,
                                              float& scale) {
    const uint8_t* block = row + kstep * kBlockBytes;
    const uint8_t* q = block + 2 + 8 * h;
    lo[0] = static_cast<int>(load_u32_unaligned(q));
    lo[1] = static_cast<int>(load_u32_unaligned(q + 4));
    hi[0] = static_cast<int>(load_u32_unaligned(q + 16));
    hi[1] = static_cast<int>(load_u32_unaligned(q + 20));
    scale = __half2float(*reinterpret_cast<const half*>(block));
  }
};

template <typename Dequant>
__global__ __launch_bounds__(kThreads) void moe_prefill_wmma(
    const uint8_t* __restrict__ cold_weight,
    const uint8_t* __restrict__ hot_weight,
    const uint8_t* __restrict__ cache_weight,
    const int* __restrict__ hot_map, const int* __restrict__ cold_map,
    const int* __restrict__ cache_map, const int* __restrict__ input,
    bf16* __restrict__ output, const int* __restrict__ sorted_route_ids,
    const int* __restrict__ block_expert_ids,
    const int* __restrict__ num_routes_post_padded, int num_experts,
    int hot_count, int cold_count, int cache_count, int top_k, int cols,
    int nrows, int token_stride_ints, int output_groups,
    int sorted_route_capacity) {
  __shared__ const uint8_t* selected_weight;
  __shared__ int selected_routes[kGroup];
  __shared__ int selected_valid;

  const int grouped_block = blockIdx.y;
  const int route_base = grouped_block * kGroup;
  if (route_base >= num_routes_post_padded[0] ||
      route_base >= sorted_route_capacity) {
    return;
  }
  if (threadIdx.x < kGroup) {
    const int slot = route_base + threadIdx.x;
    const int route = slot < sorted_route_capacity ? sorted_route_ids[slot] : -1;
    selected_routes[threadIdx.x] =
        route >= 0 && route < output_groups ? route : -1;
  }
  if (threadIdx.x == 0) {
    selected_weight = nullptr;
    selected_valid = 0;
    const int expert = block_expert_ids[grouped_block];
    if (expert >= 0 && expert < num_experts) {
      const int64_t expert_bytes =
          static_cast<int64_t>(nrows) * (cols / Dequant::kBlockCols) *
          Dequant::kBlockBytes;
      const int hot_slot = hot_map[expert];
      const int cache_slot = cache_count > 0 ? cache_map[expert] : -1;
      const int cold_slot = cold_map[expert];
      if (hot_slot >= 0 && hot_slot < hot_count) {
        selected_weight = hot_weight + hot_slot * expert_bytes;
        selected_valid = 1;
      } else if (cache_slot >= 0 && cache_slot < cache_count) {
        selected_weight = cache_weight + cache_slot * expert_bytes;
        selected_valid = 1;
      } else if (cold_slot >= 0 && cold_slot < cold_count) {
        selected_weight = cold_weight + cold_slot * expert_bytes;
        selected_valid = 1;
      }
    }
  }
  __syncthreads();
  if (!selected_valid) return;

  const int lane = threadIdx.x & (kWave - 1);
  const int wave = threadIdx.x / kWave;
  const int half = lane >> 4;            // which 8-wide K slice of the 16-wide sub-step
  const int idx = lane & 15;             // row within the M tile / token within the N tile
  const int row0 = blockIdx.x * kRowsPerBlock + wave * kRowsPerWave;
  // M tiles never straddle nrows (a multiple of 16); waves past the end still
  // take part in the block-wide barriers below and simply stage/compute nothing.
  const bool wave_active = row0 < nrows;

  // Activation rows for this lane's two tokens (one per N tile); -1 = padding.
  const int* act[2];
#pragma unroll
  for (int n = 0; n < 2; ++n) {
    const int route = selected_routes[n * 16 + idx];
    act[n] = route >= 0 ? input + static_cast<int64_t>(route / top_k) * token_stride_ints
                        : nullptr;
  }

  const int64_t row_bytes = static_cast<int64_t>(cols / Dequant::kBlockCols) *
                            Dequant::kBlockBytes;
  // Every wave stages the 256-column span of its 16 rows through LDS with
  // contiguous per-row dword copies, so packed bytes (VRAM, cache or pinned
  // host memory over PCIe) are fetched once and in cache-line order.  Blocks
  // are only 2-byte aligned, so each row is copied from the enclosing
  // dword-aligned range and read back at its 0/2-byte offset.
  // Packed rows are staged through LDS in fetches of kFetchCols columns.  A
  // fetch copies, for each of the wave's 16 rows, the contiguous byte range of
  // those columns from its enclosing 16-byte-aligned range, lanes walking the
  // row in address order: pinned host memory over PCIe only streams near link
  // rate when consecutive loads continue along a row.  The loads of fetch f+1
  // are issued into registers before fetch f is computed so cold-expert round
  // trips overlap the WMMA work.  Only the last rows of the last expert can run
  // past the tensor; they take the byte path.
  constexpr int kSpan = Dequant::kBlockBytes * (256 / Dequant::kBlockCols);
  constexpr int kFetchSpans = kSpan > 200 ? 1 : 2;
  constexpr int kFetchCols = kFetchSpans * 256;
  constexpr int kFetchBytes = kFetchSpans * kSpan;
  constexpr int kStageRow = (kFetchBytes + 15 + 15) / 16 * 16;  // room for a 16-byte alignment shift
  constexpr int kQuadsPerRow = kStageRow / 16;
  constexpr int kLoadsPerLane = (kRowsPerWave * kQuadsPerRow + kWave - 1) / kWave;
  __shared__ __align__(16) uint8_t staged[kWaves][kRowsPerWave * kStageRow];
  uint8_t* wave_stage = staged[wave];
  const uint8_t* expert_end = selected_weight + static_cast<int64_t>(nrows) * row_bytes;

  float acc[2][8];
#pragma unroll
  for (int n = 0; n < 2; ++n)
#pragma unroll
    for (int i = 0; i < 8; ++i) acc[n][i] = 0.0f;

  const int ksteps = cols / 32;
  const int fetches = (ksteps + kFetchCols / 32 - 1) / (kFetchCols / 32);
  const int64_t wave_end = static_cast<int64_t>(row0 + kRowsPerWave) * row_bytes;
  const bool interior = wave_active &&
      selected_weight + wave_end + 32 <= expert_end;   // last quad may run 30 bytes past a row end
  uint4 values[kLoadsPerLane];
  auto fetch = [&](int f) {
    const int fetch_bytes = min(kFetchBytes, static_cast<int>(row_bytes - static_cast<int64_t>(f) * kFetchBytes));
    const int quads = (fetch_bytes + 15 + 15) / 16;
    const int total = kRowsPerWave * quads;
#pragma unroll
    for (int j = 0; j < kLoadsPerLane; ++j) {
      const int unit = lane + j * kWave;
      const int r = unit / quads;
      const int q = unit - r * quads;
      const int64_t row_start = static_cast<int64_t>(row0 + r) * row_bytes + static_cast<int64_t>(f) * kFetchBytes;
      const uint8_t* src = selected_weight + (row_start & ~static_cast<int64_t>(15)) + 16 * q;
      if (interior) {
        values[j] = unit < total ? *reinterpret_cast<const uint4*>(src) : make_uint4(0, 0, 0, 0);
      } else {
        uint32_t words[4] = {0, 0, 0, 0};
        if (wave_active && unit < total) {
          for (int b = 0; b < 16 && src + b < expert_end; ++b) words[b / 4] |= static_cast<uint32_t>(src[b]) << (8 * (b % 4));
        }
        values[j] = make_uint4(words[0], words[1], words[2], words[3]);
      }
    }
  };
  auto stage = [&](int f) {
    const int fetch_bytes = min(kFetchBytes, static_cast<int>(row_bytes - static_cast<int64_t>(f) * kFetchBytes));
    const int quads = (fetch_bytes + 15 + 15) / 16;
    const int total = kRowsPerWave * quads;
#pragma unroll
    for (int j = 0; j < kLoadsPerLane; ++j) {
      const int unit = lane + j * kWave;
      if (wave_active && unit < total) {
        const int r = unit / quads;
        const int q = unit - r * quads;
        *reinterpret_cast<uint4*>(wave_stage + r * kStageRow + 16 * q) = values[j];
      }
    }
  };

  fetch(0);
  stage(0);
  for (int f = 0; f < fetches; ++f) {
    __syncthreads();                      // staged fetch visible to the whole wave
    if (f + 1 < fetches) fetch(f + 1);
    const int shift = static_cast<int>((static_cast<int64_t>(row0 + idx) * row_bytes + static_cast<int64_t>(f) * kFetchBytes) & 15);
    const uint8_t* staged_row = wave_stage + idx * kStageRow + shift;
    const int kstep_begin = f * (kFetchCols / 32);
    const int kstep_end = wave_active ? min(ksteps, kstep_begin + kFetchCols / 32) : 0;
    for (int kstep = kstep_begin; kstep < kstep_end; ++kstep) {
      v2i a_lo, a_hi;
      float row_scale;
      Dequant::load(staged_row, kstep - kstep_begin, half, a_lo, a_hi, row_scale);
      // Scales of the eight rows this lane accumulates: rows 8*half + i.
      float scales[8];
#pragma unroll
      for (int i = 0; i < 8; ++i) scales[i] = __shfl(row_scale, 8 * half + i, kWave);

#pragma unroll
      for (int n = 0; n < 2; ++n) {
        v2i b_lo = {0, 0}, b_hi = {0, 0};
        float d8 = 0.0f;
        if (act[n] != nullptr) {
          // block_q8_1 = { half2 ds; int8_t qs[32]; }: scales first, then 8 ints of quants.
          const int* block = act[n] + kstep * 9;
          d8 = __low2float(*reinterpret_cast<const half2*>(block));
          b_lo[0] = block[1 + 2 * half];
          b_lo[1] = block[2 + 2 * half];
          b_hi[0] = block[5 + 2 * half];
          b_hi[1] = block[6 + 2 * half];
        }
        v8i c = {0, 0, 0, 0, 0, 0, 0, 0};
        c = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(true, a_lo, true, b_lo, c, false);
        c = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(true, a_hi, true, b_hi, c, false);
        // c[i] = dot(row 8*half+i, token idx) over this 32-wide K step; d8 is the
        // scale of this lane's token, which is exactly the accumulator's column.
#pragma unroll
        for (int i = 0; i < 8; ++i) acc[n][i] += static_cast<float>(c[i]) * scales[i] * d8;
      }
    }
    if (f + 1 < fetches) {
      __syncthreads();                    // every wave finished reading this fetch
      stage(f + 1);
    }
  }

  // Store: lane owns column (token idx) and rows 8*half .. 8*half+7 of the tile.
#pragma unroll
  for (int n = 0; n < 2; ++n) {
    const int route = selected_routes[n * 16 + idx];
    if (!wave_active || route < 0) continue;
    bf16* out = output + static_cast<int64_t>(route) * nrows + row0 + 8 * half;
    uint32_t packed[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const bf16 lo_v = __float2bfloat16(acc[n][2 * i]);
      const bf16 hi_v = __float2bfloat16(acc[n][2 * i + 1]);
      packed[i] = static_cast<uint32_t>(*reinterpret_cast<const uint16_t*>(&lo_v)) |
                  (static_cast<uint32_t>(*reinterpret_cast<const uint16_t*>(&hi_v)) << 16);
    }
    *reinterpret_cast<uint4*>(out) = make_uint4(packed[0], packed[1], packed[2], packed[3]);
  }
}

template <typename Dequant>
void launch(const torch::Tensor& cold_weight, const torch::Tensor& hot_weight,
            const torch::Tensor* cache_weight, const torch::Tensor& hot_map,
            const torch::Tensor& cold_map, const torch::Tensor* cache_map,
            const torch::Tensor& quantized, torch::Tensor& output,
            const torch::Tensor& sorted_route_ids,
            const torch::Tensor& block_expert_ids,
            const torch::Tensor& num_routes_post_padded, int top_k, int cols,
            int nrows, int tokens, hipStream_t stream) {
  const dim3 grid((nrows + kRowsPerBlock - 1) / kRowsPerBlock,
                  block_expert_ids.numel(), 1);
  moe_prefill_wmma<Dequant><<<grid, kThreads, 0, stream>>>(
      cold_weight.data_ptr<uint8_t>(), hot_weight.data_ptr<uint8_t>(),
      cache_weight == nullptr ? nullptr : cache_weight->data_ptr<uint8_t>(),
      hot_map.data_ptr<int>(), cold_map.data_ptr<int>(),
      cache_map == nullptr ? nullptr : cache_map->data_ptr<int>(),
      quantized.data_ptr<int>(), reinterpret_cast<bf16*>(output.data_ptr()),
      sorted_route_ids.data_ptr<int>(), block_expert_ids.data_ptr<int>(),
      num_routes_post_padded.data_ptr<int>(), hot_map.numel(),
      hot_weight.size(0), cold_weight.size(0),
      cache_weight == nullptr ? 0 : cache_weight->size(0), top_k, cols, nrows,
      quantized.stride(0), tokens * top_k, sorted_route_ids.numel());
}

torch::Tensor prefill_impl(torch::Tensor x, torch::Tensor cold_weight,
                           torch::Tensor hot_weight, torch::Tensor hot_map,
                           torch::Tensor cold_map,
                           torch::Tensor sorted_route_ids,
                           torch::Tensor block_expert_ids,
                           torch::Tensor num_routes_post_padded, int64_t top_k,
                           int64_t qtype, int64_t rows, int64_t tokens,
                           int64_t group_size,
                           const torch::Tensor* cache_weight,
                           const torch::Tensor* cache_map) {
  TORCH_CHECK(x.is_cuda() && cold_weight.is_cuda() && hot_weight.is_cuda() &&
                  hot_map.is_cuda() && cold_map.is_cuda() &&
                  sorted_route_ids.is_cuda() && block_expert_ids.is_cuda() &&
                  num_routes_post_padded.is_cuda(),
              "all grouped-prefill tensors must be GPU or UVA tensors");
  TORCH_CHECK(x.scalar_type() == torch::kBFloat16, "input must be BF16");
  TORCH_CHECK(cold_weight.scalar_type() == torch::kUInt8 &&
                  hot_weight.scalar_type() == torch::kUInt8,
              "weights must contain packed GGUF bytes");
  TORCH_CHECK(hot_map.scalar_type() == torch::kInt32 &&
                  cold_map.scalar_type() == torch::kInt32 &&
                  sorted_route_ids.scalar_type() == torch::kInt32 &&
                  block_expert_ids.scalar_type() == torch::kInt32 &&
                  num_routes_post_padded.scalar_type() == torch::kInt32,
              "grouped-prefill routing tensors must be int32");
  TORCH_CHECK(x.is_contiguous() && cold_weight.is_contiguous() &&
                  hot_weight.is_contiguous() && hot_map.is_contiguous() &&
                  cold_map.is_contiguous() && sorted_route_ids.is_contiguous() &&
                  block_expert_ids.is_contiguous() &&
                  num_routes_post_padded.is_contiguous(),
              "all grouped-prefill tensors must be contiguous");
  TORCH_CHECK(qtype == 8 || qtype == 20 || qtype == 21 || qtype == 23,
              "WMMA prefill supports Q8_0, IQ4_NL, IQ3_S and IQ4_XS");
  TORCH_CHECK(group_size == kGroup, "WMMA prefill group size must be 32");
  TORCH_CHECK(top_k > 0 && tokens > 0 && rows > 0 && rows % kRowsPerWave == 0,
              "rows must be a positive multiple of 16");
  TORCH_CHECK(cold_weight.dim() == 3 && hot_weight.dim() == 3 &&
                  cold_weight.size(1) == rows && hot_weight.size(1) == rows &&
                  cold_weight.size(2) == hot_weight.size(2),
              "packed expert weights must be [experts, rows, bytes] with equal strides");
  TORCH_CHECK(hot_map.numel() == cold_map.numel(),
              "hot_map and cold_map must cover the same logical experts");
  TORCH_CHECK(x.size(0) >= tokens, "input does not cover every token");
  TORCH_CHECK(sorted_route_ids.numel() > 0 && block_expert_ids.numel() > 0 &&
                  block_expert_ids.numel() <= 65535,
              "grouped-prefill block count exceeds HIP grid-y capacity");
  const auto device = x.device();
  if (cache_weight != nullptr || cache_map != nullptr) {
    TORCH_CHECK(cache_weight != nullptr && cache_map != nullptr &&
                    cache_weight->is_cuda() && cache_map->is_cuda() &&
                    cache_weight->scalar_type() == torch::kUInt8 &&
                    cache_map->scalar_type() == torch::kInt32 &&
                    cache_weight->is_contiguous() && cache_map->is_contiguous() &&
                    cache_weight->dim() == 3 && cache_weight->size(1) == rows &&
                    cache_weight->size(2) == cold_weight.size(2) &&
                    cache_map->numel() == hot_map.numel() &&
                    cache_weight->device() == device && cache_map->device() == device,
                "cache tensors must match the packed expert layout");
  }
  const int64_t cols = x.size(1);
  const int64_t block_cols = qtype == 8 || qtype == 20 ? 32 : 256;
  const int64_t block_bytes = qtype == 8 ? 34 : qtype == 20 ? 18 : qtype == 21 ? 110 : 136;
  TORCH_CHECK(cols % block_cols == 0 && cols % 32 == 0,
              "input width is not aligned to the GGUF block");
  TORCH_CHECK(cold_weight.size(2) == (cols / block_cols) * block_bytes,
              "packed row byte width does not match qtype and input width");

  c10::cuda::CUDAGuard guard(device);
  const int64_t padded = (cols + 511) / 512 * 512;
  auto quantized = torch::empty({tokens, padded / 32 * 9},
                                torch::TensorOptions().dtype(torch::kInt32).device(device));
  auto output = torch::zeros({tokens * top_k, rows}, x.options());
  const auto stream = at::cuda::getCurrentHIPStreamMasqueradingAsCUDA();
  // Rows in grid.x so a large token count never exceeds the 65535 grid.y limit.
  const dim3 quant_grid(tokens, (padded + 255) / 256, 1);
  quantize_q8_1_rows<<<quant_grid, 256, 0, stream>>>(
      reinterpret_cast<const bf16*>(x.data_ptr()),
      reinterpret_cast<block_q8_1*>(quantized.data_ptr()), cols, padded);
  switch (qtype) {
    case 8:
      launch<DequantQ80>(cold_weight, hot_weight, cache_weight, hot_map, cold_map, cache_map,
                         quantized, output, sorted_route_ids, block_expert_ids,
                         num_routes_post_padded, top_k, cols, rows, tokens, stream);
      break;
    case 20:
      launch<DequantIQ4NL>(cold_weight, hot_weight, cache_weight, hot_map, cold_map, cache_map,
                           quantized, output, sorted_route_ids, block_expert_ids,
                           num_routes_post_padded, top_k, cols, rows, tokens, stream);
      break;
    case 21:
      launch<DequantIQ3S>(cold_weight, hot_weight, cache_weight, hot_map, cold_map, cache_map,
                          quantized, output, sorted_route_ids, block_expert_ids,
                          num_routes_post_padded, top_k, cols, rows, tokens, stream);
      break;
    default:
      launch<DequantIQ4XS>(cold_weight, hot_weight, cache_weight, hot_map, cold_map, cache_map,
                           quantized, output, sorted_route_ids, block_expert_ids,
                           num_routes_post_padded, top_k, cols, rows, tokens, stream);
      break;
  }
  AT_CUDA_CHECK(hipGetLastError());
  return output;
}

torch::Tensor tiered_iq_moe_prefill_wmma(
    torch::Tensor x, torch::Tensor cold_weight, torch::Tensor hot_weight,
    torch::Tensor hot_map, torch::Tensor cold_map,
    torch::Tensor sorted_route_ids, torch::Tensor block_expert_ids,
    torch::Tensor num_routes_post_padded, int64_t top_k, int64_t qtype,
    int64_t rows, int64_t tokens, int64_t group_size) {
  return prefill_impl(x, cold_weight, hot_weight, hot_map, cold_map,
                      sorted_route_ids, block_expert_ids, num_routes_post_padded,
                      top_k, qtype, rows, tokens, group_size, nullptr, nullptr);
}

torch::Tensor tiered_iq_moe_cached_prefill_wmma(
    torch::Tensor x, torch::Tensor cold_weight, torch::Tensor hot_weight,
    torch::Tensor cache_weight, torch::Tensor hot_map, torch::Tensor cold_map,
    torch::Tensor cache_map, torch::Tensor sorted_route_ids,
    torch::Tensor block_expert_ids, torch::Tensor num_routes_post_padded,
    int64_t top_k, int64_t qtype, int64_t rows, int64_t tokens,
    int64_t group_size) {
  return prefill_impl(x, cold_weight, hot_weight, hot_map, cold_map,
                      sorted_route_ids, block_expert_ids, num_routes_post_padded,
                      top_k, qtype, rows, tokens, group_size, &cache_weight,
                      &cache_map);
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def("tiered_iq_moe_prefill_wmma", &tiered_iq_moe_prefill_wmma,
             "Expert-grouped GGUF MoE prefill on gfx12 int8 WMMA (HIP)");
  module.def("tiered_iq_moe_cached_prefill_wmma", &tiered_iq_moe_cached_prefill_wmma,
             "Expert-grouped GGUF MoE prefill with dynamic cache on gfx12 int8 WMMA (HIP)");
}
