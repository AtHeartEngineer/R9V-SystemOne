// SPDX-License-Identifier: Apache-2.0
// Extracted unchanged planner/copy/publication from R9V cacheparallel, capacity80.
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>
#include <hip/hip_runtime.h>
namespace {
constexpr int kMaxLogicalCacheSlots=80;
constexpr int kCacheCopyBlocks=128;
__global__ __launch_bounds__(256) void plan_expert_cache_lru_fill(
    const int* __restrict__ hot_map,
    const int* __restrict__ cold_map,
    const int* __restrict__ expert_ids,
    const int* __restrict__ cache_map,
    const int* __restrict__ cache_tags,
    int* __restrict__ cache_clock,
    int* __restrict__ stats,
    int* __restrict__ pending,
    int num_experts, int cold_count, int cache_slots, int routes) {
  // Exact stable LRU ordering: untouched slots retain their old relative
  // order; touched slots follow in order of their LAST route occurrence.
  // No approximate recency or policy change. Publication/copy stay unchanged.
  if (routes <= 0 || routes > 64 || cache_slots <= 0 || cache_slots > kMaxLogicalCacheSlots) {
    if (threadIdx.x == 0) pending[0] = 0;
    return;
  }
  __shared__ int ids[64], tags[kMaxLogicalCacheSlots], old_rank[kMaxLogicalCacheSlots], last[kMaxLogicalCacheSlots], rank[kMaxLogicalCacheSlots];
  __shared__ int hits[64], occurrences[64], cold_slots[64];
  const int tid = threadIdx.x;
  if (tid < routes) ids[tid] = expert_ids[tid];
  if (tid < cache_slots) {
    tags[tid] = cache_tags[tid];
    old_rank[tid] = cache_clock[tid + 1];
  }
  __syncthreads();
  if (tid < routes) {
    const int e = ids[tid];
    hits[tid] = 0; occurrences[tid] = 0; cold_slots[tid] = -1;
    if (e >= 0 && e < num_experts) {
      const int slot = cache_map[e];
      const bool cached = slot >= 0 && slot < cache_slots && tags[slot] == e;
      hits[tid] = cached ? 1 : 0;
      const int cold = cold_map[e];
      if (!cached && hot_map[e] < 0 && cold >= 0 && cold < cold_count) {
        bool first = true;
        int count = 0;
        for (int other = 0; other < routes; ++other) {
          const bool same = ids[other] == e;
          count += same;
          if (same && other < tid) first = false;
        }
        if (first) { occurrences[tid] = count; cold_slots[tid] = cold; }
      }
    }
  }
  if (tid < cache_slots) {
    const int e = tags[tid];
    int final_route = -1;
    if (e >= 0 && e < num_experts && cache_map[e] == tid) {
      for (int route = 0; route < routes; ++route)
        if (ids[route] == e) final_route = route;
    }
    last[tid] = final_route;
  }
  __syncthreads();
  if (tid < cache_slots) {
    int position = old_rank[tid];
    if (tags[tid] >= 0) {
      position = 0;
      for (int other = 0; other < cache_slots; ++other) {
        if (tags[other] < 0 || other == tid) continue;
        if (last[tid] >= 0) {
          position += last[other] < last[tid];
        } else {
          position += last[other] < 0 && old_rank[other] < old_rank[tid];
        }
      }
    }
    rank[tid] = position;
    cache_clock[tid + 1] = position;
  }
  __syncthreads();
  if (tid != 0) return;
  pending[0] = 0;
  stats[0] += 1;
  int selected = -1, count = 0, hit_count = 0;
  for (int route = 0; route < routes; ++route) {
    hit_count += hits[route];
    if (occurrences[route] > count) {
      count = occurrences[route]; selected = route;
    }
  }
  stats[2] += hit_count;
  if (selected < 0) return;
  int target = -1;
  for (int slot = 0; slot < cache_slots; ++slot) {
    if (tags[slot] < 0) { target = slot; break; }
  }
  if (target < 0) {
    int oldest = cache_slots + 1;
    for (int slot = 0; slot < cache_slots; ++slot) {
      if (rank[slot] < oldest) { oldest = rank[slot]; target = slot; }
    }
  }
  if (target < 0) return;
  pending[1] = ids[selected];
  pending[2] = cold_slots[selected];
  pending[3] = target;
  pending[4] = target;
  pending[5] = tags[target];
  pending[6] = 3;
  __threadfence();
  pending[0] = 1;
  stats[7] += 1;
  stats[8] += count;
}


__global__ void publish_expert_cache_lru(
    int* __restrict__ cache_map,
    int* __restrict__ cache_tags,
    int* __restrict__ cache_clock,
    int* __restrict__ admission,
    int* __restrict__ stats,
    int* __restrict__ pending,
    int num_experts, int cache_slots) {
  if (threadIdx.x != 0 || pending[0] == 0 || pending[6] != 3) return;
  const int expert = pending[1];
  const int target = pending[3];
  const int old_expert = pending[5];
  if (expert < 0 || expert >= num_experts || target < 0 ||
      target >= cache_slots) {
    pending[0] = 0;
    return;
  }

  int old_rank = -1;
  if (old_expert >= 0 && old_expert < num_experts &&
      cache_tags[target] == old_expert) {
    old_rank = cache_clock[target + 1];
    if (cache_map[old_expert] == target) cache_map[old_expert] = -1;
    cache_tags[target] = -1;
    stats[4] += 1;
  }

  int published = 0;
  for (int slot = 0; slot < cache_slots; ++slot) {
    if (cache_tags[slot] < 0) continue;
    ++published;
    if (old_rank >= 0 && cache_clock[slot + 1] > old_rank) {
      cache_clock[slot + 1] -= 1;
    }
  }
  cache_clock[target + 1] = published;
  __threadfence();
  cache_tags[target] = expert;
  __threadfence();
  cache_map[expert] = target;
  admission[expert] = 2;
  stats[1] += 1;
  __threadfence();
  pending[0] = 0;
}


__global__ __launch_bounds__(256) void copy_planned_expert_cache(
    const uint8_t* __restrict__ cold_w13,
    const uint8_t* __restrict__ cold_w2,
    uint8_t* __restrict__ cache_w13,
    uint8_t* __restrict__ cache_w2,
    const int* __restrict__ pending,
    int64_t w13_expert_bytes, int64_t w2_expert_bytes,
    int expected_mode) {
  if (pending[0] == 0 || pending[6] != expected_mode) return;
  const int cold_slot = pending[2];
  const int cache_slot = pending[3];
  const uint8_t* source_w13 =
      cold_w13 + static_cast<int64_t>(cold_slot) * w13_expert_bytes;
  const uint8_t* source_w2 =
      cold_w2 + static_cast<int64_t>(cold_slot) * w2_expert_bytes;
  uint8_t* target_w13 =
      cache_w13 + static_cast<int64_t>(cache_slot) * w13_expert_bytes;
  uint8_t* target_w2 =
      cache_w2 + static_cast<int64_t>(cache_slot) * w2_expert_bytes;

  const int64_t w13_vectors = w13_expert_bytes / sizeof(uint4);
  const int64_t w2_vectors = w2_expert_bytes / sizeof(uint4);
  const int64_t vector_count = w13_vectors + w2_vectors;
  const int64_t thread = static_cast<int64_t>(blockIdx.x) * blockDim.x +
                         threadIdx.x;
  const int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;
  for (int64_t index = thread; index < vector_count; index += stride) {
    if (index < w13_vectors) {
      reinterpret_cast<uint4*>(target_w13)[index] =
          reinterpret_cast<const uint4*>(source_w13)[index];
    } else {
      const int64_t w2_index = index - w13_vectors;
      reinterpret_cast<uint4*>(target_w2)[w2_index] =
          reinterpret_cast<const uint4*>(source_w2)[w2_index];
    }
  }
  for (int64_t index = w13_vectors * sizeof(uint4) + thread;
       index < w13_expert_bytes; index += stride) {
    target_w13[index] = source_w13[index];
  }
  for (int64_t index = w2_vectors * sizeof(uint4) + thread;
       index < w2_expert_bytes; index += stride) {
    target_w2[index] = source_w2[index];
  }
}


}
void tiered_iq_moe_cache_lru_prepare(
    torch::Tensor cold_w13, torch::Tensor cold_w2, torch::Tensor hot_map,
    torch::Tensor cold_map, torch::Tensor expert_ids,
    torch::Tensor cache_w13, torch::Tensor cache_w2,
    torch::Tensor cache_map, torch::Tensor cache_tags,
    torch::Tensor cache_clock, torch::Tensor admission, torch::Tensor stats,
    torch::Tensor pending) {
  TORCH_CHECK(cold_w13.is_cuda() && cold_w2.is_cuda() && hot_map.is_cuda() &&
                  cold_map.is_cuda() && expert_ids.is_cuda() &&
                  cache_w13.is_cuda() && cache_w2.is_cuda() &&
                  cache_map.is_cuda() && cache_tags.is_cuda() &&
                  cache_clock.is_cuda() && admission.is_cuda() &&
                  stats.is_cuda() && pending.is_cuda(),
              "LRU cache tensors must be GPU or UVA tensors");
  const auto device = cold_w13.device();
  TORCH_CHECK(cold_w2.device() == device && hot_map.device() == device &&
                  cold_map.device() == device && expert_ids.device() == device &&
                  cache_w13.device() == device && cache_w2.device() == device &&
                  cache_map.device() == device && cache_tags.device() == device &&
                  cache_clock.device() == device && admission.device() == device &&
                  stats.device() == device && pending.device() == device,
              "LRU cache tensors must share one device");
  TORCH_CHECK(cold_w13.scalar_type() == torch::kUInt8 &&
                  cold_w2.scalar_type() == torch::kUInt8 &&
                  cache_w13.scalar_type() == torch::kUInt8 &&
                  cache_w2.scalar_type() == torch::kUInt8,
              "LRU cache weights must contain packed uint8 bytes");
  TORCH_CHECK(hot_map.scalar_type() == torch::kInt32 &&
                  cold_map.scalar_type() == torch::kInt32 &&
                  expert_ids.scalar_type() == torch::kInt32 &&
                  cache_map.scalar_type() == torch::kInt32 &&
                  cache_tags.scalar_type() == torch::kInt32 &&
                  cache_clock.scalar_type() == torch::kInt32 &&
                  admission.scalar_type() == torch::kInt32 &&
                  stats.scalar_type() == torch::kInt32 &&
                  pending.scalar_type() == torch::kInt32,
              "LRU cache maps and state must be int32");
  TORCH_CHECK(cold_w13.is_contiguous() && cold_w2.is_contiguous() &&
                  hot_map.is_contiguous() && cold_map.is_contiguous() &&
                  expert_ids.is_contiguous() && cache_w13.is_contiguous() &&
                  cache_w2.is_contiguous() && cache_map.is_contiguous() &&
                  cache_tags.is_contiguous() && cache_clock.is_contiguous() &&
                  admission.is_contiguous() && stats.is_contiguous() &&
                  pending.is_contiguous(),
              "LRU cache tensors must be contiguous");
  TORCH_CHECK(cold_w13.dim() == 3 && cold_w2.dim() == 3 &&
                  cache_w13.dim() == 3 && cache_w2.dim() == 3,
              "LRU expert weights must be three-dimensional");
  TORCH_CHECK(cold_w13.size(0) == cold_w2.size(0),
              "LRU cold projections must contain the same experts");
  TORCH_CHECK(cache_w13.size(0) == cache_w2.size(0) &&
                  cache_w13.size(0) == cache_tags.numel(),
              "LRU cache projections and tags must contain the same slots");
  TORCH_CHECK(cache_w13.size(1) == cold_w13.size(1) &&
                  cache_w13.size(2) == cold_w13.size(2) &&
                  cache_w2.size(1) == cold_w2.size(1) &&
                  cache_w2.size(2) == cold_w2.size(2),
              "LRU cache projection layouts must match cold expert layouts");
  TORCH_CHECK(cache_w13.size(0) > 0 &&
                  cache_w13.size(0) <= kMaxLogicalCacheSlots,
              "synchronous LRU cache supports one through eighty slots");
  TORCH_CHECK(hot_map.numel() == cold_map.numel() &&
                  hot_map.numel() == cache_map.numel() &&
                  hot_map.numel() == admission.numel(),
              "all LRU expert maps/state must cover the same logical experts");
  TORCH_CHECK(cache_clock.numel() >= cache_tags.numel() + 1 &&
                  stats.numel() >= 9 && pending.numel() >= 7,
              "LRU cache clock/stats/pending tensors are too small");

  const int routes = expert_ids.numel();
  if (routes == 0 || routes > 64) return;
  c10::cuda::CUDAGuard guard(device);
  const auto stream = at::cuda::getCurrentHIPStreamMasqueradingAsCUDA();
  const int64_t w13_expert_bytes = cold_w13.size(1) * cold_w13.size(2);
  const int64_t w2_expert_bytes = cold_w2.size(1) * cold_w2.size(2);
  plan_expert_cache_lru_fill<<<1, 256, 0, stream>>>(
      hot_map.data_ptr<int>(), cold_map.data_ptr<int>(),
      expert_ids.data_ptr<int>(), cache_map.data_ptr<int>(),
      cache_tags.data_ptr<int>(), cache_clock.data_ptr<int>(),
      stats.data_ptr<int>(), pending.data_ptr<int>(), hot_map.numel(),
      cold_w13.size(0), cache_w13.size(0), routes);
  copy_planned_expert_cache<<<kCacheCopyBlocks, 256, 0, stream>>>(
      cold_w13.data_ptr<uint8_t>(), cold_w2.data_ptr<uint8_t>(),
      cache_w13.data_ptr<uint8_t>(), cache_w2.data_ptr<uint8_t>(),
      pending.data_ptr<int>(), w13_expert_bytes, w2_expert_bytes, 3);
  publish_expert_cache_lru<<<1, 1, 0, stream>>>(
      cache_map.data_ptr<int>(), cache_tags.data_ptr<int>(),
      cache_clock.data_ptr<int>(), admission.data_ptr<int>(),
      stats.data_ptr<int>(), pending.data_ptr<int>(), cache_map.numel(),
      cache_tags.numel());
  AT_CUDA_CHECK(hipGetLastError());
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){m.def("prepare",&tiered_iq_moe_cache_lru_prepare);}
