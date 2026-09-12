#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Bounded loader-byte parity: 48 tiny layers, both TP partitions, real UVA owners.

This exercises storage layout/lifetime, not quantized GEMM arithmetic. Synthetic
packed byte dimensions preserve 416/224 partition boundaries without full weights.
Requires the retained ROCm image and two GPUs; no model assets or network access.
"""
import gc
import json
import os
from pathlib import Path
import tempfile

import torch
from torch.nn.parameter import UninitializedParameter
from vllm_gguf_plugin.quantization import params, tiered_experts as tiered


class Layer(torch.nn.Module):
    def __init__(self, index, rank):
        super().__init__()
        self.layer_name = f"model.layers.{index}.mlp.experts"
        self.local_num_experts = 512
        self._gguf_expert_partition = (0, 416, 640) if rank == 0 else (416, 224, 640)
        for name in ("w13_qweight", "w2_qweight"):
            p = UninitializedParameter(requires_grad=False, device="cpu", dtype=torch.uint8)
            p._vllm_is_uva_offloaded = True
            p._vllm_uva_pin_memory = False
            p.tensor_shape = (512,)
            self.register_parameter(name, p)


class Model(torch.nn.Module):
    def __init__(self, rank):
        super().__init__()
        self.layers = torch.nn.ModuleList([Layer(i, rank) for i in range(48)])


def weights(index):
    value = torch.arange(512 * 20 * 8, dtype=torch.int32)
    return {
        "w1": ((value + index * 7) % 251).to(torch.uint8).reshape(512, 20, 8),
        "w3": ((value * 3 + index * 11) % 253).to(torch.uint8).reshape(512, 20, 8),
        "w2": ((value * 5 + index * 17) % 255).to(torch.uint8).reshape(512, 8, 20),
    }


def must_refuse(call):
    try:
        call()
    except RuntimeError:
        return
    raise AssertionError("invalid/incomplete streaming copy was accepted")


def check_layer(layer, index, rank, hot):
    source = weights(index)
    start, width = ((0, 13) if rank == 0 else (13, 7))
    full = {
        "w13": torch.cat([source["w1"][:, start:start+width], source["w3"][:, start:start+width]], dim=1),
        "w2": source["w2"][:, :, start:start+width],
    }
    hot_set = set(hot)
    cold = [i for i in range(512) if i not in hot_set]
    for name, expected in full.items():
        p = getattr(layer, name + "_qweight")
        assert not tiered.is_tiered_expert_master(p)
        owner = p._vllm_uva_cpu_data
        assert owner.is_pinned() and p.device.type == "cuda"
        assert torch.equal(getattr(layer, "_gguf_hot_" + name).cpu(), expected[hot])
        assert torch.equal(owner, expected[cold])
        # Read the actual accelerator view after temporary masters have gone.
        assert torch.equal(p.data.cpu(), expected[cold])
    hot_map, cold_map = layer._gguf_global_to_hot.cpu(), layer._gguf_global_to_cold.cpu()
    assert torch.equal(hot_map[hot], torch.arange(len(hot), dtype=torch.int32))
    assert torch.equal(cold_map[cold], torch.arange(len(cold), dtype=torch.int32))
    assert (hot_map[cold] == -1).all() and (cold_map[hot] == -1).all()
    if rank == 0:
        assert layer._gguf_cache_w13.shape[0] == 80
        assert (layer._gguf_global_to_cache == -1).all()
        assert (layer._gguf_cache_tags == -1).all()


def run(rank, streaming, manifest, per_expert=False):
    torch.cuda.set_device(rank)
    os.environ["QWEN38_TIERED_STREAM_COMPACTION"] = str(int(streaming))
    tiered.get_tensor_model_parallel_rank = lambda: rank
    tiered._tiered_expert_modules = lambda model: enumerate(model.layers)
    model = Model(rank)
    assert tiered.prepare_tiered_expert_masters(model) == 48
    if streaming:
        must_refuse(lambda model=model: tiered.materialize_hot_expert_cache(model))
        layer = model.layers[0]
        must_refuse(lambda: params._gguf_moe_weight_loader(layer, None, layer.w13_qweight,
                    torch.empty((20, 8), dtype=torch.uint8), "weight", "w1", -1))
        assert isinstance(layer.w13_qweight, UninitializedParameter)
    peak_live_master_bytes_sampled = 0
    # Adjacent layers interleave; every pair ends with w3 as its last copy.
    for base in range(0, 48, 2):
        sources = {index: weights(index) for index in (base, base + 1)}
        for shard in ("w2", "w1", "w3"):
            for index in (base + 1, base):
                layer = model.layers[index]
                p = layer.w2_qweight if shard == "w2" else layer.w13_qweight
                source = sources[index][shard]
                # This is the actual RoutedExperts.load_weights behavior: fused
                # checkpoint tensors are unbound into 512 individual callbacks.
                deliveries = enumerate(source.unbind()) if per_expert else [(0, source)]
                for expert_id, weight in deliveries:
                    assert params._gguf_moe_weight_loader(layer, None, p, weight, "weight", shard, expert_id, True) is True
                    if streaming and shard == "w1" and expert_id == 0:
                        must_refuse(lambda: params._gguf_moe_weight_loader(layer, None, p, weight, "weight", shard, expert_id))
                live = sum(getattr(p, "_vllm_uva_cpu_data").numel() for current_layer in model.layers
                           for p in (current_layer.w13_qweight, current_layer.w2_qweight)
                           if tiered.is_tiered_expert_master(p) and hasattr(p, "_vllm_uva_cpu_data"))
                peak_live_master_bytes_sampled = max(peak_live_master_bytes_sampled, live)
    tiered.materialize_hot_expert_cache(model)
    gc.collect()
    for index, layer in enumerate(model.layers):
        check_layer(layer, index, rank, manifest["ranks"][str(rank)]["hot_experts_by_layer"][index])
        assert not hasattr(layer.w13_qweight, "_r9v_tiered_stream_copy_hook")
    assert not hasattr(model, "_r9v_stream_hooks")
    result = {"rank": rank, "streaming": streaming, "per_expert": per_expert, "layers_exact": 48,
              "sampled_peak_live_master_bytes": peak_live_master_bytes_sampled}
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return result


def main():
    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        raise SystemExit("SKIP: requires two ROCm GPUs")
    os.environ.update(QWEN38_TIERED_EXPERT_CACHE_SLOTS="80", QWEN38_TIERED_EXPERT_CACHE_RANKS="0",
                      QWEN38_TIERED_EXPERT_CACHE_POLICY="lru", QWEN38_TIERED_EXPERT_CACHE_ASYNC="0",
                      R9V_CACHE_FILL_BATCH="1", R9V_CACHE192="0")
    manifest = {"version": 1, "num_layers": 48, "num_experts": 512,
                "ranks": {str(rank): {"hot_experts_by_layer": [
                    [(i + layer * 13) % 512 for i in range(64 if rank == 0 else 320)]
                    for layer in range(48)]} for rank in (0, 1)}}
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "manifest.json"
        path.write_text(json.dumps(manifest))
        os.environ["RADIANCE_TIERED_EXPERT_MANIFEST"] = str(path)
        results = [run(rank, streaming, manifest, per_expert)
                   for per_expert in (False, True) for rank in (0, 1) for streaming in (False, True)]
    for index in range(0, len(results), 2):
        legacy, streamed = results[index:index + 2]
        assert streamed["sampled_peak_live_master_bytes"] < legacy["sampled_peak_live_master_bytes"] / 8
    print(json.dumps({"passed": True, "results": results, "scope": "synthetic packed storage/lifetime parity; not full model or GEMM qualification"}))


if __name__ == "__main__":
    main()
