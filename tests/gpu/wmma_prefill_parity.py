#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Parity and timing of the gfx12 WMMA grouped MoE prefill against the released kernel.

Runs on one gfx1201 GPU inside the runtime image, e.g.

    python tests/gpu/wmma_prefill_parity.py \
        --reference /opt/r9v/kernels/retained-mtp4/qwen38_tiered_iq_moe_hip.so \
        --candidate /opt/r9v/kernels/retained-mtp4/r9v_moe_wmma.so --output parity.json

Both kernels quantize activations identically and accumulate int8 dot products with
the same per-32 scales; only the fp32 summation order differs, so outputs must agree
to bf16 rounding of the largest output magnitude.
"""
import argparse
import functools
import gc
import importlib.util
import json
import statistics
from pathlib import Path

BLOCKS = {21: (256, 110), 23: (256, 136), 20: (32, 18), 8: (32, 34)}
NAMES = {21: "IQ3_S", 23: "IQ4_XS", 20: "IQ4_NL", 8: "Q8_0"}
# One bf16 ulp is 2^-8 of the magnitude; allow two ulps of the largest output.
RELATIVE_TOLERANCE = 2 * 2**-8


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--tokens", type=int, nargs="+", default=[1024, 4096])
    parser.add_argument("--qtypes", type=int, nargs="+", choices=sorted(BLOCKS), default=[21, 23, 20, 8])
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args()
    import torch
    from vllm.utils.torch_utils import get_accelerator_view_from_cpu_tensor

    if not torch.cuda.is_available():
        raise SystemExit("SKIP: gfx1201 GPU required")
    properties = torch.cuda.get_device_properties(args.device)
    if not str(getattr(properties, "gcnArchName", "")).startswith("gfx1201"):
        raise SystemExit("SKIP: this test requires gfx1201")
    reference = load("qwen38_tiered_iq_moe_hip", args.reference)
    candidate = load("r9v_moe_wmma", args.candidate)
    torch.cuda.set_device(args.device)
    device = torch.device("cuda", args.device)
    width = 416

    def grouping(ids, group, num_experts=512):
        per_expert = [[] for _ in range(num_experts)]
        for index, expert in enumerate(ids):
            per_expert[expert].append(index)
        sorted_ids, experts = [], []
        for expert, routes in enumerate(per_expert):
            routes = routes + [len(ids)] * ((-len(routes)) % group)
            sorted_ids.extend(routes)
            experts.extend([expert] * (len(routes) // group))
        return tuple(torch.tensor(v, dtype=torch.int32, device=device) for v in (sorted_ids, experts, [len(sorted_ids)]))

    def timed(fn, repeats=6):
        times = []
        for repeat in range(repeats):
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            out = fn()
            end.record()
            end.synchronize()
            if repeat:
                times.append(start.elapsed_time(end))
        return out, statistics.median(times)

    records = []
    failures = []
    for qtype in args.qtypes:
        cols, rows = (2560, 2 * width) if qtype in (21, 23) else (width, 2560)
        block, size = BLOCKS[qtype]
        gen = torch.Generator().manual_seed(14092026 + qtype)
        packed = torch.randint(0, 256, (512, rows, cols // block, size), dtype=torch.uint8, generator=gen)
        scales = (torch.rand((512, rows, cols // block), generator=gen) * 0.01 + 0.002).half()
        packed[..., :2] = scales.unsqueeze(-1).view(torch.uint8)
        packed = packed.reshape(512, rows, -1)
        for resident in (67, 512):
            hot = packed[:resident].to(device)
            cold_count = max(1, 512 - resident)
            owner = torch.empty((cold_count, *packed.shape[1:]), dtype=torch.uint8, pin_memory=True)
            owner.copy_(packed[resident:resident + cold_count] if resident < 512 else packed[:1])
            cold = get_accelerator_view_from_cpu_tensor(owner)
            hot_map = torch.full((512,), -1, dtype=torch.int32, device=device)
            cold_map = hot_map.clone()
            hot_map[:resident] = torch.arange(resident, dtype=torch.int32, device=device)
            if resident < 512:
                cold_map[resident:] = torch.arange(512 - resident, dtype=torch.int32, device=device)
            for tokens in args.tokens:
                route_k = 10 if qtype in (21, 23) else 1
                input_rows = tokens if route_k == 10 else tokens * 10
                ids = torch.stack([torch.randperm(512, generator=gen)[:10] for _ in range(tokens)]).flatten().tolist()
                x = torch.randn((input_rows, cols), generator=gen).bfloat16().to(device)
                g16, g32 = grouping(ids, 16), grouping(ids, 32)
                ref_args = (x, cold, hot, hot_map, cold_map, *g16, route_k, qtype, rows, input_rows, 16)
                new_args = (x, cold, hot, hot_map, cold_map, *g32, route_k, qtype, rows, input_rows, 32)
                ref, ref_ms = timed(functools.partial(reference.tiered_iq_moe_prefill_grouped, *ref_args))
                out, new_ms = timed(functools.partial(candidate.tiered_iq_moe_prefill_wmma, *new_args))
                torch.cuda.synchronize()
                diff = (out.float() - ref.float()).abs()
                magnitude = ref.float().abs().max().item()
                row = {"qtype": NAMES[qtype], "resident": resident, "tokens": tokens,
                       "reference_ms": ref_ms, "wmma_ms": new_ms, "speedup": ref_ms / new_ms,
                       "max_abs_diff": diff.max().item(), "reference_max_abs": magnitude,
                       "mean_abs_diff": diff.mean().item(), "finite": bool(torch.isfinite(out).all())}
                row["passed"] = row["finite"] and row["max_abs_diff"] <= RELATIVE_TOLERANCE * magnitude
                records.append(row)
                if not row["passed"]:
                    failures.append(row)
                print(json.dumps(row), flush=True)
                del ref, out, x, g16, g32
            torch.cuda.synchronize()
            del hot, owner, cold, hot_map, cold_map
            gc.collect()
            torch.cuda.empty_cache()
        del packed
    summary = {"passed": not failures, "cases": len(records), "failures": len(failures),
               "relative_tolerance": RELATIVE_TOLERANCE, "device": properties.name, "records": records}
    if args.output:
        args.output.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k != "records"}), flush=True)
    raise SystemExit(0 if summary["passed"] else 1)


if __name__ == "__main__":
    main()
