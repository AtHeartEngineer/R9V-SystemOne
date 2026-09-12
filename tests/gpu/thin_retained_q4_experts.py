#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Bounded nonzero GGUF parity with real UVA/cached expert routing."""
import argparse
import importlib.util
import itertools
import json
import re
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--extension', required=True, type=Path)
    parser.add_argument('--spec', type=Path, required=True)
    parser.add_argument('--limit', type=int)
    parser.add_argument('--qtypes', type=int, nargs='+', choices=(7, 8, 12, 13, 20, 21, 23), default=[12, 13, 7, 8])
    args = parser.parse_args()
    import torch
    from vllm.utils.torch_utils import get_accelerator_view_from_cpu_tensor
    from vllm_gguf_plugin import ops
    if not torch.cuda.is_available():
        raise SystemExit('SKIP: gfx1201 GPU required')
    properties = torch.cuda.get_device_properties(0)
    if not str(getattr(properties, 'gcnArchName', '')).startswith('gfx1201'):
        raise SystemExit('SKIP: this test requires gfx1201')
    match = re.search(r'f16/bf16 paths abs ([\deE.+-]+) / rel ([\deE.+-]+)', args.spec.read_text())
    if not match:
        raise ValueError('Missing Spec 1 section 6.1 numeric tolerance')
    atol, rtol = map(float, match.groups())
    spec = importlib.util.spec_from_file_location('qwen38_tiered_iq_moe_hip', args.extension)
    extension = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(extension)
    blocks = {7: (32, 24), 8: (32, 34), 12: (256, 144), 13: (256, 176), 20: (32, 18), 21: (256, 110), 23: (256, 136)}
    cases = list(itertools.product(args.qtypes, (416, 224), (1, 2, 3, 4, 5, 17)))
    if args.limit is not None:
        cases = cases[:args.limit]
    for index, (qtype, width, tokens) in enumerate(cases):
        cols, rows = (2560, 2 * width) if qtype in (12, 13, 21, 23) else (width, 2560)
        block, size = blocks[qtype]
        generator = torch.Generator().manual_seed(90210 + index)
        packed = torch.randint(0, 256, (3, rows, cols // block, size), dtype=torch.uint8, generator=generator)
        # Valid finite GGUF d (and m/dmin) half values, with nonuniform packed codes.
        scales = (torch.rand((3, rows, cols // block), generator=generator) * .01 + .002).half()
        packed[..., :2] = scales.unsqueeze(-1).view(torch.uint8)
        if qtype in (7, 12, 13):
            minima = (torch.rand(scales.shape, generator=generator) * .003).half()
            packed[..., 2:4] = minima.unsqueeze(-1).view(torch.uint8)
        packed = packed.reshape(3, rows, -1).contiguous()
        logical = packed.cuda()
        hot = logical[:1].contiguous()
        # Hold the pinned owner through the final synchronize; never pass a dangling UVA view.
        owner = torch.empty(packed[1:].shape, dtype=torch.uint8, pin_memory=True)
        owner.copy_(packed[1:])
        cold = get_accelerator_view_from_cpu_tensor(owner)
        cache = logical[2:].contiguous()
        hot_map = torch.tensor([0, -1, -1], dtype=torch.int32, device='cuda')
        cold_map = torch.tensor([-1, 0, 1], dtype=torch.int32, device='cuda')
        cache_map = torch.tensor([-1, -1, 0], dtype=torch.int32, device='cuda')
        ids_cpu = torch.tensor([[(t + j) % 3 for j in range(3)] for t in range(tokens)], dtype=torch.int32)
        ids = ids_cpu.cuda().flatten()
        x = torch.randn((tokens, cols), generator=generator).bfloat16().cuda()
        reference = torch.cat([ops.ggml_mul_mat_vec_a8(logical[int(ids_cpu[t,j])], x[t:t+1], qtype, rows)
                               for t in range(tokens) for j in range(3)], dim=0)
        if not torch.isfinite(reference).all() or reference.abs().max() == 0:
            raise AssertionError('Synthetic oracle must be finite and nonzero')
        generic_args = (x, cold, hot, cache, hot_map, cold_map, cache_map, ids, 3, qtype, rows, tokens)
        for variant in (0, 31):
            actual = extension.tiered_iq_moe_cached_gemv_variant(*generic_args, variant)
            torch.testing.assert_close(actual, reference, atol=atol, rtol=rtol)
            repeated = extension.tiered_iq_moe_cached_gemv_variant(*generic_args, variant)
            if not torch.equal(actual, repeated):
                raise AssertionError('Repeated cached GEMV changed output')
        # Group each logical expert's route IDs; padding uses the total-route sentinel.
        for group in (4, 16):
            sorted_ids, experts = [], []
            for expert in range(3):
                routes = [i for i,v in enumerate(ids_cpu.flatten().tolist()) if v == expert]
                routes += [tokens * 3] * ((-len(routes)) % group)
                sorted_ids.extend(routes)
                experts.extend([expert] * (len(routes) // group))
            grouping = tuple(torch.tensor(v, dtype=torch.int32, device='cuda') for v in (sorted_ids, experts, [len(sorted_ids)]))
            actual = extension.tiered_iq_moe_cached_prefill_grouped(
                x, cold, hot, cache, hot_map, cold_map, cache_map, *grouping, 3, qtype, rows, tokens, group)
            torch.testing.assert_close(actual, reference, atol=atol, rtol=rtol)
        torch.cuda.synchronize()
        print(json.dumps({'case': index, 'qtype': qtype, 'width': width, 'tokens': tokens, 'passed': True}), flush=True)
    print(json.dumps({'passed': True, 'cases': len(cases), 'scope': 'nonzero synthetic packed GGUF, hot/UVA/cache, token rows1..5/17, generic/reuse fallback and grouped prefill'}))


if __name__ == '__main__':
    main()
