"""Tiny matched callback-overhead measurement; not an inference benchmark."""
import json
import os
import statistics
import time
import torch

def benchmark():
    torch.cuda.set_device(int(os.environ['R9V_SMOKE_DEVICE']))
    a = torch.arange(4096, device='cuda', dtype=torch.float32)
    for _ in range(20):
        b = a * 2 + 1
    torch.cuda.synchronize()
    samples = []
    for _ in range(3):
        start = time.perf_counter()
        for _ in range(200):
            b = a * 2 + 1
        torch.cuda.synchronize()
        samples.append(time.perf_counter() - start)
    assert torch.equal(b.cpu(), torch.arange(4096, dtype=torch.float32) * 2 + 1)
    print('R9V_BENCH ' + json.dumps(dict(device=int(os.environ['R9V_SMOKE_DEVICE']),
          instrumented=os.environ.get('R9V_GPU_ATTRIBUTION') == '1',
          median_seconds=statistics.median(samples), samples=samples)), flush=True)

if __name__ == '__main__':
    benchmark()
