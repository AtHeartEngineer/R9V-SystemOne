"""Small healthy two-worker probe, numerically checked; no deliberate fault."""
import json
import os
from pathlib import Path
import signal
import time
import torch

import cpuinfo
assert cpuinfo.get_cpu_info().get('arch'), 'CPU helper stdout contract failed'
device = int(os.environ['R9V_SMOKE_DEVICE'])
torch.cuda.set_device(device)
# Exercise the same RCCL initialization that the serving workers require.
import torch.distributed as dist
from datetime import timedelta
dist.init_process_group('nccl', init_method='file:///capture/smoke-rendezvous',
                        rank=device, world_size=2, timeout=timedelta(seconds=30))
value = torch.tensor([device + 1.0], device='cuda')
dist.all_reduce(value)
assert value.item() == 3.0
print('R9V_RCCL_PASSED ' + json.dumps({'device': device}), flush=True)
from bench import benchmark
benchmark()
a = torch.arange(4096, device='cuda', dtype=torch.float32)
for _ in range(10):
    b = a * 2 + 1
torch.cuda.synchronize()
assert torch.equal(b.cpu(), torch.arange(4096, dtype=torch.float32) * 2 + 1)
ready = {'pid': os.getpid(), 'device': device, 'rank': device, 'correct': True}
print('R9V_SMOKE ' + json.dumps(ready), flush=True)
# Keep ordinary finite work in flight while the controller exercises capture.
matrix = torch.eye(2048, device='cuda', dtype=torch.float32)
torch.cuda.synchronize()
Path('/capture/smoke-ready-' + str(device) + '.json').write_text(json.dumps(ready))
# Ordinary finite kernels keep workers alive for serving-admission validation.
until = time.monotonic() + 2
while time.monotonic() < until:
    for _ in range(4):
        product = matrix @ matrix
    torch.cuda.synchronize()
assert torch.equal(product, matrix)
print('R9V_SMOKE_DONE ' + json.dumps(ready), flush=True)

dist.destroy_process_group()
