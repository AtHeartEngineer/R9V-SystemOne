"""Host-side bounded smoke dispatch. Called only after remote capture admission."""
import json
import os
from pathlib import Path
import subprocess
import time
from line_stream import run as stream_run

ROOT = Path(__file__).resolve().parent
IMAGE = 'sha256:550fe8d298ea1c2455613068f2a6b9a234f1ca24fb686731b478b1ebba982a7d'
NAME = 'r9v-attribution-smoke-232505'


def run():
    # The inherited environment of each fresh interpreter is configured by sitecustomize.
    inside = '''import os,subprocess,time,json,pathlib,signal
children=[]
try:
 for device in range(2):
  children.append(subprocess.Popen(['python3','/capture/smoke.py'],env=dict(os.environ,R9V_SMOKE_DEVICE=str(device))))
 deadline=time.monotonic()+45
 while time.monotonic()<deadline:
  if all(pathlib.Path('/capture/smoke-ready-'+str(d)+'.json').exists() for d in range(2)):break
  if any(p.poll() is not None for p in children):raise RuntimeError('smoke exited before ready')
  time.sleep(.1)
 else:raise RuntimeError('smoke readiness timeout')
 for device,p in enumerate(children):
  ready=json.loads(pathlib.Path('/capture/smoke-ready-'+str(device)+'.json').read_text())
  assert ready['pid']==p.pid
 import sys
 sys.path.insert(0,"/capture")
 from serving_capture import read_workers
 rows=read_workers(pathlib.Path('/capture'),pattern='smoke-ready-*.json')
 assert {r['rank'] for r in rows}=={0,1} and {r['pid'] for r in rows}=={p.pid for p in children}
 print('R9V_SERVING_CAPTURE_PASSED',flush=True)
 for p in children:
  if p.wait(timeout=30)!=0:raise RuntimeError('smoke worker failed')
 print('R9V_SMOKE_ALL_PASSED',flush=True)
finally:
 for p in children:
  if p.poll() is None:p.terminate()
'''
    args = ['docker', 'run', '--name', NAME, '--network', 'none', '--device', '/dev/kfd',
            '--device', '/dev/dri', '--security-opt', 'seccomp=unconfined', '--security-opt', 'label=disable',
            '--log-driver', 'json-file', '--log-opt', 'max-size=20m', '--log-opt', 'max-file=2',
            '--volume', str(ROOT) + ':/capture', '--env', 'R9V_GPU_ATTRIBUTION=1',
            '--env', 'PYTHONPATH=/capture/hooks', '--env', 'HSA_ENABLE_DEBUG=0',
            '--env', 'HSA_TOOLS_ROCPROFILER_V1_TOOLS=1',
            '--env', 'HSA_TOOLS_LIB=',
            '--env', 'ROCP_TOOL_LIBRARIES=/capture/libgpu_trace.so',
            '--env', 'LD_PRELOAD=/opt/rocm/lib/librocprofiler-sdk.so',
            '--env', 'HIP_VISIBLE_DEVICES=0,1', '--env', 'ROCR_VISIBLE_DEVICES=0,1',
            '--entrypoint', 'python3', IMAGE, '-u', '-c', inside]
    if any(ROOT.glob('smoke-ready-*.json')):
        raise SystemExit('Smoke already attempted; no automatic retry')
    baseline_inside = '''import os,subprocess
children=[subprocess.Popen(['python3','/capture/bench.py'],env=dict(os.environ,R9V_SMOKE_DEVICE=str(d))) for d in range(2)]
raise SystemExit(any(p.wait()!=0 for p in children))
'''
    baseline = list(args)
    baseline[baseline.index(NAME)] = 'r9v-attribution-baseline-232505'
    baseline[baseline.index('R9V_GPU_ATTRIBUTION=1')] = 'R9V_GPU_ATTRIBUTION=0'
    baseline[baseline.index('ROCP_TOOL_LIBRARIES=/capture/libgpu_trace.so')] = 'ROCP_TOOL_LIBRARIES='
    baseline[baseline.index('LD_PRELOAD=/opt/rocm/lib/librocprofiler-sdk.so')] = 'LD_PRELOAD='
    baseline[baseline.index('HSA_TOOLS_LIB=')] = 'HSA_TOOLS_LIB='
    baseline[-1] = baseline_inside
    try:
        stream_run(baseline, timeout=45)
        stream_run(args, timeout=90)
    finally:
        subprocess.run(['docker', 'stop', '--time', '3', NAME, 'r9v-attribution-baseline-232505'], timeout=10, check=False)


if __name__ == '__main__':
    run()
