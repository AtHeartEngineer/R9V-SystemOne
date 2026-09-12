"""Per-process command-state capture; no debugger attachment or GPU serialization."""
import os
from pathlib import Path


def stalled(record, pid, now_ns):
    return (record.get('pid') == pid and record.get('stage') == 'execute_model_enter'
            and record.get('steps', 0) >= 20
            and 0 < record.get('last_batch_tokens', 0) <= 8
            and 1_000_000_000 < now_ns - record.get('monotonic_ns', now_ns) < 30_000_000_000)


def watch():
    import json
    import signal
    import time
    pid = os.getpid()
    while True:
        time.sleep(0.1)
        try:
            for path in Path('/tmp').glob('r9v-stage-*.json'):
                record = json.loads(path.read_text())
                if not stalled(record, pid, time.monotonic_ns()):
                    continue
                detail = dict(pid=pid, wall_time=time.time(), stage=record,
                              action='freeze remotely received dispatch/completion set and saved code objects')
                (Path('/capture') / ('stall-' + str(pid) + '.json')).write_text(json.dumps(detail))
                print('R9V_CAPTURE ' + json.dumps(detail), flush=True)
                return
        except (OSError, ValueError, KeyError):
            continue

if os.environ.get('R9V_GPU_ATTRIBUTION') == '1':
    root = Path('/capture') / ('debug-' + str(os.getpid()))
    root.mkdir(mode=0o700, exist_ok=True)
    import threading
    threading.Thread(target=watch, name='r9v-stall-capture', daemon=True).start()
