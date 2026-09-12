"""Check real smoke capture artifacts without initializing a GPU runtime."""
import json
from pathlib import Path
import re

root = Path(__file__).resolve().parent
proof = []
for device in range(2):
    ready = json.loads((root / ('smoke-ready-' + str(device) + '.json')).read_text())
    directory = root / ('debug-' + str(ready['pid']))
    log = directory / 'waves.log'
    with log.open(errors='replace') if log.exists() else open('/dev/null') as stream:
        text = stream.read(2*2**20)
    objects = [p for p in directory.iterdir() if p.is_file() and p.name != 'waves.log']
    proof.append(dict(device=device, pid=ready['pid'], wave_log=text[:131072],
                      waves_observed=bool(re.search(r'wave_\d+.*pc=', text)),
                      code_objects=len(objects), code_object_bytes=sum(p.stat().st_size for p in objects)))
print(json.dumps(dict(passed=all(p['waves_observed'] and p['code_objects'] for p in proof), workers=proof)))
