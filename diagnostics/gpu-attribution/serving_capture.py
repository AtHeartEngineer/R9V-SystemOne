"""Read actual worker identities with the dispatch capture library mapped."""
import json
from pathlib import Path

def read_workers(stage_root=Path('/tmp'), proc_root=Path('/proc'), pattern='r9v-stage-*.json'):
    rows=[]
    for path in stage_root.glob(pattern):
        row=json.loads(path.read_text())
        pid=int(row['pid'])
        maps=(proc_root/str(pid)/'maps').read_text()
        if 'libgpu_trace.so' in maps:
            rows.append(row)
    return rows

if __name__=='__main__':
    print(json.dumps(read_workers()))
