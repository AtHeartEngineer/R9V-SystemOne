"""Summarize retained attribution evidence; no remote commands or GPU access."""
import json
from pathlib import Path
import re

root = Path(__file__).resolve().parent
summary = dict(root=str(root), cause_established=False)
for name in ('gate-status.json','gate-STOP.json','STOP.json','first-timeout.json',
             'instrumentation-overhead.json','gpu-inflight-at-stop.json','gpu-inflight-latest.json',
             'request-status.json','wave-capture-proof.json','artifact-mirror.json'):
    path = root / name
    if path.exists():
        data = json.loads(path.read_text())
        if name == 'wave-capture-proof.json':
            data = dict(data, workers=[{k:v for k,v in p.items() if k!='wave_log'} for p in data['workers']])
        summary[name] = data
fault = summary.get('first-timeout.json', {})
pasid = re.search(r'\bpasid=(\d+)', fault.get('trace',''))
if pasid:
    target = int(pasid.group(1))
    cutoff = fault.get('monotonic_ns', 0)
    candidates = {}
    for path in (root / 'gate-windows/identity.jsonl').rglob('identity.jsonl*'):
        if not path.is_file():
            continue
        with path.open(errors='replace') as stream:
            for line in stream:
                try:
                    record = json.loads(line)
                    if record.get('boot_id') != fault.get('boot_id'):
                        continue
                    stamp = record.get('monotonic_ns', 0)
                    if not 0 <= cutoff - stamp <= 30_000_000_000:
                        continue
                    for row in record.get('rows', []):
                        if row.get('pasid') == target:
                            key = (row['host_pid'], row['start_ticks'], row.get('bdf'))
                            candidates[key] = dict(row, snapshot_monotonic_ns=stamp)
                except (ValueError, KeyError, TypeError):
                    pass
    summary['timeout_pasid_candidates'] = list(candidates.values())
    summary['attribution_limit'] = 'PASID matches across recorded devices; owner is not automatically the initiating cause. Correlate BDF/ring and command evidence.'
summary['wave_logs'] = [str(p.relative_to(root)) for p in (root/'worker-capture').glob('debug-*/waves.log')]
summary['device_dumps'] = [str(p.relative_to(root)) for p in (root/'worker-capture/privileged').glob('*.dump')]
(root/'CAPTURE-SUMMARY.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps({k:v for k,v in summary.items() if k in ('cause_established','timeout_pasid_candidates','wave_logs','device_dumps')},indent=2))
