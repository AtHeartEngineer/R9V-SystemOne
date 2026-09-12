"""Freeze local blackbox buffers before a bounded stop if the supervisor dies."""
import json
from pathlib import Path
import subprocess
import time
from fault_window import FaultWindow

root = Path(__file__).resolve().parent
config = json.loads((root / 'controller-config.json').read_text())
for group in ('windows', 'gate-windows'):
    for directory in (root / group).glob('*'):
        if directory.is_dir():
            FaultWindow(directory).freeze(dict(time=time.time(), reason='supervisor exit'))
if not (root / 'campaign-end.json').exists():
    for unit in (config['unit'], 'r9v-attribution-smoke-232505'):
        try:
            subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=3','workstation',
                            'systemctl','--user','kill','--kill-whom=main','--signal=TERM',unit],
                           timeout=6, check=False)
        except subprocess.SubprocessError:
            pass
