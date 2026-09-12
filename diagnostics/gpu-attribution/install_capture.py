#!/usr/bin/python3
"""Install the reviewed capture service; no permissions are granted to future commands."""
import grp
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile


def main():
    if os.geteuid() != 0:
        raise SystemExit('Run this prepared installer with sudo on the workstation')
    root = Path(__file__).resolve().parent
    config = json.loads((root / 'controller-config.json').read_text())
    boot = Path('/proc/sys/kernel/random/boot_id').read_text().strip()
    if boot != config['required_boot_id']:
        raise SystemExit('Boot changed: this campaign requires a new preflight')
    manifest = json.loads((root / 'install-hashes.json').read_text())
    for name, expected in manifest.items():
        path = root / name
        if path.is_symlink() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise SystemExit('Reviewed capture input changed: ' + name)
    program = Path('/usr/local/libexec/r9v-gpu-attribution.py')
    unit = Path('/etc/systemd/system/r9v-gpu-attribution.service')
    if program.exists() or unit.exists():
        raise SystemExit('Capture already installed; inspect its state rather than replacing it')
    output = root / 'privileged'
    output.mkdir(mode=0o750)
    os.chown(output, 0, grp.getgrnam('dylan').gr_gid)
    os.chmod(output, 0o2750)
    program.parent.mkdir(exist_ok=True)
    shutil.copyfile(root / 'host_capture.py', program)
    os.chown(program, 0, 0)
    os.chmod(program, 0o755)
    unit.write_text(f'''[Unit]
Description=R9V GPU process attribution and completed device-dump capture
After=network-online.target
[Service]
Type=simple
ExecStart=/usr/bin/python3 {program} --output {output}
Restart=no
RuntimeMaxSec=10800
TimeoutStopSec=10
KillMode=control-group
UMask=0027
NoNewPrivileges=yes
ProtectSystem=strict
ProtectHome=read-only
ReadWritePaths={output}
PrivateTmp=yes
LogRateLimitIntervalSec=0
MemoryMax=256M
TasksMax=16
''')
    os.chmod(unit, 0o644)
    subprocess.run(['systemctl', 'daemon-reload'], check=True)
    subprocess.run(['systemctl', 'start', unit.name], check=True)
    subprocess.run(['systemctl', 'start', 'r9v-timeout-trace.service'], check=True)
    print('Collectors started. The waiting blackbox gate will validate capture and run the prepared experiment automatically.')


if __name__ == '__main__':
    main()
