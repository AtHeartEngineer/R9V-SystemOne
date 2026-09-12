#!/usr/bin/python3
"""Root-scoped GPU client identity and completed device-dump preservation."""
import argparse
import concurrent.futures
import grp
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import threading
import time

BDFS = {'0000:03:00.0', '0000:13:00.0'}


def identities(proc=Path('/proc')):
    rows, errors = [], []
    for directory in proc.iterdir():
        if not directory.name.isdecimal():
            continue
        try:
            relevant = []
            for fd in (directory / 'fd').iterdir():
                try:
                    target = os.readlink(fd)
                    if target.startswith('/dev/dri/') or target == '/dev/kfd':
                        relevant.append((fd.name, target))
                except FileNotFoundError:
                    continue
            if not relevant:
                continue
            status = dict(line.split(':', 1) for line in (directory / 'status').read_text().splitlines() if ':' in line)
            stat = (directory / 'stat').read_text().rsplit(')', 1)[1].split()
            common = dict(host_pid=int(directory.name), start_ticks=int(stat[19]),
                          namespace_pids=[int(p) for p in status.get('NSpid', directory.name).split()],
                          comm=status['Name'].strip(), uid=int(status['Uid'].split()[0]))
            for fd, target in relevant:
                try:
                    fields = dict(line.split(':', 1) for line in (directory / 'fdinfo' / fd).read_text().splitlines() if ':' in line)
                    fields = {k: v.strip() for k, v in fields.items()}
                    if target == '/dev/kfd':
                        rows.append(dict(common, fd=int(fd), device=target, kind='kfd', pasid=fields.get('pasid')))
                    elif fields.get('drm-pdev') in BDFS:
                        rows.append(dict(common, fd=int(fd), device=target, kind='drm',
                                         bdf=fields['drm-pdev'], pasid=int(fields['pasid']),
                                         client_id=fields.get('drm-client-id')))
                except FileNotFoundError:
                    pass  # A closed FD is not a persistent access gap.
                except (OSError, ValueError, KeyError) as error:
                    errors.append(dict(pid=directory.name, fd=fd, error=str(error)))
        except (FileNotFoundError, ProcessLookupError):
            pass
        except (OSError, ValueError, KeyError, IndexError) as error:
            errors.append(dict(pid=directory.name, error=str(error)))
    return dict(rows=rows, errors=errors)


def copy_dump(source, destination, limit=256 * 2**20):
    """Copy a completed sysfs dump; never write device controls or trigger recovery."""
    total = 0
    digest = hashlib.sha256()
    partial = destination.with_suffix('.partial')
    with source.open('rb', buffering=0) as src, partial.open('xb', buffering=0) as dst:
        os.chmod(partial, 0o640)
        while True:
            block = src.read(min(2**20, limit - total + 1))
            if not block:
                break
            total += len(block)
            if total > limit:
                raise ValueError('Device dump exceeds capture budget')
            dst.write(block)
            digest.update(block)
        os.fsync(dst.fileno())
    partial.replace(destination)
    return dict(bytes=total, sha256=digest.hexdigest(), file=destination.name)


def run(output):
    if os.geteuid() != 0:
        raise SystemExit('Root is required for complete GPU fdinfo coverage')
    output.mkdir(mode=0o750, parents=True, exist_ok=True)
    os.chown(output, 0, grp.getgrnam('dylan').gr_gid)
    os.chmod(output, 0o2750)
    boot = Path('/proc/sys/kernel/random/boot_id').read_text().strip()
    stopping = threading.Event()
    signal.signal(signal.SIGTERM, lambda *_: stopping.set())
    signal.signal(signal.SIGINT, lambda *_: stopping.set())
    pool = concurrent.futures.ThreadPoolExecutor(max_workers=2)
    pending, seen = {}, set()
    sequence = 0
    while not stopping.is_set():
        sequence += 1
        result = dict(source='r9v-gpu-identity', boot_id=boot, sequence=sequence,
                      wall_time=time.time(), monotonic_ns=time.monotonic_ns(), **identities())
        raw = json.dumps(result) + '\n'
        temp = output / 'identity.tmp'
        with temp.open('w') as stream:
            os.chmod(temp, 0o640)
            stream.write(raw)
        temp.replace(output / 'identity.json')
        print(raw, end='', flush=True)
        for device in Path('/sys/class/devcoredump').glob('devcd*'):
            if str(device) in seen or len(seen) >= 2:
                continue
            try:
                bdf = (device / 'failing_device').resolve(strict=True).name
                if bdf not in BDFS:
                    continue
                seen.add(str(device))
                destination = output / (bdf + '-' + device.name + '.dump')
                pending[pool.submit(copy_dump, device / 'data', destination)] = (str(device), time.monotonic())
            except OSError as error:
                print(json.dumps(dict(source='r9v-device-dump', error=str(error))), flush=True)
        for future, (device, started) in list(pending.items()):
            if future.done():
                try:
                    detail = future.result()
                except Exception as error:
                    detail = dict(error=str(error))
                print(json.dumps(dict(source='r9v-device-dump', boot_id=boot, device=device, **detail)), flush=True)
                del pending[future]
            elif time.monotonic() - started > 10:
                # A blocked driver read must not block identities or spawn repeated readers.
                print(json.dumps(dict(source='r9v-device-dump', device=device, state='read-stalled')), flush=True)
        stopping.wait(1)
    pool.shutdown(wait=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run(args.output)
