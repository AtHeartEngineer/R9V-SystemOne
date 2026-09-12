"""Blackbox owns admission, proof capture, and exactly one sustained dispatch."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import threading
import time

from fault_window import FaultWindow
from trace_check import TraceCheck

ROOT = Path(__file__).resolve().parent
CONFIG = json.loads((ROOT / 'controller-config.json').read_text())
REMOTE = str(Path(CONFIG['remote_output']).parent)
FAIL = threading.Event()
FINISH = threading.Event()
LOCK = threading.Lock()
START = time.time()
ACTIVE_UNIT = None
IDENTITY_AT = None
CONTROLLER_RUNNING = False
WINDOWS = []
PROCESSES = []


def save(name, data):
    target = ROOT / name
    temporary = target.with_suffix('.tmp')
    with temporary.open('w') as stream:
        json.dump(data, stream, indent=2)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(target)


def ssh(*args, timeout=15):
    return subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=3',
                           '-o', 'ServerAliveInterval=3', '-o', 'ServerAliveCountMax=2',
                           'workstation', shlex.join(args)], capture_output=True, text=True,
                          timeout=timeout, check=True).stdout


def fail(reason):
    with LOCK:
        if FAIL.is_set():
            return
        FAIL.set()
        save('gate-STOP.json', dict(time=time.time(), reason=reason, automatic_restart=False))
        for window, _ in WINDOWS:
            window.freeze(dict(reason=reason, time=time.time()))
    if CONTROLLER_RUNNING:
        # The controller owns its log buffers and freezes them before any SSH stop.
        save('external-stop.json', dict(time=time.time(), reason=reason))
    elif ACTIVE_UNIT:
        try:
            ssh('systemctl', '--user', 'kill', '--kill-whom=main', '--signal=TERM', ACTIVE_UNIT, timeout=6)
        except subprocess.SubprocessError:
            pass


def stream(args, name, inspect=None):
    window = FaultWindow(ROOT / 'gate-windows' / name, limit=32*2**20, segments=4)
    with LOCK:
        WINDOWS.append((window, name))
    proc = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=0)
    PROCESSES.append(proc)
    def read():
        pending = b''
        try:
            while not FINISH.is_set():
                chunk = os.read(proc.stdout.fileno(), 65536)
                if not chunk:
                    break
                with LOCK:
                    window.append(name, chunk)
                pending += chunk
                while b'\n' in pending:
                    line, pending = pending.split(b'\n', 1)
                    if inspect:
                        inspect(line.decode(errors='replace'))
                if len(pending) > 1024*1024:
                    fail(name + ': oversized line')
                    break
        except Exception as error:
            fail(name + ': ' + str(error))
        finally:
            with LOCK:
                window.flush()
    thread = threading.Thread(target=read, daemon=True)
    thread.start()
    return proc, thread


def identity():
    row = json.loads(ssh('cat', REMOTE + '/privileged/identity.json'))
    if row.get('boot_id') != CONFIG['required_boot_id']:
        raise ValueError('Root identity boot mismatch')
    # Use workstation wall time, not a cross-host subtraction.
    remote_now = float(ssh('date', '+%s'))
    if not -2 <= remote_now - row['wall_time'] <= 4 or row['errors']:
        raise ValueError('Root identity stale or incomplete')
    if not any(r['comm'] == 'kwin_wayland' and r.get('pasid') is not None for r in row['rows']):
        raise ValueError('KWin GPU address-space identity is still missing')
    return row


def mirror_artifacts():
    """Copy code objects, wave logs, and completed dumps while the host is responsive."""
    target = ROOT / 'worker-capture'
    target.mkdir(exist_ok=True)
    command = ['rsync','-a','--timeout=8','--max-size=256m','--partial-dir=.rsync-partial',
               '--include=/debug-*/','--include=/debug-*/*','--include=/privileged/',
               '--include=/privileged/*.dump','--include=/privileged/identity.json',
               '--include=/stall-*.json','--exclude=*',
               '-e','ssh -o BatchMode=yes -o ConnectTimeout=3',
               'workstation:'+REMOTE+'/',str(target)+'/']
    while not FINISH.is_set():
        try:
            size_code = "from pathlib import Path; p=Path("+repr(REMOTE)+"); print(sum(x.stat().st_size for d in list(p.glob('debug-*'))+[p/'privileged'] if d.exists() for x in d.rglob('*') if x.is_file()))"
            size = int(ssh('python3','-c',size_code,timeout=10))
            if size > 2*2**30:
                fail('Capture assets exceeded 2 GiB; remaining budget reserved for logs')
                return
            result = subprocess.run(command, capture_output=True, text=True, timeout=30)
            save('artifact-mirror.json', dict(time=time.time(), exit_code=result.returncode,
                                            source_bytes=size, stderr=result.stderr[:1500]))
        except Exception as error:
            save('artifact-mirror.json', dict(time=time.time(), error=str(error)))
        FINISH.wait(2)


def main():
    global ACTIVE_UNIT, CONTROLLER_RUNNING
    if (ROOT / 'gate-start.json').exists() or (ROOT / 'campaign-start.json').exists():
        raise SystemExit('Campaign already attempted; no automatic retry')
    save('gate-start.json', dict(time=START, boot=CONFIG['required_boot_id']))
    save('gate-status.json', dict(phase='waiting-for-local-sudo', time=time.time()))
    # No GPU command is issued while installation is missing.
    while time.time() - START < 24*3600:
        try:
            states = ssh('systemctl', 'is-active', 'r9v-gpu-attribution.service', 'r9v-timeout-trace.service')
            if states.splitlines() == ['active', 'active']:
                break
        except (subprocess.SubprocessError, ValueError, KeyError) as error:
            save('gate-status.json', dict(phase='waiting-for-local-sudo', time=time.time(), detail=str(error)[:500]))
        time.sleep(3)
    else:
        raise RuntimeError('Collector installation wait expired; no GPU work dispatched')
    time.sleep(2)  # Allow the new collector's first complete snapshot to be published.
    initial_identity = identity()
    save('identity-preflight.json', initial_identity)
    # Existing source/kernel identity gate plus the new complete-root coverage gate.
    result = ssh('python3', REMOTE + '/preflight.py')
    reviewed = json.loads(result)
    from trace_gate import validate_identity
    validate_identity(reviewed, reviewed)
    save('preflight-initial.json', reviewed)
    cutoff = time.time()
    def inspect_kernel(line):
        try:
            envelope = json.loads(line)
            received = datetime.datetime.fromisoformat(envelope['received_utc']).timestamp()
            if received < cutoff:
                return
            message = envelope.get('message', '')
            fault = (envelope.get('channel') == 'netconsole' and re.search(
                r'Dumping IP State(?! Completed)|device lost from bus|ring .*timeout|GPU reset|GPU fault|page fault|hard LOCKUP|Kernel panic', message, re.I))
            if '"kind": "first_timeout"' in message or fault:
                fail('fault during capture preflight: ' + message[:1500])
        except (ValueError, KeyError, TypeError):
            pass
    net, _ = stream(['sudo','-n','/usr/local/bin/r9v-crash-logs','--follow','--lines','1'], 'kernel.jsonl', inspect_kernel)
    identity_seen = threading.Event()
    def inspect_identity(line):
        global IDENTITY_AT
        try:
            row = json.loads(line)
            if row.get('source') == 'r9v-gpu-identity' and row.get('boot_id') == CONFIG['required_boot_id']:
                if row.get('errors'):
                    fail('GPU identity coverage lost')
                IDENTITY_AT = time.monotonic()
                identity_seen.set()
        except ValueError:
            pass
    ident, _ = stream(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=3','-o','ServerAliveInterval=3','-o','ServerAliveCountMax=2',
                       'workstation', 'journalctl -u r9v-gpu-attribution.service --output=cat --follow --since now'],
                      'identity.jsonl', inspect_identity)
    if not identity_seen.wait(10) or net.poll() is not None or ident.poll() is not None or FAIL.is_set():
        raise RuntimeError('Live remote evidence stream failed before smoke')
    threading.Thread(target=mirror_artifacts, daemon=True).start()
    # Wait for independently received, uncaptured timeout heartbeat before even the smoke.
    from trace_gate import TraceGate
    gate = TraceGate(reviewed)
    def inspect_heartbeat(line):
        try:
            event = gate.observe(json.loads(line), time.time())
            if event == 'fault':
                fail('Timeout recorder already latched')
        except ValueError:
            pass
    heart, _ = stream(['sudo','-n','/usr/local/bin/r9v-crash-logs','--follow','--lines','1'], 'admission-heartbeats.jsonl', inspect_heartbeat)
    deadline = time.monotonic() + 55
    while not gate.ready(time.time()):
        if FAIL.is_set() or time.monotonic() > deadline or heart.poll() is not None:
            raise RuntimeError('Fresh independent heartbeat gate unavailable')
        time.sleep(.2)
    save('gate-status.json', dict(phase='healthy-two-worker-smoke', time=time.time()))
    trace = TraceCheck()
    benchmarks = []
    done = threading.Event()
    def inspect_smoke(line):
        trace.feed(line)
        if 'R9V_BENCH ' in line:
            benchmarks.append(json.loads(line.split('R9V_BENCH ', 1)[1]))
        if 'R9V_SMOKE_ALL_PASSED' in line:
            done.set()
    ACTIVE_UNIT = 'r9v-attribution-smoke-232505'
    command = ['systemd-run','--user','--wait','--pipe','--unit='+ACTIVE_UNIT,'--property=RuntimeMaxSec=150',
               '--property=TimeoutStopSec=10', '--property=Restart=no',
               '--property=ExecStopPost=-/usr/bin/timeout 10 /home/dylan/bin/docker stop --time 3 r9v-attribution-smoke-232505 r9v-attribution-baseline-232505',
               '/usr/bin/python3',REMOTE+'/smoke_driver.py']
    proc, thread = stream(['ssh','-o','ConnectTimeout=3','-o','ServerAliveInterval=3','-o','ServerAliveCountMax=2',
                           'workstation',shlex.join(command)], 'smoke.log', inspect_smoke)
    deadline = time.monotonic() + 160
    while proc.poll() is None:
        if FAIL.is_set() or time.monotonic() > deadline or gate.failure(time.time()) or time.monotonic() - IDENTITY_AT > 5:
            raise RuntimeError('Smoke failed or its evidence heartbeat disappeared')
        time.sleep(.2)
    thread.join(timeout=3)
    save('smoke-trace-summary.json', trace.result())
    if proc.returncode or not done.is_set():
        raise RuntimeError('Healthy smoke did not complete successfully')
    trace.validate_smoke()
    comparisons = []
    for device in range(2):
        baseline = next(r for r in benchmarks if r['device'] == device and not r['instrumented'])
        instrumented = next(r for r in benchmarks if r['device'] == device and r['instrumented'])
        comparisons.append(dict(device=device, ratio=instrumented['median_seconds']/baseline['median_seconds']))
    save('instrumentation-overhead.json', dict(samples=benchmarks, comparisons=comparisons,
                                              note='Tiny kernel workload; not an inference throughput estimate'))
    proof = json.loads(ssh('python3', REMOTE + '/code_object_proof.py'))
    save('code-object-proof.json', proof)
    if not proof['passed']:
        raise RuntimeError('Code-object ELF capture proof failed')
    # Verify the independent receiver holds byte-identical code objects before admission.
    target = ROOT / 'worker-capture'
    result = subprocess.run(['rsync','-a','--timeout=8','--include=/debug-*/','--include=/debug-*/*.elf','--exclude=*','-e','ssh -o BatchMode=yes -o ConnectTimeout=3','workstation:'+REMOTE+'/',str(target)+'/'],capture_output=True,text=True,timeout=30,check=True)
    for worker in proof['workers']:
        for obj in worker['objects']:
            local = target / obj['file']
            if hashlib.sha256(local.read_bytes()).hexdigest() != obj['sha256']:
                raise RuntimeError('Remote code object mismatch: '+obj['file'])
    save('code-object-receipt.json',dict(time=time.time(),verified=True,workers=[w['pid'] for w in proof['workers']]))
    # Recheck compute use after smoke cleanup, then admit a fresh inference campaign.
    identity()
    reviewed = json.loads(ssh('python3',REMOTE+'/preflight.py'))
    validate_identity(reviewed, json.loads((ROOT/'preflight-initial.json').read_text()))
    if FAIL.is_set():
        raise RuntimeError('Capture preflight latched a fault')
    save('gate-status.json', dict(phase='sustained-workload', time=time.time(), smoke_verified=True))
    ACTIVE_UNIT = CONFIG['unit']
    CONTROLLER_RUNNING = True
    proc = subprocess.Popen(['python3', str(ROOT/'controller.py')])
    try:
        while proc.poll() is None:
            if ident.poll() is not None or net.poll() is not None:
                fail('Evidence stream disconnected')
            if time.monotonic() - IDENTITY_AT > 5:
                fail('Privileged GPU identity heartbeat disappeared')
            time.sleep(1)
        proc.wait(timeout=15)
    finally:
        if proc.poll() is None:
            proc.terminate()
    save('gate-status.json', dict(phase='finished', time=time.time(), controller_exit=proc.returncode))


if __name__ == '__main__':
    try:
        main()
    except BaseException as error:
        fail(str(error))
        save('gate-status.json', dict(phase='stopped', time=time.time(), reason=str(error)))
        raise
    finally:
        FINISH.set()
        for process in PROCESSES:
            if process.poll() is None:
                process.terminate()
        for window, _ in WINDOWS:
            window.freeze(dict(reason='capture gate finished', time=time.time()))
