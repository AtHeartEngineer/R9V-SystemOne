#!/usr/bin/python3
"""Exact-kernel timeout tracing without an external module. No GPU workload."""
import hashlib
import json
import os
from pathlib import Path
import platform
import select
import signal
import socket
import subprocess
import sys
import tempfile
import time

TAG = "r9v-timeout-trace"
ROOT = Path("/sys/kernel/tracing")
INSTANCE = ROOT / "instances/r9v_timeout_identity"
GROUP = "r9v_timeout_identity"
EVENT = "timeout_entry"
DEFINITION = (
    f"p:{GROUP}/{EVENT} amdgpu:amdgpu_job_timedout "
    "job=%di:x64 sched=+8(%di):x64 vmid=+328(%di):u32 "
    "pasid=+332(%di):u32 ring=+0(+24(+8(%di))):string "
    "signaled=-84(+8(%di)):u32 emitted=-88(+8(%di)):u32"
)
UNIT_PATH = Path("/etc/systemd/system/r9v-timeout-trace.service")
PROGRAM = Path("/usr/local/libexec/r9v-timeout-trace.py")
UNIT = """# Managed by r9v-timeout-trace
[Unit]
Description=R9V exact-kernel first timeout identity capture
After=network-online.target
Wants=network-online.target
[Service]
Type=notify
NotifyAccess=main
ExecStart=/usr/bin/python3 /usr/local/libexec/r9v-timeout-trace.py --run
Restart=no
TimeoutStartSec=20
TimeoutStopSec=10
UMask=0077
"""


def check_kernel():
    if platform.release() != "7.1.5-ogc5.1.fc44.x86_64" or platform.machine() != "x86_64":
        raise RuntimeError("Kernel/architecture changed; regenerate BTF offsets first")
    for name, expected in {
        "vmlinux": "8269e5b4a6f181c771e1f9ae0e8604f1de254df032e06a32ab47ab833e606f3a",
        "amdgpu": "d1e6c9c5376d7d10888aecd75c0aea5754e932441c642095fb397c03c5b95908",
    }.items():
        if hashlib.sha256(Path(f"/sys/kernel/btf/{name}").read_bytes()).hexdigest() != expected:
            raise RuntimeError(f"Live BTF hash mismatch: {name}")
    lockdown = Path("/sys/kernel/security/lockdown")
    if lockdown.exists() and "[none]" not in lockdown.read_text():
        raise RuntimeError("Kernel lockdown is active; no security settings will be changed")


def append_event(value):
    # Python's append-mode FileIO seeks to EOF during open. tracefs uses
    # seq_lseek, which rejects SEEK_END with EINVAL. Do not switch to "w":
    # O_TRUNC on this control file removes other users' probes.
    payload = (value + "\n").encode("ascii")
    fd = os.open(ROOT / "kprobe_events", os.O_WRONLY | os.O_CLOEXEC)
    try:
        if os.write(fd, payload) != len(payload):
            raise RuntimeError("Incomplete kprobe control write; inspect tracefs before retrying")
    finally:
        os.close(fd)


def notify_ready():
    address = os.environ.get("NOTIFY_SOCKET")
    if address:
        if address.startswith("@"):
            address = "\0" + address[1:]
        with socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM) as sock:
            sock.sendto(b"READY=1", address)


def run():
    check_kernel()
    if not (ROOT / "kprobe_events").exists():
        raise RuntimeError("tracefs kprobe_events unavailable; inspect mount/access before proceeding")
    if INSTANCE.exists() or (ROOT / f"events/{GROUP}/{EVENT}").exists():
        raise RuntimeError("Probe/instance already exists; inspect it before removing or replacing it")
    created = False
    instance_created = False
    stopped = False
    fd = None
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setblocking(False)
    boot = Path("/proc/sys/kernel/random/boot_id").read_text().strip()

    def emit(kind, **fields):
        payload = json.dumps(dict(source=TAG, kind=kind, boot_id=boot,
                                  monotonic_ns=time.monotonic_ns(), **fields))
        # UDP goes first: local journal delivery must not delay remote delivery.
        try:
            sock.sendto(payload.encode(), ("192.168.1.80", 6667))
        except OSError as error:
            print(f"UDP forwarding failed: {error}", file=sys.stderr, flush=True)
        print(payload, flush=True)

    def stop(signum, frame):
        nonlocal stopped
        stopped = True

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    try:
        append_event(DEFINITION)
        created = True
        INSTANCE.mkdir()
        instance_created = True
        (INSTANCE / "tracing_on").write_text("0\n")
        (INSTANCE / "buffer_size_kb").write_text("64\n")
        (INSTANCE / "options/overwrite").write_text("0\n")
        (INSTANCE / "buffer_percent").write_text("0\n")
        (INSTANCE / "trace_clock").write_text("mono\n")
        event_path = INSTANCE / f"events/{GROUP}/{EVENT}"
        fd = os.open(INSTANCE / "trace_pipe", os.O_RDONLY | os.O_NONBLOCK)
        (event_path / "enable").write_text("1\n")
        (INSTANCE / "tracing_on").write_text("1\n")
        emit("ready", definition=DEFINITION, instance=str(INSTANCE))
        notify_ready()
        pending = b""
        captured = False
        heartbeat_at = time.monotonic() + 30
        while not stopped:
            if captured:
                time.sleep(1)
                readable = []
            else:
                readable, _, _ = select.select([fd], [], [], 1)
            if readable:
                try:
                    pending += os.read(fd, 65536)
                except BlockingIOError:
                    pass
                while b"\n" in pending:
                    line, pending = pending.split(b"\n", 1)
                    if b"timeout_entry:" in line and not captured:
                        # Forward the first delivered event before any control writes.
                        emit("first_timeout", trace=line.decode(errors="replace"))
                        captured = True
                        (INSTANCE / "tracing_on").write_text("0\n")
                        (event_path / "enable").write_text("0\n")
                if len(pending) > 65536:
                    raise RuntimeError("Unexpected oversized trace record")
            if time.monotonic() >= heartbeat_at:
                emit("heartbeat", captured=captured)
                heartbeat_at = time.monotonic() + 30
    finally:
        if fd is not None:
            os.close(fd)
        if instance_created:
            (INSTANCE / "tracing_on").write_text("0\n")
            (INSTANCE / f"events/{GROUP}/{EVENT}/enable").write_text("0\n")
            INSTANCE.rmdir()
        if created:
            append_event(f"-:{GROUP}/{EVENT}")
        sock.close()


def validate_installation(source):
    # Permit repair of the exact failed version, or reinstall of this version.
    # Never replace an unknown program/unit or restart a running recorder.
    for path in (PROGRAM, UNIT_PATH):
        if path.is_symlink():
            raise RuntimeError(f"Refusing installation symlink: {path}")
    if UNIT_PATH.exists() and UNIT_PATH.read_text() != UNIT:
        raise RuntimeError("Existing service differs from the reviewed unit")
    if PROGRAM.exists():
        known = {
            "63d49612a45c3bb4be40c79e71b05f0799ed1a3c4538ea4fd8baa190df120dac",
            hashlib.sha256(source).hexdigest(),
        }
        if hashlib.sha256(PROGRAM.read_bytes()).hexdigest() not in known:
            raise RuntimeError("Existing recorder differs from the reviewed versions")
    state = subprocess.run(
        ["systemctl", "show", UNIT_PATH.name, "--property=ActiveState", "--value"],
        check=True, capture_output=True, text=True,
    ).stdout.strip()
    if state not in {"inactive", "failed"}:
        raise RuntimeError(f"Recorder is {state!r}; will not interrupt it")


def atomic_install(path, data, mode):
    fd, temporary = tempfile.mkstemp(prefix="." + path.name + ".", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fchmod(stream.fileno(), mode)
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def install():
    check_kernel()
    source = Path(__file__).read_bytes()
    validate_installation(source)
    PROGRAM.parent.mkdir(parents=True, exist_ok=True)
    atomic_install(PROGRAM, source, 0o755)
    atomic_install(UNIT_PATH, UNIT.encode(), 0o644)
    subprocess.run(["systemctl", "daemon-reload"], check=True)
    try:
        subprocess.run(["systemctl", "start", UNIT_PATH.name], check=True)
    except subprocess.CalledProcessError:
        subprocess.run(["journalctl", "-u", UNIT_PATH.name, "-n", "30", "--no-pager"])
        raise
    print("Trace probe attached; no GPU test started. Stop with: sudo systemctl stop " + UNIT_PATH.name)


if __name__ == "__main__":
    if os.geteuid() != 0:
        sys.exit("Run with sudo on the workstation")
    if sys.argv[1:] == ["--install"]:
        install()
    elif sys.argv[1:] == ["--run"]:
        run()
    else:
        sys.exit("Expected --install or --run")
