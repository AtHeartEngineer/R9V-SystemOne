import json
import socket
import sys
import time
import uuid

import pytest

from tools.observability import DriverSampler, Reporter
from tools.remote_watch import RunTracker


def test_wedged_probe_is_bounded_and_never_respawned():
    probe = DriverSampler(
        timeout=0.03, command=[sys.executable, "-c", "import time;time.sleep(60)"]
    )
    try:
        probe.poll()
        pid = probe.process.pid
        time.sleep(0.05)
        start = time.monotonic()
        for _ in range(10):
            assert probe.poll()["status"] == "stalled"
            assert probe.process.pid == pid
        assert time.monotonic() - start < 0.5
    finally:
        probe.close()
    assert probe.process.poll() is not None


def test_remote_records_have_shared_identity_and_sequence():
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as receiver:
        receiver.bind(("127.0.0.1", 0))
        receiver.settimeout(1)
        reporter = Reporter(target="127.0.0.1:" + str(receiver.getsockname()[1]))
        reporter.send("r9v.event", event="request_start")
        first = json.loads(receiver.recv(12000))
        reporter.send("r9v.event", event="response_saved")
        second = json.loads(receiver.recv(12000))
    assert first["run_id"] == second["run_id"]
    assert second["sequence"] == first["sequence"] + 1
    assert second["monotonic_ns"] >= first["monotonic_ns"]


def test_blackbox_requires_explicit_arm_and_captures_once_per_container():
    tracker = RunTracker()
    run = str(uuid.uuid4())
    progress = {"kind": "r9v.progress", "run_id": run, "container": "r9v-test"}
    tracker.observe(progress, 0)
    assert tracker.due(30) == []
    tracker.observe({"kind": "r9v.session_start", "run_id": run, "seconds": 180}, 30)
    tracker.observe(progress, 31)
    assert len(tracker.due(47)) == 1
    assert tracker.due(48) == []
    tracker.observe({**progress, "container": "r9v-next"}, 49)
    assert len(tracker.due(65)) == 1
    tracker.observe({"kind": "r9v.session_end", "run_id": run}, 66)
    assert tracker.due(100) == []


def test_blackbox_rejects_shell_injection_container():
    tracker = RunTracker()
    run = str(uuid.uuid4())
    tracker.observe({"kind": "r9v.session_start", "run_id": run, "seconds": 180}, 0)
    tracker.observe(
        {"kind": "r9v.progress", "run_id": run, "container": "r9v-$(reboot)"}, 1
    )
    assert tracker.due(100) == []


def test_kernel_fault_captures_before_progress_timeout_and_only_once():
    tracker = RunTracker()
    run = str(uuid.uuid4())
    envelope = {
        "channel": "netconsole",
        "received_utc": "1970-01-01T00:01:40+00:00",
        "message": "6,116931,28637523645,-;amdgpu 0000:03:00.0: Dumping IP State\n",
    }
    tracker.observe_kernel(envelope, 1, 100)
    assert tracker.due(1) == []  # An unrelated/unarmed workstation stays untouched.
    tracker.observe({"kind": "r9v.session_start", "run_id": run, "seconds": 180}, 1)
    tracker.observe({"kind": "r9v.progress", "run_id": run, "container": "r9v-test"}, 2)
    tracker.observe_kernel(envelope, 3, 100)
    due = tracker.due(3)
    assert len(due) == 1 and due[0][:2] == (run, "r9v-test")
    assert due[0][2].startswith("kernel fault:")
    tracker.observe_kernel(envelope, 4, 101)
    assert tracker.due(4) == []


@pytest.mark.parametrize(
    "change",
    [
        {"channel": "userspace"},
        {"received_utc": "1970-01-01T00:00:00+00:00"},
        {"received_utc": "1970-01-01T00:02:00+00:00"},
        {"received_utc": "not a date"},
        {"message": "amdgpu 0000:03:00.0: Dumping IP State Completed\n"},
        {"message": "normal unrelated kernel message"},
    ],
)
def test_kernel_capture_rejects_stale_or_nonfault_records(change):
    tracker = RunTracker()
    run = str(uuid.uuid4())
    tracker.observe({"kind": "r9v.session_start", "run_id": run, "seconds": 180}, 1)
    tracker.observe({"kind": "r9v.progress", "run_id": run, "container": "r9v-test"}, 2)
    tracker.observe_kernel(
        {
            "channel": "netconsole",
            "received_utc": "1970-01-01T00:01:40+00:00",
            "message": "amdgpu: device lost from bus!",
            **change,
        },
        3,
        100,
    )
    assert tracker.due(3) == []


def test_host_ram_guard_continues_when_driver_is_stalled(tmp_path, monkeypatch):
    from tools import qualify_runtime as q

    data = {
        "config": {"R9V_IMAGE": "test", "R9V_EXPECTED_GPU_BDFS": "a,b"},
        "prompt": "x",
        "arms": [{"name": "a"}],
    }
    session = q.Session(data, tmp_path)
    calls = []
    monkeypatch.setattr(q, "memory_snapshot", lambda _: {"host_available_bytes": 2**30})

    class Driver:
        def poll(self):
            calls.append(True)
            if len(calls) == 3:
                session.stop.set()
            return {"status": "stalled"}

    session.driver = Driver()
    session.monitor()
    assert len(calls) == 3
    assert len((tmp_path / "memory.jsonl").read_text().splitlines()) == 3
    assert session.failure


def test_capture_happens_before_command_is_killed(tmp_path, monkeypatch):
    from tools import qualify_runtime as q

    data = {
        "config": {"R9V_IMAGE": "test", "R9V_EXPECTED_GPU_BDFS": "a,b"},
        "prompt": "x",
        "arms": [{"name": "a"}],
    }
    session = q.Session(data, tmp_path)
    session.container_name = "r9v-test"
    session.failure = "driver stalled"
    events = []

    class Process:
        def poll(self):
            return None

        def kill(self):
            events.append("kill")

        def wait(self, **kw):
            return 0

    monkeypatch.setattr(q.subprocess, "Popen", lambda *a, **kw: Process())
    monkeypatch.setattr(q, "early_capture", lambda *a: events.append("capture"))
    with pytest.raises(RuntimeError, match="driver stalled"):
        session.command(["fixture"], tmp_path / "log")
    assert events == ["capture", "kill"]


def test_blackbox_detects_frozen_workers_with_live_host_progress():
    tracker = RunTracker()
    run = str(uuid.uuid4())
    tracker.observe({"kind": "r9v.session_start", "run_id": run, "seconds": 180}, 0)
    row = {
        "kind": "r9v.progress",
        "run_id": run,
        "container": "r9v-test",
        "phase": "arm-routes",
        "workers": [{"rank": 0, "steps": 10}, {"rank": 1, "steps": 10}],
    }
    for now in range(1, 62):
        tracker.observe(row, now)
        assert tracker.due(now) == []
    tracker.observe(row, 62)
    assert tracker.due(62)[0][2] == "worker progress unchanged for 60 seconds"


def test_early_capture_continues_after_one_command_hangs(tmp_path):
    from tools.early_capture import bounded

    start = time.monotonic()
    result = bounded(
        [sys.executable, "-c", "import time;time.sleep(60)"],
        tmp_path / "blocked",
        seconds=0.05,
    )
    assert result["timeout"]
    assert time.monotonic() - start < 1


def test_stage_handler_can_produce_python_stacks_without_importing_torch(tmp_path):
    import ast
    import os
    import subprocess
    from pathlib import Path

    source = Path("vendor/vllm/vllm/v1/worker/r9v_diagnostics.py").read_text()
    node = next(
        n
        for n in ast.parse(source).body
        if isinstance(n, ast.FunctionDef) and n.name == "stage"
    )
    script = (
        """import json, os, time
from pathlib import Path as RealPath
from types import SimpleNamespace
import sys
root = RealPath(sys.argv[1])
def Path(value):
    return root / RealPath(value).name if str(value).startswith('/tmp/') else RealPath(value)
"""
        + ast.unparse(node)
        + """
worker=SimpleNamespace(rank=0)
stage(worker, 'execute_model_enter')
print('ready', flush=True)
sys.stdin.read()
"""
    )
    with (tmp_path / "stacks").open("wb") as error:
        proc = subprocess.Popen(
            [sys.executable, "-c", script, str(tmp_path)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=error,
            env={**os.environ, "R9V_STAGE_DIAGNOSTICS": "1"},
        )
        try:
            import select

            assert select.select([proc.stdout], [], [], 3)[0]
            assert proc.stdout.readline() == b"ready\n"
            from tools.early_capture import PYTHON_STACKS

            command = PYTHON_STACKS.replace(
                "pathlib.Path('/tmp')", "pathlib.Path(" + repr(str(tmp_path)) + ")"
            )
            result = subprocess.run(
                [sys.executable, "-c", command],
                capture_output=True,
                check=False,
                text=True,
                timeout=3,
            )
            assert "stack_requested" in result.stdout
            deadline = time.monotonic() + 2
            while (
                tmp_path / "stacks"
            ).stat().st_size == 0 and time.monotonic() < deadline:
                time.sleep(0.01)
            assert b"Current thread" in (tmp_path / "stacks").read_bytes()
        finally:
            proc.kill()
            proc.wait(timeout=1)


def test_driver_eio_is_a_failure_even_when_the_probe_returns():
    sample = {
        "kind": "sample",
        "finished_at": time.time(),
        "driver_faults": [{"errno": 5}],
    }
    script = (
        "import json,sys,time;sys.stdin.readline();print("
        + repr(json.dumps(sample))
        + ",flush=True);time.sleep(5)"
    )
    probe = DriverSampler(command=[sys.executable, "-c", script])
    try:
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline:
            row = probe.poll()
            if row["status"] == "failed":
                break
            time.sleep(0.01)
        assert row["status"] == "failed"
        assert "Driver read faults" in row["error"]
    finally:
        probe.close()


def test_gpu_selection_prevents_reads_from_unrelated_devices(tmp_path, monkeypatch):
    from tools import capture_runtime as capture

    device = tmp_path / "bus/pci/devices/unrelated"
    device.mkdir(parents=True)

    def forbidden(path):
        raise AssertionError("unselected device was probed")

    monkeypatch.setattr(capture, "read_text", forbidden)
    assert capture.gpu_snapshot(tmp_path, bdfs=["selected"]) == []
    assert capture.pcie_snapshot(tmp_path, bdfs=["selected"]) == []


def test_first_fault_window_survives_cleanup_rotation_and_restart(tmp_path):
    from tools.fault_window import FaultWindow

    window = FaultWindow(tmp_path, limit=4, segments=3)
    for data in (b"old1", b"old2", b"old3", b"last"):
        window.append("hip.log", data)
    frozen = window.freeze({"reason": "first GPU fault"})
    before = {p.name: p.read_bytes() for p in frozen.iterdir()}
    assert before["hip.log"] == b"last"
    assert before["hip.log.1"] == b"old3"
    assert before["hip.log.2"] == b"old2"
    for _ in range(10):
        window.append("hip.log", b"stop")
    restarted = FaultWindow(tmp_path, limit=4, segments=3)
    restarted.append("hip.log", b"more")
    restarted.freeze({"reason": "later cleanup error"})
    assert {p.name: p.read_bytes() for p in frozen.iterdir()} == before
    assert not (tmp_path / "hip.log").exists()


def test_worker_operation_journals_and_remote_records_do_not_interleave(tmp_path):
    import ast
    import os
    import subprocess
    from pathlib import Path

    source = Path("vendor/vllm/vllm/v1/worker/r9v_diagnostics.py").read_text()
    node = next(
        n
        for n in ast.parse(source).body
        if isinstance(n, ast.FunctionDef) and n.name == "stage"
    )
    script = (
        """import json, os, time, sys
from types import SimpleNamespace
from pathlib import Path as RealPath
root = RealPath(sys.argv[1])
def Path(value):
    return root / RealPath(value).name if str(value).startswith('/tmp/') else RealPath(value)
"""
        + ast.unparse(node)
        + """
worker=SimpleNamespace(rank=int(sys.argv[2]))
for i in range(20):
    stage(worker, 'execute_model_enter' if i % 2 == 0 else 'execute_model_return')
"""
    )
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as receiver:
        receiver.bind(("127.0.0.1", 0))
        receiver.settimeout(3)
        env = {
            **os.environ,
            "R9V_STAGE_DIAGNOSTICS": "1",
            "R9V_CONTAINER_NAME": "r9v-fixture",
            "R9V_OBSERVABILITY_TARGET": "127.0.0.1:" + str(receiver.getsockname()[1]),
        }
        children = [
            subprocess.Popen(
                [sys.executable, "-c", script, str(tmp_path), str(rank)], env=env
            )
            for rank in (0, 1)
        ]
        try:
            records = [json.loads(receiver.recv(4096)) for _ in range(40)]
            for child in children:
                assert child.wait(timeout=3) == 0
        finally:
            for child in children:
                if child.poll() is None:
                    child.kill()
                child.wait(timeout=3)
    journals = list(tmp_path.glob("r9v-operations-*.jsonl"))
    assert len(journals) == 2
    for rank in (0, 1):
        remote = [r for r in records if r["rank"] == rank]
        local = [
            json.loads(line)
            for line in next(tmp_path.glob(f"r9v-operations-{rank}-*.jsonl"))
            .read_text()
            .splitlines()
        ]
        assert local == remote
        assert [r["sequence"] for r in local] == list(range(1, 21))
        assert len({(r["pid"], r["start_ticks"], r["thread_id"]) for r in local}) == 1
        assert local[-1]["stage"] == "execute_model_return"


def test_worker_operation_attribution_requires_armed_container():
    tracker = RunTracker()
    run = str(uuid.uuid4())
    record = {
        "kind": "r9v.worker_operation",
        "container": "r9v-test",
        "rank": 1,
        "pid": 603,
        "start_ticks": "12345",
        "sequence": 1,
    }
    assert tracker.operation_owner(record) is None
    tracker.observe({"kind": "r9v.session_start", "run_id": run, "seconds": 180}, 0)
    tracker.observe({"kind": "r9v.progress", "run_id": run, "container": "r9v-test"}, 1)
    assert tracker.operation_owner(record) == (
        run,
        "r9v-test",
        "rank-1-pid-603-12345.jsonl",
    )
    assert tracker.operation_owner({**record, "start_ticks": "../escape"}) is None
    assert tracker.operation_owner({**record, "container": "r9v-other"}) is None


def test_early_job_identity_triggers_capture_before_ip_dump():
    tracker = RunTracker()
    run = str(uuid.uuid4())
    tracker.observe({"kind": "r9v.session_start", "run_id": run, "seconds": 180}, 1)
    tracker.observe({"kind": "r9v.progress", "run_id": run, "container": "r9v-test"}, 2)
    tracker.observe_kernel(
        {
            "channel": "netconsole",
            "received_utc": "1970-01-01T00:01:40+00:00",
            "message": "r9v amdgpu job timeout entry before IP dump",
        },
        3,
        100,
    )
    assert len(tracker.due(3)) == 1


@pytest.mark.parametrize("trace_source", [False, True])
def test_observer_freezes_worker_history_before_requesting_remote_capture(
    tmp_path, trace_source
):
    import datetime
    import os
    import signal
    import subprocess
    from pathlib import Path

    run = str(uuid.uuid4())
    output = tmp_path / "evidence"
    binaries = tmp_path / "bin"
    binaries.mkdir()
    records = [
        {"kind": "r9v.session_start", "run_id": run, "seconds": 180},
        {"kind": "r9v.progress", "run_id": run, "container": "r9v-fixture"},
        {
            "kind": "r9v.worker_operation",
            "container": "r9v-fixture",
            "rank": 0,
            "pid": 603,
            "start_ticks": "12345",
            "sequence": 1,
            "stage": "execute_model_enter",
        },
    ]
    envelopes = [{"channel": "userspace", "message": json.dumps(r)} for r in records]
    envelopes.append(
        {
            "channel": "netconsole",
            "message": "amdgpu: device lost from bus!",
            "received_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        }
    )
    if trace_source:
        output.mkdir()
        identity = {
            "sender": "192.168.1.231",
            "boot_id": "fixture-boot",
            "wall_time": time.time(),
            "monotonic_ns": 1000_000_000_000,
            "recorder_start_ns": 900_000_000_000,
        }
        (output / "trace-identity.json").write_text(json.dumps(identity))
        envelopes[-1] = {
            "channel": "userspace",
            "source": [identity["sender"], 123],
            "received_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "message": json.dumps(
                {
                    "source": "r9v-timeout-trace",
                    "kind": "first_timeout",
                    "boot_id": identity["boot_id"],
                    "monotonic_ns": identity["monotonic_ns"],
                    "trace": "timeout_entry: ring=comp_1 pasid=7",
                }
            ),
        }
    receiver = binaries / "sudo"
    receiver.write_text(
        f"#!{sys.executable}\nimport time\n"
        + "\n".join(
            "print(" + repr(json.dumps(row)) + ", flush=True)" for row in envelopes
        )
        + "\ntime.sleep(20)\n"
    )
    frozen = output / run / "r9v-fixture-operations" / "first-fault"
    marker = tmp_path / "ssh-invoked"
    remote = binaries / "ssh"
    remote.write_text(
        f"#!{sys.executable}\nfrom pathlib import Path\n"
        f"assert (Path({str(frozen)!r}) / 'rank-0-pid-603-12345.jsonl').exists()\n"
        f"Path({str(marker)!r}).write_text('capture after freeze')\n"
    )
    for path in (receiver, remote):
        path.chmod(0o700)
    proc = subprocess.Popen(
        [
            sys.executable,
            str(Path("tools/remote_watch.py").resolve()),
            "--output",
            str(output),
            "--remote-helper",
            "/fixture-unused",
        ],
        env={**os.environ, "PATH": str(binaries) + os.pathsep + os.environ["PATH"]},
        start_new_session=True,
    )
    try:
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline and not marker.exists():
            assert proc.poll() is None
            time.sleep(0.02)
        assert marker.read_text() == "capture after freeze"
        checkpoint = json.loads((output / "observer-state.json").read_text())
        assert checkpoint["runs"][run]["captured"] == ["r9v-fixture"]
        assert json.loads((frozen / "trigger.json").read_text())["reason"].startswith(
            "timeout recorder" if trace_source else "kernel fault:"
        )
    finally:
        os.killpg(proc.pid, signal.SIGTERM)
        proc.wait(timeout=3)
