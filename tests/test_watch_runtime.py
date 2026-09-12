import json
import time
from types import SimpleNamespace

import pytest

from tools.watch_runtime import append


def test_long_prefill_progress_does_not_consume_the_crash_bundle():
    from tools.capture_runtime import ProgressTracker

    tracker = ProgressTracker(180)
    state = {"id": "same", "status": "running"}
    metrics = {
        "vllm:generation_tokens_total": 0,
        "vllm:prompt_tokens_total": 0,
        "vllm:num_requests_running": 1,
    }
    for now, steps in [(0, 1), (200, 5), (400, 10)]:
        assert "possible_request_stall" not in tracker.observe(
            state,
            metrics,
            now,
            [{"rank": 0, "steps": steps}, {"rank": 1, "steps": steps}],
        )
    assert "possible_request_stall" in tracker.observe(
        state, metrics, 600, [{"rank": 0, "steps": 10}, {"rank": 1, "steps": 10}]
    )


def test_capture_detects_preemption_while_public_metrics_are_stale():
    from tools.capture_runtime import ResourceTracker

    tracker = ResourceTracker()
    row = {"host": {}, "cgroup": {}, "pcie": []}
    assert tracker.observe(row) == []
    row["scheduler"] = {"total_preemptions": 1}
    assert "counter_increased:scheduler_preemptions" in tracker.observe(row)


def test_rolling_capture_retains_recent_records_with_bounded_files(tmp_path):
    for i in range(30):
        append(tmp_path, {"sequence": i, "payload": "x" * 40}, limit=160)
    files = sorted(tmp_path.glob("timeline*.jsonl"))
    assert len(files) == 3
    assert all(p.stat().st_size <= 160 for p in files)
    assert (
        json.loads((tmp_path / "timeline.jsonl").read_text().splitlines()[-1])[
            "sequence"
        ]
        == 29
    )


def test_watcher_stops_on_container_replacement(tmp_path, monkeypatch):
    from tools import watch_runtime as watcher

    monkeypatch.setattr(watcher.time, "sleep", lambda _: None)
    states = iter(
        [{"id": "old", "status": "running"}, {"id": "new", "status": "running"}]
    )
    monkeypatch.setattr(watcher.capture, "command_json", lambda *a: next(states))
    for name in ("fetch_metrics", "host_snapshot", "cgroup_snapshot"):
        monkeypatch.setattr(watcher.capture, name, lambda *a: {})
    for name in ("gpu_snapshot", "pcie_snapshot"):
        monkeypatch.setattr(watcher.capture, name, list)
    watcher.watch("name", 8004, tmp_path / "watch")
    rows = [
        json.loads(line)
        for line in (tmp_path / "watch/timeline.jsonl").read_text().splitlines()
    ]
    assert len(rows) == 2
    assert any("replaced" in event for event in rows[-1]["events"])


def test_watcher_saves_oom_evidence_after_container_exit(tmp_path, monkeypatch):
    from tools import watch_runtime as watcher

    state = {
        "id": "failed-id",
        "status": "exited",
        "oom_killed": True,
        "exit_code": 137,
    }
    monkeypatch.setattr(watcher.capture, "command_json", lambda *a: state)
    for name in ("fetch_metrics", "host_snapshot", "cgroup_snapshot"):
        monkeypatch.setattr(watcher.capture, name, lambda *a: {})
    for name in ("gpu_snapshot", "pcie_snapshot"):
        monkeypatch.setattr(watcher.capture, name, list)
    bundles = []
    monkeypatch.setattr(
        watcher, "collect", lambda output, container, port: bundles.append(container)
    )
    watcher.watch("name", 8004, tmp_path / "watch")
    assert bundles == ["failed-id"]
    assert json.loads((tmp_path / "watch/timeline.jsonl").read_text())["container"][
        "oom_killed"
    ]


def test_watcher_records_collector_timeout(tmp_path, monkeypatch):
    from tools import watch_runtime as watcher

    class Collector:
        def poll(self):
            return None
        def kill(self):
            pass
        def wait(self, **kwargs):
            return -9

    state = {"id": "failed-id", "status": "exited", "exit_code": 1}
    monkeypatch.setattr(watcher.capture, "command_json", lambda *a: state)
    monkeypatch.setattr(watcher.capture, "fetch_metrics", lambda *a: {})
    monkeypatch.setattr(watcher.capture, "host_snapshot", lambda *a: {})
    monkeypatch.setattr(watcher.capture, "scheduler_snapshot", lambda *a: {})
    monkeypatch.setattr(watcher.capture, "cgroup_snapshot", lambda *a: {})
    monkeypatch.setattr(watcher.capture, "scheduler_snapshot", lambda *a: {})
    monkeypatch.setattr(watcher.capture, "cgroup_snapshot", lambda *a: {})
    monkeypatch.setattr(watcher.capture, "gpu_snapshot", lambda *a: [])
    monkeypatch.setattr(watcher.capture, "pcie_snapshot", lambda *a: [])
    monkeypatch.setattr(watcher.capture, "worker_snapshot", lambda *a: [])
    monkeypatch.setattr(watcher.capture, "scheduler_snapshot", lambda *a: {})
    monkeypatch.setattr(watcher, "collect", lambda *a: None)
    watcher._watch("name", 8004, tmp_path / "watch", SimpleNamespace(poll=lambda: {"status": "ok"}), SimpleNamespace(send=lambda *a, **k: None), [(Collector(), time.monotonic() - 121)])
    events = json.loads((tmp_path / "watch/collector-events.json").read_text())
    assert events[0]["event"] == "collector_timeout"


def test_stop_capture_uses_owned_process_group_when_available(monkeypatch):
    from tools import watch_runtime as watcher
    calls = []

    class Process:
        pid = 4321
        _r9v_group_owned = True
        def poll(self):
            return None
        def wait(self, **kwargs):
            return -9

    monkeypatch.setattr(watcher.os, "killpg", lambda pid, sig: calls.append((pid, sig)))
    watcher.stop_capture(Process())
    assert calls == [(4321, watcher.signal.SIGKILL)]


def test_watcher_records_support_spawn_failure(tmp_path, monkeypatch):
    from tools import watch_runtime as watcher
    state = {"id": "failed-id", "status": "exited", "exit_code": 1}
    monkeypatch.setattr(watcher.capture, "command_json", lambda *a: state)
    monkeypatch.setattr(watcher.capture, "fetch_metrics", lambda *a: {})
    for name in ("host_snapshot", "cgroup_snapshot", "gpu_snapshot", "pcie_snapshot", "worker_snapshot", "scheduler_snapshot"):
        monkeypatch.setattr(watcher.capture, name, lambda *a: [])
    monkeypatch.setattr(watcher.capture, "host_snapshot", lambda *a: {})
    monkeypatch.setattr(watcher.capture, "scheduler_snapshot", lambda *a: {})
    monkeypatch.setattr(watcher.capture, "cgroup_snapshot", lambda *a: {})
    monkeypatch.setattr(watcher, "collect", lambda *a: (_ for _ in ()).throw(OSError("spawn denied")))
    watcher._watch("name", 8004, tmp_path / "watch", SimpleNamespace(poll=lambda: {"status": "ok"}), SimpleNamespace(send=lambda *a, **k: None), [])
    events = json.loads((tmp_path / "watch/collector-events.json").read_text())
    assert events[0]["event"] == "collector_spawn_error"


def test_watch_finally_preserves_events_and_waits_remaining_budget(tmp_path, monkeypatch):
    from tools import watch_runtime as watcher
    waits = []

    class Collector:
        pid = 9876
        def poll(self):
            return None
        def wait(self, timeout=None):
            waits.append(timeout)
            raise watcher.subprocess.TimeoutExpired("collector", timeout)
        def kill(self):
            pass

    def fake_watch(container, port, output, driver, reporter, captures):
        output.mkdir(mode=0o700, parents=True)
        (output / "collector-events.json").write_text(json.dumps([{"event": "prior_timeout"}]))
        captures.append((Collector(), watcher.time.monotonic()))

    monkeypatch.setattr(watcher, "_watch", fake_watch)
    monkeypatch.setattr(watcher, "DriverSampler", lambda: SimpleNamespace(close=lambda: None))
    watcher.watch("name", 8004, tmp_path / "watch")
    events = json.loads((tmp_path / "watch/collector-events.json").read_text())
    assert events[0]["event"] == "prior_timeout"
    assert events[1]["event"] == "collector_timeout"
    assert waits and 110 < waits[0] <= 120


def test_early_preemption_bundle_does_not_hide_a_later_crash(tmp_path, monkeypatch):
    from tools import watch_runtime as watcher

    states = iter(
        [
            {"id": "same", "status": "running"},
            {"id": "same", "status": "running"},
            {"id": "same", "status": "exited", "exit_code": 137, "oom_killed": True},
        ]
    )
    monkeypatch.setattr(watcher.time, "sleep", lambda _: None)
    monkeypatch.setattr(watcher.capture, "command_json", lambda *a: next(states))
    for name in ("fetch_metrics", "host_snapshot", "cgroup_snapshot"):
        monkeypatch.setattr(watcher.capture, name, lambda *a: {})
    counts = iter([0, 1, 1])
    monkeypatch.setattr(
        watcher.capture,
        "scheduler_snapshot",
        lambda *a: {"total_preemptions": next(counts)},
    )
    for name in ("gpu_snapshot", "pcie_snapshot", "worker_snapshot"):
        monkeypatch.setattr(watcher.capture, name, lambda *a: [])
    bundles = []
    monkeypatch.setattr(
        watcher, "collect", lambda output, *a: bundles.append(output.name)
    )
    watcher.watch("name", 8004, tmp_path / "watch")
    assert bundles == ["support", "support-final"]


@pytest.fixture(autouse=True)
def no_real_driver_probes(monkeypatch):
    from types import SimpleNamespace

    from tools import qualify_runtime, watch_runtime

    def factory(*a, **kw):
        return SimpleNamespace(poll=lambda: {"status": "ok"}, close=lambda: None)

    monkeypatch.setattr(watch_runtime, "DriverSampler", factory)
    monkeypatch.setattr(qualify_runtime, "DriverSampler", factory)


def test_collector_does_not_precreate_bundle_and_caps_both_output_streams(tmp_path, monkeypatch):
    from tools import watch_runtime as watcher

    helper = tmp_path / "support_bundle.py"
    helper.write_text("import pathlib,sys\np=pathlib.Path(sys.argv[sys.argv.index('--output')+1])\nassert not p.exists()\np.mkdir()\nsys.stderr.write('stderr retained\\n');sys.stderr.flush()\nsys.stdout.write('x'*400000);sys.stdout.flush()\n")
    monkeypatch.setattr(watcher, "__file__", str(tmp_path / "watch_runtime.py"))
    output = tmp_path / "new-support"
    process = watcher.collect(output, "fixture", 8004)
    assert process.wait(timeout=10) == 0
    watcher.finish_log(process)
    log = output.with_name("new-support.collector.log")
    assert output.is_dir()
    assert log.read_bytes().startswith(b"stderr retained")
    assert log.stat().st_size == watcher.COLLECTOR_LOG_LIMIT
    assert log.stat().st_mode & 0o777 == 0o600
    assert not process._r9v_log_thread.is_alive()
