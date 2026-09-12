# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from tools import capture_runtime as capture


def test_request_stall_requires_pending_work_and_stops_after_progress() -> None:
    tracker = capture.ProgressTracker(180)
    state = {
        "id": "first",
        "status": "running",
        "started_at": "start",
        "restart_count": 0,
    }
    metrics = {"vllm:num_requests_running": 0, "vllm:generation_tokens_total": 10}
    assert tracker.observe(state, metrics, 0) == []
    assert tracker.observe(state, metrics, 3600) == []
    metrics["vllm:num_requests_running"] = 1
    assert tracker.observe(state, metrics, 3601) == []
    assert tracker.observe(state, metrics, 3780) == []
    assert tracker.observe(state, metrics, 3781) == ["possible_request_stall"]
    metrics["vllm:generation_tokens_total"] += 1
    assert tracker.observe(state, metrics, 3782) == []
    state["restart_count"] = 1
    assert tracker.observe(state, metrics, 4000) == ["container_restarted_or_replaced"]
    assert tracker.observe(state, {"error": "timed out"}, 4200) == [
        "metrics_unavailable"
    ]
    assert tracker.observe(state, metrics, 4400) == []


@pytest.mark.parametrize("oom", [False, True])
def test_exit_137_is_distinguished_from_confirmed_docker_oom(oom) -> None:
    events = capture.ProgressTracker(180).observe(
        {"status": "exited", "exit_code": 137, "oom_killed": oom}, {}, 1
    )
    assert "container_exited" in events
    assert ("docker_reports_oom_killed" in events) is oom
    assert "possible_request_stall" not in events


@pytest.mark.parametrize(
    ("code", "expected"),
    [
        ('print(\'{"status": "running"}\')', {"status": "running"}),
        ('print("x" * 300000)', {"error": "command output exceeded capture limit"}),
        ("import time; time.sleep(60)", {"error": "command timed out"}),
        ('print("not json")', {"error": "invalid command JSON"}),
    ],
)
def test_docker_probe_survives_timeout_and_oversized_output(code, expected) -> None:
    assert capture.command_json([sys.executable, "-c", code], timeout=1) == expected


@pytest.fixture
def metrics_server():
    class Handler(BaseHTTPRequestHandler):
        status = 200
        body = b""

        def do_GET(self):
            assert self.path == "/metrics"
            self.send_response(self.status)
            self.end_headers()
            self.wfile.write(self.body)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server.server_port, Handler
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def test_metrics_capture_keeps_only_finite_aggregate_counters(metrics_server) -> None:
    port, handler = metrics_server
    handler.body = (
        b'vllm:generation_tokens_total{engine="0"} 10\n'
        b'vllm:generation_tokens_total{engine="1"} 20\n'
        b"vllm:num_requests_running 1\n"
        b"vllm:kv_cache_usage_perc NaN\n"
        b'private_metric{prompt="do not capture"} 5\n'
    )
    assert capture.fetch_metrics(port) == {
        "vllm:generation_tokens_total": 30,
        "vllm:num_requests_running": 1,
    }


def test_metrics_capture_rejects_oversized_responses_and_http_errors(
    metrics_server,
) -> None:
    port, handler = metrics_server
    handler.body = b"x" * (capture.MAX_RESPONSE_BYTES + 1)
    assert capture.fetch_metrics(port) == {"error": "metrics exceeded capture limit"}
    handler.status = 503
    handler.body = b""
    assert capture.fetch_metrics(port) == {"error": "metrics HTTP 503"}


def test_missing_progress_counters_are_reported_as_unavailable(metrics_server) -> None:
    port, handler = metrics_server
    handler.body = b"unrelated_counter 1\n"
    assert capture.fetch_metrics(port) == {
        "error": "required progress metrics unavailable"
    }


def test_cgroup_capture_retains_oom_counters_and_handles_exited_pid(tmp_path) -> None:
    proc = tmp_path / "proc"
    sys_root = tmp_path / "sys"
    (proc / "123").mkdir(parents=True)
    (proc / "123/cgroup").write_text("0::/system.slice/docker-test.scope\n")
    group = sys_root / "fs/cgroup/system.slice/docker-test.scope"
    group.mkdir(parents=True)
    (group / "memory.events").write_text("oom 3\noom_kill 1\n")
    (group / "memory.current").write_text("123456")
    snapshot = capture.cgroup_snapshot(123, proc, sys_root)
    assert snapshot["memory.events"] == "oom 3\noom_kill 1"
    assert snapshot["memory.current"] == "123456"
    assert snapshot["memory.max"] is None
    assert "unavailable" in capture.cgroup_snapshot(0, proc, sys_root)
    assert "unavailable" in capture.cgroup_snapshot(456, proc, sys_root)


def test_pcie_capture_preserves_negotiated_and_maximum_values_for_each_hop(
    tmp_path,
) -> None:
    bridge = tmp_path / "devices/pci0000:00/0000:00:01.0"
    endpoint = bridge / "0000:03:00.0"
    endpoint.mkdir(parents=True)
    by_bus = tmp_path / "bus/pci/devices"
    by_bus.mkdir(parents=True)
    (by_bus / endpoint.name).symlink_to(endpoint)
    for node, width in ((bridge, "4"), (endpoint, "16")):
        (node / "vendor").write_text("0x1002")
        (node / "current_link_speed").write_text("2.5 GT/s PCIe")
        (node / "current_link_width").write_text(width)
        (node / "max_link_speed").write_text("16.0 GT/s PCIe")
        (node / "max_link_width").write_text("16")
    (endpoint / "class").write_text("0x030000")
    (endpoint / "device").write_text("0x7551")
    (device,) = capture.pcie_snapshot(tmp_path)
    assert [hop["bdf"] for hop in device["hops"]] == [endpoint.name, bridge.name]
    assert [hop["current_link_width"] for hop in device["hops"]] == ["16", "4"]
    assert device["hops"][1]["max_link_width"] == "16"
    assert device["hops"][1]["current_link_speed"] == "2.5 GT/s PCIe"


def test_capture_stops_at_byte_limit_without_corrupting_prior_records(tmp_path) -> None:
    path = tmp_path / "capture.jsonl"
    with path.open("wb") as output:
        remaining = capture.write_record(output, {"event": "first"}, 40)
        with pytest.raises(RuntimeError, match="byte limit"):
            capture.write_record(output, {"event": "x" * 100}, remaining)
    assert json.loads(path.read_text()) == {"event": "first"}
    assert path.stat().st_size <= 40


def test_capture_refuses_to_overwrite_prior_evidence(tmp_path) -> None:
    path = tmp_path / "capture.jsonl"
    path.write_text("original evidence")
    assert capture.main(["--output", str(path), "--duration", "1"]) == 1
    assert path.read_text() == "original evidence"


def test_resource_tracker_reports_new_faults_not_historical_counts():
    tracker = capture.ResourceTracker()
    record = {
        "host": {"vmstat": {"oom_kill": 2}},
        "cgroup": {"memory.events": "oom 3\noom_kill 1"},
        "pcie": [{"hops": [{"bdf": "bridge", "aer_dev_fatal": "TOTAL_ERR_FATAL 1"}]}],
    }
    assert tracker.observe(record) == []
    assert tracker.observe(record) == []
    record["host"]["vmstat"]["oom_kill"] += 1
    record["cgroup"]["memory.events"] = "oom 4\noom_kill 2"
    record["pcie"][0]["hops"][0]["aer_dev_fatal"] = "TOTAL_ERR_FATAL 2"
    assert set(tracker.observe(record)) == {
        "counter_increased:host_oom_kill",
        "counter_increased:cgroup_oom",
        "counter_increased:cgroup_oom_kill",
        "counter_increased:bridge:aer_dev_fatal",
    }


def test_queued_requests_without_running_work_can_stall():
    tracker = capture.ProgressTracker(10)
    state = {"status": "running"}
    metrics = {
        "vllm:num_requests_running": 0,
        "vllm:num_requests_waiting": 1,
        "vllm:generation_tokens_total": 10,
    }
    assert tracker.observe(state, metrics, 0) == []
    assert tracker.observe(state, metrics, 10) == ["possible_request_stall"]


def test_cgroup_process_residency_preserves_shared_memory_without_summing(tmp_path):
    from tools.capture_runtime import process_residency

    group = tmp_path / "group"
    group.mkdir()
    (group / "cgroup.procs").write_text("123\n456\n789\n")
    for pid, rank_pid in ((123, 603), (456, 627)):
        proc = tmp_path / str(pid)
        proc.mkdir()
        (proc / "stat").write_text(f"{pid} (worker name) " + " ".join(["S"] + ["0"] * 18 + ["42"]))
        (proc / "status").write_text(f"Name:\tworker\nNSpid:\t{pid} {rank_pid}\nRssAnon:\t1024 kB\nRssShmem:\t8192 kB\nVmPin:\t0 kB\nVmSwap:\t2048 kB\n")
    result = process_residency(group, tmp_path)
    assert result["listed_processes"] == 3  # Exited PID is omitted, not fatal.
    assert len(result["processes"]) == 2
    assert result["rss_additive"] is False
    assert not result["truncated"]
    assert result["processes"][0]["RssShmem_bytes"] == 8192 * 1024
    assert result["processes"][1]["NSpid"] == "456 627"
    assert result["processes"][1]["start_ticks"] == "42"


def test_host_snapshot_includes_bounded_normal_zone_telemetry(tmp_path):
    (tmp_path / "zoneinfo").write_text(
        "Node 0, zone Normal\n  pages free     10\n        min      5\n        low      20\n        high     30\n"
    )
    snapshot = capture.host_snapshot(tmp_path)
    assert snapshot["normal_zones"][0]["free_bytes"] < snapshot["normal_zones"][0]["low_bytes"]
