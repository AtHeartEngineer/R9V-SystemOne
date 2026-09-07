# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace

import pytest

from tools import capture_runtime as capture
from tools import soak_runtime as soak
from tools import support_bundle as support
from tools.host_preflight import check_container_limits, check_resources
from tools.profile_doctor import Reporter


@pytest.fixture
def api():
    class Handler(BaseHTTPRequestHandler):
        status = 200
        payload = {
            "choices": [
                {"finish_reason": "stop", "message": {"content": "Test answer"}}
            ],
            "usage": {"prompt_tokens": 8, "completion_tokens": 2},
        }
        requests = []

        def do_POST(self):
            type(self).requests.append(
                json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            )
            assert self.path == "/v1/chat/completions"
            self.send_response(self.status)
            self.end_headers()
            self.wfile.write(json.dumps(self.payload).encode())

        def log_message(self, *_):
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


def test_soak_real_worker_keeps_token_evidence_without_generated_text(api):
    port, handler = api
    command = [
        sys.executable,
        str(Path(soak.__file__)),
        "--worker",
        "--port",
        str(port),
        "--prompt-repeats",
        "8",
    ]
    result = capture.command_json(command, timeout=3)
    assert result["completion_tokens"] == 2
    assert "Test answer" not in json.dumps(result)
    assert handler.requests[0]["stream"] is False
    assert handler.requests[0]["chat_template_kwargs"]["enable_thinking"] is False


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"choices": [{"finish_reason": None}]},
        {
            "choices": [{"finish_reason": "stop", "message": {"content": ""}}],
            "usage": {"completion_tokens": 2},
        },
        {"choices": [{"finish_reason": "stop"}], "usage": {"completion_tokens": 0}},
    ],
)
def test_soak_rejects_incomplete_or_empty_responses(api, payload):
    port, handler = api
    handler.payload = payload
    assert "error" in soak.request_once(port, "test", 8, 32, 1)


def test_soak_http_failure_is_not_a_pass(api):
    port, handler = api
    handler.status = 500
    assert soak.request_once(port, "test", 8, 32, 1) == {"error": "HTTP 500"}


def test_timeout_retains_partial_raw_evidence_but_json_probe_stays_structured():
    result = capture.command_output(
        [
            sys.executable,
            "-u",
            "-c",
            "import time; print('before crash'); time.sleep(10)",
        ],
        timeout=0.1,
    )
    assert result["error"] == "command timed out"
    assert "before crash" in result["text"]


def test_support_bundle_survives_missing_tools_and_never_overwrites(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(
        capture, "command_output", lambda *_a, **_k: {"error": "not installed"}
    )
    monkeypatch.setattr(capture, "fetch_metrics", lambda *_: {"error": "offline"})
    destination = tmp_path / "evidence"
    manifest = support.collect(destination, "stopped", 1)
    assert set(manifest["probes"].values()) == {"unavailable"}
    assert (
        json.loads((destination / "snapshot.json").read_text())["metrics"]["error"]
        == "offline"
    )
    assert destination.stat().st_mode & 0o777 == 0o700
    assert (destination / "manifest.json").stat().st_mode & 0o777 == 0o600
    with pytest.raises(FileExistsError):
        support.collect(destination, "stopped", 1)


def test_soak_saves_failed_request_before_support_collection(monkeypatch, tmp_path):
    class Monitor:
        def __init__(self, command, **_):
            Path(command[-1]).write_text('{"events": []}\n')

        def poll(self):
            return None

        def send_signal(self, _):
            pass

        def wait(self, **_):
            return 0

    monkeypatch.setattr(soak.subprocess, "Popen", Monitor)
    monkeypatch.setattr(
        capture, "command_json", lambda *_a, **_k: {"error": "command timed out"}
    )

    def collect(output, *_):
        requests = [
            json.loads(line)
            for line in (output.parent / "requests.jsonl").read_text().splitlines()
        ]
        assert requests[-2]["event"] == "request_started"
        assert requests[-1]["error"] == "command timed out"

    monkeypatch.setattr(soak, "collect", collect)
    destination = tmp_path / "soak"
    assert soak.main(["--output", str(destination), "--duration", "1"]) == 1
    assert json.loads((destination / "summary.json").read_text())["status"] == "failed"


def test_container_caps_and_lost_logging_are_actionable():
    config = {
        "Memory": 8 * 1024**3,
        "PidsLimit": 32,
        "AutoRemove": True,
        "LogConfig": {"Type": "none"},
        "Ulimits": [],
    }
    reporter = Reporter()
    check_container_limits(
        reporter,
        lambda *_: subprocess.CompletedProcess([], 0, json.dumps(config)),
        "test",
    )
    assert {check.name for check in reporter.checks if check.status == "FAIL"} == {
        "runtime-auto-remove",
        "runtime-log-retention",
    }
    assert sum(check.name == "runtime-resource-cap" for check in reporter.checks) == 2
    assert any(check.name == "runtime-memlock" for check in reporter.checks)


def test_resource_checks_use_bdf_vram_and_do_not_certify_zero_ram_policy(
    monkeypatch, tmp_path
):
    monkeypatch.delenv("R9V_MODEL_DIR", raising=False)
    monkeypatch.setenv("R9V_MIN_HOST_RAM_BYTES", "0")
    monkeypatch.setenv("R9V_MIN_HOST_AVAILABLE_BYTES", "0")
    pci = tmp_path / "bus/pci/devices/0000:03:00.0"
    pci.mkdir(parents=True)
    (pci / "mem_info_vram_total").write_text(str(16 * 1024**3))
    (pci / "mem_info_vram_used").write_text("0")
    reporter = Reporter()
    check_resources(
        reporter,
        [(0, SimpleNamespace(bdf=pci.name), None, None, None)],
        tmp_path,
        tmp_path,
        tmp_path,
        False,
    )
    assert any(c.name == "gpu-vram" and c.status == "FAIL" for c in reporter.checks)
    assert sum(c.name == "memory-qualification" for c in reporter.checks) == 2


@pytest.mark.parametrize("external", [False, True])
def test_launcher_retains_logs_and_uses_device_groups(tmp_path, external):
    # Execute the launcher against fake Docker/device-free config, not a source-text assertion.
    root = Path(__file__).resolve().parents[1]
    model = tmp_path / "models"
    files = [
        "target/a",
        "target/b",
        "target/c",
        "metadata/config.json",
        "mtp/config.json",
        "mtp/model.safetensors",
        "vision/mmproj",
        "manifests/hot",
        "ple",
    ]
    for relative in files:
        path = model / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    docker = tmp_path / "docker"
    docker.write_text(
        '#!/usr/bin/env python3\nimport sys, json, os\nif sys.argv[1] == "container": sys.exit(1)\nopen(os.environ["TEST_ARGS"], "w").write(json.dumps(sys.argv[1:]))\n'
    )
    docker.chmod(0o755)
    target = tmp_path / "args.json"
    env = dict(
        os.environ,
        PATH=str(tmp_path) + os.pathsep + os.environ["PATH"],
        TEST_ARGS=str(target),
        R9V_MODEL_DIR=str(model),
        R9V_PLE_PATH=str(model / "ple"),
        R9V_CACHE_DIR=str(tmp_path / "cache"),
        R9V_PREFLIGHT="0",
        R9V_TARGET_REL="target/a",
        R9V_TARGET_SHARD2_REL="target/b",
        R9V_TARGET_SHARD3_REL="target/c",
        R9V_MMPROJ_REL="vision/mmproj",
        R9V_MANIFEST_REL="manifests/hot",
    )
    env.pop("R9V_EXPERT_MANIFEST_PATH", None)
    if external:
        (model / "manifests/hot").unlink()
        external_path = tmp_path / "external.json"
        external_path.write_text("{}")
        env["R9V_EXPERT_MANIFEST_PATH"] = str(external_path)
    env.pop("R9V_CONFIG_FILE", None)
    env.pop("R9V_PROFILE", None)
    result = subprocess.run(
        ["bash", str(root / "scripts/launch.sh")],
        env=env,
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0, result.stderr
    args = json.loads(target.read_text())
    assert args[args.index("--log-driver") + 1] == "json-file"
    assert "max-size=20m" in args and "max-file=5" in args
    assert "memlock=-1:-1" in args
    assert "--rm" not in args
    assert "PYTHONFAULTHANDLER=1" in args
    if external:
        assert f"{external_path}:/placement/experts.json:ro" in args
        assert "RADIANCE_TIERED_EXPERT_MANIFEST=/placement/experts.json" in args


@pytest.mark.parametrize(
    "events,code", [([], 0), (["container_restarted_or_replaced"], 1)]
)
def test_completed_soak_requires_clean_independent_telemetry(
    monkeypatch, tmp_path, events, code
):
    class Monitor:
        def __init__(self, command, **_):
            Path(command[-1]).write_text(json.dumps({"events": events}) + "\n")

        def poll(self):
            return None

        def send_signal(self, _):
            pass

        def wait(self, **_):
            return 0

    monkeypatch.setattr(soak.subprocess, "Popen", Monitor)
    monkeypatch.setattr(
        capture,
        "command_json",
        lambda *_a, **_k: {"completion_tokens": 2, "finish_reason": "stop"},
    )
    monkeypatch.setattr(soak, "collect", lambda *_: None)
    destination = tmp_path / "soak"
    assert (
        soak.main(
            [
                "--output",
                str(destination),
                "--duration",
                "1",
                "--prompt-repeats",
                "8",
                "--idle-seconds",
                "1",
            ]
        )
        == code
    )
    summary = json.loads((destination / "summary.json").read_text())
    assert summary["completed_requests"] == 1
    assert summary["status"] == ("passed" if code == 0 else "failed")


def test_doctor_rejects_unconfigured_required_model_payloads(monkeypatch):
    from tools import profile_doctor as doctor

    monkeypatch.delenv("R9V_MODEL_DIR", raising=False)
    monkeypatch.delenv("R9V_PLE_PATH", raising=False)
    reporter = Reporter()
    doctor._check_model_package(reporter, Path("/unused"), "qwen38")
    doctor._check_ple_storage(reporter)
    assert len(reporter.checks) == 2
    assert all(check.status == "FAIL" for check in reporter.checks)


def test_runtime_does_not_apply_prelaunch_available_ram_gate(monkeypatch, tmp_path):
    from tools import profile_doctor as doctor

    (tmp_path / "meminfo").write_text(
        "MemTotal: 128000000 kB\nMemAvailable: 16000000 kB\n"
    )
    monkeypatch.setenv("R9V_MIN_HOST_RAM_BYTES", "64000000000")
    monkeypatch.setenv("R9V_MIN_HOST_AVAILABLE_BYTES", "64000000000")
    monkeypatch.setenv("R9V_REFERENCE_HOST_RAM_BYTES", "0")
    before, during = Reporter(), Reporter()
    doctor._check_host_memory(before, tmp_path)
    doctor._check_host_memory(during, tmp_path, runtime=True)
    assert before.checks[0].status == "FAIL"
    assert during.checks[0].status == "PASS"
