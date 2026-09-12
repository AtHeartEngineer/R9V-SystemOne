#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Passively retain rolling crash evidence until the owned container stops."""

import argparse
import json
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

try:
    from tools import capture_runtime as capture
    from tools.observability import DriverSampler, Reporter
except ModuleNotFoundError:
    import capture_runtime as capture
    from observability import DriverSampler, Reporter

COLLECTOR_LOG_LIMIT = 256 * 1024


def spawn_collector(command, log_path):
    """Drain both streams continuously, retaining only a bounded private prefix."""
    descriptor = os.open(log_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    stream = os.fdopen(descriptor, "wb")
    try:
        process = subprocess.Popen(command, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, start_new_session=True)
    except OSError:
        stream.close()
        raise
    process._r9v_group_owned = True
    process._r9v_log_path = str(log_path)

    def drain():
        remaining = COLLECTOR_LOG_LIMIT
        try:
            with process.stdout, stream:
                while block := process.stdout.read(8192):
                    if remaining:
                        kept = block[:remaining]
                        stream.write(kept)
                        stream.flush()
                        remaining -= len(kept)
        except OSError:
            # The collector exit/timeout remains recorded by the supervisor.
            pass

    thread = threading.Thread(target=drain, name="r9v-collector-log", daemon=True)
    process._r9v_log_thread = thread
    thread.start()
    return process


def collect(output, container, port):
    # The support collector exclusively creates output. Its supervisor log is a sibling.
    return spawn_collector([
        sys.executable, str(Path(__file__).with_name("support_bundle.py")),
        "--output", str(output), "--container", container, "--port", str(port),
    ], output.with_name(output.name + ".collector.log"))


def stop_capture(process):
    if getattr(process, "_r9v_group_owned", False):
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except OSError:
            if process.poll() is None:
                process.kill()
    elif process.poll() is None:
        process.kill()
    try:
        process.wait(timeout=1)
    except subprocess.TimeoutExpired:
        pass
    thread = getattr(process, "_r9v_log_thread", None)
    if thread is not None:
        thread.join(timeout=1)


def finish_log(process):
    thread = getattr(process, "_r9v_log_thread", None)
    if thread is not None:
        thread.join(timeout=0.2)
        if thread.is_alive():
            stop_capture(process)


def append(directory, record, limit=8 * 2**20):
    payload = (json.dumps(record) + "\n").encode()
    if len(payload) > limit:
        raise ValueError("individual telemetry record exceeds rotation limit")
    current = directory / "timeline.jsonl"
    if current.exists() and current.stat().st_size + len(payload) > limit:
        for previous, target in [
            ("timeline.1.jsonl", "timeline.2.jsonl"),
            ("timeline.jsonl", "timeline.1.jsonl"),
        ]:
            source = directory / previous
            if source.exists():
                source.replace(directory / target)
    descriptor = os.open(current, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o600)
    with os.fdopen(descriptor, "ab") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def _watch(container, port, output, driver, reporter, captures):
    output.mkdir(mode=0o700, parents=True, exist_ok=False)
    tracker = capture.ProgressTracker(180)
    resources = capture.ResourceTracker()
    bundled = False
    identity = None
    unavailable = 0
    collector_events = []

    def save_collector_events():
        (output / "collector-events.json").write_text(json.dumps(collector_events[-64:], indent=2) + "\n")

    while True:
        for process, started in list(captures):
            code = process.poll()
            if code is not None:
                finish_log(process)
                collector_events.append({"time": time.time(), "event": "collector_exit", "returncode": code})
                captures.remove((process, started))
                save_collector_events()
            elif time.monotonic() - started > 120:
                stop_capture(process)
                finish_log(process)
                captures.remove((process, started))
                collector_events.append({"time": time.time(), "event": "collector_timeout", "budget_seconds": 120})
                save_collector_events()
        sample = driver.poll()
        state = capture.command_json(
            ["docker", "inspect", "--format", capture.INSPECT_FORMAT, container]
        )
        unavailable = unavailable + 1 if "error" in state else 0
        replaced = identity is not None and state.get("id") not in (None, identity)
        identity = identity or state.get("id")
        metrics = capture.fetch_metrics(port)
        workers = capture.worker_snapshot(int(state.get("pid", 0)))
        record = {
            "time": time.time(),
            "container": state,
            "metrics": metrics,
            "workers": workers,
            "scheduler": capture.scheduler_snapshot(int(state.get("pid", 0))),
            "host": capture.host_snapshot(),
            "gpus": (sample.get("sample") or {}).get("gpus", []),
            "pcie": (sample.get("sample") or {}).get("pcie", []),
            "driver": sample,
            "cgroup": capture.cgroup_snapshot(int(state.get("pid", 0))),
        }
        record["events"] = tracker.observe(state, metrics, time.monotonic(), workers)
        record["events"] += resources.observe(record)
        if replaced:
            record["events"].append(
                "container_replaced: ending capture for original container"
            )
        if unavailable >= 3:
            record["events"].append(
                "container_error: identity unavailable for three polls"
            )
        if sample["status"] in ("stalled", "failed"):
            record["events"].append("driver_error: " + sample["status"])
        reporter.send(
            "r9v.watch_progress",
            container=container,
            driver_status=sample["status"],
            events=record["events"],
        )
        append(output, record)
        failed = state.get("oom_killed") or (
            state.get("status") in ("exited", "dead") and state.get("exit_code", 0) != 0
        )
        suspicious = failed or any(
            "stall" in event
            or "error" in event
            or event.startswith("counter_increased:")
            for event in record["events"]
        )
        terminal_failure = failed and state.get("status") in ("exited", "dead")
        if suspicious and (not bundled or terminal_failure):
            # An early pressure warning must not consume the later crash evidence.
            reporter.send(
                "r9v.suspect",
                container=container,
                reason="; ".join(record["events"]) or "container failed",
            )
            early_path = output / ("early-final" if bundled else "early")
            if container.startswith("r9v-") and not early_path.exists():
                try:
                    early = spawn_collector([
                        sys.executable, str(Path(__file__).with_name("early_capture.py")),
                        "--container", container, "--output", str(early_path),
                        "--reason", "watcher detected failure",
                    ], early_path.with_name(early_path.name + ".collector.log"))
                    captures.append((early, time.monotonic()))
                except OSError as error:
                    collector_events.append({"time": time.time(), "event": "collector_spawn_error", "kind": "early", "error": str(error)})
                    save_collector_events()
            try:
                process = collect(output / ("support-final" if bundled else "support"), identity or container, port)
            except OSError as error:
                collector_events.append({"time": time.time(), "event": "collector_spawn_error", "kind": "support", "error": str(error)})
                save_collector_events()
                process = None
            if process is not None:
                captures.append((process, time.monotonic()))
            bundled = True
        if replaced or unavailable >= 3 or state.get("status") in ("exited", "dead"):
            return
        time.sleep(5)


def watch(container, port, output):
    driver, reporter, captures = DriverSampler(), Reporter(), []
    try:
        _watch(container, port, output, driver, reporter, captures)
    finally:
        driver.close()
        for process, started in captures:
            try:
                process.wait(timeout=max(0, started + 120 - time.monotonic()))
                finish_log(process)
                event = {"time": time.time(), "event": "collector_exit", "returncode": process.poll()}
            except subprocess.TimeoutExpired:
                stop_capture(process)
                finish_log(process)
                event = {"time": time.time(), "event": "collector_timeout", "budget_seconds": 120}
            events_path = output / "collector-events.json"
            prior = json.loads(events_path.read_text()) if events_path.is_file() else []
            prior.append(event)
            events_path.write_text(json.dumps(prior[-64:], indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--container", required=True)
    parser.add_argument("--port", type=int, default=8004)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.container.startswith("-") or not 1 <= args.port <= 65535:
        parser.error("invalid container/port")
    watch(args.container, args.port, args.output)


if __name__ == "__main__":
    main()
