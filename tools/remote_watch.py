#!/usr/bin/env python3
"""Blackbox observer: capture once per arm if an explicitly armed run goes silent."""

import argparse
import datetime
import json
import os
import re
import select
import shlex
import subprocess
import time
import uuid
from pathlib import Path

try:
    from tools.fault_window import FaultWindow
    from tools.trace_gate import TraceGate
except ModuleNotFoundError:
    from fault_window import FaultWindow
    from trace_gate import TraceGate


class RunTracker:
    def __init__(self, silence=15):
        self.silence = silence
        self.runs = {}

    def observe_trace(self, gate, envelope, now, wall_now):
        event = gate.observe(envelope, wall_now)
        if event == "fault":
            for state in self.runs.values():
                if now <= state["deadline"] and state["container"]:
                    state.setdefault("reason", "timeout recorder captured a fault")
        return event

    def observe_kernel(self, envelope, now, wall_now):
        """Use the first fresh kernel fault while SSH may still be responsive."""
        if envelope.get("channel") != "netconsole":
            return
        message = envelope.get("message")
        if not isinstance(message, str):
            return
        try:
            received = datetime.datetime.fromisoformat(envelope["received_utc"])
            if (
                received.tzinfo is None
                or not 0 <= wall_now - received.timestamp() <= 10
            ):
                return
        except (KeyError, ValueError, TypeError):
            return
        if not re.search(
            r"amdgpu.*(?:Dumping IP State(?! Completed)|device lost from bus|"
            r"ring .*timeout|job timeout|first timeout|GPU reset|GPU fault|page fault)|"
            r"watchdog:.*lockup|page allocation failure|Kernel panic",
            message,
            re.IGNORECASE,
        ):
            return
        for state in self.runs.values():
            if now <= state["deadline"] and state["container"]:
                state.setdefault(
                    "reason", "kernel fault: " + message.splitlines()[0][:240]
                )

    def observe(self, record, now):
        try:
            run_id = str(uuid.UUID(record.get("run_id", "")))
        except (ValueError, AttributeError, TypeError):
            return
        kind = record.get("kind")
        if kind == "r9v.session_start":
            seconds = record.get("seconds")
            if not isinstance(seconds, (int, float)) or not 1 <= seconds <= 2700:
                return
            # Repeated starts cannot reset an existing arm's capture latch.
            self.runs.setdefault(
                run_id,
                {
                    "deadline": now + seconds + 60,
                    "last": now,
                    "container": None,
                    "captured": set(),
                },
            )
        state = self.runs.get(run_id)
        if state is None:
            return
        if kind == "r9v.session_lease":
            seconds = record.get("seconds")
            # Renew only the existing live container; never create/rearm a latch.
            if (type(seconds) is int and 1 <= seconds <= 900
                    and record.get("container") == state["container"]
                    and state["container"] is not None
                    and now <= state["deadline"]):
                state["deadline"] = max(state["deadline"], now + seconds + 60)
            return
        if kind == "r9v.session_end":
            del self.runs[run_id]
            return
        if kind == "r9v.progress":
            state["last"] = now
            container = record.get("container") or ""
            if not isinstance(container, str):
                return
            if re.fullmatch(r"r9v-[a-zA-Z0-9_.-]{1,100}", container):
                if container != state["container"]:
                    state.pop("reason", None)
                    state.pop("signature", None)
                state["container"] = container
            workers = record.get("workers") or []
            phase = record.get("phase") or ""
            if not isinstance(phase, str) or not isinstance(workers, list):
                return
            signature = (
                phase,
                tuple(
                    (w.get("rank"), w.get("steps"))
                    for w in workers
                    if isinstance(w, dict)
                ),
            )
            if signature != state.get("signature"):
                state["signature"], state["advanced"] = signature, now
            if (
                workers
                and phase.endswith(
                    ("-warmup", "-measure", "-routes", "-workload", "-pressure")
                )
                and now - state.get("advanced", now) > 60
            ):
                state["reason"] = "worker progress unchanged for 60 seconds"
            if record.get("driver_status") in ("failed", "stalled"):
                state["reason"] = "driver telemetry " + record["driver_status"]
        if kind == "r9v.suspect" and record.get("container") == state["container"]:
            state["reason"] = "workstation requested early evidence"

    def operation_owner(self, record):
        """Accept worker records only for a currently armed container."""
        if record.get("kind") != "r9v.worker_operation":
            return None
        for run_id, state in self.runs.items():
            if record.get("container") == state["container"] and state["container"]:
                rank, pid = record.get("rank"), record.get("pid")
                ticks, sequence = record.get("start_ticks"), record.get("sequence")
                if (
                    type(rank) is int
                    and 0 <= rank < 64
                    and type(pid) is int
                    and pid > 0
                    and isinstance(ticks, str)
                    and ticks.isdecimal()
                    and type(sequence) is int
                    and sequence > 0
                ):
                    return (
                        run_id,
                        state["container"],
                        f"rank-{rank}-pid-{pid}-{ticks}.jsonl",
                    )
        return None

    def due(self, now):
        result = []
        for run_id, state in list(self.runs.items()):
            if now > state["deadline"]:
                del self.runs[run_id]
                continue
            container = state["container"]
            if not container or container in state["captured"]:
                continue
            reason = state.get("reason")
            if now - state["last"] > self.silence:
                reason = "workstation progress channel silent"
            if reason:
                state["captured"].add(container)
                result.append((run_id, container, reason))
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--remote-helper", required=True)
    args = parser.parse_args()
    args.output.mkdir(mode=0o700, parents=True, exist_ok=True)
    tracker = RunTracker()
    identity_path = args.output / "trace-identity.json"
    trace_gate = (
        TraceGate(json.loads(identity_path.read_text()))
        if identity_path.exists()
        else None
    )
    checkpoint = args.output / "observer-state.json"
    boot_id = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
    try:
        saved = json.loads(checkpoint.read_text())
        if saved.get("boot_id") == boot_id:
            tracker.runs = saved["runs"]
            if (
                trace_gate
                and saved.get("trace_boot_id") == trace_gate.identity["boot_id"]
            ):
                trace_gate.highwater = saved.get(
                    "trace_highwater", trace_gate.highwater
                )
                trace_gate.fault = saved.get("trace_fault")
            for state in tracker.runs.values():
                state["captured"] = set(state["captured"])
                state.pop("signature", None)
    except (OSError, ValueError, KeyError, TypeError):
        pass
    reader = subprocess.Popen(
        ["sudo", "-n", "/usr/local/bin/r9v-crash-logs", "--follow", "--lines", "1"],
        stdout=subprocess.PIPE,
        bufsize=0,
    )
    buffer, children = b"", []
    windows = {}
    try:
        while reader.poll() is None:
            if select.select([reader.stdout], [], [], 1)[0]:
                buffer += os.read(reader.stdout.fileno(), 65536)
                if len(buffer) > 1024 * 1024:
                    raise RuntimeError("receiver line exceeded 1 MiB")
                while b"\n" in buffer:
                    line, buffer = buffer.split(b"\n", 1)
                    try:
                        envelope = json.loads(line)
                        if isinstance(envelope, dict):
                            if (
                                trace_gate
                                and tracker.observe_trace(
                                    trace_gate, envelope, time.monotonic(), time.time()
                                )
                                == "fault"
                            ):
                                # Freeze before reading any subsequent operation or issuing SSH.
                                for window in windows.values():
                                    window.freeze(
                                        {
                                            "time": time.time(),
                                            "reason": "timeout recorder",
                                            "event": trace_gate.fault,
                                        }
                                    )
                            tracker.observe_kernel(
                                envelope, time.monotonic(), time.time()
                            )
                        record = json.loads(envelope["message"])
                        if isinstance(record, dict):
                            tracker.observe(record, time.monotonic())
                            owner = tracker.operation_owner(record)
                            if owner:
                                run_id, container, name = owner
                                key = (run_id, container)
                                if key not in windows:
                                    windows[key] = FaultWindow(
                                        args.output
                                        / run_id
                                        / (container + "-operations"),
                                        limit=2**20,
                                        segments=2,
                                    )
                                windows[key].append(
                                    name, json.dumps(record).encode() + b"\n"
                                )
                    except (ValueError, KeyError, TypeError):
                        continue
            now = time.monotonic()
            due = tracker.due(now)
            freeze_errors = {}
            # Freeze locally before the once-only checkpoint and any SSH/cleanup.
            for run_id, container, reason in due:
                window = windows.get((run_id, container))
                if window is None:
                    window = FaultWindow(
                        args.output / run_id / (container + "-operations")
                    )
                try:
                    window.freeze({"time": time.time(), "reason": reason})
                except OSError as error:
                    freeze_errors[(run_id, container)] = str(error)
            saved = {
                key: {**state, "captured": sorted(state["captured"])}
                for key, state in tracker.runs.items()
            }
            temporary = checkpoint.with_suffix(".tmp")
            with temporary.open("w") as state_file:
                json.dump(
                    {
                        "boot_id": boot_id,
                        "runs": saved,
                        "trace_boot_id": trace_gate.identity["boot_id"]
                        if trace_gate
                        else None,
                        "trace_highwater": trace_gate.highwater if trace_gate else None,
                        "trace_fault": trace_gate.fault if trace_gate else None,
                    },
                    state_file,
                )
                state_file.flush()
                os.fsync(state_file.fileno())
            temporary.replace(checkpoint)
            for run_id, container, reason in due:
                directory = args.output / run_id / container
                directory.mkdir(mode=0o700, parents=True, exist_ok=False)
                with (directory / "trigger.json").open("x") as stream:
                    json.dump(
                        {
                            "time": time.time(),
                            "run_id": run_id,
                            "container": container,
                            "reason": reason,
                            "freeze_error": freeze_errors.get((run_id, container)),
                        },
                        stream,
                    )
                    stream.flush()
                    os.fsync(stream.fileno())
                # Only validated IDs cross into the remote shell; helper path is administrator config.
                command = shlex.join(
                    [
                        "python3",
                        args.remote_helper,
                        "--container",
                        container,
                        "--output",
                        "/var/home/dylan/AI-Work/r9v-early-capture/"
                        + run_id
                        + "/"
                        + container,
                        "--reason",
                        reason,
                        "--stream",
                    ]
                )
                with (directory / "evidence.jsonl").open("xb") as stream:
                    process = subprocess.Popen(
                        [
                            "ssh",
                            "-o",
                            "BatchMode=yes",
                            "-o",
                            "ConnectTimeout=3",
                            "-o",
                            "ServerAliveInterval=3",
                            "-o",
                            "ServerAliveCountMax=2",
                            "workstation",
                            command,
                        ],
                        stdout=stream,
                        stderr=subprocess.STDOUT,
                    )
                children.append((process, now, directory))
            for process, started, directory in list(children):
                if process.poll() is None and now - started > 30:
                    process.kill()
                if process.poll() is not None:
                    with (directory / "completion.json").open("x") as stream:
                        json.dump(
                            {"time": time.time(), "exit_code": process.returncode},
                            stream,
                        )
                        stream.flush()
                        os.fsync(stream.fileno())
                    children.remove((process, started, directory))
    finally:
        reader.terminate()
        for process, _, _ in children:
            process.kill()
    raise SystemExit("log reader exited; systemd will restart the observer")


if __name__ == "__main__":
    main()
