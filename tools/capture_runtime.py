#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Capture a bounded, read-only timeline around intermittent serving failures."""

from __future__ import annotations

import argparse
import http.client
import json
import math
import os
import selectors
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

try:
    from tools.host_pressure import normal_zones
    from tools.profile_doctor import parse_prometheus_metrics
except ModuleNotFoundError:
    from host_pressure import normal_zones
    from profile_doctor import parse_prometheus_metrics


MAX_RESPONSE_BYTES = 256 * 1024
METRICS = {
    "vllm:num_requests_running",
    "vllm:num_requests_waiting",
    "vllm:generation_tokens_total",
    "vllm:prompt_tokens_total",
    "vllm:request_success_total",
    "vllm:kv_cache_usage_perc",
}
INSPECT_FORMAT = (
    '{"id":{{json .Id}},"image":{{json .Image}},'
    '"status":{{json .State.Status}},"pid":{{json .State.Pid}},'
    '"exit_code":{{json .State.ExitCode}},"oom_killed":{{json .State.OOMKilled}},'
    '"started_at":{{json .State.StartedAt}},"finished_at":{{json .State.FinishedAt}},'
    '"restart_count":{{json .RestartCount}}}'
)


def command_output(command: list[str], timeout: float = 4.0) -> dict:
    """Bound both output and wait time, even if the Docker daemon is stuck."""
    try:
        proc = subprocess.Popen(
            command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT
        )
    except OSError as error:
        return {"error": str(error)}
    output = bytearray()
    deadline = time.monotonic() + timeout
    try:
        assert proc.stdout is not None
        with selectors.DefaultSelector() as selector:
            selector.register(proc.stdout, selectors.EVENT_READ)
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0 or not selector.select(remaining):
                    return {
                        "error": "command timed out",
                        "text": output.decode("utf-8", errors="replace"),
                    }
                chunk = os.read(proc.stdout.fileno(), 8192)
                if not chunk:
                    break
                output.extend(chunk)
                if len(output) > MAX_RESPONSE_BYTES:
                    return {
                        "error": "command output exceeded capture limit",
                        "text": output[:MAX_RESPONSE_BYTES].decode(
                            "utf-8", errors="replace"
                        ),
                    }
        try:
            code = proc.wait(timeout=max(0.01, deadline - time.monotonic()))
        except subprocess.TimeoutExpired:
            return {
                "error": "command timed out",
                "text": output.decode("utf-8", errors="replace"),
            }
        if code:
            return {
                "error": output.decode("utf-8", errors="replace")[:1024],
                "exit_code": code,
            }
        return {"text": output.decode("utf-8", errors="replace")}
    finally:
        if proc.poll() is None:
            proc.kill()
        proc.wait()
        if proc.stdout is not None:
            proc.stdout.close()


def command_json(command: list[str], timeout: float = 4.0) -> dict:
    result = command_output(command, timeout)
    if "error" in result:
        return {key: value for key, value in result.items() if key != "text"}
    try:
        value = json.loads(result["text"])
        return value if isinstance(value, dict) else {"error": "expected JSON object"}
    except ValueError:
        return {"error": "invalid command JSON"}


def sensor_snapshot(device: Path) -> dict:
    """Read hwmon clocks (Hz), temperatures (mC), and power (uW) with labels."""
    return {
        str(path.relative_to(device)): read_text(path)
        for pattern in (
            "freq*_input",
            "freq*_label",
            "temp*_input",
            "temp*_crit",
            "temp*_label",
            "power*_average",
            "power*_cap",
            "fan*_input",
        )
        for path in sorted(device.glob("hwmon/hwmon*/" + pattern))
    }


def gpu_snapshot(sys_root: Path = Path("/sys"), bdfs=None) -> list[dict]:
    result = []
    for device in sorted((sys_root / "bus/pci/devices").glob("*")):
        if bdfs is not None and device.name not in bdfs:
            continue
        if read_text(device / "vendor") != "0x1002":
            continue
        if not (read_text(device / "class") or "").startswith("0x03"):
            continue
        result.append(
            {
                "bdf": device.name,
                **{
                    name: read_text(device / name)
                    for name in (
                        "mem_info_vram_total",
                        "mem_info_vram_used",
                        "mem_info_vis_vram_used",
                        "mem_info_gtt_total",
                        "mem_info_gtt_used",
                        "gpu_busy_percent",
                    )
                },
                "sensors": sensor_snapshot(device),
            }
        )
    return result


def log_snapshot(container: str, since: str) -> dict:
    return {
        "server": command_output(
            [
                "docker",
                "logs",
                "--timestamps",
                "--since",
                since,
                "--tail",
                "200",
                container,
            ]
        ),
        "kernel": command_output(
            [
                "journalctl",
                "-k",
                "--no-pager",
                "--quiet",
                "--since",
                since,
                "--lines",
                "100",
                "--output=short-iso-precise",
            ]
        ),
    }


def read_text(path: Path) -> str | None:
    try:
        with path.open() as source:
            return source.read(8192).strip()
    except OSError:
        return None


def pcie_snapshot(sys_root: Path = Path("/sys"), bdfs=None) -> list[dict]:
    """Keep negotiated and maximum link values separate for every AMD GPU hop."""
    result = []
    for device in sorted((sys_root / "bus/pci/devices").glob("*")):
        if bdfs is not None and device.name not in bdfs:
            continue
        if read_text(device / "vendor") != "0x1002":
            continue
        if not (read_text(device / "class") or "").startswith("0x03"):
            continue
        resolved = device.resolve()
        hops = []
        for node in (resolved, *resolved.parents):
            if not (node / "vendor").exists():
                continue
            hops.append(
                {
                    "bdf": node.name,
                    **{
                        name: read_text(node / name)
                        for name in (
                            "current_link_speed",
                            "current_link_width",
                            "max_link_speed",
                            "max_link_width",
                            "aer_dev_correctable",
                            "aer_dev_nonfatal",
                            "aer_dev_fatal",
                        )
                    },
                }
            )
        result.append(
            {
                "bdf": device.name,
                "device_id": read_text(device / "device"),
                "hops": hops,
            }
        )
    return result


def host_snapshot(proc_root: Path = Path("/proc")) -> dict:
    meminfo = read_text(proc_root / "meminfo") or ""
    zoneinfo = read_text(proc_root / "zoneinfo") or ""
    keys = {
        "MemTotal",
        "MemAvailable",
        "SwapFree",
        "SwapTotal",
        "Mlocked",
        "Unevictable",
    }
    return {
        "boot_id": read_text(proc_root / "sys/kernel/random/boot_id"),
        "uptime": read_text(proc_root / "uptime"),
        "memory_kib": {
            parts[0].rstrip(":"): int(parts[1])
            for line in meminfo.splitlines()
            if len(parts := line.split()) >= 2 and parts[0].rstrip(":") in keys
        },
        "memory_pressure": read_text(proc_root / "pressure/memory"),
        "normal_zones": normal_zones(zoneinfo) if zoneinfo else [],
        "normal_zone_status": "available" if zoneinfo and normal_zones(zoneinfo) else "unavailable",
        "io_pressure": read_text(proc_root / "pressure/io"),
        "vmstat": {
            parts[0]: int(parts[1])
            for line in (read_text(proc_root / "vmstat") or "").splitlines()
            if len(parts := line.split()) == 2
            and parts[0] in {"oom_kill", "pgmajfault", "pswpin", "pswpout"}
        },
    }


def process_residency(cgroup_path: Path, proc_root: Path = Path("/proc")) -> dict:
    """Read host-only status for owned cgroup processes; shared RSS is not additive."""
    pids = (read_text(cgroup_path / "cgroup.procs") or "").split()
    rows = []
    for pid in pids[:256]:
        if not pid.isdecimal():
            continue
        path = proc_root / pid
        try:
            before = (path / "stat").read_text().rsplit(")", 1)[1].split()[19]
            status = (path / "status").read_text()
            after = (path / "stat").read_text().rsplit(")", 1)[1].split()[19]
            if before != after:
                continue
            row = {"host_pid": int(pid), "start_ticks": before}
            for line in status.splitlines():
                key, _, value = line.partition(":")
                if key in ("Name", "NSpid"):
                    row[key] = value.strip()
                if key in ("VmRSS", "RssAnon", "RssFile", "RssShmem", "VmLck", "VmPin", "VmSwap"):
                    row[key + "_bytes"] = int(value.split()[0]) * 1024
            rows.append(row)
        except (OSError, ValueError, IndexError):
            continue  # A process can exit between procfs reads.
    return {"processes": rows, "truncated": len(pids) > 256,
            "rss_additive": False, "listed_processes": len(pids)}


def cgroup_snapshot(
    pid: int, proc_root: Path = Path("/proc"), sys_root: Path = Path("/sys")
) -> dict:
    if pid <= 0:
        return {"unavailable": "container has no running PID"}
    membership = read_text(proc_root / str(pid) / "cgroup") or ""
    for line in membership.splitlines():
        if line.startswith("0::"):
            root = (sys_root / "fs/cgroup").resolve()
            path = (root / line[3:].lstrip("/")).resolve()
            if not path.is_relative_to(root):
                break
            values = {
                name: read_text(path / name)
                for name in (
                    "memory.current",
                    "memory.peak",
                    "memory.max",
                    "memory.events",
                    "memory.stat",
                    "memory.swap.current",
                    "memory.pressure",
                    "cpu.stat",
                    "cpu.max",
                    "cpu.pressure",
                    "cpuset.cpus.effective",
                    "cpuset.mems.effective",
                    "io.pressure",
                    "pids.current",
                )
            }
            values["process_residency"] = process_residency(path, proc_root)
            return values
    return {"unavailable": "cgroup v2 membership unreadable or unsupported"}


def fetch_metrics(port: int) -> dict:
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=3)
    deadline = time.monotonic() + 3
    try:
        # Direct loopback connection: no proxies or redirects.
        connection.request("GET", "/metrics")
        response = connection.getresponse()
        if response.status != 200:
            return {"error": f"metrics HTTP {response.status}"}
        body = bytearray()
        while True:
            if time.monotonic() >= deadline:
                return {"error": "metrics timed out"}
            chunk = response.read1(8192)
            if not chunk:
                break
            body.extend(chunk)
            if len(body) > MAX_RESPONSE_BYTES:
                return {"error": "metrics exceeded capture limit"}
        values = parse_prometheus_metrics(body.decode("utf-8", errors="replace"))
        selected = {
            key: value
            for key, value in values.items()
            if key in METRICS and math.isfinite(value)
        }
        if (
            not {"vllm:num_requests_running", "vllm:generation_tokens_total"}
            <= selected.keys()
        ):
            selected["error"] = "required progress metrics unavailable"
        return selected
    except (OSError, http.client.HTTPException) as error:
        return {"error": str(error)}
    finally:
        connection.close()


def worker_snapshot(pid: int) -> list[dict]:
    """Read bounded worker records through the running container's mount view."""
    records = []
    if pid <= 0:
        return records
    for rank in range(2):
        path = Path(f"/proc/{pid}/root/tmp/r9v-worker-{rank}.json")
        try:
            if path.is_symlink():
                continue
            with path.open("rb") as stream:
                raw = stream.read(65537)
            if len(raw) > 65536:
                continue
            value = json.loads(raw)
            if value.get("rank") == rank:
                stage_path = path.with_name(f"r9v-stage-{rank}.json")
                try:
                    if not stage_path.is_symlink():
                        with stage_path.open("rb") as source:
                            marker = json.loads(source.read(65536))
                        if marker.get("pid") == value.get("pid"):
                            value["stage_marker"] = marker
                except (OSError, ValueError, AttributeError):
                    pass
                records.append(value)
        except (OSError, ValueError, TypeError, AttributeError):
            continue
    return records


def scheduler_snapshot(pid: int) -> dict:
    if pid <= 0:
        return {}
    path = Path(f"/proc/{pid}/root/tmp/r9v-scheduler.json")
    try:
        if path.is_symlink():
            return {}
        with path.open("rb") as stream:
            raw = stream.read(65537)
        value = json.loads(raw) if len(raw) <= 65536 else {}
        return value if isinstance(value, dict) else {}
    except (OSError, ValueError):
        return {}


class ProgressTracker:
    def __init__(self, stall_seconds: float) -> None:
        self.stall_seconds = stall_seconds
        self.identity = None
        self.progress = None
        self.since = None

    def observe(
        self, state: dict, metrics: dict, now: float, workers: list[dict] | None = None
    ) -> list[str]:
        events = []
        identity = (
            state.get("id"),
            state.get("started_at"),
            state.get("restart_count"),
        )
        if self.identity is not None and identity != self.identity and state.get("id"):
            events.append("container_restarted_or_replaced")
            self.since = None
        if state.get("id"):
            self.identity = identity
        if state.get("status") in {"exited", "dead", "restarting", "paused"}:
            events.append("container_" + state["status"])
        if state.get("oom_killed"):
            events.append("docker_reports_oom_killed")
        if "error" in state:
            events.append("container_inspect_unavailable")
        if "error" in metrics:
            events.append("metrics_unavailable")
        progress = tuple(
            metrics.get(key)
            for key in (
                "vllm:generation_tokens_total",
                "vllm:prompt_tokens_total",
                "vllm:request_success_total",
            )
        )
        # Token counters may only update after an entire long prefill. Worker
        # step counters distinguish chunk progress from an unchanging request.
        progress += tuple(
            (row.get("rank"), row.get("steps"))
            for row in (workers or [])
            if type(row.get("steps")) is int
        )
        running = metrics.get("vllm:num_requests_running", 0) + metrics.get(
            "vllm:num_requests_waiting", 0
        )
        if (
            state.get("status") != "running"
            or "error" in metrics
            or running <= 0
            or progress[0] is None
        ):
            self.since = None
        elif self.since is None or progress != self.progress:
            self.since = now
        elif now - self.since >= self.stall_seconds:
            events.append("possible_request_stall")
        self.progress = progress
        return events


class ResourceTracker:
    """Compare counters with the run baseline; old faults are not new failures."""

    def __init__(self):
        self.previous = {}

    def observe(self, record: dict) -> list[str]:
        counters = {
            "host_oom_kill": record["host"].get("vmstat", {}).get("oom_kill", 0),
            "scheduler_preemptions": record.get("scheduler", {}).get(
                "total_preemptions", 0
            ),
        }
        for line in (record["cgroup"].get("memory.events") or "").splitlines():
            parts = line.split()
            if len(parts) == 2 and parts[0] in {"oom", "oom_kill"}:
                counters["cgroup_" + parts[0]] = int(parts[1])
        for device in record["pcie"]:
            for hop in device["hops"]:
                for kind in ("aer_dev_nonfatal", "aer_dev_fatal"):
                    text = hop.get(kind)
                    if text is not None:
                        counters[hop["bdf"] + ":" + kind] = sum(
                            int(parts[-1])
                            for line in text.splitlines()
                            if (parts := line.split()) and parts[-1].isdigit()
                        )
        events = [
            "counter_increased:" + key
            for key, value in counters.items()
            if key in self.previous and value > self.previous[key]
        ]
        self.previous.update(counters)
        return events


def write_record(output, record: dict, remaining: int) -> int:
    data = (json.dumps(record, allow_nan=False) + "\n").encode()
    if len(data) > remaining:
        raise RuntimeError("capture byte limit reached")
    output.write(data)
    output.flush()
    os.fsync(output.fileno())
    return remaining - len(data)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--container", default="r9v-qwen38-flash-next")
    parser.add_argument("--port", type=int, default=8004, help="local metrics port")
    parser.add_argument(
        "--duration",
        type=int,
        default=7200,
        help="seconds, up to 25 hours (allows a final soak request)",
    )
    parser.add_argument(
        "--interval", type=int, default=30, help="seconds between samples"
    )
    parser.add_argument("--stall-seconds", type=int, default=180)
    parser.add_argument("--max-mib", type=int, default=32, help="maximum output size")
    parser.add_argument(
        "--output", type=Path, required=True, help="new JSONL file; never overwritten"
    )
    args = parser.parse_args(argv)
    if not (
        1 <= args.port <= 65535
        and 1 <= args.duration <= 90000
        and args.interval >= 1
        and args.stall_seconds >= 1
        and 1 <= args.max_mib <= 256
    ):
        parser.error(
            "port, duration, interval, stall-seconds, or max-mib is outside its allowed range"
        )
    if not args.container or args.container.startswith("-"):
        parser.error("container must be a name or ID")
    tracker = ProgressTracker(args.stall_seconds)
    resources = ResourceTracker()
    started = time.monotonic()
    remaining = args.max_mib * 1024 * 1024
    since = datetime.now(timezone.utc).isoformat()
    try:
        # Exclusive creation preserves earlier evidence; mode limits local access.
        fd = os.open(args.output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "wb") as output:
            while time.monotonic() - started < args.duration:
                sampled = time.monotonic()
                sample_time = datetime.now(timezone.utc).isoformat()
                state = command_json(
                    ["docker", "inspect", "--format", INSPECT_FORMAT, args.container]
                )
                metrics = fetch_metrics(args.port)
                workers = worker_snapshot(int(state.get("pid", 0)))
                record = {
                    "timestamp": sample_time,
                    "elapsed_seconds": round(time.monotonic() - started, 3),
                    "container": state,
                    "stats": command_json(
                        [
                            "docker",
                            "stats",
                            "--no-stream",
                            "--format",
                            "{{json .}}",
                            args.container,
                        ]
                    ),
                    "metrics": metrics,
                    "workers": workers,
                    "scheduler": scheduler_snapshot(int(state.get("pid", 0))),
                    "host": host_snapshot(),
                    "cgroup": cgroup_snapshot(int(state.get("pid", 0))),
                    "pcie": pcie_snapshot(),
                    "gpus": gpu_snapshot(),
                    "logs": log_snapshot(args.container, since),
                    "events": tracker.observe(
                        state, metrics, time.monotonic(), workers
                    ),
                }
                record["events"].extend(resources.observe(record))
                remaining = write_record(output, record, remaining)
                since = sample_time
                time.sleep(
                    max(
                        0,
                        min(sampled + args.interval, started + args.duration)
                        - time.monotonic(),
                    )
                )
    except KeyboardInterrupt:
        return 130
    except (OSError, RuntimeError) as error:
        print(f"Capture stopped: {error}")
        return 1
    print(f"Capture saved to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
