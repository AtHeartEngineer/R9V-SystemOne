#!/usr/bin/env python3
"""Independent driver sampler and small, best-effort remote progress records."""

from __future__ import annotations

import argparse
import errno
import json
import os
import select
import socket
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path


class Reporter:
    def __init__(self, run_id=None, target=None):
        self.run_id = (
            run_id or os.environ.get("R9V_OBSERVABILITY_RUN_ID") or str(uuid.uuid4())
        )
        self.producer_id = str(uuid.uuid4())
        self.sequence = 0
        self.lock = threading.Lock()
        self.target = None
        address = (
            target
            if target is not None
            else os.environ.get("R9V_OBSERVABILITY_TARGET", "")
        )
        if address:
            host, port = address.rsplit(":", 1)
            socket.inet_pton(socket.AF_INET, host)  # Numeric only: no blocking DNS.
            if not 1024 <= int(port) <= 65535:
                raise ValueError("observability UDP port must be 1024..65535")
            self.target = (host, int(port))
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.socket.setblocking(False)
        self.boot_id = Path("/proc/sys/kernel/random/boot_id").read_text().strip()

    def send(self, kind, **fields):
        with self.lock:
            self.sequence += 1
            sequence = self.sequence
        record = dict(
            kind=kind,
            run_id=self.run_id,
            sequence=sequence,
            boot_id=self.boot_id,
            producer_id=self.producer_id,
            pid=os.getpid(),
            time=time.time(),
            monotonic_ns=time.monotonic_ns(),
            **fields,
        )
        if self.target:
            raw = json.dumps(record).encode()
            if len(raw) > 12000:
                raise ValueError("remote progress record exceeds 12 KiB")
            try:
                self.socket.sendto(raw, self.target)
            except OSError:
                pass  # Network failure must never block the host guard.
        return record


class DriverSampler:
    """One child only. A wedged driver never blocks or multiplies probe processes."""

    def __init__(self, bdfs=None, timeout=15, command=None):
        self.bdfs, self.timeout = bdfs, timeout
        self.command = command or [
            sys.executable,
            str(Path(__file__).resolve()),
            "--driver-worker",
        ]
        self.process = None
        self.pending = None
        self.latest = None
        self.last_read = None
        self.buffer = b""
        self.failed = None
        self.closed = False

    def poll(self):
        if self.closed:
            return {"status": "closed", "sample": self.latest}
        now = time.monotonic()
        if self.process is None:
            self.process = subprocess.Popen(
                self.command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                bufsize=0,
            )
        while select.select([self.process.stdout], [], [], 0)[0]:
            chunk = os.read(self.process.stdout.fileno(), 65536)
            if not chunk:
                self.failed = f"driver worker exited: {self.process.poll()}"
                break
            self.buffer += chunk
            if len(self.buffer) > 1024 * 1024:
                self.failed = "driver output exceeded 1 MiB"
                break
            while b"\n" in self.buffer:
                line, self.buffer = self.buffer.split(b"\n", 1)
                value = json.loads(line)
                if value.get("kind") == "read_start":
                    self.last_read = value
                elif value.get("kind") == "sample":
                    self.latest = value
                    self.pending = None
                    if value.get("driver_faults"):
                        self.failed = "Driver read faults: " + str(
                            value["driver_faults"]
                        )
        if self.process.poll() is not None:
            self.failed = f"driver worker exited: {self.process.returncode}"
        if not self.failed and self.pending is None:
            try:
                self.process.stdin.write(
                    (json.dumps({"bdfs": self.bdfs}) + "\n").encode()
                )
                self.pending = now
            except OSError as error:
                self.failed = str(error)
        age = now - self.pending if self.pending is not None else 0
        status = (
            "failed"
            if self.failed
            else "stalled"
            if age > self.timeout
            else "ok"
            if self.latest
            else "pending"
        )
        return {
            "status": status,
            "pending_seconds": age,
            "error": self.failed,
            "last_read": self.last_read,
            "sample": self.latest,
            "sample_age_seconds": time.time() - self.latest["finished_at"]
            if self.latest
            else None,
        }

    def close(self):
        self.closed = True
        if self.process:
            if self.process.poll() is None:
                self.process.terminate()
                try:
                    self.process.wait(timeout=0.2)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    try:
                        self.process.wait(timeout=0.2)
                    except subprocess.TimeoutExpired:
                        pass  # An uninterruptible kernel wait cannot be fixed by waiting here.
            for pipe in (self.process.stdin, self.process.stdout):
                pipe.close()


def driver_worker():
    try:
        from tools import capture_runtime as capture
    except ModuleNotFoundError:
        import capture_runtime as capture
    errors, slow = [], []

    def read(path):
        start = time.monotonic()
        print(
            json.dumps({"kind": "read_start", "path": str(path), "time": time.time()}),
            flush=True,
        )
        try:
            with Path(path).open() as stream:
                return stream.read(8192).strip()
        except OSError as error:
            errors.append(
                {"path": str(path), "errno": error.errno, "error": str(error)}
            )
            return None
        finally:
            elapsed = time.monotonic() - start
            if elapsed >= 0.05:
                slow.append({"path": str(path), "seconds": elapsed})

    capture.read_text = read
    for line in sys.stdin:
        request = json.loads(line)
        errors.clear()
        slow.clear()
        started = time.time()
        bdfs = request.get("bdfs")
        gpus, pcie = capture.gpu_snapshot(bdfs=bdfs), capture.pcie_snapshot(bdfs=bdfs)
        for gpu in gpus:
            device = Path("/sys/bus/pci/devices") / gpu["bdf"]
            for key in ("mem_busy_percent", "pp_dpm_sclk", "pp_dpm_mclk"):
                gpu[key] = read(device / key)
            total, used = gpu.get("mem_info_vram_total"), gpu.get("mem_info_vram_used")
            if total is not None and used is not None:
                gpu.update(total=int(total), free=int(total) - int(used))
        print(
            json.dumps(
                {
                    "kind": "sample",
                    "started_at": started,
                    "finished_at": time.time(),
                    "gpus": gpus,
                    "pcie": pcie,
                    "read_errors": errors,
                    "driver_faults": [
                        error
                        for error in errors
                        if error["errno"] in (errno.EIO, errno.ENODEV, errno.ETIMEDOUT)
                    ],
                    "slow_reads": slow,
                }
            ),
            flush=True,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--driver-worker", action="store_true")
    args = parser.parse_args()
    if args.driver_worker:
        driver_worker()
