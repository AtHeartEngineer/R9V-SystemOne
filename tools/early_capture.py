#!/usr/bin/env python3
"""Bounded, non-destructive evidence collection before runtime cleanup."""

import argparse
import json
import os
import re
import subprocess
import time
from pathlib import Path


def bounded(command, output, seconds=3):
    with output.open("xb") as stream:
        proc = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT)
        try:
            code = proc.wait(timeout=seconds)
            return {"command": command[0], "exit_code": code}
        except subprocess.TimeoutExpired:
            proc.kill()
            try:
                proc.wait(timeout=0.2)
            except subprocess.TimeoutExpired:
                pass
            return {"command": command[0], "timeout": True}


# This executes inside the container namespace. pidfd + start time protects against
# PID reuse; only workers advertising an installed faulthandler are signalled.
PYTHON_STACKS = r"""
import json, os, pathlib, signal
for path in pathlib.Path('/tmp').glob('r9v-stage-*.json'):
    try:
        value = json.loads(path.read_text())
        pid = value['pid']
        if type(pid) is not int or pid <= 1 or not value.get('signal_ready'):
            continue
        fd = os.pidfd_open(pid)
        try:
            actual = pathlib.Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()[19]
            if actual == value.get('start_ticks'):
                signal.pidfd_send_signal(fd, signal.SIGUSR2)
                print(json.dumps({'pid': pid, 'stage': value['stage'], 'stack_requested': True}), flush=True)
        finally:
            os.close(fd)
    except (OSError, ValueError, KeyError) as error:
        print(str(error), flush=True)
"""


def capture(container, output, reason, stream=False):
    if not re.fullmatch(r"r9v-[a-zA-Z0-9_.-]{1,100}", container):
        raise ValueError("early capture only accepts named r9v containers")
    output.mkdir(parents=True, mode=0o700, exist_ok=False)
    records = []
    commands = [
        ("sysrq.log", ["sudo", "-n", "/usr/local/bin/r9v-sysrq-dump"]),
        (
            "python-stacks.log",
            ["docker", "exec", container, "python3", "-c", PYTHON_STACKS],
        ),
        (
            "kernel.log",
            ["journalctl", "-k", "-n", "300", "--no-pager", "-o", "short-monotonic"],
        ),
        ("processes.log", ["ps", "-eLo", "pid,tid,ppid,stat,wchan:32,comm"]),
        (
            "container.json",
            ["docker", "inspect", "--format", "{{json .State}}", container],
        ),
        ("server.log", ["docker", "logs", "--tail", "150", "--timestamps", container]),
        (
            "tasks.log",
            ["docker", "top", container, "-eo", "pid,ppid,stat,wchan:32,comm"],
        ),
    ]
    if stream:
        print(
            json.dumps(
                {"event": "capture_start", "container": container, "reason": reason}
            ),
            flush=True,
        )
    for name, command in commands:
        try:
            records.append({"file": name, **bounded(command, output / name)})
        except OSError as error:
            records.append({"file": name, "error": str(error)})
        if stream:
            raw = (
                (output / name).read_bytes()[: 1024 * 1024]
                if (output / name).exists()
                else b""
            )
            print(
                json.dumps(
                    {
                        "file": name,
                        "result": records[-1],
                        "text": raw.decode(errors="replace"),
                    }
                ),
                flush=True,
            )
    with (output / "result.json").open("x") as result_stream:
        json.dump(
            {
                "time": time.time(),
                "reason": reason,
                "container": container,
                "commands": records,
            },
            result_stream,
        )
        result_stream.flush()
        os.fsync(result_stream.fileno())
    return records


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--container", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reason", required=True)
    parser.add_argument("--stream", action="store_true")
    args = parser.parse_args()
    capture(args.container, args.output, args.reason, stream=args.stream)
