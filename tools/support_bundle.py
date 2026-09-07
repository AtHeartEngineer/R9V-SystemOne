#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Save crash evidence before a container is removed, including after a reboot."""

from __future__ import annotations

import argparse
import os
from datetime import datetime, timezone
from pathlib import Path

try:
    from tools import capture_runtime as capture
except ModuleNotFoundError:
    import capture_runtime as capture


def save_json(path: Path, value: dict) -> None:
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "wb") as output:
        capture.write_record(output, value, 8 * 1024 * 1024)


def collect(output: Path, container: str, port: int) -> dict:
    output.mkdir(mode=0o700, parents=True, exist_ok=False)
    # No full environment, model data, or generated completions are collected.
    # Raw server/kernel logs can still contain user data and need review.
    probes = {
        "container": [
            "docker",
            "inspect",
            "--format",
            capture.INSPECT_FORMAT,
            container,
        ],
        "server-log": ["docker", "logs", "--timestamps", "--tail", "2000", container],
        "kernel-current": [
            "journalctl",
            "-k",
            "-b",
            "0",
            "--no-pager",
            "-n",
            "1000",
            "-o",
            "short-iso-precise",
        ],
        "kernel-previous": [
            "journalctl",
            "-k",
            "-b",
            "-1",
            "--no-pager",
            "-n",
            "1000",
            "-o",
            "short-iso-precise",
        ],
        "gpu-inventory": ["amd-smi", "list"],
        "kernel-version": ["uname", "-a"],
        "revision": [
            "git",
            "-C",
            str(Path(__file__).resolve().parents[1]),
            "rev-parse",
            "HEAD",
        ],
        "runtime-versions": [
            "docker",
            "exec",
            container,
            "python3",
            "-c",
            "import json, torch, importlib.metadata as m; "
            "print(json.dumps({'torch':torch.__version__,'hip':torch.version.hip,"
            "'vllm':m.version('vllm')}))",
        ],
    }
    results = {}
    for name, command in probes.items():
        result = capture.command_output(command, timeout=10)
        save_json(output / (name + ".json"), result)
        results[name] = (
            ("partial" if result.get("text") else "unavailable")
            if "error" in result
            else ("collected" if result.get("text", "").strip() else "empty")
        )
    state = capture.command_json(probes["container"])
    save_json(
        output / "snapshot.json",
        {
            "host": capture.host_snapshot(),
            "gpus": capture.gpu_snapshot(),
            "pcie": capture.pcie_snapshot(),
            "metrics": capture.fetch_metrics(port),
            "cgroup": capture.cgroup_snapshot(int(state.get("pid", 0))),
        },
    )
    manifest = {
        "schema": "r9v.support.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "container": container,
        "runtime_state": state,
        "probes": results,
        "limits": "Each command is limited to 256 KiB and 10 seconds; unavailable evidence is explicit.",
        "review": "Review raw server and kernel logs for private data before sharing. Nothing is uploaded.",
    }
    save_json(output / "manifest.json", manifest)
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--container",
        default=os.environ.get("R9V_CONTAINER_NAME", "r9v-qwen38-flash-next"),
    )
    parser.add_argument(
        "--port", type=int, default=int(os.environ.get("R9V_HOST_PORT", "8004"))
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="new private directory; never overwritten",
    )
    args = parser.parse_args(argv)
    if (
        not args.container
        or args.container.startswith("-")
        or not 1 <= args.port <= 65535
    ):
        parser.error("invalid container name or port")
    try:
        result = collect(args.output, args.container, args.port)
    except (OSError, RuntimeError) as error:
        print(f"Support collection failed: {error}")
        return 1
    print(f"Support evidence: {args.output}")
    print(result["review"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
