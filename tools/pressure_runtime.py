#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Isolate VRAM pressure with fixed experts and a temporary idle allocation actor.

Only for an owned qualification container. The actor runs inside that container,
so its memory and process disappear when the qualification supervisor stops it.
"""

import argparse
import json
import math
import statistics
import subprocess
import sys
import time
from pathlib import Path


def reserves(values):
    if (
        not isinstance(values, list)
        or len(values) != 2
        or any(
            type(v) not in (int, float) or not math.isfinite(v) or not 0 <= v <= 4
            for v in values
        )
    ):
        raise ValueError("pressure probe needs two finite reservations of 0..4 GiB")
    return [int(v * 2**30) for v in values]


def actor(directory):
    import torch

    directory.mkdir(exist_ok=True)
    seq = None
    buffers = []
    deadline = time.monotonic() + 2400
    while time.monotonic() < deadline:
        request_path = directory / "request.json"
        if request_path.exists():
            request = json.loads(request_path.read_text())
            if request["sequence"] != seq:
                seq = request["sequence"]
                result = {"sequence": seq, "passed": False}
                try:
                    amounts = reserves(request["gib"])
                    buffers.clear()
                    for rank in range(2):
                        with torch.cuda.device(rank):
                            torch.cuda.synchronize()
                            torch.cuda.empty_cache()
                    for rank, amount in enumerate(amounts):
                        with torch.cuda.device(rank):
                            free, _ = torch.cuda.mem_get_info()
                            if free < amount + 512 * 2**20:
                                raise ValueError(
                                    f"rank {rank}: pressure reservation would leave less than 512 MiB HIP free"
                                )
                            buffers.append(
                                torch.empty(
                                    max(1, amount),
                                    dtype=torch.uint8,
                                    device=f"cuda:{rank}",
                                ).fill_(1)
                            )
                            torch.cuda.synchronize()
                    result.update(
                        passed=True,
                        requested_bytes=amounts,
                        hip_free_bytes=[
                            torch.cuda.mem_get_info(r)[0] for r in range(2)
                        ],
                    )
                except Exception as error:
                    result["error"] = str(error)
                temporary = directory / "status.tmp"
                temporary.write_text(json.dumps(result))
                temporary.replace(directory / "status.json")
        time.sleep(0.2)


def measure(container, url, model, prompt, output, trials=7):
    output.mkdir(mode=0o700, parents=True, exist_ok=False)
    control = "/tmp/r9v-pressure-qualification"

    def command(args, **kwargs):
        return subprocess.run(
            args, check=True, capture_output=True, text=True, timeout=30, **kwargs
        )

    # A named marker prevents accidentally attaching to an unrelated prior actor.
    command(["docker", "exec", container, "mkdir", control])
    command(
        [
            "docker",
            "cp",
            str(Path(__file__).resolve()),
            container + ":" + control + "/actor.py",
        ]
    )
    command(
        [
            "docker",
            "exec",
            "-d",
            container,
            "python3",
            control + "/actor.py",
            "--actor",
            "--directory",
            control,
        ]
    )
    results = []
    for sequence, gib in enumerate(([0, 0], [1.3, 2.0], [0, 0])):
        request = output / f"pressure-{sequence}.json"
        request.write_text(json.dumps({"sequence": sequence, "gib": gib}))
        command(
            ["docker", "cp", str(request), container + ":" + control + "/request.tmp"]
        )
        command(
            [
                "docker",
                "exec",
                container,
                "mv",
                control + "/request.tmp",
                control + "/request.json",
            ]
        )
        deadline = time.monotonic() + 30
        status = {}
        while time.monotonic() < deadline:
            result = subprocess.run(
                ["docker", "exec", container, "cat", control + "/status.json"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                status = json.loads(result.stdout)
                if status.get("sequence") == sequence:
                    break
            time.sleep(0.5)
        if status.get("sequence") != sequence or not status.get("passed"):
            raise ValueError(f"pressure actor failed admission: {status}")
        samples = []
        for trial in range(trials + 2):
            destination = output / f"stage-{sequence}-request-{trial}.json"
            with destination.open("x") as stream:
                subprocess.run(
                    [
                        sys.executable,
                        str(Path(__file__).with_name("benchmark_openai.py")),
                        "--url",
                        url.rstrip("/") + "/v1",
                        "--model",
                        model,
                        "--prompt-file",
                        str(prompt),
                        "--max-tokens",
                        "256",
                        "--disable-thinking",
                        "--timeout",
                        "120",
                    ],
                    stdout=stream,
                    check=True,
                    timeout=130,
                )
            value = json.loads(destination.read_text())
            if value.get("finish_reason") not in ("stop", "length") or not value.get(
                "tg_tokens_per_second"
            ):
                raise ValueError("incomplete pressure benchmark response")
            if trial >= 2:
                samples.append(value["tg_tokens_per_second"])
        results.append(
            {
                "reservation": status,
                "tg_samples": samples,
                "tg_median": statistics.median(samples),
            }
        )
        (output / f"stage-{sequence}.json").write_text(
            json.dumps(results[-1], indent=2)
        )
    (output / "result.json").write_text(
        json.dumps(
            {
                "passed": True,
                "stages": results,
                "limits": "Fixed placement and same idle actor for zero-reservation / pressure / zero-reservation; not a production allocation policy.",
            },
            indent=2,
        )
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--actor", action="store_true")
    parser.add_argument("--directory", type=Path)
    parser.add_argument("--container")
    parser.add_argument("--url")
    parser.add_argument("--model")
    parser.add_argument("--prompt", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.actor:
        actor(args.directory)
    else:
        measure(args.container, args.url, args.model, args.prompt, args.output)


if __name__ == "__main__":
    main()
