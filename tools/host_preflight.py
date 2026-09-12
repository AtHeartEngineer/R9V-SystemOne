# SPDX-License-Identifier: Apache-2.0
"""Resource and evidence checks shared by the Qwen Flash Next doctor."""

from __future__ import annotations

import json
import os
import platform
import shutil
from pathlib import Path

try:
    from tools.expert_budget import headroom_bytes
except ModuleNotFoundError:
    from expert_budget import headroom_bytes


def check_resources(
    reporter, selected, repo_root: Path, sys_root: Path, proc_root: Path, runtime: bool
) -> None:
    reporter.note("host-kernel", platform.release())
    driver = sys_root / "module/amdgpu/version"
    if driver.exists():
        reporter.note("amdgpu-version", driver.read_text().strip())
    try:
        margins = headroom_bytes(
            os.environ.get("R9V_MIN_FREE_VRAM_GIB_BY_RANK", "3,3"), 2
        )
    except ValueError as error:
        reporter.fail(
            "vram-headroom-policy",
            str(error),
            "Set one non-negative GiB target per TP rank.",
        )
        margins = None
    # Sysfs values follow physical BDFs, not amd-smi display indices.
    for rank, gpu, *_ in selected:
        pci = sys_root / "bus/pci/devices" / gpu.bdf
        try:
            total = int((pci / "mem_info_vram_total").read_text())
            used = int((pci / "mem_info_vram_used").read_text())
            if total < 31 * 1024**3:
                reporter.fail(
                    "gpu-vram",
                    f"rank {rank} {gpu.bdf}: {total / 1024**3:.2f} GiB; dual-R9700 profile needs 32 GiB cards",
                    "Select the two R9700 devices and verify the BDF order.",
                )
            elif margins is not None and (
                margins[rank] > total or total - used < margins[rank]
            ):
                reporter.fail(
                    "gpu-headroom",
                    f"rank {rank} {gpu.bdf}: {(total - used) / 1024**3:.2f} GiB free, requested {margins[rank] / 1024**3:g} GiB",
                    "Reduce hot-expert residency or other GPU use. This check does not resize the manifest or promise future peak headroom.",
                )
            else:
                reporter.passed(
                    "gpu-vram",
                    f"rank {rank}: {total / 1024**3:.2f} GiB total, {(total - used) / 1024**3:.2f} GiB free now; snapshot only",
                )
        except (OSError, ValueError):
            reporter.warn(
                "gpu-vram",
                f"rank {rank}: VRAM capacity/use unavailable",
                "Inspect amd-smi memory telemetry; device presence alone cannot certify capacity.",
            )
        for hop in (pci.resolve(), *pci.resolve().parents):
            for kind in ("aer_dev_correctable", "aer_dev_nonfatal", "aer_dev_fatal"):
                try:
                    text = (hop / kind).read_text()
                except OSError:
                    continue
                if any(
                    parts[-1].isdigit() and int(parts[-1]) > 0
                    for line in text.splitlines()
                    if (parts := line.split())
                ):
                    reporter.warn(
                        "pcie-error-history",
                        f"{hop.name}: nonzero {kind} counters",
                        "Capture a soak and compare counter deltas with failure times. These cumulative counts alone do not prove a current fault.",
                    )
    cache = Path(
        os.environ.get("R9V_CACHE_DIR", str(repo_root / ".cache"))
    ).expanduser()
    while not cache.exists() and cache != cache.parent:
        cache = cache.parent
    try:
        free = shutil.disk_usage(cache).free
        if not os.access(cache, os.W_OK | os.X_OK):
            reporter.fail(
                "cache-storage",
                f"cache ancestor {cache} is not writable",
                "Set R9V_CACHE_DIR to writable storage.",
            )
        elif free < 1024**3:
            reporter.warn(
                "cache-storage",
                f"only {free / 1024**3:.2f} GiB free at {cache}",
                "Free cache space before runtime compilation; keep evidence on storage with headroom.",
            )
        else:
            reporter.passed(
                "cache-storage", f"{free / 1024**3:.1f} GiB free at {cache}"
            )
    except OSError as error:
        reporter.fail("cache-storage", str(error), "Check the cache filesystem.")
    if not (Path("/var/log/journal").is_dir()):
        reporter.warn(
            "persistent-kernel-logs",
            "persistent journal directory is absent",
            "Enable persistent journald storage on the host to recover GPU-reset/OOM evidence after reboot; run support before removing the container.",
        )
    for name in ("R9V_MIN_HOST_RAM_BYTES", "R9V_MIN_HOST_AVAILABLE_BYTES"):
        if os.environ.get(name, "0") == "0":
            reporter.warn(
                "memory-qualification",
                f"{name} is unset; no measured host RAM minimum is enforced",
                "Record peak host/cgroup memory during qualification and set an explicit minimum with startup headroom. The logical offload GB value is not a RAM requirement.",
            )


def check_container_limits(reporter, run, container: str) -> None:
    probe = run(["docker", "inspect", "--format", "{{json .HostConfig}}", container])
    try:
        config = json.loads(probe.stdout) if probe.returncode == 0 else None
        if not isinstance(config, dict):
            raise ValueError("container settings unavailable")
    except (ValueError, TypeError):
        reporter.warn(
            "runtime-limits",
            "cannot inspect container resource limits",
            "Check Docker access.",
        )
        return
    logs = config.get("LogConfig", {})
    if logs.get("Type") not in {"json-file", "local", "journald"}:
        reporter.fail(
            "runtime-log-retention",
            f"logging driver {logs.get('Type')!r} may not retain locally retrievable logs",
            "Recreate with the current launcher after saving existing evidence.",
        )
    elif logs.get("Type") in {"json-file", "local"} and not all(
        logs.get("Config", {}).get(key) for key in ("max-size", "max-file")
    ):
        reporter.warn(
            "runtime-log-retention",
            "log rotation bounds are not explicit",
            "Use the current launcher for bounded retained logs.",
        )
    else:
        reporter.passed("runtime-log-retention", f"local logging: {logs.get('Type')}")
    if config.get("AutoRemove"):
        reporter.fail(
            "runtime-auto-remove",
            "container is removed on exit, losing evidence",
            "Use the launcher without --rm.",
        )
    for setting in ("Memory", "MemorySwap", "PidsLimit"):
        value = config.get(setting)
        if value and value > 0:
            reporter.warn(
                "runtime-resource-cap",
                f"{setting}={value}",
                "Compare this cap with the captured cgroup peaks/events; sufficient host RAM does not override a container cap.",
            )
    memlock = next(
        (
            limit
            for limit in config.get("Ulimits") or []
            if limit.get("Name") == "memlock"
        ),
        {},
    )
    if memlock.get("Soft") != -1 or memlock.get("Hard") != -1:
        reporter.warn(
            "runtime-memlock",
            "memlock policy is inherited or limited",
            "Inspect the daemon memlock limits and serving-worker pinned-UVA probes. "
            "Rootless Docker inherits its daemon hard limit; forcing "
            "--ulimit memlock=-1:-1 can prevent startup.",
        )


def check_runtime_arguments(reporter, run, container: str) -> None:
    result = run(["docker", "inspect", "--format", "{{json .Config.Cmd}}", container])
    try:
        command = json.loads(result.stdout) if result.returncode == 0 else None
        if not isinstance(command, list) or not all(
            isinstance(item, str) for item in command
        ):
            raise ValueError("command unavailable")
    except (ValueError, TypeError):
        reporter.warn(
            "runtime-arguments",
            "cannot inspect the server launch arguments",
            "Verify the container was created by the selected profile.",
        )
        return
    expected = {
        "--max-model-len": "R9V_MAX_MODEL_LEN",
        "--max-num-seqs": "R9V_MAX_NUM_SEQS",
        "--max-num-batched-tokens": "R9V_MAX_NUM_BATCHED_TOKENS",
        "--kv-cache-memory-bytes": "R9V_KV_CACHE_MEMORY_BYTES",
        "--tensor-parallel-size": "R9V_TENSOR_PARALLEL_SIZE",
    }
    mismatches = {}
    for flag, key in expected.items():
        value = os.environ.get(key)
        if value is None:
            continue
        actual = [
            command[index + 1]
            for index, token in enumerate(command[:-1])
            if token == flag
        ]
        actual += [
            token.split("=", 1)[1] for token in command if token.startswith(flag + "=")
        ]
        if actual != [value]:
            mismatches[flag] = {"expected": value, "actual": actual}
    if mismatches:
        reporter.fail(
            "runtime-arguments",
            "live context, concurrency, prefill, KV, or TP arguments differ from this config",
            "Rerun doctor with the config used to launch, or recreate the container with the intended settings after saving evidence.",
            mismatches=mismatches,
        )
    else:
        reporter.note(
            "runtime-arguments",
            "configured context/concurrency/prefill/KV/TP flags match container launch arguments; this does not inspect worker-internal state",
        )
