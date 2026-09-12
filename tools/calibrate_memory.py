#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Derive a conservative local memory envelope from a completed qualification arm."""

import argparse
import hashlib
import json
from pathlib import Path

try:
    from tools.expert_budget import expert_memory, headroom_bytes
    from tools.plan_experts import live_contract, read_runtime
except ModuleNotFoundError:
    from expert_budget import expert_memory, headroom_bytes
    from plan_experts import live_contract, read_runtime


def calibrate(session, arm, source, *, capacity_baseline=False, contract_provider=None):
    path = session / arm
    config = json.loads((path / "config.json").read_text())
    arm_result = json.loads((path / "result.json").read_text())
    workload = json.loads((path / "workload/result.json").read_text())
    expected_context = int(config["R9V_MAX_MODEL_LEN"])
    required_checks = {"text", "tool", "vision", "vision-wide", "vision-tall", "context", "idle_resume"}
    if capacity_baseline:
        if arm_result.get("passed") is not False or workload.get("passed") is not False:
            raise ValueError("capacity baseline requires a failed headroom-only arm")
        if workload.get("headroom_passed") is not False or workload.get("telemetry_errors") or workload.get("error"):
            raise ValueError("capacity baseline requires headroom-only failure with valid telemetry")
        if workload.get("runtime_failures") != []:
            raise ValueError("capacity baseline has runtime failures")
        if set(workload.get("checks", [])) != required_checks:
            raise ValueError("capacity baseline is missing a required workload check")
        if workload.get("context_limit") != expected_context:
            raise ValueError("maximum configured workload was not exercised")
        context_response = path / "workload" / "context.json"
        if not context_response.is_file():
            raise ValueError("capacity baseline is missing context response evidence")
        usage = json.loads(context_response.read_text()).get("response", {}).get("usage", {})
        prompt_tokens = usage.get("prompt_tokens")
        if not isinstance(prompt_tokens, int) or not expected_context - 160 <= prompt_tokens <= expected_context - 128:
            raise ValueError("maximum configured context envelope was not reached")
        if not isinstance(workload.get("minimum_physical_free_bytes"), list):
            raise ValueError("capacity baseline is missing observed VRAM headroom")
        requested = headroom_bytes(config["R9V_MIN_FREE_VRAM_GIB_BY_RANK"], 2)
        observed = workload["minimum_physical_free_bytes"]
        if len(observed) != 2 or any(type(free) is not int or free < 0 for free in observed) or not any(free < target for free, target in zip(observed, requested)):
            raise ValueError("capacity baseline did not prove a requested headroom shortfall")
        if arm_result.get("error") != f"Command exited 1: see {path / 'workload.log'}":
            raise ValueError("capacity baseline arm did not fail at the workload command")
        reclamation = path / "cleanup-reclamation.json"
        inspect = path / "inspect.json"
        if not reclamation.is_file() or not inspect.is_file():
            raise ValueError("capacity baseline is missing cleanup or container identity evidence")
        reclaim = json.loads(reclamation.read_text())
        if reclaim.get("recovered") is not True or reclaim.get("error"):
            raise ValueError("capacity baseline cleanup reclamation did not pass")
        if arm_result.get("cleanup_errors"):
            raise ValueError("capacity baseline has cleanup errors")
    elif not arm_result.get("passed"):
        raise ValueError("qualification arm did not pass")
    if not capacity_baseline and (not workload.get("passed") or workload["context_limit"] != expected_context):
        raise ValueError("maximum configured workload was not qualified")
    workers = json.loads((path / "workers.json").read_text())
    expected = config["R9V_EXPECTED_GPU_BDFS"].split(",")
    workers.sort(key=lambda row: row["rank"])
    if (
        len(workers) != 2
        or [row["bdf"] for row in workers] != expected
        or not all(
            row.get("probes")
            == {"pinned_uva": True, "copy": True, "tp_all_reduce": True}
            for row in workers
        )
    ):
        raise ValueError("worker identity/transport qualification is missing")
    if capacity_baseline and ([row.get("rank") for row in workers] != [0, 1] or any(
            row.get("allocator", {}).get("num_ooms") != 0
            or row.get("schema") != "r9v.worker.v1"
            or row.get("phase") != "serving" for row in workers
    )):
        raise ValueError("capacity baseline needs complete serving worker and no-OOM evidence")
    raw = source.read_bytes()
    configured_source = Path(
        config.get("R9V_EXPERT_MANIFEST_PATH")
        or str(
            Path(config["R9V_MODEL_DIR"])
            / config.get(
                "R9V_MANIFEST_REL",
                "manifests/hot-manifest-q4-vision-128k-multiprompt-r1-lru16-neutral.json",
            )
        )
    )
    actual_raw = configured_source.read_bytes()
    actual, catalog = json.loads(actual_raw), json.loads(raw)
    if actual_raw != raw:
        if (
            actual.get("r9v_derivation", {}).get("source_sha256")
            != hashlib.sha256(raw).hexdigest()
        ):
            raise ValueError("calibration source does not match the measured placement")
        for rank in ("0", "1"):
            rows = actual["ranks"][rank]["hot_experts_by_layer"]
            priorities = catalog["ranks"][rank]["hot_experts_by_layer"]
            if (
                len(rows) != 48
                or len(priorities) != 48
                or any(
                    row != priority[: len(row)]
                    for row, priority in zip(rows, priorities)
                )
            ):
                raise ValueError(
                    "measured placement is not a priority prefix of the calibration source"
                )
    if capacity_baseline:
        inspect_data = json.loads((path / "inspect.json").read_text())
        inspect_rows = inspect_data if isinstance(inspect_data, list) else [inspect_data]
        if len(inspect_rows) != 1 or inspect_rows[0].get("Image") != config["R9V_IMAGE"]:
            raise ValueError("container inspect identity does not match configured image")
        state = inspect_rows[0].get("State", {})
        if state.get("Running") is not False or state.get("OOMKilled") is not False or state.get("ExitCode") != 0:
            raise ValueError("capacity baseline container did not stop cleanly")
        baseline_path = path / "gpu-baseline.json"
        baseline_records = json.loads(baseline_path.read_text())
        if baseline_records != reclaim.get("baseline") or [r.get("bdf") for r in baseline_records] != expected:
            raise ValueError("GPU baseline does not match cleanup reclamation evidence")
        protocol = json.loads((session / "protocol.json").read_text())
        if len(protocol.get("arms", [])) != 1 or protocol["arms"][0].get("name") != arm or protocol.get("check_headroom", True) is not True:
            raise ValueError("capacity baseline requires one arm with the headroom guard enabled")
        source_index = session / "source-inputs.json"
        if not source_index.is_file():
            raise ValueError("capacity baseline is missing frozen source index")
        frozen = json.loads(source_index.read_text())
        for name, expected_hash in frozen.items():
            if Path(name).is_absolute() or ".." in Path(name).parts:
                raise ValueError("invalid frozen source path")
            item = session / "source" / name
            if not item.is_file() or hashlib.sha256(item.read_bytes()).hexdigest() != expected_hash:
                raise ValueError(f"frozen source hash mismatch: {name}")
        required_hashes = {
            hashlib.sha256(actual_raw).hexdigest(),
            hashlib.sha256(Path(config["R9V_RUNTIME_DESCRIPTOR"]).read_bytes()).hexdigest(),
            config["R9V_MODEL_PACKAGE_SHA256"],
        }
        if not required_hashes.issubset(set(frozen.values())):
            raise ValueError("measured manifest/runtime/package identity is absent from frozen source")
    memory = expert_memory(
        actual,
        int(config["R9V_TIERED_EXPERT_CACHE_SLOTS"]),
        {int(r) for r in config["R9V_TIERED_EXPERT_CACHE_RANKS"].split(",") if r},
        config.get("R9V_TIERED_EXPERT_CACHE_ASYNC", "0") == "1",
        runtime=read_runtime(config),
    )
    samples = [
        json.loads(line) for line in (session / "memory.jsonl").read_text().splitlines()
    ]
    # First arm must begin from idle cards. Later arms get their own startup sample.
    phases = {
        arm + "-" + phase
        for phase in ("startup", "warmup", "measure", "workload", "profile")
    }
    selected = [row for row in samples if row.get("phase") in phases]
    if not selected or not any(row["phase"] == arm + "-workload" for row in selected):
        raise ValueError("missing workload memory telemetry")
    complete_index = next((i for i, row in enumerate(selected) if len(row.get("gpus", [])) == 2), None)
    if complete_index is None:
        raise ValueError("missing complete GPU memory telemetry")
    if any(row.get("gpus") for row in selected[:complete_index]):
        raise ValueError("GPU telemetry disappeared before the selected startup baseline")
    selected = selected[complete_index:]
    baseline = selected[0]
    if baseline["phase"] != arm + "-startup":
        raise ValueError("missing idle startup baseline")
    if capacity_baseline and any(abs(gpu["free"] - row["free_bytes"]) > 256 * 2**20 for gpu, row in zip(baseline["gpus"], baseline_records)):
        raise ValueError("first complete startup sample is not the idle GPU baseline")
    for sample in selected:
        if [gpu["bdf"] for gpu in sample["gpus"]] != expected or any(
            gpu["total"] != baseline["gpus"][rank]["total"]
            or not 0 <= gpu["free"] <= gpu["total"]
            for rank, gpu in enumerate(sample["gpus"])
        ):
            raise ValueError("memory telemetry device identity/capacity changed")
    ranks = []
    for rank in range(2):
        total = baseline["gpus"][rank]["total"]
        external = total - baseline["gpus"][rank]["free"]
        peak_used = total - min(row["gpus"][rank]["free"] for row in selected)
        payload = (
            memory[rank]["static_packed_bytes"] + memory[rank]["cache_packed_bytes"]
        )
        ranks.append(
            {
                "non_expert_peak_bytes": max(0, peak_used - external - payload),
                "external_allowance_bytes": external + 256 * 2**20,
                "transient_margin_bytes": 512 * 2**20,
            }
        )
    inputs = [
        session / "memory.jsonl",
        path / "workers.json",
        path / "workload/result.json",
        source,
    ]
    if configured_source.resolve() != source.resolve():
        inputs.append(configured_source)
    output = {
        "schema": "r9v.memory-capacity-baseline.v1" if capacity_baseline else "r9v.memory-calibration.v1",
        "contract": (contract_provider or live_contract)(config, hashlib.sha256(raw).hexdigest()),
        "ranks": ranks,
        "reference_cold_bytes": sum(row["cold_pinned_packed_bytes"] for row in memory),
        "startup_required_available_bytes": max(
            0,
            baseline["host_available_bytes"]
            - min(row["host_available_bytes"] for row in selected),
        ),
        "host_reserve_bytes": 4 * 2**30,
        "workload_passed": not capacity_baseline,
        "evidence": [
            {
                "path": str(p.resolve()),
                "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
            }
            for p in inputs
        ],
        "limitations": "Local bounded workload envelope; polling plus 512 MiB transient margin. External allowance includes 256 MiB variability. Revalidate every derived placement.",
    }
    if capacity_baseline:
        # The workload sampler can observe a lower point between session samples.
        for rank in range(2):
            total = baseline["gpus"][rank]["total"]
            if observed[rank] > total:
                raise ValueError("observed VRAM exceeds device capacity")
            payload = memory[rank]["static_packed_bytes"] + memory[rank]["cache_packed_bytes"]
            external = total - baseline["gpus"][rank]["free"]
            output["ranks"][rank]["non_expert_peak_bytes"] = max(
                output["ranks"][rank]["non_expert_peak_bytes"],
                total - observed[rank] - external - payload,
            )
        output.update(
            capacity_baseline_passed=True,
            qualification_status="failed_headroom_only",
            measured_hot_counts=[actual["ranks"][str(rank)].get("hot_count") for rank in range(2)],
            requested_headroom_bytes=requested,
            observed_minimum_physical_free_bytes=[int(value) for value in observed],
            evidence=list(output["evidence"]),
        )
        extra = [path / "config.json", path / "result.json", path / "workload/result.json", path / "workload/context.json",
                 path / "workers.json", path / "inspect.json", path / "cleanup-reclamation.json",
                 path / "gpu-baseline.json", path / "workload/context.json",
                 session / "protocol.json", session / "source-inputs.json"]
        for item in extra:
            if item.is_file() and not any(entry["path"] == str(item.resolve()) for entry in output["evidence"]):
                output["evidence"].append({"path": str(item.resolve()), "sha256": hashlib.sha256(item.read_bytes()).hexdigest()})
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("session", "source", "output"):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--arm", required=True)
    parser.add_argument("--capacity-baseline", action="store_true")
    args = parser.parse_args()
    try:
        result = calibrate(args.session, args.arm, args.source, capacity_baseline=args.capacity_baseline)
        with args.output.open("x") as stream:
            json.dump(result, stream, indent=2)
            stream.write("\n")
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(f"Calibration not generated: {error}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
