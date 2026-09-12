#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Package a verified reference envelope; every local placement needs validation."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import tempfile
from pathlib import Path

try:
    from tools.calibrate_memory import calibrate
    from tools.expert_budget import headroom_bytes
    from tools.plan_experts import digest, runtime_contract
except ModuleNotFoundError:
    from calibrate_memory import calibrate
    from expert_budget import headroom_bytes
    from plan_experts import digest, runtime_contract

CHECKS = {"text", "tool", "vision", "vision-wide", "vision-tall", "context", "idle_resume"}


def read(path):
    return json.loads(path.read_text())


def file_sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def build_seed(calibration, session, source, manifest, plan, output, *, arm="baseline"):
    require(Path(arm).name == arm and arm not in (".", ".."), "invalid arm")
    cal = read(calibration)
    require(isinstance(cal, dict) and cal.get("schema") == "r9v.memory-calibration.v1"
            and cal.get("workload_passed") is True, "successful normal calibration required")
    contract = cal.get("contract", {})
    path = session / arm
    config = read(path / "config.json")
    workload = read(path / "workload/result.json")
    result = read(path / "result.json")
    results = read(session / "results.json")
    require(isinstance(results, list) and sum(r.get("arm") == arm and r.get("passed") is True
            for r in results if isinstance(r, dict)) == 1, "session arm did not pass")
    require(result.get("passed") is True and not result.get("cleanup_errors"), "arm cleanup failed")
    require(workload.get("passed") is True and set(workload.get("checks", [])) == CHECKS
            and workload.get("runtime_failures") == [] and not workload.get("telemetry_errors")
            and not workload.get("error"), "complete workload evidence required")
    require(config.get("R9V_MAX_MODEL_LEN") == "131072" and workload.get("context_limit") == 131072,
            "full 128K context required")
    tokens = read(path / "workload/context.json").get("response", {}).get("usage", {}).get("prompt_tokens")
    require(type(tokens) is int and 131072 - 160 <= tokens <= 131072 - 128,
            "actual context envelope was not reached")
    targets = headroom_bytes(config["R9V_MIN_FREE_VRAM_GIB_BY_RANK"], 2)
    observed = workload.get("minimum_physical_free_bytes")
    require(workload.get("headroom_passed") is True and isinstance(observed, list)
            and len(observed) == 2 and all(type(v) is int and v >= t for v, t in zip(observed, targets)),
            "measured headroom does not meet request")
    inspected = read(path / "inspect.json")
    require(isinstance(inspected, list) and len(inspected) == 1, "one container inspect required")
    info = inspected[0]
    state = info.get("State", {})
    require(info.get("Image") == config.get("R9V_IMAGE") and state.get("Running") is False
            and state.get("OOMKilled") is False and type(state.get("ExitCode")) is int
            and state["ExitCode"] == 0, "configured container did not stop cleanly")
    require(info.get("Name", "").lstrip("/") == result.get("container"), "container identity mismatch")
    workers = read(path / "workers.json")
    require(isinstance(workers, list) and len(workers) == 2
            and all(isinstance(w, dict) and type(w.get("rank")) is int for w in workers), "invalid workers")
    workers = sorted(workers, key=lambda w: w["rank"])
    require([w["rank"] for w in workers] == [0, 1]
            and [w.get("bdf") for w in workers] == config["R9V_EXPECTED_GPU_BDFS"].split(",")
            and all(w.get("schema") == "r9v.worker.v1" and w.get("phase") == "serving"
                    and w.get("probes") == {"pinned_uva": True, "copy": True, "tp_all_reduce": True}
                    and w.get("allocator", {}).get("num_ooms") == 0 for w in workers),
            "serving worker identity/transport/no-OOM evidence missing")
    reclaim = read(path / "cleanup-reclamation.json")
    require(reclaim.get("recovered") is True and not reclaim.get("error")
            and reclaim.get("baseline") == read(path / "gpu-baseline.json"), "reclamation evidence mismatch")
    frozen = read(session / "source-inputs.json")
    require(isinstance(frozen, dict) and bool(frozen), "frozen source index missing")
    for relative, expected in frozen.items():
        rel = Path(relative)
        require(not rel.is_absolute() and ".." not in rel.parts, "unsafe frozen path")
        f = session / "source" / rel
        require(f.resolve().is_relative_to((session / "source").resolve())
                and file_sha(f) == expected, "frozen source hash mismatch: " + relative)
    actual = Path(config["R9V_EXPERT_MANIFEST_PATH"])
    require(actual.resolve().is_relative_to(session.resolve()) and file_sha(actual) == file_sha(manifest),
            "supplied manifest was not the frozen executed manifest")
    require(actual.name == file_sha(actual) + ".manifest.json", "frozen manifest filename/hash mismatch")
    required_hashes = {file_sha(Path(config["R9V_RUNTIME_DESCRIPTOR"])),
                       config["R9V_MODEL_PACKAGE_SHA256"]}
    require(required_hashes.issubset(set(frozen.values())), "frozen model/runtime/manifest identity missing")
    require(file_sha(plan) == file_sha(Path(config["R9V_PLACEMENT_PLAN"])), "supplied plan was not executed")
    source_sha = file_sha(source)
    require(source_sha == contract.get("source_sha256"), "source catalog differs from calibration")
    plan_data = read(plan)
    require(plan_data.get("manifest_sha256") == digest(read(manifest))
            and plan_data.get("contract") == contract, "plan manifest/contract mismatch")
    evidence = cal.get("evidence")
    require(isinstance(evidence, list) and bool(evidence), "empty calibration evidence")
    for item in evidence:
        require(isinstance(item, dict) and isinstance(item.get("path"), str)
                and isinstance(item.get("sha256"), str)
                and file_sha(Path(item["path"])) == item["sha256"], "calibration evidence hash mismatch")

    # Recompute allocation arithmetic from original measurements, without probing
    # the build host. The current config must still reproduce the bound contract.
    def offline_contract(env, source_hash):
        current = runtime_contract(env, source_hash, contract["devices"], contract["driver"])
        require(current == contract, "executed configuration differs from calibration")
        return current

    computed = calibrate(session, arm, source, contract_provider=offline_contract)
    require(computed == cal, "calibration differs from recomputed measurements")
    require(not output.exists(), "output already exists")

    # Freeze every supplied session byte, including raw context response and the
    # complete source snapshot. No output is committed until all copies verify.
    files = {p for p in session.rglob("*") if p.is_file()}
    files.update([calibration, source, manifest, plan])
    files.update(Path(e["path"]) for e in evidence)
    payloads = {}
    for f in sorted(files):
        raw = f.read_bytes()
        require(len(raw) <= 64 * 2**20, "oversized evidence file")
        payloads[f.resolve()] = raw
    require(sum(map(len, payloads.values())) <= 128 * 2**20, "oversized evidence package")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".memory-seed-", dir=output.parent) as temporary:
        stage = Path(temporary)
        (stage / "evidence").mkdir()
        records = {}
        for f, raw in payloads.items():
            sha = hashlib.sha256(raw).hexdigest()
            relative = "evidence/" + sha + "-" + f.name
            (stage / relative).write_bytes(raw)
            records[str(f)] = {"path": relative, "sha256": sha, "bytes": len(raw)}
        def record(f):
            return records[str(f.resolve())]
        seed = {"schema": "r9v.memory-seed.v1", "calibration": copy.deepcopy(cal),
                "placement_qualification": {"passed": True, "manifest": record(manifest),
                    "plan": record(plan), "source_catalog": record(source),
                    "session_result": record(session / "results.json"),
                    "workload": record(path / "workload/result.json"),
                    "context": record(path / "workload/context.json")},
                "provenance": {"original_calibration": record(calibration),
                    "original_evidence_identities": records,
                    "frozen_source": {rel: record(session / "source" / rel) for rel in frozen}},
                "limitations": "Reference estimate from one bounded 128K workload. Local model/image/runtime identity must match; every new placement requires local workload validation."}
        seed["calibration"]["evidence"] = [record(Path(e["path"])) for e in evidence]
        (stage / "memory-seed.json").write_text(json.dumps(seed, indent=2) + "\n")
        os.rename(stage, output)
    return seed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("calibration", "session", "source", "manifest", "plan", "output"):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--arm", default="baseline")
    args = parser.parse_args()
    try:
        build_seed(args.calibration, args.session, args.source, args.manifest, args.plan, args.output, arm=args.arm)
    except (OSError, ValueError, TypeError, KeyError) as error:
        parser.exit(1, f"Memory seed not generated: {error}\n")


if __name__ == "__main__":
    main()
