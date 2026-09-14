# SPDX-License-Identifier: Apache-2.0
"""Public reference measurements; private qualification archives stay private.

The public bundle proves its memory arithmetic from numeric samples. The
publisher attests to the bounded workload; first start must qualify locally.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import re
import tempfile
from pathlib import Path

try:
    from tools.expert_budget import expert_memory
    from tools.plan_experts import digest
except ModuleNotFoundError:
    from expert_budget import expert_memory
    from plan_experts import digest

SCHEMA = "r9v.public-memory-seed.v2"
CHECKS = {
    "text",
    "tool",
    "vision",
    "vision-wide",
    "vision-tall",
    "context",
    "idle_resume",
}


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def integer(value, minimum=0):
    require(
        type(value) is int and value >= minimum, "invalid numeric reference measurement"
    )
    return value


def read_record(base, row):
    require(
        isinstance(row, dict) and set(row) == {"path", "sha256", "bytes"},
        "invalid public evidence record",
    )
    relative = Path(row["path"])
    require(
        not relative.is_absolute()
        and len(relative.parts) == 1
        and relative.name not in (".", ".."),
        "unsafe public evidence path",
    )
    path = base / relative
    require(
        not path.is_symlink() and path.resolve().is_relative_to(base.resolve()),
        "public evidence escapes package",
    )
    require(
        path.stat().st_size == integer(row["bytes"], 1) <= 16 * 2**20,
        "public evidence size mismatch",
    )
    raw = path.read_bytes()
    require(sha(raw) == row["sha256"], "public evidence hash mismatch")
    return raw, json.loads(raw)


def catalog_content(catalog):
    """Canonical measured content, excluding only provenance file locations."""
    value = copy.deepcopy(catalog)
    inputs = value.get("ranking", {}).get("inputs")
    require(isinstance(inputs, list) and len(inputs) == 3, "three ranking inputs required")
    for row in inputs:
        require(isinstance(row, dict) and set(row) == {"path", "sha256"},
                "unexpected ranking input fields")
        require(isinstance(row["path"], str) and row["path"], "missing input location")
        require(isinstance(row["sha256"], str) and re.fullmatch("[0-9a-f]{64}", row["sha256"]),
                "invalid ranking input identity")
        del row["path"]
    return value


def manifest_content(manifest):
    value = copy.deepcopy(manifest)
    require(isinstance(value.get("r9v_derivation"), dict), "derived manifest required for projection")
    value["r9v_derivation"].pop("source_sha256", None)
    return value


def project_public_catalog(catalog_raw, manifest_raw):
    """Relocate provenance labels; retain commitments to the executed originals.

    This does not requalify an edited placement. Only three input locations and
    the manifest's catalog link change; all measured/placement content is kept.
    """
    catalog, manifest = json.loads(catalog_raw), json.loads(manifest_raw)
    if not any(marker in catalog_raw for marker in (b"/var/home/", b"/home/dylan/", b"/var/mnt/")):
        return catalog_raw, manifest_raw, None
    content = catalog_content(catalog)
    require(manifest.get("r9v_derivation", {}).get("source_sha256") == sha(catalog_raw),
            "original manifest does not bind original catalog")
    original_manifest_content = manifest_content(manifest)
    published = copy.deepcopy(catalog)
    for role, row in zip(("source", "train", "holdout"), published["ranking"]["inputs"]):
        row["path"] = "inputs/" + role + "-" + row["sha256"] + ".json"
    require(catalog_content(published) == content, "projection changed measured catalog")
    def encode(value):
        return (json.dumps(value, indent=2) + "\n").encode()
    new_catalog = encode(published)
    public_manifest = copy.deepcopy(manifest)
    public_manifest["r9v_derivation"]["source_sha256"] = sha(new_catalog)
    require(manifest_content(public_manifest) == original_manifest_content,
            "projection changed executed placement")
    new_manifest = encode(public_manifest)
    projection = {
        "schema": "r9v.public-catalog-projection.v1",
        "original_catalog_sha256": sha(catalog_raw),
        "original_manifest_file_sha256": sha(manifest_raw),
        "executed_manifest_sha256": digest(manifest),
        "public_catalog_sha256": sha(new_catalog),
        "public_manifest_sha256": digest(public_manifest),
        "catalog_content_sha256": digest(content),
        "manifest_content_sha256": digest(original_manifest_content),
        "scope": "Only ranking input locations and the derived catalog link were relocated. Original execution evidence remains privately archived; local qualification is still required.",
    }
    return new_catalog, new_manifest, projection


def verify_projection(projection, catalog_raw, manifest, proof):
    require(projection.get("schema") == "r9v.public-catalog-projection.v1",
            "unsupported public catalog projection")
    for key in ("original_catalog_sha256", "original_manifest_file_sha256", "executed_manifest_sha256",
                "public_catalog_sha256", "public_manifest_sha256", "catalog_content_sha256", "manifest_content_sha256"):
        require(isinstance(projection.get(key), str) and re.fullmatch("[0-9a-f]{64}", projection[key]),
                "missing projection identity: " + key)
    catalog = json.loads(catalog_raw)
    require(sha(catalog_raw) == projection["public_catalog_sha256"]
            and digest(manifest) == projection["public_manifest_sha256"], "public projection identity differs")
    require(digest(catalog_content(catalog)) == projection["catalog_content_sha256"]
            and digest(manifest_content(manifest)) == projection["manifest_content_sha256"],
            "public projection content differs")
    for role, row in zip(("source", "train", "holdout"), catalog["ranking"]["inputs"]):
        require(row["path"] == "inputs/" + role + "-" + row["sha256"] + ".json",
                "unexpected public input location")
    require(proof.get("executed_manifest_sha256") == projection["executed_manifest_sha256"],
            "executed placement commitment differs")


def verify_public(seed, base):
    """Recompute the exported envelope without raw logs or host-local paths."""
    require(seed.get("schema") == SCHEMA, "unsupported public reference schema")
    records = seed.get("public_evidence", {})
    require(
        set(records) == {"catalog", "manifest", "runtime", "measurements", "workload"},
        "missing public reference evidence",
    )
    payload = {key: read_record(Path(base), row) for key, row in records.items()}
    require(
        sum(row["bytes"] for row in records.values()) <= 32 * 2**20,
        "public bundle exceeds bound",
    )
    cal = seed["calibration"]
    require(
        cal.get("schema") == "r9v.memory-calibration.v1"
        and cal.get("workload_passed") is True,
        "reference calibration did not pass",
    )
    require(
        seed.get("placement_qualification", {}).get("passed") is True,
        "reference placement did not pass",
    )
    contract = cal["contract"]
    settings = contract["settings"]
    catalog_raw, catalog = payload["catalog"]
    _, manifest = payload["manifest"]
    runtime_raw, runtime = payload["runtime"]
    require(
        sha(catalog_raw) == contract["source_sha256"],
        "reference catalog identity differs",
    )
    require(
        sha(runtime_raw) == settings["R9V_RUNTIME_DESCRIPTOR"],
        "reference runtime identity differs",
    )
    require(settings["R9V_MAX_MODEL_LEN"] == "131072", "reference must retain128K")
    if manifest != catalog:
        require(
            manifest.get("r9v_derivation", {}).get("source_sha256") == sha(catalog_raw),
            "reference manifest derivation differs",
        )
        for rank in ("0", "1"):
            rows = manifest["ranks"][rank]["hot_experts_by_layer"]
            priorities = catalog["ranks"][rank]["hot_experts_by_layer"]
            require(
                len(rows) == len(priorities) == 48
                and all(
                    row == ordered[: len(row)] for row, ordered in zip(rows, priorities)
                ),
                "reference placement is not a catalog prefix",
            )
    memory = expert_memory(
        manifest,
        int(settings["R9V_TIERED_EXPERT_CACHE_SLOTS"]),
        {int(r) for r in settings["R9V_TIERED_EXPERT_CACHE_RANKS"].split(",") if r},
        settings.get("R9V_TIERED_EXPERT_CACHE_ASYNC", "0") == "1",
        runtime=runtime,
    )
    measurement = payload["measurements"][1]
    samples = measurement["samples"]
    require(
        isinstance(samples, list) and 2 <= len(samples) <= 100000,
        "missing bounded memory samples",
    )
    require(
        samples[0]["phase"] == "startup"
        and any(r["phase"] == "workload" for r in samples),
        "missing startup/workload samples",
    )
    devices = contract["devices"]
    require(len(devices) == 2, "two reference devices required")
    for row in samples:
        require(
            set(row) == {"phase", "host_available_bytes", "free_bytes"},
            "unexpected telemetry fields",
        )
        require(
            row["phase"] in {"startup", "warmup", "measure", "workload", "profile"},
            "unexpected telemetry phase",
        )
        integer(row["host_available_bytes"])
        require(len(row["free_bytes"]) == 2, "incomplete telemetry row")
        for rank, free in enumerate(row["free_bytes"]):
            require(
                integer(free) <= integer(devices[rank]["total_bytes"], 1),
                "invalid free memory",
            )
    calculated = []
    for rank in range(2):
        total = devices[rank]["total_bytes"]
        external = total - samples[0]["free_bytes"][rank]
        peak_used = total - min(row["free_bytes"][rank] for row in samples)
        packed = (
            memory[rank]["static_packed_bytes"] + memory[rank]["cache_packed_bytes"]
        )
        calculated.append(
            {
                "non_expert_peak_bytes": max(0, peak_used - external - packed),
                "external_allowance_bytes": external + 256 * 2**20,
                "transient_margin_bytes": 512 * 2**20,
            }
        )
    require(cal["ranks"] == calculated, "reference GPU envelope arithmetic differs")
    require(
        cal["reference_cold_bytes"]
        == sum(r["cold_pinned_packed_bytes"] for r in memory),
        "reference host expert bytes differ",
    )
    host_peak = max(
        0,
        samples[0]["host_available_bytes"]
        - min(r["host_available_bytes"] for r in samples),
    )
    require(
        cal["startup_required_available_bytes"] == host_peak
        and cal["host_reserve_bytes"] == 4 * 2**30,
        "reference host envelope arithmetic differs",
    )
    proof = payload["workload"][1]
    require(
        proof["checks"] == sorted(CHECKS)
        and proof["passed"] is True
        and proof["runtime_failures"] == [],
        "reference workload did not pass all checks",
    )
    require(
        proof["context_limit"] == 131072
        and 130912 <= integer(proof["prompt_tokens"]) <= 130944,
        "reference context envelope absent",
    )
    require(
        proof["image_id"] == settings["R9V_IMAGE"]
        and proof["manifest_sha256"] == digest(manifest),
        "reference image/placement mismatch",
    )
    require(
        proof["clean_stop"] is True and proof["reclaimed"] is True,
        "reference cleanup did not pass",
    )
    require(
        len(proof["workers"]) == 2 and [r["rank"] for r in proof["workers"]] == [0, 1],
        "reference worker ranks differ",
    )
    for worker in proof["workers"]:
        require(
            worker["probes"]
            == {"pinned_uva": True, "copy": True, "tp_all_reduce": True}
            and worker["num_ooms"] == 0,
            "reference transport/no-OOM proof absent",
        )
    require(
        len(proof["requested_free_bytes"]) == len(proof["minimum_free_bytes"]) == 2,
        "reference headroom incomplete",
    )
    require(
        all(
            integer(free) >= integer(requested, 1)
            for free, requested in zip(
                proof["minimum_free_bytes"], proof["requested_free_bytes"]
            )
        ),
        "reference requested headroom not met",
    )
    provenance = seed["provenance"]
    require(("catalog_projection" in provenance) == ("executed_manifest_sha256" in proof),
            "public projection marker missing")
    if "catalog_projection" in provenance:
        verify_projection(provenance["catalog_projection"], catalog_raw, manifest, proof)
    for key in (
        "private_seed_sha256",
        "private_payload_index_sha256",
        "original_calibration_sha256",
    ):
        require(
            isinstance(provenance.get(key), str)
            and re.fullmatch("[0-9a-f]{64}", provenance[key]),
            "missing private provenance commitment",
        )
    return len(records)


def export_public(private_path, output):
    try:
        from tools.memory_seed import PORTABLE, verify_bundle
    except ModuleNotFoundError:
        from memory_seed import PORTABLE, verify_bundle
    private_path, output = Path(private_path), Path(output)
    original_raw = private_path.read_bytes()
    seed = json.loads(original_raw)
    require(
        seed.get("schema") == "r9v.memory-seed.v1",
        "export requires original private seed",
    )
    base = private_path.parent
    verify_bundle(seed, base)

    def private(row):
        return (base / row["path"]).read_bytes()

    def decoded(row):
        return json.loads(private(row))

    origins = seed["provenance"]["original_evidence_identities"]

    def named(suffix):
        found = [row for name, row in origins.items() if name.endswith(suffix)]
        require(len(found) == 1, "private source record not unique: " + suffix)
        return decoded(found[0])

    original_cal_record = seed["provenance"]["original_calibration"]
    original_cal = decoded(original_cal_record)
    cal = copy.deepcopy(seed["calibration"])
    require(
        {k: v for k, v in cal.items() if k != "evidence"}
        == {k: v for k, v in original_cal.items() if k != "evidence"},
        "packaged calibration differs from original",
    )
    pq = seed["placement_qualification"]
    plan = decoded(pq["plan"])
    manifest = decoded(pq["manifest"])
    require(
        plan["contract"] == cal["contract"]
        and plan["manifest_sha256"] == digest(manifest),
        "executed private plan binding differs",
    )
    runtime_sha = cal["contract"]["settings"]["R9V_RUNTIME_DESCRIPTOR"]
    runtimes = {
        row["sha256"]: row for row in origins.values() if row["sha256"] == runtime_sha
    }
    require(runtime_sha in runtimes, "original runtime source absent")
    memory_rows = [
        json.loads(line)
        for line in private(
            next(
                row for row in cal["evidence"] if row["path"].endswith("-memory.jsonl")
            )
        ).splitlines()
    ]
    arm = (
        named("/baseline/result.json")["arm"]
        if any(k.endswith("/baseline/result.json") for k in origins)
        and "arm" in named("/baseline/result.json")
        else "baseline"
    )
    phases = {
        arm + "-" + phase: phase
        for phase in ("startup", "warmup", "measure", "workload", "profile")
    }
    selected = [row for row in memory_rows if row.get("phase") in phases]
    first = next(i for i, row in enumerate(selected) if len(row.get("gpus", [])) == 2)
    require(
        not any(row.get("gpus") for row in selected[:first]),
        "incomplete startup telemetry",
    )
    expected_bdfs = [r["bdf"] for r in cal["contract"]["devices"]]
    samples = []
    for row in selected[first:]:
        require(
            [g["bdf"] for g in row["gpus"]] == expected_bdfs
            and [g["total"] for g in row["gpus"]]
            == [r["total_bytes"] for r in cal["contract"]["devices"]],
            "private telemetry identity differs",
        )
        samples.append(
            {
                "phase": phases[row["phase"]],
                "host_available_bytes": row["host_available_bytes"],
                "free_bytes": [g["free"] for g in row["gpus"]],
            }
        )
    workload = decoded(pq["workload"])
    context = decoded(pq["context"])
    inspected = named("/baseline/inspect.json")[0]
    workers = sorted(named("/baseline/workers.json"), key=lambda w: w["rank"])
    reclaim = named("/baseline/cleanup-reclamation.json")
    baseline = named("/baseline/gpu-baseline.json")
    require(
        reclaim["baseline"] == baseline
        and reclaim.get("recovered") is True
        and not reclaim.get("error"),
        "private reclamation invalid",
    )
    require(
        workload.get("headroom_passed") is True
        and not workload.get("telemetry_errors")
        and not workload.get("error"),
        "private headroom workload incomplete",
    )
    state = inspected["State"]
    proof = {
        "passed": workload["passed"],
        "checks": sorted(workload["checks"]),
        "runtime_failures": workload["runtime_failures"],
        "context_limit": workload["context_limit"],
        "prompt_tokens": context["response"]["usage"]["prompt_tokens"],
        "minimum_free_bytes": workload["minimum_physical_free_bytes"],
        "requested_free_bytes": [r["target_free_bytes"] for r in plan["ranks"]],
        "image_id": inspected["Image"],
        "manifest_sha256": plan["manifest_sha256"],
        "clean_stop": state["Running"] is False
        and state["OOMKilled"] is False
        and type(state["ExitCode"]) is int
        and state["ExitCode"] == 0,
        "reclaimed": reclaim["recovered"],
        "workers": [
            {
                "rank": w["rank"],
                "probes": w["probes"],
                "num_ooms": w["allocator"]["num_ooms"],
            }
            for w in workers
        ],
    }
    cal.pop("evidence", None)
    cal["contract"]["settings"] = {
        k: v for k, v in cal["contract"]["settings"].items() if k not in PORTABLE
    }
    exported = {
        "schema": SCHEMA,
        "calibration": cal,
        "placement_qualification": {"passed": True},
        "provenance": {
            "private_seed_sha256": sha(original_raw),
            "private_payload_index_sha256": digest(
                sorted((r["sha256"], r["bytes"]) for r in origins.values())
            ),
            "original_calibration_sha256": original_cal_record["sha256"],
            "scope": "Publisher-attested reference workload; public numeric samples independently reproduce memory arithmetic. Raw private qualification remains archived.",
        },
        "limitations": "Reference estimate only. Exact image/model/runtime identity must match. Every local placement requires full workload qualification.",
    }

    def encoded(value):
        return (json.dumps(value, indent=2) + "\n").encode()

    catalog_raw, manifest_raw, projection = project_public_catalog(
        private(pq["source_catalog"]), private(pq["manifest"])
    )
    if projection is not None:
        # The verified private calibration is unchanged. This public reference
        # names a metadata-only projection, explicitly linked to original bytes.
        cal["contract"]["source_sha256"] = projection["public_catalog_sha256"]
        proof["executed_manifest_sha256"] = proof["manifest_sha256"]
        proof["manifest_sha256"] = projection["public_manifest_sha256"]
        exported["provenance"]["catalog_projection"] = projection
    files = {
        "catalog": catalog_raw,
        "manifest": manifest_raw,
        "runtime": private(runtimes[runtime_sha]),
        "measurements": encoded({"samples": samples}),
        "workload": encoded(proof),
    }
    require(not output.exists(), "public output already exists")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=".public-reference-", dir=output.parent
    ) as temporary:
        stage = Path(temporary)
        exported["public_evidence"] = {}
        for name, raw in files.items():
            require(
                b"/var/home/" not in raw
                and b"/home/dylan/" not in raw
                and b"/var/mnt/" not in raw,
                "private path in public payload: " + name,
            )
            filename = name + ".json"
            (stage / filename).write_bytes(raw)
            exported["public_evidence"][name] = {
                "path": filename,
                "bytes": len(raw),
                "sha256": sha(raw),
            }
        (stage / "memory-seed.json").write_bytes(encoded(exported))
        verify_public(exported, stage)
        os.rename(stage, output)
    return exported


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("private_seed", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    export_public(args.private_seed, args.output)
