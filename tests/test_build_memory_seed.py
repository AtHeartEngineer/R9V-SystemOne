# SPDX-License-Identifier: Apache-2.0
import hashlib
import json

import pytest

from tools.build_memory_seed import CHECKS, build_seed
from tools.calibrate_memory import calibrate
from tools.plan_experts import digest, runtime_contract


def fixture(tmp_path):
    session = tmp_path / "external-session"
    arm = session / "baseline"
    (arm / "workload").mkdir(parents=True)
    (session / "source").mkdir()

    def write(path, value):
        path.write_text(json.dumps(value, indent=2))
        return path

    source = write(tmp_path / "catalog.json", {
        "version": 1, "num_layers": 48, "num_experts": 512, "top_k": 10,
        "ranks": {str(r): {"hot_count": 1, "hot_experts_by_layer": [[0]] * 48} for r in range(2)}})
    def sha(p):
        return hashlib.sha256(p.read_bytes()).hexdigest()
    manifest = session / (sha(source) + ".manifest.json")
    manifest.write_bytes(source.read_bytes())
    runtime = write(session / "source/runtime.json", {})
    package = write(session / "source/package.json", {})
    write(session / "source-inputs.json", {p.name: sha(p) for p in (runtime, package)})
    plan = tmp_path / "executed-plan.json"
    config = {"R9V_IMAGE": "sha256:" + "a" * 64, "R9V_MAX_MODEL_LEN": "131072",
              "R9V_EXPECTED_GPU_BDFS": "a,b", "R9V_TIERED_EXPERT_CACHE_SLOTS": "0",
              "R9V_TIERED_EXPERT_CACHE_RANKS": "", "R9V_MIN_FREE_VRAM_GIB_BY_RANK": "3,3",
              "R9V_EXPERT_MANIFEST_PATH": str(manifest), "R9V_PLACEMENT_PLAN": str(plan),
              "R9V_MODEL_PACKAGE_SHA256": sha(package), "R9V_RUNTIME_DESCRIPTOR": str(runtime)}
    write(arm / "config.json", config)
    write(arm / "result.json", {"passed": True, "container": "owned"})
    write(session / "results.json", [{"arm": "baseline", "passed": True}])
    write(arm / "workload/result.json", {"passed": True, "context_limit": 131072,
          "checks": sorted(CHECKS), "runtime_failures": [], "headroom_passed": True,
          "minimum_physical_free_bytes": [20 * 2**30] * 2})
    write(arm / "workload/context.json", {"response": {"usage": {"prompt_tokens": 130941}}})
    write(arm / "inspect.json", [{"Name": "/owned", "Image": config["R9V_IMAGE"],
          "State": {"Running": False, "OOMKilled": False, "ExitCode": 0}}])
    write(arm / "workers.json", [{"schema": "r9v.worker.v1", "phase": "serving", "rank": r,
          "bdf": bdf, "probes": {"pinned_uva": True, "copy": True, "tp_all_reduce": True},
          "allocator": {"num_ooms": 0}} for r, bdf in enumerate(("a", "b"))])
    baseline = [{"bdf": b, "free_bytes": 31 * 2**30} for b in ("a", "b")]
    write(arm / "gpu-baseline.json", baseline)
    write(arm / "cleanup-reclamation.json", {"recovered": True, "baseline": baseline, "error": None})
    samples = [{"phase": "baseline-" + phase, "host_available_bytes": host * 2**30,
                "gpus": [{"bdf": b, "total": 32 * 2**30, "free": free * 2**30} for b in ("a", "b")]}
               for phase, host, free in [("startup", 100, 31), ("workload", 20, 20)]]
    (session / "memory.jsonl").write_text("\n".join(map(json.dumps, samples)))
    devices = [{"bdf": b, "total_bytes": 32 * 2**30} for b in ("a", "b")]
    cal = calibrate(session, "baseline", source,
                    contract_provider=lambda env, sh: runtime_contract(env, sh, devices, {}))
    calibration = write(tmp_path / "calibration.json", cal)
    write(plan, {"contract": cal["contract"], "manifest_sha256": digest(json.loads(manifest.read_text()))})
    return calibration, session, source, manifest, plan


def test_packages_external_evidence_and_every_declared_file_is_verifiable(tmp_path):
    args = fixture(tmp_path)
    output = tmp_path / "repo/seed"
    seed = build_seed(*args, output)
    assert (output / "memory-seed.json").is_file()
    for row in seed["provenance"]["original_evidence_identities"].values():
        data = (output / row["path"]).read_bytes()
        assert len(data) == row["bytes"]
        assert hashlib.sha256(data).hexdigest() == row["sha256"]
    assert len(seed["provenance"]["frozen_source"]) == 2
    assert not any(row["path"].startswith("/") for row in seed["calibration"]["evidence"])
    assert seed["calibration"]["workload_passed"] is True
    with pytest.raises(ValueError, match="already exists"):
        build_seed(*args, output)


@pytest.mark.parametrize("change", ["headroom", "short_context", "worker", "image", "reclaim",
                                    "frozen", "calibration", "empty_evidence", "plan", "manifest"])
def test_refuses_corrupt_or_unrelated_evidence_before_writing(tmp_path, change):
    args = list(fixture(tmp_path))
    cal, session, _, _, plan = args
    path = {"headroom": session / "baseline/workload/result.json",
            "short_context": session / "baseline/workload/context.json",
            "worker": session / "baseline/workers.json", "image": session / "baseline/inspect.json",
            "reclaim": session / "baseline/cleanup-reclamation.json", "frozen": session / "source/runtime.json",
            "calibration": cal, "empty_evidence": cal, "plan": plan}.get(change)
    if change == "manifest":
        args[3] = tmp_path / "unrelated.json"
        args[3].write_text("{}")
    else:
        data = json.loads(path.read_text())
        if change == "headroom":
            data["minimum_physical_free_bytes"][0] = 1
        elif change == "short_context":
            data["response"]["usage"]["prompt_tokens"] = 4096
        elif change == "worker":
            data[0]["allocator"]["num_ooms"] = 1
        elif change == "image":
            data[0]["Image"] = "unrelated"
        elif change == "reclaim":
            data["recovered"] = False
        elif change == "frozen":
            data["changed"] = True
        elif change == "calibration":
            data["ranks"][0]["non_expert_peak_bytes"] = 0
        elif change == "empty_evidence":
            data["evidence"] = []
        elif change == "plan":
            data["manifest_sha256"] = "wrong"
        path.write_text(json.dumps(data))
    output = tmp_path / "not-created"
    with pytest.raises(ValueError):
        build_seed(*args, output)
    assert not output.exists()


def test_seed_evidence_is_verified_before_reference_localization(tmp_path):
    from tools.memory_seed import verify_bundle

    seed = build_seed(*fixture(tmp_path), tmp_path / "bundle")
    base = tmp_path / "bundle"
    assert verify_bundle(seed, base) > 10
    record = seed["placement_qualification"]["context"]
    path = base / record["path"]
    raw = path.read_bytes()
    path.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="changed"):
        verify_bundle(seed, base)
    path.write_bytes(raw)
    record["path"] = "../outside.json"
    with pytest.raises(ValueError, match="escapes"):
        verify_bundle(seed, base)
