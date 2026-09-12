import hashlib
import json

import pytest

from tools import calibrate_memory as calibration


def fixture(tmp_path, monkeypatch):
    source = tmp_path / "source.json"
    source.write_text(
        json.dumps(
            {
                "version": 1,
                "num_layers": 48,
                "num_experts": 512,
                "top_k": 10,
                "ranks": {
                    str(r): {
                        "hot_count": 1,
                        "hot_experts_by_layer": [[0] for _ in range(48)],
                    }
                    for r in range(2)
                },
            }
        )
    )
    path = tmp_path / "a"
    (path / "workload").mkdir(parents=True)
    config = {
        "R9V_MAX_MODEL_LEN": "131072",
        "R9V_EXPECTED_GPU_BDFS": "a,b",
        "R9V_EXPERT_MANIFEST_PATH": str(source),
        "R9V_TIERED_EXPERT_CACHE_SLOTS": "0",
        "R9V_TIERED_EXPERT_CACHE_RANKS": "",
    }
    (path / "config.json").write_text(json.dumps(config))
    (path / "result.json").write_text('{"passed":true}')
    (path / "workload/result.json").write_text('{"passed":true,"context_limit":131072}')
    (path / "workers.json").write_text(
        json.dumps(
            [
                {
                    "rank": r,
                    "bdf": bdf,
                    "probes": {"pinned_uva": True, "copy": True, "tp_all_reduce": True},
                }
                for r, bdf in enumerate(["a", "b"])
            ]
        )
    )
    samples = [
        {
            "phase": "a-" + phase,
            "host_available_bytes": host * 2**30,
            "gpus": [
                {"bdf": bdf, "total": 32 * 2**30, "free": free * 2**30}
                for bdf in ["a", "b"]
            ],
        }
        for phase, host, free in [("startup", 100, 31), ("workload", 20, 20)]
    ]
    (tmp_path / "memory.jsonl").write_text("\n".join(map(json.dumps, samples)))
    monkeypatch.setattr(calibration, "live_contract", lambda *a: {"fixture": True})
    return source, path, samples


def test_calibration_accounts_for_observed_peak_and_external_usage(
    tmp_path, monkeypatch
):
    source, _, _ = fixture(tmp_path, monkeypatch)
    result = calibration.calibrate(tmp_path, "a", source)
    assert result["startup_required_available_bytes"] == 80 * 2**30
    assert result["ranks"][0]["external_allowance_bytes"] == 2**30 + 256 * 2**20
    assert len(result["evidence"]) == 4


def test_calibration_subtracts_measured_prefix_payload_not_full_catalog(
    tmp_path, monkeypatch
):
    source, path, _ = fixture(tmp_path, monkeypatch)
    placement = json.loads(source.read_text())
    catalog = json.loads(source.read_text())
    for rank in catalog["ranks"].values():
        rank.update(hot_count=2, hot_experts_by_layer=[[0, 1] for _ in range(48)])
    source.write_text(json.dumps(catalog))
    placement["r9v_derivation"] = {
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest()
    }
    actual = tmp_path / "measured.json"
    actual.write_text(json.dumps(placement))
    config = json.loads((path / "config.json").read_text())
    config["R9V_EXPERT_MANIFEST_PATH"] = str(actual)
    (path / "config.json").write_text(json.dumps(config))
    result = calibration.calibrate(tmp_path, "a", source)
    payload = calibration.expert_memory(placement, 0, set(), False)[0][
        "static_packed_bytes"
    ]
    assert result["ranks"][0]["non_expert_peak_bytes"] == 11 * 2**30 - payload
    placement["ranks"]["0"]["hot_experts_by_layer"][0] = [1]
    actual.write_text(json.dumps(placement))
    with pytest.raises(ValueError, match="priority prefix"):
        calibration.calibrate(tmp_path, "a", source)


def capacity_fixture(tmp_path, monkeypatch):
    source, path, samples = fixture(tmp_path, monkeypatch)
    config = json.loads((path / "config.json").read_text())
    config["R9V_MIN_FREE_VRAM_GIB_BY_RANK"] = "3,3"
    config["R9V_IMAGE"] = "test"
    (path / "config.json").write_text(json.dumps(config))
    (path / "result.json").write_text(
        json.dumps({"passed": False, "error": f"Command exited 1: see {path / 'workload.log'}"})
    )
    (path / "workload/result.json").write_text(json.dumps({
        "passed": False, "context_limit": 131072,
        "checks": ["text", "tool", "vision", "vision-wide", "vision-tall", "context", "idle_resume"],
        "runtime_failures": [], "semantic_observations": ["text arithmetic mismatch"], "headroom_passed": False,
        "minimum_physical_free_bytes": [1, 2 * 2**30],
    }))
    (path / "workload/context.json").write_text(json.dumps({"response": {"usage": {"prompt_tokens": 130930}}}))
    (path / "inspect.json").write_text(json.dumps({"Image": "test", "State": {"Running": False, "OOMKilled": False, "ExitCode": 0}}))
    baseline = [{"bdf": bdf, "free_bytes": 31 * 2**30} for bdf in ["a", "b"]]
    (path / "gpu-baseline.json").write_text(json.dumps(baseline))
    (path / "cleanup-reclamation.json").write_text(json.dumps({"recovered": True, "baseline": baseline}))
    (tmp_path / "protocol.json").write_text(json.dumps({"arms": [{"name": "a"}], "check_headroom": True}))
    workers = json.loads((path / "workers.json").read_text())
    for worker in workers:
        worker.update(schema="r9v.worker.v1", phase="serving", allocator={"num_ooms": 0})
    (path / "workers.json").write_text(json.dumps(workers))
    runtime = tmp_path / "runtime.json"
    runtime.write_text("{}")
    config["R9V_RUNTIME_DESCRIPTOR"] = str(runtime)
    config["R9V_MODEL_PACKAGE_SHA256"] = hashlib.sha256(b"{}").hexdigest()
    (path / "config.json").write_text(json.dumps(config))
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    frozen = {}
    for name, data in [("fixture.json", source.read_bytes()), ("runtime.json", b"{}")]:
        (source_dir / name).write_bytes(data)
        frozen[name] = hashlib.sha256(data).hexdigest()
    (tmp_path / "source-inputs.json").write_text(json.dumps(frozen))
    return source, path, samples


def test_capacity_baseline_accepts_only_failed_headroom_arm(tmp_path, monkeypatch):
    source, _path, _samples = capacity_fixture(tmp_path, monkeypatch)
    result = calibration.calibrate(tmp_path, "a", source, capacity_baseline=True)
    assert result["schema"] == "r9v.memory-capacity-baseline.v1"
    assert result["workload_passed"] is False
    assert result["capacity_baseline_passed"] is True
    assert result["qualification_status"] == "failed_headroom_only"
    assert result["measured_hot_counts"] == [1, 1]


@pytest.mark.parametrize("failure", ["workload", "identity", "telemetry", "source"])
def test_calibration_refuses_incomplete_or_mismatched_evidence(
    tmp_path, monkeypatch, failure
):
    source, path, samples = fixture(tmp_path, monkeypatch)
    if failure == "workload":
        (path / "workload/result.json").write_text(
            '{"passed":true,"context_limit":4096}'
        )
    elif failure == "identity":
        (path / "workers.json").write_text("[]")
    elif failure == "telemetry":
        samples[1]["gpus"][1]["bdf"] = "a"
        (tmp_path / "memory.jsonl").write_text("\n".join(map(json.dumps, samples)))
    else:
        other = tmp_path / "different.json"
        other.write_text(source.read_text() + " ")
        source = other
    with pytest.raises(ValueError):
        calibration.calibrate(tmp_path, "a", source)


@pytest.mark.parametrize("file,key,value", [
    ("workload/result.json", "headroom_passed", True),
    ("workload/result.json", "runtime_failures", None),
    ("workload/result.json", "telemetry_errors", ["gap"]),
    ("workload/result.json", "error", "API failed"),
    ("workload/result.json", "checks", ["text"]),
    ("workload/result.json", "minimum_physical_free_bytes", [-1, 0]),
    ("workload/result.json", "minimum_physical_free_bytes", [True, 0]),
    ("workload/result.json", "minimum_physical_free_bytes", [4 * 2**30] * 2),
    ("result.json", "cleanup_errors", ["stop failed"]),
    ("result.json", "error", "Command exited 1: see other-workload.log"),
    ("cleanup-reclamation.json", "recovered", False),
    ("inspect.json", "Image", "wrong"),
])
def test_capacity_rejects_failures_beyond_headroom(tmp_path, monkeypatch, file, key, value):
    source, path, _ = capacity_fixture(tmp_path, monkeypatch)
    target = path / file
    data = json.loads(target.read_text())
    data[key] = value
    target.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        calibration.calibrate(tmp_path, "a", source, capacity_baseline=True)


@pytest.mark.parametrize("corruption", ["frozen", "runtime", "context", "workers", "late-gap", "baseline"])
def test_capacity_validates_evidence_content(tmp_path, monkeypatch, corruption):
    source, path, samples = capacity_fixture(tmp_path, monkeypatch)
    if corruption == "frozen":
        (tmp_path / "source/fixture.json").write_text("changed")
    elif corruption == "runtime":
        (tmp_path / "runtime.json").write_text("{ }")
    elif corruption == "context":
        (path / "workload/context.json").write_text('{"response":{"usage":{"prompt_tokens":100}}}')
    elif corruption == "workers":
        workers = json.loads((path / "workers.json").read_text())
        workers[0]["allocator"]["num_ooms"] = 1
        (path / "workers.json").write_text(json.dumps(workers))
    elif corruption == "late-gap":
        samples.append({"phase": "a-workload", "gpus": [], "host_available_bytes": 1})
        (tmp_path / "memory.jsonl").write_text("\n".join(map(json.dumps, samples)))
    else:
        (path / "gpu-baseline.json").write_text("[]")
    with pytest.raises(ValueError):
        calibration.calibrate(tmp_path, "a", source, capacity_baseline=True)
