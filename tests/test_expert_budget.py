# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import copy
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from tools import profile_doctor as doctor
from tools import trim_experts
from tools.expert_budget import (
    COST_PATH,
    expert_memory,
    headroom_bytes,
    validate_manifest,
)
from tools.host_preflight import check_resources, check_runtime_arguments


def manifest(counts=(329, 369)):
    return {
        "version": 1,
        "num_layers": 48,
        "num_experts": 512,
        "top_k": 10,
        "ranks": {
            str(rank): {
                "hot_count": count,
                "hot_experts_by_layer": [list(range(count)) for _ in range(48)],
            }
            for rank, count in enumerate(counts)
        },
    }


@pytest.mark.parametrize(
    "corruption",
    [
        "lying_count",
        "duplicate",
        "boolean",
        "negative",
        "missing_layer",
        "empty_layer",
        "wrong_rank",
        "wrong_bytes",
    ],
)
def test_manifest_rejects_bad_contents_even_with_small_declared_counts(corruption):
    data = manifest()
    entry = data["ranks"]["1"]
    if corruption == "lying_count":
        entry["hot_count"] = 1
    elif corruption == "duplicate":
        entry["hot_experts_by_layer"][0][1] = 0
    elif corruption == "boolean":
        entry["hot_experts_by_layer"][0][0] = True
    elif corruption == "negative":
        entry["hot_experts_by_layer"][0][0] = -1
    elif corruption == "missing_layer":
        entry["hot_experts_by_layer"].pop()
    elif corruption == "empty_layer":
        entry["hot_experts_by_layer"][0] = []
    elif corruption == "wrong_rank":
        data["ranks"]["2"] = data["ranks"].pop("1")
    elif corruption == "wrong_bytes":
        entry["hot_bytes"] = 1
    with pytest.raises(ValueError):
        expert_memory(data, 16, {1})


def test_pinned_catalog_tracks_exact_package_and_mixed_layer_sizes():
    catalog = json.loads(COST_PATH.read_text())
    root = Path(__file__).resolve().parents[1]
    package = json.loads(
        (
            root
            / "packages/models/qwen38-flash-next/ud-iq4-xs--mtp-blockfp8--mmproj-q8/package.json"
        ).read_text()
    )
    assert catalog["model_package"] == package["id"]
    assert catalog["target_artifacts"] == [
        {k: a[k] for k in ("path", "bytes", "sha256")}
        for a in package["artifacts"]
        if a["role"] == "target"
    ]
    costs = catalog["packed_bytes_per_expert_by_layer_per_rank"]
    assert (
        len(set(costs)) == 3
    )  # Mixed quantization; global expert count is insufficient.
    assert sum(costs) * 512 * 2 == 59_519_795_200
    budgets = expert_memory(manifest(), 16, {1})
    assert budgets[0]["static_packed_bytes"] == 19_123_059_200
    assert budgets[1]["static_packed_bytes"] == 21_448_051_200
    assert budgets[1]["cache_packed_bytes"] == 929_996_800


def test_byte_accounting_distinguishes_which_layer_has_more_experts():
    data = manifest((1, 1))
    for rank in data["ranks"].values():
        rank.pop("hot_count")
    costs = json.loads(COST_PATH.read_text())[
        "packed_bytes_per_expert_by_layer_per_rank"
    ]
    cheap, expensive = costs.index(min(costs)), costs.index(max(costs))
    data["ranks"]["0"]["hot_experts_by_layer"][cheap].append(2)
    data["ranks"]["1"]["hot_experts_by_layer"][expensive].append(2)
    result = expert_memory(data, 0, set())
    assert result[1]["static_packed_bytes"] > result[0]["static_packed_bytes"]


def test_async_cache_accounts_for_extra_physical_slot_without_growing_cold_owner():
    sync = expert_memory(manifest(), 16, {1})
    async_result = expert_memory(manifest(), 16, {1}, True)
    assert async_result[1]["cache_physical_slots"] == 17
    assert (
        async_result[1]["cache_packed_bytes"] - sync[1]["cache_packed_bytes"]
        == 58_124_800
    )
    assert (
        async_result[1]["cold_pinned_packed_bytes"]
        == sync[1]["cold_pinned_packed_bytes"]
    )
    assert async_result[0] == sync[0]


def test_trim_preserves_ids_and_source_and_drops_invalidated_route_statistics():
    source = manifest()
    source["holdout"] = {"fake_old_metric": 1}
    source["ranks"]["1"]["tail_route_mass"] = 0.001
    original = copy.deepcopy(source)
    result, note = trim_experts.trim(source, [329, 329], "test-sha")
    assert source == original
    assert "holdout" not in result and "tail_route_mass" not in result["ranks"]["1"]
    assert (
        result["ranks"]["1"]["hot_experts_by_layer"]
        == source["ranks"]["0"]["hot_experts_by_layer"]
    )
    assert note["deltas"][1]["gpu_packed_bytes_released"] == 2_324_992_000
    assert note["deltas"][1]["host_pinned_bytes_added"] == 2_324_992_000
    assert note["deltas"][0]["gpu_packed_bytes_released"] == 0
    validate_manifest(result)


@pytest.mark.parametrize("counts", [[0, 329], [330, 369], [329, 370], [329], [True, 1]])
def test_trim_cannot_invent_unranked_experts_or_emit_unsupported_empty_layers(counts):
    with pytest.raises(ValueError):
        trim_experts.trim(manifest(), counts, "test")


def test_trim_cli_refuses_to_overwrite_evidence(tmp_path):
    source = tmp_path / "source.json"
    source.write_text(json.dumps(manifest()))
    assert (
        trim_experts.main(
            [
                "--source",
                str(source),
                "--hot-counts",
                "329,329",
                "--output",
                str(source),
            ]
        )
        == 1
    )
    assert json.loads(source.read_text()) == manifest()


@pytest.mark.parametrize(
    "value", ["5", "5,NaN", "inf,1", "-1,3", "1,2,3", "true,4", "1e308,5"]
)
def test_headroom_rejects_invalid_values(value):
    with pytest.raises(ValueError):
        headroom_bytes(value, 2)


def test_both_ranks_enforce_user_headroom_without_display_assumption(
    monkeypatch, tmp_path
):
    monkeypatch.setenv("R9V_MIN_FREE_VRAM_GIB_BY_RANK", "5,5")
    selected = []
    for rank, used in enumerate((24, 28)):
        bdf = f"0000:0{rank + 1}:00.0"
        path = tmp_path / "bus/pci/devices" / bdf
        path.mkdir(parents=True)
        (path / "mem_info_vram_total").write_text(str(32 * 1024**3))
        (path / "mem_info_vram_used").write_text(str(used * 1024**3))
        selected.append((rank, SimpleNamespace(bdf=bdf), None, None, None))
    reporter = doctor.Reporter()
    check_resources(reporter, selected, tmp_path, tmp_path, tmp_path, True)
    failures = [c for c in reporter.checks if c.name == "gpu-headroom"]
    assert len(failures) == 1 and "rank 1" in failures[0].message
    assert failures[0].status == "FAIL"


def test_doctor_rejects_async_lru_and_invalid_kernel(monkeypatch):
    monkeypatch.setenv("R9V_TIERED_IQ_MOE_VARIANT", "does-not-exist")
    monkeypatch.setenv("R9V_TIERED_EXPERT_CACHE_SLOTS", "16")
    monkeypatch.setenv("R9V_TIERED_EXPERT_CACHE_RANKS", "1")
    monkeypatch.setenv("R9V_TIERED_EXPERT_CACHE_POLICY", "lru")
    monkeypatch.setenv("R9V_TIERED_EXPERT_CACHE_ASYNC", "1")
    monkeypatch.setenv("R9V_PLE_RESIDENCY_MODE", "ssd")
    reporter = doctor.Reporter()
    doctor._check_profile_policy(reporter, 2, [])
    result = next(c for c in reporter.checks if c.name == "decode-policy")
    assert result.status == "FAIL"
    assert "unknown MoE variant" in result.message and "asynchronous" in result.message


@pytest.mark.parametrize("value", ["nan,1", "-2,3", "1,inf"])
def test_nonfinite_or_negative_pcie_policy_cannot_silently_bypass_checks(value):
    with pytest.raises(ValueError):
        doctor._csv_float(value, 2, "test")


def test_runtime_contract_detects_context_override(monkeypatch):
    monkeypatch.setenv("R9V_MAX_MODEL_LEN", "131072")
    reporter = doctor.Reporter()
    check_runtime_arguments(
        reporter,
        lambda *_: subprocess.CompletedProcess(
            [], 0, json.dumps(["--max-model-len=262144"])
        ),
        "test",
    )
    (check,) = reporter.checks
    assert check.status == "FAIL"
    assert check.details["mismatches"]["--max-model-len"]["actual"] == ["262144"]


@pytest.mark.parametrize(
    "actual,status",
    [
        (["0000:03:00.0", "0000:13:00.0"], "PASS"),
        (["0000:13:00.0", "0000:03:00.0"], "FAIL"),
    ],
)
def test_container_hip_identity_must_match_host_bdf_order(monkeypatch, actual, status):
    monkeypatch.setattr(
        doctor,
        "_run",
        lambda *_a, **_k: subprocess.CompletedProcess([], 0, json.dumps(actual)),
    )
    selected = [
        (rank, SimpleNamespace(bdf=bdf))
        for rank, bdf in enumerate(("0000:03:00.0", "0000:13:00.0"))
    ]
    reporter = doctor.Reporter()
    doctor._check_runtime_identity(reporter, selected)
    assert reporter.checks[0].status == status


def test_count_ceiling_does_not_certify_vram_fit(monkeypatch, tmp_path):
    source = tmp_path / "placement.json"
    source.write_text(json.dumps(manifest()))
    monkeypatch.setenv("R9V_MODEL_DIR", str(tmp_path))
    monkeypatch.setenv("R9V_EXPERT_MANIFEST_PATH", str(source))
    monkeypatch.setenv("R9V_MIN_FREE_VRAM_GIB_BY_RANK", "5,5")
    monkeypatch.setenv("R9V_TIERED_EXPERT_CACHE_SLOTS", "16")
    monkeypatch.setenv("R9V_TIERED_EXPERT_CACHE_RANKS", "1")
    monkeypatch.setenv("R9V_TIERED_EXPERT_CACHE_ASYNC", "0")
    monkeypatch.setenv("R9V_MAX_EFFECTIVE_EXPERTS_PER_RANK", "329,385")
    monkeypatch.setenv("R9V_KV_CACHE_MEMORY_BYTES", "2285670400")
    bdf = "0000:03:00.0"
    pci = tmp_path / "bus/pci/devices" / bdf
    pci.mkdir(parents=True)
    (pci / "mem_info_vram_total").write_text(str(32 * 1024**3))
    (pci / "mem_info_vram_used").write_text(str(20 * 1024**3))
    reporter = doctor.Reporter()
    doctor._check_manifest_budget(
        reporter,
        2,
        selected=[(0, SimpleNamespace(bdf=bdf))],
        sys_root=tmp_path,
        proc_root=tmp_path,
    )
    assert any(
        c.name == "expert-budget" and c.status == "PASS" for c in reporter.checks
    )
    assert any(
        c.name == "vram-budget-floor" and c.status == "FAIL" for c in reporter.checks
    )
