import hashlib

import pytest

from tools.prepare_placement import _seed_source_contract


def _seed(source_hash):
    return {"calibration": {"contract": {"source_sha256": source_hash}}}


def test_direct_catalog_source_does_not_require_legacy_manifest(tmp_path):
    selected = tmp_path / "catalog.json"
    selected.write_text("catalog")
    selected_hash = hashlib.sha256(selected.read_bytes()).hexdigest()
    contract = {"source_sha256": selected_hash, "settings": {"catalog": "selected"}}
    result = _seed_source_contract(_seed(selected_hash), contract, selected_hash,
                                   tmp_path / "missing-legacy.json")
    assert result is contract


def test_legacy_seed_can_localize_against_selected_catalog_rebase(tmp_path):
    legacy = tmp_path / "legacy.json"
    legacy.write_text("legacy")
    legacy_hash = hashlib.sha256(legacy.read_bytes()).hexdigest()
    selected_hash = hashlib.sha256(b"catalog").hexdigest()
    contract = {"source_sha256": selected_hash, "settings": {"image": "same"}}
    result = _seed_source_contract(_seed(legacy_hash), contract, selected_hash, legacy)
    assert result["source_sha256"] == legacy_hash
    assert result["settings"] == contract["settings"]


def test_wrong_seed_source_is_rejected(tmp_path):
    legacy = tmp_path / "legacy.json"
    legacy.write_text("legacy")
    selected_hash = hashlib.sha256(b"catalog").hexdigest()
    with pytest.raises(ValueError, match="neither"):
        _seed_source_contract(_seed("wrong"), {"source_sha256": selected_hash},
                              selected_hash, legacy)


def test_direct_catalog_keeps_changed_runtime_image_for_strict_plan_checks(tmp_path):
    selected = tmp_path / "catalog.json"
    selected.write_text("catalog")
    selected_hash = hashlib.sha256(selected.read_bytes()).hexdigest()
    contract = {"source_sha256": selected_hash,
                "settings": {"R9V_IMAGE": "changed", "R9V_RUNTIME_DESCRIPTOR": "changed"}}
    result = _seed_source_contract(_seed(selected_hash), contract, selected_hash,
                                   tmp_path / "missing-legacy.json")
    assert result["settings"] == contract["settings"]


@pytest.mark.parametrize("key", ["R9V_IMAGE", "R9V_RUNTIME_DESCRIPTOR", "R9V_MODEL_PACKAGE_SHA256"])
def test_direct_catalog_still_rejects_changed_runtime_and_model(tmp_path, key):
    import copy

    from tools.memory_seed import localize

    contract = {"source_sha256": "catalog-hash", "settings": {key: "tested"},
                "devices": [{"total_bytes": 32 * 2**30}] * 2, "driver": {}}
    seed = {"schema": "r9v.memory-seed.v1", "placement_qualification": ["qualified"],
            "calibration": {"contract": copy.deepcopy(contract), "workload_passed": True,
                            "ranks": [{}, {}]}}
    contract["settings"][key] = "changed"
    selected = _seed_source_contract(seed, contract, "catalog-hash", tmp_path / "missing")
    with pytest.raises(ValueError, match="image/model/workload"):
        localize(seed, selected, [0, 0])


@pytest.mark.parametrize("seed", [None, [], {"calibration": None}, {"calibration": []}])
def test_malformed_seed_refuses_before_touching_legacy_manifest(tmp_path, seed):
    with pytest.raises(ValueError, match="calibration"):
        _seed_source_contract(seed, {}, "catalog", tmp_path / "missing")
