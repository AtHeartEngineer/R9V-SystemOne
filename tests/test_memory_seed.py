import copy

import pytest

from tools.memory_seed import localize


def fixture():
    contract = {
        "settings": {
            "R9V_IMAGE": "fixed-image",
            "R9V_MAX_MODEL_LEN": "131072",
            "R9V_MODEL_DIR": "/old",
            "R9V_EXPECTED_GPU_BDFS": "a,b",
        },
        "source_sha256": "fixed-source",
        "driver": {"kernel": "reference"},
        "devices": [{"bdf": bdf, "total_bytes": 32 * 2**30} for bdf in ("a", "b")],
    }
    seed = {
        "schema": "r9v.memory-seed.v1",
        "placement_qualification": ["qualified-plan-sha"],
        "calibration": {
            "contract": copy.deepcopy(contract),
            "workload_passed": True,
            "ranks": [{}, {}],
        },
    }
    return seed, contract


def test_reference_seed_adapts_external_use_but_requires_local_qualification():
    seed, contract = fixture()
    contract["settings"]["R9V_MODEL_DIR"] = "/new"
    contract["settings"]["R9V_EXPECTED_GPU_BDFS"] = "c,d"
    contract["driver"]["kernel"] = "local"
    result = localize(seed, contract, [5 * 2**30, 2**30])
    assert not result["workload_passed"]
    assert result["reference_workload_passed"]
    assert result["ranks"][0]["external_allowance_bytes"] == 5 * 2**30 + 256 * 2**20
    assert seed["calibration"]["workload_passed"]
    assert result["reference_driver"] == {"kernel": "reference"}


@pytest.mark.parametrize(
    "failure", ["image", "context", "model", "capacity", "qualification"]
)
def test_seed_refuses_unmeasured_workload_or_insufficient_hardware(failure):
    seed, contract = fixture()
    if failure == "image":
        contract["settings"]["R9V_IMAGE"] = "other-image"
    elif failure == "context":
        contract["settings"]["R9V_MAX_MODEL_LEN"] = "262144"
    elif failure == "model":
        contract["source_sha256"] = "other-source"
    elif failure == "capacity":
        contract["devices"][0]["total_bytes"] = 0
    else:
        seed["placement_qualification"] = []
    with pytest.raises(ValueError):
        localize(seed, contract, [0, 0])


def test_reference_capacity_is_not_a_literal_admission_floor():
    seed, contract = fixture()
    contract["devices"][0]["total_bytes"] -= 128 * 2**20
    result = localize(seed, contract, [0, 0])
    assert result["contract"]["devices"][0]["total_bytes"] == 32 * 2**30 - 128 * 2**20
    assert not result["workload_passed"]
