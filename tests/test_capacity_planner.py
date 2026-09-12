# SPDX-License-Identifier: Apache-2.0
import copy

import pytest

from tests.test_expert_budget import calibrated_plan
from tools.plan_experts import plan


def capacity_fixture():
    source, contract, calibration = calibrated_plan()
    calibration.update(schema="r9v.memory-capacity-baseline.v1", workload_passed=False,
                       capacity_baseline_passed=True,
                       qualification_status="failed_headroom_only",
                       measured_hot_counts=[100, 120])
    return source, contract, calibration


def test_capacity_input_requires_explicit_opt_in():
    source, contract, calibration = capacity_fixture()
    with pytest.raises(ValueError, match="explicit"):
        plan(source, calibration, contract, [3 * 2**30] * 2, 128 * 2**30)


def test_capacity_candidate_never_grows_residency_or_claims_qualification():
    source, contract, calibration = capacity_fixture()
    original = copy.deepcopy(calibration)
    _, result = plan(source, calibration, contract, [3 * 2**30] * 2,
                     128 * 2**30, allow_capacity_baseline=True)
    assert result["hot_counts"] == [100, 120]
    assert result["capacity_baseline_estimate"] is True
    assert result["source_qualification_status"] == "failed_headroom_only"
    assert "requires fresh full workload" in result["qualification"]
    assert all(r["estimated_free_bytes"] >= r["target_free_bytes"] for r in result["ranks"])
    assert calibration == original


@pytest.mark.parametrize("key,value", [
    ("workload_passed", True), ("capacity_baseline_passed", False),
    ("qualification_status", "passed"), ("measured_hot_counts", [True, 100]),
    ("measured_hot_counts", [0, 100]), ("measured_hot_counts", [100]),
    ("evidence", []),
])
def test_capacity_rejects_invalid_provenance(key, value):
    source, contract, calibration = capacity_fixture()
    calibration[key] = value
    with pytest.raises(ValueError):
        plan(source, calibration, contract, [3 * 2**30] * 2,
             128 * 2**30, allow_capacity_baseline=True)


def test_capacity_keeps_contract_and_host_checks():
    source, contract, calibration = capacity_fixture()
    with pytest.raises(ValueError, match="stale"):
        plan(source, calibration, {**contract, "driver": {"changed": True}},
             [3 * 2**30] * 2, 128 * 2**30, allow_capacity_baseline=True)
    with pytest.raises(ValueError, match="Host RAM"):
        plan(source, calibration, contract, [3 * 2**30] * 2,
             10 * 2**30, allow_capacity_baseline=True)


def test_logging_run_id_does_not_invalidate_plan_or_compile_cache():
    from tools.plan_experts import runtime_contract
    from tools.runtime_cache_key import cache_key
    base = {"R9V_MTP_SPEC_TOKENS": "4"}
    first = {**base, "R9V_OBSERVABILITY_RUN_ID": "first", "R9V_OBSERVABILITY_TARGET": "first-log"}
    second = {**base, "R9V_OBSERVABILITY_RUN_ID": "second", "R9V_OBSERVABILITY_TARGET": "second-log"}
    assert runtime_contract(first, "source", [], {}) == runtime_contract(base, "source", [], {})
    assert runtime_contract(first, "source", [], {}) == runtime_contract(second, "source", [], {})
    assert cache_key("image", b"manifest", first, {}) == cache_key("image", b"manifest", second, {})
    changed = {**second, "R9V_MTP_SPEC_TOKENS": "3"}
    assert cache_key("image", b"manifest", changed, {}) != cache_key("image", b"manifest", first, {})


def test_stale_plan_reports_changed_keys_without_values():
    from tools.plan_experts import check_plan
    source, contract, calibration = calibrated_plan()
    manifest, result = plan(source, calibration, contract, [3 * 2**30] * 2, 128 * 2**30)
    changed = {**contract, "settings": {**contract["settings"], "R9V_PRIVATE_SETTING": "do-not-print"}}
    with pytest.raises(ValueError) as error:
        check_plan(result, manifest, changed, 128 * 2**30, [31 * 2**30] * 2)
    assert "R9V_PRIVATE_SETTING" in str(error.value)
    assert "do-not-print" not in str(error.value)
