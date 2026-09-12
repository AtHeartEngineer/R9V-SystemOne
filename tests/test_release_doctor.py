# SPDX-License-Identifier: Apache-2.0
import json

from tools import profile_doctor as doctor
from tools import plan_experts


def test_doctor_reports_missing_plan_for_saved_custom_headroom(monkeypatch):
    monkeypatch.delenv('R9V_PLACEMENT_PLAN', raising=False)
    monkeypatch.setenv('R9V_HEADROOM_SELECTION', '1')
    reporter = doctor.Reporter()
    doctor._check_placement_plan(reporter)
    assert reporter.checks[0].status == 'FAIL'
    assert 'no calibrated placement' in reporter.checks[0].message


def test_doctor_rejects_changed_headroom_even_when_workload_contract_matches(tmp_path, monkeypatch):
    source = tmp_path / 'source.json'
    source.write_text('{}')
    manifest = tmp_path / 'manifest.json'
    manifest.write_text('{}')
    plan = tmp_path / 'plan.json'
    plan.write_text(json.dumps({'source_path': str(source), 'ranks': [
        {'target_free_bytes': 3 * 2**30}, {'target_free_bytes': 3 * 2**30}]}))
    monkeypatch.setenv('R9V_PLACEMENT_PLAN', str(plan))
    monkeypatch.setenv('R9V_EXPERT_MANIFEST_PATH', str(manifest))
    monkeypatch.setenv('R9V_MIN_FREE_VRAM_GIB_BY_RANK', '6,4')
    reporter = doctor.Reporter()
    doctor._check_placement_plan(reporter, runtime=True)
    assert reporter.checks[0].status == 'FAIL'
    assert 'Requested headroom differs' in reporter.checks[0].message


def test_doctor_uses_selected_runtime_cache_and_mtp_capability(tmp_path, monkeypatch):
    descriptor = tmp_path / 'runtime.json'
    descriptor.write_text(json.dumps({'capabilities': {'max_cache_slots': 192, 'reference_mtp_depth': 4}}))
    for key, value in {'R9V_RUNTIME_DESCRIPTOR': str(descriptor), 'R9V_TIERED_IQ_MOE_VARIANT': 'reuse3v2',
                       'R9V_TIERED_EXPERT_CACHE_SLOTS': '160', 'R9V_TIERED_EXPERT_CACHE_RANKS': '0',
                       'R9V_TIERED_EXPERT_CACHE_POLICY': 'lru', 'R9V_MTP_SPEC_TOKENS': '4',
                       'R9V_PLE_RESIDENCY_MODE': 'ssd'}.items():
        monkeypatch.setenv(key, value)
    reporter = doctor.Reporter()
    doctor._check_profile_policy(reporter, 2, [])
    assert next(c for c in reporter.checks if c.name == 'cache-policy').status == 'PASS'
    assert not any(c.name == 'workload-envelope' for c in reporter.checks)
    descriptor.write_text(json.dumps({'capabilities': {'max_cache_slots': 16}}))
    reporter = doctor.Reporter()
    doctor._check_profile_policy(reporter, 2, [])
    assert next(c for c in reporter.checks if c.name == 'cache-policy').status == 'FAIL'


def test_planner_reports_both_infeasible_headroom_targets():
    from tests.test_expert_budget import calibrated_plan
    source, contract, calibration = calibrated_plan()
    import pytest
    with pytest.raises(ValueError) as error:
        plan_experts.plan(source, calibration, contract, [40 * 2**30] * 2, 100 * 2**30)
    assert 'Rank 0' in str(error.value) and 'Rank 1' in str(error.value)
    assert 'shortfall' in str(error.value)


def test_doctor_rejects_ranking_from_another_runtime_image(tmp_path, monkeypatch):
    import hashlib
    from pathlib import Path
    from tools.rank_experts import rank_catalog
    root = Path(__file__).resolve().parents[1]
    profile_root = root / 'profiles/qwen38-flash-next/dual-r9700-q4-k-xl'
    profile = json.loads((profile_root / 'profile.json').read_text())
    package_path = root / profile['descriptors']['model_package']
    package = json.loads(package_path.read_text())
    runtime = root / profile['descriptors']['runtime']
    image = 'sha256:' + 'd' * 64
    binding = {'model_package': package['id'], 'model_hash': hashlib.sha256(package_path.read_bytes()).hexdigest(),
               'runtime_hash': hashlib.sha256((hashlib.sha256(runtime.read_bytes()).hexdigest() + '\0' + image).encode()).hexdigest(), 'config_hash': 'a' * 64}
    histogram = {'schema': 'r9v.routes.v1', 'metadata': binding,
                 'decode_counts': [[1] * 512 for _ in range(48)],
                 'prefill_counts': [[1] * 512 for _ in range(48)]}
    source = json.loads((root / 'packages/placements/qwen38-flash-next/ud-q4-k-xl/dual-r9700/bootstrap-manifest.json').read_text())
    catalog = tmp_path / 'catalog.json'
    catalog.write_text(json.dumps(rank_catalog(source, histogram, histogram)))
    for key, value in {'R9V_PROFILE_ROOT': str(profile_root), 'R9V_PROFILE_ID': profile['id'],
                       'R9V_MODEL_PACKAGE': package['id'], 'R9V_RUNTIME_DESCRIPTOR': str(runtime),
                       'R9V_EXPERT_CATALOG_PATH': str(catalog), 'R9V_EXPERT_TP_SPLIT': '416,224',
                       'R9V_IMAGE': image}.items():
        monkeypatch.setenv(key, value)
    monkeypatch.delenv('R9V_EXPERT_MANIFEST_PATH', raising=False)
    monkeypatch.delenv('R9V_MODEL_PACKAGE_SHA256', raising=False)
    report = doctor.Reporter()
    doctor._check_release_assets(report, root)
    assert next(c for c in report.checks if c.name == 'expert-ranking').status == 'PASS'
    monkeypatch.setenv('R9V_IMAGE', 'sha256:' + 'e' * 64)
    report = doctor.Reporter()
    doctor._check_release_assets(report, root)
    failure = next(c for c in report.checks if c.name == 'expert-ranking')
    assert failure.status == 'FAIL'
    assert 'different runtime image' in failure.message


def test_doctor_rejects_cache_backend_before_loading_weights(tmp_path, monkeypatch):
    descriptor = tmp_path / 'runtime.json'
    descriptor.write_text(json.dumps({'capabilities': {'max_cache_slots': 192, 'reference_mtp_depth': 4}}))
    for key, value in {'R9V_RUNTIME_DESCRIPTOR': str(descriptor), 'R9V_TIERED_EXPERT_CACHE_SLOTS': '80',
                       'R9V_TIERED_EXPERT_CACHE_RANKS': '0', 'R9V_TIERED_EXPERT_CACHE_POLICY': 'lru',
                       'R9V_CACHE80': '1', 'R9V_CACHE192': '1', 'R9V_MTP_SPEC_TOKENS': '4',
                       'R9V_TIERED_IQ_MOE_VARIANT': 'reuse3v2', 'R9V_PLE_RESIDENCY_MODE': 'ssd'}.items():
        monkeypatch.setenv(key, value)
    report = doctor.Reporter()
    doctor._check_profile_policy(report, 2, [])
    assert any(c.status == 'FAIL' and 'cache prepare backend' in c.message for c in report.checks)
    monkeypatch.setenv('R9V_CACHE192', '0')
    report = doctor.Reporter()
    doctor._check_profile_policy(report, 2, [])
    assert not any(c.status == 'FAIL' and 'cache prepare backend' in c.message for c in report.checks)
