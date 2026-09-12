# SPDX-License-Identifier: Apache-2.0
import copy
import hashlib
import json
import shutil
from pathlib import Path

import pytest

from tools.memory_seed import localize, verify_bundle

ROOT = Path(__file__).resolve().parents[1]
SEEDS = [
    "packages/placements/qwen38-flash-next/ud-iq4-xs/dual-r9700/qualified-128k-r3",
    "packages/placements/qwen38-flash-next/ud-q4-k-xl/dual-r9700/image6-routing-r4/qualified-128k-r2",
]


@pytest.fixture(params=SEEDS)
def bundle(request, tmp_path):
    source = ROOT / request.param
    shutil.copytree(source, tmp_path / "seed")
    base = tmp_path / "seed"
    return json.loads((base / "memory-seed.json").read_text()), base


def rewrite(seed, base, key, mutate):
    row = seed["public_evidence"][key]
    path = base / row["path"]
    value = json.loads(path.read_text())
    mutate(value)
    raw = (json.dumps(value) + "\n").encode()
    path.write_bytes(raw)
    row.update(bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())


def test_public_numeric_evidence_recomputes_and_localizes_as_unqualified(bundle):
    seed, base = bundle
    assert verify_bundle(seed, base) == 5
    estimate = localize(seed, copy.deepcopy(seed["calibration"]["contract"]), [0, 0])
    assert estimate["workload_passed"] is False
    assert estimate["reference_workload_passed"] is True
    for path in base.iterdir():
        raw = path.read_bytes()
        assert b"/var/mnt/" not in raw and b"/home/dylan/" not in raw
        assert b'"choices"' not in raw and b'"prompt"' not in raw


@pytest.mark.parametrize(
    "field",
    ["non_expert_peak_bytes", "external_allowance_bytes", "transient_margin_bytes"],
)
def test_modified_envelope_fails_even_when_files_are_intact(bundle, field):
    seed, base = bundle
    seed["calibration"]["ranks"][0][field] += 1
    with pytest.raises(ValueError, match="arithmetic"):
        verify_bundle(seed, base)


def test_public_seed_reaches_planner_but_still_requires_local_qualification(bundle):
    from tools.plan_experts import plan

    seed, base = bundle
    verify_bundle(seed, base)
    contract = copy.deepcopy(seed['calibration']['contract'])
    estimate = localize(seed, contract, [0, 0])
    records = seed['public_evidence']
    source = json.loads((base / records['catalog']['path']).read_text())
    runtime = json.loads((base / records['runtime']['path']).read_text())
    _, result = plan(source, estimate, contract, [3 * 2**30] * 2,
                     512 * 2**30, allow_reference=True, runtime=runtime)
    assert estimate['evidence'] == list(records.values())
    assert estimate['workload_passed'] is False
    assert result['reference_estimate'] is True
    assert result['qualification'] == 'requires admission and workload validation after loading'
    with pytest.raises(ValueError, match='passing workload evidence'):
        plan(source, estimate, contract, [3 * 2**30] * 2,
             512 * 2**30, allow_reference=False, runtime=runtime)


def test_rehashed_incomplete_context_is_not_qualification(bundle):
    seed, base = bundle
    rewrite(seed, base, "workload", lambda value: value.update(prompt_tokens=1000))
    with pytest.raises(ValueError, match="context"):
        verify_bundle(seed, base)


def test_rehashed_incomplete_workload_is_rejected(bundle):
    seed, base = bundle
    rewrite(seed, base, "workload", lambda value: value["checks"].remove("tool"))
    with pytest.raises(ValueError, match="workload"):
        verify_bundle(seed, base)


def test_rehashed_wrong_image_is_rejected(bundle):
    seed, base = bundle
    rewrite(
        seed,
        base,
        "workload",
        lambda value: value.update(image_id="sha256:" + "0" * 64),
    )
    with pytest.raises(ValueError, match="image"):
        verify_bundle(seed, base)


def test_rehashed_telemetry_peak_does_not_match_envelope(bundle):
    seed, base = bundle
    rewrite(
        seed,
        base,
        "measurements",
        lambda value: value["samples"][-1]["free_bytes"].__setitem__(0, 0),
    )
    with pytest.raises(ValueError, match="arithmetic"):
        verify_bundle(seed, base)


def test_missing_startup_and_symlink_evidence_rejected(bundle):
    seed, base = bundle
    rewrite(
        seed,
        base,
        "measurements",
        lambda value: value["samples"][0].update(phase="workload"),
    )
    with pytest.raises(ValueError, match="startup"):
        verify_bundle(seed, base)
    path = base / seed["public_evidence"]["catalog"]["path"]
    target = base.parent / "external.json"
    path.rename(target)
    path.symlink_to(target)
    with pytest.raises(ValueError, match="escapes"):
        verify_bundle(seed, base)
