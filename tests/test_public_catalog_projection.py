# SPDX-License-Identifier: Apache-2.0
import json

import pytest

from tools.plan_experts import digest
from tools.public_memory_seed import project_public_catalog, sha, verify_projection


def encode(value):
    return (json.dumps(value, indent=2) + "\n").encode()


@pytest.fixture
def originals():
    catalog = {
        "ranks": {"0": {"hot_experts_by_layer": [[2, 1, 0]]}},
        "expert_memory": {"packed_bytes": [100, 200]},
        "ranking": {
            "inputs": [{"path": "/var/home/private/" + role, "sha256": str(i) * 64}
                       for i, role in enumerate(("source", "train", "holdout"))],
            "training_counts": [[3, 2, 1]],
            "capture_binding": {"runtime_hash": "a" * 64},
        },
    }
    raw = encode(catalog)
    manifest = {"ranks": {"0": {"hot_experts_by_layer": [[2, 1]]}},
                "expert_memory": {"packed_bytes": [100, 200]},
                "r9v_derivation": {"source_sha256": sha(raw), "method": "prefix"}}
    return raw, encode(manifest)


def test_projection_is_deterministic_and_keeps_original_bytes_and_execution_commitment(originals):
    raw, manifest_raw = originals
    before = (raw, manifest_raw)
    projected, published_manifest, record = project_public_catalog(*originals)
    assert project_public_catalog(*originals) == (projected, published_manifest, record)
    assert originals == before
    assert record["original_catalog_sha256"] == sha(raw)
    assert record["original_manifest_file_sha256"] == sha(manifest_raw)
    assert record["executed_manifest_sha256"] == digest(json.loads(manifest_raw))
    assert b"/var/home/" not in projected
    assert json.loads(projected)["ranks"] == json.loads(raw)["ranks"]
    assert json.loads(published_manifest)["ranks"] == json.loads(manifest_raw)["ranks"]
    proof = {"executed_manifest_sha256": record["executed_manifest_sha256"]}
    verify_projection(record, projected, json.loads(published_manifest), proof)


@pytest.mark.parametrize("field", ["ids", "counts", "costs", "runtime"])
def test_projection_rejects_changed_catalog_content_even_with_updated_file_hash(originals, field):
    raw, mr, record = project_public_catalog(*originals)
    value = json.loads(raw)
    if field == "ids":
        value["ranks"]["0"]["hot_experts_by_layer"][0].reverse()
    if field == "counts":
        value["ranking"]["training_counts"][0][0] += 1
    if field == "costs":
        value["expert_memory"]["packed_bytes"][0] += 1
    if field == "runtime":
        value["ranking"]["capture_binding"]["runtime_hash"] = "b" * 64
    raw = encode(value)
    record["public_catalog_sha256"] = sha(raw)
    with pytest.raises(ValueError, match="content differs"):
        verify_projection(record, raw, json.loads(mr), {"executed_manifest_sha256": record["executed_manifest_sha256"]})


def test_projection_rejects_changed_executed_prefix_even_with_updated_manifest_hash(originals):
    raw, mr, record = project_public_catalog(*originals)
    manifest = json.loads(mr)
    manifest["ranks"]["0"]["hot_experts_by_layer"][0].reverse()
    record["public_manifest_sha256"] = digest(manifest)
    with pytest.raises(ValueError, match="content differs"):
        verify_projection(record, raw, manifest, {"executed_manifest_sha256": record["executed_manifest_sha256"]})


def test_projection_rejects_unbound_original_manifest(originals):
    raw, mr = originals
    manifest = json.loads(mr)
    manifest["r9v_derivation"]["source_sha256"] = "f" * 64
    with pytest.raises(ValueError, match="original manifest"):
        project_public_catalog(raw, encode(manifest))


def test_public_input_locations_need_no_projection_and_keep_byte_identity(originals):
    raw, mr = originals
    raw = raw.replace(b"/var/home/private/", b"inputs/")
    assert project_public_catalog(raw, mr) == (raw, mr, None)


def test_unexpected_provenance_fields_are_not_silently_removed(originals):
    raw, mr = originals
    value = json.loads(raw)
    value["ranking"]["inputs"][0]["extra"] = "private"
    with pytest.raises(ValueError, match="unexpected ranking input"):
        project_public_catalog(encode(value), mr)
