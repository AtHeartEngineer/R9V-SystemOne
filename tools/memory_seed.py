#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Adapt a qualified release memory envelope for provisional local admission.

A reference envelope is an estimate, not local qualification. Setup must run the
advertised workload before saving a reusable qualification receipt.
"""

import copy
import hashlib
from pathlib import Path

# These vary by installation without changing model tensors or serving work.
PORTABLE = {
    "R9V_MODEL_DIR",
    "R9V_PLE_PATH",
    "R9V_MANIFEST_REL",
    "R9V_EXPECTED_GPU_BDFS",
    "R9V_VISIBLE_DEVICES",
    "R9V_EXPECTED_PCIE_LINKS",
    "R9V_MIN_PCIE_BANDWIDTH_GBPS",
    "R9V_REFERENCE_PCIE_BANDWIDTH_GBPS",
    "R9V_MIN_HOST_RAM_BYTES",
    "R9V_MIN_HOST_AVAILABLE_BYTES",
    "R9V_REFERENCE_HOST_RAM_BYTES",
    "R9V_DOCTOR_STRICT",
    "R9V_SERVED_MODEL_NAME",
}


def localize(seed, contract, external):
    if seed.get("schema") not in ("r9v.memory-seed.v1", "r9v.public-memory-seed.v2"):
        raise ValueError("unsupported release memory seed")
    reference = seed.get("calibration", {})
    if not reference.get("workload_passed") or not seed.get("placement_qualification"):
        raise ValueError(
            "release memory seed needs workload and placement qualification"
        )
    measured = reference["contract"]
    before = {k: v for k, v in measured["settings"].items() if k not in PORTABLE}
    after = {k: v for k, v in contract["settings"].items() if k not in PORTABLE}
    if before != after or measured["source_sha256"] != contract["source_sha256"]:
        raise ValueError(
            "release memory seed does not match this image/model/workload; use local calibration"
        )
    if len(external) != 2 or len(contract["devices"]) != 2:
        raise ValueError("release memory seed requires two GPUs")
    result = copy.deepcopy(reference)
    if seed['schema'] == 'r9v.public-memory-seed.v2':
        # verify_bundle() validates these public payloads before admission.
        # Keep their references when adapting the envelope for the planner.
        records = seed.get('public_evidence', {})
        if not isinstance(records, dict) or not records:
            raise ValueError('public memory seed needs evidence references')
        result['evidence'] = copy.deepcopy(list(records.values()))
    result["contract"] = contract
    result["workload_passed"] = False
    result["reference_workload_passed"] = True
    result["scope"] = "reference estimate awaiting local qualification"
    result["reference_driver"] = measured["driver"]
    for rank, used in enumerate(external):
        capacity = contract["devices"][rank]["total_bytes"]
        if (
            type(capacity) is not int
            or capacity <= 0
            or type(used) is not int
            or not 0 <= used < capacity
        ):
            raise ValueError("invalid external GPU allocation")
        # Allocation sizes are byte counts, not fractions of reference VRAM.
        # The planner must fit them to the actual capacity; slightly different
        # firmware/driver reservations do not invalidate the reference estimate.
        result["ranks"][rank]["external_allowance_bytes"] = used + 256 * 2**20
    return result


def verify_bundle(seed, base):
    """Verify packaged evidence before using a shipped reference envelope."""
    if not isinstance(seed, dict):
        raise ValueError("memory seed must be an object")  # noqa: TRY004 - public validation contract
    if seed.get("schema") == "r9v.public-memory-seed.v2":
        try:
            from tools.public_memory_seed import verify_public
        except ModuleNotFoundError:
            from public_memory_seed import verify_public
        return verify_public(seed, base)
    provenance = seed.get("provenance", {})
    identities = provenance.get("original_evidence_identities")
    if not isinstance(identities, dict) or not identities:
        raise ValueError("memory seed has no packaged evidence identities")
    base = Path(base).resolve()
    records = list(identities.values())
    records += seed.get("calibration", {}).get("evidence", [])
    placement = seed.get("placement_qualification", {})
    if not isinstance(placement, dict) or placement.get("passed") is not True:
        raise ValueError("memory seed has no passing reference placement")
    for key in ("manifest", "plan", "source_catalog", "session_result", "workload", "context"):
        records.append(placement.get(key))
    checked = {}
    total = 0
    for row in records:
        if not isinstance(row, dict) or not isinstance(row.get("path"), str):
            raise ValueError("malformed memory seed evidence record")  # noqa: TRY004 - public validation contract
        relative = Path(row["path"])
        path = (base / relative).resolve()
        if relative.is_absolute() or ".." in relative.parts or not path.is_relative_to(base):
            raise ValueError("memory seed evidence escapes its package")
        if path not in checked:
            size = path.stat().st_size
            total += size
            if size > 64 * 2**20 or total > 128 * 2**20:
                raise ValueError("memory seed evidence exceeds package limit")
            checked[path] = (size, hashlib.sha256(path.read_bytes()).hexdigest())
        if checked[path] != (row.get("bytes"), row.get("sha256")):
            raise ValueError("memory seed evidence changed: " + relative.name)
    return len(checked)
