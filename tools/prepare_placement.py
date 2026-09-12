# SPDX-License-Identifier: Apache-2.0
"""Prepare provisional reference-based placement and reuse validated contracts."""

import hashlib
import json
import time
from pathlib import Path

try:
    from tools.expert_budget import headroom_bytes
    from tools.memory_seed import localize, verify_bundle
    from tools.plan_experts import check_plan, live_contract, plan
except ModuleNotFoundError:
    from expert_budget import headroom_bytes
    from memory_seed import localize, verify_bundle
    from plan_experts import check_plan, live_contract, plan

ROOT = Path(__file__).resolve().parents[1]


def _seed_source_contract(seed, contract, selected_source_hash, original):
    """Choose the seed basis without weakening plan() identity checks."""
    calibration = seed.get("calibration") if isinstance(seed, dict) else None
    if not isinstance(calibration, dict):
        raise ValueError("seed calibration must be an object")  # noqa: TRY004 - public validation contract
    seed_contract = calibration.get("contract", {})
    seed_source_hash = seed_contract.get("source_sha256") if isinstance(seed_contract, dict) else None
    if seed_source_hash == selected_source_hash:
        return contract
    try:
        original_hash = hashlib.sha256(original.read_bytes()).hexdigest()
    except OSError as error:
        raise ValueError(
            "seed source does not match selected catalog and legacy manifest is unavailable"
        ) from error
    if seed_source_hash != original_hash:
        raise ValueError("seed source matches neither selected catalog nor legacy manifest")
    return {**contract, "source_sha256": original_hash}


def apply(env, runtime, state_dir):
    location = env.get("R9V_MEMORY_SEED_PATH") or runtime.get("distribution", {}).get(
        "memory_seed"
    )
    if not location:
        return False
    env.setdefault("R9V_MIN_FREE_VRAM_GIB_BY_RANK", "3,3")
    seed_path = Path(location)
    if not seed_path.is_absolute():
        seed_path = ROOT / seed_path
    raw = seed_path.read_bytes()
    seed_sha = hashlib.sha256(raw).hexdigest()
    seed = json.loads(raw)
    verify_bundle(seed, seed_path.parent)
    original = Path(env["R9V_MODEL_DIR"]) / env.get(
        "R9V_MANIFEST_REL",
        "manifests/hot-manifest-q4-vision-128k-multiprompt-r1-lru16-neutral.json",
    )
    source = Path(env.get("R9V_EXPERT_CATALOG_PATH") or original)
    contract = live_contract(env, hashlib.sha256(source.read_bytes()).hexdigest())
    targets = headroom_bytes(env.get("R9V_MIN_FREE_VRAM_GIB_BY_RANK", "3,3"), 2)
    available = (
        int(
            next(
                line.split()[1]
                for line in Path("/proc/meminfo").read_text().splitlines()
                if line.startswith("MemAvailable:")
            )
        )
        * 1024
    )
    external = [
        int(
            (
                Path("/sys/bus/pci/devices") / device["bdf"] / "mem_info_vram_used"
            ).read_text()
        )
        for device in contract["devices"]
    ]
    free = [
        device["total_bytes"] - used
        for device, used in zip(contract["devices"], external)
    ]
    if env.get("R9V_PLACEMENT_PLAN") and env.get("R9V_EXPERT_MANIFEST_PATH"):
        try:
            old = json.loads(Path(env["R9V_PLACEMENT_PLAN"]).read_text())
            manifest = json.loads(Path(env["R9V_EXPERT_MANIFEST_PATH"]).read_text())
            if (
                old.get("memory_seed_sha256") == seed_sha
                and [r["target_free_bytes"] for r in old["ranks"]] == targets
            ):
                check_plan(old, manifest, contract, available, free)
                return True
        except (OSError, ValueError, KeyError, TypeError):
            pass
    selected_source_hash = contract["source_sha256"]
    localization_contract = _seed_source_contract(
        seed, contract, selected_source_hash, original
    )
    estimate = localize(seed, localization_contract, external)
    manifest, result = plan(
        json.loads(source.read_text()),
        estimate,
        contract,
        targets,
        available,
        allow_reference=True,
        runtime=runtime,
    )
    result.update(source_path=str(source.resolve()), memory_seed_sha256=seed_sha)
    destination = state_dir / f"placement-{time.time_ns()}"
    destination.mkdir(mode=0o700)
    for name, value in [("manifest.json", manifest), ("plan.json", result)]:
        with (destination / name).open("x") as stream:
            json.dump(value, stream, indent=2)
            stream.write("\n")
    env.update(
        R9V_EXPERT_MANIFEST_PATH=str(destination / "manifest.json"),
        R9V_PLACEMENT_PLAN=str(destination / "plan.json"),
        R9V_MAX_EFFECTIVE_EXPERTS_PER_RANK=",".join(
            str(n + row["cache_physical_slots"])
            for n, row in zip(result["hot_counts"], result["expert_memory"])
        ),
    )
    print(
        "Provisional local placement: hot experts "
        + str(result["hot_counts"])
        + "; first start requires the complete workload check.",
        flush=True,
    )
    return True
