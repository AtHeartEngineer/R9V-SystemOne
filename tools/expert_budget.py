# SPDX-License-Identifier: Apache-2.0
"""Validate placement contents and account for packed expert bytes, not total VRAM."""

from __future__ import annotations

import json
import hashlib
import math
import re
from pathlib import Path

COST_PATH = (
    Path(__file__).resolve().parents[1]
    / "packages/placements/qwen38-flash-next/ud-iq4-xs/dual-r9700/expert-memory.json"
)
DEFAULT_RUNTIME_MAX_CACHE_SLOTS = 16


def _read_catalog(catalog):
    if catalog is None:
        catalog = COST_PATH
    if isinstance(catalog, (str, Path)):
        catalog = json.loads(Path(catalog).read_text())
    if not isinstance(catalog, dict):
        raise ValueError("expert cost catalog must be an object or JSON path")
    costs = catalog.get("packed_bytes_per_expert_by_layer_per_rank")
    if catalog.get("schema") != "r9v.expert-memory.v1":
        raise ValueError("unsupported expert cost catalog")
    by_rank = catalog.get("packed_bytes_per_expert_by_layer_by_rank")
    if costs is None and by_rank is None:
        raise ValueError("expert cost catalog needs packed per-layer costs")
    if costs is not None and (not isinstance(costs, list) or len(costs) != 48 or any(
        type(n) is not int or n <= 0 for n in costs
    )):
        raise ValueError("expert cost catalog needs 48 positive integer layer costs")
    if catalog.get("num_layers", 48) != 48 or catalog.get("num_experts", 512) != 512:
        raise ValueError("expert cost catalog dimensions must be 48 layers and 512 experts")
    if not isinstance(catalog.get("model_package"), str) or not catalog["model_package"]:
        raise ValueError("expert cost catalog needs a model_package identity")
    if catalog.get("tensor_parallel_size", 2) != 2:
        raise ValueError("expert cost catalog requires tensor parallel size 2")
    artifacts = catalog.get("target_artifacts")
    if artifacts is not None:
        if not isinstance(artifacts, list) or not artifacts:
            raise ValueError("target_artifacts must be a nonempty list")
        paths = set()
        for artifact in artifacts:
            if not isinstance(artifact, dict):
                raise ValueError("target artifact must be an object")
            path, size, sha = (artifact.get(k) for k in ("path", "bytes", "sha256"))
            if (not isinstance(path, str) or not path or Path(path).is_absolute()
                    or ".." in Path(path).parts or path in paths
                    or type(size) is not int or size <= 0
                    or not isinstance(sha, str) or not re.fullmatch(r"[0-9a-f]{64}", sha)):
                raise ValueError("target artifact requires unique relative path, positive bytes and complete SHA-256")
            paths.add(path)
    partition = catalog.get("expert_partition", [320, 320])
    if (not isinstance(partition, list) or len(partition) != 2 or
            any(type(n) is not int or n <= 0 for n in partition) or
            sum(partition) != 640 or any(n not in (192,224,320,416,448) for n in partition)):
        raise ValueError("expert cost catalog expert_partition needs supported channel widths summing to 640")
    by_rank = catalog.get("packed_bytes_per_expert_by_layer_by_rank")
    if by_rank is not None:
        if not isinstance(by_rank, dict) or set(by_rank) != {"0", "1"}:
            raise ValueError("rank-specific expert costs must contain ranks 0 and 1")
        for rank in ("0", "1"):
            values = by_rank[rank]
            if (not isinstance(values, list) or len(values) != 48 or
                    any(type(n) is not int or n <= 0 for n in values)):
                raise ValueError(f"rank {rank} needs 48 positive integer layer costs")
        if costs is not None and any(by_rank[rank] != costs for rank in ("0", "1")):
            raise ValueError("legacy and rank-specific cost vectors conflict")
        if partition == [320, 320] and any(by_rank["0"][i] != by_rank["1"][i] for i in range(48)):
            raise ValueError("asymmetric rank costs require an asymmetric expert_partition")
    elif partition != [320, 320]:
        raise ValueError("asymmetric expert_partition requires rank-specific packed costs")
    return catalog


def cost_vectors(catalog=None) -> list[list[int]]:
    """Return the exact packed bytes per layer for each TP rank."""
    value = _read_catalog(catalog)
    by_rank = value.get("packed_bytes_per_expert_by_layer_by_rank")
    if by_rank is not None:
        return [list(by_rank[str(rank)]) for rank in range(2)]
    legacy = list(value["packed_bytes_per_expert_by_layer_per_rank"])
    return [legacy, list(legacy)]


def cost_catalog(data=None, catalog=None) -> dict:
    """Resolve explicit or manifest-embedded costs, retaining legacy fallback."""
    if catalog is not None:
        resolved = _read_catalog(catalog)
    else:
        resolved = None
    if isinstance(data, dict):
        for key in ("expert_memory", "packed_cost_catalog", "expert_cost_catalog_data"):
            embedded = data.get(key)
            if isinstance(embedded, dict) and (
                "packed_bytes_per_expert_by_layer_per_rank" in embedded or
                "packed_bytes_per_expert_by_layer_by_rank" in embedded
            ):
                if resolved is not None and catalog_identity(resolved) != catalog_identity(embedded):
                    raise ValueError("explicit expert cost catalog does not match embedded catalog")
                resolved = _read_catalog(embedded)
                break
        identity = data.get("expert_cost_catalog")
        if identity is not None and not isinstance(identity, dict):
            raise ValueError("manifest expert cost catalog identity must be an object")
        if identity is not None and resolved is None:
            resolved = _read_catalog(None)
        if identity is not None and resolved is not None and catalog_identity(resolved) != identity:
            raise ValueError("manifest expert cost catalog identity does not match selected catalog")
    resolved = resolved if resolved is not None else _read_catalog(None)
    if isinstance(data, dict) and data.get('model_package') not in (None, resolved['model_package']):
        raise ValueError('Manifest model package differs from expert cost catalog')
    return resolved


def validate_cost_contract(catalog, config):
    costs = _read_catalog(catalog)
    expected = config.get('R9V_MODEL_PACKAGE')
    if expected and expected != costs['model_package']:
        raise ValueError('Expert cost catalog belongs to a different model package')
    raw = config.get('R9V_EXPERT_TP_SPLIT', '320,320')
    partition = [int(x) for x in raw.split(',')]
    if partition != costs.get('expert_partition', [320, 320]):
        raise ValueError('Expert cost catalog channel partition differs from the runtime')


def catalog_identity(catalog=None) -> dict:
    """Return a stable identity for the exact packed tensor byte costs."""
    value = _read_catalog(catalog)
    costs = json.dumps(cost_vectors(value), separators=(",", ":"))
    result = {
        "schema": value["schema"],
        "model_package": value["model_package"],
        "packed_costs_sha256": hashlib.sha256(costs.encode()).hexdigest(),
        "catalog_sha256": hashlib.sha256(
            json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
    }
    if "quantization" in value:
        result["quantization"] = value["quantization"]
    return result


def runtime_cache_limit(runtime=None) -> int:
    if runtime is None:
        return DEFAULT_RUNTIME_MAX_CACHE_SLOTS
    if isinstance(runtime, (str, Path)):
        runtime = json.loads(Path(runtime).read_text())
    if not isinstance(runtime, dict):
        raise ValueError("runtime descriptor must be an object or JSON path")
    capabilities = runtime.get("capabilities", {})
    if not isinstance(capabilities, dict):
        raise ValueError("runtime capabilities must be an object")
    limit = capabilities.get("max_cache_slots", DEFAULT_RUNTIME_MAX_CACHE_SLOTS)
    if type(limit) is not int:
        raise ValueError("runtime capabilities.max_cache_slots must be an integer")
    if not 0 <= limit <= 192:
        raise ValueError("runtime max_cache_slots must be 0..192")
    return limit


def headroom_bytes(value: str, count: int) -> list[int]:
    try:
        values = [float(item.strip()) for item in value.split(",")]
    except ValueError as error:
        raise ValueError(
            "R9V_MIN_FREE_VRAM_GIB_BY_RANK must contain numeric GiB values"
        ) from error
    if len(values) != count or any(
        not math.isfinite(item) or item < 0 or not math.isfinite(item * 1024**3)
        for item in values
    ):
        raise ValueError(
            f"R9V_MIN_FREE_VRAM_GIB_BY_RANK needs {count} finite non-negative GiB values"
        )
    return [math.ceil(item * 1024**3) for item in values]


def validate_manifest(data: dict, count: int = 2) -> list[list[int]]:
    if (
        not isinstance(data, dict)
        or data.get("version") != 1
        or data.get("num_layers") != 48
        or data.get("num_experts") != 512
    ):
        raise ValueError("manifest must describe version 1, 48 layers, 512 experts")
    if data.get("top_k", 10) != 10:
        raise ValueError("manifest top_k must match the target's 10 routed experts")
    ranks = data.get("ranks")
    if not isinstance(ranks, dict) or set(ranks) != {str(i) for i in range(count)}:
        raise ValueError(f"manifest must contain exactly TP ranks 0..{count - 1}")
    counts = []
    for rank in range(count):
        entry = ranks[str(rank)]
        if not isinstance(entry, dict):
            raise ValueError(f"rank {rank}: expected an object")
        layers = entry.get("hot_experts_by_layer")
        if not isinstance(layers, list) or len(layers) != 48:
            raise ValueError(f"rank {rank}: expected 48 hot-expert lists")
        row_counts = []
        for layer, ids in enumerate(layers):
            if not isinstance(ids, list) or not ids:
                raise ValueError(
                    f"rank {rank} layer {layer}: at least one hot expert is required by this runtime"
                )
            if any(type(i) is not int or not 0 <= i < 512 for i in ids):
                raise ValueError(
                    f"rank {rank} layer {layer}: IDs must be integers in 0..511"
                )
            if len(set(ids)) != len(ids):
                raise ValueError(f"rank {rank} layer {layer}: duplicate expert IDs")
            row_counts.append(len(ids))
        declared = entry.get("hot_count")
        if declared is not None and (
            type(declared) is not int or any(n != declared for n in row_counts)
        ):
            raise ValueError(
                f"rank {rank}: declared hot_count does not match the actual per-layer lists"
            )
        if "hot_slots" in entry and entry["hot_slots"] != sum(row_counts):
            raise ValueError(f"rank {rank}: hot_slots does not match actual lists")
        counts.append(row_counts)
    return counts


def expert_memory(
    data: dict, slots: int, cache_ranks: set[int], asynchronous: bool = False,
    *, catalog=None, runtime=None
) -> list[dict]:
    counts = validate_manifest(data)
    catalog = cost_catalog(data, catalog)
    identity = catalog_identity(catalog)
    if data.get("expert_cost_catalog") is not None and data["expert_cost_catalog"] != identity:
        raise ValueError("manifest expert cost catalog identity does not match supplied packed costs")
    limit = runtime_cache_limit(runtime)
    if type(slots) is not int or not 0 <= slots <= limit or not cache_ranks <= {0, 1}:
        raise ValueError(f"cache must use 0..{limit} slots and TP ranks 0,1")
    if slots > 128 and (slots not in (160, 192) or asynchronous):
        raise ValueError('Extended cache requires 160 or 192 synchronous slots')
    vectors = cost_vectors(catalog)
    result = []
    for rank, layers in enumerate(counts):
        costs = vectors[rank]
        hot = sum(n * cost for n, cost in zip(layers, costs))
        declared = data["ranks"][str(rank)].get("hot_bytes")
        if declared is not None and declared != hot:
            raise ValueError(
                f"rank {rank}: hot_bytes disagrees with the pinned target's packed tensor sizes"
            )
        physical = (
            (slots + int(asynchronous and slots > 0)) if rank in cache_ranks else 0
        )
        result.append(
            {
                "rank": rank,
                "hot_counts_by_layer": layers,
                "static_packed_bytes": hot,
                "cold_pinned_packed_bytes": sum(
                    (512 - n) * cost for n, cost in zip(layers, costs)
                ),
                "cache_physical_slots": physical,
                "cache_packed_bytes": physical * sum(costs),
                "full_pageable_master_bytes": 512 * sum(costs),
                "expert_cost_catalog": identity,
            }
        )
    return result
