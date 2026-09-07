# SPDX-License-Identifier: Apache-2.0
"""Validate placement contents and account for packed expert bytes, not total VRAM."""

from __future__ import annotations

import json
import math
from pathlib import Path

COST_PATH = (
    Path(__file__).resolve().parents[1]
    / "packages/placements/qwen38-flash-next/ud-iq4-xs/dual-r9700/expert-memory.json"
)


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
    data: dict, slots: int, cache_ranks: set[int], asynchronous: bool = False
) -> list[dict]:
    counts = validate_manifest(data)
    if type(slots) is not int or not 0 <= slots <= 16 or not cache_ranks <= {0, 1}:
        raise ValueError("cache must use 0..16 slots and TP ranks 0,1")
    catalog = json.loads(COST_PATH.read_text())
    costs = catalog["packed_bytes_per_expert_by_layer_per_rank"]
    if len(costs) != 48 or any(type(n) is not int or n <= 0 for n in costs):
        raise ValueError("invalid pinned expert byte catalog")
    result = []
    for rank, layers in enumerate(counts):
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
            }
        )
    return result
