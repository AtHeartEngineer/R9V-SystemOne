# SPDX-License-Identifier: Apache-2.0
"""Explicit intermediate-channel partitions for tiered Qwen target experts."""
from __future__ import annotations

import os
import re


def configure_expert_partition(layer, hidden_size: int, num_experts: int):
    raw = os.environ.get("QWEN38_EXPERT_TP_SPLIT")
    if raw is None:
        return None
    match = re.search(r"(?:^|\.)layers\.(\d+)\.mlp\.experts$", layer.layer_name)
    if match is None or not 0 <= int(match.group(1)) < 48:
        return None
    config = layer.moe_config
    parallel = config.moe_parallel_config
    if (hidden_size, num_experts, config.intermediate_size) != (2560, 512, 640):
        raise ValueError("Unequal expert partition requires the Qwen3.8 target layout")
    if parallel.tp_size != 2 or parallel.ep_size != 1 or parallel.enable_eplb:
        raise ValueError("Unequal expert partition requires TP2 without EP/EPLB")
    widths = tuple(int(x.strip()) for x in raw.split(","))
    if (
        len(widths) != 2
        or sum(widths) != 640
        or any(x not in (192, 224, 320, 416, 448) for x in widths)
    ):
        raise ValueError(
            "QWEN38_EXPERT_TP_SPLIT needs supported widths "
            "(192, 224, 320, 416, 448) summing to 640"
        )
    rank = config.tp_rank
    if rank not in (0, 1):
        raise ValueError("Invalid expert TP rank")
    width = widths[rank]
    partition = (sum(widths[:rank]), width, 640)
    layer._gguf_expert_partition = partition
    layer.intermediate_size_per_partition = width
    config.intermediate_size_per_partition = width
    return partition


def packed_partition_bounds(full_size: int, partition):
    offset, width, total = partition
    if full_size * offset % total or full_size * width % total:
        raise ValueError("Expert partition cuts a packed weight block")
    return full_size * offset // total, full_size * width // total
