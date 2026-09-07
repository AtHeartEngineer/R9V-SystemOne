#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Derive a smaller experimental placement from an existing ordered hot map."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

try:
    from tools.expert_budget import expert_memory, validate_manifest
except ModuleNotFoundError:
    from expert_budget import expert_memory, validate_manifest


def trim(source: dict, counts: list[int], source_hash: str) -> tuple[dict, dict]:
    old = validate_manifest(source)
    if len(counts) != 2 or any(
        type(n) is not int or n < 1 or n > min(old[r]) for r, n in enumerate(counts)
    ):
        raise ValueError(
            "choose two positive hot counts no larger than the smallest source layer on each rank; truncated maps cannot be expanded"
        )
    before = expert_memory(source, 0, set())
    result = {key: source[key] for key in ("version", "num_layers", "num_experts")}
    result.update({"top_k": 10, "placement": "uniform", "ranks": {}})
    for rank, count in enumerate(counts):
        result["ranks"][str(rank)] = {
            "hot_count": count,
            "hot_slots": count * 48,
            "hot_experts_by_layer": [
                ids[:count]
                for ids in source["ranks"][str(rank)]["hot_experts_by_layer"]
            ],
        }
    after = expert_memory(result, 0, set())
    deltas = []
    for rank in range(2):
        result["ranks"][str(rank)]["hot_bytes"] = after[rank]["static_packed_bytes"]
        removed = (
            before[rank]["static_packed_bytes"] - after[rank]["static_packed_bytes"]
        )
        deltas.append(
            {
                "rank": rank,
                "gpu_packed_bytes_released": removed,
                "host_pinned_bytes_added": removed,
            }
        )
    provenance = {
        "source_sha256": source_hash,
        "method": "preserve source priority prefixes",
        "qualification": "unqualified; routing hit rates and total VRAM headroom must be measured",
        "deltas": deltas,
        "cache": "unchanged; cache slots are configured separately at launch",
    }
    # Source holdout/route statistics no longer describe this placement.
    result["r9v_derivation"] = provenance
    return result, provenance


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path)
    parser.add_argument(
        "--hot-counts",
        required=True,
        help="per-layer counts in TP-rank order, e.g. 329,329",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="new manifest outside the immutable model package",
    )
    args = parser.parse_args(argv)
    try:
        source = args.source
        if source is None:
            override = os.environ.get("R9V_EXPERT_MANIFEST_PATH")
            if override:
                source = Path(override)
            else:
                model = os.environ.get("R9V_MODEL_DIR")
                if not model:
                    raise ValueError("provide --source or R9V_MODEL_DIR")
                source = Path(model) / os.environ.get(
                    "R9V_MANIFEST_REL",
                    "manifests/hot-manifest-q4-vision-128k-multiprompt-r1-lru16-neutral.json",
                )
        raw = source.read_bytes()
        result, provenance = trim(
            json.loads(raw),
            [int(n) for n in args.hot_counts.split(",")],
            hashlib.sha256(raw).hexdigest(),
        )
        fd = os.open(args.output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "w") as output:
            json.dump(result, output, indent=2)
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())
    except (OSError, ValueError, TypeError, KeyError) as error:
        print(f"Placement not generated: {error}")
        return 1
    print(json.dumps(provenance, indent=2))
    print(f"Experimental placement: {args.output.resolve()}")
    print(
        "Set R9V_EXPERT_MANIFEST_PATH to this absolute path, rerun doctor, then qualify the new placement."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
