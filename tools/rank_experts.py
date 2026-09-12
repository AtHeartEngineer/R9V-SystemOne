#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Build a complete expert catalog with explicit ties and held-out comparison."""

import argparse
import hashlib
import json
from pathlib import Path

try:
    from tools.expert_budget import catalog_identity, cost_catalog, validate_manifest
except ModuleNotFoundError:
    from expert_budget import catalog_identity, cost_catalog, validate_manifest


def counts(histogram):
    if histogram.get("schema") != "r9v.routes.v1":
        raise ValueError("unsupported route histogram")
    phases = [histogram.get(k) for k in ("decode_counts", "prefill_counts")]
    for phase in phases:
        if not isinstance(phase, list) or len(phase) != 48:
            raise ValueError("every phase must contain all 48 layers")
        for row in phase:
            if (
                not isinstance(row, list)
                or len(row) != 512
                or any(type(n) is not int or n < 0 for n in row)
                or sum(row) == 0
            ):
                raise ValueError(
                    "each phase/layer needs 512 nonnegative counts and observed routes"
                )
    return [[a + b for a, b in zip(*rows)] for rows in zip(*phases)]


def _capture_binding(histogram):
    """Extract model/config identity fields that must agree across captures."""
    metadata = histogram.get("metadata", {})
    if not isinstance(metadata, dict):
        raise ValueError("route histogram metadata must be an object")
    binding = {}
    for key in ("model", "model_hash", "model_package", "config", "config_hash",
                "runtime", "runtime_hash", "served_model"):
        value = histogram.get(key, metadata.get(key))
        if value is not None:
            binding[key] = value
    return binding


def _check_capture_binding(train, holdout):
    left, right = _capture_binding(train), _capture_binding(holdout)
    mismatches = {key: (left[key], right[key]) for key in left.keys() & right.keys()
                  if left[key] != right[key]}
    if mismatches:
        raise ValueError(f"training and held-out captures have different model/config identity: {mismatches}")
    # A field present in only one capture is useful evidence but cannot bind the
    # pair; retain it for the generated catalog so the gap is reviewable.
    return {**left, **right}, sorted(set(left) ^ set(right))


def rank_catalog(source, train, holdout, *, preserve_prefixes=False, catalog=None):
    old = validate_manifest(source)
    source_identity = source.get("expert_cost_catalog")
    resolved_catalog = cost_catalog(source, catalog)
    selected_identity = catalog_identity(resolved_catalog)
    if source_identity is not None and source_identity != selected_identity:
        raise ValueError("source manifest expert cost catalog identity does not match ranking catalog")
    capture_binding, unpaired_binding_fields = _check_capture_binding(train, holdout)
    for capture in (train, holdout):
        package = _capture_binding(capture).get('model_package')
        if package is not None and package != resolved_catalog['model_package']:
            raise ValueError('Route capture model package differs from the expert cost catalog')
    binding_verified = (not unpaired_binding_fields and all(
        key in _capture_binding(train) for key in ('model_package', 'model_hash', 'runtime_hash', 'config_hash')))
    train_counts, held_counts = counts(train), counts(holdout)
    ranked = [
        sorted(range(512), key=lambda expert: (-row[expert], expert))
        for row in train_counts
    ]
    ranked_by_rank = [ranked, ranked]
    if preserve_prefixes:
        ranked_by_rank = [
            [
                prefix + [i for i in ranked[layer] if i not in set(prefix)]
                for layer, prefix in enumerate(
                    source["ranks"][str(rank)]["hot_experts_by_layer"]
                )
            ]
            for rank in range(2)
        ]
    result = {
        "version": 1,
        "num_layers": 48,
        "num_experts": 512,
        "top_k": 10,
        "ranks": {
            str(rank): {"hot_count": 512, "hot_experts_by_layer": ranked_by_rank[rank]}
            for rank in range(2)
        },
        "expert_cost_catalog": selected_identity,
        "expert_memory": resolved_catalog,
    }
    comparisons = []
    for rank in range(2):
        for phase in ("decode_counts", "prefill_counts"):
            total = sum(map(sum, holdout[phase]))
            before = sum(
                sum(
                    row[i]
                    for i in source["ranks"][str(rank)]["hot_experts_by_layer"][layer]
                )
                for layer, row in enumerate(holdout[phase])
            )
            after = sum(
                sum(row[i] for i in ranked_by_rank[rank][layer][: old[rank][layer]])
                for layer, row in enumerate(holdout[phase])
            )
            comparisons.append(
                {
                    "rank": rank,
                    "phase": phase,
                    "source_hit_fraction": before / total,
                    "catalog_hit_fraction": after / total,
                }
            )
    result["ranking"] = {
        "binding_verified": binding_verified,
        "method": (
            "preserve source prefixes; rank remaining IDs by measured route count"
            if preserve_prefixes
            else "descending measured route count"
        )
        + "; expert ID breaks ties without implying benefit",
        "preserved_source_prefixes": preserve_prefixes,
        "training_counts": train_counts,
        # Keep measured heat separate from any preserved placement priorities.
        "measured_cold_to_hot_by_layer": [
            sorted(range(512), key=lambda expert: (row[expert], expert))
            for row in train_counts
        ],
        "phase_route_totals": {
            split: {phase: sum(map(sum, capture[phase]))
                    for phase in ("decode_counts", "prefill_counts")}
            for split, capture in (("train", train), ("holdout", holdout))
        },
        "heldout_counts": held_counts,
        "unobserved_ids_by_layer": [
            [i for i, n in enumerate(row) if n == 0] for row in train_counts
        ],
        "heldout_comparison": comparisons,
        "heldout_non_regression": all(
            c["catalog_hit_fraction"] >= c["source_hit_fraction"] for c in comparisons
        ),
        "qualification": "catalog only; a derived placement still needs throughput and memory qualification",
        "packed_cost_catalog": resolved_catalog,
        "capture_binding": capture_binding,
        "unpaired_binding_fields": unpaired_binding_fields,
        "measurement_scope": "all 512 expert IDs per layer",
        "ranking_semantics": "fully measured route counts; zero-count IDs are deterministic ties",
    }
    if preserve_prefixes:
        result["ranking"]["ranking_semantics"] = (
            "source hot prefixes are preserved; the remaining IDs are fully measured and sorted, "
            "with zero-count IDs deterministic ties"
        )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "train", "holdout", "output"):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--require-identity", action="store_true", help="require matching model/package/runtime/workload fingerprints in both captures")
    parser.add_argument(
        "--preserve-prefixes",
        action="store_true",
        help="extend the existing priorities to 512 IDs without reranking their qualified prefixes",
    )
    parser.add_argument("--expert-catalog", type=Path)
    args = parser.parse_args()
    try:
        paths = [args.source, args.train, args.holdout]
        raw = [p.read_bytes() for p in paths]
        if hashlib.sha256(raw[1]).digest() == hashlib.sha256(raw[2]).digest():
            raise ValueError("training and held-out captures must be independent")
        result = rank_catalog(
            *(json.loads(data) for data in raw),
            preserve_prefixes=args.preserve_prefixes,
            catalog=args.expert_catalog,
        )
        if args.require_identity and not result['ranking']['binding_verified']:
            raise ValueError('Both captures need matching model/package/runtime/workload identities')
        result["ranking"]["inputs"] = [
            {"path": str(p), "sha256": hashlib.sha256(data).hexdigest()}
            for p, data in zip(paths, raw)
        ]
        with args.output.open("x") as stream:
            json.dump(result, stream, indent=2)
            stream.write("\n")
        print(json.dumps(result["ranking"]["heldout_comparison"], indent=2))
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(f"Catalog not generated: {error}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
