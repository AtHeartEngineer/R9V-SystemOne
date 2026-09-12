import pytest

from tools.rank_experts import counts, rank_catalog


def test_complete_catalog_records_unseen_ties_without_inventing_scores():
    source = {
        "version": 1,
        "num_layers": 48,
        "num_experts": 512,
        "ranks": {
            str(r): {
                "hot_count": 2,
                "hot_experts_by_layer": [[0, 1] for _ in range(48)],
            }
            for r in range(2)
        },
    }
    train = {
        "schema": "r9v.routes.v1",
        "decode_counts": [[0, 1, 2] + [0] * 509 for _ in range(48)],
        "prefill_counts": [[0, 1, 2] + [0] * 509 for _ in range(48)],
    }
    catalog = rank_catalog(source, train, train)
    row = catalog["ranks"]["0"]["hot_experts_by_layer"][0]
    assert sorted(row) == list(range(512))
    assert row[:3] == [2, 1, 0]
    assert catalog["ranking"]["unobserved_ids_by_layer"][0] == [0] + list(range(3, 512))
    assert catalog["ranking"]["heldout_non_regression"]


def test_catalog_requires_actual_prefill_coverage():
    with pytest.raises(ValueError, match="observed routes"):
        counts(
            {
                "schema": "r9v.routes.v1",
                "decode_counts": [[1] * 512 for _ in range(48)],
                "prefill_counts": [[0] * 512 for _ in range(48)],
            }
        )


def test_catalog_extension_preserves_each_ranks_original_priorities():
    source = {
        "version": 1,
        "num_layers": 48,
        "num_experts": 512,
        "ranks": {
            str(r): {
                "hot_count": 2,
                "hot_experts_by_layer": [[r, r + 2] for _ in range(48)],
            }
            for r in range(2)
        },
    }
    histogram = {
        "schema": "r9v.routes.v1",
        "decode_counts": [[1, 3, 2] + [0] * 509 for _ in range(48)],
        "prefill_counts": [[1, 3, 2] + [0] * 509 for _ in range(48)],
    }
    result = rank_catalog(source, histogram, histogram, preserve_prefixes=True)
    for rank in range(2):
        row = result["ranks"][str(rank)]["hot_experts_by_layer"][0]
        assert row[:2] == [rank, rank + 2]
        assert sorted(row) == list(range(512))
    assert all(
        row["catalog_hit_fraction"] == row["source_hit_fraction"]
        for row in result["ranking"]["heldout_comparison"]
    )


    assert result["ranking"]["measured_cold_to_hot_by_layer"][0][-3:] == [0, 2, 1]
    assert result["ranking"]["phase_route_totals"]["train"]["decode_counts"] == 48 * 6

def test_catalog_rejects_captures_bound_to_different_model_configs():
    source = {
        "version": 1, "num_layers": 48, "num_experts": 512,
        "ranks": {str(r): {"hot_count": 1,
                           "hot_experts_by_layer": [[0] for _ in range(48)]}
                  for r in range(2)},
    }
    base = {"schema": "r9v.routes.v1", "metadata": {"model_hash": "a"},
            "decode_counts": [[1] * 512 for _ in range(48)],
            "prefill_counts": [[1] * 512 for _ in range(48)]}
    other = {**base, "metadata": {"model_hash": "b"}}
    with pytest.raises(ValueError, match="model/config identity"):
        rank_catalog(source, base, other)
