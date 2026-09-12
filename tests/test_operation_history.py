import pytest

from tools.operation_history import analyze


def record(sequence, stage, stamp=100):
    return {
        "container": "r9v-fixture",
        "rank": 0,
        "pid": 603,
        "start_ticks": "12",
        "thread_id": 603,
        "steps": 1,
        "sequence": sequence,
        "stage": stage,
        "monotonic_ns": stamp,
    }


def test_missing_record_cannot_be_paired_as_a_long_operation():
    result = analyze(
        [record(1, "execute_model_enter"), record(4, "execute_model_return", 1000)]
    )
    assert result["missing"] == 2
    assert result["pairs"] == []


def test_contiguous_pair_measures_host_boundary():
    result = analyze(
        [record(1, "execute_model_enter"), record(2, "execute_model_return", 1_000_100)]
    )
    assert result["pairs"][0]["host_ms"] == 1
    assert result["missing"] == 0


def test_mixed_workers_rejected():
    with pytest.raises(ValueError):
        analyze(
            [
                record(1, "execute_model_enter"),
                {**record(2, "execute_model_return"), "rank": 1},
            ]
        )


def test_replayed_record_reported_and_different_step_not_paired():
    first = record(1, "execute_model_enter")
    result = analyze([first, first, {**record(2, "execute_model_return"), "steps": 2}])
    assert result["duplicates"] == 1
    assert result["pairs"] == []
