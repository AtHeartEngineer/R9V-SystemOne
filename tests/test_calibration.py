import math

import pytest
from pydantic import ValidationError

from r9v_systemone.calibration import CalibrationStore, normalize_logprobs


def test_normalizes_only_supplied_candidates():
    actual = normalize_logprobs({"a": -10.0, "b": -11.0})

    assert actual["a"] == pytest.approx(0.73105858)
    assert actual["b"] == pytest.approx(0.26894142)
    assert sum(actual.values()) == pytest.approx(1.0)


def test_extreme_logprobs_produce_a_finite_distribution():
    actual = normalize_logprobs({"likely": -10_000.0, "unlikely": -11_000.0})

    assert actual == {"likely": 1.0, "unlikely": 0.0}
    assert all(math.isfinite(value) for value in actual.values())
    assert math.fsum(actual.values()) == pytest.approx(1.0)


def test_normalization_preserves_candidate_insertion_order():
    actual = normalize_logprobs({"third": -3.0, "first": -1.0, "second": -2.0})

    assert list(actual) == ["third", "first", "second"]


def test_temperature_changes_sharpness():
    raw = {"a": -1.0, "b": -2.0}

    assert normalize_logprobs(raw, 0.5)["a"] > normalize_logprobs(raw, 2.0)["a"]


def test_normalization_rejects_empty_candidates():
    with pytest.raises(ValueError, match="at least one candidate"):
        normalize_logprobs({})


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf, True, False])
def test_normalization_rejects_invalid_logprobs(value):
    with pytest.raises(ValueError, match="finite real number"):
        normalize_logprobs({"candidate": value})


@pytest.mark.parametrize(
    "temperature", [0.0, -1.0, math.nan, math.inf, -math.inf, True, False]
)
def test_normalization_rejects_invalid_temperature(temperature):
    with pytest.raises(ValueError, match="finite positive real number"):
        normalize_logprobs({"candidate": -1.0}, temperature)


def test_unconfigured_store_uses_uncalibrated_temperature():
    store = CalibrationStore(None)

    assert store.temperature_for(None) == (1.0, False)
    assert store.temperature_for("occupancy") == (1.0, False)


def test_configured_store_returns_family_temperature(tmp_path):
    path = tmp_path / "calibration.json"
    path.write_text(
        '{"version": 1, "families": {"occupancy": {"temperature": 1.25}}}',
        encoding="utf-8",
    )
    store = CalibrationStore(path)

    assert store.temperature_for("occupancy") == (1.25, True)
    assert store.temperature_for("lighting") == (1.0, False)
    assert store.temperature_for(None) == (1.0, False)


def test_configured_store_validates_the_versioned_model(tmp_path):
    path = tmp_path / "calibration.json"
    path.write_text(
        '{"version": 2, "families": {"occupancy": {"temperature": 1.25}}}',
        encoding="utf-8",
    )

    with pytest.raises(ValidationError):
        CalibrationStore(path)
