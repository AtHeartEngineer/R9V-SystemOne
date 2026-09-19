"""Candidate-score normalization and optional temperature lookup."""

import math
from collections.abc import Mapping
from numbers import Real
from pathlib import Path

from .models import CalibrationFile


def _is_finite_real(value: object) -> bool:
    if isinstance(value, bool) or not isinstance(value, Real):
        return False
    try:
        return math.isfinite(value)
    except (OverflowError, TypeError, ValueError):
        return False


def normalize_logprobs(
    logprobs: Mapping[str, float], temperature: float = 1.0
) -> dict[str, float]:
    """Normalize supplied candidate log probabilities after temperature scaling."""

    if not logprobs:
        raise ValueError("logprobs must contain at least one candidate")
    if not _is_finite_real(temperature) or temperature <= 0:
        raise ValueError("temperature must be a finite positive real number")
    if any(not _is_finite_real(value) for value in logprobs.values()):
        raise ValueError("logprobs values must be finite real numbers")

    peak = max(logprobs.values())
    weights = {
        key: math.exp((value - peak) / temperature)
        for key, value in logprobs.items()
    }
    total = math.fsum(weights.values())
    return {key: value / total for key, value in weights.items()}


class CalibrationStore:
    """Load optional versioned per-family temperatures from a JSON file."""

    def __init__(self, path: Path | None = None) -> None:
        self._calibration = (
            None
            if path is None
            else CalibrationFile.model_validate_json(path.read_text(encoding="utf-8"))
        )

    def temperature_for(self, family: str | None) -> tuple[float, bool]:
        if self._calibration is None or family is None:
            return 1.0, False
        entry = self._calibration.families.get(family)
        if entry is None:
            return 1.0, False
        return entry.temperature, True
