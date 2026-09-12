import pytest

from tools.pressure_runtime import reserves


@pytest.mark.parametrize("value", [[-1, 0], [0, 5], [float("nan"), 0], [True, 0], [0]])
def test_pressure_probe_rejects_unsafe_reservations(value):
    with pytest.raises(ValueError):
        reserves(value)


def test_pressure_probe_uses_explicit_binary_gib():
    assert reserves([0, 1.5]) == [0, 1610612736]
