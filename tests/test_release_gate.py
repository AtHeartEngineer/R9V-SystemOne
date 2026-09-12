import pytest

from tools.release_gate import GATES, validate_receipt


def receipt():
    return {
        "schema": "r9v.release-qualification.v1",
        "image_id": "sha256:tested",
        "source_revision": "source",
        "gates": dict.fromkeys(GATES, True),
        "evidence": ["qualification"],
        "limitations": ["bounded local sessions"],
    }


def test_publishing_rejects_rebuilt_image_and_missing_gates():
    value = receipt()
    validate_receipt(value, "sha256:tested", "source")
    with pytest.raises(ValueError, match="exact image"):
        validate_receipt(value, "sha256:rebuilt", "source")
    value["gates"]["headroom"] = False
    with pytest.raises(ValueError, match="headroom"):
        validate_receipt(value, "sha256:tested", "source")
