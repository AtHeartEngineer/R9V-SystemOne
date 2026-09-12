# SPDX-License-Identifier: Apache-2.0
"""CPU regression for MTP-aware routing phase attribution."""
import ast
from pathlib import Path
from types import SimpleNamespace

import pytest

SOURCE = Path(__file__).resolve().parents[1] / "runtimes/qwen38-flash-next-gfx1201-v1/retained-mtp4/python/vllm_gguf_plugin/quantization/route_profile.py"


def classifier():
    # The classifier is pure Python; do not import the optional ROCm stack.
    module = ast.parse(SOURCE.read_text())
    function = next(n for n in module.body if isinstance(n, ast.FunctionDef)
                    and n.name == "_histogram_phase")
    namespace = {}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(SOURCE), "exec"), namespace)
    return namespace["_histogram_phase"]


def meta(prefill=0, decode=0, speculative=0, actual=None):
    return SimpleNamespace(num_prefill_tokens=prefill, num_decode_tokens=decode,
                           num_spec_decode_tokens=speculative,
                           num_actual_tokens=(prefill + decode + speculative if actual is None else actual))


@pytest.mark.parametrize("metadata,rows,expected", [
    (meta(speculative=5), 5, (0, 5)),
    (meta(speculative=6), 6, (0, 6)),
    (meta(decode=1), 1, (0, 1)),
    (meta(prefill=2), 2, (1, 2)),
    (meta(prefill=5), 5, (1, 5)),
    (meta(prefill=1024), 1024, (1, 1024)),
    (meta(speculative=5), 8, (0, 5)),
])
def test_actual_scheduled_phase_not_row_threshold(metadata, rows, expected):
    assert classifier()({"gdn0": metadata, "gdn1": metadata, "full_attention": object()}, rows) == expected


@pytest.mark.parametrize("metadata,rows", [
    ({}, 5), (None, 5),
    ({"gdn": meta(prefill=3, speculative=5)}, 8),
    ({"gdn": meta(speculative=5, actual=4)}, 5),
    ({"gdn": meta(speculative=5)}, 4),
    ({"gdn": meta(decode=-1, prefill=3)}, 2),
    ({"gdn": meta()}, 1),
    ({"gdn0": meta(speculative=5), "gdn1": meta(prefill=5)}, 5),
])
def test_unsupported_or_inconsistent_capture_fails_explicitly(metadata, rows):
    with pytest.raises(ValueError):
        classifier()(metadata, rows)
