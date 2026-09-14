# SPDX-License-Identifier: Apache-2.0
"""Cold owner backend selection; GPU byte/lifetime checks run separately."""
import ast
from pathlib import Path
from types import SimpleNamespace
import pytest

QUANT = Path(__file__).resolve().parents[1] / "runtimes/qwen38-flash-next-gfx1201-v1/retained-mtp4/python/vllm_gguf_plugin/quantization"

def extract(filename, name, scope):
    node = next(n for n in ast.parse((QUANT / filename).read_text()).body if getattr(n, "name", None) == name)
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    tree = ast.fix_missing_locations(ast.Module(body=[future, node], type_ignores=[]))
    exec(compile(tree, str(QUANT / filename), "exec"), scope)
    return scope[name]

@pytest.mark.parametrize("mode", ["default", "coherent", "noncoherent"])
@pytest.mark.parametrize("hip", [None, "7.14"])
def test_cold_owner_preserves_requested_policy_and_backend(mode, hip):
    calls = []
    def record(kind):
        def allocation(*args, **kwargs):
            calls.append((kind, args, kwargs))
            return "owner"
        return allocation
    fn = extract("params.py", "allocate_tiered_cold_host_empty", {
        "torch": SimpleNamespace(version=SimpleNamespace(hip=hip)),
        "_uva_host_coherence": lambda: mode,
        "_hip_host_empty": record("direct"),
        "_explicit_hip_uva_empty": record("explicit"),
        "allocate_uva_host_empty": record("legacy"),
    })
    assert fn((441, 832, 1100), "uint8") == "owner"
    if hip is None:
        assert calls == [("legacy", ((441, 832, 1100), "uint8"), {})]
    elif mode == "default":
        assert calls == [("direct", ((441, 832, 1100), "uint8"), {"flag": 0, "mode": "default"})]
    else:
        assert calls == [("explicit", ((441, 832, 1100), "uint8", mode), {})]

def test_invalid_policy_fails_before_allocating():
    def bad_policy():
        raise ValueError("invalid host policy")
    fn = extract("params.py", "allocate_tiered_cold_host_empty", {"_uva_host_coherence": bad_policy})
    with pytest.raises(ValueError, match="invalid host policy"):
        fn((1,), "uint8")

@pytest.mark.parametrize("streaming", [False, True])
def test_only_streaming_compaction_selects_exact_cold_allocator(streaming):
    legacy, exact = object(), object()
    class StopAfterChoice(Exception):
        pass
    chosen = []
    def compact(*args, **kwargs):
        chosen.append(kwargs["cold_empty"])
        raise StopAfterChoice
    fn = extract("tiered_experts.py", "_compact_expert_parameter", {
        "validate_expert_master": lambda *args: None,
        "is_tiered_expert_master": lambda p: True,
        "_compaction_device": lambda p: "cuda:0",
        "compact_expert_master": compact,
        "allocate_uva_host_empty": legacy,
        "allocate_tiered_cold_host_empty": exact,
        "_stream_compaction_enabled": lambda: streaming,
    })
    param = SimpleNamespace(_vllm_is_uva_offloaded=True, _vllm_uva_cpu_data=object())
    with pytest.raises(StopAfterChoice):
        fn(param, [0], 512)
    assert chosen == [exact if streaming else legacy]
