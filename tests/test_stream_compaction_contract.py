# SPDX-License-Identifier: Apache-2.0
"""CPU behavior checks for streaming copies, including the real loader hook order."""
import ast
import itertools
from pathlib import Path
from types import SimpleNamespace
import weakref

import pytest

ROOT = Path(__file__).resolve().parents[1]
QUANT = ROOT / "runtimes/qwen38-flash-next-gfx1201-v1/retained-mtp4/python/vllm_gguf_plugin/quantization"
HOOK_ATTR = "_r9v_tiered_stream_copy_hook"


def extract(path, name, namespace):
    tree = ast.parse(path.read_text())
    node = next(n for n in tree.body if getattr(n, "name", None) == name)
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    module = ast.fix_missing_locations(ast.Module(body=[future, node], type_ignores=[]))
    exec(compile(module, str(path), "exec"), namespace)
    return namespace[name]


class Parameter:
    master = True


class Module:
    def __init__(self):
        self.w13_qweight = Parameter()
        self.w2_qweight = Parameter()
        self._gguf_expert_partition = (0, 416, 640)


class Model:
    _r9v_stream_manifest = {"version": 1}


class Weight:
    ndim = 3
    shape = (512, 20, 8)


def fixture(fail_compaction=False):
    calls = []

    def compact(model, **kwargs):
        calls.append(kwargs)
        if fail_compaction:
            raise RuntimeError("copy/allocation failed")
        module = kwargs["only_module"]
        module.w13_qweight.master = False
        module.w2_qweight.master = False

    hook_class = extract(QUANT / "tiered_experts.py", "_StreamCompactionHook", {
        "weakref": weakref, "is_tiered_expert_master": lambda p: p.master,
        "materialize_hot_expert_cache": compact,
    })
    model, module = Model(), Module()
    hook = hook_class(model, module, 0, [1], 512, "manifest")
    for p in (module.w13_qweight, module.w2_qweight):
        setattr(p, HOOK_ATTR, hook)
    return model, module, hook, calls


def parameter(module, shard):
    return module.w2_qweight if shard == "w2" else module.w13_qweight


@pytest.mark.parametrize("order", list(itertools.permutations(("w1", "w2", "w3"))))
def test_all_copy_orders_finalize_once_after_the_last_completed_copy(order):
    model, module, hook, calls = fixture()
    for index, shard in enumerate(order):
        p = parameter(module, shard)
        hook.before_copy(module, p, shard, Weight())
        assert len(calls) == 0
        hook.after_copy(module, p, shard, Weight())
        assert len(calls) == int(index == 2)
    hook.require_complete()
    assert not module.w13_qweight.master and not module.w2_qweight.master
    assert hook.model_ref() is model


def test_incomplete_or_failed_compaction_cannot_finalize():
    model, module, hook, calls = fixture(fail_compaction=True)
    for shard in ("w1", "w2"):
        hook.after_copy(module, parameter(module, shard), shard, Weight())
    with pytest.raises(RuntimeError, match="incomplete"):
        hook.require_complete()
    with pytest.raises(RuntimeError, match="allocation failed"):
        hook.after_copy(module, module.w13_qweight, "w3", Weight())
    assert len(calls) == 1 and not hook.finalized
    with pytest.raises(RuntimeError, match="incomplete"):
        hook.require_complete()
    assert model is not None


@pytest.mark.parametrize("bad", ["duplicate", "late", "partial", "wrong_parameter", "wrong_module", "missing_partition", "wrong_shard"])
def test_invalid_write_rejected_before_materialization(bad):
    model, module, hook, calls = fixture()
    shard, p, weight, target = "w1", module.w13_qweight, Weight(), module
    if bad == "duplicate":
        hook.after_copy(module, p, shard, weight)
    elif bad == "late":
        for s in ("w1", "w2", "w3"):
            hook.after_copy(module, parameter(module, s), s, weight)
    elif bad == "partial":
        weight = SimpleNamespace(ndim=3, shape=(511, 20, 8))
    elif bad == "wrong_parameter":
        p = module.w2_qweight
    elif bad == "wrong_module":
        target = Module()
    elif bad == "missing_partition":
        module._gguf_expert_partition = None
    else:
        shard = "other"
    touched = []
    loader = extract(QUANT / "params.py", "_gguf_moe_weight_loader", {
        "_TIERED_STREAM_COPY_HOOK_ATTR": HOOK_ATTR,
        "_materialize_gguf_moe_param": lambda *a: touched.append(True),
    })
    with pytest.raises(RuntimeError):
        loader(target, None, p, weight, "weight", shard, 0)
    assert touched == []
    assert model is not None


class CopyTensor:
    dtype = "uint8"

    def __init__(self, shape, state):
        self.shape, self.state = shape, state
        self.ndim = len(shape)

    def narrow(self, dim, start, width):
        shape = list(self.shape)
        assert 0 <= start <= start + width <= shape[dim]
        shape[dim] = width
        return CopyTensor(tuple(shape), self.state)

    def copy_(self, source):
        assert self.shape == source.shape
        if self.state["fail"]:
            raise RuntimeError("actual copy failed")
        self.state["copies"] += 1


def test_actual_loader_does_not_count_failed_copy_and_can_retry():
    model, module, hook, calls = fixture()
    state = {"fail": True, "copies": 0}
    p = module.w13_qweight
    p.data = CopyTensor((512, 26, 8), state)
    module.local_num_experts = 512
    source = CopyTensor((512, 20, 8), state)
    loader = extract(QUANT / "params.py", "_gguf_moe_weight_loader", {
        "_TIERED_STREAM_COPY_HOOK_ATTR": HOOK_ATTR,
        "_materialize_gguf_moe_param": lambda *a: None,
        "torch": SimpleNamespace(uint8="uint8"),
        "packed_partition_bounds": lambda size, part: (size * part[0] // part[2], size * part[1] // part[2]),
    })
    with pytest.raises(RuntimeError, match="actual copy failed"):
        loader(module, None, p, source, "weight", "w1", 0)
    assert hook.seen == set() and calls == []
    state["fail"] = False
    assert loader(module, None, p, source, "weight", "w1", 0, True) is True
    assert hook.seen == {"w1"} and state["copies"] == 1
    assert model is not None


@pytest.mark.parametrize("last_expert", [0, 511, 137])
def test_per_expert_copies_require_every_expert_in_every_projection(last_expert):
    model, module, hook, calls = fixture()
    weight = SimpleNamespace(ndim=2, shape=(20, 8))
    order = [i for i in reversed(range(512)) if i != last_expert]
    for shard in ("w3", "w1", "w2"):
        for expert in order + ([last_expert] if shard != "w2" else []):
            p = parameter(module, shard)
            hook.before_copy(module, p, shard, weight, expert)
            hook.after_copy(module, p, shard, weight, expert)
    assert calls == []
    with pytest.raises(RuntimeError, match="incomplete"):
        hook.require_complete()
    hook.after_copy(module, module.w2_qweight, "w2", weight, last_expert)
    assert len(calls) == 1
    hook.require_complete()
    assert model is not None


def test_failed_individual_copy_and_mixed_bulk_overlap_cannot_complete():
    model, module, hook, calls = fixture()
    weight = SimpleNamespace(ndim=2, shape=(20, 8))
    hook.before_copy(module, module.w13_qweight, "w1", weight, 42)
    assert hook.loaded_experts["w1"] == set()
    hook.after_copy(module, module.w13_qweight, "w1", weight, 42)
    with pytest.raises(RuntimeError, match="Duplicate"):
        hook.before_copy(module, module.w13_qweight, "w1", weight, 42)
    with pytest.raises(RuntimeError, match="Duplicate"):
        hook.before_copy(module, module.w13_qweight, "w1", Weight(), 0)
    assert hook.seen == set() and calls == [] and model is not None


@pytest.mark.parametrize("expert_id", [-1, 512, None, True, 2.0])
def test_individual_copy_requires_exact_in_range_expert_id(expert_id):
    model, module, hook, calls = fixture()
    with pytest.raises(RuntimeError, match="valid ID"):
        hook.before_copy(module, module.w13_qweight, "w1", SimpleNamespace(ndim=2, shape=(20, 8)), expert_id)
    assert hook.seen == set() and calls == [] and model is not None
