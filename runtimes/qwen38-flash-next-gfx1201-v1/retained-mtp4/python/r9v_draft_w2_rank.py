"""Draft-only W2 coarse ranking with exact indexed Q6_K reranking."""
from __future__ import annotations

import importlib.util
import os
from types import SimpleNamespace

import torch

_MODULE = None
_INDEXED = None


def module():
    global _MODULE
    if _MODULE is None:
        path = os.environ["R9V_DRAFT_W2_RANK_SO"]
        spec = importlib.util.spec_from_file_location("r9v_draft_w2_rank", path)
        assert spec is not None and spec.loader is not None
        _MODULE = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(_MODULE)
    return _MODULE


def indexed_module():
    global _INDEXED
    if _INDEXED is None:
        from r9v_draft_indexed import extension

        _INDEXED = extension()
    return _INDEXED


def pack_rows_w2(weight):
    """Pack a bounded BF16 chunk; codes map to (-3,-1,+1,+3)."""
    if weight.ndim != 2 or weight.dtype != torch.bfloat16 or weight.shape[1] % 128:
        raise ValueError("Expected BF16 rows with K divisible by128")
    n,k=weight.shape
    w=weight.float().reshape(n,k//128,128)
    bound=w.abs().amax(-1)
    best_error=torch.full_like(bound,float('inf'))
    chosen=torch.zeros_like(bound)
    for clip in (1.0,.9,.8,.7,.6,.5,.4):
        scale=(bound*(clip/3)).half().float().clamp_min(2**-24)
        q=torch.round((w/scale[...,None]+3)*.5).clamp(0,3)
        error=(w-(2*q-3)*scale[...,None]).square().sum(-1)
        better=error<best_error
        chosen=torch.where(better,scale,chosen)
        best_error=torch.minimum(best_error,error)
    chosen=torch.where(bound==0,torch.zeros_like(chosen),chosen)
    divisor=torch.where(chosen>0,chosen,torch.ones_like(chosen))
    q=torch.round((w/divisor[...,None]+3)*.5).clamp(0,3).to(torch.uint8).reshape(n,k)
    packed=q[:,0::4] | (q[:,1::4]<<2) | (q[:,2::4]<<4) | (q[:,3::4]<<6)
    return packed.contiguous(),chosen.half().contiguous()


def pack_head_w2(head):
    """Only chunk-sized BF16 temporaries; no full dequantized head copy."""
    from vllm_gguf_plugin import ops
    source=head.qweight
    if (source.dtype!=torch.uint8 or source.shape!=(124160,2100)
            or int(head.qweight_type.weight_type)!=14 or head.embedding_dim!=2560):
        raise ValueError("Expected the pinned original Q6_K head")
    n,k=source.shape[0],head.embedding_dim
    packed=torch.empty((n,k//4),dtype=torch.uint8,device=source.device)
    scales=torch.empty((n,k//128),dtype=torch.float16,device=source.device)
    for start in range(0,n,512):
        chunk=source[start:start+512].contiguous()
        bf16=ops.ggml_dequantize(chunk,14,k,chunk.shape[0],torch.bfloat16).reshape(chunk.shape[0],k)
        q,scale=pack_rows_w2(bf16)
        packed[start:start+len(chunk)]=q
        scales[start:start+len(chunk)]=scale
    return packed,scales


class W2RankMethod:
    def __init__(self, original, packed, scales, count):
        self.original = original
        self.packed = packed
        self.scales = scales
        self.count = count

    def apply(self, head, x, bias=None):
        if (x.ndim != 2 or x.shape[0] != 1 or x.dtype != torch.bfloat16 or
                not x.is_contiguous() or bias is not None):
            return self.original.quant_method.apply(self.original, x, bias=bias)
        coarse = module().project(self.packed, self.scales, x)
        ids = torch.topk(coarse, self.count, dim=-1, sorted=False).indices.reshape(-1)
        exact = indexed_module().indexed(self.original.qweight, x, ids)
        out = torch.full((1, self.original.qweight.shape[0]), -float("inf"),
                         dtype=exact.dtype, device=x.device)
        out.scatter_(1, ids.view(1, -1), exact)
        return out


def draft_head(model):
    count = int(os.environ.get("R9V_DRAFT_W2_RANK_ROWS", "0"))
    if count == 0:
        return model.lm_head
    if count not in (128, 512, 2048, 8192):
        raise ValueError("R9V_DRAFT_W2_RANK_ROWS must be 0, 128, 512, 2048, or 8192")
    cached = getattr(model, "_r9v_w2_rank_head", None)
    if cached is not None:
        if cached.weight is not model.lm_head.qweight or cached.quant_method.count != count:
            raise RuntimeError("W2 draft head changed after packing")
        return cached
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError("W2 draft head must be packed during eager warmup")
    original = model.lm_head
    if (model.logits_processor.head_dtype not in (None, torch.bfloat16) or
            model.logits_processor.soft_cap is not None or
            original.quant_method.layout is not None or original.qweight.dtype != torch.uint8 or not original.qweight.is_contiguous() or
            int(original.qweight_type.weight_type) != 14 or
            original.qweight.shape != (124160, 2100) or original.embedding_dim != 2560 or
            original.qweight.shard_id):
        raise ValueError("W2 draft requires the pinned local M1 Q6_K head")
    module(); indexed_module()
    packed, scales = pack_head_w2(original)
    cached = SimpleNamespace(weight=original.qweight, original_head=original,
                             tp_size=original.tp_size, shard_indices=original.shard_indices,
                             quant_method=W2RankMethod(original, packed, scales, count))
    model._r9v_w2_rank_head = cached
    print(f"[r9v] draft-only W2 coarse/Q6 exact rows={count}; target unchanged", flush=True)
    return cached
