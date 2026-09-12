"""Experimental draft-only coarse-to-exact Q6_K vocabulary search.

Target weights and full-vocabulary target verification are untouched.
Coarse ranks are an untrained heuristic; selected logits use exact Q6_K math.
"""
import importlib.util, logging, os
from types import SimpleNamespace
import torch

_EXTENSION = None

def extension():
 global _EXTENSION
 if _EXTENSION is None:
  path=os.environ['R9V_DRAFT_INDEXED_SO']
  spec=importlib.util.spec_from_file_location('r9v_draft_indexed_q6',path)
  assert spec is not None and spec.loader is not None
  _EXTENSION=importlib.util.module_from_spec(spec);spec.loader.exec_module(_EXTENSION)
 return _EXTENSION

class IndexedMethod:
 def __init__(self,original,count,blocks):
  self.original=original;self.count=count;self.blocks=blocks;self.ext=extension()
 def apply(self,head,x,bias=None):
  if x.ndim!=2 or x.shape[0]!=1 or x.dtype!=torch.bfloat16 or not x.is_contiguous() or bias is not None:
   return self.original.quant_method.apply(self.original,x,bias=bias)
  coarse=self.ext.coarse(self.original.qweight,x,*self.blocks)
  ids=torch.topk(coarse,self.count,dim=-1,sorted=False).indices.reshape(-1)
  small=self.ext.indexed(self.original.qweight,x,ids)
  out=torch.full((1,self.original.qweight.shape[0]),-float('inf'),dtype=small.dtype,device=x.device)
  out.scatter_(1,ids.view(1,-1),small)
  return out

def draft_head(model):
 count=int(os.environ.get('R9V_DRAFT_INDEXED_ROWS','0'))
 if count==0:return model.lm_head
 assert count in [4096,8192]
 blocks=tuple(int(v) for v in os.environ.get('R9V_DRAFT_COARSE_BLOCKS','0,5').split(','))
 assert len(blocks)==2 and len(set(blocks))==2 and all(0<=b<10 for b in blocks)
 cached=getattr(model,'_r9v_indexed_head',None)
 if cached is not None:
  assert cached.weight is model.lm_head.qweight and cached.quant_method.count==count and cached.quant_method.blocks==blocks
  return cached
 assert not torch.cuda.is_current_stream_capturing(),'Initialize indexed draft head in eager warmup'
 original=model.lm_head
 assert original.qweight.dtype==torch.uint8 and original.qweight.is_contiguous()
 assert int(original.qweight_type.weight_type)==14 and original.embedding_dim==2560
 assert original.qweight.shape==(124160,2100) and not original.qweight.shard_id
 assert original.quant_method.layout is None
 assert model.logits_processor.head_dtype in [None,torch.bfloat16] and model.logits_processor.soft_cap is None
 method=IndexedMethod(original,count,blocks)
 cached=SimpleNamespace(weight=original.qweight,original_head=original,tp_size=original.tp_size,shard_indices=original.shard_indices,quant_method=method)
 model._r9v_indexed_head=cached
 print(f'[r9v] draft-only indexed Q6_K head rows={count}/124160 blocks={blocks}; target unchanged',flush=True)
 return cached
