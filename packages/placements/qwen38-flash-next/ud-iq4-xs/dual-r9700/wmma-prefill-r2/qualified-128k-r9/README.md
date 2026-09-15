# WMMA prefill IQ4 reference placement

Reference for the gfx12 int8 WMMA grouped MoE prefill runtime at 4096-token prompt
chunks. The measured 46/434 static-expert placement with cache160/0 passed all
seven workload checks at 131,072 context, including a 130,941-token
prompt, with at least 3 GiB free on each GPU (observed minimum
3,974,066,176 / 3,786,801,152 bytes).

The public seed provides numeric measurements and commitments to independently preserved
private evidence. Only provenance locations and the catalog link are projected; model
tensors, runtime settings, expert ordering, counts, and memory arithmetic are unchanged.
Every local placement must qualify through setup/start before its receipt is reusable.

Route captures, ranking, calibration and the derived qualification all ran on image
`sha256:2dac17a215fb5b0e3461e4c3e36a2981eec8ac3d6021e73183d247e819740c03` with `R9V_MAX_NUM_BATCHED_TOKENS=4096` and
`R9V_TIERED_PREFILL_GROUP_SIZE=32`; see [the qualification note](../../../../../../../docs/qualification/wmma-prefill-20260914.md).
