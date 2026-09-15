# WMMA grouped MoE prefill — 2026-09-14

## Why prompt processing regressed and what changed

The v0.2.0 IQ4 placement keeps 71–76 static experts on rank 0 plus a 160-slot decode
cache, against 329 static experts in the August V1 placement. Prefill touches nearly
every expert in every layer, so each 1024-token chunk streamed roughly 33 GB of packed
expert weights over PCIe on rank 0 at link rate. That is the whole difference between
V1's 1,512 tok/s and v0.2.0's 989 tok/s at 8K; per-token compute was unchanged.

Larger chunks amortise that streaming, but the released grouped prefill kernel is a
dot-product GEMV that runs at about 9 TFLOPS with every expert resident, so above
4096-token chunks prefill became compute-bound at roughly 1,330 tok/s regardless of
placement. `r9v_moe_wmma` replaces it for prompt chunks: one workgroup per (expert
group of 32 routes, 128-row tile), each wave dequantizes IQ3_S/IQ4_XS/IQ4_NL/Q8_0 rows
straight into wave32 int8 WMMA operands, applies the per-32 weight and activation
scales in fp32, and stages packed rows through LDS with row-sequential 16-byte loads
issued one fetch ahead so pinned-host reads over PCIe stream near link rate. The
activation quantizer, the routing alignment, the decode GEMV kernels, the cache and the
placement are unchanged; only prompt chunks with more than 64 tokens take the new path.

The IQ4 profile now uses 4096-token chunks (`R9V_MAX_NUM_BATCHED_TOKENS=4096`) and
`R9V_TIERED_PREFILL_GROUP_SIZE=32`. Both are part of the memory-seed contract, so the
placement assets were re-derived for the new image with the release tools: route
captures, expert ranking, calibration, planner placement, derived 128K qualification,
memory seed and public export (`packages/placements/.../wmma-prefill-r1/qualified-128k-r8`).

## Kernel parity

`tests/gpu/wmma_prefill_parity.py` compares the new kernel with the released kernel on
synthetic packed experts for all four quant types the IQ4 package uses, with 67 and 512
of 512 experts resident and 1024/4096 tokens. Both kernels quantize activations
identically and accumulate int8 dot products with the same per-32 scales; only the fp32
summation order differs, and every case agrees within two bf16 ulps of the largest output
(observed maximum 1.2 ulp, mean absolute difference below 1e-4 of the output scale).
Timing on the same cases: 2.4–3.9× faster with experts resident at 4096 tokens.

Model-level check on the release-candidate image at 4096-token chunks versus the released
image at 1024-token chunks (same server, three 8,192-token slices of the pinned Aider
corpus, `prompt_logprobs`): mean prompt NLL 0.808 / 0.446 / 0.214 versus
0.806 / 0.447 / 0.245. Twelve short greedy generations: MTP acceptance 3.12 versus 3.10
tokens per step and 44.7 versus 43.9 ms per decode step. Decode kernels are unchanged;
greedy continuations differ between chunk settings on the released image as well, so
decode is compared per step rather than by generated text.

## Benchmarks

Measured on the dual Radeon AI PRO R9700 workstation through the ordinary product path:
a fresh `./r9v setup qwen38-mtp4 --headroom 3,3` and `./r9v start`, whose first-start
workload qualification passed on the planner's 54/428 static-expert placement (cache160/0,
4,167 / 4,362 MiB free at ready). The procedure is the original release's sustained PP
suite: one unmeasured 1,024-token warmup, then ten 8K / one-output requests (one-second
pauses), three 32K and two 64K / 16-output requests (two-second pauses), serial. Rates are
prompt tokens per second, `prompt_tokens / TTFT`; prefix-cache queries were zero.

| Runtime | 8K mean / median | 32K mean / median | 64K mean / median | All trials |
|---|---:|---:|---:|---|
| WMMA prefill, 4096-token chunks (this branch) | **1,669.6 / 1,670.7** | **1,617.1 / 1,626.9** | **1,601.4 / 1,601.4** | [JSON](results/iq4-wmma-prefill-20260914.json) |
| v0.2.0 IQ4, 1024-token chunks | 989.3 / 984.6 | 982.0 / 984.7 | 968.2 / 968.2 | [JSON](results/iq4-v020-pp-20260914.json) |
| Original IQ4 MTP2 reference (August) | 1,512.0 / 1,510.2 | 1,401.8 / 1,365.3 | 1,357.0 / 1,357.0 | [JSON](results/qwen38-group16-pp-v1.json) |

8K trials ranged 1,637–1,726; 32K 1,592–1,632; 64K 1,599–1,604. The 8K and 32K
results are +69% and +65% over v0.2.0 and above the August reference at every length.

Generation on the same server afterwards: the 256-token reference prompt decoded at
77.2 and 78.7 tok/s (38.6–39.3 ms per step, 3.04 accepted tokens per step); twelve
128-token greedy generations averaged 71.3 tok/s at 43.3 ms per step and 3.08 tokens
per step, against 70.7 tok/s, 43.9 ms and 3.10 on the released image in the same harness.
Decode kernels and placement policy are unchanged; the reference prompt's lower single
number is a different greedy continuation with lower MTP acceptance at 4096-token chunks,
not a slower step. Generation is reported per step for that reason; it is not a new
throughput qualification.

## Reproduce

Same procedure as [v0.2.0](v020-prefill-20260914.md), from this branch's checkout with
the `qwen38-mtp4` profile started through `./r9v setup` / `./r9v start`.
