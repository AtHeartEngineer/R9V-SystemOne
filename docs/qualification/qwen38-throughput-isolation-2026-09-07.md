# Qwen3.8 Flash Next throughput and VRAM isolation — 2026-09-07

The experiment measured the effect of trimming either card's expert placement
and found a concrete synchronization hotspot. It did **not** establish the
complete cause of the roughly 55 versus 78 token/s variation. No placement,
ROCm version, or production runtime change is promoted by these results.

## Results

All current-image arms used the same weights, MTP2, 16-slot synchronous LRU
cache on rank 1, fixed 2,285,670,400-byte KV budget per rank, 131,072 configured
maximum context, 1 concurrent request, and 1,024-token prefill batches.
Rank 0 is the display card; rank 1 is headless. The complete PCIe paths are
Gen5 x16 and a Gen4 x4 bottleneck, respectively.

| Arm | Expert counts | Historical prompt TG256 median | Short-prompt TG512 median | Sampled free VRAM rank 0 / rank 1 |
|---|---:|---:|---:|---:|
| Current, timing off | 329 / 369 | 54.41 | 54.71 | 1.85 / 0.87 GiB |
| Current, timing on | 329 / 369 | 54.59 initially | 77.47, after a mid-run speed change | 1.88 / 0.87 GiB |
| Display trim, timing off | 300 / 369 | 54.60 | 56.38 | 3.77 / 0.86 GiB |
| Headless trim, timing off | 329 / 320 | 50.41 | 51.00 | 2.37 / 3.64 GiB |

Historical-prompt and short-prompt medians each use three requests. The
historical prompt has 278 input tokens and 256 output tokens; its SHA-256 is
`5223ff2fe77292258301122e1773f1d9ceb9459d02de2d1e5d4fc7e4da3b7b08`, matching
the previously recorded benchmark provenance. The separate short prompts have
268 input tokens and either 256 or 512 output tokens. Thinking is disabled,
temperature is zero, and all measured requests finish at the requested length.
Outputs are not byte-identical across repeats, so routing and MTP behavior
are not perfectly controlled even with identical input text.

Free VRAM is the minimum of one-second physical sysfs samples during the
measured requests, including the 8K requests where present. This can miss
shorter peaks. The headless arm completed short requests only: its 8K and
61K context checks and GPU trace were omitted at the deadline. Its extra
warmup history also differs after a controller resume. Therefore the table
is observational evidence, not a strictly controlled percentage penalty for
each placement and not equal-workload peak-memory certification.

The display arm additionally completed 61,463 input tokens. Its minimum free
VRAM stayed approximately 3.77 / 0.86 GiB during that request. Text, tools,
and the one-image red fixture passed on both trimmed placements. Full 128K,
hours-long stability, and the combined 300 / 320 placement were not tested.

## What was isolated

The original placement already reproduces roughly 54–55 token/s on the exact
historical prompt, before long-context and vision requests and with PLE
timing logging off. The difference from the earlier 78.11 mean is therefore
not explained solely by changing from 256 to 512 output tokens, or by the
order of long-context/vision checks.

Current-image baseline reference TG remained 54.87 after 61K context and
54.67 after vision. The display trim similarly measured 55.13 and 54.35.

The logging-on arm started near 55, then reached 70–82 token/s without a
restart or configuration change. Its historical prompt measured 81.06 after
long context and 79.37 after vision. Free VRAM did not materially change at
that transition. Per-request MTP acceptance changed much less than speed:
for the second short TG512 request, baseline versus logging acceptance was
74.5% versus 77.7%, while speed was 53.75 versus 78.52 token/s.

Server decode histograms corroborate client streaming timing. For example,
short TG256 trial 2 took 4.507 server decode seconds in the baseline and
3.124 seconds in the logging arm. Client measurements agree closely.
This is serving-time variation, not just SSE/client measurement overhead.

PLE timing medians around the logging transition changed from approximately
1.43 to 1.25 ms per worker operation, too small by themselves to account for
the much larger throughput change. GPU core clocks rose under load in both
modes, and sampled memory clocks were unchanged. These observations do not
rule out all scheduling, power, or shared-host effects. In particular, the
planned final original-placement repeat did not fit the time bound, so the
causal effect of logging is unresolved. Do not turn logging on as a claimed
performance fix based on this run.

## Synchronization lead from the GPU trace

A separate eight-active-iteration PyTorch trace was captured after the display
arm's unprofiled benchmarks. The trace is not used as a throughput sample.

Across an approximately 534 ms device-event span, recorded GPU kernel time
summed to 84.64 ms on rank 0 and 102.44 ms on rank 1. Each rank recorded sixteen
`hipMemcpyWithStream` calls totaling approximately 449–450 ms of CPU time.
All sixteen copied **eight bytes from host to device**. Their matching device
copy events totaled only 43–48 microseconds. One post-first-step blocking call
lasted approximately 45.93 ms for a 2.60-microsecond device transfer.

These are waits associated with tiny metadata transfers, not hundreds of
milliseconds spent transmitting expert weights in those copy calls. CPU and
GPU intervals overlap; totals must not be added or interpreted as an exact
recoverable fraction of latency. Profiling adds overhead, and a blocking copy
can inherit the cost of previously queued work.

The immediately surrounding CPU operations are three `nonzero` operations,
index extraction, `cat`, and repeated `to`/`_to_copy`/`copy_`. This matches the
PLE short-convolution metadata builder in
`vendor/vllm/vllm/v1/attention/backends/short_conv_attn.py`, particularly its
CPU request-index copies around lines 333–334 and its repeated speculative
index copy around line 401. This is a source/trace inference, not Python-stack
attribution: the trace was recorded without stacks.

The next focused ablation should instrument that builder, then compare
existing blocking copies against pinned asynchronous copies with correct
source lifetime and reuse of already-device-resident indices. The existing
`async_tensor_h2d` helper is already imported there. Preserve request ordering,
accepted-token indexing, CPU-device operation, and mixed prefill/decode/spec
metadata semantics. Confirm whether removing these waits improves unprofiled
throughput or merely moves the waits elsewhere. Also trace the faster state;
this run captured only a slower state.

## Expert storage and cache traffic

The display trim removes 1,685,619,200 packed bytes (1.57 GiB) from GPU storage
and adds the same payload to pinned cold storage. The headless trim moves
2,848,115,200 bytes (2.65 GiB). Physical free VRAM also depends on allocator
reservations, driver allocations, and desktop activity; its change need not
match the packed expert payload exactly.

Worker allocator snapshots for the original placement separate major stages:

| Stage | Rank 0 allocated / reserved | Rank 1 allocated / reserved |
|---|---:|---:|
| After model loading | 22.838 / 23.629 GiB | 25.886 / 26.725 GiB |
| After profiling | 23.009 / 23.879 GiB | 26.058 / 26.975 GiB |
| After KV initialization | 25.138 / 26.008 GiB | 28.187 / 29.104 GiB |
| After graph/warmup | 25.435 / 26.254 GiB | 28.484 / 29.350 GiB |

These are PyTorch allocations, not total physical usage. HIP free-memory
snapshots and physical sysfs free memory differed, so the report keeps those
measurements separate rather than constructing an exact cross-API residual.

Existing kernel counters were read between requests on the trimmed arms.
For the three historical TG256 requests, rank 1 filled approximately
2.016–2.042 GiB of expert cache with 369 hot experts (display arm), versus
3.135–3.204 GiB with 320 hot experts (headless arm). That is about 1.1 GiB
more cache-fill payload per request in this sample. The observed headless
trim has more transfer work as well as more free VRAM.

These counts exclude direct UVA reads of cold experts. Cache fills are not
total PCIe traffic, and the counters do not provide a denominator for an
all-route cache-hit percentage. A different rank-0 map and differing generated
outputs also prevent treating this comparison as a pure rank-1 cache ablation.

The results support pursuing per-card headroom rather than equal expert counts.
They do not establish a safe combined map or a universal throughput guarantee.

## Historical-image comparison and limits

The earlier clean image documented in the qualification report,
`sha256:09411bb3e4782eff8c45fd90be620a8d4f808bfb55b8210045c106eef8b3e23a`,
was available locally. All installed vLLM Python files and the three shipped
HIP kernel binaries matched the current candidate byte-for-byte. Installed
GGUF differences were limited to `loader.py`, `params.py`, `tiered_experts.py`,
and the added `tiered_compaction.py`. Image environment and entrypoint matched;
this was not a complete byte comparison of every framework/runtime file.

The historical image's pinned-master loading path failed to reach serving on
this occupied host. Sampled available host RAM fell to approximately 0.025 GiB,
with substantial memory and I/O pressure; Docker inspection timed out. It was
manually stopped with a five-second grace period, yielding exit 137 and
`OOMKilled=false`. This exit was our stop escalation, not proof of an OOM kill.
No throughput result from that image exists. Available RAM recovered to over
70 GiB after stopping it. The current loader's first two startup minima were
approximately 11.5–11.7 GiB available. Retain the pageable-master improvement;
do not roll back the loader based on this incomplete speed comparison.

## Provenance and evidence

Current binary image:
`sha256:ade48fbf9e16485397fc74964f82ad707502bd78b8f979b4376d6d001af7820a`.
Host launcher source began at `adc6164`. No vendor source, production defaults,
ROCm stack, model assets, or kernel binaries were modified by this experiment.
The test used a local worker overlay for lifecycle memory snapshots and one
snapshot every 128 execute calls, without explicit synchronization. Later arms
also exposed local-only development RPC for between-request counters and
post-benchmark profiling. That diagnostic overlay is not a release image.

The bounded run and collection lasted 45m26s. Controller adaptations preserved
completed arms; the headless controller resume added warmup history. The
historical failure consumed the time reserved for a final baseline repeat.
All test containers are stopped and retained; current-image containers exited
zero. The final headless 61K probe was omitted, despite a subsequently reused
log label saying “after long context.” The absence of its context result file
and the explicit omission event are authoritative.

Local private evidence is under
`/var/home/dylan/AI-Work/r9v-throughput-isolation-20260907/`: `protocol.json`,
`events.jsonl`, `memory.jsonl`, per-arm responses/metrics/logs, worker snapshots,
`analysis.json`, `cache-analysis.json`, `blocking-copy-analysis.json`, and the
two display-arm GPU traces. The one-off controller, overlay, and analyzers are
preserved there. Raw evidence has not been uploaded or committed.
