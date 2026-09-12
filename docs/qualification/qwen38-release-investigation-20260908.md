# Flash Next release investigation, 2026-09-08

Status: qualification in progress. No runtime promotion is justified yet.
This investigation covers the vLLM/GGUF profile only.

The reference host has a Ryzen 5 9600X, 128 GiB installed RAM (123.4 GiB
reported usable), and two R9700s. Rank 0 is the display card at `03:00.0` on
Gen5 x16. Rank 1 at `13:00.0` has a Gen4 x4 upstream bottleneck. Endpoint
link speed alone does not describe that path. All GPU sessions are bounded
to 45 minutes including cleanup; these results do not establish hours of
stability or qualify other host topologies.

## Findings

* **Fixed KV exhaustion is reproducible despite free VRAM.** With
  2,285,670,400 KV bytes per rank, a 130,941-token request was preempted at
  129,024 computed tokens, then again at 127,424. The scheduler reset its
  work and repeated prefill. The startup estimate of 133,719 KV tokens did
  not account adequately for this workload's working blocks. The public
  metrics counter remained zero while scheduler-owned records captured
  both rewinds.
* **An additional 256 MiB removed that timeout in one comparison.** With
  2,554,105,856 KV bytes, the same request completed in 83.74 seconds,
  with no recorded rewind. It returned `R9V` instead of `R9V-731` and
  therefore failed retrieval correctness. This is not a passing memory
  calibration or proof of the minimum sufficient KV allocation.
* **Expert storage needs an explicit physical reserve on both cards.**
  The original map left about 924 MB physical free on the headless card
  during one failed full-context run. The 300/320 map with rank-1 LRU16
  left at least 3,517,366,272 and 3,329,507,328 bytes during the larger-KV
  workload. Reserve must include dense/MTP/vision weights, KV, graphs,
  workspace, allocator overhead, and external display use.
* **Removing the LRU is a regression on this host.** At equal packed
  expert payload, 300/320 plus rank-1 LRU16 measured 65.18 median decode
  tokens/s; static 300/336 measured 41.16. Both passed the 32K functional
  workload. Keep LRU16 while investigating the remaining throughput loss.
* **Free VRAM does not explain the throughput loss by itself.** In a
  controlled reservation/release experiment, throughput fell from about
  67 to 51 tokens/s and stayed low after the same HIP free-byte counts
  were restored. The same output hash occurred at both speeds. This
  does not identify the persistent state responsible for the slowdown.
* **Cold/warm compilation dependencies differed without code changes.**
  Reconstructing the recorded SHA values showed that an unreadable
  `<frozen os>` marker was the only difference. Normalizing both traces
  to the actual standard-library source preserves invalidation and
  passes focused tests.

Subsequent restarts reused the same compilation hash. Main compilation dropped
from 157.12 to 4.46 seconds with MTP2, and 165.19 to 4.28 seconds without MTP.
Startup kernels compiled before that stage were still going to disposable
container storage; the launcher now persists their Triton cache and autotune
results too. The additional startup-cache change still needs a real restart test.

With MTP disabled, the 130,941-token retrieval returned the complete code in
80.39 seconds. Two exact-format vision checks failed because the answer repeated
`red`. With MTP enabled, retrieval repeatedly returned only `R9V`. Neither arm
passes the complete qualification. The optimized QSA scorer matched its reference
GPU implementation exactly on visible scores at 32K and 128K, and agreed with
independent PyTorch calculations within 1.5e-6 on the sampled cases.

Captures also retain the driver's GTT accounting alongside physical VRAM and
worker allocator statistics. GTT usage alone is not proof of eviction: host
offload uses GPU-addressable system memory intentionally. Counter definitions
are documented in the [Linux AMDGPU memory accounting reference](https://www.kernel.org/doc/html/latest/gpu/amdgpu/driver-misc.html#gpu-memory-usage-information).

The complete ROCm 10 candidate passed custom-kernel checks and serving-worker
transport probes after correcting SDK library discovery. It reproduced the same
near-128K retrieval failure in 84.04 seconds. Timed decode samples were 64.23,
49.65, and 50.00 tokens/s; physical free-byte minima were 2,790,100,992 and
2,596,503,552. This does not support promoting the upgrade as a stability or
throughput fix. The matching 7.14 image remains available.

The driver's `pp_dpm_mclk` table lacked an active-state marker during most fast
and slow pressure samples, so those records do not establish the memory clock.
Captures now include direct hwmon clock readings and their labels, temperature,
and power. Thread scheduling observations are also being collected for the
kernel isolation runs.

Disabling the fused GDN MTP kernel did not fix retrieval; neither did additionally
disabling the optimized dense M3, HC-up mixing, and shared-expert epilogue paths.
All other functional checks passed in both arms. These tests retained the tiered
expert kernel and prefix caching, so they do not exclude those paths.

The asynchronous ShortConv metadata experiment did not recover throughput
and was reverted. Real worker probes passed with reversed discrete-card
order; a separate HIP-only probe verified a nonconsecutive selection.
Numeric device selection is the hardware-verified path.

## Evidence and release gates

Local immutable experiment records are under
`/var/home/dylan/AI-Work/r9v-release-20260908`: `kv-capacity-session`,
`headroom3-session`, `cache-context-session`, `pressure-session`, and
`pressure-correlation.json`. Failed runs retain requests, responses,
memory samples, container logs, and worker/scheduler records.

Full-context correctness, sustained throughput, strict 3/5 GiB placement
qualification, cache reuse, clean setup, and the ROCm 10 comparison remain
release gates. A failed retrieval must not be relabeled as a passing
calibration. Prebuilt distribution and a reference memory seed require
qualification of the exact image that will be published.
