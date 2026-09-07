# Qwen3.8 Flash Next stability and ROCm 10.0 qualification

Status: proposed qualification plan, 2026-09-07. No ROCm 10 image has been built
or qualified by this change. The release candidate remains on its existing
pinned ROCm 7.14 runtime. This work concerns the vLLM/GGUF Flash Next profile.

## What ROCm 10 changes

AMD calls ROCm 10.0 a production release. The new installation layout uses
`/opt/rocm/core-10.0`; package-manager installations provide compatibility
symlinks, while tarball installations need explicit paths. R9700/gfx1201 is
part of the `gfx120X-all` package family. See AMD's
[transition guide](https://rocm.docs.amd.com/en/latest/about/transition-guide-TheRock.html).

The [release notes](https://rocm.docs.amd.com/en/latest/about/release-notes.html)
list PyTorch 2.13 and vLLM 0.27 support, HIP event improvements, GPU core-dump
improvements, and several profiler stability fixes. These are reasons to test
an upgrade; they do not identify the cause of the reported R9V failures or
establish that ordinary unprofiled Flash Next serving is fixed.

Check the exact GPU, operating system, driver, and firmware combination against
AMD's [compatibility matrix](https://rocm.docs.amd.com/en/latest/compatibility/compatibility-matrix.html).
A userspace container cannot update the host amdgpu driver or repair faulty
PCIe links.

## Build an isolated candidate

The current vLLM Dockerfile pins a ROCm 7.14 base digest, builds PyTorch 2.11,
Triton 3.6, torchvision 0.24.1, and AITER 0.1.19, and embeds `+rocm7.14` in the
PyTorch build version. Changing only the base tag would mix assumptions and
mislabel the result. The runtime Dockerfile then rebuilds the GGUF plugin and
custom HIP kernels against that stack.

Prepare a separately named `rocm10` runtime descriptor and immutable base-image
digest. Resolve the matching framework/torchvision/Triton/AITER versions and
rebuild vLLM, GGUF plugin, and every custom HIP extension. Keep the current
model package and placement fixed for the first comparison. Verify paths,
linked HIP/RCCL libraries, torch.version.hip, architecture targets, and image
ID inside the candidate. Keep compilation caches separate by image/runtime.

First try the existing application pins with the supported framework stack;
if ABI/API compatibility requires a vLLM rebase, record that as a separate
variable. Run tensor/kernel correctness gates before performance and soak
measurements. Preserve the working 7.14 image and cache for rollback. Do not
promote a mutable `latest` tag as upgrade evidence.

## Required evidence

| Gate | Workload | Required evidence |
|---|---|---|
| Clean install | fetch, package/hash verification, PLE derivation, build, doctor, launch | Complete commands, versions, BDF/rank mapping, image digest; no development overlays |
| Startup memory | At least three fresh container starts on the lowest supported RAM host | Host MemAvailable, cgroup peak/events, VRAM per BDF, PLE mode; set measured total/available RAM minima with reserve |
| Functional | Text, streaming, tool call, one image, MTP enabled; bounded output correctness fixtures | Valid completions and expected content/tool/image behavior; no silent fallback |
| Context | Unique cache-miss 8K/32K/64K and near-128K prompts plus output | Actual tokenizer/server counts, successful finish, resource peaks; stay within total context |
| Short soak | Two hours cycling prompt sizes | Requests/timeline/support bundle, no timeouts, restarts, OOMs, or new uncorrectable AER |
| Long soak | Eight hours, then 24 hours for the release candidate | Same evidence; memory and token latency trends, corrected AER rate and thermal/power samples reviewed |
| Idle/wake | Repeated ten-minute idle intervals followed by new requests | No stalled wake, new faults, or unexplained resource growth |
| Portability | Reference asymmetric PCIe host and another supported dual-R9700 topology; lowest claimed RAM size | Passes with documented rank/placement settings and explicit performance expectations |
| Upgrade A/B | Same model, prompts, host driver, placement and policies on 7.14 and 10.0 | Correctness, completion/failure rate, memory peaks, PP/TG, startup time; changes attributable to the candidate |

`./r9v soak qwen38` implements sequential synthetic text liveness trials with
hard request deadlines. It is only the short/long/idle workload driver; it
cannot certify the other rows. Its default token lengths are approximate and
it does not grade generated text or benchmark streaming inter-token latency.
Use the existing streaming benchmark for PP/TG and targeted functional tests
for API capabilities.

Before removing a failed container, run `./r9v support qwen38 --output <new-dir>`.
Correlate all evidence by UTC time and physical BDF. A SIGTERM cleanup sequence
or exit 137 alone does not prove the source of a crash. A cumulative corrected
PCIe error count alone does not prove the GPU path failed. Preserve the actual
incident window and compare new counters with the baseline.

Release promotion requires measured clean-host and endurance results, not
just passing the CPU diagnostics tests. Unreproduced user failures remain open
until affected hosts provide a comparable capture or a reproducible workload.
