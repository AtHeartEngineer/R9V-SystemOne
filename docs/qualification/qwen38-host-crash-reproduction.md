# Qwen3.8 Flash Next host crash: capture, reproduce, isolate

Status: a subsequent blackbox-controlled campaign captured display-GPU access loss and a kernel hard lockup during the first warmup of its third fresh server, with route collection disabled. The initial GPU-failure cause remains unresolved. The preparation and proposed comparisons below are historical; they are not a qualification result.

## September 8 follow-up: memory pressure and HIP operation capture

Recovered local kernel evidence contains an order-0 page-allocation failure during
model loading, about 97 seconds before GPU access was lost. The Normal memory
zone had 35,596 KiB free against a 66,456 KiB minimum, despite sampled
MemAvailable remaining above 61 GiB. Memory pressure reached 33.74% `some avg10`.
This establishes a gap in the old MemAvailable-only guard; it does not prove that
RAM pressure caused the GPU failure.

`tools/qualify_runtime.py` now records host memory-zone watermarks and stops after
two consecutive samples with all Normal zones below their minimum. The guard
uses host-only `/proc` reads. It supplements the existing available-RAM guard;
it is not a production admission guarantee or a substitute for measuring peak
load and serving residency.

For a bounded diagnostic session, set `R9V_HIP_FAULT_DIAGNOSTICS=1` in its protocol
config. The launcher sets `AMD_SERIALIZE_KERNEL=3`, `AMD_SERIALIZE_COPY=3`,
`AMD_LOG_LEVEL=4`, and decimal `AMD_LOG_MASK=2097155`. This captures API calls,
commands and shader names while synchronizing operations. HIP's integer flag
parser uses decimal conversion, so a `0x`-prefixed mask can silently disable the
requested logging. Verify actual operation records before dispatching requests.
Logging remains off by default and is expensive; compare neither its throughput
nor its passing behavior directly with ordinary asynchronous serving. Bound the
session, rotate logs and preserve them on an independent machine.

`tools/remote_watch.py` also starts early capture on fresh netconsole GPU faults
while an explicitly armed session is active. It retains the once-per-container
latch and rejects stale records; waiting only for lost progress can miss the
window before SSH stops responding.

The examined OGC `v7.1.5-ogc5` source calls the IP-state dump before printing the
normal ring-timeout identity. The failed capture has a dump-start message without
its completion or ring identity. Logging ring/PASID before that dump is a focused
kernel instrumentation change if userspace evidence cannot identify the trigger.
It is not itself a stability fix.

Sources: [HIP debugging](https://rocm.docs.amd.com/projects/HIP/en/docs-7.1.1/how-to/debugging.html),
[HIP flag parsing](https://github.com/ROCm/clr/blob/develop/rocclr/utils/flags.cpp),
[examined kernel timeout path](https://github.com/OpenGamingCollective/linux/blob/a48e710b0766bc42ff292581cb1b6915a3ec1798/drivers/gpu/drm/amd/amdgpu/amdgpu_job.c).

## What the incident narrows down

The route-collection marker was created at 06:58:44 CDT. Response files 0, 1, and 2 were created by 06:58:59, but their contents did not survive. Final fsynced telemetry began at 06:59:02 and lost the display GPU's live sensor readings. The next request (index 3) is a candidate, not a proven faulting request. Both the collection path and monitoring activity remain possible contributors.

The original process used HIP 7.14.60850, the release-probe-v6 image, TP2, MTP2, eager route collection, the 300/320 expert map with 16 rank-1 cache slots, and 2,554,105,856 bytes of KV per rank. Preserve the original image ID, source snapshot, warmup sequence, input order, and map hashes from the incident artifacts. Do not rebuild over the incident tag. Changing warmups, image, VRAM budget, and profiler simultaneously would prevent attribution.

## Capture before another GPU run

Inspection on this machine found kexec_crash_loaded=0, kexec_crash_size=0, no kdump.service, no archived pstore dump, and no panic-on-lockup configuration. The NMI watchdog is enabled, but that is not evidence that a crash dump will be produced. kernel.sysrq=16 allows keyboard sync, not the diagnostic task/backtrace commands. The AMD driver has gpu_recovery=-1 (automatic/default), not explicit recovery disabled. Do not infer a fault from those defaults alone.

1. Prefer a second **physical machine** on the wired LAN receiving kernel netconsole output. A VM on the affected workstation does not survive its host locking up. Verify delivery with a harmless uniquely named kernel log message before testing; preserve receiver timestamps. Netconsole uses UDP and can lose messages or fail if the NIC/bus stops. It improves evidence, not guarantees it.
2. Configure and validate kdump on the host OS as a complementary path. It requires a reserved crash-memory region and a loaded capture kernel; a container alone cannot provide this. Save vmcore to the approved large-data storage with an explicit size budget before provisioning it. Configuration requires administrator access and typically a reboot. Do not turn on forced panics or trigger a deliberate crash before the dump path is ready and the workstation is prepared for interruption.
3. If remote access still works during a display freeze, collect the kernel journal and SysRq blocked tasks (w), task states (t), CPU backtraces (l), and memory state (m) before rebooting. Root can invoke these through /proc/sysrq-trigger independently of the keyboard bitmask. These are diagnostic commands; do not substitute c (deliberate crash) or b (immediate reboot).
4. Keep request events, host RAM/PSI, GPU physical VRAM, worker progress, and container state in separate recording paths. Isolate driver-backed sensor polling from the host heartbeat so a blocked sysfs read cannot stop both. Record each read error and duration: a null value alone does not distinguish EIO, permission errors, timeout, or missing files. This monitor isolation remains to be implemented.
5. Record the kernel build, firmware, boot arguments, and any active GPU/CPU/RAM tuning. Current boot includes a custom 7.1.5-ogc5.1 kernel, amdgpu.dcdebugmask=0x20000, and amdgpu.ppfeaturemask=0xffffffff. These are controlled variables, not evidence of an overclock or a cause.

Sources: [kernel netconsole](https://docs.kernel.org/networking/netconsole.html), [kdump](https://docs.kernel.org/admin-guide/kdump/kdump.html), [SysRq](https://docs.kernel.org/admin-guide/sysrq.html), [AMD driver parameters](https://docs.kernel.org/gpu/amdgpu/module-parameters.html).

## First controlled comparison

Use fresh containers, the same immutable image and effective environment, and the same original two warmups plus ten benchmark requests before the corpus. Set R9V_ROUTE_PROFILE_DIR for **both** arms, so both use the same eager execution mode; only the collection marker differs. Each arm gets its own unused directory mounted at the matching server path. Verify that mapping before dispatch. Merely omitting R9V_ROUTE_PROFILE_DIR would also change server execution configuration and spoil the comparison.

Start with the first four training requests (the smallest ordered prefix spanning the candidate failure); only expand to the full eleven-request corpus if needed. Cap each GPU session at 45 minutes including startup and cleanup, using the existing qualification supervisor. Stop after the first captured failure; do not loop through host crashes automatically. The replay client alone is not a host watchdog and cannot guarantee cleanup after a kernel lockup.

Against an already prepared server, the new client controls are:

```bash
python3 tools/capture_routes.py --directory "$RUN_DIR" --split train --limit 4 --no-collect
```

For the matching collection arm in a fresh server and directory, omit `--no-collect`. RUN_DIR must be the actual shared path specified in that server's R9V_ROUTE_PROFILE_DIR. These commands do not create the server or replace its supervisor. Add supervisor protocol support for this A/B sequence before automated execution.

The client now fsyncs the exact request before dispatch, response after receipt, corpus, and event journal, including directory entries. events.jsonl identifies request_start without response_saved after an interruption. Results distinguish replay-only and truncated-corpus runs from complete collection. Fsync changes timing, so a non-reproduction must not be treated as proof the original fault is fixed.

## Follow the evidence

| Result | Next isolation step |
| --- | --- |
| Collection on fails, off survives | Reduce to the same top-k shapes and index values; test the histogram allocation/scatter path independently with bounded GPU memory. Add index-range validation and stream synchronization only in a diagnostic build. Do not assume scatter_add is faulty merely because it is newly active. |
| Both fail | Compare with driver-backed sensor polling disabled while retaining independent host/kernel capture; then isolate MTP, custom expert-cache kernels, and inter-GPU transfers one at a time. |
| GPU page fault or ring timeout with process/VM identification | Correlate the identified process and queue with the durable request and worker stage; reduce the implicated operation. |
| CPU lockup or blocked-task trace | Use the kernel stack to identify the lock/driver path, then compare kernel builds with userspace fixed. |
| OOM or allocation failures | Change only the expert budget, keeping KV/context/MTP fixed; compare measured 3 versus 5 GiB free headroom. |
| Abrupt loss without a trace despite verified remote capture | Power, PCIe, firmware, CPU/RAM, or a hard bus lock remain possible. Compare stock tuning and hardware independently; lack of logs does not prove PSU failure. |

Only after a reproducible failing case exists should we compare ROCm 7.14 with ROCm 10, then any kernel change separately. Both container runtimes use the host kernel driver. A fix needs the same reproducer to change from failing to passing, followed by repeated bounded runs and the original throughput, VRAM, text/tool/vision, and long-context checks. One successful retry cannot establish stability.


## Additional observability before the next reproduction

`tools/observability.py` isolates GPU sysfs reads in one subprocess, retaining
last-read paths, errno, slow reads and sample age. Qualification host RAM checks
continue while that subprocess is stuck. Driver I/O failures trigger evidence
collection before cleanup. Passive capture also runs support bundles outside its
monitor loop.

Set `R9V_OBSERVABILITY_TARGET` in the qualification protocol config to the
collector's numeric IPv4 address and UDP port. Run IDs connect qualification
phases, route request/persistence events, worker progress, VRAM/sensor samples,
and early capture requests. Producer IDs disambiguate sequence counters.
`tools/remote_watch.py` runs on the independent log receiver and captures over
SSH after an armed run goes silent or reports a driver failure. It never resets
hardware or starts workloads.

Optional `R9V_STAGE_DIAGNOSTICS=1` adds host-only worker markers around model
execution, token sampling, and HIP diagnostic queries. These describe host call
boundaries; they do not synchronize GPU completion. This mode also registers
Python stack capture through SIGUSR2 when that signal is available. Early capture
uses worker-advertised readiness, start time and pidfds to avoid signalling
unrelated/reused PIDs. Use the matching runtime build, or freeze both
`R9V_DEV_GPU_WORKER_PY` and `R9V_DEV_WORKER_DIAGNOSTICS_PY` source overlays in the
protocol. Leave stage instrumentation off for normal throughput qualification.

The workstation-specific installer, service definitions, CPU integration proof,
and pending kdump/debug-symbol requirements are recorded under
`/var/home/dylan/AI-Work/r9v-release-20260908/observability/README.md`.
CPU checks do not qualify GPU execution or demonstrate successful kernel dump
capture. Kdump still requires administrator setup, reserved crash memory, a
planned reboot and a validated dump before automatic lockup panics are armed.

## Blackbox continuation: independently attributable evidence

The September 9 autonomous continuation added bounded per-process operation
journals and optional UDP operation records under `R9V_STAGE_DIAGNOSTICS=1`.
Each record carries container, rank, PID, process start ticks, native thread ID
and sequence. `R9V_OBSERVABILITY_TARGET` is forwarded to workers by the launcher.
The local journal retains two 1 MiB segments per worker; the blackbox observer
keeps separate journals for armed worker identities and freezes them before
requesting early remote capture. UDP sequence gaps mean lost evidence; a
host operation return is not proof of asynchronous GPU completion. Shared HIP
stderr remains unsuitable for attributing an unmatched individual kernel call.

`tools/fault_window.py` also supplies first-fault retention for the prepared
blackbox controller's rotating HIP streams. It latches before moving/fsyncing
the window, and later cleanup output cannot rotate the snapshot away. CPU
fixtures exercise interprocess attribution, restart retention and actual
observer freeze-before-SSH ordering without dispatching GPU work.

Cgroup capture now includes per-process host status, namespace PID mapping,
RSS categories, swap and lock/pin counters. Shared RSS is explicitly not summed.
Historical cgroup extraction found 49.214 GiB peak anonymous charge and
11.987 GiB peak swap.current during startup. No loader-pressure reduction or
causal connection to the initiating GPU fault has been established.

A small exact-kernel timeout probe is built under
`diagnostics/timeout-probe-7.1.5-ogc5/`. Its README records running-BTF offsets,
compilation limitations and the root privilege required to load it. It is
not loaded and is not a stability fix. No new GPU run or host crash occurred
in this continuation. Full asynchronous serving and release qualification
remain unresolved; passing CPU and earlier serialized tests do not waive them.

Durable continuation evidence is on blackbox at
`/home/dylan/AI-Work/r9v-crash-20260908/root-cause/autonomous-20260909T024558Z/FINAL.md`.

## Tracefs integration and September 9 normal-execution arm

The signed-module restriction was handled with the reviewed built-in tracefs
recorder; the unsigned external module was not loaded. The blackbox controller
now admits only after verifying the recorder's source/unit hashes, root ownership,
active service and process identity, kernel/live BTF, pinned runtime/map/source,
free host/Normal-zone memory and absence of reported KFD users. It separately
requires fresh matching-boot host and recorder heartbeats (`captured=false`).
Recorder traffic cannot refresh host liveness. Stale, wrong-source/boot and
replayed recorder messages are rejected; first_timeout or captured=true latches
a fault, freezes the local window before SSH, and stops owned work once.
The standalone qualification harness does not supply these controller gates.

The authorized single asynchronous arm completed 2 warmups and 3 measured
256-token requests, then exited cleanly. Measured decode rates were 38.00,
43.54 and 23.33 tok/s (median 38.00). Sampled serving free VRAM reached
3.614/2.806 GiB. These short requests neither reproduce the original third-start
failure nor qualify long context, vision, strict headroom or release.

Startup captured about 27.8 GiB anonymous RSS in each TP worker, 57.659 GiB peak
cgroup anonymous charge, and Normal-zone free memory within 1.910 MiB of its
minimum. Full expert-master staging before compaction is a concrete optimization
target; a bounded-loader implementation and matched runtime validation are still
required. Startup pressure was not shown to cause a GPU fault.

The slow final request had the same output text hash as the other measured
requests, zero sampled memory PSI, increased CPU pressure and doubled median
host execute_model duration. No initiating GPU operation/queue/PASID was captured.
The first delivered tracefs timeout is not a globally atomic first fault and
userspace forwarding still depends on scheduling; netconsole remains independent.

UDP operation retention missed 190 of 3990 records in this run. Complete
per-worker journals were recovered from the stopped container. The CPU-tested
`tools/operation_history.py` accepts only contiguous same-worker/thread/step
enter/return pairs, preventing gaps from being reported as a long GPU operation.
It measures host boundaries, not asynchronous GPU completion.

Evidence: blackbox
`/home/dylan/AI-Work/r9v-crash-20260908/root-cause/resume-20260909T202840Z/FINAL.md`.
No additional GPU arm, fault induction, reset, reboot or recorder rearm followed.
