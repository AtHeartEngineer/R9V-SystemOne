# Flash Next hardware assumptions and configurable expert residency

Reviewed 2026-09-07 against the pinned vLLM/GGUF Flash Next runtime. This is a
serving-profile review; it does not change the separate native inference engine.

## Decision

Make per-rank physical free-VRAM targets the eventual user contract. Named
presets should choose headroom/workload settings, not maintain unrelated expert
maps. Keep the current placement as a performance reference while qualifying a
more conservative arm. A target of 5 GiB on each card is a reasonable experiment,
not a demonstrated universal safety margin or crash cure.

Implementing the full contract requires a calibrated planner and runtime checks.
The doctor changes here add validation and accounting; they do not automatically
resize a loaded model or guarantee minimum free VRAM throughout arbitrary future
workloads.

## Is rank 1 too full?

The saved September 2 user log records these values:

| Component | Rank 0 | Rank 1 |
|---|---:|---:|
| Static hot expert payload | 17.810 GiB | 19.975 GiB |
| Dynamic expert cache payload | 0 | 0.866 GiB |
| Total reported model load allocation | 22.81 GiB | 25.85 GiB |
| Explicit KV reservation | 2.13 GiB | 2.13 GiB |
| Reported graph capture allocation | 0.38 GiB | 0.38 GiB |

The expert/cache rows are included in the model-load row; do not add them again.
Against the logged 31.34 GiB initially free, subtracting model load, KV and graph
allocation leaves roughly 6.02/2.98 GiB. This is an approximate accounting
reconstruction, not a measured free-memory low watermark: allocator reservations,
non-PyTorch allocations, vision/workspace peaks and concurrent applications can
change it. It nevertheless makes the headless rank a sensible pressure-reduction
experiment.

The cache is allocated up front with a fixed number of physical slots. A long
run does not accumulate a new resident expert on every cache miss. Late failures
can still expose lazy workspace allocations, fragmentation, new image/prompt
shapes, driver faults or protocol bugs. A successful startup does not prove the
workload's memory high-water mark fits.

Start the A/B trial by retaining the LRU16 cache and reducing rank 1's static
count from 369 to 329 per layer. This releases 2,324,992,000 bytes (2.165 GiB) of
packed GPU weights and adds the same amount to pinned cold host storage. Keep
rank 0 at 329, context, MTP, image limits and runtime versions constant. A second
arm can disable the 16 cache slots: that releases another 929,996,800 bytes
(0.866 GiB) without enlarging the cold owner, because cached experts already
have a host copy. Measure the performance cost separately, particularly on a
slow PCIe rank.

The published placement is unchanged. The generated 329/329 arm is explicitly
unqualified. Neither changing headroom targets nor lowering count ceilings
changes residency by itself.

## The minimum is not the advertised active parameter count

The target routes ten experts per token, but cold experts can be read through
UVA from pinned host memory. Routing to an expert does not imply that its
weights must be permanently resident in VRAM. The current manifest validator
requires at least one hot expert per layer; that is an implementation constraint,
not a requirement that all ten active experts be hot. Extreme low-hot placements
still require kernel correctness and performance qualification.

Mandatory or workload-dependent GPU allocations include the target's non-routed
weights and shared experts, MTP weights/state, vision encoder/projector, attention
and recurrent state, KV reservation, graph pools, communication buffers,
activations, quantization/route workspaces, and allocator/runtime overhead. They
must be budgeted using actual TP layout and runtime behavior. A single “active
parameters × bytes” calculation misses this structure.

Reducing hot residency increases pinned host RAM use and PCIe traffic. It also
does not remove the full pageable expert masters used during initial loading.
A 55.43-GiB aggregate packed-master component exists before compaction, while the
cold owners and temporary staging evolve as layers are compacted. File sizes,
steady RSS, and startup peak are different quantities. In pinned PLE mode the
26.82-GiB table is an additional pinned component; in SSD mode its pages are
file-backed and should not be counted again as a mandatory full pinned copy.

## Hardware assumptions found

| Assumption | Why it is insufficient | Correction |
|---|---|---|
| Rank 0 needs headroom; rank 1 may use everything | Displays and co-tenants can be on either card; both ranks have transients | Configure a physical free-memory target for every rank, tied to stable BDF/UUID identity |
| 329/385 counts establish VRAM safety | Dense/MTP/vision, KV, graphs and workspaces are absent; a manifest can lie about its lists | Validate lists and packed tensor bytes; enforce a complete calibrated memory budget |
| Every additional hot expert costs the same across layers | This target mixes quantizations; per-layer expert bytes take three different values | Use a per-layer cost catalog for the exact package and TP split |
| 15/7 GB/s PCIe floors are a minimum requirement | They express reference performance, not whether the computation is valid | Default hard floors are now disabled; reference speed shortfalls warn; explicit user floors still fail |
| Endpoint maximum speed/width describes the path | Upstream bridges, slot wiring and current negotiation can limit it | Keep the path traversal; distinguish advertised capacity from measured transfer behavior |
| Correct doctor enumeration fixes all runtime enumeration | The pinned ROCm platform still indexes some AMD-SMI handles numerically | Match properties by HIP-reported BDF/UUID inside workers; doctor now detects host index differences and probes container HIP order |
| Explicit KV bytes are automatically protected by gpu_memory_utilization | The pinned worker skips available-memory sizing in that branch | Budget KV explicitly with the whole workload; warn now and add worker memory admission later |
| The shell config describes the running process | Container arguments/environments can be stale or overridden | Compare launch arguments and critical environment values; ultimately store an immutable worker plan fingerprint |
| A metadata/idle check proves readiness | It performs no representative allocation, host-UVA read, collective, graph or image execution | Separate prerequisite checks from an explicit runtime qualification suite |
| A driver version or two device nodes establish transport health | Host pinning, peer transport, NUMA placement and real kernel execution may fail | Add small correctness probes for HIP allocation, pinned UVA reads, copies and TP collectives; record actual transport and NUMA binding |

There are also non-memory candidates. The PLE worker retains only the first
request per DP rank when draining a batch and logs that duplicates were skipped.
The saved older report contains those warnings. Requests need explicit sequence
identity, buffer ownership and completion/error propagation before this path can
be considered robust; blindly replaying discarded metadata could read overwritten
input. This is a protocol investigation, not something lowering hot residency
can resolve. Corrected PCIe errors in the saved report also require physical-BDF
correlation; earlier analysis found the reported root port belonged to another
GPU, not proof that a serving R9700 failed.

## What the larger map should contain

Publish a **complete ranked catalog**, not a larger fixed resident set:

- Every expert ID 0..511 for every layer, with no duplicates or missing IDs.
- Packed cost of each expert shard, exact target artifact hashes, TP layout and
  kernel/runtime compatibility.
- Route-frequency/benefit data from a representative training corpus and held-out
  results. Include varied text, tools, vision, long-context and MTP behavior;
  the current eager route profiler collects only one-to-three-row events and
  does not establish long-prefill coverage.
- Ranking provenance and uncertainty for unseen/tied experts. Do not invent a
  meaningful order for experts whose scores were not collected.

The current maps contain only prefixes (329 and 369 IDs per layer). They are
enough to trim safely while preserving the existing order, but cannot recover
the ranking of the omitted experts. Existing rank 0 lists match prefixes of rank
1 in the installed reference artifact. The new trim tool stays within each
rank's supplied list; it does not invent or expand a catalog. It also removes
old route/holdout statistics, which no longer describe the smaller placement.

For the first automatic planner, retain uniform counts per rank and account for
all mixed layer costs. Later, nonuniform counts can optimize benefit per byte,
but their host-staging allocation shapes, graph behavior and runtime kernels
need independent qualification. Optimize the slower rank's step time, not merely
aggregate cache-hit rate; TP progress is constrained by the slowest participant.

## How a real headroom planner should work

For rank r, solve:

    hot payload + dynamic cache payload
      <= physical capacity - external-use allowance - requested free headroom
         - mandatory model/KV allocations - measured transient/runtime reserve

Keep persistent allocations and transient allowances separate so none are counted
twice. Use GiB explicitly in the UI/config, show the resulting byte budget and
expected pinned-host increase, and reject infeasible requests with the largest
headroom the chosen workload can support. Offer reducing context, concurrency,
image limits or MTP as explicit alternatives; never silently change those
features or pretend the requested free space was achieved.

The sequence should be:

1. Resolve worker HIP identities and validate the exact model/runtime/workload
   contract, including host RAM and device permissions.
2. Establish a calibrated non-expert envelope using an intentionally small hot
   placement and representative maximum prefill/decode/vision/graph workloads.
   Ensure that calibration itself fits host memory. Measure both per-process
   allocation/reservation peaks and physical device free memory; polling alone
   can miss a transient peak.
3. Compute the resident map and cache allocation within each rank's budget.
   Generate a separate immutable manifest and plan. Recheck host pinned storage
   and loading-phase headroom after choosing the map.
4. Load the chosen placement and validate the same workload envelope. Freeze it
   before graph capture/serving; moving or resizing weights under captured graphs
   requires a separate design. If admission fails, exit with the measured deficit
   and preserve evidence rather than retrying indefinitely.
5. Monitor the observed free-memory low watermark and pressure while serving.
   Other applications can invalidate the external-use allowance, so a target is
   guaranteed only within an explicit workload/co-tenancy contract.

Fingerprint plans with model and manifest contents, kernel/plugin/image identity,
HIP/PyTorch/runtime/driver versions, GPU identity/capacity, TP order, KV/context,
prefill and sequence limits, MTP, vision limits, graph policy and cache settings.
The existing vLLM startup-plan cache is for KV sizing; it does not explicitly
include our expert-manifest contents, cache policy or HIP version in its factor
list. It is not a ready-made expert-placement planner.

## Doctor changes completed in this pass

Doctor validates all per-layer IDs/counts and optional declared bytes, accounts
for static/cold/cache payloads including the extra async cache slot, reports the
pinned-host floor and pageable-master component, and rejects impossible partial
VRAM budgets. It rejects unsupported kernel/cache combinations, invalid numeric
budgets, TP mismatches and stale context/KV/concurrency launch flags. Headroom
checks apply to both cards with `R9V_MIN_FREE_VRAM_GIB_BY_RANK=5,5` (default 3,3).

The old target-file-size + fixed 16-GiB RAM estimate was removed: it counted
file-backed PLE misleadingly and could double count it in pinned mode. New
component accounting explicitly does not certify the unknown total peak.

Container HIP probing is a fresh process using that container's visibility
masks; it is not proof of an already-running worker's identity. Worker startup
BDF records, measured peak-memory admission, full API stress, driver/NUMA/transport
qualification and the PLE protocol fix remain necessary follow-up work.
