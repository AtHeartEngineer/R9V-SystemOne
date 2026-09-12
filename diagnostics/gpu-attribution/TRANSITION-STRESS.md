# PLE transition stress test

Developed September 10, 2026. Eleven CPU regression tests pass. Latest live run stopped after52.61minutes on lost independent identity coverage. The full-hour qualification is **inconclusive**, not passed. See RESULT.md for findings and limitations.

The previous stability test repeated a request catalog. This test targets ownership and scheduling transitions in the host-fenced PLE implementation: prefill/decode shape changes, stream cancellation, queued requests, idle/resume, and graphics starting/stopping. It retains the patched image, 131072 model context, MTP2, P2P, 300/320 placement, four rank-1 LRU slots, and existing model/cache assets. No source overlays, per-step GPU probes, artificial OOM, panic, reset, or power action.

## One-hour workload

Each deterministic round performs:

1. A short prompt requesting 1024 generated tokens.
2. A fresh approximately 32768-token prompt with 3 generated tokens.
3. A fresh approximately 130816-token prompt with 1 generated token, followed by a short 3-token decode.
4. A long streaming response disconnected after three content-bearing SSE chunks, followed by an arithmetic recovery request.
5. Two simultaneous client requests, one disconnected and one allowed to finish. The server keeps its tested maximum of one executing sequence; this exercises queue transitions.
6. Every third round: 30 seconds idle, then arithmetic recovery.

One owned 512x512 X11 fill/copy control runs at 30fps on alternate rounds. Unique service names and bounded stop/kill cleanup prevent prior rendering controls accumulating. Each long prompt starts with a deterministic changing nonce, reducing cross-round prefix reuse. `/tokenize` chooses actual prompt size; returned usage must confirm the envelope. Seeds, repeat counts, hashes, request outcomes, and graphics logs allow reconstruction.

Default duration is 3600 seconds, with at least three complete rounds required. No new round starts in the final 600 seconds; that remainder retains telemetry while idle. The result reports actual completed rounds and elapsed time. This is not a claim of continuous full utilization for every second.

## Outcomes and bounds

- Requests run in disposable child processes with a total parent-enforced deadline (300 seconds per request group; 30 seconds per tokenizer call), including unresponsive sockets. At most two HTTP children exist at once. Response size is capped at 2MiB.
- Intended disconnections are recorded separately. A stream that ends before the planned disconnect, an unexpected EOF, HTTP error, deadline, or lost telemetry fails the test; there is no blind request retry.
- Arithmetic errors are recorded separately from transport/stability. The one-token context probes test execution and shape transitions, **not** the known failing full-code retrieval test.
- Headroom and kernel/process attribution come from the existing independent campaign collectors. A client exception alone does not prove a GPU crash.
- Success requires fresh controller heartbeat, fresh two-worker identities from the expected boot, the pinned patched image, no model source overlays, complete requests, planned disconnections, rendering progress, and at least three rounds.
- On completion or failure the client requests the existing controller's owned stop. Its `result.json` deliberately leaves `aftermath_verified=false`: review the controller's frozen evidence and 90-second aftermath before issuing a final stability verdict.
- Expected new client data is below 32MiB; existing capture/log budgets still apply. Reuse the existing 1.2GiB compilation cache and model weights. Check blackbox space first (about640MiB remained at the last audit).

## Running under the existing controller

The driver runs **as dylan on blackbox**. It requires a **fresh** campaign assembled using the validated `215000` admission/paused-harness pattern. Historical STOP files must never be removed. Before starting:

- Use current-boot root identity, driver/kernel heartbeat, arithmetic and graph IPC smoke gates.
- Keep the tested immutable image `sha256:977c0de0ee05c415a8f8fca7d38e78e5a9c8fbb083b1dad8e0a87f927ea370a0` and `ple-host-fixed-20260910` cache namespace.
- Give the parent model unit at least 5400 seconds total runtime, including startup; keep its paused-between-requests loop and active driver monitoring. Its independent controller must cover the whole test plus 90 seconds aftermath.
- Wait for `BOUNDARY-PAUSED.json` after the two initial warmups. No catalog or benchmark client may still be issuing requests.
- Preserve the fresh campaign's `x11_copy_workload.py` on workstation. Stop any earlier owned rendering controls before admission. Do not stop the normal desktop or unrelated CPU containers.
- The blackbox campaign must expose `status.json`, `capture-health.json`, `identity-admitted.json`, and `controller-config.json` as in the validated admission harness. Use atomic_json.save_json for every controller snapshot; in-place truncation can expose empty JSON to readers. The controller must atomically publish `capture-health.json` immediately after each accepted host/trace heartbeat or fault, with the TraceGate values `host_at`, `trace_at`, and `fault`. The driver checks primary receipt ages (host <10 seconds, trace <45 seconds), controller summary age <10 seconds, and worker identity age <5 seconds. Sample the clock after reading atomic publications. Periodic summaries alone can hide fresh receipts and falsely reject healthy telemetry.

Then, from a dylan shell on blackbox (substitute the **new** campaign and output paths):

```bash
systemd-run --user --unit=r9v-transition-stress --collect \
  --property=RuntimeMaxSec=65min --property=TimeoutStopSec=20 \
  --property=MemoryMax=512M --property=Restart=no \
  /usr/bin/python3 /home/dylan/AI-Work/r9v-transition-stress-20260910/transition_stress.py \
  --campaign /home/dylan/AI-Work/r9v-crash-20260908/root-cause/NEW-CAMPAIGN \
  --output /home/dylan/AI-Work/r9v-crash-20260908/root-cause/NEW-CAMPAIGN/transition-results \
  --seconds 3600 --seed 731
```

The test itself never powers the workstation. The existing cumulative recovery ledger remains **5 used / 1 remaining**, and the authorized recovery sequence applies only after a confirmed lockout. Developing this test consumed no recovery cycle.

## Offline validation

```bash
python3 transition_stress.py --plan
python3 -m unittest test_transition_stress -v
```

The fake HTTP server tests simultaneous completion/cancellation, premature end, truncated SSE, total deadlines, active coverage loss, large tokenizer payload handling, real-token calibration, and stop/reboot/stale/missing-worker rejection. These are harness tests, not GPU stability evidence.
