# Rougher release stress test — September 11, 2026

The 61.3-minute sustained stress run passed its stability checks. The subsequent warm restart, explicit retrieval requests, and 90.104-second aftermath also passed review. This is evidence of improved stability on this workstation, not an unconditional shipping clearance.

## Measured workload

- 16 continuous waves, 8 competing clients per wave, with the supported single executing model sequence.
- 80 completed normal responses, 32 content-confirmed streaming disconnects, and 32 timed client drops. Timed drops do not prove server execution.
- 16 near-128K prefills, each 130806–130816 input tokens, plus 16 additional large prefills.
- 50,231 completion tokens in normal responses, including 48,991 from the two long-decode jobs per wave. Requested decode limits are not assumed to equal actual output length.
- Alternating 1920x1080 X11 rendering and uncapped Vulkan graphics load. A retained telemetry sample measured both GPUs at 100% busy, with reported power averages of 336W and 281W and junction temperatures of 94°C and 81°C. These are individual observed samples, not whole-run averages or maxima.
- All 16 arithmetic recovery checks passed. A separately labeled 65K retrieval request during the load also passed.

## Stability evidence

No GPU timeout, allocation warning, OOM kill, host reboot, or automatic recovery was recorded in the main campaign. The model container exited 0. The harness unit records 143 because the controller intentionally terminated it at completion; this is distinct from the container exit status. Both actual inference worker PIDs and KWin appeared throughout 3,602 identity snapshots; the largest interval was 1.046 seconds. Minimum free VRAM was 4,029,440,000 and 3,442,548,736 bytes. The main aftermath lasted 90.027 seconds and passed review.

The frozen image is `sha256:977c0de0ee05c415a8f8fca7d38e78e5a9c8fbb083b1dad8e0a87f927ea370a0`. TP2, expert placement, P2P, 131072 context, MTP2, and decode graphs remained unchanged. No production model code was changed for this test.

## Findings that remain relevant to shipping

The original retrieval oracle produced 26 mismatches and the raw main `release_passed` remains false. Its wording ambiguously included both a run identifier and a code, but some answers also repeated filler or refused the task. Those results are retained, not erased. Explicitly labeled follow-up probes passed at 32K, 64K, 96K and near 128K after a warm restart. This narrows the problem but does not establish general long-context accuracy across varied user prompts.

The warm restart did not clear file caches and began with approximately 57 GiB free host RAM. Its successful startup does not resolve the earlier reproduced failure under lower-memory conditions. The earlier roughly 45-token/s benchmark versus a 70-token/s target was not requalified by this run. Neither issue should be silently converted into a pass.

An initial separate attempt, 021000, aborted before stress requests because the independent archive acknowledgement had incorrect ownership. Publication now uses the actual workload user and admission verifies readability as that user. No GPU failure occurred in that aborted attempt. A supplementary helper also had a syntax error before issuing requests; it was corrected, compiled and rerun successfully. Both histories are retained.

## Artifacts and reusable checks

`attribution-20260911T022000Z/REVIEWED-RELEASE-STRESS.json` contains the reviewed main metrics and original mismatches. The warm review is in `attribution-20260911T033000Z/REVIEWED-WARM-STRESS.json`. Full captures are preserved as `capture-final.tar.gz` under each campaign, independently SHA256-verified on this machine and blackbox. Temporary RAM captures are retired only after comparing every current file with the archive.

`RELEASE-STRESS.md` describes the workload and limitations. The reusable client now uses explicit retrieval labels. Eighteen CPU regression tests passed against revised helpers; the frozen main run used the preceding 17-test version. The source is also in workstation `projects/inference/r9v/diagnostics/gpu-attribution`. No commit, push, deployment, or power cycle was performed.

Cleanup verified: the workstation remains on its original boot, the desktop and all four unrelated CPU containers remain active, owned graphics/model/capture services are stopped, and all three temporary RAM mounts were retired after full archival verification. The recovery ledger remains at five completed power cycles out of six; none was used here.
