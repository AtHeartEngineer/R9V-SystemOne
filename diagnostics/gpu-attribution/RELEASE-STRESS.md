# Release stress workload, September 11, 2026

The live 022000 campaign exercises the pinned host-fenced PLE image on two R9700 GPUs with the desktop active. It preserves TP2, 300/320 placement, four rank-1 LRU slots, P2P, 131072 context, MTP2, max sequences 1, and decode graph shapes 1/3.

Each wave submits eight competing clients: near-128K prefill, another 32K/64K/96K prefill, 2048- and 1024-token decodes, two content-confirmed streaming disconnects, and two timed client drops. Timed drops do not prove server execution. Queue order is randomized with seed 911731. One arithmetic recovery request follows each wave. This stresses the supported one-sequence configuration; it does not test simultaneous model execution at a higher sequence setting.

Even waves add a 1920x1080 X11 copy/fill workload at 60fps plus an uncapped 1920x1080 Vulkan cube. Odd waves remove both. The workload starts waves continuously for at least an hour and allows the final wave to finish. There is no idle padding. Root process identity and external driver telemetry remain active; the model has no per-step diagnostic probes.

The blackbox capture directory is a bounded 512MiB tmpfs, with independent compressed snapshots on the local machine every 20 seconds. Admission verifies acknowledgements as the actual workload user. Runtime checks bound archive staleness, host/trace/identity freshness, request deadlines, and graphics lifetime. Stops retain a 90-second aftermath. No automatic reboot or power cycle is implemented.

`release_stress.py --help` documents the client arguments. It requires an already prepared, frozen, admitted campaign and recorder. The deployment scripts under AI-Work are incident-specific: do not replay their boot IDs, paths, or campaign names on another machine. `python3 -m unittest test_release_stress test_transition_stress -v` runs 18 CPU regression tests against the revised source. The frozen 022000 run used the preceding 17-test version.

The frozen original retrieval prompt included a run identifier and a code without an explicit target label. Its 26 mismatches remain in the raw results; several responses were more problematic than merely returning the identifier. A separate clearly labeled 65K request passed during load. Reusable source now labels the target explicitly; the warm restart follow-up applies that oracle at 32K, 64K, 96K and near 128K. Never silently reclassify the original raw release result as a pass.

This is a bounded stress qualification, not exhaustive release coverage. It does not establish multi-day endurance, arbitrary user prompts, multimodal correctness, different hardware/configurations, or the previous 70-token/s performance target. See RESULT.md for measured outcomes and remaining issues.
