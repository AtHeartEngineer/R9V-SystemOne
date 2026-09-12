# Qwen MTP4 release candidate: placement and support

`qwen38-mtp4` (IQ4_XS) and `qwen38-q4-xl` (Q4_K_XL) are experimental profiles. The public source gates (PR32 and PR33), image bundle and fresh-download checks passed for both profiles. The IQ4 image6 streaming reference passed its bounded 128K baseline, while clean IQ4 setup/restart and Q4 clean user-flow qualification remain pending. The latest BetterBench speed/latency evaluation also remains pending. The public runtime image bundle is available from the [v0.2.0-rc1 GitHub Release](https://github.com/Dyluhn/R9V/releases/tag/v0.2.0-rc1-images); setup verifies its parts and loads the exact image ID.

## Reference qualification

The reference machine has two 32 GiB R9700 GPUs and 128 GiB host RAM. Rank 0 uses PCIe Gen5 x16; rank 1 crosses a Gen4 x4 link. The tests retain 131,072 context and 2,562,215,936 KV bytes on each card, with MTP4 and prefix caching disabled.

| Profile | Reference static experts, ranks 0/1 | Dynamic cache slots, ranks 0/1 | Minimum free VRAM, ranks 0/1 |
|---|---:|---:|---:|
| `qwen38-mtp4` | 71 / 450 | 160 / 0 | 3.79 / 3.79 GiB |
| `qwen38-q4-xl` | 97 / 349 | 80 / 0 | 3.79 / 3.76 GiB |

The IQ4 image6 streaming reference retained **131,072 context tokens** and passed seven bounded checks, including text, tools, three image shapes, idle resume and an actual **130,941-token prompt**. Its median was **93.026774 TG tok/s**; measured free VRAM was 4,072,144,896 and 4,075,905,024 bytes (about 3.79 GiB per card), followed by a clean 90-second aftermath and GPU reclaim. This reference used existing verified assets and does not establish clean download-to-run setup/restart or answer quality. The Q4 clean user flow and latest BetterBench evaluation remain pending.

Prior reference throughput samples measured IQ4 at 93.825 tok/s with static 72/455 and Q4 at 53.431 tok/s with static 97/349. The initial Q4 bootstrap placement measured 25.285 tok/s. These are fixed-prompt reference samples with MTP4, not throughput measurements of the newer user-planned placements, mixed traffic, or decode at full context. They do not establish TG100.

The qualification archive identities are recorded in [the release evidence index](qualification/results/qwen38-mtp4-userstart-20260912.json). Profiles remain bound to their exact tested runtime images; a rebuilt image requires matching calibration and fresh qualification.

## Selecting free memory on each card

`--headroom 5,3` requests at least 5 GiB free on the first selected card and 3 GiB on the second. Card order follows the saved GPU BDF selection; verify it with doctor. Complete measured expert catalogs and portable reference memory seeds are included for both profiles. The planner accounts separately for static experts, dynamic cache, other model/runtime allocations, context and external GPU usage, and rejects an impossible request with each card's shortfall.

This is a measured workload budget, not a reservation against applications allocating memory later. Context is retained. More headroom can require fewer experts in VRAM and lower throughput. Each map records all 512 experts in each of 48 layers per rank, with separate training and held-out routing captures. Unobserved experts remain explicit ties; their order is not evidence of measured coldness. A workload with different routing can perform differently.

The 416/224 split describes intermediate channels on the cards, not expert counts. IQ4 and Q4 have separate packed-cost catalogs, measured maps and memory seeds. Their maps and budgets cannot be interchanged.

Setup automatically selects and verifies the profile image bundle:

```bash
./r9v setup qwen38-mtp4 --model-dir "$MODEL_DIR" \
  --headroom 3,3 \
  --ple-path "$EXISTING_PLE" --accept-model-license
./r9v start qwen38-mtp4
```

Use `qwen38-q4-xl` with its own model directory and image for Q4. The profiles select their matching catalog and seed automatically. Explicit `--calibration` and `--expert-catalog` remain available for separately measured configurations; stale or mismatched evidence is rejected.

First start plans the placement, loads the model and runs the complete local workload qualification before reporting ready. An unchanged restart reuses the verified receipt. To change the requested budget, stop the selected container, then run `./r9v start qwen38-mtp4 --headroom 5,3`; the new placement must qualify. A plain `run` does not perform this setup workflow. Pass the same `--state-dir` to setup, start and support when using a custom state location.

The measured reference image IDs are:

- IQ4: `sha256:2e50016cfcc9cd22f15d3f69ccf001e4877236e12ebb4ab458cc9c16caaef9e3`
- Q4: `sha256:2e50016cfcc9cd22f15d3f69ccf001e4877236e12ebb4ab458cc9c16caaef9e3`

The image identities are carried by the release bundle and verified during setup. Building from source does not imply the resulting image has either identity.

## Q4_K_XL assets

Q4 uses the original upstream shards pinned by revision and SHA-256; it does not requantize IQ4. Its original Q8_0 target head and IQ4's original Q6_K target head are unchanged. Q6-specific draft-head acceleration is disabled for Q4. The image6 streaming loader is enabled for both current profiles.

`--reuse-from "$EXISTING_MODEL_DIR"` reuses matching auxiliary assets through verified hard links on the same filesystem. `--ple-path "$EXISTING_PLE"` selects an existing matching PLE tensor. Incompatible assets are rejected. The four Q4 target shards occupy 103.69 GiB; reserve additional space for auxiliary assets, the runtime and caches. The original unranked bootstrap manifest remains available as historical evidence; the new profile selects the measured ranked placement.

## Doctor and support evidence

Run support before removing a failed container. Select its profile and saved state:

```bash
./r9v doctor qwen38-q4-xl
./r9v support qwen38-q4-xl --state-dir "$STATE_DIR" \
  --output "$SUPPORT_DIR/run-001" --archive "$SUPPORT_DIR/run-001.tar.gz"
```

Support records sanitized setup/placement context, image identity, selected host source-file hashes, worker records, host memory and Normal-zone pressure, GPU/PCIe state, available server/kernel logs and rolling capture tails. File sizes and SHA-256 values accompany explicit probe availability. Source hashes remain available in source archives without Git metadata. Both reference profiles successfully collected the required evidence after stopping; a stopped server's live metrics and `docker exec` probes can be unavailable without invalidating retained logs and worker records.

Directories and archives are private, bounded and never overwritten. Configuration summaries hide credentials and personal paths. Raw application/kernel logs may contain identifying information; review them before sharing. Collection writes locally and never uploads. Include the failed action and approximate time; do not attach model weights or private prompts.

Doctor separates configured settings from execution evidence. A missing startup kernel marker does not prove the wrong kernel ran. Rootless Docker intentionally inherits its daemon's memlock hard limit; blindly forcing unlimited memlock can prevent startup. Inspect the daemon policy and actual serving-worker pinned-UVA probes when investigating that warning.
