# Qwen MTP4 release candidate: placement and support

`qwen38-mtp4` (IQ4_XS) and `qwen38-q4-xl` (Q4_K_XL) are experimental profiles. The public source gates (PR32 and PR33) passed for both profiles. The IQ4 image7 streaming reference passed its bounded 128K baseline; a second startup passed, while ordinary public setup/start/restart remains pending. Q4's image6 ranked reference is recorded, while its ordinary user flow remains pending. The latest BetterBench plan (GGZ14 v0.6.0, `d00ad5e`) has not been executed. Anonymous public downloads and a resumed Docker import passed; no model ran during the image check. The public runtime image bundle is available from the [v0.2.0-rc2 GitHub Release](https://github.com/Dyluhn/R9V/releases/tag/v0.2.0-rc2-images); setup verifies its parts and loads the exact image ID.

## Reference qualification

The reference machine has two 32 GiB R9700 GPUs and 128 GiB host RAM. Rank 0 uses PCIe Gen5 x16; rank 1 crosses a Gen4 x4 link. The tests retain 131,072 context and 2,562,215,936 KV bytes on each card, with MTP4 and prefix caching disabled.

| Profile | Reference static experts, ranks 0/1 | Dynamic cache slots, ranks 0/1 | Minimum free VRAM, ranks 0/1 |
|---|---:|---:|---:|
| `qwen38-mtp4` | 71 / 450 | 160 / 0 | 3.96 / 3.81 GiB |
| `qwen38-q4-xl` | 97 / 349 | 80 / 0 | 3.79 / 3.76 GiB |

The IQ4 image7 streaming reference retained **131,072 context tokens** and passed seven bounded checks, including text, tools, three image shapes, idle resume and an actual **130,941-token prompt**. Its median was **89.45196 TG tok/s**; measured free VRAM was 4,253,020,160 and 4,090,036,224 bytes, with minimum Normal-zone free value 450,269,184 bytes, followed by a clean 90-second aftermath and GPU reclaim. The supervisor incorrectly reported failure because its cleanup check required exact VRAM equality: rank 0 had 185.203 MiB more free and rank 1 was unchanged. Independent review confirmed no per-card shortfall throughout the aftermath. This reference used existing verified assets and does not establish ordinary public setup/start/restart or answer quality. The Q4 ordinary public user flow and latest BetterBench plan remain pending.

The Q4 image6 ranked reference measured 53.431 TG tok/s with static 97/349. Older comparator samples remain historical evidence in the qualification archive and are not measurements of the current user flows, mixed traffic or decode at full context. No result here establishes TG100 or the planned BetterBench evaluation.

The qualification archive identities are recorded in [the release evidence index](qualification/results/qwen38-mtp4-userstart-20260912.json). Profiles remain bound to their exact tested runtime images; a rebuilt image requires matching calibration and fresh qualification.

## Selecting free memory on each card

`--headroom 5,3` requests at least 5 GiB free on the first selected card and 3 GiB on the second. Card order follows the saved GPU BDF selection; verify it with doctor. Complete measured expert catalogs and portable reference memory seeds are included for both profiles. The planner accounts separately for static experts, dynamic cache, other model/runtime allocations, context and external GPU usage, and rejects an impossible request with each card's shortfall.

This is a measured workload budget, not a reservation against applications allocating memory later. Context is retained. More headroom can require fewer experts in VRAM and lower throughput. Each map records all 512 experts in each of 48 layers per rank, with separate training and held-out routing captures. Placement order retains existing hot prefixes and orders remaining experts by observed frequency. Unobserved experts remain deterministic ties, not evidence of measured coldness. A workload with different routing can perform differently.

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

- IQ4: `sha256:46ab688af195643e61322a72b4e7b7fa0999c12299bffb2a4515f8363c59393c`
- Q4: `sha256:2e50016cfcc9cd22f15d3f69ccf001e4877236e12ebb4ab458cc9c16caaef9e3`

The image identities are carried by the release bundle and verified during setup. Building from source does not imply the resulting image has either identity.

## Model assets and runtime images

IQ4 image7 uses exact-sized HIP cold owners with the streaming loader. Its original model bytes and Q6_K target head are unchanged. Q4 uses the original upstream shards pinned by revision and SHA-256, its original Q8_0 target head is unchanged, and image6 remains its runtime image. Neither profile's model or head was changed for this reference.

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
