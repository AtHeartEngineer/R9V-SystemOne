# Install and run Qwen3.8 Flash Next

This guide covers the two newest dual-R9700 MTP4 profiles: `qwen38-mtp4`
(UD-IQ4_XS, experimental) and `qwen38-q4-xl` (UD-Q4_K_XL, experimental).
Both require two 32 GiB `gfx1201` Radeon AI PRO R9700 GPUs, ROCm device access,
Docker, Python 3.10+, Git, `curl`, and storage for the model, 28,800,138,240-byte
PLE payload, image layers, and runtime cache. Device order is semantic.

## Check the host

```bash
git clone --recursive https://github.com/Dyluhn/R9V.git
cd R9V
./r9v list --by-topology
./r9v validate qwen38-mtp4
./r9v validate qwen38-q4-xl
./r9v doctor qwen38-mtp4 -- --host-only
./r9v doctor qwen38-q4-xl -- --host-only
docker info >/dev/null
```

Keep recursive submodule revisions pinned. Host-only checks do not qualify
serving or memory headroom.

## Fetch and setup

Read the Qwen Community License before accepting it. Fetch one profile into its
own model directory; never interchange the IQ4 and Q4 packages or placements.
Install the Hugging Face CLI if model artifacts are not already present:

```bash
python3 -m venv ~/.local/share/r9v/download-tools
~/.local/share/r9v/download-tools/bin/pip install -U huggingface_hub
export PATH="$HOME/.local/share/r9v/download-tools/bin:$PATH"
```

```bash
export MODEL_DIR=/fast-storage/qwen38-r9v
./r9v fetch qwen38-mtp4 --model-dir "$MODEL_DIR" --accept-model-license
./r9v verify qwen38-mtp4 --model-dir "$MODEL_DIR" -- --hash
```

Use `qwen38-q4-xl` in both commands for Q4. Setup selects the profile's
[`release/image-bundle-20260912.json`](https://github.com/Dyluhn/R9V/releases/tag/v0.2.0-rc1-images), downloads and SHA-256 verifies its parts,
and loads the exact original image ID. Docker 29 must use the containerd image
store so the loaded image keeps its exact ID; check `docker info` before setup:

```bash
./r9v setup qwen38-mtp4 --model-dir "$MODEL_DIR" -- \
  --accept-model-license
```

For Q4, substitute `qwen38-q4-xl`. Use
`--gpu-bdfs BDF0,BDF1` for rank order, `--data-dir` for SSD data, `--ple-path`
to reuse a PLE file, and `--state-dir` for isolated resumable state. `--build`
is an explicit source-build choice and cannot accompany image options.

Setup verifies model artifacts, prepares/checks the PLE payload, saves machine
identity, and runs preflight. Repeat it after interruption.

## Start and qualify

```bash
./r9v start qwen38-mtp4
curl -fsS http://127.0.0.1:8004/health
curl -fsS http://127.0.0.1:8004/v1/models
```

Use `qwen38-q4-xl` for Q4. Start waits for health, runs the runtime doctor, and
performs the bounded workload qualification for the selected placement. A
reference seed is an estimate and still requires local validation. A different
headroom target can be planned from the portable release seed; matching local
calibration is an alternative when available:

```bash
./r9v start qwen38-mtp4 -- --headroom 5,5
```

Use `./r9v support PROFILE --state-dir DIR` for private diagnostics. Do not
publish prompts, completions, raw token IDs, or logs. Start refuses to replace
an existing profile container; inspect and deliberately stop that exact
container before retrying.

## Development builds

The public installation path uses the GitHub Release image bundle. Source builds
are a developer workflow requiring the repository's private pinned base and all
dependency inputs; they are outside the clean public reproduction path and are
not a substitute for the released image identity.

## Troubleshooting

| Symptom | Action |
|---|---|
| Download or hash verification fails | Check disk and `hf` access, repair or re-fetch the exact package, and do not continue with a failed hash. |
| Wrong GPU order or BDF mismatch | Run `amd-smi list`, then rerun setup with `--gpu-bdfs BDF0,BDF1`. |
| Insufficient requested headroom | Preserve the per-rank shortfalls. Adjust the target deliberately or let the release seed plan it; complete local workload qualification afterward. |
| Host normal-zone pressure | Free host memory or reduce CPU-offloaded residency; swap does not satisfy pinned-RAM requirements. |
| Startup/JIT timeout | Increase `--timeout`, inspect retained Docker logs, and check image/cache space before retrying. |
| Existing container | Inspect the exact named container, save diagnostics, then deliberately stop/remove it before retrying. |
| Support needed | Run `./r9v support PROFILE --state-dir DIR`; keep the bundle private. |
