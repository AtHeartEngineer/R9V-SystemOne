# Setup guide for AI assistants

This guide covers `qwen38-mtp4` (UD-IQ4_XS, MTP4, 128K, experimental) and `qwen38-q4-xl`
(UD-Q4_K_XL, MTP4, 128K) on two 32 GiB `gfx1201` Radeon AI PRO R9700 GPUs.
Keep their model packages, catalogs, manifests, runtime descriptors, and
calibration records separate.

Check ROCm, `/dev/kfd`, render access, Docker, Python 3.10+, Git, `curl`, and
storage first:

```bash
./r9v list --by-topology
./r9v validate qwen38-mtp4
./r9v validate qwen38-q4-xl
./r9v doctor qwen38-mtp4 -- --host-only
```

If model files are not already present, install the Hugging Face CLI in an
isolated environment: `python3 -m venv /tmp/r9v-tools && /tmp/r9v-tools/bin/pip
install -U huggingface_hub`, then add `/tmp/r9v-tools/bin` to `PATH`.

After the user reads and accepts the Qwen Community License:

```bash
export MODEL_DIR=/fast-storage/qwen38-r9v
./r9v fetch qwen38-mtp4 --model-dir "$MODEL_DIR" --accept-model-license
./r9v verify qwen38-mtp4 --model-dir "$MODEL_DIR" -- --hash
./r9v setup qwen38-mtp4 --model-dir "$MODEL_DIR" -- \
  --accept-model-license
```

For Q4, substitute `qwen38-q4-xl` in every command. Setup automatically reads
[`release/image-bundle-exact-host-20260912.json`](https://github.com/Dyluhn/R9V/releases/tag/v0.2.0-rc2-images), verifies every SHA-256 part, and loads the
exact image ID from the GitHub Release bundle. Docker 29 must use the containerd
image store; verify this with `docker info`. `--build` is the explicit source
build alternative. Never mix IQ4 and Q4 placement artifacts.

Start and check readiness:

```bash
./r9v start qwen38-mtp4
curl -fsS http://127.0.0.1:8004/health
curl -fsS http://127.0.0.1:8004/v1/models
```

Use `qwen38-q4-xl` for Q4. Start runs runtime checks and bounded qualification;
health alone is not qualification. A reference memory seed remains an estimate
and each new placement needs local workload validation. The portable release
seed can plan another headroom target; matching local calibration is also
accepted:

```bash
./r9v start qwen38-mtp4 -- --headroom 5,5
```

Do not expose prompts, completions, raw token IDs, or private logs. Ordinary
public setup/start/restart and the planned BetterBench speed/latency evaluation
remain pending. Use
`./r9v support PROFILE --state-dir DIR` for private diagnostics. Preserve the
fail-closed checks and pinned recursive submodules.

Full command details: [docs/installation.md](docs/installation.md).
