# R9V

[![CI](https://github.com/Dyluhn/R9V/actions/workflows/ci.yml/badge.svg)](https://github.com/Dyluhn/R9V/actions/workflows/ci.yml)

R9V runs Qwen3.8 Flash Next on two AMD Radeon AI PRO R9700 GPUs. It combines a pinned vLLM fork, GGUF loading, specialized `gfx1201` kernels, expert offloading and four-token MTP speculative decoding. The server exposes an OpenAI-compatible API for text, tool calls and single-image requests.

Each profile binds a model package, runtime, hardware layout and expert placement. Downloads are checked against pinned revisions and file hashes. Setup records the selected configuration, and first start qualifies its workload and memory headroom before reporting ready.

**Current status:** the IQ4_XS and Q4_K_XL MTP4 profiles passed setup, first start and restart on the reference machine using existing verified assets. They remain experimental. Setup now selects the [GitHub Release image bundle](https://github.com/Dyluhn/R9V/releases/tag/v0.2.0-rc1-images), verifies its parts, and loads the exact original image ID. Clean download-to-run reproduction remains pending; BetterBench speed/latency evaluation is separate and pending. See [release status and evidence](docs/qwen-release-candidate.md).

## Profiles and features

| Alias | Model package | Runtime | Status |
|---|---|---|---|
| `qwen38-mtp4` | IQ4_XS | MTP4, dual R9700, 128K context | Reference setup/start/restart passed |
| `qwen38-q4-xl` | Q4_K_XL | MTP4, dual R9700, 128K context | Reference setup/start/restart passed |

Use the explicit MTP4 aliases for the current workflow.

- **Per-card headroom:** choose the free VRAM to retain on each card with `--headroom 3,3` or an asymmetric budget such as `--headroom 5,3`.
- **Measured expert maps:** separate complete catalogs for IQ4 and Q4 rank all 512 experts across 48 layers per rank, using training and separate held-out routing captures. The planner retains frequently used experts in VRAM within the memory budget.
- **Q4_K_XL support:** a dedicated package, packed expert costs, ranked placement and streaming loader. Q4 uses its upstream model bytes and original Q8_0 target head; IQ4 retains its original Q6_K target head.
- **Resumable setup and qualified restart:** reuse matching model assets, save configuration, qualify a newly planned placement, and reuse its verified receipt on an unchanged restart.
- **Doctor:** inspect model/runtime/placement compatibility, GPU ordering, memory pressure, PCIe information, cache identity and available worker evidence.
- **Support bundles:** collect configuration summaries, source identities, worker records, memory and GPU diagnostics, logs and capture tails into a bounded local archive with file hashes.

Unobserved experts are explicit ties in the maps. Routing frequency depends on workload; a different prompt mix can change the best placement. Requested headroom is checked against the qualification workload, and cannot prevent another application from allocating VRAM later.

## Hardware and storage

The reference system uses:

- Two **32 GiB Radeon AI PRO R9700** cards (`gfx1201`).
- Linux with working AMD GPU drivers, `amd-smi`, `/dev/kfd` and `/dev/dri` access.
- **128 GiB host RAM**. Smaller hosts are untested; cold expert allocations use host memory.
- An asymmetric PCIe layout: rank 0 on Gen5 x16 and rank 1 across Gen4 x4. GPU ordering matters to placement and performance.
- Git, Python 3.10+, Docker and the Hugging Face CLI described in the [installation guide](docs/installation.md).

The IQ4 package occupies approximately **90.36 GiB**. The four Q4 target shards alone occupy **103.69 GiB**, with auxiliary assets additional. The derived PLE file occupies **26.82 GiB**. Leave further room for runtime images, compilation caches and diagnostics. Reuse verified assets instead of duplicating model files.

## Setup and start

Clone the source and its exact dependencies:

```bash
git clone --recursive https://github.com/Dyluhn/R9V.git
cd R9V
./r9v show qwen38-mtp4
./r9v doctor qwen38-mtp4 -- --host-only
```

The profile distribution entries select the matching GitHub Release image bundle and exact image IDs. Docker 29 must use the containerd image store so `docker load` preserves those IDs. These are the reference identities:

| Profile | Tested local image ID |
|---|---|
| `qwen38-mtp4` | `sha256:987468f3f9991dfad8b07f51a18bbd5e5c01dc164d9f82c4143e77bfd14ca80d` |
| `qwen38-q4-xl` | `sha256:2e50016cfcc9cd22f15d3f69ccf001e4877236e12ebb4ab458cc9c16caaef9e3` |

Install the download CLI in an isolated environment if it is not already available:

```bash
python3 -m venv ~/.local/share/r9v/download-tools
~/.local/share/r9v/download-tools/bin/pip install huggingface_hub
export PATH="$HOME/.local/share/r9v/download-tools/bin:$PATH"
```

For IQ4, choose persistent SSD directories with enough free space:

```bash
MODEL_DIR=/path/to/qwen-iq4
STATE_DIR=/path/to/r9v-state/iq4
./r9v setup qwen38-mtp4 \
  --model-dir "$MODEL_DIR" --state-dir "$STATE_DIR" \
  --headroom 3,3 --accept-model-license
./r9v start qwen38-mtp4 --state-dir "$STATE_DIR"
```

For Q4, use its own model and state directories:

```bash
./r9v setup qwen38-q4-xl --model-dir /path/to/qwen-q4 \
  --state-dir /path/to/r9v-state/q4 --headroom 3,3 --accept-model-license
./r9v start qwen38-q4-xl --state-dir /path/to/r9v-state/q4
```

Run one profile at a time. Before switching, inspect `docker ps`, save any needed support evidence, and stop the selected R9V container by its exact name using `docker stop NAME`. Setup supports `--reuse-from /path/to/existing/assets` for matching assets and `--ple-path /path/to/existing/ple.bin` for an existing verified PLE file. The profiles select their corresponding expert catalog and reference memory seed automatically.

First start runs the local workload qualification, including context and headroom checks. An unchanged restart can reuse its verified receipt. To change the budget, stop the selected container, then run:

```bash
./r9v start qwen38-mtp4 --state-dir "$STATE_DIR" --headroom 5,3
```

The new placement must qualify. Card order follows the saved GPU selection; inspect it with doctor. The planner reports per-card shortfalls when a request cannot fit, while retaining the configured context.

The default API endpoint is `http://127.0.0.1:8004/v1`. Use the address recorded by your selected setup if you override the port. See the [release guide](docs/qwen-release-candidate.md), [installation guide](docs/installation.md), and [troubleshooting guide](docs/troubleshooting.md). Rebuilding an image does not reproduce a qualified image identity automatically.

## Measured results

The following fixed-prompt reference samples used MTP4 on the dual-R9700 system:

| Profile / placement | Static experts, ranks 0/1 | Generation tokens/s |
|---|---:|---:|
| IQ4 reference | 72 / 455 | **93.825** |
| Q4 measured ranked placement | 97 / 349 | **53.431** |
| Q4 initial bootstrap placement | 64 / 320 | **25.285** |

These samples are not measurements of mixed traffic or generation at a full context. They also differ slightly from the placements selected by the final user setup flow:

| User setup profile | Static experts, ranks 0/1 | Dynamic cache slots, ranks 0/1 | Minimum free VRAM, ranks 0/1 |
|---|---:|---:|---:|
| IQ4, requested 3 / 3 GiB | 71 / 457 | 160 / 0 | 3.82 / 3.67 GiB |
| Q4, requested 3 / 3 GiB | 97 / 348 | 80 / 0 | 3.79 / 3.74 GiB |

Both user setups retained **131,072 context tokens** and passed seven workload checks, including an actual **130,941-token prompt**, text, tools, three image shapes and idle resume. Stop/restart reused the unchanged verified receipt and completed a short request. These checks establish bounded runtime behavior, not answer quality or a 100 tok/s qualification.

The [evidence index](docs/qualification/results/qwen38-mtp4-userstart-20260912.json) records the preserved qualification archives. Historical prefill and comparator results remain in the [earlier Qwen qualification](docs/qualification/qwen38-ud-iq4-xs-dual-r9700.md); they should not be substituted for measurements of the new placements.

## Diagnostics and reporting a problem

Run doctor for the selected profile:

```bash
./r9v doctor qwen38-mtp4 --state-dir "$STATE_DIR"
```

Collect support evidence before removing a failed container, using the same state directory as setup:

```bash
SUPPORT_DIR=/path/to/private/r9v-support
./r9v support qwen38-mtp4 --state-dir "$STATE_DIR" \
  --output "$SUPPORT_DIR/run-001" \
  --archive "$SUPPORT_DIR/run-001.tar.gz"
```

Collection stays local and never uploads automatically. Configuration summaries hide credentials and personal paths, but raw application and kernel logs can contain identifying information or request content. Review the archive before attaching it to a GitHub issue. Include the profile, failed command, approximate failure time and whether the server, container or whole host stopped responding. Do not attach model weights or private prompts.

Doctor distinguishes configured settings from observed execution. Missing live metrics after a container stops are reported as unavailable; a missing kernel marker alone does not prove the wrong kernel ran. See the [configuration reference](profiles/qwen38-flash-next/dual-r9700/README.md) for available controls and corrective actions.

## Source and development

```text
profiles/             model/runtime/hardware compositions and launch settings
packages/models/      pinned model sources, artifact hashes and licenses
packages/placements/  expert maps, memory seeds and placement manifests
runtimes/             runtime descriptors and retained kernel/source overlays
hardware/             GPU, RAM, PCIe and rank contracts
kernels/              pinned R9V kernel submodule
vendor/               pinned vLLM and GGUF-plugin forks
tools/                setup, planning, qualification, doctor and support
tests/                CPU checks and explicit GPU qualification tests
```

The kernels are specialized for supported shapes, quantizations and `gfx1201`. A different model, GPU, topology or runtime image requires its own validation. Current reference qualification covers one active sequence at a time.

Run the CPU and static checks with:

```bash
python -m pip install --requirement requirements-ci.txt
python -m pytest -q tests
./scripts/ci-static.sh
```

CPU CI checks tooling and source contracts. GPU parity, graph replay, full-model qualification and throughput measurements require the matching hardware. Clean-host reproduction and BetterBench speed/latency evaluation remain pending release work.

Read [CONVENTIONS.md](CONVENTIONS.md) before changing code. Dependency gitlinks are release inputs: use the committed revisions rather than replacing them with moving branch heads.

## License and provenance

R9V-owned code and kernels use Apache-2.0. The vLLM and GGUF-plugin forks retain their licenses; llama.cpp/ggml-derived quantization primitives retain their MIT notices. Qwen model weights are separately governed by the Qwen Community License 1.0.

See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md), [licensing](docs/licensing.md) and the [provenance audit](docs/provenance-audit.md). Runtime source publication, public image distribution and complete installation qualification are tracked separately; this README reports only the gates that have actually passed.
