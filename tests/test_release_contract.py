# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import json
import os
import re
import subprocess
from pathlib import Path

from tools.upload_model import build_uploads

ROOT = Path(__file__).resolve().parents[1]
QWEN_PACKAGE = (
    ROOT
    / "packages/models/qwen38-flash-next/"
    "ud-iq4-xs--mtp-blockfp8--mmproj-q8/package.json"
)
QWEN_PROFILE = ROOT / "profiles/qwen38-flash-next/dual-r9700-mtp4/profile.json"
QWEN_PLACEMENT = (
    ROOT
    / "packages/placements/qwen38-flash-next/ud-iq4-xs/dual-r9700/"
    "mtp4-headroom3.json"
)


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_qwen_public_status_matches_published_distribution() -> None:
    profile = _load(QWEN_PROFILE)
    package = _load(QWEN_PACKAGE)

    assert profile["status"] == "experimental"
    assert package["distribution"]["status"] == "published"
    assert package["distribution"]["revision"] == (
        "bf836f0c20b6c92fcad4226ad3115eb8a19f7582"
    )


def test_qwen_package_contains_every_launch_contract_file() -> None:
    package = _load(QWEN_PACKAGE)
    _load(QWEN_PLACEMENT)
    paths = {artifact["path"] for artifact in package["artifacts"]}

    required = {
        "LICENSE",
        "THIRD_PARTY_NOTICES.md",
        "metadata/config.json",
        "metadata/tokenizer.json",
        "metadata/tokenizer_config.json",
        "metadata/chat_template.jinja",
        "metadata/preprocessor_config.json",
        "metadata/video_preprocessor_config.json",
        "metadata/generation_config.json",
        "metadata/merges.txt",
        "metadata/vocab.json",
        "mtp/config.json",
        "mtp/model.safetensors",
        "mtp/mtp-fp8-block-manifest.json",
        "vision/mmproj-Qwen3.8-Flash-Next-Q8_0.gguf",
        "manifests/hot-manifest-q4-vision-128k-multiprompt-r1-lru16-neutral.json",
    }
    assert required <= paths
    assert len([path for path in paths if path.endswith(".gguf")]) == 4


def test_qwen_uploader_covers_required_package_artifacts() -> None:
    package = _load(QWEN_PACKAGE)
    required = {
        artifact["path"]: artifact
        for artifact in package["artifacts"]
        if artifact.get("required", True)
    }
    uploads = build_uploads(Path("/model-root"), Path("/manifest.json"), ROOT)
    by_destination = {upload.destination: upload for upload in uploads}

    assert required.keys() <= by_destination.keys()
    for destination, artifact in required.items():
        upload = by_destination[destination]
        assert upload.expected_bytes == artifact["bytes"]
        assert upload.expected_sha256 == artifact["sha256"]
    assert "package.json" in by_destination


def test_derived_ggml_sources_ship_the_historical_mit_notice() -> None:
    notice_paths = (
        ROOT / "THIRD_PARTY_NOTICES.md",
        ROOT / "vendor/vllm-gguf-plugin/THIRD_PARTY_NOTICES.md",
        ROOT / "kernels/r9v-gfx1201/THIRD_PARTY_NOTICES.md",
    )
    for path in notice_paths:
        text = path.read_text(encoding="utf-8")
        assert "MIT License" in text
        assert "Copyright (c) 2023-2024 The ggml authors" in text

    manifest = (ROOT / "vendor/vllm-gguf-plugin/MANIFEST.in").read_text()
    project = (ROOT / "vendor/vllm-gguf-plugin/pyproject.toml").read_text()
    assert "include THIRD_PARTY_NOTICES.md" in manifest
    assert 'license-files = ["LICENSE", "THIRD_PARTY_NOTICES.md"]' in project


def test_historical_radiance_patch_text_is_not_distributed() -> None:
    runner = (
        ROOT / "vendor/vllm/vllm/v1/worker/gpu_model_runner.py"
    ).read_text(encoding="utf-8")

    assert "# RADIANCE: align the placeholder mask" not in runner
    assert "def align_draft_multimodal_mask(" in runner


def test_root_readme_does_not_overstate_release_readiness() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")

    assert "Qwen3.8 Flash Next" in readme
    assert "qwen38-mtp4" in readme
    assert "qwen38-q4-xl" in readme
    assert "MTP4" in readme


def test_root_readme_local_links_exist() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    targets = re.findall(r"\[[^]]+\]\(([^)]+)\)", readme)

    local_targets = [target for target in targets if "://" not in target]
    assert local_targets
    for target in local_targets:
        path = target.split("#", maxsplit=1)[0]
        assert (ROOT / path).exists(), target


def test_clean_clone_runbooks_are_ordered_and_fail_closed() -> None:
    qwen = (ROOT / "docs/installation.md").read_text(encoding="utf-8")
    model_card = (ROOT / "model/README.md").read_text(encoding="utf-8")

    assert qwen.index("git clone --recursive") < qwen.index("./r9v list")
    assert qwen.index("export MODEL_DIR=") < qwen.index('"$MODEL_DIR"')
    assert "./r9v fetch qwen38-mtp4" in qwen
    assert "./r9v verify qwen38-mtp4" in qwen and "-- --hash" in qwen
    assert "28,800,138,240" in qwen
    assert "amd-smi list" in qwen
    assert "./r9v start qwen38-mtp4" in qwen
    assert "Troubleshooting" in qwen
    assert "./r9v support PROFILE" in qwen

    assert "python tools/prepare_ple.py" not in model_card
    assert "docs/installation.md" in model_card


def test_image_build_requires_buildx_and_loads_local_images() -> None:
    script = (ROOT / "scripts/build-image.sh").read_text(encoding="utf-8")
    dockerfile = (
        ROOT / "vendor/vllm/docker/Dockerfile.r9v_rocm714"
    ).read_text(encoding="utf-8")

    assert "docker buildx version" in script
    assert script.count("docker buildx build --load") == 2
    assert 'R9V_VLLM_VERSION="$vllm_version"' in script
    assert "Dockerfile.r9v_rocm714" in script
    assert "rocm/dev-ubuntu-24.04:7.14.0-full@sha256:" in dockerfile
    assert "TORCH_VERSION=2.11.0" in dockerfile
    assert "TRITON_VERSION=3.6.0" in dockerfile
    assert "FLYDSL_VERSION=0.2.4" in dockerfile


def test_ci_is_read_only_pinned_and_covers_release_checks() -> None:
    workflow = (ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8")
    script = (ROOT / "scripts/ci-static.sh").read_text(encoding="utf-8")

    assert "permissions:\n  contents: read" in workflow
    assert "pull_request:" in workflow
    assert "submodules: recursive" in workflow
    assert (
        "actions/checkout@d23441a48e516b6c34aea4fa41551a30e30af803"
        in workflow
    )
    assert (
        "actions/setup-python@ece7cb06caefa5fff74198d8649806c4678c61a1"
        in workflow
    )
    assert "self-hosted" not in workflow
    for check in (
        "ruff check",
        "shellcheck",
        "./r9v validate",
        "git submodule status",
    ):
        assert check in script


def test_qwen_launch_pins_measured_rocm_dispatch_policy() -> None:
    launcher = (ROOT / "scripts/launch.sh").read_text(encoding="utf-8")

    assert "--env NCCL_ALGO=Ring" in launcher
    assert "--env NCCL_PROTO=Simple" in launcher
    assert "--env VLLM_ROCM_USE_AITER=1" in launcher
    assert "--env VLLM_ROCM_USE_AITER_UNIFIED_ATTENTION=1" in launcher
    for subsystem in (
        "LINEAR",
        "MHA",
        "MLA",
        "MOE",
        "RMSNORM",
        "FP8BMM",
        "FP4BMM",
    ):
        assert f"--env VLLM_ROCM_USE_AITER_{subsystem}=0" in launcher


def _run_qwen_launcher(tmp_path: Path, **overrides: str) -> list[str]:
    model_dir = tmp_path / "model"
    required = (
        "target/Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf",
        "target/Qwen3.8-Flash-Next-UD-IQ4_XS-00002-of-00003.gguf",
        "target/Qwen3.8-Flash-Next-UD-IQ4_XS-00003-of-00003.gguf",
        "metadata/config.json",
        "mtp/config.json",
        "mtp/model.safetensors",
        "vision/mmproj-Qwen3.8-Flash-Next-Q8_0.gguf",
        "manifests/hot-manifest-q4-vision-128k-multiprompt-r1-lru16-neutral.json",
    )
    for relative in required:
        path = model_dir / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    ple_path = tmp_path / "per_layer_token_embd.iq4_nl.bin"
    ple_path.touch()

    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    docker_args = tmp_path / "docker-args"
    fake_docker = bin_dir / "docker"
    fake_docker.write_text(
        "#!/bin/sh\n"
        "if [ \"$1\" = container ]; then exit 1; fi\n"
        "if [ \"$1\" = info ]; then printf '[]'; exit 0; fi\n"
        "if [ \"$1\" = image ] && [ \"$2\" = inspect ]; then "
        "printf 'sha256:test'; exit 0; fi\n"
        "printf '%s\\n' \"$@\" >\"$R9V_TEST_DOCKER_ARGS\"\n",
        encoding="utf-8",
    )
    fake_docker.chmod(0o755)

    env = os.environ.copy()
    env.update(
        {
            "PATH": f"{bin_dir}:{env['PATH']}",
            "R9V_MODEL_DIR": str(model_dir),
            "R9V_PLE_PATH": str(ple_path),
            "R9V_CACHE_DIR": str(tmp_path / "cache"),
            "R9V_CACHE_NAMESPACE": "test",
            "R9V_CONTAINER_NAME": "r9v-launch-test",
            "R9V_PREFLIGHT": "0",
            "R9V_TEST_DOCKER_ARGS": str(docker_args),
            **overrides,
        }
    )

    result = subprocess.run(
        [str(ROOT / "scripts/launch.sh")],
        cwd=ROOT,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    return docker_args.read_text(encoding="utf-8").splitlines()


def test_qwen_launch_forwards_each_cpu_offload_parameter(tmp_path: Path) -> None:
    args = _run_qwen_launcher(
        tmp_path,
        R9V_CPU_OFFLOAD_PARAMS="experts visual embed_tokens",
    )
    start = args.index("--cpu-offload-params")
    assert args[start : start + 4] == [
        "--cpu-offload-params",
        "experts",
        "visual",
        "embed_tokens",
    ]


def test_qwen_launch_host_network_does_not_publish_a_bridge_port(
    tmp_path: Path,
) -> None:
    args = _run_qwen_launcher(tmp_path, R9V_NETWORK_MODE="host")

    start = args.index("--network")
    assert args[start : start + 2] == ["--network", "host"]
    assert "--publish" not in args


def test_qwen_launch_builds_compilation_config_without_host_python() -> None:
    launcher = (ROOT / "scripts/launch.sh").read_text(encoding="utf-8")

    assert "compilation_config=$(python3" not in launcher
    assert "max_capture_size=$((R9V_MTP_SPEC_TOKENS + 1))" in launcher


def test_qwen_launch_resolves_cache_key_with_configured_host_python(
    tmp_path: Path,
) -> None:
    args = _run_qwen_launcher(
        tmp_path,
        R9V_CACHE_NAMESPACE="",
        R9V_HOST_PYTHON="/run/current-system/sw/bin/python3",
    )

    assert any(
        value.startswith("VLLM_CACHE_ROOT=/cache/vllm/") for value in args
    )


def test_vllm_wheel_retains_r9v_provenance_notice() -> None:
    pyproject = (ROOT / "vendor/vllm/pyproject.toml").read_text(encoding="utf-8")
    manifest = (ROOT / "vendor/vllm/MANIFEST.in").read_text(encoding="utf-8")

    assert 'license-files = ["LICENSE", "THIRD_PARTY_NOTICES.md"]' in pyproject
    assert "include THIRD_PARTY_NOTICES.md" in manifest


def test_runtime_source_pins_match_checked_out_submodules() -> None:
    runtime = json.loads(
        (ROOT / "runtimes/qwen38-flash-next-gfx1201-v1/runtime.json").read_text(
            encoding="utf-8"
        )
    )
    lock = json.loads(
        (ROOT / "release/sources.lock.json").read_text(encoding="utf-8")
    )
    expected = {
        "vllm_revision": ("vendor/vllm", "vllm"),
        "gguf_plugin_revision": (
            "vendor/vllm-gguf-plugin",
            "vllm_gguf_plugin",
        ),
        "kernel_revision": ("kernels/r9v-gfx1201", "r9v_gfx1201_kernels"),
    }

    for runtime_key, (relative_path, lock_key) in expected.items():
        revision = subprocess.check_output(
            ["git", "-C", str(ROOT / relative_path), "rev-parse", "HEAD"],
            text=True,
        ).strip()
        assert runtime["source"][runtime_key] == revision
        assert lock["code"][lock_key]["release_revision"] == revision
