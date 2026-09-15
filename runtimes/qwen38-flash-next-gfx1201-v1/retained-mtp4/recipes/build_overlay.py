#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Build the retained MTP4 extensions into a clean, deterministic output tree.

The script intentionally performs no model or runtime work.  It requires the
ROCm/PyTorch toolchain supplied by the pinned base image and a GGUF plugin
checkout whose headers match that image.
"""
from __future__ import annotations

import argparse
import importlib.util
import os
from pathlib import Path

from torch.utils.cpp_extension import load


MODULES = (
    ("qwen38_tiered_iq_moe_hip", "tiered_iq_moe_hip.cu", True, "auto,u2,u5,u10,reuse3,reuse3v2"),
    ("qwen38_dense_mmvq_hip", "dense_mmvq_hip.cu", True, None),
    ("r9v_q8_mmvq5", "q8_mmvq5.cu", True, None),
    ("r9v_hc_mmq4", "hc_mmq4.cu", True, None),
    ("r9v_hc_mixmmq4", "hc_mixmmq4.cu", True, None),
    ("r9v_hc_mixmmq45", "hc_mixmmq45.cu", True, None),
    ("r9v_cache80", "cache80_prepare.cu", False, None),
    ("r9v_cache192", "cache192_prepare.cu", False, None),
    ("r9v_draft_w2_rank", "draft_w2_rank.cu", True, None),
    ("r9v_draft_indexed_q6", "draft_indexed_q6.cu", True, None),
    ("r9v_moe_wmma", "r9v_moe_wmma.cu", True, None),
)


def plugin_csrc() -> Path:
    configured = os.environ.get("VLLM_GGUF_PLUGIN_CSRC")
    if configured:
        result = Path(configured).expanduser().resolve()
    else:
        spec = importlib.util.find_spec("vllm_gguf_plugin")
        locations = () if spec is None else spec.submodule_search_locations or ()
        if not locations:
            raise SystemExit("set VLLM_GGUF_PLUGIN_CSRC to the pinned GGUF csrc")
        result = Path(next(iter(locations))) / "csrc"
    if not (result / "gguf/ggml-common_hip.h").is_file():
        raise SystemExit(f"missing GGUF HIP headers under {result}")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=Path(__file__).resolve().parents[1] / "kernels")
    parser.add_argument("--output", type=Path, default=Path("/opt/r9v/kernels"))
    parser.add_argument("--build", type=Path, default=Path("/opt/r9v/build/retained-mtp4"))
    args = parser.parse_args()
    source = args.source.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    args.build.mkdir(parents=True, exist_ok=True)
    headers = plugin_csrc()
    for name, filename, needs_gguf, variants in MODULES:
        source_path = source / filename
        if not source_path.is_file():
            raise SystemExit(f"missing retained source: {source_path}")
        include = [str(headers), str(headers / "gguf")] if needs_gguf else []
        cuda_flags = ["-O3", "-std=c++17", "-DUSE_ROCM", "--offload-arch=gfx1201"]
        (args.build / name).mkdir(parents=True, exist_ok=True)
        if variants:
            # The selected MTP4 build emitted only auto + reuse3v2 (1 | 32).
            # Keep this exact specialization mask; enabling every historical
            # variant changes the compiled artifact and its performance claim.
            cuda_flags.append("-DQWEN38_TIERED_IQ_MOE_SPECIALIZED_VARIANTS=33")
        load(
            name=name,
            sources=[str(source_path)],
            extra_include_paths=include,
            extra_cflags=["-O3", "-std=c++17"],
            extra_cuda_cflags=cuda_flags,
            build_directory=str(args.build / name),
            verbose=True,
        )
        built = args.build / name / f"{name}.so"
        if not built.is_file():
            raise SystemExit(f"compiler did not produce {built}")
        target = args.output / built.name
        target.write_bytes(built.read_bytes())
        target.chmod(0o755)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
