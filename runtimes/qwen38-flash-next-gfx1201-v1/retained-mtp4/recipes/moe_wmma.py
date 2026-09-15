# SPDX-License-Identifier: Apache-2.0
"""Build the gfx12 int8 WMMA grouped MoE prefill extension against the image's pinned GGUF headers."""
import os
from pathlib import Path

from torch.utils.cpp_extension import load

root = Path(__file__).resolve().parent
source = root.parent / "kernels" / "r9v_moe_wmma.cu"
build = Path(os.environ.get("R9V_MOE_WMMA_BUILD", root / "build"))
output = Path(os.environ.get("R9V_MOE_WMMA_SO", build / "r9v_moe_wmma.so"))
headers = Path(os.environ.get("VLLM_GGUF_PLUGIN_CSRC", "/opt/r9v/lib/python3.12/site-packages/vllm_gguf_plugin/csrc"))
if not (headers / "gguf" / "ggml-common_hip.h").is_file():
    raise SystemExit(f"missing GGUF HIP headers under {headers}")
build.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("PYTORCH_ROCM_ARCH", "gfx1201")
load(name="r9v_moe_wmma", sources=[str(source)],
     extra_include_paths=[str(headers), str(headers / "gguf")],
     extra_cflags=["-O3", "-std=c++17"],
     extra_cuda_cflags=["-O3", "-std=c++17", "-DUSE_ROCM", "--offload-arch=gfx1201"],
     build_directory=str(build), verbose=True)
built = build / "r9v_moe_wmma.so"
if output != built:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(built.read_bytes())
print(f"built {output}")
