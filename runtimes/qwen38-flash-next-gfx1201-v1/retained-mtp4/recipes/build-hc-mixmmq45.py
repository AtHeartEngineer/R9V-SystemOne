from pathlib import Path
import importlib.util,os
from torch.utils.cpp_extension import load
root=Path(__file__).resolve().parent
headers=Path(next(iter(importlib.util.find_spec('vllm_gguf_plugin').submodule_search_locations)))/'csrc'
build=root/'build';build.mkdir(exist_ok=True)
load(name='r9v_hc_mixmmq45',sources=[str(root/'hc_mixmmq45.cu')],extra_include_paths=[str(headers),str(headers/'gguf')],extra_cflags=['-O3','-std=c++17'],extra_cuda_cflags=['-O3','-std=c++17','-DUSE_ROCM','--offload-arch=gfx1201'],build_directory=str(build),verbose=True)
