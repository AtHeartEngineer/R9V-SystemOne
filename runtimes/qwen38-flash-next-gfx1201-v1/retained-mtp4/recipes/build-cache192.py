from pathlib import Path
from torch.utils.cpp_extension import load
root=Path(__file__).resolve().parent
build=root/'build';build.mkdir(exist_ok=True)
load(name='r9v_cache192',sources=[str(root/'cache192_prepare.cu')],extra_cflags=['-O3','-std=c++17'],extra_cuda_cflags=['-O3','-std=c++17','-DUSE_ROCM','--offload-arch=gfx1201'],build_directory=str(build),verbose=True)
