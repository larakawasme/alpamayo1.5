import os
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

CUTLASS_ROOT = os.environ.get("CUTLASS_ROOT", "/home/lara/cutlass")

setup(
    name="gemv_cutlass_topk_ext",
    ext_modules=[CUDAExtension(
        name="gemv_cutlass_topk_ext",
        sources=["gemv_cutlass_topk_ext.cu"],
        include_dirs=["include", f"{CUTLASS_ROOT}/include"],
        extra_compile_args={
            "nvcc": ["-arch=sm_120", "-std=c++17", "-O3", "--expt-relaxed-constexpr"],
            "cxx": ["-std=c++17", "-O3"],
        },
    )],
    cmdclass={"build_ext": BuildExtension},
)
