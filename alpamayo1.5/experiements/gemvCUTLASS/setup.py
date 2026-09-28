import os
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

CUTLASS_ROOT = os.environ.get("CUTLASS_ROOT", "")
assert CUTLASS_ROOT, "Set CUTLASS_ROOT env var"

setup(
    name="gemv_ext",
    ext_modules=[
        CUDAExtension(
            name="gemv_ext",
            sources=["gemv_ext.cu"],
            include_dirs=[
                f"{CUTLASS_ROOT}/include",
                f"{CUTLASS_ROOT}/tools/util/include",
            ],
            extra_compile_args={
                "nvcc": [
                    "-arch=sm_120",           # RTX 5070 Ti = Blackwell sm_120
                    "-std=c++17",
                    "-O3",
                    "--expt-relaxed-constexpr",
                    "-U__CUDA_NO_HALF_OPERATORS__",
                ],
                "cxx": ["-std=c++17", "-O3"],
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)