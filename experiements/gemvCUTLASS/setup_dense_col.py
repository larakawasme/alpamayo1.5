from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="gemv_dense_col_ext",
    ext_modules=[CUDAExtension(
        name="gemv_dense_col_ext",
        sources=["gemv_dense_col_ext.cu"],
        extra_compile_args={
            "nvcc": ["-arch=sm_120", "-std=c++17", "-O3"],
            "cxx": ["-std=c++17", "-O3"],
        },
    )],
    cmdclass={"build_ext": BuildExtension},
)
