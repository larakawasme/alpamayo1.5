from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


setup(
    name="gemv_sparse_ext",
    ext_modules=[    
        CUDAExtension(
            name="gemv_sparse_ext",
            sources=["gemv_sparse_ext.cu"],
            extra_compile_args={
                "nvcc": ["-arch=sm_120", "-std=c++17", "-O3"],
                "cxx": ["-std=c++17", "-O3"],
            },
        ),

        CUDAExtension(
            name="gemv_sparse_no_colision_splitk",
            sources=["gemv_sparse_no_colision_splitk.cu"],
            extra_compile_args={
                "nvcc": ["-arch=sm_120", "-std=c++17", "-O3"],
                "cxx": ["-std=c++17", "-O3"],
            },
        ),
                CUDAExtension(
            name="gemv_sparse_ext_no_malloc",
            sources=["gemv_sparse_ext_no_malloc.cu"],
            extra_compile_args={
                "nvcc": ["-arch=sm_120", "-std=c++17", "-O3"],
                "cxx": ["-std=c++17", "-O3"],
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
