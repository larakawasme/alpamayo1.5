from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


setup(
    name="rough_topk_sparse_gemv",
    ext_modules=[    
        CUDAExtension(
            name="rough_topk_sparse_gemv",
            sources=["rough_topk_sparse_gemv.cu"],
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
            name="gemv_standalone_ext",
            sources=["gemv_standalone_ext.cu"],
            extra_compile_args={
                "nvcc": ["-arch=sm_120", "-std=c++17", "-O3"],
                "cxx": ["-std=c++17", "-O3"],
            },
        ),
        CUDAExtension(
            name="rough_topk_sparse_gemv_improved",
            sources=["rough_topk_sparse_gemv_improved.cu"],
            extra_compile_args={
                "nvcc": ["-arch=sm_120", "-std=c++17", "-O3"],
                "cxx": ["-std=c++17", "-O3"],
            },
        ),
        CUDAExtension(
            name="exact_topk_sparse_gemv_v3",
            sources=["exact_topk_sparse_gemv_v3.cu"],
            extra_compile_args={
                "nvcc": ["-arch=sm_120", "-std=c++17", "-O3"],
                "cxx": ["-std=c++17", "-O3"],
            },
        )
    ],

    
    cmdclass={"build_ext": BuildExtension},
)
