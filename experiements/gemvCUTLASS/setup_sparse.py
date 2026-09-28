from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


setup(
    name="gemv_sparse_ext",
    ext_modules=[    
        CUDAExtension(
            name="gemv_sparse_ext",
            sources=["cutlass_library_cuda_files/gemv_cutlass_rowmajor_ext.cu"],
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
            name="gemv_sparse_ext_rough_topk_improved",
            sources=["gemv_sparse_ext_rough_topk_improved.cu"],
            extra_compile_args={
                "nvcc": ["-arch=sm_120", "-std=c++17", "-O3"],
                "cxx": ["-std=c++17", "-O3"],
            },
        ),
        CUDAExtension(
            name="gemv_sparse_extt_true_topk",
            sources=["gemv_sparse_extt_true_topk.cu"],
            extra_compile_args={
                "nvcc": ["-arch=sm_120", "-std=c++17", "-O3"],
                "cxx": ["-std=c++17", "-O3"],
            },
        )
    ],

    
    cmdclass={"build_ext": BuildExtension},
)
