from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='cuda_kernels',
    version='0.0.0',
    ext_modules=[
        CUDAExtension('cuda_kernels', [
            'kernels.cpp',
            'feature_extraction.cu',
            'adjacency.cu',
            'sinkhorn_log.cu',
            'distance.cu',
            'regularize.cu',
            "update_quasi_permutation.cu",
            "hungarian/hungarian.cu",
        ],
            extra_compile_args={
            'nvcc': ["--threads", "20", "-O3", "--use_fast_math"],
        }),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
