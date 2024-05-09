from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='cuda_kernels',
    version='0.0.0',
    ext_modules=[
        CUDAExtension('cuda_kernels', [
            'kernels.cpp',
            'hungarian.cu',
            'feature_extraction.cu',
            'adjacency.cu',
            'sinkhorn_log.cu',
            'distance.cu',
            'regularize.cu',
        ],
        extra_compile_args={
            'nvcc': ["--threads", "20",
                     "-gencode", "arch=compute_87,code=sm_87",
                     "-gencode", "arch=compute_86,code=sm_86",
                     "--library-path=/CuLAP/lib/libculap.a", "--library=libculap",],
        }),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
