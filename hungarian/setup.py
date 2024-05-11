from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='cuda_hungarian',
    version='0.0.0',
    ext_modules=[
        CUDAExtension('cuda_hungarian', [
            'hungarian_binds.cpp',
            'hungarian.cu',
        ],
        extra_compile_args={
            'nvcc': ["--threads", "20",
                     "-gencode", "arch=compute_87,code=sm_87",
                     "-gencode", "arch=compute_86,code=sm_86",
                     "-std=c++20"
            ]        }),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })