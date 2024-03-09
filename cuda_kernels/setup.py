from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='cuda_kernels',
    version='0.0.0',
    ext_modules=[
        CUDAExtension('cuda_kernels', [
            'kernels.cpp',
            'sinkhorn_log.cu',
        ]),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
