from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='sinkhorn_cuda',
    version='0.0.0',
    ext_modules=[
        CUDAExtension('sinkhorn_cuda', [
            'sinkhorn.cpp',
            'kernel.cu',
        ]),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
