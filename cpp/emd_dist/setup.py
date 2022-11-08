from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='emd',
    version='1.0.0',
    ext_modules=[
        CUDAExtension('emd', [
            'emd.cpp',
            'emd_cuda.cu',
        ]),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })