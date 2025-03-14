from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np
from glob import glob
setup(
    name='cython_merge function',
    ext_modules=cythonize(
        Extension(
            'cython_merge',
            glob('*.pyx'),
            include_dirs=[np.get_include(),"."],
            extra_compile_args=["-std=c++11"],
            extra_link_args=["-std=c++11"],
        ),
        language_level = "3",
    ),
    zip_safe=False,
)
