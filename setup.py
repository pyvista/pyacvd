"""Setup for pyacvd"""
from io import open as io_open
import os

from Cython.Build import cythonize
import numpy as np
from setuptools import Extension, setup

# Get version from version info
__version__ = None
version_file = os.path.join(os.path.dirname(__file__), "pyacvd", "_version.py")
with io_open(version_file, mode="r") as fd:
    exec(fd.read())


def read(*paths):
    with open(os.path.join(*paths), "r") as fid:
        return fid.read()


setup(
    name="pyacvd",
    packages=["pyacvd"],
    version=__version__,
    description="Uniformly remeshes surface meshes",
    long_description=read("README.rst"),
    long_description_content_type="text/x-rst",
    ext_modules=cythonize(
        [
            Extension(
                "pyacvd._clustering",
                ["pyacvd/cython/_clustering.pyx"],
                language="c++",
                include_dirs=[np.get_include()],
            )
        ]
    ),
    url="https://github.com/pyvista/pyacvd",
    author="Alex Kaszynski",
    author_email="akascap@gmail.com",
    license="MIT",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "License :: OSI Approved :: MIT License",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: POSIX",
        "Operating System :: MacOS",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    python_requires=">=3.7",
    install_requires=["pyvista>=0.37.0", "numpy", "scipy"],
    keywords="vtk uniform meshing remeshing, acvd",
)
