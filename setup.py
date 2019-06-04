"""Setup for pyacvd"""
import os
from io import open as io_open

from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext as _build_ext

# Get version from version info
__version__ = None
version_file = os.path.join(os.path.dirname(__file__), 'pyacvd', '_version.py')
with io_open(version_file, mode='r') as fd:
    # execute file from raw string
    exec(fd.read())


class build_ext(_build_ext):
    """ build class that includes numpy directory """
    def finalize_options(self):
        _build_ext.finalize_options(self)
        # Prevent numpy from thinking it is still in its setup process:
        __builtins__.__NUMPY_SETUP__ = False
        import numpy
        self.include_dirs.append(numpy.get_include())
        

def read(*paths):
    with open(os.path.join(*paths), 'r') as f:
        return f.read()


setup(
    name='pyacvd',
    packages = ['pyacvd'],
    version=__version__,
    description='Uniformly remeshes surface meshes',
    long_description=read('README.rst'),

    # Cython directives
    cmdclass = {'build_ext': build_ext},
    ext_modules=[Extension("pyacvd._clustering",
                           ["pyacvd/cython/_clustering.pyx"],
                           language='c++')],

    url='https://github.com/akaszynski/pyacvd',
    author='Alex Kaszynski',
    author_email='akascap@gmail.com',
    license='MIT',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Information Analysis',
        'License :: OSI Approved :: MIT License',
        'Operating System :: Microsoft :: Windows',
        'Operating System :: POSIX',
        'Operating System :: MacOS',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
    ],

    python_requires='>=2.7,!=3.0.*,!=3.1.*,!=3.2.*,!=3.3.*,!=3.4.*',
    install_requires=['pyvista', 'scipy'],
    keywords='vtk uniform meshing remeshing, acvd',
    
)
