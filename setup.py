""" Setup.py for ACVD.py """

import os
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext as _build_ext
from distutils.core import setup
from Cython.Build import cythonize

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
    name='PyACVD',
    packages = ['PyACVD', 'PyACVD.Tests'], # this must be the same as the name above

    version='0.1.1',
    description='Uniformly remeshes vtk surface meshes',
    long_description=read('README.rst'),

    # Cython directives
    cmdclass = {'build_ext': build_ext},
    ext_modules= cythonize([Extension("PyACVD.Clustering_Cython",
                           ["PyACVD/cython/Clustering_Cython.pyx"],
                           language='c++')]),

    url='https://github.com/akaszynski/PyACVD',
    author='Alex Kaszynski',
    author_email='akascap@gmail.com',

    # Choose your license
    license='CECILL-B',
    classifiers=[
        'Development Status :: 3 - Alpha',

        # Target audience
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Information Analysis',

        # CECIL-B license
        'License :: OSI Approved :: CEA CNRS Inria Logiciel Libre License, version 2.1 (CeCILL-2.1)',

        # Tested only on Python 2.7
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
    ],

    keywords='vtk uniform meshing remeshing, acvd',
    package_data={'PyACVD.Tests': ['StanfordBunny.ply']}
    
)
