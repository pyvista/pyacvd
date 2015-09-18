""" Setup.py for ACVD.py """
from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import os
import numpy

def read(*paths):
    with open(os.path.join(*paths), 'r') as f:
        return f.read()

setup(
    name='PyACVD',
    packages = ['PyACVD', 'PyACVD.Tests'], # this must be the same as the name above
#    py_modules=['PyACVD'],

    version='0.1.0', # alpha release
    description='Uniformly remeshes vtk surface meshes',
    long_description=read('description.rst'),

    # Cython directives
    cmdclass = {'build_ext': build_ext},
    ext_modules=[Extension("PyACVD.Clustering_Cython",["PyACVD/cython/Clustering_Cython.pyx"],
                           language='c++', include_dirs=[numpy.get_include()])],

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
    ],

    keywords='vtk uniform meshing remeshing',
    package_data={'PyACVD.Tests': ['StanfordBunny.stl']},
    include_dirs=[numpy.get_include()]
    
)
