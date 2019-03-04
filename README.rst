PyACVD
======

This module takes a vtk surface mesh (vtkPolyData) surface and returns a
uniformly meshed surface also as a vtkPolyData.  It is based on research by:
S. Valette, and J. M. Chassery in `ACVD <https://github.com/valette/ACVD>`_.

Much of this code was translated from the C++ source code available on the
above website.  Cython was used as much of the remeshing process is of an
iterative nature.  This is currently a work in progress and any bugs within
this module do not reflect the true nature of ACVD developed by S. Valette.


Installation
------------

Installation is straightforward using pip::

    $ pip install PyACVD
    
You can also visit `GitHub <https://github.com/akaszynski/PyACVD>`_ to download the latest source and install it running the following from the source directory::

    $ pip install .

You will need a working copy of VTK.  This can be obtained by either building for the source or installing it using a Python distrubution like `Anaconda <https://www.continuum.io/downloads>`_.  The other dependencies are ``numpy`` and ``cython``.

Tests
-----

You can test if your installation works by running the following tests included in the package.

.. code:: python

   from PyACVD import Tests

   # Run Stanford bunny remeshing example
   Tests.Remesh.Bunny()

   # Run non-uniform sphere remeshing example
   Tests.Remesh.Sphere()


Example
-------

This example loads a surface mesh, generates 10000 clusters, and creates a uniform mesh.

.. code:: python

    from PyACVD import Clustering
    
    # Load mesh from file.
    filename = 'file.stl'
    stlReader = vtk.vtkSTLReader() 
    stlReader.SetFileName(filename) 
    stlReader.Update()
    mesh = stlReader.GetOutput()
    
    # Create clustering object
    cobj = Clustering.Cluster(mesh)

    # Generate clusters
    cobj.GenClusters(10000)
    
    # Generate uniform mesh
    cobj.GenMesh()

    # Get mesh
    remesh = cobj.ReturnMesh()
    
    
    # The clustered original mesh and new mesh can be viewed with:
    cobj.PlotClusters()   # must run cobj.GenClusters first
    cobj.PlotRemesh()     # must run cobj.GenMesh first


Python Algorthim Restrictions
-----------------------------

The `vtkPolyData` mesh should not contain duplicate points (i.e. adjcent faces
should share identical points).  If not already done so, clean the mesh
using `vtk.vtkCleanPolyData`
    
The number of resulting points is limited by the available memory of the
host computer.  If approaching the upper limit of your available memory,
reduce the "subratio" option when generating the mesh.  As it will be
pointed out below, the coarser the mesh, the less accurate the solution.
    
The input mesh should be composed of one surface.  Unexpected behavior
may result from a multiple input meshes, though some testing has shown
that it is stable.
    
Holes in the input mesh may not be filled by the module and will result in
a non-manifold output.


Known bugs
----------

- Cluster sizes are highly dependent on initial cluster placement.
- Clusters one face (or point) large will generate highly non-uniform meshes.
    
