pyacvd
======
.. image:: https://img.shields.io/pypi/v/pyacvd.svg
    :target: https://pypi.org/project/pyacvd/


This module takes a surface mesh and returns a uniformly meshed surface using voronoi clustering.  This approach is loosely based on research by S. Valette, and J. M. Chassery in `ACVD <https://github.com/valette/ACVD>`_.


Installation
------------
Installation is straightforward using pip::

    $ pip install pyacvd


Example
-------
This example remeshes a non-uniform quad mesh into a uniform triangular mesh.

.. code:: python

    from pyvista import examples
    import pyacvd

    # download cow mesh
    cow = examples.download_cow()

    # plot original mesh
    cow.plot(show_edges=True, color='w')

.. image:: https://github.com/pyvista/pyacvd/raw/main/docs/images/cow.png
    :alt: original cow mesh

.. image:: https://github.com/pyvista/pyacvd/raw/main/docs/images/cow_zoom.png
    :alt: zoomed cow mesh

.. code:: python

    clus = pyacvd.Clustering(cow)
    # mesh is not dense enough for uniform remeshing
    clus.subdivide(3)
    clus.cluster(20000)

    # plot clustered cow mesh
    clus.plot()

.. image:: https://github.com/pyvista/pyacvd/raw/main/docs/images/cow_clus.png
    :alt: zoomed cow mesh

.. code:: python

    # remesh
    remesh = clus.create_mesh()

    # plot uniformly remeshed cow
    remesh.plot(color='w', show_edges=True)

.. image:: https://github.com/pyvista/pyacvd/raw/main/docs/images/cow_remesh.png
    :alt: zoomed cow mesh
