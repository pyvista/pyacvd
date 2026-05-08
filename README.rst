########
 pyacvd
########

.. image:: https://img.shields.io/pypi/v/pyacvd.svg
   :target: https://pypi.org/project/pyacvd/

This module takes a surface mesh and returns a uniformly meshed surface
using voronoi clustering. This approach is loosely based on research by
S. Valette, and J. M. Chassery in `ACVD
<https://github.com/valette/ACVD>`_.

**************
 Installation
**************

Installation is straightforward using pip:

.. code::

   $ pip install pyacvd

**********
 Examples
**********

PyVista accessor (recommended, ``pyvista >= 0.48``)
===================================================

Installing ``pyacvd`` registers an ``acvd`` namespace on every
``pyvista.PolyData``, so uniform remeshing slots straight into a PyVista
pipeline. Once ``pyacvd`` is imported (or auto-discovered via PyVista's
entry-point system) you can call ``mesh.acvd.<method>(...)`` directly:

.. code:: python

   import pyacvd  # registers ``mesh.acvd``
   from pyvista import examples

   cow = examples.download_cow().triangulate()

   # one-shot uniform remesh: subdivide → cluster → rebuild
   remesh = cow.acvd.remesh(20000, subdivide=3)
   remesh.plot(color='w', show_edges=True)

The accessor never mutates the source mesh and always returns a fresh
``pv.PolyData``. Available methods:

-  ``mesh.acvd.remesh(n_clusters, subdivide=0, fast=False, ...)`` —
   one-shot uniform remesh.

-  ``mesh.acvd.clustering(weights=None, subdivide=0)`` — return a
   configured ``pyacvd.Clustering`` for fine-grained control (plotting
   clusters, mixing fast/uniform clustering, etc.).

-  ``mesh.acvd.cluster_ids(n_clusters, ...)`` — per-point cluster id
   array, useful for visualization.

-  ``mesh.acvd.subdivide(n)`` — linearly subdivide the surface.

Classic ``Clustering`` API
==========================

This example remeshes a non-uniform quad mesh into a uniform triangular
mesh. The classic API works on every supported PyVista version; the
accessor above is a thin convenience layer on top.

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
