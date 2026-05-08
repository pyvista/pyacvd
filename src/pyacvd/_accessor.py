"""PyVista accessor for pyacvd.

Importing this module registers the ``acvd`` namespace on
:class:`pyvista.PolyData`, exposing pyacvd's uniform remeshing as
``mesh.acvd.<method>(...)``. This module is imported automatically by
``pyacvd`` when the installed PyVista is recent enough
(``>= 0.48``); on older versions the accessor simply isn't registered
and the classic :class:`pyacvd.Clustering` API still works.
"""

from typing import TYPE_CHECKING

import numpy as np
import pyvista as pv

from pyacvd.clustering import Clustering

if TYPE_CHECKING:
    from pyacvd.clustering import NDArray_FLOAT32_64


@pv.register_dataset_accessor("acvd", pv.PolyData)
class ACVDAccessor:
    """ACVD uniform remeshing exposed as ``polydata.acvd.<method>(...)``.

    The accessor wraps :class:`pyacvd.Clustering` so you can stay inside
    a PyVista pipeline. :meth:`remesh` is the one-shot entry point;
    :meth:`clustering` returns a configured :class:`~pyacvd.Clustering`
    object when you need finer control (plotting clusters, inspecting
    cluster centroids, mixing :meth:`~pyacvd.Clustering.cluster` with
    :meth:`~pyacvd.Clustering.fast_cluster`).

    The input mesh is triangulated on-the-fly when needed; the original
    is never mutated.

    Parameters
    ----------
    mesh : pyvista.PolyData
        Surface mesh the accessor operates on.

    Examples
    --------
    Uniformly remesh the Stanford bunny.

    >>> from pyvista import examples
    >>> import pyacvd  # noqa: F401  (registers the accessor)
    >>> bunny = examples.download_bunny()
    >>> remeshed = bunny.acvd.remesh(5000)
    >>> remeshed.n_points
    5000

    Inspect clusters before generating the final mesh.

    >>> import pyvista as pv
    >>> import pyacvd  # noqa: F401
    >>> cyl = pv.Cylinder().triangulate()
    >>> clus = cyl.acvd.clustering(subdivide=3)
    >>> _ = clus.cluster(500)
    >>> clus.create_mesh().n_points
    500

    """

    def __init__(self, mesh: pv.PolyData) -> None:
        """Initialize the accessor with a source mesh."""
        self._mesh = mesh

    def _triangulated(self) -> pv.PolyData:
        """Return an all-triangle copy of the wrapped mesh."""
        if self._mesh.is_all_triangles:
            return self._mesh.copy(deep=False)
        return self._mesh.triangulate()

    def clustering(
        self,
        weights: "NDArray_FLOAT32_64 | None" = None,
        *,
        subdivide: int = 0,
    ) -> Clustering:
        """Return a :class:`~pyacvd.Clustering` configured from this mesh.

        Parameters
        ----------
        weights : numpy.ndarray, optional
            Per-point weights forwarded to :class:`~pyacvd.Clustering`.
            Larger weights pull cluster density toward those points.

        subdivide : int, default: 0
            If positive, perform this many linear subdivisions on the
            (triangulated) mesh before clustering. Coarse meshes need
            subdivision before they are dense enough for uniform
            remeshing.

        Returns
        -------
        pyacvd.Clustering
            Clustering object operating on a triangulated copy of the
            wrapped mesh; the source dataset is unchanged.

        """
        clus = Clustering(self._triangulated(), weights=weights)
        if subdivide:
            clus.subdivide(subdivide)
        return clus

    def remesh(
        self,
        n_clusters: int,
        *,
        subdivide: int = 0,
        weights: "NDArray_FLOAT32_64 | None" = None,
        maxiter: int = 100,
        iso_try: int = 10,
        fast: bool = False,
        moveclus: bool = True,
        flipnorm: bool = True,
        clean: bool = True,
    ) -> pv.PolyData:
        """Uniformly remesh the surface using ACVD.

        One-shot wrapper that triangulates (if needed), optionally
        subdivides, clusters, and rebuilds a new uniform triangular
        mesh.

        Parameters
        ----------
        n_clusters : int
            Target number of points (clusters) in the remeshed surface.

        subdivide : int, default: 0
            Linear subdivisions to apply before clustering. Required
            when the input mesh is coarser than ``n_clusters``.

        weights : numpy.ndarray, optional
            Per-point weights forwarded to :class:`~pyacvd.Clustering`.

        maxiter : int, default: 100
            Maximum clustering iterations. Ignored when ``fast=True``.

        iso_try : int, default: 10
            Attempts to remove isolated clusters. Ignored when
            ``fast=True``.

        fast : bool, default: False
            Use :meth:`~pyacvd.Clustering.fast_cluster` instead of the
            uniform clustering. Faster, but cluster sizes are not
            guaranteed to be uniform.

        moveclus : bool, default: True
            Move generated points onto the original surface.

        flipnorm : bool, default: True
            Align output face normals with the source mesh.

        clean : bool, default: True
            Run :meth:`pyvista.PolyData.clean` on the result.

        Returns
        -------
        pyvista.PolyData
            A new uniformly remeshed all-triangle surface. The source
            mesh is not modified.

        See Also
        --------
        clustering : Return the underlying :class:`~pyacvd.Clustering`
            for fine-grained control.
        cluster_ids : Compute cluster ids without rebuilding the
            surface.

        Examples
        --------
        >>> import pyvista as pv
        >>> import pyacvd  # noqa: F401
        >>> mesh = pv.Cylinder().triangulate()
        >>> uniform = mesh.acvd.remesh(500, subdivide=3)
        >>> uniform.n_points
        500

        """
        clus = self.clustering(weights=weights, subdivide=subdivide)
        if fast:
            clus.fast_cluster(n_clusters)
        else:
            clus.cluster(n_clusters, maxiter=maxiter, iso_try=iso_try)
        return clus.create_mesh(moveclus=moveclus, flipnorm=flipnorm, clean=clean)

    def subdivide(self, nsub: int) -> pv.PolyData:
        """Linearly subdivide the surface ``nsub`` times.

        Parameters
        ----------
        nsub : int
            Number of subdivisions.

        Returns
        -------
        pyvista.PolyData
            A new subdivided all-triangle surface.

        Examples
        --------
        >>> import pyvista as pv
        >>> import pyacvd  # noqa: F401
        >>> mesh = pv.Cylinder().triangulate()
        >>> mesh.acvd.subdivide(2).n_points > mesh.n_points
        True

        """
        return self.clustering(subdivide=nsub).mesh

    def cluster_ids(
        self,
        n_clusters: int,
        *,
        subdivide: int = 0,
        weights: "NDArray_FLOAT32_64 | None" = None,
        maxiter: int = 100,
        iso_try: int = 10,
        fast: bool = False,
    ) -> np.ndarray:
        """Compute cluster ids without building the remeshed surface.

        Useful for visualizing clusters on the (subdivided) source mesh
        or driving downstream analysis.

        Parameters
        ----------
        n_clusters : int
            Target number of clusters.

        subdivide : int, default: 0
            See :meth:`remesh`.

        weights : numpy.ndarray, optional
            See :meth:`remesh`.

        maxiter : int, default: 100
            See :meth:`remesh`.

        iso_try : int, default: 10
            See :meth:`remesh`.

        fast : bool, default: False
            See :meth:`remesh`.

        Returns
        -------
        numpy.ndarray
            ``int32`` array of length ``n_points`` of the (possibly
            subdivided) mesh, with one cluster id per point. ``-1``
            marks points that were not assigned to a cluster.

        """
        clus = self.clustering(weights=weights, subdivide=subdivide)
        if fast:
            return clus.fast_cluster(n_clusters)
        return clus.cluster(n_clusters, maxiter=maxiter, iso_try=iso_try)


__all__ = ["ACVDAccessor"]
