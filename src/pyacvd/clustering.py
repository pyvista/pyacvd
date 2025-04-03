"""Point based clustering module"""

import logging
import warnings
from typing import Any, Optional, Sequence, Tuple, TypeVar, Union, cast

import numpy as np
import numpy.typing as npt
import pyvista as pv
from numpy.typing import NDArray
from pykdtree.kdtree import KDTree
from pyvista import ID_TYPE
from pyvista.core.pointset import PolyData
from vtkmodules.vtkCommonDataModel import vtkCellArray

from pyacvd import _clustering

NumpyNumArray = npt.NDArray[np.number]
NumpyIntArray = npt.NDArray[np.int_]
Number = Union[int, float]
Vector = Union[NumpyNumArray, Sequence[Number]]
Matrix = Union[NumpyNumArray, Sequence[Vector]]


NDArray_INT32 = NDArray[np.int32]
NDArray_INT32_64 = npt.NDArray[Union[np.int32, np.int64]]
NDArray_FLOAT32 = NDArray[np.float32]
NDArray_FLOAT64 = NDArray[np.float64]
NDArray_FLOAT32_64 = Union[NDArray_FLOAT32, NDArray_FLOAT64]
NDArray_UINT32 = npt.NDArray[np.uint32]

U = TypeVar("U", np.int32, np.int64)
T = TypeVar("T", np.float32, np.float64)

LOG = logging.getLogger(__name__)

MAX_THREADS = 4


def point_normals(mesh: PolyData) -> NDArray[T]:
    """
    Return point normals of an all triangle mesh.

    This is about 10x faster than :func:`pyvista.PolyData.compute_normals`.

    Parameters
    ----------
    mesh : pyvista.PolyData
        All triangular surface mesh.

    Returns
    -------
    numpy.ndarray
        Normalized point normal array.

    Examples
    --------
    >>> import femorph_ext
    >>> import pyvista as pv
    >>> mesh = pv.Icosphere(nsub=8)
    >>> pnorm = utilities.point_normals(mesh)
    >>> pnorm
    array([[ 0.        , -0.52573908,  0.85064588],
           [ 0.        , -0.52573908, -0.85064588],
           [ 0.        ,  0.52573908, -0.85064588],
           ...,
           [-0.57508381,  0.57767968, -0.57927955],
           [-0.57928493,  0.57508047, -0.57767761],
           [-0.57767968,  0.57927955, -0.57508381]])

    """
    return _clustering.point_normals(mesh.points, _tri_faces_from_poly(mesh))


def _tri_faces_from_poly(mesh: PolyData) -> NDArray[U]:
    """Return the triangle faces from a polydata."""
    return cast(NDArray[U], mesh._connectivity_array.reshape(-1, 3))


def unique_edges(neigh: NDArray_INT32, neigh_off: NDArray_INT32) -> NDArray_INT32:
    """
    Identify unique edges from a list of neighbors and their offsets.

    This function processes an array of neighbors and their corresponding
    offsets to construct a unique set of edges. The function ensures that each
    edge is represented only once by filtering out duplicate connections.

    Parameters
    ----------
    neigh : np.ndarray
        An array of integers where each element is an index into a set of
        points, representing the neighboring points for each node.
    neigh_off : np.ndarray
        An array of integers that act as offsets into the ``neigh`` array,
        indicating the start of the neighbor list for each node. The difference
        between successive elements in `neigh_off` gives the count of neighbors
        for a node.

    Returns
    -------
    np.ndarray
        A 2D array where each row represents an edge. Each edge is denoted by
        two integers, the indices of the nodes it connects. All edges are
        unique.

    """
    return _clustering.unique_edges(neigh, neigh_off)


class Clustering:
    """Uniform point clustering based on ACVD.

    Parameters
    ----------
    mesh : pyvista.PolyData
        Mesh to cluster.

    Examples
    --------
    Perform a uniform recluster of the stanford bunny.

    >>> from pyvista import examples
    >>> import pyacvd
    >>> mesh = examples.download_bunny()
    >>> clus = pyacvd.Clustering()
    >>> clus.cluster()
    >>> remeshed = clus.create_mesh()

    """

    def __init__(self, mesh: pv.PolyData, weights: Optional[NDArray_FLOAT32_64] = None) -> None:
        """Initialize neighbors."""
        self.mesh = mesh
        self.clusters: Optional[NDArray_INT32] = None
        self.nclus: Optional[int] = None

        if not self.mesh.is_all_triangles:
            raise ValueError(
                "Input mesh must be composed of all triangles. Hint: `mesh.triangulate` first."
            )

        # Compute point weights and weighted points
        area, wcent = point_weights(self.mesh, additional_weights=weights, force_double=True)
        self.area = cast(NDArray_FLOAT64, area)
        self.area[self.area == 0] = 1e-10

        self.wcent = cast(NDArray_FLOAT64, wcent)

        # neighbors and edges
        self.neigh, self.neigh_off = _clustering.neighbors_from_trimesh(
            mesh.n_points, _tri_faces_from_poly(mesh)
        )
        self.edges = unique_edges(self.neigh, self.neigh_off)

    def cluster(
        self,
        nclus: int,
        maxiter: int = 100,
        debug: bool = False,
        iso_try: int = 10,
        init_only: bool = False,
    ) -> NDArray_INT32:
        """
        Generate clusters.

        Parameters
        ----------
        nclus : int
            Number of clusters to generate.
        maxiter : int, default: 100
            Maximum number of iterations to attempt to uniformly distribute the
            clusters.
        debug : bool, default: False
            Enable debug output.
        iso_try : int, default: 10
            Number of times to attempt to remove isolated clusters.
        init_only : bool, default: False
            Only perform initial clustering. Resulting clusters will be
            non-uniformly distributed.

        """
        # do not allow number of clusters to exceed the number of points
        invalid = False
        if nclus > self.area.size:
            self.nclus = self.area.size
            self.clusters = np.arange(self.nclus, dtype=np.int32)
        else:
            self.clusters, invalid, self.nclus = _clustering.cluster(
                self.neigh,
                self.neigh_off,
                nclus,
                self.area,
                self.wcent,
                self.edges,
                maxiter,
                iso_try,
                init_only,
            )
        return self.clusters

    def fast_cluster(self, nclus: int) -> NDArray_INT32:
        """
        Rapidly cluster points at the expense of uniform cluster size.

        Parameters
        ----------
        nclus : int
            Number of clusters to generate.

        """
        LOG.info("Fast clustering using %d clusters", nclus)
        self.clusters, self.nclus = _clustering.fast_cluster(
            self.neigh, self.neigh_off, nclus, self.area, self.wcent, self.edges
        )
        return self.clusters

    def subdivide(self, nsub: int) -> None:
        """Perform a linear subdivision of the mesh.

        Parameters
        ----------
        nsub : int
            Number of subdivisions
        """
        self.mesh.copy_from(_subdivide(self.mesh, nsub))
        self._update_data()

    def _update_data(self, weights: Optional[NDArray_FLOAT32_64] = None) -> None:
        """Recompute neighbors and weights."""
        # Compute point weights and weighted points
        if weights is not None:
            weights = weights.astype(np.float64, copy=False)
        self.area, self.wcent = point_weights(
            self.mesh, additional_weights=weights, force_double=True
        )
        self.area[self.area == 0] = 1e-10

        # neighbors and edges
        self.neigh, self.neigh_off = _clustering.neighbors_from_trimesh(
            self.mesh.n_points, _tri_faces_from_poly(self.mesh)
        )
        self.edges = unique_edges(self.neigh, self.neigh_off)

    def plot(self, random_color: bool = True, **kwargs: Any) -> Any:
        """
        Plot clusters if available.

        Parameters
        ----------
        random_color : bool, default: True
            Plots clusters with a random color rather than a color
            based on the order of creation.
        **kwargs : keyword arguments, optional
            See :func:`pyvista.plot`.

        Returns
        -------
        cpos : list
            Camera position. See :func:`pyvista.plot`.
        """
        if self.clusters is None or self.nclus is None:
            LOG.warning("No clusters to plot.")
            return self.mesh.plot(notebook=False, **kwargs)

        # Setup color
        if random_color:
            rand_color = np.random.random(self.nclus)
        else:
            rand_color = np.linspace(0, 1, self.nclus, dtype=np.float64)
        colors = rand_color[self.clusters]

        # Set color range depending if null clusters exist
        if np.any(colors == -1):
            colors[colors == -1] = -0.25
            rng = [-0.25, 1]
        else:
            rng = [0, 1]

        return self.mesh.plot(notebook=False, scalars=colors, rng=rng, **kwargs)

    def create_mesh(self, moveclus: bool = True, flipnorm: bool = True) -> pv.PolyData:
        """
        Generate mesh from clusters.

        Parameters
        ----------
        moveclus : bool, default: True
            Move the created points to the surface of the original mesh.
        flipnorm : bool, default: True
            Ensure the normals of the clustered mesh match the normals of the
            original mesh.

        """
        if self.clusters is None or self.nclus is None:
            raise RuntimeError("Clusters have not been initialized.")

        LOG.info("Generating mesh from clusters")
        if flipnorm or moveclus:
            cnorm = self.cluster_norm
        else:
            cnorm = None

        # Generate mesh
        self.remesh = create_mesh(
            self.mesh, self.area, self.clusters, cnorm, self.nclus, moveclus, flipnorm
        )
        return self.remesh

    @property
    def cluster_norm(self) -> NDArray_FLOAT64:
        """Return cluster norms."""
        if self.clusters is None or self.nclus is None:
            raise RuntimeError("Clusters have not been initialized.")

        # Normals of original mesh
        n = point_normals(self.mesh)

        # Compute normalized mean cluster normals
        cnorm = np.empty((self.nclus, 3))
        cnorm[:, 0] = np.bincount(self.clusters, weights=n[:, 0] * self.area)
        cnorm[:, 1] = np.bincount(self.clusters, weights=n[:, 1] * self.area)
        cnorm[:, 2] = np.bincount(self.clusters, weights=n[:, 2] * self.area)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cnorm /= ((cnorm * cnorm).sum(1) ** 0.5).reshape((-1, 1))

        return cnorm

    @property
    def cluster_centroid(self) -> NDArray_FLOAT32_64:
        """Compute an area normalized value for each cluster."""
        if self.clusters is None:
            raise RuntimeError("Clusters have not been initialized.")

        wval = self.mesh.points * self.area.reshape(-1, 1)
        cval: NDArray_FLOAT32_64 = np.vstack(
            (
                np.bincount(self.clusters, weights=wval[:, 0]),
                np.bincount(self.clusters, weights=wval[:, 1]),
                np.bincount(self.clusters, weights=wval[:, 2]),
            )
        ) / np.bincount(self.clusters, weights=self.area)
        return cval.T


def point_weights(
    mesh: PolyData,
    additional_weights: Optional[NDArray[T]] = None,
    num_threads: int = 2,
    force_double: bool = False,
) -> Tuple[NDArray[T], NDArray[T]]:
    """
    Return weighted points.

    Point weights are equal to the sum of the face areas contributing to the point.

    Parameters
    ----------
    mesh : pyvista.PolyData
        All triangular surface mesh.
    additional_weights : np.ndarray, optional
        Additional weights to supply to each point. Accounted for in ``wpoints``.
    num_threads : int, default: True
        Number of threads. Recommended to use a low value due to cache thrashing.
    force_double : bool, default: False
        Force computation in double precision.

    Returns
    -------
    pweights : numpy.ndarray
        Point weights array, shaped ``(n, 1)``.
    wpoints : numpy.ndarray
        Weighted points, shaped ``(n, 3)``.

    """
    points: NDArray[T] = mesh.points
    if force_double:
        points = points.astype(np.float64, copy=False)
    faces = _tri_faces_from_poly(mesh)
    if additional_weights is None:
        additional_weights = np.empty(0, dtype=points.dtype)
    if additional_weights.dtype != points.dtype:
        additional_weights = additional_weights.astype(points.dtype)

    return _clustering.weighted_points(points, faces, additional_weights, num_threads)


def polydata_from_faces(points: NDArray_FLOAT32_64, faces: NDArray_INT32_64) -> PolyData:
    """
    Generate a polydata from a faces array containing no padding and all triangles.

    Parameters
    ----------
    points : np.ndarray
        Points array.
    faces : np.ndarray
        ``(n, 3)`` faces array.

    Returns
    -------
    PolyData
        New mesh.

    """
    if faces.ndim != 2:
        raise ValueError("Expected a two dimensional face array.")

    pdata = PolyData()
    pdata.points = points

    carr = vtkCellArray()
    offset = np.arange(0, faces.size + 1, faces.shape[1], dtype=ID_TYPE)
    carr.SetData(pv.numpy_to_idarr(offset, deep=True), pv.numpy_to_idarr(faces, deep=True))
    pdata.SetPolys(carr)
    return pdata


def face_centroid_arrays(points: NDArray[T], faces: NDArray[U]) -> NDArray[T]:
    """
    Return the centroid of each face of a triangular mesh.

    Parameters
    ----------
    points : np.ndarray
        Points array.
    faces : np.ndarray
        Faces array.

    Returns
    -------
    numpy.ndarray
        Face centroids.

    """
    return _clustering.face_centroid(points, faces)


def neighbors(
    source: NDArray_FLOAT32_64,
    target: NDArray_FLOAT32_64,
    n: int = 1,
    eps: float = 0.0,
    distance_upper_bound: Optional[float] = None,
    sqr_dists: bool = False,
) -> Tuple[NDArray_FLOAT32_64, NDArray_UINT32]:
    """
    Return the indices of the nearest n neighbors between source and target.

    Parameters
    ----------
    source : numpy.ndarray
        The points being searched against.
    target : numpy.ndarray
        The points searching from.
    n : int, default: 1
        Number of neighbors to find.
    eps : float, default: 0.0
        Accuracy.
    distance_upper_bound : float, optional
        Return only neighbors within this distance. Vastly speeds up search.
    sqr_dists : bool, default: False
        Internally pykdtree works with squared distances. Determines if the
        squared or Euclidean distances are returned. If you don't need the
        distances, set this to ``True`` to slightly improve performance.

    Returns
    -------
    numpy.ndarray
        Distances between a target point and its neighbors (squared if
        ``sqr_dists=True``.
    numpy.ndarray
        Indices between a target point and its neighbors.

    """
    if n > source.shape[0]:
        # warnings.warn(
        #     f"Number of requested neighbors {n} exceeds number of possible "
        #     f"neighbors {source.shape[0]}"
        # )
        n = source.shape[0]

    tree = KDTree(source)

    # source and target datatypes must match
    if source.dtype != target.dtype:
        target = target.astype(source.dtype)

    # target array must be contiguous
    if not target.flags.c_contiguous:
        target = np.ascontiguousarray(target)

    d, ind = tree.query(
        target,
        n,
        eps=eps,
        distance_upper_bound=distance_upper_bound,
        sqr_dists=sqr_dists,
    )

    return d, ind


def ray_trace(
    source_v: NDArray[T],
    source_n: NDArray[T],
    target_v: NDArray[T],
    target_f: NDArray[U],
    neigh: NDArray_UINT32,
    no_inf: bool = True,
    num_threads: int = 8,
    out_of_bounds_idx: Optional[int] = None,
    in_vector: bool = False,
) -> Tuple[NDArray[T], NDArray[T]]:
    if out_of_bounds_idx is None:
        out_of_bounds_idx = 0
    if out_of_bounds_idx == -1:
        raise ValueError
    dist, ind = _clustering.ray_trace(
        source_v,
        source_n,
        target_v,
        target_f,
        neigh,
        no_inf,
        num_threads,
        out_of_bounds_idx,
        in_vector,
    )
    return dist, ind


def _unique_row_indices(a: Matrix) -> NDArray_INT32_64:
    """Return indices of unique rows."""
    arr = np.asarray(a)
    b = np.ascontiguousarray(arr).view(np.dtype((np.void, arr.dtype.itemsize * arr.shape[1])))
    _, idx = np.unique(b, return_index=True)
    return idx


def face_normals_array(points: NDArray[T], faces: NDArray[U]) -> NDArray[T]:
    """
    Return the normals of faces in a mesh or a set of vertices and faces.

    Parameters
    ----------
    points : numpy.ndarray
        Vertex array.
    faces : numpy.ndarray
        All triangular face array. Shaped ``(n, 3)``.

    Returns
    -------
    numpy.ndarray
        Normalized face normal array.

    """
    return _clustering.face_normals(points, faces)


def create_mesh(
    mesh: pv.PolyData,
    area: NDArray_FLOAT64,
    clusters: NDArray_INT32,
    cnorm: Optional[NDArray_FLOAT64],
    nclus: int,
    moveclus: bool,
    flipnorm: bool = True,
) -> pv.PolyData:
    """
    Generate a new mesh given cluster data and other parameters.

    Parameters
    ----------
    mesh : pyvista.PolyData
        Input mesh object that needs to be reshaped.
    area : numpy.ndarray
        An array representing the area of each face of the mesh.
    clusters : numpy.ndarray
        An array representing clusters in the mesh.
    cnorm : numpy.ndarray
        An array representing the centroids of the clusters.
    nclus : int
        The number of clusters.
    moveclus : bool
        A boolean flag to move cluster centers to the surface of their
        corresponding cluster.
    flipnorm : bool, default: True
        If ``True``, flip the normals of the faces.

    Returns
    -------
    object
        Returns a PolyData object representing the new generated mesh.

    Notes
    -----
    This function uses the ray-tracing algorithm to compute intersections, and
    applies the logic of clustering, centroid calculation, face normal
    calculation, and the creation of a sparse matrix. The function supports the
    flipping of face normals and moving of cluster centers based on provided
    flags.

    It also involves removing duplicate faces, computing mean normals of
    clusters, comparing with the normals of each face, and reversing the order
    if required.  Finally, it creates a new surface using vtk and returns it.
    """
    faces = mesh._connectivity_array.reshape(-1, 3)
    points = mesh.points.astype(np.float64, copy=False)

    # Compute centroids
    ccent = np.ascontiguousarray(_cluster_centroid(points, area, clusters)).astype(
        np.float64, copy=False
    )

    # Update cluster centroid locations based on intersections
    if moveclus and cnorm is not None:
        tgt_nbr = min(faces.shape[0] - 1, 1000)
        fcent = face_centroid_arrays(points, faces)
        _, ind = neighbors(fcent, ccent, tgt_nbr, sqr_dists=True)
        dist, _ = ray_trace(ccent, cnorm, points, faces, ind, num_threads=MAX_THREADS)
        ccent += cnorm * dist.reshape((-1, 1))

    # Ignore faces that do not connect to different clusters
    f_clus = np.sort(clusters[faces], axis=1)
    mask = np.all(np.diff(f_clus, axis=1) != 0, axis=1)
    f = f_clus[mask]

    v = ccent
    f = f[_unique_row_indices(f)]  # Remove duplicate faces

    # Mean normals of clusters each face is build from
    if flipnorm and cnorm is not None:
        adjcnorm = cnorm[f].sum(1)
        adjcnorm /= np.linalg.norm(adjcnorm, axis=1).reshape(-1, 1)

        # and compare this with the normals of each face
        newnorm = face_normals_array(v, f)
        agg = (adjcnorm * newnorm).sum(1)  # dot product

        # If the dot is negative, reverse the order of those faces
        flip_ind = np.flatnonzero(agg < 0.0)
        f[flip_ind] = f[flip_ind, ::-1]

    return polydata_from_faces(v, f)


def _cluster_centroid(
    cent: NDArray_FLOAT32_64, area: NDArray_FLOAT32_64, clusters: NDArray_INT32
) -> NDArray_FLOAT32_64:
    """Compute an area normalized centroid for each cluster."""
    # Check if null cluster exists
    null_clusters = np.any(clusters == -1)
    if null_clusters:
        clusters = clusters.copy()
        clusters[clusters == -1] = clusters.max() + 1

    wval = cent * area.reshape(-1, 1)
    cweight = np.bincount(clusters, weights=area)

    cval = (
        np.vstack(
            (
                np.bincount(clusters, weights=wval[:, 0]),
                np.bincount(clusters, weights=wval[:, 1]),
                np.bincount(clusters, weights=wval[:, 2]),
            )
        )
        / cweight
    )

    if null_clusters:
        cval[:, -1] = np.inf

    return cval.T


def _subdivide(mesh: PolyData, nsub: int) -> PolyData:
    """Perform a linear subdivision of a mesh"""
    new_faces = mesh.faces.reshape(-1, 4)[:, 1:]
    if new_faces.dtype != np.int32:
        new_faces = new_faces.astype(np.int32)

    new_points = mesh.points
    if new_points.dtype != np.double:
        new_points = new_points.astype(np.double)

    for _ in range(nsub):
        new_points, new_faces, _ = _clustering.subdivision(new_points, new_faces, 0.0)

    sub_mesh = polydata_from_faces(new_points, new_faces)
    return sub_mesh.clean()
