"""Point based clustering module"""

import ctypes

import numpy as np
import pyvista as pv
from pyvista import ID_TYPE
from vtkmodules.vtkCommonDataModel import vtkCellArray

from pyacvd import _clustering


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

    def __init__(self, mesh):
        """Check inputs and initializes neighbors"""
        # mesh must be triangular
        if not mesh.is_all_triangles:
            mesh = mesh.triangulate()

        self.mesh = mesh.copy()
        self.clusters = None
        self.nclus = None
        self.remesh = None
        self._area = None
        self._wcent = None
        self._neigh = None
        self._nneigh = None
        self._edges = None
        self._update_data(None)

    def _update_data(self, weights=None):
        # Compute point weights and weighted points
        self._area, self._wcent = weighted_points(self.mesh, additional_weights=weights)

        # neighbors and edges
        self._neigh, self._nneigh = neighbors_from_mesh(self.mesh)
        self._edges = _clustering.edge_id(self._neigh, self._nneigh)

    def cluster(self, nclus, maxiter=100, debug=False, iso_try=10):
        """Cluster points"""
        self.clusters, _, self.nclus = _clustering.cluster(
            self._neigh,
            self._nneigh,
            nclus,
            self._area,
            self._wcent,
            self._edges,
            maxiter,
            debug,
            iso_try,
        )

        return self.clusters

    def subdivide(self, nsub):
        """Perform a linear subdivision of the mesh.

        Parameters
        ----------
        nsub : int
            Number of subdivisions
        """
        self.mesh.copy_from(_subdivide(self.mesh, nsub))
        self._update_data()

    def plot(self, random_color=True, **kwargs):
        """Plot clusters if available.

        Parameters
        ----------
        random_color : bool, optional
            Plots clusters with a random color rather than a color
            based on the order of creation.

        **kwargs : keyword arguments, optional
            See help(pyvista.plot)

        Returns
        -------
        cpos : list
            Camera position.  See help(pyvista.plot)
        """
        if not hasattr(self, "clusters"):
            return self.mesh.plot(**kwargs)

        # Setup color
        if random_color:
            rand_color = np.random.random(self.nclus)
        else:
            rand_color = np.linspace(0, 1, self.nclus)
        colors = rand_color[self.clusters]

        # Set color range depending if null clusters exist
        if np.any(colors == -1):
            colors[colors == -1] = -0.25
            rng = [-0.25, 1]
        else:
            rng = [0, 1]

        return self.mesh.plot(scalars=colors, rng=rng, **kwargs)

    def create_mesh(self, flipnorm=True):
        """Generates mesh from clusters"""
        if flipnorm:
            cnorm = self.cluster_norm
        else:
            cnorm = None

        # Generate mesh
        self.remesh = create_mesh(self.mesh, self._area, self.clusters, cnorm, flipnorm)
        return self.remesh

    @property
    def cluster_norm(self):
        """Return cluster norms"""
        if not hasattr(self, "clusters"):
            raise RuntimeError("No clusters available")

        # Normals of original mesh
        self.mesh.compute_normals(cell_normals=False, inplace=True, consistent_normals=False)
        norm = self.mesh.point_data["Normals"]

        # Compute normalized mean cluster normals
        cnorm = np.empty((self.nclus, 3))
        cnorm[:, 0] = np.bincount(self.clusters, weights=norm[:, 0] * self._area)
        cnorm[:, 1] = np.bincount(self.clusters, weights=norm[:, 1] * self._area)
        cnorm[:, 2] = np.bincount(self.clusters, weights=norm[:, 2] * self._area)
        weights = ((cnorm * cnorm).sum(1) ** 0.5).reshape((-1, 1))
        weights[weights == 0] = 1
        cnorm /= weights
        return cnorm

    @property
    def cluster_centroid(self):
        """Computes an area normalized value for each cluster"""
        wval = self.mesh.points * self._area.reshape(-1, 1)
        cval = np.vstack(
            (
                np.bincount(self.clusters, weights=wval[:, 0]),
                np.bincount(self.clusters, weights=wval[:, 1]),
                np.bincount(self.clusters, weights=wval[:, 2]),
            )
        )
        weights = np.bincount(self.clusters, weights=self._area)
        weights[weights == 0] = 1
        cval /= weights
        return cval.T


def cluster_centroid(cent, area, clusters):
    """Computes an area normalized centroid for each cluster"""

    # Check if null cluster exists
    null_clusters = np.any(clusters == -1)
    if null_clusters:
        clusters = clusters.copy()
        clusters[clusters == -1] = clusters.max() + 1

    wval = cent * area.reshape(-1, 1)
    cweight = np.bincount(clusters, weights=area)
    cweight[cweight == 0] = 1

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


def polydata_from_faces(points: np.ndarray, faces: np.ndarray) -> pv.PolyData:
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

    pdata = pv.PolyData()
    pdata.points = points

    carr = vtkCellArray()
    offset = np.arange(0, faces.size + 1, faces.shape[1], dtype=ID_TYPE)
    carr.SetData(pv.numpy_to_idarr(offset, deep=True), pv.numpy_to_idarr(faces, deep=True))
    pdata.SetPolys(carr)
    return pdata


def create_mesh(
    mesh: pv.PolyData,
    area: np.ndarray,
    clusters: np.ndarray,
    cnorm: np.ndarray,
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
    flipnorm : bool, default: True
        If ``True``, flip the normals of the faces.

    Returns
    -------
    pyvista.PolyData
        Returns a PolyData object representing the new generated mesh.

    """
    # Cluster centroids
    ccent = np.ascontiguousarray(cluster_centroid(mesh.points, area, clusters))

    # Include only faces that connect to different clusters
    faces = mesh._connectivity_array.reshape(-1, 3)
    f_clus = np.sort(clusters[faces], axis=1)
    mask = np.all(np.diff(f_clus, axis=1) != 0, axis=1)
    f = f_clus[mask]

    # Remove duplicate faces
    f = f[unique_row_indices(f)]

    # New mesh generated from unique faces built and cluster centroids
    mesh_out = polydata_from_faces(ccent, f)

    # Reorder faces to be consistent with the original mesh
    if flipnorm:
        # Mean normals of clusters each face is build from
        adjcnorm = cnorm[f].sum(1)
        adjcnorm /= np.linalg.norm(adjcnorm, axis=1).reshape(-1, 1)

        newnorm = mesh_out.compute_normals(point_normals=False, consistent_normals=False)["Normals"]

        # If the dot is negative, reverse the order of those faces
        agg = (adjcnorm * newnorm).sum(1)  # dot product
        mask = agg < 0.0
        f_out = mesh_out._connectivity_array.reshape(-1, 3)
        f_out[mask] = f_out[mask, ::-1]

    return mesh_out


def unique_row_indices(a):
    """Indices of unique rows"""
    b = np.ascontiguousarray(a).view(np.dtype((np.void, a.dtype.itemsize * a.shape[1])))
    _, idx = np.unique(b, return_index=True)
    return idx


def weighted_points(mesh, return_weighted=True, additional_weights=None):
    """Returns point weight based on area weight and weighted points.

    Points are weighted by adjacent area faces.

    Parameters
    ----------
    mesh : pv.PolyData
        All triangular surface mesh.

    return_weighted : bool, optional
        Returns vertices mutlipled by point weights.

    Returns
    -------
    pweight : np.ndarray, np.double
        Point weight array

    wvertex : np.ndarray, np.double
        Vertices mutlipled by their corresponding weights.  Returned only
        when return_weighted is True.

    """
    faces = mesh.faces.reshape(-1, 4)
    if faces.dtype != np.int32:
        faces = faces.astype(np.int32)
    points = mesh.points

    if additional_weights is not None:
        weights = additional_weights
        return_weighted = True
        if not weights.flags["C_CONTIGUOUS"]:
            weights = np.ascontiguousarray(weights, dtype=ctypes.c_double)
        elif weights.dtype != ctypes.c_double:
            weights = weights.astype(ctypes.c_double)

        if (weights < 0).any():
            raise Exception("Negative weights not allowed.")

    else:
        weights = np.array([])

    if points.dtype == np.float64:
        weighted_point_func = _clustering.weighted_points_double
    else:
        weighted_point_func = _clustering.weighted_points_float

    return weighted_point_func(points, faces, weights, return_weighted)


def neighbors_from_mesh(mesh):
    """Assemble neighbor array.  Assumes all-triangular mesh.

    Parameters
    ----------
    mesh : pyvista.PolyData
        Mesh to assemble neighbors from.

    Returns
    -------
    neigh : int np.ndarray [:, ::1]
        Indices of each neighboring node for each node.

    nneigh : int np.ndarray [::1]
        Number of neighbors for each node.
    """
    npoints = mesh.number_of_points
    faces = mesh.faces.reshape(-1, 4)
    if faces.dtype != np.int32:
        faces = faces.astype(np.int32)

    return _clustering.neighbors_from_faces(npoints, faces)


def _subdivide(mesh, nsub):
    """Perform a linear subdivision of a mesh"""
    new_faces = mesh.faces.reshape(-1, 4)
    if new_faces.dtype != np.int32:
        new_faces = new_faces.astype(np.int32)

    new_points = mesh.points
    if new_points.dtype != np.double:
        new_points = new_points.astype(np.double)

    for _ in range(nsub):
        new_points, new_faces = _clustering.subdivision(new_points, new_faces)

    sub_mesh = pv.PolyData(new_points, new_faces)
    sub_mesh.clean(inplace=True)
    return sub_mesh
