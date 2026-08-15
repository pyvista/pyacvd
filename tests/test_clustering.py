import os

import numpy as np
import pyacvd
import pytest
import pyvista as pv
from pyacvd import _clustering, clustering
from pyvista import examples
from pyvista.plotting import system_supports_plotting

NO_PLOTTING = not system_supports_plotting()


def _supports_fixed_size_storage() -> bool:
    """Return True when pyvista stores regular cells with fixed size storage.

    pyvista 0.49 rewrote ``CellArray.from_regular_cells`` to use VTK's fixed size
    cell storage, which drops the explicit offsets array and stops widening an
    int32 connectivity array to the VTK id type. This probes that behaviour rather
    than a private symbol, which can be renamed without deprecation.

    Delete this and the two skips it gates once ``pyvista>=0.49`` is the floor in
    ``pyproject.toml``; the behaviour is unconditional from that release on.
    """
    points = np.zeros((3, 3))
    faces = np.array([[0, 1, 2]], dtype=np.int32)
    return bool(pv.PolyData.from_regular_faces(points, faces).regular_faces.dtype == np.int32)


SUPPORTS_FIXED_SIZE_STORAGE = _supports_fixed_size_storage()

# skip plotting on windows. This occurs specifically on Python 3.13, skipping
# all for the time being
if os.name == "nt":
    NO_PLOTTING = True

try:
    bunny = examples.download_bunny()
except:
    bunny = None

try:
    cow = examples.download_cow()
except:
    cow = None


@pytest.mark.skipif(bunny is None, reason="Requires example data")
def test_bunny() -> None:
    clus = pyacvd.Clustering(bunny)
    clus.cluster(5000)
    remesh = clus.create_mesh(clean=False)
    assert remesh.n_points == 5000

    remesh = clus.create_mesh(clean=True)
    assert remesh.n_points == remesh.clean().n_points


def test_cylinder() -> None:
    cylinder = pv.Cylinder().triangulate()
    # cylinder.clean(inplace=True)

    clus = pyacvd.Clustering(cylinder)
    clus.subdivide(3)
    nclus = 500
    clus.cluster(nclus)

    remesh = clus.create_mesh()
    assert remesh.n_points == nclus


@pytest.mark.skipif(cow is None, reason="Requires example data")
@pytest.mark.skipif(NO_PLOTTING, reason="Requires system to support plotting")
def test_cow() -> None:
    # must be an all triangular mesh to sub-divide
    cow.triangulate(inplace=True)

    # mesh is not dense enough for uniform remeshing
    clus = pyacvd.Clustering(cow)
    clus.subdivide(3)
    clus.cluster(20000)

    clus.plot(off_screen=True)
    remesh = clus.create_mesh()
    assert remesh.n_points


def _slotted_wall() -> pv.PolyData:
    """Return a wall with a slot cut into it and a second wall parallel behind it.

    The slot is one row of faces wide, which is what issue #70 looks like: a
    cluster covering the wall either side of it has its centroid land in the
    slot, a fraction of the cluster across from the material it sits on, and the
    ray cast from it goes straight through and hits the wall behind.

    Built by hand rather than with a boolean so the point order, and with it the
    clustering, does not depend on the VTK version.
    """
    n, size, gap = 20, 10.0, 5.0
    t = np.linspace(0.0, size, n + 1)
    xx, yy = np.meshgrid(t, t, indexing="ij")
    plate = np.column_stack((xx.ravel(), yy.ravel(), np.zeros(xx.size)))

    ind = np.arange(plate.shape[0]).reshape(n + 1, n + 1)
    a, b = ind[:-1, :-1].ravel(), ind[1:, :-1].ravel()
    c, d = ind[1:, 1:].ravel(), ind[:-1, 1:].ravel()
    tri = np.vstack((np.column_stack((a, b, c)), np.column_stack((a, c, d))))

    # cut a slot into the front wall, running in from one edge
    cent = plate[tri].mean(axis=1)
    half = size / n / 2
    in_slot = (np.abs(cent[:, 1] - size / 2) < half) & (cent[:, 0] < 0.7 * size)

    points = np.vstack((plate, plate - [0.0, 0.0, gap]))
    faces = np.vstack((tri[~in_slot], tri + plate.shape[0]))
    used, faces = np.unique(faces, return_inverse=True)  # drop the slot's points
    return pv.PolyData.from_regular_faces(points[used], faces.reshape(-1, 3).astype(np.int32))


@pytest.mark.parametrize("nclus", [40, 50, 60, 110, 120, 130, 150, 180, 320])
def test_create_mesh_over_slot(nclus: int) -> None:
    """Cluster centroids must not be projected onto the far wall of a hollow part.

    A cluster covering the wall either side of the slot has its centroid land
    over the void. The ray cast from it misses the wall the cluster sits on and
    hits the wall behind, dragging the point across the part and leaving a wedge
    of long triangles behind it.

    Both walls are flat, so every centroid already lies on the wall it belongs
    to and projecting must not move a single one. Each of these cluster counts
    moves one of them by the full five unit gap without the fix.

    https://github.com/pyvista/pyacvd/issues/70
    """
    clus = pyacvd.Clustering(_slotted_wall())
    clus.cluster(nclus)

    assert np.array_equal(clus.create_mesh().points, clus.create_mesh(moveclus=False).points)


def test_create_mesh_onto_cone_tip() -> None:
    """A centroid that is genuinely off the surface must still be projected.

    Clustering a surface of revolution gives rings of clusters. A ring's area
    weighted centroid sits on the axis and its mean normal cancels radially and
    points along it, so the ray legitimately runs out of the tip of the cone.
    That is several cluster radii, further than the runaway of the slot above,
    and refusing it leaves the point stranded inside the cone.
    """
    clus = pyacvd.Clustering(pv.Cone(height=2.0, resolution=40).triangulate())
    clus.subdivide(3)
    clus.cluster(25)

    moved = clus.create_mesh()
    unmoved = clus.create_mesh(moveclus=False)

    # the projection has work to do here, unlike on the slotted wall
    assert np.linalg.norm(moved.points - unmoved.points, axis=1).max() > 0.5

    # and every point it moves lands on the cone
    _, closest = clus.mesh.find_closest_cell(moved.points, return_closest_point=True)
    assert np.linalg.norm(moved.points - closest, axis=1).max() < 1e-9

    _, closest = clus.mesh.find_closest_cell(unmoved.points, return_closest_point=True)
    assert np.linalg.norm(unmoved.points - closest, axis=1).max() > 0.1


def test_runaway_projection() -> None:
    """Only a centroid already on the surface may have its projection refused."""
    # three clusters of two points, each one unit either side of its centroid
    points = np.array([[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]] * 3)
    clusters = np.array([0, 0, 1, 1, 2, 2], dtype=np.int32)
    ccent = np.zeros((3, 3))

    # the first two sit on the surface, a fifth of a cluster from it, the third
    # is six tenths away and so is closing a real gap
    sqr_dist = np.repeat([[0.04], [0.04], [0.36]], 4, axis=1)

    dist = np.array([3.001, 2.999, 100.0])
    mask = clustering._runaway_projection(points, ccent, clusters, dist, sqr_dist)

    assert mask.tolist() == [True, False, False]
    assert clustering._runaway_projection(points, ccent, clusters, -dist, sqr_dist).tolist() == [
        True,
        False,
        False,
    ]


def test_runaway_projection_reaches_past_the_nearest_face() -> None:
    """A cluster of one point is sized by its neighbourhood, not the nearest face.

    Such a cluster has no radius of its own and the nearest face centroid is only
    about a third of an edge away, so sizing it by that alone refuses ordinary
    projections. The faces here run out from ``0.1`` to ``1.2``; the fourth of
    them gives a limit of ``1.2`` rather than the ``0.3`` of the nearest, and
    reaching further out than that stops refusing anything at all.
    """
    points = np.zeros((1, 3))
    clusters = np.zeros(1, dtype=np.int32)
    ccent = np.zeros((1, 3))
    sqr_dist = (np.arange(1, 13) / 10.0) ** 2
    sqr_dist = sqr_dist.reshape(1, -1)

    kept = clustering._runaway_projection(points, ccent, clusters, np.array([1.0]), sqr_dist)
    refused = clustering._runaway_projection(points, ccent, clusters, np.array([1.3]), sqr_dist)

    assert not kept[0]
    assert refused[0]


def test_polydata_from_faces() -> None:
    points = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]])
    faces = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int64)

    mesh = clustering.polydata_from_faces(points, faces)

    assert mesh.n_points == 4
    assert mesh.n_cells == 2
    assert np.array_equal(mesh.regular_faces, faces)


def test_polydata_from_faces_invalid() -> None:
    points = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0]])
    with pytest.raises(ValueError, match="two dimensional"):
        clustering.polydata_from_faces(points, np.array([0, 1, 2], dtype=np.int64))


@pytest.mark.skipif(not SUPPORTS_FIXED_SIZE_STORAGE, reason="Requires fixed size cell storage")
def test_polydata_from_faces_int32() -> None:
    """An int32 faces array must not be widened to the VTK id type.

    The probe above uses a shallow copy, so this also covers the ``deep=True``
    copy that ``polydata_from_faces`` makes.
    """
    points = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]])
    faces = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)

    mesh = clustering.polydata_from_faces(points, faces)

    assert mesh.regular_faces.dtype == np.int32
    assert np.array_equal(mesh.regular_faces, faces)


@pytest.mark.skipif(not SUPPORTS_FIXED_SIZE_STORAGE, reason="Requires fixed size cell storage")
def test_polydata_from_faces_fixed_size_storage() -> None:
    """Triangles are stored without an explicit offsets array."""
    points = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]])
    faces = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int64)

    mesh = clustering.polydata_from_faces(points, faces)

    assert mesh.GetPolys().IsStorageFixedSize()


def test_point_normals_non_triangular() -> None:
    """A non-triangular mesh must be rejected rather than silently misread."""
    with pytest.raises(ValueError, match="all triangles"):
        clustering.point_normals(pv.Plane(i_resolution=2, j_resolution=2))


def test_faces_are_int32() -> None:
    """Faces stay int32 through a full clustering round trip."""
    clus = pyacvd.Clustering(pv.Sphere().triangulate())
    assert clustering._tri_faces_from_poly(clus.mesh).dtype == np.int32

    clus.subdivide(2)
    sub_faces = clustering._tri_faces_from_poly(clus.mesh)
    assert sub_faces.dtype == np.int32
    assert np.array_equal(sub_faces, clus.mesh.regular_faces)

    clus.cluster(500)
    remesh = clus.create_mesh()
    assert remesh.n_points == 500
    assert clustering._tri_faces_from_poly(remesh).dtype == np.int32


def test_subdivision_returns_int32() -> None:
    """The subdivision extension returns an int32 faces array."""
    mesh = pv.Sphere().triangulate()
    points = mesh.points.astype(np.float64)
    faces = clustering._tri_faces_from_poly(mesh)

    new_points, new_faces, nsub = _clustering.subdivision(points, faces, 0.0)

    assert new_faces.dtype == np.int32
    assert nsub == faces.shape[0]
    assert new_faces.shape == (faces.shape[0] * 4, 3)
    assert new_faces.max() == new_points.shape[0] - 1


def test_tri_faces_from_poly_too_many_points(monkeypatch: pytest.MonkeyPatch) -> None:
    """A mesh larger than an int32 face index can address must be rejected."""
    mesh = pv.Sphere().triangulate()
    monkeypatch.setattr(clustering, "MAX_POINTS", mesh.n_points - 1)

    with pytest.raises(ValueError, match="exceeds the maximum"):
        clustering._tri_faces_from_poly(mesh)


def test_tri_faces_from_poly_empty() -> None:
    """An empty mesh has no faces rather than being an error."""
    faces = clustering._tri_faces_from_poly(pv.PolyData())
    assert faces.shape == (0, 3)
    assert faces.dtype == np.int32


def test_tri_faces_from_poly_quad() -> None:
    with pytest.raises(ValueError, match="all triangles"):
        clustering._tri_faces_from_poly(pv.Plane(i_resolution=1, j_resolution=1))


def test_tri_faces_from_poly_mixed() -> None:
    """A mesh of mixed cell types is rejected with the same message as a quad."""
    points = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]])
    mixed = pv.PolyData(points, faces=[3, 0, 1, 2, 4, 0, 1, 2, 3])
    assert mixed.n_cells == 2

    with pytest.raises(ValueError, match="all triangles"):
        clustering._tri_faces_from_poly(mixed)
