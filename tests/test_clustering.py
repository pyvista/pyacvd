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
