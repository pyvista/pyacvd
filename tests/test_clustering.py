import ast
import os
from pathlib import Path

import numpy as np
import pyacvd
import pytest
import pyvista as pv
from pyacvd import clustering
from pyvista import examples
from pyvista.plotting import system_supports_plotting

NO_PLOTTING = not system_supports_plotting()

try:
    # pyvista 0.49 rewrote ``CellArray.from_regular_cells`` to use VTK's fixed
    # size cell storage and to stop widening an int32 connectivity array to the
    # VTK id type. This constant was added by that change, so its absence means
    # the installed pyvista still writes an explicit offsets array and upcasts.
    from pyvista.core._vtk_utilities import _SUPPORTS_FIXED_SIZE_STORAGE
except ImportError:
    _SUPPORTS_FIXED_SIZE_STORAGE = False

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


def test_no_direct_vtk_import() -> None:
    """Meshes must be built through pyvista rather than a VTK binding.

    pyvista is not always built on the stock ``vtkmodules`` wheel. A
    ``vtkCellArray`` from a different binding is a different C++ type and
    ``vtkPolyData.SetPolys`` rejects it.
    """
    tree = ast.parse(Path(clustering.__file__).read_text())

    roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            roots.add(node.module.split(".")[0])

    assert "vtkmodules" not in roots
    assert "vtk" not in roots


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


@pytest.mark.skipif(not _SUPPORTS_FIXED_SIZE_STORAGE, reason="Requires fixed size cell storage")
def test_polydata_from_faces_int32() -> None:
    """An int32 faces array must not be widened to the VTK id type."""
    points = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]])
    faces = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)

    mesh = clustering.polydata_from_faces(points, faces)

    assert mesh.regular_faces.dtype == np.int32
    assert np.array_equal(mesh.regular_faces, faces)


@pytest.mark.skipif(not _SUPPORTS_FIXED_SIZE_STORAGE, reason="Requires fixed size cell storage")
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
