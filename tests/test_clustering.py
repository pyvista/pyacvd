import os
import pyacvd
import pytest
import pyvista as pv
from pyvista import examples
from pyvista.plotting import system_supports_plotting

NO_PLOTTING = not system_supports_plotting()

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
