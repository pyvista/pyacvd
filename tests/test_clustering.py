import pytest
from pyvista.plotting import system_supports_plotting
from pyvista import examples
import pyvista as pv

import pyacvd


NO_PLOTTING = not system_supports_plotting()


bunny = examples.download_bunny()
cow = examples.download_cow()


def test_bunny():
    clus = pyacvd.Clustering(bunny)
    clus.cluster(5000)
    remesh = clus.create_mesh()
    assert remesh.n_points == 5000


def test_cylinder():
    cylinder = pv.Cylinder().tri_filter()
    cylinder.clean(inplace=True)
    cylinder.subdivide(3, inplace=True)

    clus = pyacvd.Clustering(cylinder)
    nclus = 500
    clus.cluster(nclus)

    remesh = clus.create_mesh()
    assert remesh.n_points == nclus


@pytest.mark.skipif(NO_PLOTTING, reason="Requires system to support plotting")
def test_cow():
    # must be an all triangular mesh to sub-divide
    cow.tri_filter(inplace=True)

    # mesh is not dense enough for uniform remeshing
    cow.subdivide(4, inplace=True)

    clus = pyacvd.Clustering(cow)
    clus.cluster(20000)
    
    clus.plot(off_screen=True)
    remesh = clus.create_mesh()
