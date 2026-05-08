"""Tests for the pyacvd PyVista accessor (``mesh.acvd.*``)."""

import numpy as np
import pytest
import pyvista as pv

import pyacvd

# The accessor is only available on pyvista >= 0.48; skip the whole
# module otherwise so pyacvd remains usable on older releases.
if not hasattr(pv, "register_dataset_accessor"):
    pytest.skip("pyvista >= 0.48 required for the accessor", allow_module_level=True)


@pytest.fixture
def cylinder() -> pv.PolyData:
    return pv.Cylinder().triangulate()


def test_accessor_registered() -> None:
    mesh = pv.Cylinder().triangulate()
    assert hasattr(mesh, "acvd")
    assert isinstance(mesh.acvd, pyacvd._accessor.ACVDAccessor)


def test_accessor_cached_on_instance(cylinder: pv.PolyData) -> None:
    # PyVista caches accessor instances per dataset.
    assert cylinder.acvd is cylinder.acvd


def test_remesh_returns_uniform_polydata(cylinder: pv.PolyData) -> None:
    nclus = 500
    remeshed = cylinder.acvd.remesh(nclus, subdivide=3)
    assert isinstance(remeshed, pv.PolyData)
    assert remeshed.n_points == nclus
    assert remeshed.is_all_triangles


def test_remesh_does_not_mutate_source(cylinder: pv.PolyData) -> None:
    n_points_before = cylinder.n_points
    n_cells_before = cylinder.n_cells
    cylinder.acvd.remesh(500, subdivide=3)
    assert cylinder.n_points == n_points_before
    assert cylinder.n_cells == n_cells_before


def test_remesh_triangulates_non_triangle_input() -> None:
    # Cylinder() is quad/triangle mixed by default; the accessor should
    # silently triangulate rather than raising.
    quad_cyl = pv.Cylinder()
    assert not quad_cyl.is_all_triangles
    remeshed = quad_cyl.acvd.remesh(200, subdivide=3)
    assert remeshed.n_points == 200


def test_remesh_fast(cylinder: pv.PolyData) -> None:
    remeshed = cylinder.acvd.remesh(200, subdivide=3, fast=True)
    assert remeshed.is_all_triangles
    # fast_cluster does not guarantee an exact match, but it should be
    # in the same ballpark and never empty.
    assert remeshed.n_points > 0


def test_clustering_returns_clustering(cylinder: pv.PolyData) -> None:
    clus = cylinder.acvd.clustering(subdivide=2)
    assert isinstance(clus, pyacvd.Clustering)
    # Subdivide grew the mesh; original is untouched.
    assert clus.mesh.n_points > cylinder.n_points


def test_subdivide_returns_polydata(cylinder: pv.PolyData) -> None:
    out = cylinder.acvd.subdivide(2)
    assert isinstance(out, pv.PolyData)
    assert out.n_points > cylinder.n_points
    assert out.is_all_triangles


def test_cluster_ids_shape(cylinder: pv.PolyData) -> None:
    clus = cylinder.acvd.clustering(subdivide=3)
    ids = cylinder.acvd.cluster_ids(200, subdivide=3)
    assert ids.dtype == np.int32
    assert ids.shape == (clus.mesh.n_points,)
    # Cluster ids should index a sensible range.
    valid = ids[ids >= 0]
    assert valid.min() >= 0
    assert valid.max() < 200


def test_remesh_weights_accepted(cylinder: pv.PolyData) -> None:
    # Smoke test: weights of all ones should match the unweighted run
    # closely enough to produce the requested cluster count.
    n = pv.Cylinder().triangulate().subdivide(3).n_points
    weights = np.ones(n, dtype=np.float64)
    remeshed = cylinder.acvd.remesh(200, subdivide=3, weights=weights)
    assert remeshed.n_points == 200
