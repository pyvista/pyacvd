import os
from typing import Callable

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


def _brute_force_ray_trace(
    origins: np.ndarray,
    dirs: np.ndarray,
    points: np.ndarray,
    faces: np.ndarray,
    in_vector: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the nearest intersection, testing every ray against every face.

    The reference the tree is checked against: same Moller-Trumbore test and same
    tolerances, none of the traversal.
    """
    v0 = points[faces[:, 0]]
    e1 = points[faces[:, 1]] - v0
    e2 = points[faces[:, 2]] - v0
    # the determinant is compared against the area scale of the face, not a
    # fixed number, exactly as the tree does
    sqr_scale = (np.cross(e1, e2) ** 2).sum(axis=1)

    dist = np.zeros(len(origins))
    hit = np.full(len(origins), -1, dtype=np.int32)

    for i, (origin, direction) in enumerate(zip(origins, dirs)):
        p = np.cross(direction, e2)
        det = np.einsum("ij,ij->i", e1, p)
        ok = det**2 >= 1e-9**2 * sqr_scale
        inv_det = np.divide(1.0, det, out=np.zeros_like(det), where=ok)

        s = origin - v0
        u = np.einsum("ij,ij->i", s, p) * inv_det
        ok &= (u >= -1e-6) & (u <= 1.0 + 1e-6)

        q = np.cross(s, e1)
        v = (q @ direction) * inv_det
        ok &= (v >= -1e-6) & (u + v <= 1.0 + 1e-6)

        t = np.einsum("ij,ij->i", e2, q) * inv_det
        if in_vector:
            ok &= t > 0.0
        if not ok.any():
            continue

        key = np.where(ok, t if in_vector else np.abs(t), np.inf)
        nearest = int(np.argmin(key))
        dist[i] = t[nearest]
        hit[i] = nearest
    return dist, hit


def _random_rays(mesh: pv.PolyData, n: int = 300, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Random rays over the mesh, originating clear of its surface.

    Deliberately not from its vertices. A ray starting on the surface meets every
    face of that vertex's fan at zero distance, and which of them is reported is
    an arbitrary tie that says nothing about whether the traversal is right.
    """
    rng = np.random.default_rng(seed)
    lower, upper = np.array(mesh.bounds).reshape(3, 2).T
    span = upper - lower

    origins = lower - 0.25 * span + rng.random((n, 3)) * 1.5 * span
    dirs = rng.normal(size=(n, 3))
    return origins, dirs / np.linalg.norm(dirs, axis=1, keepdims=True)


@pytest.mark.parametrize("in_vector", [False, True], ids=["both_ways", "forwards"])
@pytest.mark.parametrize(
    "make_mesh",
    [
        lambda: pv.Sphere(),
        lambda: pv.Cone(resolution=40),
        lambda: pv.ParametricTorus(),
        lambda: examples.load_airplane(),
    ],
    ids=["sphere", "cone", "torus", "airplane"],
)
def test_ray_trace_matches_brute_force(
    make_mesh: Callable[[], pv.PolyData], in_vector: bool
) -> None:
    """The tree must find exactly what testing every face finds."""
    mesh = make_mesh().triangulate().clean()
    points = mesh.points.astype(np.float64)
    faces = clustering._tri_faces_from_poly(mesh)
    origins, dirs = _random_rays(mesh)

    dist, hit = clustering.ray_trace(origins, dirs, points, faces, in_vector)
    expected_dist, expected_hit = _brute_force_ray_trace(origins, dirs, points, faces, in_vector)

    assert np.array_equal(hit, expected_hit)
    assert np.allclose(dist, expected_dist, rtol=0, atol=1e-12)
    assert hit.max() >= 0, "the rays missed the mesh entirely, so nothing was compared"


def _offset_centroid_mesh() -> tuple[np.ndarray, np.ndarray]:
    """Two faces where the nearer one along the ray has the further centroid.

    The near face contains the ray but is stretched away across the xy plane, so
    its centroid is 10.2 from the origin against the far face's 5.0. Ordering
    candidate faces by how near their centroid is therefore offers the far face
    first.
    """
    points = np.array(
        [
            [-1.0, -1.0, 1.0],
            [1.0, -1.0, 1.0],
            [20.0, 25.0, 1.0],
            [-1.0, -1.0, 5.0],
            [1.0, -1.0, 5.0],
            [0.0, 1.0, 5.0],
        ]
    )
    return points, np.array([[0, 1, 2], [3, 4, 5]], dtype=np.int32)


def test_ray_trace_takes_the_nearest_face_not_the_nearest_centroid() -> None:
    """A ray must stop at the first face it meets.

    Casting against the faces with the nearest centroids and stopping at the
    first of them that is hit returns the face at 5.0 here, and with it a point
    projected four units too far.
    """
    points, faces = _offset_centroid_mesh()
    centroid_dist = np.linalg.norm(points[faces].mean(axis=1), axis=1)
    assert centroid_dist[0] > centroid_dist[1], "the near face must have the further centroid"

    dist, hit = clustering.ray_trace(np.zeros((1, 3)), np.array([[0.0, 0.0, 1.0]]), points, faces)

    assert hit[0] == 0
    assert dist[0] == pytest.approx(1.0)


def test_ray_trace_reaches_past_a_thousand_faces() -> None:
    """Every face is a candidate, however many there are.

    A fixed candidate list cannot see a face beyond its end. Here the only face
    the ray meets is the last of several thousand, and all the nearer centroids
    belong to faces off to one side that it misses.
    """
    n = 4000
    rng = np.random.default_rng(0)
    decoys = rng.random((n, 3, 3)) * 2.0 + np.array([5.0, 0.0, 0.0])
    target = np.array([[[-1.0, -1.0, 1.0], [1.0, -1.0, 1.0], [0.0, 1.0, 1.0]]])

    points = np.vstack((decoys, target)).reshape(-1, 3)
    faces = np.arange(points.shape[0], dtype=np.int32).reshape(-1, 3)

    dist, hit = clustering.ray_trace(np.zeros((1, 3)), np.array([[0.0, 0.0, 1.0]]), points, faces)

    assert hit[0] == n
    assert dist[0] == pytest.approx(1.0)


@pytest.mark.parametrize("scale", [1.0, 1e-3, 1e-6], ids=["unit", "milli", "micro"])
def test_ray_trace_is_independent_of_mesh_scale(scale: float) -> None:
    """A small mesh must be traced like a large one.

    The Moller-Trumbore determinant carries the area of the face, so comparing
    it against a fixed number rejects a face for being small rather than for
    lying along the ray. At the tolerance this replaced, 99.7% of the faces of
    the Stanford bunny subdivided once were unhittable and its cluster
    centroids were left where they were.
    """
    points, faces = _offset_centroid_mesh()

    dist, hit = clustering.ray_trace(
        np.zeros((1, 3)), np.array([[0.0, 0.0, 1.0]]), points * scale, faces
    )

    assert hit[0] == 0
    assert dist[0] == pytest.approx(scale)


def test_ray_trace_ignores_a_face_of_no_area() -> None:
    """A degenerate face has no plane to meet and must not be reported."""
    points = np.array(
        [
            [-1.0, -1.0, 1.0],
            [1.0, -1.0, 1.0],
            [1.0, -1.0, 1.0],  # repeated, so the face is a line
            [-1.0, -1.0, 5.0],
            [1.0, -1.0, 5.0],
            [0.0, 1.0, 5.0],
        ]
    )
    faces = np.array([[0, 1, 2], [3, 4, 5]], dtype=np.int32)

    dist, hit = clustering.ray_trace(np.zeros((1, 3)), np.array([[0.0, 0.0, 1.0]]), points, faces)

    assert hit[0] == 1
    assert dist[0] == pytest.approx(5.0)


def test_ray_trace_in_vector_only_travels_forwards() -> None:
    """A ray restricted to its direction must ignore what lies behind it."""
    points, faces = _offset_centroid_mesh()
    origins = np.array([[0.0, 0.0, 3.0]])
    dirs = np.array([[0.0, 0.0, 1.0]])

    # the near face is two behind, the far face is two ahead
    both_ways, hit_both = clustering.ray_trace(origins, dirs, points, faces, False)
    forwards, hit_forwards = clustering.ray_trace(origins, dirs, points, faces, True)

    assert hit_both[0] == 0
    assert both_ways[0] == pytest.approx(-2.0)
    assert hit_forwards[0] == 1
    assert forwards[0] == pytest.approx(2.0)


def test_ray_trace_miss_leaves_the_point_alone() -> None:
    """A ray that meets nothing reports no face and no distance to travel."""
    points, faces = _offset_centroid_mesh()

    dist, hit = clustering.ray_trace(
        np.array([[100.0, 100.0, 0.0]]), np.array([[0.0, 0.0, 1.0]]), points, faces
    )

    assert hit[0] == -1
    assert dist[0] == 0.0


def test_ray_trace_empty_surface() -> None:
    """Casting against nothing is a miss rather than an error."""
    dist, hit = clustering.ray_trace(
        np.zeros((1, 3)),
        np.array([[0.0, 0.0, 1.0]]),
        np.zeros((0, 3)),
        np.zeros((0, 3), dtype=np.int32),
    )

    assert hit[0] == -1
    assert dist[0] == 0.0


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_ray_trace_dtypes(dtype: type) -> None:
    """A float32 surface is traced at the same precision as a float64 one."""
    points, faces = _offset_centroid_mesh()

    dist, hit = clustering.ray_trace(
        np.zeros((1, 3), dtype=dtype),
        np.array([[0.0, 0.0, 1.0]], dtype=dtype),
        points.astype(dtype),
        faces,
    )

    assert dist.dtype == dtype
    assert hit[0] == 0
    assert dist[0] == pytest.approx(1.0)


def test_create_mesh_projects_onto_the_surface() -> None:
    """Every point ``moveclus`` moves must end up on the surface.

    The airplane is the mesh in the example set that most exercises this: it is
    thin, so a cluster normal often runs nearly along a panel rather than across
    it, and the face the ray meets is regularly not one of those whose centroid
    is nearest the cluster.
    """
    clus = pyacvd.Clustering(examples.load_airplane().triangulate().clean())
    clus.subdivide(3)
    clus.cluster(3000)

    moved = clus.create_mesh()
    _, closest = clus.mesh.find_closest_cell(moved.points, return_closest_point=True)
    deviation = np.linalg.norm(moved.points - closest, axis=1)

    # relative to the model, which is around 2000 across
    assert deviation.max() < 1e-6
