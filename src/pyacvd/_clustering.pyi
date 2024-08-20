from typing import Tuple, TypeVar, Union

import numpy as np
from numpy.typing import NDArray

NDArray_UINT32 = NDArray[np.uint32]
NDArray_INT32 = NDArray[np.int32]
NDArray_FLOAT32 = NDArray[np.float32]
NDArray_FLOAT64 = NDArray[np.float64]
NDArray_FLOAT32_64 = Union[NDArray_FLOAT32, NDArray_FLOAT64]

T = TypeVar("T", np.float32, np.float64)
U = TypeVar("U", np.int32, np.int64)

def cluster(
    neigh_arr: NDArray_INT32,
    neigh_off_arr: NDArray_INT32,
    n_clus: int,
    area_arr: NDArray[T],
    cent_arr: NDArray[T],
    edges_arr: NDArray_INT32,
    max_iter: int,
    iso_try: int,
    init_only: bool,
) -> Tuple[NDArray_INT32, bool, int]: ...
def fast_cluster(
    neigh_arr: NDArray_INT32,
    neigh_off_arr: NDArray_INT32,
    n_clus: int,
    area_arr: NDArray[T],
    cent_arr: NDArray[T],
    edges_arr: NDArray_INT32,
) -> Tuple[NDArray_INT32, int]: ...
def unique_edges(
    neigh_arr: NDArray_INT32,
    neigh_off_arr: NDArray_INT32,
) -> NDArray_INT32: ...
def face_normals(
    points: NDArray[T],
    faces: NDArray[U],
) -> NDArray[T]: ...
def point_normals(
    points: NDArray[T],
    faces: NDArray[U],
) -> NDArray[T]: ...
def face_centroid(
    points: NDArray[T],
    faces: NDArray[U],
) -> NDArray[T]: ...
def weighted_points(
    points: NDArray[T],
    faces: NDArray[U],
    aweights: NDArray[T],
    n_threads: int,
) -> Tuple[NDArray[T], NDArray[T]]: ...
def ray_trace(
    source_pt: NDArray[T],
    source_n: NDArray[T],
    target_v: NDArray[T],
    target_f: NDArray[U],
    idx: NDArray_UINT32,
    no_inf: bool,
    num_threads: int,
    out_of_bounds_idx: int,
    in_vector: bool,
) -> NDArray[T]: ...
def neighbors_from_trimesh(
    n_points: int,
    faces: NDArray[U],
) -> Tuple[NDArray_INT32, NDArray_INT32]: ...
def subdivision(
    points: NDArray[T],
    faces: NDArray[U],
    tgtlen: float,
) -> Tuple[NDArray_FLOAT64, NDArray[U], int]: ...
