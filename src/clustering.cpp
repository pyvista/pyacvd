#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

#include "array_support.h"

#ifdef _MSC_VER
#define restrict __restrict
#elif defined(__GNUC__) || defined(__clang__)
#define restrict __restrict__
#else
#error unsupported compiler
#endif

// Branch prediction hints compatibility macros
#if defined(__GNUC__) || defined(__clang__)
#define LIKELY(x) __builtin_expect(!!(x), 1)
#define UNLIKELY(x) __builtin_expect(!!(x), 0)
#elif defined(_MSC_VER)
// MSVC does not support __builtin_expect or [[likely]]/[[unlikely]]
// Define the macros as no-ops
#define LIKELY(x) (x)
#define UNLIKELY(x) (x)
#else
#define LIKELY(x) (x)
#define UNLIKELY(x) (x)
#endif

namespace nb = nanobind;
using namespace nb::literals;

using IntVector = std::vector<int>;

template <typename T> using vec3 = T[3];

#define ALIGNMENT 32
#define VAL_SQRT_2 1.414213562373095    // sqrt(2)
#define VAL_2_SQRT_3 1.1547005383792517 // 2/sqrt(3)
#define VAL_1_3 0.3333333333333333333   // 1/3
#define VAL_1_6 0.1666666666666666666   // 1/6
#define TAU_WEG 0.3472704594677839      // (4/5.0)*(1 - 2**0.5*1.0/39**(1/4.0))

#define VTK_EMPTY_CELL 0
#define VTK_HEXAHEDRON 12
#define VTK_PYRAMID 14
#define VTK_TETRA 10
#define VTK_WEDGE 13
#define VTK_QUADRATIC_TETRA 24
#define VTK_QUADRATIC_PYRAMID 27
#define VTK_QUADRATIC_WEDGE 26
#define VTK_QUADRATIC_HEXAHEDRON 25

#define EPSILON 1E-6

int TRI_EDGES[3][2] = {{0, 1}, {1, 2}, {2, 0}};

template <typename T> inline void ArraySubInplace(T *arr_a, const T *arr_b) noexcept {
    arr_a[0] -= arr_b[0];
    arr_a[1] -= arr_b[1];
    arr_a[2] -= arr_b[2];
}

template <typename T> inline void ArrayAddInplace(T *arr_a, const T *arr_b) noexcept {
    arr_a[0] += arr_b[0];
    arr_a[1] += arr_b[1];
    arr_a[2] += arr_b[2];
}

template <typename T>
NDArray<T, 2>
PointNormals(NDArray<const T, 2> points_arr, NDArray<const int64_t, 2> faces_arr) {
    int n_faces = faces_arr.shape(0);
    int n_points = points_arr.shape(0);

    size_t j;
    auto pnorm_arr = MakeNDArray<T, 2>({n_points, 3}, true);
    T *pnorm = pnorm_arr.data();

    T *fnorm = AllocateArray<T>(n_faces * 3);
    const T *v = points_arr.data();
    const int64_t *f = faces_arr.data();

    for (size_t i = 0; i < n_faces; i++) {
        int64_t point0 = f[i * 3 + 0];
        int64_t point1 = f[i * 3 + 1];
        int64_t point2 = f[i * 3 + 2];

        T v0_0 = v[point0 * 3 + 0];
        T v0_1 = v[point0 * 3 + 1];
        T v0_2 = v[point0 * 3 + 2];

        T v1_0 = v[point1 * 3 + 0];
        T v1_1 = v[point1 * 3 + 1];
        T v1_2 = v[point1 * 3 + 2];

        T v2_0 = v[point2 * 3 + 0];
        T v2_1 = v[point2 * 3 + 1];
        T v2_2 = v[point2 * 3 + 2];

        T e0_0 = v1_0 - v0_0;
        T e0_1 = v1_1 - v0_1;
        T e0_2 = v1_2 - v0_2;

        T e1_0 = v2_0 - v0_0;
        T e1_1 = v2_1 - v0_1;
        T e1_2 = v2_2 - v0_2;

        T c0 = e0_1 * e1_2 - e0_2 * e1_1;
        T c1 = e0_2 * e1_0 - e0_0 * e1_2;
        T c2 = e0_0 * e1_1 - e0_1 * e1_0;

        T fnorm_0, fnorm_1, fnorm_2;
        T c_len = sqrt(c0 * c0 + c1 * c1 + c2 * c2);
        if (LIKELY(c_len > 0)) {
            T c_len_inv = 1 / c_len;
            fnorm[i * 3 + 0] = c0 * c_len_inv;
            fnorm[i * 3 + 1] = c1 * c_len_inv;
            fnorm[i * 3 + 2] = c2 * c_len_inv;
        } else {
            fnorm[i * 3 + 0] = c0;
            fnorm[i * 3 + 1] = c1;
            fnorm[i * 3 + 2] = c2;
        }
    }

    // Sum to pnorm in a single thread to avoid race condition
    for (size_t i = 0; i < n_faces; i++) {
        int64_t point0 = f[i * 3 + 0];
        int64_t point1 = f[i * 3 + 1];
        int64_t point2 = f[i * 3 + 2];

        pnorm[point0 * 3 + 0] += fnorm[i * 3 + 0];
        pnorm[point1 * 3 + 0] += fnorm[i * 3 + 0];
        pnorm[point2 * 3 + 0] += fnorm[i * 3 + 0];

        pnorm[point0 * 3 + 1] += fnorm[i * 3 + 1];
        pnorm[point1 * 3 + 1] += fnorm[i * 3 + 1];
        pnorm[point2 * 3 + 1] += fnorm[i * 3 + 1];

        pnorm[point0 * 3 + 2] += fnorm[i * 3 + 2];
        pnorm[point1 * 3 + 2] += fnorm[i * 3 + 2];
        pnorm[point2 * 3 + 2] += fnorm[i * 3 + 2];
    }

    // Normalize point normals
    for (size_t i = 0; i < n_points; i++) {
        T plen = sqrt(
            pnorm[i * 3 + 0] * pnorm[i * 3 + 0] + pnorm[i * 3 + 1] * pnorm[i * 3 + 1] +
            pnorm[i * 3 + 2] * pnorm[i * 3 + 2]);
        if (LIKELY(plen != 0)) {
            T plen_inv = 1 / plen;
            pnorm[i * 3 + 0] *= plen_inv;
            pnorm[i * 3 + 1] *= plen_inv;
            pnorm[i * 3 + 2] *= plen_inv;
        }
    }

    delete[] fnorm;

    return pnorm_arr;
}

template <typename T>
NDArray<T, 2>
FaceCentroid(const NDArray<const T, 2> points, const NDArray<const int64_t, 2> faces) {
    const T *v = points.data();
    const int64_t *f = faces.data();

    int n_faces = faces.shape(0);
    auto fmean_arr = MakeNDArray<T, 2>({n_faces, 3});
    T *fmean = fmean_arr.data();

    for (size_t i = 0; i < n_faces; i++) {
        const int64_t point0 = f[i * 3 + 0];
        const int64_t point1 = f[i * 3 + 1];
        const int64_t point2 = f[i * 3 + 2];

        fmean[i * 3 + 0] =
            (v[point0 * 3 + 0] + v[point1 * 3 + 0] + v[point2 * 3 + 0]) * VAL_1_3;
        fmean[i * 3 + 1] =
            (v[point0 * 3 + 1] + v[point1 * 3 + 1] + v[point2 * 3 + 1]) * VAL_1_3;
        fmean[i * 3 + 2] =
            (v[point0 * 3 + 2] + v[point1 * 3 + 2] + v[point2 * 3 + 2]) * VAL_1_3;
    }

    return fmean_arr;
}

template <typename T>
NDArray<T, 2>
FaceNormals(const NDArray<const T, 2> points, const NDArray<const int64_t, 2> faces) {

    int n_faces = faces.shape(0);
    int n_points = points.shape(0);
    auto fnorm_arr = MakeNDArray<T, 2>({n_faces, 3});
    T *fnorm = fnorm_arr.data();

    const T *v = points.data();
    const int64_t *f = faces.data();

    for (size_t i = 0; i < n_faces; i++) {
        int64_t point0 = f[i * 3 + 0];
        int64_t point1 = f[i * 3 + 1];
        int64_t point2 = f[i * 3 + 2];

        T v0_0 = v[point0 * 3 + 0];
        T v0_1 = v[point0 * 3 + 1];
        T v0_2 = v[point0 * 3 + 2];

        T v1_0 = v[point1 * 3 + 0];
        T v1_1 = v[point1 * 3 + 1];
        T v1_2 = v[point1 * 3 + 2];

        T v2_0 = v[point2 * 3 + 0];
        T v2_1 = v[point2 * 3 + 1];
        T v2_2 = v[point2 * 3 + 2];

        T e0_0 = v1_0 - v0_0;
        T e0_1 = v1_1 - v0_1;
        T e0_2 = v1_2 - v0_2;

        T e1_0 = v2_0 - v0_0;
        T e1_1 = v2_1 - v0_1;
        T e1_2 = v2_2 - v0_2;

        T c0 = e0_1 * e1_2 - e0_2 * e1_1;
        T c1 = e0_2 * e1_0 - e0_0 * e1_2;
        T c2 = e0_0 * e1_1 - e0_1 * e1_0;

        // Normalize cross products to get normal vector
        T c_len = sqrt(c0 * c0 + c1 * c1 + c2 * c2);
        if (LIKELY(c_len > 0)) {
            T c_len_inv = 1 / c_len;
            fnorm[i * 3 + 0] = c0 * c_len_inv;
            fnorm[i * 3 + 1] = c1 * c_len_inv;
            fnorm[i * 3 + 2] = c2 * c_len_inv;
        } else {
            fnorm[i * 3 + 0] = c0;
            fnorm[i * 3 + 1] = c1;
            fnorm[i * 3 + 2] = c2;
        }
    }

    return fnorm_arr;
}

template <typename T>
nb::tuple RayTrace(
    NDArray<const T, 2> source_pt_arr,
    NDArray<const T, 2> source_n_arr,
    NDArray<const T, 2> v_arr,
    NDArray<const int64_t, 2> f_arr,
    NDArray<const uint32_t, 2> idx_arr,
    const bool no_inf,                // true
    const int num_threads,            // -1
    const uint32_t out_of_bounds_idx, // 0 or max targets
    const bool in_vector) {           // false

    const T *source_pt = source_pt_arr.data();
    const T *source_n = source_n_arr.data();
    const T *v = v_arr.data();
    const int64_t *f = f_arr.data();
    const uint32_t *idx = idx_arr.data();

    const size_t nfaces = f_arr.shape(0);

    int npoints = source_pt_arr.shape(0);
    size_t tgt_nbr = idx_arr.shape(1);                  // number of target neighbors
    auto dists_arr = MakeNDArray<T, 1>({(int)npoints}); // dist to nearest face
    T *dists = dists_arr.data();

    // Index of the nearest face
    auto near_ind_arr = MakeNDArray<int, 1>({npoints});
    int *near_ind = near_ind_arr.data();

    // Loop through each face and determine intersections
    for (size_t i = 0; i < npoints; i++) {
        T prev_dist = std::numeric_limits<T>::infinity();
        int near_idx = -1;

        T source_n0 = source_n[i * 3 + 0];
        T source_n1 = source_n[i * 3 + 1];
        T source_n2 = source_n[i * 3 + 2];

        T source_p0 = source_pt[i * 3 + 0];
        T source_p1 = source_pt[i * 3 + 1];
        T source_p2 = source_pt[i * 3 + 2];

        for (size_t j = 0; j < tgt_nbr; ++j) {
            uint32_t ind = idx[i * tgt_nbr + j];
            if (UNLIKELY(out_of_bounds_idx && out_of_bounds_idx == ind))
                break;

            // Compute edges for this triangle. We do this here rather than
            // pre-computing all since we're exiting on the first intersection
            int64_t i0 = f[ind * 3 + 0];
            int64_t i1 = f[ind * 3 + 1];
            int64_t i2 = f[ind * 3 + 2];

            T e1[3], e2[3];
            for (int k = 0; k < 3; k++) {
                T vertex = v[i0 * 3 + k];
                e1[k] = v[i1 * 3 + k] - vertex;
                e2[k] = v[i2 * 3 + k] - vertex;
            }

            // calculate the determinant
            T p0 = source_n1 * e2[2] - source_n2 * e2[1];
            T p1 = source_n2 * e2[0] - source_n0 * e2[2];
            T p2 = source_n0 * e2[1] - source_n1 * e2[0];

            T det = e1[0] * p0 + e1[1] * p1 + e1[2] * p2;
            if (UNLIKELY(std::abs(det) < EPSILON))
                continue;
            T inv_det = 1.0 / det;

            T t0 = source_p0 - v[i0 * 3 + 0];
            T t1 = source_p1 - v[i0 * 3 + 1];
            T t2 = source_p2 - v[i0 * 3 + 2];

            T u = (t0 * p0 + t1 * p1 + t2 * p2) * inv_det;
            if (u < -EPSILON || u > 1.0 + EPSILON)
                continue;

            T q0 = t1 * e1[2] - t2 * e1[1];
            T q1 = t2 * e1[0] - t0 * e1[2];
            T q2 = t0 * e1[1] - t1 * e1[0];
            T vtest = (source_n0 * q0 + source_n1 * q1 + source_n2 * q2) * inv_det;
            if (vtest < -EPSILON || u + vtest > 1.0 + EPSILON)
                continue;

            if (u + vtest < 1.0 + EPSILON) {
                T dist = (e2[0] * q0 + e2[1] * q1 + e2[2] * q2) * inv_det;
                if (in_vector) {
                    if (dist > 0.0 && dist < prev_dist) {
                        prev_dist = dist;
                        near_idx = ind;
                        break; // exit on first intersection
                    }
                } else {
                    if (std::abs(dist) < std::abs(prev_dist)) {
                        prev_dist = dist;
                        near_idx = ind;
                        break; // exit on first intersection
                    }
                }
            }
        }

        if (no_inf && near_idx == -1) { // no intersection and no inf
            dists[i] = 0.0;
            near_ind[i] = near_idx;
        } else {
            dists[i] = prev_dist;
            near_ind[i] = near_idx;
        }
    }

    return nb::make_tuple(dists_arr, near_ind_arr);
}

nb::tuple NeighborsFromTriFaces(const int n_points, const NDArray<int64_t, 2> faces_arr) {

    // internal array: # of adjacent points per point
    int *n_nbr = AllocateArray<int>(n_points, true);

    int edge_a, edge_b;
    int n_faces = faces_arr.shape(0);
    auto faces = faces_arr.data();

    // First, determine the total number of neighbors for each point
    for (size_t i = 0; i < n_faces; i++) {
        int face_offset = i * 3;

        // Increment adjacency counts for each edge in the cell
        for (size_t j = 0; j < 3; j++) {
            edge_a = faces[face_offset + TRI_EDGES[j][0]];
            edge_b = faces[face_offset + TRI_EDGES[j][1]];
            n_nbr[edge_a] += 1;
            n_nbr[edge_b] += 1;
        }
    }

    // Assemble the lookup table now that we've determined the number of
    // node neighbors
    int *lookup = AllocateArray<int>(n_points + 1);
    int cum_sum = 0; // cumulative sum
    for (size_t i = 0; i < n_points; i++) {
        lookup[i] = cum_sum;
        cum_sum += n_nbr[i];
    }
    // supply the last entry
    lookup[n_points] = cum_sum;

    // now reset the number of connections per node and use that to help
    // populate the flat neighbors array
    std::fill(n_nbr, n_nbr + n_points, 0);

    // Now, populate the adjacent neighbors array. Same logic as above except
    // here we check for duplicate nodes
    int *adj_array = AllocateArray<int>(cum_sum);
    std::fill(adj_array, adj_array + cum_sum, -1);

    // Determine the number of elements adjacent to each node
    for (size_t i = 0; i < n_faces; i++) {
        int face_offset = i * 3;

        // Increment adjacency counts for each edge in the cell
        for (size_t j = 0; j < 3; j++) {
            edge_a = faces[face_offset + TRI_EDGES[j][0]];
            edge_b = faces[face_offset + TRI_EDGES[j][1]];

            // Edge a: Check if neighbor already exists
            bool found = false;
            for (int k = 0; k < n_nbr[edge_a]; k++) {
                if (adj_array[lookup[edge_a] + k] == edge_b) {
                    found = true;
                    break;
                }
            }
            if (!found) {
                adj_array[lookup[edge_a] + n_nbr[edge_a]] = edge_b;
                n_nbr[edge_a] += 1;
            }

            // Same logic for edge b
            found = false;
            for (int k = 0; k < n_nbr[edge_b]; k++) {
                if (adj_array[lookup[edge_b] + k] == edge_a) {
                    found = true;
                    break;
                }
            }
            if (!found) {
                adj_array[lookup[edge_b] + n_nbr[edge_b]] = edge_a;
                n_nbr[edge_b] += 1;
            }
        } // edge edge

    } // each cell

    // finally, compress the neighbor array now that we know the true number of
    // neighbors
    int *adj_comp_array = AllocateArray<int>(cum_sum);
    cum_sum = 0; // reset cumulative sum
    for (size_t i = 0; i < n_points; i++) {
        int index = lookup[i]; // NOTE: use pre-updated lookup
        int n = n_nbr[i];
        std::copy(adj_array + index, adj_array + index + n, adj_comp_array + cum_sum);

        // now, update the lookup
        lookup[i] = cum_sum;
        cum_sum += n_nbr[i];
    }
    // supply the last entry
    lookup[n_points] = cum_sum;

    // free internal arrays
    delete[] n_nbr;
    delete[] adj_array;

    auto adj_np = WrapNDarray<int, 1>(adj_comp_array, {cum_sum});
    auto lookup_np = WrapNDarray<int, 1>(lookup, {n_points + 1});

    return nb::make_tuple(adj_np, lookup_np);
}

template <typename T>
nb::tuple PointWeights(
    NDArray<const T, 2> points_arr,
    NDArray<const int64_t, 2> faces_arr,
    NDArray<const T, 1> aweights_arr,
    int n_threads) {

    const int n_faces = faces_arr.shape(0);
    const int n_points = points_arr.shape(0);

    // additional weights
    const int n_add_weights = aweights_arr.size();
    const T *aweights = aweights_arr.data();

    // point weights
    NDArray<T, 1> pweight_arr = MakeNDArray<T, 1>({n_points}, true);
    T *pweight = pweight_arr.data();

    // Weighted vertices
    NDArray<T, 2> wvertex_arr = MakeNDArray<T, 2>({n_points, 3});
    T *wvertex = wvertex_arr.data();

    const T *v = points_arr.data();
    const int64_t *f = faces_arr.data();

    T *local_pweight = AllocateArray<T>(n_points, true);

    for (size_t i = 0; i < n_faces; i++) {
        int64_t point0 = f[i * 3 + 0];
        int64_t point1 = f[i * 3 + 1];
        int64_t point2 = f[i * 3 + 2];

        const T *v0 = v + point0 * 3, *v1 = v + point1 * 3, *v2 = v + point2 * 3;

        T e0[3] = {v1[0] - v0[0], v1[1] - v0[1], v1[2] - v0[2]};
        T e1[3] = {v2[0] - v0[0], v2[1] - v0[1], v2[2] - v0[2]};

        T c[3] = {
            e0[1] * e1[2] - e0[2] * e1[1],
            e0[2] * e1[0] - e0[0] * e1[2],
            e0[0] * e1[1] - e0[1] * e1[0]};

        T c_len = sqrt(c[0] * c[0] + c[1] * c[1] + c[2] * c[2]);
        T farea_l = 0.5 * c_len;

        local_pweight[point0] += farea_l;
        local_pweight[point1] += farea_l;
        local_pweight[point2] += farea_l;
    }

    for (size_t i = 0; i < n_points; i++) {
        pweight[i] += local_pweight[i];
    }

    delete[] local_pweight;

    // ensure this actually helps
    const T *pweight_const = pweight;

    if (n_add_weights) {
        for (size_t i = 0; i < n_points; i++) {
            const T wgt = aweights[i] * pweight_const[i];
            wvertex[i * 3 + 0] = wgt * v[i * 3 + 0];
            wvertex[i * 3 + 1] = wgt * v[i * 3 + 1];
            wvertex[i * 3 + 2] = wgt * v[i * 3 + 2];
        }
    } else {
        for (size_t i = 0; i < n_points; i++) {
            const T wgt = pweight[i];
            wvertex[i * 3 + 0] = wgt * v[i * 3 + 0];
            wvertex[i * 3 + 1] = wgt * v[i * 3 + 1];
            wvertex[i * 3 + 2] = wgt * v[i * 3 + 2];
        }
    }

    return nb::make_tuple(pweight_arr, wvertex_arr);
}

NDArray<int, 2>
UniqueEdges(NDArray<const int, 1> neigh_arr, NDArray<const int, 1> neigh_off_arr) {
    const int *neigh = neigh_arr.data();
    const int *neigh_off = neigh_off_arr.data();
    int n_points = neigh_off_arr.size() - 1;
    int neigh_arr_sz = neigh_arr.size();

    int *edges = AllocateArray<int>(neigh_arr_sz * 2);

    // Create a visited array to mark when nodes have been connected
    bool *visited = AllocateArray<bool>(neigh_arr_sz, true);

    int c = 0;
    for (size_t i = 0; i < n_points; i++) {
        for (int j = neigh_off[i]; j < neigh_off[i + 1]; j++) {
            if (!visited[j]) {
                edges[c * 2] = i;
                edges[c * 2 + 1] = neigh[j];
                c++;
                visited[j] = true;
                // Mark the reciprocal edge as visited
                for (int k = neigh_off[neigh[j]]; k < neigh_off[neigh[j] + 1]; k++) {
                    if (neigh[k] == i) {
                        visited[k] = true;
                        break;
                    }
                }
            }
        }
    }

    delete[] visited;

    return WrapNDarray<int, 2>(edges, {c, 2});
}

template <typename T>
void InitializeClusters(
    int *clusters,
    const int *neigh,
    const int *neigh_off,
    const T *area,
    int nclus,
    const int n_points) {

    // first, compute the total area
    T area_remain = 0;
    for (size_t i = 0; i < n_points; i++) {
        area_remain += area[i];
    }

    // last index (starting point to search for first free point)
    int lst_index = 0;

    // points on the current ring
    std::vector<int> c_ring;
    c_ring.reserve(16384); // should be sufficient for most applications

    // next ring count and indices
    std::vector<int> n_ring;
    n_ring.reserve(16384);

    // assign clusters
    T ct_area = area_remain / nclus; // cluster target area
    for (size_t i = 0; i < nclus; i++) {

        // Get target area and reset current area
        T t_area = area_remain - ct_area * (nclus - i - 1);
        T c_area = 0.0; // cluster area

        // Find the starting point for the cluster
        while (lst_index < n_points && clusters[lst_index] != -1) {
            lst_index++;
        }
        if (lst_index >= n_points)
            break; // No free points left

        c_area += area[lst_index];
        c_ring.push_back(lst_index);
        clusters[lst_index] = i;

        // nothing allocated, exit early
        if (!c_ring.size()) {
            break;
        }

        while (!c_ring.empty()) {

            // ring over the neighbors for each point and add it to the cluster
            // if free
            for (int point_idx : c_ring) {
                for (size_t k = neigh_off[point_idx]; k < neigh_off[point_idx + 1]; k++) {
                    int nbr = neigh[k];

                    // add if cluster is unassigned and the new area remains
                    // under the target area
                    if (clusters[nbr] == -1 && area[nbr] + c_area < t_area) {
                        c_area += area[nbr];
                        clusters[nbr] = i; // where i is the current cluster
                        n_ring.push_back(nbr);
                    }

                } // for each neighbor
            } // for all the points in the current ring

            // swap next ring as current ring
            c_ring.swap(n_ring);
            n_ring.clear();

        } // while current ring is not empty

        area_remain -= c_area;

    } // for each cluster
}

// Eliminate any null clusters by propagating along edges
int GrowUnvisited(const int *edges, const int n_edges, int *clusters, bool *visited) {
    int point_a, point_b, clus_a, clus_b;

    // unassigned edges, or edges where both points are zero
    std::vector<int> u_edges;
    u_edges.reserve(4096);

    // Simply assign to adjcent cluster
    int n_iso = 0; // number of edges with isolated points
    for (size_t i = 0; i < n_edges; ++i) {
        point_a = edges[2 * i];
        point_b = edges[2 * i + 1];
        clus_a = clusters[point_a];
        clus_b = clusters[point_b];

        if (!visited[point_a] && !visited[point_b]) {
            u_edges.push_back(i);
            n_iso++;
        } else {
            if (!visited[point_a] && visited[point_b]) {
                clusters[point_a] = clus_b;
                visited[point_a] = true;
                n_iso++;
            }
            if (!visited[point_b] && visited[point_a]) {
                clusters[point_b] = clus_a;
                visited[point_b] = true;
                n_iso++;
            }
        }
    }

    // For cases where both edges are null. Potentially this is in a deep
    // pocket where all the points are null except for the edges
    bool change = true;
    while (change && !u_edges.empty()) {
        change = false;
        for (auto it = u_edges.begin(); it != u_edges.end();) {
            size_t i = *it;
            point_a = edges[2 * i];
            point_b = edges[2 * i + 1];
            clus_a = clusters[point_a];
            clus_b = clusters[point_b];

            // check if the points of the edge can be assigned
            if (~visited[point_a] && visited[point_b]) {
                clusters[point_a] = clus_b;
                change = true;
            } else if (visited[point_a] && ~visited[point_b]) {
                clusters[point_b] = clus_a;
                change = true;
            } else {
                ++it;
                continue;
            }
            it = u_edges.erase(it);
        }
    }

    // return number of edges with isolated points
    return n_iso;
}

template <typename T>
inline T EnergyWith(const T *sgamma, const T cent[3], const T srho, T p_area) {
    // Energy when adding both item to cluster
    return ((sgamma[0] + cent[0]) * (sgamma[0] + cent[0]) +
            (sgamma[1] + cent[1]) * (sgamma[1] + cent[1]) +
            (sgamma[2] + cent[2]) * (sgamma[2] + cent[2])) /
           (srho + p_area);
}
template <typename T>
inline T EnergyWithout(const T *sgamma, const T cent[3], const T srho, T p_area) {
    // Energy when removing item from cluster
    return ((sgamma[0] - cent[0]) * (sgamma[0] - cent[0]) +
            (sgamma[1] - cent[1]) * (sgamma[1] - cent[1]) +
            (sgamma[2] - cent[2]) * (sgamma[2] - cent[2])) /
           (srho - p_area);
}

// minimize cluster energy
template <typename T>
int MinimizeEnergy(
    const int *restrict edges,
    int *restrict clusters,
    const T *restrict area,
    const T *restrict cent,
    T *restrict sgamma,
    T *restrict srho,
    T *restrict energy,
    const int n_clus,
    const int n_edges,
    const int n_points,
    const int max_iter) {

    // // count points per cluster
    // int *clus_count = AllocateArray<int>(n_clus, true);
    // for (size_t i = 0; i < n_points; i++) {
    //     clus_count[clusters[i]]++;
    // }

    std::vector<int> clus_count(n_clus, 0);
    for (size_t i = 0; i < n_points; i++) {
        clus_count.at(clusters[i])++; // Use .at() for bounds-checked access
        // clus_count[i]++; // Use .at() for bounds-checked access
    }

    // modification arrays (current and next iteration)
    bool *mod_c = AllocateArray<bool>(n_clus);
    bool *mod_n = AllocateArray<bool>(n_clus);

    // all clusters start out being modified
    std::fill(mod_n, mod_n + n_clus, true);

    bool changed = true;
    int n_iter = 0;
    while (changed && n_iter < max_iter) {

        // reset modification arrays
        std::swap(mod_c, mod_n);
        std::fill(mod_n, mod_n + n_clus, false);

        changed = false;
#pragma unroll 8
        for (size_t i = 0; i < n_edges; i++) {

            T area_a, area_b, e_orig, e_a, e_b;
            const T *cent_a, *cent_b;
            T *sgamma_a, *sgamma_b;

            int point_a = edges[2 * i];
            int point_b = edges[2 * i + 1];
            int clus_a = clusters[point_a];
            int clus_b = clusters[point_b];

            // skip if edge does not cross two different clusters or if both
            // clusters are unmodified
            if (clus_a == clus_b || (!mod_c[clus_a] && !mod_c[clus_b]))
                continue;

            // if both clusters have more than once face, we can modify either
            // one
            if (clus_count[clus_a] > 1 && clus_count[clus_b] > 1) {
                area_a = area[point_a];
                cent_a = &cent[point_a * 3];
                sgamma_a = &sgamma[clus_a * 3];

                area_b = area[point_b];
                cent_b = &cent[point_b * 3];
                sgamma_b = &sgamma[clus_b * 3];

                // original sum of the energy of the clusters
                e_orig = energy[clus_a] + energy[clus_b];

                // energy with both assigned to cluster a (move b to a)
                T e_awb = EnergyWith(sgamma_a, cent_b, srho[clus_a], area_b);
                T e_bnb = EnergyWithout(sgamma_b, cent_b, srho[clus_b], area_b);
                T e_a = e_awb + e_bnb;

                // energy with both assigned to cluster b (move a to b)
                T e_ana = EnergyWithout(sgamma_a, cent_a, srho[clus_a], area_a);
                T e_bwa = EnergyWith(sgamma_b, cent_a, srho[clus_b], area_a);
                T e_b = e_ana + e_bwa;

                // Select the configuration with the least amount of energy
                if (e_a > e_orig && e_a > e_b) {
                    // flag both clusters as modified
                    mod_n[clus_a] = true;
                    mod_n[clus_b] = true;
                    changed = true;

                    // reassign point b to cluster a
                    clusters[point_b] = clus_a;
                    clus_count[clus_b]--;
                    clus_count[clus_a]++;

                    // Update cluster mass and centroid
                    srho[clus_a] += area_b;
                    ArrayAddInplace(sgamma_a, cent_b);
                    srho[clus_b] -= area_b;
                    ArraySubInplace(sgamma_b, cent_b);

                    // Update cluster energy
                    energy[clus_a] = e_awb;
                    energy[clus_b] = e_bnb;

                } else if (e_b > e_orig && e_b > e_a) {
                    // flag both clusters as modified
                    mod_n[clus_a] = true;
                    mod_n[clus_b] = true;
                    changed = true;

                    // reassign point a to cluster b
                    clusters[point_a] = clus_b;
                    clus_count[clus_a]--;
                    clus_count[clus_b]++;

                    // Update cluster  mass and centroid
                    srho[clus_b] += area_a;
                    ArrayAddInplace(sgamma_b, cent_a);
                    srho[clus_a] -= area_a;
                    ArraySubInplace(sgamma_a, cent_a);

                    // Update cluster energy
                    energy[clus_a] = e_ana;
                    energy[clus_b] = e_bwa;
                }

            } else if (clus_count[clus_a] > 1) {
                // determine if point a can be reassigned to clus_b
                area_a = area[point_a];
                cent_a = &cent[point_a * 3];
                sgamma_a = &sgamma[clus_a * 3];

                sgamma_b = &sgamma[clus_b * 3];

                // original sum of the energy of the clusters
                e_orig = energy[clus_a] + energy[clus_b];

                // energy with both assigned to cluster b (move a to b)
                T e_ana = EnergyWithout(sgamma_a, cent_a, srho[clus_a], area_a);
                T e_bwa = EnergyWith(sgamma_b, cent_a, srho[clus_b], area_a);
                T e_b = e_ana + e_bwa;

                if (e_b > e_orig) {
                    // flag both clusters as modified
                    mod_n[clus_a] = true;
                    mod_n[clus_b] = true;
                    changed = true;

                    // reassign point a to cluster b
                    clusters[point_a] = clus_b;
                    clus_count[clus_a]--;
                    clus_count[clus_b]++;

                    // Update cluster  mass and centroid
                    srho[clus_b] += area_a;
                    ArrayAddInplace(sgamma_b, cent_a);
                    srho[clus_a] -= area_a;
                    ArraySubInplace(sgamma_a, cent_a);

                    // Update cluster energy
                    energy[clus_a] = e_ana;
                    energy[clus_b] = e_bwa;
                }
            } else if (clus_count[clus_b] > 1) {
                // determine if point b can be reassigned to clus_a

                sgamma_a = &sgamma[clus_a * 3];

                area_b = area[point_b];
                cent_b = &cent[point_b * 3];
                sgamma_b = &sgamma[clus_b * 3];

                // original sum of the energy of the clusters
                e_orig = energy[clus_a] + energy[clus_b];

                // energy with both assigned to cluster a (move b to a)
                T e_awb = EnergyWith(sgamma_a, cent_b, srho[clus_a], area_b);
                T e_bnb = EnergyWithout(sgamma_b, cent_b, srho[clus_b], area_b);
                e_a = e_awb + e_bnb;

                if (e_a > e_orig) {
                    // flag both clusters as modified
                    mod_n[clus_a] = true;
                    mod_n[clus_b] = true;
                    changed = true;

                    // reassign point b to cluster a
                    clusters[point_b] = clus_a;
                    clus_count[clus_b]--;
                    clus_count[clus_a]++;

                    // Update cluster mass and centroid
                    srho[clus_a] += area_b;
                    ArrayAddInplace(sgamma_a, cent_b);
                    srho[clus_b] -= area_b;
                    ArraySubInplace(sgamma_b, cent_b);

                    // Update cluster energy
                    energy[clus_a] = e_awb;
                    energy[clus_b] = e_bnb;
                }

            } // can be reassigned (point b >> clus a)

        } // for each edge

        n_iter++;
    } // until converged

    // delete[] clus_count;
    delete[] mod_c;
    delete[] mod_n;

    return n_iter;
}

// Remove isolated clusters.
//
// This works by starting at a point and adding all neighboring points that
// have the same cluster as the current "active" one (the one with the first
// point) until there are no more points to be added to the expanding front.
//
// This repeats until each cluster has been visited in full, and any points
// that have not been visited are nulled.
int GrowIsolatedClusters(
    const int n_clus,
    const int n_points,
    const int *neigh,
    const int *neigh_off,
    int *clusters,
    const int *edges,
    const int n_edges) {

    bool *visited = AllocateArray<bool>(n_points, true);
    bool *visited_cluster = AllocateArray<bool>(n_clus, true);

    // current and next front
    std::vector<int> front_c;
    front_c.reserve(4096);
    std::vector<int> front_n;
    front_n.reserve(4096);

    int n_clus_checked = 0;
    int ifound = 0;
    int cur_clus, c, i_front_old, i_front_new;

    while (n_clus_checked < n_clus) {
        // Find first unvisited point and corresponding unvisited cluster
        bool found_unvisited = false;
        for (size_t i = ifound; i < n_points; ++i) {
            if (!visited[i] && !visited_cluster[clusters[i]]) {
                ifound = i;
                n_clus_checked++;
                found_unvisited = true;
                break;
            }
        }

        // exit if no starting point is found;
        if (!found_unvisited) {
            break;
        }

        // store cluster data and check that this has been visited
        cur_clus = clusters[ifound];
        visited[ifound] = 1;
        visited_cluster[cur_clus] = 1;

        // Start front propagation
        front_n.push_back(ifound);
        ifound++;
        do {
            // Make the active front the next front
            std::swap(front_c, front_n);
            front_n.clear();

            for (int ind : front_c) {
                for (size_t i = neigh_off[ind]; i < neigh_off[ind + 1]; ++i) {
                    int nbr = neigh[i];
                    if (clusters[nbr] == cur_clus && !visited[nbr]) {
                        front_n.push_back(nbr);
                        visited[nbr] = true;
                    }
                }
            }
        } while (front_n.size());
    }

    // Expand all unvisited points
    int ndisc = GrowUnvisited(edges, n_edges, clusters, visited);
    delete[] visited;
    delete[] visited_cluster;

    return ndisc;
}

int RenumberClusters(int *clusters, const int n_points, const int n_clus) {
    // Renumber clusters ensuring consecutive indexing
    bool *assigned = AllocateArray<bool>(n_clus, true);
    int *ref_arr = AllocateArray<int>(n_clus);

    int c = 0; // active cluster number
    for (size_t i = 0; i < n_points; ++i) {
        int cnum = clusters[i];
        if (!assigned[cnum]) {
            assigned[cnum] = true;
            ref_arr[cnum] = c++;
        }
        clusters[i] = ref_arr[cnum];
    }

    delete[] assigned;
    delete[] ref_arr;

    return c; // Return the number of unique clusters
}

// Eliminate any null clusters by propagating along edges
bool GrowNull(const int *edges, int n_edges, int *clusters) {
    int point_a, point_b, clus_a, clus_b;

    // unassigned edges, or edges where both points are zero
    std::vector<int> u_edges;
    u_edges.reserve(4096);

    for (size_t i = 0; i < n_edges; ++i) {
        point_a = edges[2 * i];
        point_b = edges[2 * i + 1];
        clus_a = clusters[point_a];
        clus_b = clusters[point_b];

        if (clus_a == -1 && clus_b == -1) {
            u_edges.push_back(i);
        } else {
            if (clus_a == -1) {
                clusters[point_a] = clus_b;
            }
            if (clus_b == -1) {
                clusters[point_b] = clus_a;
            }
        }
    }

    bool change = true;
    while (change && !u_edges.empty()) {
        change = false;
        for (auto it = u_edges.begin(); it != u_edges.end();) {
            size_t i = *it;
            point_a = edges[2 * i];
            point_b = edges[2 * i + 1];
            clus_a = clusters[point_a];
            clus_b = clusters[point_b];

            // check if the points of the edge can be assigned
            if (clus_a == -1 && clus_b != -1) {
                clusters[point_a] = clus_b;
                change = true;
            } else if (clus_b == -1 && clus_a != -1) {
                clusters[point_b] = clus_a;
                change = true;
            } else {
                ++it;
                continue;
            }
            it = u_edges.erase(it);
        }
    }

    // return true if failed
    return !u_edges.empty();
}

// Cluster with minimal optimization. Energy is not minimized and disconnected
// clusters are not eliminated.
template <typename T>
nb::tuple FastCluster(
    NDArray<const int, 1> neigh_arr,
    NDArray<const int, 1> neigh_off_arr,
    const int n_clus,
    NDArray<const T, 1> area_arr,
    NDArray<const T, 2> cent_arr,
    NDArray<const int, 2> edges_arr) {
    const int n_points = neigh_off_arr.shape(0) - 1;
    const int n_edges = edges_arr.shape(0);

    // initialize clusters array
    NDArray<int, 1> clusters_arr = MakeNDArray<int, 1>({n_points});
    int *clusters = clusters_arr.data();
    std::fill(clusters, clusters + n_points, -1);

    const int *neigh = neigh_arr.data();
    const int *neigh_off = neigh_off_arr.data();
    const T *area = area_arr.data();
    const int *edges = edges_arr.data();
    const T *cent = cent_arr.data();

    InitializeClusters(clusters, neigh, neigh_off, area, n_clus, n_points);

    // Grow any null clusters and assign them to 0 if it fails
    bool any_null = GrowNull(edges, n_edges, clusters);
    if (any_null) {
        for (size_t i = 0; i < n_points; i++) {
            if (clusters[i] == -1) {
                clusters[i] = 0;
            }
        }
    }

    // Finalize clusters by renumbering them. This compresses the cluster
    // numbering and returns the actual number of clusters generated
    int n_clus_actual = RenumberClusters(clusters, n_points, n_clus);
    return nb::make_tuple(clusters_arr, n_clus_actual);
}

template <typename T>
nb::tuple Cluster(
    NDArray<const int, 1> neigh_arr,
    NDArray<const int, 1> neigh_off_arr,
    const int n_clus,
    NDArray<const T, 1> area_arr,
    NDArray<const T, 2> cent_arr,
    NDArray<const int, 2> edges_arr,
    const int max_iter,
    int iso_try, // times to identify isolated clusters
    bool init_only) {

    const int n_points = neigh_off_arr.shape(0) - 1;
    const int n_edges = edges_arr.shape(0);

    // initialize clusters array
    NDArray<int, 1> clusters_arr = MakeNDArray<int, 1>({n_points});
    int *clusters = clusters_arr.data();
    std::fill(clusters, clusters + n_points, -1);

    const int *neigh = neigh_arr.data();
    const int *neigh_off = neigh_off_arr.data();
    const T *area = area_arr.data();
    const int *edges = edges_arr.data();
    const T *cent = cent_arr.data();

    InitializeClusters(clusters, neigh, neigh_off, area, n_clus, n_points);

    // Grow any null clusters and assign them to 0 if it fails
    bool any_null = GrowNull(edges, n_edges, clusters);
    if (any_null) {
        for (size_t i = 0; i < n_points; i++) {
            if (clusters[i] == -1) {
                clusters[i] = 0;
            }
        }
    }

    // Return unoptimized clusters
    if (init_only) {
        return nb::make_tuple(clusters_arr, false, n_clus);
    }

    // computer cluster centers and mass
    T *sgamma = AllocateArray<T>(n_clus * 3, true);
    T *srho = AllocateArray<T>(n_clus, true);
    for (size_t i = 0; i < n_points; i++) {
        int clus = clusters[i];
        srho[clus] += area[i];
        sgamma[clus * 3 + 0] += cent[i * 3 + 0];
        sgamma[clus * 3 + 1] += cent[i * 3 + 1];
        sgamma[clus * 3 + 2] += cent[i * 3 + 2];
    }

    // compute cluster energy
    T *energy = AllocateArray<T>(n_clus);
    for (size_t i = 0; i < n_clus; i++) {
        energy[i] =
            (sgamma[i * 3 + 0] * sgamma[i * 3 + 0] + sgamma[i * 3 + 1] * sgamma[i * 3 + 1] +
             sgamma[i * 3 + 2] * sgamma[i * 3 + 2]) /
            srho[i];
    }

    // Minimize the energy and remove any isolated clusters
    int n_iter = MinimizeEnergy(
        edges,
        clusters,
        area,
        cent,
        sgamma,
        srho,
        energy,
        n_clus,
        n_edges,
        n_points,
        max_iter);

    // Attempt to remove all isolated clusters
    int n_disc =
        GrowIsolatedClusters(n_clus, n_points, neigh, neigh_off, clusters, edges, n_edges);
    int n_iter_iso = 0;
    while (n_disc && n_iter_iso < iso_try) {
        MinimizeEnergy(
            edges,
            clusters,
            area,
            cent,
            sgamma,
            srho,
            energy,
            n_clus,
            n_edges,
            n_points,
            max_iter);
        n_disc = GrowIsolatedClusters(
            n_clus, n_points, neigh, neigh_off, clusters, edges, n_edges);
        n_iter_iso++;
    }

    delete[] energy;
    delete[] sgamma;
    delete[] srho;

    // final zero out invalid clusters, otherwise risk segfaulting in renumber
    // clusters
    for (size_t i = 0; i < n_points; i++) {
        if (clusters[i] == -1) {
            clusters[i] = 0;
        }
    }

    // Finalize clusters by renumbering them. This compresses the cluster
    // numbering and returns the actual number of clusters generated
    int n_clus_actual = RenumberClusters(clusters, n_points, n_clus);
    return nb::make_tuple(clusters_arr, n_disc > 0, n_clus_actual);
}

template <typename T> NDArray<T, 1> TriArea(NDArray<T, 2> points, NDArray<int64_t, 2> faces) {

    int n_faces = faces.shape(0);
    int n_points = points.shape(0);
    auto tria = MakeNDArray<T, 1>({n_faces});
    auto tria_view = tria.view();

    auto v = points.view();
    auto f = faces.view();

    for (size_t i = 0; i < n_faces; i++) {
        int64_t point0 = f(i, 0);
        int64_t point1 = f(i, 1);
        int64_t point2 = f(i, 2);

        T v0_0 = v(point0, 0);
        T v0_1 = v(point0, 1);
        T v0_2 = v(point0, 2);

        T v1_0 = v(point1, 0);
        T v1_1 = v(point1, 1);
        T v1_2 = v(point1, 2);

        T v2_0 = v(point2, 0);
        T v2_1 = v(point2, 1);
        T v2_2 = v(point2, 2);

        T e0_0 = v1_0 - v0_0;
        T e0_1 = v1_1 - v0_1;
        T e0_2 = v1_2 - v0_2;

        T e1_0 = v2_0 - v0_0;
        T e1_1 = v2_1 - v0_1;
        T e1_2 = v2_2 - v0_2;

        T c0 = e0_1 * e1_2 - e0_2 * e1_1;
        T c1 = e0_2 * e1_0 - e0_0 * e1_2;
        T c2 = e0_0 * e1_1 - e0_1 * e1_0;

        tria(i) = 0.5 * sqrt(c0 * c0 + c1 * c1 + c2 * c2);
    }

    return tria;
}

template <typename T>
nb::tuple SubdivideTriangles(
    const NDArray<T, 2> points, const NDArray<int64_t, 2> faces, const double tgtlen) {

    const int nvert = points.shape(0);
    const int nface = faces.shape(0);
    auto f = faces.view();
    auto v = points.view();

    // Compute current mesh length
    const auto tarea = TriArea(points, faces);

    // determine the total number of new faces
    int nface_new = nface;
    for (size_t i = 0; i < nface; i++) {
        if (tarea(i) > tgtlen) {
            nface_new += 3;
        }
    }

    const T *v_data = points.data();

    // Size Vertex and face arrays for maximum possible array sizes
    T *newv = new T[(nvert + nface_new) * 3];
    int64_t *newf = new int64_t[nface_new * 3];

    // copy existing vertex array
    std::copy(v_data, v_data + nvert * 3, newv);

    // split triangles into three new ones
    int nsub = 0;
    int fc = 0;
    int vc = nvert;
    for (size_t i = 0; i < nface; i++) {
        const int point0 = f(i, 0);
        const int point1 = f(i, 1);
        const int point2 = f(i, 2);

        // Split if triangle length exceeds target area
        if (tarea(i) > tgtlen) {

            // Face 0
            newf[fc * 3 + 0] = point0;
            newf[fc * 3 + 1] = vc;
            newf[fc * 3 + 2] = vc + 2;
            fc += 1;

            // Face 1
            newf[fc * 3 + 0] = point1;
            newf[fc * 3 + 1] = vc + 1;
            newf[fc * 3 + 2] = vc;
            fc += 1;

            // Face 2
            newf[fc * 3 + 0] = point2;
            newf[fc * 3 + 1] = vc + 2;
            newf[fc * 3 + 2] = vc + 1;
            fc += 1;

            // Face 3
            newf[fc * 3 + 0] = vc;
            newf[fc * 3 + 1] = vc + 1;
            newf[fc * 3 + 2] = vc + 2;
            fc += 1;

            // New Vertices
            newv[vc * 3 + 0] = (v(point0, 0) + v(point1, 0)) * 0.5;
            newv[vc * 3 + 1] = (v(point0, 1) + v(point1, 1)) * 0.5;
            newv[vc * 3 + 2] = (v(point0, 2) + v(point1, 2)) * 0.5;
            vc += 1;

            newv[vc * 3 + 0] = (v(point1, 0) + v(point2, 0)) * 0.5;
            newv[vc * 3 + 1] = (v(point1, 1) + v(point2, 1)) * 0.5;
            newv[vc * 3 + 2] = (v(point1, 2) + v(point2, 2)) * 0.5;
            vc += 1;

            newv[vc * 3 + 0] = (v(point0, 0) + v(point2, 0)) * 0.5;
            newv[vc * 3 + 1] = (v(point0, 1) + v(point2, 1)) * 0.5;
            newv[vc * 3 + 2] = (v(point0, 2) + v(point2, 2)) * 0.5;
            vc += 1;

            nsub += 1;

        } else {
            newf[fc * 3 + 0] = point0;
            newf[fc * 3 + 1] = point1;
            newf[fc * 3 + 2] = point2;
            fc += 1;
        }
    }

    auto newv_arr = WrapNDarray<T, 2>(newv, {vc, 3});
    auto newf_arr = WrapNDarray<int64_t, 2>(newf, {fc, 3});
    return nb::make_tuple(newv_arr, newf_arr, nsub);
}

NB_MODULE(_clustering, m) {
    m.def("face_normals", &FaceNormals<float>);
    m.def("face_normals", &FaceNormals<double>);

    m.def("point_normals", &PointNormals<float>);
    m.def("point_normals", &PointNormals<double>);

    m.def("face_centroid", &FaceCentroid<float>);
    m.def("face_centroid", &FaceCentroid<double>);

    m.def("ray_trace", &RayTrace<float>);
    m.def("ray_trace", &RayTrace<double>);

    m.def("neighbors_from_trimesh", &NeighborsFromTriFaces);

    m.def("weighted_points", &PointWeights<float>);
    m.def("weighted_points", &PointWeights<double>);

    m.def("unique_edges", &UniqueEdges);

    m.def("cluster", &Cluster<float>);
    m.def("cluster", &Cluster<double>);

    m.def("fast_cluster", &FastCluster<float>);
    m.def("fast_cluster", &FastCluster<double>);

    m.def("subdivision", &SubdivideTriangles<float>);
    m.def("subdivision", &SubdivideTriangles<double>);
};
