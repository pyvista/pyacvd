#ifndef PYACVD_BVH_HEADER_H
#define PYACVD_BVH_HEADER_H

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

// -----------------------------------------------------------------------------
// Bounding volume hierarchy over a triangle surface
// -----------------------------------------------------------------------------
//
// A top-down, median-split BVH used to find, for each of many rays, the
// intersection with the surface nearest the ray's origin.
//
// The alternative it replaces is to take the faces whose centroids are nearest
// the origin and test those. That is approximate twice over: the face a ray
// actually hits need not be among the nearest by centroid, and a ray can hit a
// face whose centroid is far away while missing every near one. Widening the
// candidate list narrows the error without closing it, and the k-nearest search
// needed to build the list costs more than this whole structure.
//
// Build is O(n log n) and single-threaded. Queries are independent and read
// only, so a caller is free to run them in parallel, but nothing here does.
// -----------------------------------------------------------------------------

namespace pyacvd_bvh {

// Faces per leaf. Ray-triangle tests are cheap, so large leaves trade a little
// query time for a shallower tree and a faster build.
static constexpr int BVH_LEAF_SIZE = 128;

// Deep enough for any tree a median split can produce: it halves the face count
// at every level, so depth stays under 64 for face counts an int can hold.
static constexpr int BVH_STACK_SIZE = 128;

// Rejects rays within rounding of parallel to a triangle's plane.
static constexpr double BVH_DET_EPS = 1e-6;

// Barycentric slack, so a ray passing exactly along a shared edge hits one of
// the two faces rather than slipping between them.
static constexpr double BVH_TRI_EPS = 1e-6;

// Guards the split axis against a set of faces whose centroids coincide.
static constexpr double BVH_BUILD_EPS = 1e-12;

// A node is a leaf when ``count > 0``, in which case ``left`` indexes the first
// of its faces in ``prim_indices``. Otherwise ``left`` indexes its left child
// and the right child follows immediately after it.
struct Node {
    double bmin[3];
    double bmax[3];
    int32_t left;
    int32_t count;
};

// ``T`` is the scalar the caller stores its vertices in. Everything computed
// here is double regardless, so a float32 surface is traced at the same
// precision without being copied up first.
template <typename T> struct BVH {
    std::vector<Node> nodes;
    std::vector<int32_t> prim_indices;
    std::vector<double> tri_bmin;
    std::vector<double> tri_bmax;
    std::vector<double> tri_centroid;

    // Borrowed. The caller keeps the arrays alive for the lifetime of the tree.
    const T *vertices = nullptr;
    const int32_t *faces = nullptr;
    int nfaces = 0;
};

template <typename T> inline void compute_tri_aabbs(BVH<T> &bvh) {
    const int nfaces = bvh.nfaces;
    const T *v = bvh.vertices;
    const int32_t *f = bvh.faces;
    for (int i = 0; i < nfaces; ++i) {
        bvh.prim_indices[i] = i;
        const int64_t i0 = f[i * 3 + 0];
        const int64_t i1 = f[i * 3 + 1];
        const int64_t i2 = f[i * 3 + 2];
        for (int k = 0; k < 3; ++k) {
            const double a = v[i0 * 3 + k];
            const double b = v[i1 * 3 + k];
            const double c = v[i2 * 3 + k];
            bvh.tri_bmin[i * 3 + k] = std::min(a, std::min(b, c));
            bvh.tri_bmax[i * 3 + k] = std::max(a, std::max(b, c));
            bvh.tri_centroid[i * 3 + k] = (a + b + c) * (1.0 / 3.0);
        }
    }
}

// Build a tree over ``nfaces`` triangles, splitting each node at the median
// face centroid along the axis its centroids are most spread over.
template <typename T>
inline void bvh_build(BVH<T> &bvh, const T *vertices, const int32_t *faces, int nfaces) {
    bvh.vertices = vertices;
    bvh.faces = faces;
    bvh.nfaces = nfaces;
    bvh.nodes.clear();
    if (nfaces <= 0) {
        return;
    }

    bvh.tri_bmin.assign(static_cast<size_t>(nfaces) * 3, 0.0);
    bvh.tri_bmax.assign(static_cast<size_t>(nfaces) * 3, 0.0);
    bvh.tri_centroid.assign(static_cast<size_t>(nfaces) * 3, 0.0);
    bvh.prim_indices.assign(nfaces, 0);
    compute_tri_aabbs(bvh);

    const size_t max_leaves =
        (static_cast<size_t>(nfaces) + BVH_LEAF_SIZE - 1) / BVH_LEAF_SIZE;
    bvh.nodes.reserve(max_leaves * 2 + 1);
    bvh.nodes.emplace_back();

    struct Task {
        int node_idx;
        int first;
        int count;
    };
    std::vector<Task> stack;
    stack.reserve(64);
    stack.push_back({0, 0, nfaces});

    const double inf = std::numeric_limits<double>::infinity();

    while (!stack.empty()) {
        const Task task = stack.back();
        stack.pop_back();

        // Bounds over the faces themselves, and separately over their
        // centroids, which is what the split is chosen from
        double bmin[3] = {inf, inf, inf};
        double bmax[3] = {-inf, -inf, -inf};
        double cmin[3] = {inf, inf, inf};
        double cmax[3] = {-inf, -inf, -inf};
        for (int i = task.first; i < task.first + task.count; ++i) {
            const int tri = bvh.prim_indices[i];
            for (int k = 0; k < 3; ++k) {
                bmin[k] = std::min(bmin[k], bvh.tri_bmin[tri * 3 + k]);
                bmax[k] = std::max(bmax[k], bvh.tri_bmax[tri * 3 + k]);
                cmin[k] = std::min(cmin[k], bvh.tri_centroid[tri * 3 + k]);
                cmax[k] = std::max(cmax[k], bvh.tri_centroid[tri * 3 + k]);
            }
        }
        for (int k = 0; k < 3; ++k) {
            bvh.nodes[task.node_idx].bmin[k] = bmin[k];
            bvh.nodes[task.node_idx].bmax[k] = bmax[k];
        }

        int axis = 0;
        const double ext[3] = {cmax[0] - cmin[0], cmax[1] - cmin[1], cmax[2] - cmin[2]};
        if (ext[1] > ext[0] && ext[1] >= ext[2])
            axis = 1;
        else if (ext[2] > ext[0] && ext[2] >= ext[1])
            axis = 2;

        // Stop when the node is small enough, or when its centroids are so
        // nearly coincident that a split cannot separate them
        if (task.count <= BVH_LEAF_SIZE || ext[axis] < BVH_BUILD_EPS) {
            bvh.nodes[task.node_idx].left = task.first;
            bvh.nodes[task.node_idx].count = task.count;
            continue;
        }

        const int mid = task.first + task.count / 2;
        const double *centroids = bvh.tri_centroid.data();
        std::nth_element(
            bvh.prim_indices.begin() + task.first,
            bvh.prim_indices.begin() + mid,
            bvh.prim_indices.begin() + task.first + task.count,
            [centroids, axis](int32_t a, int32_t b) {
                return centroids[a * 3 + axis] < centroids[b * 3 + axis];
            });

        const int left_idx = static_cast<int>(bvh.nodes.size());
        bvh.nodes.emplace_back();
        bvh.nodes.emplace_back();
        bvh.nodes[task.node_idx].left = left_idx;
        bvh.nodes[task.node_idx].count = 0;

        stack.push_back({left_idx + 1, mid, task.first + task.count - mid});
        stack.push_back({left_idx, task.first, mid - task.first});
    }
}

// Slab test of a ray against an axis-aligned box, writing the time at which the
// ray enters it into ``t_enter``. Returns whether the box is met within
// ``[t_lo, t_hi]``.
inline bool ray_aabb(
    const double origin[3],
    const double dir[3],
    const double inv_dir[3],
    const double bmin[3],
    const double bmax[3],
    const double t_lo,
    const double t_hi,
    double &t_enter) {
    double tmin = -std::numeric_limits<double>::infinity();
    double tmax = std::numeric_limits<double>::infinity();

    for (int k = 0; k < 3; ++k) {
        if (std::abs(dir[k]) < 1e-30) {
            // Parallel to this pair of slabs: either inside them for all t or
            // outside for all t
            if (origin[k] < bmin[k] || origin[k] > bmax[k])
                return false;
            continue;
        }
        double t1 = (bmin[k] - origin[k]) * inv_dir[k];
        double t2 = (bmax[k] - origin[k]) * inv_dir[k];
        if (t1 > t2)
            std::swap(t1, t2);
        tmin = std::max(tmin, t1);
        tmax = std::min(tmax, t2);
        if (tmin > tmax)
            return false;
    }

    t_enter = tmin;
    return tmax >= t_lo && tmin <= t_hi;
}

// Moller-Trumbore ray-triangle intersection. On a hit, ``t`` is the signed
// distance along ``dir``, which is assumed to be of unit length.
template <typename T>
inline bool ray_triangle(
    const double origin[3],
    const double dir[3],
    const T *v0,
    const T *v1,
    const T *v2,
    double &t) {
    const double e1[3] = {
        double(v1[0]) - double(v0[0]),
        double(v1[1]) - double(v0[1]),
        double(v1[2]) - double(v0[2])};
    const double e2[3] = {
        double(v2[0]) - double(v0[0]),
        double(v2[1]) - double(v0[1]),
        double(v2[2]) - double(v0[2])};

    const double p[3] = {
        dir[1] * e2[2] - dir[2] * e2[1],
        dir[2] * e2[0] - dir[0] * e2[2],
        dir[0] * e2[1] - dir[1] * e2[0]};

    const double det = e1[0] * p[0] + e1[1] * p[1] + e1[2] * p[2];
    if (std::abs(det) < BVH_DET_EPS)
        return false;
    const double inv_det = 1.0 / det;

    const double s[3] = {
        origin[0] - double(v0[0]), origin[1] - double(v0[1]), origin[2] - double(v0[2])};
    const double u = (s[0] * p[0] + s[1] * p[1] + s[2] * p[2]) * inv_det;
    if (u < -BVH_TRI_EPS || u > 1.0 + BVH_TRI_EPS)
        return false;

    const double q[3] = {
        s[1] * e1[2] - s[2] * e1[1],
        s[2] * e1[0] - s[0] * e1[2],
        s[0] * e1[1] - s[1] * e1[0]};

    const double vv = (dir[0] * q[0] + dir[1] * q[1] + dir[2] * q[2]) * inv_det;
    if (vv < -BVH_TRI_EPS || u + vv > 1.0 + BVH_TRI_EPS)
        return false;

    t = (e2[0] * q[0] + e2[1] * q[1] + e2[2] * q[2]) * inv_det;
    return true;
}

// Trace one ray, returning the index of the face it hits, or -1 for a miss.
//
// ``t_hit`` is set to the signed distance along ``dir``. When ``in_vector`` the
// ray only travels forwards and the hit is the one with the smallest positive
// distance; otherwise it travels both ways and the hit is the one nearest the
// origin in either direction.
template <typename T>
inline int bvh_trace_ray(
    const BVH<T> &bvh,
    const double origin[3],
    const double dir[3],
    const bool in_vector,
    double &t_hit) {
    t_hit = std::numeric_limits<double>::infinity();
    if (bvh.nodes.empty()) {
        return -1;
    }

    double inv_dir[3];
    for (int k = 0; k < 3; ++k) {
        inv_dir[k] = (std::abs(dir[k]) >= 1e-30) ? 1.0 / dir[k] : 0.0;
    }

    double best_t = std::numeric_limits<double>::infinity();
    double best_abs = std::numeric_limits<double>::infinity();
    int best_idx = -1;

    int stack[BVH_STACK_SIZE];
    int sp = 0;
    stack[sp++] = 0;

    while (sp > 0) {
        const Node &node = bvh.nodes[stack[--sp]];

        // Prune against the best hit so far. Travelling both ways, that is an
        // interval straddling the origin rather than one running forward
        const double t_lo = in_vector ? 0.0 : -best_abs;
        const double t_hi = in_vector ? best_t : best_abs;

        double t_enter;
        if (!ray_aabb(origin, dir, inv_dir, node.bmin, node.bmax, t_lo, t_hi, t_enter)) {
            continue;
        }

        if (node.count > 0) {
            const T *v = bvh.vertices;
            const int32_t *f = bvh.faces;
            const int32_t *prims = bvh.prim_indices.data();
            for (int i = node.left; i < node.left + node.count; ++i) {
                const int tri = prims[i];
                const int64_t i0 = f[tri * 3 + 0];
                const int64_t i1 = f[tri * 3 + 1];
                const int64_t i2 = f[tri * 3 + 2];
                double t;
                if (!ray_triangle(origin, dir, &v[i0 * 3], &v[i1 * 3], &v[i2 * 3], t))
                    continue;
                if (in_vector) {
                    if (t > 0.0 && t < best_t) {
                        best_t = t;
                        best_idx = tri;
                    }
                } else if (std::abs(t) < best_abs) {
                    best_abs = std::abs(t);
                    best_t = t;
                    best_idx = tri;
                }
            }
            continue;
        }

        // Internal. Push the further child first so the nearer one is popped
        // next and tightens the pruning interval before the far one is opened.
        const int left = node.left;
        const Node &nl = bvh.nodes[left];
        const Node &nr = bvh.nodes[left + 1];

        double tl, tr;
        const bool hit_l = ray_aabb(origin, dir, inv_dir, nl.bmin, nl.bmax, t_lo, t_hi, tl);
        const bool hit_r = ray_aabb(origin, dir, inv_dir, nr.bmin, nr.bmax, t_lo, t_hi, tr);

        if (hit_l && hit_r) {
            if (sp + 2 > BVH_STACK_SIZE)
                continue;
            const double key_l = in_vector ? tl : std::abs(tl);
            const double key_r = in_vector ? tr : std::abs(tr);
            if (key_l < key_r) {
                stack[sp++] = left + 1;
                stack[sp++] = left;
            } else {
                stack[sp++] = left;
                stack[sp++] = left + 1;
            }
        } else if (hit_l) {
            if (sp + 1 > BVH_STACK_SIZE)
                continue;
            stack[sp++] = left;
        } else if (hit_r) {
            if (sp + 1 > BVH_STACK_SIZE)
                continue;
            stack[sp++] = left + 1;
        }
    }

    t_hit = best_t;
    return best_idx;
}

} // namespace pyacvd_bvh

#endif
