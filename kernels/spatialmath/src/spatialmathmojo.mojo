"""Clean-room batched SE(3)/SO(3) pose algebra on float64 matrices.

Written fresh from the group definitions of SE(3) and SO(3) (see e.g.
Corke, "Robotics, Vision and Control", Springer, for the conventions the
reference toolbox uses):

    compose:   C = A @ B                    (full 4x4 / 3x3 matrix product)
    inverse:   SE(3): [R t; 0 1]^-1 = [R^T  -R^T t; 0 0 0 1]
               SO(3): R^-1 = R^T
    transform: SE(3): out = h2e(T @ e2h(p)) = (T @ [p;1])[:3] / (T @ [p;1])[3]
               SO(3): out = R @ p

Every exported function processes a whole batch of poses (and, for
transform, a whole batch of points) in one call. The per-element dot
products accumulate over k in ascending order — the same IEEE-754 float64
operation order as the reference software's per-pose ``x @ y`` dispatch —
so batch outputs agree with that oracle to within a few ulps (measured
~1e-15; compiled code, like the oracle's own BLAS, may contract a*b+c
into a fused multiply-add). No third-party Mojo code is used or adapted.

Exported C ABI (stateless: every call computes one whole batch):

    int32_t spatialmathmojo_abi_version(void)
    int32_t spatialmathmojo_se3_compose(a, a_stride, b, b_stride, n, out)
    int32_t spatialmathmojo_so3_compose(a, a_stride, b, b_stride, n, out)
    int32_t spatialmathmojo_se3_inverse(t, n, out)
    int32_t spatialmathmojo_so3_inverse(r, n, out)
    int32_t spatialmathmojo_se3_transform(t, n, points, m, out)
    int32_t spatialmathmojo_so3_transform(r, n, points, m, out)

    a, b, t, r   pose batches, C-order row-major: 16 float64 per SE(3)
                 pose, 9 per SO(3) pose
    a_stride     element stride between consecutive left poses: 16 (SE3)
                 or 9 (SO3) for per-pose pairing, 0 to broadcast a
                 singleton left pose across the batch (same for b_stride)
    n            number of output poses
    points       m C-order row-major float64 triples (one point per row)
    dst          compose/inverse: n poses like the inputs;
                 transform: n*m C-order row-major float64 triples,
                 out[i*m + j] = pose_i applied to point_j (fully
                 overwritten)
"""

from std.memory import Pointer
from std.origin import MutUntrackedOrigin

comptime ABI_VERSION: Int32 = 1

# C-side pointer spelling (untracked origin: the caller owns every buffer).
comptime F64Ptr = Pointer[Float64, MutUntrackedOrigin]


@export
def spatialmathmojo_abi_version() abi("C") -> Int32:
    return ABI_VERSION


@export
def spatialmathmojo_se3_compose(
    a: F64Ptr,
    a_stride: Int64,
    b: F64Ptr,
    b_stride: Int64,
    n: Int64,
    dst: F64Ptr,
) abi("C") -> Int32:
    """C[p] = A[p] @ B[p] for p in [0, n): full 4x4 products, k ascending.

    Returns 0 on success, 2 on invalid arguments. A stride of 0 broadcasts
    that operand's single pose across the batch; otherwise the stride must
    be 16 elements. `out` holds n*16 float64 slots, fully overwritten.
    """
    if n < 0:
        return 2
    if n > 0:
        if a_stride != 0 and a_stride != 16:
            return 2
        if b_stride != 0 and b_stride != 16:
            return 2
    var sa = Int(a_stride)
    var sb = Int(b_stride)
    for p in range(Int(n)):
        var pa = a.unsafe_offset(p * sa)
        var pb = b.unsafe_offset(p * sb)
        var po = dst.unsafe_offset(p * 16)
        for i in range(4):
            for j in range(4):
                var s = Float64(0.0)
                for k in range(4):
                    s += pa[unsafe_offset=i * 4 + k] * pb[
                        unsafe_offset=k * 4 + j
                    ]
                po[unsafe_offset=i * 4 + j] = s
    return 0


@export
def spatialmathmojo_so3_compose(
    a: F64Ptr,
    a_stride: Int64,
    b: F64Ptr,
    b_stride: Int64,
    n: Int64,
    dst: F64Ptr,
) abi("C") -> Int32:
    """C[p] = A[p] @ B[p] for p in [0, n): full 3x3 products, k ascending.

    Strides are 9 elements (per-pose pairing) or 0 (broadcast). `out`
    holds n*9 float64 slots, fully overwritten.
    """
    if n < 0:
        return 2
    if n > 0:
        if a_stride != 0 and a_stride != 9:
            return 2
        if b_stride != 0 and b_stride != 9:
            return 2
    var sa = Int(a_stride)
    var sb = Int(b_stride)
    for p in range(Int(n)):
        var pa = a.unsafe_offset(p * sa)
        var pb = b.unsafe_offset(p * sb)
        var po = dst.unsafe_offset(p * 9)
        for i in range(3):
            for j in range(3):
                var s = Float64(0.0)
                for k in range(3):
                    s += pa[unsafe_offset=i * 3 + k] * pb[
                        unsafe_offset=k * 3 + j
                    ]
                po[unsafe_offset=i * 3 + j] = s
    return 0


@export
def spatialmathmojo_se3_inverse(
    t: F64Ptr,
    n: Int64,
    dst: F64Ptr,
) abi("C") -> Int32:
    """Batch SE(3) inverse: out[p] = [R^T  -R^T t; 0 0 0 1], k ascending.

    The structured inverse reads only R = T[:3, :3] and t = T[:3, 3]; the
    input's last row is never read and the output's last row is exactly
    [0, 0, 0, 1] (matching the reference `trinv`, which builds the result
    from a zero matrix). The negation of -R^T t is applied after the dot
    product, as in the reference. `out` holds n*16 float64 slots.
    """
    if n < 0:
        return 2
    for p in range(Int(n)):
        var pt = t.unsafe_offset(p * 16)
        var po = dst.unsafe_offset(p * 16)
        var t0 = pt[unsafe_offset=3]
        var t1 = pt[unsafe_offset=7]
        var t2 = pt[unsafe_offset=11]
        for i in range(3):
            # Row i of R^T is column i of R; -(R^T t)[i] = -sum_k R[k,i]*t[k].
            var r0i = pt[unsafe_offset=i]  # R[0, i]
            var r1i = pt[unsafe_offset=4 + i]  # R[1, i]
            var r2i = pt[unsafe_offset=8 + i]  # R[2, i]
            po[unsafe_offset=i * 4] = r0i
            po[unsafe_offset=i * 4 + 1] = r1i
            po[unsafe_offset=i * 4 + 2] = r2i
            po[unsafe_offset=i * 4 + 3] = -(
                r0i * t0 + r1i * t1 + r2i * t2
            )
        po[unsafe_offset=12] = 0.0
        po[unsafe_offset=13] = 0.0
        po[unsafe_offset=14] = 0.0
        po[unsafe_offset=15] = 1.0
    return 0


@export
def spatialmathmojo_so3_inverse(
    r: F64Ptr,
    n: Int64,
    dst: F64Ptr,
) abi("C") -> Int32:
    """Batch SO(3) inverse: out[p] = R[p]^T (exact). n*9 float64 slots."""
    if n < 0:
        return 2
    for p in range(Int(n)):
        var pr = r.unsafe_offset(p * 9)
        var po = dst.unsafe_offset(p * 9)
        for i in range(3):
            for j in range(3):
                po[unsafe_offset=i * 3 + j] = pr[unsafe_offset=j * 3 + i]
    return 0


@export
def spatialmathmojo_se3_transform(
    t: F64Ptr,
    n: Int64,
    points: F64Ptr,
    m: Int64,
    dst: F64Ptr,
) abi("C") -> Int32:
    """Batch SE(3) point transform: out[i,j] = h2e(T[i] @ e2h(p[j])).

    Homogeneous product w = T @ [px, py, pz, 1] accumulated with k
    ascending, then Euclidean division out = w[:3] / w[3] — the reference
    `h2e` divides by the last homogeneous component even when it is
    exactly 1. `out` holds n*m row-major triples: out[(i*m + j)*3 + c].
    """
    if n < 0 or m < 0:
        return 2
    for i in range(Int(n)):
        var pt = t.unsafe_offset(i * 16)
        for j in range(Int(m)):
            var pp = points.unsafe_offset(j * 3)
            var px = pp[unsafe_offset=0]
            var py = pp[unsafe_offset=1]
            var pz = pp[unsafe_offset=2]
            var po = dst.unsafe_offset((i * Int(m) + j) * 3)
            # w = T @ [px, py, pz, 1], each row accumulated k ascending
            # (T[r,3] * 1.0 == T[r,3] exactly).
            var w0 = (
                pt[unsafe_offset=0] * px
                + pt[unsafe_offset=1] * py
                + pt[unsafe_offset=2] * pz
                + pt[unsafe_offset=3] * 1.0
            )
            var w1 = (
                pt[unsafe_offset=4] * px
                + pt[unsafe_offset=5] * py
                + pt[unsafe_offset=6] * pz
                + pt[unsafe_offset=7] * 1.0
            )
            var w2 = (
                pt[unsafe_offset=8] * px
                + pt[unsafe_offset=9] * py
                + pt[unsafe_offset=10] * pz
                + pt[unsafe_offset=11] * 1.0
            )
            var w3 = (
                pt[unsafe_offset=12] * px
                + pt[unsafe_offset=13] * py
                + pt[unsafe_offset=14] * pz
                + pt[unsafe_offset=15] * 1.0
            )
            po[unsafe_offset=0] = w0 / w3
            po[unsafe_offset=1] = w1 / w3
            po[unsafe_offset=2] = w2 / w3
    return 0


@export
def spatialmathmojo_so3_transform(
    r: F64Ptr,
    n: Int64,
    points: F64Ptr,
    m: Int64,
    dst: F64Ptr,
) abi("C") -> Int32:
    """Batch SO(3) vector rotation: out[i,j] = R[i] @ p[j], k ascending.

    `out` holds n*m row-major triples: out[(i*m + j)*3 + c].
    """
    if n < 0 or m < 0:
        return 2
    for i in range(Int(n)):
        var pr = r.unsafe_offset(i * 9)
        for j in range(Int(m)):
            var pp = points.unsafe_offset(j * 3)
            var px = pp[unsafe_offset=0]
            var py = pp[unsafe_offset=1]
            var pz = pp[unsafe_offset=2]
            var po = dst.unsafe_offset((i * Int(m) + j) * 3)
            for r in range(3):
                po[unsafe_offset=r] = (
                    pr[unsafe_offset=r * 3] * px
                    + pr[unsafe_offset=r * 3 + 1] * py
                    + pr[unsafe_offset=r * 3 + 2] * pz
                )
    return 0
