"""Clean-room nuScenes detection-eval matching kernel.

Written fresh from the published metric definition (center-distance greedy
matching of confidence-sorted predictions against per-sample ground truth).
No third-party Mojo code is used or adapted.

Exported C ABI (batch-shaped: the Python wrapper hands over every box once,
grouped per class, and this kernel runs the greedy matching pass for every
class x distance-threshold pair — the "x40 fan-out" — in a single call):

    int32_t  nuscenesevalmojo_abi_version(void)
    int32_t  nuscenesevalmojo_match(n_classes, n_samples, n_gt_total,
                                    n_pred_total,
                                    gt_class_offsets, gt_sample_offsets,
                                    gt_vals, gt_attr,
                                    pred_class_offsets, pred_sample,
                                    pred_vals, pred_attr,
                                    periods, dist_ths, n_ths, out)

Buffers (all caller-owned, read-only except `out`):

    gt_class_offsets   int64[n_classes+1]      class blocks of the gt arrays
    gt_sample_offsets  int64[n_classes*(n_samples+1)]
                                               per class, CSR over samples
    gt_vals            float64[n_gt_total*8]   x,y,w,l,h,yaw,vx,vy per gt box
    gt_attr            int32[n_gt_total]       attribute id, -1 = missing
    pred_class_offsets int64[n_classes+1]      class blocks of the pred arrays
    pred_sample        int32[n_pred_total]     sample index of each pred
    pred_vals          float64[n_pred_total*8] same layout as gt_vals
    pred_attr          int32[n_pred_total]
    periods            float64[n_classes]      orientation period per class
                                               (pi for 'barrier', 2*pi else)
    dist_ths           float64[n_ths]          matching distance thresholds
    out                float64[n_ths*n_pred_total*6]
                                               per (threshold, pred): tp flag,
                                               trans/vel/scale/orient/attr
                                               errors (NaN when not matched)

Predictions arrive per class in the wrapper's confidence-sorted order, which
the metric defines as descending score with ties broken by descending original
index. For each (class, threshold) pass the kernel walks that order, matching
each prediction against the nearest still-unmatched ground-truth box of the
same class in the same sample (strict `<` scans, first minimum wins). All
arithmetic is IEEE-754 float64 in the same operation order as the reference
NumPy implementation (PyPI nuscenes-devkit). Element-wise agreement with the
wrapper's pure-Python fallback is within ~1 ulp: compiled code may contract
`a*b+c` into a fused multiply-add, which CPython bytecode cannot (the same
situation as the oracle's own compiled NumPy/BLAS paths).
"""

from std.ffi import external_call
from std.math import sqrt
from std.memory import Pointer
from std.memory.alloc import unsafe_alloc
from std.origin import MutUntrackedOrigin

comptime ABI_VERSION: Int32 = 1

# Float64 fields per box record in gt_vals / pred_vals.
comptime BOX_STRIDE: Int = 8
# Float64 outputs per (threshold, prediction) in `out`.
comptime OUT_STRIDE: Int = 6

# The double closest to pi (same value as numpy.pi).
comptime PI: Float64 = 3.141592653589793

comptime F64Ptr = Pointer[Float64, MutUntrackedOrigin]
comptime I32Ptr = Pointer[Int32, MutUntrackedOrigin]
comptime I64Ptr = Pointer[Int64, MutUntrackedOrigin]
comptime U8Ptr = Pointer[UInt8, MutUntrackedOrigin]


def _nan() -> Float64:
    """Quiet NaN marker for "not matched / not applicable" outputs."""
    return Float64(0.0) / Float64(0.0)


def _py_rem(v: Float64, period: Float64) -> Float64:
    """CPython `float % float` (float_rem) for a positive period."""
    var r = external_call["fmod", Float64, Float64, Float64](v, period)
    if r != 0.0:
        if r < 0.0:
            r += period
    else:
        r = 0.0  # normalize -0.0 to +0.0, like copysign(0.0, period)
    return r


def _angle_diff(x: Float64, y: Float64, period: Float64) -> Float64:
    """Smallest signed difference from angle y to angle x, modulo period."""
    var half = period / 2.0
    var diff = _py_rem(x - y + half, period) - half
    if diff > PI:
        diff -= 2.0 * PI
    return diff


def _match_one(
    p: Int,
    gt_begin: Int,
    gt_end: Int,
    pred_vals: F64Ptr,
    gt_vals: F64Ptr,
    gt_attr: I32Ptr,
    pred_attr: I32Ptr,
    taken: U8Ptr,
    period: Float64,
    dist_th: Float64,
    out_buf: F64Ptr,
    out_off: Int,
):
    """Greedily match one prediction; write its 6 output values."""
    var pb = p * BOX_STRIDE
    var px = pred_vals[unsafe_offset=pb]
    var py = pred_vals[unsafe_offset=pb + 1]
    var min_dist = Float64.MAX
    var best = -1
    for g in range(gt_begin, gt_end):
        if taken[unsafe_offset=g] != 0:
            continue
        var gb = g * BOX_STRIDE
        var dx = px - gt_vals[unsafe_offset=gb]
        var dy = py - gt_vals[unsafe_offset=gb + 1]
        var d = sqrt(dx * dx + dy * dy)
        if d < min_dist:
            min_dist = d
            best = g
    if best < 0 or not (min_dist < dist_th):
        out_buf[unsafe_offset=out_off] = 0.0
        for k in range(1, OUT_STRIDE):
            out_buf[unsafe_offset=out_off + k] = _nan()
        return

    taken[unsafe_offset=best] = 1
    var gb = best * BOX_STRIDE
    out_buf[unsafe_offset=out_off] = 1.0
    # Translation error: the matched center distance (same operands, same ops).
    out_buf[unsafe_offset=out_off + 1] = min_dist
    # Velocity error: L2 norm of the velocity difference (xy).
    var dvx = pred_vals[unsafe_offset=pb + 6] - gt_vals[unsafe_offset=gb + 6]
    var dvy = pred_vals[unsafe_offset=pb + 7] - gt_vals[unsafe_offset=gb + 7]
    out_buf[unsafe_offset=out_off + 2] = sqrt(dvx * dvx + dvy * dvy)
    # Scale error: 1 - aligned-boxes IoU.
    var mw = min(pred_vals[unsafe_offset=pb + 2], gt_vals[unsafe_offset=gb + 2])
    var ml = min(pred_vals[unsafe_offset=pb + 3], gt_vals[unsafe_offset=gb + 3])
    var mh = min(pred_vals[unsafe_offset=pb + 4], gt_vals[unsafe_offset=gb + 4])
    var inter = mw * ml * mh
    var vol_pred = (
        pred_vals[unsafe_offset=pb + 2]
        * pred_vals[unsafe_offset=pb + 3]
        * pred_vals[unsafe_offset=pb + 4]
    )
    var vol_gt = (
        gt_vals[unsafe_offset=gb + 2]
        * gt_vals[unsafe_offset=gb + 3]
        * gt_vals[unsafe_offset=gb + 4]
    )
    out_buf[unsafe_offset=out_off + 3] = 1.0 - inter / (vol_gt + vol_pred - inter)
    # Orientation error: absolute periodic yaw difference.
    var yaw_diff = _angle_diff(
        gt_vals[unsafe_offset=gb + 5], pred_vals[unsafe_offset=pb + 5], period
    )
    out_buf[unsafe_offset=out_off + 4] = abs(yaw_diff)
    # Attribute error: 1 - accuracy, NaN when the gt box has no attribute.
    var ga = gt_attr[unsafe_offset=best]
    if ga < 0:
        out_buf[unsafe_offset=out_off + 5] = _nan()
    elif ga != pred_attr[unsafe_offset=p]:
        out_buf[unsafe_offset=out_off + 5] = 1.0
    else:
        out_buf[unsafe_offset=out_off + 5] = 0.0


@export
def nuscenesevalmojo_abi_version() abi("C") -> Int32:
    return ABI_VERSION


@export
def nuscenesevalmojo_match(
    n_classes: Int64,
    n_samples: Int64,
    n_gt_total: Int64,
    n_pred_total: Int64,
    gt_class_offsets: I64Ptr,
    gt_sample_offsets: I64Ptr,
    gt_vals: F64Ptr,
    gt_attr: I32Ptr,
    pred_class_offsets: I64Ptr,
    pred_sample: I32Ptr,
    pred_vals: F64Ptr,
    pred_attr: I32Ptr,
    periods: F64Ptr,
    dist_ths: F64Ptr,
    n_ths: Int64,
    out_buf: F64Ptr,
) abi("C") -> Int32:
    """Run greedy matching for every class x threshold pair. 0 on success."""
    if (
        n_classes <= 0
        or n_samples <= 0
        or n_gt_total < 0
        or n_pred_total < 0
        or n_ths <= 0
    ):
        return 1

    var taken = unsafe_alloc[UInt8](Int(n_gt_total))
    for c in range(Int(n_classes)):
        var gt_c0 = Int(gt_class_offsets[unsafe_offset=c])
        var gt_c1 = Int(gt_class_offsets[unsafe_offset=c + 1])
        var pred_c0 = Int(pred_class_offsets[unsafe_offset=c])
        var pred_c1 = Int(pred_class_offsets[unsafe_offset=c + 1])
        var period = periods[unsafe_offset=c]
        var csr_base = c * (Int(n_samples) + 1)
        for t in range(Int(n_ths)):
            var dist_th = dist_ths[unsafe_offset=t]
            # Reset the per-(class, threshold) matched-gt flags.
            for g in range(gt_c0, gt_c1):
                taken[unsafe_offset=g] = 0
            var out_base = t * Int(n_pred_total) * OUT_STRIDE
            for p in range(pred_c0, pred_c1):
                var s = Int(pred_sample[unsafe_offset=p])
                var gt_begin = Int(gt_sample_offsets[unsafe_offset=csr_base + s])
                var gt_end = Int(gt_sample_offsets[unsafe_offset=csr_base + s + 1])
                _match_one(
                    p,
                    gt_begin,
                    gt_end,
                    pred_vals,
                    gt_vals,
                    gt_attr,
                    pred_attr,
                    taken,
                    period,
                    dist_th,
                    out_buf,
                    out_base + p * OUT_STRIDE,
                )
    taken.unsafe_free()
    return 0
