"""Clean-room Kalman filter + Rauch-Tung-Striebel smoother kernel.

Written fresh from the textbook linear-Gaussian recursions (Kalman 1960;
Rauch, Tung & Striebel 1965), matching the published pykalman package's
conventions and operation order so outputs agree element-wise:

    predict:  P(x_t | z_0..t-1)  mean = A @ x + b
                                 cov  = A @ (P @ A.T) + Q
    correct:  S = C @ (P_pred @ C.T) + R
              K = P_pred @ (C.T @ inv(S))
              mean = mean_pred + K @ (z - (C @ mean_pred + d))
              cov  = P_pred - K @ (C @ P_pred)
    smooth:   J = P_filt[t] @ (A.T @ inv(P_pred[t+1]))
              mean_s[t] = mean_f[t] + J @ (mean_s[t+1] - mean_pred[t+1])
              cov_s[t]  = P_filt[t] + J @ ((cov_s[t+1] - P_pred[t+1]) @ J.T)

A timestep whose observation is masked is skipped whole (gain is zero,
filtered equals predicted) — the same all-or-nothing rule pykalman applies
when any component of z_t is masked. All arithmetic is IEEE-754 float64 with
sequential accumulation; small dense matrices only (state/obs dimensions are
runtime values, typically < 16). The matrix inverse is Gauss-Jordan
elimination with partial pivoting — deterministic, and for the
well-conditioned innovation/predicted covariances of stable systems it
agrees with the reference's SVD pseudo-inverse to ~1e-13, far inside the
documented 1e-10 parity tolerance. No third-party Mojo code is used.

Exported C ABI (batch-shaped: a model handle owns the system matrices; each
call processes a whole observation series in one FFI crossing):

    int32_t  pykalmanmojo_abi_version(void)
    void*    pykalmanmojo_model_create(n_dim_state, n_dim_obs,
                                       A, b, Q, C, d, R, x0, P0)
    int32_t  pykalmanmojo_filter(handle, n_timesteps, obs, mask,
                                 out_filt_mean, out_filt_cov)
    int32_t  pykalmanmojo_smooth(handle, n_timesteps, obs, mask,
                                 out_smooth_mean, out_smooth_cov)
    void     pykalmanmojo_model_destroy(handle)

Return codes: 0 success, 1 NULL handle, 2 invalid n_timesteps, 3 singular
matrix encountered during an inverse (the wrapper then serves that call
from the pure-Python reference, whose pseudo-inverse handles singularity).
"""

from std.memory import Pointer, unsafe_memcpy
from std.memory.alloc import unsafe_alloc
from std.origin import MutUntrackedOrigin

comptime ABI_VERSION: Int32 = 1

# C-side pointer spellings (untracked origin: the caller owns the lifetime
# of anything passed in; the library owns what it allocates).
comptime F64Ptr = Pointer[Float64, MutUntrackedOrigin]
comptime I32Ptr = Pointer[Int32, MutUntrackedOrigin]
comptime Handle = Optional[Pointer[UInt8, MutUntrackedOrigin]]


struct KalmanModel(Copyable, Movable):
    """Owned, native copy of one time-invariant linear-Gaussian system."""

    var n_s: Int64  # state dimension
    var n_o: Int64  # observation dimension
    var A: F64Ptr  # [n_s × n_s] transition matrix (row-major)
    var b: F64Ptr  # [n_s] transition offset
    var Q: F64Ptr  # [n_s × n_s] transition covariance
    var C: F64Ptr  # [n_o × n_s] observation matrix
    var d: F64Ptr  # [n_o] observation offset
    var R: F64Ptr  # [n_o × n_o] observation covariance
    var x0: F64Ptr  # [n_s] initial state mean
    var P0: F64Ptr  # [n_s × n_s] initial state covariance

    def __init__(
        out self,
        n_s: Int64,
        n_o: Int64,
        A: F64Ptr,
        b: F64Ptr,
        Q: F64Ptr,
        C: F64Ptr,
        d: F64Ptr,
        R: F64Ptr,
        x0: F64Ptr,
        P0: F64Ptr,
    ):
        self.n_s = n_s
        self.n_o = n_o
        self.A = A
        self.b = b
        self.Q = Q
        self.C = C
        self.d = d
        self.R = R
        self.x0 = x0
        self.P0 = P0


# --- small dense float64 linear algebra (row-major, sequential accumulation) --


def _matmul_nn(a: F64Ptr, b: F64Ptr, dst: F64Ptr, m: Int, k: Int, n: Int):
    """dst[m×n] = a[m×k] @ b[k×n] (overwrite)."""
    for i in range(m):
        for j in range(n):
            var acc = Float64(0.0)
            for p in range(k):
                acc += a[unsafe_offset=i * k + p] * b[unsafe_offset=p * n + j]
            dst[unsafe_offset=i * n + j] = acc


def _matmul_bt(a: F64Ptr, b: F64Ptr, dst: F64Ptr, m: Int, k: Int, n: Int):
    """dst[m×n] = a[m×k] @ b[n×k]^T (overwrite, no transpose materialized)."""
    for i in range(m):
        for j in range(n):
            var acc = Float64(0.0)
            for p in range(k):
                acc += a[unsafe_offset=i * k + p] * b[unsafe_offset=j * k + p]
            dst[unsafe_offset=i * n + j] = acc


def _matmul_at(a: F64Ptr, b: F64Ptr, dst: F64Ptr, m: Int, k: Int, n: Int):
    """dst[m×n] = a[k×m]^T @ b[k×n] (overwrite, no transpose materialized)."""
    for i in range(m):
        for j in range(n):
            var acc = Float64(0.0)
            for p in range(k):
                acc += a[unsafe_offset=p * m + i] * b[unsafe_offset=p * n + j]
            dst[unsafe_offset=i * n + j] = acc


def _matvec(a: F64Ptr, x: F64Ptr, dst: F64Ptr, m: Int, n: Int):
    """dst[m] = a[m×n] @ x[n] (overwrite)."""
    for i in range(m):
        var acc = Float64(0.0)
        for p in range(n):
            acc += a[unsafe_offset=i * n + p] * x[unsafe_offset=p]
        dst[unsafe_offset=i] = acc


def _inverse(a: F64Ptr, dst: F64Ptr, aug: F64Ptr, n: Int) -> Int32:
    """dst[n×n] = inv(a[n×n]) via Gauss-Jordan with partial pivoting.

    `aug` is caller workspace of n × 2n. Returns 0 on success, 3 when an
    exactly-zero pivot is hit (singular matrix).
    """
    var w = 2 * n
    for i in range(n):
        for j in range(n):
            aug[unsafe_offset=i * w + j] = a[unsafe_offset=i * n + j]
            if i == j:
                aug[unsafe_offset=i * w + n + j] = 1.0
            else:
                aug[unsafe_offset=i * w + n + j] = 0.0
    for col in range(n):
        var piv = col
        var best = aug[unsafe_offset=col * w + col]
        if best < 0.0:
            best = -best
        for r in range(col + 1, n):
            var v = aug[unsafe_offset=r * w + col]
            if v < 0.0:
                v = -v
            if v > best:
                best = v
                piv = r
        if best == 0.0:
            return 3
        if piv != col:
            for j in range(w):
                var tmp = aug[unsafe_offset=col * w + j]
                aug[unsafe_offset=col * w + j] = aug[unsafe_offset=piv * w + j]
                aug[unsafe_offset=piv * w + j] = tmp
        var dv = aug[unsafe_offset=col * w + col]
        for j in range(w):
            aug[unsafe_offset=col * w + j] /= dv
        for r in range(n):
            if r != col:
                var f = aug[unsafe_offset=r * w + col]
                if f != 0.0:
                    for j in range(w):
                        aug[unsafe_offset=r * w + j] -= f * aug[
                            unsafe_offset=col * w + j
                        ]
    for i in range(n):
        for j in range(n):
            dst[unsafe_offset=i * n + j] = aug[unsafe_offset=i * w + n + j]
    return 0


# --- recursion steps (operate on caller-provided per-step buffers) ---


def _predict(m: KalmanModel, fm_prev: F64Ptr, fc_prev: F64Ptr, pm: F64Ptr,
             pc: F64Ptr, tmp: F64Ptr):
    """P(x_t | z_0..t-1) from filtered (t-1): pm = A@fm + b, pc = A@(fc@A.T)+Q."""
    var n = Int(m.n_s)
    _matvec(m.A, fm_prev, pm, n, n)
    for i in range(n):
        pm[unsafe_offset=i] += m.b[unsafe_offset=i]
    _matmul_bt(fc_prev, m.A, tmp, n, n, n)  # fc_prev @ A.T
    _matmul_nn(m.A, tmp, pc, n, n, n)  # A @ (fc_prev @ A.T)
    for i in range(n * n):
        pc[unsafe_offset=i] += m.Q[unsafe_offset=i]


def _correct(
    m: KalmanModel,
    obs: F64Ptr,
    t: Int,
    pm: F64Ptr,
    pc: F64Ptr,
    fm: F64Ptr,
    fc: F64Ptr,
    K: F64Ptr,
    S: F64Ptr,
    invS: F64Ptr,
    aug: F64Ptr,
    tmp2: F64Ptr,
    tmp3: F64Ptr,
    tmp4: F64Ptr,
    tmp5: F64Ptr,
    pom: F64Ptr,
    resid: F64Ptr,
) -> Int32:
    """Filtered (t) from predicted (t) and observation row t of `obs`."""
    var ns = Int(m.n_s)
    var no = Int(m.n_o)
    _matmul_bt(pc, m.C, tmp2, ns, ns, no)  # P_pred @ C.T  [ns×no]
    _matmul_nn(m.C, tmp2, S, no, ns, no)  # C @ (P_pred @ C.T)  [no×no]
    for i in range(no * no):
        S[unsafe_offset=i] += m.R[unsafe_offset=i]
    var rc = _inverse(S, invS, aug, no)
    if rc != 0:
        return rc
    _matmul_at(m.C, invS, tmp3, ns, no, no)  # C.T @ inv(S)  [ns×no]
    _matmul_nn(pc, tmp3, K, ns, ns, no)  # K = P_pred @ (C.T @ inv(S))  [ns×no]
    _matvec(m.C, pm, pom, no, ns)  # C @ mean_pred
    for i in range(no):
        pom[unsafe_offset=i] += m.d[unsafe_offset=i]
    for i in range(no):
        resid[unsafe_offset=i] = obs[unsafe_offset=t * no + i] - pom[unsafe_offset=i]
    _matvec(K, resid, fm, ns, no)
    for i in range(ns):
        fm[unsafe_offset=i] += pm[unsafe_offset=i]
    _matmul_nn(m.C, pc, tmp4, no, ns, ns)  # C @ P_pred  [no×ns]
    _matmul_nn(K, tmp4, tmp5, ns, no, ns)  # K @ (C @ P_pred)  [ns×ns]
    for i in range(ns * ns):
        fc[unsafe_offset=i] = pc[unsafe_offset=i] - tmp5[unsafe_offset=i]
    return 0


def _filter_all(
    m: KalmanModel,
    T: Int,
    obs: F64Ptr,
    mask: I32Ptr,
    out_fm: F64Ptr,
    out_fc: F64Ptr,
) -> Int32:
    """Forward pass over the whole series; filtered rows written per timestep.

    out_fm is [T×n_s], out_fc is [T×n_s×n_s]. Masked timesteps (mask[t] != 0)
    skip the correction: filtered equals predicted, pykalman's rule.
    """
    var ns = Int(m.n_s)
    var no = Int(m.n_o)
    var nmax = ns
    if no > nmax:
        nmax = no

    var pm = unsafe_alloc[Float64](ns)
    var pc = unsafe_alloc[Float64](ns * ns)
    var fm_prev = unsafe_alloc[Float64](ns)
    var fc_prev = unsafe_alloc[Float64](ns * ns)
    var K = unsafe_alloc[Float64](ns * no)
    var S = unsafe_alloc[Float64](no * no)
    var invS = unsafe_alloc[Float64](no * no)
    var aug = unsafe_alloc[Float64](nmax * 2 * nmax)
    var tmp2 = unsafe_alloc[Float64](ns * no)
    var tmp3 = unsafe_alloc[Float64](ns * no)
    var tmp4 = unsafe_alloc[Float64](no * ns)
    var tmp5 = unsafe_alloc[Float64](ns * ns)
    var pom = unsafe_alloc[Float64](no)
    var resid = unsafe_alloc[Float64](no)

    var rc = Int32(0)
    for t in range(T):
        var fm_t = out_fm.unsafe_offset(t * ns)
        var fc_t = out_fc.unsafe_offset(t * ns * ns)
        if t == 0:
            unsafe_memcpy(dest=pm, src=m.x0, count=ns)
            unsafe_memcpy(dest=pc, src=m.P0, count=ns * ns)
        else:
            _predict(m, fm_prev, fc_prev, pm, pc, tmp5)
        if mask[unsafe_offset=t] != 0:
            # Missing observation: gain is zero, filtered equals predicted.
            unsafe_memcpy(dest=fm_t, src=pm, count=ns)
            unsafe_memcpy(dest=fc_t, src=pc, count=ns * ns)
        else:
            rc = _correct(
                m, obs, t, pm, pc, fm_t, fc_t, K, S, invS, aug, tmp2, tmp3,
                tmp4, tmp5, pom, resid,
            )
            if rc != 0:
                break
        unsafe_memcpy(dest=fm_prev, src=fm_t, count=ns)
        unsafe_memcpy(dest=fc_prev, src=fc_t, count=ns * ns)

    pm.unsafe_free()
    pc.unsafe_free()
    fm_prev.unsafe_free()
    fc_prev.unsafe_free()
    K.unsafe_free()
    S.unsafe_free()
    invS.unsafe_free()
    aug.unsafe_free()
    tmp2.unsafe_free()
    tmp3.unsafe_free()
    tmp4.unsafe_free()
    tmp5.unsafe_free()
    pom.unsafe_free()
    resid.unsafe_free()
    return rc


@export
def pykalmanmojo_abi_version() abi("C") -> Int32:
    return ABI_VERSION


@export
def pykalmanmojo_model_create(
    n_dim_state: Int64,
    n_dim_obs: Int64,
    A: F64Ptr,
    b: F64Ptr,
    Q: F64Ptr,
    C: F64Ptr,
    d: F64Ptr,
    R: F64Ptr,
    x0: F64Ptr,
    P0: F64Ptr,
) abi("C") -> Handle:
    """Copy the system matrices into a native model; NULL on invalid input."""
    if n_dim_state < 1 or n_dim_obs < 1:
        return None
    var ns = Int(n_dim_state)
    var no = Int(n_dim_obs)

    var A_c = unsafe_alloc[Float64](ns * ns)
    unsafe_memcpy(dest=A_c, src=A, count=ns * ns)
    var b_c = unsafe_alloc[Float64](ns)
    unsafe_memcpy(dest=b_c, src=b, count=ns)
    var Q_c = unsafe_alloc[Float64](ns * ns)
    unsafe_memcpy(dest=Q_c, src=Q, count=ns * ns)
    var C_c = unsafe_alloc[Float64](no * ns)
    unsafe_memcpy(dest=C_c, src=C, count=no * ns)
    var d_c = unsafe_alloc[Float64](no)
    unsafe_memcpy(dest=d_c, src=d, count=no)
    var R_c = unsafe_alloc[Float64](no * no)
    unsafe_memcpy(dest=R_c, src=R, count=no * no)
    var x0_c = unsafe_alloc[Float64](ns)
    unsafe_memcpy(dest=x0_c, src=x0, count=ns)
    var P0_c = unsafe_alloc[Float64](ns * ns)
    unsafe_memcpy(dest=P0_c, src=P0, count=ns * ns)

    var m = unsafe_alloc[KalmanModel](1)
    m[] = KalmanModel(n_dim_state, n_dim_obs, A_c, b_c, Q_c, C_c, d_c, R_c,
                      x0_c, P0_c)
    return m.unsafe_bitcast[UInt8]()


@export
def pykalmanmojo_filter(
    handle: Handle,
    n_timesteps: Int64,
    obs: F64Ptr,
    mask: I32Ptr,
    out_filt_mean: F64Ptr,
    out_filt_cov: F64Ptr,
) abi("C") -> Int32:
    """Filter the whole observation series.

    `obs` is [T×n_o] row-major, `mask` is [T] (nonzero = missing timestep),
    outputs are [T×n_s] and [T×n_s×n_s]. Returns 0 on success.
    """
    if not handle:
        return 1
    if n_timesteps < 1:
        return 2
    var m = handle.value().unsafe_bitcast[KalmanModel]()
    return _filter_all(m[], Int(n_timesteps), obs, mask, out_filt_mean,
                       out_filt_cov)


@export
def pykalmanmojo_smooth(
    handle: Handle,
    n_timesteps: Int64,
    obs: F64Ptr,
    mask: I32Ptr,
    out_smooth_mean: F64Ptr,
    out_smooth_cov: F64Ptr,
) abi("C") -> Int32:
    """Rauch-Tung-Striebel smooth the whole observation series.

    Runs the forward filter internally, then the backward pass. The backward
    pass recomputes predicted (t+1) from stored filtered (t) with the same
    `_predict` routine the forward pass uses, so values are bit-identical to
    a stored-predictions implementation while halving peak memory.
    """
    if not handle:
        return 1
    if n_timesteps < 1:
        return 2
    var m = handle.value().unsafe_bitcast[KalmanModel]()
    var ns = Int(m[].n_s)
    var T = Int(n_timesteps)

    var fm_all = unsafe_alloc[Float64](T * ns)
    var fc_all = unsafe_alloc[Float64](T * ns * ns)
    var rc = _filter_all(m[], T, obs, mask, fm_all, fc_all)
    if rc != 0:
        fm_all.unsafe_free()
        fc_all.unsafe_free()
        return rc

    # Backward pass workspace.
    var pm1 = unsafe_alloc[Float64](ns)
    var pc1 = unsafe_alloc[Float64](ns * ns)
    var invP = unsafe_alloc[Float64](ns * ns)
    var aug = unsafe_alloc[Float64](ns * 2 * ns)
    var tmpA = unsafe_alloc[Float64](ns * ns)
    var J = unsafe_alloc[Float64](ns * ns)
    var tmpB = unsafe_alloc[Float64](ns * ns)
    var tmpC = unsafe_alloc[Float64](ns * ns)
    var dm = unsafe_alloc[Float64](ns)
    var dC = unsafe_alloc[Float64](ns * ns)
    var s_next = unsafe_alloc[Float64](ns)
    var sc_next = unsafe_alloc[Float64](ns * ns)

    # t = T-1: smoothed equals filtered.
    for i in range(ns):
        var v = fm_all[unsafe_offset=(T - 1) * ns + i]
        s_next[unsafe_offset=i] = v
        out_smooth_mean[unsafe_offset=(T - 1) * ns + i] = v
    for i in range(ns * ns):
        var v = fc_all[unsafe_offset=(T - 1) * ns * ns + i]
        sc_next[unsafe_offset=i] = v
        out_smooth_cov[unsafe_offset=(T - 1) * ns * ns + i] = v

    var t = T - 2
    while t >= 0 and rc == 0:
        # predicted (t+1) from filtered (t) — identical to the forward pass.
        _predict(m[], fm_all.unsafe_offset(t * ns), fc_all.unsafe_offset(t * ns * ns), pm1,
                 pc1, tmpC)
        rc = _inverse(pc1, invP, aug, ns)
        if rc != 0:
            break
        _matmul_at(m[].A, invP, tmpA, ns, ns, ns)  # A.T @ inv(P_pred)
        _matmul_nn(fc_all.unsafe_offset(t * ns * ns), tmpA, J, ns, ns, ns)  # J
        for i in range(ns):
            dm[unsafe_offset=i] = s_next[unsafe_offset=i] - pm1[unsafe_offset=i]
        _matvec(J, dm, out_smooth_mean.unsafe_offset(t * ns), ns, ns)
        for i in range(ns):
            out_smooth_mean[unsafe_offset=t * ns + i] += fm_all[
                unsafe_offset=t * ns + i
            ]
        for i in range(ns * ns):
            dC[unsafe_offset=i] = sc_next[unsafe_offset=i] - pc1[unsafe_offset=i]
        _matmul_bt(dC, J, tmpB, ns, ns, ns)  # (S_next - P_pred) @ J.T
        _matmul_nn(J, tmpB, tmpC, ns, ns, ns)  # J @ (...) @ J.T
        for i in range(ns * ns):
            var v = fc_all[unsafe_offset=t * ns * ns + i] + tmpC[unsafe_offset=i]
            out_smooth_cov[unsafe_offset=t * ns * ns + i] = v
            sc_next[unsafe_offset=i] = v
        for i in range(ns):
            s_next[unsafe_offset=i] = out_smooth_mean[unsafe_offset=t * ns + i]
        t -= 1

    fm_all.unsafe_free()
    fc_all.unsafe_free()
    pm1.unsafe_free()
    pc1.unsafe_free()
    invP.unsafe_free()
    aug.unsafe_free()
    tmpA.unsafe_free()
    J.unsafe_free()
    tmpB.unsafe_free()
    tmpC.unsafe_free()
    dm.unsafe_free()
    dC.unsafe_free()
    s_next.unsafe_free()
    sc_next.unsafe_free()
    return rc


@export
def pykalmanmojo_model_destroy(handle: Handle) abi("C"):
    if not handle:
        return
    var m = handle.value().unsafe_bitcast[KalmanModel]()
    m[].A.unsafe_free()
    m[].b.unsafe_free()
    m[].Q.unsafe_free()
    m[].C.unsafe_free()
    m[].d.unsafe_free()
    m[].R.unsafe_free()
    m[].x0.unsafe_free()
    m[].P0.unsafe_free()
    m.unsafe_free()
