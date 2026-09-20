"""Clean-room parcel CAPE/CIN kernel for standard atmospheric soundings.

Written fresh from textbook thermodynamics (Bolton 1980; Emanuel
"Atmospheric Convection" 1994; Wallace & Hobbs 3.54) — the same algorithm
as the vendored pure-Python reference `metpy_mojo/_reference.py`, which
documents the formula-by-formula provenance and the black-box parity
measurements against the MetPy oracle. No third-party Mojo code is used.

Exported C ABI (batched: one call computes many columns):

    int32_t metpycapemojo_abi_version(void)
    int32_t metpycapemojo_cape_cin(int64_t ncols, int64_t nlev,
                                   const double* p, const double* t,
                                   const double* td, int32_t which,
                                   double mu_depth_hpa,
                                   double* out_cape, double* out_cin,
                                   double* out_lclp, double* out_lfcp,
                                   double* out_elp)

Column c occupies p/t/td[c*nlev, (c+1)*nlev) with pressure strictly
decreasing (hPa); temperatures in K. `which`: 0 = surface parcel,
1 = most-unstable parcel (max Bolton theta-e within mu_depth_hpa of the
surface). Outputs are per-column CAPE/CIN (J/kg) and LCL/LFC/EL pressures
(hPa); a nonexistent LFC/EL is NaN (no LFC => cape = cin = 0; no EL
crossing => EL = NaN and CAPE integrates to the profile top).

Return codes: 0 = success; 1 = invalid dimensions; 2 = invalid which;
3 = a column holds non-finite data, non-positive pressure, or es >= p
(outside the standard-atmosphere scope). All arithmetic is IEEE-754
float64 in the same operation order as the Python reference.
"""

from std.math import exp, log, pow, nan
from std.memory import Pointer
from std.memory.alloc import unsafe_alloc
from std.origin import MutUntrackedOrigin

comptime ABI_VERSION: Int32 = 1

# Physical constants (mirrored from metpy_mojo/_reference.py).
comptime RD: Float64 = 287.05
comptime RV: Float64 = 461.51
comptime CPD: Float64 = 1004.0
comptime LV: Float64 = 2.5e6
comptime EPS: Float64 = RD / RV
comptime KAPPA: Float64 = 2.0 / 7.0
comptime P_REF: Float64 = 1000.0
comptime RK45_RTOL: Float64 = 1e-9
comptime RK45_ATOL: Float64 = 1e-11

comptime F64Ptr = Pointer[Float64, MutUntrackedOrigin]


def _es(t: Float64) -> Float64:
    """Saturation vapor pressure (hPa), Bolton 1980 eq. 10."""
    return 6.112 * exp(17.67 * (t - 273.15) / (t - 29.65))


def _sat_mr(t: Float64, p: Float64) -> Float64:
    var e = _es(t)
    return EPS * e / (p - e)


def _mr_from_td(td: Float64, p: Float64) -> Float64:
    var e = _es(td)
    return EPS * e / (p - e)


def _virtual_t(t: Float64, r: Float64) -> Float64:
    return t * (1.0 + r / EPS) / (1.0 + r)


def _bolton_lcl_t(t: Float64, td: Float64) -> Float64:
    """Closed-form LCL temperature estimate (K), Bolton 1980 eq. 15."""
    return 1.0 / (1.0 / (td - 56.0) + log(t / td) / 800.0) + 56.0


def _dry_kappa(r: Float64) -> Float64:
    """Moist-air Poisson exponent used inside the LCL solve."""
    return KAPPA * (1.0 + 0.608 * r) / (1.0 + 0.84 * r)


def _lcl(t0: Float64, td0: Float64, p0: Float64) -> Tuple[Float64, Float64]:
    """Exact LCL (pressure hPa, temperature K); see _reference.lcl."""
    if td0 >= t0:
        return Tuple(p0, t0)
    var r0 = _mr_from_td(td0, p0)
    var kappa_m = _dry_kappa(r0)
    var t_b = _bolton_lcl_t(t0, td0)
    var lo = t_b - 15.0
    var hi = t_b + 15.0
    for _ in range(20):
        var p_l = p0 * pow(hi / t0, 1.0 / kappa_m)
        if _sat_mr(hi, p_l) - r0 > 0.0:
            break
        hi += 15.0
    for _ in range(20):
        var p_l = p0 * pow(lo / t0, 1.0 / kappa_m)
        if _sat_mr(lo, p_l) - r0 < 0.0:
            break
        lo -= 15.0
    for _ in range(60):
        var mid = 0.5 * (lo + hi)
        var p_l = p0 * pow(mid / t0, 1.0 / kappa_m)
        if _sat_mr(mid, p_l) - r0 < 0.0:
            lo = mid
        else:
            hi = mid
    var t_lcl = 0.5 * (lo + hi)
    return Tuple(p0 * pow(t_lcl / t0, 1.0 / kappa_m), t_lcl)


def _moist_rhs(t: Float64, p: Float64) -> Float64:
    """Pseudoadiabatic moist lapse rate dT/dp (K/hPa)."""
    var rs = _sat_mr(t, p)
    var numerator = RD * t + LV * rs
    var denominator = p * (CPD + LV * LV * rs / (RV * t * t))
    return numerator / denominator


def _rk45_to_grid(
    t0: Float64, p0: Float64, targets: F64Ptr, start: Int, n_targets: Int, out_buf: F64Ptr
):
    """Adaptive Dormand-Prince RK45 on the moist adiabat, sampled exactly
    at targets[start .. start+n_targets) (strictly decreasing, below p0).
    Deterministic controller identical to the Python reference."""
    var t = t0
    var p = p0
    var h = 0.25 * (targets[unsafe_offset=start] - p0)
    for j in range(n_targets):
        var target = targets[unsafe_offset=start + j]
        while p > target:
            var step = h
            if p + step <= target:
                step = target - p
            var k1 = _moist_rhs(t, p)
            var k2 = _moist_rhs(t + step * 0.2 * k1, p + 0.2 * step)
            var k3 = _moist_rhs(
                t + step * (3.0 / 40.0 * k1 + 9.0 / 40.0 * k2), p + 0.3 * step
            )
            var k4 = _moist_rhs(
                t + step * (44.0 / 45.0 * k1 - 56.0 / 15.0 * k2 + 32.0 / 9.0 * k3),
                p + 0.8 * step,
            )
            var k5 = _moist_rhs(
                t
                + step
                * (
                    19372.0 / 6561.0 * k1
                    - 25360.0 / 2187.0 * k2
                    + 64448.0 / 6561.0 * k3
                    - 212.0 / 729.0 * k4
                ),
                p + 8.0 / 9.0 * step,
            )
            var k6 = _moist_rhs(
                t
                + step
                * (
                    9017.0 / 3168.0 * k1
                    - 355.0 / 33.0 * k2
                    + 46732.0 / 5247.0 * k3
                    + 49.0 / 176.0 * k4
                    - 5103.0 / 18656.0 * k5
                ),
                p + step,
            )
            var t5 = t + step * (
                35.0 / 384.0 * k1 + 500.0 / 1113.0 * k3 + 125.0 / 192.0 * k4
                - 2187.0 / 6784.0 * k5 + 11.0 / 84.0 * k6
            )
            var k7 = _moist_rhs(t5, p + step)
            var err = step * (
                71.0 / 57600.0 * k1 - 71.0 / 16695.0 * k3 + 71.0 / 1920.0 * k4
                - 17253.0 / 339200.0 * k5 + 22.0 / 525.0 * k6 - 1.0 / 40.0 * k7
            )
            var scale = RK45_ATOL + RK45_RTOL * max(abs(t), abs(t5))
            var err_norm = abs(err) / scale
            if err_norm <= 1.0:
                t = t5
                p = p + step
                var factor: Float64
                if err_norm == 0.0:
                    factor = 10.0
                else:
                    factor = min(10.0, max(0.2, 0.9 * pow(err_norm, -0.2)))
                h = step * factor
            else:
                h = step * max(0.2, 0.9 * pow(err_norm, -0.2))
        out_buf[unsafe_offset=j] = t


def _theta_e(t: Float64, td: Float64, p: Float64) -> Float64:
    """Equivalent potential temperature (K), Bolton 1980 eq. 38."""
    var r_gkg = 1e3 * _mr_from_td(td, p)
    var t_lcl = _bolton_lcl_t(t, td)
    return (
        t
        * pow(P_REF / p, 0.2854 * (1.0 - 0.28e-3 * r_gkg))
        * exp((3376.0 / t_lcl - 2.54) * 1e-3 * r_gkg * (1.0 + 0.81e-3 * r_gkg))
    )


def _interp_aug(aug_x: F64Ptr, aug_b: F64Ptr, na: Int, xq: Float64) -> Float64:
    """Linear interpolation on the descending augmented grid (clamped)."""
    if xq >= aug_x[unsafe_offset=0]:
        return aug_b[unsafe_offset=0]
    if xq <= aug_x[unsafe_offset=na - 1]:
        return aug_b[unsafe_offset=na - 1]
    var j = 0
    while not (aug_x[unsafe_offset=j] >= xq and xq >= aug_x[unsafe_offset=j + 1]):
        j += 1
    var x0 = aug_x[unsafe_offset=j]
    var x1 = aug_x[unsafe_offset=j + 1]
    if x1 == x0:
        return aug_b[unsafe_offset=j]
    return aug_b[unsafe_offset=j] + (aug_b[unsafe_offset=j + 1] - aug_b[
        unsafe_offset=j
    ]) * (xq - x0) / (x1 - x0)


def _trapz(
    aug_x: F64Ptr, aug_b: F64Ptr, na: Int, x_lo: Float64, x_hi: Float64, clip_pos: Bool
) -> Float64:
    """Trapezoid over the augmented grid from x_lo down to x_hi (both
    interpolated), optionally clipped to the positive part. The result is
    negative for positive integrands (x decreases); the caller signs it."""
    var total = 0.0
    var prev_x = x_lo
    var prev_b = _interp_aug(aug_x, aug_b, na, x_lo)
    if clip_pos:
        prev_b = max(prev_b, 0.0)
    for j in range(na):
        var xxj = aug_x[unsafe_offset=j]
        if xxj > x_lo or xxj < x_hi:
            continue
        var bbj = aug_b[unsafe_offset=j]
        if clip_pos:
            bbj = max(bbj, 0.0)
        total += 0.5 * (prev_b + bbj) * (xxj - prev_x)
        prev_x = xxj
        prev_b = bbj
    var last_b = _interp_aug(aug_x, aug_b, na, x_hi)
    if clip_pos:
        last_b = max(last_b, 0.0)
    total += 0.5 * (prev_b + last_b) * (x_hi - prev_x)
    return total


def _column(
    p: F64Ptr,
    t: F64Ptr,
    td: F64Ptr,
    n: Int,
    which: Int32,
    mu_depth: Float64,
    gp: F64Ptr,
    gt: F64Ptr,
    gtd: F64Ptr,
    tp: F64Ptr,
    bb: F64Ptr,
    xx: F64Ptr,
    ups: F64Ptr,
    downs: F64Ptr,
    aug_x: F64Ptr,
    aug_b: F64Ptr,
    rk_out: F64Ptr,
) -> Tuple[Float64, Float64, Float64, Float64, Float64]:
    """One column: returns (cape, cin, p_lcl, p_lfc, p_el).

    Scratch buffers: gp/gt/gtd/tp/bb/xx hold n+1; ups/downs hold n;
    aug_x/aug_b hold 2n+2; rk_out holds n.
    """
    var nanv = nan[DType.float64]()

    # Parcel start level.
    var i0 = 0
    if which == 1:
        var best = 0
        var best_te = -1.0e300
        var p_floor = p[unsafe_offset=0] - mu_depth
        for i in range(n):
            if p[unsafe_offset=i] >= p_floor:
                var te = _theta_e(
                    t[unsafe_offset=i], td[unsafe_offset=i], p[unsafe_offset=i]
                )
                if te > best_te:
                    best_te = te
                    best = i
        i0 = best

    var t0 = t[unsafe_offset=i0]
    var td0 = td[unsafe_offset=i0]
    var p0 = p[unsafe_offset=i0]
    var r0 = _mr_from_td(td0, p0)
    var lcl_res = _lcl(t0, td0, p0)
    var p_lcl = lcl_res[0]
    var t_lcl = lcl_res[1]

    # Integration grid: env levels from the start level, LCL inserted when
    # it falls strictly between two levels.
    var m_env = n - i0
    var m = m_env
    var k_insert = -1
    if p_lcl >= p[unsafe_offset=i0] - 1e-12:
        p_lcl = p[unsafe_offset=i0]
        t_lcl = t[unsafe_offset=i0]
        for j in range(m_env):
            gp[unsafe_offset=j] = p[unsafe_offset=i0 + j]
            gt[unsafe_offset=j] = t[unsafe_offset=i0 + j]
            gtd[unsafe_offset=j] = td[unsafe_offset=i0 + j]
    elif p_lcl <= p[unsafe_offset=n - 1]:
        for j in range(m_env):
            gp[unsafe_offset=j] = p[unsafe_offset=i0 + j]
            gt[unsafe_offset=j] = t[unsafe_offset=i0 + j]
            gtd[unsafe_offset=j] = td[unsafe_offset=i0 + j]
    else:
        var k = 1
        while p[unsafe_offset=i0 + k] > p_lcl:
            k += 1
        # env[k-1] > p_lcl > env[k]; env values linear-in-p interpolated.
        var pe_hi = p[unsafe_offset=i0 + k - 1]
        var pe_lo = p[unsafe_offset=i0 + k]
        var frac = (pe_hi - p_lcl) / (pe_hi - pe_lo)
        var t_at = t[unsafe_offset=i0 + k - 1] + (
            t[unsafe_offset=i0 + k] - t[unsafe_offset=i0 + k - 1]
        ) * frac
        var td_at = td[unsafe_offset=i0 + k - 1] + (
            td[unsafe_offset=i0 + k] - td[unsafe_offset=i0 + k - 1]
        ) * frac
        for j in range(k):
            gp[unsafe_offset=j] = p[unsafe_offset=i0 + j]
            gt[unsafe_offset=j] = t[unsafe_offset=i0 + j]
            gtd[unsafe_offset=j] = td[unsafe_offset=i0 + j]
        gp[unsafe_offset=k] = p_lcl
        gt[unsafe_offset=k] = t_at
        gtd[unsafe_offset=k] = td_at
        for j in range(k, m_env):
            gp[unsafe_offset=j + 1] = p[unsafe_offset=i0 + j]
            gt[unsafe_offset=j + 1] = t[unsafe_offset=i0 + j]
            gtd[unsafe_offset=j + 1] = td[unsafe_offset=i0 + j]
        m = m_env + 1
        k_insert = k

    # Parcel temperature on the grid.
    for j in range(m):
        if gp[unsafe_offset=j] >= p_lcl:
            tp[unsafe_offset=j] = t0 * pow(gp[unsafe_offset=j] / p0, KAPPA)
    if k_insert >= 0:
        tp[unsafe_offset=k_insert] = t_lcl
    var jsa = 0
    while jsa < m and gp[unsafe_offset=jsa] >= p_lcl:
        jsa += 1
    var n_above = m - jsa
    if n_above > 0:
        var t_dry_at_lcl = t0 * pow(p_lcl / p0, KAPPA)
        _rk45_to_grid(t_dry_at_lcl, p_lcl, gp, jsa, n_above, rk_out)
        for j in range(n_above):
            tp[unsafe_offset=jsa + j] = rk_out[unsafe_offset=j]

    # Buoyancy with exact virtual temperatures; x = ln p.
    for j in range(m):
        var r_env = _mr_from_td(gtd[unsafe_offset=j], gp[unsafe_offset=j])
        var r_par = _sat_mr(tp[unsafe_offset=j], gp[unsafe_offset=j])
        if r_par > r0:
            r_par = r0
        bb[unsafe_offset=j] = RD * (
            _virtual_t(tp[unsafe_offset=j], r_par)
            - _virtual_t(gt[unsafe_offset=j], r_env)
        )
        xx[unsafe_offset=j] = log(gp[unsafe_offset=j])

    # Buoyancy zero crossings (LFC/EL gating; see _reference for the
    # decoded MetPy cape_cin semantics).
    var n_up = 0
    var n_down = 0
    for j in range(1, m):
        var b0 = bb[unsafe_offset=j - 1]
        var b1 = bb[unsafe_offset=j]
        if b0 <= 0.0 and b1 > 0.0:
            var frac = b0 / (b0 - b1)
            ups[unsafe_offset=n_up] = xx[unsafe_offset=j - 1] + frac * (
                xx[unsafe_offset=j] - xx[unsafe_offset=j - 1]
            )
            n_up += 1
        elif b0 >= 0.0 and b1 < 0.0:
            var frac = b0 / (b0 - b1)
            downs[unsafe_offset=n_down] = xx[unsafe_offset=j - 1] + frac * (
                xx[unsafe_offset=j] - xx[unsafe_offset=j - 1]
            )
            n_down += 1
    if n_up == 0:
        return Tuple(0.0, 0.0, p_lcl, nanv, nanv)

    var x_lfc = ups[unsafe_offset=0]
    var x_el = nanv
    if n_down > 0:
        var x_down_last = downs[unsafe_offset=n_down - 1]
        var found = 0.0
        var has = False
        for j in range(n_up):
            if ups[unsafe_offset=j] > x_down_last:
                has = True
                found = ups[unsafe_offset=j]
        if has:
            x_lfc = found
            x_el = x_down_last
    var p_lfc = exp(x_lfc)
    var p_el = nanv
    var x_top = xx[unsafe_offset=m - 1]
    if not (x_el != x_el):  # x_el is not NaN
        p_el = exp(x_el)
        x_top = x_el

    # Augment the grid with every strictly interior zero crossing.
    var na = 0
    for j in range(m):
        aug_x[unsafe_offset=na] = xx[unsafe_offset=j]
        aug_b[unsafe_offset=na] = bb[unsafe_offset=j]
        na += 1
        if j + 1 < m:
            var b0 = bb[unsafe_offset=j]
            var b1 = bb[unsafe_offset=j + 1]
            if ((b0 > 0.0) != (b1 > 0.0)) and b0 != b1:
                var frac = b0 / (b0 - b1)
                if frac > 0.0 and frac < 1.0:
                    aug_x[unsafe_offset=na] = xx[unsafe_offset=j] + frac * (
                        xx[unsafe_offset=j + 1] - xx[unsafe_offset=j]
                    )
                    aug_b[unsafe_offset=na] = 0.0
                    na += 1

    var cape = -_trapz(aug_x, aug_b, na, x_lfc, x_top, True)
    if cape < 0.0:
        cape = 0.0
    var cin = -_trapz(aug_x, aug_b, na, xx[unsafe_offset=0], x_lfc, False)
    if cin > 0.0:
        cin = 0.0
    return Tuple(cape, cin, p_lcl, p_lfc, p_el)


@export
def metpycapemojo_abi_version() abi("C") -> Int32:
    return ABI_VERSION


@export
def metpycapemojo_cape_cin(
    ncols: Int64,
    nlev: Int64,
    p: F64Ptr,
    t: F64Ptr,
    td: F64Ptr,
    which: Int32,
    mu_depth: Float64,
    out_cape: F64Ptr,
    out_cin: F64Ptr,
    out_lclp: F64Ptr,
    out_lfcp: F64Ptr,
    out_elp: F64Ptr,
) abi("C") -> Int32:
    """Batched CAPE/CIN driver. See the module docstring for the contract."""
    if ncols <= 0 or nlev < 2:
        return 1
    if which < 0 or which > 1:
        return 2
    var n = Int(nlev)
    # Defensive data validation (the wrapper validates first and raises;
    # this keeps the C ABI safe against direct misuse).
    for i in range(Int(ncols) * n):
        var pv = p[unsafe_offset=i]
        var tv = t[unsafe_offset=i]
        var tdv = td[unsafe_offset=i]
        if not (pv > 0.0) or pv != pv or tv != tv or tdv != tdv:
            return 3
        if _es(tv) >= pv or _es(tdv) >= pv:
            return 3
    for c in range(Int(ncols)):
        var base = c * n
        for j in range(1, n):
            if p[unsafe_offset=base + j] >= p[unsafe_offset=base + j - 1]:
                return 3

    # Scratch for one column at a time (reused across columns).
    var gp = unsafe_alloc[Float64](n + 1)
    var gt = unsafe_alloc[Float64](n + 1)
    var gtd = unsafe_alloc[Float64](n + 1)
    var tp = unsafe_alloc[Float64](n + 1)
    var bb = unsafe_alloc[Float64](n + 1)
    var xx = unsafe_alloc[Float64](n + 1)
    var ups = unsafe_alloc[Float64](n)
    var downs = unsafe_alloc[Float64](n)
    var aug_x = unsafe_alloc[Float64](2 * n + 2)
    var aug_b = unsafe_alloc[Float64](2 * n + 2)
    var rk_out = unsafe_alloc[Float64](n)

    for c in range(Int(ncols)):
        var base = c * n
        var res = _column(
            p.unsafe_offset(base),
            t.unsafe_offset(base),
            td.unsafe_offset(base),
            n,
            which,
            mu_depth,
            gp,
            gt,
            gtd,
            tp,
            bb,
            xx,
            ups,
            downs,
            aug_x,
            aug_b,
            rk_out,
        )
        out_cape[unsafe_offset=c] = res[0]
        out_cin[unsafe_offset=c] = res[1]
        out_lclp[unsafe_offset=c] = res[2]
        out_lfcp[unsafe_offset=c] = res[3]
        out_elp[unsafe_offset=c] = res[4]

    gp.unsafe_free()
    gt.unsafe_free()
    gtd.unsafe_free()
    tp.unsafe_free()
    bb.unsafe_free()
    xx.unsafe_free()
    ups.unsafe_free()
    downs.unsafe_free()
    aug_x.unsafe_free()
    aug_b.unsafe_free()
    rk_out.unsafe_free()
    return 0
