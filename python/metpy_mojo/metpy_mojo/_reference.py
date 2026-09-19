"""Vendored pure-Python reference: parcel CAPE/CIN for standard soundings.

Clean-room implementation of the textbook pseudoadiabatic parcel method:

- Saturation vapor pressure: Bolton (1980) eq. 10.
- LCL: exact solve of ``w_sat(T_lcl, p_lcl) = r0`` along the dry adiabat
  (Bolton's closed form brackets the root; bisection refines it).
- Moist ascent: the classic pseudoadiabatic lapse rate in pressure
  coordinates (Emanuel, "Atmospheric Convection", 1994; Wallace & Hobbs
  eq. 3.54 converted with the hydrostatic equation),

      dT/dp = (Rd * T + Lv * rs) / (p * (cpd + Lv^2 * rs / (Rv * T^2)))

  integrated with an adaptive Dormand-Prince RK45 (deterministic step
  controller), sampled exactly on the environmental grid pressures. This
  is the theta_e-conserving form: rs = eps*es/(p - es) is the saturation
  mixing ratio and Clausius-Clapeyron gives drs/dT = Lv*rs/(Rv*T^2).
- Buoyancy uses the exact virtual temperature ``T*(1 + r/eps)/(1 + r)``;
  CAPE/CIN are trapezoid integrals of ``Rd*(Tv_parcel - Tv_env)`` in
  log-pressure over the LCL-augmented environmental grid, split at every
  interpolated buoyancy zero crossing. The LFC/EL gating mirrors the
  MetPy oracle's ``cape_cin`` semantics, decoded black-box on the
  standard-sounding suite (see ``cape_cin_column`` for the full rules):
  LFC = the buoyancy - -> + crossing paired with the last + -> -
  crossing (EL); CAPE = positive-only area LFC -> EL (or profile top);
  CIN = signed area from the parcel start to the LFC, floored at 0;
  no - -> + crossing => CAPE = CIN = 0.
- Most-unstable parcel: argmax of Bolton (1980) eq. 38 equivalent
  potential temperature within a depth below the surface (default 300 hPa).

This module is the fallback backend on platforms without the Mojo kernel
and the documented-algorithm reference for the differential suite. It is
deliberately per-column scalar math so its operation order is obvious;
the Mojo kernel ports this same algorithm.
"""

from __future__ import annotations

import math

import numpy as np

# Physical constants. CPD/LV are the values whose combination minimizes the
# parcel-temperature drift against the metpy oracle (max 0.04 K over a deep
# moist adiabat; see the package README "Parity" section) — consistent with
# the constants MetPy documents (Cp_d = 1004 J/(kg K), Lv = 2.5e6 J/kg).
RD = 287.05  # dry-air gas constant, J/(kg K)
RV = 461.51  # water-vapor gas constant, J/(kg K)
CPD = 1004.0  # dry-air specific heat at constant pressure, J/(kg K)
LV = 2.5e6  # latent heat of vaporization, J/kg
EPS = RD / RV  # molecular-weight ratio, ~0.622
KAPPA = 2.0 / 7.0  # Poisson exponent for dry adiabats (the classic 2/7 —
# matches the MetPy oracle's below-LCL dry-lapse profile to 1e-7)
P_REF = 1000.0  # reference pressure for theta-e, hPa

# RK45 error-control tolerances (deterministic Dormand-Prince controller).
_RK45_RTOL = 1e-9
_RK45_ATOL = 1e-11

# Most-unstable parcel search depth below the surface, hPa.
MU_DEPTH_HPA = 300.0

__all__ = ["MU_DEPTH_HPA", "cape_cin_column", "cape_cin_columns"]


def saturation_vapor_pressure(t_k: float) -> float:
    """Saturation vapor pressure over liquid water (hPa), Bolton 1980 eq. 10."""
    return 6.112 * math.exp(17.67 * (t_k - 273.15) / (t_k - 29.65))


def saturation_mixing_ratio(t_k: float, p_hpa: float) -> float:
    """Saturation mixing ratio (kg/kg) at temperature/pressure."""
    es = saturation_vapor_pressure(t_k)
    return EPS * es / (p_hpa - es)


def mixing_ratio_from_dewpoint(td_k: float, p_hpa: float) -> float:
    """Mixing ratio (kg/kg) implied by dewpoint at the given pressure."""
    es = saturation_vapor_pressure(td_k)
    return EPS * es / (p_hpa - es)


def virtual_temperature(t_k: float, r: float) -> float:
    """Exact virtual temperature (K) for temperature t_k and mixing ratio r."""
    return t_k * (1.0 + r / EPS) / (1.0 + r)


def _bolton_lcl_temperature(t_k: float, td_k: float) -> float:
    """Closed-form LCL temperature estimate (K), Bolton 1980 eq. 15."""
    return 1.0 / (1.0 / (td_k - 56.0) + math.log(t_k / td_k) / 800.0) + 56.0


def dry_kappa(r: float) -> float:
    """Moist-air Poisson exponent used inside the LCL solve.

    kappa = (2/7) * (1 + 0.608*r)/(1 + 0.84*r): the dry 2/7 corrected for
    the water-vapor heat capacity (R = Rd*(1 + r/eps)/(1 + r),
    cp = cpd*(1 + 0.84*r)). Matches the MetPy oracle's LCL placement to
    ~4e-5 in kappa across the sounding suite (verified black-box); the
    below-LCL parcel profile itself uses the plain 2/7, as the oracle
    does.
    """
    return KAPPA * (1.0 + 0.608 * r) / (1.0 + 0.84 * r)


def lcl(t0_k: float, td0_k: float, p0_hpa: float) -> tuple[float, float]:
    """Exact LCL pressure (hPa) and temperature (K) for one parcel.

    Solves ``w_sat(T_lcl, p_lcl(T_lcl)) = r0`` with the dry adiabat
    ``p_lcl = p0 * (T_lcl/T0)^(1/kappa_moist)``. Along the adiabat the
    saturation mixing ratio falls monotonically as T_lcl drops, so the
    root is unique with T_lcl <= Td0 <= T0; Bolton's closed form brackets
    it, bounded expansion covers extreme soundings, and a fixed
    60-iteration bisection converges deterministically. A saturated
    parcel (Td0 >= T0) has its LCL at its own level.
    """
    if td0_k >= t0_k:
        return p0_hpa, t0_k
    r0 = mixing_ratio_from_dewpoint(td0_k, p0_hpa)
    kappa_m = dry_kappa(r0)

    def residual(t_l: float) -> float:
        p_l = p0_hpa * (t_l / t0_k) ** (1.0 / kappa_m)
        return saturation_mixing_ratio(t_l, p_l) - r0

    t_b = _bolton_lcl_temperature(t0_k, td0_k)
    lo, hi = t_b - 15.0, t_b + 15.0
    for _ in range(20):  # f(hi) must be > 0 (hi is at most T0)
        if residual(hi) > 0.0:
            break
        hi += 15.0
    for _ in range(20):  # f(lo) must be < 0
        if residual(lo) < 0.0:
            break
        lo -= 15.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if residual(mid) < 0.0:
            lo = mid
        else:
            hi = mid
    t_lcl = 0.5 * (lo + hi)
    return p0_hpa * (t_lcl / t0_k) ** (1.0 / kappa_m), t_lcl


def moist_lapse_rhs(t_k: float, p_hpa: float) -> float:
    """Pseudoadiabatic moist lapse rate dT/dp (K/hPa) at (t_k, p_hpa)."""
    rs = saturation_mixing_ratio(t_k, p_hpa)
    numerator = RD * t_k + LV * rs
    denominator = p_hpa * (CPD + LV * LV * rs / (RV * t_k * t_k))
    return numerator / denominator


# Dormand-Prince RK45 5th-order weights and the embedded error
# coefficients (b5 - b4), Dormand & Prince (1980).
_DP_B5 = (35.0 / 384.0, 500.0 / 1113.0, 125.0 / 192.0,
          -2187.0 / 6784.0, 11.0 / 84.0)
_DP_E = (71.0 / 57600.0, -71.0 / 16695.0, 71.0 / 1920.0,
         -17253.0 / 339200.0, 22.0 / 525.0, -1.0 / 40.0)


def _rk45_to_grid(t0: float, p0: float, p_targets: np.ndarray) -> np.ndarray:
    """Integrate the moist adiabat from (t0, p0) down to each target pressure.

    `p_targets` must be strictly decreasing and below p0. Adaptive steps
    are clamped to land exactly on each target pressure; the deterministic
    controller (fixed tolerances and safety factors) makes the sampled
    solution reproducible run-to-run.
    """
    out = np.empty(p_targets.shape[0], dtype=np.float64)
    t, p = t0, p0
    h = 0.25 * (p_targets[0] - p0)  # negative: integrate toward lower p
    for j in range(p_targets.shape[0]):
        target = p_targets[j]
        while p > target:  # the clamped final step lands exactly on target
            step = h
            if p + step <= target:  # clamp onto the grid boundary
                step = target - p
            k1 = moist_lapse_rhs(t, p)
            k2 = moist_lapse_rhs(t + step * 0.2 * k1, p + 0.2 * step)
            k3 = moist_lapse_rhs(
                t + step * (3.0 / 40.0 * k1 + 9.0 / 40.0 * k2), p + 0.3 * step)
            k4 = moist_lapse_rhs(
                t + step * (44.0 / 45.0 * k1 - 56.0 / 15.0 * k2 + 32.0 / 9.0 * k3),
                p + 0.8 * step)
            k5 = moist_lapse_rhs(
                t + step * (19372.0 / 6561.0 * k1 - 25360.0 / 2187.0 * k2
                            + 64448.0 / 6561.0 * k3 - 212.0 / 729.0 * k4),
                p + 8.0 / 9.0 * step)
            k6 = moist_lapse_rhs(
                t + step * (9017.0 / 3168.0 * k1 - 355.0 / 33.0 * k2
                            + 46732.0 / 5247.0 * k3 + 49.0 / 176.0 * k4
                            - 5103.0 / 18656.0 * k5),
                p + step)
            t5 = t + step * (_DP_B5[0] * k1 + _DP_B5[1] * k3 + _DP_B5[2] * k4
                             + _DP_B5[3] * k5 + _DP_B5[4] * k6)
            k7 = moist_lapse_rhs(t5, p + step)
            err = step * (_DP_E[0] * k1 + _DP_E[1] * k3 + _DP_E[2] * k4
                          + _DP_E[3] * k5 + _DP_E[4] * k6 + _DP_E[5] * k7)
            scale = _RK45_ATOL + _RK45_RTOL * max(abs(t), abs(t5))
            err_norm = abs(err) / scale
            if err_norm <= 1.0:
                t, p = t5, p + step
                factor = 10.0 if err_norm == 0.0 else min(
                    10.0, max(0.2, 0.9 * err_norm ** (-0.2)))
                h = step * factor
            else:
                h = step * max(0.2, 0.9 * err_norm ** (-0.2))
        out[j] = t
    return out


def equivalent_potential_temperature(t_k: float, td_k: float, p_hpa: float) -> float:
    """Equivalent potential temperature (K), Bolton 1980 eq. 38."""
    r_gkg = 1e3 * mixing_ratio_from_dewpoint(td_k, p_hpa)
    t_lcl = _bolton_lcl_temperature(t_k, td_k)
    return (t_k
            * (P_REF / p_hpa) ** (0.2854 * (1.0 - 0.28e-3 * r_gkg))
            * math.exp((3376.0 / t_lcl - 2.54) * 1e-3 * r_gkg * (1.0 + 0.81e-3 * r_gkg)))


def _parcel_start_index(p: np.ndarray, t: np.ndarray, td: np.ndarray,
                        which: int, mu_depth_hpa: float) -> int:
    """0 = surface parcel; 1 = most-unstable (max theta-e in the lowest depth)."""
    if which == 0:
        return 0
    theta_e = np.full(p.shape[0], -np.inf)
    for i in range(p.shape[0]):
        if p[i] >= p[0] - mu_depth_hpa:
            theta_e[i] = equivalent_potential_temperature(t[i], td[i], p[i])
    return int(np.argmax(theta_e))


def _interp_linear_p(x_hpa: float, xs_hpa: np.ndarray, ys: np.ndarray) -> float:
    """Linear interpolation in pressure (xs strictly decreasing).

    Used for the environmental values at the inserted LCL grid point — the
    MetPy oracle interpolates linearly in pressure there (verified black-box
    against its buoyancy at the LCL point), while everything else
    (crossings, quadrature) is in log-pressure.
    """
    return float(np.interp(x_hpa, xs_hpa[::-1], ys[::-1]))


def cape_cin_column(p: np.ndarray, t: np.ndarray, td: np.ndarray,
                    which: int = 0, mu_depth_hpa: float = MU_DEPTH_HPA
                    ) -> tuple[float, float, float, float, float]:
    """CAPE/CIN (J/kg) plus LCL/LFC/EL pressures (hPa) for one column.

    `p` strictly decreasing (hPa); `t`, `td` in K. Returns
    ``(cape, cin, p_lcl, p_lfc, p_el)``; an LFC/EL that does not exist is
    NaN (mirroring the MetPy oracle: no LFC => cape = cin = 0; no EL
    crossing => EL is NaN and CAPE integrates to the profile top).
    """
    n = p.shape[0]
    i0 = _parcel_start_index(p, t, td, which, mu_depth_hpa)
    t0, td0, p0 = float(t[i0]), float(td[i0]), float(p[i0])
    r0 = mixing_ratio_from_dewpoint(td0, p0)
    p_lcl, t_lcl = lcl(t0, td0, p0)

    # Integration grid: environmental levels from the start level up, with
    # the LCL inserted when it falls between levels (the same grid shape as
    # the documented MetPy parcel_profile_with_lcl behavior). If the LCL is
    # above the profile top there is no saturated ascent on the grid.
    env_p = p[i0:n]
    env_t = t[i0:n]
    env_td = td[i0:n]
    if p_lcl >= env_p[0] - 1e-12:
        p_lcl, t_lcl = float(env_p[0]), float(env_t[0])
        grid_p, grid_t, grid_td = env_p, env_t, env_td
    elif p_lcl <= env_p[-1]:
        grid_p, grid_t, grid_td = env_p, env_t, env_td
    else:
        t_at_lcl = _interp_linear_p(p_lcl, env_p, env_t)
        td_at_lcl = _interp_linear_p(p_lcl, env_p, env_td)
        grid_p = np.concatenate([env_p, [p_lcl]])
        order = np.argsort(grid_p)[::-1]
        grid_p = grid_p[order]
        merged_t = np.concatenate([env_t, [t_at_lcl]])
        merged_td = np.concatenate([env_td, [td_at_lcl]])
        grid_t, grid_td = merged_t[order], merged_td[order]
    m = grid_p.shape[0]

    # Parcel temperature on the grid. Below the LCL: dry adiabat (kappa =
    # 2/7). At the inserted LCL point: the LCL temperature from the
    # moist-kappa solve (matching the oracle's profile). Above the LCL:
    # RK45 moist adiabat — started from the 2/7 dry-adiabat value AT the
    # LCL pressure, which is where the MetPy oracle's own profile
    # initializes its saturated ascent (its reported T_lcl sits ~0.03 K
    # above the dry adiabat; verified black-box on fine grids).
    above = grid_p < p_lcl
    t_par = np.empty(m)
    t_par[~above] = t0 * (grid_p[~above] / p0) ** KAPPA
    if p_lcl < env_p[0] - 1e-12 and p_lcl > env_p[-1]:
        idx_lcl = int(np.argmin(np.abs(grid_p - p_lcl)))
        t_par[idx_lcl] = t_lcl
    if np.any(above):
        t_dry_at_lcl = t0 * (p_lcl / p0) ** KAPPA
        t_par[above] = _rk45_to_grid(t_dry_at_lcl, p_lcl, grid_p[above])

    # Buoyancy profile with exact virtual temperatures (the integrand, and
    # the crossing metric). The gating below mirrors the MetPy oracle's
    # cape_cin semantics, decoded black-box on the standard-sounding suite
    # (max rel CAPE error 0.7% using the oracle's own profiles):
    #   * LFC = the buoyancy - -> + crossing immediately before the LAST
    #     + -> - crossing (the widest buoyant window adjacent to the EL);
    #     if buoyancy never turns negative again, the first - -> + crossing.
    #   * EL = that last + -> - crossing (NaN if none; CAPE then integrates
    #     to the profile top).
    #   * CAPE = positive-only trapezoid of buoyancy in ln p from LFC to EL,
    #     split at every interior zero crossing.
    #   * CIN = signed (not clipped) buoyancy integral from the parcel start
    #     to the LFC, floored at 0. No - -> + crossing => cape = cin = 0.
    b = np.empty(m)
    for i in range(m):
        r_env = mixing_ratio_from_dewpoint(float(grid_td[i]), float(grid_p[i]))
        r_par = min(saturation_mixing_ratio(float(t_par[i]), float(grid_p[i])), r0)
        b[i] = RD * (virtual_temperature(float(t_par[i]), r_par)
                     - virtual_temperature(float(grid_t[i]), r_env))
    x = np.log(grid_p)

    up_crossings: list[float] = []  # x positions, - -> +
    down_crossings: list[float] = []  # + -> -
    for i in range(1, m):
        if b[i - 1] <= 0.0 < b[i]:
            frac = b[i - 1] / (b[i - 1] - b[i])
            up_crossings.append(float(x[i - 1] + frac * (x[i] - x[i - 1])))
        elif b[i - 1] >= 0.0 > b[i]:
            frac = b[i - 1] / (b[i - 1] - b[i])
            down_crossings.append(float(x[i - 1] + frac * (x[i] - x[i - 1])))
    if not up_crossings:
        return 0.0, 0.0, p_lcl, math.nan, math.nan

    x_lfc = up_crossings[0]
    x_el = math.nan
    if down_crossings:
        x_down_last = down_crossings[-1]
        before = [u for u in up_crossings if u > x_down_last]  # below the EL
        if before:
            x_lfc = before[-1]  # the up-crossing paired with the EL
            x_el = x_down_last
    p_lfc = math.exp(x_lfc)
    p_el = math.exp(x_el) if not math.isnan(x_el) else math.nan
    x_top = x_el if not math.isnan(x_el) else float(x[-1])

    # Quadrature grid: the original grid augmented with every strictly
    # interior buoyancy zero crossing, so the trapezoid splits exactly at
    # sign changes (profile order: surface -> top).
    xs_aug: list[float] = []
    b_aug: list[float] = []
    for i in range(m):
        xs_aug.append(float(x[i]))
        b_aug.append(float(b[i]))
        if i + 1 < m and ((b[i] > 0.0) != (b[i + 1] > 0.0)) and b[i] != b[i + 1]:
            frac = b[i] / (b[i] - b[i + 1])
            if 0.0 < frac < 1.0:
                xs_aug.append(float(x[i] + frac * (x[i + 1] - x[i])))
                b_aug.append(0.0)

    # Trapezoid in ln p (surface -> top order, so x decreases; the leading
    # sign turns CAPE positive and leaves CIN negative, matching the oracle).
    def _trapz(x_lo: float, x_hi: float, mode: str) -> float:
        seg = [(xx, bb) for xx, bb in zip(xs_aug, b_aug) if x_hi <= xx <= x_lo]
        xs = np.array([x_lo] + [s[0] for s in seg] + [x_hi])
        bb = np.array([np.interp(x_lo, xs_aug[::-1], b_aug[::-1])]
                      + [s[1] for s in seg]
                      + [np.interp(x_hi, xs_aug[::-1], b_aug[::-1])])
        if mode == "positive":
            bb = np.clip(bb, 0.0, None)
        return float(np.trapezoid(bb, xs))

    cape = -_trapz(x_lfc, x_top, "positive")
    cin = min(0.0, -_trapz(float(x[0]), x_lfc, "signed"))
    return max(cape, 0.0), cin, p_lcl, p_lfc, p_el


def cape_cin_columns(p: np.ndarray, t: np.ndarray, td: np.ndarray,
                     which: int = 0, mu_depth_hpa: float = MU_DEPTH_HPA
                     ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Batched driver: (ncols, nlev) arrays -> five (ncols,) result arrays.

    Column c occupies row c of each input; pressure must be strictly
    decreasing along each row.
    """
    ncols = p.shape[0]
    cape = np.empty(ncols)
    cin = np.empty(ncols)
    lclp = np.empty(ncols)
    lfcp = np.empty(ncols)
    elp = np.empty(ncols)
    for c in range(ncols):
        cape[c], cin[c], lclp[c], lfcp[c], elp[c] = cape_cin_column(
            p[c], t[c], td[c], which, mu_depth_hpa)
    return cape, cin, lclp, lfcp, elp
