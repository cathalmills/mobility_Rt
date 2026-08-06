#!/usr/bin/env python3
"""
Mobility-Informed Renewal Equation Framework
=============================================

General-purpose implementation of the mobility-informed renewal equation
framework (Mills et al. 2026) for computing reproduction numbers, generation
time distributions, and epidemic transience measures on human mobility networks.

Quick start
-----------
    import numpy as np
    from mobility_rt.framework import MobilityRtFramework, discretise_gamma

    p = discretise_gamma(mean=5.5, sd=1.8, max_days=25)
    model = MobilityRtFramework(
        f_jk        = f_jk,          # (T, N, N) or (N, N) row-stochastic matrix
        populations = populations,    # (N,) resident populations
        infectiousness_profile = p,   # (max_days,) normalised p(a_E)
        contact_rate_home = 13.0,     # lambda_W contacts/day at home location
        contact_rate_away = 3.9,      # lambda_B contacts/day away from home
        R0_target   = 1.5,           # calibration target
        initial_infections = seed,    # (N,) seed
        T           = 200,           # days
    )
    results = model.simulate()
    model.plot_overview(results, save_path="overview.png")

Reference
---------
Mills, C. (2026). Reproduction numbers for epidemics on human mobility networks.
Department of Statistics and Pandemic Sciences Institute, University of Oxford.
"""

from __future__ import annotations

import warnings
import numpy as np
from scipy.optimize import brentq
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

warnings.filterwarnings("ignore", category=RuntimeWarning)

# Okabe-Ito colorblind-safe palette (Wong 2011 Nature Methods)
OKABE_ITO = ['#E69F00', '#56B4E9', '#009E73', '#F0E442',
             '#0072B2', '#D55E00', '#CC79A7', '#000000']

__version__ = "1.1.0"


# ═══════════════════════════════════════════════════════════════════════════════
# 1.  INPUT VALIDATION
# ═══════════════════════════════════════════════════════════════════════════════

def validate_inputs(
    f_jk: np.ndarray,
    populations: np.ndarray,
    infectiousness_profile: np.ndarray,
    contact_rate_home: float,
    contact_rate_away: float,
    initial_infections: np.ndarray,
    T: int,
    *,
    R0_target: float | None = None,
    beta: float | None = None,
    tol: float = 1e-8,
    verbose: bool = True,
) -> dict:
    """
    Validate all inputs and return processed versions.

    Parametrisation modes
    ---------------------
    Two mutually exclusive ways to specify transmission intensity:

    **Mode A — β (recommended for outbreak-start analysis):**
        Set ``beta``: per-contact transmission probability (0 < β ≤ 1).
        Combined with ``contact_rate_home`` and ``contact_rate_away``, this
        uniquely determines all reproduction numbers without any calibration step.
        R₀ is then an *output*, derived from the network structure and β.
        Suitable when you know (or want to scan) the transmission probability
        but do not know R₀ in advance.

    **Mode B — R₀ target (for scenario analysis with a known R₀):**
        Set ``R0_target``: the desired basic reproduction number.
        Contact rates are scaled internally so that R(t=0) = R₀_target.
        β is then an internal, derived quantity.

    Exactly one of ``beta`` or ``R0_target`` must be provided.

    Parameters
    ----------
    f_jk : ndarray, shape (T, N, N) or (N, N)
        Row-stochastic mobility matrix.  f_jk[t,j,k] = P(resident j in location k
        at time t).  If shape (N, N) it is treated as static and broadcast to
        (T, N, N).
    populations : ndarray, shape (N,)
        Resident population N_j in each location (strictly positive).
    infectiousness_profile : ndarray, shape (max_days,)
        Discretised infectiousness p(a_E) summing to 1 (normalised internally
        if not already).
    contact_rate_home : float
        lambda_W — per-person contacts/day in the home location (e.g. 13.0 from
        POLYMOD Mossong 2008).
    contact_rate_away : float
        lambda_B — per-person contacts/day away from home (e.g. 0.30 × lambda_W).
    initial_infections : ndarray, shape (N,)
        Initial infected seed per location (non-negative, sums > 0).
    T : int
        Simulation duration (days, >= 2).
    R0_target : float, optional
        Target R₀ for calibration (Mode B). Mutually exclusive with ``beta``.
    beta : float, optional
        Per-contact transmission probability (Mode A). Mutually exclusive with
        ``R0_target``.  Must satisfy 0 < β ≤ 1.
    tol : float
        Tolerance for row-stochasticity check.
    verbose : bool
        Print summary after validation.

    Returns
    -------
    dict with processed 'f_jk', 'populations', 'infectiousness_profile',
    'initial_infections', 'N', 'T', 'max_days'.

    Raises
    ------
    ValueError with a clear message listing every detected problem.
    """
    errors: list[str] = []
    warn_msgs: list[str] = []

    # populations
    populations = np.asarray(populations, dtype=float)
    if populations.ndim != 1:
        errors.append(f"populations must be 1-D, got shape {populations.shape}")
    elif np.any(populations <= 0):
        errors.append("All populations must be > 0")
    N = len(populations)

    # T
    if not isinstance(T, (int, np.integer)) or T < 2:
        errors.append(f"T must be an integer >= 2, got {T!r}")
    T = int(T)

    # f_jk
    f_jk = np.asarray(f_jk, dtype=float)
    if f_jk.ndim == 2:
        if f_jk.shape != (N, N):
            errors.append(f"Static f_jk must be ({N},{N}), got {f_jk.shape}")
        else:
            f_jk = np.broadcast_to(f_jk[np.newaxis], (T, N, N)).copy()
            warn_msgs.append("f_jk is 2-D (static) — broadcast to (T,N,N)")
    elif f_jk.ndim == 3:
        if f_jk.shape[1] != N or f_jk.shape[2] != N:
            errors.append(f"f_jk shape[1:] must be ({N},{N}), got {f_jk.shape}")
        if f_jk.shape[0] != T:
            errors.append(f"f_jk.shape[0]={f_jk.shape[0]} != T={T}")
    else:
        errors.append(f"f_jk must be 2-D or 3-D, got {f_jk.ndim}-D")

    if not errors:
        if np.any(f_jk < -tol):
            errors.append("f_jk contains negative entries")
        row_sums = f_jk.sum(axis=-1)
        max_dev = float(np.abs(row_sums - 1.0).max())
        if max_dev > tol:
            errors.append(
                f"f_jk rows must sum to 1 (max deviation = {max_dev:.2e}); "
                "normalise each row before passing."
            )

    # infectiousness_profile
    infectiousness_profile = np.asarray(infectiousness_profile, dtype=float)
    if infectiousness_profile.ndim != 1 or len(infectiousness_profile) < 2:
        errors.append("infectiousness_profile must be 1-D with length >= 2")
    elif np.any(infectiousness_profile < 0):
        errors.append("infectiousness_profile must be non-negative everywhere")
    else:
        tot = infectiousness_profile.sum()
        if tot < 1e-15:
            errors.append("infectiousness_profile sums to zero")
        elif not np.isclose(tot, 1.0, atol=1e-4):
            warn_msgs.append(f"infectiousness_profile sums to {tot:.6f} — normalising")
            infectiousness_profile = infectiousness_profile / tot
    max_days = len(infectiousness_profile)

    # contact rates
    if not (isinstance(contact_rate_home, (int, float)) and contact_rate_home > 0):
        errors.append(f"contact_rate_home must be > 0, got {contact_rate_home!r}")
    if not (isinstance(contact_rate_away, (int, float)) and contact_rate_away >= 0):
        errors.append(f"contact_rate_away must be >= 0, got {contact_rate_away!r}")
    if not errors and contact_rate_away > contact_rate_home:
        warn_msgs.append(
            f"contact_rate_away ({contact_rate_away}) > contact_rate_home "
            f"({contact_rate_home}) — unusual but allowed"
        )

    # Transmission parametrisation: exactly one of beta or R0_target
    if beta is not None and R0_target is not None:
        errors.append("Provide exactly one of 'beta' or 'R0_target', not both.")
    elif beta is None and R0_target is None:
        errors.append("Must provide either 'beta' (per-contact transmission probability) "
                      "or 'R0_target' (calibration target).")
    elif beta is not None:
        if not (isinstance(beta, (int, float)) and 0 < beta <= 1):
            errors.append(f"beta must satisfy 0 < beta <= 1, got {beta!r}")
        elif beta > 0.3:
            warn_msgs.append(f"beta = {beta:.3f} is high (> 0.3); typical range 0.01–0.15")
    elif R0_target is not None:
        if not (isinstance(R0_target, (int, float)) and R0_target > 0):
            errors.append(f"R0_target must be > 0, got {R0_target!r}")
        elif R0_target > 15:
            warn_msgs.append(f"R0_target = {R0_target} is very high (> 15)")

    # initial_infections
    initial_infections = np.asarray(initial_infections, dtype=float)
    if not errors:
        if initial_infections.shape != (N,):
            errors.append(
                f"initial_infections must have shape ({N},), got {initial_infections.shape}"
            )
        elif np.any(initial_infections < 0):
            errors.append("initial_infections must be non-negative")
        elif initial_infections.sum() == 0:
            errors.append("initial_infections must have at least one positive entry")
        else:
            for j in range(N):
                if initial_infections[j] > populations[j]:
                    errors.append(
                        f"initial_infections[{j}]={initial_infections[j]:.0f} "
                        f"> populations[{j}]={populations[j]:.0f}"
                    )

    if errors:
        raise ValueError(
            "Input validation failed:\n" + "\n".join(f"  • {e}" for e in errors)
        )

    if verbose:
        print("✓ Inputs validated")
        print(f"  Locations N={N}, Duration T={T}, max_days={max_days}")
        print(f"  Total population: {populations.sum():,.0f}")
        print(f"  Seed infections:  {initial_infections.sum():.0f}")
        print(f"  λ_W={contact_rate_home}, λ_B={contact_rate_away} "
              f"(ratio {contact_rate_away/contact_rate_home:.2f})")
        if beta is not None:
            print(f"  Mode: β = {beta}  (R₀ computed as output)")
        else:
            print(f"  Mode: R₀ target = {R0_target}  (β calibrated internally)")
        for w in warn_msgs:
            print(f"  ⚠ {w}")

    return {
        "f_jk": f_jk,
        "populations": populations,
        "infectiousness_profile": infectiousness_profile,
        "initial_infections": initial_infections,
        "N": N, "T": T, "max_days": max_days,
        "beta": beta, "R0_target": R0_target,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# 2.  INFECTIOUSNESS PROFILE UTILITIES
# ═══════════════════════════════════════════════════════════════════════════════

# discretise_gamma is the single canonical implementation, imported from the core
# package so the framework and core can never drift (see mobility_rt.distributions).
from mobility_rt.distributions import discretise_gamma


# ═══════════════════════════════════════════════════════════════════════════════
# 2b.  β-FIRST UTILITIES: R₀ COMPUTATION, ESTIMATION, AND PARAMETER SWEEP
# ═══════════════════════════════════════════════════════════════════════════════

def compute_R0_from_beta(
    f0: np.ndarray,
    populations: np.ndarray,
    infectiousness_profile: np.ndarray,
    beta: float,
    contact_rate_home: float,
    contact_rate_away: float,
) -> float:
    """
    Compute R₀ given per-contact transmission probability β.

    R₀ = ρ(R(t=0)) evaluated at fully susceptible S = populations,
    with lw = β × λ_W, lb = β × λ_B.

    Parameters
    ----------
    f0 : (N, N) mobility matrix at t=0.
    populations : (N,) resident populations.
    infectiousness_profile : (max_days,) normalised p(a_E).
    beta : per-contact transmission probability β ∈ (0, 1].
    contact_rate_home : λ_W contacts/day at home.
    contact_rate_away : λ_B contacts/day away from home.

    Returns
    -------
    float : R₀ spectral radius.
    """
    lw = beta * contact_rate_home
    lb = beta * contact_rate_away
    K_base, _, _, _ = _kernel_base(f0, populations, lw, lb)
    R0_mat = infectiousness_profile[1:].sum() * populations[np.newaxis, :] * K_base
    return R_network(R0_mat)


def estimate_beta_from_growth_rate(
    r_observed: float,
    infectiousness_profile: np.ndarray,
    f0: np.ndarray,
    populations: np.ndarray,
    contact_rate_home: float,
    contact_rate_away: float,
    *,
    beta_bounds: tuple = (1e-4, 0.5),
) -> dict:
    """
    Back-calculate β from an observed early epidemic growth rate r (1/day).

    Uses the Euler-Lotka equation  1 = R₀ × Σ_a p(a) exp(−r a)
    to infer R₀, then inverts R₀(β) to find β.

    This answers the key outbreak-start question: *given an observed doubling
    time and known contact rates + GT, what is the implied β?*

    Parameters
    ----------
    r_observed : float
        Exponential growth rate (1/day).  r = log(2)/doubling_time.
    infectiousness_profile : (max_days,) normalised p(a_E).
    f0 : (N, N) mobility matrix at t=0 (used for R₀(β) mapping).
    populations : (N,)
    contact_rate_home : λ_W
    contact_rate_away : λ_B
    beta_bounds : (lo, hi) search interval for β.

    Returns
    -------
    dict with keys 'beta', 'R0', 'R0_check', 'r_check', 'doubling_time'.
    """
    p    = infectiousness_profile
    # Renewal Euler-Lotka uses ages a ≥ 1 (exclude same-day a=0), matching the
    # simulator and compute_R0_from_beta.
    p_r  = p.copy(); p_r[0] = 0.0
    p_r  = p_r / p_r.sum() if p_r.sum() > 1e-15 else p_r
    days = np.arange(len(p_r))

    # Step 1: solve Euler-Lotka for R₀ given r
    def _el(R0_val):
        return R0_val * float(np.sum(p_r * np.exp(-r_observed * days))) - 1.0

    if _el(0.05) * _el(25.0) > 0:
        raise ValueError(
            f"Cannot solve Euler-Lotka for r={r_observed:.4f}. "
            "Verify r is realistic for the given GT profile."
        )
    R0_req = float(brentq(_el, 0.05, 25.0, xtol=1e-10))

    # Step 2: find β such that R₀(β) = R0_req
    def _rb(b):
        return compute_R0_from_beta(f0, populations, p, b,
                                     contact_rate_home, contact_rate_away) - R0_req

    b_lo, b_hi = beta_bounds
    if _rb(b_lo) * _rb(b_hi) > 0:
        raise ValueError(
            f"β not in {beta_bounds} achieves R₀={R0_req:.3f}. "
            "Try wider beta_bounds or check contact rates."
        )
    beta_est = float(brentq(_rb, b_lo, b_hi, xtol=1e-10))
    R0_check = compute_R0_from_beta(f0, populations, p, beta_est,
                                     contact_rate_home, contact_rate_away)
    r_check  = euler_lotka_r(R0_check, p)

    return {
        "beta":          beta_est,
        "R0":            R0_req,
        "R0_check":      R0_check,
        "r_check":       r_check,
        "doubling_time": np.log(2) / r_observed if r_observed > 0 else np.nan,
    }


def scan_beta_to_R0(
    beta_values: np.ndarray,
    f0: np.ndarray,
    populations: np.ndarray,
    infectiousness_profile: np.ndarray,
    contact_rate_home: float,
    contact_rate_away: float,
) -> np.ndarray:
    """
    Map an array of β values to their corresponding R₀ values (no simulation).

    Returns (M,) array of R₀ values matching ``beta_values``.
    """
    return np.array([
        compute_R0_from_beta(f0, populations, infectiousness_profile, b,
                              contact_rate_home, contact_rate_away)
        for b in beta_values
    ])


def parameter_sweep(
    f_jk_series: np.ndarray,
    populations: np.ndarray,
    initial_infections: np.ndarray,
    T: int,
    param_grid: dict,
    *,
    contact_rate_home: float = 13.0,
    contact_rate_away_ratio: float = 0.30,
    infectiousness_profile: np.ndarray | None = None,
    gt_mean_default: float = 5.5,
    gt_sd_default: float = 1.8,
    max_days: int = 25,
    birth_rate: float = 3e-5,
    death_rate: float = 3e-5,
    verbose: bool = False,
) -> dict:
    """
    Full-factorial parameter sweep — run the epidemic model across a grid.

    Sweep axes (keys in ``param_grid``, each a 1-D array of values):
      'beta'           — per-contact transmission probability
      'lb_lw_ratio'    — λ_B/λ_W ratio (scales away contact rate)
      'gt_mean'        — mean generation time (days); rebuilds p(a_E)
      'gt_sd'          — SD of generation time (days); rebuilds p(a_E)
      'commuting_scale'— multiplier on off-diagonal f entries (0–1)

    Any axes not specified keep their defaults.

    Returns
    -------
    dict with keys:
      'axes'            — dict of sweep axes {name: values}
      'R0_grid'         — computed R₀ for each combo, shape = grid shape
      'attack_rate_grid'— final attack rate %, same shape
      'peak_day_grid'   — epidemic peak day, same shape
      'R_network_grid'  — R(t) trajectory, shape grid_shape + (T,)
      'reactivity_grid' — σ(t) trajectory, same shape
      'param_combinations' — list of dicts for each combination
    """
    from itertools import product as _product

    if infectiousness_profile is None:
        infectiousness_profile = discretise_gamma(gt_mean_default, gt_sd_default, max_days)

    keys   = list(param_grid.keys())
    vals   = [np.atleast_1d(param_grid[k]) for k in keys]
    combos = list(_product(*vals))
    nc     = len(combos)
    grid_shape = tuple(len(v) for v in vals)
    N = len(populations)
    f0 = f_jk_series[0]

    R0_arr   = np.zeros(nc)
    att_arr  = np.zeros(nc)
    peak_arr = np.zeros(nc, dtype=int)
    Rnet_arr = np.zeros((nc, T))
    sig_arr  = np.zeros((nc, T))
    combo_dicts = []

    for i, combo in enumerate(combos):
        p = dict(zip(keys, combo))
        combo_dicts.append(p)

        beta_i  = float(p.get("beta", 0.035))
        ratio_i = float(p.get("lb_lw_ratio", contact_rate_away_ratio))
        lw_i    = beta_i * contact_rate_home
        lb_i    = beta_i * contact_rate_home * ratio_i

        # Optionally rebuild GT profile
        if "gt_mean" in p or "gt_sd" in p:
            gmean = float(p.get("gt_mean", gt_mean_default))
            gsd   = float(p.get("gt_sd",   gt_sd_default))
            p_i   = discretise_gamma(gmean, gsd, max_days)
        else:
            p_i = infectiousness_profile

        # Optionally scale commuting
        if "commuting_scale" in p:
            f_i  = _scale_commuting(f_jk_series, float(p["commuting_scale"]))
            f0_i = f_i[0]
        else:
            f_i, f0_i = f_jk_series, f0

        R0_arr[i] = compute_R0_from_beta(f0_i, populations, p_i, beta_i,
                                          contact_rate_home,
                                          contact_rate_home * ratio_i)

        res = simulate_epidemic(
            f_i, populations, p_i, lw_i, lb_i, None,
            initial_infections, T,
            birth_rate=birth_rate, death_rate=death_rate,
            verbose=False, _beta_mode=True,
        )
        inc = res["incidence"]
        att_arr[i]   = inc.sum() / populations.sum() * 100
        peak_arr[i]  = int(inc.sum(axis=1).argmax())
        Rnet_arr[i]  = res["R_network"]
        sig_arr[i]   = res["reactivity"]

        if verbose:
            print(f"  [{i+1:4d}/{nc}] β={beta_i:.4f} → R₀={R0_arr[i]:.3f}, "
                  f"AR={att_arr[i]:.1f}%, peak={peak_arr[i]}")

    return {
        "axes":               {k: np.asarray(v) for k, v in zip(keys, vals)},
        "R0_grid":            R0_arr.reshape(grid_shape),
        "attack_rate_grid":   att_arr.reshape(grid_shape),
        "peak_day_grid":      peak_arr.reshape(grid_shape),
        "R_network_grid":     Rnet_arr.reshape(grid_shape + (T,)),
        "reactivity_grid":    sig_arr.reshape(grid_shape + (T,)),
        "param_combinations": combo_dicts,
    }


def _scale_commuting(f_series: np.ndarray, scale: float) -> np.ndarray:
    """Scale off-diagonal entries of f_series by ``scale`` and re-normalise."""
    f = f_series.copy()
    T_f, N, _ = f.shape
    for t in range(T_f):
        for j in range(N):
            d   = f[t, j, j]
            off = (f[t, j] - np.eye(N)[j] * d) * scale
            f[t, j]    = off
            f[t, j, j] = max(0.0, 1.0 - off.sum())
            rs = f[t, j].sum()
            if rs > 1e-15:
                f[t, j] /= rs
    return f


# ═══════════════════════════════════════════════════════════════════════════════
# 3.  KERNEL AND REPRODUCTION NUMBERS  (Eqs 7, 12, 14, 15, 44, 49)
# ═══════════════════════════════════════════════════════════════════════════════

def _kernel_base(
    f_t: np.ndarray,
    populations: np.ndarray,
    lw: float,
    lb: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute K_base[k,j] for one time slice (vectorised O(N²)).

    K_base[k,j] = lambda_W * f^{jk}(t) * f^{kk}(t) / N^k_eff(t)    [within]
                + lambda_B * sum_{l≠k} f^{jl}(t)*f^{kl}(t)/N^l_eff  [between]

    N^l_eff = sum_q f^{ql}(t) * N_q   (effective population at meeting loc l)

    Returns K_base (N,N), K_within (N,N), K_between (N,N), N_eff (N,).
    Rows = infectors k, columns = infectees j.
    """
    N_eff = f_t.T @ populations                          # (N,)
    inv_N = np.where(N_eff > 0, 1.0 / N_eff, 0.0)
    A = f_t * inv_N[np.newaxis, :]                       # A[j,l] = f^{jl}/N^l_eff
    D = A @ f_t.T                                         # D[j,k] = Σ_l A[j,l]*f^{kl}
    within = f_t.T * (f_t.diagonal() * inv_N)[:, np.newaxis]  # [k,j]
    K_within  = lw * within
    K_between = lb * np.maximum(D.T - within, 0.0)
    return K_within + K_between, K_within, K_between, N_eff


def compute_R_matrix(
    f_t: np.ndarray,
    S: np.ndarray,
    populations: np.ndarray,
    infectiousness_profile: np.ndarray,
    lw: float,
    lb: float,
) -> np.ndarray:
    """
    R^{kj}(t) = S_j(t) * K_base[k,j] * integral p(a_E) da_E.   (Eq 14)

    Since p is normalised, integral = 1 and R^{kj} = S_j * K_base[k,j].
    Rows = infectors k, columns = infectees j.
    """
    K_base, _, _, _ = _kernel_base(f_t, populations, lw, lb)
    # Integrate over infection ages a ≥ 1 (a=0 same-day transmission is excluded, as in
    # the forward simulator) — keeps the NGM consistent with the realised dynamics.
    return infectiousness_profile[1:].sum() * S[np.newaxis, :] * K_base


def R_outward(R_mat: np.ndarray) -> np.ndarray:
    """R^k_out(t) = sum_j R^{kj}(t).  (Eq 15)  Shape (N,)."""
    return R_mat.sum(axis=1)


def R_inward(R_mat: np.ndarray) -> np.ndarray:
    """R^j_in(t) = sum_k R^{kj}(t).  (Eq 45)  Shape (N,)."""
    return R_mat.sum(axis=0)


def R_network(R_mat: np.ndarray) -> float:
    """R(t) = rho(R(t)) = spectral radius.  (Eq 23)"""
    return float(np.max(np.abs(np.linalg.eigvals(R_mat))))


def E_risk_averse(R_mat: np.ndarray) -> float:
    """
    E(t) = sum_j [R^j_out]^2 / sum_k R^k_out.  (Eq 42)

    Complementary spatial risk indicator.  E(t) > 1 and R(t) < 1 signals
    potential localised resurgence in high-R_out but low-centrality locations.
    """
    Ro = R_outward(R_mat)
    d = Ro.sum()
    return float(np.sum(Ro**2) / d) if d > 1e-15 else 0.0


def R_meeting(
    f_t: np.ndarray,
    S: np.ndarray,
    populations: np.ndarray,
    infectiousness_profile: np.ndarray,
    lw: float,
    lb: float,
) -> np.ndarray:
    """
    R^l_meeting(t) for each venue location l.  (Eq 49)

    R^l_meeting measures the mixing capacity of location l as a transmission
    arena regardless of where residents come from.
    """
    N = f_t.shape[0]
    _, _, _, N_eff = _kernel_base(f_t, populations, lw, lb)
    inv_N = np.where(N_eff > 0, 1.0 / N_eff, 0.0)
    S_eff = f_t.T @ S
    sp = infectiousness_profile[1:].sum()   # a ≥ 1
    out = np.zeros(N)
    for l in range(N):
        kap = lw * f_t[l, l] + lb * max(f_t[:, l].sum() - f_t[l, l], 0.0)
        out[l] = sp * S_eff[l] * kap * inv_N[l]
    return out


# ═══════════════════════════════════════════════════════════════════════════════
# 4.  SPECTRAL ANALYSIS  (Section 3.1.3-3.1.5, Eqs 23-40)
# ═══════════════════════════════════════════════════════════════════════════════

def _perron_real(evec_col: np.ndarray) -> np.ndarray:
    """Real dominant (Perron) eigenvector from a complex eigenvector column.

    Rotates out the arbitrary global phase using the largest-magnitude entry, takes
    the real part, and fixes the sign to non-negative.  Exact for the primitive
    non-negative NGMs here (matches np.abs for a strictly-positive Perron vector) but,
    unlike np.abs, does not silently discard sign on degenerate/reducible matrices.
    """
    k = int(np.argmax(np.abs(evec_col)))
    phase = evec_col[k] / (np.abs(evec_col[k]) + 1e-300)
    v = (evec_col / phase).real.astype(float)
    if v[k] < 0:
        v = -v
    return v


def spectral_analysis(R_mat: np.ndarray) -> dict:
    """
    Full spectral decomposition of the NGM R(t).

    Returns
    -------
    dict with keys:
        rho             — spectral radius R(t) (Eq 23)
        lambda2         — |second eigenvalue|
        mixing_ratio    — s(t) = |λ₂|/ρ  (Eq 36)
        spectral_gap    — ζ(t) = ρ - |λ₂|
        damping_ratio   — d(t) = ρ/|λ₂|
        sigma           — reactivity σ(t) = ||R||₂  (Eq 37)
        condition_number— κ(R(t))  (Eq 38)
        right_eigvec    — v(t): reproductive value vector  (Eq 26), sum=1
        left_eigvec     — v*(t): stable spatial distribution (Eq 25), sum=1
        eigenvalues     — all eigenvalues, sorted by |·|
        transient_possible — True if σ > 1 and ρ < 1
    """
    ev_r, evec_r = np.linalg.eig(R_mat)
    ev_l, evec_l = np.linalg.eig(R_mat.T)

    ir = np.argsort(np.abs(ev_r))[::-1]
    il = np.argsort(np.abs(ev_l))[::-1]

    rho     = float(np.abs(ev_r[ir[0]]))
    lam2    = float(np.abs(ev_r[ir[1]])) if len(ir) > 1 else 0.0
    sigma   = float(np.linalg.svd(R_mat, compute_uv=False)[0])

    # Reproductive value v: right eigvec of R_mat  (manuscript's v)
    v = _perron_real(evec_r[:, ir[0]])
    v /= (v.sum() + 1e-300)

    # Stable distribution v*: right eigvec of R_mat^T  (manuscript's v*)
    vs = _perron_real(evec_l[:, il[0]])
    vs /= (vs.sum() + 1e-300)

    # Eigenvalue condition number κ = ||v_l||₂ ||v_r||₂ / |v_l·v_r|
    wr = evec_r[:, ir[0]]
    vl = evec_l[:, il[0]]
    denom = abs(float(vl @ wr))
    kappa = float(np.linalg.norm(vl) * np.linalg.norm(wr)) / (denom + 1e-300)

    return {
        "rho":              rho,
        "lambda2":          lam2,
        "mixing_ratio":     lam2 / rho if rho > 1e-15 else 0.0,
        "spectral_gap":     rho - lam2,
        "damping_ratio":    rho / lam2 if lam2 > 1e-15 else np.inf,
        "sigma":            sigma,
        "condition_number": kappa,
        "right_eigvec":     v,    # reproductive value
        "left_eigvec":      vs,   # stable distribution
        "eigenvalues":      ev_r[ir],
        "transient_possible": (sigma > 1.0) and (rho < 1.0),
        "amplification_ratio": sigma / rho if rho > 1e-15 else np.inf,
    }


def amplification_envelope(R_mat: np.ndarray, n_max: int = 30) -> dict:
    """
    Compute amplification envelopes A(n) and A₁(n) over n generations.

    A(n)  = ||R^n||₂   (ℓ² norm, worst-case multi-location seeding)  (Eq 39)
    A₁(n) = ||R^n||₁   (ℓ¹ norm, worst-case single-location seeding) (Eq 40)

    A(0)=1, A(1)=σ(t);  A₁(0)=1, A₁(1)=max_k R^k_out(t).
    """
    rho = R_network(R_mat)
    A  = np.zeros(n_max + 1)
    A1 = np.zeros(n_max + 1)
    Rn = np.eye(R_mat.shape[0])
    for n in range(n_max + 1):
        A[n]  = float(np.linalg.svd(Rn, compute_uv=False)[0])
        A1[n] = float(Rn.sum(axis=1).max())   # max row sum = max_k R^k_out (code stores R^T)
        Rn    = Rn @ R_mat
    ns = np.arange(n_max + 1)
    return {"n": ns, "A": A, "A1": A1, "rho_n": rho ** ns, "rho": rho}


# ═══════════════════════════════════════════════════════════════════════════════
# 5.  SENSITIVITY AND ELASTICITY  (Section 3.1.4, Eqs 29-35)
# ═══════════════════════════════════════════════════════════════════════════════

def sensitivity_elasticity(R_mat: np.ndarray) -> dict:
    """
    Sensitivity s^{kj} = v_j * v*_k / (v·v*)  and  elasticity ε^{kj}.  (Eqs 31,33)

    s^{kj}(t) = ∂R(t)/∂R^{kj}(t): sensitivity of spectral radius to entry (k,j).
    ε^{kj}(t) = (R^{kj}/R) * s^{kj}: 1% change in R^{kj} → ε^{kj}% change in R.
    sum_{k,j} ε^{kj} = 1 identically.

    Returns
    -------
    dict with keys:
        sensitivity   (N,N)  s^{kj}
        elasticity    (N,N)  ε^{kj}
        elasticity_out (N,)  ε^k_out = sum_j ε^{kj}  infector elasticity (Eq 34)
        elasticity_in  (N,)  ε^j_in  = sum_k ε^{kj}  infectee elasticity (Eq 35)
        rho            float
        condition_number float
    """
    spec = spectral_analysis(R_mat)
    rho  = spec["rho"]
    v    = spec["right_eigvec"]   # reproductive value
    vs   = spec["left_eigvec"]    # stable distribution

    vw   = max(float(vs @ v), 1e-15)
    # s^{kj} = v*_k * v_j / (v*·v)  →  S[k,j] = vs[k]*v[j]/vw
    S_mat = np.outer(vs, v) / vw
    E_mat = (R_mat / rho) * S_mat if rho > 1e-15 else np.zeros_like(R_mat)

    return {
        "sensitivity":     S_mat,
        "elasticity":      E_mat,
        "elasticity_out":  E_mat.sum(axis=1),
        "elasticity_in":   E_mat.sum(axis=0),
        "rho":             rho,
        "condition_number": spec["condition_number"],
    }


# ═══════════════════════════════════════════════════════════════════════════════
# 6.  TYPE REPRODUCTION NUMBERS  (Section 3.1.7, Eqs 52-55)
# ═══════════════════════════════════════════════════════════════════════════════

def type_R_number(R_mat: np.ndarray, j: int) -> float:
    """
    Instantaneous type reproduction number T^j_type(t) for location j.  (Eq 53)

    T^j_type = R_jj + R_jJ (I - R_JJ)^{-1} R_Jj   where J = {0..L}\\{j}

    Returns np.nan when ρ(R_JJ) >= 1 (background sustains epidemic without j).
    T^j_type > 1 <=> R(t) > 1  (for irreducible R(t)).
    """
    N = R_mat.shape[0]
    J  = [k for k in range(N) if k != j]
    R_jj  = R_mat[j, j]
    if not J:                               # single-location network: T_j = R_jj
        return float(R_jj)
    R_jJ  = R_mat[j, J]
    R_Jj  = R_mat[np.ix_(J, [j])].flatten()
    R_JJ  = R_mat[np.ix_(J, J)]
    if float(np.max(np.abs(np.linalg.eigvals(R_JJ)))) >= 1.0:
        return np.nan
    try:
        res = np.linalg.solve(np.eye(len(J)) - R_JJ, np.eye(len(J)))
    except np.linalg.LinAlgError:
        return np.nan
    return float(R_jj + R_jJ @ res @ R_Jj)


def type_R_all(R_mat: np.ndarray) -> np.ndarray:
    """T^j_type(t) for every location j.  NaN where undefined."""
    return np.array([type_R_number(R_mat, j) for j in range(R_mat.shape[0])])


def group_type_R(R_mat: np.ndarray, P: list) -> float:
    """
    Group type reproduction number T^P_type(t).  (Eq 55)

    T^P_type = rho( R_PP + R_PQ (I-R_QQ)^{-1} R_QP )   where Q = complement(P).
    Returns np.nan when ρ(R_QQ) >= 1.
    """
    P = sorted(set(P))
    N = R_mat.shape[0]
    Q = [k for k in range(N) if k not in P]
    if not Q:
        return R_network(R_mat)
    R_PP = R_mat[np.ix_(P, P)]
    R_PQ = R_mat[np.ix_(P, Q)]
    R_QP = R_mat[np.ix_(Q, P)]
    R_QQ = R_mat[np.ix_(Q, Q)]
    if float(np.max(np.abs(np.linalg.eigvals(R_QQ)))) >= 1.0:
        return np.nan
    try:
        res = np.linalg.solve(np.eye(len(Q)) - R_QQ, np.eye(len(Q)))
    except np.linalg.LinAlgError:
        return np.nan
    return R_network(R_PP + R_PQ @ res @ R_QP)


# ═══════════════════════════════════════════════════════════════════════════════
# 7.  GENERATION TIME DISTRIBUTIONS  (Section 3.1.2, Eqs 16-17)
# ═══════════════════════════════════════════════════════════════════════════════

def compute_generation_times(
    f_t: np.ndarray,
    S: np.ndarray,
    populations: np.ndarray,
    infectiousness_profile: np.ndarray,
    lw: float,
    lb: float,
) -> dict:
    """
    All GT distributions at one time point.

    Under the separable kernel (Eq 7), the GT is UNIVERSAL:
        g^{kj}(t,a_E) = p(a_E)  for ALL (k,j) and ALL t.

    This is because K^{kj}(t,a_E) = p(a_E) * S_j * K_base[k,j], so
    S_j*K_base cancels in g = K/R.  Contact rate heterogeneity (lw ≠ lb)
    affects R magnitudes but not GT shapes.

    Returns
    -------
    dict with keys:
        g_universal  (max_days,)  — universal p(a_E) (all types collapse here)
        mean_gt      float        — expected generation time (days)
        R_matrix     (N,N)        — R^{kj} at this time
        R_meeting    (N,)         — R^l_meeting
        g_pairwise   (max_days,N,N) — g^{kj} (= universal for separable kernel)
        g_outward    (max_days,N)  — g^k_out
        g_inward     (max_days,N)  — g^j_in
        g_meeting    (max_days,N)  — g^l_meeting
        g_network    (max_days,)   — network GT (universal p(a)/Σp; see note above)
    """
    max_days = len(infectiousness_profile)
    p = infectiousness_profile
    # Renewal generation-time profile: exclude same-day (a=0) transmission and
    # renormalise over a ≥ 1, matching the forward simulator and compute_R_matrix.
    p_r = p.copy(); p_r[0] = 0.0
    p_r = p_r / p_r.sum() if p_r.sum() > 1e-15 else p_r
    N = f_t.shape[0]
    K_base, _, _, N_eff = _kernel_base(f_t, populations, lw, lb)
    inv_N = np.where(N_eff > 0, 1.0 / N_eff, 0.0)
    S_eff = f_t.T @ S

    R_mat  = p[1:].sum() * S[np.newaxis, :] * K_base
    R_meet = R_meeting(f_t, S, populations, p, lw, lb)

    # Universal GT (a ≥ 1)
    g_univ = p_r.copy()
    mean_gt = float(np.sum(np.arange(max_days) * p))   # GT distribution mean (full support)

    # Pairwise: tile universal
    g_pw = np.zeros((max_days, N, N))
    for a in range(max_days):
        g_pw[a] = p_r[a]   # same for all (k,j)

    g_out  = np.tile(p_r[:, np.newaxis], (1, N))
    g_in   = np.tile(p_r[:, np.newaxis], (1, N))
    g_net  = p_r.copy()

    g_meet = np.zeros((max_days, N))
    for l in range(N):
        if R_meet[l] > 1e-15:
            g_meet[:, l] = p_r

    return {
        "g_universal": g_univ,
        "mean_gt":     mean_gt,
        "R_matrix":    R_mat,
        "R_meeting":   R_meet,
        "g_pairwise":  g_pw,
        "g_outward":   g_out,
        "g_inward":    g_in,
        "g_meeting":   g_meet,
        "g_network":   g_net,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# 8.  INDEPENDENT PER-LOCATION R̂(t)  (Eq 56)
# ═══════════════════════════════════════════════════════════════════════════════

# Cori/EpiEstim default prior Gamma(shape=1, scale=5) → mean 5; the posterior-mean
# denominator adds the RATE = 1/scale = 0.2 (not the scale, 5).  R̂ is reported only
# once the windowed incidence is non-trivial (EpiEstim default ≥ 12).
R_PRIOR_SHAPE = 1.0
R_PRIOR_RATE  = 0.2
R_MIN_WINDOW_INCIDENCE = 12.0


def estimate_R_independent(
    incidence: np.ndarray,
    infectiousness_profile: np.ndarray,
    window: int = 7,
) -> np.ndarray:
    """
    Sliding-window Bayesian R̂^j_ind(t) (Cori 2013) per location.  (Eq 56)

    Assumes each location is a closed population — ignores import/export.
    Compare with R^j_in and R^j_out to reveal spatial-neglect bias.

    Parameters
    ----------
    incidence : (T, N)
    infectiousness_profile : (max_days,)
    window : int
        Width of sliding window in days.

    Returns
    -------
    (T, N) array with np.nan for days lacking sufficient data.
    """
    T, N  = incidence.shape
    max_s = len(infectiousness_profile)
    R_est = np.full((T, N), np.nan)
    pa, pb = R_PRIOR_SHAPE, R_PRIOR_RATE   # EpiEstim default prior (mean 5)
    for j in range(N):
        for t in range(window, T):
            t0  = max(0, t - window + 1)
            obs = incidence[t0:t+1, j].sum()
            lam = 0.0
            for tw in range(t0, t+1):
                # lag a weighted by infectiousness_profile[a] = p(a), matching the
                # forward simulator so the estimator uses the SAME generation interval.
                for a in range(1, max_s):
                    if tw - a >= 0:
                        lam += infectiousness_profile[a] * incidence[tw-a, j]
            if lam > 1e-4 and obs >= R_MIN_WINDOW_INCIDENCE:
                R_est[t, j] = (pa + obs) / (pb + lam)
    return R_est


# ═══════════════════════════════════════════════════════════════════════════════
# 9.  GROWTH RATE AND SOURCE-SINK UTILITIES
# ═══════════════════════════════════════════════════════════════════════════════

def euler_lotka_r(rho: float, g_tilde: np.ndarray) -> float:
    """Solve Euler-Lotka 1 = R(t) Σ_{a≥1} g(a) e^{-ra} for r.

    Excludes the a=0 (same-day) term and renormalises the generation-time
    distribution over a ≥ 1, matching the renewal process the simulator obeys.
    """
    if rho <= 0 or g_tilde.sum() < 1e-15:
        return np.nan
    g = np.asarray(g_tilde, dtype=float).copy()
    g[0] = 0.0
    gs = g.sum()
    if gs < 1e-15:
        return np.nan
    g = g / gs
    days = np.arange(len(g))
    def f(r): return rho * float(np.sum(g * np.exp(-r * days))) - 1.0
    try:
        if abs(rho - 1.0) < 1e-8:
            return 0.0
        lo, hi = (-2.0, 0.0) if rho < 1 else (0.0, 2.0)
        if f(lo) * f(hi) >= 0:
            return np.nan
        return float(brentq(f, lo, hi, xtol=1e-8, maxiter=200))
    except Exception:
        return np.nan


def empirical_growth_rate(incidence: np.ndarray, window: int = 3) -> np.ndarray:
    """r̂(t) = [log I(t+w) - log I(t-w)] / (2w).  Shape (T,)."""
    tot  = incidence.sum(axis=1).astype(float)
    T    = len(tot)
    rhat = np.full(T, np.nan)
    for t in range(window, T - window):
        if tot[t+window] > 0 and tot[t-window] > 0:
            rhat[t] = (np.log(tot[t+window]) - np.log(tot[t-window])) / (2*window)
    return rhat


def source_sink_analysis(R_mat: np.ndarray) -> dict:
    """
    Source-sink decomposition at one time point.

    Returns dict with R_outward, R_inward, net_export (= R_out - R_in),
    sources (net > 0), sinks (net < 0), pi_within, pi_between, eta (Eq 46).
    """
    Ro  = R_outward(R_mat)
    Ri  = R_inward(R_mat)
    net = Ro - Ri
    tot = R_mat.sum()
    pi_w = np.trace(R_mat) / tot if tot > 0 else 0.0
    cv_o = Ro.std() / Ro.mean() if Ro.mean() > 0 else 0.0
    cv_i = Ri.std() / Ri.mean() if Ri.mean() > 0 else 0.0
    return {
        "R_outward":  Ro, "R_inward": Ri,
        "net_export": net,
        "sources":    np.where(net > 0)[0],
        "sinks":      np.where(net < 0)[0],
        "pi_within":  pi_w, "pi_between": 1.0 - pi_w,
        "eta":        cv_o / cv_i if cv_i > 0 else np.inf,
    }


def effective_populations_series(
    f_jk: np.ndarray,
    S_series: np.ndarray,
    incidence: np.ndarray,
    infectiousness_profile: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    S^l_eff(t) = Σ_j f^{jl}(t) S_j(t)  and  I^l_eff(t) = Σ_k f^{kl}(t) Λ_k(t).

    Returns (S_eff, I_eff) each shape (T, N).
    """
    T, N = incidence.shape
    max_s = len(infectiousness_profile)
    S_eff = np.zeros((T, N))
    I_eff = np.zeros((T, N))
    for t in range(T):
        S_eff[t] = f_jk[t].T @ S_series[t]
        lam = np.zeros(N)
        for s in range(1, min(max_s, t+1)):
            lam += infectiousness_profile[s] * incidence[t-s]
        I_eff[t] = f_jk[t].T @ lam
    return S_eff, I_eff


# ═══════════════════════════════════════════════════════════════════════════════
# 10.  PDE SIMULATION ENGINE  (Section 2.4, Eqs 4-6)
# ═══════════════════════════════════════════════════════════════════════════════

def simulate_epidemic(
    f_jk: np.ndarray,
    populations: np.ndarray,
    infectiousness_profile: np.ndarray,
    lw_raw: float,
    lb_raw: float,
    R0_target: float | None,
    initial_infections: np.ndarray,
    T: int,
    *,
    birth_rate: float = 3e-5,
    death_rate: float = 3e-5,
    susceptible_depletion: bool = True,
    compute_type_r: bool = False,
    type_r_stride: int = 5,
    verbose: bool = True,
    _beta_mode: bool = False,
) -> dict:
    """
    First-order upwind PDE simulation of ∂E^j/∂t + ∂E^j/∂a_E = 0.  (Eq 4)

    Boundary condition (Eq 6):
        E^j(t,0) = S_j(t) * Σ_k K_base[k,j] * Σ_{a_E} p(a_E) E^k(t,a_E)

    Two modes:
    - **β mode** (``_beta_mode=True`` or ``R0_target=None``): lw_raw and lb_raw
      are already the effective transmission rates lw=β×λ_W, lb=β×λ_B.  No
      calibration is performed.  R₀ is reported as an *output*.
    - **R₀ calibration mode** (``R0_target`` is a positive float): lw_raw and
      lb_raw are contact rates that are scaled internally so that R(0)=R₀_target.

    Parameters
    ----------
    f_jk : (T, N, N)   pre-validated row-stochastic mobility matrix.
    populations : (N,)
    infectiousness_profile : (max_days,)  normalised.
    lw_raw, lb_raw : float
        In β-mode: effective rates lw=β×λ_W, lb=β×λ_B (used directly).
        In R₀-mode: raw contact rates that are scaled to hit R₀_target.
    R0_target : float or None
        Calibration target (R₀-mode) or None (β-mode).
    initial_infections : (N,) seed.
    T : int                  simulation length.
    birth_rate, death_rate : demographic rates (default 3e-5/day).
    susceptible_depletion : deplete S as infections occur.
    compute_type_r : compute T^j_type every `type_r_stride` days (expensive).
    type_r_stride : days between type-R computations.
    verbose : print progress.
    _beta_mode : internal flag — set True to bypass calibration.

    Returns
    -------
    dict with the full output suite — all quantities from Tables 1-2.
    """
    N       = len(populations)
    max_days = len(infectiousness_profile)
    p        = infectiousness_profile
    S        = populations.copy().astype(float)

    # Storage
    S_ser   = np.zeros((T, N))
    E_pde   = np.zeros((T, N, max_days))
    inc_mat = np.zeros((T, N, N))
    R_pw    = np.zeros((T, N, N))
    R_meet  = np.zeros((T, N))
    R_out_s = np.zeros((T, N))
    R_in_s  = np.zeros((T, N))
    R_net_s = np.zeros(T)
    E_ra_s  = np.zeros(T)
    sens_s  = np.zeros((T, N, N))
    elas_s  = np.zeros((T, N, N))
    elo_s   = np.zeros((T, N))
    eli_s   = np.zeros((T, N))
    mix_s   = np.zeros(T)
    gap_s   = np.zeros(T)
    sig_s   = np.zeros(T)
    cond_s  = np.zeros(T)
    rev_s   = np.zeros((T, N))   # reproductive value
    sd_s    = np.zeros((T, N))   # stable distribution
    type_r_s = np.full((T, N), np.nan) if compute_type_r else None

    # Initial conditions
    E_pde[0, :, 0] = initial_infections
    S -= initial_infections
    S_ser[0] = S.copy()

    # ── Calibration / β-mode ─────────────────────────────────────────────────
    f0 = f_jk[0]
    if _beta_mode or R0_target is None:
        # β-mode: lw_raw and lb_raw are the final effective rates, no scaling
        lw, lb = lw_raw, lb_raw
        R0_check    = compute_R_matrix(f0, populations, populations, p, lw, lb)
        R0_achieved = R_network(R0_check)
        if verbose:
            print(f"  β-mode: lw={lw:.4f}, lb={lb:.4f}, R₀={R0_achieved:.4f}")
    else:
        # R₀-calibration mode
        R0_mat = compute_R_matrix(f0, populations, populations, p, lw_raw, lb_raw)
        rho0   = R_network(R0_mat)
        if rho0 < 1e-15:
            raise RuntimeError(
                f"Initial R(0) = {rho0:.2e} ≈ 0. "
                "Check that f_jk has non-trivial off-diagonal entries and "
                "contact rates are positive."
            )
        scale = R0_target / rho0
        lw, lb = lw_raw * scale, lb_raw * scale
        R0_check    = compute_R_matrix(f0, populations, populations, p, lw, lb)
        R0_achieved = R_network(R0_check)
        if verbose:
            print(f"  Calibration: raw ρ₀={rho0:.4f}, scale={scale:.4f}, "
                  f"achieved R₀={R0_achieved:.4f} (target={R0_target})")

    def _store(t, R_mat, f_t, S_t):
        Ro = R_outward(R_mat)
        Ri = R_inward(R_mat)
        rho = R_network(R_mat)
        R_out_s[t] = Ro
        R_in_s[t]  = Ri
        R_net_s[t] = rho
        E_ra_s[t]  = E_risk_averse(R_mat)
        spec = spectral_analysis(R_mat)
        mix_s[t]  = spec["mixing_ratio"]
        gap_s[t]  = spec["spectral_gap"]
        sig_s[t]  = spec["sigma"]
        cond_s[t] = spec["condition_number"]
        rev_s[t]  = spec["right_eigvec"]
        sd_s[t]   = spec["left_eigvec"]
        se = sensitivity_elasticity(R_mat)
        sens_s[t] = se["sensitivity"]
        elas_s[t] = se["elasticity"]
        elo_s[t]  = se["elasticity_out"]
        eli_s[t]  = se["elasticity_in"]
        R_meet[t] = R_meeting(f_t, S_t, populations, p, lw, lb)
        if compute_type_r and (t % type_r_stride == 0):
            type_r_s[t] = type_R_all(R_mat)

    R_pw[0] = R0_check
    _store(0, R0_check, f0, S)

    # ── Time loop ─────────────────────────────────────────────────────────────
    for t in range(1, T):
        f_t = f_jk[min(t, len(f_jk)-1)]

        # Upwind shift: E(t, a) = E(t-1, a-1)
        E_pde[t, :, 1:] = E_pde[t-1, :, :-1]

        # Force of infection
        K_base, _, _, _ = _kernel_base(f_t, populations, lw, lb)
        wE = E_pde[t, :, 1:] @ p[1:]          # Σ_{a>=1} p(a) E_k(t,a), shape (N,)

        contrib = K_base * wE[:, np.newaxis]   # [k, j]
        expected = np.maximum(S * contrib.sum(axis=0), 0.0)
        new_j = np.minimum(expected, S) if susceptible_depletion else expected

        # Pairwise incidence decomposition
        col_sum = contrib.sum(axis=0)
        for j in range(N):
            if col_sum[j] > 1e-15 and new_j[j] > 0:
                inc_mat[t, :, j] = new_j[j] * contrib[:, j] / col_sum[j]

        E_pde[t, :, 0] = new_j

        R_mat_t = compute_R_matrix(f_t, S, populations, p, lw, lb)
        R_pw[t] = R_mat_t
        _store(t, R_mat_t, f_t, S)

        if susceptible_depletion:
            S = np.maximum(S - new_j + birth_rate*populations - death_rate*S, 0.0)
        S_ser[t] = S.copy()

        if verbose and t % 50 == 0:
            print(f"  t={t:4d}: R={R_net_s[t]:.3f}, "
                  f"σ={sig_s[t]:.3f}, inc={new_j.sum():.0f}")

    return {
        # Core epidemic
        "incidence":          E_pde[:, :, 0],
        "incidence_matrix":   inc_mat,
        "susceptibles":       S_ser,
        "E_pde_state":        E_pde,
        # Reproduction numbers (all time)
        "R_pairwise":         R_pw,
        "R_outward":          R_out_s,
        "R_inward":           R_in_s,
        "R_network":          R_net_s,
        "R_meeting":          R_meet,
        "E_risk_averse":      E_ra_s,
        # Spectral
        "mixing_ratio":       mix_s,
        "spectral_gap":       gap_s,
        "reactivity":         sig_s,
        "condition_number":   cond_s,
        "reproductive_value": rev_s,
        "stable_distribution": sd_s,
        # Sensitivity/elasticity
        "sensitivity":        sens_s,
        "elasticity":         elas_s,
        "elasticity_out":     elo_s,
        "elasticity_in":      eli_s,
        # Type R (if requested)
        "type_R":             type_r_s,
        # Calibration
        "lw": lw, "lb": lb, "R0_achieved": R0_achieved,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# 11.  MAIN CLASS
# ═══════════════════════════════════════════════════════════════════════════════

class MobilityRtFramework:
    """
    Mobility-Informed Renewal Equation Framework.

    Wraps validate → simulate → post-process into a single object.
    All outputs from Tables 1-2 of Mills (2026) are returned by .simulate().

    Two transmission parametrisation modes — provide exactly one:

    **Mode A — β (recommended for outbreak-start analysis):**
        ``beta`` = per-contact transmission probability ∈ (0, 1].
        R₀ is an *output* of the simulation, not an input.
        Use this when you know (or wish to scan) the transmission biology
        rather than assuming a particular R₀.

    **Mode B — R₀ target (for scenario analysis):**
        ``R0_target`` = desired basic reproduction number.
        Contact rates are scaled so R(t=0) = R₀_target.

    Parameters
    ----------
    f_jk : (T, N, N) or (N, N) row-stochastic mobility matrix.
    populations : (N,) resident populations.
    infectiousness_profile : (max_days,) discretised p(a_E), normalised.
    contact_rate_home : λ_W contacts/day at home location (e.g. 13.0, POLYMOD).
    contact_rate_away : λ_B contacts/day away from home (e.g. 0.30 × λ_W).
    beta : per-contact transmission probability (Mode A). Mutually exclusive
           with R0_target.
    R0_target : calibration target (Mode B). Mutually exclusive with beta.
    initial_infections : (N,) seed infections.
    T : simulation duration (days).
    birth_rate, death_rate : demographic rates (default 3×10⁻⁵/day).
    susceptible_depletion : bool (default True).
    location_names : list of str, optional.
    """

    def __init__(
        self,
        f_jk: np.ndarray,
        populations: np.ndarray,
        infectiousness_profile: np.ndarray,
        contact_rate_home: float,
        contact_rate_away: float,
        initial_infections: np.ndarray,
        T: int,
        *,
        beta: float | None = None,
        R0_target: float | None = None,
        birth_rate: float = 3e-5,
        death_rate: float = 3e-5,
        susceptible_depletion: bool = True,
        location_names: list | None = None,
        verbose: bool = True,
    ):
        validated = validate_inputs(
            f_jk, populations, infectiousness_profile,
            contact_rate_home, contact_rate_away,
            initial_infections, T,
            beta=beta, R0_target=R0_target,
            verbose=verbose,
        )
        self.f_jk         = validated["f_jk"]
        self.populations  = validated["populations"]
        self.p            = validated["infectiousness_profile"]
        self.lw_raw       = float(contact_rate_home)
        self.lb_raw       = float(contact_rate_away)
        self.beta         = validated["beta"]
        self.R0_target    = validated["R0_target"]
        self.seed         = validated["initial_infections"]
        self.T            = validated["T"]
        self.N            = validated["N"]
        self.max_days     = validated["max_days"]
        self.birth_rate   = birth_rate
        self.death_rate   = death_rate
        self.susceptible_depletion = susceptible_depletion
        self.verbose      = verbose
        self.location_names = (
            location_names if location_names is not None
            else [f"L{i+1}" for i in range(self.N)]
        )
        self._results: dict | None = None

    # ── simulation ────────────────────────────────────────────────────────────

    def simulate(
        self,
        *,
        compute_type_r: bool = False,
        type_r_stride: int = 5,
    ) -> dict:
        """
        Run the full epidemic simulation and compute all framework quantities.

        Parameters
        ----------
        compute_type_r : bool
            Compute T^j_type every `type_r_stride` days (expensive; O(N³) per step).
        type_r_stride : int
            Days between type-R computations.

        Returns
        -------
        dict
            Full output suite (all quantities from Tables 1-2 of manuscript).
            Also accessible as self.results after calling simulate().
        """
        if self.verbose:
            print(f"\nRunning simulation: N={self.N}, T={self.T} days")

        # Determine effective contact rates for whichever mode is active
        if self.beta is not None:
            lw_eff = self.beta * self.lw_raw
            lb_eff = self.beta * self.lb_raw
        else:
            lw_eff = self.lw_raw
            lb_eff = self.lb_raw

        results = simulate_epidemic(
            self.f_jk, self.populations, self.p,
            lw_eff, lb_eff, self.R0_target,
            self.seed, self.T,
            birth_rate=self.birth_rate, death_rate=self.death_rate,
            susceptible_depletion=self.susceptible_depletion,
            compute_type_r=compute_type_r,
            type_r_stride=type_r_stride,
            verbose=self.verbose,
            _beta_mode=(self.beta is not None),
        )

        # Augment with post-simulation quantities
        inc = results["incidence"]
        results["R_independent"] = estimate_R_independent(inc, self.p)
        results["growth_rate_empirical"] = empirical_growth_rate(inc)
        results["growth_rate_EL"] = np.array([
            euler_lotka_r(results["R_network"][t], self.p)
            for t in range(self.T)
        ])
        S_eff, I_eff = effective_populations_series(
            self.f_jk, results["susceptibles"], inc, self.p)
        results["S_eff"] = S_eff
        results["I_eff"] = I_eff

        self._results = results
        if self.verbose:
            peak = int(inc.sum(axis=1).argmax())
            att  = inc.sum() / self.populations.sum() * 100
            print(f"\nSummary: peak day={peak}, attack rate={att:.1f}%, "
                  f"R₀_achieved={results['R0_achieved']:.4f}")
        return results

    @property
    def results(self) -> dict:
        if self._results is None:
            raise RuntimeError("Call .simulate() first")
        return self._results

    # ── convenience analysis methods ──────────────────────────────────────────

    def get_generation_times_snapshot(self, t: int) -> dict:
        """GT distributions at day t (uses calibrated lw/lb)."""
        if self._results is None:
            raise RuntimeError("Call .simulate() first")
        r = self._results
        return compute_generation_times(
            self.f_jk[t], r["susceptibles"][t], self.populations,
            self.p, r["lw"], r["lb"]
        )

    def get_amplification_envelope(self, t: int, n_max: int = 30) -> dict:
        """Amplification envelope A(n) and A₁(n) at day t."""
        return amplification_envelope(self.results["R_pairwise"][t], n_max)

    def source_sink_at(self, t: int) -> dict:
        """Source-sink decomposition at day t."""
        return source_sink_analysis(self.results["R_pairwise"][t])

    def type_R_at(self, t: int) -> np.ndarray:
        """Compute T^j_type at day t (always recomputed, not cached)."""
        return type_R_all(self.results["R_pairwise"][t])

    def group_type_R_at(self, t: int, P: list) -> float:
        """T^P_type at day t for group P."""
        return group_type_R(self.results["R_pairwise"][t], P)

    def controllability_effort(self, t: int) -> dict:
        """
        Minimum fractional reduction needed to drive R < 1 at day t.

        Returns homogeneous effort (same reduction everywhere) and
        heterogeneous effort (prioritised by elasticity).
        """
        R_mat = self.results["R_pairwise"][t]
        rho = R_network(R_mat)
        u_homog = max(0.0, 1.0 - 1.0/rho) if rho > 0 else 0.0
        se = sensitivity_elasticity(R_mat)
        order = np.argsort(se["elasticity_out"])[::-1]
        u_het = np.zeros(self.N)
        R_cur = R_mat.copy()
        for idx in order:
            if R_network(R_cur) <= 1.0:
                break
            lo, hi = 0.0, 1.0
            for _ in range(30):
                mid = (lo + hi) / 2.0
                Rt = R_cur.copy(); Rt[idx, :] *= (1.0 - mid)
                (hi := mid) if R_network(Rt) <= 1.0 else (lo := mid)
            u_het[idx] = hi
            R_cur[idx, :] *= (1.0 - u_het[idx])
        return {"u_homogeneous": u_homog, "u_heterogeneous": u_het,
                "rho": rho, "priority_order": order}

    # ═══════════════════════════════════════════════════════════════════════════
    # 12.  VISUALISATION SUITE
    # ═══════════════════════════════════════════════════════════════════════════

    # Shared style
    @staticmethod
    def _style():
        plt.rcParams.update({
            "font.family": "sans-serif", "font.size": 9,
            "axes.spines.top": False, "axes.spines.right": False,
            "axes.linewidth": 0.7, "xtick.major.size": 3,
            "ytick.major.size": 3, "legend.frameon": False,
            "lines.linewidth": 1.1, "savefig.dpi": 200,
            "savefig.bbox": "tight",
        })

    @staticmethod
    def _panel(ax, letter, x=-0.13, y=1.03):
        ax.text(x, y, letter, transform=ax.transAxes,
                fontsize=11, fontweight="bold", va="top")

    @staticmethod
    def _heatmap(ax, Z, T, N, *, cbar_label, letter=None,
                 cmap="viridis", vmin=None, vmax=None, ylabels=None):
        kw = dict(aspect="auto", origin="upper", cmap=cmap,
                  extent=[-0.5, T-0.5, N-0.5, -0.5], interpolation="nearest")
        if vmin is not None: kw["vmin"] = vmin
        if vmax is not None: kw["vmax"] = vmax
        im = ax.imshow(Z.T, **kw)
        ax.set_xlim(-0.5, T-0.5); ax.set_ylim(N-0.5, -0.5)
        ax.set_yticks(range(N))
        ax.set_yticklabels(ylabels if ylabels else [f"L{i+1}" for i in range(N)],
                           fontsize=7)
        ax.set_xlabel("Day $t$", fontsize=9)
        cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cb.set_label(cbar_label, fontsize=8)
        cb.ax.tick_params(labelsize=7)
        if letter:
            ax.text(-0.13, 1.03, letter, transform=ax.transAxes,
                    fontsize=11, fontweight="bold", va="top")

    def plot_overview(
        self,
        results: dict | None = None,
        *,
        save_path: str | None = None,
    ) -> plt.Figure:
        """
        6-panel epidemic overview figure (Figure 2 equivalent).

        Panels:
          A — mean mobility matrix f̄_{jk}
          B — home fraction f_{jj}(t) over time
          C — incidence E^j(t,0) heatmap
          D — effective susceptibles S^l_eff(t)
          E — R(t), σ(t), E(t) with total incidence
          F — infector elasticity ε^k_out(t) heatmap
        """
        self._style()
        r = results if results is not None else self.results
        inc   = r["incidence"]
        T, N  = inc.shape
        loc   = self.location_names
        OK    = OKABE_ITO

        f_mean = self.f_jk.mean(axis=0)
        f_diag = np.array([[self.f_jk[t, j, j] for j in range(N)] for t in range(T)])

        fig = plt.figure(figsize=(13, 8))
        gs  = gridspec.GridSpec(2, 3, hspace=0.55, wspace=0.48,
                                left=0.07, right=0.97, top=0.96, bottom=0.09)

        # A — mean mobility matrix
        ax = fig.add_subplot(gs[0, 0])
        im = ax.imshow(f_mean, cmap="Blues", aspect="auto")
        ax.set_xticks(range(N)); ax.set_xticklabels(loc, fontsize=7, rotation=45, ha="right")
        ax.set_yticks(range(N)); ax.set_yticklabels(loc, fontsize=7)
        ax.set_xlabel("Activity $k$"); ax.set_ylabel("Residence $j$")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04).set_label(r"$\bar{f}^{jk}$", fontsize=8)
        self._panel(ax, "A")

        # B — home fraction
        ax = fig.add_subplot(gs[1, 0])
        self._heatmap(ax, f_diag, T, N, cbar_label=r"$f^{jj}(t)$",
                      cmap="RdYlGn", vmin=0.3, vmax=1.0, letter="D", ylabels=loc)

        # C — incidence heatmap
        ax = fig.add_subplot(gs[0, 1])
        self._heatmap(ax, inc, T, N, cbar_label=r"$E^j(t,0)$",
                      cmap="YlOrRd", letter="B", ylabels=loc)
        ax.set_ylabel("Residence $j$")

        # D — S_eff
        ax = fig.add_subplot(gs[1, 1])
        self._heatmap(ax, r["S_eff"]/1e3, T, N,
                      cbar_label=r"$S^l_{\mathrm{eff}}$ ($10^3$)",
                      cmap="Blues", letter="E", ylabels=loc)
        ax.set_ylabel("Meeting $l$")

        # E — R(t), sigma(t), E(t) with incidence
        ax  = fig.add_subplot(gs[0, 2])
        ax2 = ax.twinx(); ax2.spines["right"].set_visible(True)
        t_arr = np.arange(T)
        Rn  = r["R_network"]; sg = r["reactivity"]; Et = r["E_risk_averse"]
        vR  = Rn > 0; vs = sg > 0; vE = Et > 0
        ax.plot(t_arr[vR], Rn[vR], color=OK[4], lw=1.1, label=r"$\mathcal{R}(t)$")
        ax.plot(t_arr[vs], sg[vs], color=OK[5], lw=1.0, ls="--", label=r"$\sigma(t)$")
        ax.plot(t_arr[vE], Et[vE], color=OK[6], lw=0.9, ls=":", label=r"$\mathcal{E}(t)$")
        ax.axhline(1.0, color="#666", ls="--", lw=0.8)
        tmask = (sg > 1) & (Rn < 1)
        if tmask.any():
            ax.fill_between(t_arr, 1.0, sg, where=tmask,
                            color="orange", alpha=0.18, label=r"$\sigma>1,\mathcal{R}<1$")
        total = inc.sum(axis=1)
        ax2.fill_between(t_arr, total/1e3, alpha=0.16, color=OK[1])
        ax2.plot(t_arr, total/1e3, color=OK[1], lw=0.9)
        ax.set_ylabel(r"$\mathcal{R}(t)$, $\sigma(t)$, $\mathcal{E}(t)$", color=OK[4])
        ax2.set_ylabel(r"Incidence ($10^3$)", color=OK[1])
        ax.set_xlabel("Day $t$"); ax.legend(fontsize=6.5, ncol=2, loc="upper right")
        ax.tick_params(axis="y", labelcolor=OK[4]); ax2.tick_params(axis="y", labelcolor=OK[1])
        self._panel(ax, "C")

        # F — elasticity
        ax = fig.add_subplot(gs[1, 2])
        elo = r["elasticity_out"]
        vmax_e = float(np.percentile(elo[elo > 0], 97)) if (elo > 0).any() else 1.0
        self._heatmap(ax, elo, T, N, cbar_label=r"$\varepsilon^k_{\mathrm{out}}(t)$",
                      cmap="YlOrRd", vmin=0, vmax=vmax_e, letter="F", ylabels=loc)
        ax.set_ylabel("Infector $k$")

        if save_path:
            plt.savefig(save_path)
        return fig

    def plot_R_taxonomy(
        self,
        results: dict | None = None,
        *,
        save_path: str | None = None,
    ) -> plt.Figure:
        """
        Taxonomy figure: R^j_out, R^j_in, pairwise R^{kj}, meeting R, type R.
        """
        self._style()
        r   = results if results is not None else self.results
        inc = r["incidence"]
        T, N = inc.shape
        loc  = self.location_names
        peak = int(inc.sum(axis=1).argmax())
        OK   = OKABE_ITO

        fig = plt.figure(figsize=(14, 9))
        gs  = gridspec.GridSpec(2, 4, hspace=0.55, wspace=0.50,
                                left=0.06, right=0.97, top=0.96, bottom=0.09)

        # A — R_out
        ax = fig.add_subplot(gs[0, 0])
        pos = r["R_outward"][r["R_outward"] > 0]
        self._heatmap(ax, r["R_outward"], T, N,
                      cbar_label=r"$R^k_{\mathrm{out}}(t)$", cmap="plasma",
                      vmin=np.percentile(pos, 2) if pos.size else 0,
                      vmax=np.percentile(pos, 97) if pos.size else 3,
                      letter="A", ylabels=loc)
        ax.set_ylabel("Infector $k$")

        # B — R_in
        ax = fig.add_subplot(gs[0, 1])
        pos = r["R_inward"][r["R_inward"] > 0]
        self._heatmap(ax, r["R_inward"], T, N,
                      cbar_label=r"$R^j_{\mathrm{in}}(t)$", cmap="viridis",
                      vmin=np.percentile(pos, 2) if pos.size else 0,
                      vmax=np.percentile(pos, 97) if pos.size else 3,
                      letter="B", ylabels=loc)
        ax.set_ylabel("Infectee $j$")

        # C — pairwise R^{kj} at peak
        ax = fig.add_subplot(gs[0, 2])
        Rpk = r["R_pairwise"][peak]
        im = ax.imshow(Rpk, cmap="PuOr", aspect="auto")
        ax.set_xticks(range(N)); ax.set_xticklabels(loc, fontsize=7, rotation=45, ha="right")
        ax.set_yticks(range(N)); ax.set_yticklabels(loc, fontsize=7)
        ax.set_xlabel("Infectee $j$"); ax.set_ylabel("Infector $k$")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04).set_label(
            rf"$R^{{kj}}$ (day {peak})", fontsize=8)
        self._panel(ax, "C")

        # D — R_meeting
        ax = fig.add_subplot(gs[0, 3])
        pos = r["R_meeting"][r["R_meeting"] > 0]
        self._heatmap(ax, r["R_meeting"], T, N,
                      cbar_label=r"$R^l_{\mathrm{meeting}}(t)$", cmap="cividis",
                      vmin=np.percentile(pos, 2) if pos.size else 0,
                      vmax=np.percentile(pos, 97) if pos.size else 3,
                      letter="D", ylabels=loc)
        ax.set_ylabel("Meeting $l$")

        # E — Source-sink at peak
        ax = fig.add_subplot(gs[1, 0])
        ss  = source_sink_analysis(r["R_pairwise"][peak])
        net = ss["net_export"]
        clrs = [OK[5] if x > 0 else OK[4] for x in net]
        ax.barh(range(N), net, color=clrs, height=0.65, edgecolor="none")
        ax.axvline(0, color="k", lw=0.8)
        ax.set_yticks(range(N)); ax.set_yticklabels(loc, fontsize=7)
        ax.set_xlabel(r"$R^k_{\mathrm{out}} - R^k_{\mathrm{in}}$")
        ax.set_title(f"Source-sink (day {peak})", fontsize=8)
        self._panel(ax, "E")

        # F — R_independent vs R_out comparison
        ax = fig.add_subplot(gs[1, 1])
        Rind = r.get("R_independent", None)
        if Rind is not None:
            for j in range(min(N, 3)):
                vi = ~np.isnan(Rind[:, j])
                col = OK[j % len(OK)]
                ax.plot(np.where(vi)[0], Rind[vi, j], "--", color=col, lw=0.9, alpha=0.7)
                ax.plot(np.arange(T), r["R_outward"][:, j], "-", color=col,
                        lw=0.9, label=loc[j])
        ax.axhline(1, color="#555", ls="--", lw=0.8)
        ax.set_xlabel("Day $t$"); ax.set_ylabel("$R$")
        ax.set_title(r"$R^k_{\mathrm{out}}$ (solid) vs $\hat{R}^j_{\mathrm{ind}}$ (dashed)",
                     fontsize=7.5)
        ax.legend(fontsize=6.5, ncol=1)
        self._panel(ax, "F")

        # G — elasticity scatter (time-mean)
        ax = fig.add_subplot(gs[1, 2])
        me_out = r["elasticity_out"].mean(axis=0)
        me_in  = r["elasticity_in"].mean(axis=0)
        sc = ax.scatter(me_out, me_in, s=55, c=np.arange(N), cmap="tab10",
                        edgecolors="#222", linewidths=0.5)
        for j in range(N):
            ax.annotate(loc[j], (me_out[j], me_in[j]),
                        textcoords="offset points", xytext=(4,3), fontsize=7)
        mx = max(me_out.max(), me_in.max(), 1e-9)
        ax.plot([0, mx], [0, mx], color="#999", ls="--", lw=0.8)
        ax.set_xlabel(r"time-mean $\varepsilon^k_{\mathrm{out}}$")
        ax.set_ylabel(r"time-mean $\varepsilon^j_{\mathrm{in}}$")
        self._panel(ax, "G")

        # H — Type R numbers (if available)
        ax = fig.add_subplot(gs[1, 3])
        TR = r.get("type_R", None)
        if TR is not None and not np.all(np.isnan(TR)):
            TR_plot = np.nan_to_num(TR, nan=0.0)
            self._heatmap(ax, TR_plot, T, N,
                          cbar_label=r"$T^j_{\mathrm{type}}(t)$", cmap="plasma",
                          ylabels=loc)
            ax.set_title(r"Type $R$ (0 = undefined)", fontsize=8)
        else:
            ax.text(0.5, 0.5, "type_R not computed\n(set compute_type_r=True)",
                    ha="center", va="center", transform=ax.transAxes, fontsize=9)
        self._panel(ax, "H")

        if save_path:
            plt.savefig(save_path)
        return fig

    def plot_transience(
        self,
        results: dict | None = None,
        *,
        n_max_env: int = 20,
        save_path: str | None = None,
    ) -> plt.Figure:
        """
        Transience figure: mixing ratio, reactivity/R comparison, amplification
        envelope at three epidemic stages, eigenvalue magnitudes, condition number.
        """
        self._style()
        r   = results if results is not None else self.results
        inc = r["incidence"]
        T   = inc.shape[0]
        OK  = OKABE_ITO

        t_arr = np.arange(T)
        Rn    = r["R_network"]
        sg    = r["reactivity"]
        Et    = r["E_risk_averse"]
        mix   = r["mixing_ratio"]
        cond  = r["condition_number"]
        peak  = int(inc.sum(axis=1).argmax())
        early = max(1, peak // 3)
        late  = min(T-1, peak + 30)

        fig = plt.figure(figsize=(13, 9))
        gs  = gridspec.GridSpec(2, 3, hspace=0.55, wspace=0.50,
                                left=0.07, right=0.97, top=0.96, bottom=0.09)

        # A — mixing ratio s(t)
        ax = fig.add_subplot(gs[0, 0])
        vld = mix > 0
        ax.plot(t_arr[vld], mix[vld], color=OK[2], lw=1.1, label=r"$s(t)=|\lambda_2|/\mathcal{R}$")
        ax.axvline(peak, color="#777", ls=":", lw=0.8)
        ax.set_ylim(0, 1.05); ax.set_xlabel("Day $t$")
        ax.set_ylabel(r"$s(t)$"); ax.legend(fontsize=7.5)
        self._panel(ax, "A")

        # B — R(t), sigma(t), E(t), A_1(1)
        ax = fig.add_subplot(gs[0, 1])
        A1_1 = r["R_outward"].max(axis=1)   # A₁(1) = max_k R^k_out
        vR = Rn > 0; vs = sg > 0; vE = Et > 0; va = A1_1 > 0
        ax.plot(t_arr[vR], Rn[vR], color=OK[4], lw=1.1, label=r"$\mathcal{R}(t)$")
        ax.plot(t_arr[vs], sg[vs], color=OK[5], lw=1.0, label=r"$\sigma(t)$")
        ax.plot(t_arr[vE], Et[vE], color=OK[6], lw=0.9, ls="--", label=r"$\mathcal{E}(t)$")
        ax.plot(t_arr[va], A1_1[va], color=OK[1], lw=0.9, ls="-.", label=r"$\mathcal{A}_1(1)$")
        ax.axhline(1.0, color="#666", ls="--", lw=0.8)
        tmask = (sg > 1) & (Rn < 1)
        if tmask.any():
            ax.fill_between(t_arr, 1.0, sg, where=tmask, color="orange",
                            alpha=0.18, label=r"False-action zone")
        ax.set_xlabel("Day $t$"); ax.legend(fontsize=6.5, ncol=2, loc="upper right")
        self._panel(ax, "B")

        # C — amplification envelope at early/peak/late
        ax = fig.add_subplot(gs[0, 2])
        for t_phase, name, col in [(early,"early",OK[2]),(peak,"peak",OK[5]),(late,"late",OK[0])]:
            env = amplification_envelope(r["R_pairwise"][t_phase], n_max=n_max_env)
            rn  = env["rho_n"]
            ax.plot(env["n"], env["A"]  / (rn + 1e-300), color=col, lw=1.0,
                    label=f"$A(n)$ {name}")
            ax.plot(env["n"], env["A1"] / (rn + 1e-300), color=col, lw=0.7, ls=":")
        ax.axhline(1.0, color="#888", ls="--", lw=0.75)
        ax.set_xlabel("$n$ (generations)")
        ax.set_ylabel(r"Envelope / $\mathcal{R}^n$")
        ax.set_title(r"$A(n)/\mathcal{R}^n$ (solid), $\mathcal{A}_1(n)/\mathcal{R}^n$ (dotted)",
                     fontsize=7.5)
        ax.legend(fontsize=6, ncol=2)
        self._panel(ax, "C")

        # D — eigenvalue magnitudes
        ax = fig.add_subplot(gs[1, 0])
        lam1 = Rn
        lam2 = r["spectral_gap"]   # gap = lam1 - lam2, but store mix_ratio*Rn = lam2
        lam2_abs = r["mixing_ratio"] * Rn
        ax.plot(t_arr, lam1, color=OK[4], lw=1.1, label=r"$|\lambda_1|$")
        ax.plot(t_arr, lam2_abs, color=OK[2], lw=1.0, ls="--", label=r"$|\lambda_2|$")
        ax.axvline(peak, color="#777", ls=":", lw=0.8)
        ax.set_xlabel("Day $t$"); ax.set_ylabel("Eigenvalue magnitude")
        ax.legend(fontsize=7.5)
        self._panel(ax, "D")

        # E — condition number
        ax = fig.add_subplot(gs[1, 1])
        cond_clip = np.minimum(cond, 1e4)
        ax.semilogy(t_arr, cond_clip, color=OK[1], lw=1.0)
        ax.axvline(peak, color="#777", ls=":", lw=0.8)
        ax.set_xlabel("Day $t$"); ax.set_ylabel(r"$\kappa(\mathbf{R})$ (log)")
        self._panel(ax, "E")

        # F — within-location fraction pi_j(t)
        ax = fig.add_subplot(gs[1, 2])
        diag   = np.array([r["incidence_matrix"][t].diagonal() for t in range(T)])
        colsum = r["incidence"].clip(1e-15)
        pi_w   = diag / colsum
        N      = pi_w.shape[1]
        loc    = self.location_names
        im     = ax.imshow(pi_w.T, aspect="auto", origin="upper", cmap="RdYlGn",
                           vmin=0, vmax=1, extent=[-0.5, T-0.5, N-0.5, -0.5])
        ax.set_yticks(range(N)); ax.set_yticklabels(loc, fontsize=7)
        ax.set_xlabel("Day $t$"); ax.set_ylabel("Location $j$")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04).set_label(
            r"$\pi_j(t)$ within-frac", fontsize=8)
        self._panel(ax, "F")

        if save_path:
            plt.savefig(save_path)
        return fig

    def plot_generation_times(
        self,
        results: dict | None = None,
        *,
        save_path: str | None = None,
    ) -> plt.Figure:
        """
        Generation time distributions at early / peak / late epidemic stages.
        """
        self._style()
        r   = results if results is not None else self.results
        inc = r["incidence"]
        T   = inc.shape[0]
        peak  = int(inc.sum(axis=1).argmax())
        early = max(1, peak // 3)
        late  = min(T-1, peak + 30)
        OK = OKABE_ITO

        fig, axes = plt.subplots(1, 3, figsize=(11, 3.5),
                                 gridspec_kw=dict(left=0.07, right=0.97,
                                                  top=0.88, bottom=0.18, wspace=0.4))
        days = np.arange(self.max_days)
        for ax, t_phase, name in zip(axes, [early, peak, late], ["Early","Peak","Late"]):
            gt = self.get_generation_times_snapshot(t_phase)
            ax.plot(days, gt["g_universal"], color=OK[0], lw=1.6,
                    label=f"Universal $p(a_E)$\n(mean={gt['mean_gt']:.1f}d)")
            ax.fill_between(days, gt["g_universal"], alpha=0.2, color=OK[0])
            ax.set_xlabel("Infection age $a_E$ (days)")
            ax.set_ylabel("Probability density" if name == "Early" else "")
            ax.set_title(f"{name} (day {t_phase}): all GT types coincide", fontsize=8.5)
            ax.legend(fontsize=8)
            self._panel(ax, {"Early":"A","Peak":"B","Late":"C"}[name])

        if save_path:
            plt.savefig(save_path)
        return fig

    def plot_eigenvectors(
        self,
        results: dict | None = None,
        *,
        save_path: str | None = None,
    ) -> plt.Figure:
        """
        Reproductive value v(t) and stable distribution v*(t) over time.
        """
        self._style()
        r   = results if results is not None else self.results
        inc = r["incidence"]
        T, N = inc.shape
        loc  = self.location_names

        fig, axes = plt.subplots(1, 2, figsize=(10, 4),
                                 gridspec_kw=dict(left=0.07, right=0.97,
                                                  top=0.90, bottom=0.12, wspace=0.42))

        self._heatmap(axes[0], r["reproductive_value"], T, N,
                      cbar_label=r"$v_j(t)$ reproductive value",
                      cmap="YlGnBu", vmin=0, letter="A", ylabels=loc)
        axes[0].set_ylabel("Location $j$")

        self._heatmap(axes[1], r["stable_distribution"], T, N,
                      cbar_label=r"$v^*_j(t)$ stable distribution",
                      cmap="OrRd", vmin=0, letter="B", ylabels=loc)
        axes[1].set_ylabel("Location $j$")

        if save_path:
            plt.savefig(save_path)
        return fig


# ═══════════════════════════════════════════════════════════════════════════════
# 13.  UNIT TESTS AND VALIDATION
# ═══════════════════════════════════════════════════════════════════════════════

class _Tester:
    """Minimal test runner (no external dependencies)."""
    def __init__(self):
        self.passed = 0; self.failed = 0; self.errors: list[str] = []

    def check(self, name, cond, msg=""):
        if cond:
            self.passed += 1
        else:
            self.failed += 1
            self.errors.append(f"FAIL [{name}]: {msg}")

    def approx(self, name, a, b, atol=1e-8, rtol=1e-5):
        err = abs(float(a) - float(b))
        thr = atol + rtol * abs(float(b))
        self.check(name, err <= thr, f"|{a:.6g} - {b:.6g}| = {err:.2e} > {thr:.2e}")

    def summary(self):
        print(f"\n{'='*60}")
        print(f"Unit tests: {self.passed} passed, {self.failed} failed")
        if self.errors:
            for e in self.errors: print(f"  {e}")
        else:
            print("  ALL TESTS PASSED ✓")
        print("="*60)
        return self.failed == 0


def run_all_tests(verbose: bool = True) -> bool:
    """
    Run the full validation suite for mobility_rt.framework.

    Covers:
      1. Row-stochasticity of f_jk
      2. Non-negativity and normalisation of p(a_E)
      3. R_matrix non-negativity and column scaling
      4. R_outward / R_inward conservation (sum equality)
      5. Sensitivity formula vs finite difference
      6. Elasticity sum = 1
      7. Spectral radius threshold property
      8. Reactivity sigma >= R for normal matrix; sigma = R for symmetric
      9. Amplification envelope: A(0)=1, A(1)=sigma, non-decreasing at start
      10. Type R equivalence (singleton P vs scalar formula)
      11. Euler-Lotka identity
      12. Universal GT property (g = p for all (k,j))
      13. GT sums to 1
      14. Source-sink: sum(R_out) = sum(R_in) = total
      15. Validate_inputs catches bad inputs
      16. PDE simulation: R₀ achieved, incidence non-negative, S non-increasing
      17. Within-between decomposition sums to 1

    Returns True if all tests pass.
    """
    t = _Tester()
    rng = np.random.default_rng(2026)

    # ── helpers ───────────────────────────────────────────────────────────────
    def rand_f(N):
        raw = rng.uniform(0.1, 1.0, (N, N))
        return raw / raw.sum(axis=1, keepdims=True)

    def rand_R(N, scale=1.5):
        M = rng.uniform(0.1, scale, (N, N))
        return M

    if verbose: print("Running validation tests...")

    # 1. Row-stochastic
    N = 5
    f = rand_f(N)
    t.check("row_stochastic", np.allclose(f.sum(axis=1), 1.0, atol=1e-12),
            "f rows don't sum to 1")

    # 2. p(a_E) non-negative and sums to 1
    p = discretise_gamma(5.5, 1.8, 25)
    t.check("profile_nonneg", np.all(p >= 0), "p has negative values")
    t.approx("profile_sum1", p.sum(), 1.0, atol=1e-10)

    # 3. R_matrix non-negativity and scaling with S
    pops = rng.uniform(1e5, 1e6, N)
    S    = pops.copy()
    lw, lb = 13.0, 4.0
    R = compute_R_matrix(f, S, pops, p, lw, lb)
    t.check("R_nonneg", np.all(R >= -1e-14), "R_matrix has negative entries")
    # If S halved, R should halve column-wise
    R2 = compute_R_matrix(f, S/2, pops, p, lw, lb)
    t.check("R_scales_with_S", np.allclose(R2, R/2, atol=1e-10),
            "R doesn't scale linearly with S")

    # 4. Source-sink conservation
    t.approx("Rout_sum_eq_Rin_sum", R_outward(R).sum(), R_inward(R).sum(), atol=1e-10)

    # 5. Sensitivity vs finite difference
    R_test = rand_R(4)
    se = sensitivity_elasticity(R_test)
    S_mat = se["sensitivity"]
    delta = 1e-6
    for (ki, ji) in [(0,0),(1,2),(2,1),(3,3)]:
        R_plus = R_test.copy(); R_plus[ki, ji] += delta
        fd = (R_network(R_plus) - R_network(R_test)) / delta
        t.approx(f"sens_fd_{ki}{ji}", S_mat[ki, ji], fd, atol=1e-4)

    # 6. Elasticity sum = 1
    t.approx("elasticity_sum1", se["elasticity"].sum(), 1.0, atol=1e-10)

    # 7. R_network threshold
    R_big = rand_R(3, scale=3.0)
    R_sml = rand_R(3, scale=0.4)
    t.check("Rnet_big>1", R_network(R_big) > 1.0, f"rho={R_network(R_big):.4f}")
    t.check("Rnet_sml<1", R_network(R_sml) < 1.0, f"rho={R_network(R_sml):.4f}")

    # 8. Reactivity: sigma >= rho; equality for symmetric
    spec = spectral_analysis(R_test)
    t.check("sigma_geq_rho", spec["sigma"] >= spec["rho"] - 1e-12,
            f"sigma={spec['sigma']:.4f} < rho={spec['rho']:.4f}")
    Rsym = R_test + R_test.T  # symmetric → sigma == rho
    ss   = spectral_analysis(Rsym)
    t.approx("sigma_eq_rho_symmetric", ss["sigma"], ss["rho"], atol=1e-8)

    # 9. Amplification envelope
    env = amplification_envelope(R_test, n_max=15)
    t.approx("A0_eq_1", env["A"][0], 1.0, atol=1e-10)
    t.approx("A1_eq_sigma", env["A"][1], spec["sigma"], atol=1e-8)
    t.check("A_nonneg", np.all(env["A"] >= -1e-12), "A(n) has negatives")
    t.check("A1_0_eq_1", abs(env["A1"][0] - 1.0) < 1e-10, f"A1(0)={env['A1'][0]}")

    # 10. Group type R (P=full) = R_network
    R_irred = rand_R(3, 0.8)   # may have rho < 1
    gtr = group_type_R(R_irred, list(range(3)))
    t.approx("group_typeR_full", gtr, R_network(R_irred), atol=1e-8)

    # 11. Euler-Lotka identity (renewal profile: a ≥ 1, renormalised)
    p_r = p.copy(); p_r[0] = 0.0; p_r = p_r / p_r.sum()
    for rho_test in [0.5, 1.0, 2.0]:
        r_sol = euler_lotka_r(rho_test, p)
        if not np.isnan(r_sol):
            days = np.arange(len(p_r))
            resid = rho_test * float(np.sum(p_r * np.exp(-r_sol * days))) - 1.0
            t.approx(f"EL_identity_{rho_test}", resid, 0.0, atol=1e-6)

    # 12. Universal GT property: g^{kj} = p_r (a≥1 renormalised) for all (k,j)
    gt = compute_generation_times(f, S, pops, p, lw, lb)
    for a in range(25):
        t.check(f"universal_gt_a{a}",
                np.allclose(gt["g_pairwise"][a], p_r[a], atol=1e-12),
                "GT not universal at age {a}")

    # 13. GT sums to 1
    t.approx("gnet_sum1", gt["g_network"].sum(), 1.0, atol=1e-10)

    # 14. Source-sink
    ss2 = source_sink_analysis(R_test)
    t.approx("ss_out_eq_in", ss2["R_outward"].sum(), ss2["R_inward"].sum(), atol=1e-10)
    t.approx("pi_w+pi_b=1", ss2["pi_within"] + ss2["pi_between"], 1.0, atol=1e-12)

    # 15. validate_inputs catches bad inputs (updated to new signature)
    ok_f   = rand_f(3)
    ok_f3d = ok_f[np.newaxis].repeat(10, axis=0)
    ok_pop = np.array([1e5, 2e5, 3e5])
    ok_p   = discretise_gamma(5.0, 1.5, 10)
    ok_s   = np.array([1.0, 0.0, 0.0])
    for bad, desc in [
        (lambda: validate_inputs(ok_f3d[0,:2,:3], ok_pop, ok_p, 13, 4, ok_s, 10,
                                 beta=0.03),
         "bad f shape"),
        (lambda: validate_inputs(ok_f3d, ok_pop, ok_p, -1, 4, ok_s, 10,
                                 beta=0.03),
         "negative lw"),
        (lambda: validate_inputs(ok_f3d, ok_pop, ok_p, 13, 4, ok_s, 10,
                                 R0_target=0),
         "zero R0"),
        (lambda: validate_inputs(ok_f3d, ok_pop, ok_p, 13, 4, np.zeros(3), 10,
                                 beta=0.03),
         "zero seed"),
        (lambda: validate_inputs(ok_f3d, ok_pop, ok_p, 13, 4, ok_s, 10),
         "neither beta nor R0"),
        (lambda: validate_inputs(ok_f3d, ok_pop, ok_p, 13, 4, ok_s, 10,
                                 beta=0.03, R0_target=1.5),
         "both beta and R0"),
    ]:
        caught = False
        try: bad()
        except ValueError: caught = True
        t.check(f"validates_{desc}", caught, f"Did not raise ValueError for {desc}")

    # 16. PDE simulation smoke test — both modes
    N_s, T_s = 4, 30
    f_s = np.stack([rand_f(N_s)] * T_s)
    pop_s  = np.ones(N_s) * 1e5
    seed_s = np.array([5.0, 0, 0, 0])
    p_s    = discretise_gamma(5.0, 1.5, 12)
    # R₀-calibration mode
    res_s = simulate_epidemic(f_s, pop_s, p_s, 13.0, 3.9, 1.5, seed_s, T_s,
                              verbose=False)
    t.approx("R0_achieved_cal", res_s["R0_achieved"], 1.5, atol=0.05)
    # β-mode (same β as calibration gives same result)
    beta_eq = res_s["lw"] / 13.0
    res_b = simulate_epidemic(f_s, pop_s, p_s, beta_eq*13.0, beta_eq*3.9, None,
                              seed_s, T_s, verbose=False, _beta_mode=True)
    t.approx("R0_achieved_beta", res_b["R0_achieved"], res_s["R0_achieved"], atol=0.02)
    t.check("incidence_nonneg", np.all(res_s["incidence"] >= -1e-12),
            "negative incidence detected")
    S_tot = res_s["susceptibles"].sum(axis=1)
    t.check("S_nonincreasing", np.all(np.diff(S_tot) <= 1.0),
            "Susceptibles increased unexpectedly")

    # 17. Within-between decomposition
    D = np.diag([1.0, 2.0, 3.0])
    ss_d = source_sink_analysis(D)
    t.approx("pi_within_diag", ss_d["pi_within"], 1.0, atol=1e-12)

    t.summary()
    return t.failed == 0


# ═══════════════════════════════════════════════════════════════════════════════
# 14.  CONVENIENCE: SYNTHETIC NETWORK BUILDER
# ═══════════════════════════════════════════════════════════════════════════════

def build_synthetic_network(
    N: int = 6,
    *,
    node_types: list[str] | None = None,
    populations: np.ndarray | None = None,
    T: int = 200,
    seed: int = 42,
    day_variation_sd: float = 0.12,
    hub_attraction_power: float = 0.5,
    commuting_fracs: np.ndarray | None = None,
    decay_scale: float = 20.0,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Build a synthetic mobility matrix for demonstration purposes.

    Uses exponential distance-decay with population gravity weighting,
    day-of-week scaling (Mon-Thu=1.0, Fri=0.95, Sat=0.85, Sun=0.70),
    and daily lognormal noise.

    Parameters
    ----------
    N : int
        Number of locations (default 6).
    node_types : list of str, optional
        Node type labels (length N). Used for commuting_fracs if not provided.
    populations : (N,) array, optional
        Resident populations. If None, generated from node_types.
    T : int
        Number of time steps.
    seed : int
        Random seed.
    day_variation_sd : float
        SD of daily lognormal noise (0 = no day-to-day variation).
    hub_attraction_power : float
        Exponent on population in gravity model (0.5 = Simini radiation).
    commuting_fracs : (N,) array, optional
        Fraction of each day spent away from home for each location.
    decay_scale : float
        Distance decay scale (km).

    Returns
    -------
    f_jk : (T, N, N) row-stochastic mobility matrix
    populations : (N,) resident populations
    node_types : list of str
    """
    rng = np.random.default_rng(seed)

    # Default node types and populations
    if node_types is None:
        type_seq = (["core"]     * max(1, int(0.20*N)) +
                    ["dense"]    * max(1, int(0.30*N)) +
                    ["suburban"] * max(1, int(0.30*N)) +
                    ["peripheral"] * max(1, N - int(0.80*N)))
        node_types = type_seq[:N]
        while len(node_types) < N:
            node_types.append("suburban")

    type_pop  = {"core":800_000,"dense":600_000,"suburban":300_000,"peripheral":120_000}
    type_cfrac = {"core":0.40,"dense":0.35,"suburban":0.28,"peripheral":0.18}

    if populations is None:
        pop_raw = np.array([type_pop.get(t, 200_000) for t in node_types], float)
        pop_raw *= np.exp(rng.normal(0, 0.2, N))
        populations = np.round(pop_raw).astype(float)

    if commuting_fracs is None:
        commuting_fracs = np.array([type_cfrac.get(t, 0.25) for t in node_types])

    # Synthetic spatial coordinates (km)
    radii  = rng.exponential(decay_scale * 0.4, N)
    angles = rng.uniform(0, 2*np.pi, N)
    coords = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
    from scipy.spatial.distance import cdist
    dists  = cdist(coords, coords)

    # Base f matrix
    pop_mean = populations.mean()
    d_safe   = np.maximum(dists, 0.5)
    W = np.exp(-d_safe / decay_scale) * (populations / pop_mean)[np.newaxis, :] ** hub_attraction_power
    np.fill_diagonal(W, 0.0)
    base_f = np.zeros((N, N))
    row_W  = W.sum(axis=1)
    for j in range(N):
        if row_W[j] > 0:
            base_f[j] = commuting_fracs[j] * W[j] / row_W[j]
        base_f[j, j] = 1.0 - commuting_fracs[j]

    # Time-varying f_jk
    dow_scale = np.array([1.00, 1.00, 1.00, 1.00, 0.95, 0.85, 0.70])
    f_jk = np.zeros((T, N, N))
    for t in range(T):
        scale = float(dow_scale[t % 7] *
                      np.clip(rng.lognormal(0.0, day_variation_sd), 0.4, 2.0))
        for j in range(N):
            cf_j = commuting_fracs[j]
            scaled = float(np.clip(cf_j * scale, 0.0, 0.95))
            if cf_j > 1e-12:
                ratio     = scaled / cf_j
                f_jk[t,j] = base_f[j] * ratio
                f_jk[t,j,j] = 0.0
            f_jk[t,j,j] = max(0.0, 1.0 - f_jk[t,j].sum())
            rs = f_jk[t,j].sum()
            if rs > 1e-15:
                f_jk[t,j] /= rs

    return f_jk, populations, node_types


# ═══════════════════════════════════════════════════════════════════════════════
# 15.  ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys
    ok = run_all_tests(verbose=True)
    sys.exit(0 if ok else 1)
