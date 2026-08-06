"""mobility_rt.transmissibility."""
import numpy as np
from scipy.optimize import brentq
from mobility_rt.kernel import R_inward, R_outward, R_system
from mobility_rt.spectral import sensitivity_elasticity


def source_sink_analysis(R_mat):
    R_out = R_outward(R_mat);  R_in = R_inward(R_mat);  net = R_out - R_in
    return {"R_outward": R_out, "R_inward": R_in, "net_export": net,
            "sources": np.where(net > 0)[0], "sinks": np.where(net < 0)[0]}


def within_between_decomposition(R_mat):
    d = np.trace(R_mat);  s = R_mat.sum()
    pi = d / s if s > 0 else 0.0
    return {"pi_within": pi, "pi_between": 1.0 - pi}


def euler_lotka_r(rho_val, g_tilde):
    """Solve 1 = R(t) Σ_{a≥1} g̃(a) e^{-ra} for r.

    The a=0 (same-day) term is excluded and the generation-time distribution
    renormalised over a ≥ 1, matching the renewal process the simulator obeys
    (which drops a=0).  For a distribution with g̃(0)≈0 this is a no-op.
    """
    if rho_val <= 0 or g_tilde.sum() < 1e-15:
        return np.nan
    g = np.asarray(g_tilde, dtype=float).copy()
    g[0] = 0.0
    gs = g.sum()
    if gs < 1e-15:
        return np.nan
    g = g / gs
    days = np.arange(len(g))
    def f(r):
        return rho_val * float(np.sum(g * np.exp(-r * days))) - 1.0
    try:
        if abs(rho_val - 1.0) < 1e-8:
            return 0.0
        lo, hi = (-2.0, 0.0) if rho_val < 1 else (0.0, 2.0)
        if f(lo) * f(hi) >= 0:
            return np.nan
        return float(brentq(f, lo, hi, xtol=1e-8, maxiter=200))
    except Exception:
        return np.nan


def empirical_growth_rate(incidence, window=3):
    """r̂(t) = [log I(t+w) − log I(t−w)] / (2w)."""
    tot  = incidence.sum(axis=1).astype(float)
    T    = len(tot)
    rhat = np.full(T, np.nan)
    for t in range(window, T - window):
        if tot[t + window] > 0 and tot[t - window] > 0:
            rhat[t] = (np.log(tot[t+window]) - np.log(tot[t-window])) / (2*window)
    return rhat


def epidemic_speed(R_mat, distances, gen_time_pmf):
    tot = R_mat.sum()
    if tot < 1e-15:
        return {"mean_distance": 0.0, "mean_gen_time": 0.0, "speed": 0.0}
    d_bar = float(np.sum(distances * R_mat) / tot)
    g_bar = float(np.sum(np.arange(len(gen_time_pmf)) * gen_time_pmf))
    return {"mean_distance": d_bar, "mean_gen_time": g_bar,
            "speed": d_bar / g_bar if g_bar > 0 else 0.0}


def minimum_control_effort(R_mat, costs=None):
    """Homogeneous and greedy per-location control effort to bring ρ(R) ≤ 1.

    u_homogeneous = 1 - 1/ρ is the exact uniform reduction. u_heterogeneous is a
    GREEDY heuristic: locations are ranked once by outward-elasticity/cost and each
    infector row is reduced (via bisection) in that fixed order until ρ ≤ 1.  It returns
    a FEASIBLE control (ρ ≤ 1 is reached) but total_effort_hetero is an upper bound, not
    a guaranteed minimum (the ranking is not recomputed after each reduction).
    """
    rho = R_system(R_mat);  N = R_mat.shape[0]
    if costs is None:
        costs = np.ones(N)
    costs     = np.asarray(costs, float)
    safe_cost = np.where(costs > 0, costs, np.inf)  # zero-cost → deprioritise in ranking only
    u_homog = max(0.0, 1.0 - 1.0/rho) if rho > 0 else 0.0
    se      = sensitivity_elasticity(R_mat)
    order   = np.argsort(se["elasticity"].sum(axis=1) / safe_cost)[::-1]
    u_het   = np.zeros(N)
    R_cur   = R_mat.copy()
    for idx in order:
        if R_system(R_cur) <= 1.0:
            break
        lo, hi = 0.0, 1.0
        for _ in range(30):
            mid   = (lo + hi) / 2.0
            R_tst = R_cur.copy();  R_tst[idx, :] *= (1.0 - mid)
            if R_system(R_tst) <= 1.0:
                hi = mid
            else:
                lo = mid
        u_het[idx] = hi
        R_cur[idx, :] *= (1.0 - u_het[idx])
    return {"u_homogeneous": u_homog, "u_heterogeneous": u_het,
            "total_effort_homog":  u_homog * costs.sum(),
            "total_effort_hetero": float((u_het * costs).sum()),
            "priority_order": order}


def _compute_power_mean_spectrum(R_out_series, alpha_arr):
    """Compute the power-mean spectrum X(α,t) for an array of α values.

    X(α,t) = Σ_j ω_j(α) R^j_out(t),  ω_j(α) = (R^j_out)^α / Σ_k (R^k_out)^α

    For α=0 all weights are equal (arithmetic mean of R^j_out).
    For α=1 this equals E(t) = Σ_j (R^j_out)² / Σ_k R^k_out.
    As α→∞ this approaches max_j R^j_out = A₁(1).

    Parameters
    ----------
    R_out_series : (T, N) array  — outward reproduction numbers per location
    alpha_arr    : (n_alpha,) array — α values to evaluate

    Returns
    -------
    X : (n_alpha, T) array  — X(α,t)
    """
    T, N = R_out_series.shape
    n_alpha = len(alpha_arr)
    X = np.full((n_alpha, T), np.nan)
    for ia, alpha in enumerate(alpha_arr):
        for t in range(T):
            rv = R_out_series[t]
            if rv.sum() < 1e-12:
                continue
            if alpha == 0:
                X[ia, t] = float(np.mean(rv))
            else:
                w = rv ** alpha
                denom = w.sum()
                if denom > 1e-300:
                    X[ia, t] = float(np.dot(w, rv) / denom)
    return X
