"""mobility_rt.transmissibility."""
import numpy as np
from scipy.optimize import brentq


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

    if abs(rho_val - 1.0) < 1e-8:
        return 0.0
    lo, hi = (-2.0, 0.0) if rho_val < 1 else (0.0, 2.0)
    if f(lo) * f(hi) >= 0:
        return np.nan          # no root in the bracket (|r| would exceed 2 per day)
    # brentq raises only if it fails to converge; that must surface, not become NaN.
    return float(brentq(f, lo, hi, xtol=1e-8, maxiter=200))


def risk_averse_R(R_out):
    """Spatial risk-averse reproduction number E(t) (manuscript Eq 42).

      E(t) = Σ_j (R^j_out)² / Σ_j R^j_out,

    the α = 1 member of the Lehmer-mean spectrum X(α, t) (Eq 41).  R_out is (N,) for
    one day, giving a float, or (T, N), giving (T,).  NaN where Σ_j R^j_out = 0.
    """
    R_out = np.asarray(R_out, dtype=float)
    num = (R_out ** 2).sum(axis=-1)
    den = R_out.sum(axis=-1)
    if R_out.ndim == 1:
        return float(num / den) if den > 0 else np.nan
    return np.divide(num, den, out=np.full(den.shape, np.nan), where=den > 0)


def within_fraction(incidence_matrix):
    """Share of new infections that are within-location, per location and overall.

    incidence_matrix[t, k, j] is the number of new infections on day t in residents
    of j caused by residents of k.  Returns
      pi_j[t, j] = E^{jj}(t) / Σ_k E^{kj}(t)       (infections in residents of j
                                                   caused by residents of j)
      pi_bar[t]  = Σ_j E^{jj}(t) / Σ_{k,j} E^{kj}(t)
    with NaN wherever the denominator is zero (e.g. day 0, which has no transmission).
    """
    M = np.asarray(incidence_matrix, dtype=float)
    diag = np.diagonal(M, axis1=1, axis2=2)
    col = M.sum(axis=1)
    pi_j = np.divide(diag, col, out=np.full(diag.shape, np.nan), where=col > 0)
    total = M.sum(axis=(1, 2))
    trace = np.trace(M, axis1=1, axis2=2)
    pi_bar = np.divide(trace, total, out=np.full(total.shape, np.nan), where=total > 0)
    return pi_j, pi_bar


def lehmer_mean_spectrum(R_out_series, alpha_arr):
    """Lehmer-mean spectrum X(α,t) of the outward reproduction numbers (Eq 41).

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
