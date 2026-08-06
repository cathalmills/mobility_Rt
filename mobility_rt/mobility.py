"""mobility_rt.mobility."""
import numpy as np


def generate_mobility(n_locations, T, populations, distances, node_types, meta,
                      day_variation_sd=0.15, seed=42,
                      commuting_frac_scale=1.0,
                      hub_attraction_power=0.5):
    """
    Build time-varying row-stochastic f_{jk}(t).

    Base matrix f_{jk} (before time variation) built from:
      - Home fraction:  f_{jj} = 1 - c_j
      - Off-diagonal weight:  w_{jk} = exp(-d_{jk}/δ) × (N_k/N̄)^p
        where p=hub_attraction_power (default 0.5 = sqrt, as in Simini 2012).
        Larger p concentrates flows at high-population hubs; p=1.0 is full
        gravity; p=1.5 creates a strong hub-and-spoke topology.
      - Normalised so off-diagonal row sums to c_j.

    Time variation :
      - Day-of-week scaling: Mon–Thu=1.00, Fri=0.95, Sat=0.90, Sun=0.75
      - Daily lognormal noise σ=day_variation_sd, clipped to [0.50, 1.80]
    """
    rng = np.random.default_rng(seed)
    N   = n_locations
    cf  = meta["commuting_fracs"].copy()
    if commuting_frac_scale != 1.0:
        cf = np.clip(cf * commuting_frac_scale, 0.0, 0.85)
    ds  = meta["decay_scale"]
    pop_mean = populations.mean()

    # ── base off-diagonal weights ──────────────────────────────────────────
    # w_{jk} = exp(-d_{jk}/δ) × (N_k / N̄)^p   [j ≠ k]
    # hub_attraction_power=0.5 
    # hub_attraction_power=1.5 → strong hub-and-spoke flow asymmetry
    d_safe = np.maximum(distances, 0.5)
    W = np.exp(-d_safe / ds) * (populations / pop_mean)[np.newaxis, :] ** hub_attraction_power
    np.fill_diagonal(W, 0.0)

    # ── normalise rows to commuting fraction ──────────────────────────────
    base_f = np.zeros((N, N))
    row_sums = W.sum(axis=1)
    for j in range(N):
        if row_sums[j] > 0:
            base_f[j] = cf[j] * W[j] / row_sums[j]
            base_f[j, j] = 1.0 - cf[j]
        else:
            base_f[j, j] = 1.0   # no reachable destinations (e.g. N=1): stay home, row sums to 1

    # ── time-varying f_jk(t) ─────────────────────────────────────────────
    # Day-of-week: Mon–Thu 1.0, Fri 0.95, Sat 0.90, Sun 0.75
    dow_scale = np.array([1.00, 1.00, 1.00, 1.00, 0.95, 0.90, 0.75])

    f_jk = np.zeros((T, N, N))
    for t in range(T):
        scale = dow_scale[t % 7] * float(
            np.clip(rng.lognormal(0.0, day_variation_sd), 0.50, 1.80))

        for j in range(N):
            away_base   = cf[j]
            scaled_away = float(np.clip(away_base * scale, 0.0, 0.95))
            if away_base > 1e-12:
                ratio          = scaled_away / away_base
                f_jk[t, j]    = base_f[j] * ratio
                f_jk[t, j, j] = 0.0
            f_jk[t, j, j] = max(0.0, 1.0 - f_jk[t, j].sum())
            rs = f_jk[t, j].sum()
            if rs > 1e-15:
                f_jk[t, j] /= rs

    return f_jk, base_f
