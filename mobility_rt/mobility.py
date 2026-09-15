"""mobility_rt.mobility — time-varying mobility matrices f^{jk}(t) (manuscript Appendix A)."""
import numpy as np
from mobility_rt.config import AWAY_FRACTION_CAP, DISTANCE_FLOOR_KM, DOW_SCALE, XI_CLIP


def generate_mobility(n_locations, T, populations, distances, meta,
                      day_variation_sd=0.15, seed=42, hub_attraction_power=0.5):
    """
    Time-varying row-stochastic mobility series f_jk of shape (T, N, N), where
    f_jk[t, j, k] is the probability that a resident of j is in location k on day t.

    Base matrix (Appendix A), with c_j = meta["commuting_fracs"][j] and
    δ = meta["decay_scale"]:
      base_f[j, j] = 1 - c_j
      w_jk = exp(-max(d_jk, DISTANCE_FLOOR_KM) / δ) · (N_k / N̄)^p      (k ≠ j)
      base_f[j, k] = c_j · w_jk / Σ_{k'≠j} w_jk'
    p = hub_attraction_power.  p = 0.5 is Appendix A's square-root population
    attractiveness; Scenario C uses p = 8, which concentrates satellite flows on the
    most populous location.

    Time variation, one factor shared by every location on day t (day 0 is a Monday):
      s(t) = DOW_SCALE[t mod 7] · clip(ξ_t, XI_CLIP),  ξ_t ~ LogNormal(0, day_variation_sd).
    Each row's away fraction becomes min(c_j s(t), AWAY_FRACTION_CAP) with its
    off-diagonal proportions kept, the diagonal takes the remainder, and the row is
    renormalised to sum to 1.

    The ξ_t draws depend only on seed and T, not on populations or distances, so two
    geographies generated with the same seed share their day-to-day variation.

    Returns (f_jk, base_f).
    """
    rng = np.random.default_rng(seed)
    N   = n_locations
    cf  = np.asarray(meta["commuting_fracs"], dtype=float).copy()
    ds  = meta["decay_scale"]
    pop_mean = populations.mean()

    # ── base off-diagonal weights w_jk = exp(-d_jk/δ) · (N_k/N̄)^p  [j ≠ k] ──
    d_safe = np.maximum(distances, DISTANCE_FLOOR_KM)
    W = np.exp(-d_safe / ds) * (populations / pop_mean)[np.newaxis, :] ** hub_attraction_power
    np.fill_diagonal(W, 0.0)

    # ── normalise rows to the commuting fraction ─────────────────────────────
    base_f = np.zeros((N, N))
    row_sums = W.sum(axis=1)
    for j in range(N):
        if row_sums[j] > 0:
            base_f[j] = cf[j] * W[j] / row_sums[j]
            base_f[j, j] = 1.0 - cf[j]
        else:
            base_f[j, j] = 1.0   # no reachable destinations (e.g. N=1): stay home, row sums to 1

    # ── time-varying f_jk(t) ────────────────────────────────────────────────
    dow_scale = np.asarray(DOW_SCALE, dtype=float)
    f_jk = np.zeros((T, N, N))
    for t in range(T):
        scale = dow_scale[t % 7] * float(
            np.clip(rng.lognormal(0.0, day_variation_sd), XI_CLIP[0], XI_CLIP[1]))

        for j in range(N):
            away_base   = cf[j]
            scaled_away = float(np.clip(away_base * scale, 0.0, AWAY_FRACTION_CAP))
            if away_base > 1e-12:
                ratio          = scaled_away / away_base
                f_jk[t, j]    = base_f[j] * ratio
                f_jk[t, j, j] = 0.0
            f_jk[t, j, j] = max(0.0, 1.0 - f_jk[t, j].sum())
            rs = f_jk[t, j].sum()
            if rs > 1e-15:
                f_jk[t, j] /= rs

    return f_jk, base_f
