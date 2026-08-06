"""mobility_rt.estimators."""
import numpy as np
from mobility_rt.config import R_MIN_WINDOW_INCIDENCE, R_PRIOR_RATE, R_PRIOR_SHAPE
from mobility_rt.kernel import R_outward, R_system


def estimate_R_independent(incidence, gen_time, window=7):
    """Sliding-window Cori/EpiEstim independent reproduction-number estimator.

    EpiEstim-default prior Gamma(shape=1, scale=5) → prior mean 5; posterior mean over
    the window W = [t-window+1, t] is
      R̂_j(t) = (R_PRIOR_SHAPE + Σ_{s∈W} I_j(s)) / (R_PRIOR_RATE + Σ_{s∈W} Λ_j(s)),
    with Λ_j(s) = Σ_{a=1}^{max_s-1} p(a) I_j(s-a)  [total infectiousness at day s].
    gen_time is 0-indexed and lag a is weighted by gen_time[a] = p(a), matching the
    forward simulator's force of infection (E_pde[:,:,1:] @ infect_profile[1:]).
    R̂ is reported only where the windowed incidence ≥ R_MIN_WINDOW_INCIDENCE (else NaN),
    so the estimate is data-driven rather than the prior mean.
    """
    T, N   = incidence.shape
    max_s  = len(gen_time)
    R_est  = np.full((T, N), np.nan)
    pa, pb = R_PRIOR_SHAPE, R_PRIOR_RATE
    for j in range(N):
        for t in range(window, T):
            t0  = max(0, t - window + 1)
            obs = incidence[t0:t + 1, j].sum()
            lam = 0.0
            for tw in range(t0, t + 1):
                for a in range(1, max_s):
                    if tw - a >= 0:
                        lam += gen_time[a] * incidence[tw - a, j]
            if lam > 1e-4 and obs >= R_MIN_WINDOW_INCIDENCE:
                R_est[t, j] = (pa + obs) / (pb + lam)
    return R_est


def compute_naive_R_aggregate(incidence_total, gen_time_pmf, window=7):
    """Sliding-window independent estimator applied to aggregate (network-total) incidence.

    Treats the entire network as one homogeneous location, using the same Cori/EpiEstim
    prior and reporting threshold as estimate_R_independent:
      R̂(t) = (R_PRIOR_SHAPE + Σ_{s∈W} I(s)) / (R_PRIOR_RATE + Σ_{s∈W} Λ(s))
    where W = [t-window+1, t] and Λ(s) = Σ_{a=1}^{max_s-1} p(a) I(s-a).
    Lag a is weighted by gen_time_pmf[a] = p(a), matching the forward simulator's
    generation interval (so any residual bias is spatial, not a GT-alignment artefact).
    Returns R̂[T] (nan where the windowed incidence < R_MIN_WINDOW_INCIDENCE).
    """
    T      = len(incidence_total)
    max_s  = len(gen_time_pmf)
    pa, pb = R_PRIOR_SHAPE, R_PRIOR_RATE
    R_naive = np.full(T, np.nan)
    for t in range(window, T):
        t0    = max(0, t - window + 1)
        obs   = incidence_total[t0:t + 1].sum()
        denom = pb
        for tw in range(t0, t + 1):
            for a in range(1, max_s):
                if tw - a >= 0:
                    denom += gen_time_pmf[a] * incidence_total[tw - a]
        if obs >= R_MIN_WINDOW_INCIDENCE and denom > 1e-4:
            R_naive[t] = (pa + obs) / denom
    return R_naive


def _compute_naive_suite(sim, city, gen_time_pmf):
    """Compute every naive estimator for one simulation.

    Returns a dict with keys:
      rho_t, E_t, inc_tot, peak_t, T, N
      ind_iw, ind_pw, ind_Rw, ind_am   (Part 1: per-loc independent R, then aggregated)
      out_iw, out_pw, out_Rw, out_am   (Part 2: R^j_out, then aggregated)
      agg                               (Part 3: sliding-window R on total incidence)
    """
    inc    = sim["incidence"]          # (T, N)
    R_mats = sim["R_matrices"]
    T_len, N = inc.shape
    pops   = city[1].astype(float)
    w_pop  = pops / (pops.sum() + 1e-300)

    inc_tot = inc.sum(axis=1)
    rho_t   = np.array([R_system(R_mats[t]) for t in range(T_len)])
    R_out_t = np.array([R_outward(R_mats[t]) for t in range(T_len)])   # (T, N)

    # E(t) = X(1,t) = R_out-weighted mean R_out  [= Σ(R^j_out)² / Σ R^j_out]
    E_t = np.array([
        np.sum(R_out_t[t]**2) / (np.sum(R_out_t[t]) + 1e-300)
        for t in range(T_len)])

    # ── Part 3: aggregate independent R on total incidence ───────────────
    agg = compute_naive_R_aggregate(inc_tot, gen_time_pmf, window=7)

    # ── Per-location independent R estimates R_loc[t, j] ─────────────────
    # Same Cori/EpiEstim prior + windowed-incidence reporting threshold as
    # estimate_R_independent / compute_naive_R_aggregate.
    max_s = len(gen_time_pmf); pa, pb = R_PRIOR_SHAPE, R_PRIOR_RATE; window = 7
    R_loc = np.full((T_len, N), np.nan)
    for j in range(N):
        for t in range(window, T_len):
            t0    = max(0, t - window + 1)
            obs   = inc[t0:t + 1, j].sum()
            denom = pb
            for tw in range(t0, t + 1):
                for a in range(1, max_s):
                    if tw - a >= 0:
                        denom += gen_time_pmf[a] * inc[tw - a, j]
            if obs >= R_MIN_WINDOW_INCIDENCE and denom > 1e-4:
                R_loc[t, j] = (pa + obs) / denom

    # ── Part 1: per-location independent R, then aggregated ──────────────
    ind_iw = np.full(T_len, np.nan)
    ind_pw = np.full(T_len, np.nan)
    ind_Rw = np.full(T_len, np.nan)
    ind_am = np.full(T_len, np.nan)

    for t in range(T_len):
        vj = ~np.isnan(R_loc[t])
        if not vj.any():
            continue
        Rv = R_loc[t, vj]
        Iv = inc[t, vj]
        pv = pops[vj]
        # i) incidence-weighted
        if Iv.sum() > 1e-6:
            ind_iw[t] = float(np.sum(Iv * Rv) / Iv.sum())
        # ii) population-weighted
        wpv = pv / pv.sum()
        ind_pw[t] = float(np.sum(wpv * Rv))
        # iii) relative-transmissibility (self-)weighted: weight each independent
        #      estimate R^ind_j by its OWN relative transmissibility
        #      w_j = R^ind_j / Σ_k R^ind_k  →  R^{ind,Rw} = Σ_j (R^ind_j)² / Σ_j R^ind_j
        #      (manuscript §3.2.2 / Fig 4; Parag et al. [68]).
        if Rv.sum() > 1e-6:
            ind_Rw[t] = float(np.sum(Rv * Rv) / Rv.sum())
        # iv) arithmetic mean
        ind_am[t] = float(Rv.mean())

    # ── Part 2: R_out from model, aggregated ─────────────────────────────
    out_iw = np.full(T_len, np.nan)
    out_pw = np.zeros(T_len)
    for t in range(T_len):
        if inc_tot[t] > 1e-6:
            out_iw[t] = float(np.sum(inc[t] * R_out_t[t]) / inc_tot[t])
        out_pw[t] = float(np.sum(w_pop * R_out_t[t]))

    out_Rw = E_t.copy()            # R_out-weighted mean R_out = E(t) by definition
    out_am = R_out_t.mean(axis=1)  # arithmetic mean

    return {
        "rho_t":   rho_t,
        "E_t":     E_t,
        "inc_tot": inc_tot,
        "peak_t":  int(inc_tot.argmax()),
        "T":       T_len,
        "N":       N,
        "ind_iw":  ind_iw,
        "ind_pw":  ind_pw,
        "ind_Rw":  ind_Rw,
        "ind_am":  ind_am,
        "out_iw":  out_iw,
        "out_pw":  out_pw,
        "out_Rw":  out_Rw,
        "out_am":  out_am,
        "agg":     agg,
    }
