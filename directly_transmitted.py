#!/usr/bin/env python3
"""
Mobility-Informed and Mechanism-Led Renewal Equations
— Directly Transmitted Diseases (Section 3.1) —

Notation (Diekmann convention): rows=infectors k, cols=infectees j.
  K_{kj}(t,a_E) = Σ_l f^{jl}(t) S_j(t) f^{kl}(t) λ^{kl}_E(t,a_E)   [Eq 9]
  R_{kj}(t)     = Σ_{a_E} K_{kj}(t,a_E)                              [Eq 12]
  R^k_out(t)    = Σ_j R_{kj}(t)   [row sums]                         [Eq 13]
  R^j_in(t)     = Σ_k R_{kj}(t)   [col sums]                         [Eq 23]
  R^l_meeting(t)= meeting-location reproduction number                [Sec 3.1.4]
  R(t)          = ρ(R(t))                                             [Eq 20]
  σ(t)          = ρ((R+Rᵀ)/2) = ‖R‖₂                                 [Eq 21]

Mobility uses empirically-grounded approach (not gravity model):
  Scenario A — Dense urban  (Wesolowski 2015, Tizzoni 2014, Azman 2014)
  Scenario B — Zambia-like sparse national (Wesolowski 2021 eLife)
  Both:  exponential distance decay, node-type heterogeneity, hub attraction,
         day-of-week scaling, lognormal daily noise.
"""

import numpy as np
from scipy.stats import gamma as gamma_dist
from scipy.spatial.distance import cdist
from scipy.optimize import brentq
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
import warnings
warnings.filterwarnings("ignore")

np.set_printoptions(precision=4, suppress=True)


# ══════════════════════════════════════════════════════════════════════════════
# 0.  PARAMETERS
# ══════════════════════════════════════════════════════════════════════════════

COVID_PARAMS = {
    "gen_time_mean":          5.5,    # days  [Hart et al. 2022 PLOS CB]
    "gen_time_sd":            1.8,
    "max_gen_time":           25,
    "base_contact_rate":      13.0,   # contacts/day [POLYMOD Mossong 2008]
    "prob_transmission_peak": 0.035,
    "R0_target":              2.5,
    # Two-component infectiousness profiles: within-location (household) contacts
    # have a shorter, earlier-peaking generation time than between-location
    # (workplace/community) contacts [Fraser 2011; Cereda et al. 2020].
    # The mixture of the two shapes generates genuine spatio-temporal variation
    # in g_{kj}(t, a_E) as the within/between balance shifts across location
    # pairs and across days of the week.
    "within_gt_mean":  4.0,   # days — household contact GT [Cereda 2020 short serial]
    "within_gt_sd":    1.5,
    "between_gt_mean": 7.0,   # days — community/workplace GT [longer tail]
    "between_gt_sd":   2.5,
}


# ══════════════════════════════════════════════════════════════════════════════
# 1.  GENERATION TIME PMF: For comparison and independent fitting
# ══════════════════════════════════════════════════════════════════════════════

def discretise_gamma(mean, sd, max_days):
    shape = (mean / sd) ** 2
    scale = sd ** 2 / mean
    pmf = np.array([
        gamma_dist.cdf(d+1, a=shape, scale=scale) -
        gamma_dist.cdf(d,   a=shape, scale=scale)
        for d in range(max_days)
    ])
    return pmf / pmf.sum()


# ══════════════════════════════════════════════════════════════════════════════
# 2.  CITY GEOMETRY  (empirically-grounded scenarios)
# ══════════════════════════════════════════════════════════════════════════════

def haversine_km(lat1, lon1, lat2, lon2):
    """Great-circle distance in km."""
    R = 6371.0
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = (np.sin(dlat/2)**2
         + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon/2)**2)
    return 2 * R * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def generate_city(n_locations=10, scenario="lagos", seed=42):
    """
    Generate synthetic city with empirically-grounded structure.

    scenario="lagos"  — dense urban megacity (Lagos-inspired).
        N locations, tight spatial clustering (~40 km span),
        node types: core/dense/suburban/peripheral,
        commuting fractions 18–40% (Wesolowski 2015, Tizzoni 2014).

    scenario="zambia" — sparse national network (Zambia-inspired).
        N locations spread over ~600 km,
        node types: capital/peri-capital/urban-industrial/semi-urban/rural/remote-rural,
        commuting fractions 1.5–8% (Wesolowski 2021 eLife).

    Returns: coords [N,2] (km), populations [N], distances [N,N] (km),
             node_types [N], node_metadata dict.
    """
    rng = np.random.default_rng(seed)
    N   = n_locations

    if scenario == "lagos":
        # Type assignment: ~20% core, 30% dense, 30% suburban, 20% peripheral
        type_seq = (["core"]     * max(1, int(0.20 * N)) +
                    ["dense"]    * max(1, int(0.30 * N)) +
                    ["suburban"] * max(1, int(0.30 * N)) +
                    ["peripheral"] * max(1, N - int(0.20*N) - int(0.30*N) - int(0.30*N)))
        type_seq = type_seq[:N]
        rng.shuffle(type_seq)
        node_types = type_seq

        # Spatial: tight urban spread, ~40 km radius
        radii  = rng.exponential(8.0, N)
        angles = rng.uniform(0, 2*np.pi, N)
        coords = np.column_stack([radii * np.cos(angles),
                                  radii * np.sin(angles)])

        # Population: cores largest, peripherals smallest
        type_pop = {"core": 900_000, "dense": 700_000,
                    "suburban": 400_000, "peripheral": 200_000}
        pop_raw = np.array([type_pop[t] for t in node_types], dtype=float)
        pop_raw *= np.exp(rng.normal(0, 0.25, N))

        # Commuting fractions per type (Wesolowski 2015, Tizzoni 2014)
        cf_map = {"core": 0.40, "dense": 0.35, "suburban": 0.28, "peripheral": 0.18}

        # Distance decay scale: 7 km (most intra-city trips < 10 km)
        decay_scale = 7.0

    elif scenario == "zambia":
        # Types: 1 capital, 1 peri-capital, 1 urban-industrial, 1 semi-urban,
        #        rest split rural/remote
        n_remote = max(1, N // 5)
        n_rural  = max(1, N // 3)
        n_urban  = N - 4 - n_rural - n_remote
        type_seq = (["capital"] + ["peri-capital"] + ["urban-industrial"] +
                    ["semi-urban"] * max(1, n_urban) +
                    ["rural"] * n_rural +
                    ["remote-rural"] * n_remote)
        type_seq = type_seq[:N]
        while len(type_seq) < N:
            type_seq.append("rural")
        node_types = type_seq

        # Spatial: spread over ~600 km
        radii  = rng.exponential(120.0, N)
        radii[0] = 0.0   # capital at centre
        angles = rng.uniform(0, 2*np.pi, N)
        coords = np.column_stack([radii * np.cos(angles),
                                  radii * np.sin(angles)])

        # Population: capital largest
        type_pop = {"capital": 3_000_000, "peri-capital": 1_800_000,
                    "urban-industrial": 2_500_000, "semi-urban": 1_500_000,
                    "rural": 900_000, "remote-rural": 600_000}
        pop_raw = np.array([type_pop[t] for t in node_types], dtype=float)
        pop_raw *= np.exp(rng.normal(0, 0.20, N))

        # Commuting fractions (Wesolowski 2021 eLife)
        cf_map = {"capital": 0.06, "peri-capital": 0.08, "urban-industrial": 0.05,
                  "semi-urban": 0.04, "rural": 0.025, "remote-rural": 0.015}

        # Distance decay scale: 200 km (Wesolowski 2021: exponential best fit)
        decay_scale = 200.0

    else:
        raise ValueError(f"Unknown scenario: {scenario!r}. Use 'lagos' or 'zambia'.")

    populations = np.round(pop_raw).astype(float)
    distances   = cdist(coords, coords, metric="euclidean")
    commuting_fracs = np.array([cf_map[t] for t in node_types])

    meta = {
        "node_types":       node_types,
        "commuting_fracs":  commuting_fracs,
        "decay_scale":      decay_scale,
        "scenario":         scenario,
        "cf_map":           cf_map,
    }
    return coords, populations, distances, node_types, meta


# ══════════════════════════════════════════════════════════════════════════════
# 3.  MOBILITY  f_{jk}(t)  — empirically-grounded
# ══════════════════════════════════════════════════════════════════════════════

def generate_mobility(n_locations, T, populations, distances, node_types, meta,
                      day_variation_sd=0.15, seed=42,
                      hub_mult_scale=1.0, commuting_frac_scale=1.0):
    """
    Build time-varying row-stochastic f_{jk}(t) using the empirical approach
    of Wesolowski et al. 2015/2021 and Azman et al. 2014.

    Base matrix construction:
      1. Diagonal = commuting_fracs[j] (home fraction = 1 - commuting_frac)
      2. Off-diagonal weights:
           base = exp(-d_{jk} / decay_scale)            [exponential decay]
           × trip-type multiplier (urban-urban > urban-rural > rural-rural)
           × hub attraction (capital/core nodes)
           × adjacency bonus (for zambia: nodes within 0.5×max_dist)
           × lognormal asymmetry perturbation (σ≈0.15 at construction time)
      3. Normalise off-diagonal row to commuting_frac[j]

    Time variation:
      Day-of-week scaling: Mon–Thu=1.00, Fri=0.90, Sat=0.60, Sun=0.50
      Daily lognormal noise on away-fraction: σ=day_variation_sd
    """
    rng    = np.random.default_rng(seed)
    N      = n_locations
    types  = node_types
    cf     = meta["commuting_fracs"].copy()        # 1 - home fraction
    if commuting_frac_scale != 1.0:
        cf = np.clip(cf * commuting_frac_scale, 0.0, 0.85)
    ds     = meta["decay_scale"]
    scen   = meta["scenario"]

    # ── define type groupings ──────────────────────────────────────────────
    if scen == "lagos":
        urban_strong = {"core", "dense"}
        urban_weak   = {"suburban"}
        hubs         = {"core"}
        hub_mult     = 2.5 * hub_mult_scale
        urban_urban_mult = 4.0
        urban_rural_mult = 2.0
    else:  # zambia
        urban_strong = {"capital", "peri-capital", "urban-industrial"}
        urban_weak   = {"semi-urban"}
        hubs         = {"capital"}
        secondary_hubs = {"urban-industrial"}
        hub_mult      = 4.0
        sec_hub_mult  = 2.0
        urban_urban_mult = 2.5
        urban_rural_mult = 1.5

    # ── compute adjacency (nodes within 40% of max dist are "adjacent") ───
    max_dist   = distances.max()
    adj_thresh = 0.40 * max_dist if scen == "zambia" else 0.35 * max_dist

    # ── build base off-diagonal weight matrix ─────────────────────────────
    W = np.zeros((N, N))
    for j in range(N):
        for k in range(N):
            if j == k:
                continue
            d = max(distances[j, k], 0.5)
            w = np.exp(-d / ds)

            # trip-type multiplier
            tj, tk = types[j], types[k]
            if tj in urban_strong and tk in urban_strong:
                w *= urban_urban_mult
            elif tj in urban_strong | urban_weak or tk in urban_strong | urban_weak:
                w *= urban_rural_mult

            # hub attraction (destination k is a hub)
            if tk in hubs:
                w *= hub_mult
            if scen == "zambia" and tk in secondary_hubs:
                w *= sec_hub_mult

            # population attractiveness (mild sqrt scaling, not full gravity)
            w *= np.sqrt(populations[k] / populations.mean())

            # adjacency bonus for zambia
            if scen == "zambia" and d < adj_thresh:
                w *= 3.0

            # construction-time lognormal asymmetry (Wesolowski: real data is asymmetric)
            w *= float(rng.lognormal(0.0, 0.12))

            W[j, k] = w

    # ── normalise to get base_f ───────────────────────────────────────────
    base_f = np.zeros((N, N))
    for j in range(N):
        row_w = W[j].sum()
        if row_w > 0:
            base_f[j] = cf[j] * W[j] / row_w      # away fraction distributed
        base_f[j, j] = 1.0 - cf[j]                 # home fraction

    # ── time-varying f_jk(t) ─────────────────────────────────────────────
    # Day-of-week: Mon–Thu 1.0, Fri 0.9, Sat 0.6, Sun 0.5
    dow_scale = np.array([1.00, 1.00, 1.00, 1.00, 0.90, 0.60, 0.50])

    f_jk = np.zeros((T, N, N))
    for t in range(T):
        scale = dow_scale[t % 7] * float(
            np.clip(rng.lognormal(0.0, day_variation_sd), 0.50, 1.80))

        for j in range(N):
            away_base   = cf[j]
            scaled_away = float(np.clip(away_base * scale, 0.0, 0.95))
            if away_base > 1e-12:
                ratio       = scaled_away / away_base
                f_jk[t, j] = base_f[j] * ratio
                f_jk[t, j, j] = 0.0
            f_jk[t, j, j] = max(0.0, 1.0 - f_jk[t, j].sum())
            # renormalise (numerical safety)
            rs = f_jk[t, j].sum()
            if rs > 1e-15:
                f_jk[t, j] /= rs

    return f_jk, base_f


# ══════════════════════════════════════════════════════════════════════════════
# 4.  VECTORISED KERNEL BASE
# ══════════════════════════════════════════════════════════════════════════════

def _kernel_base(f_t, populations, lw, lb):
    """
    Precompute base_K[k,j] and its within/between split for one time-slice.

    Two-component decomposition:
      bK_within[k,j] = lw * f[j,k] * f[k,k] / N_eff[k]
          (infector k and infectee j both at home location k)
      bK_between[k,j] = lb * Σ_{l≠k} f[j,l] * f[k,l] / N_eff[l]
          (infector k and infectee j meet away from k)
      base_K[k,j] = bK_within[k,j] + bK_between[k,j]

    D[j,k] = Σ_l f[j,l]*f[k,l]/N_eff[l]  (all meeting locations)
    D^T[k,j] - within[k,j] = Σ_{l≠k} f[j,l]*f[k,l]/N_eff[l]  (away meetings)

    Returns: base_K, bK_within, bK_between, N_eff, inv_Neff
    """
    N_eff    = f_t.T @ populations
    inv_Neff = np.where(N_eff > 0, 1.0 / N_eff, 0.0)
    A        = f_t * inv_Neff[np.newaxis, :]        # A[j,l] = f[j,l]/N_eff[l]
    D        = A @ f_t.T                             # D[j,k] = Σ_l A[j,l]*f[k,l]
    within   = f_t.T * (f_t.diagonal() * inv_Neff)[:, np.newaxis]  # [k,j]
    bK_within  = lw * within
    bK_between = lb * np.maximum(D.T - within, 0.0)  # away-meeting component ≥ 0
    base_K     = bK_within + bK_between
    return base_K, bK_within, bK_between, N_eff, inv_Neff


# ══════════════════════════════════════════════════════════════════════════════
# 5.  REPRODUCTION NUMBERS
# ══════════════════════════════════════════════════════════════════════════════

def compute_R_matrix(f_t, S, populations, prob_peak, infect_profile, lw, lb):
    """R_{kj}(t) = prob_peak * Σ_a p(a) * S[j] * base_K[k,j].

    R values are independent of the within/between GT split because both PMFs
    sum to 1; the split only affects the temporal *shape* of K_{kj}(t,a_E).
    """
    base_K, _, _, _, _ = _kernel_base(f_t, populations, lw, lb)
    return prob_peak * infect_profile.sum() * S[np.newaxis, :] * base_K


def R_outward(R_mat):
    return R_mat.sum(axis=1)

def R_inward(R_mat):
    return R_mat.sum(axis=0)

def R_system(R_mat):
    ev = np.linalg.eigvals(R_mat)
    return float(np.max(np.abs(ev)))


def compute_R_meeting(f_t, S, populations, prob_peak, infect_profile, lw, lb):
    """
    R^l_meeting(t) = S^l_eff * κ_eff(l) * prob_peak * Σ_a p(a) / N^l_eff
    κ_eff(l) = lw*f[l,l] + lb*Σ_{k≠l} f[k,l]
    """
    N        = f_t.shape[0]
    _, _, _, N_eff, inv_Neff = _kernel_base(f_t, populations, lw, lb)
    S_eff    = f_t.T @ S
    sum_w    = infect_profile.sum()
    R_meet   = np.zeros(N)
    for l in range(N):
        kappa = lw * f_t[l, l] + lb * (f_t[:, l].sum() - f_t[l, l])
        R_meet[l] = prob_peak * sum_w * S_eff[l] * kappa * inv_Neff[l]
    return R_meet


def spectral_analysis(R_mat):
    """ρ, λ₂, mixing ratio s=|λ₂|/ρ, right eigvec w, left eigvec v."""
    ev_r, evec_r = np.linalg.eig(R_mat)
    ev_l, evec_l = np.linalg.eig(R_mat.T)

    idx_r = np.argsort(np.abs(ev_r))[::-1]
    idx_l = np.argsort(np.abs(ev_l))[::-1]

    rho     = float(np.abs(ev_r[idx_r[0]]))
    lambda2 = float(np.abs(ev_r[idx_r[1]])) if len(idx_r) > 1 else 0.0

    w = np.abs(evec_r[:, idx_r[0]]).real.astype(float);  w /= (w.sum() + 1e-300)
    v = np.abs(evec_l[:, idx_l[0]]).real.astype(float);  v /= (v.sum() + 1e-300)

    mix = lambda2 / rho if rho > 0 else 0.0
    cv  = (R_mat.sum(axis=1).std() / R_mat.sum(axis=1).mean()
           if R_mat.sum(axis=1).mean() > 0 else 0.0)

    return {"rho": rho, "lambda2": lambda2, "mixing_ratio": mix,
            "damping_ratio": rho / lambda2 if lambda2 > 0 else np.inf,
            "right_eigvec": w,        # right eigvec of R_mat (infector=rows); = reprod. values in Diekmann
            "left_eigvec": v,         # right eigvec of R_mat.T = right eigvec of Diekmann K^D
            "stable_distribution": v, # alias: right eigvec of Diekmann K^D = stable spatial distribution
            "reprod_value": w,        # alias: left eigvec of Diekmann K^D = reproductive value vector
            "cv_row_sums": cv, "eigenvalues": ev_r[idx_r]}


# ══════════════════════════════════════════════════════════════════════════════
# 6.  GENERATION TIME DISTRIBUTIONS
# ══════════════════════════════════════════════════════════════════════════════

def compute_generation_times(f_t, S, populations, prob_peak, w_within, w_between,
                              max_days, lw, lb):
    """
    All GT distributions at one time point, using two infectiousness profiles.

    K_{kj}(t,a_E) = S_j * prob_peak * [bKw[k,j]*w_within[a_E] + bKb[k,j]*w_between[a_E]]

    Because bKw and bKb vary by (k,j) pair, the pairwise GT:
      g_{kj}(t,a_E) = [bKw[k,j]*w_within[a_E] + bKb[k,j]*w_between[a_E]]
                      / (bKw[k,j] + bKb[k,j])
    is a *mixture* whose weight depends on the within/between balance for that
    specific infector–infectee pair.  This generates genuine spatio-temporal
    variation:
      • Peripheral–peripheral pairs (high home fraction) → closer to w_within
      • Central–distant pairs (low home fraction) → closer to w_between
      • Weekends (high f_{kk}) → shift toward w_within; weekdays → w_between
    Since both w_within and w_between are PMFs (sum to 1), R_kj = Σ_a K_kj(t,a_E)
    = prob_peak * S_j * base_K[k,j] regardless of the within/between split.
    The split only affects the temporal *shape* of the kernel (when transmission
    occurs), not the total number of secondary cases.
    """
    N    = len(populations)
    base_K, bKw, bKb, N_eff, inv_Neff = _kernel_base(f_t, populations, lw, lb)
    S_eff = f_t.T @ S

    # K[a, k, j] = S[j] * prob_peak * (bKw[k,j]*w_within[a] + bKb[k,j]*w_between[a])
    # Vectorised: K[a] = prob_peak * S[np.newaxis,:] * (bKw*w_within[a] + bKb*w_between[a])
    K_series = np.zeros((max_days, N, N))
    for a in range(max_days):
        K_series[a] = prob_peak * S[np.newaxis, :] * (
            bKw * w_within[a] + bKb * w_between[a])

    R_mat = K_series.sum(axis=0)  # same as prob_peak * S[j] * base_K (since both PMFs sum to 1)

    # Pairwise: g_{kj}(a) varies across (k,j) pairs — genuine spatial variation
    g_pw = np.zeros_like(K_series)
    for k in range(N):
        for j in range(N):
            if R_mat[k, j] > 1e-15:
                g_pw[:, k, j] = K_series[:, k, j] / R_mat[k, j]

    # Outward: g^k_out(a) — mixture weighted by all infectees j of infector k
    K_out = K_series.sum(axis=2);  R_out = R_mat.sum(axis=1)
    g_out = np.zeros_like(K_out)
    for k in range(N):
        if R_out[k] > 1e-15:
            g_out[:, k] = K_out[:, k] / R_out[k]

    # Inward: g^j_in(a) — mixture weighted by all infectors k of infectee j
    K_in = K_series.sum(axis=1);  R_in = R_mat.sum(axis=0)
    g_in = np.zeros_like(K_in)
    for j in range(N):
        if R_in[j] > 1e-15:
            g_in[:, j] = K_in[:, j] / R_in[j]

    # Network-level: g_net(a) — overall mixture (varies with day-of-week!)
    K_tot = K_series.sum(axis=(1, 2));  R_tot = R_mat.sum()
    g_net = K_tot / R_tot if R_tot > 1e-15 else np.zeros(max_days)

    # Meeting-location: g^l_meeting(a) — mixture at location l
    K_meet = np.zeros((max_days, N))
    for l in range(N):
        kappa_w = lw * f_t[l, l]
        kappa_b = lb * max(f_t[:, l].sum() - f_t[l, l], 0.0)
        for a in range(max_days):
            K_meet[a, l] = (prob_peak * (kappa_w * w_within[a] + kappa_b * w_between[a])
                            * S_eff[l] * inv_Neff[l])
    R_meet = K_meet.sum(axis=0)
    g_meet = np.zeros_like(K_meet)
    for l in range(N):
        if R_meet[l] > 1e-15:
            g_meet[:, l] = K_meet[:, l] / R_meet[l]

    return {"g_pairwise": g_pw, "g_outward": g_out, "g_inward": g_in,
            "g_meeting": g_meet, "g_network": g_net,
            "K_series": K_series, "R_matrix": R_mat, "R_meeting": R_meet}


def eigenvector_weighted_gt(K_series, R_mat):
    """
    g̃(a_E) = Σ_{k,j} [v_k w_j / (v^T w)] K_{kj}(a_E) / ρ  [sensitivity-weighted GT]
    """
    spec = spectral_analysis(R_mat)
    rho  = spec["rho"]
    if rho < 1e-15:
        return np.zeros(K_series.shape[0])
    v, w = spec["left_eigvec"], spec["right_eigvec"]
    vw   = max(float(v @ w), 1e-15)
    S_mat  = np.outer(v, w) / vw
    g_t    = np.einsum("kj,akj->a", S_mat, K_series) / rho
    tot    = g_t.sum()
    return g_t / tot if tot > 1e-15 else g_t


# ══════════════════════════════════════════════════════════════════════════════
# 7.  EFFECTIVE POPULATIONS AT MEETING LOCATIONS
# ══════════════════════════════════════════════════════════════════════════════

def compute_effective_populations_series(f_jk, S_series, incidence,
                                          gen_time_pmf, max_days):
    """S^l_eff(t) = Σ_j f_{jl}(t) S_j(t),   I^l_eff(t) = Σ_k f_{kl}(t) Λ_k(t)."""
    T, N = incidence.shape
    S_eff = np.zeros((T, N))
    I_eff = np.zeros((T, N))
    for t in range(T):
        S_eff[t] = f_jk[t].T @ S_series[t]
        lam = np.zeros(N)
        for s in range(1, min(max_days, t + 1)):
            lam += gen_time_pmf[s] * incidence[t - s]
        I_eff[t] = f_jk[t].T @ lam
    return S_eff, I_eff


# ══════════════════════════════════════════════════════════════════════════════
# 8.  FORWARD SIMULATION — UPWIND PDE + DEMOGRAPHICS
# ══════════════════════════════════════════════════════════════════════════════

def simulate_epidemic_pde(T, n_locations, populations, f_jk_series,
                           prob_peak, infect_profile, max_days, R0_target,
                           initial_infections, lambda_within, lambda_between,
                           w_within, w_between,
                           birth_rate=0.00003, death_rate=0.00003,
                           stochastic=True, susceptible_depletion=True, seed=42):
    """
    Upwind finite-difference PDE  ∂E/∂t + ∂E/∂a_E = 0  [Eq 2].
    Boundary condition: E_j(t,0) = Σ_k Σ_{a_E} K_{kj}(t,a_E) E_k(t,a_E)  [Eq 4].

    Two-component force of infection:
      K_{kj}(t,a_E) = S_j(t)*prob_peak*(bKw[k,j]*w_within[a_E] + bKb[k,j]*w_between[a_E])

    The split gives realistic GT variation while keeping R values unchanged
    (since both PMFs sum to 1, R_{kj} = prob_peak * S_j * base_K as before).
    infect_profile is used only for calibration of the R0 scaling factor.
    Vectorised O(N²) per step.
    """
    rng = np.random.default_rng(seed)
    N   = n_locations
    S   = populations.copy().astype(float)

    S_series         = np.zeros((T, N))
    E_pde            = np.zeros((T, N, max_days))
    incidence_matrix = np.zeros((T, N, N))
    R_matrices       = np.zeros((T, N, N))
    R_meeting_series = np.zeros((T, N))

    E_pde[0, :, 0]   = initial_infections
    S               -= initial_infections
    S_series[0]      = S.copy()

    # calibrate
    f0     = f_jk_series[0]
    R0_mat = compute_R_matrix(f0, populations, populations, prob_peak,
                               infect_profile, lambda_within, lambda_between)
    rho0   = R_system(R0_mat)
    scale  = R0_target / rho0 if rho0 > 0 else 1.0
    lw, lb = lambda_within * scale, lambda_between * scale

    R_check = compute_R_matrix(f0, populations, populations, prob_peak,
                                infect_profile, lw, lb)
    print(f"  Calibration: ρ₀={rho0:.4f}  scale={scale:.4f}  "
          f"ρ_check={R_system(R_check):.4f}  (target={R0_target})")

    R_matrices[0]       = R_check
    R_meeting_series[0] = compute_R_meeting(f0, populations, populations,
                                             prob_peak, infect_profile, lw, lb)

    for t in range(1, T):
        f_t = f_jk_series[min(t, len(f_jk_series) - 1)]

        # PDE upwind step
        E_pde[t, :, 1:] = E_pde[t - 1, :, :-1]

        # Two-component force of infection
        base_K, bKw, bKb, _, _ = _kernel_base(f_t, populations, lw, lb)
        # Weighted infectious pressure per infector k under each GT profile
        wE_w = prob_peak * (E_pde[t, :, 1:] @ w_within[1:])   # within-location GT
        wE_b = prob_peak * (E_pde[t, :, 1:] @ w_between[1:])  # between-location GT
        # New infections in j: S_j * Σ_k [bKw[k,j]*wE_w[k] + bKb[k,j]*wE_b[k]]
        contrib_kj = bKw * wE_w[:, np.newaxis] + bKb * wE_b[:, np.newaxis]  # [k,j]
        expected_j = np.maximum(S * contrib_kj.sum(axis=0), 0.0)

        if stochastic:
            new_j = rng.poisson(np.minimum(expected_j, 1e7)).astype(float)
        else:
            new_j = expected_j.copy()
        if susceptible_depletion:
            new_j = np.minimum(new_j, S)

        # Distribute pairwise incidence proportional to contrib_kj
        col_sum = contrib_kj.sum(axis=0)
        for j in range(N):
            if col_sum[j] > 1e-15 and new_j[j] > 0:
                incidence_matrix[t, :, j] = new_j[j] * contrib_kj[:, j] / col_sum[j]

        E_pde[t, :, 0] = new_j

        R_mat_t              = compute_R_matrix(f_t, S, populations, prob_peak,
                                                infect_profile, lw, lb)
        R_matrices[t]        = R_mat_t
        R_meeting_series[t]  = compute_R_meeting(f_t, S, populations, prob_peak,
                                                  infect_profile, lw, lb)

        if susceptible_depletion:
            S = np.maximum(S - new_j
                           + birth_rate * populations
                           - death_rate * S, 0.0)
        S_series[t] = S.copy()

    return {"incidence":            E_pde[:, :, 0],
            "incidence_matrix":     incidence_matrix,
            "R_matrices":           R_matrices,
            "R_meeting_series":     R_meeting_series,
            "susceptibles":         S_series,
            "lambda_within_scaled": lw,
            "lambda_between_scaled": lb,
            "E_pde_state":          E_pde}


# ══════════════════════════════════════════════════════════════════════════════
# 9.  INDEPENDENT R̂(t) 
# ══════════════════════════════════════════════════════════════════════════════

def estimate_R_independent(incidence, gen_time, window=7):
    """Prior Gamma(1,5) → mean=0.2; suppressed when incidence→0."""
    T, N   = incidence.shape
    max_s  = len(gen_time)
    R_est  = np.full((T, N), np.nan)
    pa, pb = 1.0, 5.0
    for j in range(N):
        for t in range(max_s, T):
            t0  = max(max_s, t - window + 1)
            obs = incidence[t0:t+1, j].sum()
            lam = sum(gen_time[s] * incidence[tw - s, j]
                      for tw in range(t0, t+1)
                      for s  in range(1, min(max_s, tw)))
            if lam > 1e-4 and obs >= 1:
                R_est[t, j] = (pa + obs) / (pb + lam)
    return R_est


# ══════════════════════════════════════════════════════════════════════════════
# 10. SOURCE-SINK AND DECOMPOSITION
# ══════════════════════════════════════════════════════════════════════════════

def source_sink_analysis(R_mat):
    R_out = R_outward(R_mat);  R_in = R_inward(R_mat);  net = R_out - R_in
    return {"R_outward": R_out, "R_inward": R_in, "net_export": net,
            "sources": np.where(net > 0)[0], "sinks": np.where(net < 0)[0]}

def within_between_decomposition(R_mat):
    d = np.trace(R_mat);  s = R_mat.sum()
    pi = d / s if s > 0 else 0.0
    return {"pi_within": pi, "pi_between": 1.0 - pi}


# ══════════════════════════════════════════════════════════════════════════════
# 11. SENSITIVITY AND ELASTICITY
# ══════════════════════════════════════════════════════════════════════════════

def sensitivity_elasticity(R_mat):
    """S_{kj}=∂ρ/∂R_{kj}=v_k w_j/(v^T w),  E_{kj}=(R_{kj}/ρ)*S_{kj}."""
    spec = spectral_analysis(R_mat)
    rho  = spec["rho"];  v = spec["left_eigvec"];  w = spec["right_eigvec"]
    vw   = max(float(v @ w), 1e-15)
    S_m  = np.outer(v, w) / vw
    E_m  = (R_mat / rho) * S_m if rho > 1e-15 else np.zeros_like(R_mat)
    return {"sensitivity": S_m, "elasticity": E_m, "rho": rho,
            "condition_number": float(np.linalg.cond(R_mat))}


# ══════════════════════════════════════════════════════════════════════════════
# 12. REACTIVITY AND TRANSIENT AMPLIFICATION  [Eq 21]
# ══════════════════════════════════════════════════════════════════════════════

def reactivity(R_mat):
    """σ(t) = ρ((R+Rᵀ)/2) = ‖R‖₂."""
    sigma = float(np.linalg.eigvalsh((R_mat + R_mat.T) / 2).max())
    rho   = R_system(R_mat)
    return {"sigma": sigma, "rho": rho,
            "transient_possible": sigma > 1 and rho < 1,
            "amplification_ratio": sigma / rho if rho > 0 else np.inf}

def amplification_envelope(R_mat, n_max=30):
    """A(n)=‖R^n‖₂=σ_max(R^n).  Captures non-normal transient growth."""
    rho = R_system(R_mat)
    A   = np.zeros(n_max + 1)
    Rn  = np.eye(R_mat.shape[0])
    for n in range(n_max + 1):
        A[n] = float(np.linalg.svd(Rn, compute_uv=False)[0])
        Rn   = Rn @ R_mat
    return {"A": A, "rho_n": rho ** np.arange(n_max + 1),
            "n": np.arange(n_max + 1), "rho": rho}

def convergence_cosine(incidence, R_matrices):
    """cos(p(t), w(t)) where p=observed spatial distrib, w=dominant eigvec."""
    T = incidence.shape[0]
    cs = np.full(T, np.nan)
    for t in range(T):
        tot = incidence[t].sum()
        if tot < 1:
            continue
        p = incidence[t] / tot
        # stable distribution = right eigvec of Diekmann K^D = left eigvec of R_mat
        w = spectral_analysis(R_matrices[t])["stable_distribution"]
        d = np.linalg.norm(p) * np.linalg.norm(w)
        if d > 1e-15:
            cs[t] = float(p @ w) / d
    return cs


# ══════════════════════════════════════════════════════════════════════════════
# 13. EULER-LOTKA  r(t)
# ══════════════════════════════════════════════════════════════════════════════

def euler_lotka_r(rho_val, g_tilde):
    """Solve 1 = R(t) Σ_a g̃(a) e^{-ra} for r."""
    if rho_val <= 0 or g_tilde.sum() < 1e-15:
        return np.nan
    days = np.arange(len(g_tilde))
    def f(r):
        return rho_val * float(np.sum(g_tilde * np.exp(-r * days))) - 1.0
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


# ══════════════════════════════════════════════════════════════════════════════
# 14. EPIDEMIC SPATIAL SPEED
# ══════════════════════════════════════════════════════════════════════════════

def epidemic_speed(R_mat, distances, gen_time_pmf):
    tot = R_mat.sum()
    if tot < 1e-15:
        return {"mean_distance": 0.0, "mean_gen_time": 0.0, "speed": 0.0}
    d_bar = float(np.sum(distances * R_mat) / tot)
    g_bar = float(np.sum(np.arange(len(gen_time_pmf)) * gen_time_pmf))
    return {"mean_distance": d_bar, "mean_gen_time": g_bar,
            "speed": d_bar / g_bar if g_bar > 0 else 0.0}


# ══════════════════════════════════════════════════════════════════════════════
# 15. CONTROLLABILITY
# ══════════════════════════════════════════════════════════════════════════════

def minimum_control_effort(R_mat, costs=None):
    rho = R_system(R_mat);  N = R_mat.shape[0]
    if costs is None:
        costs = np.ones(N)
    u_homog = max(0.0, 1.0 - 1.0/rho) if rho > 0 else 0.0
    se      = sensitivity_elasticity(R_mat)
    order   = np.argsort(se["elasticity"].sum(axis=1) / costs)[::-1]
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


# ══════════════════════════════════════════════════════════════════════════════
# 16. HELPER
# ══════════════════════════════════════════════════════════════════════════════

def _contiguous_segments(bool_arr):
    segs, in_s, start = [], False, 0
    for i, v in enumerate(bool_arr):
        if v and not in_s:
            start, in_s = i, True
        elif not v and in_s:
            segs.append((start, i)); in_s = False
    if in_s:
        segs.append((start, len(bool_arr)))
    return segs


# ══════════════════════════════════════════════════════════════════════════════
# 17. PUBLICATION STYLE + FIGURE 2 — SIMULATED DATA AND OVERALL OUTPUTS
# ══════════════════════════════════════════════════════════════════════════════

# Okabe-Ito colorblind-safe palette [Okabe & Ito 2008; Wong 2011 Nature Methods]
OKABE_ITO = ['#E69F00', '#56B4E9', '#009E73', '#F0E442',
             '#0072B2', '#D55E00', '#CC79A7', '#000000']


def _set_pub_style():
    """Publication-quality matplotlib defaults for Nature-family journals.

    Nature guidelines: sans-serif font, 7 pt minimum, no top/right spines,
    figures ≤ 183 mm wide (double-column) or ≤ 89 mm (single-column).
    """
    plt.rcParams.update({
        "font.family":          "sans-serif",
        "font.sans-serif":      ["Helvetica Neue", "Helvetica", "Arial",
                                 "DejaVu Sans"],
        "font.size":            8,
        "axes.titlesize":       8,
        "axes.labelsize":       8,
        "xtick.labelsize":      7,
        "ytick.labelsize":      7,
        "legend.fontsize":      7,
        "legend.frameon":       False,
        "legend.handlelength":  1.5,
        "axes.spines.top":      False,
        "axes.spines.right":    False,
        "axes.linewidth":       0.7,
        "xtick.major.width":    0.6,
        "ytick.major.width":    0.6,
        "xtick.major.size":     3,
        "ytick.major.size":     3,
        "xtick.minor.size":     1.5,
        "ytick.minor.size":     1.5,
        "lines.linewidth":      0.9,
        "patch.linewidth":      0.6,
        "figure.dpi":           150,
        "savefig.dpi":          300,
        "savefig.bbox":         "tight",
        "savefig.pad_inches":   0.05,
        "image.cmap":           "viridis",
        "pdf.fonttype":         42,   # embed fonts properly in PDFs
        "ps.fonttype":          42,
    })


def _panel_label(ax, letter, x=-0.14, y=1.04):
    """Add bold panel letter (a, b, c …) in top-left corner of axes."""
    ax.text(x, y, letter, transform=ax.transAxes,
            fontsize=10, fontweight="bold", va="top", ha="left")

def _panel_label_3d(ax, letter):
    """Add bold panel letter to 3D axes via text2D (transAxes coords)."""
    ax.text2D(-0.06, 1.06, letter, transform=ax.transAxes,
              fontsize=10, fontweight="bold", va="top", ha="left")

def _bar3d_Rkj(ax, R_mat, day_label):
    """3D bar chart of pairwise R_kj matrix. Infectors on x-axis, infectees on y-axis."""
    N = R_mat.shape[0]
    xpos = np.repeat(np.arange(N), N)   # infector k
    ypos = np.tile(np.arange(N), N)     # infectee j
    dz   = np.maximum(R_mat.flatten(), 0.0)
    colors = [OKABE_ITO[k % len(OKABE_ITO)] for k in xpos]
    ax.bar3d(xpos - 0.38, ypos - 0.38, np.zeros(N * N),
             0.75, 0.75, dz, color=colors, alpha=0.85, shade=True, linewidth=0.0)
    # Clean panes: no fill, subtle edges, no grid
    for pane in [ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane]:
        pane.fill = False
        pane.set_edgecolor("#cccccc")
    ax.xaxis._axinfo["grid"]["color"] = (0, 0, 0, 0)
    ax.yaxis._axinfo["grid"]["color"] = (0, 0, 0, 0)
    ax.zaxis._axinfo["grid"]["color"] = (0, 0, 0, 0)
    ax.set_xticks(range(N)); ax.set_yticks(range(N))
    ax.set_xticklabels([f"L{i+1}" for i in range(N)], fontsize=4)
    ax.set_yticklabels([f"L{i+1}" for i in range(N)], fontsize=4)
    ax.set_xlabel("Infector $k$", fontsize=6, labelpad=2)
    ax.set_ylabel("Infectee $j$", fontsize=6, labelpad=2)
    ax.zaxis.set_rotate_label(False)
    ax.zaxis.set_tick_params(labelsize=5, pad=0)
    ax.view_init(elev=28, azim=-55)
    ax.tick_params(axis='z', labelsize=5, pad=0)
    ax.set_title(f"$R_{{kj}}$ — {day_label}", fontsize=7, pad=18, fontweight="bold")


def _bar3d_inc(ax, inc_mat, day_label):
    """3D bar chart of pairwise new infections E_{kj}. Infectors on x-axis, infectees on y-axis."""
    N = inc_mat.shape[0]
    xpos = np.repeat(np.arange(N), N)
    ypos = np.tile(np.arange(N), N)
    dz   = np.maximum(inc_mat.flatten(), 0.0)
    colors = [OKABE_ITO[k % len(OKABE_ITO)] for k in xpos]
    ax.bar3d(xpos - 0.38, ypos - 0.38, np.zeros(N * N),
             0.75, 0.75, dz, color=colors, alpha=0.85, shade=True, linewidth=0.0)
    for pane in [ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane]:
        pane.fill = False
        pane.set_edgecolor("#cccccc")
    ax.xaxis._axinfo["grid"]["color"] = (0, 0, 0, 0)
    ax.yaxis._axinfo["grid"]["color"] = (0, 0, 0, 0)
    ax.zaxis._axinfo["grid"]["color"] = (0, 0, 0, 0)
    ax.set_xticks(range(N)); ax.set_yticks(range(N))
    ax.set_xticklabels([f"L{i+1}" for i in range(N)], fontsize=4)
    ax.set_yticklabels([f"L{i+1}" for i in range(N)], fontsize=4)
    ax.set_xlabel("Infector $k$", fontsize=6, labelpad=2)
    ax.set_ylabel("Infectee $j$", fontsize=6, labelpad=2)
    ax.zaxis.set_rotate_label(False)
    ax.zaxis.set_tick_params(labelsize=5, pad=0)
    ax.view_init(elev=28, azim=-55)
    ax.tick_params(axis='z', labelsize=5, pad=0)
    ax.set_title(f"$E_{{kj}}$ — {day_label}", fontsize=7, pad=18, fontweight="bold")


def plot_fig2(sim, city_data, f_jk, gen_time_pmf, max_days, scenario_name,
              save_prefix="fig"):
    """Figure 2: Simulated epidemic — mobility inputs and epidemic outputs.

    Panels:
      a  Mean mobility matrix f̄_{jk}
      b  Home fraction f_{jj}(t) over time (shows weekly commuting cycles)
      c  Incidence E_j(t,0) by location and time
      d  Effective susceptibles S^{eff}_m(t) at meeting locations
      e  System R(t) = ρ(R(t)) with total incidence on twin axis
      f  Column elasticity Σ_j ε_{jk}(t) as heatmap (infector × time)
    """
    inc      = sim["incidence"]
    R_mats   = sim["R_matrices"]
    S_series = sim["susceptibles"]
    coords, pops, dists, node_types, meta = city_data
    T, N = inc.shape

    S_eff_s, _ = compute_effective_populations_series(
        f_jk, S_series, inc, gen_time_pmf, max_days)
    R_sys = np.array([R_system(R_mats[t]) for t in range(T)])
    loc   = [f"L{i+1}" for i in range(N)]

    fig = plt.figure(figsize=(7.2, 4.8))
    gs  = gridspec.GridSpec(2, 3, hspace=0.58, wspace=0.52,
                            left=0.09, right=0.96, top=0.97, bottom=0.10)

    # ── a: mean mobility matrix ────────────────────────────────────────────
    ax = fig.add_subplot(gs[0, 0])
    im = ax.imshow(f_jk.mean(axis=0), cmap="Blues", aspect="auto")
    ax.set_xticks(range(N)); ax.set_xticklabels(loc, fontsize=5, rotation=45)
    ax.set_yticks(range(N)); ax.set_yticklabels(loc, fontsize=5)
    ax.set_xlabel("Activity location $k$"); ax.set_ylabel("Residence $j$")
    cb_a = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb_a.ax.set_title("$\\bar{f}_{jk}$", fontsize=6, pad=3)
    _panel_label(ax, "a")

    # ── b: home fraction f_{jj}(t) over time ─────────────────────────────
    ax = fig.add_subplot(gs[1, 0])
    diag_f = np.array([[f_jk[t, j, j] for j in range(N)]
                        for t in range(T)]).T   # shape (N, T)
    im = ax.imshow(diag_f, cmap="RdYlGn", aspect="auto",
                   vmin=0.3, vmax=1.0, origin="upper")
    ax.set_yticks(range(N)); ax.set_yticklabels(loc, fontsize=5)
    ax.set_xlabel("Day $t$"); ax.set_ylabel("Location")
    cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.ax.set_title("$f_{jj}(t)$", fontsize=6, pad=3)
    _panel_label(ax, "d")

    # ── b: incidence E_j(t,0) heatmap ─────────────────────────────────────
    ax = fig.add_subplot(gs[0, 1])
    im = ax.imshow(inc.T, cmap="YlOrRd", aspect="auto", origin="upper")
    ax.set_yticks(range(N)); ax.set_yticklabels(loc, fontsize=5)
    ax.set_xlabel("Day $t$"); ax.set_ylabel("Residence")
    cb_c = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb_c.ax.set_title("Incidence", fontsize=6, pad=3)
    _panel_label(ax, "b")

    # ── e: effective susceptibles S^l_eff(t) at meeting locations ──────────
    ax = fig.add_subplot(gs[1, 1])
    im = ax.imshow(S_eff_s.T / 1e3, cmap="Blues", aspect="auto", origin="upper")
    ax.set_yticks(range(N)); ax.set_yticklabels(loc, fontsize=5)
    ax.set_xlabel("Day $t$"); ax.set_ylabel("Activity location $l$")
    cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.ax.set_title("$S^l_{\\rm eff}$ ($\\times 10^3$)", fontsize=6, pad=3)
    _panel_label(ax, "e")

    # ── c: system R(t) with total incidence on twin axis ──────────────────
    ax  = fig.add_subplot(gs[0, 2])
    ax2 = ax.twinx()
    ax2.spines["right"].set_visible(True)
    ax2.spines["top"].set_visible(False)
    vld = R_sys > 0
    ax.plot(np.where(vld)[0], R_sys[vld], color=OKABE_ITO[4], lw=0.9,
            label="$\\mathcal{R}(t)$")
    ax.axhline(1.0, color="0.55", ls="--", lw=0.8)
    total_inc = inc.sum(axis=1)
    ax2.fill_between(range(T), total_inc / 1e3, alpha=0.22, color=OKABE_ITO[1])
    ax2.plot(total_inc / 1e3, color=OKABE_ITO[1], lw=1.0)
    ax.set_ylabel("$\\mathcal{R}(t)$", color=OKABE_ITO[4])
    ax2.set_ylabel("Incidence ($\\times 10^3$)", color=OKABE_ITO[1])
    ax.set_xlabel("Day $t$")
    ax.set_ylim(0, max(3.5, float(R_sys[vld].max()) * 1.1) if vld.any() else 3.5)
    ax.tick_params(axis="y", labelcolor=OKABE_ITO[4])
    ax2.tick_params(axis="y", labelcolor=OKABE_ITO[1])
    _panel_label(ax, "c")

    # ── f: column elasticity Σ_j ε_{jk}(t) as heatmap ────────────────────
    ax = fig.add_subplot(gs[1, 2])
    elas = np.zeros((T, N))
    for t in range(T):
        elas[t] = sensitivity_elasticity(R_mats[t])["elasticity"].sum(axis=1)
    vmax_e = np.percentile(elas[elas > 0], 97) if (elas > 0).any() else 1.0
    im = ax.imshow(elas.T, cmap="YlOrRd", aspect="auto", origin="upper",
                   vmin=0, vmax=vmax_e)
    ax.set_yticks(range(N)); ax.set_yticklabels(loc, fontsize=5)
    ax.set_xlabel("Day $t$"); ax.set_ylabel("Infector $k$")
    cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.ax.set_title("$\\sum_j \\varepsilon_{jk}$", fontsize=6, pad=3)
    ax.text(0.02, 0.02,
            "$\\varepsilon_{kj} = (R_{kj}/\\mathcal{R})\\,\\partial\\mathcal{R}/\\partial R_{kj}$",
            transform=ax.transAxes, fontsize=5, va="bottom", color="0.35")
    _panel_label(ax, "f")

    plt.savefig(f"{save_prefix}_02_overview.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved {save_prefix}_02_overview.png")


# ══════════════════════════════════════════════════════════════════════════════
# 18. FIGURE 3 — TAXONOMY OF R AND GT
# ══════════════════════════════════════════════════════════════════════════════

def plot_fig3(sim, city_data, R_independent, gt_snaps, w_within, w_between,
              scenario_name, save_prefix="fig"):
    """Figure 3: Taxonomy of R types and generation time distributions.

    Panels (3×3 grid, row 2 via GridSpecFromSubplotSpec for equal g/h widths):
      a  GT distributions at peak — g_{kk} within, g^k_out outward, g^j_in inward
         for hub vs peripheral locations
      b  R^k_out(t) heatmap (infector × time) — plasma colormap
      c  R^j_in(t) heatmap (infectee × time) — viridis colormap
      d  3D bar chart of R_{kj} at epidemic peak
      e  3D bar chart of pairwise new infections E_{kj} at epidemic peak
      f  Source–sink decomposition at peak (row 1, col 2)
      g  Bias: R_indep (dashed) vs R^j_in (solid) for hub/mid/peripheral
      h  R^k_out vs R_indep comparison for hub/mid/peripheral
    Letters follow strict left-to-right, top-to-bottom reading order.
    Panels g and h are equal width (each half of the bottom row).
    """
    inc    = sim["incidence"]
    inc_mat= sim["incidence_matrix"]   # shape (T, N, N)
    R_mats = sim["R_matrices"]
    coords, pops, dists, node_types, meta = city_data
    T, N = inc.shape

    R_out_s = np.array([R_outward(R_mats[t]) for t in range(T)])
    R_in_s  = np.array([R_inward(R_mats[t])  for t in range(T)])
    peak    = int(inc.sum(axis=1).argmax())
    early   = max(1, peak // 3)
    late    = min(T - 1, peak + 30)
    loc     = [f"L{i+1}" for i in range(N)]

    dc    = np.linalg.norm(coords - coords.mean(axis=0), axis=1)
    i_hub = int(np.argmin(dc))
    i_per = int(np.argmax(dc))
    i_mid = int(np.argsort(dc)[N // 2])

    show_locs = [i_hub, i_mid, i_per]
    show_lbls = [f"hub L{i_hub+1}", f"mid L{i_mid+1}", f"periph L{i_per+1}"]

    fig = plt.figure(figsize=(7.2, 9.0))
    gs  = gridspec.GridSpec(3, 3, hspace=0.72, wspace=0.60,
                            left=0.09, right=0.97, top=0.97, bottom=0.04)

    # ── a: GT distributions at peak ───────────────────────────────────────
    ax   = fig.add_subplot(gs[0, 0])
    days = np.arange(len(w_within))
    gt_p = gt_snaps["peak"]
    g_pw  = gt_p["g_pairwise"]
    g_out = gt_p["g_outward"]
    g_in  = gt_p["g_inward"]

    for i, short, col in [(i_hub, "hub", OKABE_ITO[0]),
                           (i_per, "periph", OKABE_ITO[5])]:
        g_kk = g_pw[:, i, i]
        if g_kk.sum() > 0.5:
            ax.plot(days, g_kk,        color=col, lw=1.2,
                    label=f"$g_{{kk}}$ {short}")
        if g_out[:, i].sum() > 0.5:
            ax.plot(days, g_out[:, i], color=col, lw=0.9, ls="--",
                    label=f"$g^k_{{\\rm out}}$ {short}")
        if g_in[:, i].sum() > 0.5:
            ax.plot(days, g_in[:, i],  color=col, lw=0.9, ls=":",
                    label=f"$g^j_{{\\rm in}}$ {short}")

    for i, col in [(i_hub, OKABE_ITO[0]), (i_per, OKABE_ITO[5])]:
        g_kk = g_pw[:, i, i]
        if g_kk.sum() > 0.5:
            mu = float(np.sum(days * g_kk))
            ax.axvline(mu, color=col, lw=0.5, ls=":", alpha=0.5)

    ax.set_xlabel("Infection age $a_E$ (days)")
    ax.set_ylabel("Probability")
    ax.legend(fontsize=5.0, ncol=2, borderpad=0.3, labelspacing=0.15,
              handlelength=1.2)
    ax.text(0.97, 0.97, "Dashed: $g^k_{\\rm out}$, dotted: $g^j_{\\rm in}$",
            transform=ax.transAxes, fontsize=4.5, ha="right", va="top",
            color="0.4", style="italic")
    _panel_label(ax, "a")

    # ── b: R^k_out(t) heatmap ─────────────────────────────────────────────
    ax = fig.add_subplot(gs[0, 1])
    pos_out = R_out_s[R_out_s > 0]
    vmin_out = np.percentile(pos_out, 2)  if pos_out.size else 0.0
    vmax_out = np.percentile(pos_out, 97) if pos_out.size else 3.0
    im = ax.imshow(R_out_s.T, cmap="plasma", aspect="auto", origin="upper",
                   vmin=vmin_out, vmax=vmax_out)
    ax.set_yticks(range(N)); ax.set_yticklabels(loc, fontsize=5)
    ax.set_xlabel("Day $t$"); ax.set_ylabel("Infector $k$")
    cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.ax.set_title("$R^k_{\\rm out}$", fontsize=6, pad=3)
    _panel_label(ax, "b")

    # ── c: R^j_in(t) heatmap ──────────────────────────────────────────────
    ax = fig.add_subplot(gs[0, 2])
    pos_in = R_in_s[R_in_s > 0]
    vmin_in = np.percentile(pos_in, 2)  if pos_in.size else 0.0
    vmax_in = np.percentile(pos_in, 97) if pos_in.size else 3.0
    im = ax.imshow(R_in_s.T, cmap="viridis", aspect="auto", origin="upper",
                   vmin=vmin_in, vmax=vmax_in)
    ax.set_yticks(range(N)); ax.set_yticklabels(loc, fontsize=5)
    ax.set_xlabel("Day $t$"); ax.set_ylabel("Infectee $j$")
    cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.ax.set_title("$R^j_{\\rm in}$", fontsize=6, pad=3)
    _panel_label(ax, "c")

    # ── d: 3D R_kj at epidemic peak ────────────────────────────────────────
    ax_3d = fig.add_subplot(gs[1, 0], projection="3d")
    _bar3d_Rkj(ax_3d, R_mats[peak], f"peak (day {peak})")
    ax_3d.text2D(-0.08, 1.05, "d", transform=ax_3d.transAxes,
                 fontsize=10, fontweight="bold", va="top", ha="left")

    # ── e: 3D pairwise new infections E_{kj} at epidemic peak ─────────────
    ax_3d2 = fig.add_subplot(gs[1, 1], projection="3d")
    _bar3d_inc(ax_3d2, inc_mat[peak], f"peak (day {peak})")
    ax_3d2.text2D(-0.08, 1.05, "e", transform=ax_3d2.transAxes,
                  fontsize=10, fontweight="bold", va="top", ha="left")

    # ── f: Source–sink decomposition at peak — row 1 col 2 ────────────────
    ax = fig.add_subplot(gs[1, 2])
    ss  = source_sink_analysis(R_mats[peak])
    net = ss["net_export"]
    bc  = [OKABE_ITO[5] if x > 0 else OKABE_ITO[4] for x in net]
    ax.barh(range(N), net, color=bc, height=0.65, edgecolor="none")
    ax.axvline(0, color="k", lw=0.8)
    ax.set_yticks(range(N)); ax.set_yticklabels(loc, fontsize=6)
    ax.set_xlabel("Net export  $R^k_{\\rm out} - R^j_{\\rm in}$")
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(facecolor=OKABE_ITO[5], label="Source"),
                        Patch(facecolor=OKABE_ITO[4], label="Sink")],
              fontsize=6, loc="upper right", borderpad=0.3)
    _panel_label(ax, "f")

    # ── g and h: equal-width halves of the bottom row ─────────────────────
    gs_bottom = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[2, :],
                                                  wspace=0.55)

    # ── g: Bias — R_indep vs R^j_in ───────────────────────────────────────
    ax = fig.add_subplot(gs_bottom[0, 0])
    for j, lbl, col in zip(show_locs, show_lbls,
                            [OKABE_ITO[0], OKABE_ITO[2], OKABE_ITO[5]]):
        vi = ~np.isnan(R_independent[:, j])
        vm = R_in_s[:, j] > 0
        b  = vi & vm
        if b.sum() > 3:
            ax.plot(np.where(b)[0], R_independent[b, j], "--",
                    color=col, lw=0.9, alpha=0.75)
            ax.plot(np.where(b)[0], R_in_s[b, j], "-",
                    color=col, lw=0.9, label=lbl)
    ax.axhline(1, color="0.55", ls="--", lw=0.8)
    handles, labels_leg = ax.get_legend_handles_labels()
    handles += [Line2D([0],[0], color="0.4", lw=0.9, ls="-"),
                Line2D([0],[0], color="0.4", lw=0.9, ls="--")]
    labels_leg += ["$R^j_{\\rm in}$ (solid)",
                   "$R_{\\rm indep}$ (dashed)"]
    ax.legend(handles=handles, labels=labels_leg, fontsize=5.5, ncol=2,
              borderpad=0.3, labelspacing=0.15)
    ax.set_xlabel("Day $t$"); ax.set_ylabel("$R(t)$")
    _panel_label(ax, "g")

    # ── h: R^k_out vs R_indep — hub/mid/peripheral ────────────────────────
    ax = fig.add_subplot(gs_bottom[0, 1])
    R_out_s2 = np.array([R_outward(R_mats[t]) for t in range(T)])
    for j, lbl, col in zip(show_locs, show_lbls,
                            [OKABE_ITO[0], OKABE_ITO[2], OKABE_ITO[5]]):
        vi = ~np.isnan(R_independent[:, j])
        vo = R_out_s2[:, j] > 0
        b  = vi & vo
        if b.sum() > 3:
            ax.plot(np.where(b)[0], R_independent[b, j], "--",
                    color=col, lw=0.9, alpha=0.75)
            ax.plot(np.where(b)[0], R_out_s2[b, j], "-",
                    color=col, lw=0.9, label=lbl)
    ax.axhline(1, color="0.55", ls="--", lw=0.8)
    handles_h, labels_h = ax.get_legend_handles_labels()
    handles_h += [Line2D([0],[0], color="0.4", lw=0.9, ls="-"),
                  Line2D([0],[0], color="0.4", lw=0.9, ls="--")]
    labels_h  += ["$R^k_{\\rm out}$ (solid)", "$R_{\\rm indep}$ (dashed)"]
    ax.legend(handles=handles_h, labels=labels_h, fontsize=5.0, ncol=1,
              borderpad=0.3, labelspacing=0.15)
    ax.set_xlabel("Day $t$"); ax.set_ylabel("$R(t)$")
    _panel_label(ax, "h")

    plt.savefig(f"{save_prefix}_03_taxonomy.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved {save_prefix}_03_taxonomy.png")


# ══════════════════════════════════════════════════════════════════════════════
# 19. FIGURE 4 — SPECTRAL PROPERTIES AND TRANSIENT DYNAMICS
# ══════════════════════════════════════════════════════════════════════════════

def plot_fig4(sim, city_data, R_mat_alt_t0, scenario_name,
              w_within, w_between, max_days, save_prefix="fig"):
    """Figure 4: Spectral properties.

    Panels (2×2):
      a  Mixing ratio s(t) = |λ_2|/ρ over time with day-of-week overlay
      b  R(t) vs σ(t) over time, shading transient zone where σ>1 and R<1
      c  Amplification envelope A(n)=||R^n||_2 at early/peak/late phases
      d  Top 3 eigenvalue magnitudes |λ_1(t)|, |λ_2(t)|, |λ_3(t)| over time
    """
    inc    = sim["incidence"]
    R_mats = sim["R_matrices"]
    coords, pops, dists, node_types, meta = city_data
    T, N = inc.shape

    R_sys    = np.array([R_system(R_mats[t]) for t in range(T)])
    specs    = [spectral_analysis(R_mats[t]) for t in range(T)]
    mix_ts   = np.array([s["mixing_ratio"] for s in specs])
    sigma_ts = np.array([reactivity(R_mats[t])["sigma"] for t in range(T)])

    peak  = int(inc.sum(axis=1).argmax())
    early = max(1, peak // 3)
    late  = min(T - 1, peak + 30)

    fig = plt.figure(figsize=(7.2, 7.5))
    gs  = gridspec.GridSpec(3, 2, hspace=0.65, wspace=0.50,
                            left=0.09, right=0.97, top=0.95, bottom=0.07)

    # ── a: mixing ratio s(t) with day-of-week overlay ─────────────────────
    ax  = fig.add_subplot(gs[0, 0])
    ax2 = ax.twinx()
    ax2.spines["right"].set_visible(True); ax2.spines["top"].set_visible(False)
    vm = mix_ts > 0
    ax.plot(np.where(vm)[0], mix_ts[vm], color=OKABE_ITO[2], lw=0.9,
            label="$s(t)$", zorder=5)
    dow_scale   = np.array([1.00, 1.00, 1.00, 1.00, 0.90, 0.60, 0.50])
    dow_pattern = np.array([dow_scale[t % 7] for t in range(T)])
    ax2.plot(range(T), dow_pattern, color=OKABE_ITO[0], lw=0.6, ls=":",
             alpha=0.7, label="DoW scaling")
    ax2.set_ylabel("DoW scale", color=OKABE_ITO[0], fontsize=6)
    ax2.tick_params(axis="y", labelcolor=OKABE_ITO[0], labelsize=6)
    ax2.set_ylim(0, 1.5)
    ax.set_xlabel("Day $t$")
    ax.set_ylabel("$s(t)=|\\lambda_2|/\\mathcal{R}$")
    ax.set_ylim(0, 1.05)
    lines1, labs1 = ax.get_legend_handles_labels()
    lines2, labs2 = ax2.get_legend_handles_labels()
    ax.legend(lines1+lines2, labs1+labs2, fontsize=5.5, borderpad=0.3)
    ax.text(0.05, 0.12, "Oscillation driven by\nday-of-week mobility",
            transform=ax.transAxes, fontsize=5.0, color="0.4", style="italic")
    _panel_label(ax, "a")

    # ── b: R(t) vs σ(t) over time ─────────────────────────────────────────
    ax = fig.add_subplot(gs[0, 1])
    vR   = R_sys > 0
    vsig = sigma_ts > 0
    t_arr = np.arange(T)
    ax.plot(t_arr[vR],   R_sys[vR],      color=OKABE_ITO[4], lw=0.9,
            label="$\\mathcal{R}(t)$")
    ax.plot(t_arr[vsig], sigma_ts[vsig], color=OKABE_ITO[5], lw=0.9,
            label="$\\sigma(t)$")
    ax.axhline(1.0, color="0.55", ls="--", lw=0.8)
    transient_mask = (sigma_ts > 1) & (R_sys < 1)
    if transient_mask.any():
        ax.fill_between(t_arr, 1.0, sigma_ts,
                        where=transient_mask,
                        color=OKABE_ITO[0], alpha=0.25,
                        label="$\\sigma>1, \\mathcal{R}<1$\n(transient zone)")
    ax.set_xlabel("Day $t$"); ax.set_ylabel("Value")
    ax.legend(fontsize=5.5, borderpad=0.3, labelspacing=0.15)
    # Compute mean amplification ratio σ/ρ − 1 (%)
    _amp_arr  = np.where(R_sys > 0.1, sigma_ts / R_sys - 1.0, np.nan)
    _amp_mean = float(np.nanmean(_amp_arr)) * 100
    ax.text(0.97, 0.97,
            (f"$\\sigma \\geq \\mathcal{{R}}$ always;\n"
             f"$\\sigma/\\mathcal{{R}}-1 \\approx {_amp_mean:.2f}\\%$\n"
             "Near 0: bidirectional flows\n→ nearly symmetric $\\mathbf{{R}}$;\n"
             "directed flows needed for\nlarge gap"),
            transform=ax.transAxes, fontsize=4.5, ha="right", va="top",
            color="0.4", style="italic")
    _panel_label(ax, "b")

    # ── c: Amplification envelope A(n) at three epidemic phases ───────────
    ax = fig.add_subplot(gs[1, 0])
    phase_specs = [
        (early, "early",  OKABE_ITO[2]),
        (peak,  "peak",   OKABE_ITO[5]),
        (late,  "late",   OKABE_ITO[0]),
    ]
    for t_phase, phase_name, col in phase_specs:
        env = amplification_envelope(R_mats[t_phase], n_max=20)
        ax.plot(env["n"], env["A"] / (env["rho_n"] + 1e-300), color=col, lw=1.0,
                label=f"{phase_name} (day {t_phase})")
    ax.axhline(1.0, color="0.55", ls="--", lw=0.7, alpha=0.7, label="Normal matrix baseline")
    ax.set_xlabel("$n$ (generations)")
    ax.set_ylabel("$A(n)/\\rho^n$")
    ax.set_title("Transient amplification $A(n)/\\rho^n$", fontsize=6, pad=3)
    ax.legend(fontsize=5.5, borderpad=0.3, labelspacing=0.15)
    ax.text(0.97, 0.97,
            "$A(n)/\\rho^n > 1$:\\ntransient amplification\\n(non-normality)",
            transform=ax.transAxes, fontsize=4.8, ha="right", va="top",
            color="0.4", style="italic")
    _panel_label(ax, "c")

    # ── d: Top 3 eigenvalue magnitudes over time ──────────────────────────
    ax = fig.add_subplot(gs[1, 1])
    lam1 = np.array([float(np.abs(specs[t]["eigenvalues"][0])) for t in range(T)])
    lam2 = np.array([float(np.abs(specs[t]["eigenvalues"][1]))
                     if len(specs[t]["eigenvalues"]) > 1 else 0.0 for t in range(T)])
    lam3 = np.array([float(np.abs(specs[t]["eigenvalues"][2]))
                     if len(specs[t]["eigenvalues"]) > 2 else 0.0 for t in range(T)])
    t_arr = np.arange(T)
    ax.plot(t_arr, lam1, color=OKABE_ITO[4], lw=1.0, label="$|\\lambda_1(t)|$")
    ax.plot(t_arr, lam2, color=OKABE_ITO[2], lw=0.9, ls="--", label="$|\\lambda_2(t)|$")
    ax.plot(t_arr, lam3, color=OKABE_ITO[0], lw=0.8, ls=":", label="$|\\lambda_3(t)|$")
    ax.axvline(peak, color="0.55", ls="--", lw=0.8, alpha=0.7)
    ax.text(peak + 1, ax.get_ylim()[1] * 0.95 if ax.get_ylim()[1] > 0 else 0.1,
            f"peak\n(day {peak})", fontsize=5, color="0.4", va="top")
    ax.set_xlabel("Day $t$")
    ax.set_ylabel("Eigenvalue magnitude")
    ax.set_title("Top 3 eigenvalue magnitudes", fontsize=6, pad=3)
    ax.legend(fontsize=5.5, borderpad=0.3, labelspacing=0.15)
    _panel_label(ax, "d")

    # ── e: Within-fraction heatmap + overall π̄(t) overlay + per-loc bar ─────
    # π_j(t) = E_{jj}(t)/Σ_k E_{kj}(t)  [heatmap, per-location per-day]
    # Overall π(t) = Σ_j E_{jj}(t)/Σ_{k,j} E_{kj}(t)  [navy dash-dot overlay]
    # Right bar = time-averaged π̄_j  [per-location summary]
    gs_e = gridspec.GridSpecFromSubplotSpec(
        1, 2, subplot_spec=gs[2, :], width_ratios=[5, 1], wspace=0.08)
    ax   = fig.add_subplot(gs_e[0, 0])
    ax_r = fig.add_subplot(gs_e[0, 1])

    N_loc   = inc.shape[1]
    inc_mat = sim["incidence_matrix"]             # (T, N, N)
    col_sum = inc_mat.sum(axis=1)                 # Σ_k E_{kj}(t) → (T, N)
    diag    = np.array([inc_mat[t].diagonal() for t in range(T)])   # (T, N)
    pi_within  = np.where(col_sum > 0, diag / col_sum, np.nan)      # (T, N)
    # overall π(t) = trace / total across all (k,j) pairs
    total_mat  = inc_mat.sum(axis=(1, 2))                            # (T,)
    total_diag = np.array([inc_mat[t].trace() for t in range(T)])    # (T,)
    pi_overall = np.where(total_mat > 0, total_diag / total_mat, np.nan)  # (T,)

    im = ax.imshow(pi_within.T, aspect="auto", origin="upper",
                   cmap="RdYlGn", vmin=0, vmax=1, interpolation="nearest",
                   extent=[0, T, N_loc + 0.5, 0.5])
    ax.set_xlabel("Day $t$")
    ax.set_ylabel("Location $j$")
    ax.set_title(
        r"Within-fraction $\pi_j(t) = E_{jj}/\!\sum_k E_{kj}$"
        "  (green = local, red = imported)", fontsize=6.5, pad=4)
    ax.set_yticks(range(1, N_loc + 1))
    ax.set_yticklabels([f"L{i}" for i in range(1, N_loc + 1)], fontsize=6)
    ax.axvline(peak, color="k", lw=0.9, ls="--", alpha=0.6)
    ax.text(peak + 1, 0.75, f"peak\n(d{peak})", fontsize=5, color="k",
            va="top", transform=ax.get_xaxis_transform())
    # Overlay overall π(t) on twin-x axis (navy dash-dot)
    ax2 = ax.twinx()
    ax2.spines["right"].set_visible(True)
    ax2.spines["top"].set_visible(False)
    mask_o = ~np.isnan(pi_overall)
    ax2.plot(np.where(mask_o)[0], pi_overall[mask_o],
             color="#1a237e", lw=1.4, ls="-.", alpha=0.90,
             label=r"$\bar{\pi}(t)$ overall")
    ax2.set_ylim(0, 1.4)
    ax2.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax2.set_ylabel(r"Overall $\bar{\pi}(t)$", fontsize=6, color="#1a237e")
    ax2.tick_params(axis="y", labelcolor="#1a237e", labelsize=5)
    ax2.legend(fontsize=5.5, loc="upper right", borderpad=0.3)
    cbar = fig.colorbar(im, ax=ax, orientation="horizontal", pad=0.22,
                        fraction=0.04, aspect=40)
    cbar.set_label(r"$\pi_j(t)$  (per-location within-fraction)", fontsize=6)
    cbar.ax.tick_params(labelsize=5)
    _panel_label(ax, "e")

    # ── right panel: per-location time-averaged π̄_j bar chart ─────────────
    pi_j_avg       = np.nanmean(pi_within, axis=0)               # (N,)
    pi_overall_avg = float(np.nanmean(pi_overall[mask_o])) if mask_o.any() else 0.5
    bar_clrs = [plt.cm.RdYlGn(float(np.clip(v, 0, 1))) for v in pi_j_avg]
    ax_r.barh(range(1, N_loc + 1), pi_j_avg, color=bar_clrs,
              height=0.72, edgecolor="none")
    ax_r.axvline(pi_overall_avg, color="#1a237e", lw=1.2, ls="-.")
    ax_r.text(min(pi_overall_avg + 0.04, 0.98), N_loc + 0.6,
              f"all:{pi_overall_avg:.2f}", fontsize=4.5, color="#1a237e",
              va="top", ha="left")
    ax_r.set_xlim(0, 1.10)
    ax_r.set_xlabel(r"$\bar{\pi}_j$", fontsize=6)
    ax_r.set_yticks(range(1, N_loc + 1))
    ax_r.set_yticklabels([], fontsize=5)
    ax_r.set_title("Time-avg\n$\\bar{\\pi}_j$", fontsize=5.5, pad=3)
    ax_r.tick_params(labelsize=5)
    for j_idx, v in enumerate(pi_j_avg):
        ax_r.text(min(float(v) + 0.05, 1.02), j_idx + 1, f"{v:.2f}",
                  va="center", ha="left", fontsize=4.2, color="0.3")

    plt.savefig(f"{save_prefix}_04_spectral.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved {save_prefix}_04_spectral.png")


# ══════════════════════════════════════════════════════════════════════════════
# 20. FIGURE 5 — COUNTERFACTUAL COMMUTING
# ══════════════════════════════════════════════════════════════════════════════

def plot_fig5(sim_A, sim_B, city_A, city_B, f_A, f_B,
              R_t0_A, R_t0_B, save_prefix="fig"):
    """Figure 5: Two mobility settings compared — Dense urban vs Sparse national.

    Panels:
      a  Mean mobility matrix f̄_{jk} — Scenario B (sparse national)
      b  3D bar chart of E_kj at epidemic peak — Scenario B (sparse national)
      c  3D bar chart of R_kj at epidemic peak — Scenario B (sparse national)
      d  R(t) and σ(t) for both scenarios over time
      e  System R(t) comparison with normalised incidence on twin axis
      f  R^k_out over time for both scenarios
    """
    inc_A, inc_B = sim_A["incidence"], sim_B["incidence"]
    Rm_A,  Rm_B  = sim_A["R_matrices"], sim_B["R_matrices"]
    imat_B = sim_B.get("incidence_matrix", None)   # (T, N, N) if available
    coords_A, pops_A, dists_A, types_A, meta_A = city_A
    T, N  = inc_A.shape
    pk_B  = int(inc_B.sum(axis=1).argmax())
    loc   = [f"L{i+1}" for i in range(N)]
    col_A = OKABE_ITO[0]   # dense urban — orange
    col_B = OKABE_ITO[4]   # sparse national — blue

    R_sys_A  = np.array([R_system(Rm_A[t]) for t in range(T)])
    R_sys_B  = np.array([R_system(Rm_B[t]) for t in range(T)])
    sigma_A  = np.array([reactivity(Rm_A[t])["sigma"] for t in range(T)])
    sigma_B  = np.array([reactivity(Rm_B[t])["sigma"] for t in range(T)])

    fig = plt.figure(figsize=(7.2, 6.2))
    gs  = gridspec.GridSpec(3, 2, hspace=0.62, wspace=0.52,
                            left=0.10, right=0.96, top=0.96, bottom=0.08)

    # ── a: mean mobility matrix — Scenario B ──────────────────────────────
    ax = fig.add_subplot(gs[0, 0])
    im = ax.imshow(f_B.mean(axis=0), cmap="Oranges", aspect="auto")
    ax.set_xticks(range(N)); ax.set_xticklabels(loc, fontsize=5, rotation=45)
    ax.set_yticks(range(N)); ax.set_yticklabels(loc, fontsize=5)
    ax.set_xlabel("Activity location $k$"); ax.set_ylabel("Residence $j$")
    cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.ax.set_title("$\\bar{f}_{jk}$", fontsize=6, pad=3)
    ax.text(0.02, 0.98, "Sparse national\n(~5% commuting, ~600 km span)",
            transform=ax.transAxes, fontsize=5.5, ha="left", va="top",
            color=col_B, fontweight="bold")
    _panel_label(ax, "a")

    # ── b: 3D E_kj at peak — Scenario B ───────────────────────────────────
    ax_b = fig.add_subplot(gs[0, 1], projection="3d")
    if imat_B is not None:
        _bar3d_inc(ax_b, imat_B[pk_B], f"Sparse national — peak (day {pk_B})")
    else:
        ax_b.text(0.5, 0.5, 0.5, "incidence matrix\nnot available",
                  ha="center", va="center", fontsize=7, color="0.5")
    ax_b.text2D(-0.10, 1.05, "b", transform=ax_b.transAxes,
                fontsize=10, fontweight="bold", va="top", ha="left")

    # ── c: 3D R_kj at peak — Scenario B ───────────────────────────────────
    ax_c = fig.add_subplot(gs[1, 0], projection="3d")
    _bar3d_Rkj(ax_c, Rm_B[pk_B], f"Sparse national — peak (day {pk_B})")
    ax_c.text2D(-0.10, 1.05, "c", transform=ax_c.transAxes,
                fontsize=10, fontweight="bold", va="top", ha="left")

    # ── d: R(t) and σ(t) for both scenarios ───────────────────────────────
    ax = fig.add_subplot(gs[1, 1])
    t_arr = np.arange(T)
    vA = R_sys_A > 0; vB = R_sys_B > 0
    ax.plot(t_arr[vA], R_sys_A[vA],   color=col_A, lw=1.0,
            label="$\\mathcal{R}$ — dense urban")
    ax.plot(t_arr[vB], R_sys_B[vB],   color=col_B, lw=1.0,
            label="$\\mathcal{R}$ — sparse national")
    vsa = sigma_A > 0; vsb = sigma_B > 0
    ax.plot(t_arr[vsa], sigma_A[vsa], color=col_A, lw=0.7, ls="--",
            alpha=0.7, label="$\\sigma$ — dense urban")
    ax.plot(t_arr[vsb], sigma_B[vsb], color=col_B, lw=0.7, ls="--",
            alpha=0.7, label="$\\sigma$ — sparse national")
    ax.axhline(1.0, color="0.55", ls="--", lw=0.8)
    ax.set_xlabel("Day $t$"); ax.set_ylabel("Value")
    ax.legend(fontsize=5.0, ncol=2, borderpad=0.3, labelspacing=0.15,
              handlelength=1.2)
    ax.text(0.97, 0.97, "Solid: $\\mathcal{R}(t)$; dashed: $\\sigma(t)$",
            transform=ax.transAxes, fontsize=4.5, ha="right", va="top",
            color="0.4", style="italic")
    _panel_label(ax, "d")

    # ── e: system R(t) with normalised incidence on twin axis ──────────────
    ax  = fig.add_subplot(gs[2, 0])
    ax2 = ax.twinx()
    ax2.spines["right"].set_visible(True); ax2.spines["top"].set_visible(False)
    ax.plot(t_arr[vA], R_sys_A[vA], color=col_A, lw=0.9, label="Dense urban")
    ax.plot(t_arr[vB], R_sys_B[vB], color=col_B, lw=0.9, label="Sparse national")
    ax.axhline(1.0, color="0.55", ls="--", lw=0.8)
    peak_inc = float(max(inc_A.sum(axis=1).max(), inc_B.sum(axis=1).max()))
    ax2.fill_between(range(T), inc_A.sum(axis=1) / peak_inc, alpha=0.14, color=col_A)
    ax2.fill_between(range(T), inc_B.sum(axis=1) / peak_inc, alpha=0.14, color=col_B)
    ax.set_xlabel("Day $t$"); ax.set_ylabel("$\\mathcal{R}(t)$")
    ax2.set_ylabel("Normalised incidence", fontsize=6)
    ax2.tick_params(labelsize=6)
    ax.legend(fontsize=6, borderpad=0.3)
    _panel_label(ax, "e")

    # ── f: R^k_out over time for both scenarios ────────────────────────────
    ax = fig.add_subplot(gs[2, 1])
    R_out_A = np.array([R_outward(Rm_A[t]) for t in range(T)])
    R_out_B = np.array([R_outward(Rm_B[t]) for t in range(T)])
    dc_A    = np.linalg.norm(coords_A - coords_A.mean(axis=0), axis=1)
    i_hub_A = int(np.argmin(dc_A))
    i_per_A = int(np.argmax(dc_A))
    for k in range(N):
        lw_k = 1.2 if k in [i_hub_A, i_per_A] else 0.5
        al_k = 1.0 if k in [i_hub_A, i_per_A] else 0.3
        ax.plot(R_out_A[:, k], color=col_A, lw=lw_k, alpha=al_k)
        ax.plot(R_out_B[:, k], color=col_B, lw=lw_k, alpha=al_k, ls="--")
    ax.axhline(1.0, color="0.55", ls="--", lw=0.8)
    from matplotlib.lines import Line2D as L2
    ax.legend(handles=[L2([0],[0], color=col_A, lw=1.0, label="Dense urban"),
                        L2([0],[0], color=col_B, lw=1.0, ls="--", label="Sparse national")],
              fontsize=5.5, borderpad=0.3)
    ax.set_xlabel("Day $t$"); ax.set_ylabel("$R^k_{\\rm out}(t)$")
    ax.text(0.03, 0.97, "Hub: thick; peripheral: thick dashed\nAll other locations: muted",
            transform=ax.transAxes, fontsize=4.5, va="top", color="0.5", style="italic")
    _panel_label(ax, "f")

    fig.suptitle(
        "Figure 5: Two distinct mobility settings — Dense urban vs Sparse national, "
        "same $R_0 = 2.5$.\n"
        "Dense urban: 10 locations, ~18–40\\% commuting, ~40 km span (Figs 2–4). "
        "Sparse national: 10 locations, ~1.5–8\\% commuting, ~600 km span.",
        fontsize=5.5, y=1.005, ha="center", style="italic", color="0.3")
    plt.savefig(f"{save_prefix}_05_comparison.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved {save_prefix}_05_comparison.png")


# ══════════════════════════════════════════════════════════════════════════════
# 21. FIGURE 6 — MEETING-LOCATION ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════

def plot_fig6(sim, city_data, f_jk, prob_peak, infect_profile, lw, lb,
              scenario_name, save_prefix="fig", fig_tag="SI5"):
    """Figure 6: Meeting-location analysis.

    Panels:
      a  R^l_meeting(t) as heatmap (meeting location × time)
      b  R^j_in vs R^l_meeting at peak — actual values (not ranks)
      c  Counterfactual R(t) when top meeting locations are closed
    """
    inc       = sim["incidence"]
    R_mats    = sim["R_matrices"]
    R_meet_s  = sim["R_meeting_series"]
    S_series  = sim["susceptibles"]
    coords, pops, dists, node_types, meta = city_data
    T, N   = inc.shape
    peak   = int(inc.sum(axis=1).argmax())
    loc    = [f"L{i+1}" for i in range(N)]

    fig = plt.figure(figsize=(4.8, 3.0))
    gs  = gridspec.GridSpec(1, 2, hspace=0.45, wspace=0.52,
                            left=0.08, right=0.97, top=0.95, bottom=0.16)

    # ── a: R^l_meeting(t) heatmap ─────────────────────────────────────────
    ax = fig.add_subplot(gs[0, 0])
    im = ax.imshow(R_meet_s.T, cmap="YlOrRd", aspect="auto", origin="upper")
    ax.set_yticks(range(N)); ax.set_yticklabels(loc, fontsize=5)
    ax.set_xlabel("Day $t$"); ax.set_ylabel("Activity location $l$")
    cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.ax.set_title("$R^l_{\\rm meeting}$", fontsize=6, pad=3)
    _panel_label(ax, "a")

    # ── b: R^j_in vs R^l_meeting actual values at peak ───────────────────
    ax = fig.add_subplot(gs[0, 1])
    R_in_p   = R_inward(R_mats[peak])
    R_meet_p = R_meet_s[peak]
    for j in range(N):
        ax.scatter(R_in_p[j], R_meet_p[j], s=35,
                   color=OKABE_ITO[j % len(OKABE_ITO)],
                   edgecolors="k", linewidths=0.4, zorder=5)
        ax.annotate(loc[j], (R_in_p[j], R_meet_p[j]),
                    fontsize=5.5, xytext=(3, 3), textcoords="offset points",
                    color=OKABE_ITO[j % len(OKABE_ITO)])
    all_v = np.concatenate([R_in_p, R_meet_p])
    vmin_v, vmax_v = 0, all_v.max() * 1.1
    ax.plot([vmin_v, vmax_v], [vmin_v, vmax_v], color="0.55", ls="--", lw=0.8)
    ax.set_xlabel("$R^j_{\\rm in}$  (residence-based)")
    ax.set_ylabel("$R^l_{\\rm meeting}$  (activity-based)")
    ax.text(0.05, 0.97, f"peak (day {peak})", transform=ax.transAxes,
            fontsize=6, va="top", style="italic")
    _panel_label(ax, "b")

    fname = f"{save_prefix}_{fig_tag}_meeting.png"
    plt.savefig(fname, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved {fname}")


# ══════════════════════════════════════════════════════════════════════════════
# 22. FIGURE 7 — GROWTH RATE AND GENERATION TIME DYNAMICS
# ══════════════════════════════════════════════════════════════════════════════

def plot_fig7(sim, city_data, f_jk, w_within, w_between, max_days,
              prob_peak, scenario_name, save_prefix="fig"):
    inc    = sim["incidence"]
    R_mats = sim["R_matrices"]
    coords, pops, dists, node_types, meta = city_data
    T, N   = inc.shape

    R_sys  = np.array([R_system(R_mats[t]) for t in range(T)])
    r_hat  = empirical_growth_rate(inc, window=3)
    days   = np.arange(max_days)

    # Network-level GT: mixture of w_within and w_between weighted by K totals
    # Use the peak-time network GT as the representative g̃ for Euler-Lotka
    lw_sim = sim["lambda_within_scaled"]
    lb_sim = sim["lambda_between_scaled"]
    S_ser  = sim["susceptibles"]

    peak_t  = int(inc.sum(axis=1).argmax())
    early_t = max(1, peak_t // 3)
    late_t  = min(T - 1, peak_t + 30)

    gt_early = compute_generation_times(
        f_jk[early_t], S_ser[early_t], pops,
        prob_peak, w_within, w_between, max_days, lw_sim, lb_sim)
    gt_peak  = compute_generation_times(
        f_jk[peak_t],  S_ser[peak_t],  pops,
        prob_peak, w_within, w_between, max_days, lw_sim, lb_sim)
    gt_late  = compute_generation_times(
        f_jk[late_t],  S_ser[late_t],  pops,
        prob_peak, w_within, w_between, max_days, lw_sim, lb_sim)

    g_tilde = gt_peak["g_network"]
    r_el    = np.array([euler_lotka_r(R_sys[t], g_tilde) for t in range(T)])

    speed_ts = np.array([epidemic_speed(R_mats[t], dists, g_tilde)["speed"]
                         for t in range(T)])
    dist_ts  = np.array([epidemic_speed(R_mats[t], dists, g_tilde)["mean_distance"]
                         for t in range(T)])

    fig = plt.figure(figsize=(3.6, 2.8))
    gs  = gridspec.GridSpec(1, 1, hspace=0.45, wspace=0.50,
                            left=0.14, right=0.97, top=0.95, bottom=0.18)

    # b: GT type comparison at peak — within (g_kk), outward, inward for hub vs periph
    ax = fig.add_subplot(gs[0, 0])
    days_arr = np.arange(max_days)
    dc_fig7 = np.linalg.norm(coords - coords.mean(axis=0), axis=1)
    i_hub_7 = int(np.argmin(dc_fig7))
    i_per_7 = int(np.argmax(dc_fig7))
    g_pw_pk  = gt_peak["g_pairwise"]
    g_out_pk = gt_peak["g_outward"]
    g_in_pk  = gt_peak["g_inward"]
    for i, short, col in [(i_hub_7, "hub", OKABE_ITO[0]),
                           (i_per_7, "periph", OKABE_ITO[5])]:
        if g_pw_pk[:, i, i].sum() > 0.5:
            ax.plot(days_arr, g_pw_pk[:, i, i], color=col, lw=1.2,
                    label=f"$g_{{kk}}$ {short}")
        if g_out_pk[:, i].sum() > 0.5:
            ax.plot(days_arr, g_out_pk[:, i], color=col, lw=0.9, ls="--",
                    label=f"$g^k_{{\\rm out}}$ {short}")
        if g_in_pk[:, i].sum() > 0.5:
            ax.plot(days_arr, g_in_pk[:, i],  color=col, lw=0.9, ls=":",
                    label=f"$g^j_{{\\rm in}}$ {short}")
    ax.set_xlabel("Infection age $a_E$ (days)")
    ax.set_ylabel("Probability")
    ax.legend(fontsize=5.0, ncol=2, borderpad=0.3, labelspacing=0.15)
    _panel_label(ax, "b")

    plt.savefig(f"{save_prefix}_SI6_gt_comparison.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved {save_prefix}_SI6_gt_comparison.png")


# ══════════════════════════════════════════════════════════════════════════════
# 23. SI FIGURE 1 — SENSITIVITY METRICS
# ══════════════════════════════════════════════════════════════════════════════

def plot_SI1(sim, city_data, scenario_name, save_prefix="fig"):
    """SI Figure 1: Sensitivity and elasticity matrices.

    S_{kj} = ∂ρ/∂R_{kj} = v_k w_j / (v^T w)  [sensitivity of system R to R_{kj}]
    ε_{kj} = (R_{kj}/ρ) × S_{kj}              [proportional elasticity]

    Panels (2×3):
      a  Sensitivity matrix S_{kj} at early time
      b  Sensitivity matrix S_{kj} at epidemic peak
      c  CV of R^k_out over time
      d  Elasticity matrix ε_{kj} at early time
      e  Elasticity matrix ε_{kj} at epidemic peak
      f  Column elasticity Σ_j ε_{jk} at peak (bar chart)
    """
    inc    = sim["incidence"]
    R_mats = sim["R_matrices"]
    coords, pops, dists, node_types, meta = city_data
    T, N   = inc.shape
    peak   = int(inc.sum(axis=1).argmax())
    early  = max(1, peak // 3)
    loc    = [f"L{i+1}" for i in range(N)]

    fig = plt.figure(figsize=(7.2, 4.8))
    gs  = gridspec.GridSpec(2, 3, hspace=0.85, wspace=0.55,
                            left=0.08, right=0.97, top=0.97, bottom=0.08)

    # ── a/b: sensitivity matrices at early and peak ────────────────────────
    for ci, (day, dlbl) in enumerate([(early, f"early (day {early})"),
                                       (peak,  f"peak (day {peak})")]):
        se = sensitivity_elasticity(R_mats[day])
        ax = fig.add_subplot(gs[0, ci])
        im = ax.imshow(se["sensitivity"], cmap="Blues", aspect="auto")
        ax.set_xticks(range(N)); ax.set_xticklabels(loc, fontsize=5, rotation=45)
        ax.set_yticks(range(N)); ax.set_yticklabels(loc, fontsize=5)
        ax.set_xlabel("Infectee $j$"); ax.set_ylabel("Infector $k$")
        cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cb.ax.set_title(dlbl, fontsize=5, pad=3)
        ax.set_title(f"Sensitivity $S_{{kj}} = \\partial\\mathcal{{R}}/\\partial R_{{kj}}$",
                     fontsize=6, pad=3)
        _panel_label(ax, ["a", "b"][ci])

    # ── c: CV of outward R over time ───────────────────────────────────────
    ax = fig.add_subplot(gs[0, 2])
    cv_ts = np.array([spectral_analysis(R_mats[t])["cv_row_sums"] for t in range(T)])
    vc    = cv_ts > 0
    ax.plot(np.where(vc)[0], cv_ts[vc], color=OKABE_ITO[2], lw=1.2)
    ax.set_xlabel("Day $t$"); ax.set_ylabel("CV of $R^k_{\\rm out}$")
    ax.set_title("Heterogeneity in\noutward $R$", fontsize=6, pad=3)
    _panel_label(ax, "c")

    # ── d/e: elasticity matrices at early and peak ─────────────────────────
    for ci, (day, dlbl) in enumerate([(early, f"early (day {early})"),
                                       (peak,  f"peak (day {peak})")]):
        se = sensitivity_elasticity(R_mats[day])
        ax = fig.add_subplot(gs[1, ci])
        im = ax.imshow(se["elasticity"], cmap="Purples", aspect="auto")
        ax.set_xticks(range(N)); ax.set_xticklabels(loc, fontsize=5, rotation=45)
        ax.set_yticks(range(N)); ax.set_yticklabels(loc, fontsize=5)
        ax.set_xlabel("Infectee $j$"); ax.set_ylabel("Infector $k$")
        cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cb.ax.set_title(dlbl, fontsize=5, pad=3)
        ax.set_title(f"Elasticity $\\varepsilon_{{kj}} = (R_{{kj}}/\\mathcal{{R}})\\,S_{{kj}}$",
                     fontsize=6, pad=3)
        _panel_label(ax, ["d", "e"][ci])

    # ── f: column elasticity bar chart at peak ─────────────────────────────
    ax = fig.add_subplot(gs[1, 2])
    se_peak = sensitivity_elasticity(R_mats[peak])
    col_elas = se_peak["elasticity"].sum(axis=1)   # Σ_j ε_{jk}
    for k in range(N):
        ax.bar(k, col_elas[k], color=OKABE_ITO[k % len(OKABE_ITO)])
    ax.set_xticks(range(N)); ax.set_xticklabels(loc, rotation=45, fontsize=6)
    ax.set_ylabel("$\\sum_j \\varepsilon_{jk}$")
    ax.set_title("Column elasticity\n(infector importance)", fontsize=6, pad=5)
    ax.text(0.98, 0.98, f"peak (day {peak})", transform=ax.transAxes,
            fontsize=5, ha="right", va="top", style="italic")
    _panel_label(ax, "f")

    plt.savefig(f"{save_prefix}_SI1_sensitivity.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved {save_prefix}_SI1_sensitivity.png")


# ══════════════════════════════════════════════════════════════════════════════
# 24. SI FIGURE 2 — COUNTERFACTUAL PREDICTIONS
# ══════════════════════════════════════════════════════════════════════════════

def plot_SI2(sim, city_data, f_jk, prob_peak, infect_profile, lw, lb,
             scenario_name, save_prefix="fig"):
    """SI Figure 2: supplementary counterfactual and per-location analysis.

    This supplements Figure 2 (Dense urban) and is compared to Figure 5 (Sparse national).

    Panels (2×2):
      a  Total daily incidence over time
      b  R(t) = ρ(R_mat(t)) with twin-axis incidence
      c  R^k_out as heatmap (location × time)
      d  R^j_in as heatmap (location × time)
    """
    inc      = sim["incidence"]
    R_mats   = sim["R_matrices"]
    S_series = sim["susceptibles"]
    coords, pops, dists, node_types, meta = city_data
    T, N     = inc.shape
    loc      = [f"L{i+1}" for i in range(N)]

    R_sys   = np.array([R_system(R_mats[t]) for t in range(T)])
    R_out_s = np.array([R_outward(R_mats[t]) for t in range(T)])
    R_in_s  = np.array([R_inward(R_mats[t])  for t in range(T)])
    total_inc = inc.sum(axis=1)

    fig = plt.figure(figsize=(7.2, 4.0))
    gs  = gridspec.GridSpec(2, 2, hspace=0.58, wspace=0.48,
                            left=0.09, right=0.97, top=0.97, bottom=0.09)

    # a: total daily incidence over time
    ax = fig.add_subplot(gs[0, 0])
    ax.fill_between(range(T), total_inc / 1e3, alpha=0.25, color=OKABE_ITO[1])
    ax.plot(total_inc / 1e3, color=OKABE_ITO[1], lw=1.4)
    ax.set_xlabel("Day $t$")
    ax.set_ylabel("Total daily incidence ($\\times 10^3$)")
    ax.text(0.97, 0.97, f"Scenario A (Dense urban, $R_0=2.5$)",
            transform=ax.transAxes, fontsize=5.5, ha="right", va="top",
            style="italic", color="0.4")
    _panel_label(ax, "a")

    # b: R(t) only — no twin axis
    ax = fig.add_subplot(gs[0, 1])
    vld = R_sys > 0
    ax.plot(np.where(vld)[0], R_sys[vld], color=OKABE_ITO[4], lw=0.9,
            label="$\\mathcal{R}(t)$")
    ax.axhline(1.0, color="0.55", ls="--", lw=0.8)
    ax.set_ylabel("$\\mathcal{R}(t)$")
    ax.set_xlabel("Day $t$")
    ax.set_ylim(0, max(3.5, R_sys[vld].max() * 1.1) if vld.any() else 3.5)
    ax.legend(fontsize=6, borderpad=0.3)
    _panel_label(ax, "b")

    # c: R_out heatmap
    ax = fig.add_subplot(gs[1, 0])
    vmax_out = np.percentile(R_out_s[R_out_s > 0], 98) if (R_out_s > 0).any() else 3.0
    im = ax.imshow(R_out_s.T, cmap="YlOrRd", aspect="auto", origin="upper",
                   vmin=0, vmax=vmax_out)
    ax.set_yticks(range(N)); ax.set_yticklabels(loc, fontsize=5)
    ax.set_xlabel("Day $t$"); ax.set_ylabel("Location $k$")
    cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.ax.set_title("$R^k_{\\rm out}$", fontsize=6, pad=3)
    _panel_label(ax, "c")

    # d: R_in heatmap
    ax = fig.add_subplot(gs[1, 1])
    vmax_in = np.percentile(R_in_s[R_in_s > 0], 98) if (R_in_s > 0).any() else 3.0
    im = ax.imshow(R_in_s.T, cmap="Blues", aspect="auto", origin="upper",
                   vmin=0, vmax=vmax_in)
    ax.set_yticks(range(N)); ax.set_yticklabels(loc, fontsize=5)
    ax.set_xlabel("Day $t$"); ax.set_ylabel("Location $j$")
    cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.ax.set_title("$R^j_{\\rm in}$", fontsize=6, pad=3)
    _panel_label(ax, "d")

    plt.savefig(f"{save_prefix}_SI2_counterfactual.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved {save_prefix}_SI2_counterfactual.png")


# ══════════════════════════════════════════════════════════════════════════════
# 26. SI FIGURE 3 — EPIDEMIOLOGICAL PARAMETER ASSUMPTIONS
# ══════════════════════════════════════════════════════════════════════════════

def plot_SI_epi_params(params, w_within, w_between, gen_time_pmf, max_days,
                       sim, f_jk, populations=None, save_prefix="fig"):
    """SI Figure 3: Epidemiological parameter assumptions with literature citations.

    Panels (2×3):
      a  Generation time distributions (overall, household, community)
      b  Cumulative GT distributions showing median and 95th percentile
      c  Contact rate parameterisation (β/N_eff pattern)
      d  Day-of-week mobility scaling
      e  Effective population N_eff^l at peak time (bar chart per location)
      f  Effective per-contact transmission rate λ_eff^l = β_w/N_eff^l at peak
    """
    days = np.arange(max_days)

    # compute N_eff at peak using f_jk and populations
    inc  = sim["incidence"]
    peak = int(inc.sum(axis=1).argmax())
    if populations is None:
        # approximate from initial susceptibles + initial incidence
        populations = sim["susceptibles"][0] + inc[0]
    N_eff_peak = f_jk[peak].T @ populations   # shape (N,)
    N = len(N_eff_peak)
    loc_labels = [f"L{i+1}" for i in range(N)]

    LW = params["base_contact_rate"]
    LB = params["base_contact_rate"] * 0.30

    fig = plt.figure(figsize=(7.2, 4.8))
    gs  = gridspec.GridSpec(2, 3, hspace=0.70, wspace=0.55,
                            left=0.09, right=0.97, top=0.97, bottom=0.10)

    # ── a: GT probability density functions ───────────────────────────────
    ax = fig.add_subplot(gs[0, 0])  # row 0, col 0
    ax.plot(days, gen_time_pmf, color=OKABE_ITO[4], lw=1.4,
            label=(f"$p(a_E)$ overall "
                   f"(mean={params['gen_time_mean']} d, "
                   f"SD={params['gen_time_sd']} d)"))
    ax.plot(days, w_within,  color=OKABE_ITO[0], lw=1.0, ls="--",
            label=(f"$p_{{\\rm w}}(a_E)$ household "
                   f"(mean={params['within_gt_mean']} d, "
                   f"SD={params['within_gt_sd']} d)"))
    ax.plot(days, w_between, color=OKABE_ITO[5], lw=1.0, ls=":",
            label=(f"$p_{{\\rm b}}(a_E)$ community "
                   f"(mean={params['between_gt_mean']} d, "
                   f"SD={params['between_gt_sd']} d)"))
    ax.set_xlabel("Infection age $a_E$ (days)")
    ax.set_ylabel("Probability")
    ax.legend(fontsize=5.2, borderpad=0.3, labelspacing=0.2)
    # Citations: Hart et al. (2022) PLOS Comput. Biol.; Cereda et al. (2020)
    ax.text(0.97, 0.60,
            "$p_{\\rm w}(a_E)$: within-location\n(household) infectiousness\nprofile in $K_{kj}$",
            transform=ax.transAxes, fontsize=4.5, ha="right", va="top",
            color=OKABE_ITO[0], style="italic")
    ax.text(0.97, 0.40,
            "$p_{\\rm b}(a_E)$: between-location\n(community) infectiousness\nprofile in $K_{kj}$",
            transform=ax.transAxes, fontsize=4.5, ha="right", va="top",
            color=OKABE_ITO[5], style="italic")
    ax.text(0.97, 0.18,
            "Overall $p(a_E)$: used in\nindependent $\\hat{R}_j(t)$\nestimator",
            transform=ax.transAxes, fontsize=4.5, ha="right", va="top",
            color=OKABE_ITO[4], style="italic")
    _panel_label(ax, "a")

    # ── b: Cumulative GT distributions ────────────────────────────────────
    ax = fig.add_subplot(gs[0, 1])  # row 0, col 1
    ax.plot(days, np.cumsum(gen_time_pmf), color=OKABE_ITO[4], lw=1.4,
            label="$p(a_E)$ overall")
    ax.plot(days, np.cumsum(w_within),     color=OKABE_ITO[0], lw=1.0, ls="--",
            label="$p_{\\rm w}(a_E)$ household")
    ax.plot(days, np.cumsum(w_between),    color=OKABE_ITO[5], lw=1.0, ls=":",
            label="$p_{\\rm b}(a_E)$ community")
    ax.axhline(0.50, color="0.65", lw=0.7, ls="--")
    ax.axhline(0.95, color="0.65", lw=0.7, ls="--")
    ax.text(max_days * 0.62, 0.52, "50%", fontsize=5.5, color="0.5")
    ax.text(max_days * 0.62, 0.97, "95%", fontsize=5.5, color="0.5")
    ax.set_xlabel("Infection age $a_E$ (days)")
    ax.set_ylabel("Cumulative probability")
    ax.legend(fontsize=6, borderpad=0.3)
    _panel_label(ax, "b")

    # ── c: Contact rate parameterisation ──────────────────────────────────
    ax = fig.add_subplot(gs[0, 2])  # row 0, col 2
    labels_c = ["Base rate\n(POLYMOD)\n$\\beta$",
                "Within-location\n$\\beta_{\\rm w}$",
                "Between-location\n$\\beta_{\\rm b}$"]
    values_c = [params["base_contact_rate"], params["base_contact_rate"],
                params["base_contact_rate"] * 0.30]
    bars = ax.bar(labels_c, values_c,
                  color=[OKABE_ITO[4], OKABE_ITO[0], OKABE_ITO[5]],
                  alpha=0.85, edgecolor="none", width=0.55)
    ax.set_ylabel("Contacts per day ($\\beta$)")
    ax.set_ylim(0, max(values_c) * 1.45)
    for bar, val in zip(bars, values_c):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.2,
                f"{val:.1f}", ha="center", va="bottom", fontsize=7)
    ax.text(0.50, 0.78,
            "Actual $\\lambda^{kl}(t) = \\beta / N^l_{\\rm eff}(t)$",
            transform=ax.transAxes, fontsize=4.8, ha="center", va="bottom",
            color="0.35", style="italic")
    ax.text(0.50, 0.69,
            "[POLYMOD: Mossong et al. 2008 PLOS Med;\nCauchemez et al. 2011 Nat Med]",
            transform=ax.transAxes, fontsize=4.0, ha="center", va="top",
            color="0.45", style="italic")
    _panel_label(ax, "c")

    # ── d: Day-of-week mobility scaling ───────────────────────────────────
    ax = fig.add_subplot(gs[1, 0])  # row 1, col 0
    dow_scale  = np.array([1.00, 1.00, 1.00, 1.00, 0.90, 0.60, 0.50])
    dow_labels = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    dow_cols   = ([OKABE_ITO[1]] * 4 + [OKABE_ITO[0]] +
                  [OKABE_ITO[5]] * 2)
    bars = ax.bar(dow_labels, dow_scale, color=dow_cols, alpha=0.85,
                  edgecolor="none", width=0.65)
    ax.axhline(1.0, color="0.6", lw=0.7, ls="--")
    ax.set_ylabel("Multiplier on base commuting fraction")
    ax.set_ylim(0, 1.35)
    for bar, val in zip(bars, dow_scale):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"{val:.2f}", ha="center", va="bottom", fontsize=6.5)
    # Note: base commuting fraction varies by node type (18–40% away for urban,
    # 1.5–8% for sparse national). Scale 1.0 ≠ all time away — it is the
    # node-specific commuting fraction (Wesolowski et al. 2015 PNAS; Mossong 2008 PLOS Med).
    ax.text(0.02, 0.97, ("Scale applied to node-specific\nbase commuting fraction\n"
                          "(core: 40%, peripheral: 18% away)\n"
                          "[Wesolowski et al. 2015 PNAS;\nMossong et al. 2008 PLOS Med]"),
            transform=ax.transAxes, fontsize=4.5, va="top", ha="left", color="0.4",
            style="italic")
    _panel_label(ax, "d")

    # ── e: Effective population N_eff^l at peak time ──────────────────────
    ax = fig.add_subplot(gs[1, 1])  # row 1, col 1
    bar_cols_e = [OKABE_ITO[i % len(OKABE_ITO)] for i in range(N)]
    bars_e = ax.bar(range(N), N_eff_peak / 1e3, color=bar_cols_e,
                    alpha=0.85, edgecolor="none", width=0.7)
    ax.set_xticks(range(N)); ax.set_xticklabels(loc_labels, rotation=45, fontsize=6)
    ax.set_ylabel("$N^l_{\\rm eff}$ ($\\times 10^3$)")
    ax.set_title(f"Effective population $N^l_{{\\rm eff}}$\nat peak (day {peak})",
                 fontsize=6, pad=3)
    ax.text(0.97, 0.97,
            "$N^l_{\\rm eff}(t) = \\sum_j f_{{jl}}(t)\\,N_j$",
            transform=ax.transAxes, fontsize=5.0, ha="right", va="top",
            color="0.35", style="italic")
    _panel_label(ax, "e")

    # ── f: Effective per-contact transmission rate λ_eff^l = β/N_eff^l ──
    ax = fig.add_subplot(gs[1, 2])  # row 1, col 2
    lam_w_eff = np.where(N_eff_peak > 0, LW / N_eff_peak, 0.0)
    lam_b_eff = np.where(N_eff_peak > 0, LB / N_eff_peak, 0.0)
    x = np.arange(N)
    width = 0.38
    bars_w = ax.bar(x - width/2, lam_w_eff * 1e4, width=width,
                    color=OKABE_ITO[0], alpha=0.85, edgecolor="none",
                    label="$\\beta_{\\rm w}/N^l_{\\rm eff}$")
    bars_b = ax.bar(x + width/2, lam_b_eff * 1e4, width=width,
                    color=OKABE_ITO[5], alpha=0.85, edgecolor="none",
                    label="$\\beta_{\\rm b}/N^l_{\\rm eff}$")
    ax.set_xticks(range(N)); ax.set_xticklabels(loc_labels, rotation=45, fontsize=6)
    ax.set_ylabel("$\\lambda^l_{\\rm eff}$ ($\\times 10^{-4}$)")
    ax.set_title(f"Per-contact transmission rate\n$\\lambda^l = \\beta / N^l_{{\\rm eff}}$ at peak",
                 fontsize=6, pad=3)
    ax.legend(fontsize=5.5, borderpad=0.3)
    ax.text(0.97, 0.97,
            "Higher $\\lambda^l$ where fewer\npeople gather (low $N^l_{\\rm eff}$)",
            transform=ax.transAxes, fontsize=4.5, ha="right", va="top",
            color="0.4", style="italic")
    _panel_label(ax, "f")

    plt.savefig(f"{save_prefix}_SI3_epi_params.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved {save_prefix}_SI3_epi_params.png")


# ══════════════════════════════════════════════════════════════════════════════
# 27. SI FIGURE 4 — PDE NUMERICAL CONVERGENCE
# ══════════════════════════════════════════════════════════════════════════════

def _simulate_nloc(n_loc, T_test, params, LW, LB, seed=42):
    """Generate a mini lagos-like city with n_loc locations and simulate.
    Returns dict with 'R_matrices', 'incidence' keys.
    Used for spatial grid convergence test.
    """
    city = generate_city(n_loc, scenario="lagos", seed=seed)
    coords_l, pops_l, dists_l, types_l, meta_l = city
    f_l, _ = generate_mobility(n_loc, T_test, pops_l, dists_l, types_l, meta_l,
                                day_variation_sd=0.10, seed=seed)
    max_days = params["max_gen_time"]
    gtp  = discretise_gamma(params["gen_time_mean"], params["gen_time_sd"], max_days)
    ww   = discretise_gamma(params["within_gt_mean"],  params["within_gt_sd"],  max_days)
    wb   = discretise_gamma(params["between_gt_mean"], params["between_gt_sd"], max_days)
    init = np.zeros(n_loc); init[0] = 10
    s = simulate_epidemic_pde(
        T_test, n_loc, pops_l, f_l,
        params["prob_transmission_peak"], gtp, max_days,
        params["R0_target"], init, LW, LB, ww, wb,
        birth_rate=0.00003, death_rate=0.00003,
        stochastic=False, susceptible_depletion=True, seed=seed)
    return s


def _simulate_heun(T, n_locations, populations, f_jk_series,
                   prob_peak, infect_profile, max_days, R0_target,
                   initial_infections, lambda_within, lambda_between,
                   w_within, w_between, seed=42):
    """Deterministic epidemic using Heun's method (2nd-order Runge-Kutta)
    for the susceptible depletion step. The upwind PDE step is unchanged.
    Used for numerical validation in SI Figure 4."""
    N  = n_locations
    S  = populations.copy().astype(float)
    incidence  = np.zeros((T, N))
    R_matrices = np.zeros((T, N, N))
    f0     = f_jk_series[0]
    R0_mat = compute_R_matrix(f0, populations, populations, prob_peak,
                               infect_profile, lambda_within, lambda_between)
    rho0   = R_system(R0_mat)
    scale  = R0_target / rho0 if rho0 > 0 else 1.0
    lw, lb = lambda_within * scale, lambda_between * scale
    R_matrices[0] = compute_R_matrix(f0, populations, populations, prob_peak,
                                      infect_profile, lw, lb)
    E_pde = np.zeros((T, N, max_days))
    E_pde[0, :, 0] = initial_infections
    S -= initial_infections
    for t in range(1, T):
        f_t = f_jk_series[min(t, len(f_jk_series) - 1)]
        E_pde[t, :, 1:] = E_pde[t - 1, :, :-1]
        base_K, bKw, bKb, _, _ = _kernel_base(f_t, populations, lw, lb)
        wE_w = prob_peak * (E_pde[t, :, 1:] @ w_within[1:])
        wE_b = prob_peak * (E_pde[t, :, 1:] @ w_between[1:])
        contrib_kj = bKw * wE_w[:, np.newaxis] + bKb * wE_b[:, np.newaxis]
        # Predictor (Forward Euler)
        new_j  = np.minimum(np.maximum(S * contrib_kj.sum(axis=0), 0.0), S)
        S_pred = np.maximum(S - new_j, 0.0)
        # Corrector: re-evaluate force of infection at S_pred
        new_j2 = np.minimum(np.maximum(S_pred * contrib_kj.sum(axis=0), 0.0), S_pred)
        # Average slopes (Heun's rule)
        S = np.maximum(S - 0.5 * (new_j + new_j2), 0.0)
        E_pde[t, :, 0] = 0.5 * (new_j + new_j2)
        incidence[t]   = E_pde[t, :, 0]
        R_matrices[t]  = compute_R_matrix(f_t, S, populations, prob_peak,
                                           infect_profile, lw, lb)
    return {"incidence": incidence, "R_matrices": R_matrices}


def _simulate_rk4(T, n_locations, populations, f_jk_series,
                  prob_peak, infect_profile, max_days, R0_target,
                  initial_infections, lambda_within, lambda_between,
                  w_within, w_between, seed=42):
    """Deterministic epidemic using 4th-order Runge-Kutta for S(t).
    The upwind PDE step is unchanged. Used for numerical validation."""
    N  = n_locations
    S  = populations.copy().astype(float)
    incidence  = np.zeros((T, N))
    R_matrices = np.zeros((T, N, N))
    f0     = f_jk_series[0]
    R0_mat = compute_R_matrix(f0, populations, populations, prob_peak,
                               infect_profile, lambda_within, lambda_between)
    rho0   = R_system(R0_mat)
    scale  = R0_target / rho0 if rho0 > 0 else 1.0
    lw, lb = lambda_within * scale, lambda_between * scale
    R_matrices[0] = compute_R_matrix(f0, populations, populations, prob_peak,
                                      infect_profile, lw, lb)
    E_pde = np.zeros((T, N, max_days))
    E_pde[0, :, 0] = initial_infections
    S -= initial_infections
    for t in range(1, T):
        f_t = f_jk_series[min(t, len(f_jk_series) - 1)]
        E_pde[t, :, 1:] = E_pde[t - 1, :, :-1]
        base_K, bKw, bKb, _, _ = _kernel_base(f_t, populations, lw, lb)
        wE_w = prob_peak * (E_pde[t, :, 1:] @ w_within[1:])
        wE_b = prob_peak * (E_pde[t, :, 1:] @ w_between[1:])
        contrib = (bKw * wE_w[:, np.newaxis] + bKb * wE_b[:, np.newaxis]).sum(axis=0)
        # RK4 for dS/dt = -S * contrib (treating contrib as fixed over the step)
        k1 = -np.minimum(S * contrib, S)
        k2 = -np.minimum(np.maximum(S + 0.5*k1, 0.0) * contrib, np.maximum(S + 0.5*k1, 0.0))
        k3 = -np.minimum(np.maximum(S + 0.5*k2, 0.0) * contrib, np.maximum(S + 0.5*k2, 0.0))
        k4 = -np.minimum(np.maximum(S + k3, 0.0) * contrib, np.maximum(S + k3, 0.0))
        delta_S = (k1 + 2*k2 + 2*k3 + k4) / 6.0
        new_j  = np.minimum(np.maximum(-delta_S, 0.0), S)
        S = np.maximum(S - new_j, 0.0)
        E_pde[t, :, 0] = new_j
        incidence[t]   = new_j
        R_matrices[t]  = compute_R_matrix(f_t, S, populations, prob_peak,
                                           infect_profile, lw, lb)
    return {"incidence": incidence, "R_matrices": R_matrices}


def plot_SI_pde_convergence(city_data, f_jk, params, initial_infections, LW, LB,
                             w_within, w_between, T_test=90, save_prefix="fig"):
    """SI Figure 4: Numerical validation of the PDE solver.

    Tests epidemic outputs across:
      (a/b/c) Different GT truncation values max_days ∈ {10,15,20,25}
      (d)     Spatial grid size: N_LOC ∈ {5, 10, 15, 20} locations
      (e)     Alternative solver: Forward Euler vs Heun's method
      (f)     R0 consistency check (calibrated R_0 should ≈ 2.5)

    Panels:
      a  Total incidence for max_days ∈ {10, 15, 20, 25} (deterministic)
      b  System R(t) for max_days ∈ {10, 15, 20, 25}
      c  Total incidence: deterministic vs stochastic (5 seeds)
      d  Spatial grid convergence: system R(t) for N_LOC ∈ {5,10,15,20}
      e  Solver comparison: Forward Euler vs Heun (2nd-order RK) for S equation
      f  R0 consistency check — calibrated ρ(R(t=0)) for each max_days
    """
    coords, pops, dists, node_types, meta = city_data
    N_LOC = len(pops)
    f_sub = f_jk[:T_test]

    max_days_ref = params["max_gen_time"]
    gen_time_pmf = discretise_gamma(params["gen_time_mean"],
                                     params["gen_time_sd"], max_days_ref)
    infect_profile = gen_time_pmf.copy()

    test_md  = [10, 15, 20, 25]
    md_cols  = [OKABE_ITO[0], OKABE_ITO[2], OKABE_ITO[4], OKABE_ITO[5]]

    print("  Running PDE convergence tests (deterministic, T=90)...")
    sims_md = {}
    for md, col in zip(test_md, md_cols):
        gtp = discretise_gamma(params["gen_time_mean"], params["gen_time_sd"], md)
        ww  = discretise_gamma(params["within_gt_mean"],  params["within_gt_sd"],  md)
        wb  = discretise_gamma(params["between_gt_mean"], params["between_gt_sd"], md)
        s   = simulate_epidemic_pde(
            T_test, N_LOC, pops, f_sub,
            params["prob_transmission_peak"], gtp, md,
            params["R0_target"], initial_infections, LW, LB, ww, wb,
            birth_rate=0.00003, death_rate=0.00003,
            stochastic=False, susceptible_depletion=True, seed=42)
        sims_md[md] = s

    fig = plt.figure(figsize=(7.2, 8.0))
    gs  = gridspec.GridSpec(3, 2, hspace=0.60, wspace=0.45,
                            left=0.09, right=0.97, top=0.95, bottom=0.08)

    # ── a: Total incidence for different max_days ──────────────────────────
    ax = fig.add_subplot(gs[0, 0])
    for md, col in zip(test_md, md_cols):
        inc = sims_md[md]["incidence"].sum(axis=1)
        ax.plot(inc / 1e3, color=col, lw=0.9, label=f"$\\tau={md}$ d")
    ax.set_xlabel("Day $t$")
    ax.set_ylabel("Total incidence ($\\times 10^3$)")
    ax.set_title("GT truncation convergence", fontsize=6, pad=3)
    ax.legend(fontsize=6, borderpad=0.3)
    _panel_label(ax, "a")

    # ── b: System R(t) for different max_days ─────────────────────────────
    ax = fig.add_subplot(gs[0, 1])
    for md, col in zip(test_md, md_cols):
        R_sys = np.array([R_system(sims_md[md]["R_matrices"][t])
                          for t in range(T_test)])
        vld = R_sys > 0
        ax.plot(np.where(vld)[0], R_sys[vld], color=col, lw=0.9,
                label=f"$\\tau={md}$ d")
    ax.axhline(1.0, color="0.55", ls="--", lw=0.8)
    ax.set_xlabel("Day $t$"); ax.set_ylabel("$\\mathcal{R}(t)$")
    ax.legend(fontsize=6, borderpad=0.3)
    _panel_label(ax, "b")

    # ── c: Stochastic variability (5 seeds vs deterministic) ──────────────
    ax = fig.add_subplot(gs[1, 0])
    s_det = simulate_epidemic_pde(
        T_test, N_LOC, pops, f_sub,
        params["prob_transmission_peak"], infect_profile, max_days_ref,
        params["R0_target"], initial_infections, LW, LB, w_within, w_between,
        birth_rate=0.00003, death_rate=0.00003,
        stochastic=False, susceptible_depletion=True, seed=42)
    ax.plot(s_det["incidence"].sum(axis=1) / 1e3,
            color="k", lw=1.2, label="Deterministic", zorder=5)
    for seed_v in [42, 123, 456, 789, 1337]:
        s_st = simulate_epidemic_pde(
            T_test, N_LOC, pops, f_sub,
            params["prob_transmission_peak"], infect_profile, max_days_ref,
            params["R0_target"], initial_infections, LW, LB, w_within, w_between,
            birth_rate=0.00003, death_rate=0.00003,
            stochastic=True, susceptible_depletion=True, seed=seed_v)
        ax.plot(s_st["incidence"].sum(axis=1) / 1e3,
                color=OKABE_ITO[1], lw=0.6, alpha=0.6)
    from matplotlib.lines import Line2D as _L2
    ax.legend(handles=[_L2([0],[0], color="k", lw=1.2, label="Deterministic"),
                        _L2([0],[0], color=OKABE_ITO[1], lw=0.8, alpha=0.7,
                            label="Stochastic (5 seeds)")],
              fontsize=6, borderpad=0.3)
    ax.set_xlabel("Day $t$"); ax.set_ylabel("Total incidence ($\\times 10^3$)")
    _panel_label(ax, "c")

    # ── d: Spatial grid convergence (N_LOC ∈ {5,10,15,20}) ───────────────
    ax = fig.add_subplot(gs[1, 1])
    nloc_vals = [5, 10, 15, 20]
    nloc_cols = [OKABE_ITO[0], OKABE_ITO[2], OKABE_ITO[4], OKABE_ITO[5]]
    print("  Spatial grid convergence (varying N_LOC)...")
    for nl, col in zip(nloc_vals, nloc_cols):
        try:
            s_nl = _simulate_nloc(nl, T_test, params, LW, LB, seed=42)
            R_nl = np.array([R_system(s_nl["R_matrices"][t]) for t in range(T_test)])
            vld  = R_nl > 0
            ax.plot(np.where(vld)[0], R_nl[vld], color=col, lw=0.9,
                    label=f"$N_{{\\rm loc}}={nl}$")
        except Exception as e:
            print(f"    N_LOC={nl} failed: {e}")
    ax.axhline(1.0, color="0.55", ls="--", lw=0.8)
    ax.set_xlabel("Day $t$"); ax.set_ylabel("$\\mathcal{R}(t)$")
    ax.set_title("Spatial grid convergence\n($N_{{\\rm loc}}$ locations, same $R_0=2.5$)",
                 fontsize=6, pad=3)
    ax.legend(fontsize=6, borderpad=0.3)
    _panel_label(ax, "d")

    # ── e: Solver comparison ──────────────────────────────────────────────
    ax = fig.add_subplot(gs[2, 0])
    print("  Alternative solvers (Euler vs Heun vs RK4)...")
    s_euler = simulate_epidemic_pde(
        T_test, N_LOC, pops, f_sub, params["prob_transmission_peak"],
        infect_profile, max_days_ref, params["R0_target"],
        initial_infections, LW, LB, w_within, w_between,
        birth_rate=0.00003, death_rate=0.00003,
        stochastic=False, susceptible_depletion=True, seed=42)
    s_heun = _simulate_heun(
        T_test, N_LOC, pops, f_sub, params["prob_transmission_peak"],
        infect_profile, max_days_ref, params["R0_target"],
        initial_infections, LW, LB, w_within, w_between, seed=42)
    s_rk4  = _simulate_rk4(
        T_test, N_LOC, pops, f_sub, params["prob_transmission_peak"],
        infect_profile, max_days_ref, params["R0_target"],
        initial_infections, LW, LB, w_within, w_between, seed=42)
    ax.plot(s_euler["incidence"].sum(axis=1) / 1e3,
            color=OKABE_ITO[4], lw=1.1, label="Forward Euler")
    ax.plot(s_heun["incidence"].sum(axis=1) / 1e3,
            color=OKABE_ITO[5], lw=0.9, ls="--", label="Heun (2nd-order RK)")
    ax.plot(s_rk4["incidence"].sum(axis=1) / 1e3,
            color=OKABE_ITO[2], lw=0.8, ls=":", label="RK4 (4th-order)")
    ax.set_xlabel("Day $t$"); ax.set_ylabel("Incidence ($\\times 10^3$)")
    ax.set_title("Solver comparison\n(three methods should agree)",
                 fontsize=6, pad=3)
    ax.legend(fontsize=6, borderpad=0.3)
    _panel_label(ax, "e")

    # ── f: R0 consistency check ────────────────────────────────────────────
    ax = fig.add_subplot(gs[2, 1])
    rho0_vals = []
    for md in test_md:
        rho0_vals.append(R_system(sims_md[md]["R_matrices"][0]))
    ax.plot(test_md, rho0_vals, color=OKABE_ITO[4], lw=1.0, marker="o",
            markersize=5, markeredgecolor="k", markeredgewidth=0.5)
    ax.axhline(params["R0_target"], color="0.4", ls="--", lw=0.8,
               label=f"$R_0 = {params['R0_target']}$")
    ax.set_xlabel("Truncation $\\tau$ (days)")
    ax.set_ylabel("$\\mathcal{R}_0 = \\rho(\\mathbf{R}(t=0))$")
    ax.set_title("$\\mathcal{R}_0$ consistency\n(calibrated, should $\\approx 2.5$)",
                 fontsize=6, pad=3)
    ax.set_ylim(0, 3.5)
    ax.legend(fontsize=6, borderpad=0.3)
    ax.set_xticks(test_md)
    _panel_label(ax, "f")

    plt.savefig(f"{save_prefix}_SI4_convergence.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved {save_prefix}_SI4_convergence.png")


# ══════════════════════════════════════════════════════════════════════════════
# 28. SI FIGURE 7 — SENSITIVITY TO PARAMETERS
# ══════════════════════════════════════════════════════════════════════════════

def plot_SI_sensitivity(city_data, f_jk, params, initial_infections,
                        LW, LB, w_within, w_between, T_test=120, save_prefix="fig"):
    """SI Figure 7: Sensitivity of epidemic dynamics to key parameters.

    Panels:
      a  R(t) for different R0 values (1.5, 2.0, 2.5, 3.0)
      b  Incidence for different initial infection patterns (hub-seeded vs
         periph-seeded vs uniform)
      c  R(t) for different between-location fraction λ_b/λ_w (0.15, 0.30, 0.45)
      d  R(t) for different generation time means (4.0, 5.5, 7.0 days)
    """
    coords, pops, dists, node_types, meta = city_data
    N_LOC = len(pops)
    f_sub = f_jk[:T_test]
    max_days = params["max_gen_time"]
    gen_time_pmf = discretise_gamma(params["gen_time_mean"],
                                     params["gen_time_sd"], max_days)
    infect_profile = gen_time_pmf.copy()

    dc = np.linalg.norm(coords - coords.mean(axis=0), axis=1)
    i_hub = int(np.argmin(dc))
    i_per = int(np.argmax(dc))

    print("  Running SI7 sensitivity simulations...")

    fig = plt.figure(figsize=(7.2, 5.2))
    gs  = gridspec.GridSpec(2, 2, hspace=0.55, wspace=0.48,
                            left=0.09, right=0.97, top=0.95, bottom=0.09)

    # ── a: R0 sensitivity ─────────────────────────────────────────────────
    ax = fig.add_subplot(gs[0, 0])
    R0_vals  = [1.5, 2.0, 2.5, 3.0]
    R0_cols  = [OKABE_ITO[2], OKABE_ITO[0], OKABE_ITO[4], OKABE_ITO[5]]
    for R0v, col in zip(R0_vals, R0_cols):
        s = simulate_epidemic_pde(
            T_test, N_LOC, pops, f_sub,
            params["prob_transmission_peak"], infect_profile, max_days,
            R0v, initial_infections, LW, LB, w_within, w_between,
            birth_rate=0.00003, death_rate=0.00003,
            stochastic=False, susceptible_depletion=True, seed=42)
        R_sys = np.array([R_system(s["R_matrices"][t]) for t in range(T_test)])
        vld = R_sys > 0
        ax.plot(np.where(vld)[0], R_sys[vld], color=col, lw=0.9,
                label=f"$R_0={R0v}$")
    ax.axhline(1.0, color="0.55", ls="--", lw=0.8)
    ax.set_xlabel("Day $t$"); ax.set_ylabel("$\\mathcal{R}(t)$")
    ax.legend(fontsize=6, borderpad=0.3)
    _panel_label(ax, "a")

    # ── b: initial condition sensitivity ─────────────────────────────────
    ax = fig.add_subplot(gs[0, 1])
    init_scenarios = {
        "hub-seeded": ("seed hub only", OKABE_ITO[0]),
        "periph-seeded": ("seed periph only", OKABE_ITO[5]),
        "uniform": ("uniform seeding", OKABE_ITO[4]),
    }
    n_seed = initial_infections.sum()
    for scen_name, (lbl, col) in init_scenarios.items():
        init_v = np.zeros(N_LOC)
        if scen_name == "hub-seeded":
            init_v[i_hub] = n_seed
        elif scen_name == "periph-seeded":
            init_v[i_per] = n_seed
        else:
            init_v[:] = n_seed / N_LOC
        s = simulate_epidemic_pde(
            T_test, N_LOC, pops, f_sub,
            params["prob_transmission_peak"], infect_profile, max_days,
            params["R0_target"], init_v, LW, LB, w_within, w_between,
            birth_rate=0.00003, death_rate=0.00003,
            stochastic=False, susceptible_depletion=True, seed=42)
        inc_tot = s["incidence"].sum(axis=1)
        ax.plot(inc_tot / 1e3, color=col, lw=0.9, label=lbl)
    ax.set_xlabel("Day $t$"); ax.set_ylabel("Incidence ($\\times 10^3$)")
    ax.legend(fontsize=6, borderpad=0.3)
    _panel_label(ax, "b")

    # ── c: between-fraction sensitivity ──────────────────────────────────
    ax = fig.add_subplot(gs[1, 0])
    beta_fracs = [0.15, 0.30, 0.45]
    bf_cols    = [OKABE_ITO[2], OKABE_ITO[4], OKABE_ITO[5]]
    for bf, col in zip(beta_fracs, bf_cols):
        lb_v = LW * bf
        s = simulate_epidemic_pde(
            T_test, N_LOC, pops, f_sub,
            params["prob_transmission_peak"], infect_profile, max_days,
            params["R0_target"], initial_infections, LW, lb_v, w_within, w_between,
            birth_rate=0.00003, death_rate=0.00003,
            stochastic=False, susceptible_depletion=True, seed=42)
        R_sys = np.array([R_system(s["R_matrices"][t]) for t in range(T_test)])
        vld = R_sys > 0
        ax.plot(np.where(vld)[0], R_sys[vld], color=col, lw=0.9,
                label=f"$\\lambda_b/\\lambda_w={bf}$")
    ax.axhline(1.0, color="0.55", ls="--", lw=0.8)
    ax.set_xlabel("Day $t$"); ax.set_ylabel("$\\mathcal{R}(t)$")
    ax.legend(fontsize=6, borderpad=0.3)
    _panel_label(ax, "c")

    # ── d: infectiousness profile shape sensitivity ────────────────────────
    # Vary the within-household (w_within) shape: early-peaking (COVID-like,
    # Cereda 2020), standard (current), and late-peaking profiles.
    # The overall GT emerges from w_within, w_between, and mobility — it is
    # NOT a free parameter. This tests sensitivity to the biological timing
    # of household transmission while keeping R_0=2.5 and mobility fixed.
    ax = fig.add_subplot(gs[1, 1])
    profile_specs = [
        ("$p_{\\rm w}$: early-peaking\n(Cereda 2020, mean=2.5 d)",  2.5, 1.0,  OKABE_ITO[0]),
        ("$p_{\\rm w}$: standard\n(Hart 2022, mean=4.0 d)",          4.0, 1.5,  OKABE_ITO[4]),
        ("$p_{\\rm w}$: late-peaking\n(longer serial, mean=6.0 d)",  6.0, 2.0,  OKABE_ITO[5]),
    ]
    gtp_base = discretise_gamma(params["gen_time_mean"], params["gen_time_sd"], max_days)
    wb_base  = discretise_gamma(params["between_gt_mean"], params["between_gt_sd"], max_days)
    for lbl, ww_mean, ww_sd, col in profile_specs:
        ww_v = discretise_gamma(ww_mean, ww_sd, max_days)
        s = simulate_epidemic_pde(
            T_test, N_LOC, pops, f_sub,
            params["prob_transmission_peak"], gtp_base, max_days,
            params["R0_target"], initial_infections, LW, LB, ww_v, wb_base,
            birth_rate=0.00003, death_rate=0.00003,
            stochastic=False, susceptible_depletion=True, seed=42)
        inc_tot = s["incidence"].sum(axis=1)
        ax.plot(inc_tot / 1e3, color=col, lw=0.9, label=lbl)
    ax.set_xlabel("Day $t$")
    ax.set_ylabel("Incidence ($\\times 10^3$)")
    ax.set_title("Sensitivity to within-location\ninfectiousness profile $p_{{\\rm w}}(a_E)$",
                 fontsize=6, pad=3)
    ax.text(0.03, 0.97,
            ("$p_{\\rm w}(a_E)$ shape varied;\n"
             "$p_{\\rm b}(a_E)$, $R_0$, mobility\n"
             "held constant. $\\lambda = \\beta/N_{\\rm eff}$;\n"
             "GT emerges from mechanism."),
            transform=ax.transAxes, fontsize=4.5, va="top", color="0.4", style="italic")
    ax.legend(fontsize=5.5, borderpad=0.3)
    _panel_label(ax, "d")

    plt.savefig(f"{save_prefix}_SI7_sensitivity.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved {save_prefix}_SI7_sensitivity.png")


# ══════════════════════════════════════════════════════════════════════════════
# 29. SI FIGURE 8 — 3-D PAIRWISE INCIDENCE: EARLY vs PEAK, BOTH SCENARIOS
# ══════════════════════════════════════════════════════════════════════════════

def plot_SI_3d_earlypeak(sim_A, sim_B, save_prefix="fig"):
    """SI Figure 8: Pairwise new-infection matrices E_{kj} at early and peak
    time points for both Dense-urban and Sparse-national scenarios.

    Panels (2×2):
      a  Dense urban     — early epidemic  (E_{kj})
      b  Dense urban     — epidemic peak   (E_{kj})
      c  Sparse national — early epidemic  (E_{kj})
      d  Sparse national — epidemic peak   (E_{kj})
    """
    inc_A  = sim_A["incidence"]
    inc_B  = sim_B["incidence"]
    imat_A = sim_A["incidence_matrix"]   # shape (T, N, N)
    imat_B = sim_B["incidence_matrix"]

    pk_A    = int(inc_A.sum(axis=1).argmax())
    pk_B    = int(inc_B.sum(axis=1).argmax())
    early_A = max(1, pk_A // 3)
    early_B = max(1, pk_B // 3)

    fig = plt.figure(figsize=(7.2, 6.4))
    gs  = gridspec.GridSpec(2, 2, hspace=0.30, wspace=0.25,
                            left=0.04, right=0.98, top=0.92, bottom=0.04)

    panel_specs = [
        (0, 0, "a", imat_A[early_A], f"Dense urban — early (day {early_A})"),
        (0, 1, "b", imat_A[pk_A],    f"Dense urban — peak  (day {pk_A})"),
        (1, 0, "c", imat_B[early_B], f"Sparse national — early (day {early_B})"),
        (1, 1, "d", imat_B[pk_B],    f"Sparse national — peak  (day {pk_B})"),
    ]

    for row, col, lbl, mat, title in panel_specs:
        ax = fig.add_subplot(gs[row, col], projection="3d")
        _bar3d_inc(ax, mat, title)
        ax.text2D(-0.04, 1.12, lbl, transform=ax.transAxes,
                  fontsize=10, fontweight="bold", va="top", ha="left")

    fig.suptitle(
        "SI Figure 8: Pairwise new-infection matrix $E_{kj}$ at early and peak time points.\n"
        "Colours identify infector $k$; bar height is daily new infections transmitted to $j$.\n"
        "Panels a–b: Dense-urban scenario (Figs 2–4).  "
        "Panels c–d: Sparse-national scenario (Fig 5).",
        fontsize=6, y=0.97, ha="center", style="italic", color="0.3")
    plt.savefig(f"{save_prefix}_SI8_3d_earlypeak.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved {save_prefix}_SI8_3d_earlypeak.png")


# ══════════════════════════════════════════════════════════════════════════════
# 30b. FIGURE — 3D SURFACES: INCIDENCE, GT MEAN, GT WATERFALL
# ══════════════════════════════════════════════════════════════════════════════

def plot_3d_surfaces(sim, city_data, f_jk, w_within, w_between, max_days,
                     prob_peak, save_prefix="fig"):
    """3D surface visualisations of epidemic dynamics.

    Panels:
      a  3D surface: incidence E_j(t,0) — axes time × location × cases
      b  3D surface: outward GT mean μ^k_out(t) — time × location × days
         (sampled every 5 days for computational efficiency)
      c  GT network distribution waterfall — g_network(a_E) plotted as coloured
         ribbons at sampled time snapshots (colour = epidemic day).
         Near-constant ribbons confirm time-invariance; subtle shifts reveal
         day-of-week and susceptible-depletion effects on the within/between
         mixture weights.

    GT variation is inherently modest in this model because:
      - lw >> lb  (within >> between contact rates) → most pairs dominated by
        w_within (shorter GT)
      - With lw/lb = 1/0.30 and typical bKw/bKb ≈ 3:1, effective GT mean
        ranges ~4.0–5.0 d across hub vs peripheral pairs
      - Larger GT spread requires stronger between-location transmission
        (higher lb/lw) or greater separation of within/between GT means.
    """
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    inc    = sim["incidence"]              # (T, N)
    lw_sim = sim["lambda_within_scaled"]
    lb_sim = sim["lambda_between_scaled"]
    S_ser  = sim["susceptibles"]
    coords, pops, dists, node_types, meta = city_data
    T, N   = inc.shape
    peak   = int(inc.sum(axis=1).argmax())
    days_a = np.arange(max_days)

    # ── pre-compute GT metrics at sampled time points ─────────────────────
    t_sample = np.arange(0, T, 5)          # every 5 days (~30 evaluations)
    n_samp   = len(t_sample)
    gt_out_means = np.zeros((n_samp, N))   # outward GT mean per location
    gt_net_pmfs  = np.zeros((n_samp, max_days))  # network GT distribution
    print("  Computing GT surfaces (sampled every 5 days)...")
    for ti, t_idx in enumerate(t_sample):
        gt_t = compute_generation_times(
            f_jk[t_idx], S_ser[t_idx], pops, prob_peak,
            w_within, w_between, max_days, lw_sim, lb_sim)
        g_out = gt_t["g_outward"]          # (max_days, N)
        g_net = gt_t["g_network"]          # (max_days,)
        for j in range(N):
            if g_out[:, j].sum() > 0.5:
                gt_out_means[ti, j] = float(g_out[:, j] @ days_a)
        if g_net.sum() > 0.5:
            gt_net_pmfs[ti] = g_net

    loc_labels = [f"L{i+1}" for i in range(N)]

    # ── helper: clean 3D pane styling ────────────────────────────────────
    def _clean_panes(ax3):
        for pane in [ax3.xaxis.pane, ax3.yaxis.pane, ax3.zaxis.pane]:
            pane.fill = False
            pane.set_edgecolor("#dddddd")
        ax3.xaxis._axinfo["grid"]["color"] = (0, 0, 0, 0.06)
        ax3.yaxis._axinfo["grid"]["color"] = (0, 0, 0, 0.06)
        ax3.zaxis._axinfo["grid"]["color"] = (0, 0, 0, 0.06)

    fig = plt.figure(figsize=(7.2, 9.5))
    gs  = gridspec.GridSpec(2, 2, hspace=0.45, wspace=0.42,
                            left=0.06, right=0.97, top=0.95, bottom=0.04)

    # ── a: incidence surface ─────────────────────────────────────────────
    ax = fig.add_subplot(gs[0, 0], projection="3d")
    t_arr = np.arange(T, dtype=float)
    j_arr = np.arange(N, dtype=float)
    T_g, J_g = np.meshgrid(t_arr, j_arr)       # (N, T)
    Z_inc     = inc.T / 1e3                     # (N, T)
    surf_a = ax.plot_surface(T_g, J_g, Z_inc, cmap="YlOrRd", alpha=0.88,
                              linewidth=0, antialiased=True,
                              rcount=min(T, 60), ccount=N)
    ax.set_xlabel("Day $t$", fontsize=5, labelpad=0)
    ax.set_ylabel("Location $j$", fontsize=5, labelpad=0)
    ax.set_zlabel("(×10³)", fontsize=4.5, labelpad=-2)
    ax.set_yticks(j_arr); ax.set_yticklabels(loc_labels, fontsize=3.5)
    ax.tick_params(axis="x", labelsize=4.5, pad=0)
    ax.tick_params(axis="z", labelsize=4.5, pad=0)
    ax.view_init(elev=32, azim=-52)
    ax.set_title("$E_j(t,0)$ — incidence\n(time × location)", fontsize=7, pad=6)
    _clean_panes(ax)
    fig.colorbar(surf_a, ax=ax, shrink=0.42, pad=0.06,
                 label="Incidence (×10³)", format="%.1f",
                 orientation="vertical", aspect=15)
    ax.text2D(-0.06, 1.06, "a", transform=ax.transAxes,
              fontsize=10, fontweight="bold", va="top", ha="left")

    # ── b: outward GT mean surface ────────────────────────────────────────
    ax = fig.add_subplot(gs[0, 1], projection="3d")
    T_s, J_s = np.meshgrid(t_sample.astype(float), j_arr)  # (N, n_samp)
    Z_gt      = gt_out_means.T                              # (N, n_samp)
    Z_gt      = np.where(Z_gt > 0, Z_gt, np.nan)
    surf_b = ax.plot_surface(T_s, J_s, Z_gt, cmap="viridis", alpha=0.88,
                              linewidth=0, antialiased=True)
    ax.set_xlabel("Day $t$", fontsize=5, labelpad=0)
    ax.set_ylabel("Location $j$", fontsize=5, labelpad=0)
    ax.set_zlabel("GT mean (days)", fontsize=4.5, labelpad=-2)
    ax.set_yticks(j_arr); ax.set_yticklabels(loc_labels, fontsize=3.5)
    ax.tick_params(axis="x", labelsize=4.5, pad=0)
    ax.tick_params(axis="z", labelsize=4.5, pad=0)
    ax.view_init(elev=32, azim=-52)
    ax.set_title("$\\bar{g}^k_{\\rm out}(t)$ — outward GT mean\n(time × location)",
                 fontsize=7, pad=6)
    _clean_panes(ax)
    fig.colorbar(surf_b, ax=ax, shrink=0.42, pad=0.06,
                 label="GT mean (days)", format="%.2f",
                 orientation="vertical", aspect=15)
    ax.text2D(-0.06, 1.06, "b", transform=ax.transAxes,
              fontsize=10, fontweight="bold", va="top", ha="left")
    # Annotate variation magnitude
    gt_range = float(np.nanmax(Z_gt) - np.nanmin(Z_gt[Z_gt > 0])) if (Z_gt > 0).any() else 0
    ax.text2D(0.02, 0.02, f"range ≈ {gt_range:.2f} d", transform=ax.transAxes,
              fontsize=5, color="0.4", style="italic")

    # ── c: GT network waterfall ───────────────────────────────────────────
    ax = fig.add_subplot(gs[1, :], projection="3d")
    cmap_wf = plt.cm.plasma
    norm_t  = plt.Normalize(float(t_sample.min()), float(t_sample.max()))
    for ti in range(n_samp):
        g_net = gt_net_pmfs[ti]
        if g_net.sum() < 0.5:
            continue
        t_val = float(t_sample[ti])
        col   = cmap_wf(norm_t(t_val))
        y_arr = np.full(max_days, t_val)
        # Filled ribbon polygon under the curve
        xs = np.concatenate([[days_a[0]], days_a, [days_a[-1]]])
        zs = np.concatenate([[0.0], g_net, [0.0]])
        ys = np.full_like(xs, t_val)
        verts_3d = [list(zip(xs, ys, zs))]
        poly = Poly3DCollection(verts_3d, alpha=0.20, facecolor=col,
                                edgecolor="none", linewidths=0)
        ax.add_collection3d(poly)
        # Top edge line
        ax.plot(days_a, y_arr, g_net, color=col, lw=0.65, alpha=0.85)
    ax.set_xlabel("Infection age $a_E$ (days)", fontsize=6, labelpad=1)
    ax.set_ylabel("Epidemic day $t$", fontsize=6, labelpad=1)
    ax.set_zlabel("$g_{\\rm net}(a_E)$", fontsize=5, labelpad=-1)
    ax.tick_params(axis="x", labelsize=5, pad=0)
    ax.tick_params(axis="y", labelsize=5, pad=0)
    ax.tick_params(axis="z", labelsize=4.5, pad=0)
    ax.view_init(elev=28, azim=-58)
    ax.set_title(
        "Network GT distribution $g_{\\rm net}(a_E,\\,t)$ waterfall — "
        "each ribbon = one time snapshot (sampled every 5 days)\n"
        "Near-parallel ribbons = time-invariance of GT shape; "
        "colour gradient = epidemic progression",
        fontsize=6.5, pad=8)
    ax.set_xlim(0, max_days - 1)
    ax.set_zlim(bottom=0)
    _clean_panes(ax)
    sm = plt.cm.ScalarMappable(cmap=cmap_wf, norm=norm_t)
    sm.set_array([])
    fig.colorbar(sm, ax=ax, shrink=0.32, pad=0.12, aspect=18,
                 label="Epidemic day $t$", orientation="horizontal",
                 format="%.0f")
    ax.text2D(-0.04, 1.05, "c", transform=ax.transAxes,
              fontsize=10, fontweight="bold", va="top", ha="left")

    plt.savefig(f"{save_prefix}_3d_surfaces.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved {save_prefix}_3d_surfaces.png")


# ══════════════════════════════════════════════════════════════════════════════
# 30c. SI FIGURE — THREE-INGREDIENT LAMBDA DECOMPOSITION
# ══════════════════════════════════════════════════════════════════════════════

def plot_SI_lambda_decomposition(sim, city_data, f_jk, params,
                                  gen_time_pmf, w_within, w_between, max_days,
                                  save_prefix="fig"):
    """
    SI figure: Three-ingredient decomposition of λ^{kl}_E(t, a_E).

    PDF Eq. 9:  K_{kj}(t,a_E) = Σ_l f_{jl}·S_j·f_{kl}·λ^{kl}_E(t,a_E)
    λ^{kl}_E(t,a_E) is composed of exactly three ingredients:
      1) 1/N^l_eff(t)   — frequency-dependent density at meeting location l
                         (N^l_eff(t) = Σ_j f_{jl}(t)·N_j)
      2) β^{kl}          — location-pair contact rate
                         (β_w = lw for l=k  /  β_b = lb for l≠k)
                         [POLYMOD: Mossong et al. 2008; Cauchemez et al. 2011]
      3) p^{kl}(a_E)     — biological infectiousness at infection age a_E
                         (p_w for household; p_b for community contacts)
                         [Hart et al. 2022 PLOS CB; Cereda et al. 2020]

    Panels (3×2):
      a  Ingredient 1: N^l_eff(t) time series — hub / mid / peripheral
         Shows weekly oscillations due to commuting patterns.
      b  Ingredient 2: β^{kl} calibrated contact rates with citations and ratio
      c  Ingredient 3: infectiousness profiles p(a_E) — w_within, w_between, overall
      d  Combined λ^{kl}_E(a_E) at peak — within vs between, all 3 ingredients labelled
      e  Kernel K_{kj}(a_E) for 4 representative pairs at epidemic peak
         Shaded area = R_{kj}; shape = g_{kj}(a_E)
      f  Within-fraction bKw / base_K heatmap at epidemic peak
         Green = household-dominated; red = community-dominated
    """
    inc    = sim["incidence"]           # (T, N)
    S_ser  = sim["susceptibles"]        # (T, N)
    lw_sim = sim["lambda_within_scaled"]
    lb_sim = sim["lambda_between_scaled"]
    coords, pops, dists, node_types, meta = city_data
    T, N   = inc.shape
    days_a = np.arange(max_days)

    # ── helper: identify hub / mid / peripheral ────────────────────────────
    peak_t   = int(inc.sum(axis=1).argmax())
    early_t  = max(1, peak_t // 3)
    i_hub    = meta.get("hub_idx",  0)
    i_per    = meta.get("periph_idx", N - 1)
    i_mid    = (i_hub + i_per) // 2

    # ── precompute N_eff time series ───────────────────────────────────────
    N_eff_ts = np.array([f_jk[t].T @ pops for t in range(T)])  # (T, N)

    # ── precompute kernel ingredients at epidemic peak ─────────────────────
    f_pk = f_jk[peak_t]
    S_pk = S_ser[peak_t]
    base_K_pk, bKw_pk, bKb_pk, N_eff_pk, inv_Neff_pk = (
        _kernel_base(f_pk, pops, lw_sim, lb_sim))
    prob_peak = params["prob_transmission_peak"]

    # K_{kj}(a) at epidemic peak
    K_pk = np.zeros((max_days, N, N))
    for a in range(max_days):
        K_pk[a] = (prob_peak * S_pk[np.newaxis, :] *
                   (bKw_pk * w_within[a] + bKb_pk * w_between[a]))
    R_pk = K_pk.sum(axis=0)          # (N, N) — same as compute_R_matrix

    # within-fraction
    with np.errstate(divide="ignore", invalid="ignore"):
        wfrac_pk = np.where(base_K_pk > 0, bKw_pk / base_K_pk, np.nan)

    # ── figure ────────────────────────────────────────────────────────────
    COLS = OKABE_ITO            # colour palette (Wong 2011)
    t_arr = np.arange(T)

    fig = plt.figure(figsize=(7.2, 9.6))
    gs  = gridspec.GridSpec(3, 2, hspace=0.74, wspace=0.52,
                            left=0.09, right=0.97, top=0.96, bottom=0.06)

    # ─────────────────────────────────────────────────────────────────────
    # Panel a — Ingredient 1: N^l_eff(t) time series
    # ─────────────────────────────────────────────────────────────────────
    ax = fig.add_subplot(gs[0, 0])

    show_locs  = [i_hub, i_mid, i_per]
    show_lbls  = [f"Hub L{i_hub+1} ({node_types[i_hub]})",
                  f"Mid L{i_mid+1} ({node_types[i_mid]})",
                  f"Periph L{i_per+1} ({node_types[i_per]})"]
    show_cols  = [COLS[0], COLS[2], COLS[5]]

    for idx, (loc, lbl, col) in enumerate(
            zip(show_locs, show_lbls, show_cols)):
        ax.plot(t_arr, N_eff_ts[:, loc] / 1e3, color=col, lw=1.0,
                label=lbl, alpha=0.9)

    ax.axvline(peak_t, color="0.55", ls="--", lw=0.8, alpha=0.7)
    ax.text(peak_t + 1, ax.get_ylim()[1] * 0.95, f"peak\nd{peak_t}",
            fontsize=4.5, color="0.5", va="top")

    ax.set_xlabel("Day $t$")
    ax.set_ylabel("$N^l_{\\rm eff}(t)$ ($\\times 10^3$)")
    ax.set_title("Ingredient 1: effective density\n"
                 "$N^l_{\\rm eff}(t)=\\sum_j f_{jl}(t)\\,N_j$",
                 fontsize=6.5, pad=3)
    ax.legend(fontsize=5.2, borderpad=0.3, labelspacing=0.15, loc="lower right")
    ax.text(0.02, 0.97,
            ("Weekly oscillations: weekday commuting\n"
             "↑ $N^l_{\\rm eff}$ at hubs; weekend ↓\n"
             "→ $\\lambda^{kl}_{E}\\propto 1/N^l_{\\rm eff}$ oscillates inversely"),
            transform=ax.transAxes, fontsize=4.5, va="top", ha="left",
            color="0.4", style="italic")
    _panel_label(ax, "a")

    # ─────────────────────────────────────────────────────────────────────
    # Panel b — Ingredient 2: β^{kl} calibrated contact rates
    # ─────────────────────────────────────────────────────────────────────
    ax = fig.add_subplot(gs[0, 1])

    base_rate  = params["base_contact_rate"]   # 13.0 (pre-calibration)
    # Determine calibration scale from sim
    scale_val  = lw_sim / base_rate

    labels_b   = ["$\\beta$ (POLYMOD\nbase)",
                  "$\\beta_{\\rm w}$ (within,\ncalibrated)",
                  "$\\beta_{\\rm b}$ (between,\ncalibrated)"]
    values_b   = [base_rate, lw_sim, lb_sim]
    bar_cols_b = [COLS[4], COLS[0], COLS[5]]
    bars = ax.bar(labels_b, values_b, color=bar_cols_b,
                  alpha=0.85, edgecolor="none", width=0.55)
    for bar, val in zip(bars, values_b):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.15,
                f"{val:.2f}", ha="center", va="bottom", fontsize=7.0,
                fontweight="bold")

    ax.set_ylabel("Contacts per day ($\\beta$)", fontsize=8)
    ax.set_ylim(0, max(values_b) * 1.52)
    ax.text(0.97, 0.97,
            (f"$\\beta_{{\\rm b}}/\\beta_{{\\rm w}} = {lb_sim/lw_sim:.2f}$"
             " (Cauchemez 2011)\n"
             f"Calibration scale = {scale_val:.3f}\n"
             f"(to achieve $R_0={params['R0_target']}$)\n\n"
             "Refs:\n"
             "• POLYMOD (Mossong 2008 PLOS Med)\n"
             "• Cauchemez et al. 2011 Nat Med\n"
             "• Ferguson et al. 2005 Nature"),
            transform=ax.transAxes, fontsize=4.5, ha="right", va="top",
            color="0.40", style="italic")
    ax.set_title("Ingredient 2: location-pair contact rate\n"
                 "$\\beta^{kl}$: within $l=k$ vs between $l\\neq k$",
                 fontsize=6.5, pad=3)
    _panel_label(ax, "b")

    # ─────────────────────────────────────────────────────────────────────
    # Panel c — Ingredient 3: infectiousness profiles p(a_E)
    # ─────────────────────────────────────────────────────────────────────
    ax = fig.add_subplot(gs[1, 0])

    # primary axis: PDF
    mu_ww = float(days_a @ w_within)
    mu_wb = float(days_a @ w_between)
    mu_gp = float(days_a @ gen_time_pmf)

    ax.plot(days_a, w_within,    color=COLS[0], lw=1.2, ls="--",
            label=f"$p_{{\\rm w}}(a_E)$ household  $\\bar{{a}}={mu_ww:.1f}\\,$d")
    ax.plot(days_a, w_between,   color=COLS[5], lw=1.2, ls=":",
            label=f"$p_{{\\rm b}}(a_E)$ community  $\\bar{{a}}={mu_wb:.1f}\\,$d")
    ax.plot(days_a, gen_time_pmf, color=COLS[4], lw=1.4,
            label=f"$p(a_E)$ overall  $\\bar{{a}}={mu_gp:.1f}\\,$d")

    # vertical mean lines
    for mu, col in [(mu_ww, COLS[0]), (mu_wb, COLS[5]), (mu_gp, COLS[4])]:
        ax.axvline(mu, color=col, lw=0.6, ls="-", alpha=0.55)

    ax.set_xlabel("Infection age $a_E$ (days)")
    ax.set_ylabel("Probability")
    ax.set_xlim(0, max_days - 1)
    ax.legend(fontsize=5.0, borderpad=0.3, labelspacing=0.15)
    ax.set_title("Ingredient 3: biological infectiousness\n"
                 "$p^{kl}(a_E)$: household vs community profiles",
                 fontsize=6.5, pad=3)
    ax.text(0.97, 0.97,
            ("Hart et al. 2022 PLOS CB\n"
             "(overall $p$, mean 5.5 d)\n"
             "Cereda et al. 2020\n"
             "(household $p_{{\\rm w}}$, mean 4.0 d)"),
            transform=ax.transAxes, fontsize=4.5, ha="right", va="top",
            color="0.40", style="italic")
    _panel_label(ax, "c")

    # ─────────────────────────────────────────────────────────────────────
    # Panel d — Combined: λ^{kl}_E(a_E) at epidemic peak (all 3 ingredients)
    # ─────────────────────────────────────────────────────────────────────
    ax = fig.add_subplot(gs[1, 1])

    # Representative N_eff: use the median location at peak
    N_eff_med = float(np.median(N_eff_pk))
    # λ_w(a_E) = β_w × p_w(a_E) / N_eff   [l = k, within-home meeting]
    lam_w = lw_sim * prob_peak * w_within  / N_eff_med
    # λ_b(a_E) = β_b × p_b(a_E) / N_eff   [l ≠ k, away meeting]
    lam_b = lb_sim * prob_peak * w_between / N_eff_med

    ax.plot(days_a, lam_w * 1e5, color=COLS[0], lw=1.3, ls="--",
            label=(f"$\\lambda_{{\\rm w}}^{{kk}}(a_E)$  "
                   f"($\\int=${(lam_w.sum()*1e5):.3f}$\\times10^{{-5}}$)"))
    ax.plot(days_a, lam_b * 1e5, color=COLS[5], lw=1.3, ls=":",
            label=(f"$\\lambda_{{\\rm b}}^{{kl}}(a_E)$  "
                   f"($\\int=${(lam_b.sum()*1e5):.3f}$\\times10^{{-5}}$)"))
    ax.fill_between(days_a, lam_w * 1e5, alpha=0.12, color=COLS[0])
    ax.fill_between(days_a, lam_b * 1e5, alpha=0.12, color=COLS[5])

    # Annotate the three ingredients with arrows
    peak_a_w = int(np.argmax(w_within))
    peak_a_b = int(np.argmax(w_between))
    ax.annotate("$\\beta_{\\rm w}/N^k_{\\rm eff}$\n(Ingred. 1+2)",
                xy=(peak_a_w, float(lam_w[peak_a_w] * 1e5)),
                xytext=(peak_a_w + 3, float(lam_w[peak_a_w] * 1e5) * 1.35),
                fontsize=4.5, color=COLS[0], ha="left",
                arrowprops=dict(arrowstyle="-", color=COLS[0],
                                lw=0.7, alpha=0.8))
    ax.annotate("$p_{\\rm w}(a_E)$ shape\n(Ingred. 3)",
                xy=(peak_a_w, float(lam_w[peak_a_w] * 1e5)),
                xytext=(peak_a_w - 1, float(lam_w[peak_a_w] * 1e5) * 0.5),
                fontsize=4.5, color=COLS[0], ha="right",
                arrowprops=dict(arrowstyle="-", color=COLS[0],
                                lw=0.7, alpha=0.8))
    ax.set_xlabel("Infection age $a_E$ (days)")
    ax.set_ylabel("$\\lambda^{kl}_E(a_E)$ ($\\times 10^{-5}$/day)")
    ax.set_xlim(0, max_days - 1)
    ax.legend(fontsize=5.0, borderpad=0.3, labelspacing=0.15)
    ax.set_title(
        "Combined $\\lambda^{kl}_E = \\beta^{kl}\\,p^{kl}(a_E)/N^l_{\\rm eff}$\n"
        f"at peak (day {peak_t}; median $N_{{\\rm eff}}=${N_eff_med/1e3:.1f}k)",
        fontsize=6.5, pad=3)
    ax.text(0.97, 0.35,
            ("$\\int \\lambda^{kl}_E\\,da_E$ = per-contact\n"
             "transmission probability\n"
             "(summed over infection life)"),
            transform=ax.transAxes, fontsize=4.3, ha="right", va="bottom",
            color="0.4", style="italic")
    _panel_label(ax, "d")

    # ─────────────────────────────────────────────────────────────────────
    # Panel e — K_{kj}(a_E) for 4 representative infector–infectee pairs
    # ─────────────────────────────────────────────────────────────────────
    ax = fig.add_subplot(gs[2, 0])

    # Four pairs: hub→hub, periph→periph, hub→periph, periph→hub
    pairs = [
        (i_hub,  i_hub,  COLS[0], "-",  f"hub→hub L{i_hub+1}→L{i_hub+1}"),
        (i_per,  i_per,  COLS[5], "--", f"periph→periph L{i_per+1}→L{i_per+1}"),
        (i_hub,  i_per,  COLS[2], "-.", f"hub→periph L{i_hub+1}→L{i_per+1}"),
        (i_per,  i_hub,  COLS[1], ":", f"periph→hub L{i_per+1}→L{i_hub+1}"),
    ]
    scale_k = 1e4   # display scale
    for (k, j, col, ls, lbl) in pairs:
        K_kj = K_pk[:, k, j]
        R_kj = float(R_pk[k, j])
        if R_kj > 1e-12:
            ax.plot(days_a, K_kj * scale_k, color=col, lw=1.1, ls=ls,
                    label=f"{lbl}  $R_{{kj}}={R_kj:.2f}$")
            ax.fill_between(days_a, K_kj * scale_k, alpha=0.07, color=col)

    ax.set_xlabel("Infection age $a_E$ (days)")
    ax.set_ylabel(f"$K_{{kj}}(a_E)$ ($\\times 10^{{-{int(np.log10(scale_k))}}}$)")
    ax.set_xlim(0, max_days - 1)
    ax.legend(fontsize=4.5, borderpad=0.3, labelspacing=0.15, loc="upper right")
    ax.set_title(
        f"Kernel $K_{{kj}}(t_{{\\rm pk}}, a_E)$ — representative pairs\n"
        f"$\\int K_{{kj}}\\,da_E = R_{{kj}}(t_{{\\rm pk}})$  (day {peak_t})",
        fontsize=6.5, pad=3)
    ax.text(0.98, 0.56,
            ("Filled area $=R_{kj}$\n"
             "Shape $=g_{kj}(a_E)$\n"
             "hub→periph later peak\n"
             "due to community $p_b$"),
            transform=ax.transAxes, fontsize=4.5, ha="right", va="top",
            color="0.4", style="italic")
    _panel_label(ax, "e")

    # ─────────────────────────────────────────────────────────────────────
    # Panel f — Within-fraction bKw/base_K heatmap
    # ─────────────────────────────────────────────────────────────────────
    ax = fig.add_subplot(gs[2, 1])

    im = ax.imshow(wfrac_pk, vmin=0.0, vmax=1.0,
                   cmap="RdYlGn", origin="upper", aspect="equal",
                   interpolation="nearest")
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.ax.tick_params(labelsize=5)
    cb.set_label("Within-fraction $\\beta_{\\rm w}/(\\beta_{\\rm w}+\\beta_{\\rm b})$",
                 fontsize=5.5)

    loc_ticks = list(range(N))
    ax.set_xticks(loc_ticks)
    ax.set_yticks(loc_ticks)
    ax.set_xticklabels([f"L{i+1}" for i in range(N)], fontsize=4.5,
                       rotation=45, ha="right")
    ax.set_yticklabels([f"L{i+1}" for i in range(N)], fontsize=4.5)
    ax.set_xlabel("Infectee $j$", fontsize=7)
    ax.set_ylabel("Infector $k$", fontsize=7)
    ax.set_title(
        f"Within-fraction $\\beta_{{\\rm w}}K^{{\\rm w}}_{{kj}}/K^{{\\rm base}}_{{kj}}$\n"
        f"at peak (day {peak_t}) — household vs community mix",
        fontsize=6.5, pad=3)

    # Mean within-fraction annotation
    mean_wf = float(np.nanmean(wfrac_pk))
    ax.text(0.03, 0.97,
            (f"Mean = {mean_wf:.2f}\n"
             "Green: household-dominated\n"
             "→ GT shape closer to $p_w$\n"
             "Red: community-dominated\n"
             "→ GT shape closer to $p_b$"),
            transform=ax.transAxes, fontsize=4.5, ha="left", va="top",
            color="0.35", style="italic")
    _panel_label(ax, "f")

    # ── super-title ───────────────────────────────────────────────────────
    fig.text(0.50, 0.993,
             ("Three-ingredient decomposition of $\\lambda^{kl}_E(t,a_E) = "
              "\\beta^{kl}\\,p^{kl}(a_E)/N^l_{\\rm eff}(t)$ [PDF Eq. 4/9]"),
             ha="center", va="top", fontsize=7.5, fontweight="bold")

    plt.savefig(f"{save_prefix}_SI_lambda_decomp.png",
                dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved {save_prefix}_SI_lambda_decomp.png")


# ══════════════════════════════════════════════════════════════════════════════
# 30. SI FIGURE 5 — COMBINED MEETING-LOCATION ANALYSIS (BOTH SCENARIOS)
# ══════════════════════════════════════════════════════════════════════════════

def plot_SI5_combined(sim_A, sim_B, city_A, city_B, f_A, f_B,
                      prob_peak, infect_profile, lw_A, lb_A, lw_B, lb_B,
                      save_prefix="fig"):
    """SI Figure 5: Meeting-location analysis for both scenarios.

    Panels (2×2):
      a  Dense urban  — R^l_meeting heatmap
      b  Dense urban  — R^j_in vs R^l_meeting scatter at peak
      c  Sparse national — R^l_meeting heatmap
      d  Sparse national — R^j_in vs R^l_meeting scatter at peak
    """
    fig = plt.figure(figsize=(7.2, 5.5))
    gs  = gridspec.GridSpec(2, 2, hspace=0.62, wspace=0.52,
                            left=0.08, right=0.97, top=0.95, bottom=0.08)

    scenarios = [
        (sim_A, city_A, lw_A, lb_A, "Dense urban",     0, OKABE_ITO[0]),
        (sim_B, city_B, lw_B, lb_B, "Sparse national", 1, OKABE_ITO[4]),
    ]
    panel_pairs = [("a", "b"), ("c", "d")]

    for (sim, city_d, lw, lb, name, row, col_theme), (lbl_heat, lbl_scat) in zip(scenarios, panel_pairs):
        inc      = sim["incidence"]
        R_mats   = sim["R_matrices"]
        R_meet_s = sim["R_meeting_series"]
        coords, pops, dists, node_types, meta = city_d
        T, N  = inc.shape
        peak  = int(inc.sum(axis=1).argmax())
        loc   = [f"L{i+1}" for i in range(N)]

        # Heatmap panel
        ax = fig.add_subplot(gs[row, 0])
        im = ax.imshow(R_meet_s.T, cmap="YlOrRd", aspect="auto", origin="upper")
        ax.set_yticks(range(N)); ax.set_yticklabels(loc, fontsize=5)
        ax.set_xlabel("Day $t$"); ax.set_ylabel("Activity location $l$")
        cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cb.ax.set_title("$R^l_{\\rm meeting}$", fontsize=6, pad=3)
        ax.text(0.02, 0.98, name, transform=ax.transAxes, fontsize=6,
                fontweight="bold", va="top", color=col_theme)
        _panel_label(ax, lbl_heat)

        # Scatter panel
        ax = fig.add_subplot(gs[row, 1])
        R_in_p   = R_inward(R_mats[peak])
        R_meet_p = R_meet_s[peak]
        for j in range(N):
            ax.scatter(R_in_p[j], R_meet_p[j], s=35,
                       color=OKABE_ITO[j % len(OKABE_ITO)],
                       edgecolors="k", linewidths=0.4, zorder=5)
            ax.annotate(loc[j], (R_in_p[j], R_meet_p[j]),
                        fontsize=5.5, xytext=(3, 3), textcoords="offset points",
                        color=OKABE_ITO[j % len(OKABE_ITO)])
        all_v = np.concatenate([R_in_p, R_meet_p])
        vmax_v = all_v.max() * 1.1
        ax.plot([0, vmax_v], [0, vmax_v], color="0.55", ls="--", lw=0.8)
        ax.set_xlabel("$R^j_{\\rm in}$  (residence-based)")
        ax.set_ylabel("$R^l_{\\rm meeting}$  (activity-based)")
        ax.text(0.05, 0.97, f"{name} — peak (day {peak})",
                transform=ax.transAxes, fontsize=5.5, va="top", style="italic",
                color=col_theme)
        _panel_label(ax, lbl_scat)

    fig.suptitle(
        "SI Figure 5: Meeting-location reproduction number $R^l_{\\rm meeting}$ "
        "for Dense urban (top) and Sparse national (bottom).",
        fontsize=6, y=0.99, ha="center", style="italic", color="0.3")
    fname = f"{save_prefix}_SI5_meeting_combined.png"
    plt.savefig(fname, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved {fname}")


# ══════════════════════════════════════════════════════════════════════════════
# 30. FIGURE — NON-NORMAL MOBILITY COUNTERFACTUAL (Scenario C)
# ══════════════════════════════════════════════════════════════════════════════

def plot_counterfactual_nonnormal(sim_A, sim_C, city_A, f_A, f_C,
                                   w_within, w_between,
                                   max_days, lw_A, lb_A, lw_C, lb_C,
                                   params, save_prefix="fig"):
    """Counterfactual comparison: baseline (A) vs hub-amplified non-normal (C).

    Layout 3×3:
      a  Incidence curves (all locations, both scenarios)
      b  Network-level R(t) comparison
      c  Reactivity σ(t) and R(t) with transient zone (sigma>1, R<1)
      d  Generation time distributions at epidemic peak (both scenarios)
      e  R_kj heatmap at peak — Baseline A
      f  R_kj heatmap at peak — Counterfactual C
      g  Mixing ratio s(t) = |λ_2|/ρ comparison
      h  E_kj (elasticity) heatmap at peak — Baseline A
      i  E_kj (elasticity) heatmap at peak — Counterfactual C

    Non-normal counterfactual inspired by Colizza et al. (2006 Science) hub-driven
    epidemic dynamics: stronger hub attraction creates more directed, asymmetric
    flows and hence larger non-normality in R_mat.
    """
    inc_A  = sim_A["incidence"];     inc_C  = sim_C["incidence"]
    R_A    = sim_A["R_matrices"];    R_C    = sim_C["R_matrices"]
    T, N   = inc_A.shape

    peak_A = int(inc_A.sum(axis=1).argmax())
    peak_C = int(inc_C.sum(axis=1).argmax())

    # ── pre-compute spectral quantities ───────────────────────────────────
    def _get_spectral(R_mats, T_len):
        rho_t   = np.array([R_system(R_mats[t]) for t in range(T_len)])
        sigma_t = np.array([reactivity(R_mats[t])["sigma"] for t in range(T_len)])
        mix_t   = np.array([spectral_analysis(R_mats[t])["mixing_ratio"] for t in range(T_len)])
        return rho_t, sigma_t, mix_t

    rho_A, sigma_A, mix_A = _get_spectral(R_A, T)
    rho_C, sigma_C, mix_C = _get_spectral(R_C, T)

    # ── pre-compute GT at peak ─────────────────────────────────────────────
    prob_peak = params["prob_transmission_peak"]
    pops_A    = city_A[1]

    gt_A = compute_generation_times(
        f_A[peak_A], sim_A["susceptibles"][peak_A], pops_A,
        prob_peak, w_within, w_between, max_days, lw_A, lb_A)
    # Use same populations for C (same city geometry)
    gt_C = compute_generation_times(
        f_C[peak_C], sim_C["susceptibles"][peak_C], pops_A,
        prob_peak, w_within, w_between, max_days, lw_C, lb_C)

    # ── elasticity at peak ────────────────────────────────────────────────
    ela_A = sensitivity_elasticity(R_A[peak_A])
    ela_C = sensitivity_elasticity(R_C[peak_C])

    # ── figure layout ──────────────────────────────────────────────────────
    fig = plt.figure(figsize=(7.2, 8.0))
    gs  = gridspec.GridSpec(3, 3, hspace=0.75, wspace=0.55,
                            left=0.09, right=0.97, top=0.96, bottom=0.07)

    t_arr = np.arange(T)
    col_A = OKABE_ITO[4]   # blue  — baseline
    col_C = OKABE_ITO[5]   # red   — counterfactual

    # ── a: Incidence curves ────────────────────────────────────────────────
    ax = fig.add_subplot(gs[0, 0])
    inc_tot_A = inc_A.sum(axis=1)
    inc_tot_C = inc_C.sum(axis=1)
    ax.plot(t_arr, inc_tot_A / 1e3, color=col_A, lw=1.2, label="Baseline (A)")
    ax.plot(t_arr, inc_tot_C / 1e3, color=col_C, lw=1.2, ls="--", label="Hub-amplified (C)")
    ax.axvline(peak_A, color=col_A, lw=0.7, ls=":", alpha=0.7)
    ax.axvline(peak_C, color=col_C, lw=0.7, ls=":", alpha=0.7)
    ax.set_xlabel("Day $t$")
    ax.set_ylabel("Daily incidence (×10³)")
    ax.set_title("Incidence", fontsize=7, pad=3)
    ax.legend(fontsize=5.5, borderpad=0.3)
    _panel_label(ax, "a")

    # ── b: Network-level R(t) ──────────────────────────────────────────────
    ax = fig.add_subplot(gs[0, 1])
    ax.plot(t_arr, rho_A, color=col_A, lw=1.0, label="$\\mathcal{R}(t)$ Baseline")
    ax.plot(t_arr, rho_C, color=col_C, lw=1.0, ls="--", label="$\\mathcal{R}(t)$ Hub-amp.")
    ax.axhline(1.0, color="0.55", ls="--", lw=0.7)
    ax.set_xlabel("Day $t$")
    ax.set_ylabel("$\\mathcal{R}(t)$")
    ax.set_title("Network reproduction number", fontsize=7, pad=3)
    ax.legend(fontsize=5.5, borderpad=0.3)
    _panel_label(ax, "b")

    # ── c: Sigma(t) and R(t) with transient zone ───────────────────────────
    ax = fig.add_subplot(gs[0, 2])
    ax.plot(t_arr, rho_C,   color=col_C, lw=1.0, label="$\\mathcal{R}(t)$")
    ax.plot(t_arr, sigma_C, color=OKABE_ITO[6], lw=1.0, ls="-.",
            label="$\\sigma(t)$ (reactivity)")
    ax.axhline(1.0, color="0.55", ls="--", lw=0.7)
    transient_mask = (sigma_C > 1) & (rho_C < 1)
    if transient_mask.any():
        ax.fill_between(t_arr, 1.0, sigma_C, where=transient_mask,
                        color=OKABE_ITO[0], alpha=0.30,
                        label="$\\sigma>1, \\mathcal{R}<1$\n(transient zone)")
    ax.set_xlabel("Day $t$")
    ax.set_ylabel("Value")
    ax.set_title("Transience: hub-amplified (C)", fontsize=7, pad=3)
    ax.legend(fontsize=5.5, borderpad=0.3, labelspacing=0.15)
    _panel_label(ax, "c")

    # ── d: GT distributions at peak ────────────────────────────────────────
    ax = fig.add_subplot(gs[1, 0])
    days_arr = np.arange(max_days)
    ax.plot(days_arr, gt_A["g_network"], color=col_A, lw=1.1,
            label=f"Baseline $\\bar{{g}}$={float(np.sum(days_arr*gt_A['g_network'])):.1f}d")
    ax.plot(days_arr, gt_C["g_network"], color=col_C, lw=1.1, ls="--",
            label=f"Hub-amp. $\\bar{{g}}$={float(np.sum(days_arr*gt_C['g_network'])):.1f}d")
    ax.plot(days_arr, w_within,  color=OKABE_ITO[0], lw=0.8, ls=":",
            alpha=0.7, label="$p_w$ (household)")
    ax.plot(days_arr, w_between, color=OKABE_ITO[5], lw=0.8, ls=":",
            alpha=0.7, label="$p_b$ (community)")
    ax.set_xlabel("Infection age $a_E$ (days)")
    ax.set_ylabel("Probability")
    ax.set_title("Gen. time at peak", fontsize=7, pad=3)
    ax.legend(fontsize=5.0, borderpad=0.3, labelspacing=0.1)
    _panel_label(ax, "d")

    # ── e/f: R_kj heatmaps at peak ────────────────────────────────────────
    R_pk_A = R_A[peak_A];  R_pk_C = R_C[peak_C]
    vmax_R  = np.percentile(np.concatenate([R_pk_A.flatten(), R_pk_C.flatten()]), 98)

    for col_idx, (R_pk, label_sc, pklabel) in enumerate(
            [(R_pk_A, "Baseline (A)", f"day {peak_A}"),
             (R_pk_C, "Hub-amplified (C)", f"day {peak_C}")]):
        ax = fig.add_subplot(gs[1, col_idx + 1])
        im = ax.imshow(R_pk, vmin=0, vmax=vmax_R, cmap="plasma",
                       origin="upper", aspect="equal", interpolation="nearest")
        ax.set_xticks(range(N)); ax.set_yticks(range(N))
        ax.set_xticklabels([f"L{i+1}" for i in range(N)], fontsize=4, rotation=45)
        ax.set_yticklabels([f"L{i+1}" for i in range(N)], fontsize=4)
        ax.set_xlabel("Infectee $j$", fontsize=7)
        ax.set_ylabel("Infector $k$", fontsize=7)
        ax.set_title(f"$R_{{kj}}$ — {label_sc}\n({pklabel})", fontsize=6, pad=3)
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cb.ax.tick_params(labelsize=5)
        _panel_label(ax, "ef"[col_idx])

    # ── g: Mixing ratio s(t) ──────────────────────────────────────────────
    ax = fig.add_subplot(gs[2, 0])
    vm_A = mix_A > 0;  vm_C = mix_C > 0
    ax.plot(t_arr[vm_A], mix_A[vm_A], color=col_A, lw=1.0, label="Baseline")
    ax.plot(t_arr[vm_C], mix_C[vm_C], color=col_C, lw=1.0, ls="--", label="Hub-amp.")
    ax.set_xlabel("Day $t$")
    ax.set_ylabel("$s(t)=|\\lambda_2|/\\mathcal{R}$")
    ax.set_title("Mixing ratio $s(t)$", fontsize=7, pad=3)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=5.5, borderpad=0.3)
    _panel_label(ax, "g")

    # ── h/i: Elasticity heatmaps at peak ──────────────────────────────────
    E_pk_A = ela_A["elasticity"];  E_pk_C = ela_C["elasticity"]
    vmax_E = np.percentile(np.concatenate([E_pk_A.flatten(), E_pk_C.flatten()]), 98)

    for col_idx, (E_pk, label_sc, pklabel) in enumerate(
            [(E_pk_A, "Baseline (A)", f"day {peak_A}"),
             (E_pk_C, "Hub-amplified (C)", f"day {peak_C}")]):
        ax = fig.add_subplot(gs[2, col_idx + 1])
        im = ax.imshow(E_pk, vmin=0, vmax=vmax_E, cmap="viridis",
                       origin="upper", aspect="equal", interpolation="nearest")
        ax.set_xticks(range(N)); ax.set_yticks(range(N))
        ax.set_xticklabels([f"L{i+1}" for i in range(N)], fontsize=4, rotation=45)
        ax.set_yticklabels([f"L{i+1}" for i in range(N)], fontsize=4)
        ax.set_xlabel("Infectee $j$", fontsize=7)
        ax.set_ylabel("Infector $k$", fontsize=7)
        ax.set_title(f"$E_{{kj}}$ (elasticity) — {label_sc}\n({pklabel})", fontsize=6, pad=3)
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cb.ax.tick_params(labelsize=5)
        _panel_label(ax, "hi"[col_idx])

    plt.savefig(f"{save_prefix}_counterfactual_nonnormal.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved {save_prefix}_counterfactual_nonnormal.png")


# ══════════════════════════════════════════════════════════════════════════════
# 25. MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 72)
    print("MOBILITY-INFORMED RENEWAL EQUATIONS — DIRECTLY TRANSMITTED DISEASES")
    print("=" * 72)

    N_LOC    = 10
    T        = 150
    SEED     = 42
    params   = COVID_PARAMS
    max_days = params["max_gen_time"]
    # λ_within: per-contact transmission rate at home location = POLYMOD community rate
    # λ_between: inter-location contacts ~30% as intense as home contacts
    #   [Cauchemez et al. 2011 Nat Med; Ferguson et al. 2005 Nature; Keeling & Rohani 2008]
    BETWEEN_FRACTION = 0.30
    LW       = params["base_contact_rate"]          # 13.0 contacts/day [POLYMOD; Mossong et al. 2008]
    LB       = params["base_contact_rate"] * BETWEEN_FRACTION

    # ── infectiousness profile ─────────────────────────────────────────────
    gen_time_pmf   = discretise_gamma(params["gen_time_mean"],
                                       params["gen_time_sd"], max_days)
    infect_profile = gen_time_pmf.copy()   # w(a_E): surv=1 absorbed
    w_within  = discretise_gamma(params["within_gt_mean"],  params["within_gt_sd"],  max_days)
    w_between = discretise_gamma(params["between_gt_mean"], params["between_gt_sd"], max_days)
    print(f"\n  GT mean = {float(np.sum(np.arange(max_days)*gen_time_pmf)):.2f} d")

    # ── Scenario A: Dense urban ───────────────────────────────────────────
    print("\n─── Scenario A: Dense urban ───")
    city_A = generate_city(N_LOC, scenario="lagos", seed=SEED)
    coords_A, pops_A, dists_A, types_A, meta_A = city_A
    print(f"  Node types: {types_A}")
    print(f"  Pops: {pops_A.astype(int)}")
    print(f"  Commuting fracs: {np.round(meta_A['commuting_fracs'], 3)}")

    f_A, base_fA = generate_mobility(N_LOC, T, pops_A, dists_A, types_A, meta_A,
                                      day_variation_sd=0.15, seed=SEED)
    print(f"  Row-stochastic: {np.allclose(f_A.sum(axis=-1), 1.0)}")

    R0_A = compute_R_matrix(f_A[0], pops_A, pops_A,
                             params["prob_transmission_peak"], infect_profile, LW, LB)
    spec0_A = spectral_analysis(R0_A)
    print(f"  ρ(R₀) = {spec0_A['rho']:.4f}  s = {spec0_A['mixing_ratio']:.4f}")

    initial_A = np.zeros(N_LOC);  initial_A[0] = 10
    sim_A = simulate_epidemic_pde(
        T, N_LOC, pops_A, f_A,
        params["prob_transmission_peak"], infect_profile, max_days,
        params["R0_target"], initial_A, LW, LB, w_within, w_between,
        birth_rate=0.00003, death_rate=0.00003,
        stochastic=True, susceptible_depletion=True, seed=SEED)

    lw_A, lb_A = sim_A["lambda_within_scaled"], sim_A["lambda_between_scaled"]
    inc_A = sim_A["incidence"]
    pk_A  = int(inc_A.sum(axis=1).argmax())
    print(f"  Total infections: {inc_A.sum():.0f}  "
          f"Peak day: {pk_A}  "
          f"Attack rate: {inc_A.sum()/pops_A.sum()*100:.1f}%")

    # ── Scenario B: Zambia-like sparse ────────────────────────────────────
    print("\n─── Scenario B: Sparse national ───")
    city_B = generate_city(N_LOC, scenario="zambia", seed=SEED)
    coords_B, pops_B, dists_B, types_B, meta_B = city_B
    print(f"  Node types: {types_B}")
    print(f"  Commuting fracs: {np.round(meta_B['commuting_fracs'], 3)}")

    f_B, base_fB = generate_mobility(N_LOC, T, pops_B, dists_B, types_B, meta_B,
                                      day_variation_sd=0.12, seed=SEED)
    initial_B = np.zeros(N_LOC);  initial_B[0] = 10
    sim_B = simulate_epidemic_pde(
        T, N_LOC, pops_B, f_B,
        params["prob_transmission_peak"], infect_profile, max_days,
        params["R0_target"], initial_B, LW, LB, w_within, w_between,
        birth_rate=0.00003, death_rate=0.00003,
        stochastic=True, susceptible_depletion=True, seed=SEED)

    lw_B, lb_B = sim_B["lambda_within_scaled"], sim_B["lambda_between_scaled"]
    inc_B = sim_B["incidence"]
    print(f"  Attack rate (B): {inc_B.sum()/pops_B.sum()*100:.1f}%  "
          f"Peak day: {int(inc_B.sum(axis=1).argmax())}")

    # ── Zambia mobility validation ────────────────────────────────────────
    # Checks whether meaningful between-location transmission occurs despite
    # very low commuting fracs (1.5–8%) over large distances (~600 km).
    # With lw/N_eff per-capita rates, f[j→k] ~1–4% for rural nodes means
    # cross-location R_kj is ~1–4% of home-location R. Epidemics seeded
    # at the capital (loc 0) will predominantly burn locally; rural spread
    # depends on per-day flow × force of infection at the capital.
    print("\n─── Zambia mobility validation ───")
    off_diag_mask = ~np.eye(N_LOC, dtype=bool)
    off_vals_B    = base_fB[off_diag_mask]
    print(f"  Home fracs (diagonal): {np.diag(base_fB).round(3)}")
    print(f"  Off-diagonal f range:  [{off_vals_B.min():.6f}, {off_vals_B.max():.6f}]"
          f"  mean={off_vals_B.mean():.6f}")
    imat_B = sim_B["incidence_matrix"]
    between_B = sum(imat_B[t].sum() - imat_B[t].trace() for t in range(T))
    within_B  = sum(imat_B[t].trace() for t in range(T))
    total_B   = between_B + within_B
    print(f"  Between-location infections: {between_B:.1f}  "
          f"({between_B / (total_B + 1e-10) * 100:.2f}% of total)")
    print(f"  Within-location infections:  {within_B:.1f}  "
          f"({within_B  / (total_B + 1e-10) * 100:.2f}% of total)")
    print(f"  Interpretation: low commuting fracs + exponential distance"
          f" decay → valid sparse-network behaviour, not a bug.")

    # ── Scenario C: Hub-amplified non-normal mobility ─────────────────────
    print("\n─── Scenario C: Hub-amplified dense urban (non-normal counterfactual) ───")
    # Inspired by Colizza et al. (2006 Science) hub-driven metapopulation dynamics.
    # Increased hub attraction (hub_mult_scale=2.4x: 2.5→6.0) and commuting fracs
    # (×1.5 for core/dense nodes) to maximise non-normality of R_mat.
    f_C, base_fC = generate_mobility(N_LOC, T, pops_A, dists_A, types_A, meta_A,
                                      day_variation_sd=0.20, seed=SEED+1,
                                      hub_mult_scale=2.4, commuting_frac_scale=1.5)
    initial_C = np.zeros(N_LOC);  initial_C[0] = 10
    sim_C = simulate_epidemic_pde(
        T, N_LOC, pops_A, f_C,
        params["prob_transmission_peak"], infect_profile, max_days,
        params["R0_target"], initial_C, LW, LB, w_within, w_between,
        birth_rate=0.00003, death_rate=0.00003,
        stochastic=True, susceptible_depletion=True, seed=SEED+1)
    lw_C, lb_C = sim_C["lambda_within_scaled"], sim_C["lambda_between_scaled"]
    inc_C = sim_C["incidence"]
    pk_C  = int(inc_C.sum(axis=1).argmax())
    print(f"  Total infections: {inc_C.sum():.0f}  "
          f"Peak day: {pk_C}  "
          f"Attack rate: {inc_C.sum()/pops_A.sum()*100:.1f}%")
    spec0_C = spectral_analysis(compute_R_matrix(f_C[0], pops_A, pops_A,
                 params["prob_transmission_peak"], infect_profile, LW, LB))
    react_C = reactivity(compute_R_matrix(f_C[0], pops_A, pops_A,
                 params["prob_transmission_peak"], infect_profile, LW, LB))
    print(f"  ρ(R₀)_C = {spec0_C['rho']:.4f}  σ/ρ_C = {react_C['amplification_ratio']:.4f}")

    # ── Independent R estimation ───────────────────────────────────────────
    print("\n─── Independent R̂(t) estimation ───")
    R_ind_A = estimate_R_independent(inc_A, gen_time_pmf, window=7)
    R_in_s  = np.array([R_inward(sim_A["R_matrices"][t]) for t in range(T)])
    for j in range(min(N_LOC, 4)):
        ind_m = float(np.nanmean(R_ind_A[30:60, j]))
        mob_m = float(np.nanmean(R_in_s[30:60, j]))
        bias  = (ind_m - mob_m) / mob_m * 100 if mob_m > 0 else np.nan
        print(f"    L{j+1}: Indep={ind_m:.3f}  R^in={mob_m:.3f}  Bias={bias:+.1f}%")

    # ── GT snapshots ──────────────────────────────────────────────────────
    print("\n─── GT snapshots ───")
    early_A = max(1, pk_A // 3);  late_A = min(T-1, pk_A+30)
    gt_snaps_A = {}
    for nm, day in [("early", early_A), ("peak", pk_A), ("late", late_A)]:
        gt_snaps_A[nm] = compute_generation_times(
            f_A[day], sim_A["susceptibles"][day], pops_A,
            params["prob_transmission_peak"], w_within, w_between, max_days, lw_A, lb_A)
        gn_mean = float(np.sum(np.arange(max_days) * gt_snaps_A[nm]["g_network"]))
        print(f"    {nm:5s} (d{day:3d}): g_network mean = {gn_mean:.4f} d")

    # ── GT spatial variation at peak ──────────────────────────────────────
    # Why GT variation is modest: w_within (mean 4d) and w_between (mean 7d)
    # are mixed proportional to bKw[k,j]/bKb[k,j] = lw*f[j,k]*f[k,k] /
    # (lb * Σ_{l≠k} f[j,l]*f[k,l]/N_eff[l]).  With lw/lb=1/0.30 and high
    # home fracs (60–80%), bKw >> bKb for most pairs → GT dominated by w_within.
    # Hub–hub pairs have most away contact but still bKw/bKb ≈ 1.5–3.
    # Larger GT spread needs higher lb/lw or wider within/between GT gap.
    print("\n─── GT spatial variation at peak (g^k_out means per location) ───")
    _days_arr = np.arange(max_days)
    _g_out_pk  = gt_snaps_A["peak"]["g_outward"]   # (max_days, N)
    _bKw_bKb_pk, _bKw_pk, _bKb_pk, _, _ = _kernel_base(
        f_A[pk_A], pops_A, lw_A, lb_A)
    for j in range(N_LOC):
        if _g_out_pk[:, j].sum() > 0.5:
            mu_j  = float(_g_out_pk[:, j] @ _days_arr)
            # within-weight: mean of bKw[j,:] / (bKw[j,:]+bKb[j,:])
            denom = (_bKw_pk[j] + _bKb_pk[j]).sum()
            ww_frac = float(_bKw_pk[j].sum() / denom) if denom > 0 else np.nan
            print(f"    L{j+1} ({types_A[j]:11s}): g_out_mean={mu_j:.3f}d  "
                  f"within-weight={ww_frac:.3f}")
    # Lambda value diagnostics
    _N_eff_peak = f_A[pk_A].T @ pops_A
    _lam_w_eff  = np.where(_N_eff_peak > 0, lw_A / _N_eff_peak, 0.0)
    _lam_b_eff  = np.where(_N_eff_peak > 0, lb_A / _N_eff_peak, 0.0)
    print(f"\n─── Lambda value diagnostics ───")
    print(f"  lw (scaled) = {lw_A:.4f}  lb (scaled) = {lb_A:.4f}  "
          f"lb/lw = {lb_A/lw_A:.3f}")
    print(f"  N_eff at peak: [{_N_eff_peak.min():.0f}, {_N_eff_peak.max():.0f}]")
    print(f"  λ_w = lw/N_eff: [{_lam_w_eff.min()*1e5:.2f}e-5, "
          f"{_lam_w_eff.max()*1e5:.2f}e-5] per contact per day")
    print(f"  λ_b = lb/N_eff: [{_lam_b_eff.min()*1e5:.2f}e-5, "
          f"{_lam_b_eff.max()*1e5:.2f}e-5] per contact per day")
    print(f"  [These are λ^k_E(t,a_E) base rates; final rate = λ × p(a_E)]")

    # ── Spectral + reactivity summary ─────────────────────────────────────
    react_A = reactivity(sim_A["R_matrices"][0])
    print(f"\n  σ(0)/ρ(0) = {react_A['amplification_ratio']:.4f}")
    print(f"  Spectral gap = {spec0_A['rho'] - spec0_A['lambda2']:.4f}")

    # ── Controllability ───────────────────────────────────────────────────
    ctrl = minimum_control_effort(sim_A["R_matrices"][0],
                                   costs=pops_A / pops_A.mean())
    print(f"\n  Homogeneous u_min = {ctrl['u_homogeneous']*100:.1f}%")

    # ── Figures ───────────────────────────────────────────────────────────
    print("\n─── Generating figures ───")
    _set_pub_style()
    R_t0_B = compute_R_matrix(f_B[0], pops_B, pops_B,
                               params["prob_transmission_peak"], infect_profile,
                               sim_B["lambda_within_scaled"], sim_B["lambda_between_scaled"])

    import os
    os.makedirs("out_figs/main", exist_ok=True)
    os.makedirs("out_figs/SI",   exist_ok=True)
    mpfx = "out_figs/main/fig"   # main figure prefix
    spfx = "out_figs/SI/fig"     # SI figure prefix

    plot_fig2(sim_A, city_A, f_A, gen_time_pmf, max_days, "Dense urban", mpfx)
    plot_fig3(sim_A, city_A, R_ind_A, gt_snaps_A, w_within, w_between, "Dense urban", mpfx)
    plot_fig4(sim_A, city_A, R_t0_B, "Dense urban", w_within, w_between, max_days, mpfx)
    plot_fig5(sim_A, sim_B, city_A, city_B, f_A, f_B, R0_A, R_t0_B, mpfx)
    plot_SI5_combined(sim_A, sim_B, city_A, city_B, f_A, f_B,
                      params["prob_transmission_peak"], infect_profile,
                      lw_A, lb_A, lw_B, lb_B, spfx)
    plot_fig7(sim_A, city_A, f_A, w_within, w_between, max_days,
              params["prob_transmission_peak"], "Dense urban", spfx)
    plot_SI1(sim_A, city_A, "Dense urban", spfx)
    plot_SI2(sim_A, city_A, f_A,
             params["prob_transmission_peak"], infect_profile, lw_A, lb_A,
             "Dense urban", spfx)
    plot_SI_epi_params(params, w_within, w_between, gen_time_pmf, max_days,
                       sim_A, f_A, populations=pops_A, save_prefix=spfx)
    plot_SI_pde_convergence(city_A, f_A,
                             params, initial_A, LW, LB, w_within, w_between,
                             T_test=90, save_prefix=spfx)
    plot_SI_sensitivity(city_A, f_A, params, initial_A, LW, LB, w_within, w_between,
                        T_test=120, save_prefix=spfx)
    plot_SI_3d_earlypeak(sim_A, sim_B, spfx)
    plot_counterfactual_nonnormal(sim_A, sim_C, city_A, f_A, f_C,
                                   w_within, w_between, max_days,
                                   lw_A, lb_A, lw_C, lb_C, params, mpfx)
    plot_3d_surfaces(sim_A, city_A, f_A, w_within, w_between, max_days,
                     params["prob_transmission_peak"], mpfx)
    plot_SI_lambda_decomposition(sim_A, city_A, f_A, params,
                                  gen_time_pmf, w_within, w_between, max_days,
                                  spfx)

    print("\nAll figures saved.  Done.")


if __name__ == "__main__":
    main()
