#!/usr/bin/env python3
"""
Mobility-Informed and Mechanism-Led Renewal Equations
— Directly Transmitted Diseases (Section 3.1) —

Complete numerical implementation of:
  1. Synthetic city geometry and mobility (phone-ping time-fraction f_{jk})
  2. PDE-derived instantaneous kernels K_{kj}(t, a_E)
  3. Forward simulation from the renewal equation (Finite Difference PDE)
  4. Explicit birth and death demographic processes
  5. Full taxonomy of reproduction numbers: R_{kj}, R^out_k, R^in_j, R(t)=ρ(R)
  6. Location-specific and network-level generation time distributions
  7. Independent (Cori-style) vs mobility-informed estimation + bias quantification
  8. Source–sink decomposition
  9. Spectral analysis: mixing time, CV of row sums, dominant eigenvector
 10. Reactivity and transient amplification
 11. Sensitivity / elasticity of ρ(R) to each matrix entry and mobility flow
 12. Epidemic speed (spatial generation distance / velocity)
 13. Controllability: minimum effort optimisation, targeted vs homogeneous
 14. Comprehensive diagnostics and visualisation
 15. Vector-Borne Quadruple Integral Foundation (Section 3.2)

COVID-19 parameters from peer-reviewed sources:
  - Generation time: Gamma(mean=5.5, sd=1.8)  [Hart et al. 2022, PLOS Comp Biol]
  - Infectiousness profile: peaks ~2 days post-symptom onset
  - R0 ≈ 2.5 (original strain) [Liu et al. 2020, J Travel Med]
  - Serial interval ≈ 5.2 days [He et al. 2020, Nature Medicine]
"""

import numpy as np
from scipy.stats import gamma as gamma_dist
from scipy.spatial.distance import cdist
from scipy.optimize import minimize_scalar
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize, TwoSlopeNorm
import matplotlib.cm as cm
import warnings
warnings.filterwarnings("ignore")

np.set_printoptions(precision=4, suppress=True)

# ══════════════════════════════════════════════════════════════════════════════
# 0. COVID-19 EPIDEMIOLOGICAL PARAMETERS (peer-reviewed)
# ══════════════════════════════════════════════════════════════════════════════

COVID_PARAMS = {
    "gen_time_mean": 5.5,      
    "gen_time_sd": 1.8,        
    "max_gen_time": 25,        

    "infectiousness_peak": 2.5,   
    "infectiousness_sd": 1.5,

    "base_contact_rate": 13.0,    
    "prob_transmission_peak": 0.035,
    "mean_infectious_period": 7.0,   
    "R0_target": 2.5,
}

# ══════════════════════════════════════════════════════════════════════════════
# 1. GENERATION TIME AND INFECTIOUSNESS PROFILE
# ══════════════════════════════════════════════════════════════════════════════

def discretise_gamma(mean, sd, max_days):
    if sd <= 0 or mean <= 0:
        raise ValueError("mean and sd must be positive")
    shape = (mean / sd) ** 2
    scale = sd ** 2 / mean
    pmf = np.zeros(max_days)
    for d in range(max_days):
        pmf[d] = gamma_dist.cdf(d + 1, a=shape, scale=scale) - gamma_dist.cdf(d, a=shape, scale=scale)
    pmf /= pmf.sum()
    return pmf

def infectiousness_profile(max_days, peak=2.5, sd=1.5):
    shape = (peak / sd) ** 2
    scale = sd ** 2 / peak
    profile = np.array([gamma_dist.pdf(d + 0.5, a=shape, scale=scale) for d in range(max_days)])
    profile /= profile.max()  
    return profile

def survival_function(max_days, mean_infectious_period):
    psi = 1.0 / mean_infectious_period
    return np.exp(-psi * np.arange(max_days))

# ══════════════════════════════════════════════════════════════════════════════
# 2. SYNTHETIC CITY AND MOBILITY DATA
# ══════════════════════════════════════════════════════════════════════════════

def generate_city(n_locations=10, city_type="polycentric", seed=42):
    rng = np.random.default_rng(seed)

    if city_type == "polycentric":
        n_centres = max(3, n_locations // 3)
        centre_angles = np.linspace(0, 2 * np.pi, n_centres, endpoint=False)
        centre_radii = rng.uniform(3, 10, n_centres)
        centre_coords = np.column_stack([
            centre_radii * np.cos(centre_angles),
            centre_radii * np.sin(centre_angles),
        ])
        assignments = rng.choice(n_centres, n_locations)
        coords = centre_coords[assignments] + rng.normal(0, 2.0, (n_locations, 2))
        dist_to_centre = np.linalg.norm(coords - centre_coords[assignments], axis=1)
        pop_raw = 80000 / (1.0 + dist_to_centre) + rng.exponential(15000, n_locations)
        populations = np.round(pop_raw).astype(float)
    elif city_type == "monocentric":
        angles = rng.uniform(0, 2 * np.pi, n_locations)
        radii = rng.exponential(5.0, n_locations)
        radii[0] = 0.0
        coords = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
        pop_raw = 100000 / (1.0 + radii) + rng.exponential(10000, n_locations)
        populations = np.round(pop_raw).astype(float)
    else:
        raise ValueError(f"Unknown city_type: {city_type}")

    distances = cdist(coords, coords, metric="euclidean")
    return coords, populations, distances


def generate_mobility(
    n_locations, T, populations, distances, coords,
    distance_decay=1.5,
    day_variation_sd=0.15,
    seed=42,
):
    rng = np.random.default_rng(seed)
    N = n_locations
    attractiveness = populations / populations.sum()

    centroid = coords.mean(axis=0)
    dist_to_centroid = np.linalg.norm(coords - centroid, axis=1)
    d_norm = dist_to_centroid / (dist_to_centroid.max() + 1e-10) 
    home_fracs = 0.50 + 0.25 * d_norm          
    home_fracs += rng.normal(0, 0.03, N)        
    home_fracs = np.clip(home_fracs, 0.35, 0.85)

    dist_kernel = np.zeros((N, N))
    for j in range(N):
        for k in range(N):
            if j == k:
                continue
            d = max(distances[j, k], 0.5)
            dist_kernel[j, k] = attractiveness[k] / (d ** distance_decay)

    base_f = np.zeros((N, N))
    for j in range(N):
        total = dist_kernel[j, :].sum()
        if total > 0:
            away_frac = 1.0 - home_fracs[j]
            base_f[j, :] = away_frac * dist_kernel[j, :] / total
        base_f[j, j] = home_fracs[j]

    dow_scale = np.array([1.00, 1.00, 1.00, 1.00, 0.90, 0.60, 0.50])

    f_jk = np.zeros((T, N, N))
    for t in range(T):
        dow = t % 7
        scale = dow_scale[dow]

        noise = rng.lognormal(0.0, day_variation_sd)
        noise = np.clip(noise, 0.30, 3.00)
        scale *= noise

        for j in range(N):
            away_base = 1.0 - home_fracs[j]
            scaled_away = np.clip(away_base * scale, 0.0, 0.90)

            if away_base > 1e-12:
                ratio = scaled_away / away_base
                f_jk[t, j, :] = base_f[j, :] * ratio
                f_jk[t, j, j] = 0.0            

            f_jk[t, j, j] = 1.0 - f_jk[t, j, :].sum()
            f_jk[t, j, :] = np.clip(f_jk[t, j, :], 0.0, 1.0)
            f_jk[t, j, :] /= f_jk[t, j, :].sum()

    return f_jk

# ══════════════════════════════════════════════════════════════════════════════
# 3. INSTANTANEOUS KERNELS
# ══════════════════════════════════════════════════════════════════════════════

def compute_lambda(
    a_E, contact_rate, prob_peak, infectiousness, survival,
    N_eff_m, lambda_within=None, lambda_between=None,
    is_home=True,
):
    p_aE = prob_peak * infectiousness[a_E] * survival[a_E]

    if lambda_within is not None and lambda_between is not None:
        kappa = lambda_within if is_home else lambda_between
    else:
        kappa = contact_rate

    if N_eff_m > 0:
        return kappa * p_aE / N_eff_m
    return 0.0

def compute_kernel_matrix(
    f_S, f_E, S, populations,
    contact_rate, prob_peak, infectiousness, survival, a_E,
    lambda_within=None, lambda_between=None,
):
    N = len(populations)
    N_eff = f_S.T @ populations 

    K = np.zeros((N, N))
    for k in range(N):
        for j in range(N):
            val = 0.0
            for m in range(N):
                is_home = (m == k)
                lam = compute_lambda(
                    a_E, contact_rate, prob_peak,
                    infectiousness, survival, N_eff[m],
                    lambda_within, lambda_between, is_home,
                )
                val += f_S[j, m] * f_E[k, m] * lam
            K[k, j] = S[j] * val
    return K

def compute_kernel_matrix_fast(
    f_S, f_E, S, populations,
    contact_rate, prob_peak, infectiousness, survival, a_E,
    lambda_within=None, lambda_between=None,
):
    N = len(populations)
    N_eff = f_S.T @ populations
    inv_N_eff = np.where(N_eff > 0, 1.0 / N_eff, 0.0)

    p_aE = prob_peak * infectiousness[a_E] * survival[a_E]

    if lambda_within is not None and lambda_between is not None:
        K = np.zeros((N, N))
        for k in range(N):
            for j in range(N):
                val = 0.0
                for m in range(N):
                    kappa = lambda_within if m == k else lambda_between
                    val += f_S[j, m] * f_E[k, m] * kappa * p_aE * inv_N_eff[m]
                K[k, j] = S[j] * val
        return K
    else:
        weighted_f_E = f_E * inv_N_eff[np.newaxis, :]  
        K = contact_rate * p_aE * (weighted_f_E @ f_S.T)
        K *= S[np.newaxis, :]
        return K

# ══════════════════════════════════════════════════════════════════════════════
# 4. REPRODUCTION NUMBERS
# ══════════════════════════════════════════════════════════════════════════════

def compute_R_matrix(
    f_S, f_E, S, populations,
    contact_rate, prob_peak, infectiousness, survival, max_days,
    lambda_within=None, lambda_between=None,
):
    N = len(populations)
    R_mat = np.zeros((N, N))
    for a_E in range(max_days):
        K = compute_kernel_matrix_fast(
            f_S, f_E, S, populations,
            contact_rate, prob_peak, infectiousness, survival, a_E,
            lambda_within, lambda_between,
        )
        R_mat += K
    return R_mat

def R_outward(R_mat):
    return R_mat.sum(axis=1)

def R_inward(R_mat):
    return R_mat.sum(axis=0)

def R_system(R_mat):
    eigvals = np.linalg.eigvals(R_mat)
    return np.max(np.abs(eigvals)).real

def spectral_analysis(R_mat):
    eigvals = np.linalg.eigvals(R_mat)
    idx = np.argsort(np.abs(eigvals))[::-1]
    eigvals_sorted = eigvals[idx]

    rho = np.abs(eigvals_sorted[0]).real
    lambda2 = np.abs(eigvals_sorted[1]).real if len(eigvals_sorted) > 1 else 0.0

    eigvals_r, eigvecs_r = np.linalg.eig(R_mat)
    eigvals_l, eigvecs_l = np.linalg.eig(R_mat.T)

    idx_r = np.argmax(np.abs(eigvals_r))
    idx_l = np.argmax(np.abs(eigvals_l))

    w = np.abs(eigvecs_r[:, idx_r].real)  
    w /= w.sum()
    v = np.abs(eigvecs_l[:, idx_l].real)  
    v /= v.sum()

    mixing_ratio = lambda2 / rho if rho > 0 else 0.0

    row_sums = R_mat.sum(axis=1)
    cv = row_sums.std() / row_sums.mean() if row_sums.mean() > 0 else 0.0

    return {
        "rho": rho,
        "lambda2": lambda2,
        "mixing_ratio": mixing_ratio,
        "right_eigvec": w,
        "left_eigvec": v,
        "cv_row_sums": cv,
        "eigenvalues": eigvals_sorted,
    }

# ══════════════════════════════════════════════════════════════════════════════
# 5. GENERATION TIME DISTRIBUTIONS
# ══════════════════════════════════════════════════════════════════════════════

def compute_generation_times(
    f_S, f_E, S, populations,
    contact_rate, prob_peak, infectiousness, survival, max_days,
    lambda_within=None, lambda_between=None,
):
    N = len(populations)
    K_series = np.zeros((max_days, N, N))

    for a_E in range(max_days):
        K_series[a_E] = compute_kernel_matrix_fast(
            f_S, f_E, S, populations,
            contact_rate, prob_peak, infectiousness, survival, a_E,
            lambda_within, lambda_between,
        )

    R_mat = K_series.sum(axis=0)
    g_pairwise = np.zeros_like(K_series)
    for k in range(N):
        for j in range(N):
            if R_mat[k, j] > 1e-15:
                g_pairwise[:, k, j] = K_series[:, k, j] / R_mat[k, j]

    K_out = K_series.sum(axis=2)  
    R_out = R_mat.sum(axis=1)     
    g_out = np.zeros_like(K_out)
    for k in range(N):
        if R_out[k] > 1e-15:
            g_out[:, k] = K_out[:, k] / R_out[k]

    K_in = K_series.sum(axis=1)   
    R_in = R_mat.sum(axis=0)      
    g_in = np.zeros_like(K_in)
    for j in range(N):
        if R_in[j] > 1e-15:
            g_in[:, j] = K_in[:, j] / R_in[j]

    K_total = K_series.sum(axis=(1, 2))  
    R_total = R_mat.sum()
    g_network = K_total / R_total if R_total > 1e-15 else K_total

    return {
        "g_pairwise": g_pairwise,
        "g_outward": g_out,
        "g_inward": g_in,
        "g_network": g_network,
        "K_series": K_series,
        "R_matrix": R_mat,
    }

# ══════════════════════════════════════════════════════════════════════════════
# 6. FORWARD SIMULATION (FINITE DIFFERENCE PDE & DEMOGRAPHICS)
# ══════════════════════════════════════════════════════════════════════════════

def simulate_epidemic_pde(
    T, n_locations, populations, f_jk_series,
    contact_rate, prob_peak, infectiousness, survival, max_days,
    R0_target, initial_infections,
    lambda_within=None, lambda_between=None,
    birth_rate=0.00003, death_rate=0.00003,  
    stochastic=True, susceptible_depletion=True,
    seed=42,
):
    """
    Forward simulation using a Finite Difference scheme for the age-structured PDE,
    incorporating explicit birth and death processes.
    """
    rng = np.random.default_rng(seed)
    N = n_locations
    
    # State arrays
    S_series = np.zeros((T, N))
    S = populations.copy().astype(float)
    
    # E[t, j, a_E] tracks the density of individuals in location j at time t with infection age a_E
    E_pde = np.zeros((T, N, max_days)) 
    incidence_matrix = np.zeros((T, N, N))
    R_matrices = np.zeros((T, N, N))
    
    # Initialize t=0
    E_pde[0, :, 0] = initial_infections
    S -= initial_infections
    S_series[0, :] = S.copy()

    # Calibrate: compute R matrix at t=0, scale contact rate so ρ = R0_target
    f0 = f_jk_series[0]
    R_mat_0 = compute_R_matrix(
        f0, f0, populations, populations,
        contact_rate, prob_peak, infectiousness, survival, max_days,
        lambda_within, lambda_between,
    )
    rho_0 = R_system(R_mat_0)
    calibration_scale = R0_target / rho_0 if rho_0 > 0 else 1.0

    if lambda_within is not None:
        lw = lambda_within * calibration_scale
        lb = lambda_between * calibration_scale
    else:
        lw, lb = None, None
    scaled_contact = contact_rate * calibration_scale

    # Verify calibration
    R_check = compute_R_matrix(
        f0, f0, populations, populations,
        scaled_contact, prob_peak, infectiousness, survival, max_days,
        lw, lb,
    )
    rho_check = R_system(R_check)
    print(f"  Calibration: base ρ = {rho_0:.4f}, scale = {calibration_scale:.4f}")
    print(f"  Post-calibration ρ = {rho_check:.4f} (target: {R0_target})")

    R_matrices[0] = R_check.copy()

    for t in range(1, T):
        f_t = f_jk_series[min(t, len(f_jk_series) - 1)]
        
        # 1. Advance the PDE interior via Finite Difference (Upwind scheme: a_E -> a_E + 1)
        for a in range(1, max_days):
            E_pde[t, :, a] = E_pde[t-1, :, a-1] 
            
        # 2. Compute Instantaneous Kernels and Boundary Condition E^j(t, 0)
        R_mat_t = compute_R_matrix(
            f_t, f_t, S, populations, scaled_contact, prob_peak, 
            infectiousness, survival, max_days, lw, lb
        )
        R_matrices[t] = R_mat_t
        
        # Evaluate the integral boundary condition across all infection ages
        for k in range(N):
            for j in range(N):
                expected = 0.0
                for a_E in range(1, max_days):
                    K_val = compute_kernel_matrix_fast(
                        f_t, f_t, S, populations, scaled_contact, prob_peak, 
                        infectiousness, survival, a_E, lw, lb
                    )[k, j]
                    expected += K_val * E_pde[t, k, a_E]
                
                if stochastic and expected > 0:
                    new = rng.poisson(min(expected, 1e7))
                else:
                    new = max(expected, 0)
                
                if susceptible_depletion:
                    new = min(new, S[j])
                
                incidence_matrix[t, k, j] = new
                
        new_total = incidence_matrix[t].sum(axis=0)
        E_pde[t, :, 0] = new_total

        # 3. Explicit Birth and Death Processes for Susceptibles
        if susceptible_depletion:
            births = birth_rate * populations
            deaths = death_rate * S
            S = S - new_total + births - deaths
            S = np.maximum(S, 0)
            
        S_series[t, :] = S.copy()

    incidence = E_pde[:, :, 0] 
    
    return {
        "incidence": incidence,
        "incidence_matrix": incidence_matrix,
        "R_matrices": R_matrices,
        "susceptibles": S_series,
        "calibration_scale": calibration_scale,
        "scaled_contact_rate": scaled_contact,
        "lambda_within_scaled": lw,
        "lambda_between_scaled": lb,
        "E_pde_state": E_pde
    }

# ══════════════════════════════════════════════════════════════════════════════
# 7. INDEPENDENT (CORI-STYLE) R(t) ESTIMATION
# ══════════════════════════════════════════════════════════════════════════════

def estimate_R_independent(incidence, gen_time, window=7):
    T, N = incidence.shape
    max_s = len(gen_time)
    R_est = np.full((T, N), np.nan)
    prior_a, prior_b = 1.0, 5.0   

    for j in range(N):
        for t in range(max_s, T):
            t_start = max(max_s, t - window + 1)
            total_obs = incidence[t_start:t + 1, j].sum()
            total_lambda = 0.0
            for t_w in range(t_start, t + 1):
                for s in range(1, min(max_s, t_w)):
                    total_lambda += gen_time[s] * incidence[t_w - s, j]
            if total_lambda > 1e-4 and total_obs >= 1:
                R_est[t, j] = (prior_a + total_obs) / (prior_b + total_lambda)
    return R_est

# ══════════════════════════════════════════════════════════════════════════════
# 8. SOURCE-SINK, SENSITIVITY, REACTIVITY, CONTROLLABILITY
# ══════════════════════════════════════════════════════════════════════════════

def source_sink_analysis(R_mat):
    R_out = R_outward(R_mat)
    R_in = R_inward(R_mat)
    net = R_out - R_in  
    return {
        "R_outward": R_out,
        "R_inward": R_in,
        "net_export": net,
        "sources": np.where(net > 0)[0],
        "sinks": np.where(net < 0)[0],
    }

def within_between_decomposition(R_mat):
    diag_sum = np.trace(R_mat)
    total_sum = R_mat.sum()
    pi_within = diag_sum / total_sum if total_sum > 0 else 0
    return {
        "pi_within": pi_within,
        "pi_between": 1 - pi_within,
        "diag_sum": diag_sum,
        "total_sum": total_sum,
    }

def sensitivity_elasticity(R_mat):
    spec = spectral_analysis(R_mat)
    rho = spec["rho"]
    v = spec["left_eigvec"]
    w = spec["right_eigvec"]

    N = len(v)
    vw = v @ w
    if vw < 1e-15:
        vw = 1e-15

    sensitivity = np.outer(v, w) / vw
    elasticity = np.zeros((N, N))
    if rho > 1e-15:
        elasticity = (R_mat / rho) * sensitivity

    return {
        "sensitivity": sensitivity,
        "elasticity": elasticity,
        "rho": rho,
    }

def reactivity(R_mat):
    symmetric_part = (R_mat + R_mat.T) / 2
    eigvals = np.linalg.eigvalsh(symmetric_part)
    sigma = eigvals.max()
    rho = R_system(R_mat)
    return {
        "sigma": sigma,
        "rho": rho,
        "transient_possible": sigma > 1 and rho < 1,
        "amplification_ratio": sigma / rho if rho > 0 else np.inf,
    }

def epidemic_speed(R_mat, distances, gen_time_pmf):
    N = R_mat.shape[0]
    total = R_mat.sum()
    if total < 1e-15:
        return {"mean_distance": 0, "mean_gen_time": 0, "speed": 0}

    d_bar = np.sum(distances * R_mat) / total
    g_bar = np.sum(np.arange(len(gen_time_pmf)) * gen_time_pmf)
    speed = d_bar / g_bar if g_bar > 0 else 0

    return {"mean_distance": d_bar, "mean_gen_time": g_bar, "speed": speed}

def minimum_control_effort(R_mat, costs=None):
    rho = R_system(R_mat)
    N = R_mat.shape[0]
    if costs is None:
        costs = np.ones(N)

    u_homog = max(0, 1 - 1 / rho) if rho > 0 else 0

    se = sensitivity_elasticity(R_mat)
    row_elasticity = se["elasticity"].sum(axis=1)
    priority = row_elasticity / costs
    order = np.argsort(priority)[::-1]

    u_het = np.zeros(N)
    R_current = R_mat.copy()
    for idx in order:
        rho_current = R_system(R_current)
        if rho_current <= 1.0:
            break
        lo, hi = 0.0, 1.0
        for _ in range(30):
            mid = (lo + hi) / 2
            R_test = R_current.copy()
            R_test[idx, :] *= (1 - mid)
            if R_system(R_test) <= 1.0:
                hi = mid
            else:
                lo = mid
        u_het[idx] = hi
        R_current[idx, :] *= (1 - u_het[idx])

    return {
        "u_homogeneous": u_homog,
        "u_heterogeneous": u_het,
        "total_effort_homog": u_homog * costs.sum(),
        "total_effort_hetero": (u_het * costs).sum(),
        "priority_order": order,
    }

# ══════════════════════════════════════════════════════════════════════════════
# 9. COMPREHENSIVE VISUALISATION
# ══════════════════════════════════════════════════════════════════════════════

def plot_all(
    sim_result, city_data, epi_params, gen_time_snapshots,
    R_independent, spectral_data, save_prefix="fig",
):
    incidence = sim_result["incidence"]
    R_matrices = sim_result["R_matrices"]
    S_series = sim_result["susceptibles"]
    coords, populations, distances = city_data
    T, N = incidence.shape

    R_sys = np.array([R_system(R_matrices[t]) for t in range(T)])
    R_out_series = np.array([R_outward(R_matrices[t]) for t in range(T)])
    R_in_series = np.array([R_inward(R_matrices[t]) for t in range(T)])

    loc_names = [f"L{i}" for i in range(N)]
    colors = plt.cm.Set2(np.linspace(0, 1, min(N, 10)))

    # ─── Figure 1: Epidemic overview ───
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Figure 1: Epidemic Overview (COVID-19, Mobility-Informed)", fontsize=14, fontweight="bold")

    ax = axes[0, 0]
    ax.plot(incidence.sum(axis=1), "k-", lw=2, label="Total")
    for j in range(min(N, 6)):
        ax.plot(incidence[:, j], color=colors[j], alpha=0.7, label=loc_names[j])
    ax.set_xlabel("Day"); ax.set_ylabel("New infections")
    ax.set_title("Incidence"); ax.legend(fontsize=7, ncol=2)

    ax = axes[0, 1]
    ax.plot(S_series.sum(axis=1) / populations.sum(), "k-", lw=2, label="Total")
    for j in range(min(N, 4)):
        ax.plot(S_series[:, j] / populations[j], color=colors[j], alpha=0.7, label=loc_names[j])
    ax.set_xlabel("Day"); ax.set_ylabel("Fraction susceptible")
    ax.set_title("Susceptible depletion"); ax.legend(fontsize=7)

    ax = axes[1, 0]
    valid = R_sys > 0
    ax.plot(np.where(valid)[0], R_sys[valid], "b-", lw=2, label="ρ(R(t)) — mobility-informed")
    ax.axhline(1.0, color="gray", ls="--", alpha=0.5)
    ax.set_xlabel("Day"); ax.set_ylabel("R(t)")
    ax.set_title("System-level R(t)"); ax.legend(fontsize=8)
    ax.set_ylim(0, max(3, R_sys[valid].max() * 1.1) if valid.any() else 3)

    ax = axes[1, 1]
    cum = incidence.cumsum(axis=0)
    for j in range(min(N, 6)):
        ax.plot(cum[:, j], color=colors[j], label=loc_names[j])
    ax.set_xlabel("Day"); ax.set_ylabel("Cumulative infections")
    ax.set_title("Cumulative incidence"); ax.legend(fontsize=7, ncol=2)

    plt.tight_layout()
    plt.savefig(f"{save_prefix}_01_overview.png", dpi=150, bbox_inches="tight")
    plt.close()

    # ─── Figure 2: R matrix heatmap and spectral ───
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Figure 2: Reproduction Number Matrix & Spectral Properties", fontsize=14, fontweight="bold")

    peak_day = incidence.sum(axis=1).argmax()
    peak_day = max(peak_day, 5)
    R_peak = R_matrices[peak_day]

    ax = axes[0]
    im = ax.imshow(R_peak, cmap="YlOrRd", aspect="auto")
    plt.colorbar(im, ax=ax, label="R_{kj}")
    ax.set_xticks(range(N)); ax.set_xticklabels(loc_names, fontsize=7, rotation=45)
    ax.set_yticks(range(N)); ax.set_yticklabels(loc_names, fontsize=7)
    ax.set_xlabel("Target j"); ax.set_ylabel("Source k")
    ax.set_title(f"R_{{kj}} matrix (day {peak_day})")

    ax = axes[1]
    se = sensitivity_elasticity(R_peak)
    im = ax.imshow(se["elasticity"], cmap="Purples", aspect="auto")
    plt.colorbar(im, ax=ax, label="Elasticity e_{kj}")
    ax.set_xticks(range(N)); ax.set_xticklabels(loc_names, fontsize=7, rotation=45)
    ax.set_yticks(range(N)); ax.set_yticklabels(loc_names, fontsize=7)
    ax.set_xlabel("Target j"); ax.set_ylabel("Source k")
    ax.set_title(f"Elasticity of ρ(R) (day {peak_day})")

    ax = axes[2]
    spec = spectral_data
    eigvals = spec["eigenvalues"]
    ax.scatter(eigvals.real, eigvals.imag, s=80, c="steelblue", edgecolors="k", zorder=5)
    ax.axhline(0, color="gray", ls="-", alpha=0.3)
    ax.axvline(0, color="gray", ls="-", alpha=0.3)
    theta = np.linspace(0, 2 * np.pi, 100)
    ax.plot(spec["rho"] * np.cos(theta), spec["rho"] * np.sin(theta), "r--", alpha=0.5, label=f"ρ = {spec['rho']:.3f}")
    ax.set_xlabel("Real"); ax.set_ylabel("Imaginary")
    ax.set_title("Eigenvalue spectrum of R(t=0)")
    ax.legend(); ax.set_aspect("equal")

    plt.tight_layout()
    plt.savefig(f"{save_prefix}_02_R_matrix.png", dpi=150, bbox_inches="tight")
    plt.close()

    # ─── Figure 3: R types comparison ───
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Figure 3: Taxonomy of Reproduction Numbers", fontsize=14, fontweight="bold")

    ax = axes[0, 0]
    for j in range(min(N, 5)):
        valid_j = R_out_series[:, j] > 0
        ax.plot(np.where(valid_j)[0], R_out_series[valid_j, j], color=colors[j], label=loc_names[j])
    ax.axhline(1, color="gray", ls="--", alpha=0.5)
    ax.set_xlabel("Day"); ax.set_ylabel("R^out_k(t)")
    ax.set_title("Outward R (source potential)"); ax.legend(fontsize=7)

    ax = axes[0, 1]
    for j in range(min(N, 5)):
        valid_j = R_in_series[:, j] > 0
        ax.plot(np.where(valid_j)[0], R_in_series[valid_j, j], color=colors[j], label=loc_names[j])
    ax.axhline(1, color="gray", ls="--", alpha=0.5)
    ax.set_xlabel("Day"); ax.set_ylabel("R^in_j(t)")
    ax.set_title("Inward R (growth indicator)"); ax.legend(fontsize=7)

    ax = axes[1, 0]
    for j in range(min(N, 3)):
        valid_ind = ~np.isnan(R_independent[:, j])
        valid_mob = R_in_series[:, j] > 0
        both = valid_ind & valid_mob
        if both.any():
            ax.plot(np.where(both)[0], R_independent[both, j], "--", color=colors[j], alpha=0.7, label=f"Indep {loc_names[j]}")
            ax.plot(np.where(both)[0], R_in_series[both, j], "-", color=colors[j], label=f"Mob {loc_names[j]}")
    ax.axhline(1, color="gray", ls="--", alpha=0.5)
    ax.set_xlabel("Day"); ax.set_ylabel("R(t)")
    ax.set_title("Independent vs Mobility-informed R(t)"); ax.legend(fontsize=7)

    ax = axes[1, 1]
    ss = source_sink_analysis(R_matrices[0])
    bar_colors = ["#d63031" if n > 0 else "#0984e3" for n in ss["net_export"]]
    ax.barh(range(N), ss["net_export"], color=bar_colors)
    ax.set_yticks(range(N)); ax.set_yticklabels(loc_names, fontsize=7)
    ax.axvline(0, color="k", lw=0.5)
    ax.set_xlabel("Net export (R^out - R^in)")
    ax.set_title("Source–Sink (day 0): Red=Source, Blue=Sink")

    plt.tight_layout()
    plt.savefig(f"{save_prefix}_03_R_types.png", dpi=150, bbox_inches="tight")
    plt.close()

    # ─── Figure 4: Generation time distributions ───
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Figure 4: Generation Time Distributions", fontsize=14, fontweight="bold")

    gt_ref = gen_time_snapshots["early"]
    days = np.arange(len(gt_ref["g_network"]))

    ax = axes[0]
    ax.bar(days, gt_ref["g_network"], color="steelblue", alpha=0.8,
           label="Intrinsic GT: w(a)/Σw")
    bio_gt = discretise_gamma(COVID_PARAMS["gen_time_mean"], COVID_PARAMS["gen_time_sd"], len(days))
    ax.plot(days, bio_gt, "r--", lw=2, label="Biological Gamma (input)")
    ax.set_xlabel("Infection age $a_E$ (days)")
    ax.set_ylabel("Probability")
    ax.set_title("Intrinsic GT = infectiousness × survival\n(identical for all pairs; equals $w(a_E)/\\Sigma w$)")
    ax.legend(fontsize=8)

    ax = axes[1]
    phase_styles = {
        "early":  ("Early epidemic",   "steelblue",  "-",  2.5),
        "peak":   ("Peak epidemic",    "darkorange",  "--", 2.0),
        "late":   ("Post-peak (late)", "forestgreen", ":",  2.0),
    }
    plotted_labels = set()
    for phase, (label, col, ls, lw_) in phase_styles.items():
        if phase not in gen_time_snapshots:
            continue
        gt_snap = gen_time_snapshots[phase]
        g_net = gt_snap["g_network"]
        if label not in plotted_labels:
            ax.plot(days[:len(g_net)], g_net, color=col, ls=ls, lw=lw_, label=label)
            plotted_labels.add(label)
        for k in range(min(N, 3)):
            g_k = gt_snap["g_outward"][:, k]
            if g_k.sum() > 0.5:
                ax.plot(days[:len(g_k)], g_k, color=col, ls=ls, lw=0.8, alpha=0.35)
    ax.set_xlabel("Infection age $a_E$ (days)")
    ax.set_ylabel("Probability")
    ax.set_title("GT snapshots at early / peak / late epidemic\n"
                 r"(curves collapse: $g_{kj}(t,a_E)$ is time-invariant)")
    ax.legend(fontsize=8)

    ax = axes[2]
    R_sys = np.array([R_system(R_matrices[t]) for t in range(T)])
    R_out_ts = np.array([R_outward(R_matrices[t]) for t in range(T)])
    valid = R_sys > 0
    ax.plot(np.where(valid)[0], R_sys[valid], "k-", lw=2, label="ρ(R(t)) — system")
    for k in range(min(N, 4)):
        v_k = R_out_ts[:, k] > 0
        ax.plot(np.where(v_k)[0], R_out_ts[v_k, k],
                color=colors[k], alpha=0.7, lw=1.2, label=f"$R^{{out}}_{{{loc_names[k]}}}(t)$")
    ax.axhline(1.0, color="gray", ls="--", alpha=0.5)
    ax.set_xlabel("Day $t$")
    ax.set_ylabel("Reproduction number")
    ax.set_title("What DOES change: $R_{kj}(t) = c_{kj}(t)\\cdot\\Sigma_a w$\n"
                 "(driven by $S_j(t)$, $f_{jk}(t)$ — not the shape of $g$)")
    ax.legend(fontsize=7)

    plt.tight_layout()
    plt.savefig(f"{save_prefix}_04_generation_times.png", dpi=150, bbox_inches="tight")
    plt.close()

    # ─── Figure 5: City map with R values ───
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle("Figure 5: Spatial Structure and Mobility", fontsize=14, fontweight="bold")

    f_avg = city_data[3] if len(city_data) > 3 else None

    for ax_idx, (values, title, cmap) in enumerate([
        (populations, "Population", "Blues"),
        (R_outward(R_matrices[0]), "R^out_k (day 0)", "Reds"),
        (R_inward(R_matrices[0]), "R^in_j (day 0)", "Oranges"),
    ]):
        ax = axes[ax_idx]
        sizes = populations / populations.max() * 400 + 50
        sc = ax.scatter(coords[:, 0], coords[:, 1], s=sizes, c=values,
                       cmap=cmap, edgecolors="black", lw=0.5, zorder=5)
        plt.colorbar(sc, ax=ax, label=title)
        for j in range(N):
            ax.annotate(str(j), coords[j], fontsize=7, ha="center", va="center", zorder=10)
        ax.set_xlabel("x (km)"); ax.set_ylabel("y (km)")
        ax.set_title(title); ax.set_aspect("equal")

    plt.tight_layout()
    plt.savefig(f"{save_prefix}_05_spatial.png", dpi=150, bbox_inches="tight")
    plt.close()

    # ─── Figure 6: Diagnostics ───
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Figure 6: Diagnostics & Additional Quantities", fontsize=14, fontweight="bold")

    ax = axes[0, 0]
    pi_within = np.zeros(T)
    for t in range(T):
        wb = within_between_decomposition(R_matrices[t])
        pi_within[t] = wb["pi_within"]
    valid = pi_within > 0
    ax.plot(np.where(valid)[0], pi_within[valid], "b-", lw=1.2)
    if valid.sum() > 14:
        from numpy.lib.stride_tricks import sliding_window_view
        pi_smooth = np.full(T, np.nan)
        pw = sliding_window_view(pi_within[valid], 7)
        pi_smooth_vals = pw.mean(axis=1)
        valid_idx = np.where(valid)[0]
        centre_idx = valid_idx[3:3 + len(pi_smooth_vals)]
        ax.plot(centre_idx, pi_smooth_vals, "r-", lw=2.5,
                label="7-day mean (secular trend)")
        ax.legend(fontsize=7)
    ax.set_xlabel("Day"); ax.set_ylabel("π_within")
    ax.set_title("Fraction of TX within locations\n"
                 "(oscillation = weekly mobility; trend = heterogeneous S depletion)")
    ax.set_ylim(0, 1)

    ax = axes[0, 1]
    sigma_series = np.zeros(T)
    for t in range(T):
        r = reactivity(R_matrices[t])
        sigma_series[t] = r["sigma"]
    valid_s = sigma_series > 0
    ax.plot(np.where(valid_s)[0], sigma_series[valid_s], "r-", lw=2, label="σ(t) reactivity")
    ax.plot(np.where(valid)[0], R_sys[valid], "b--", lw=1.5, label="ρ(t) spectral radius")
    ax.axhline(1, color="gray", ls="--", alpha=0.5)
    ax.set_xlabel("Day"); ax.set_ylabel("Value")
    ax.set_title("Reactivity σ(t) vs spectral radius ρ(t)\n"
                 "(gap = asymmetry from heterogeneous pops & home fractions)")
    ax.legend(fontsize=8)

    ax = axes[1, 0]
    mixing = np.zeros(T)
    for t in range(T):
        sp = spectral_analysis(R_matrices[t])
        mixing[t] = sp["mixing_ratio"]
    valid_m = mixing > 0
    ax.plot(np.where(valid_m)[0], mixing[valid_m], "g-", lw=2)
    ax.set_xlabel("Day"); ax.set_ylabel("|λ₂|/ρ")
    ax.set_title("Mixing time ratio (Birello et al. bias indicator)")
    ax.set_ylim(0, 1)

    ax = axes[1, 1]
    gt_intrinsic = gen_time_snapshots["early"]["g_network"]
    speeds = np.zeros(T)
    for t in range(T):
        es = epidemic_speed(R_matrices[t], distances, gt_intrinsic)
        speeds[t] = es["speed"]
    valid_sp = speeds > 0
    ax.plot(np.where(valid_sp)[0], speeds[valid_sp], "m-", lw=2)
    ax.set_xlabel("Day"); ax.set_ylabel("km/day")
    ax.set_title("Epidemic spatial speed v(t) = d̄/ḡ")

    plt.tight_layout()
    plt.savefig(f"{save_prefix}_06_diagnostics.png", dpi=150, bbox_inches="tight")
    plt.close()

    print(f"  Saved 6 figures with prefix '{save_prefix}'")


# ══════════════════════════════════════════════════════════════════════════════
# 10. VECTOR-BORNE QUADRUPLE INTEGRAL (FOR SECTION 3.2 - SAVE FOR NEXT SCRIPT)
# ══════════════════════════════════════════════════════════════════════════════

def compute_vector_borne_quadruple_integral(
    t, N, E_human_history, f_human, f_mosquito,
    b_VE, b_WV, b_IW, b_EI, mu_mosquito,
    max_aV, max_aW, max_aI, max_aE
):
    """
    Evaluates the Quadruple Integral Renewal Equation for Vector-Borne Diseases (Eq 42).
    Vectorized using einsum to avoid deeply nested loops.
    Note: For integration into a separate vector-borne tracking pipeline.
    """
    E_new = np.zeros(N)
    aV_grid, aW_grid, aI_grid, aE_grid = np.ogrid[0:max_aV, 0:max_aW, 0:max_aI, 0:max_aE]
    
    t_mosq_infected = t - aV_grid - aW_grid
    t_human_infected = t_mosq_infected - aI_grid - aE_grid
    valid_mask = (t_human_infected >= 0)
    
    for j in range(N):
        for k in range(N):
            for c in range(N):
                kernel_density = (
                    b_VE[k, j] * np.exp(-mu_mosquito * aV_grid) *
                    b_WV[k] * b_IW[c, k] * b_EI[c] 
                )
                
                historical_E = np.where(
                    valid_mask,
                    E_human_history[c, np.maximum(t_human_infected, 0)], 
                    0
                )
                
                integral_val = np.sum(kernel_density * historical_E * valid_mask)
                E_new[j] += integral_val
                
    return E_new


# ══════════════════════════════════════════════════════════════════════════════
# 11. MAIN ANALYSIS PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 72)
    print("MOBILITY-INFORMED RENEWAL EQUATIONS — DIRECTLY TRANSMITTED DISEASES")
    print("Full numerical implementation of Section 3.1 with PDE & Demographics")
    print("=" * 72)

    N_LOC = 10
    T = 150
    SEED = 42
    params = COVID_PARAMS
    max_days = params["max_gen_time"]

    LAMBDA_WITHIN = params["base_contact_rate"] * 1.2
    LAMBDA_BETWEEN = params["base_contact_rate"] * 0.4

    print("\n─── Step 1: City geometry and populations ───")
    coords, populations, distances = generate_city(N_LOC, "polycentric", SEED)
    print(f"  Locations: {N_LOC}")
    print(f"  Populations: {populations.astype(int)}")

    print("\n─── Step 2: Generating mobility data f_jk(t) ───")
    f_jk = generate_mobility(N_LOC, T, populations, distances, coords, seed=SEED)
    print(f"  Row sums check (all 1.0): {np.allclose(f_jk.sum(axis=-1), 1.0)}")

    print("\n─── Step 3: Epidemiological components ───")
    gen_time_pmf = discretise_gamma(params["gen_time_mean"], params["gen_time_sd"], max_days)

    infect_profile = gen_time_pmf.copy()   
    surv = np.ones(max_days)               

    gt_mean = np.sum(np.arange(max_days) * gen_time_pmf)
    print(f"  Generation time: mean={gt_mean:.2f} days (Gamma target)")

    print("\n─── Step 4: Initial R matrix and spectral analysis ───")
    R0_mat = compute_R_matrix(
        f_jk[0], f_jk[0], populations, populations,
        params["base_contact_rate"], params["prob_transmission_peak"],
        infect_profile, surv, max_days,
        LAMBDA_WITHIN, LAMBDA_BETWEEN,
    )
    spec0 = spectral_analysis(R0_mat)
    print(f"  ρ(R₀) = {spec0['rho']:.4f} (before calibration)")

    print("\n─── Step 5: Generation time distributions ───")
    gt_data = compute_generation_times(
        f_jk[0], f_jk[0], populations, populations,
        params["base_contact_rate"], params["prob_transmission_peak"],
        infect_profile, surv, max_days,
        LAMBDA_WITHIN, LAMBDA_BETWEEN,
    )
    net_gt_mean = np.sum(np.arange(max_days) * gt_data["g_network"])
    print(f"  Network-level GT mean: {net_gt_mean:.2f} days")

    print("\n─── Step 6: Source-sink analysis ───")
    ss = source_sink_analysis(R0_mat)
    print(f"  Net export: {ss['net_export']}")

    print("\n─── Step 7: Reactivity ───")
    react = reactivity(R0_mat)
    print(f"  σ = {react['sigma']:.4f}")

    print("\n─── Step 8: Sensitivity and elasticity ───")
    se = sensitivity_elasticity(R0_mat)
    print(f"  Elasticity sum (should be ≈1): {se['elasticity'].sum():.4f}")

    print("\n─── Step 9: Epidemic speed ───")
    es = epidemic_speed(R0_mat, distances, gt_data["g_network"])
    print(f"  Spatial speed: {es['speed']:.2f} km/day")

    print("\n─── Step 10: Forward simulation (PDE & Demographics) ───")
    initial = np.zeros(N_LOC)
    initial[0] = 10  

    sim = simulate_epidemic_pde(
        T, N_LOC, populations, f_jk,
        params["base_contact_rate"], params["prob_transmission_peak"],
        infect_profile, surv, max_days,
        params["R0_target"], initial,
        LAMBDA_WITHIN, LAMBDA_BETWEEN,
        birth_rate=0.00003,
        death_rate=0.00003,
        stochastic=True, susceptible_depletion=True,
        seed=SEED,
    )

    lw_scaled = sim["lambda_within_scaled"]
    lb_scaled = sim["lambda_between_scaled"]
    sc_contact = sim["scaled_contact_rate"]
    incidence = sim["incidence"]

    print(f"  Total infections: {incidence.sum():.0f}")
    print(f"  Peak day: {incidence.sum(axis=1).argmax()}")
    print(f"  Attack rate: {incidence.sum() / populations.sum() * 100:.1f}%")

    print("\n─── Step 11: Independent R(t) estimation (Cori et al.) ───")
    R_ind = estimate_R_independent(incidence, gen_time_pmf, window=7)

    R_mob_in = np.array([R_inward(sim["R_matrices"][t]) for t in range(T)])
    for j in range(min(N_LOC, 4)):
        ind_vals = R_ind[30:60, j]
        mob_vals = R_mob_in[30:60, j]
        ind_mean = np.nanmean(ind_vals)
        mob_mean = np.nanmean(mob_vals[mob_vals > 0])
        bias = (ind_mean - mob_mean) / mob_mean * 100 if mob_mean > 0 else np.nan
        print(f"    L{j}: Indep={ind_mean:.3f}, Mob-in={mob_mean:.3f}, Bias={bias:+.1f}%")

    print("\n─── Step 12: Controllability analysis ───")
    R_control = sim["R_matrices"][0]
    ctrl = minimum_control_effort(R_control, costs=populations / populations.mean())
    print(f"  Homogeneous u_min = {ctrl['u_homogeneous']:.4f} ({ctrl['u_homogeneous']*100:.1f}%)")
    print(f"  Total effort — homog: {ctrl['total_effort_homog']:.3f}, hetero: {ctrl['total_effort_hetero']:.3f}")

    print("\n─── Step 13: Generating visualisations ───")
    peak_day_idx = int(incidence.sum(axis=1).argmax())
    early_day = max(1, min(7, peak_day_idx // 3))   
    late_day  = min(T - 1, peak_day_idx + 30)        

    def _gt_snap(t_idx):
        return compute_generation_times(
            f_jk[t_idx], f_jk[t_idx], sim["susceptibles"][t_idx], populations,
            sc_contact, params["prob_transmission_peak"],
            infect_profile, surv, max_days, lw_scaled, lb_scaled,
        )

    gt_snapshots = {
        "early": _gt_snap(early_day),
        "peak":  _gt_snap(peak_day_idx),
        "late":  _gt_snap(late_day),
    }

    city_data_ext = (coords, populations, distances)
    plot_all(
        sim, city_data_ext, params, gt_snapshots,
        R_ind, spec0
    )

    print("\nDone.")
