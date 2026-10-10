"""mobility_rt.simulation — forward simulation of the mobility-informed renewal PDE."""
import numpy as np
from mobility_rt.kernel import R_system, _as_multiplier, _kernel_base, compute_R_matrix, compute_R_meeting


def compute_effective_populations_series(f_jk, S_start, incidence,
                                         gen_time_pmf, max_days):
    """S^l_eff(t) = Σ_j f_{jl}(t) S_j(t)  and  I^l_eff(t) = Σ_k f_{kl}(t) Λ_k(t).

    Two simplifications, both deliberate: there is no same-day transmission (the
    force of infection runs over ages a ≥ 1), and mobility is one share per day, so
    nobody moves within a day.  Susceptible depletion WITHIN the day is not a
    simplification any more — the default "exponential" update depletes S through
    the day — but these series are start-of-day quantities regardless.

    S_start must be sim["S_start"], the START-of-day susceptibles: the S from which
    R_matrices[t] and the day-t force of infection are computed.  Λ_k(t) =
    Σ_{a=1}^{max_days-1} p(a) I_k(t-a) uses incidence up to day t-1 only, so both
    series describe the start of day t.
    """
    T, N = incidence.shape
    S_eff = np.zeros((T, N))
    I_eff = np.zeros((T, N))
    for t in range(T):
        S_eff[t] = f_jk[t].T @ S_start[t]
        lam = np.zeros(N)
        for s in range(1, min(max_days, t + 1)):
            lam += gen_time_pmf[s] * incidence[t - s]
        I_eff[t] = f_jk[t].T @ lam
    return S_eff, I_eff


def phase_days(incidence, T=None):
    """Early, peak and late days of one run, used by the snapshot figures.

      peak  = day of maximum network incidence
      early = max(1, peak // 3), a third of the way to the peak; day 0 is excluded
              because R_matrices[0] uses S = N and no transmission has happened yet
      late  = min(T - 1, peak + 30), about 5.5 mean generation times after the peak

    The days are relative to each run's own peak, so runs that spread at different
    speeds are compared at the same stage.
    """
    inc = np.asarray(incidence, dtype=float)
    T = inc.shape[0] if T is None else int(T)
    peak = int(inc.sum(axis=1).argmax())
    return {"early": max(1, peak // 3), "peak": peak, "late": min(T - 1, peak + 30)}


def _calibrate(f_calibration, populations, prob_peak, infect_profile, R0_target,
               lambda_within, lambda_between, contact_mult, infect_mult):
    """One common scale on (λ_W, λ_B) giving ρ(R) = R0_target at S = N under f_calibration.

    R is linear in (λ_W, λ_B) at a fixed ratio
    Returns (lw, lb, rho_unscaled, scale).
    """
    f_cal = np.asarray(f_calibration, dtype=float)
    N = len(populations)
    if f_cal.shape != (N, N):
        raise ValueError(f"f_calibration must have shape ({N}, {N}), got {f_cal.shape}")
    if not np.allclose(f_cal.sum(axis=1), 1.0, rtol=0.0, atol=1e-10):
        raise ValueError("f_calibration must be row-stochastic")
    rho_unscaled = R_system(compute_R_matrix(f_cal, populations, populations, prob_peak,
                                             infect_profile, lambda_within, lambda_between,
                                             contact_mult, infect_mult))
    if not rho_unscaled > 0.0:
        raise ValueError("the uncalibrated R matrix has spectral radius 0; cannot calibrate")
    scale = R0_target / rho_unscaled
    return lambda_within * scale, lambda_between * scale, rho_unscaled, scale


def simulate_epidemic_pde(T, n_locations, populations, f_jk_series, prob_peak,
                          infect_profile, max_days, R0_target, initial_infections,
                          lambda_within, lambda_between, *, f_calibration,
                          contact_mult=None, infect_mult=None,
                          update="exponential"):
    """
    Deterministic upwind scheme for ∂E/∂t + ∂E/∂a_E = 0 (Eq 4) on a uniform grid
    Δt = Δa_E = 1 day.  The day-t force of infection on a susceptible resident of j
    is the boundary condition (Eq 6),

      φ_j(t) = Σ_k base_K[k, j] m_k prob_peak Σ_{a≥1} p(a) E_k(t, a),

    evaluated at the day-t mobility and the START-of-day susceptibles S_j(t), with
    base_K the mobility kernel of _kernel_base.
    Age and time in same units, so the upwind shift
    E[t, k, a] = E[t-1, k, a-1] gives E[t, k, a] = E[t-a, k, 0] exactly.

    `update` selects how S is depleted over the day, at the same Δt = 1 day:
      "exponential" (default)  E_j(t, 0) = S_j(t) (1 − exp(−φ_j(t))), the exact
          solution of dS_j/dt = −S_j φ_j over the day at fixed φ: susceptibles are
          depleted THROUGHOUT the day, as in the continuous model from which the
          renewal equation is derived.  It never exceeds S_j(t), so no cap is needed.
      "euler"  E_j(t, 0) = min(S_j(t) φ_j(t), S_j(t)), forward Euler: every infection
          on day t draws on the start-of-day susceptibles.  First order in Δt.  Being
          linear in S, it alone makes the discrete renewal identity
          I_j(t) = [R(t)ᵀ Λ(t)]_j / Σ_{a≥1} p(a) hold exactly; under "exponential" it
          holds to O(φ/2) (φ/2 ≤ 1.2 % at R0 = 1.5, ≤ 10.5 % at R0 = 3.0, Scenario A).
          That identity is a property of the Euler scheme, not of the model.  Kept for
          the SI4 comparison.

    Accuracy against a refined grid (Δt = Δa_E = 1/64 d, Scenario A, T = 365 d, the
    continuous Gamma(5.5, 1.8) profile binned at the refined step; reproduced by
    test_refined_grid_limit).  Attack rate in percentage points of the population,
    peak in per cent of the refined-grid peak:

                        R0 = 1.5              R0 = 3.0
                    attack pp   peak %    attack pp   peak %
      exponential     +0.014     -0.20      -0.009     +1.45
      euler           +0.611     +2.30      +1.147     +8.18

    Both daily schemes recover the attack rate to well under a tenth of a percentage
    point on the exponential update; forward Euler does not, and overstates the peak
    several-fold more.  The refined grid is itself converged: m = 16 differs from
    m = 64 by 0.011 pp (R0 = 1.5) and 0.003 pp (R0 = 3.0).

    Calibration: λ_W and λ_B are multiplied by one common factor so that
    ρ(R) = R0_target for a fully susceptible population (S = N) under the mobility
    matrix f_calibration (see _calibrate).

    Spatial heterogeneity for simulations (both default None ≡ homogeneous):
      contact_mult c[l]  contact-rate multiplier at MEETING location l
      infect_mult  m[k]  transmission-probability multiplier of an infector resident
                         in k (scales row k of R and k's force of infection)

    Closed population: there are no births or deaths, so
    S_end[t] + Σ_{s<=t} incidence[s] = N for every t.

    Returned arrays, t = 0..T-1:
      incidence[t, j]           new infections in residents of j on day t
                                (incidence[0] is the seed)
      S_start[t, j]             susceptibles at the start of day t; S_start[0] = N
      S_end[t, j]               susceptibles at the end of day t; S_start[t+1] = S_end[t]
      R_matrices[t, k, j]       R^{kj}(t), from f_jk_series[t] and S_start[t]
      R_meeting_series[t, l]    R^l_meeting(t), from the same f and S
      incidence_matrix[t, k, j] new infections on day t in residents of j caused by
                                residents of k (zero on day 0)
      E_pde_state[t, k, a]      E_k(t, a) for ages a = 0..max_days-1
    """
    N = n_locations
    if update not in ("exponential", "euler"):
        raise ValueError(f"update must be 'exponential' or 'euler', got {update!r}")
    populations = np.asarray(populations, dtype=float)
    infect_mult_v = _as_multiplier(infect_mult, N, "infect_mult")
    lw, lb, rho_unscaled, scale = _calibrate(
        f_calibration, populations, prob_peak, infect_profile, R0_target,
        lambda_within, lambda_between, contact_mult, infect_mult)
    print(f"  Calibration: ρ(R) at S = N under f_calibration = {rho_unscaled:.4f} "
          f"before scaling; scale = {scale:.4f}; target = {R0_target}")

    S = populations.copy()
    S_start          = np.zeros((T, N))
    S_end            = np.zeros((T, N))
    E_pde            = np.zeros((T, N, max_days))
    incidence_matrix = np.zeros((T, N, N))
    R_matrices       = np.zeros((T, N, N))
    R_meeting_series = np.zeros((T, N))

    S_start[0]     = S
    E_pde[0, :, 0] = initial_infections
    S             -= initial_infections
    S_end[0]       = S

    f0 = f_jk_series[0]
    R_matrices[0]       = compute_R_matrix(f0, populations, populations, prob_peak,
                                           infect_profile, lw, lb,
                                           contact_mult, infect_mult)
    R_meeting_series[0] = compute_R_meeting(f0, populations, populations,
                                            prob_peak, infect_profile, lw, lb,
                                            contact_mult, infect_mult)

    for t in range(1, T):
        S_start[t] = S                  # S here is the end of day t-1
        f_t = f_jk_series[min(t, len(f_jk_series) - 1)]

        # PDE upwind step
        E_pde[t, :, 1:] = E_pde[t - 1, :, :-1]

        # Force of infection from the single universal profile p(a_E), with each
        # infector k's own transmissibility m[k]
        base_K, _, _, _, _ = _kernel_base(f_t, populations, lw, lb, contact_mult)
        wE = infect_mult_v * prob_peak * (E_pde[t, :, 1:] @ infect_profile[1:])
        contrib_kj = base_K * wE[:, np.newaxis]  # [k,j]
        foi_j = np.maximum(contrib_kj.sum(axis=0), 0.0)   # φ_j(t), per susceptible
        if update == "exponential": #the default is exponential (which I recommend)
            new_j = S * (-np.expm1(-foi_j))    # depletion throughout the day
        else:
            new_j = np.minimum(S * foi_j, S)   # forward Euler, capped at S

        # Distribute pairwise incidence proportional to contrib_kj
        col_sum = contrib_kj.sum(axis=0) #sum over rows
        for j in range(N): #incidence for each location j 
            if col_sum[j] > 1e-15 and new_j[j] > 0:
                incidence_matrix[t, :, j] = new_j[j] * contrib_kj[:, j] / col_sum[j]

        E_pde[t, :, 0] = new_j

        R_matrices[t]       = compute_R_matrix(f_t, S, populations, prob_peak,
                                               infect_profile, lw, lb,
                                               contact_mult, infect_mult)
        R_meeting_series[t] = compute_R_meeting(f_t, S, populations, prob_peak,
                                                infect_profile, lw, lb,
                                                contact_mult, infect_mult)

        S = np.maximum(S - new_j, 0.0)
        S_end[t] = S

    return {"incidence":             E_pde[:, :, 0],
            "incidence_matrix":      incidence_matrix,
            "R_matrices":            R_matrices,
            "R_meeting_series":      R_meeting_series,
            "S_start":               S_start,
            "S_end":                 S_end,
            "lambda_within_scaled":  lw,
            "lambda_between_scaled": lb,
            "contact_mult":          contact_mult,
            "infect_mult":           infect_mult,
            "update":                update,
            "E_pde_state":           E_pde}



#Simulators for checks
def _simulate_heun(T, n_locations, populations, f_jk_series, prob_peak,
                   infect_profile, max_days, R0_target, initial_infections,
                   lambda_within, lambda_between, *, f_calibration,
                   contact_mult=None, infect_mult=None):
    """Heun (predictor-corrector) update of the susceptible depletion, with the upwind
    age step, boundary condition and calibration of simulate_epidemic_pde.  A
    different update rule at the same Δt = 1 day, not a Δt refinement.  Used by
    Fig SI4C."""
    N  = n_locations
    populations = np.asarray(populations, dtype=float)
    infect_mult_v = _as_multiplier(infect_mult, N, "infect_mult")
    lw, lb, _, _ = _calibrate(f_calibration, populations, prob_peak, infect_profile,
                              R0_target, lambda_within, lambda_between,
                              contact_mult, infect_mult)
    S  = populations.copy()
    incidence  = np.zeros((T, N))
    R_matrices = np.zeros((T, N, N))
    f0     = f_jk_series[0]
    R_matrices[0] = compute_R_matrix(f0, populations, populations, prob_peak,
                                     infect_profile, lw, lb,
                                     contact_mult, infect_mult)
    E_pde = np.zeros((T, N, max_days))
    E_pde[0, :, 0] = initial_infections
    incidence[0]   = initial_infections
    S -= initial_infections
    for t in range(1, T):
        f_t = f_jk_series[min(t, len(f_jk_series) - 1)]
        E_pde[t, :, 1:] = E_pde[t - 1, :, :-1]
        base_K, _, _, _, _ = _kernel_base(f_t, populations, lw, lb, contact_mult)
        wE = infect_mult_v * prob_peak * (E_pde[t, :, 1:] @ infect_profile[1:])
        contrib_kj = base_K * wE[:, np.newaxis]
        # Predictor (Forward Euler)
        new_j  = np.minimum(np.maximum(S * contrib_kj.sum(axis=0), 0.0), S)
        S_pred = np.maximum(S - new_j, 0.0)
        # Corrector: re-evaluate force of infection at S_pred
        new_j2 = np.minimum(np.maximum(S_pred * contrib_kj.sum(axis=0), 0.0), S_pred)
        # R_matrices[t] uses the start-of-day susceptibles, as in simulate_epidemic_pde.
        R_matrices[t]  = compute_R_matrix(f_t, S, populations, prob_peak,
                                          infect_profile, lw, lb,
                                          contact_mult, infect_mult)
        # Average slopes (Heun's rule).
        S = np.maximum(S - 0.5 * (new_j + new_j2), 0.0)
        E_pde[t, :, 0] = 0.5 * (new_j + new_j2)
        incidence[t]   = E_pde[t, :, 0]
    return {"incidence": incidence, "R_matrices": R_matrices}


def _simulate_rk4(T, n_locations, populations, f_jk_series, prob_peak,
                  infect_profile, max_days, R0_target, initial_infections,
                  lambda_within, lambda_between, *, f_calibration,
                  contact_mult=None, infect_mult=None):
    """Fourth-order Runge-Kutta update of S over one day with the force of infection
    held fixed, and the upwind age step, boundary condition and calibration of
    simulate_epidemic_pde.  A different update rule at the same Δt = 1 day, not a
    Δt refinement.  Used by Fig SI4C."""
    N  = n_locations
    populations = np.asarray(populations, dtype=float)
    infect_mult_v = _as_multiplier(infect_mult, N, "infect_mult")
    lw, lb, _, _ = _calibrate(f_calibration, populations, prob_peak, infect_profile,
                              R0_target, lambda_within, lambda_between,
                              contact_mult, infect_mult)
    S  = populations.copy()
    incidence  = np.zeros((T, N))
    R_matrices = np.zeros((T, N, N))
    f0     = f_jk_series[0]
    R_matrices[0] = compute_R_matrix(f0, populations, populations, prob_peak,
                                     infect_profile, lw, lb,
                                     contact_mult, infect_mult)
    E_pde = np.zeros((T, N, max_days))
    E_pde[0, :, 0] = initial_infections
    incidence[0]   = initial_infections
    S -= initial_infections
    for t in range(1, T):
        f_t = f_jk_series[min(t, len(f_jk_series) - 1)]
        E_pde[t, :, 1:] = E_pde[t - 1, :, :-1]
        base_K, _, _, _, _ = _kernel_base(f_t, populations, lw, lb, contact_mult)
        wE = infect_mult_v * prob_peak * (E_pde[t, :, 1:] @ infect_profile[1:])
        contrib = (base_K * wE[:, np.newaxis]).sum(axis=0)
        # RK4 for dS/dt = -S * contrib (treating contrib as fixed over the step)
        k1 = -np.minimum(S * contrib, S)
        k2 = -np.minimum(np.maximum(S + 0.5*k1, 0.0) * contrib, np.maximum(S + 0.5*k1, 0.0))
        k3 = -np.minimum(np.maximum(S + 0.5*k2, 0.0) * contrib, np.maximum(S + 0.5*k2, 0.0))
        k4 = -np.minimum(np.maximum(S + k3, 0.0) * contrib, np.maximum(S + k3, 0.0))
        delta_S = (k1 + 2*k2 + 2*k3 + k4) / 6.0
        new_j  = np.minimum(np.maximum(-delta_S, 0.0), S)
        # R_matrices[t] uses the start-of-day susceptibles, as in simulate_epidemic_pde.
        R_matrices[t]  = compute_R_matrix(f_t, S, populations, prob_peak,
                                          infect_profile, lw, lb,
                                          contact_mult, infect_mult)
        S = np.maximum(S - new_j, 0.0)
        E_pde[t, :, 0] = new_j
        incidence[t]   = new_j
    return {"incidence": incidence, "R_matrices": R_matrices}
