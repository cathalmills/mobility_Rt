"""mobility_rt.simulation."""
import numpy as np
from mobility_rt.kernel import R_system, _kernel_base, compute_R_matrix, compute_R_meeting


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


def simulate_epidemic_pde(T, n_locations, populations, f_jk_series,
                           prob_peak, infect_profile, max_days, R0_target,
                           initial_infections, lambda_within, lambda_between,
                           w_within=None, w_between=None,
                           birth_rate=0.00003, death_rate=0.00003,
                           stochastic=False, susceptible_depletion=True, seed=42):
    """
    Deterministic upwind finite-difference PDE  ∂E/∂t + ∂E/∂a_E = 0  [Eq 4].
    Boundary condition (Eq 6; renewal form Eq 8/18, evaluated exactly w/o noise):
      E_j(t,0) = S_j(t) * Σ_k base_K[k,j] * prob_peak * Σ_{a_E} p(a_E) E_k(t,a_E)

    The upwind shift E[t,k,a] = E[t-1,k,a-1] ensures E[t,k,a] = E[t-a,k,0] exactly,
    so the PDE boundary condition is algebraically identical to the discrete renewal
    equation.  No Poisson or other stochastic noise is applied; incidence is the
    direct evaluation of the BC formula.

    w_within, w_between: accepted but ignored (kept for backward compat).
    stochastic, seed: accepted but ignored (kept for backward compat).
    Vectorised O(N²) per step.
    """
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

        # Single-profile force of infection (PDF model: universal p(a_E))
        base_K, bKw, bKb, _, _ = _kernel_base(f_t, populations, lw, lb)
        # Weighted infectious pressure per infector k using single profile
        wE = prob_peak * (E_pde[t, :, 1:] @ infect_profile[1:])
        # New infections in j: S_j * Σ_k base_K[k,j] * wE[k]
        contrib_kj = base_K * wE[:, np.newaxis]  # [k,j]
        expected_j = np.maximum(S * contrib_kj.sum(axis=0), 0.0)

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


def _simulate_heun(T, n_locations, populations, f_jk_series,
                   prob_peak, infect_profile, max_days, R0_target,
                   initial_infections, lambda_within, lambda_between,
                   w_within=None, w_between=None,
                   birth_rate=0.00003, death_rate=0.00003, seed=42):
    """Deterministic epidemic using Heun's method (2nd-order Runge-Kutta)
    for the susceptible depletion step. The upwind PDE step is unchanged.
    Used for numerical validation in SI Figure 4.
    w_within and w_between accepted but ignored (single-profile PDF model)."""
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
    incidence[0]   = initial_infections
    S -= initial_infections
    for t in range(1, T):
        f_t = f_jk_series[min(t, len(f_jk_series) - 1)]
        E_pde[t, :, 1:] = E_pde[t - 1, :, :-1]
        base_K, bKw, bKb, _, _ = _kernel_base(f_t, populations, lw, lb)
        wE = prob_peak * (E_pde[t, :, 1:] @ infect_profile[1:])
        contrib_kj = base_K * wE[:, np.newaxis]
        # Predictor (Forward Euler)
        new_j  = np.minimum(np.maximum(S * contrib_kj.sum(axis=0), 0.0), S)
        S_pred = np.maximum(S - new_j, 0.0)
        # Corrector: re-evaluate force of infection at S_pred
        new_j2 = np.minimum(np.maximum(S_pred * contrib_kj.sum(axis=0), 0.0), S_pred)
        # R_matrices[t] uses the pre-update susceptibles S(t-1), matching the reference
        # integrator (so each solver's R series is like-for-like with its incidence).
        R_matrices[t]  = compute_R_matrix(f_t, S, populations, prob_peak,
                                           infect_profile, lw, lb)
        # Average slopes (Heun's rule); same birth/death demography as the
        # reference upwind integrator so the convergence comparison is like-for-like.
        S = np.maximum(S - 0.5 * (new_j + new_j2)
                       + birth_rate * populations - death_rate * S, 0.0)
        E_pde[t, :, 0] = 0.5 * (new_j + new_j2)
        incidence[t]   = E_pde[t, :, 0]
    return {"incidence": incidence, "R_matrices": R_matrices}


def _simulate_rk4(T, n_locations, populations, f_jk_series,
                  prob_peak, infect_profile, max_days, R0_target,
                  initial_infections, lambda_within, lambda_between,
                  w_within=None, w_between=None,
                  birth_rate=0.00003, death_rate=0.00003, seed=42):
    """Deterministic epidemic using 4th-order Runge-Kutta for S(t).
    The upwind PDE step is unchanged. Used for numerical validation.
    w_within and w_between accepted but ignored (single-profile PDF model)."""
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
    incidence[0]   = initial_infections
    S -= initial_infections
    for t in range(1, T):
        f_t = f_jk_series[min(t, len(f_jk_series) - 1)]
        E_pde[t, :, 1:] = E_pde[t - 1, :, :-1]
        base_K, bKw, bKb, _, _ = _kernel_base(f_t, populations, lw, lb)
        wE = prob_peak * (E_pde[t, :, 1:] @ infect_profile[1:])
        contrib = (base_K * wE[:, np.newaxis]).sum(axis=0)
        # RK4 for dS/dt = -S * contrib (treating contrib as fixed over the step)
        k1 = -np.minimum(S * contrib, S)
        k2 = -np.minimum(np.maximum(S + 0.5*k1, 0.0) * contrib, np.maximum(S + 0.5*k1, 0.0))
        k3 = -np.minimum(np.maximum(S + 0.5*k2, 0.0) * contrib, np.maximum(S + 0.5*k2, 0.0))
        k4 = -np.minimum(np.maximum(S + k3, 0.0) * contrib, np.maximum(S + k3, 0.0))
        delta_S = (k1 + 2*k2 + 2*k3 + k4) / 6.0
        new_j  = np.minimum(np.maximum(-delta_S, 0.0), S)
        # R_matrices[t] uses the pre-update susceptibles S(t-1), matching the reference.
        R_matrices[t]  = compute_R_matrix(f_t, S, populations, prob_peak,
                                           infect_profile, lw, lb)
        # Same birth/death demography as the reference upwind integrator so the
        # convergence comparison is like-for-like.
        S = np.maximum(S - new_j
                       + birth_rate * populations - death_rate * S, 0.0)
        E_pde[t, :, 0] = new_j
        incidence[t]   = new_j
    return {"incidence": incidence, "R_matrices": R_matrices}
