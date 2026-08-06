"""mobility_rt.kernel."""
import numpy as np


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


def compute_R_matrix(f_t, S, populations, prob_peak, infect_profile, lw, lb):
    """R_{kj}(t) = prob_peak * Σ_{a≥1} p(a) * S[j] * base_K[k,j].

    The integral over infection age uses ages a ≥ 1 (infect_profile[1:]), matching
    the forward simulator's force of infection (E_pde[:,:,1:] @ infect_profile[1:]),
    which excludes same-day (a=0) transmission.  Using the full Σ_{a≥0} p(a) would
    overstate R by 1/(1-p[0]) and make the realised R0 = R0_target·(1-p[0]); for the
    Hart GT p[0]≈4e-6 (negligible) but this keeps the NGM exactly consistent with the
    dynamics for any (e.g. short) generation time.
    """
    base_K, _, _, _, _ = _kernel_base(f_t, populations, lw, lb)
    return prob_peak * infect_profile[1:].sum() * S[np.newaxis, :] * base_K


def R_outward(R_mat):
    return R_mat.sum(axis=1)


def R_inward(R_mat):
    return R_mat.sum(axis=0)


def R_system(R_mat):
    ev = np.linalg.eigvals(R_mat)
    return float(np.max(np.abs(ev)))


def compute_R_meeting(f_t, S, populations, prob_peak, infect_profile, lw, lb):
    """
    R^l_meeting(t) = S^l_eff * κ_eff(l) * prob_peak * Σ_{a≥1} p(a) / N^l_eff
    κ_eff(l) = lw*f[l,l] + lb*Σ_{k≠l} f[k,l]
    (Σ over a ≥ 1, consistent with compute_R_matrix and the forward simulator.)
    """
    N        = f_t.shape[0]
    _, _, _, N_eff, inv_Neff = _kernel_base(f_t, populations, lw, lb)
    S_eff    = f_t.T @ S
    sum_w    = infect_profile[1:].sum()
    R_meet   = np.zeros(N)
    for l in range(N):
        kappa = lw * f_t[l, l] + lb * (f_t[:, l].sum() - f_t[l, l])
        R_meet[l] = prob_peak * sum_w * S_eff[l] * kappa * inv_Neff[l]
    return R_meet
