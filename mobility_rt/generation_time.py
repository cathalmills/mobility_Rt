"""mobility_rt.generation_time."""
import numpy as np
from mobility_rt.kernel import _kernel_base
from mobility_rt.spectral import spectral_analysis


def compute_generation_times(f_t, S, populations, prob_peak, p_aE,
                              max_days, lw, lb):
    """
    All GT distributions at one time point, using a single infectiousness profile.

    PDF model (Section 3.1):
      λ^{kl}_E(t,a_E) = κ^{kl}(t) / N^l_eff(t) × p(a_E)
      K_{kj}(t,a_E) = S_j(t) * base_K[k,j] * p(a_E)

    where base_K[k,j] = Σ_l f^{jl}·f^{kl}·κ^{kl}/N^l_eff is a scalar
    (no a_E dependence).  Normalising:

      g_{kj}(t,a_E) = K_{kj}(t,a_E) / R_{kj}(t)
                    = [S_j * base_K[k,j] * p(a_E)] / [S_j * base_K[k,j] * ∫p]
                    = p(a_E) / ∫p

    The scalar base_K[k,j] cancels exactly.  GT shape is therefore UNIVERSAL —
    identical for ALL (k,j) pairs and ALL times t.  κ^{kl} variation (lw vs lb)
    affects R_{kj} magnitudes but NOT GT shapes.
    """
    N    = len(populations)
    base_K, bKw, bKb, N_eff, inv_Neff = _kernel_base(f_t, populations, lw, lb)
    S_eff = f_t.T @ S

    # K[a, k, j] = S[j] * prob_peak * base_K[k,j] * p_aE[a], for ages a ≥ 1 only
    # (a=0 excluded to match compute_R_matrix and the forward simulator; the a=0 slot
    # stays zero so R_mat integrates Σ_{a≥1} p and the GTs normalise over a ≥ 1).
    K_series = np.zeros((max_days, N, N))
    for a in range(1, max_days):
        K_series[a] = prob_peak * S[np.newaxis, :] * base_K * p_aE[a]

    R_mat = K_series.sum(axis=0)  # = prob_peak * S[j] * base_K[k,j] * Σ_{a≥1} p_aE

    # Pairwise: g_{kj}(a) = p_aE/∫p for all (k,j) — universal GT
    g_pw = np.zeros_like(K_series)
    for k in range(N):
        for j in range(N):
            if R_mat[k, j] > 1e-15:
                g_pw[:, k, j] = K_series[:, k, j] / R_mat[k, j]

    # Outward: g^k_out(a) — same universal shape
    K_out = K_series.sum(axis=2);  R_out = R_mat.sum(axis=1)
    g_out = np.zeros_like(K_out)
    for k in range(N):
        if R_out[k] > 1e-15:
            g_out[:, k] = K_out[:, k] / R_out[k]

    # Inward: g^j_in(a) — same universal shape
    K_in = K_series.sum(axis=1);  R_in = R_mat.sum(axis=0)
    g_in = np.zeros_like(K_in)
    for j in range(N):
        if R_in[j] > 1e-15:
            g_in[:, j] = K_in[:, j] / R_in[j]

    # Network-level: g_net(a) — universal p_aE/∫p
    K_tot = K_series.sum(axis=(1, 2));  R_tot = R_mat.sum()
    g_net = K_tot / R_tot if R_tot > 1e-15 else np.zeros(max_days)

    # Meeting-location: g^l_meeting(a) — same universal shape
    K_meet = np.zeros((max_days, N))
    for l in range(N):
        kappa_w = lw * f_t[l, l]
        kappa_b = lb * max(f_t[:, l].sum() - f_t[l, l], 0.0)
        for a in range(1, max_days):   # a ≥ 1, as elsewhere
            K_meet[a, l] = prob_peak * (kappa_w + kappa_b) * p_aE[a] * S_eff[l] * inv_Neff[l]
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
