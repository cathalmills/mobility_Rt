"""mobility_rt.generation_time."""
import numpy as np
from mobility_rt.kernel import _as_multiplier, _kernel_base, compute_R_meeting
from mobility_rt.spectral import spectral_analysis


def compute_generation_times(f_t, S, populations, prob_peak, p_aE,
                              max_days, lw, lb,
                              contact_mult=None, infect_mult=None):
    """
    All GT distributions at one time point, using a single infectiousness profile.

    PDF model (Section 3.1):
      λ^{kl}_E(t,a_E) = κ^{kl}(t) / N^l_eff(t) × p(a_E)
      K_{kj}(t,a_E) = m[k] * S_j(t) * base_K[k,j] * p(a_E)

    where base_K[k,j] = Σ_l c[l]·f^{jl}·f^{kl}·κ^{kl}/N^l_eff is a scalar
    (no a_E dependence).  Normalising:

      g_{kj}(t,a_E) = K_{kj}(t,a_E) / R_{kj}(t)
                    = [m_k S_j base_K[k,j] p(a_E)] / [m_k S_j base_K[k,j] ∫p]
                    = p(a_E) / ∫p

    The scalar m[k]·S_j·base_K[k,j] cancels exactly.  GT shape is therefore
    UNIVERSAL — identical for ALL (k,j) pairs and ALL times t.  κ^{kl} variation
    (lw vs lb) affects R_{kj} magnitudes but NOT GT shapes.

    SPATIAL HETEROGENEITY.  contact_mult c[l] (contact rate at meeting location l)
    and infect_mult m[k] (transmissibility of an infector resident in k) are the
    heterogeneous-scenario multipliers of mobility_rt.heterogeneity.  Both are
    scalars in infection age a_E, so they rescale the LEVEL of K_{kj} and cancel in
    the normalisation exactly as S_j and base_K do: the forward generation time
    stays universal under spatial heterogeneity, and only R_{kj} changes.  A
    location-dependent generation time would require a location-dependent PROFILE
    SHAPE p_k(a_E), which this model does not have.

    S is the susceptible vector the snapshot is taken at; pass sim["S_start"][t]
    to match R_matrices[t].  p_aE must have length max_days.

    Both multipliers default to None (≡ 1), reproducing the homogeneous model.
    """
    N    = len(populations)
    if len(p_aE) != max_days:
        raise ValueError(f"p_aE has length {len(p_aE)} but max_days = {max_days}")
    base_K, _, _, _, _ = _kernel_base(f_t, populations, lw, lb, contact_mult)
    m    = _as_multiplier(infect_mult, N, "infect_mult")

    # K[a, k, j] = m[k] * prob_peak * S[j] * base_K[k,j] * p_aE[a], ages a ≥ 1 only
    # (a=0 excluded to match compute_R_matrix and the forward simulator; the a=0 slot
    # stays zero so R_mat integrates Σ_{a≥1} p and the GTs normalise over a ≥ 1).
    row = m * prob_peak
    K_series = np.zeros((max_days, N, N))
    for a in range(1, max_days):
        K_series[a] = row[:, np.newaxis] * S[np.newaxis, :] * base_K * p_aE[a]

    R_mat = K_series.sum(axis=0)  # = m[k]*prob_peak*S[j]*base_K[k,j]*Σ_{a≥1} p_aE

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

    # Network level: Eqs 27-28, the pairwise kernels weighted by the sensitivities
    # s^{kj} = v_j v*_k / (vᵀv*).  Under this separable kernel it equals p/∫p too.
    g_net = eigenvector_weighted_gt(K_series, R_mat)

    # Meeting-location: g^l_meeting(a) — same universal shape.  R^l_meeting is taken
    # from compute_R_meeting (Eq 50) rather than re-derived here, so the two cannot
    # drift apart.
    R_meet = compute_R_meeting(f_t, S, populations, prob_peak, p_aE, lw, lb,
                               contact_mult, infect_mult)
    K_meet = np.zeros((max_days, N))
    tail   = float(p_aE[1:].sum())
    if tail > 1e-15:
        K_meet[1:, :] = np.outer(p_aE[1:] / tail, R_meet)
    g_meet = np.zeros_like(K_meet)
    for l in range(N):
        if R_meet[l] > 1e-15:
            g_meet[:, l] = K_meet[:, l] / R_meet[l]

    return {"g_pairwise": g_pw, "g_outward": g_out, "g_inward": g_in,
            "g_meeting": g_meet, "g_network": g_net,
            "K_series": K_series, "R_matrix": R_mat, "R_meeting": R_meet}


def eigenvector_weighted_gt(K_series, R_mat):
    """Network kernel and generation-time distribution, manuscript Eqs 27-28.

      K_network(a_E) = Σ_j Σ_k v_j v*_k K^{kj}(a_E) / (vᵀv*)        (Eq 27)
      g_network(a_E) = K_network(a_E) / ℛ                           (Eq 28)

    K_series[a, k, j] = K^{kj}(a_E) (infector k, infectee j) and R_mat[k, j] = R^{kj},
    the code orientation, i.e. the transpose of the manuscript NGM.  So the
    manuscript's v (reproductive value, Eq 26) is spectral_analysis "right_eigvec"
    and v* (stable distribution, Eq 25) is "left_eigvec", and the weight on K^{kj}
    is the sensitivity s^{kj} of Eq 31.

    When Σ_a K^{kj}(a) = R^{kj} for every pair, Σ_a g_network(a) = Σ_{kj} ε^{kj} = 1,
    so no renormalisation is applied.  Returns zeros when ℛ = 0.
    """
    spec = spectral_analysis(R_mat)
    rho  = spec["rho"]
    if rho < 1e-15:
        return np.zeros(K_series.shape[0])
    vstar, v = spec["left_eigvec"], spec["right_eigvec"]
    s = np.outer(vstar, v) / float(v @ vstar)       # s[k, j] = v_j v*_k / (vᵀv*)
    return np.einsum("kj,akj->a", s, K_series) / rho
