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
    SHAPE p_k(a_E), which this model does not have.  What does vary across space is
    the REALISED (backward) generation interval — see realised_generation_times.

    Both multipliers default to None (≡ 1), reproducing the homogeneous model.
    """
    N    = len(populations)
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

    # Network-level: g_net(a) — universal p_aE/∫p
    K_tot = K_series.sum(axis=(1, 2));  R_tot = R_mat.sum()
    g_net = K_tot / R_tot if R_tot > 1e-15 else np.zeros(max_days)

    # Meeting-location: g^l_meeting(a) — same universal shape.  R^l_meeting is taken
    # from compute_R_meeting rather than re-derived here, so the meeting-location
    # κ_eff(l) = c[l]·(lw·f[l,l] + lb·Σ_{k≠l} f[k,l]) and the presence-weighted
    # infectiousness m̄[l] cannot drift between the two implementations.
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


def realised_generation_times(E_pde, incidence_matrix, p_aE):
    """Backward (cohort) generation intervals actually realised in a simulation.

    compute_generation_times returns the FORWARD generation time built into the
    kernel, which is universal: every (k,j) pair at every time has the same shape
    p(a_E)/∫p, with or without spatial heterogeneity in contact rate and
    transmissibility.  The distribution an analyst would actually reconstruct from
    transmission pairs is the BACKWARD one — how long ago the infector of each new
    case was itself infected — and that is NOT universal.  It reweights p(a_E) by
    the infector's own infection-age composition, so it is early-shifted while a
    location is growing and late-shifted while it is declining.  Because local
    epidemics peak at different times, it varies across space and over time.

    New infections at t attributable to an infector resident in k of infection age a
    are proportional to base_K[k,j]·m[k]·E^k(t,a)·p(a) (the simulator's boundary
    condition, resolved by age).  Splitting that product gives an age weight that
    depends on k only,

        φ_k(t,a) ∝ E^k(t,a) · p(a),   a ≥ 1,

    so the infector-side backward GT needs nothing but the infection-age state, and
    the infectee-side and network distributions are mixtures of the φ_k with weights
    given by the pairwise incidence E^{kj}(t) already stored by the simulator.  No
    part of this reconstruction re-derives the kernel, so it stays exact under any
    contact_mult / infect_mult.

    Parameters
    ----------
    E_pde : (T, N, max_days)      sim["E_pde_state"], E^k(t,a).
    incidence_matrix : (T, N, N)  sim["incidence_matrix"], E^{kj}(t) as [t, k, j].
    p_aE : (max_days,)            infectiousness profile p(a_E).

    Returns
    -------
    dict with, for infector k, infectee j and the network as a whole, the backward
    GT pmf over a_E (rows sum to 1 where defined, to 0 where no transmission is
    happening) and its mean in days (NaN where undefined):
      g_infector (T,N,max_days)  mean_infector (T,N)
      g_infectee (T,N,max_days)  mean_infectee (T,N)
      g_network  (T,max_days)    mean_network  (T,)
    """
    E_pde = np.asarray(E_pde, dtype=float)
    incidence_matrix = np.asarray(incidence_matrix, dtype=float)
    p_aE = np.asarray(p_aE, dtype=float)
    if E_pde.ndim != 3:
        raise ValueError(f"E_pde must be (T, N, max_days), got shape {E_pde.shape}")
    T, N, A = E_pde.shape
    if p_aE.shape != (A,):
        raise ValueError(f"p_aE must have length {A}, got {p_aE.shape}")
    if incidence_matrix.shape != (T, N, N):
        raise ValueError(f"incidence_matrix must be ({T}, {N}, {N}), "
                         f"got {incidence_matrix.shape}")
    days = np.arange(A, dtype=float)

    def _norm(W):
        """Normalise the last axis where it carries mass; zero rows elsewhere."""
        tot = W.sum(axis=-1)
        out = np.zeros_like(W)
        ok  = tot > 0.0
        out[ok] = W[ok] / tot[ok][..., np.newaxis]
        return out, ok

    # φ_k(t,a) ∝ E^k(t,a)·p(a), ages a ≥ 1 (a = 0 cannot be an infector age here,
    # exactly as in compute_R_matrix and the forward simulator).
    W_inf = E_pde * p_aE[np.newaxis, np.newaxis, :]
    W_inf[:, :, 0] = 0.0
    g_infector, ok_inf = _norm(W_inf)

    # Infectee j mixes the infector distributions with the pairwise incidence it
    # actually received: b_j(t,a) = Σ_k E^{kj}(t)·φ_k(t,a) / Σ_k E^{kj}(t).
    W_intee = np.einsum("tkj,tka->tja", incidence_matrix, g_infector)
    g_infectee, ok_intee = _norm(W_intee)

    # Network: the same mixture over every infectee as well.
    W_net = np.einsum("tja->ta", W_intee)
    g_network, ok_net = _norm(W_net)

    def _mean(g, ok):
        mu = g @ days
        mu[~ok] = np.nan
        return mu

    return {"g_infector":   g_infector,
            "g_infectee":   g_infectee,
            "g_network":    g_network,
            "mean_infector": _mean(g_infector, ok_inf),
            "mean_infectee": _mean(g_infectee, ok_intee),
            "mean_network":  _mean(g_network,  ok_net)}


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
