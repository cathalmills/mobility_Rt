"""mobility_rt.kernel."""
import numpy as np


def _as_multiplier(mult, n, name):
    """Validate an optional per-location multiplier; None → ones (no-op).

    Returned as float64 of length n.  A multiplier must be finite and non-negative:
    a negative contact rate or infectiousness has no meaning and would silently
    produce negative entries in R.
    """
    if mult is None:
        return np.ones(n, dtype=float)
    arr = np.asarray(mult, dtype=float)
    if arr.ndim == 0:
        arr = np.full(n, float(arr))
    if arr.shape != (n,):
        raise ValueError(f"{name} must be a scalar or length-{n} vector, got shape {arr.shape}")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must be finite")
    if np.any(arr < 0.0):
        raise ValueError(f"{name} must be non-negative")
    return arr


def _kernel_base(f_t, populations, lw, lb, contact_mult=None):
    """Precompute base_K[k,j] and its within/between split for ONE time-slice.

    Inputs:
      f_t          mobility for this day, f_t[j, l] = share of j's residents at l
      populations  resident population of each location
      lw           base contact rate (POLYMOD)
      lb           lw x the away-from-home multiplier (lower transmission away
                   from the home location)

    Two-component decomposition (c[l] = contact_mult at MEETING location l):
      bK_within[k,j] = lw * c[k] * f[j,k] * f[k,k] / N_eff[k]
          (infector k and infectee j both at home location k, so they meet at k)
      bK_between[k,j] = lb * Σ_{l≠k} c[l] * f[j,l] * f[k,l] / N_eff[l]
          (infector k and infectee j meet away from k, at l)
      base_K[k,j] = bK_within[k,j] + bK_between[k,j]

    contact_mult scales the contact rate of the location where the contact HAPPENS,
    not of the infector's home: a resident of a low-contact suburb who commutes into
    a high-contact core makes core-rate contacts while there.  contact_mult=None
    (the default) is c ≡ 1 and reproduces the homogeneous kernel exactly.

    D[j,k] = Σ_l c[l]*f[j,l]*f[k,l]/N_eff[l]  (all meeting locations)
    D^T[k,j] - within[k,j] = Σ_{l≠k} c[l]*f[j,l]*f[k,l]/N_eff[l]  (away meetings)

    Returns: base_K, bK_within, bK_between, N_eff, inv_Neff
    """

    # validate the multiplier (None -> ones)
    c        = _as_multiplier(contact_mult, f_t.shape[0], "contact_mult")
    # effective population present at each meeting location l on this day
    N_eff    = f_t.T @ populations
    inv_Neff = np.where(N_eff > 0, 1.0 / N_eff, 0.0)
    c_inv    = c * inv_Neff                          # c[l]/N_eff[l]
    A        = f_t * c_inv[np.newaxis, :]            # A[j,l] = c[l]*f[j,l]/N_eff[l]
    D        = A @ f_t.T                             # D[j,k] = Σ_l A[j,l]*f[k,l] at time t
    within   = f_t.T * (f_t.diagonal() * c_inv)[:, np.newaxis]  # [k,j], meeting at k
    bK_within  = lw * within
    bK_between = lb * np.maximum(D.T - within, 0.0)  # away-meeting component ≥ 0
    base_K     = bK_within + bK_between
    return base_K, bK_within, bK_between, N_eff, inv_Neff


def compute_R_matrix(f_t, S, populations, prob_peak, infect_profile, lw, lb,
                     contact_mult=None, infect_mult=None):
    """R_{kj}(t) = m[k] * prob_peak * Σ_{a≥1} p(a) * S[j] * base_K[k,j].

    ORIENTATION (deliberate, and the transpose of the manuscript's NGM).  Row k is
    the INFECTOR's home location, column j the INFECTEE's, so R[k, j] is transmission
    from residents of k to residents of j.  The manuscript's next-generation matrix
    K^D is the transpose of this, K^D[j, k] = R[k, j]: that is why the forward step
    is R.T @ Λ, why spectral_analysis's "right_eigvec" (of R) is the manuscript's
    reproductive value and its "left_eigvec" the stable distribution, and why
    ρ is unchanged (a matrix and its transpose share a spectrum).

    The integral over infection age uses ages a ≥ 1 (infect_profile[1:]), matching
    the forward simulator's force of infection (E_pde[:,:,1:] @ infect_profile[1:]),
    which excludes same-day (a=0) transmission.  Using the full Σ_{a≥0} p(a) would
    overstate R by 1/(1-p[0]); for the Hart generation time p[0] ≈ 4e-6, negligible,
    but the a ≥ 1 convention keeps the NGM exactly consistent with the dynamics for
    any (e.g. short) generation time.

    contact_mult c[l] scales the contact rate at MEETING location l (see _kernel_base).    
    infect_mult  m[k] scales the transmission probability of an infector resident in k,
    so it multiplies row k, i.e. everyone that infector k infects. 
    I've included to allow for spatially heterogeneous contact rates in different locations and 
        spatially heterogenous transmission probabilities from residents of different locations.
    However, I/you can change components of R_{kj} in so many more ways, and likely more interesting ways!
    Both default to None (≡ 1)
    """
    base_K, _, _, _, _ = _kernel_base(f_t, populations, lw, lb, contact_mult)
    m = _as_multiplier(infect_mult, f_t.shape[0], "infect_mult")
    row = m * prob_peak * infect_profile[1:].sum()       # per-infector scale, [k]
    return row[:, np.newaxis] * S[np.newaxis, :] * base_K


def R_outward(R_mat):
    return R_mat.sum(axis=1) #sum over cols 


def R_inward(R_mat):
    return R_mat.sum(axis=0)


def R_system(R_mat):
    ev = np.linalg.eigvals(R_mat)
    return float(np.max(np.abs(ev))) #spectral radius (same for transpose)


def compute_R_meeting(f_t, S, populations, prob_peak, infect_profile, lw, lb,
                      contact_mult=None, infect_mult=None):
    """Meeting-location reproduction numbers R^l_meeting(t), manuscript Eq 50.

      R^l_meeting(t) = Σ_j Σ_k f^{jl}(t) S_j(t) f^{kl}(t) ∫_{a≥1} λ^{kl}_E(t, a_E) da_E,

    with, as everywhere in the model (Eq 7 and compute_R_matrix),
      λ^{kl}_E(t, a_E) = χ^{kl} m_k prob_peak p(a_E) / N^l_eff(t),
      χ^{kl} = c_l · (lw if k == l else lb)   (lw when the infector is at home),
      N^l_eff(t) = Σ_j f^{jl}(t) N_j.
    Hence
      R^l_meeting = c_l · prob_peak · Σ_{a≥1} p(a) · S^l_eff / N^l_eff
                    · Σ_k f^{kl} χ̃^{kl} m_k,     χ̃^{kl} = lw if k == l else lb,
    with S^l_eff = Σ_j f^{jl} S_j.  Summing over l recovers Σ_{k,j} R^{kj}: every
    infection is attributed to exactly one meeting location.
    """
    N        = f_t.shape[0]
    c        = _as_multiplier(contact_mult, N, "contact_mult")
    m        = _as_multiplier(infect_mult,  N, "infect_mult")
    _, _, _, _, inv_Neff = _kernel_base(f_t, populations, lw, lb, contact_mult)
    S_eff    = f_t.T @ S
    sum_w    = infect_profile[1:].sum()
    R_meet   = np.zeros(N)
    for l in range(N):
        chi_tilde = np.full(N, float(lb))
        chi_tilde[l] = lw                            # infector resident in l is at home
        R_meet[l] = (c[l] * prob_peak * sum_w * S_eff[l] * inv_Neff[l]
                     * np.sum(f_t[:, l] * chi_tilde * m))
    return R_meet
