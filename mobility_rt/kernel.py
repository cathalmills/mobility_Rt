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
    """
    Precompute base_K[k,j] and its within/between split for one time-slice.

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
    c        = _as_multiplier(contact_mult, f_t.shape[0], "contact_mult")
    N_eff    = f_t.T @ populations
    inv_Neff = np.where(N_eff > 0, 1.0 / N_eff, 0.0)
    c_inv    = c * inv_Neff                          # c[l]/N_eff[l]
    A        = f_t * c_inv[np.newaxis, :]            # A[j,l] = c[l]*f[j,l]/N_eff[l]
    D        = A @ f_t.T                             # D[j,k] = Σ_l A[j,l]*f[k,l]
    within   = f_t.T * (f_t.diagonal() * c_inv)[:, np.newaxis]  # [k,j], meeting at k
    bK_within  = lw * within
    bK_between = lb * np.maximum(D.T - within, 0.0)  # away-meeting component ≥ 0
    base_K     = bK_within + bK_between
    return base_K, bK_within, bK_between, N_eff, inv_Neff


def compute_R_matrix(f_t, S, populations, prob_peak, infect_profile, lw, lb,
                     contact_mult=None, infect_mult=None):
    """R_{kj}(t) = m[k] * prob_peak * Σ_{a≥1} p(a) * S[j] * base_K[k,j].

    The integral over infection age uses ages a ≥ 1 (infect_profile[1:]), matching
    the forward simulator's force of infection (E_pde[:,:,1:] @ infect_profile[1:]),
    which excludes same-day (a=0) transmission.  Using the full Σ_{a≥0} p(a) would
    overstate R by 1/(1-p[0]) and make the realised R0 = R0_target·(1-p[0]); for the
    Hart GT p[0]≈4e-6 (negligible) but this keeps the NGM exactly consistent with the
    dynamics for any (e.g. short) generation time.

    contact_mult c[l] scales the contact rate at MEETING location l (see _kernel_base).
    infect_mult  m[k] scales the transmission probability of an infector resident in k,
    so it multiplies row k, i.e. everyone that infector k infects.  Both default to
    None (≡ 1) and then reproduce the homogeneous R matrix exactly.
    """
    base_K, _, _, _, _ = _kernel_base(f_t, populations, lw, lb, contact_mult)
    m = _as_multiplier(infect_mult, f_t.shape[0], "infect_mult")
    row = m * prob_peak * infect_profile[1:].sum()       # per-infector scale, [k]
    return row[:, np.newaxis] * S[np.newaxis, :] * base_K


def R_outward(R_mat):
    return R_mat.sum(axis=1)


def R_inward(R_mat):
    return R_mat.sum(axis=0)


def R_system(R_mat):
    ev = np.linalg.eigvals(R_mat)
    return float(np.max(np.abs(ev)))


def compute_R_meeting(f_t, S, populations, prob_peak, infect_profile, lw, lb,
                      contact_mult=None, infect_mult=None):
    """
    R^l_meeting(t) = m̄[l] * S^l_eff * κ_eff(l) * prob_peak * Σ_{a≥1} p(a) / N^l_eff
    κ_eff(l) = c[l] * (lw*f[l,l] + lb*Σ_{k≠l} f[k,l])
    (Σ over a ≥ 1, consistent with compute_R_matrix and the forward simulator.)

    R^l_meeting is a per-infectious-person quantity at meeting location l, so with
    heterogeneous infectiousness it needs the infectiousness of a TYPICAL person
    present at l.  m̄[l] is therefore the presence-weighted mean
      m̄[l] = Σ_k f[k,l] N_k m[k] / Σ_k f[k,l] N_k = Σ_k f[k,l] N_k m[k] / N_eff[l],
    which reduces to m[l] when nobody travels and to 1 when m ≡ 1.
    """
    N        = f_t.shape[0]
    c        = _as_multiplier(contact_mult, N, "contact_mult")
    m        = _as_multiplier(infect_mult,  N, "infect_mult")
    _, _, _, N_eff, inv_Neff = _kernel_base(f_t, populations, lw, lb, contact_mult)
    S_eff    = f_t.T @ S
    # m ≡ 1 must give m̄ ≡ 1 EXACTLY; N_eff*(1/N_eff) is only 1 to rounding, so
    # short-circuit rather than computing a ratio that is analytically one.
    m_bar    = (np.ones(N) if infect_mult is None
                else (f_t.T @ (populations * m)) * inv_Neff)  # presence-weighted mean m
    sum_w    = infect_profile[1:].sum()
    R_meet   = np.zeros(N)
    for l in range(N):
        kappa = c[l] * (lw * f_t[l, l] + lb * (f_t[:, l].sum() - f_t[l, l]))
        R_meet[l] = m_bar[l] * prob_peak * sum_w * S_eff[l] * kappa * inv_Neff[l]
    return R_meet
