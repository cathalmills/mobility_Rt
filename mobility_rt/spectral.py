"""mobility_rt.spectral."""
import numpy as np
from mobility_rt.kernel import R_system


def _perron_real(evec_col):
    """Recover the real dominant (Perron) eigenvector from a complex eigenvector column.

    np.linalg.eig returns eigenvectors with an arbitrary complex phase; for a primitive
    non-negative matrix the dominant eigenvector is real and single-signed.  We rotate out
    the global phase using the largest-magnitude entry, take the real part, and fix the sign
    to non-negative.  This is exact for the primitive non-negative NGMs used here (identical
    to the old np.abs for a strictly-positive Perron vector) but, unlike np.abs, does not
    silently discard sign information on degenerate/reducible matrices.
    """
    k = int(np.argmax(np.abs(evec_col)))
    phase = evec_col[k] / (np.abs(evec_col[k]) + 1e-300)
    v = (evec_col / phase).real.astype(float)
    if v[k] < 0:
        v = -v
    return v


def spectral_analysis(R_mat):
    """ρ, λ₂, mixing ratio s=|λ₂|/ρ, right eigvec w, left eigvec v."""
    ev_r, evec_r = np.linalg.eig(R_mat) #right eigenvec-val pair w
    ev_l, evec_l = np.linalg.eig(R_mat.T)

    idx_r = np.argsort(np.abs(ev_r))[::-1]
    idx_l = np.argsort(np.abs(ev_l))[::-1]

    rho     = float(np.abs(ev_r[idx_r[0]]))
    #sub-dominant eigenvalue (which governs transience)
    lambda2 = float(np.abs(ev_r[idx_r[1]])) if len(idx_r) > 1 else 0.0

    w = _perron_real(evec_r[:, idx_r[0]]);  w /= (w.sum() + 1e-300)
    v = _perron_real(evec_l[:, idx_l[0]]);  v /= (v.sum() + 1e-300)

    mix = lambda2 / rho if rho > 0 else 0.0 #mixing ratio, how close eigenvals are as a ratio
    cv  = (R_mat.sum(axis=1).std() / R_mat.sum(axis=1).mean()
           if R_mat.sum(axis=1).mean() > 0 else 0.0) #how variable outward reproduction numbers are (closest to per-location indicator)

    return {"rho": rho, "lambda2": lambda2, "mixing_ratio": mix,
            "damping_ratio": rho / lambda2 if lambda2 > 0 else np.inf,
            # Paper's notation (§3.1.4): v = asymptotic REPRODUCTIVE VALUE (Fisher's,
            # in Diekmann), v* = stable INFECTION DISTRIBUTION.  R_mat here is the
            # transpose of the paper's NGM K^D, so the sides swap: the paper's v is a
            # LEFT eigenvector of K^D and therefore a RIGHT eigenvector of R_mat.
            "right_eigvec": w,        # = v  (reproductive value): right eigvec of R_mat, left eigvec of K^D
            "left_eigvec": v,         # = v* (stable distribution): left eigvec of R_mat, right eigvec of K^D
            "stable_distribution": v, # alias for left_eigvec  = v*
            "reprod_value": w,        # alias for right_eigvec = v
            "cv_row_sums": cv, "eigenvalues": ev_r[idx_r]}


def sensitivity_elasticity(R_mat):
    """Sensitivity, elasticity and eigenvalue condition number of ρ(R) (Eqs 29-33, 38).

    In code orientation (R_mat[k, j] = R^{kj}, the transpose of the manuscript NGM):
    This is because I want interpretation k->j transmission from source to destination
      v = spectral_analysis "left_eigvec"  = manuscript v* (stable distribution, Eq 25)
      w = spectral_analysis "right_eigvec" = manuscript v  (reproductive value, Eq 26)
      sensitivity[k, j] = v_k w_j / (vᵀw) = v_j v*_k / (vᵀv*) = s^{kj}     (Eq 31)
      elasticity[k, j]  = (R_mat[k, j] / ρ) · sensitivity[k, j] = ε^{kj}  (Eq 33)
      condition_number  = ||v||₂ ||w||₂ / |vᵀw|                           (Eq 38)
    κ does not depend on how either eigenvector is scaled, so the real, sum-normalised
    Perron vectors give it exactly.  It is the eigenvalue condition number, not
    np.linalg.cond.
    """
    spec = spectral_analysis(R_mat)
    rho  = spec["rho"];  v = spec["left_eigvec"];  w = spec["right_eigvec"]
    vw   = max(float(v @ w), 1e-15)
    S_m  = np.outer(v, w) / vw
    E_m  = (R_mat / rho) * S_m if rho > 1e-15 else np.zeros_like(R_mat)
    kappa = float(np.linalg.norm(v) * np.linalg.norm(w) / abs(float(v @ w)))
    return {"sensitivity": S_m, "elasticity": E_m, "rho": rho,
            "condition_number": kappa}


def reactivity(R_mat):
    """σ(t) = ‖R(t)‖₂  (largest singular value; Eq. 37 in ms.)."""
    sigma = float(np.linalg.svd(R_mat, compute_uv=False)[0])
    rho   = R_system(R_mat)
    return {"sigma": sigma, "rho": rho,
            "transient_possible": sigma > 1 and rho < 1,
            "amplification_ratio": sigma / rho if rho > 0 else np.inf}


def amplification_envelope(R_mat, n_max=30):
    """Transient amplification envelopes of R^n for n = 0..n_max.

      A[n]  = ||R^n||₂ = σ_max(R^n)                                        (Eq 39)
      A1[n] = ||R_ms^n||₁ = largest column sum of the manuscript NGM power  (Eq 40)
            = largest row sum of R_mat^n in code orientation (R_mat = R_msᵀ)
    """
    rho = R_system(R_mat)
    A   = np.zeros(n_max + 1)
    A1  = np.zeros(n_max + 1)
    Rn  = np.eye(R_mat.shape[0])
    for n in range(n_max + 1):
        A[n]  = float(np.linalg.svd(Rn, compute_uv=False)[0])
        A1[n] = float(np.abs(Rn).sum(axis=1).max())
        Rn    = Rn @ R_mat
    return {"A": A, "A1": A1, "rho_n": rho ** np.arange(n_max + 1),
            "n": np.arange(n_max + 1), "rho": rho}

