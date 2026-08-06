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
    ev_r, evec_r = np.linalg.eig(R_mat)
    ev_l, evec_l = np.linalg.eig(R_mat.T)

    idx_r = np.argsort(np.abs(ev_r))[::-1]
    idx_l = np.argsort(np.abs(ev_l))[::-1]

    rho     = float(np.abs(ev_r[idx_r[0]]))
    lambda2 = float(np.abs(ev_r[idx_r[1]])) if len(idx_r) > 1 else 0.0

    w = _perron_real(evec_r[:, idx_r[0]]);  w /= (w.sum() + 1e-300)
    v = _perron_real(evec_l[:, idx_l[0]]);  v /= (v.sum() + 1e-300)

    mix = lambda2 / rho if rho > 0 else 0.0
    cv  = (R_mat.sum(axis=1).std() / R_mat.sum(axis=1).mean()
           if R_mat.sum(axis=1).mean() > 0 else 0.0)

    return {"rho": rho, "lambda2": lambda2, "mixing_ratio": mix,
            "damping_ratio": rho / lambda2 if lambda2 > 0 else np.inf,
            "right_eigvec": w,        # right eigvec of R_mat (infector=rows); = reprod. values in Diekmann
            "left_eigvec": v,         # right eigvec of R_mat.T = right eigvec of Diekmann K^D
            "stable_distribution": v, # alias: right eigvec of Diekmann K^D = stable spatial distribution
            "reprod_value": w,        # alias: left eigvec of Diekmann K^D = reproductive value vector
            "cv_row_sums": cv, "eigenvalues": ev_r[idx_r]}


def sensitivity_elasticity(R_mat):
    """S_{kj}=∂ρ/∂R_{kj}=v_k w_j/(v^T w),  E_{kj}=(R_{kj}/ρ)*S_{kj}.

    Here v=left_eigvec (code) = stable distribution v* (manuscript),
    w=right_eigvec (code) = reproductive value v (manuscript).

    Condition number: κ(R(t)) = ||v||₂ ||w||₂ / |v·w|  (Eq 38 in ms).
    Since np.linalg.eig returns unit-L2-norm vectors we have ||v||=||w||=1,
    so κ = 1/|v·w|.  Note: this is the eigenvalue condition number, NOT the
    matrix condition number np.linalg.cond().
    """
    spec = spectral_analysis(R_mat)
    rho  = spec["rho"];  v = spec["left_eigvec"];  w = spec["right_eigvec"]
    vw   = max(float(v @ w), 1e-15)
    S_m  = np.outer(v, w) / vw
    E_m  = (R_mat / rho) * S_m if rho > 1e-15 else np.zeros_like(R_mat)
    # Eigenvalue condition number (Eq 38): κ = ||v||₂ ||w||₂ / |v^T w|
    # Use unnormalised eigenvectors for correct L2 norms
    ev_r, evec_r = np.linalg.eig(R_mat)
    ev_l, evec_l = np.linalg.eig(R_mat.T)
    ir = int(np.argmax(np.abs(ev_r))); il = int(np.argmax(np.abs(ev_l)))
    wr = evec_r[:, ir]; vl = evec_l[:, il]
    denom = abs(float(vl @ wr))
    kappa = float(np.linalg.norm(vl) * np.linalg.norm(wr)) / (denom + 1e-300)
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
    """A(n)=‖R^n‖₂=σ_max(R^n).  Captures non-normal transient growth."""
    rho = R_system(R_mat)
    A   = np.zeros(n_max + 1)
    Rn  = np.eye(R_mat.shape[0])
    for n in range(n_max + 1):
        A[n] = float(np.linalg.svd(Rn, compute_uv=False)[0])
        Rn   = Rn @ R_mat
    return {"A": A, "rho_n": rho ** np.arange(n_max + 1),
            "n": np.arange(n_max + 1), "rho": rho}


def convergence_cosine(incidence, R_matrices):
    """cos(p(t), w(t)) where p=observed spatial distrib, w=dominant eigvec."""
    T = incidence.shape[0]
    cs = np.full(T, np.nan)
    for t in range(T):
        tot = incidence[t].sum()
        if tot < 1:
            continue
        p = incidence[t] / tot
        # stable distribution = right eigvec of Diekmann K^D = left eigvec of R_mat
        w = spectral_analysis(R_matrices[t])["stable_distribution"]
        d = np.linalg.norm(p) * np.linalg.norm(w)
        if d > 1e-15:
            cs[t] = float(p @ w) / d
    return cs
