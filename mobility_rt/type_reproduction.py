"""mobility_rt.type_reproduction."""
import numpy as np
from mobility_rt.kernel import R_system


def type_reproduction_number(R_mat, j):
    r"""
    Instantaneous type reproduction number T_j(t) for location j.

        T_j = R_{jj} + R_{jJ} (I - R_{JJ})^{-1} R_{Jj}          [Eq 54]

    where J = {0,...,L-1} \ {j} (all locations except j).  This is the
    decomposed form of Eq 53; the two are algebraically equivalent when the
    Neumann series for R_{JJ} converges (ρ(R_{JJ}) < 1).

    Note on convention: R_mat here is stored as rows=infector, cols=infectee
    (the transpose of the manuscript NGM), but T_j is a bilinear form and is
    invariant to this transpose because (I - R_{JJ}^T)^{-1} = ((I - R_{JJ})^{-1})^T,
    so the result matches Eq 54 exactly.

    Returns
    -------
    float or np.nan
        np.nan when ρ(R_{JJ}) >= 1, i.e. the background network J can sustain
        an epidemic without location j, so T_j is undefined (diverges to ∞ and
        control of j alone cannot eliminate the epidemic).
    """
    N = R_mat.shape[0]
    J = [k for k in range(N) if k != j]
    R_jj = R_mat[j, j]                        # scalar
    if not J:                                 # single-location network: T_j = R_jj
        return float(R_jj)
    R_jJ = R_mat[j, J]                        # (L-1,)
    R_Jj = R_mat[J, j]                        # (L-1,)
    R_JJ = R_mat[np.ix_(J, J)]                # (L-1, L-1)

    # Check convergence of Neumann series for R_JJ
    rho_JJ = np.max(np.abs(np.linalg.eigvals(R_JJ)))
    if rho_JJ >= 1.0:
        return np.nan          # T_j undefined: background can sustain epidemic alone

    I_JJ = np.eye(len(J))
    try:
        resolvent = np.linalg.solve(I_JJ - R_JJ, I_JJ)  # (I - R_JJ)^{-1}
    except np.linalg.LinAlgError:
        return np.nan

    T_j = float(R_jj + R_jJ @ resolvent @ R_Jj)
    return T_j


def type_reproduction_numbers(R_mat):

     """

     Compute T_j(t) for all locations j.  Returns array of length L,

     with np.nan where T_j is undefined (ρ(R_{JJ}) >= 1).

     """

     N = R_mat.shape[0]

     return np.array([type_reproduction_number(R_mat, j) for j in range(N)])


def type_reproduction_number_group(R_mat, P):
    """
    Group type reproduction number T^P_type(t) for a set P of locations.

    T^P_type = ρ( R_PP + R_PQ (I - R_QQ)^{-1} R_QP )               [Eq 56]

    where Q = {0,...,L-1} \\ P (all locations outside P).

    When P = {j} (singleton), this reduces to the scalar T_j.

    Parameters
    ----------
    R_mat : (L, L) ndarray
        Next-generation matrix R(t).
    P : list-like of int
        Indices of locations in the target group.

    Returns
    -------
    float or np.nan
        np.nan when ρ(R_QQ) >= 1, i.e. the complementary network Q can
        sustain an epidemic without P, so T^P is undefined (diverges to ∞).
    """
    P = sorted(set(P))
    N = R_mat.shape[0]
    Q = [k for k in range(N) if k not in P]

    if not Q:
        # P = full network → T^P = R(t)
        return float(R_system(R_mat))

    R_PP = R_mat[np.ix_(P, P)]
    R_PQ = R_mat[np.ix_(P, Q)]
    R_QP = R_mat[np.ix_(Q, P)]
    R_QQ = R_mat[np.ix_(Q, Q)]

    rho_QQ = float(np.max(np.abs(np.linalg.eigvals(R_QQ))))
    if rho_QQ >= 1.0:
        return np.nan          # background Q sustains epidemic: T^P undefined

    try:
        resolvent = np.linalg.solve(np.eye(len(Q)) - R_QQ, np.eye(len(Q)))
    except np.linalg.LinAlgError:
        return np.nan

    M = R_PP + R_PQ @ resolvent @ R_QP   # effective within-P NGM
    return float(R_system(M))


def _group_indices(node_types, type_set):
    """Return sorted list of location indices whose type is in *type_set*."""
    return [j for j, t in enumerate(node_types) if t in type_set]
