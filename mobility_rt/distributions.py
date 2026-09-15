"""mobility_rt.distributions."""
import numpy as np
from scipy.stats import gamma as gamma_dist


def discretise_gamma(mean, sd, max_days):
    """Double-interval-censored discretisation of Gamma(mean, sd) to a daily pmf.

    Returns the pmf of the daily-resolution generation interval D = floor(U + X),
    where X ~ Gamma(shape=(mean/sd)^2, scale=sd^2/mean) and U ~ Uniform(0,1) is the
    unobserved within-day offset of the primary event.  This recovers E[X] exactly
    and removes the ~0.5-day downward mean bias of the naive scheme
    pmf[d] = F(d+1) - F(d) (which is the distribution of floor(X)); see Park et al.
    2024 and Charniga et al. 2024.  Analytic (primary-censored) form, as in the
    primarycensored package:

        P(D=d) = ∫_d^{d+1} F(v) dv - ∫_{d-1}^{d} F(v) dv = IF(d+1) - 2·IF(d) + IF(d-1)
        IF(x)  = ∫_0^x F(v) dv = x·F(x; a, θ) - a·θ·F(x; a+1, θ)   (0 for x <= 0).

    Double-interval-censoring recovers the mean exactly but, by construction, inflates
    the variance by E[frac(1-frac)] ~ 1/6: the realised daily-pmf sd is sqrt(sd^2 + 1/6),
    e.g. Gamma(5.5, 1.8) -> sd ~ 1.846.  The stated sd (1.8) is the latent continuous
    distribution's; the discrete censored interval is genuinely more spread.  This is a
    property of the censoring method, not a discretisation error.

    max_days must be large enough that the truncated tail mass is negligible: the pmf
    is renormalised after truncation, so a max_days that cuts non-trivial tail mass
    biases the mean low (e.g. Gamma(5.5,1.8): max_days=25 removes ~4e-10 = safe;
    max_days=10 removes ~2.7% and shifts the mean ~0.14 d).  The SI convergence figure
    varies max_days deliberately to demonstrate this.
    """
    shape = (mean / sd) ** 2
    scale = sd ** 2 / mean

    def _int_F(x):  # ∫_0^x F(v) dv for Gamma(shape, scale)
        if x <= 0:
            return 0.0
        return (x * gamma_dist.cdf(x, a=shape, scale=scale)
                - shape * scale * gamma_dist.cdf(x, a=shape + 1, scale=scale))

    pmf = np.array([_int_F(d + 1) - 2.0 * _int_F(d) + _int_F(d - 1)
                    for d in range(max_days)])
    pmf = np.clip(pmf, 0.0, None)
    total = pmf.sum()
    if total < 1e-300:
        raise ValueError(
            f"discretise_gamma: all probability mass falls beyond max_days={max_days} "
            f"for Gamma(mean={mean}, sd={sd}); increase max_days.")
    return pmf / total


def renewal_gt(p):
    """The generation-time distribution the renewal model actually runs on.

    Every reproduction number, kernel normalisation and force of infection in the
    model integrates the infectiousness profile over a_E >= 1 (there is no same-day
    transmission), so the generation time is p(a_E) with the a_E = 0 entry set to
    zero and renormalised: g(a_E) = p(a_E) / Σ_{a>=1} p(a) for a_E >= 1, g(0) = 0.
    """
    g = np.asarray(p, dtype=float).copy()
    g[0] = 0.0
    total = g.sum()
    if not total > 0.0:
        raise ValueError("the profile carries no mass at ages a_E >= 1")
    return g / total
