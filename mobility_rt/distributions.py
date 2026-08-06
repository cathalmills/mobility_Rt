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

    max_days must be large enough that the truncated tail mass is negligible: the pmf
    is renormalised after truncation, so a max_days that cuts non-trivial tail mass
    biases the mean low (e.g. Gamma(5.5,1.8): max_days=25 removes ~2e-10 = safe;
    max_days=10 removes ~0.6% and shifts the mean).  The SI convergence figure varies
    max_days deliberately to demonstrate this.
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
    return pmf / pmf.sum()
