"""mobility_rt.heterogeneity — spatially heterogeneous contact and infectiousness.

The baseline model gives every location the same per-contact transmission rate and
the same contact rate, so all variation in the reproduction numbers comes from
mobility geometry and local susceptible depletion.  This module builds the two
multiplier vectors that relax that assumption:

  contact_mult c[l]  contact rate at MEETING location l, relative to the network
                     geometric mean.  Denser settlement types are assumed to
                     generate more contacts per person-day.
  infect_mult  m[k]  transmission probability per contact for an infector resident
                     in k, relative to the network geometric mean.  Modelled as
                     unstructured lognormal variation (housing, ventilation,
                     occupation, care-seeking) rather than a settlement gradient.

Both are normalised to GEOMETRIC mean 1.  Only relative variation matters: the
simulator rescales lambda_within/lambda_between so that rho(R(0)) = R0_target,
which divides out any common factor.

IMPORTANT — these are illustrative modelling assumptions, not fitted or
literature-calibrated values.  They are stated as round numbers so that the
heterogeneous scenarios are a clearly-labelled sensitivity analysis rather than a
tuned result.  Any quantitative claim needs them replaced by setting-specific
contact-survey estimates.
"""
import numpy as np


# Relative contact rate by settlement type.  Dense/central types are assumed to
# generate about twice the contacts per person-day of the sparsest type; the
# within-scenario ratio is 1.30/0.65 = 2.0 for both geographies.
CONTACT_MULT_BY_TYPE = {
    # lagos (dense urban megacity)
    "core":             1.30,
    "dense":            1.15,
    "suburban":         0.90,
    "peripheral":       0.65,
    # zambia (sparse national network)
    "capital":          1.30,
    "peri-capital":     1.15,
    "urban-industrial": 1.10,
    "semi-urban":       0.95,
    "rural":            0.75,
    "remote-rural":     0.65,
}

# Lognormal sigma for the unstructured infectiousness multiplier.
# sigma = 0.35 gives a coefficient of variation of about 36%.
INFECT_LOGSD = 0.35


def contact_multipliers(node_types, mult_by_type=None):
    """c[l] from settlement type, normalised to geometric mean 1.

    Raises on an unknown node type rather than silently defaulting it to 1, which
    would quietly turn a heterogeneous scenario into a partly homogeneous one.
    """
    if len(node_types) == 0:
        raise ValueError("node_types is empty; nothing to build multipliers for")
    table = CONTACT_MULT_BY_TYPE if mult_by_type is None else mult_by_type
    unknown = sorted({t for t in node_types if t not in table})
    if unknown:
        raise KeyError(f"no contact multiplier defined for node type(s): {unknown}")
    c = np.array([table[t] for t in node_types], dtype=float)
    if np.any(c <= 0):
        raise ValueError("contact multipliers must be strictly positive")
    return c / np.exp(np.mean(np.log(c)))


def infectiousness_multipliers(n_locations, log_sd=INFECT_LOGSD, seed=42):
    """m[k] as lognormal noise, normalised to geometric mean exactly 1.

    Dividing by the REALISED geometric mean rather than by exp(0) keeps the draw
    centred at 1 for any finite n, so two seeds differ in pattern and not in overall
    level.  Note this fixes the geometric, not the arithmetic, mean: by Jensen the
    arithmetic mean of the returned vector exceeds 1.  That residual common factor
    is irrelevant here because the simulator rescales lambda_within/lambda_between
    to hit rho(R(0)) = R0_target, which divides any common factor out.

    n_locations = 1 correctly returns exactly [1.0]: one location cannot be
    heterogeneous with respect to itself.
    """
    if n_locations < 1:
        raise ValueError("n_locations must be at least 1")
    if log_sd < 0:
        raise ValueError("log_sd must be non-negative")
    if log_sd == 0:
        return np.ones(n_locations, dtype=float)
    z = np.random.default_rng(seed).normal(0.0, log_sd, n_locations)
    return np.exp(z - z.mean())


def heterogeneity_summary(contact_mult, infect_mult):
    """Human-readable dispersion diagnostics for the console log."""
    def _d(v):
        v = np.asarray(v, dtype=float)
        return {"min": float(v.min()), "max": float(v.max()),
                "cv": float(v.std() / v.mean())}
    return {"contact": _d(contact_mult), "infect": _d(infect_mult)}
