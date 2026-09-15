"""mobility_rt.config — model parameters and constants (manuscript §2.4, Appendix A)."""


COVID_PARAMS = {
    # Infectiousness profile Gamma(mean 5.5 d, sd 1.8 d), truncated at 25 d
    # (manuscript §2.4, ref [59]: Hart et al. 2022, eLife 11:e70767).
    "gen_time_mean":          5.5,
    "gen_time_sd":            1.8,
    "max_gen_time":           25,
    # λ_W, contacts per person per day (manuscript §2.4, ref [60]: POLYMOD, Mossong
    # et al. 2008).  The calibration rescales λ_W and λ_B by one common factor, so
    # this value and prob_transmission_peak have no effect on any output; only
    # λ_B/λ_W is identified (finding F01).
    "base_contact_rate":      13.03,
    "prob_transmission_peak": 0.035,
    # Calibration target: ρ(R) for a fully susceptible population (S = N) under the
    # calibration mobility matrix passed to simulate_epidemic_pde.
    "R0_target":              1.5,
}


R_PRIOR_SHAPE = 1.0     # Gamma prior shape a  (EpiEstim default)


R_PRIOR_RATE  = 0.2     # Gamma prior rate 1/scale  →  prior mean a/rate = 5


# An estimate is reported only where at least ONE case falls in the window.
#
# This is a unit, not a tuned parameter.  A case count below one is not observable:
# the deterministic PDE decays to fractions of a case in the tail, which no
# surveillance system could ever record as anything but zero.  Left in, those
# fractions carry no information, the posterior collapses onto the prior mean of 5,
# and the estimator climbs away from the true R(t) in the tail.
#
# Any LARGER value would be arbitrary, and a larger value is also actively harmful
# when per-location estimates are aggregated: the aggregate is then taken over
# whichever locations currently clear the rule, so its composition changes daily,
# and the reported error becomes a function of the rule's value rather than of the
# aggregation scheme.  One case is the smallest value that is both observable and
# not a free choice.
R_MIN_WINDOW_CASES = 1.0


# ── mobility (manuscript Appendix A) ────────────────────────────────────────
# λ_B/λ_W: between-location contacts assumed 30% as intense as home contacts (§2.4).
BETWEEN_FRACTION = 0.30

# Day-of-week multipliers on each location's away-from-home fraction, Monday first
# (day 0 is a Monday).  Appendix A of the 11 Sep 2026 manuscript states Sunday = 0.80;
# the code and every figure use 0.75, and the manuscript is to be corrected (D7).
DOW_SCALE = (1.00, 1.00, 1.00, 1.00, 0.95, 0.90, 0.75)

# Clip applied to the shared daily lognormal factor ξ_t (not stated in the manuscript;
# decision D19).
XI_CLIP = (0.50, 1.80)

# Cap on a location's scaled away-from-home fraction (Appendix A).
AWAY_FRACTION_CAP = 0.95

# Distances below this are raised to it before the exp(-d/δ) distance weight, so
# co-located nodes do not dominate (not stated in the manuscript; decision D19).
DISTANCE_FLOOR_KM = 0.5

# σ_ξ of the daily lognormal mobility factor per scenario (decision D19).
DAY_VARIATION_SD = {"A": 0.15, "B": 0.12, "C": 0.15}


OKABE_ITO = ['#E69F00', '#56B4E9', '#009E73', '#F0E442',
             '#0072B2', '#D55E00', '#CC79A7', '#000000']
