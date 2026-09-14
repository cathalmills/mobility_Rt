"""mobility_rt.config."""


COVID_PARAMS = {
    "gen_time_mean":          5.5,    # days  [Hart et al. 2022 Lancet Infect Dis]
    "gen_time_sd":            1.8,
    "max_gen_time":           25,
    "base_contact_rate":      13.03,  # contacts/day [POLYMOD Mossong 2008]
    "prob_transmission_peak": 0.035,
    "R0_target":              1.5,
}


R_PRIOR_SHAPE = 1.0     # Gamma prior shape a  (EpiEstim default)


R_PRIOR_RATE  = 0.2     # Gamma prior rate 1/scale  →  prior mean a/rate = 5


# An estimate is reported only where at least ONE case falls in the window.
#
# This is a unit, not a tuned parameter.  A case count below one is not observable:
# the deterministic PDE decays to fractions of a case (down to 1e-4 by day 364),
# which no surveillance system could ever record as anything but zero.  Left in,
# those fractions carry no information, the posterior collapses onto the prior mean
# of 5, and the estimator climbs away from a true R(t) of about 0.6 in the tail.
#
# Any LARGER value would be arbitrary, and a larger value is also actively harmful
# when per-location estimates are aggregated: the aggregate is then taken over
# whichever locations currently clear the rule, so its composition changes daily,
# and the reported error becomes a function of the rule's value rather than of the
# aggregation scheme (it varies ~30-fold as the rule moves from 1 to 50 cases).
# One case is the smallest value that is both observable and not a free choice.
R_MIN_WINDOW_CASES = 1.0


OKABE_ITO = ['#E69F00', '#56B4E9', '#009E73', '#F0E442',
             '#0072B2', '#D55E00', '#CC79A7', '#000000']
