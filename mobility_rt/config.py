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


R_MIN_WINDOW_INCIDENCE = 12.0


OKABE_ITO = ['#E69F00', '#56B4E9', '#009E73', '#F0E442',
             '#0072B2', '#D55E00', '#CC79A7', '#000000']
