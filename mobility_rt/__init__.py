"""mobility_rt — mobility-informed renewal equations (core math, matplotlib-free)."""
import warnings
import numpy as np
warnings.filterwarnings("ignore")
np.set_printoptions(precision=4, suppress=True)

__version__ = "1.0.0"

# ── core public API (importing this package pulls in NO matplotlib) ──
from mobility_rt.config import COVID_PARAMS
from mobility_rt.distributions import discretise_gamma
from mobility_rt.geometry import generate_city
from mobility_rt.geometry import representative_locs
from mobility_rt.mobility import generate_mobility
from mobility_rt.kernel import compute_R_matrix
from mobility_rt.kernel import R_outward
from mobility_rt.kernel import R_inward
from mobility_rt.kernel import R_system
from mobility_rt.kernel import compute_R_meeting
from mobility_rt.spectral import spectral_analysis
from mobility_rt.generation_time import compute_generation_times
from mobility_rt.generation_time import eigenvector_weighted_gt
from mobility_rt.simulation import compute_effective_populations_series
from mobility_rt.type_reproduction import type_reproduction_number
from mobility_rt.type_reproduction import type_reproduction_numbers
from mobility_rt.type_reproduction import type_reproduction_number_group
from mobility_rt.simulation import simulate_epidemic_pde
from mobility_rt.config import R_PRIOR_SHAPE
from mobility_rt.config import R_PRIOR_RATE
from mobility_rt.config import R_MIN_WINDOW_INCIDENCE
from mobility_rt.estimators import estimate_R_independent
from mobility_rt.transmissibility import source_sink_analysis
from mobility_rt.transmissibility import within_between_decomposition
from mobility_rt.spectral import sensitivity_elasticity
from mobility_rt.spectral import reactivity
from mobility_rt.spectral import amplification_envelope
from mobility_rt.spectral import convergence_cosine
from mobility_rt.transmissibility import euler_lotka_r
from mobility_rt.transmissibility import empirical_growth_rate
from mobility_rt.transmissibility import epidemic_speed
from mobility_rt.transmissibility import minimum_control_effort
from mobility_rt.config import OKABE_ITO
from mobility_rt.estimators import compute_naive_R_aggregate
# selected private helpers used by tests / downstream
from mobility_rt.kernel import _kernel_base
from mobility_rt.spectral import _perron_real
from mobility_rt.type_reproduction import _group_indices
from mobility_rt.simulation import _simulate_heun
from mobility_rt.simulation import _simulate_rk4
from mobility_rt.estimators import _compute_naive_suite
from mobility_rt.transmissibility import _compute_power_mean_spectrum
