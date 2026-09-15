"""mobility_rt.scripts.run_all — build the manuscript scenarios and write every
manuscript figure and table."""
import os
import sys

# Make this runnable both as `python -m mobility_rt.scripts.run_all` and as a plain
# script (`python mobility_rt/scripts/run_all.py`) from any directory: put the repo
# root on sys.path when the package is not already importable.
if __package__ in (None, ""):
    _root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if _root not in sys.path:
        sys.path.insert(0, _root)

import numpy as np
from mobility_rt.config import BETWEEN_FRACTION, COVID_PARAMS, DAY_VARIATION_SD
from mobility_rt.distributions import discretise_gamma
from mobility_rt.error_tables import build_estimator_error_tables
from mobility_rt.estimators import WINDOW, estimate_R_independent
from mobility_rt.generation_time import compute_generation_times
from mobility_rt.geometry import generate_city
from mobility_rt.heterogeneity import contact_multipliers, infectiousness_multipliers
from mobility_rt.mobility import generate_mobility
from mobility_rt.plotting.figures_corrected import run_corrected_figures
from mobility_rt.plotting.figures_main import (_compute_main_bias_data, plot_estimator_si_suite,
                                               plot_fig5, plot_fig_C_transience,
                                               plot_lehmer_mean_spectrum,
                                               plot_main_bias_figure, plot_type_repro)
from mobility_rt.plotting.figures_si import (plot_SI0_population, plot_SI1,
                                             plot_SI5_combined, plot_SI_3d_earlypeak,
                                             plot_SI_R_comparison, plot_SI_epi_params,
                                             plot_SI_pde_convergence, plot_SI_sensitivity)
from mobility_rt.plotting.style import _set_pub_style
from mobility_rt.simulation import phase_days, simulate_epidemic_pde

N_LOC = 10
T = 365
SEED = 42


def _report(tag, sim, pops):
    inc = sim["incidence"]
    print(f"  {tag}: total infections {inc.sum():.0f}, "
          f"peak day {int(inc.sum(axis=1).argmax())}, "
          f"attack rate {inc.sum() / pops.sum() * 100:.1f}%")


def hub_representative_locations(distances, hub_idx):
    """Hub, mid and peripheral representative locations of the hub-amplified scenario.

    hub    = hub_idx.  The other locations (satellites) are ranked by their distance
             from the hub.
    periph = the satellite farthest from the hub.
    mid    = the satellite at the lower median of that ranking.
    Distance is used rather than the share of the day a satellite's residents spend at
    the hub: with hub_attraction_power = 8 that share equals the satellite's commuting
    fraction to within 1e-8, so satellites of one type would be separated only by
    differences of order 1e-10.  Raises ValueError if either choice is tied (within
    1e-9 km), so no label depends on index order.  Returns (hub, mid, periph).
    """
    d = np.asarray(distances, dtype=float)
    sats = [j for j in range(d.shape[0]) if j != hub_idx]
    if len(sats) < 3:
        raise ValueError("need at least three satellites for distinct mid and periph")
    dist = d[sats, hub_idx]
    order = np.argsort(dist, kind="stable")
    picks = {}
    for name, rank in (("periph", len(sats) - 1), ("mid", (len(sats) - 1) // 2)):
        value = dist[order[rank]]
        if np.count_nonzero(np.abs(dist - value) <= 1e-9) > 1:
            raise ValueError(f"{name} representative location is tied at distance {value} km")
        picks[name] = sats[int(order[rank])]
    return int(hub_idx), picks["mid"], picks["periph"]


def build_scenarios():
    """Simulate every scenario the manuscript uses; return them in one dict.

    A        dense urban (Lagos-like geometry)
    A-static A with time-invariant mobility base_fA
    A-het    A with heterogeneous contact rate c[l] and infectiousness m[k]
    B        sparse national (Zambia-like geometry), and B-het
    C        hub-amplified: A's geometry with one 3M hub and hub_attraction_power 8,
             and C-het
    """
    params = COVID_PARAMS
    max_days = params["max_gen_time"]
    LW = params["base_contact_rate"]
    LB = params["base_contact_rate"] * BETWEEN_FRACTION
    gen_time_pmf = discretise_gamma(params["gen_time_mean"], params["gen_time_sd"],
                                    max_days)
    infect_profile = gen_time_pmf.copy()

    def _simulate(pops, f_series, base_f, initial, contact_mult=None, infect_mult=None):
        # Calibration (D5): λ_W and λ_B are scaled so that ρ(R) = R0_target for a fully
        # susceptible population under the scenario's noise-free weekday mobility
        # base_f, the same matrix for a scenario and its -het and -static variants.
        return simulate_epidemic_pde(
            T, N_LOC, pops, f_series, params["prob_transmission_peak"], infect_profile,
            max_days, params["R0_target"], initial, LW, LB,
            f_calibration=base_f,
            contact_mult=contact_mult, infect_mult=infect_mult)

    # ── Scenario A: dense urban ────────────────────────────────────────────
    city_A = generate_city(N_LOC, scenario="lagos", seed=SEED)
    coords_A, pops_A, dists_A, types_A, meta_A = city_A
    f_A, base_fA = generate_mobility(N_LOC, T, pops_A, dists_A, meta_A,
                                     day_variation_sd=DAY_VARIATION_SD["A"], seed=SEED)
    initial_A = np.zeros(N_LOC)
    initial_A[0] = 10
    sim_A = _simulate(pops_A, f_A, base_fA, initial_A)
    # A-static: same everything, but f(t) = base_fA on every day
    f_A_static = np.stack([base_fA] * T, axis=0)
    sim_A_static = _simulate(pops_A, f_A_static, base_fA, initial_A)
    sim_A_het = _simulate(pops_A, f_A, base_fA, initial_A, contact_multipliers(types_A),
                          infectiousness_multipliers(N_LOC, seed=SEED + 0))

    # ── Scenario B: sparse national ────────────────────────────────────────
    city_B = generate_city(N_LOC, scenario="zambia", seed=SEED)
    _, pops_B, dists_B, types_B, meta_B = city_B
    f_B, base_fB = generate_mobility(N_LOC, T, pops_B, dists_B, meta_B,
                                     day_variation_sd=DAY_VARIATION_SD["B"], seed=SEED)
    initial_B = np.zeros(N_LOC)
    initial_B[0] = 10
    sim_B = _simulate(pops_B, f_B, base_fB, initial_B)
    sim_B_het = _simulate(pops_B, f_B, base_fB, initial_B, contact_multipliers(types_B),
                          infectiousness_multipliers(N_LOC, seed=SEED + 1))

    # ── Scenario C: hub-amplified ──────────────────────────────────────────
    # C differs from A only in its resident populations (a 3M hub at L1 and nine
    # 200k satellites) and in hub_attraction_power = 8, which concentrates satellite
    # commuting on the hub.  Geometry, commuting fractions, the mobility noise seed
    # and the seeding (10 infections at L1) are A's, so any difference between A and
    # C is due to the hub (decision D16).  Node types follow the populations: the hub
    # is "core" and each 200k satellite "peripheral" (the Lagos type whose base
    # population is 200k).  Types feed only C-het's contact multipliers and labels.
    pops_C = np.full(N_LOC, 200_000.0)
    pops_C[0] = 3_000_000.0
    types_C = ["core"] + ["peripheral"] * (N_LOC - 1)
    meta_C = dict(meta_A)
    meta_C["node_types"] = types_C
    f_C, base_fC = generate_mobility(N_LOC, T, pops_C, dists_A, meta_C,
                                     day_variation_sd=DAY_VARIATION_SD["C"],
                                     seed=SEED, hub_attraction_power=8.0)
    hub_C, mid_C, periph_C = hub_representative_locations(dists_A, hub_idx=0)
    meta_C.update(hub_idx=hub_C, mid_idx=mid_C, periph_idx=periph_C)
    city_C = (coords_A, pops_C, dists_A, types_C, meta_C)
    initial_C = initial_A.copy()
    sim_C = _simulate(pops_C, f_C, base_fC, initial_C)
    sim_C_het = _simulate(pops_C, f_C, base_fC, initial_C, contact_multipliers(types_C),
                          infectiousness_multipliers(N_LOC, seed=SEED + 2))

    return {
        "params": params, "max_days": max_days, "LW": LW, "LB": LB,
        "gen_time_pmf": gen_time_pmf, "infect_profile": infect_profile,
        "city_A": city_A, "f_A": f_A, "base_fA": base_fA, "f_A_static": f_A_static,
        "initial_A": initial_A,
        "sim_A": sim_A, "sim_A_static": sim_A_static, "sim_A_het": sim_A_het,
        "city_B": city_B, "f_B": f_B, "base_fB": base_fB, "initial_B": initial_B,
        "sim_B": sim_B, "sim_B_het": sim_B_het,
        "city_C": city_C, "f_C": f_C, "base_fC": base_fC, "initial_C": initial_C,
        "sim_C": sim_C, "sim_C_het": sim_C_het,
    }


def _gt_snapshots(sim, f_series, pops, params, p_aE, max_days):
    """Generation-time snapshots at the early / peak / late days (phase_days) of one
    scenario, built with that scenario's own calibrated rates and multipliers and
    the start-of-day susceptibles that R_matrices[t] uses."""
    snaps = {}
    for name, day in phase_days(sim["incidence"]).items():
        snaps[name] = compute_generation_times(
            f_series[day], sim["S_start"][day], pops,
            params["prob_transmission_peak"], p_aE, max_days,
            sim["lambda_within_scaled"], sim["lambda_between_scaled"],
            sim.get("contact_mult"), sim.get("infect_mult"))
    return snaps


def main():
    print("=" * 72)
    print("MOBILITY-INFORMED RENEWAL EQUATIONS — MANUSCRIPT FIGURES AND TABLES")
    print("=" * 72)

    sc = build_scenarios()
    params, max_days = sc["params"], sc["max_days"]
    LW, LB = sc["LW"], sc["LB"]
    gen_time_pmf = sc["gen_time_pmf"]
    city_A, city_B, city_C = sc["city_A"], sc["city_B"], sc["city_C"]
    pops_A = city_A[1]
    f_A, f_A_static, f_B, f_C = sc["f_A"], sc["f_A_static"], sc["f_B"], sc["f_C"]
    sim_A, sim_A_static, sim_A_het = sc["sim_A"], sc["sim_A_static"], sc["sim_A_het"]
    sim_B, sim_B_het = sc["sim_B"], sc["sim_B_het"]
    sim_C, sim_C_het = sc["sim_C"], sc["sim_C_het"]
    for tag, sim, city in (("A", sim_A, city_A), ("A-static", sim_A_static, city_A),
                           ("A-het", sim_A_het, city_A), ("B", sim_B, city_B),
                           ("B-het", sim_B_het, city_B), ("C", sim_C, city_C),
                           ("C-het", sim_C_het, city_C)):
        _report(tag, sim, city[1])

    R_ind_A = estimate_R_independent(sim_A["incidence"], gen_time_pmf, window=WINDOW)
    R_ind_As = estimate_R_independent(sim_A_static["incidence"], gen_time_pmf,
                                      window=WINDOW)
    gt_snaps_A = _gt_snapshots(sim_A, f_A, pops_A, params, gen_time_pmf, max_days)
    gt_snaps_As = _gt_snapshots(sim_A_static, f_A_static, pops_A, params, gen_time_pmf,
                                max_days)

    print("\n─── Generating figures ───")
    _set_pub_style()
    os.makedirs("out_figs/main", exist_ok=True)
    os.makedirs("out_figs/SI", exist_ok=True)
    os.makedirs("out_figs/tables", exist_ok=True)
    mpfx = "out_figs/main/fig"
    spfx = "out_figs/SI/fig"

    plot_fig_C_transience(sim_C, city_C, f_C, mpfx)                       # Fig SI32
    plot_fig5(sim_A, sim_B, f_B, mpfx)                                     # Fig SI31
    plot_SI5_combined(sim_A, sim_B, spfx)                                  # Fig SI27
    plot_SI1(sim_A, spfx)                                                  # Fig SI5
    plot_SI_epi_params(params, gen_time_pmf, max_days, sim_A, f_A, pops_A,
                       save_prefix=spfx)                                   # Fig SI2
    plot_SI_pde_convergence(city_A, f_A, sc["base_fA"], params, sc["initial_A"],
                            LW, LB, T_test=T, save_prefix=spfx)            # Fig SI4
    plot_SI_sensitivity(city_A, f_A, sc["base_fA"], params, sc["initial_A"],
                        LW, LB, T_test=T, save_prefix=spfx)                # Fig SI6
    plot_SI_3d_earlypeak(sim_A, sim_B, spfx)                               # Fig SI1

    print("\n─── Spatial aggregation bias: Fig 4 and Figs SI10-SI25 ───")
    bias_data = _compute_main_bias_data(sim_A, sim_B, sim_C, city_A, city_B, city_C,
                                        gen_time_pmf)
    plot_main_bias_figure(sim_A, sim_B, sim_C, city_A, city_B, city_C, gen_time_pmf,
                          save_prefix=mpfx, all_data=bias_data)            # Fig 4
    plot_estimator_si_suite(bias_data, save_prefix=mpfx)                  # Figs SI10-SI25

    print("\n─── Estimator mean-relative-error tables (Tables SI1-SI2) ───")
    scenarios = [("Scenario A", sim_A, city_A),
                 ("Scenario A-het", sim_A_het, city_A),
                 ("Scenario B", sim_B, city_B),
                 ("Scenario B-het", sim_B_het, city_B),
                 ("Scenario C", sim_C, city_C),
                 ("Scenario C-het", sim_C_het, city_C)]
    build_estimator_error_tables(scenarios, gen_time_pmf, out_dir="out_figs/tables",
                                 prefix="estimator_error")

    plot_SI0_population(city_A, city_B, spfx)                              # Fig SI3
    plot_SI_R_comparison(sim_A, city_A, R_ind_A, spfx)                     # Fig SI26

    print("\n─── Type reproduction number figures (Figs SI28-SI30) ───")
    plot_type_repro(sim_A, sim_B, city_A, city_B, mpfx)

    print("\n─── Figures 2, 3, 5 and their static-mobility twins (Figs SI7-SI9) ───")
    run_corrected_figures(sim_A, sim_A_static, city_A, f_A, f_A_static,
                          R_ind_A, R_ind_As, gt_snaps_A, gt_snaps_As,
                          gen_time_pmf, max_days,
                          scenario_name="Dense urban",
                          out_dir="out_figs/corrected")

    print("\n─── Lehmer-mean spectrum, Scenario C (Fig SI33) ───")
    plot_lehmer_mean_spectrum(sim_C, "Scenario_C", mpfx)

    print("\nAll figures saved.  Done.")


if __name__ == "__main__":
    main()
