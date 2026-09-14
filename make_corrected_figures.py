#!/usr/bin/env python3
"""Back-compat shim for the corrected manuscript figure workflow.

The plotting code now lives in mobility_rt.plotting.figures_corrected and is called
from mobility_rt.scripts.run_all, so the whole pipeline is self-contained and the
corrected figures are drawn from the same Scenario A simulation as everything else.

Prefer:  python3 -m mobility_rt.scripts.run_all

This script is kept so that `python3 make_corrected_figures.py` still produces the
corrected figures on their own.  It rebuilds Scenario A with exactly the parameters
run_all uses, then calls the same drawing code.
"""
from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import numpy as np

import mobility_rt as dt
from mobility_rt.plotting.figures_corrected import (  # noqa: F401  (re-exported)
    plot_fig02_overview,
    plot_fig03_taxonomy,
    plot_fig04_spectral,
    run_corrected_figures,
    set_modern_style,
)


def run_corrected_figure_suite(
    *,
    out_dir: str = "out_figs/corrected",
    N_LOC: int = 10,
    T: int = 365,
    SEED: int = 42,
):
    """Rebuild Scenario A (time-varying and static mobility) and draw the figures.

    Parameters are kept identical to mobility_rt.scripts.run_all, so this standalone
    path and the pipeline produce the same figures.
    """
    os.makedirs(out_dir, exist_ok=True)

    params = dt.COVID_PARAMS
    max_days = params["max_gen_time"]
    gen_time_pmf = dt.discretise_gamma(params["gen_time_mean"],
                                       params["gen_time_sd"], max_days)
    infect_profile = gen_time_pmf.copy()
    w_within = w_between = infect_profile
    LW = params["base_contact_rate"]
    LB = params["base_contact_rate"] * 0.30

    print("─── Scenario A (dense urban) ───")
    city_A = dt.generate_city(N_LOC, scenario="lagos", seed=SEED)
    coords_A, pops_A, dists_A, types_A, meta_A = city_A
    f_A, base_fA = dt.generate_mobility(N_LOC, T, pops_A, dists_A, types_A, meta_A,
                                        day_variation_sd=0.15, seed=SEED)
    initial_A = np.zeros(N_LOC)
    initial_A[0] = 10

    def _sim(f_series):
        return dt.simulate_epidemic_pde(
            T, N_LOC, pops_A, f_series,
            params["prob_transmission_peak"], infect_profile, max_days,
            params["R0_target"], initial_A, LW, LB, w_within, w_between,
            birth_rate=3e-5, death_rate=3e-5,
            stochastic=False, susceptible_depletion=True, seed=SEED)

    sim_A = _sim(f_A)
    f_static = np.stack([base_fA] * T, axis=0)
    sim_static = _sim(f_static)

    R_ind_A = dt.estimate_R_independent(sim_A["incidence"], gen_time_pmf, window=7)
    R_ind_s = dt.estimate_R_independent(sim_static["incidence"], gen_time_pmf, window=7)

    def _gt_snaps(sim, f_series):
        pk = int(sim["incidence"].sum(axis=1).argmax())
        lw, lb = sim["lambda_within_scaled"], sim["lambda_between_scaled"]
        snaps = {}
        for nm, day in [("early", max(1, pk // 3)), ("peak", pk),
                        ("late", min(T - 1, pk + 30))]:
            snaps[nm] = dt.compute_generation_times(
                f_series[day], sim["susceptibles"][day], pops_A,
                params["prob_transmission_peak"], w_within, max_days, lw, lb)
        return snaps

    return run_corrected_figures(
        sim_A, sim_static, city_A, f_A, f_static,
        R_ind_A, R_ind_s, _gt_snaps(sim_A, f_A), _gt_snaps(sim_static, f_static),
        gen_time_pmf, w_within, w_between, max_days,
        scenario_name="Dense urban", out_dir=out_dir)


if __name__ == "__main__":
    run_corrected_figure_suite()
