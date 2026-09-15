# Multi-scale measures of time-varying epidemic spread on human mobility networks

Reference implementation for the manuscript

> **Multi-scale measures of time-varying epidemic spread on human mobility networks**
> Cathal Mills, Benjamin Reddy, William Hart, Robin Thompson, Kris V. Parag,
> Moritz U. G. Kraemer, Christl A. Donnelly, Ben Lambert.
> arXiv: https://doi.org/10.48550/arXiv.2607.28514

The code implements the network-based modelling framework of the paper, which folds
within-day human movement into mechanism-led renewal equations and derives a family of
instantaneous reproduction numbers, kernels, generation-time distributions and
transience measures at the network, location, transmission-corridor and
meeting-location scales. It contains only what the manuscript's figures and tables use.

---

## Repository layout

```
mobility_rt/
    config.py             parameters and constants (COVID_PARAMS, estimator prior,
                          day-of-week factors, λ_B/λ_W, σ_ξ per scenario)
    geometry.py           synthetic city geometry                     (generate_city)
    mobility.py           time-varying mobility matrices f^{jk}(t)    (generate_mobility)
    distributions.py      generation-time discretisation              (discretise_gamma, renewal_gt)
    kernel.py             kernel, R^{kj}(t) (Eq 14) and R^l_meeting (Eq 50)
    spectral.py           ρ(R), eigenvectors, sensitivity/elasticity, κ, σ, A(n), A1(n)
    generation_time.py    generation-time distributions, g_network (Eqs 27-28)
    type_reproduction.py  type reproduction numbers (Eqs 53-56)
    transmissibility.py   Euler-Lotka, E(t) (Eq 42), Lehmer-mean spectrum (Eq 41),
                          within-location fraction
    simulation.py         renewal-PDE forward simulation, phase days
    estimators.py         independent (Cori) estimators and network aggregations
    error_tables.py       Tables SI1-SI2
    heterogeneity.py      contact and infectiousness multipliers of the -het scenarios
    plotting/             figure code (imports matplotlib)
    scripts/run_all.py    builds every scenario and writes every manuscript figure and table
scripts/manifest.py       numerical fingerprint of every scenario (for comparing code versions)
test_mobility_rt.py       tests
requirements-lock.txt     exact package versions used for the manuscript
```

## Requirements

- Python ≥ 3.12
- `numpy`, `scipy`, `matplotlib` (and `pytest` for the tests)
- The exact versions that produced the manuscript's outputs are in `requirements-lock.txt`.

```bash
pip install -r requirements-lock.txt
pip install -e .
```

## Quick start

From the repository root:

```bash
python3 -m mobility_rt.scripts.run_all          # every figure and table, about 1 minute
python3 -m pytest -q test_mobility_rt.py         # tests, about 15 s
python3 scripts/manifest.py --out out_manifest/run.json          # numerical manifest
python3 scripts/manifest.py --compare old.json new.json          # what changed
```

Outputs are written to `out_figs/{main,SI,corrected,tables}/` (not versioned).

---

## Model choices

**Scenarios** (built in `run_all.build_scenarios`):

| Scenario | Network | Construction |
|---|---|---|
| A | Dense urban (Lagos-like), 10 locations | seed 42, σ_ξ = 0.15, 10 infections at L1 |
| A-static | A with f(t) = `base_f` on every day | |
| B | Sparse national (Zambia-like), 10 locations | σ_ξ = 0.12, 10 infections at L1 |
| C | Hub-amplified | A's geometry, commuting fractions, mobility seed and seeding; only the populations (a 3M hub at L1, nine 200k satellites) and `hub_attraction_power = 8` differ. Types: hub "core", satellites "peripheral" |
| A-het, B-het, C-het | as A, B, C | with contact multipliers c[l] by settlement type and lognormal infectiousness multipliers m[k] (illustrative assumptions, `heterogeneity.py`) |

**Calibration.** λ_W and λ_B are scaled by one common factor so that ρ(R) = R0 = 1.5 for
a fully susceptible population under each scenario's noise-free weekday mobility
`base_f`. Only λ_B/λ_W = 0.30 is identified; the POLYMOD contact rate and
`prob_transmission_peak` have no effect on any output. ρ(R(t)) on day 0 uses that day's
mobility draw and so differs slightly from 1.5.

**Population.** Closed: no births or deaths, so S_end(t) + cumulative incidence = N.

**Mobility.** Day-of-week factors on the away-from-home fraction are
(1, 1, 1, 1, 0.95, 0.90, 0.75), Monday first; a shared daily lognormal factor ξ_t is
clipped to [0.5, 1.8]; distances are floored at 0.5 km.

**Infectiousness profile.** Gamma(mean 5.5 d, sd 1.8 d), double-interval-censored to a daily
pmf and truncated at 25 d. Transmission uses infection ages a_E ≥ 1 only.

## Conventions

- `R_matrices[t, k, j]` is R^{kj}(t), infector k and infectee j: the transpose of the
  manuscript's next-generation matrix. Hence `spectral_analysis` "right_eigvec" is the
  manuscript's reproductive value v and "left_eigvec" its stable distribution v*.
- `S_start[t]` is the susceptible count at the start of day t (`S_start[0] = N`); `R_matrices[t]`,
  `R_meeting_series[t]` and the day-t force of infection use it. `S_end[t]` is the count at
  the end of day t.
- A sliding-window estimate at day t summarises days t−6 … t.

## Estimator comparisons (Fig 4, Figs SI10–SI25, Tables SI1–SI2)

Nine network-level summaries are compared with ρ(R(t)) (and E(t)): the aggregate-incidence
estimate, and the arithmetic-mean, population-weighted, infectiousness-weighted and
relative-transmissibility-weighted means of the per-location independent estimates and of
the outward reproduction numbers.

- **Infectiousness weighting** weights location j by its total infectiousness
  Λ_j = Σ_{a≥1} p(a) I_j(t−a) (ref [68]); for a windowed estimate Λ_j is summed over the same
  window, which makes the weighted independent mean close to the aggregate estimate.
- **Matched reference.** Windowed estimates are scored against the 7-day trailing mean of
  the target; unwindowed quantities against the target on day t.
- **Common days.** Within a scenario every scheme is scored on the same days (ρ > 0.05,
  every scheme and reference defined); growth is ρ > 1.
- **Threshold crossing.** The day a series settles at or below 1, counted on those scored
  days, reported as the estimate's crossing minus that of the daily target.

The statistics printed on the bias figures equal the corresponding table cells (tested).

## Manuscript figure and table map

| Manuscript | File | Function |
|---|---|---|
| Fig 2 / SI7 | `corrected/fig_02_overview_surface.pdf` / `fig_SI_static_02_overview.pdf` | `figures_corrected.plot_fig02_overview` |
| Fig 3 / SI8 | `corrected/fig_03_taxonomy_surface.pdf` / `fig_SI_static_03_taxonomy.pdf` | `figures_corrected.plot_fig03_taxonomy` |
| Fig 4 | `main/fig_main_bias_combined.pdf` | `figures_main.plot_main_bias_figure` |
| Fig 5 / SI9 | `corrected/fig_04_spectral.pdf` / `fig_SI_static_04_spectral.pdf` | `figures_corrected.plot_fig04_spectral` |
| SI1 | `SI/fig_SI8_3d_earlypeak.pdf` | `figures_si.plot_SI_3d_earlypeak` |
| SI2 | `SI/fig_SI3_epi_params.pdf` | `figures_si.plot_SI_epi_params` |
| SI3 | `SI/fig_SI0_population.pdf` | `figures_si.plot_SI0_population` |
| SI4 | `SI/fig_SI4_convergence.pdf` | `figures_si.plot_SI_pde_convergence` |
| SI5 | `SI/fig_SI1_sensitivity.pdf` | `figures_si.plot_SI1` |
| SI6 | `SI/fig_SI7_sensitivity.pdf` | `figures_si.plot_SI_sensitivity` |
| SI10–SI25 | `main/fig_naive_{scheme}_vs_{Rt,Et}.pdf` | `figures_main.plot_estimator_si_suite` |
| SI26 | `SI/fig_SI_R_comparison.pdf` | `figures_si.plot_SI_R_comparison` |
| SI27 | `SI/fig_SI5_meeting_combined.png` | `figures_si.plot_SI5_combined` |
| SI28–SI30 | `main/fig_type_{surfaces,heatmaps,groups}.png` | `figures_main.plot_type_repro` |
| SI31 | `main/fig_05_comparison.pdf` | `figures_main.plot_fig5` |
| SI32 | `main/fig_scenario_C_transience.pdf` | `figures_main.plot_fig_C_transience` |
| SI33 | `main/fig_lehmer_mean_spectrum_Scenario_C.pdf` | `figures_main.plot_lehmer_mean_spectrum` |
| Tables SI1–SI2 | `tables/estimator_error_{full,growth}.{csv,tex}` | `error_tables.build_estimator_error_tables` |

Fig 1 is a schematic and is not produced by the code.

## Tests

`python3 -m pytest -q test_mobility_rt.py` runs oracle tests (brute-force and closed-form
checks of Eqs 14, 27–28, 31–35, 38–42, 50, 53–56, the renewal identity, the discretisation
and the estimators), the scenario decisions (calibration, closed population, Scenario C
construction), the error tables, and the plotting fixes.

## License

For the Author Accepted Manuscript, a CC BY public copyright licence applies (see the
manuscript's copyright statement).
