# Multi-scale measures of time-varying epidemic spread on human mobility networks

Reference implementation for the manuscript

> **Multi-scale measures of time-varying epidemic spread on human mobility networks**
> Cathal Mills, Benjamin Reddy, William Hart, Robin Thompson, Kris V. Parag,
> Moritz U. G. Kraemer, Christl A. Donnelly, Ben Lambert.
> arXiv: https://doi.org/10.48550/arXiv.2607.28514

The code implements the network-based modelling framework from the paper, which folds
within-day human movement into mechanism-led renewal equations and derives a family of
instantaneous reproduction numbers, kernels, generation-time distributions, and
transience measures. These indicators describe transmission in space and time at the
network, location, transmission-corridor, and meeting-location scales.

---

## Repository layout

Analysis code lives in the `mobility_rt` package. 

```
mobility_rt/
    config.py             parameters (COVID_PARAMS, estimator priors, seeds)
    geometry.py           synthetic city geometry            (generate_city)
    mobility.py           within-day mobility matrices        (generate_mobility)
    distributions.py      generation-time discretisation      (discretise_gamma)
    kernel.py             transmission kernel and R^{kj} matrix
    spectral.py           spectral radius, eigenvectors, sensitivity/elasticity
    generation_time.py    generation-time distributions
    type_reproduction.py  type-reproduction numbers
    transmissibility.py   source-sink, power-mean spectrum, Euler-Lotka
    estimators.py         naive and independent R estimators
    simulation.py         renewal-PDE forward simulation
    framework.py          MobilityRtFramework class used by the notebooks
    plotting/             all figure and animation code (imports matplotlib)
    scripts/run_all.py    runs the three scenarios and writes every figure
```

The top-level files:

| File | Role |
|---|---|
| `make_corrected_figures.py` | Redraws the main-text overview, taxonomy, and spectral figures as tile heatmaps and 3-D surfaces. |
| `demo.ipynb` | End-to-end walkthrough of the framework on one synthetic network. |
| `exploration.ipynb` | Parameter exploration (β↔R₀, growth-rate inversion, sensitivity surfaces). |
| `test_mobility_rt.py` | Unit tests for the mathematical machinery. |

The figures are not stored in the repository. Running either pipeline creates an
`out_figs/` directory, with `main/`, `SI/`, and `corrected/` subfolders, and writes
the PDFs, PNGs, and GIFs there.

---

## Requirements

- Python ≥ 3.9
- `numpy`, `scipy`, `matplotlib`
- `pillow` — only for writing the animated GIFs
- `jupyter` (or JupyterLab) — only for the notebooks

The simplest way to set up is an editable install from the repository root, which
pulls in the dependencies and puts the `mobility_rt` package on the import path so the
commands below work from any directory:

```bash
pip install -e .            # add [notebooks] to also install jupyter
```

If you would rather not install anything, just install the dependencies directly and
run the commands from the repository root:

```bash
pip install numpy scipy matplotlib pillow jupyter
```

---

## Quick start

Run these from the repository root (or from anywhere after `pip install -e .`):

```bash
# Full pipeline: all main + SI figures + animations (~5-8 min)
python3 -m mobility_rt.scripts.run_all

# Redrawn main-text figures only (~1-2 min)
python3 make_corrected_figures.py

# Unit tests
python3 test_mobility_rt.py

# Notebooks (just for illustration)
jupyter lab demo.ipynb
jupyter lab exploration.ipynb
```

After `pip install -e .` the pipeline is also available as a console command,
`mobility-rt-run`.

---

## The three scenarios

Both pipelines simulate the same three synthetic networks (Methods §2.3–2.4):

| Scenario | Label in code | Network | Notes |
|----------|---------------|---------|-------|
| A | `lagos` | Dense-urban megacity (10 nodes: core / dense / suburban / peripheral) | Primary scenario. |
| B | `zambia` | Sparse-national, capital-dominated hub-and-spoke | Counterfactual. |
| C | hub-amplified | Scenario-A geometry with a single amplified hub (`hub_attraction_power` raised) | Non-normal, false-action-zone counterfactual. |

The shared epidemiological parameters are in `mobility_rt.COVID_PARAMS`: a
SARS-CoV-2-like generation time `Gamma(mean = 5.5 d, sd = 1.8 d)` truncated at 25 d
(Hart et al. 2022); `R(t=0) = R₀ = 1.5`; home contact rate `λ_W = 13.03`
contacts/day (POLYMOD, Mossong et al. 2008); away contact rate `λ_B = 0.30 · λ_W`.
The PDE is integrated with a first-order upwind scheme on a uniform grid
`Δt = Δa_E = 1 day`.

The continuous generation time is turned into a daily pmf by `discretise_gamma`,
using double-interval-censored discretisation (the distribution of `floor(U + X)`,
`U ~ Uniform(0,1)`), which recovers the specified mean exactly. This avoids the
roughly half-day downward mean bias of the naïve `F(d+1) − F(d)` scheme
(Park et al. 2024; Charniga et al. 2024) and matches the approach of the
`primarycensored` R package.

---

## Reproducing the main-text figures

Figure 1 in the manuscript is a schematic and is not produced by the code. The
internal names of the remaining figures do not all match the final manuscript
numbers. File `fig_04_spectral` is manuscript Figure 5
(transience/importation), and manuscript Figure 4 (estimator bias) is the file
`fig_main_bias_combined`.

The full pipeline builds the three scenarios once, computes the derived quantities,
and then calls a sequence of `plot_*` functions, which live in `mobility_rt/plotting/`.
The table below maps each manuscript figure to the function that draws it and the file
it writes. Rows marked `corrected` come from `make_corrected_figures.py`.

| Manuscript figure | Content | Drawn by | Output file(s) |
|---|---|---|---|
| Fig 1 | Conceptual schematic | — | — |
| Fig 2 | Overall mobility & epidemic dynamics (Scenario A) | `plot_fig2` | `out_figs/main/fig_02_overview.pdf` |
| | | `plot_fig02_overview` (corrected) | `out_figs/corrected/fig_02_overview.pdf` (2-D tiles) and `…_02_overview_surface.pdf` (3-D) |
| Fig 3 | Taxonomy of reproduction numbers & generation-time distributions | `plot_fig3` | `out_figs/main/fig_03_taxonomy.pdf` |
| | | `plot_fig03_taxonomy` (corrected) | `out_figs/corrected/fig_03_taxonomy.pdf` and `…_03_taxonomy_surface.pdf` |
| Fig 4 | Bias in network-level R̂ estimators (rows = Scenarios A/B/C) | `plot_main_bias_figure` | `out_figs/main/fig_main_bias_combined.pdf` |
| Fig 5 | Transience & importation (mixing ratio, reactivity σ, amplification envelopes, eigenvalues, condition number, within-location fraction) | `plot_fig4` | `out_figs/main/fig_04_spectral.pdf` |
| | | `plot_fig04_spectral` (corrected) | `out_figs/corrected/fig_04_spectral.pdf` |

`make_corrected_figures.py` produces only the redrawn Fig 2, Fig 3, and Fig 5.
Everything else, including the Fig 4 bias figure, comes from the full pipeline. The
corrected script re-simulates Scenario A (time-varying and static mobility) plus
Scenario B for the reference NGM, and also writes static-mobility counterparts
(`fig_SI_static_02_overview*`, `…_03_taxonomy*`, `…_04_spectral`).

---

## Supplementary (SI) figures

Written by the full pipeline into `out_figs/SI/`:

| Output file | Content | Function |
|---|---|---|
| `fig_SI0_population.pdf` | Population & node-type structure, Scenarios A vs B | `plot_SI0_population` |
| `fig_SI1_sensitivity.pdf` | Sensitivity & elasticity maps, Scenario A | `plot_SI1` |
| `fig_SI2_counterfactual.pdf` | R₀ = 1.2 counterfactual run | `plot_SI2` |
| `fig_SI3_epi_params.pdf` | Epidemiological / mobility parameter assumptions (GT, day-of-week scaling, N_eff, λ_eff) | `plot_SI_epi_params` |
| `fig_SI4_convergence.pdf` | Numerical validation: GT-truncation & solver (upwind / Heun / RK4) convergence, exact boundary-condition check | `plot_SI_pde_convergence` |
| `fig_SI5_meeting_combined.png` / `fig_SI5*_meeting.png` | Meeting-location reproduction numbers R^l_meeting | `plot_SI5_combined`, `plot_fig6` |
| `fig_SI6_gt_comparison.pdf` | Generation-time comparison across quantities | `plot_fig7` |
| `fig_SI7_sensitivity.pdf` | Univariate parameter sensitivity of R(t) | `plot_SI_sensitivity` |
| `fig_SI8_3d_earlypeak.pdf` | 3-D early-vs-peak reproduction-number surfaces | `plot_SI_3d_earlypeak` |
| `fig_SI_gt_varying_beta.pdf` | Generation time under varying β | `plot_SI_gt_varying_beta` |
| `fig_SI_lambda_decomp.pdf` | Within- vs between-location kernel decomposition | `plot_SI_lambda_decomposition` |
| `fig_SI_R_comparison.pdf` | Independent R̂ⱼ vs framework R^j_out / R^j_in | `plot_SI_R_comparison` |
| `fig_SI_gt_spatial.pdf` | Spatial variation of generation times | `plot_SI_gt_spatial` |
| `fig_SI_elasticity_surfaces.pdf` | 3-D elasticity surfaces over time | `plot_elasticity_surfaces` |
| `fig_SI_static_02_overview*`, `…_03_taxonomy*`, `…_04_spectral*` | Static-mobility counterparts of the main figures | `plot_fig2/3/4` (static run) |

### Additional outputs in `out_figs/main/`

The full pipeline also writes several analysis figures and animations that support the
main text:

- Estimator-bias building blocks. `plot_naive_R_comparison_suite` produces the
  per-scheme comparison PDFs (`fig_naive_*`, `fig_aggregate_*`) for the aggregate,
  population-weighted, incidence-weighted, arithmetic-mean, and
  relative-transmissibility-weighted estimators. These are summarised in the combined
  Fig 4 (`fig_main_bias_combined.pdf`).
- Scenario-C transience. `fig_scenario_C_transience.pdf` (`plot_fig_C_transience`), the
  hub-amplified false-action-zone figure (σ > 1 while R < 1).
- Type-reproduction numbers. `fig_type_heatmaps.png`, `fig_type_surfaces.png`,
  `fig_type_groups.png` (`plot_type_repro`).
- Power-mean (risk-averse) spectrum. `fig_power_mean_spectrum_Scenario_{A,B,C}.pdf`
  (`plot_power_mean_spectrum`).
- Counterfactual non-normality. `fig_counterfactual_nonnormal.pdf`.
- Animations. `fig_anim_01*_network_{incidence,cum_incidence,r_out,eigvec}.gif`,
  `fig_anim_02_R_heatmap.gif`, `fig_anim_03_scenario_AC.gif`,
  `fig_anim_04_elasticity.gif` (these need `pillow`).

---

## The notebooks

Both notebooks import from `mobility_rt.framework` and run top to bottom without
arguments.

### `demo.ipynb` — framework walkthrough

A tour of every framework output on a single synthetic network, using the
`MobilityRtFramework` class:

1. Setup & infectiousness profile
2. Build a synthetic mobility network
3. Run the simulation
4. Network-level indicators: R(t) (threshold), reactivity σ(t), risk-averse E(t)
5. Location-level reproduction numbers: outward R^k_out(t), inward R^j_in(t)
6. Pairwise R^{kj}(t) matrix, sensitivity & elasticity (Eqs. 29–35)
7. Generation-time distributions, showing the universal-GT property
8. Transience: mixing ratio, amplification envelope, condition number (§3.1.5)
9. Type-reproduction numbers T^j_type(t) (Eq. 54)
10. Spatial source–sink analysis and minimum control effort
11. Bias of the independent (closed-population) estimator
12. Counterfactual on the sparse-rural network

### `exploration.ipynb` — parameter exploration

How the framework's outputs respond to epidemiological and network parameters:

1. Realistic parameter set-up
2. β → R₀ mapping, and inverting an observed growth rate / doubling time to estimate β
3. Full simulation in β-mode
4. Univariate sensitivity analyses
5. Bivariate parameter surfaces (e.g. GT mean × GT sd)
6. Cross-network comparison
7. Uncertainty propagation (β uncertainty → R₀ uncertainty)

The final section is a static preview of a planned interactive dashboard.

---

## Tests

```bash
python3 test_mobility_rt.py
```

The suite has 40 tests in two groups. `TestMobilityRt` checks the model itself: the
row-stochasticity of the mobility matrices, the eigenvector and stable-distribution
identities, the sensitivity and elasticity relations, the Euler–Lotka relation, the
double-censored discretisation (it recovers the target mean), and the universal
generation-time property (g is identical across all location pairs, times, and
day-of-week mobility). 

---

## License

For the Author Accepted Manuscript, a CC BY public copyright licence applies (see the
manuscript's copyright statement).
