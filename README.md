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

All of the analysis code lives in the `mobility_rt` package. The mathematical core is
kept separate from the plotting code, so the model can be imported and used without
matplotlib installed.

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

# Notebooks
jupyter lab demo.ipynb
jupyter lab exploration.ipynb
```

After `pip install -e .` the pipeline is also available as a console command,
`mobility-rt-run`.

---

## Using your own data

Everything below this section reproduces the manuscript. This section is for applying
the framework to your own network.

The package is a **forward model with a diagnostic suite attached**, not an inference
engine. Given a mobility network, resident populations, and a generation-time
distribution, it simulates the renewal PDE and returns the whole family of
reproduction numbers — `R^{kj}`, `R^k_out`, `R^j_in`, `R^l_meeting`, `ρ(R)`,
`T^j_type` — alongside the spectral diagnostics (reactivity `σ`, mixing ratio,
spectral gap, sensitivity, elasticity). It does **not** fit parameters to observed
case counts: there is no likelihood and no optimiser anywhere in the package. What it
offers instead is one data-anchored entry point, `estimate_beta_from_growth_rate`,
described under [Route 2](#route-2--anchoring-the-model-on-observed-data).

### What you have to supply

| Input | Symbol | Shape | Requirement |
|---|---|---|---|
| Mobility matrix | `f^{jk}(t)` | `(T,N,N)` or `(N,N)` | Row-stochastic; a 2-D matrix is broadcast over `T` |
| Resident populations | `N_j` | `(N,)` | Strictly positive |
| Infectiousness profile | `p(a_E)` | `(max_days,)` | Non-negative daily pmf; normalised internally |
| Seed infections | — | `(N,)` | Non-negative, at least one positive, none exceeding its population |
| Contact rates | `λ_W`, `λ_B` | scalars | Contacts/day at home and away |
| Transmission intensity | `β` **or** `R₀` | scalar | Exactly one of the two — see below |

`validate_inputs` checks all of this up front and collects the problems it finds into
a single error message rather than failing on the first one, so a malformed input
surfaces as one readable list.

#### Choosing β or R₀

These are two different questions and the class refuses to let you supply both.

- **`R0_target=...`** scales `λ_W` and `λ_B` so that `ρ(R(0))` equals your target.
  Use it for scenario analysis where `R₀` is a known quantity you want to impose.
- **`beta=...`** is the per-contact transmission probability; `λ_W` and `λ_B` are used
  as given and `R₀` comes out as a *result*. Use it when the transmission biology is
  what you know, or when sweeping.

#### Building `f_jk` — the step people get wrong

`f_jk[t, j, k]` is the probability that a **resident of `j`** is present in
**location `k`** at time `t`. Rows index residence, columns index presence, the
diagonal `f_jk[t,j,j]` is the fraction of the day spent at home, and **every row must
sum to 1** (to within `1e-8`). This is not a commuter-flow matrix; convert one by
row-normalising:

```python
f = commuter_counts / commuter_counts.sum(axis=1, keepdims=True)
```

If instead you have off-diagonal flows plus a separate "fraction of the day spent away
from home" `c_j` — the usual shape of a CDR or travel-survey summary — scale the
off-diagonal row to `c_j` and put the remainder on the diagonal:

```python
flows = raw_flows.astype(float)
np.fill_diagonal(flows, 0.0)
f = c[:, None] * flows / flows.sum(axis=1, keepdims=True)
np.fill_diagonal(f, 1.0 - c)
```

For a time-varying network pass `(T,N,N)` with `f_jk.shape[0] == T` exactly. If you
have no network at all and want a plausible synthetic one to test against,
`build_synthetic_network(N=..., T=...)` returns a gravity-model `f_jk` with
distance decay, day-of-week structure, and daily noise.

#### One caveat on the generation time

The model excludes same-day transmission: every reproduction number integrates
`p(a_E)` over `a_E ≥ 1`, and the mass at `p[0]` is dropped rather than redistributed.
For the SARS-CoV-2-like default, `Gamma(5.5, 1.8)`, that mass is `4×10⁻⁶` and nothing
changes. For a short generation time it is not negligible — `Gamma(1.5, 0.9)` puts
11.8 % of its mass at `a_E = 0`, and the *effective* mean generation interval the model
runs on is then 1.70 d rather than the 1.5 d you asked for. Under `R0_target` the
calibration absorbs the lost mass so `R₀` is still hit exactly, but the epidemic's
speed reflects the longer effective interval. Check `p[0]` before trusting a
fast-pathogen run.

### Route 1 — forward scenario modelling

This is the fully supported path. Supply the network and run:

```python
import numpy as np
from mobility_rt.framework import MobilityRtFramework, discretise_gamma

model = MobilityRtFramework(
    f_jk                   = f,                 # (T,N,N) or (N,N), row-stochastic
    populations            = np.array([820_000., 260_000., 140_000., 95_000.]),
    infectiousness_profile = discretise_gamma(mean=5.5, sd=1.8, max_days=25),
    contact_rate_home      = 13.0,              # λ_W, POLYMOD
    contact_rate_away      = 3.9,               # λ_B = 0.30 · λ_W
    R0_target              = 2.0,               # or beta=0.04 — exactly one
    initial_infections     = np.array([10., 0., 0., 0.]),
    T                      = 180,
    location_names         = ["Central", "North", "East", "Rural"],
)
results = model.simulate(compute_type_r=True, type_r_stride=5)
```

`simulate()` returns a dict of 29 entries — 25 arrays, three scalars, and `type_R`,
which is `None` unless you ask for it. The ones most people want:

| Key | Shape | Quantity |
|---|---|---|
| `incidence` | `(T,N)` | New infections per location per day |
| `R_pairwise` | `(T,N,N)` | `R^{kj}(t)` — infections in `j` caused by an infector resident in `k` |
| `R_outward`, `R_inward` | `(T,N)` | Row and column sums of `R^{kj}` |
| `R_network` | `(T,)` | `ρ(R(t))` — the system reproduction number |
| `R_meeting` | `(T,N)` | `R^l_meeting(t)` — transmission capacity of `l` as a venue |
| `reactivity` | `(T,)` | `σ(t) = ‖R(t)‖₂`; `σ>1` with `ρ<1` is the false-action zone |
| `type_R` | `(T,N)` | `T^j_type(t)`, `NaN` where undefined — only if `compute_type_r=True` |
| `reproductive_value`, `stable_distribution` | `(T,N)` | Right and left Perron vectors |
| `sensitivity`, `elasticity` | `(T,N,N)` | `∂ρ/∂R_{kj}` and its elasticity |
| `R_independent` | `(T,N)` | Mobility-blind Cori estimate, for comparison |
| `R0_achieved` | scalar | `ρ(R(0))` actually realised |

Point diagnostics are available per day — `model.source_sink_at(t)`,
`model.type_R_at(t)`, `model.controllability_effort(t)`,
`model.get_amplification_envelope(t)` — and five figure methods
(`plot_overview`, `plot_R_taxonomy`, `plot_transience`, `plot_generation_times`,
`plot_eigenvectors`) each take an optional `save_path`.

To sweep rather than run a single scenario, `parameter_sweep` takes a `param_grid`
over any of `beta`, `lb_lw_ratio`, `gt_mean`, `gt_sd`, `commuting_scale` and returns
`R₀`, attack-rate, peak-day, `R(t)` and `σ(t)` grids.

### Route 2 — anchoring the model on observed data

There is no likelihood, so you cannot fit this model to a case series. What you *can*
do is pin its transmission intensity to an observed early growth rate and then
evaluate the reproduction numbers against your own susceptible reconstruction. Three
steps.

**Step 1 — invert the observed growth rate for β.**
`estimate_beta_from_growth_rate` solves Euler–Lotka for the `R₀` implied by `r` and
your generation time, then inverts `R₀(β)` on your network:

```python
from mobility_rt.framework import estimate_beta_from_growth_rate, empirical_growth_rate

r_hat = empirical_growth_rate(observed_incidence, window=3)   # expects (T, N)
r_obs = float(np.nanmedian(r_hat[10:35]))                     # early exponential phase

cal = estimate_beta_from_growth_rate(
    r_observed=r_obs, infectiousness_profile=p, f0=f,
    populations=populations, contact_rate_home=13.0, contact_rate_away=3.9,
)
# {'beta': 0.150, 'R0': 1.61, 'R0_check': ..., 'r_check': ..., 'doubling_time': 7.8}
```

Pass a 1-D total as `total[:, None]`, since `empirical_growth_rate` sums over axis 1.

**Step 2 — reconstruct `S_j(t)` yourself.** The package does not do this. A cumulative
sum of ascertained cases is the crude version and ignores under-ascertainment, so
substitute whatever reconstruction you would normally defend:

```python
S = np.maximum(populations[None, :] - np.cumsum(observed_incidence, axis=0), 0.0)
```

**Step 3 — evaluate the reproduction numbers on your observed network and `S(t)`.**

```python
from mobility_rt import compute_R_matrix, R_outward, R_system, reactivity

lw, lb = cal["beta"] * 13.0, cal["beta"] * 3.9      # β folded into the contact rates
rho, sigma = np.empty(T), np.empty(T)
for t in range(T):
    R_t      = compute_R_matrix(f_series[t], S[t], populations, 1.0, p, lw, lb)
    rho[t]   = R_system(R_t)
    sigma[t] = reactivity(R_t)["sigma"]
```

> **The `1.0` is load-bearing.** `mobility_rt.compute_R_matrix` takes `prob_peak` as
> its fourth positional argument. Because `β` has already been folded into `lw` and
> `lb` on the line above, `prob_peak` must be `1.0` or `β` is applied twice and every
> reproduction number is scaled by `β²`. Supply `β` in exactly one of the two places.

Finally, compare against what a mobility-blind analysis of the same data would have
told you — the gap is the object of study:

```python
from mobility_rt import estimate_R_independent, compute_naive_R_aggregate

R_naive_loc = estimate_R_independent(observed_incidence, p, window=7)             # (T,N)
R_naive_agg = compute_naive_R_aggregate(observed_incidence.sum(axis=1), p, window=7)
```

Both apply the Cori/EpiEstim `Gamma(1, 5)` prior and report `NaN` in any window
holding less than one case.

### Two entry points, one set of equations

There are two public surfaces and they use **different argument orders**:

| | `mobility_rt` (core) | `mobility_rt.framework` |
|---|---|---|
| Import | `from mobility_rt import compute_R_matrix` | `from mobility_rt.framework import ...` |
| `compute_R_matrix` | `(f, S, pops, prob_peak, p, lw, lb)` | `(f, S, pops, p, lw, lb)` |
| Spatial heterogeneity | `contact_mult`, `infect_mult` supported | not supported |
| matplotlib required | no | yes, at import |

The framework's variant is the core's with `prob_peak = 1.0` and no heterogeneity
multipliers; `β` lives in `lw`/`lb` instead. The two are numerically identical —
`TestFrameworkCoreConsistency` asserts parity for all twenty duplicated functions,
including bit-identical output from the two simulators — but because they share
function *names* while differing in argument *order*, mixing the two imports in one
script will silently compute the wrong thing. Pick one and import it explicitly.

Use the core API if you need `contact_mult`/`infect_mult` spatial heterogeneity or a
matplotlib-free dependency footprint; use the framework API otherwise.

### Scale

Memory is `O(T·N²)` — `R_pairwise`, `incidence_matrix`, `sensitivity` and `elasticity`
are each `(T,N,N)` — and each simulated day costs two eigendecompositions plus an SVD.
Measured on one core, `T = 365`:

| `N` | Runtime | Peak memory |
|---|---|---|
| 20 | 2.5 s | 7 MB |
| 60 | 9.1 s | 49 MB |
| 120 | 25.7 s | 181 MB |

That extrapolates to roughly 3 GB and several minutes at `N = 500`. The framework is
comfortable at the scale of districts or health zones; it is not built for
thousand-node mobile-phone networks without first aggregating the network.

### What the package does not do

- **No parameter fitting.** No likelihood, no MCMC, no optimiser. `R₀`, `β`, `λ_W`,
  `λ_B` and the generation time are all inputs, not estimates.
- **No susceptible reconstruction.** You supply `S(t)` or let the simulator generate it.
- **No observation model.** Incidence is infections, not reported cases: no reporting
  delay, no under-ascertainment, no day-of-week reporting effect.
- **No uncertainty propagation.** The simulator is deterministic; the only credible
  intervals anywhere are the Gamma posteriors inside the Cori estimators.
- **No file I/O for inputs.** Everything is passed as NumPy arrays; loading your own
  network and case data is on you.

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
`primarycensored` package.

---

## Reproducing the main-text figures

Figure 1 in the manuscript is a schematic and is not produced by the code. The
internal names of the remaining figures do not all match the final manuscript
numbers. In particular, the file `fig_04_spectral` is manuscript Figure 5
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

The corrected figures are now drawn by the full pipeline too, from the same
Scenario A simulation as every other figure, so `python3 -m mobility_rt.scripts.run_all`
is self-contained. Their code lives in `mobility_rt/plotting/figures_corrected.py`;
`make_corrected_figures.py` is a back-compat shim that rebuilds Scenario A with the
pipeline's parameters and calls the same drawing code, for anyone who wants the
corrected set on its own. Static-mobility counterparts (`fig_SI_static_02_overview*`,
`…_03_taxonomy*`, `…_04_spectral`) are written alongside.

The same corrected panel set is written for every alternative scenario, so any two of
them can be read side by side without adjusting for a different layout. Each prefix
below carries `_02_overview.pdf`, `_02_overview_surface.pdf`, `_03_taxonomy.pdf`,
`_03_taxonomy_surface.pdf` and `_04_spectral.pdf` in `out_figs/corrected/`:

| Prefix | Scenario |
|---|---|
| `fig` | Dense urban (A), the manuscript run |
| `fig_SI_static` | Dense urban (A) with time-invariant mobility |
| `fig_A_het` | Dense urban with heterogeneous contact rate and transmissibility |
| `fig_B`, `fig_B_het` | Sparse national, baseline and heterogeneous |
| `fig_C`, `fig_C_het` | Hub-amplified, baseline and heterogeneous |

The `-het` runs switch on the two multipliers of `mobility_rt/heterogeneity.py`: a
contact-rate multiplier `c[l]` at the meeting location, by settlement type, and a
lognormal transmissibility multiplier `m[k]` for an infector resident in `k`.
Everything else — geometry, populations, mobility series, seeding, R₀ target,
demography, solver — is identical to the baseline it sits next to.

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
| `fig_SI_gt_heterogeneity.pdf` | Generation time under spatially heterogeneous contact and transmissibility: forward GT universal in every scenario, realised (backward) interval varying by location | `plot_SI_gt_heterogeneity` |
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
- Weighting-scheme variants of Fig 4. `plot_main_bias_variant_figures` redraws
  `fig_main_bias_combined` seven times with the same layout, colours and statistics,
  changing only the middle column to a different aggregation scheme while the right
  column stays the E(t) benchmark:
  `fig_main_bias_combined_agg.pdf` (sliding-window R̂ on network-total incidence),
  `…_ind_am.pdf` / `…_out_am.pdf` (arithmetic mean of R^j_ind / R^j_out),
  `…_ind_iw.pdf` / `…_out_iw.pdf` (incidence-weighted), and
  `…_ind_pw.pdf` / `…_out_pw.pdf` (population-weighted).
- Heterogeneous-transmissibility scenarios. Each of Scenarios A, B and C has an
  `-het` counterpart in which the contact rate varies by settlement type at the
  meeting location and the per-contact transmission probability varies lognormally
  by location of residence (`mobility_rt/heterogeneity.py`). Everything else is
  identical, so the pair isolates the effect of heterogeneous transmissibility on
  spatial aggregation bias. The same nine bias figures are written for them with a
  `fig_het_` prefix. The multipliers are illustrative modelling assumptions, not
  fitted or literature-calibrated values.
- The single manuscript table. `out_figs/tables/manuscript_table.csv` and `.tex`
  (`mobility_rt/combined_table.py`) bring both halves of the analysis together on
  one set of days: the nine network-level aggregation schemes with their mean
  relative error and signed bias against ρ(R(t)), and the location-level
  diagnostics of what a single location's own estimate tells you. Nothing is
  recomputed there; every figure is lifted from the tables below, and the builder
  refuses to emit anything unless the two halves used identical days, which is what
  makes it legitimate to read a network row against a location row.
- Per-location discrepancy. `out_figs/tables/location_discrepancy.csv` and `.tex`
  (`mobility_rt/location_errors.py`) with the matching
  `fig_location_discrepancy.pdf` (`plot_location_discrepancy`) ask the
  location-level question underneath the network-level tables: when you estimate a
  reproduction number independently for one location, what are you learning about?
  Each location's estimate is compared with two things, both taken as their 7-day
  trailing means to match the window the estimate summarises: that location's own
  outward reproduction number, and the network ρ(R(t)). The estimate is the closer
  of the two to the *network* value, because mobility mixes the infectious pressure
  arriving in a location, so its renewal ratio reflects the whole network rather
  than its own transmissibility. These are therefore a discrepancy, not an
  estimator error: the independent estimator is not trying and failing to measure
  the location's own R, it is measuring something else.
- Per-location diagnostics. `out_figs/tables/location_diagnostics.csv` and `.tex`
  with `fig_location_diagnostics.pdf` (`plot_location_diagnostics`) turn those
  distances into the three questions a public health team actually has. Can I rank
  locations by their estimates to find the worst place (across-location rank
  correlation with the true outward reproduction numbers, which is *negative* in
  both dense heterogeneous scenarios, so ranking points the wrong way)? Does a
  panel of estimates convey how heterogeneous the network is (the estimates
  understate the true spread by 9 to 14 times under heterogeneity)? Does a
  location's estimate tell that location whether its own cases are rising (yes,
  agreeing 99.8% of the time or better everywhere, against 47 to 74% agreement
  with whether that location's own outward R exceeds 1)? The older
  `fig_location_discrepancy` is kept alongside as the raw-distance view.
- Estimator error summary. `fig_estimator_error_summary.pdf`
  (`plot_estimator_error_summary`) is the visual companion to those tables: a
  dumbbell per aggregation scheme showing where its mean relative error sits with
  homogeneous transmissibility and where it moves to with heterogeneous
  transmissibility, for all three geographies and both periods. The x-axis is
  logarithmic, so connector length reads as a ratio rather than a difference.
- Estimator error tables. `out_figs/tables/estimator_error_{growth,full}.csv` and
  the matching `.tex` files (`mobility_rt/error_tables.py`) give the mean relative
  error of all nine candidate network-level summaries against ρ(R(t)), for all six
  scenarios, over the growth phase (ρ > 1) and over the full period. The nine are
  four weightings (arithmetic mean, population, incidence, relative R) of the
  per-location independent estimates, the same four weightings of the model's
  outward reproduction numbers, and the single aggregate-incidence estimate.
  Two conventions make the comparison fair.
  First, each scheme is compared with the reference matched to its own smoothing: a
  sliding-window (Cori/EpiEstim) estimate summarises the preceding 7 days, so it is
  scored against the matching 7-day trailing mean of ρ(R(t)), while schemes carrying
  no window are scored against ρ(R(t)) itself. ρ oscillates with the day-of-week
  commuting cycle, so scoring a smoothed estimate against an unsmoothed reference
  charges it for smoothing and inflates its error several-fold.
  Second, all schemes are scored on an identical set of days within each scenario.
  The independent estimators use the Cori/EpiEstim defaults (7-day window,
  Gamma(1, 0.2) prior) plus a single observability floor: an estimate is reported
  only where the window holds at least one case. One case is a unit rather than a
  tuned parameter. It is needed because the deterministic solver decays to fractions
  of a case, which no surveillance system could record, and where it does the
  posterior collapses onto the prior mean of 5. Any larger floor would be arbitrary,
  and a larger floor also makes the reported error a function of its own value
  rather than of the aggregation scheme, varying roughly 30-fold between 1 and 50
  cases. A network aggregate is reported only when every location has an estimate,
  so the aggregate has a fixed composition and differences between the weighting
  schemes reflect the weights rather than which locations were in the average.
- Generation time under spatial heterogeneity.
  `out_figs/tables/gt_spatial_heterogeneity.csv` and the matching `.tex`
  (`mobility_rt/gt_spatial.py`) separate the two things that "the generation time
  varies across space" can mean, for all six scenarios. The forward generation time
  built into the kernel does not vary: the contact multiplier `c[l]` and the
  transmissibility multiplier `m[k]` carry no dependence on infection age, so they
  cancel in g = K/R exactly as the susceptible pool and the mobility term do, and the
  table reports the departure from p(a_E)/∫p over every location pair at three
  epidemic phases as a measured quantity (it is at machine precision). The realised,
  backward interval does vary: it reweights p(a_E) by each location's own
  infection-age composition, so it runs short while a location is growing and long
  while it is declining, and locations are not synchronised. The table gives its
  network mean and its across-location spread at the early, peak and late snapshots
  and at its maximum. Making the forward generation time genuinely location-specific
  would need a location-specific profile SHAPE p_k(a_E), which this model does not
  have. The companion figure is `out_figs/SI/fig_SI_gt_heterogeneity.pdf`.
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

> Note: in the λ_B/λ_W and generation-time sweeps, β is held fixed (§4b–4c), so R₀
> rises with the sweep and the early growth rate is pinned to the target doubling time
> by construction. The in-notebook comments state this explicitly.

---

## Tests

```bash
python3 test_mobility_rt.py
```

The suite has 148 tests across eleven classes. `TestMobilityRt` checks the model
itself: the row-stochasticity of the mobility matrices, the eigenvector and
stable-distribution identities, the sensitivity and elasticity relations, the
Euler–Lotka relation, the double-censored discretisation (it recovers the target
mean), and the universal generation-time property (g is identical across all location
pairs, times, and day-of-week mobility). The remaining classes cover spatial
heterogeneity, the error and generation-time tables, the location-level discrepancy
analysis, and the corrected-figure scenarios.

`TestFrameworkCoreConsistency` is the drift guard. `mobility_rt.framework` carries its
own copy of twenty core functions so that it can stand alone as the general-purpose
API, and this class pins every one of them to its core counterpart — including
bit-identical output from the two simulators (`atol=0`) and the argument mapping
(`prob_peak=1.0`) that translates between the two signatures. It also introspects the
framework module, so a *new* function appearing there fails the suite until it is
either registered as a guarded duplicate or declared framework-only. Adding that guard
turned up one real inconsistency: the framework was evaluating `R^l_meeting(0)` at the
post-seed susceptible state while every other `t = 0` quantity beside it used the
fully susceptible state.

---

## License

For the Author Accepted Manuscript, a CC BY public copyright licence applies (see the
manuscript's copyright statement).
