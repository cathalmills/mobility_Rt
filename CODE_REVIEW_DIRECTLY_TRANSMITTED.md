# Code review: `directly_transmitted.py` workflow

This document records issues found in the mathematical workflow, numerical implementation, naming consistency with the manuscript, and figure generation. It is intended to accompany `mobility_Rt_workflow_corrected.py`, which implements the same science with corrections and updated figures.

---

## 1. Notation and indexing conventions

### 1.1 Mobility superscripts vs subscripts

Throughout the code and several figures, mobility is written as `f_{jk}` or `\bar{f}_{jk}`. In the manuscript convention you cited, flows from residence `j` toward activity location `k` should appear as **`f^{jk}(t)`** (superscripts). The implementation stores `f_jk[t, j, k]` consistent with **row `j`, column `k` = `f^{jk}`**; only the *printed* LaTeX in plots and comments uses subscripts. This is a **presentation / manuscript-alignment** issue, not a silent indexing bug, but it should be fixed everywhere in axis labels, colorbar titles, and docstrings.

### 1.2 Next-generation matrix orientation

The code defines `R_mat[k, j]` as infector `k` → infectee `j` (see `compute_R_matrix` and `_bar3d_Rkj`). That matches the comment on `base_K[k, j]`. **Figure 3 panel B** labels the vertical axis “Infector `j`” while the quantity plotted is `R^j_{\mathrm{out}}(t) = \sum_a R_{ja}(t)`, i.e. the **row sum over infectees** for **infector `j`**. The label should read **infector `j`** only if the matrix were transposed; with the code’s convention the heatmap rows are infector indices **`k`** (or `j` if you rename consistently, but then the symbol in the title `R^j_{\mathrm{out}}` must match the row dimension). This is a **label/notation mismatch**, not a wrong array.

---

## 2. Elasticity and sensitivity (confirmed bug in Figure 2)

### 2.1 Infector marginal elasticity

`sensitivity_elasticity` returns `E_m[k, j]` proportional to `\partial \rho / \partial R_{kj}` in the usual left/right eigenvector construction. Then:

- `\sum_j E_{kj}` aggregates over **infectees** for fixed **infector `k`**.

In **`plot_fig2` panel F**, the code does:

```python
elas[t] = sensitivity_elasticity(R_mats[t])["elasticity"].sum(axis=1)
```

So each row index is **infector `k`**. The **y-axis label** was set to “Infector `j`” — **wrong index letter**. It should be **“Infector `k`”** (or “Infector (row)”).

### 2.2 Missing infectee marginal in Figure 2

The manuscript’s infectee elasticity marginal sums **over source locations `k`**:

\[
\varepsilon^{\cdot j}(t) = \sum_k \varepsilon_{kj}(t).
\]

That is **`elasticity.sum(axis=0)`** for each `t`. The original Figure 2 did not show this panel; the corrected workflow adds it (new panel I in the 3×3 layout).

### 2.3 `spectral_analysis`: eigenvectors for non-symmetric `R`

For nonnegative `R`, Perron–Frobenius gives a simple dominant eigenvalue and strictly positive eigenvectors in exact arithmetic. The code uses `np.linalg.eig` / `eig` on `R` and `R.T`, takes the eigenvector of largest **modulus**, then **`np.abs(...).real`** and **renormalizes to sum 1**. That discards any phase and enforces positivity; it is **reasonable for nonnegative `R`**, but:

- If `R` were ever nearly reducible / numerically defective, `eig` could return complex pairs and the “largest modulus” choice could be fragile.
- **Condition number** in `sensitivity_elasticity` uses `np.linalg.cond(R_mat)` (2-norm), while “reactivity” uses the **largest singular value** — different notions; fine if documented, confusing if read as one object.

### 2.4 Naming aliases in `spectral_analysis`

The dict keys `stable_distribution` vs `reprod_value` are documented in comments as Diekmann-style aliases, but the assignment (`stable_distribution`: `v` from `R^T`) is easy to misread against textbook notation. **Recommend** a single pair of names, e.g. `left_eigvec_dominant` and `right_eigvec_dominant`, and drop ambiguous aliases from the return dict or document them in one place only.

---

## 3. Mobility generator `generate_mobility`

### 3.1 Docstring vs implementation (day-of-week)

The **docstring** (lines ~188–190) states Mon–Thu = 1.00, Fri = 0.95, Sat = 0.90, Sun = 0.75. An **inline comment** (line ~217) mentions older values (“Fri 0.9, Sat 0.6, Sun 0.5”) that **do not match** the actual array `dow_scale = [1,1,1,1,0.95,0.90,0.75]`. Readers comparing code to comments will be misled.

### 3.2 Row-stochastic repair

After scaling commuting by a noisy factor, the code resets the diagonal so each row sums to 1 and renormalizes. That is coherent. Extreme clipping (`scaled_away` up to 0.95) can **erase** the intended lognormal variability when `away_base` is large; worth noting as a **modelling choice**, not a bug.

---

## 4. Kernel and meeting-time construction (`_kernel_base`, `compute_R_meeting`)

### 4.1 Within / between split

`bK_between = lb * np.maximum(D.T - within, 0.0)` guards against tiny negative numerical noise when subtracting the at-home meeting component from the full co-location sum. **Mathematically** `D^T - within` should already be nonnegative for genuine mobility matrices; the `maximum(..., 0)` is a **numerical hygiene** step.

### 4.2 `compute_effective_populations_series` vs notation

The docstring uses `f_{jl}` while the tensor is `f[j, l]` = **`f^{jl}`** in manuscript language. The convolution for `I_eff` uses `gen_time_pmf[s] * incidence[t-s]` with `s` from `1` to `min(max_days, t)` — **consistent with skipping `s=0`** if generation-time mass at day 0 is zero (true for discretised gamma starting at day bins `[0,1)` etc.). If `p(1)` is the first **positive** day of infectiousness, align `infect_profile` indexing with `E_pde[t, :, 0]` as the boundary mass at infection age 0 (see §5).

---

## 5. Forward “PDE” / renewal simulation (`simulate_epidemic_pde`)

### 5.1 Upwind shift and boundary

The transport `E(t, a) <- E(t-1, a-1)` with boundary data at `a=0` matches a **method-of-lines** discretisation of `\partial_t E + \partial_a E = 0` with influx at `a=0`. The claim that this is “algebraically identical” to the discrete renewal equation is **correct** provided the **same** discretisation of the convolution appears in the boundary (it does via `wE`).

### 5.2 Convolution indexing in `wE`

The code sets:

```python
wE = prob_peak * (E_pde[t, :, 1:] @ infect_profile[1:])
```

So infection **age 0** (`E[:,0]`) does **not** contribute to `wE` on the same day. That is correct if **`infect_profile[0]` is zero** (typical for a serial interval / generation-time PMF starting after exposure). Here `infect_profile = gen_time_pmf` from a gamma discretisation: the **first bin** can be **small but positive**. Then **infectiousness at age 0 is omitted** from the force of infection until the next time step’s shift moves that mass into `a \geq 1`. That introduces a **one-day lag** relative to a formulation that includes `p(0)` in the same-day kernel. For your discretisation this is likely **negligible** if `p(0)\approx 0`, but it is a **modelling detail** worth stating explicitly in the manuscript supplement.

### 5.3 Demographics and `populations` in `base_K`

`base_K` is built with **`populations` (initial N_j)**, not time-varying totals, inside `_kernel_base` via `N_eff = f^T N`. The simulation updates **`S`** but does not reflow `N_eff` with births/deaths into `N_j` for the denominator. With tiny birth/death rates this is **second order**; strictly, **`N_eff` should track total living population per location** if demographics matter.

### 5.4 Pairwise incidence allocation

New infections at `j` are split across `k` proportionally to `contrib_{kj}`. If `col_sum[j]` is tiny, the loop skips; **mass is conserved** in the normal case. Good.

---

## 6. Generation-time routines (`compute_generation_times`)

Under the **single infectiousness profile** `p(a_E)` scaled across all pairs, **`g_{kj}` collapses to `p / \sum p`** when `R_{kj}>0`. The implementation’s pairwise loops are therefore **redundant but correct**. The meeting-location and network aggregates inherit the same shape. **No bug** — but the nested Python loops are **O(N² × max_days)** per call; fine for `N=10`, poor for large `N`.

---

## 7. Type reproduction numbers (`type_reproduction_number`)

### 7.1 Formula

The scalar next-generation construction `T_j = R_{jj} + R_{jJ}(I-R_{JJ})^{-1}R_{Jj}` matches the **standard group / type reproduction** algebra when `R` is the **small-domain** NGM with the indexing used here.

### 7.2 Code quality

The function mixes indentation widths; **`py_compile` succeeds** but a **SyntaxWarning** is emitted for `\` inside a docstring (`\{j}` should be `\\{j}` or a raw docstring). Cleaning this improves maintainability.

---

## 8. Independent R estimator (`estimate_R_independent`)

Gamma(1,5) prior on `R` is implemented as `(a + \sum I)/(b + \sum \Lambda)` with `a=1, b=5`. The infectiousness denominator uses `gen_time[a-1]` for lag `a` — **consistent with 0-based PMF**. Good.

---

## 9. Figure-specific issues (original `plot_fig2`, `plot_fig3`, `plot_fig4`)

### 9.1 Figure 2 layout vs requested manuscript panels

Original layout is **2×3** with **mean `f`**, **diagonal home fraction** (not full `f^{jk}(t)` tensor), incidence, `S_{\mathrm{eff}}`, `R(t)`/`E(t)`, and one elasticity heatmap. The user’s updated layout requires a **3×3** grid including **network sketch**, **full mobility over time**, **activity prevalence**, **infectee elasticity**, and corrected axis labels.

### 9.2 Figure 3

- **Panel letter collision**: row-0 panel “D” is 3D `R_{kj}` while row-1 “F” is source–sink; letters are unique but the **docstring’s “Panels (3×3 grid…”** does not match the actual **`GridSpec(2, 5)`** (2×5) layout — **documentation drift**.
- **Panel I** title says `R^j_{\mathrm{type}}(t)` while the computed quantity is **`T_j(t)`** from `type_reproduction_numbers` — suggest harmonising notation with the manuscript.
- **Figure 3 panel B** y-label “Infector `j`” vs `R^j_{\mathrm{out}}` — same convention issue as §1.2.

### 9.3 Figure 4

- **Panel A**: twin axis `ax2` for day-of-week pattern can **crowd** panel B at fixed `wspace`; needs **wider gutter** or **spine position** adjustment.
- **Panel D** (eigenvalue magnitudes near 1.2–1.35): Matplotlib may still enable **offset/scientific** tick formatting; should force **plain** formatter / fixed decimals.
- **Panel F** (within-fraction): the **right-hand twin** for `\bar\pi(t)` can trigger **scientific notation** on a **0–1** scale depending on margins; set explicit formatter.

---

## 10. Ancillary / auxiliary issues

### 10.1 `plot_SI0_population` (Scenario B colours)

`type_colors_B` uses keys `"town"` etc., but **`zambia` scenario types** are `"capital"`, `"peri-capital"`, … — many nodes fall through to **grey**. Cosmetic / legend accuracy issue.

### 10.2 `minimum_control_effort`

The bisection scales **`R_cur[idx, :]`** (entire row) by `(1-u)`, i.e. removes **all outgoing transmission** from `idx` proportionally. That is a **specific control mechanism** (row scaling), not necessarily “effort at location `idx`” in the most general sense. Fine if that is the intended intervention; otherwise clarify.

### 10.3 Warnings globally suppressed

`warnings.filterwarnings("ignore")` hides useful numerical warnings (ill-conditioned eigenproblems, deprecated APIs). Prefer **scoped** filters.

---

## 11. Summary table

| Area                         | Severity   | Issue |
|-----------------------------|------------|--------|
| Fig 2 elasticity y-axis     | **Bug**    | Wrong infector index label (`j` vs `k`). |
| Manuscript notation         | Medium     | Subscript `f_{jk}` vs superscript `f^{jk}` in labels. |
| Fig 3 R_out heatmap label   | Medium     | Row index vs title `R^j_{\mathrm{out}}` consistency. |
| `generate_mobility` comments| Low        | Stale DoW comment vs code. |
| `wE` convolution age 0    | Low/theory | Skips `a=0` contribution; OK if `p(0)\approx 0`. |
| `N_eff` uses fixed `N`     | Low        | Ignores demographic change in denominator. |
| Fig 4 formatters / layout   | Medium     | Twin-axis crowding; tick scientific notation. |
| `spectral_analysis` aliases | Low        | Confusing key names for eigenvectors. |

---

## 12. Relation to `mobility_Rt_workflow_corrected.py`

That module:

- Keeps the **validated numerical core** by importing from `directly_transmitted`.
- Fixes **figure semantics** (elasticity marginals, axis labels, manuscript superscripts on `f^{jk}`).
- Adds **activity-location prevalence** \(\sum_a \sum_j f^{jl}(t) E_j(t,a)\).
- Rebuilds **Figure 2** (3×3) and **Figure 3** (3×4) per your specification, plus **contour** variants.
- Patches **Figure 4** layout and tick formatting as above.

No change to the **mathematical definition** of `R`, `ρ`, or the upwind scheme was required beyond the clarifications in this review; the largest **actionable** error affecting interpretation was the **elasticity panel axis labelling** and missing **infectee marginal** view in Figure 2.

---

## 13. Deliverables (this repository)

| Artifact | Path |
|----------|------|
| Detailed review (this file) | `CODE_REVIEW_DIRECTLY_TRANSMITTED.md` |
| Corrected workflow + figures | `mobility_Rt_workflow_corrected.py` |
| Generated PDFs (run the script) | `out_figs/corrected/` |

After `python3 mobility_Rt_workflow_corrected.py`, expect at least:

- `fig_02_overview.pdf`, `fig_02_overview_surface.pdf` — dense-urban 3×3 overview (2D tiles vs **3D surfaces**).
- `fig_SI_static_02_overview.pdf`, `fig_SI_static_02_overview_surface.pdf` — static-mobility counterparts.
- Panel C is **$f^{jj}(t)$** (home fraction), not all off-diagonal flows.
- `fig_03_taxonomy.pdf`, `fig_03_taxonomy_surface.pdf` — 3×4 taxonomy (tiles vs 3D surfaces; G/H use bar charts in tile mode).
- `fig_SI_static_03_taxonomy.pdf`, `fig_SI_static_03_taxonomy_surface.pdf`.
- Notation aligned with manuscript: $R^{kj}$, $E^j(t,0)$, $E^{kj}$, $v_j$ / $v^{*}_j$, $\varepsilon^{kj}$, $\varepsilon^k_{\mathrm{out}}$, $\varepsilon^j_{\mathrm{in}}$.
- `fig_04_spectral.pdf`, `fig_SI_static_04_spectral.pdf` — spectral figure with gutter and non-scientific y-ticks where appropriate.

The new module **imports** `directly_transmitted` for all dynamical and matrix algebra; it only replaces **figure construction** and adds the **activity prevalence** aggregate. Figure 3 panel **D** uses a **scatter of time-mean infector vs infectee marginals** (joint “ranking geometry”); if you prefer two separate horizontal rank bar charts, that can be swapped in without changing the simulation.
