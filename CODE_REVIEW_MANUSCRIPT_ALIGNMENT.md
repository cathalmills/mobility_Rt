# Code ↔ Manuscript alignment review

**Scope.** Equation‑by‑equation verification of the numerical core in
`directly_transmitted.py` and the figure pipeline in
`mobility_Rt_workflow_corrected.py` against the manuscript
*"Multi‑scale measures of time‑varying epidemic spread on human mobility networks"*
(`mobility_Rt-51.pdf`, 25 June 2026). Method: 5 independent equation‑grounded
verifiers + manual tracing of the kernel, type‑reproduction, calibration, and
estimator code. **This is a review of planned changes — no code has been changed.**

**Code convention (verified, intentional, not a bug).** The code's matrix
`R_mat[k,j]` has rows = infector *k*, columns = infectee *j* — the *transpose*
of the manuscript's `R(t)` (Eq 22, rows = infectee). Because ρ and singular
values are transpose‑invariant, every derived quantity remains correct; this was
checked explicitly for `R_out`/`R_in`, the eigenvector roles, the type‑R block
formula, and the ℓ¹ amplification norm.

---

## A. Verified CORRECT (no change)

These were checked directly against the stated equations and **match**:

| Manuscript | Code | Status |
|---|---|---|
| Eq 7  N^l_eff=Σ_q f^{ql}N^q; λ^{kl}=χ^{kl}/N^l_eff·p | `_kernel_base` (N_eff=fᵀ@pop; 1/N_eff at meeting loc l) | ✓ |
| Eq 12 K^{kj}=Σ_l f^{jl}S^j f^{kl}λ^{kl}; within=l=k | `_kernel_base` within/between split, S^j applied in `compute_R_matrix` | ✓ |
| Eq 14/15/44/23 R^{kj}, R^k_out, R^j_in, ρ(R) | `compute_R_matrix`, `R_outward`=axis1, `R_inward`=axis0, `R_system` | ✓ |
| Eq 48/49 R^l_meeting, S^l_eff=Σ_j f^{jl}S^j | `compute_R_meeting` (S_eff=fᵀ@S, correct sum over j) | ✓ |
| Eq 16/17 g^{kj}=K/R → p(a)/Σp (universal) | `compute_generation_times` (all collapse to p/Σp) | ✓ |
| Eq 25/26 right=stable v*, left=reproductive v | `spectral_analysis` (transpose‑aware labelling) | ✓ |
| Eq 36 s=|λ₂|/R, damping 1/s | `spectral_analysis` mixing_ratio/damping_ratio | ✓ |
| **Eq 37 σ=‖R‖₂ (largest singular value)** | `reactivity` = `svd(R)[0]` | ✓ **(your Q1)** |
| Eq 38 κ=‖v‖‖v*‖/|v·v*| | `_cond_eigvec`, `sensitivity_elasticity` (eigenvalue cond. #) | ✓ |
| Eq 39/40 A(n)=‖Rⁿ‖₂, A₁(n)=‖Rⁿ‖₁ | svd of Rⁿ; max **row** sum (correct given transpose) | ✓ |
| Eq 29‑35 s^{kj}=v_j v*_k/(v·v*), ε, ε_out, ε_in | `sensitivity_elasticity` | ✓ |
| **ε^k_out(m)=ε^j_in(m)** (Fig 2/3 claim) | marginals come out exactly equal (numerically verified) | ✓ |
| Eq 52/53/54 type‑R, ρ(R_JJ)<1 guard | `type_reproduction_number` == `..._group([j])` (verified equal) | ✓ |
| Eq 55 group type‑R | `type_reproduction_number_group` (transpose‑invariant) | ✓ |
| Eq 41/42 X(α,t), E(t)=Σ(R_out²)/Σ(R_out) | power‑mean spectrum + `E_t` at **all** call sites | ✓ |
| Eq 4/6/8/18 PDE + renewal BC, upwind Δt=Δa=1 | `simulate_epidemic_pde` (deterministic exact BC) | ✓ |
| R(0)=R0=1.5 calibration | `simulate_epidemic_pde` rescales λ so ρ(R(0))=1.5 | ✓ |
| Gamma(5.5,1.8) trunc 25; λ_B=0.30λ_W; 2/3/3/2 nodes; c_j=40/35/28/18% | `COVID_PARAMS`, `generate_city`, `discretise_gamma` | ✓ |
| Day‑of‑week weekly scaling applied + rows stay stochastic | `generate_mobility` (`dow_scale[t%7]`, renormalised) | ✓ |
| R_naive (pooled Cori), R_pw (population‑weighted) | `compute_naive_R_aggregate`, `out_pw`/`ind_pw` | ✓ |

---

## B. Planned change — CODE (results‑affecting; needs your go‑ahead)

### B1 — Independent‑estimator weighting `R_ind,RW` (Fig 4 / `fig_main_bias_combined`)  · severity HIGH
- **Where:** `directly_transmitted.py:4156‑4158` (`_compute_naive_suite`, key `ind_Rw`).
- **Now:** `ind_Rw[t] = Σ_j (R^out_j · R^ind_j) / Σ_j R^out_j` — weights the
  independent estimates by the **model outward R**.
- **Manuscript §3.2.2 + Fig 4 caption:** "weights independent, per‑location
  estimates R^ind_j(t) by their relative transmissibility
  (i.e. R^ind_j(t)/Σ_k R^ind_k(t))" → `R_ind,RW = Σ_j (R^ind_j)² / Σ_j R^ind_j`.
- **Why it matters:** the orange `R^j_{ind,Rw}(t)` trace in the main bias figure
  (and the SI naive panels) is currently computed with the wrong weights; the
  figure does not match its own caption / the Parag et al. [68] method it cites.
- **Planned fix:**
  ```python
  # iii) relative-transmissibility (self-)weighted independent estimate
  if Rv.sum() > 1e-6:
      ind_Rw[t] = float(np.sum(Rv * Rv) / Rv.sum())   # Σ(R_ind²)/Σ(R_ind)
  ```
- **Impact:** regenerates Fig 4 / `fig_main_bias_combined` (all 3 scenarios) and
  the dependent SI naive panels. Qualitative story (independent weighting is
  biased vs ρ(R)) is expected to persist; the numeric bias/MAE/MSE annotations
  will change. **Recommend applying.**

---

## C. Manuscript text / caption issues (author decision — not code)

### C1 — Fig 3 caption B/C order contradicts body text and the figure · severity MED
- **Caption (p22):** "B) and C) are the **inward and outward** reproduction
  numbers respectively."
- **Body text (p21):** "The largest values of **R^j_out(t) and R^j_in(t)** …
  (Figure 3 B and C)."
- **Figure (code):** panel B = `R_out` ("Infector k"), panel C = `R_in`
  ("Infectee j") → **B = outward, C = inward**, matching the body text.
- **Fix:** correct the caption to "B) and C) are the **outward and inward**
  reproduction numbers respectively."

### C2 — Fig 3 caption letter/layout drift · severity LOW
- The caption reuses "F)" twice and refers to a **source‑sink** panel and a
  "G) and H) … R^{kj} and E^{kj}" arrangement that does not match the current
  `mobility_Rt_workflow_corrected.py` `plot_fig03_taxonomy` layout
  (A GT, B R_out, C R_in, D elasticity scatter, E v, F v*, G R^{kj}, H E^{kj},
  I/J independent comparisons, K type‑R, L meeting). Reconcile caption ↔ figure
  (the figure is internally correct; the caption describes an older layout).

### C3 — Fig 2 panel E label · severity LOW
- Panel E plots `Σ_a Σ_j f^{jl} E^j(t,a)` = **E^l_eff(t)** (effective infected
  population, the manuscript's named quantity, p15). The figure colour‑bar shows
  the raw double sum. Optional: relabel the colour‑bar title `E^l_{eff}(t)` to
  match the caption wording ("Effective infected population in location l").

---

## D. Documentation / comment fixes (cosmetic, zero numerical effect)

| # | Where | Issue | Fix |
|---|---|---|---|
| D1 | `directly_transmitted.py:35,6955` | `base_contact_rate = 13.0` | manuscript states **13.03** (POLYMOD). Harmless (rescaled by calibration) but inconsistent with text → set `13.03`. |
| D2 | `directly_transmitted.py:~270` | stale inline DoW comment "Fri 0.9, Sat 0.6, Sun 0.5" | active array is `[…,0.95,0.90,0.75]`; fix the comment. |
| D3 | `directly_transmitted.py:817` | `reactivity` docstring cites "Eq. 35" | should be **Eq 37**. |
| D4 | `directly_transmitted.py:513,583` | type‑R docstrings cite "[Eq 45]"/"[Eq 47]" | should be **Eq 53 / Eq 55**. |
| D5 | `directly_transmitted.py:644‑645` | `simulate` docstring cites "Eq 2 / Eq 4" | PDE is **Eq 4**, BC/renewal **Eq 6/8/18**. |
| D6 | `directly_transmitted.py:~7100` vs `:7109` | Scenario‑C comment says `LB_C=LW×0.005`; code uses `LW×0.3` | **figure is valid** (verified σ/ρ≈1.60 > the ~1.3 needed, 31‑day false‑action zone); just update the stale comment to `0.3`. |

---

## E. Optional gaps / clarifications

### E1 — η(t) (Eq 45) not implemented · severity LOW (missing feature, not a bug)
- Eq 45 defines η(t)=CV(R^j_out)/CV(R^j_in). Only `CV(R^j_out)` (`cv_row_sums`)
  is computed/plotted. If η is meant to be a reported output, add
  `eta = cv(R_out)/cv(R_in)`; otherwise no action.

### E2 — `estimate_R_independent` vs Eq 56 · severity LOW (consistent, clarify text)
- Eq 56 is written as the idealised per‑day ratio
  `R^ind_j = E^j(t,0)/Σ_a g^j(a)E^j(t−a,0)`. The code
  (`directly_transmitted.py:739‑762`, and the `R_loc` loop at `:4122‑4134`)
  uses the **Cori et al. [13] windowed estimator** (7‑day window, Gamma(1,5)
  prior) — which is exactly the "estimator often used for a single closed
  population (e.g. [13])" the text cites. **Not a bug.** Optional: state the
  window/prior in Methods so Eq 56 ↔ implementation is explicit.

### E3 — Shared mobility for S and E · severity LOW (stated simplification)
- Eq 6 allows `f^{jl}(t,S) ≠ f^{jl}(t,E)`. The implementation uses one `f` for
  both (a simplification the manuscript explicitly permits). No change needed;
  worth a one‑line note in Methods if not already present.

---

## CHANGES APPLIED — 2026-06-29

All edits below are in `directly_transmitted.py` unless noted. Verified `ast.parse` OK.

**B1 — independent-estimator weighting (results-affecting; main bias figure).**
`_compute_naive_suite`, `ind_Rw`. Changed the weights from the model outward R to
the independent estimate's own relative transmissibility, per §3.2.2 / Fig 4:
- before: `ind_Rw[t] = Σ_j (R^out_j · R^ind_j) / Σ_j R^out_j`
- after:  `ind_Rw[t] = Σ_j (R^ind_j)² / Σ_j R^ind_j`   (weights `w_j = R^ind_j/Σ_k R^ind_k`)
Removed the now-unused `Ov = R_out_t[t, vj]` in that loop. Regenerates
`out_figs/main/fig_main_bias_combined.pdf` (orange `R^j_{ind,Rw}` trace + the col-1
bias/MAE/MSE annotations in all three scenario rows). Other panels unchanged.

**Panel-F legend relocation (both fig_04 pipelines)** — `directly_transmitted.py`
`plot_fig4` and `mobility_Rt_workflow_corrected.py` `plot_fig04_spectral`: the
π̄(t) legend now sits in the gutter between the heatmap's right-hand y-axis
(+colorbar) and the right-hand bar panel (placed on `ax_r`,
`loc="center right", bbox_to_anchor=(-0.30, 0.55)`), with a slightly wider gutter
(`wspace` 0.42/0.46 → 0.9). It no longer overlaps the right y-axis.

**Cosmetic / documentation (zero numerical effect):**
- D1 `COVID_PARAMS["base_contact_rate"]` 13.0 → **13.03** (+ matching comment in `main`).
  No numerical change: calibration rescales λ so ρ(R(0))=1.5 regardless.
- D2 `generate_mobility` day-of-week comment "Fri 0.9, Sat 0.6, Sun 0.5" →
  "Fri 0.95, Sat 0.90, Sun 0.75" (matches the active `dow_scale` array).
- D3 `reactivity` docstring "Eq. 35" → **"Eq. 37"**.
- D4 `type_reproduction_number` docstring "[Eq 45]" → **"[Eq 53]"**;
  `type_reproduction_number_group` "[Eq 47]" → **"[Eq 55]"**.
- D5 `simulate_epidemic_pde` docstring "[Eq 2]"/"Eq 4" → PDE **[Eq 4]**, BC **Eq 6 (renewal 8/18)**.
- D6 Scenario-C design comment "LB_C = LW × 0.005" → "LW × 0.3 (see below)" — matches
  the actual `LB_C = LW*0.3`; verified σ/ρ≈1.60 > 1.3 so the false-action zone is valid.

**Not changed (by design / your earlier instruction):** C1–C3 are manuscript caption
edits (author); E1 (η Eq 45) and E2/E3 (Cori-windowed estimator; shared S/E mobility)
are consistent-with-text non-bugs left as-is.

## Recommended order of execution
1. **B1** (Fig 4 weighting) — the one substantive code fix; then regenerate
   `fig_main_bias_combined` + SI naive panels.
2. **D1–D6** cosmetic code/comment fixes (batch; no re‑run needed except whatever
   you regenerate anyway).
3. **C1–C3** caption edits in the manuscript (author).
4. **E1–E3** optional.
