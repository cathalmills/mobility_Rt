"""mobility_rt.scripts.run_all — generate every manuscript + SI figure (former main())."""
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
from mobility_rt.config import COVID_PARAMS
from mobility_rt.distributions import discretise_gamma
from mobility_rt.estimators import estimate_R_independent
from mobility_rt.generation_time import compute_generation_times
from mobility_rt.geometry import generate_city
from mobility_rt.kernel import R_inward, R_system, _kernel_base, compute_R_matrix
from mobility_rt.mobility import generate_mobility
from mobility_rt.plotting.animations import _precompute_anim_data, plot_animation_R_heatmap, plot_animation_elasticity, plot_animation_network_metric, plot_animation_scenario_comparison
from mobility_rt.plotting.figures_main import plot_counterfactual_nonnormal, plot_fig2, plot_fig3, plot_fig4, plot_fig5, plot_fig_C_transience, plot_main_bias_figure, plot_naive_R_comparison_suite, plot_power_mean_spectrum, plot_type_repro
from mobility_rt.plotting.figures_si import plot_3d_surfaces, plot_SI0_population, plot_SI1, plot_SI2, plot_SI5_combined, plot_SI_3d_earlypeak, plot_SI_R_comparison, plot_SI_epi_params, plot_SI_gt_spatial, plot_SI_gt_varying_beta, plot_SI_lambda_decomposition, plot_SI_pde_convergence, plot_SI_sensitivity, plot_elasticity_surfaces, plot_fig7
from mobility_rt.plotting.style import _set_pub_style
from mobility_rt.simulation import simulate_epidemic_pde
from mobility_rt.spectral import reactivity, spectral_analysis
from mobility_rt.transmissibility import minimum_control_effort
from mobility_rt.type_reproduction import type_reproduction_numbers


def main():
    print("=" * 72)
    print("MOBILITY-INFORMED RENEWAL EQUATIONS — DIRECTLY TRANSMITTED DISEASES")
    print("=" * 72)

    N_LOC    = 10
    T        = 365
    SEED     = 42
    params   = COVID_PARAMS
    max_days = params["max_gen_time"]
    # λ_within: per-contact transmission rate at home location = POLYMOD community rate
    # λ_between: inter-location contacts ~30% as intense as home contacts
    #   Modelling assumption: LB/LW=0.30 reflects reduced contact intensity outside home
    BETWEEN_FRACTION = 0.30
    LW       = params["base_contact_rate"]          # 13.03 contacts/day [POLYMOD; Mossong et al. 2008]
    LB       = params["base_contact_rate"] * BETWEEN_FRACTION

    # ── infectiousness profile ─────────────────────────────────────────────
    gen_time_pmf   = discretise_gamma(params["gen_time_mean"],
                                       params["gen_time_sd"], max_days)
    infect_profile = gen_time_pmf.copy()   # w(a_E): surv=1 absorbed
    # PDF model: single infectiousness profile p(a_E) for all location pairs.
    # GT shape is universal: g_{kj}(t,a_E) = p(a_E)/∫p for all (k,j) and t.
    # κ^{kl} variation (lw/lb) affects R_{kj} magnitudes, not GT shapes.
    p_aE      = infect_profile  # alias for clarity in GT computations
    w_within  = infect_profile  # kept for plot backward-compat; now equal to p_aE
    w_between = infect_profile  # kept for plot backward-compat; now equal to p_aE
    print(f"\n  GT mean = {float(np.sum(np.arange(max_days)*gen_time_pmf)):.2f} d")

    # ── Scenario A: Dense urban ───────────────────────────────────────────
    print("\n─── Scenario A: Dense urban ───")
    city_A = generate_city(N_LOC, scenario="lagos", seed=SEED)
    coords_A, pops_A, dists_A, types_A, meta_A = city_A
    print(f"  Node types: {types_A}")
    print(f"  Pops: {pops_A.astype(int)}")
    print(f"  Commuting fracs: {np.round(meta_A['commuting_fracs'], 3)}")

    f_A, base_fA = generate_mobility(N_LOC, T, pops_A, dists_A, types_A, meta_A,
                                      day_variation_sd=0.15, seed=SEED)
    print(f"  Row-stochastic: {np.allclose(f_A.sum(axis=-1), 1.0)}")

    R0_A = compute_R_matrix(f_A[0], pops_A, pops_A,
                             params["prob_transmission_peak"], infect_profile, LW, LB)
    spec0_A = spectral_analysis(R0_A)
    print(f"  ρ(R₀) = {spec0_A['rho']:.4f}  s = {spec0_A['mixing_ratio']:.4f}")

    initial_A = np.zeros(N_LOC);  initial_A[0] = 10
    sim_A = simulate_epidemic_pde(
        T, N_LOC, pops_A, f_A,
        params["prob_transmission_peak"], infect_profile, max_days,
        params["R0_target"], initial_A, LW, LB, w_within, w_between,
        birth_rate=0.00003, death_rate=0.00003,
        stochastic=False, susceptible_depletion=True, seed=SEED)

    lw_A, lb_A = sim_A["lambda_within_scaled"], sim_A["lambda_between_scaled"]
    inc_A = sim_A["incidence"]
    pk_A  = int(inc_A.sum(axis=1).argmax())
    print(f"  Total infections: {inc_A.sum():.0f}  "
          f"Peak day: {pk_A}  "
          f"Attack rate: {inc_A.sum()/pops_A.sum()*100:.1f}%")

    # ── Scenario A-static: Dense urban with time-invariant (static) mobility ─
    # Same city geometry, populations, initial conditions and all epi params as
    # Scenario A, but f_{jk}(t) = base_fA for all t (no day-of-week scaling or
    # lognormal daily noise).  Used to produce SI figures comparing the effect
    # of time-varying vs static mobility on R(t), R-types, and spectral props.
    print("\n─── Scenario A-static: Dense urban (static mobility) ───")
    f_A_static = np.stack([base_fA] * T, axis=0)   # (T, N, N), row-stochastic
    print(f"  Row-stochastic: {np.allclose(f_A_static.sum(axis=-1), 1.0)}")
    sim_A_static = simulate_epidemic_pde(
        T, N_LOC, pops_A, f_A_static,
        params["prob_transmission_peak"], infect_profile, max_days,
        params["R0_target"], initial_A, LW, LB, w_within, w_between,
        birth_rate=0.00003, death_rate=0.00003,
        stochastic=False, susceptible_depletion=True, seed=SEED)
    inc_As   = sim_A_static["incidence"]
    pk_As    = int(inc_As.sum(axis=1).argmax())
    lw_As    = sim_A_static["lambda_within_scaled"]
    lb_As    = sim_A_static["lambda_between_scaled"]
    print(f"  Total infections: {inc_As.sum():.0f}  "
          f"Peak day: {pk_As}  "
          f"Attack rate: {inc_As.sum()/pops_A.sum()*100:.1f}%")

    # ── Scenario A-R012: Dense urban with R0=1.2 (for SI2 counterfactual) ──
    print("\n─── Scenario A-R012: Dense urban (R₀=1.2) ───")
    sim_A_R012 = simulate_epidemic_pde(
        T, N_LOC, pops_A, f_A,
        params["prob_transmission_peak"], infect_profile, max_days,
        1.2, initial_A, LW, LB, w_within, w_between,
        birth_rate=0.00003, death_rate=0.00003,
        stochastic=False, susceptible_depletion=True, seed=SEED)
    lw_A_R012 = sim_A_R012["lambda_within_scaled"]
    lb_A_R012 = sim_A_R012["lambda_between_scaled"]
    inc_A_R012 = sim_A_R012["incidence"]
    print(f"  Total infections: {inc_A_R012.sum():.0f}  "
          f"Peak day: {int(inc_A_R012.sum(axis=1).argmax())}  "
          f"Attack rate: {inc_A_R012.sum()/pops_A.sum()*100:.1f}%")

    # ── Scenario B: Zambia-like sparse ────────────────────────────────────
    print("\n─── Scenario B: Sparse national ───")
    city_B = generate_city(N_LOC, scenario="zambia", seed=SEED)
    coords_B, pops_B, dists_B, types_B, meta_B = city_B
    print(f"  Node types: {types_B}")
    print(f"  Commuting fracs: {np.round(meta_B['commuting_fracs'], 3)}")

    f_B, base_fB = generate_mobility(N_LOC, T, pops_B, dists_B, types_B, meta_B,
                                      day_variation_sd=0.12, seed=SEED)
    initial_B = np.zeros(N_LOC);  initial_B[0] = 10
    sim_B = simulate_epidemic_pde(
        T, N_LOC, pops_B, f_B,
        params["prob_transmission_peak"], infect_profile, max_days,
        params["R0_target"], initial_B, LW, LB, w_within, w_between,
        birth_rate=0.00003, death_rate=0.00003,
        stochastic=False, susceptible_depletion=True, seed=SEED)

    lw_B, lb_B = sim_B["lambda_within_scaled"], sim_B["lambda_between_scaled"]
    inc_B = sim_B["incidence"]
    print(f"  Attack rate (B): {inc_B.sum()/pops_B.sum()*100:.1f}%  "
          f"Peak day: {int(inc_B.sum(axis=1).argmax())}")

    # ── Zambia mobility validation ────────────────────────────────────────
    # Checks whether meaningful between-location transmission occurs despite
    # very low commuting fracs (1.5–8%) over large distances (~600 km).
    # With lw/N_eff per-capita rates, f[j→k] ~1–4% for rural nodes means
    # cross-location R_kj is ~1–4% of home-location R. Epidemics seeded
    # at the capital (loc 0) will predominantly burn locally; rural spread
    # depends on per-day flow × force of infection at the capital.
    print("\n─── Zambia mobility validation ───")
    off_diag_mask = ~np.eye(N_LOC, dtype=bool)
    off_vals_B    = base_fB[off_diag_mask]
    print(f"  Home fracs (diagonal): {np.diag(base_fB).round(3)}")
    print(f"  Off-diagonal f range:  [{off_vals_B.min():.6f}, {off_vals_B.max():.6f}]"
          f"  mean={off_vals_B.mean():.6f}")
    imat_B = sim_B["incidence_matrix"]
    between_B = sum(imat_B[t].sum() - imat_B[t].trace() for t in range(T))
    within_B  = sum(imat_B[t].trace() for t in range(T))
    total_B   = between_B + within_B
    print(f"  Between-location infections: {between_B:.1f}  "
          f"({between_B / (total_B + 1e-10) * 100:.2f}% of total)")
    print(f"  Within-location infections:  {within_B:.1f}  "
          f"({within_B  / (total_B + 1e-10) * 100:.2f}% of total)")
    print(f"  Interpretation: low commuting fracs + exponential distance"
          f" decay → valid sparse-network behaviour, not a bug.")

    # ── Scenario C: Mega-hub non-normal mobility (false-action zone) ─────
    print("\n─── Scenario C: Mega-hub non-normal (false-action zone) ───")
    # Design rationale — checked analytically before full simulation:
    #
    # For σ(t) > 1 while ρ(t) < 1 (false-action zone), R_mat must be highly
    # non-normal with strong upper-triangular structure: R[hub,sat] >> R[sat,hub].
    #
    # This requires f[sat,hub] >> f[hub,sat]:
    #   (a) ONE mega-hub (pops_C[0]=3M) + 9 small satellites (200k each).
    #       With hub_attraction_power=8, gravity weight ∝ N^8 concentrates
    #       ALL satellite commuting at the single hub: f[sat,hub] ≈ c_sat.
    #       Hub residents barely visit satellites: f[hub,sat] ≈ 0.
    #       → within-K term: R[hub,sat] = lw·f[sat,hub]·f[hub,hub]/N_eff[hub] (large)
    #                        R[sat,hub] = lw·f[hub,sat]·f[sat,sat]/N_eff[sat] ≈ 0
    #       → R_mat is near-upper-triangular: R[hub,*] large, R[sat,hub] ≈ 0.
    #   (b) LB_C = LW × 0.3 (see below): suppresses the symmetric between-K term
    #       (D[j,k]·D[k,j] cancels asymmetry); leaves within-K dominant.
    #   (c) Epidemic seeded at satellite node 1 (not hub) to route spread
    #       through the hub-to-satellite channel, making the asymmetry visible.
    #   (d) commuting_frac_scale=1.0 (do not inflate hub's commuting, which
    #       would reduce f[hub,hub] and weaken the within-K directional term).
    #
    # Verification: σ/ρ ≥ 1.5 at t=0 → false-action zone ≈ 30–50 days.
    # If σ/ρ < 1.3, a WARNING is printed so parameters can be adjusted.
    LB_C   = LW * 0.3
    pops_C = np.ones(N_LOC) * 200_000.0
    pops_C[0] = 3_000_000.0          # single mega-hub at node 0
    # city_C shares geometry with city_A (same coords/dists/types) but uses pops_C.
    # Rebuild the population-dependent representative indices for pops_C (mega-hub is
    # node 0) so representative_locs(city_C) is correct, rather than aliasing meta_A.
    meta_C = dict(meta_A)
    meta_C["hub_idx"]    = int(np.argmax(pops_C))
    meta_C["periph_idx"] = int(np.argmin(pops_C))
    _mid_c = [k for k in range(N_LOC) if k not in (meta_C["hub_idx"], meta_C["periph_idx"])]
    meta_C["mid_idx"]    = _mid_c[len(_mid_c) // 2] if _mid_c else 0
    city_C = (coords_A, pops_C, dists_A, types_A, meta_C)
    f_C, base_fC = generate_mobility(N_LOC, T, pops_C, dists_A, types_A, meta_C,
                                      day_variation_sd=0.15, seed=SEED+1,
                                      commuting_frac_scale=1.0,
                                      hub_attraction_power=8.0)
    initial_C = np.zeros(N_LOC);  initial_C[1] = 10   # seed at satellite node 1
    sim_C = simulate_epidemic_pde(
        T, N_LOC, pops_C, f_C,
        params["prob_transmission_peak"], infect_profile, max_days,
        params["R0_target"], initial_C, LW, LB_C, w_within, w_between,
        birth_rate=0.00003, death_rate=0.00003,
        stochastic=False, susceptible_depletion=True, seed=SEED)
    lw_C, lb_C = sim_C["lambda_within_scaled"], sim_C["lambda_between_scaled"]
    inc_C = sim_C["incidence"]
    pk_C  = int(inc_C.sum(axis=1).argmax())
    print(f"  Total infections: {inc_C.sum():.0f}  "
          f"Peak day: {pk_C}  "
          f"Attack rate: {inc_C.sum()/pops_C.sum()*100:.1f}%")
    R0_mat_C = compute_R_matrix(f_C[0], pops_C, pops_C,
                 params["prob_transmission_peak"], infect_profile, LW, LB_C)
    spec0_C  = spectral_analysis(R0_mat_C)
    react_C  = reactivity(R0_mat_C)
    _ratio_C = react_C["amplification_ratio"]
    print(f"  LB_C/LW={LB_C/LW:.4f}  hub_power=8.0  pops_C[0]=3M  → σ/ρ_C={_ratio_C:.4f}")
    if _ratio_C < 1.3:
        print("  WARNING: σ/ρ < 1.3 — false-action zone may not be clearly visible")

    # ── Independent R estimation ───────────────────────────────────────────
    print("\n─── Independent R̂(t) estimation ───")
    R_ind_A = estimate_R_independent(inc_A, gen_time_pmf, window=7)
    R_ind_C = estimate_R_independent(inc_C, gen_time_pmf, window=7)
    R_in_s  = np.array([R_inward(sim_A["R_matrices"][t]) for t in range(T)])
    for j in range(min(N_LOC, 4)):
        ind_m = float(np.nanmean(R_ind_A[30:60, j]))
        mob_m = float(np.nanmean(R_in_s[30:60, j]))
        bias  = (ind_m - mob_m) / mob_m * 100 if mob_m > 0 else np.nan
        print(f"    L{j+1}: Indep={ind_m:.3f}  R^in={mob_m:.3f}  Bias={bias:+.1f}%")

    # ── GT snapshots ──────────────────────────────────────────────────────
    print("\n─── GT snapshots ───")
    early_A = max(1, pk_A // 3);  late_A = min(T-1, pk_A+30)
    gt_snaps_A = {}
    for nm, day in [("early", early_A), ("peak", pk_A), ("late", late_A)]:
        gt_snaps_A[nm] = compute_generation_times(
            f_A[day], sim_A["susceptibles"][day], pops_A,
            params["prob_transmission_peak"], w_within, max_days, lw_A, lb_A)
        gn_mean = float(np.sum(np.arange(max_days) * gt_snaps_A[nm]["g_network"]))
        print(f"    {nm:5s} (d{day:3d}): g_network mean = {gn_mean:.4f} d")

    # ── GT snapshots for static scenario ─────────────────────────────────
    early_As = max(1, pk_As // 3);  late_As = min(T-1, pk_As+30)
    gt_snaps_As = {}
    for nm, day in [("early", early_As), ("peak", pk_As), ("late", late_As)]:
        gt_snaps_As[nm] = compute_generation_times(
            f_A_static[day], sim_A_static["susceptibles"][day], pops_A,
            params["prob_transmission_peak"], w_within, max_days, lw_As, lb_As)

    # ── Independent R for static scenario ────────────────────────────────
    R_ind_As = estimate_R_independent(inc_As, gen_time_pmf, window=7)

    # ── GT spatial variation at peak ──────────────────────────────────────
    # PDF model: g_{kj} = p(a_E)/∫p universally for all (k,j) and t.
    # GT means should all equal GT_univ_mean = mean of gen_time_pmf.
    # Any residual variation is numerical noise only.
    print("\n─── GT spatial variation at peak (g^k_out means per location) ───")
    _days_arr = np.arange(max_days)
    _g_out_pk  = gt_snaps_A["peak"]["g_outward"]   # (max_days, N)
    _bKw_bKb_pk, _bKw_pk, _bKb_pk, _, _ = _kernel_base(
        f_A[pk_A], pops_A, lw_A, lb_A)
    for j in range(N_LOC):
        if _g_out_pk[:, j].sum() > 0.5:
            mu_j  = float(_g_out_pk[:, j] @ _days_arr)
            # within-weight: mean of bKw[j,:] / (bKw[j,:]+bKb[j,:])
            denom = (_bKw_pk[j] + _bKb_pk[j]).sum()
            ww_frac = float(_bKw_pk[j].sum() / denom) if denom > 0 else np.nan
            print(f"    L{j+1} ({types_A[j]:11s}): g_out_mean={mu_j:.3f}d  "
                  f"within-weight={ww_frac:.3f}")
    # Lambda value diagnostics
    _N_eff_peak = f_A[pk_A].T @ pops_A
    _lam_w_eff  = np.where(_N_eff_peak > 0, lw_A / _N_eff_peak, 0.0)
    _lam_b_eff  = np.where(_N_eff_peak > 0, lb_A / _N_eff_peak, 0.0)
    print(f"\n─── Lambda value diagnostics ───")
    print(f"  lw (scaled) = {lw_A:.4f}  lb (scaled) = {lb_A:.4f}  "
          f"lb/lw = {lb_A/lw_A:.3f}")
    print(f"  N_eff at peak: [{_N_eff_peak.min():.0f}, {_N_eff_peak.max():.0f}]")
    print(f"  λ_w = lw/N_eff: [{_lam_w_eff.min()*1e5:.2f}e-5, "
          f"{_lam_w_eff.max()*1e5:.2f}e-5] per contact per day")
    print(f"  λ_b = lb/N_eff: [{_lam_b_eff.min()*1e5:.2f}e-5, "
          f"{_lam_b_eff.max()*1e5:.2f}e-5] per contact per day")
    print(f"  [These are λ^k_E(t,a_E) base rates; final rate = λ × p(a_E)]")

    # ── Spectral + reactivity summary ─────────────────────────────────────
    react_A = reactivity(sim_A["R_matrices"][0])
    print(f"\n  σ(0)/ρ(0) = {react_A['amplification_ratio']:.4f}")
    print(f"  Spectral gap = {spec0_A['rho'] - spec0_A['lambda2']:.4f}")

    # ── Controllability ───────────────────────────────────────────────────
    ctrl = minimum_control_effort(sim_A["R_matrices"][0],
                                   costs=pops_A / pops_A.mean())
    print(f"\n  Homogeneous u_min = {ctrl['u_homogeneous']*100:.1f}%")

    # ── Figures ───────────────────────────────────────────────────────────
    print("\n─── Generating figures ───")
    _set_pub_style()
    R_t0_B = compute_R_matrix(f_B[0], pops_B, pops_B,
                               params["prob_transmission_peak"], infect_profile,
                               sim_B["lambda_within_scaled"], sim_B["lambda_between_scaled"])

    import os
    os.makedirs("out_figs/main", exist_ok=True)
    os.makedirs("out_figs/SI",   exist_ok=True)
    mpfx = "out_figs/main/fig"   # main figure prefix
    spfx = "out_figs/SI/fig"     # SI figure prefix
    # ── Type reproduction numbers (quick console check) ──────────────────
    print("\n─── Type reproduction numbers T_j(t) quick check ───")
    # Verify: T_j > 1 ⟺ R(t) > 1  (check at three representative days)
    for chk_day in [10, pk_A, min(T - 1, pk_A + 40)]:
        R_m   = sim_A["R_matrices"][chk_day]
        rho_t = R_system(R_m)
        T_js  = type_reproduction_numbers(R_m)
        flag  = "OK" if (
            np.all((T_js[~np.isnan(T_js)] > 1) == (rho_t > 1))
        ) else "MISMATCH"
        print(f"  day {chk_day:3d}  ρ={rho_t:.3f}  "
              f"T_j=[{np.nanmin(T_js):.2f},{np.nanmax(T_js):.2f}]  [{flag}]")


    plot_fig2(sim_A, city_A, f_A, gen_time_pmf, max_days, "Dense urban", mpfx)
    plot_fig3(sim_A, city_A, R_ind_A, gt_snaps_A, w_within, w_between, "Dense urban", mpfx)
    plot_fig4(sim_A, city_A, R_t0_B, "Dense urban", w_within, w_between, max_days, mpfx)
    plot_fig4(sim_B, city_B, R_t0_B, "Sparse national", w_within, w_between, max_days,
              "out_figs/main/fig_B")
    plot_fig_C_transience(sim_C, city_C, f_C, mpfx)
    plot_fig5(sim_A, sim_B, city_A, city_B, f_A, f_B, R0_A, R_t0_B, mpfx)
    plot_SI5_combined(sim_A, sim_B, city_A, city_B, f_A, f_B,
                      params["prob_transmission_peak"], infect_profile,
                      lw_A, lb_A, lw_B, lb_B, spfx)
    plot_fig7(sim_A, city_A, f_A, w_within, w_between, max_days,
              params["prob_transmission_peak"], "Dense urban", spfx)
    plot_SI1(sim_A, city_A, "Dense urban", spfx)
    print("\n─── Elasticity surface figure ───")
    plot_elasticity_surfaces(sim_A, sim_B, city_A, city_B, save_prefix=spfx)
    plot_SI2(sim_A_R012, city_A, f_A,
             params["prob_transmission_peak"], infect_profile, lw_A_R012, lb_A_R012,
             "Dense urban (R0=1.2)", spfx)
    plot_SI_epi_params(params, w_within, w_between, gen_time_pmf, max_days,
                       sim_A, f_A, populations=pops_A, save_prefix=spfx)
    plot_SI_pde_convergence(city_A, f_A,
                             params, initial_A, LW, LB, w_within, w_between,
                             T_test=90, save_prefix=spfx)
    plot_SI_sensitivity(city_A, f_A, params, initial_A, LW, LB, w_within, w_between,
                        T_test=120, save_prefix=spfx)
    plot_SI_3d_earlypeak(sim_A, sim_B, spfx)
    plot_counterfactual_nonnormal(sim_A, sim_C, city_A, f_A, f_C,
                                   w_within, w_between, max_days,
                                   lw_A, lb_A, lw_C, lb_C, params,
                                   R_ind_A, R_ind_C, gen_time_pmf, mpfx,
                                   city_C=city_C)
    print("\n─── Naive R comparison suite (18 figures) ───")
    plot_naive_R_comparison_suite(sim_A, sim_B, sim_C,
                                   city_A, city_B, city_C,
                                   gen_time_pmf, save_prefix=mpfx)
    print("\n─── Combined spatial aggregation bias figure ───")
    plot_main_bias_figure(sim_A, sim_B, sim_C,
                          city_A, city_B, city_C,
                          gen_time_pmf, save_prefix=mpfx)
    plot_3d_surfaces(sim_A, city_A, f_A, w_within, w_between, max_days,
                     params["prob_transmission_peak"], mpfx)
    plot_SI_lambda_decomposition(sim_A, city_A, f_A, params,
                                  gen_time_pmf, w_within, w_between, max_days,
                                  spfx)
    plot_SI0_population(city_A, city_B, spfx)
    plot_SI_gt_varying_beta(city_A, f_A[pk_A], w_within, w_between, max_days,
                             lw_A, lb_A, spfx)
    plot_SI_R_comparison(sim_A, city_A, R_ind_A, gen_time_pmf, spfx)
    plot_SI_gt_spatial(sim_A, city_A, w_within, w_between, gen_time_pmf,
                       max_days, spfx)

    # ── Type reproduction number figures ──────────────────────────────────
    print("\n─── Type reproduction number figures ───")
    plot_type_repro(sim_A, sim_B, city_A, city_B, mpfx)

    # ── SI figures: Static mobility alternative scenario ──────────────────
    # Repeats fig_02_overview, fig_03_taxonomy and fig_04_spectral for
    # the dense-urban setting but with a time-invariant (static) mobility
    # matrix f_{jk}(t) = base_fA for all t.  All other parameters are
    # identical to Scenario A.
    print("\n─── SI figures: Static mobility scenario ───")
    static_sfx = f"{spfx}_SI_static"
    plot_fig2(sim_A_static, city_A, f_A_static, gen_time_pmf, max_days,
              "Dense urban (static mobility)", static_sfx)
    plot_fig3(sim_A_static, city_A, R_ind_As, gt_snaps_As, w_within, w_between,
              "Dense urban (static mobility)", static_sfx)
    plot_fig4(sim_A_static, city_A, R_t0_B, "Dense urban (static mobility)",
              w_within, w_between, max_days, static_sfx)

    # ── Power-mean spectrum figures (one per scenario) ────────────────────
    print("\n─── Power-mean spectrum figures ───")
    plot_power_mean_spectrum(sim_A, city_A, "Scenario_A", mpfx)
    plot_power_mean_spectrum(sim_B, city_B, "Scenario_B", mpfx)
    plot_power_mean_spectrum(sim_C, city_C, "Scenario_C", mpfx)

    # ── Animations ────────────────────────────────────────────────────────────
    print("\n─── Generating animations ───")
    print("  Precomputing animation data...")
    data_A = _precompute_anim_data(sim_A, pops_A, gen_time_pmf)
    data_B = _precompute_anim_data(sim_B, pops_B, gen_time_pmf)
    data_C = _precompute_anim_data(sim_C, pops_C, gen_time_pmf)

    for _metric in ("incidence", "cum_incidence", "r_out", "eigvec"):
        plot_animation_network_metric(sim_A, sim_B, city_A, city_B,
                                      f_A, f_B, data_A, data_B,
                                      metric=_metric, save_prefix=mpfx)

    plot_animation_elasticity(sim_A, sim_B, sim_C,
                               city_A, city_B, city_C,
                               base_fA, base_fB, base_fC,
                               data_A, data_B, data_C, save_prefix=mpfx)

    plot_animation_R_heatmap(sim_A, city_A, data_A, gen_time_pmf,
                              save_prefix=mpfx)

    plot_animation_scenario_comparison(sim_A, sim_C, city_A, city_C,
                                        f_A, f_C, data_A, data_C,
                                        save_prefix=mpfx)

    print("\nAll figures saved.  Done.")


if __name__ == "__main__":
    main()