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
from mobility_rt.combined_table import build_combined_table
from mobility_rt.config import COVID_PARAMS
from mobility_rt.distributions import discretise_gamma
from mobility_rt.error_tables import build_estimator_error_tables
from mobility_rt.estimators import estimate_R_independent
from mobility_rt.generation_time import compute_generation_times
from mobility_rt.geometry import generate_city
from mobility_rt.gt_spatial import build_gt_spatial_table
from mobility_rt.heterogeneity import contact_multipliers, heterogeneity_summary, infectiousness_multipliers
from mobility_rt.kernel import R_inward, R_outward, R_system, _kernel_base, compute_R_matrix
from mobility_rt.location_errors import build_diagnostics_table, build_location_table
from mobility_rt.mobility import generate_mobility
from mobility_rt.plotting.animations import _precompute_anim_data, plot_animation_R_heatmap, plot_animation_elasticity, plot_animation_network_metric, plot_animation_scenario_comparison
from mobility_rt.plotting.figures_corrected import run_corrected_figures
from mobility_rt.plotting.figures_main import plot_counterfactual_nonnormal, plot_location_diagnostics, plot_location_discrepancy, plot_fig2, plot_fig3, plot_fig4, plot_fig5, plot_fig_C_transience, plot_estimator_error_summary, plot_main_bias_figure, plot_main_bias_variant_figures, plot_naive_R_comparison_suite, plot_power_mean_spectrum, plot_type_repro
from mobility_rt.plotting.figures_si import plot_3d_surfaces, plot_SI0_population, plot_SI1, plot_SI2, plot_SI5_combined, plot_SI_3d_earlypeak, plot_SI_R_comparison, plot_SI_epi_params, plot_SI_gt_heterogeneity, plot_SI_gt_spatial, plot_SI_gt_varying_beta, plot_SI_lambda_decomposition, plot_SI_pde_convergence, plot_SI_sensitivity, plot_elasticity_surfaces, plot_fig7
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

    # ── GT snapshots, one helper for every scenario ───────────────────────
    # Every scenario needs the same three snapshots (early / peak / late) built with
    # ITS OWN calibrated lambdas and ITS OWN heterogeneity multipliers.  Deriving
    # them here from the simulation dict means a heterogeneous scenario can never be
    # handed the homogeneous kernel by accident, which would silently mis-state its
    # R^{kj} and R^l_meeting (the generation-time SHAPES are universal either way).
    def _gt_snapshots(sim, f_series, pops_s):
        inc_s = sim["incidence"]
        pk_s  = int(inc_s.sum(axis=1).argmax())
        lw_s  = sim["lambda_within_scaled"]
        lb_s  = sim["lambda_between_scaled"]
        snaps = {}
        for nm, day in (("early", max(1, pk_s // 3)),
                        ("peak",  pk_s),
                        ("late",  min(T - 1, pk_s + 30))):
            snaps[nm] = compute_generation_times(
                f_series[day], sim["susceptibles"][day], pops_s,
                params["prob_transmission_peak"], w_within, max_days, lw_s, lb_s,
                sim.get("contact_mult"), sim.get("infect_mult"))
        return snaps

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
    print(f"  Total infections: {inc_As.sum():.0f}  "
          f"Peak day: {pk_As}  "
          f"Attack rate: {inc_As.sum()/pops_A.sum()*100:.1f}%")

    # ── Heterogeneous-transmissibility counterparts (A-het / B-het / C-het) ──
    # The baseline scenarios give every location the same contact rate and the same
    # per-contact transmission probability, so ALL variation in the reproduction
    # numbers comes from mobility geometry and susceptible depletion.  Each "-het"
    # scenario is that same scenario with two multipliers switched on:
    #   c[l]  contact rate at meeting location l, by settlement type
    #   m[k]  transmission probability of an infector resident in k, lognormal
    # Everything else (geometry, populations, mobility series, seeding, R0_target,
    # demography, solver) is identical, so the pair isolates the effect of
    # heterogeneous transmissibility on spatial aggregation bias.
    # The multipliers are illustrative assumptions, not fitted values; see
    # mobility_rt/heterogeneity.py.
    def _run_het(tag, city, f_series, initial, lb_scen, seed_offset):
        pops_h, types_h = city[1], city[3]
        # seed_offset gives each scenario its own infectiousness draw, so the three
        # heterogeneous runs are not correlated through a shared random vector.
        c_h = contact_multipliers(types_h)
        m_h = infectiousness_multipliers(len(pops_h), seed=SEED + seed_offset)
        print(f"\n─── Scenario {tag}: heterogeneous contact + infectiousness ───")
        summ = heterogeneity_summary(c_h, m_h)
        print(f"  contact c[l]: [{summ['contact']['min']:.3f}, "
              f"{summ['contact']['max']:.3f}]  CV={summ['contact']['cv']*100:.1f}%")
        print(f"  infect  m[k]: [{summ['infect']['min']:.3f}, "
              f"{summ['infect']['max']:.3f}]  CV={summ['infect']['cv']*100:.1f}%")
        sim_h = simulate_epidemic_pde(
            T, len(pops_h), pops_h, f_series,
            params["prob_transmission_peak"], infect_profile, max_days,
            params["R0_target"], initial, LW, lb_scen, w_within, w_between,
            birth_rate=0.00003, death_rate=0.00003,
            stochastic=False, susceptible_depletion=True, seed=SEED,
            contact_mult=c_h, infect_mult=m_h)
        inc_h  = sim_h["incidence"]
        R_out_h = np.array([R_outward(sim_h["R_matrices"][t]) for t in range(T)])
        pk_h   = int(inc_h.sum(axis=1).argmax())
        cv_out = float(np.median([R_out_h[t].std() / R_out_h[t].mean()
                                  for t in range(20, max(21, pk_h))]))
        print(f"  Total infections: {inc_h.sum():.0f}  Peak day: {pk_h}  "
              f"Attack rate: {inc_h.sum()/pops_h.sum()*100:.1f}%")
        print(f"  Spread of R^j_out across locations (median, days 20..peak): "
              f"{cv_out*100:.1f}%")
        # The multipliers are stored on the returned dict as sim["contact_mult"]
        # and sim["infect_mult"], so there is nothing else to hand back.
        return sim_h

    sim_A_het = _run_het("A-het (dense urban)", city_A, f_A, initial_A, LB, 0)

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

    sim_B_het = _run_het("B-het (sparse national)", city_B, f_B, initial_B, LB, 1)

    # ── Scenario C: Mega-hub non-normal mobility (false-action zone) ─────
    print("\n─── Scenario C: Mega-hub non-normal (false-action zone) ───")
    # Design rationale — checked analytically before full simulation:
    #
    # For σ(t) > 1 while ρ(t) < 1 (false-action zone), R_mat must be highly
    # non-normal: R[sat,hub] and R[hub,sat] must be strongly asymmetric.
    #
    # Hub topology + the mega-hub's susceptible pool make R[sat→hub] >> R[hub→sat]:
    #   (a) ONE mega-hub (pops_C[0]=3M) + 9 small satellites (200k each).
    #       With hub_attraction_power=8, gravity weight ∝ N^8 concentrates
    #       ALL satellite commuting at the single hub: f[sat,hub] ≈ c_sat.
    #       Hub residents barely visit satellites: f[hub,sat] ≈ 0.
    #       Because R[k,j] ∝ S_j (the INFECTEE's susceptibles) and S_hub = 3M ≫
    #       S_sat = 200k, the infectee=hub entries dominate: R[sat→hub] ≫ R[hub→sat].
    #       The hub is a super-RECEIVER (R_in ≫ R_out; here ≈ 1.61 vs 0.36), so
    #       R_mat has a dominant column — the source of the non-normality.
    #   (b) LB_C = LW × 0.3 — the SAME between/within ratio as A/B (BETWEEN_FRACTION),
    #       not a C-specific suppression. A modest lb keeps the symmetric between-K
    #       term below the asymmetric term, but C's non-normality is driven by the
    #       hub topology + susceptibility asymmetry in (a), NOT by this ratio.
    #   (c) Epidemic seeded at satellite node 1 (not hub) to route spread
    #       through the satellite→hub channel, making the asymmetry visible.
    #   (d) commuting_frac_scale=1.0 (do not inflate hub's commuting, which
    #       would reduce f[hub,hub] and weaken the directional term).
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

    # Scenario C keeps its own between-location rate LB_C and its satellite seeding.
    sim_C_het = _run_het("C-het (hub-amplified)", city_C, f_C, initial_C, LB_C, 2)

    # ── Independent R estimation ───────────────────────────────────────────
    print("\n─── Independent R̂(t) estimation ───")
    R_ind_A = estimate_R_independent(inc_A, gen_time_pmf, window=7)
    R_ind_C = estimate_R_independent(inc_C, gen_time_pmf, window=7)
    # A per-location independent estimate is NOT an estimate of that location's
    # R^in or R^out.  Mobility mixes the infectious pressure arriving there, so it
    # tracks the NETWORK value instead.  Printing it as a "bias" against R^in
    # implied R^in was the target; these are discrepancies, and the network column
    # is the small one.
    R_in_s  = np.array([R_inward(sim_A["R_matrices"][t]) for t in range(T)])
    R_out_s = np.array([R_outward(sim_A["R_matrices"][t]) for t in range(T)])
    rho_s   = np.array([R_system(sim_A["R_matrices"][t]) for t in range(T)])
    print("    (discrepancies, not errors: the estimate targets none of these)")
    for j in range(min(N_LOC, 4)):
        ind_m = float(np.nanmean(R_ind_A[30:60, j]))
        def _pc(x):
            return (ind_m - x) / x * 100 if x > 0 else np.nan
        print(f"    L{j+1}: Indep={ind_m:.3f}   vs rho {_pc(float(np.mean(rho_s[30:60]))):+6.1f}%"
              f"   vs R^out {_pc(float(np.mean(R_out_s[30:60, j]))):+6.1f}%"
              f"   vs R^in {_pc(float(np.mean(R_in_s[30:60, j]))):+6.1f}%")

    # ── GT snapshots ──────────────────────────────────────────────────────
    print("\n─── GT snapshots ───")
    early_A = max(1, pk_A // 3);  late_A = min(T-1, pk_A+30)
    gt_snaps_A = _gt_snapshots(sim_A, f_A, pops_A)
    for nm, day in [("early", early_A), ("peak", pk_A), ("late", late_A)]:
        gn_mean = float(np.sum(np.arange(max_days) * gt_snaps_A[nm]["g_network"]))
        print(f"    {nm:5s} (d{day:3d}): g_network mean = {gn_mean:.4f} d")

    # ── GT snapshots for static scenario ─────────────────────────────────
    gt_snaps_As = _gt_snapshots(sim_A_static, f_A_static, pops_A)

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
    os.makedirs("out_figs/main",   exist_ok=True)
    os.makedirs("out_figs/SI",     exist_ok=True)
    os.makedirs("out_figs/tables", exist_ok=True)
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
    # The estimator suite is computed once here and reused by the seven
    # weighting-scheme variants below (identical layout, different middle column).
    bias_data = plot_main_bias_figure(sim_A, sim_B, sim_C,
                                      city_A, city_B, city_C,
                                      gen_time_pmf, save_prefix=mpfx)
    print("\n─── Spatial aggregation bias — weighting-scheme variants (7 figures) ───")
    plot_main_bias_variant_figures(sim_A, sim_B, sim_C,
                                   city_A, city_B, city_C,
                                   gen_time_pmf, save_prefix=mpfx,
                                   all_data=bias_data)

    # ── Same figures for the heterogeneous-transmissibility counterparts ──
    # Cities (populations, node types) are unchanged, so city_A/B/C are reused;
    # only the simulations differ.
    print("\n─── Combined bias figures — heterogeneous scenarios (8 figures) ───")
    het_labels = ("Dense urban (A-het)",
                  "Sparse national (B-het)",
                  "Hub-amplified (C-het)")
    hpfx = f"{mpfx}_het"
    bias_data_het = plot_main_bias_figure(sim_A_het, sim_B_het, sim_C_het,
                                          city_A, city_B, city_C,
                                          gen_time_pmf, save_prefix=hpfx,
                                          labels=het_labels)
    plot_main_bias_variant_figures(sim_A_het, sim_B_het, sim_C_het,
                                   city_A, city_B, city_C,
                                   gen_time_pmf, save_prefix=hpfx,
                                   all_data=bias_data_het)

    # ── Mean relative error tables (CSV + LaTeX), growth phase and full period ──
    # One scenario list for every table and figure below, so their columns can
    # never fall out of step; the manuscript table asserts they did not.
    _SCEN = [("Scenario A",     sim_A,     city_A),
             ("Scenario A-het", sim_A_het, city_A),
             ("Scenario B",     sim_B,     city_B),
             ("Scenario B-het", sim_B_het, city_B),
             ("Scenario C",     sim_C,     city_C),
             ("Scenario C-het", sim_C_het, city_C)]

    print("\n─── Estimator mean-relative-error tables ───")
    err_table, _ = build_estimator_error_tables(
        _SCEN, gen_time_pmf, out_dir="out_figs/tables", prefix="estimator_error")

    # ── Location level: what a single location's own estimate is tracking ──
    print("\n─── Per-location discrepancy table ───")
    loc_table, _ = build_location_table(_SCEN, gen_time_pmf,
                                        out_dir="out_figs/tables",
                                        prefix="location_discrepancy")
    _LOC_TITLES = {"Scenario A":     "Dense urban (A)",
                   "Scenario A-het": "Dense urban (A-het)",
                   "Scenario B":     "Sparse national (B)",
                   "Scenario B-het": "Sparse national (B-het)",
                   "Scenario C":     "Hub-amplified (C)",
                   "Scenario C-het": "Hub-amplified (C-het)"}
    plot_location_discrepancy(loc_table, _LOC_TITLES, save_prefix=mpfx)

    # The discrepancy table above reports distances.  This one turns them into the
    # three questions a public health team actually has: can I rank locations, is
    # the heterogeneity visible, does a local estimate track local growth?
    print("\n─── Per-location diagnostics ───")
    diag_table, _ = build_diagnostics_table(_SCEN, gen_time_pmf,
                                            out_dir="out_figs/tables",
                                            prefix="location_diagnostics")
    plot_location_diagnostics(diag_table, _LOC_TITLES, save_prefix=mpfx)

    # ── The single manuscript table: both halves, one set of days ─────────
    # Assembled from the three tables already computed above, so nothing is
    # recomputed and the numbers cannot drift from the tables they came from.
    print("\n─── Manuscript table (network + location level) ───")
    build_combined_table(_SCEN, gen_time_pmf, out_dir="out_figs/tables",
                         prefix="manuscript_table",
                         net_table=err_table, loc_table=loc_table,
                         diag_table=diag_table)

    # Visual companion to those tables: what heterogeneity does to each scheme.
    print("\n─── Estimator error summary figure ───")
    plot_estimator_error_summary(
        err_table,
        [("Scenario A", "Scenario A-het", "Dense urban (A)"),
         ("Scenario B", "Scenario B-het", "Sparse national (B)"),
         ("Scenario C", "Scenario C-het", "Hub-amplified (C)")],
        save_prefix=mpfx)
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
              w_within, w_between, max_days, static_sfx, show_dow=False)

    # ── Does the generation time vary across space? ───────────────────────
    # The heterogeneous scenarios give each location its own contact rate c[l] and
    # its own transmissibility m[k].  Neither depends on infection age, so both
    # cancel in g_{kj} = K_{kj}/R_{kj} and the FORWARD generation time stays exactly
    # universal; the REALISED (backward) interval does vary across space, because it
    # reweights p(a_E) by each location's own infection-age composition.  Both halves
    # are measured here rather than asserted.
    print("\n─── Generation time under spatial heterogeneity ───")
    gt_snaps_A_het = _gt_snapshots(sim_A_het, f_A,  pops_A)
    gt_snaps_B     = _gt_snapshots(sim_B,     f_B,  pops_B)
    gt_snaps_B_het = _gt_snapshots(sim_B_het, f_B,  pops_B)
    gt_snaps_C     = _gt_snapshots(sim_C,     f_C,  pops_C)
    gt_snaps_C_het = _gt_snapshots(sim_C_het, f_C,  pops_C)
    build_gt_spatial_table(
        [("Dense urban (A)",       sim_A,     gt_snaps_A),
         ("Dense urban (A-het)",   sim_A_het, gt_snaps_A_het),
         ("Sparse national (B)",   sim_B,     gt_snaps_B),
         ("Sparse national (B-het)", sim_B_het, gt_snaps_B_het),
         ("Hub-amplified (C)",     sim_C,     gt_snaps_C),
         ("Hub-amplified (C-het)", sim_C_het, gt_snaps_C_het)],
        p_aE, out_dir="out_figs/tables", prefix="gt_spatial_heterogeneity")
    plot_SI_gt_heterogeneity(
        [("Dense urban",     sim_A, sim_A_het, gt_snaps_A, gt_snaps_A_het),
         ("Sparse national", sim_B, sim_B_het, gt_snaps_B, gt_snaps_B_het),
         ("Hub-amplified",   sim_C, sim_C_het, gt_snaps_C, gt_snaps_C_het)],
        p_aE, max_days, spfx)

    # ── Corrected manuscript figures (formerly make_corrected_figures.py) ─
    # Uses the Scenario A simulations already built above, so the corrected and
    # main figures are guaranteed to depict the same run.  run_corrected_figures
    # restores the pipeline's rcParams on exit.
    print("\n─── Corrected manuscript figures ───")
    # The alternative scenarios get exactly the treatment the static-mobility twin
    # gets: the same corrected fig_02 / fig_03 / fig_04 panel set, from their own
    # simulation, their own mobility series, their own independent R̂ and their own
    # GT snapshots.  Each heterogeneous run is listed next to the baseline it is the
    # counterpart of, so a reader comparing A with A-het, B with B-het or C with
    # C-het is comparing two figures that differ in nothing but the multipliers.
    # (Scenario A's own baseline set is the "fig_*" one drawn by the call below.)
    _corrected_extras = [
        (sim_A_het, city_A, f_A, gt_snaps_A_het, "Dense urban (A-het)",     "fig_A_het"),
        (sim_B,     city_B, f_B, gt_snaps_B,     "Sparse national (B)",     "fig_B"),
        (sim_B_het, city_B, f_B, gt_snaps_B_het, "Sparse national (B-het)", "fig_B_het"),
        (sim_C,     city_C, f_C, gt_snaps_C,     "Hub-amplified (C)",       "fig_C"),
        (sim_C_het, city_C, f_C, gt_snaps_C_het, "Hub-amplified (C-het)",   "fig_C_het"),
    ]
    extra_scenarios = []
    for _sim_x, _city_x, _f_x, _gt_x, _name_x, _pfx_x in _corrected_extras:
        extra_scenarios.append({
            "sim":       _sim_x,
            "city_data": _city_x,
            "f_series":  _f_x,
            "R_ind":     estimate_R_independent(_sim_x["incidence"],
                                                gen_time_pmf, window=7),
            "gt_snaps":  _gt_x,
            "name":      _name_x,
            "prefix":    _pfx_x,
        })
    run_corrected_figures(sim_A, sim_A_static, city_A, f_A, f_A_static,
                          R_ind_A, R_ind_As, gt_snaps_A, gt_snaps_As,
                          gen_time_pmf, w_within, w_between, max_days,
                          scenario_name="Dense urban",
                          out_dir="out_figs/corrected",
                          extra_scenarios=extra_scenarios)

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