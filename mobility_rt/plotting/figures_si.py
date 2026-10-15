"""mobility_rt.plotting.figures_si — manuscript Figs SI1-SI6, SI26 and SI27.

  plot_SI_3d_earlypeak   Fig SI1   pairwise new infections at early and peak days
  plot_SI_epi_params     Fig SI2   epidemiological and mobility parameters
  plot_SI0_population    Fig SI3   resident populations
  plot_SI_pde_convergence Fig SI4  numerical checks of the solver
  plot_SI1               Fig SI5   sensitivity and elasticity matrices
  plot_SI_sensitivity    Fig SI6   sensitivity of R(t) and incidence to inputs
  plot_SI_R_comparison   Fig SI26  independent estimates against R^j_out and R^j_in
  plot_SI5_combined      Fig SI27  meeting-location reproduction numbers
(Function and file names predate the manuscript's SI numbering.)
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from mobility_rt.config import DOW_SCALE, OKABE_ITO
from mobility_rt.distributions import discretise_gamma
from mobility_rt.error_tables import ESTIMATOR_WINDOW, RHO_FLOOR, trailing_mean
from mobility_rt.estimators import total_infectiousness
from mobility_rt.geometry import representative_locs
from mobility_rt.kernel import R_inward, R_outward, R_system
from mobility_rt.plotting.style import _bar3d_inc, _panel_label
from mobility_rt.simulation import (_simulate_heun, _simulate_rk4, phase_days,
                                    simulate_epidemic_pde)
from mobility_rt.spectral import sensitivity_elasticity, spectral_analysis
from mobility_rt.transmissibility import euler_lotka_r


def plot_SI0_population(city_A, city_B, save_prefix="fig"):
    """Fig SI3: resident population of each location, Scenarios A and B, coloured by
    node type."""
    pops_A, types_A = city_A[1], city_A[3]
    pops_B, types_B = city_B[1], city_B[3]
    N = len(pops_A)
    loc = [f"L{i+1}" for i in range(N)]

    # Assign colours by node type using Okabe-Ito
    type_colors_A = {
        "core":       OKABE_ITO[0],
        "dense":      OKABE_ITO[1],
        "suburban":   OKABE_ITO[2],
        "peripheral": OKABE_ITO[3],
    }
    type_colors_B = {
        "capital":          OKABE_ITO[0],
        "peri-capital":     OKABE_ITO[1],
        "urban-industrial": OKABE_ITO[2],
        "semi-urban":       OKABE_ITO[4],
        "rural":            OKABE_ITO[3],
        "remote-rural":     OKABE_ITO[6],
    }

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2),
                             gridspec_kw=dict(left=0.09, right=0.97,
                                              top=0.87, bottom=0.28,
                                              wspace=0.38))

    for ax, pops, types, type_colors, title in [
        (axes[0], pops_A, types_A, type_colors_A, "Dense urban (Scenario A)"),
        (axes[1], pops_B, types_B, type_colors_B, "Sparse national (Scenario B)"),
    ]:
        clrs = [type_colors.get(t, OKABE_ITO[5]) for t in types]
        ax.bar(range(N), pops / 1e3, color=clrs, edgecolor="none", width=0.72)
        ax.set_xticks(range(N))
        ax.set_xticklabels(loc, fontsize=7, rotation=45, ha="right")
        ax.set_ylabel("Population ($\\times 10^3$)", fontsize=8)
        ax.set_xlabel("Location", fontsize=8)
        ax.axhline(float(pops.mean()) / 1e3, color="0.5", lw=0.9, ls="--")
        # Annotate total population
        ax.text(0.97, 0.97, f"Total = {pops.sum()/1e3:.0f}k",
                transform=ax.transAxes, fontsize=7, ha="right", va="top",
                color="0.3")
        # Legend: node types in order of first appearance, then the mean line
        seen = {}
        for t, c in zip(types, clrs):
            if t not in seen:
                seen[t] = c
        handles = [Patch(facecolor=c, label=t.capitalize(), edgecolor="none")
                   for t, c in seen.items()]
        ax.legend(handles=handles + [
            Line2D([0], [0], color="0.5", lw=0.9, ls="--",
                   label=f"Mean {pops.mean()/1e3:.1f}k")],
                  fontsize=6, borderpad=0.3, loc="upper center",
                  bbox_to_anchor=(0.5, -0.22), ncol=3)

    plt.savefig(f"{save_prefix}_SI0_population.pdf", dpi=300,
                bbox_inches="tight")
    plt.close()
    print(f"  Saved {save_prefix}_SI0_population.pdf")


def plot_SI1(sim, save_prefix="fig"):
    """Fig SI5: sensitivity and elasticity matrices.

    S_{kj} = ∂ℛ/∂R_{kj} = v*_k v_j / (vᵀv*)   (Eq 31)
    ε_{kj} = (R_{kj}/ℛ) S_{kj}                 (Eq 33)

    Panels (2×3):
      A  S_{kj} at the early day (phase_days)
      B  S_{kj} at the epidemic peak
      C  coefficient of variation of R^j_out over time
      D  ε_{kj} at the early day
      E  ε_{kj} at the epidemic peak
      F  location elasticity ε^k = Σ_j ε_{kj} at the peak
    """
    inc    = sim["incidence"]
    R_mats = sim["R_matrices"]
    T, N   = inc.shape
    phases = phase_days(inc)
    early, peak = phases["early"], phases["peak"]
    loc    = [f"L{i+1}" for i in range(N)]

    fig = plt.figure(figsize=(7.2, 4.8))
    gs  = gridspec.GridSpec(2, 3, hspace=0.55, wspace=0.55,
                            left=0.08, right=0.97, top=0.97, bottom=0.08)

    # ── A/B: sensitivity matrices at early and peak ────────────────────────
    # (the early and peak days are stated in the caption, not on the figure)
    for ci, day in enumerate((early, peak)):
        se = sensitivity_elasticity(R_mats[day])
        ax = fig.add_subplot(gs[0, ci])
        im = ax.imshow(se["sensitivity"], cmap="Blues", aspect="auto")
        ax.set_xticks(range(N)); ax.set_xticklabels(loc, fontsize=5, rotation=45)
        ax.set_yticks(range(N)); ax.set_yticklabels(loc, fontsize=5)
        ax.set_xlabel("Infectee $j$"); ax.set_ylabel("Infector $k$")
        cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cb.ax.set_title("$s^{kj}$", fontsize=6, pad=3)
        _panel_label(ax, ["A", "B"][ci])

    # ── C: CV of outward R over time ───────────────────────────────────────
    ax = fig.add_subplot(gs[0, 2])
    cv_ts = np.array([spectral_analysis(R_mats[t])["cv_row_sums"] for t in range(T)])
    vc    = cv_ts > 0
    ax.plot(np.where(vc)[0], cv_ts[vc], color=OKABE_ITO[2], lw=1.2)
    ax.set_xlabel("Day $t$"); ax.set_ylabel("CV of $R^j_{\\rm out}$")
    _panel_label(ax, "C")

    # ── D/E: elasticity matrices at early and peak ─────────────────────────
    for ci, day in enumerate((early, peak)):
        se = sensitivity_elasticity(R_mats[day])
        ax = fig.add_subplot(gs[1, ci])
        im = ax.imshow(se["elasticity"], cmap="Purples", aspect="auto")
        ax.set_xticks(range(N)); ax.set_xticklabels(loc, fontsize=5, rotation=45)
        ax.set_yticks(range(N)); ax.set_yticklabels(loc, fontsize=5)
        ax.set_xlabel("Infectee $j$"); ax.set_ylabel("Infector $k$")
        cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cb.ax.set_title("$\\varepsilon^{kj}$", fontsize=6, pad=3)
        _panel_label(ax, ["D", "E"][ci])

    # ── F: location elasticity ε^k = Σ_j ε^{kj} (= Σ_j ε^{jk}) at the peak ────
    ax = fig.add_subplot(gs[1, 2])
    se_peak = sensitivity_elasticity(R_mats[peak])
    col_elas = se_peak["elasticity"].sum(axis=1)   # Σ_j ε_{kj}
    for k in range(N):
        ax.bar(k, col_elas[k], color=OKABE_ITO[k % len(OKABE_ITO)])
    ax.set_xticks(range(N)); ax.set_xticklabels(loc, rotation=45, fontsize=6)
    ax.set_ylabel("Location elasticity $\\varepsilon^k$")
    _panel_label(ax, "F")

    plt.savefig(f"{save_prefix}_SI1_sensitivity.pdf", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved {save_prefix}_SI1_sensitivity.pdf")


def plot_SI_epi_params(params, gen_time_pmf, max_days, sim, f_jk, populations,
                       save_prefix="fig"):
    """Fig SI2: epidemiological and mobility parameters.

    Panels (2×3):
      A  infectiousness profile p(a_E)
      B  its cumulative distribution
      C  between- relative to within-location contact rate, λ_B/λ_W
      D  day-of-week multipliers on the away-from-home fraction
      E  effective population N^l_eff at the epidemic peak
      F  calibrated rates prob_transmission_peak·λ/N^l_eff at meeting location l,
         homogeneous scenario (c_l = m_k = 1), at the epidemic peak
    """
    days = np.arange(max_days)
    populations = np.asarray(populations, dtype=float)

    inc  = sim["incidence"]
    peak = int(inc.sum(axis=1).argmax())
    N_eff_peak = f_jk[peak].T @ populations   # shape (N,)
    N = len(N_eff_peak)
    loc_labels = [f"L{i+1}" for i in range(N)]

    # The calibrated rates the model actually runs at
    lw = sim["lambda_within_scaled"]
    lb = sim["lambda_between_scaled"]
    pp = params["prob_transmission_peak"]

    fig = plt.figure(figsize=(7.2, 4.8))
    gs  = gridspec.GridSpec(2, 3, hspace=0.70, wspace=0.55,
                            left=0.09, right=0.97, top=0.97, bottom=0.10)

    # ── A: infectiousness profile ─────────────────────────────────────────
    ax = fig.add_subplot(gs[0, 0])
    ax.plot(days, gen_time_pmf, color=OKABE_ITO[4], lw=1.6,
            label=(f"$p(a_E)$ from Gamma(mean={params['gen_time_mean']} d, "
                   f"SD={params['gen_time_sd']} d)"))
    mu_p = float(np.sum(days * gen_time_pmf))
    ax.axvline(mu_p, color=OKABE_ITO[4], lw=0.9, ls="--", alpha=0.7,
               label=f"Mean = {mu_p:.1f} d")
    ax.fill_between(days, gen_time_pmf, alpha=0.15, color=OKABE_ITO[4])
    ax.set_xlabel("Infection age $a_E$ (days)")
    ax.set_ylabel("Probability")
    ax.legend(fontsize=5.5, borderpad=0.3, labelspacing=0.2, loc="upper right")
    _panel_label(ax, "A")

    # ── B: cumulative distribution ────────────────────────────────────────
    ax = fig.add_subplot(gs[0, 1])
    ax.plot(days, np.cumsum(gen_time_pmf), color=OKABE_ITO[4], lw=1.6,
            label="$p(a_E)$ (universal profile)")
    ax.axhline(0.50, color="0.65", lw=0.7, ls="--")
    ax.axhline(0.95, color="0.65", lw=0.7, ls="--")
    ax.text(max_days * 0.62, 0.52, "50%", fontsize=5.5, color="0.5")
    ax.text(max_days * 0.62, 0.97, "95%", fontsize=5.5, color="0.5")
    ax.set_xlabel("Infection age $a_E$ (days)")
    ax.set_ylabel("Cumulative probability")
    ax.legend(fontsize=6, borderpad=0.3)
    _panel_label(ax, "B")

    # ── C: between- relative to within-location contact rate ──────────────
    ax = fig.add_subplot(gs[0, 2])
    ratio = lb / lw
    labels_c = ["Within location\n(home), $\\lambda_W$",
                "Between locations\n(away), $\\lambda_B$"]
    values_c = [1.0, ratio]
    bars = ax.bar(labels_c, values_c, color=[OKABE_ITO[0], OKABE_ITO[5]],
                  alpha=0.85, edgecolor="none", width=0.55)
    ax.set_ylabel("Contact rate relative to $\\lambda_W$")
    ax.set_ylim(0, 1.6)
    ax.tick_params(axis="x", labelsize=6)
    for bar, val in zip(bars, values_c):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"{val:.2f}", ha="center", va="bottom", fontsize=7)
    ax.text(0.5, 0.98,
            "$\\lambda_W$ and $\\lambda_B$ are rescaled together\n"
            "to hit $R_0$; only the ratio is identified",
            transform=ax.transAxes, ha="center", va="top", fontsize=5, color="0.3")
    _panel_label(ax, "C")

    # ── D: day-of-week mobility scaling ───────────────────────────────────
    ax = fig.add_subplot(gs[1, 0])
    dow_scale  = np.asarray(DOW_SCALE, dtype=float)
    dow_labels = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    dow_cols   = ([OKABE_ITO[1]] * 4 + [OKABE_ITO[0]] +
                  [OKABE_ITO[5]] * 2)
    bars = ax.bar(dow_labels, dow_scale, color=dow_cols, alpha=0.85,
                  edgecolor="none", width=0.65)
    ax.axhline(1.0, color="0.6", lw=0.7, ls="--")
    ax.set_ylabel("Multiplier on away-from-home fraction")
    ax.set_ylim(0, 1.35)
    for bar, val in zip(bars, dow_scale):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"{val:.2f}", ha="center", va="bottom", fontsize=6.5)
    _panel_label(ax, "D")

    # ── E: effective population N_eff^l at peak ───────────────────────────
    ax = fig.add_subplot(gs[1, 1])
    bar_cols_e = [OKABE_ITO[i % len(OKABE_ITO)] for i in range(N)]
    ax.bar(range(N), N_eff_peak / 1e3, color=bar_cols_e,
           alpha=0.85, edgecolor="none", width=0.7)
    ax.set_xticks(range(N)); ax.set_xticklabels(loc_labels, rotation=45, fontsize=6)
    ax.set_ylabel("$N^l_{\\rm eff}$ ($\\times 10^3$)")
    _panel_label(ax, "E")

    # ── F: calibrated rates at meeting location l ──────────────────────────
    ax = fig.add_subplot(gs[1, 2])
    rate_w = np.where(N_eff_peak > 0, pp * lw / N_eff_peak, 0.0)
    rate_b = np.where(N_eff_peak > 0, pp * lb / N_eff_peak, 0.0)
    exponent = int(np.floor(np.log10(max(rate_w.max(), rate_b.max()))))
    unit = 10.0 ** exponent
    x = np.arange(N)
    width = 0.38
    ax.bar(x - width/2, rate_w / unit, width=width,
           color=OKABE_ITO[0], alpha=0.85, edgecolor="none",
           label="$p\\,\\lambda_W/N^l_{\\rm eff}$")
    ax.bar(x + width/2, rate_b / unit, width=width,
           color=OKABE_ITO[5], alpha=0.85, edgecolor="none",
           label="$p\\,\\lambda_B/N^l_{\\rm eff}$")
    ax.set_xticks(range(N)); ax.set_xticklabels(loc_labels, rotation=45, fontsize=6)
    ax.set_ylabel(f"Rate ($\\times 10^{{{exponent}}}$ per day)")
    ax.legend(fontsize=5.5, borderpad=0.3)
    _panel_label(ax, "F")

    plt.savefig(f"{save_prefix}_SI3_epi_params.pdf", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved {save_prefix}_SI3_epi_params.pdf")


def plot_SI_pde_convergence(city_data, f_jk, base_f, params, initial_infections,
                            LW, LB, T_test=365, save_prefix="fig"):
    """Fig SI4: numerical checks of the deterministic upwind solver.

      A  total incidence for infectiousness-profile truncations τ = 10, 15, 20, 25 d
      B  ℛ(t) for the same truncations
      C  incidence under four update rules for S at Δt = 1 d: exponential escape
         (the solver used everywhere), forward Euler, Heun and RK4.  These are
         different update rules at the same step, not a refinement of Δt.  Heun and
         RK4 approximate the same within-day depletion as exponential escape, so
         forward Euler is the outlier
      D  closed-population balance |S_end(t) + Σ_{s<=t} I(s) − N| / N per location
      E  renewal identity I_j(t) = [R(t)ᵀ Λ(t)]_j / Σ_{a>=1} p(a), with
         Λ_k(t) = Σ_{a>=1} p(a) I_k(t−a) rebuilt from the stored incidence alone
         (an independent path through R_matrices), relative residual per location.
         The identity is exact only for the forward-Euler update, whose boundary
         condition is linear in S; under the exponential update it holds to O(φ/2),
         so both are drawn.  It is a property of the scheme, not of the model
      F  early growth: log-linear fit to incidence against the Euler–Lotka rate at the
         mean ℛ(t) over the fitting window

    Every run is calibrated on base_f, as in the pipeline.
    """
    pops = np.asarray(city_data[1], dtype=float)
    N_LOC = len(pops)
    f_sub = f_jk[:T_test]

    max_days_ref = params["max_gen_time"]
    infect_profile = discretise_gamma(params["gen_time_mean"],
                                      params["gen_time_sd"], max_days_ref)

    test_md  = [10, 15, 20, 25]
    md_cols  = [OKABE_ITO[0], OKABE_ITO[2], OKABE_ITO[4], OKABE_ITO[5]]
    if max_days_ref not in test_md:
        raise ValueError(f"max_gen_time {max_days_ref} must be one of {test_md}")

    print(f"  Running solver checks (deterministic, T={T_test})...")
    sims_md = {}
    for md in test_md:
        gtp = discretise_gamma(params["gen_time_mean"], params["gen_time_sd"], md)
        sims_md[md] = simulate_epidemic_pde(
            T_test, N_LOC, pops, f_sub, params["prob_transmission_peak"], gtp, md,
            params["R0_target"], initial_infections, LW, LB, f_calibration=base_f)

    # the reference run (full truncation) is the exponential-escape solver of the pipeline
    s_ref   = sims_md[max_days_ref]
    inc_ref = s_ref["incidence"]           # (T, N)

    print("  Alternative update rules (forward Euler, Heun, RK4)...")
    s_euler = simulate_epidemic_pde(
        T_test, N_LOC, pops, f_sub, params["prob_transmission_peak"],
        infect_profile, max_days_ref, params["R0_target"],
        initial_infections, LW, LB, f_calibration=base_f, update="euler")
    s_heun = _simulate_heun(
        T_test, N_LOC, pops, f_sub, params["prob_transmission_peak"],
        infect_profile, max_days_ref, params["R0_target"],
        initial_infections, LW, LB, f_calibration=base_f)
    s_rk4  = _simulate_rk4(
        T_test, N_LOC, pops, f_sub, params["prob_transmission_peak"],
        infect_profile, max_days_ref, params["R0_target"],
        initial_infections, LW, LB, f_calibration=base_f)

    # ── D: closed-population balance ──────────────────────────────────────
    cumI = np.cumsum(inc_ref, axis=0)
    mass_resid = np.abs(s_ref["S_end"] + cumI - pops[np.newaxis, :]) / pops[np.newaxis, :]

    # ── E: renewal identity from the stored incidence ─────────────────────
    lam_ref = total_infectiousness(inc_ref, infect_profile)   # Λ_k(t), (T, N)
    sum_p = infect_profile[1:].sum()
    renewal_resid = np.full((T_test, N_LOC), np.nan)
    euler_resid   = np.full((T_test, N_LOC), np.nan)
    inc_eul = s_euler["incidence"]
    lam_eul = total_infectiousness(inc_eul, infect_profile)
    for t in range(1, T_test):
        pred = s_ref["R_matrices"][t].T @ lam_ref[t] / sum_p
        renewal_resid[t] = np.abs(inc_ref[t] - pred) / np.maximum(inc_ref[t], 1.0)
        pred_e = s_euler["R_matrices"][t].T @ lam_eul[t] / sum_p
        euler_resid[t] = np.abs(inc_eul[t] - pred_e) / np.maximum(inc_eul[t], 1.0)

    # ── figure ────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(7.2, 8.0))
    gs  = gridspec.GridSpec(3, 2, hspace=0.62, wspace=0.45,
                            left=0.09, right=0.97, top=0.94, bottom=0.08)

    # ── A: total incidence for different truncations ───────────────────────
    ax = fig.add_subplot(gs[0, 0])
    for md, col in zip(test_md, md_cols):
        inc = sims_md[md]["incidence"].sum(axis=1)
        ax.plot(inc / 1e3, color=col, lw=0.9, label=f"$\\tau={md}$ d")
    ax.set_xlabel("Day $t$")
    ax.set_ylabel("Total incidence ($\\times 10^3$)")
    ax.legend(fontsize=6, borderpad=0.3, title="Max age $\\tau$", title_fontsize=5.5)
    _panel_label(ax, "A")

    # ── B: ℛ(t) for different truncations ─────────────────────────────────
    ax = fig.add_subplot(gs[0, 1])
    for md, col in zip(test_md, md_cols):
        R_sys = np.array([R_system(sims_md[md]["R_matrices"][t])
                          for t in range(T_test)])
        vld = R_sys > 0
        ax.plot(np.where(vld)[0], R_sys[vld], color=col, lw=0.9,
                label=f"$\\tau={md}$ d")
    ax.axhline(1.0, color="0.55", ls="--", lw=0.8)
    ax.set_xlabel("Day $t$"); ax.set_ylabel("$\\mathcal{R}(t)$")
    ax.legend(fontsize=6, borderpad=0.3, title="Max age $\\tau$", title_fontsize=5.5)
    _panel_label(ax, "B")

    # ── C: update rules ───────────────────────────────────────────────────
    ax = fig.add_subplot(gs[1, 0])
    ax.plot(inc_ref.sum(axis=1) / 1e3,
            color=OKABE_ITO[0], lw=1.3, label="Exponential escape (used throughout)")
    ax.plot(s_euler["incidence"].sum(axis=1) / 1e3,
            color=OKABE_ITO[1], lw=1.0, ls="-.", label="Forward Euler")
    ax.plot(s_heun["incidence"].sum(axis=1) / 1e3,
            color=OKABE_ITO[4], lw=0.9, ls="--", label="Heun")
    ax.plot(s_rk4["incidence"].sum(axis=1) / 1e3,
            color=OKABE_ITO[2], lw=0.8, ls=":", label="RK4")
    ax.set_xlabel("Day $t$"); ax.set_ylabel("Incidence ($\\times 10^3$)")
    ax.legend(fontsize=6, borderpad=0.3, loc="upper left")
    _panel_label(ax, "C")

    # ── D: closed-population balance ──────────────────────────────────────
    ax = fig.add_subplot(gs[1, 1])
    for j in range(N_LOC):
        ax.plot(np.maximum(mass_resid[:, j], 1e-18), color=OKABE_ITO[j % len(OKABE_ITO)],
                lw=0.7, alpha=0.75)
    ax.set_xlabel("Day $t$")
    ax.set_ylabel("$|S_{\\rm end}+I_{\\rm cum}-N|/N$ (floor $10^{-18}$)")
    ax.set_yscale("log")
    ax.text(0.97, 0.97, "Each line = one location",
            transform=ax.transAxes, fontsize=5, ha="right", va="top", color="0.4")
    _panel_label(ax, "D")

    # ── E: renewal identity ───────────────────────────────────────────────
    ax = fig.add_subplot(gs[2, 0])
    # day 0 is the seed and has no renewal identity, so the residual starts at day 1
    resid_days = np.arange(1, T_test)
    ax.semilogy(resid_days, np.maximum(renewal_resid[1:].mean(axis=1), 1e-18),
                color=OKABE_ITO[0], lw=1.0, label="Exponential: mean over locations")
    ax.semilogy(resid_days, np.maximum(renewal_resid[1:].max(axis=1), 1e-18),
                color=OKABE_ITO[5], lw=0.9, ls="--", label="Exponential: max over locations")
    ax.semilogy(resid_days, np.maximum(euler_resid[1:].max(axis=1), 1e-18),
                color=OKABE_ITO[1], lw=0.9, ls=":",
                label="Forward Euler: max (exact; drawn at the $10^{-18}$ floor)")
    ax.set_xlabel("Day $t$")
    ax.set_ylabel("$|I_j(t) - [\\mathbf{R}(t)^\\top\\Lambda(t)]_j/\\sum_{a\\geq1}p|$"
                  " / max$(I_j,1)$  (floor $10^{-18}$)", fontsize=6)
    ax.legend(fontsize=6, borderpad=0.3)
    _panel_label(ax, "E")

    # ── F: early growth vs Euler–Lotka ────────────────────────────────────
    ax = fig.add_subplot(gs[2, 1])
    inc_total = inc_ref.sum(axis=1)
    peak_day  = int(np.argmax(inc_total))
    # Fit log-linear growth over [fit_start, fit_end), before half the peak day
    fit_end   = max(5, peak_day // 2)
    fit_start = max(1, fit_end - 20)
    t_arr_fit = np.arange(fit_start, fit_end)
    inc_fit   = inc_total[fit_start:fit_end]
    valid_fit = inc_fit > 0
    if valid_fit.sum() > 4:
        coeffs = np.polyfit(t_arr_fit[valid_fit], np.log(inc_fit[valid_fit]), 1)
        r_fit, log_c0 = float(coeffs[0]), float(coeffs[1])
    else:
        r_fit, log_c0 = np.nan, np.nan
    # Euler–Lotka at the mean ℛ(t) over the same days
    rho_bar = float(np.mean([R_system(s_ref["R_matrices"][t]) for t in t_arr_fit]))
    r_theory = euler_lotka_r(rho_bar, infect_profile)
    t_plot = np.arange(T_test)
    ax.semilogy(t_plot[inc_total > 0], inc_total[inc_total > 0],
                color=OKABE_ITO[0], lw=1.2, label="PDE simulation")
    t_line = np.arange(fit_start, min(fit_end + 15, T_test))
    if not np.isnan(r_fit):
        ax.semilogy(t_line, np.exp(r_fit * t_line + log_c0),
                    color=OKABE_ITO[2], lw=1.0, ls="--",
                    label=f"Fitted: $r={r_fit:.4f}\\,\\mathrm{{d}}^{{-1}}$")
        if not np.isnan(r_theory):
            # anchored at the fitted value on the first day of the window
            y0 = np.exp(r_fit * fit_start + log_c0)
            ax.semilogy(t_line, y0 * np.exp(r_theory * (t_line - fit_start)),
                        color=OKABE_ITO[5], lw=1.0, ls=":",
                        label=(f"Euler–Lotka at mean $\\mathcal{{R}}$ = {rho_bar:.3f} "
                               f"over fit window: $r={r_theory:.4f}\\,\\mathrm{{d}}^{{-1}}$"))
    ax.axvspan(fit_start, fit_end, alpha=0.07, color=OKABE_ITO[0])
    ax.set_xlabel("Day $t$")
    ax.set_ylabel("Total incidence (log scale)")
    ax.legend(fontsize=5.0, borderpad=0.3)
    _panel_label(ax, "F")

    plt.savefig(f"{save_prefix}_SI4_convergence.pdf", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved {save_prefix}_SI4_convergence.pdf")


def plot_SI_sensitivity(city_data, f_jk, base_f, params, initial_infections,
                        LW, LB, T_test=365, save_prefix="fig"):
    """Fig SI6: sensitivity of ℛ(t) and incidence to model inputs.

    Panels:
      A  ℛ(t) for R0 = 1.5, 2.0, 2.5, 3.0
      B  incidence for seeding at the hub, at the peripheral location, or uniformly
      C  ℛ(t) for λ_B/λ_W = 0.15, 0.30, 0.45
      D  incidence for infectiousness profiles with means 2.5, 5.5 and 7.0 days
         (sd 1.0, 1.8 and 2.5 days)

    Every run is calibrated on base_f, so R0 is held at its target in B-D.
    """
    pops = np.asarray(city_data[1], dtype=float)
    N_LOC = len(pops)
    f_sub = f_jk[:T_test]
    max_days = params["max_gen_time"]
    infect_profile = discretise_gamma(params["gen_time_mean"],
                                      params["gen_time_sd"], max_days)

    i_hub, _, i_per, _, _ = representative_locs(city_data)

    def _run(R0, profile, init, lb):
        return simulate_epidemic_pde(
            T_test, N_LOC, pops, f_sub, params["prob_transmission_peak"], profile,
            max_days, R0, init, LW, lb, f_calibration=base_f)

    print("  Running SI sensitivity simulations...")

    fig = plt.figure(figsize=(7.2, 5.2))
    gs  = gridspec.GridSpec(2, 2, hspace=0.55, wspace=0.48,
                            left=0.09, right=0.97, top=0.95, bottom=0.09)

    # ── A: R0 sensitivity ─────────────────────────────────────────────────
    ax = fig.add_subplot(gs[0, 0])
    R0_vals  = [1.5, 2.0, 2.5, 3.0]
    R0_cols  = [OKABE_ITO[2], OKABE_ITO[0], OKABE_ITO[4], OKABE_ITO[5]]
    for R0v, col in zip(R0_vals, R0_cols):
        s = _run(R0v, infect_profile, initial_infections, LB)
        R_sys = np.array([R_system(s["R_matrices"][t]) for t in range(T_test)])
        vld = R_sys > 0
        ax.plot(np.where(vld)[0], R_sys[vld], color=col, lw=0.9,
                label=f"$R_0={R0v}$")
    ax.axhline(1.0, color="0.55", ls="--", lw=0.8)
    ax.set_xlabel("Day $t$"); ax.set_ylabel("$\\mathcal{R}(t)$")
    ax.legend(fontsize=6, borderpad=0.3)
    _panel_label(ax, "A")

    # ── B: initial condition sensitivity ─────────────────────────────────
    ax = fig.add_subplot(gs[0, 1])
    n_seed = initial_infections.sum()
    for scen_name, lbl, col in (("hub", f"seed hub only (L{i_hub + 1})", OKABE_ITO[0]),
                                ("periph", f"seed peripheral only (L{i_per + 1})", OKABE_ITO[5]),
                                ("uniform", "uniform seeding", OKABE_ITO[4])):
        init_v = np.zeros(N_LOC)
        if scen_name == "hub":
            init_v[i_hub] = n_seed
        elif scen_name == "periph":
            init_v[i_per] = n_seed
        else:
            init_v[:] = n_seed / N_LOC
        s = _run(params["R0_target"], infect_profile, init_v, LB)
        ax.plot(s["incidence"].sum(axis=1) / 1e3, color=col, lw=0.9, label=lbl)
    ax.set_xlabel("Day $t$"); ax.set_ylabel("Incidence ($\\times 10^3$)")
    ax.legend(fontsize=6, borderpad=0.3)
    _panel_label(ax, "B")

    # ── C: between-location contact fraction ──────────────────────────────
    ax = fig.add_subplot(gs[1, 0])
    for bf, col in zip([0.15, 0.30, 0.45], [OKABE_ITO[2], OKABE_ITO[4], OKABE_ITO[5]]):
        s = _run(params["R0_target"], infect_profile, initial_infections, LW * bf)
        R_sys = np.array([R_system(s["R_matrices"][t]) for t in range(T_test)])
        vld = R_sys > 0
        ax.plot(np.where(vld)[0], R_sys[vld], color=col, lw=0.9,
                label=f"$\\lambda_B/\\lambda_W={bf}$")
    ax.axhline(1.0, color="0.55", ls="--", lw=0.8)
    ax.set_xlabel("Day $t$"); ax.set_ylabel("$\\mathcal{R}(t)$")
    ax.legend(fontsize=6, borderpad=0.3)
    _panel_label(ax, "C")

    # ── D: infectiousness profile ─────────────────────────────────────────
    ax = fig.add_subplot(gs[1, 1])
    profile_specs = [
        ("$p(a_E)$: mean 2.5 d, sd 1.0 d", 2.5, 1.0, OKABE_ITO[0]),
        ("$p(a_E)$: mean 5.5 d, sd 1.8 d", 5.5, 1.8, OKABE_ITO[4]),
        ("$p(a_E)$: mean 7.0 d, sd 2.5 d", 7.0, 2.5, OKABE_ITO[5]),
    ]
    for lbl, p_mean, p_sd, col in profile_specs:
        gtp_v = discretise_gamma(p_mean, p_sd, max_days)
        s = _run(params["R0_target"], gtp_v, initial_infections, LB)
        ax.plot(s["incidence"].sum(axis=1) / 1e3, color=col, lw=0.9, label=lbl)
    ax.set_xlabel("Day $t$")
    ax.set_ylabel("Incidence ($\\times 10^3$)")
    ax.text(0.03, 0.97,
            "Single $p(a_E)$ varied;\nmobility held constant.",
            transform=ax.transAxes, fontsize=4.5, va="top", color="0.4", style="italic")
    ax.legend(fontsize=5.5, borderpad=0.3)
    _panel_label(ax, "D")

    plt.savefig(f"{save_prefix}_SI7_sensitivity.pdf", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved {save_prefix}_SI7_sensitivity.pdf")


def plot_SI_3d_earlypeak(sim_A, sim_B, save_prefix="fig"):
    """Fig SI1: pairwise new infections E_{kj} at the early and peak days (phase_days)
    of the dense-urban and sparse-national scenarios.

    Panels (2×2):
      a  Dense urban     — early day
      b  Dense urban     — epidemic peak
      c  Sparse national — early day
      d  Sparse national — epidemic peak
    """
    ph_A = phase_days(sim_A["incidence"])
    ph_B = phase_days(sim_B["incidence"])
    imat_A = sim_A["incidence_matrix"]   # shape (T, N, N)
    imat_B = sim_B["incidence_matrix"]

    fig = plt.figure(figsize=(7.2, 6.4))
    gs  = gridspec.GridSpec(2, 2, hspace=0.30, wspace=0.25,
                            left=0.04, right=0.98, top=0.92, bottom=0.04)

    panel_specs = [
        (0, 0, "a", imat_A[ph_A["early"]], f"Dense urban — early (day {ph_A['early']})"),
        (0, 1, "b", imat_A[ph_A["peak"]],  f"Dense urban — peak  (day {ph_A['peak']})"),
        (1, 0, "c", imat_B[ph_B["early"]], f"Sparse national — early (day {ph_B['early']})"),
        (1, 1, "d", imat_B[ph_B["peak"]],  f"Sparse national — peak  (day {ph_B['peak']})"),
    ]

    for row, col, lbl, mat, title in panel_specs:
        ax = fig.add_subplot(gs[row, col], projection="3d")
        _bar3d_inc(ax, mat, title)
        ax.text2D(-0.04, 1.02, lbl, transform=ax.transAxes,
                  fontsize=10, fontweight="bold", va="top", ha="left")

    plt.savefig(f"{save_prefix}_SI8_3d_earlypeak.pdf", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved {save_prefix}_SI8_3d_earlypeak.pdf")


def plot_SI5_combined(sim_A, sim_B, save_prefix="fig"):
    """Fig SI27: meeting-location reproduction numbers R^l_meeting (Eq 50).

    Panels (2×2):
      a  Dense urban     — R^l_meeting(t) heatmap
      b  Dense urban     — R^j_in against R^l_meeting at the epidemic peak
      c  Sparse national — R^l_meeting(t) heatmap
      d  Sparse national — R^j_in against R^l_meeting at the epidemic peak
    """
    fig = plt.figure(figsize=(7.2, 5.5))
    gs  = gridspec.GridSpec(2, 2, hspace=0.62, wspace=0.52,
                            left=0.08, right=0.97, top=0.93, bottom=0.08)

    scenarios = [
        (sim_A, "Dense urban",     0, OKABE_ITO[0]),
        (sim_B, "Sparse national", 1, OKABE_ITO[4]),
    ]
    panel_pairs = [("a", "b"), ("c", "d")]

    for (sim, name, row, col_theme), (lbl_heat, lbl_scat) in zip(scenarios, panel_pairs):
        inc      = sim["incidence"]
        R_mats   = sim["R_matrices"]
        R_meet_s = sim["R_meeting_series"]
        T, N  = inc.shape
        peak  = int(inc.sum(axis=1).argmax())
        loc   = [f"L{i+1}" for i in range(N)]

        # Heatmap panel
        ax = fig.add_subplot(gs[row, 0])
        im = ax.imshow(R_meet_s.T, cmap="YlOrRd", aspect="auto", origin="upper")
        ax.set_yticks(range(N)); ax.set_yticklabels(loc, fontsize=5)
        ax.set_xlabel("Day $t$"); ax.set_ylabel("Activity location $l$")
        cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cb.ax.set_title("$R^l_{\\rm meeting}$", fontsize=6, pad=3)
        _panel_label(ax, lbl_heat)

        # Scatter panel
        ax = fig.add_subplot(gs[row, 1])
        R_in_p   = R_inward(R_mats[peak])
        R_meet_p = R_meet_s[peak]
        for j in range(N):
            ax.scatter(R_in_p[j], R_meet_p[j], s=35,
                       color=OKABE_ITO[j % len(OKABE_ITO)],
                       edgecolors="k", linewidths=0.4, zorder=5)
            ax.annotate(loc[j], (R_in_p[j], R_meet_p[j]),
                        fontsize=5.5, xytext=(3, 3), textcoords="offset points",
                        color=OKABE_ITO[j % len(OKABE_ITO)])
        all_v = np.concatenate([R_in_p, R_meet_p])
        vmax_v = all_v.max() * 1.1
        ax.plot([0, vmax_v], [0, vmax_v], color="0.55", ls="--", lw=0.8)
        ax.set_xlabel("$R^j_{\\rm in}$  (residence-based)")
        ax.set_ylabel("$R^l_{\\rm meeting}$  (activity-based)")
        _panel_label(ax, lbl_scat)

    fname = f"{save_prefix}_SI5_meeting_combined.png"
    plt.savefig(fname, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved {fname}")


def plot_SI_R_comparison(sim, city_data, R_independent, save_prefix="fig"):
    """Fig SI26: independent estimates R̂^j_ind(t) against R^j_out(t) and R^j_in(t).

    R̂^j_ind(t) summarises the ESTIMATOR_WINDOW days ending at t, so every comparison
    uses the matching trailing means of R^j_out and R^j_in.  Row 1
    shows the three quantities themselves.

    3 × 3 panel layout
      A  R^j_out(t)       B  R^j_in(t)        C  R̂^j_ind(t)
      D  (R̂^j_ind − R̄^j_out)/R̄^j_out (%)    E  (R̂^j_ind − R̄^j_in)/R̄^j_in (%)
      F  per-location mean relative difference ± 1 sd, over the days with
         ρ(R(t)) > RHO_FLOOR on which every location has an estimate
      G  R̄^j_out (solid) and R̂^j_ind (dashed); hub, mid and peripheral highlighted
      H  R̄^j_in (solid) and R̂^j_ind (dashed)
      I  R̂^j_ind against R̄^j_in, all locations and days, coloured by epidemic phase
    (R̄ = trailing mean over ESTIMATOR_WINDOW days.)
    """
    inc    = sim["incidence"]           # (T, N)
    R_mats = sim["R_matrices"]
    node_types = city_data[3]
    T, N   = inc.shape

    # ── derived series ────────────────────────────────────────────────────
    R_out_s = np.array([R_outward(R_mats[t]) for t in range(T)])   # (T, N)
    R_in_s  = np.array([R_inward(R_mats[t])  for t in range(T)])   # (T, N)
    rho_s   = np.array([R_system(R_mats[t])  for t in range(T)])
    R_out_tm = np.column_stack([trailing_mean(R_out_s[:, j], ESTIMATOR_WINDOW)
                                for j in range(N)])
    R_in_tm  = np.column_stack([trailing_mean(R_in_s[:, j], ESTIMATOR_WINDOW)
                                for j in range(N)])
    R_ind   = np.asarray(R_independent, dtype=float)                 # (T, N)

    # ── location roles ────────────────────────────────────────────────────
    i_hub, i_mid, i_per, _, _ = representative_locs(city_data)
    peak    = int(inc.sum(axis=1).argmax())
    loc_lbl = [f"L{i+1}" for i in range(N)]

    col_hub = OKABE_ITO[0]   # orange
    col_mid = OKABE_ITO[2]   # bluish-green
    col_per = OKABE_ITO[5]   # vermillion

    # ── shared colour scale for row-1 heatmaps ────────────────────────────
    r_vals  = np.concatenate([R_out_s[R_out_s > 0].ravel(),
                              R_in_s[R_in_s > 0].ravel(),
                              R_ind[np.isfinite(R_ind)].ravel()])
    vmin_r  = 0.0
    vmax_r  = float(np.percentile(r_vals, 97)) if r_vals.size else 3.0
    cmap_r  = "RdYlBu_r"

    # ── relative differences (%) against the trailing means ───────────────
    def _rel_diff(num, den):
        """(num - den)/den × 100, NaN where either is undefined."""
        valid = np.isfinite(num) & np.isfinite(den) & (den > 1e-4)
        out   = np.full(num.shape, np.nan)
        out[valid] = (num[valid] - den[valid]) / den[valid] * 100.0
        return out

    diff_out = _rel_diff(R_ind, R_out_tm)   # (T, N)
    diff_in  = _rel_diff(R_ind, R_in_tm)    # (T, N)
    both = np.concatenate([diff_out[np.isfinite(diff_out)],
                           diff_in[np.isfinite(diff_in)]])
    blim = max(float(np.percentile(np.abs(both), 97)), 10.0) if both.size else 100.0

    # ── epidemic phase array for scatter colouring ────────────────────────
    phase      = np.zeros(T, dtype=float)
    phase[:peak]  = np.linspace(0.0, 0.5, peak)
    phase[peak:]  = np.linspace(0.5, 1.0, T - peak)

    fig = plt.figure(figsize=(7.2, 9.6))
    gs  = gridspec.GridSpec(3, 3, hspace=0.72, wspace=0.52,
                            left=0.09, right=0.96, top=0.96, bottom=0.05)

    def _heatmap(ax, data, cmap, vmin, vmax, title, ylabel="Location"):
        im = ax.imshow(data.T, aspect="auto", origin="upper",
                       cmap=cmap, vmin=vmin, vmax=vmax,
                       interpolation="nearest")
        ax.set_yticks(range(N))
        ax.set_yticklabels(loc_lbl, fontsize=5)
        ax.set_xlabel("Day $t$", fontsize=7)
        ax.set_ylabel(ylabel, fontsize=7)
        ax.axvline(peak, color="k", lw=0.7, ls="--", alpha=0.5)
        ax.text(peak + 0.5, 0.01, f"pk d{peak}",
                fontsize=4.5, color="k", va="bottom",
                transform=ax.get_xaxis_transform())
        return im

    # ─── Row 1: R heatmaps ────────────────────────────────────────────────
    ax_a = fig.add_subplot(gs[0, 0])
    _heatmap(ax_a, np.where(R_out_s > 0, R_out_s, np.nan), cmap_r, vmin_r, vmax_r,
             "$R^j_{\\rm out}(t)$ — outward")
    _panel_label(ax_a, "A")

    ax_b = fig.add_subplot(gs[0, 1])
    _heatmap(ax_b, np.where(R_in_s > 0, R_in_s, np.nan), cmap_r, vmin_r, vmax_r,
             "$R^j_{\\rm in}(t)$ — inward")
    _panel_label(ax_b, "B")

    ax_c = fig.add_subplot(gs[0, 2])
    im_c = _heatmap(ax_c, R_ind, cmap_r, vmin_r, vmax_r,
                    "$\\hat{R}^j_{\\mathrm{ind}}(t)$ — independent estimate")
    _panel_label(ax_c, "C")

    cbar_r = fig.colorbar(im_c, ax=[ax_a, ax_b, ax_c],
                          orientation="horizontal", fraction=0.025,
                          pad=0.18, aspect=50)
    cbar_r.set_label("Reproduction number", fontsize=6)
    cbar_r.ax.tick_params(labelsize=5)

    # ─── Row 2: relative-difference heatmaps ──────────────────────────────
    ax_d = fig.add_subplot(gs[1, 0])
    _heatmap(ax_d, diff_out, "RdBu_r", -blim, blim,
             "$(\\hat{R}^j_{\\mathrm{ind}}-\\bar{R}^j_{\\rm out})/\\bar{R}^j_{\\rm out}$ (%)")
    _panel_label(ax_d, "D")

    ax_e = fig.add_subplot(gs[1, 1])
    im_e = _heatmap(ax_e, diff_in, "RdBu_r", -blim, blim,
                    "$(\\hat{R}^j_{\\mathrm{ind}}-\\bar{R}^j_{\\rm in})/\\bar{R}^j_{\\rm in}$ (%)")
    _panel_label(ax_e, "E")

    cbar_b = fig.colorbar(im_e, ax=[ax_d, ax_e],
                          orientation="horizontal", fraction=0.030,
                          pad=0.18, aspect=40)
    cbar_b.set_label(f"Relative difference (%), $\\bar{{R}}$ = {ESTIMATOR_WINDOW}-day trailing mean",
                     fontsize=6)
    cbar_b.ax.tick_params(labelsize=5)

    # ── F: per-location mean relative difference ──────────────────────────
    ax_f = fig.add_subplot(gs[1, 2])
    win   = (rho_s > RHO_FLOOR) & np.all(np.isfinite(R_ind), axis=1)
    mu_bo = np.array([np.mean(diff_out[win, j]) if win.any() else np.nan for j in range(N)])
    sd_bo = np.array([np.std(diff_out[win, j])  if win.any() else np.nan for j in range(N)])
    mu_bi = np.array([np.mean(diff_in[win, j])  if win.any() else np.nan for j in range(N)])
    sd_bi = np.array([np.std(diff_in[win, j])   if win.any() else np.nan for j in range(N)])
    y     = np.arange(N)
    ax_f.barh(y + 0.18, mu_bo, height=0.34, color=OKABE_ITO[1],
              xerr=sd_bo, error_kw=dict(elinewidth=0.6, capsize=1.5),
              label="vs $\\bar{R}^j_{\\rm out}$", edgecolor="none")
    ax_f.barh(y - 0.18, mu_bi, height=0.34, color=OKABE_ITO[0],
              xerr=sd_bi, error_kw=dict(elinewidth=0.6, capsize=1.5),
              label="vs $\\bar{R}^j_{\\rm in}$", edgecolor="none")
    ax_f.axvline(0, color="k", lw=0.8)
    ax_f.set_yticks(y)
    ax_f.set_yticklabels(loc_lbl, fontsize=5.5)
    ax_f.set_xlabel(f"Mean relative difference (%) ± 1 s.d.\n"
                    f"({int(win.sum())} days: $\\rho>{RHO_FLOOR}$, all $\\hat{{R}}^j$ defined)",
                    fontsize=6)
    ax_f.legend(fontsize=5.5, borderpad=0.3, loc="lower right")
    _panel_label(ax_f, "F")

    # ─── Row 3: time series ───────────────────────────────────────────────
    def _ts_panel(ax, mob_s, mob_lbl, ylabel):
        """All locations; hub, mid and peripheral highlighted."""
        t_arr = np.arange(T)
        for j in range(N):
            lw_j  = 0.4; al_j = 0.25; col_j = "0.65"
            if j == i_hub: lw_j = 1.2; al_j = 1.0; col_j = col_hub
            if j == i_mid: lw_j = 1.2; al_j = 1.0; col_j = col_mid
            if j == i_per: lw_j = 1.2; al_j = 1.0; col_j = col_per
            vm = np.isfinite(mob_s[:, j]) & (mob_s[:, j] > 0)
            vi = np.isfinite(R_ind[:, j])
            if vm.sum() > 3:
                ax.plot(t_arr[vm], mob_s[vm, j],
                        color=col_j, lw=lw_j, alpha=al_j)
            if vi.sum() > 3:
                ax.plot(t_arr[vi], R_ind[vi, j],
                        color=col_j, lw=lw_j, alpha=al_j * 0.8,
                        ls="--")
        ax.axhline(1.0, color="0.5", lw=0.8, ls=":")
        ax.axvline(peak, color="0.5", lw=0.7, ls="--", alpha=0.5)
        ax.set_xlabel("Day $t$", fontsize=7)
        ax.set_ylabel(ylabel, fontsize=7)
        handles = [
            Line2D([0], [0], color=col_hub, lw=1.1,
                   label=f"hub L{i_hub+1} ({node_types[i_hub]})"),
            Line2D([0], [0], color=col_mid, lw=1.1,
                   label=f"mid L{i_mid+1} ({node_types[i_mid]})"),
            Line2D([0], [0], color=col_per, lw=1.1,
                   label=f"periph L{i_per+1} ({node_types[i_per]})"),
            Line2D([0], [0], color="0.5", lw=1.0, ls="-",
                   label=mob_lbl + " (solid)"),
            Line2D([0], [0], color="0.5", lw=1.0, ls="--",
                   label="$\\hat{R}^j_{\\mathrm{ind}}$ (dashed)"),
        ]
        ax.legend(handles=handles, fontsize=4.5, ncol=2,
                  borderpad=0.3, labelspacing=0.12, handlelength=1.2)

    ax_g = fig.add_subplot(gs[2, 0])
    _ts_panel(ax_g, R_out_tm, f"$\\bar{{R}}^j_{{\\rm out}}$ ({ESTIMATOR_WINDOW}-day mean)", "$R(t)$")
    _panel_label(ax_g, "G")

    ax_h = fig.add_subplot(gs[2, 1])
    _ts_panel(ax_h, R_in_tm, f"$\\bar{{R}}^j_{{\\rm in}}$ ({ESTIMATOR_WINDOW}-day mean)", "$R(t)$")
    _panel_label(ax_h, "H")

    # ── I: scatter R̂^j_ind against R̄^j_in, coloured by phase ──────────────
    ax_i = fig.add_subplot(gs[2, 2])
    cmap_ph = plt.cm.plasma
    ok2d = np.isfinite(R_ind) & np.isfinite(R_in_tm) & (R_in_tm > 0)
    for j in range(N):
        ok = ok2d[:, j]
        if ok.sum() < 2:
            continue
        t_idx = np.where(ok)[0]
        ax_i.scatter(R_in_tm[ok, j], R_ind[ok, j],
                     c=phase[t_idx], cmap=cmap_ph, vmin=0, vmax=1,
                     s=3, alpha=0.45, linewidths=0)
    _x = R_in_tm[ok2d]
    _y = R_ind[ok2d]
    lim_x = float(np.percentile(_x, 97)) if _x.size else 3.0
    lim_y = float(np.percentile(_y, 99)) if _y.size else 3.0
    lim_max = max(lim_x, lim_y)
    ax_i.plot([0, lim_max], [0, lim_max], color="0.4", lw=0.9, ls="--",
              zorder=5, label="1:1")
    if _x.size > 20:
        bins   = np.linspace(0, lim_max, 25)
        bx     = 0.5 * (bins[:-1] + bins[1:])
        by_mu  = np.full(bx.shape, np.nan)
        for b, (lo, hi) in enumerate(zip(bins[:-1], bins[1:])):
            in_bin = (_x >= lo) & (_x < hi)
            if in_bin.any():
                by_mu[b] = _y[in_bin].mean()
        valid  = np.isfinite(by_mu)
        ax_i.plot(bx[valid], by_mu[valid], color=OKABE_ITO[0],
                  lw=1.4, zorder=6, label="Bin mean")
    cbar_ph = fig.colorbar(plt.cm.ScalarMappable(norm=Normalize(0, 1), cmap=cmap_ph),
                           ax=ax_i, fraction=0.046, pad=0.04)
    cbar_ph.set_label("Epidemic phase\n(0=early, 1=late)", fontsize=5)
    cbar_ph.ax.tick_params(labelsize=4.5)
    ax_i.set_xlabel(f"$\\bar{{R}}^j_{{\\rm in}}(t)$ ({ESTIMATOR_WINDOW}-day mean)", fontsize=7)
    ax_i.set_ylabel("$\\hat{R}^j_{\\mathrm{ind}}(t)$", fontsize=7)
    ax_i.set_xlim(0, lim_x)
    ax_i.set_ylim(0, lim_y)
    ax_i.legend(fontsize=5.5, borderpad=0.3, loc="upper left")
    _panel_label(ax_i, "I")

    plt.savefig(f"{save_prefix}_SI_R_comparison.pdf",
                dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved {save_prefix}_SI_R_comparison.pdf")
