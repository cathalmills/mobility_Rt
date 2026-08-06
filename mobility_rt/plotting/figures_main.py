"""mobility_rt.figures_main."""
import warnings
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
from mobility_rt.config import OKABE_ITO
from mobility_rt.estimators import _compute_naive_suite
from mobility_rt.geometry import representative_locs
from mobility_rt.kernel import R_inward, R_outward, R_system
from mobility_rt.plotting.style import _bar3d_Rkj, _bar3d_inc, _panel_label, _panel_label_3d
from mobility_rt.simulation import compute_effective_populations_series
from mobility_rt.spectral import amplification_envelope, reactivity, sensitivity_elasticity, spectral_analysis
from mobility_rt.transmissibility import _compute_power_mean_spectrum, source_sink_analysis
from mobility_rt.type_reproduction import _group_indices, type_reproduction_number_group, type_reproduction_numbers


def plot_fig2(sim, city_data, f_jk, gen_time_pmf, max_days, scenario_name,
              save_prefix="fig"):
    """Figure 2: Simulated epidemic — mobility inputs and epidemic outputs.

    Panels:
      a  Mean mobility matrix f̄_{jk}
      b  Home fraction f_{jj}(t) over time (shows weekly commuting cycles)
      c  Incidence E_j(t,0) by location and time
      d  Effective susceptibles S^{eff}_m(t) at meeting locations
      e  System R(t) = ρ(R(t)) with total incidence on twin axis
      f  Column elasticity Σ_j ε_{kj}(t) as heatmap (infector × time)
    """
    inc      = sim["incidence"]
    R_mats   = sim["R_matrices"]
    S_series = sim["susceptibles"]
    coords, pops, dists, node_types, meta = city_data
    T, N = inc.shape

    S_eff_s, _ = compute_effective_populations_series(
        f_jk, S_series, inc, gen_time_pmf, max_days)
    R_sys = np.array([R_system(R_mats[t]) for t in range(T)])
    loc   = [f"L{i+1}" for i in range(N)]

    fig = plt.figure(figsize=(7.2, 4.8))
    gs  = gridspec.GridSpec(2, 3, hspace=0.58, wspace=0.52,
                            left=0.09, right=0.96, top=0.97, bottom=0.10)

    # ── a: mean mobility matrix ────────────────────────────────────────────
    ax = fig.add_subplot(gs[0, 0])
    im = ax.pcolormesh(np.arange(N+1)-0.5, np.arange(N+1)-0.5,
                       f_jk.mean(axis=0), cmap="Blues", shading="flat")
    ax.set_xlim(-0.5, N-0.5); ax.set_ylim(N-0.5, -0.5)
    ax.set_xticks(range(N)); ax.set_xticklabels(loc, fontsize=5, rotation=45)
    ax.set_yticks(range(N)); ax.set_yticklabels(loc, fontsize=5)
    ax.set_xlabel("Activity location $k$"); ax.set_ylabel("Residence $j$")
    cb_a = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb_a.ax.set_title("$\\bar{f}_{jk}$", fontsize=6, pad=3)
    _panel_label(ax, "A")

    # ── b: home fraction f_{jj}(t) over time ─────────────────────────────
    ax = fig.add_subplot(gs[1, 0])
    diag_f = np.array([[f_jk[t, j, j] for j in range(N)]
                        for t in range(T)]).T   # shape (N, T)
    im = ax.pcolormesh(np.arange(T+1)-0.5, np.arange(N+1)-0.5,
                       diag_f, cmap="RdYlGn", shading="flat",
                       vmin=0.3, vmax=1.0)
    ax.set_xlim(-0.5, T-0.5); ax.set_ylim(N-0.5, -0.5)
    ax.set_yticks(range(N)); ax.set_yticklabels(loc, fontsize=5)
    ax.set_xlabel("Day $t$"); ax.set_ylabel("Location")
    cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.ax.set_title("$f_{jj}(t)$", fontsize=6, pad=3)
    _panel_label(ax, "D")

    # ── b: incidence E_j(t,0) heatmap ─────────────────────────────────────
    ax = fig.add_subplot(gs[0, 1])
    im = ax.pcolormesh(np.arange(T+1)-0.5, np.arange(N+1)-0.5,
                       inc.T, cmap="YlOrRd", shading="flat")
    ax.set_xlim(-0.5, T-0.5); ax.set_ylim(N-0.5, -0.5)
    ax.set_yticks(range(N)); ax.set_yticklabels(loc, fontsize=5)
    ax.set_xlabel("Day $t$"); ax.set_ylabel("Residence")
    cb_c = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb_c.ax.set_title("Incidence", fontsize=6, pad=3)
    _panel_label(ax, "B")

    # ── e: effective susceptibles S^l_eff(t) at meeting locations ──────────
    ax = fig.add_subplot(gs[1, 1])
    im = ax.pcolormesh(np.arange(T+1)-0.5, np.arange(N+1)-0.5,
                       S_eff_s.T / 1e3, cmap="Blues", shading="flat")
    ax.set_xlim(-0.5, T-0.5); ax.set_ylim(N-0.5, -0.5)
    ax.set_yticks(range(N)); ax.set_yticklabels(loc, fontsize=5)
    ax.set_xlabel("Day $t$"); ax.set_ylabel("Activity location $l$")
    cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.ax.set_title("$S^l_{\\rm eff}$ ($\\times 10^3$)", fontsize=6, pad=3)
    _panel_label(ax, "E")

    # ── c: system R(t) with total incidence on twin axis ──────────────────
    ax  = fig.add_subplot(gs[0, 2])
    ax2 = ax.twinx()
    ax2.spines["right"].set_visible(True)
    ax2.spines["top"].set_visible(False)
    vld = R_sys > 0
    # Risk-aware reproduction number E(t) = Σ_j (R^j_out)² / Σ_k R^k_out
    R_out_series = np.array([R_outward(R_mats[t]) for t in range(T)])
    E_t = np.array([np.sum(R_out_series[t]**2) / (np.sum(R_out_series[t]) + 1e-300)
                    for t in range(T)])
    vE = E_t > 0
    ax.plot(np.where(vld)[0], R_sys[vld], color=OKABE_ITO[4], lw=0.9,
            label="$\\mathcal{R}(t)$")
    ax.plot(np.where(vE)[0], E_t[vE], color=OKABE_ITO[6], lw=0.9, ls="--",
            label="$\\mathcal{E}(t)$")
    ax.axhline(1.0, color="0.55", ls="--", lw=0.8)
    total_inc = inc.sum(axis=1)
    ax2.fill_between(range(T), total_inc / 1e3, alpha=0.22, color=OKABE_ITO[1])
    ax2.plot(total_inc / 1e3, color=OKABE_ITO[1], lw=1.0)
    ax.set_ylabel("$\\mathcal{R}(t)$,  $\\mathcal{E}(t)$", color=OKABE_ITO[4])
    ax2.set_ylabel("Incidence ($\\times 10^3$)", color=OKABE_ITO[1])
    ax.set_xlabel("Day $t$")
    ax.set_ylim(0, max(2.0, float(R_sys[vld].max()) * 1.1) if vld.any() else 3.5)
    ax.tick_params(axis="y", labelcolor=OKABE_ITO[4])
    ax2.tick_params(axis="y", labelcolor=OKABE_ITO[1])
    ax.legend(fontsize=5.5, borderpad=0.3, labelspacing=0.12, loc="upper right")
    # ── annotation: R0, attack rate, cumulative ────────────────────────
    R0_ann    = float(R_sys[vld][0]) if vld.any() else 0.0
    total_pop = float(pops.sum())
    cum_inf   = float(total_inc.sum())
    att_rate  = cum_inf / total_pop * 100
    ax.text(0.97, 0.55,
            f"$\\mathcal{{R}}_0 = {R0_ann:.2f}$\n"
            f"Attack rate = {att_rate:.1f}%\n"
            f"Cumulative = {cum_inf/1e6:.2f}M",
            transform=ax.transAxes, fontsize=5.5, ha="right", va="top",
            color="0.2",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.75", alpha=0.85,
                      lw=0.6))
    _panel_label(ax, "C")

    # ── f: column elasticity Σ_j ε_{kj}(t) as heatmap ────────────────────
    ax = fig.add_subplot(gs[1, 2])
    elas = np.zeros((T, N))
    for t in range(T):
        elas[t] = sensitivity_elasticity(R_mats[t])["elasticity"].sum(axis=1)
    vmax_e = np.percentile(elas[elas > 0], 97) if (elas > 0).any() else 1.0
    im = ax.pcolormesh(np.arange(T+1)-0.5, np.arange(N+1)-0.5,
                       elas.T, cmap="YlOrRd", shading="flat",
                       vmin=0, vmax=vmax_e)
    ax.set_xlim(-0.5, T-0.5); ax.set_ylim(N-0.5, -0.5)
    ax.set_yticks(range(N)); ax.set_yticklabels(loc, fontsize=5)
    ax.set_xlabel("Day $t$"); ax.set_ylabel("Infector $k$")
    cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.ax.set_title("$\\sum_j \\varepsilon_{kj}$", fontsize=6, pad=3)
    _panel_label(ax, "F")

    plt.savefig(f"{save_prefix}_02_overview.pdf", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved {save_prefix}_02_overview.pdf")


def plot_fig3(sim, city_data, R_independent, gt_snaps, w_within, w_between,
              scenario_name, save_prefix="fig"):
    """Figure 3: Taxonomy of R types and generation time distributions.

    Panels (3×3 grid, row 2 via GridSpecFromSubplotSpec for equal g/h widths):
      a  GT distributions at peak — g_{jj} within, g^j_out outward, g^j_in inward
         for hub vs peripheral locations
      b  R^j_out(t) heatmap (infector × time) — plasma colormap
      c  R^j_in(t) heatmap (infectee × time) — viridis colormap
      d  3D bar chart of R_{kj} at epidemic peak
      e  3D bar chart of pairwise new infections E_{kj} at epidemic peak
      f  Source–sink decomposition at peak (row 1, col 2)
      g  Bias: R̂^j_ind (dashed) vs R^j_in (solid) for hub/mid/peripheral
      h  R^j_out vs R̂^j_ind comparison for hub/mid/peripheral
    Letters follow strict left-to-right, top-to-bottom reading order.
    Panels g and h are equal width (each half of the bottom row).
    """
    inc    = sim["incidence"]
    inc_mat= sim["incidence_matrix"]   # shape (T, N, N)
    R_mats = sim["R_matrices"]
    coords, pops, dists, node_types, meta = city_data
    T, N = inc.shape

    R_out_s = np.array([R_outward(R_mats[t]) for t in range(T)])
    R_in_s  = np.array([R_inward(R_mats[t])  for t in range(T)])
    peak    = int(inc.sum(axis=1).argmax())
    early   = max(1, peak // 3)
    late    = min(T - 1, peak + 30)
    loc     = [f"L{i+1}" for i in range(N)]

    i_hub, i_mid, i_per, show_locs, show_lbls = representative_locs(city_data)

    fig = plt.figure(figsize=(14.0, 6.0))
    gs  = gridspec.GridSpec(2, 5, hspace=0.52, wspace=0.62,
                            left=0.06, right=0.98, top=0.97, bottom=0.10)

    # ── a: GT distributions at peak ───────────────────────────────────────
    ax   = fig.add_subplot(gs[0, 0])
    days = np.arange(len(w_within))
    gt_p  = gt_snaps["peak"]
    g_pw  = gt_p["g_pairwise"]
    g_out = gt_p["g_outward"]
    g_in  = gt_p["g_inward"]
    # Universal GT: g_univ = p/∫p (= w_within since w_within = infect_profile)
    g_univ = w_within / w_within.sum() if w_within.sum() > 0 else w_within.copy()
    GT_univ = float(np.sum(days * g_univ))

    # Show a single location (hub) — within, outward, inward
    g_kk = g_pw[:, i_hub, i_hub]
    if g_kk.sum() > 0.5:
        ax.plot(days, g_kk,            color=OKABE_ITO[0], lw=1.3,
                label=f"$g_{{kk}}$ within")
    if g_out[:, i_hub].sum() > 0.5:
        ax.plot(days, g_out[:, i_hub], color=OKABE_ITO[1], lw=1.0, ls="--",
                label=f"$g^j_{{\\rm out}}$")
    if g_in[:, i_hub].sum() > 0.5:
        ax.plot(days, g_in[:, i_hub],  color=OKABE_ITO[5], lw=1.0, ls=":",
                label=f"$g^j_{{\\rm in}}$")
    ax.plot(days, g_univ, color="0.35", lw=1.6, ls="-", zorder=0, alpha=0.45,
            label=f"$p/\\int p$ ({GT_univ:.1f}d)")

    ax.set_xlabel("Age $a_E$ (days)", fontsize=6)
    ax.set_ylabel("Probability", fontsize=6)
    ax.set_title("GT at peak: within / out / in",
                 fontsize=5.5, pad=3)
    ax.legend(fontsize=4.0, ncol=2, borderpad=0.2, labelspacing=0.12,
              handlelength=1.0)
    ax.text(0.03, 0.03,
            "$g_{kj}=p/\\int p$ universally;\nall curves overlay.",
            transform=ax.transAxes, fontsize=4.0, va="bottom",
            color="0.4", style="italic")
    _panel_label(ax, "A")

    # ── b: R^j_out(t) heatmap ─────────────────────────────────────────────
    ax = fig.add_subplot(gs[0, 1])  # row 0, col 1
    pos_out = R_out_s[R_out_s > 0]
    vmin_out = np.percentile(pos_out, 2)  if pos_out.size else 0.0
    vmax_out = np.percentile(pos_out, 97) if pos_out.size else 3.0
    im = ax.pcolormesh(np.arange(T + 1) - 0.5, np.arange(N + 1) - 0.5,
                       R_out_s.T, cmap="plasma", shading="flat",
                       vmin=vmin_out, vmax=vmax_out)
    ax.set_xlim(-0.5, T - 0.5); ax.set_ylim(N - 0.5, -0.5)
    ax.set_yticks(range(N)); ax.set_yticklabels(loc, fontsize=5)
    ax.set_xlabel("Day $t$"); ax.set_ylabel("Infector $j$")
    cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.ax.set_title("$R^j_{\\rm out}$", fontsize=6, pad=3)
    _panel_label(ax, "B")

    # ── c: R^j_in(t) heatmap ──────────────────────────────────────────────
    ax = fig.add_subplot(gs[0, 2])  # row 0, col 2
    pos_in = R_in_s[R_in_s > 0]
    vmin_in = np.percentile(pos_in, 2)  if pos_in.size else 0.0
    vmax_in = np.percentile(pos_in, 97) if pos_in.size else 3.0
    im = ax.pcolormesh(np.arange(T + 1) - 0.5, np.arange(N + 1) - 0.5,
                       R_in_s.T, cmap="viridis", shading="flat",
                       vmin=vmin_in, vmax=vmax_in)
    ax.set_xlim(-0.5, T - 0.5); ax.set_ylim(N - 0.5, -0.5)
    ax.set_yticks(range(N)); ax.set_yticklabels(loc, fontsize=5)
    ax.set_xlabel("Day $t$"); ax.set_ylabel("Infectee $j$")
    cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.ax.set_title("$R^j_{\\rm in}$", fontsize=6, pad=3)
    _panel_label(ax, "C")

    # ── d: 3D R_kj at epidemic peak ────────────────────────────────────────
    ax_3d = fig.add_subplot(gs[0, 3], projection="3d")  # row 0, col 3
    _bar3d_Rkj(ax_3d, R_mats[peak], f"peak (day {peak})")
    ax_3d.text2D(-0.08, 1.05, "D", transform=ax_3d.transAxes,
                 fontsize=10, fontweight="bold", va="top", ha="left")

    # ── e: 3D pairwise new infections E_{kj} at epidemic peak ─────────────
    ax_3d2 = fig.add_subplot(gs[0, 4], projection="3d")  # row 0, col 4
    _bar3d_inc(ax_3d2, inc_mat[peak], f"peak (day {peak})")
    ax_3d2.text2D(-0.08, 1.05, "E", transform=ax_3d2.transAxes,
                  fontsize=10, fontweight="bold", va="top", ha="left")

    # ── f: Source–sink decomposition at peak — row 1 col 0 ────────────────
    ax = fig.add_subplot(gs[1, 0])  # row 1, col 0
    ss  = source_sink_analysis(R_mats[peak])
    net = ss["net_export"]
    bc  = [OKABE_ITO[5] if x > 0 else OKABE_ITO[4] for x in net]
    ax.barh(range(N), net, color=bc, height=0.65, edgecolor="none")
    ax.axvline(0, color="k", lw=0.8)
    ax.set_yticks(range(N)); ax.set_yticklabels(loc, fontsize=6)
    ax.set_xlabel("Net export  $R^j_{\\rm out} - R^j_{\\rm in}$")
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(facecolor=OKABE_ITO[5], label="Source"),
                        Patch(facecolor=OKABE_ITO[4], label="Sink")],
              fontsize=6, loc="lower right", borderpad=0.3)
    _panel_label(ax, "F")

    # ── g: Bias — R̂^j_ind vs R^j_in ───────────────────────────────────────
    ax = fig.add_subplot(gs[1, 1])  # row 1, col 1
    for j, lbl, col in zip(show_locs, show_lbls,
                            [OKABE_ITO[0], OKABE_ITO[2], OKABE_ITO[5]]):
        vi = ~np.isnan(R_independent[:, j])
        vm = R_in_s[:, j] > 0
        b  = vi & vm
        if b.sum() > 3:
            ax.plot(np.where(b)[0], R_independent[b, j], "--",
                    color=col, lw=0.9, alpha=0.75)
            ax.plot(np.where(b)[0], R_in_s[b, j], "-",
                    color=col, lw=0.9, label=lbl)
    ax.axhline(1, color="0.55", ls="--", lw=0.8)
    handles, labels_leg = ax.get_legend_handles_labels()
    handles += [Line2D([0],[0], color="0.4", lw=0.9, ls="-"),
                Line2D([0],[0], color="0.4", lw=0.9, ls="--")]
    labels_leg += ["$R^j_{\\rm in}$ (solid)",
                   "$R^j_{\\mathrm{ind}}$ (dashed)"]
    ax.legend(handles=handles, labels=labels_leg, fontsize=5.5, ncol=2,
              borderpad=0.3, labelspacing=0.15)
    ax.set_xlabel("Day $t$"); ax.set_ylabel("$R(t)$")
    _panel_label(ax, "G")

    # ── h: R^j_out vs R̂^j_ind — hub/mid/peripheral ────────────────────────
    ax = fig.add_subplot(gs[1, 2])  # row 1, col 2
    R_out_s2 = np.array([R_outward(R_mats[t]) for t in range(T)])
    for j, lbl, col in zip(show_locs, show_lbls,
                            [OKABE_ITO[0], OKABE_ITO[2], OKABE_ITO[5]]):
        vi = ~np.isnan(R_independent[:, j])
        vo = R_out_s2[:, j] > 0
        b  = vi & vo
        if b.sum() > 3:
            ax.plot(np.where(b)[0], R_independent[b, j], "--",
                    color=col, lw=0.9, alpha=0.75)
            ax.plot(np.where(b)[0], R_out_s2[b, j], "-",
                    color=col, lw=0.9, label=lbl)
    ax.axhline(1, color="0.55", ls="--", lw=0.8)
    handles_h, labels_h = ax.get_legend_handles_labels()
    handles_h += [Line2D([0],[0], color="0.4", lw=0.9, ls="-"),
                  Line2D([0],[0], color="0.4", lw=0.9, ls="--")]
    labels_h  += ["$R^j_{\\rm out}$ (solid)", "$R^j_{\\mathrm{ind}}$ (dashed)"]
    ax.legend(handles=handles_h, labels=labels_h, fontsize=5.0, ncol=1,
              borderpad=0.3, labelspacing=0.15)
    ax.set_xlabel("Day $t$"); ax.set_ylabel("$R(t)$")
    _panel_label(ax, "H")

    # ── i: T_j(t) surface — type reproduction number over time ──────────
    # Compute T_j for all time steps (NaN when ρ(R_{JJ}) >= 1)
    T_ser = np.zeros((T, N))
    for t_idx in range(T):
        T_ser[t_idx] = type_reproduction_numbers(R_mats[t_idx])
    ax_surf = fig.add_subplot(gs[1, 3], projection="3d")  # row 1, col 3
    XX, YY = np.meshgrid(np.arange(T), np.arange(N))
    Z_T = np.ma.masked_invalid(T_ser.T)  # (N, T)
    try:
        surf = ax_surf.plot_surface(XX, YY, Z_T,
                                    cmap="plasma", linewidth=0,
                                    antialiased=True, alpha=0.88,
                                    rstride=1, cstride=max(1, T // 60))
        ax_surf.plot_surface(XX, YY, np.ones_like(Z_T.data),
                             color="grey", alpha=0.08, linewidth=0)
        fig.colorbar(surf, ax=ax_surf, fraction=0.022, pad=0.08, shrink=0.55)
    except Exception:
        pass
    # Clean pane style — transparent fill, subtle edge, no grid lines
    for _pane in [ax_surf.xaxis.pane, ax_surf.yaxis.pane, ax_surf.zaxis.pane]:
        _pane.fill = False
        _pane.set_edgecolor("#cccccc")
    ax_surf.xaxis._axinfo["grid"]["color"] = (0, 0, 0, 0)
    ax_surf.yaxis._axinfo["grid"]["color"] = (0, 0, 0, 0)
    ax_surf.zaxis._axinfo["grid"]["color"] = (0, 0, 0, 0)
    ax_surf.set_xlabel("Day $t$", fontsize=6, labelpad=3)
    ax_surf.set_ylabel("Location", fontsize=6, labelpad=3)
    ax_surf.set_zlabel("")   # suppress rotated z-label; use title instead
    ax_surf.set_title("$R^j_{\\rm type}(t)$", fontsize=6.5, pad=4)
    ax_surf.set_yticks(np.arange(N))
    ax_surf.set_yticklabels([f"L{j+1}" for j in range(N)], fontsize=4.5)
    ax_surf.tick_params(labelsize=5)
    ax_surf.view_init(elev=26, azim=-52)
    ax_surf.text2D(-0.06, 1.06, "I", transform=ax_surf.transAxes,
                   fontsize=10, fontweight="bold", va="top", ha="left")

    # ── j: R^l_meeting(t) tile plot ───────────────────────────────────────
    ax_meet = fig.add_subplot(gs[1, 4])  # row 1, col 4
    R_meet_s = sim["R_meeting_series"]  # (T, N)
    pos_m = R_meet_s[R_meet_s > 0]
    vmin_m = float(np.percentile(pos_m, 2))  if pos_m.size else 0.0
    vmax_m = float(np.percentile(pos_m, 97)) if pos_m.size else 3.0
    im_m = ax_meet.pcolormesh(
        np.arange(T + 1) - 0.5, np.arange(N + 1) - 0.5,
        R_meet_s.T, cmap="cividis", shading="flat",
        vmin=vmin_m, vmax=vmax_m)
    ax_meet.set_xlim(-0.5, T - 0.5)
    ax_meet.set_ylim(N - 0.5, -0.5)
    ax_meet.set_yticks(range(N))
    ax_meet.set_yticklabels(loc, fontsize=5)
    ax_meet.set_xlabel("Day $t$")
    ax_meet.set_ylabel("Meeting loc. $l$")
    ax_meet.set_title("$R^l_{\\rm meeting}(t)$", fontsize=6, pad=3)
    cb_m = plt.colorbar(im_m, ax=ax_meet, fraction=0.046, pad=0.04)
    cb_m.ax.set_title("$R^l_{\\rm meeting}$", fontsize=5.5, pad=2)
    _panel_label(ax_meet, "J")

    plt.savefig(f"{save_prefix}_03_taxonomy.pdf", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved {save_prefix}_03_taxonomy.pdf")


def plot_fig4(sim, city_data, R_mat_alt_t0, scenario_name,
              w_within, w_between, max_days, save_prefix="fig"):
    """Figure 4: Spectral properties.

    Panels (2×3 + full-width row):
      a  Mixing ratio s(t) = |λ_2|/ρ over time with day-of-week overlay
      b  R(t) vs σ(t) over time, shading transient zone where σ>1 and R<1
      c  Amplification envelope A(n)=||R^n||_2 at early/peak/late phases
      d  Top 3 eigenvalue magnitudes |λ_1(t)|, |λ_2(t)|, |λ_3(t)| over time
      e  Eigenvalue condition number κ(R(t)) = ‖v‖₂‖v*‖₂/|vᵀv*| over time (log scale, Eq 38)
      f  Within-fraction heatmap π_j(t) and per-location bar (was e)
    """
    inc    = sim["incidence"]
    R_mats = sim["R_matrices"]
    coords, pops, dists, node_types, meta = city_data
    T, N = inc.shape

    R_sys    = np.array([R_system(R_mats[t]) for t in range(T)])
    specs    = [spectral_analysis(R_mats[t]) for t in range(T)]
    mix_ts   = np.array([s["mixing_ratio"] for s in specs])
    sigma_ts = np.array([reactivity(R_mats[t])["sigma"] for t in range(T)])

    peak  = int(inc.sum(axis=1).argmax())
    early = max(1, peak // 3)
    late  = min(T - 1, peak + 30)

    fig = plt.figure(figsize=(7.2, 7.5))
    gs  = gridspec.GridSpec(3, 6, hspace=0.65, wspace=0.72,
                            left=0.09, right=0.97, top=0.95, bottom=0.07)

    # ── a: mixing ratio s(t) with day-of-week overlay ─────────────────────
    ax  = fig.add_subplot(gs[0, 0:3])
    ax2 = ax.twinx()
    ax2.spines["right"].set_visible(True)
    ax2.spines["right"].set_linewidth(0.8)
    ax2.spines["top"].set_visible(False)
    vm = mix_ts > 0
    ax.plot(np.where(vm)[0], mix_ts[vm], color=OKABE_ITO[2], lw=0.9,
            label="$s(t)$", zorder=5)
    dow_scale = np.array([1.00, 1.00, 1.00, 1.00, 0.95, 0.90, 0.75])
    dow_pattern = np.array([dow_scale[t % 7] for t in range(T)])
    ax2.plot(range(T), dow_pattern, color=OKABE_ITO[0], lw=1.1, ls="--",
             alpha=0.40, label="DoW scaling", zorder=3)
    ax2.set_ylabel("DoW scale", color=OKABE_ITO[0], fontsize=6)
    ax2.tick_params(axis="y", labelcolor=OKABE_ITO[0], labelsize=6,
                    direction="out", length=3, width=0.8)
    # Equally spaced right-hand ticks over the meaningful DoW range (0.75–1.0)
    ax2.set_ylim(0.7, 1.0)
    ax2.set_yticks([0.70, 0.80, 0.90, 1.00])
    ax.set_xlabel("Day $t$")
    ax.set_ylabel("$s(t)=|\\lambda_2|/\\mathcal{R}$")
    ax.set_ylim(0, 1.05)
    lines1, labs1 = ax.get_legend_handles_labels()
    lines2, labs2 = ax2.get_legend_handles_labels()
    ax.legend(lines1+lines2, labs1+labs2, fontsize=5.5, borderpad=0.3)
    _panel_label(ax, "A")

    # ── b: R(t) vs σ(t), first-generation epidemicity A1(1), and E(t) ───────
    ax = fig.add_subplot(gs[0, 3:6])
    vR   = R_sys > 0
    vsig = sigma_ts > 0
    t_arr = np.arange(T)
    # A1(1) = max_k R^k_out(t) — first-generation epidemicity (ℓ1 envelope, n=1)
    A1_1_ts = np.array([float(np.max(R_outward(R_mats[t]))) for t in range(T)])
    vA1  = A1_1_ts > 0
    # E(t) = risk-aware reproduction number = Σ_j (R^j_out)² / Σ_k R^k_out
    R_out_b = np.array([R_outward(R_mats[t]) for t in range(T)])
    E_t_b   = np.array([np.sum(R_out_b[t]**2) / (np.sum(R_out_b[t]) + 1e-300)
                        for t in range(T)])
    vEb = E_t_b > 0
    ax.plot(t_arr[vR],   R_sys[vR],      color=OKABE_ITO[4], lw=0.9,
            label="$\\mathcal{R}(t)$")
    ax.plot(t_arr[vsig], sigma_ts[vsig], color=OKABE_ITO[5], lw=0.9,
            label="$\\sigma(t)$")
    ax.plot(t_arr[vA1],  A1_1_ts[vA1],  color=OKABE_ITO[1], lw=0.9, ls="-.",
            label="$\\mathcal{A}_1(1)=\\max_k R^k_{\\rm out}$")
    ax.plot(t_arr[vEb],  E_t_b[vEb],    color=OKABE_ITO[6], lw=0.9, ls="--",
            label="$\\mathcal{E}(t)=X(1,t)$")
    ax.axhline(1.0, color="0.55", ls="--", lw=0.8)
    transient_mask = (sigma_ts > 1) & (R_sys < 1)
    if transient_mask.any():
        ax.fill_between(t_arr, 1.0, sigma_ts,
                        where=transient_mask,
                        color="orange", alpha=0.15,
                        label="Transient zone: $\\sigma>1$, $\\mathcal{R}<1$")
    ax.set_xlabel("Day $t$"); ax.set_ylabel("Value")
    ax.legend(fontsize=5.0, borderpad=0.3, labelspacing=0.12, ncol=2)
    # Gap summary text box: mean difference and ratio between σ, X(1,t) and R(t)
    _active_b = R_sys > 0.05
    if _active_b.sum() > 5:
        _sdiff  = float(np.nanmean((sigma_ts - R_sys)[_active_b]))
        _sratio = float(np.nanmean((sigma_ts / (R_sys + 1e-300))[_active_b]))
        _ediff  = float(np.nanmean((E_t_b - R_sys)[_active_b]))
        _eratio = float(np.nanmean((E_t_b / (R_sys + 1e-300))[_active_b]))
        _gap_txt = (
            f"Mean $\\sigma - \\mathcal{{R}}$: ${_sdiff:+.3f}$"
            f"  (ratio $= {_sratio:.3f}$)\n"
            f"Mean $X(1,t) - \\mathcal{{R}}$: ${_ediff:+.3f}$"
            f"  (ratio $= {_eratio:.3f}$)"
        )
        from matplotlib.transforms import blended_transform_factory as _btf
        _tr_b = _btf(ax.transAxes, ax.transData)
        ax.text(0.98, 1.06, _gap_txt,
                transform=_tr_b, fontsize=5.0, va="bottom", ha="right",
                bbox=dict(boxstyle="round,pad=0.30", fc="white", ec="0.65",
                          alpha=0.93, lw=0.6))
    _panel_label(ax, "B")

    # ── c: Amplification envelope A(n) (ℓ2) and A1(n) (ℓ1) ───────────────
    ax = fig.add_subplot(gs[1, 0:2])
    phase_specs = [
        (early, "early",  OKABE_ITO[2]),
        (peak,  "peak",   OKABE_ITO[5]),
        (late,  "late",   OKABE_ITO[0]),
    ]
    n_max_env = 20
    for t_phase, phase_name, col in phase_specs:
        env  = amplification_envelope(R_mats[t_phase], n_max=n_max_env)
        rho  = env["rho"]
        rho_n = env["rho_n"]
        # ℓ2-norm envelope A(n) = ‖R^n‖_2
        ax.plot(env["n"], env["A"] / (rho_n + 1e-300), color=col, lw=1.0,
                label=f"$A(n)$ {phase_name}")
        # ℓ1-norm envelope A1(n) = max row sum of R^n
        Rn = np.eye(R_mats[t_phase].shape[0])
        A1_n = np.zeros(n_max_env + 1)
        for n in range(n_max_env + 1):
            A1_n[n] = float(np.max(Rn.sum(axis=1)))
            Rn = Rn @ R_mats[t_phase]
        ax.plot(env["n"], A1_n / (rho_n + 1e-300), color=col, lw=0.8, ls=":",
                label=f"$\\mathcal{{A}}_1(n)$ {phase_name}")
    ax.axhline(1.0, color="0.55", ls="--", lw=0.7, alpha=0.7)
    ax.set_xlabel("$n$ (generations)")
    ax.set_ylabel("Envelope$/\\mathcal{R}^n$")
    ax.set_title("Amplification envelopes", fontsize=6, pad=3)
    ax.legend(fontsize=4.5, borderpad=0.3, labelspacing=0.10, ncol=2,
              loc="upper right")
    _panel_label(ax, "C")

    # ── d: Top 3 eigenvalue magnitudes over time ──────────────────────────
    ax = fig.add_subplot(gs[1, 2:4])
    lam1 = np.array([float(np.abs(specs[t]["eigenvalues"][0])) for t in range(T)])
    lam2 = np.array([float(np.abs(specs[t]["eigenvalues"][1]))
                     if len(specs[t]["eigenvalues"]) > 1 else 0.0 for t in range(T)])
    lam3 = np.array([float(np.abs(specs[t]["eigenvalues"][2]))
                     if len(specs[t]["eigenvalues"]) > 2 else 0.0 for t in range(T)])
    t_arr = np.arange(T)
    ax.plot(t_arr, lam1, color=OKABE_ITO[4], lw=1.0, label="$|\\lambda_1(t)|$")
    ax.plot(t_arr, lam2, color=OKABE_ITO[2], lw=0.9, ls="--", label="$|\\lambda_2(t)|$")
    ax.plot(t_arr, lam3, color=OKABE_ITO[0], lw=0.8, ls=":", label="$|\\lambda_3(t)|$")
    ax.axvline(peak, color="0.55", ls="--", lw=0.8, alpha=0.7)
    ax.text(peak + 1, ax.get_ylim()[1] * 0.95 if ax.get_ylim()[1] > 0 else 0.1,
            f"peak\n(day {peak})", fontsize=5, color="0.4", va="top")
    ax.set_xlabel("Day $t$")
    ax.set_ylabel("Eigenvalue magnitude")
    ax.set_title("Top 3 eigenvalue magnitudes", fontsize=6, pad=3)
    ax.legend(fontsize=5.5, borderpad=0.3, labelspacing=0.15)
    _panel_label(ax, "D")

    # ── e: Condition number κ(R(t)) = ‖v‖₂‖v*‖₂/|v·v*|  (Eq. 38) ──────
    ax = fig.add_subplot(gs[1, 4:6])
    def _cond_eigvec(R_m):
        """κ = ‖v‖₂‖w‖₂/|v·w| — np.eig returns L2-unit vecs, so κ = 1/|v·w|."""
        ev_r, evec_r = np.linalg.eig(R_m)
        ev_l, evec_l = np.linalg.eig(R_m.T)
        idx_r = int(np.argmax(np.abs(ev_r)))
        idx_l = int(np.argmax(np.abs(ev_l)))
        w = evec_r[:, idx_r]
        v = evec_l[:, idx_l]
        return 1.0 / (abs(float(v @ w)) + 1e-300)
    cond_ts   = np.array([_cond_eigvec(R_mats[t]) for t in range(T)])
    cond_ts   = np.minimum(cond_ts, 1e4)
    ax.semilogy(np.arange(T), cond_ts, color=OKABE_ITO[1], lw=0.9,
                label="$\\kappa(\\mathbf{R}(t))$")
    _yv_cand = [1, 2, 3, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000]
    # _yv = [y for y in _yv_cand if 0.9 * cond_ts.min() <= y <= 1.1 * cond_ts.max()]
    # if _yv:
    #     ax.set_yticks(_yv)
    # from matplotlib.ticker import FuncFormatter
    # ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:g}"))
    ax.axvline(peak, color="0.55", ls="--", lw=0.8, alpha=0.7)
    ax.set_xlabel("Day $t$")
    ax.set_ylabel("$\\kappa$ (log scale)")
    ax.set_title(
        "Condition number\n"
        r"$\kappa(\mathbf{R}) = \|\mathbf{v}\|_2\|\mathbf{v}^*\|_2\,/\,|\mathbf{v}^\top\mathbf{v}^*|$",
        fontsize=6, pad=3)
    ax.legend(fontsize=5.5, borderpad=0.3)
    _panel_label(ax, "E")

    # ── f: Within-fraction heatmap + overall π̄(t) overlay + per-loc bar ─────
    # π_j(t) = E_{jj}(t)/Σ_k E_{kj}(t)  [heatmap, per-location per-day]
    # Overall π(t) = Σ_j E_{jj}(t)/Σ_{k,j} E_{kj}(t)  [navy dash-dot overlay]
    # Right bar = time-averaged π̄_j  [per-location summary]
    # Wider gutter (wspace) so the π̄(t) legend sits between the heatmap's
    # right-hand y-axis/colorbar and the right-hand bar panel.
    gs_e = gridspec.GridSpecFromSubplotSpec(
        1, 2, subplot_spec=gs[2, :], width_ratios=[3, 1.15], wspace=0.9)
    ax   = fig.add_subplot(gs_e[0, 0])
    ax_r = fig.add_subplot(gs_e[0, 1])

    N_loc   = inc.shape[1]
    inc_mat = sim["incidence_matrix"]             # (T, N, N)
    col_sum = inc_mat.sum(axis=1)                 # Σ_k E_{kj}(t) → (T, N)
    diag    = np.array([inc_mat[t].diagonal() for t in range(T)])   # (T, N)
    pi_within  = np.where(col_sum > 0, diag / col_sum, np.nan)      # (T, N)
    # overall π(t) = trace / total across all (k,j) pairs
    total_mat  = inc_mat.sum(axis=(1, 2))                            # (T,)
    total_diag = np.array([inc_mat[t].trace() for t in range(T)])    # (T,)
    pi_overall = np.where(total_mat > 0, total_diag / total_mat, np.nan)  # (T,)

    im = ax.imshow(pi_within.T, aspect="auto", origin="upper",
                   cmap="RdYlGn", vmin=0, vmax=1, interpolation="nearest",
                   extent=[0, T, N_loc + 0.5, 0.5])
    ax.set_xlabel("Day $t$")
    ax.set_ylabel("Location $j$")
    ax.set_title(
        r"Within-fraction $\pi_j(t) = E_{jj}/\!\sum_k E_{kj}$"
        "  (green = local, red = imported)", fontsize=6.5, pad=4)
    ax.set_yticks(range(1, N_loc + 1))
    ax.set_yticklabels([f"L{i}" for i in range(1, N_loc + 1)], fontsize=6)
    ax.axvline(peak, color="k", lw=0.9, ls="--", alpha=0.6)
    ax.text(peak + 1, 0.75, f"peak\n(d{peak})", fontsize=5, color="k",
            va="top", transform=ax.get_xaxis_transform())
    # Overlay overall π(t) on twin-x axis (navy dash-dot)
    ax2 = ax.twinx()
    ax2.spines["right"].set_visible(True)
    ax2.spines["top"].set_visible(False)
    mask_o = ~np.isnan(pi_overall)
    ax2.plot(np.where(mask_o)[0], pi_overall[mask_o],
             color="#1a237e", lw=1.4, ls="-.", alpha=0.90,
             label=r"$\bar{\pi}(t)$ overall")
    ax2.set_ylim(0, 1.4)
    ax2.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax2.set_ylabel(r"Overall $\bar{\pi}(t)$", fontsize=6, color="#1a237e")
    ax2.tick_params(axis="y", labelcolor="#1a237e", labelsize=5)
    # π̄(t) legend inset INSIDE the heatmap (upper-right corner, where the dash-dot
    # π̄(t) trace does not run), framed so it reads over the cells.
    ax2.legend(loc="upper right", fontsize=5.5, frameon=True, facecolor="white",
               framealpha=0.9, edgecolor="0.8", borderpad=0.3, handlelength=1.4)
    # Vertical colorbar on the right with the label as a title above the bar
    # (horizontal placement would force the title to collide with the "Day t"
    # x-axis label). Per-location within-fraction is already named in the title.
    # Colorbar in its own inset in the gutter, to the RIGHT of the heatmap's twin
    # y-axis (so it no longer overlaps the "Overall π̄(t)" axis), left of the bar.
    cax = ax.inset_axes([1.26, 0.06, 0.05, 0.88])
    cbar = fig.colorbar(im, cax=cax)
    cbar.ax.set_title(r"$\pi_j(t)$", fontsize=6, pad=3)
    cbar.ax.tick_params(labelsize=5)
    _panel_label(ax, "F")

    # ── right panel: per-location time-averaged π̄_j bar chart ─────────────
    pi_j_avg       = np.nanmean(pi_within, axis=0)               # (N,)
    pi_overall_avg = float(np.nanmean(pi_overall[mask_o])) if mask_o.any() else 0.5
    bar_clrs = [plt.cm.RdYlGn(float(np.clip(v, 0, 1))) for v in pi_j_avg]
    ax_r.barh(range(1, N_loc + 1), pi_j_avg, color=bar_clrs,
              height=0.72, edgecolor="none")
    ax_r.axvline(pi_overall_avg, color="#1a237e", lw=1.2, ls="-.")
    ax_r.text(min(pi_overall_avg + 0.04, 0.98), N_loc + 0.6,
              f"all:{pi_overall_avg:.2f}", fontsize=4.5, color="#1a237e",
              va="top", ha="left")
    ax_r.set_xlim(0, 1.10)
    ax_r.set_xlabel(r"$\bar{\pi}_j$", fontsize=6)
    ax_r.set_yticks(range(1, N_loc + 1))
    ax_r.set_yticklabels([f"L{i}" for i in range(1, N_loc + 1)], fontsize=5)
    ax_r.set_title("Time-avg\n$\\bar{\\pi}_j$", fontsize=5.5, pad=3)
    ax_r.tick_params(labelsize=5)
    for j_idx, v in enumerate(pi_j_avg):
        ax_r.text(min(float(v) + 0.03, 1.02), j_idx + 1, f"{v:.2f}",
                  va="center", ha="left", fontsize=4.2, color="0.3")

    plt.savefig(f"{save_prefix}_04_spectral.pdf", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved {save_prefix}_04_spectral.pdf")


def plot_fig5(sim_A, sim_B, city_A, city_B, f_A, f_B,
              R_t0_A, R_t0_B, save_prefix="fig"):
    """Figure 5: Two mobility settings compared — Dense urban vs Sparse national.

    Panels:
      a  Mean mobility matrix f̄_{jk} — Scenario B (sparse national)
      b  3D bar chart of E_kj at epidemic peak — Scenario B (sparse national)
      c  3D bar chart of R_kj at epidemic peak — Scenario B (sparse national)
      d  R(t) and σ(t) for both scenarios over time
      e  System R(t) comparison with normalised incidence on twin axis
      f  Overall within-location fraction π(t)=Σ_j E_{jj}/Σ_{k,j} E_{kj} for both scenarios
    """
    inc_A, inc_B = sim_A["incidence"], sim_B["incidence"]
    Rm_A,  Rm_B  = sim_A["R_matrices"], sim_B["R_matrices"]
    imat_B = sim_B.get("incidence_matrix", None)   # (T, N, N) if available
    coords_A, pops_A, dists_A, types_A, meta_A = city_A
    T, N  = inc_A.shape
    pk_B  = int(inc_B.sum(axis=1).argmax())
    loc   = [f"L{i+1}" for i in range(N)]
    col_A = OKABE_ITO[0]   # dense urban — orange
    col_B = OKABE_ITO[4]   # sparse national — blue

    R_sys_A  = np.array([R_system(Rm_A[t]) for t in range(T)])
    R_sys_B  = np.array([R_system(Rm_B[t]) for t in range(T)])
    sigma_A  = np.array([reactivity(Rm_A[t])["sigma"] for t in range(T)])
    sigma_B  = np.array([reactivity(Rm_B[t])["sigma"] for t in range(T)])

    fig = plt.figure(figsize=(7.2, 6.2))
    gs  = gridspec.GridSpec(3, 2, hspace=0.62, wspace=0.52,
                            left=0.10, right=0.96, top=0.96, bottom=0.08)

    # ── a: mean mobility matrix — Scenario B (log scale to reveal movement) ──
    ax = fig.add_subplot(gs[0, 0])
    f_B_mean = f_B.mean(axis=0)
    import matplotlib.colors as mcolors
    lognorm = mcolors.LogNorm(vmin=max(f_B_mean[f_B_mean > 0].min(), 1e-4),
                               vmax=f_B_mean.max())
    im = ax.imshow(f_B_mean, cmap="Oranges", aspect="auto", norm=lognorm)
    ax.set_xticks(range(N)); ax.set_xticklabels(loc, fontsize=5, rotation=45)
    ax.set_yticks(range(N)); ax.set_yticklabels(loc, fontsize=5)
    ax.set_xlabel("Activity location $k$"); ax.set_ylabel("Residence $j$")
    cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.ax.set_title("$\\bar{f}_{jk}$\n(log)", fontsize=6, pad=3)
    ax.set_title("Mean mobility (log scale)", fontsize=6, pad=3)
    _panel_label(ax, "A")

    # ── b: 3D E_kj at peak — Scenario B ───────────────────────────────────
    ax_b = fig.add_subplot(gs[0, 1], projection="3d")
    if imat_B is not None:
        _bar3d_inc(ax_b, imat_B[pk_B], f"Sparse national — peak (day {pk_B})")
    else:
        ax_b.text(0.5, 0.5, 0.5, "incidence matrix\nnot available",
                  ha="center", va="center", fontsize=7, color="0.5")
    ax_b.text2D(-0.10, 1.05, "b", transform=ax_b.transAxes,
                fontsize=10, fontweight="bold", va="top", ha="left")

    # ── c: 3D R_kj at peak — Scenario B ───────────────────────────────────
    ax_c = fig.add_subplot(gs[1, 0], projection="3d")
    _bar3d_Rkj(ax_c, Rm_B[pk_B], f"Sparse national — peak (day {pk_B})")
    ax_c.text2D(-0.10, 1.05, "c", transform=ax_c.transAxes,
                fontsize=10, fontweight="bold", va="top", ha="left")

    # ── d: R(t) and σ(t) for both scenarios ───────────────────────────────
    ax = fig.add_subplot(gs[1, 1])
    t_arr = np.arange(T)
    vA = R_sys_A > 0; vB = R_sys_B > 0
    ax.plot(t_arr[vA], R_sys_A[vA],   color=col_A, lw=1.0,
            label="$\\mathcal{R}$ — dense urban")
    ax.plot(t_arr[vB], R_sys_B[vB],   color=col_B, lw=1.0,
            label="$\\mathcal{R}$ — sparse national")
    vsa = sigma_A > 0; vsb = sigma_B > 0
    ax.plot(t_arr[vsa], sigma_A[vsa], color=col_A, lw=0.7, ls="--",
            alpha=0.7, label="$\\sigma$ — dense urban")
    ax.plot(t_arr[vsb], sigma_B[vsb], color=col_B, lw=0.7, ls="--",
            alpha=0.7, label="$\\sigma$ — sparse national")
    ax.axhline(1.0, color="0.55", ls="--", lw=0.8)
    ax.set_xlabel("Day $t$"); ax.set_ylabel("Value")
    ax.legend(fontsize=5.0, ncol=2, borderpad=0.3, labelspacing=0.15,
              handlelength=1.2)
    _panel_label(ax, "D")

    # ── e: system R(t) with normalised incidence on twin axis ──────────────
    ax  = fig.add_subplot(gs[2, 0])
    ax2 = ax.twinx()
    ax2.spines["right"].set_visible(True); ax2.spines["top"].set_visible(False)
    ax.plot(t_arr[vA], R_sys_A[vA], color=col_A, lw=0.9, label="Dense urban")
    ax.plot(t_arr[vB], R_sys_B[vB], color=col_B, lw=0.9, label="Sparse national")
    ax.axhline(1.0, color="0.55", ls="--", lw=0.8)
    peak_inc = float(max(inc_A.sum(axis=1).max(), inc_B.sum(axis=1).max()))
    ax2.fill_between(range(T), inc_A.sum(axis=1) / peak_inc, alpha=0.14, color=col_A)
    ax2.fill_between(range(T), inc_B.sum(axis=1) / peak_inc, alpha=0.14, color=col_B)
    ax.set_xlabel("Day $t$"); ax.set_ylabel("$\\mathcal{R}(t)$")
    ax2.set_ylabel("Normalised incidence", fontsize=6)
    ax2.tick_params(labelsize=6)
    ax.legend(fontsize=6, borderpad=0.3)
    _panel_label(ax, "E")

    # ── f: Within-fraction π(t) for both scenarios ─────────────────────────
    # π(t) = Σ_j E_{jj}(t) / Σ_{k,j} E_{kj}(t)  [overall within-fraction]
    ax = fig.add_subplot(gs[2, 1])
    imat_A = sim_A.get("incidence_matrix", None)
    t_arr  = np.arange(T)
    for sim_xy, col_xy, lbl_xy in [
            (sim_A, col_A, "Dense urban"),
            (sim_B, col_B, "Sparse national")]:
        im_xy = sim_xy.get("incidence_matrix", None)
        if im_xy is not None:
            tot_xy  = im_xy.sum(axis=(1, 2))
            diag_xy = np.array([im_xy[t].trace() for t in range(T)])
            pi_xy   = np.where(tot_xy > 0, diag_xy / tot_xy, np.nan)
            vm_xy   = ~np.isnan(pi_xy)
            ax.plot(t_arr[vm_xy], pi_xy[vm_xy], color=col_xy, lw=1.0, label=lbl_xy)
    ax.axhline(0.5, color="0.55", ls=":", lw=0.7, alpha=0.7)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Day $t$")
    ax.set_ylabel(r"Overall $\bar{\pi}(t) = \sum_j E_{jj}/\sum_{k,j} E_{kj}$")
    ax.set_title(r"Within-fraction $\bar{\pi}(t)$: local vs imported", fontsize=6.5, pad=3)
    ax.legend(fontsize=6, borderpad=0.3)
    _panel_label(ax, "F")

    plt.savefig(f"{save_prefix}_05_comparison.pdf", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved {save_prefix}_05_comparison.pdf")


def plot_counterfactual_nonnormal(sim_A, sim_C, city_A, f_A, f_C,
                                   w_within, w_between,
                                   max_days, lw_A, lb_A, lw_C, lb_C,
                                   params, R_ind_A, R_ind_C, gen_time_pmf,
                                   save_prefix="fig", city_C=None):
    """Counterfactual: baseline (A) vs hub-amplified (C) — false-action zone.

    The operationally meaningful 'false-action zone' is: ρ(R(t)) < 1
    (epidemic controlled at the network level) but the naive per-location
    estimator R̂^j_ind(t) > 1 at satellite locations, because import-driven
    incidence inflates the local renewal denominator.  A surveillance analyst
    observing only satellite case counts would wrongly conclude that local
    transmission is growing and call for NPIs.

    Note on spectral non-normality (σ/ρ): with realistic lw/lb = 3.33 and home
    fractions 0.15–0.82, the between-location kernel term lb·D[j,k] is
    symmetric by construction (D is symmetric), so σ/ρ ≈ 1.03–1.15 and the
    spectral false-action zone (σ > 1, ρ < 1) is too narrow to plot
    meaningfully.  The figure instead shows the estimator-bias false-action zone
    (panel f) which is both achievable and policy-relevant.

    Layout 4 × 3:
      a  Aggregate incidence — A vs C
      b  Network ρ(R(t)) — A vs C
      c  σ(t)/ρ(t) non-normality ratio — A vs C
      d  Per-location incidence stacked area — Scenario C
      e  R̂^j_ind vs ρ(R) — Scenario A  (mild bias)
      f  R̂^j_ind vs ρ(R) — Scenario C  (FALSE-ACTION ZONE shaded)
      g  R_kj heatmap at peak — Baseline A
      h  R_kj heatmap at peak — Hub-amplified C
      i  Within-fraction π̄_j bar chart — A vs C
      j  Overall within-fraction π̄(t) timeseries — A vs C
      k  π_j(t) heatmap — Scenario A
      l  π_j(t) heatmap — Scenario C
    """
    inc_A  = sim_A["incidence"];     inc_C  = sim_C["incidence"]
    R_A    = sim_A["R_matrices"];    R_C    = sim_C["R_matrices"]
    T, N   = inc_A.shape
    t_arr  = np.arange(T)
    pops_A = city_A[1]
    types_A = city_A[3]
    pops_C = city_C[1] if city_C is not None else pops_A
    hub_idx_C = int(np.argmax(pops_C))   # mega-hub node index in Scenario C

    peak_A = int(inc_A.sum(axis=1).argmax())
    peak_C = int(inc_C.sum(axis=1).argmax())

    # ── spectral quantities ────────────────────────────────────────────────
    rho_A   = np.array([R_system(R_A[t]) for t in range(T)])
    rho_C   = np.array([R_system(R_C[t]) for t in range(T)])
    sigma_A = np.array([reactivity(R_A[t])["sigma"] for t in range(T)])
    sigma_C = np.array([reactivity(R_C[t])["sigma"] for t in range(T)])
    ratio_A = np.where(rho_A > 0.05, sigma_A / rho_A, np.nan)
    ratio_C = np.where(rho_C > 0.05, sigma_C / rho_C, np.nan)

    # ── within-fraction π_j(t) = E_jj / Σ_k E_kj ────────────────────────
    def _pi(sim, T_len):
        imat    = sim["incidence_matrix"]
        col_sum = imat.sum(axis=1)
        diag    = np.array([imat[t].diagonal() for t in range(T_len)])
        return np.where(col_sum > 1e-6, diag / col_sum, np.nan)

    pi_A = _pi(sim_A, T)
    pi_C = _pi(sim_C, T)

    # Per-scenario hub + two satellite locations for the false-action panels (E/F).
    # Roles MUST be scenario-specific: Scenario A uses the paper's core hub
    # (meta["hub_idx"], as everywhere else), while Scenario C's hub is its
    # highest-population node (the 3M mega-hub, hub_idx_C).  Using Scenario A's
    # indices for panel F would shade C's mega-hub as a "satellite".
    hub_A = int(city_A[4].get("hub_idx", int(np.argmax(pops_A))))
    sat_A = [j for j in range(N) if j != hub_A][:2]
    hub_C = hub_idx_C
    sat_C = [j for j in range(N) if j != hub_C][:2]

    col_A   = OKABE_ITO[4]   # sky blue  — baseline
    col_C   = OKABE_ITO[5]   # vermillion — counterfactual
    col_hub = OKABE_ITO[0]   # orange    — hub
    col_sa1 = OKABE_ITO[2]   # blue-green — satellite 1
    col_sa2 = OKABE_ITO[3]   # yellow    — satellite 2
    col_rho = "0.35"

    fig = plt.figure(figsize=(7.2, 10.5))
    gs  = gridspec.GridSpec(4, 3, hspace=0.82, wspace=0.56,
                            left=0.09, right=0.97, top=0.96, bottom=0.05)

    # ── a: aggregate incidence ─────────────────────────────────────────────
    ax = fig.add_subplot(gs[0, 0])
    ax.plot(t_arr, inc_A.sum(1)/1e3, color=col_A, lw=1.1, label="Baseline (A)")
    ax.plot(t_arr, inc_C.sum(1)/1e3, color=col_C, lw=1.1, ls="--", label="Hub-amp. (C)")
    ax.axvline(peak_A, color=col_A, lw=0.6, ls=":", alpha=0.7)
    ax.axvline(peak_C, color=col_C, lw=0.6, ls=":", alpha=0.7)
    ax.set_xlabel("Day $t$", fontsize=7); ax.set_ylabel("Incidence (×10³)", fontsize=7)
    ax.set_title("Aggregate incidence", fontsize=6.5, pad=3)
    ax.legend(fontsize=5, borderpad=0.3); ax.tick_params(labelsize=6)
    _panel_label(ax, "A")

    # ── b: network ρ(R) and E(t) ─────────────────────────────────────────
    ax = fig.add_subplot(gs[0, 1])
    # Risk-aware reproduction number E(t) = Σ_j (R^j_out)² / Σ_k R^k_out
    R_out_nn_A = np.array([R_outward(R_A[t]) for t in range(T)])
    R_out_nn_C = np.array([R_outward(R_C[t]) for t in range(T)])
    E_t_nn_A = np.array([np.sum(R_out_nn_A[t]**2) / (np.sum(R_out_nn_A[t]) + 1e-300)
                          for t in range(T)])
    E_t_nn_C = np.array([np.sum(R_out_nn_C[t]**2) / (np.sum(R_out_nn_C[t]) + 1e-300)
                          for t in range(T)])
    ax.plot(t_arr, rho_A, color=col_A, lw=1.1, label="$\\mathcal{R}(t)$ — A")
    ax.plot(t_arr, rho_C, color=col_C, lw=1.1, ls="--", label="$\\mathcal{R}(t)$ — C")
    ax.plot(t_arr, E_t_nn_A, color=col_A, lw=0.9, ls=":", alpha=0.8,
            label="$\\mathcal{E}(t)$ — A")
    ax.plot(t_arr, E_t_nn_C, color=col_C, lw=0.9, ls=":", alpha=0.8,
            label="$\\mathcal{E}(t)$ — C")
    ax.axhline(1.0, color=col_rho, ls="--", lw=0.8)
    ax.set_xlabel("Day $t$", fontsize=7)
    ax.set_ylabel("$\\mathcal{R}(t)$,  $\\mathcal{E}(t)$", fontsize=7)
    ax.set_title("Network $\\mathcal{R}(t)$ and $\\mathcal{E}(t)$", fontsize=6.5, pad=3)
    ax.legend(fontsize=4.5, borderpad=0.3, ncol=2); ax.tick_params(labelsize=6)
    _panel_label(ax, "B")

    # ── c: σ/ρ non-normality ratio ─────────────────────────────────────────
    ax = fig.add_subplot(gs[0, 2])
    vm_A = ~np.isnan(ratio_A); vm_C = ~np.isnan(ratio_C)
    ax.plot(t_arr[vm_A], ratio_A[vm_A], color=col_A, lw=1.0, label="Baseline (A)")
    ax.plot(t_arr[vm_C], ratio_C[vm_C], color=col_C, lw=1.0, ls="--", label="Hub-amp. (C)")
    ax.axhline(1.0, color=col_rho, ls="--", lw=0.8)
    ax.set_xlabel("Day $t$", fontsize=7); ax.set_ylabel("$\\sigma(t)/\\mathcal{R}(t)$", fontsize=7)
    ax.set_title("Non-normality $\\sigma/\\mathcal{R}$", fontsize=6.5, pad=3)
    ax.text(0.97, 0.97,
            "Hub-amp. (C) has\nlarger $\\sigma/\\mathcal{R}$ gap",
            transform=ax.transAxes, fontsize=4.5, ha="right", va="top",
            color="0.4", style="italic")
    ax.legend(fontsize=5, borderpad=0.3); ax.tick_params(labelsize=6)
    _panel_label(ax, "C")

    # ── d: per-location incidence stacked — Scenario C ─────────────────────
    ax = fig.add_subplot(gs[1, 0])
    order_C = np.argsort(-pops_C)   # sort by Scenario C populations
    cmap_l  = plt.cm.tab10
    def _c_lbl(i):
        return "hub" if i == hub_idx_C else "sat"
    ax.stackplot(t_arr, inc_C[:, order_C].T / 1e3,
                 colors=[cmap_l(i / N) for i in range(N)],
                 labels=[f"L{order_C[i]+1} ({_c_lbl(order_C[i])})" for i in range(N)],
                 alpha=0.85)
    ax.axvline(peak_C, color="k", lw=0.8, ls="--")
    ax.set_xlabel("Day $t$", fontsize=7); ax.set_ylabel("Incidence (×10³)", fontsize=7)
    ax.set_title("Hub-amp. (C): per-location incidence", fontsize=6.5, pad=3)
    ax.legend(fontsize=3.5, ncol=2, borderpad=0.2, labelspacing=0.1, loc="upper right")
    ax.tick_params(labelsize=6)
    _panel_label(ax, "D")

    # ── e / f: R̂^j_ind vs ρ(R) — the FALSE-ACTION ZONE ───────────────────
    for col_idx, (rho_x, R_ind_x, peak_x, label_sc, panel_id, hub_x, sat_x) in enumerate([
            (rho_A, R_ind_A, peak_A, "Baseline (A)", "E", hub_A, sat_A),
            (rho_C, R_ind_C, peak_C, "Hub-amp. (C)", "F", hub_C, sat_C),
    ]):
        ax = fig.add_subplot(gs[1, col_idx + 1])
        ax.plot(t_arr, rho_x, color=col_rho, lw=1.2,
                label="$\\mathcal{R}(t)=\\rho(\\mathbf{R})$", zorder=5)
        ax.axhline(1.0, color=col_rho, ls="--", lw=0.8, alpha=0.6)
        ax.axvline(peak_x, color=col_rho, ls=":", lw=0.6, alpha=0.5)

        loc_cols = [col_hub, col_sa1, col_sa2]
        for j, (loc_j, lc) in enumerate(zip([hub_x] + sat_x, loc_cols)):
            vm = ~np.isnan(R_ind_x[:, loc_j])
            role = "hub" if j == 0 else "sat"
            lbl = f"$\\hat{{R}}^{{\\mathrm{{ind}}}}$ L{loc_j+1} ({role})"
            ax.plot(t_arr[vm], R_ind_x[vm, loc_j], color=lc, lw=0.9,
                    ls="--" if j > 0 else "-", alpha=0.85, label=lbl)

        # shade false-action: R̂_ind_j > 1 AND ρ < 1
        any_fa = np.zeros(T, dtype=bool)
        for loc_j, lc in zip(sat_x, [col_sa1, col_sa2]):
            fa = (~np.isnan(R_ind_x[:, loc_j])) & (R_ind_x[:, loc_j] > 1.0) & (rho_x < 1.0)
            if fa.any():
                ax.fill_between(t_arr, 1.0, R_ind_x[:, loc_j],
                                where=fa, color=lc, alpha=0.18)
                any_fa |= fa
        if any_fa.any():
            fa_days = np.where(any_fa)[0]
            ax.axvspan(fa_days[0], fa_days[-1], color="#cc0000", alpha=0.06, zorder=0)
            mid = int(fa_days[len(fa_days) // 2])
            ymax = max(np.nanmax(R_ind_x[any_fa, :]) * 1.05, 1.5) if any_fa.any() else 1.5
            ax.text(mid, ymax * 0.93,
                    "False-action zone\n"
                    r"($\mathcal{R}<1$, $R^j_{\mathrm{ind}}>1$)",
                    ha="center", va="top", fontsize=4.5, color="#cc0000",
                    style="italic",
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.8))

        ax.set_xlabel("Day $t$", fontsize=7)
        ax.set_ylabel("Reproduction number", fontsize=7)
        ax.set_title(f"{label_sc}\n$\\hat{{R}}^{{\\rm ind}}_j$ vs $\\mathcal{{R}}(t)$",
                     fontsize=6.5, pad=3)
        ax.legend(fontsize=4.5, borderpad=0.3, labelspacing=0.1, loc="upper right")
        ax.tick_params(labelsize=6)
        _panel_label(ax, panel_id)

    # ── g / h: R_kj heatmaps at peak ──────────────────────────────────────
    R_pk_A = R_A[peak_A]; R_pk_C = R_C[peak_C]
    vmax_R = np.percentile(np.concatenate([R_pk_A.ravel(), R_pk_C.ravel()]), 98)
    for col_idx, (R_pk, label_sc, pklbl) in enumerate([
            (R_pk_A, "Baseline (A)", f"d{peak_A}"),
            (R_pk_C, "Hub-amp. (C)", f"d{peak_C}")]):
        ax = fig.add_subplot(gs[2, col_idx])
        im = ax.imshow(R_pk, vmin=0, vmax=vmax_R, cmap="plasma",
                       origin="upper", aspect="equal", interpolation="nearest")
        ax.set_xticks(range(N)); ax.set_yticks(range(N))
        ax.set_xticklabels([f"L{i+1}" for i in range(N)], fontsize=3.8, rotation=45)
        ax.set_yticklabels([f"L{i+1}" for i in range(N)], fontsize=3.8)
        ax.set_xlabel("Infectee $j$", fontsize=7); ax.set_ylabel("Infector $k$", fontsize=7)
        ax.set_title(f"$R_{{kj}}$ — {label_sc} ({pklbl})", fontsize=6, pad=3)
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cb.ax.tick_params(labelsize=5)
        _panel_label(ax, "GH"[col_idx])

    # ── i: resident population bar chart — Scenario C ─────────────────────
    ax = fig.add_subplot(gs[2, 2])
    COL_HUB_BAR = "#CC5500"
    bar_clrs = [COL_HUB_BAR if i == hub_idx_C else "0.55" for i in range(N)]
    ax.barh(range(N), pops_C / 1e6, color=bar_clrs, height=0.72, edgecolor="none")
    ax.set_yticks(range(N))
    ax.set_yticklabels([f"L{i+1}" for i in range(N)], fontsize=5)
    ax.set_xlabel("Resident population (millions)", fontsize=7)
    ax.set_title("Resident population\nScenario C", fontsize=6.5, pad=3)
    ax.invert_yaxis()
    ax.tick_params(labelsize=6)
    ax.text(pops_C[hub_idx_C] / 1e6 + 0.04, hub_idx_C,
            "hub", fontsize=5, color=COL_HUB_BAR, va="center")
    _panel_label(ax, "I")

    # ── j: overall π̄(t) timeseries ────────────────────────────────────────
    ax = fig.add_subplot(gs[3, 0])
    def _pi_ov(sim, T_len):
        imat = sim["incidence_matrix"]
        tot  = imat.sum(axis=(1, 2))
        diag = np.array([imat[t].trace() for t in range(T_len)])
        return np.where(tot > 0, diag / tot, np.nan)
    pi_ov_A = _pi_ov(sim_A, T); pi_ov_C = _pi_ov(sim_C, T)
    vm_oa = ~np.isnan(pi_ov_A); vm_oc = ~np.isnan(pi_ov_C)
    ax.plot(t_arr[vm_oa], pi_ov_A[vm_oa], color=col_A, lw=1.1, label="Baseline (A)")
    ax.plot(t_arr[vm_oc], pi_ov_C[vm_oc], color=col_C, lw=1.1, ls="--", label="Hub-amp. (C)")
    ax.axhline(0.9, color="0.5", ls="--", lw=0.7, alpha=0.7, label="90% local")
    ax.fill_between(t_arr[vm_oc], 0, pi_ov_C[vm_oc],
                    where=pi_ov_C[vm_oc] < 0.9, color=col_C, alpha=0.12,
                    label="Import-driven (C)")
    ax.set_xlabel("Day $t$", fontsize=7); ax.set_ylabel(r"$\bar{\pi}(t)$", fontsize=7)
    ax.set_title("Overall within-fraction $\\bar{\\pi}(t)$", fontsize=6.5, pad=3)
    ax.set_ylim(0, 1.1)
    ax.legend(fontsize=5, borderpad=0.3); ax.tick_params(labelsize=6)
    _panel_label(ax, "J")

    # ── k / l: π_j(t) heatmaps ────────────────────────────────────────────
    for col_idx, (pi_x, peak_x, label_sc, panel_id) in enumerate([
            (pi_A, peak_A, "Baseline (A)", "K"),
            (pi_C, peak_C, "Hub-amp. (C)", "L"),
    ]):
        ax = fig.add_subplot(gs[3, col_idx + 1])
        im = ax.imshow(pi_x.T, aspect="auto", origin="upper",
                       cmap="RdYlGn", vmin=0, vmax=1,
                       interpolation="nearest",
                       extent=[0, T, N + 0.5, 0.5])
        ax.axvline(peak_x, color="k", lw=0.8, ls="--", alpha=0.7)
        ax.set_xlabel("Day $t$", fontsize=7); ax.set_ylabel("Location $j$", fontsize=7)
        ax.set_yticks(range(1, N + 1))
        ax.set_yticklabels([f"L{j}" for j in range(1, N+1)], fontsize=5)
        ax.set_title(f"{label_sc}: " + r"$\pi_j(t)$"
                     + " (green=local, red=import)", fontsize=6, pad=3)
        cb = fig.colorbar(im, ax=ax, orientation="horizontal", pad=0.32,
                          fraction=0.04, aspect=40, shrink=0.80)
        cb.set_ticks([0.0, 0.25, 0.5, 0.75, 1.0])
        cb.ax.tick_params(labelsize=5)
        cb.set_label(r"$\pi_j(t)$", fontsize=5.5)
        ax.tick_params(labelsize=6)
        _panel_label(ax, panel_id)

    plt.savefig(f"{save_prefix}_counterfactual_nonnormal.pdf", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved {save_prefix}_counterfactual_nonnormal.pdf")


def _plot_naive_3x3(all_data, est_key, ref_key, est_tex, ref_tex,
                    part_title, fname):
    """One 3-row x 3-col comparison figure for one (estimator, reference) pair.

    all_data  : list of (data_dict, scenario_label, scenario_color)
    est_key   : key into data_dict for the estimator time series
    ref_key   : "rho_t" or "E_t"
    est_tex   : LaTeX string (no outer $ delimiters)
    ref_tex   : LaTeX string (no outer $ delimiters)
    part_title: short description for suptitle
    fname     : full save path (.pdf)
    """
    COL_ET   = OKABE_ITO[6]       # reddish-purple for E(t)
    REF_IS_E = (ref_key == "E_t")
    PANELS   = ["ABC", "DEF", "GHI"]

    fig = plt.figure(figsize=(7.2, 8.0))
    gs  = gridspec.GridSpec(3, 3, hspace=0.72, wspace=0.52,
                            left=0.09, right=0.97, top=0.92, bottom=0.07)

    for row, (d, sc_label, col) in enumerate(all_data):
        pids    = PANELS[row]
        t_arr   = np.arange(d["T"])
        R_est   = d[est_key]
        R_ref   = d[ref_key]
        rho_t   = d["rho_t"]
        inc_tot = d["inc_tot"]
        peak_t  = d["peak_t"]

        # validity: reference above noise floor; estimator finite
        valid = R_ref > 0.05
        if np.any(np.isnan(R_est)):
            valid = valid & ~np.isnan(R_est)
        bias_abs = np.where(valid, R_est - R_ref, np.nan)
        bias_pct = np.where(valid & (R_ref > 0.1),
                            (R_est - R_ref) / R_ref * 100.0, np.nan)

        # ── col 0: aggregate incidence ─────────────────────────────────
        ax = fig.add_subplot(gs[row, 0])
        ax.fill_between(t_arr, inc_tot / 1e3, alpha=0.28, color=col)
        ax.plot(t_arr, inc_tot / 1e3, color=col, lw=1.1)
        ax.axvline(peak_t, color="0.50", ls="--", lw=0.8)
        ax.set_xlabel("Day $t$", fontsize=7)
        ax.set_ylabel("Daily incidence ($\\times 10^3$)", fontsize=7)
        ax.set_title(f"{sc_label}\nAggregate incidence", fontsize=6.5, pad=3)
        ax.tick_params(labelsize=6)
        _panel_label(ax, pids[0])

        # ── col 1: estimator vs reference ──────────────────────────────
        ax = fig.add_subplot(gs[row, 1])
        if REF_IS_E:
            vr = rho_t > 0
            ax.plot(t_arr[vr], rho_t[vr], color=col, lw=0.8, ls=":",
                    alpha=0.50, label="$\\mathcal{R}(t)$  [context]", zorder=2)
        ref_col = COL_ET if REF_IS_E else col
        vref = R_ref > 0
        ax.plot(t_arr[vref], R_ref[vref], color=ref_col, lw=1.3,
                label=f"${ref_tex}$", zorder=4)
        if np.any(np.isnan(R_est)):
            vm = ~np.isnan(R_est)
            ax.plot(t_arr[vm], R_est[vm], color="crimson", lw=1.0,
                    ls="--", label=f"${est_tex}$", zorder=5)
        else:
            ax.plot(t_arr, R_est, color="crimson", lw=1.0,
                    ls="--", label=f"${est_tex}$", zorder=5)
        ax.axhline(1.0, color="0.55", ls=":", lw=0.7, zorder=1)
        ax.axvline(peak_t, color="0.50", ls="--", lw=0.7, alpha=0.55, zorder=1)
        ax.set_xlabel("Day $t$", fontsize=7)
        ax.set_ylabel("Reproduction number", fontsize=7)
        ax.set_title(f"{sc_label}", fontsize=6.5, pad=3)
        ax.legend(fontsize=4.8, borderpad=0.3, labelspacing=0.10,
                  loc="upper right", ncol=1)
        ax.tick_params(labelsize=6)
        mean_bias = float(np.nanmean(bias_pct))
        ax.text(0.03, 0.05, f"Mean bias: ${mean_bias:+.1f}\\%$",
                transform=ax.transAxes, fontsize=5, ha="left", va="bottom",
                color="crimson", style="italic",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none",
                          alpha=0.75))
        _panel_label(ax, pids[1])

        # ── col 2: signed bias ─────────────────────────────────────────
        ax = fig.add_subplot(gs[row, 2])
        bv = ~np.isnan(bias_abs)
        if bv.any():
            pos = bv & (bias_abs > 0)
            neg = bv & (bias_abs < 0)
            if pos.any():
                ax.fill_between(t_arr, 0,
                                np.where(pos, bias_abs, 0),
                                color="crimson", alpha=0.35,
                                label="Over-estimate")
            if neg.any():
                ax.fill_between(t_arr, 0,
                                np.where(neg, bias_abs, 0),
                                color=OKABE_ITO[2], alpha=0.35,
                                label="Under-estimate")
            ax.plot(t_arr[bv], bias_abs[bv], color="0.30", lw=0.8)
        ax.axhline(0, color="0.55", ls="--", lw=0.7)
        ax.axvline(peak_t, color="0.50", ls="--", lw=0.7, alpha=0.55)
        ax.set_xlabel("Day $t$", fontsize=7)
        ax.set_ylabel(f"Estimator $-$ ${ref_tex}$", fontsize=7)
        ax.set_title(f"{sc_label}\nBias", fontsize=6.5, pad=3)
        if bv.any():
            ax.legend(fontsize=5.0, borderpad=0.3, labelspacing=0.12)
        ax.tick_params(labelsize=6)
        _panel_label(ax, pids[2])

    fig.suptitle(
        f"{part_title}  —  ${est_tex}$ vs ${ref_tex}$",
        fontsize=7.5, y=0.988)
    plt.savefig(fname, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved {fname}")


def plot_naive_R_comparison_suite(sim_A, sim_B, sim_C, city_A, city_B, city_C,
                                   gen_time_pmf, save_prefix="fig"):
    """Generate all 18 systematic naive-R comparison figures.

    Part 1 (8 figs) — aggregated per-location independent R:
      iw (incidence-weighted), pw (population-weighted),
      Rw (R_out-weighted), am (arithmetic mean),
      each compared to rho(R(t)) and to E(t).

    Part 2 (8 figs) — aggregated R^j_out from the R matrix:
      same four weighting schemes (note: Rw = E(t) by definition),
      each compared to rho(R(t)) and to E(t).

    Part 3 (2 figs) — aggregate independent R on total (network) incidence:
      compared to rho(R(t)) and to E(t).
    """
    print("  Computing naive estimator suite (3 scenarios) ...")
    scenarios = [
        (sim_A, city_A, "Dense urban (A)",    OKABE_ITO[4]),
        (sim_B, city_B, "Sparse national (B)", OKABE_ITO[2]),
        (sim_C, city_C, "Hub-amplified (C)",  OKABE_ITO[5]),
    ]
    all_data = []
    for sim, city, label, col in scenarios:
        d = _compute_naive_suite(sim, city, gen_time_pmf)
        all_data.append((d, label, col))
        print(f"    {label} done")

    refs = [
        ("rho_t", "\\mathcal{R}(t)",           "vs_Rt"),
        ("E_t",   "\\mathcal{E}(t){=}X(1,t)", "vs_Et"),
    ]

    # ── Part 1: per-location independent R, aggregated ───────────────────
    p1 = ("Part~1 — Aggregated independent"
          " $R^j_{\\mathrm{ind}}(t)$ (per location)")
    for est_key, est_tex, sfx in [
        ("ind_iw",
         "R^j_{\\mathrm{ind},\\mathrm{iw}}",
         "ind_iw"),
        ("ind_pw",
         "R^j_{\\mathrm{ind},\\mathrm{pw}}",
         "ind_pw"),
        ("ind_Rw",
         "R^j_{\\mathrm{ind},\\mathrm{Rw}}",
         "ind_Rw"),
        ("ind_am",
         "R^j_{\\mathrm{ind},\\mathrm{am}}",
         "ind_am"),
    ]:
        for ref_key, ref_tex, ref_sfx in refs:
            _plot_naive_3x3(
                all_data, est_key, ref_key, est_tex, ref_tex,
                p1, f"{save_prefix}_naive_{sfx}_{ref_sfx}.pdf")

    # ── Part 2: R_out from model, aggregated ─────────────────────────────
    p2 = ("Part~2 — Aggregated $R^j_{\\mathrm{out}}(t)$"
          " (from $\\mathbf{R}$ matrix)")
    for est_key, est_tex, sfx in [
        ("out_iw",
         "\\hat{R}^{\\mathrm{out}}_{\\mathrm{iw}}",
         "out_iw"),
        ("out_pw",
         "\\hat{R}^{\\mathrm{out}}_{\\mathrm{pw}}",
         "out_pw"),
        ("out_Rw",
         "\\hat{R}^{\\mathrm{out}}_{\\mathrm{Rw}}{=}\\mathcal{E}(t)",
         "out_Rw"),
        ("out_am",
         "\\hat{R}^{\\mathrm{out}}_{\\mathrm{am}}",
         "out_am"),
    ]:
        for ref_key, ref_tex, ref_sfx in refs:
            _plot_naive_3x3(
                all_data, est_key, ref_key, est_tex, ref_tex,
                p2, f"{save_prefix}_naive_{sfx}_{ref_sfx}.pdf")

    # ── Part 3: aggregate independent R on total incidence ───────────────
    p3 = ("Part~3 — Aggregate independent"
          " $R_{\\mathrm{ind}}(t)$ on total incidence")
    for ref_key, ref_tex, ref_sfx in refs:
        _plot_naive_3x3(
            all_data, "agg", ref_key,
            "R_{\\mathrm{ind}}", ref_tex,
            p3, f"{save_prefix}_naive_agg_{ref_sfx}.pdf")


def plot_main_bias_figure(sim_A, sim_B, sim_C, city_A, city_B, city_C,
                          gen_time_pmf, save_prefix="fig"):
    """Main-manuscript figure: spatial aggregation bias from two sources.

    Combines in one figure the two distinct routes by which spatial aggregation
    produces bias in estimating the network reproduction number ρ(R(t)):

      i)  Weighting independent estimators: R^j_{ind,Rw}(t) — the R_out-weighted
          average of per-location sliding-window estimates R^j_{ind}(t).
      ii) Weighting outward reproduction numbers: E(t) = X(1,t) — the
          R_out-weighted mean of R^j_out(t).

    Layout — 3 rows (Scenario A / B / C) × 3 columns:
      Col 0  Time series: ρ(R(t)) [solid, dark grey], R^j_{ind,Rw}(t) [dashed,
             orange], E(t) [dotted, reddish-purple].  Colours are FIXED across
             all rows so the legend is interpretable without per-row lookup.
      Col 1  Signed bias R^j_{ind,Rw} − ρ(R).
      Col 2  Signed bias E(t) − ρ(R).
             Both bias panels use IDENTICAL fill colours:
               vermillion (#D55E00) = over-estimate (positive bias)
               sky-blue   (#56B4E9) = under-estimate (negative bias)
             Statistics (Mean bias, MAE, MSE) shown in black.

    Panel labels A–I.
    Saved as: {save_prefix}_main_bias_combined.pdf
    """
    scenarios = [
        (sim_A, city_A, "Dense urban (Scenario A)"),
        (sim_B, city_B, "Sparse national (Scenario B)"),
        (sim_C, city_C, "Hub-amplified (Scenario C)"),
    ]
    print("  Computing naive estimator suite for combined bias figure ...")
    all_data = []
    for sim, city, label in scenarios:
        d = _compute_naive_suite(sim, city, gen_time_pmf)
        all_data.append((d, label))

    # ── Fixed colours — identical in every row ────────────────────────────
    COL_RHO   = "#333333"       # dark charcoal  — ρ(R(t)), the truth
    COL_IND   = OKABE_ITO[0]    # orange         — R^j_{ind,Rw}
    COL_ET    = OKABE_ITO[6]    # reddish-purple — E(t)
    COL_OVER  = "#D55E00"       # vermillion     — positive bias (both cols 1 & 2)
    COL_UNDER = "#56B4E9"       # sky blue       — negative bias (both cols 1 & 2)
    COL_LINE  = "0.30"          # dark grey bias trace
    PANEL_IDS = list("ABCDEFGHI")

    # scenario colours used only for title accent
    SC_COLS = [OKABE_ITO[4], OKABE_ITO[2], OKABE_ITO[5]]

    fig = plt.figure(figsize=(7.2, 7.8))
    # No suptitle: extra top headroom + row spacing so the two-line scenario
    # titles and the bold panel letters are all fully visible.
    gs  = gridspec.GridSpec(3, 3, hspace=0.95, wspace=0.52,
                            left=0.11, right=0.97, top=0.95, bottom=0.06)

    panel_idx = 0
    for row, (d, sc_label) in enumerate(all_data):
        t_arr     = np.arange(d["T"])
        rho_t     = d["rho_t"]
        ind_Rw    = d["ind_Rw"]
        E_t       = d["out_Rw"]
        peak_t    = d["peak_t"]
        sc_col    = SC_COLS[row]

        valid_ind = ~np.isnan(ind_Rw)
        valid_rho = rho_t > 0.05

        bias_ind = np.where(valid_rho & valid_ind, ind_Rw - rho_t, np.nan)
        bias_E   = np.where(valid_rho, E_t - rho_t, np.nan)

        def _stats(b):
            v = b[~np.isnan(b)]
            if v.size == 0:
                return np.nan, np.nan, np.nan
            return float(v.mean()), float(np.mean(np.abs(v))), float(np.mean(v**2))

        # ── Col 0: time series (fixed colours across rows) ─────────────────
        ax = fig.add_subplot(gs[row, 0])
        ax.plot(t_arr[valid_rho], rho_t[valid_rho],
                color=COL_RHO, lw=1.5, label=r"$\mathcal{R}(t)$", zorder=4)
        ax.plot(t_arr[valid_ind], ind_Rw[valid_ind],
                color=COL_IND, lw=1.0, ls="--",
                label=r"$R^j_{\mathrm{ind},Rw}(t)$", zorder=5)
        ax.plot(t_arr, E_t, color=COL_ET, lw=1.0, ls=":",
                label=r"$\mathcal{E}(t)$", zorder=5)
        ax.axhline(1.0, color="0.60", ls=":", lw=0.7, zorder=1)
        ax.axvline(peak_t, color="0.55", ls="--", lw=0.7, alpha=0.45, zorder=1)
        ax.set_xlabel("Day $t$", fontsize=7)
        ax.set_ylabel("Reproduction number", fontsize=7)
        ax.set_title(f"{sc_label}\nEstimators vs $\\mathcal{{R}}(t)$",
                     fontsize=6.5, pad=3, color=sc_col, fontweight="bold")
        ax.legend(fontsize=4.8, borderpad=0.3, labelspacing=0.10,
                  loc="upper right", ncol=1)
        ax.tick_params(labelsize=6)
        _panel_label(ax, PANEL_IDS[panel_idx], x=-0.17, y=1.26); panel_idx += 1

        # ── helper: draw one bias panel ────────────────────────────────────
        def _bias_panel(ax_b, bias, ylabel, title_suffix):
            bv = ~np.isnan(bias)
            if bv.any():
                pos = bv & (bias > 0)
                neg = bv & (bias < 0)
                if pos.any():
                    ax_b.fill_between(t_arr, 0, np.where(pos, bias, 0),
                                      color=COL_OVER, alpha=0.38,
                                      label="Over-estimate")
                if neg.any():
                    ax_b.fill_between(t_arr, 0, np.where(neg, bias, 0),
                                      color=COL_UNDER, alpha=0.38,
                                      label="Under-estimate")
                ax_b.plot(t_arr[bv], bias[bv], color=COL_LINE, lw=0.8, zorder=3)
            ax_b.axhline(0, color="0.60", ls="--", lw=0.7)
            ax_b.axvline(peak_t, color="0.55", ls="--", lw=0.7, alpha=0.45)
            # statistics in black
            mn, mae, mse = _stats(bias)
            stats_str = (f"Bias $= {mn:+.3f}$\n"
                         f"MAE $= {mae:.3f}$\n"
                         f"MSE $= {mse:.4f}$")
            ax_b.text(0.03, 0.97, stats_str,
                      transform=ax_b.transAxes, fontsize=5,
                      ha="left", va="top", color="black",
                      bbox=dict(facecolor="white", edgecolor="0.80",
                                linewidth=0.5, alpha=0.85, pad=2.0))
            if bv.any():
                ax_b.legend(fontsize=5.0, borderpad=0.3, labelspacing=0.12,
                            loc="lower right")
            ax_b.set_xlabel("Day $t$", fontsize=7)
            ax_b.set_ylabel(ylabel, fontsize=7)
            ax_b.set_title(f"{sc_label}\n{title_suffix}",
                           fontsize=6.5, pad=3, color=sc_col, fontweight="bold")
            ax_b.tick_params(labelsize=6)

        # ── Col 1: bias from independent estimator ──────────────────────────
        ax = fig.add_subplot(gs[row, 1])
        _bias_panel(ax, bias_ind,
                    r"$R^j_{\mathrm{ind},Rw} - \mathcal{R}(t)$",
                    r"Bias — $R^j_{\mathrm{ind},Rw}$ weighting")
        _panel_label(ax, PANEL_IDS[panel_idx], x=-0.17, y=1.26); panel_idx += 1

        # ── Col 2: bias from outward R weighting ────────────────────────────
        ax = fig.add_subplot(gs[row, 2])
        _bias_panel(ax, bias_E,
                    r"$\mathcal{E}(t) - \mathcal{R}(t)$",
                    r"Bias — $\mathcal{E}(t)$ weighting")
        _panel_label(ax, PANEL_IDS[panel_idx], x=-0.17, y=1.26); panel_idx += 1

    fname = f"{save_prefix}_main_bias_combined.pdf"
    plt.savefig(fname, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved {fname}")


def plot_type_repro(sim_A, sim_B, city_A, city_B, save_prefix):
    """
    Three publication-quality figures for type reproduction numbers T_j(t).

    Key mathematical property exploited:
        T_j(t) = R_{jj} + R_{jJ}(I-R_{JJ})^{-1}R_{Jj}      [Eq 54]
        T^P_type = ρ(R_{PP} + R_{PQ}(I-R_{QQ})^{-1}R_{QP})  [Eq 56]

    T_j is UNDEFINED (=∞) when ρ(R_{JJ}) ≥ 1, meaning the background
    network J alone sustains the epidemic.  At R₀=1.5 and home fractions
    0.60–0.98, most nodes have R_{jj} > 1 individually — each district is
    self-sustaining — so T_j = ∞ throughout the epidemic growth phase.
    T_j first becomes finite precisely when ρ(R) crosses 1 (threshold
    theorem: T_j > 1 ⟺ ρ(R) > 1 for irreducible non-negative matrices).

    Figure layout
    -------------
    1. *_type_heatmaps.png  — 2×2: T_j(t) heatmap (Lagos/Zambia) +
                               R_jj(t)/ρ(t) self-sustaining fraction lines.
    2. *_type_surfaces.png  — 2×2: 3-D T_j surface + 3-D R_jj surface,
                               for both scenarios.
    3. *_type_groups.png    — 3×2: (a-b) T^P vs ρ(R) trajectory showing
                               when groups become controllable;
                               (c-d) R_jj(t) by type (epidemic drivers);
                               (e-f) verification scatter T_j vs ρ.

    Literature
    ----------
    Roberts & Heesterbeek 2003 (Proc R Soc B) — original type R number.
    Heesterbeek & Roberts 2007 — threshold properties.
    Svensson 2020 (Math Biosci) — type R number for structured populations.
    Melegaro et al. 2017 (PLOS ONE) — African urban/rural contact rates.
    """
    from mpl_toolkits.mplot3d import Axes3D          # noqa: F401
    from matplotlib.colors import TwoSlopeNorm, Normalize

    coords_A, pops_A, dists_A, types_A, meta_A = city_A
    coords_B, pops_B, dists_B, types_B, meta_B = city_B

    N = len(types_A)
    T = sim_A["R_matrices"].shape[0]
    days = np.arange(T)
    oi = OKABE_ITO

    # ── compute T_j(t), R_jj(t), and ρ(t) ──────────────────────────────────
    print("    computing R^j_type(t) — Scenario A  ...", flush=True)
    T_ser_A = np.array([type_reproduction_numbers(sim_A["R_matrices"][t])
                        for t in range(T)], dtype=float)   # (T, N)
    Rjj_A   = np.array([np.diag(sim_A["R_matrices"][t]) for t in range(T)])  # (T,N)
    rho_A   = np.array([R_system(sim_A["R_matrices"][t]) for t in range(T)])

    print("    computing R^j_type(t) — Scenario B ...", flush=True)
    T_ser_B = np.array([type_reproduction_numbers(sim_B["R_matrices"][t])
                        for t in range(T)], dtype=float)
    Rjj_B   = np.array([np.diag(sim_B["R_matrices"][t]) for t in range(T)])  # (T,N)
    rho_B   = np.array([R_system(sim_B["R_matrices"][t]) for t in range(T)])

    # ── sort locations by node-type tier ────────────────────────────────────
    type_order_A = ["core", "dense", "suburban", "peripheral"]
    type_order_B = ["capital", "peri-capital", "urban-industrial",
                    "semi-urban", "rural", "remote-rural"]

    def _sorted_idx(types, order):
        om = {t: i for i, t in enumerate(order)}
        return sorted(range(len(types)), key=lambda j: om.get(types[j], 99))

    idx_A = _sorted_idx(types_A, type_order_A)
    idx_B = _sorted_idx(types_B, type_order_B)
    lab_A = [f"{types_A[j].capitalize()} #{j+1}" for j in idx_A]
    lab_B = [f"{types_B[j].replace('-', '\u2011').title()} #{j+1}" for j in idx_B]

    def _cross1(rho):
        c = np.where(rho < 1.0)[0]
        return int(c[0]) if len(c) > 0 else T

    cross_A = _cross1(rho_A)
    cross_B = _cross1(rho_B)

    # ── canonical groups ────────────────────────────────────────────────────
    groups_A = {
        "Core only":          {"core"},
        "Dense only":         {"dense"},
        "Suburban only":      {"suburban"},
        "Peripheral only":    {"peripheral"},
        "Core + Dense":       {"core", "dense"},
        "Core+Dense+Sub":     {"core", "dense", "suburban"},
    }
    groups_B = {
        "Capital only":       {"capital"},
        "Capital + Peri":     {"capital", "peri-capital"},
        "All urban":          {"capital", "peri-capital", "urban-industrial"},
        "Urban + Semi-urban": {"capital", "peri-capital",
                               "urban-industrial", "semi-urban"},
        "Rural only":         {"rural"},
        "Remote-rural only":  {"remote-rural"},
    }

    def _group_series(R_mats, node_types, groups):
        out = {}
        for lbl, tset in groups.items():
            P = _group_indices(node_types, tset)
            if not P:
                continue
            out[lbl] = np.array(
                [type_reproduction_number_group(R_mats[t], P) for t in range(T)],
                dtype=float)
        return out

    print("    computing group R^P_type(t) — Scenario A  ...", flush=True)
    grp_A = _group_series(sim_A["R_matrices"], types_A, groups_A)
    print("    computing group R^P_type(t) — Scenario B ...", flush=True)
    grp_B = _group_series(sim_B["R_matrices"], types_B, groups_B)

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Figure 1 — T_j heatmaps (top) + R_jj/ρ(t) self-sustaining fraction (bottom)
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    from matplotlib.patches import Patch
    from matplotlib.colors import TwoSlopeNorm

    fig = plt.figure(figsize=(15, 10))
    gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.48, wspace=0.32)

    norm_div = TwoSlopeNorm(vmin=0.0, vcenter=1.0, vmax=5.0)
    cmap_div = plt.cm.RdBu_r

    # ── row 0: T_j heatmaps ──────────────────────────────────────────────
    for col, (T_ser, idx, labs, rho_s, xday, title, panel_lbl) in enumerate([
        (T_ser_A, idx_A, lab_A, rho_A, cross_A, "Scenario A: Dense urban", "a"),
        (T_ser_B, idx_B, lab_B, rho_B, cross_B, "Scenario B: Sparse national", "b"),
    ]):
        ax = fig.add_subplot(gs[0, col])
        data   = T_ser[:, idx].T          # (N, T)
        masked = np.ma.masked_invalid(data)

        X, Y = np.meshgrid(np.arange(T + 1), np.arange(N + 1))
        im = ax.pcolormesh(X, Y - 0.5, masked, cmap=cmap_div, norm=norm_div,
                           shading="flat")
        # Grey overlay for NaN (T_j = ∞: background self-sustaining)
        nan_mask_pm = np.where(np.isnan(data), 1.0, np.nan)
        ax.pcolormesh(X, Y - 0.5, nan_mask_pm, cmap="Greys_r",
                      vmin=0.5, vmax=1.5, shading="flat", alpha=0.55)
        ax.axvline(xday, color="#111111", lw=1.4, ls="--", alpha=0.9)

        # Annotation box explaining grey = R^j_type = ∞
        ax.text(xday * 0.30, N * 0.82,
                r"$R^j_{\rm type} = \infty$" + "\n" + r"(each node self-sustaining)" + "\n"
                r"$R_{jj}(0) > 1$",
                fontsize=6.5, color="#333333", ha="center", va="top",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                          edgecolor="0.65", linewidth=0.6, alpha=0.88))
        ax.text(xday + (T - xday) * 0.45, N * 0.82,
                r"$R^j_{\rm type} < 1$" + "\n" + r"(controlled)",
                fontsize=6.5, color="#1a6ab5", ha="center", va="top",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                          edgecolor="0.65", linewidth=0.6, alpha=0.88))

        ax.set_yticks(np.arange(N))
        ax.set_yticklabels(labs, fontsize=7.5)
        ax.set_xlabel("Day", fontsize=9)
        ax.set_title(title, fontsize=10, pad=6)

        handles = [
            Line2D([0], [0], color="#111111", lw=1.4, ls="--",
                   label=f"$\\rho(t)=1$, day {xday}"),
            Patch(facecolor="#aaaaaa", edgecolor="none",
                  label=r"$R^j_{\rm type}=\infty$"),
        ]
        ax.legend(handles=handles, fontsize=7, loc="lower right",
                  framealpha=0.80, handlelength=1.2)
        cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.03, extend="max")
        cb.set_label("$R^j_{\\rm type}(t)$", fontsize=8.5)
        cb.ax.axhline(norm_div(1.0), color="#333", lw=0.9, ls=":")
        _panel_label(ax, panel_lbl, x=-0.16)

    # ── row 1: R_jj(t) self-sustaining fraction ──────────────────────────
    # R_jj(t) shows WHEN each location transitions from self-sustaining (>1) to
    # network-dependent (<1). This EXPLAINS the NaN pattern above.
    for col, (Rjj, idx, labs, rho_s, xday, title, panel_lbl) in enumerate([
        (Rjj_A, idx_A, lab_A, rho_A, cross_A,
         r"$R_{jj}(t)$: within-location reproduction — Scenario A", "c"),
        (Rjj_B, idx_B, lab_B, rho_B, cross_B,
         r"$R_{jj}(t)$: within-location reproduction — Scenario B", "d"),
    ]):
        ax = fig.add_subplot(gs[1, col])
        for k, orig_j in enumerate(idx):
            ax.plot(days, Rjj[:, orig_j],
                    color=oi[k % len(oi)], lw=1.4, label=labs[k], alpha=0.85)

        ax.axhline(1.0, color="#333333", lw=1.0, ls="--", alpha=0.8,
                   label="$R_{jj}=1$", zorder=4)
        ax.axvline(xday, color="#888888", lw=0.9, ls=":", alpha=0.7)
        ax.axvspan(0, xday, alpha=0.05, color=oi[4], zorder=0)

        # Shade region where T_j is defined (Rjj < 1 for ALL j simultaneously)
        # This is approximately the post-threshold region
        ax.text(0.02, 0.97,
                r"$R_{jj}(t)>1$: node self-sustaining" + "\n"
                r"$\Rightarrow T_j=\infty$ for all other nodes" + "\n"
                r"(background can sustain epidemic alone)",
                transform=ax.transAxes, fontsize=6, va="top", ha="left",
                color="#555555",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="0.97",
                          edgecolor="0.70", linewidth=0.5))

        ax.set_xlabel("Day", fontsize=9)
        ax.set_ylabel("$R_{jj}(t)$", fontsize=9)
        ax.set_title(title, fontsize=9, pad=5)
        ax.set_ylim(bottom=0)
        ax.legend(fontsize=6.5, ncol=2, loc="upper right",
                  framealpha=0.80, handlelength=1.2)
        _panel_label(ax, panel_lbl)

    plt.suptitle(
        r"Type reproduction numbers $R^j_{\rm type}(t)$ and within-location drivers $R_{jj}(t)$",
        fontsize=11, y=1.01)
    fname1 = f"{save_prefix}_type_heatmaps.png"
    plt.savefig(fname1, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved {fname1}")

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Figure 2 — 3-D surfaces: T_j(t) and R_jj(t) side by side
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    fig = plt.figure(figsize=(16, 9))
    gs2 = gridspec.GridSpec(2, 2, figure=fig, hspace=0.10, wspace=0.02)

    surf_cmap_T   = plt.cm.plasma
    surf_cmap_Rjj = plt.cm.viridis
    panel_it = iter(["a", "b", "c", "d"])

    for row, (T_ser, Rjj, idx, rho_s, xday, title_scen) in enumerate([
        (T_ser_A, Rjj_A, idx_A, rho_A, cross_A, "Scenario A: Dense urban"),
        (T_ser_B, Rjj_B, idx_B, rho_B, cross_B, "Scenario B: Sparse national"),
    ]):
        XX, YY = np.meshgrid(days, np.arange(N))

        # left: T_j(t) surface
        ax1 = fig.add_subplot(gs2[row, 0], projection="3d")
        ax1.set_facecolor("white")
        Z_T      = T_ser[:, idx].T
        Z_T_masked = np.ma.masked_invalid(Z_T)
        surf1 = ax1.plot_surface(XX, YY, Z_T_masked, cmap=surf_cmap_T,
                                  linewidth=0, antialiased=True, alpha=0.88,
                                  rstride=1, cstride=max(1, T // 60))
        ax1.plot_surface(XX, YY, np.ones_like(Z_T_masked),
                         color="grey", alpha=0.10, linewidth=0)
        ax1.set_xlabel("Day", fontsize=7, labelpad=4)
        ax1.set_ylabel("Location", fontsize=7, labelpad=4)
        ax1.set_zlabel("$R^j_{\\rm type}(t)$", fontsize=7, labelpad=3)
        ax1.set_yticks(np.arange(N))
        ax1.set_yticklabels([f"L{idx[j]+1}" for j in range(N)], fontsize=5.5)
        ax1.set_title(f"{title_scen}: $R^j_{{\\rm type}}(t)$", fontsize=9, pad=6)
        ax1.view_init(elev=26, azim=-52)
        ax1.tick_params(labelsize=6)
        cb1 = fig.colorbar(surf1, ax=ax1, fraction=0.022, pad=0.08, shrink=0.60)
        cb1.set_label("$R^j_{\\rm type}(t)$", fontsize=7)
        _panel_label_3d(ax1, next(panel_it))

        # right: R_jj(t) surface
        ax2 = fig.add_subplot(gs2[row, 1], projection="3d")
        Z_R  = Rjj[:, idx].T
        surf2 = ax2.plot_surface(XX, YY, Z_R, cmap=surf_cmap_Rjj,
                                  linewidth=0, antialiased=True, alpha=0.88,
                                  rstride=1, cstride=max(1, T // 60))
        ax2.plot_surface(XX, YY, np.ones_like(Z_R),
                         color="crimson", alpha=0.12, linewidth=0)
        ax2.set_xlabel("Day", fontsize=7, labelpad=4)
        ax2.set_ylabel("Location", fontsize=7, labelpad=4)
        ax2.set_zlabel("$R_{jj}(t)$", fontsize=7, labelpad=3)
        ax2.set_yticks(np.arange(N))
        ax2.set_yticklabels([f"L{idx[j]+1}" for j in range(N)], fontsize=5.5)
        ax2.set_title(f"{title_scen}: $R_{{jj}}(t)$", fontsize=9, pad=6)
        ax2.view_init(elev=26, azim=-52)
        ax2.tick_params(labelsize=6)
        cb2 = fig.colorbar(surf2, ax=ax2, fraction=0.022, pad=0.08, shrink=0.60)
        cb2.set_label("$R_{jj}(t)$", fontsize=7)
        _panel_label_3d(ax2, next(panel_it))

    plt.suptitle(
        r"3-D view: $R^j_{\rm type}(t)$ (controllability) and $R_{jj}(t)$ (self-sustaining transmission)",
        fontsize=11, y=1.01)
    fname2 = f"{save_prefix}_type_surfaces.png"
    plt.savefig(fname2, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved {fname2}")

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Figure 3 — Group T^P vs ρ(R) + R_jj by type + verification
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    from matplotlib.colors import Normalize

    cmap_t = plt.cm.cividis

    fig, axes = plt.subplots(4, 2, figsize=(14, 17),
                             gridspec_kw={"hspace": 0.55, "wspace": 0.35})

    # ── Row 0: T^P vs ρ(R) trajectory ───────────────────────────────────────
    # x = ρ(R(t))  [decreasing from R0→0], y = T^P(t) [only where finite]
    # This shows the threshold relationship T^P > 1 ⟺ ρ > 1 clearly,
    # and exactly WHEN in the epidemic each group's T^P first becomes finite.
    for ax, grp, rho_s, title, panel_lbl in [
        (axes[0, 0], grp_A, rho_A,
         r"Group $R^{\mathcal{P}}_{\rm type}$ vs $\rho(t)$ — Scenario A: Dense urban", "a"),
        (axes[0, 1], grp_B, rho_B,
         r"Group $R^{\mathcal{P}}_{\rm type}$ vs $\rho(t)$ — Scenario B: Sparse national", "b"),
    ]:
        for k, (lbl, vals) in enumerate(grp.items()):
            mask = np.isfinite(vals)
            if mask.any():
                ax.plot(rho_s[mask], vals[mask],
                        color=oi[k % len(oi)], lw=1.8, label=lbl,
                        alpha=0.90, solid_capstyle="round")
                # Mark the first point where T^P becomes defined
                first = int(np.where(mask)[0][0])
                ax.scatter(rho_s[first], vals[first],
                           color=oi[k % len(oi)], s=35, zorder=5,
                           edgecolors="white", linewidths=0.6)

        ax.axhline(1.0, color="#333333", lw=1.0, ls="--", alpha=0.8,
                   label="$R^{\\mathcal{P}}_{\\rm type}=1$", zorder=4)
        ax.axvline(1.0, color="#333333", lw=1.0, ls=":", alpha=0.8,
                   label="$\\rho=1$", zorder=4)
        ax.text(0.97, 0.97,
                r"Trajectory direction: $\rho$ decreasing" + "\n"
                r"Dots = first time $R^{\mathcal{P}}_{\rm type}$ finite ($\rho(R_{QQ})<1$)",
                transform=ax.transAxes, fontsize=6, va="top", ha="right",
                color="#444444", style="italic",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="0.96",
                          edgecolor="0.65", linewidth=0.5))
        ax.set_xlabel(r"$\rho\!\left(\mathbf{R}(t)\right)$", fontsize=9)
        ax.set_ylabel(r"$R^{\mathcal{P}}_{\rm type}(t)$", fontsize=9)
        ax.set_title(title, fontsize=9.5, pad=5)
        ax.set_xlim(0, None)
        ax.set_ylim(0, None)
        ax.legend(fontsize=6.5, loc="upper left", framealpha=0.80,
                  handlelength=1.2, ncol=1)
        _panel_label(ax, panel_lbl)

    # ── Row 1: R_jj(t) by type, showing epidemic drivers ────────────────────
    # Aggregate per node-type: mean R_jj(t) within each type group
    for ax, Rjj, types, rho_s, xday, title, panel_lbl in [
        (axes[1, 0], Rjj_A, types_A, rho_A, cross_A,
         r"$R_{jj}(t)$ by node type — Scenario A: Dense urban", "c"),
        (axes[1, 1], Rjj_B, types_B, rho_B, cross_B,
         r"$R_{jj}(t)$ by node type — Scenario B: Sparse national", "d"),
    ]:
        unique_types = list(dict.fromkeys(types))   # preserve order
        for k, ntype in enumerate(unique_types):
            idxs = [j for j, t in enumerate(types) if t == ntype]
            mean_Rjj = Rjj[:, idxs].mean(axis=1)
            ax.plot(days, mean_Rjj, color=oi[k % len(oi)], lw=1.8,
                    label=ntype.capitalize(), alpha=0.88,
                    solid_capstyle="round")
            if len(idxs) > 1:
                ax.fill_between(days, Rjj[:, idxs].min(axis=1),
                                Rjj[:, idxs].max(axis=1),
                                color=oi[k % len(oi)], alpha=0.12)

        ax.axhline(1.0, color="#333333", lw=1.0, ls="--", alpha=0.8,
                   label="$R_{jj}=1$", zorder=4)
        ax.axvline(xday, color="#888888", lw=0.8, ls=":", alpha=0.7)
        ax.text(0.02, 0.97,
                r"$R_{jj}>1$: node sustains epidemic alone" + "\n"
                r"$\Rightarrow$ $R^j_{\rm type}({\rm other})=\infty$" + "\n"
                r"$R_{jj}<1$: node needs spatial coupling",
                transform=ax.transAxes, fontsize=6, va="top", ha="left",
                color="#555555",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="0.97",
                          edgecolor="0.68", linewidth=0.5))
        ax.set_xlabel("Day", fontsize=9)
        ax.set_ylabel("$R_{jj}(t)$  (mean ± range by type)", fontsize=8.5)
        ax.set_title(title, fontsize=9.5, pad=5)
        ax.set_ylim(bottom=0)
        ax.legend(fontsize=7, loc="upper right", framealpha=0.80,
                  handlelength=1.2)
        _panel_label(ax, panel_lbl)

    # ── Row 2: verification scatter ─────────────────────────────────────────
    norm_t  = Normalize(0, T)
    for ax, T_ser, rho_s, scen_lbl, panel_lbl in [
        (axes[2, 0], T_ser_A, rho_A, "Scenario A: Dense urban",      "e"),
        (axes[2, 1], T_ser_B, rho_B, "Scenario B: Sparse national",  "f"),
    ]:
        for j in range(N):
            ax.scatter(rho_s, T_ser[:, j],
                       c=days, cmap=cmap_t, norm=norm_t,
                       s=5, alpha=0.5, linewidths=0)

        ax.axhline(1.0, color="#333333", lw=1.0, ls="--", alpha=0.8,
                   label="$R^j_{\\rm type}=1$", zorder=4)
        ax.axvline(1.0, color="#333333", lw=1.0, ls=":", alpha=0.8,
                   label="$\\rho(t)=1$", zorder=4)

        xr = rho_s.max() * 1.06
        yr = max(np.nanmax(T_ser) * 1.08 if np.any(np.isfinite(T_ser)) else 2, 2)
        ax.set_xlim(0, xr); ax.set_ylim(0, yr)

        ax.text(0.02, 0.97,
                r"$\rho>1$: $R^j_{\rm type}=\infty$ (grey = NaN)" + "\n"
                r"$\rho<1$: $R^j_{\rm type}<1$ ✓ threshold theorem",
                transform=ax.transAxes, fontsize=6.5, va="top", ha="left",
                color="#444444",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="0.96",
                          edgecolor="0.68", linewidth=0.5))
        ax.text(0.98, 0.03, "Roberts & Heesterbeek 2003",
                transform=ax.transAxes, fontsize=5.5, va="bottom", ha="right",
                color="#777777", style="italic")

        ax.set_xlabel(r"$\rho\!\left(\mathbf{R}(t)\right)$", fontsize=9)
        ax.set_ylabel("$R^j_{\\rm type}(t)$", fontsize=9)
        ax.set_title(
            f"Verification: $R^j_{{\\rm type}}>1 \\Leftrightarrow \\rho>1$ — {scen_lbl}",
            fontsize=9, pad=5)
        ax.legend(fontsize=7.5, loc="lower right", framealpha=0.80,
                  handlelength=1.2)
        cb = fig.colorbar(
            plt.cm.ScalarMappable(cmap=cmap_t, norm=norm_t),
            ax=ax, fraction=0.032, pad=0.03)
        cb.set_label("Day $t$", fontsize=8)
        _panel_label(ax, panel_lbl)

    # ── Row 3: Mean R^j_type(t) by NODE TYPE — tile plot ────────────────────
    # Each row = one node type; value = nanmean of T_j(t) across all
    # locations of that type. Grey cells = all locations in type have
    # T_j=∞ (background network is self-sustaining; undefined by construction).
    for ax, T_ser, types, t_order, rho_s, xday, title, panel_lbl in [
        (axes[3, 0], T_ser_A, types_A, type_order_A, rho_A, cross_A,
         r"Mean $R^j_{\rm type}(t)$ by node type — Scenario A", "g"),
        (axes[3, 1], T_ser_B, types_B, type_order_B, rho_B, cross_B,
         r"Mean $R^j_{\rm type}(t)$ by node type — Scenario B", "h"),
    ]:
        n_types = len(t_order)
        # Aggregate: for each type compute nanmean of T_j(t) across its locations
        type_data = np.full((n_types, T), np.nan)
        for ti, ntype in enumerate(t_order):
            idxs = [j for j, nt in enumerate(types) if nt == ntype]
            if not idxs:
                continue
            Tj_group = T_ser[:, idxs]   # (T, n_locs_of_type)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                type_data[ti] = np.nanmean(Tj_group, axis=1)  # (T,)
        # Cap at 5.0 for colourmap readability
        data_plot = np.where(np.isfinite(type_data),
                             np.minimum(type_data, 5.0), np.nan)
        masked = np.ma.masked_invalid(data_plot)
        from matplotlib.colors import TwoSlopeNorm as _TSN
        norm_tile = _TSN(vmin=0.0, vcenter=1.0, vmax=5.0)
        im = ax.pcolormesh(np.arange(T + 1), np.arange(n_types + 1) - 0.5,
                           masked, cmap="RdBu_r", norm=norm_tile,
                           shading="flat")
        # Grey overlay for fully-undefined cells (all locs of this type: T_j=∞)
        nan_overlay = np.where(np.isnan(type_data), 1.0, np.nan)
        ax.pcolormesh(np.arange(T + 1), np.arange(n_types + 1) - 0.5,
                      nan_overlay, cmap="Greys_r",
                      vmin=0.5, vmax=1.5, shading="flat", alpha=0.50)
        ax.axvline(xday, color="#111111", lw=1.2, ls="--", alpha=0.85)
        ax.set_yticks(np.arange(n_types))
        ax.set_yticklabels([t.replace("-", "\u2011").title()
                            for t in t_order], fontsize=7)
        ax.set_xlabel("Day $t$", fontsize=9)
        ax.set_title(title, fontsize=9, pad=5)
        cb_t = fig.colorbar(im, ax=ax, fraction=0.032, pad=0.03, extend="max")
        cb_t.set_label(r"Mean $R^j_{\rm type}(t)$", fontsize=7.5)
        cb_t.ax.axhline(norm_tile(1.0), color="#333", lw=0.8, ls=":")
        from matplotlib.patches import Patch as _Patch
        handles_t = [
            Line2D([0], [0], color="#111111", lw=1.2, ls="--",
                   label=f"$\\rho(t)=1$, day {xday}"),
            _Patch(facecolor="#aaaaaa", edgecolor="none",
                   label=r"$R^j_{\rm type}=\infty$ (undefined)"),
        ]
        ax.legend(handles=handles_t, fontsize=6.5, loc="lower right",
                  framealpha=0.80, handlelength=1.2)
        _panel_label(ax, panel_lbl)

    plt.suptitle(
        r"Group $R^{\mathcal{P}}_{\rm type}$ vs $\rho(t)$, within-location drivers $R_{jj}(t)$, "
        r"and threshold verification",
        fontsize=11, y=1.005)
    fname3 = f"{save_prefix}_type_groups.png"
    plt.savefig(fname3, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved {fname3}")


def plot_fig_C_transience(sim_C, city_data, f_C, save_prefix="fig"):
    """Scenario C — hub-amplified transient amplification zone.

    A dedicated publication-quality figure for the non-normal hub-and-spoke
    scenario, emphasising the transient zone σ(t) > 1, ρ(t) < 1.

    Layout (2-row):
      Top row (3 panels): a mobility matrix | b incidence heatmap | c R_{kj} at peak
      Bottom (1 wide):    d ρ(t) & σ(t) — transient zone with dotted drop-lines
    """
    inc    = sim_C["incidence"]
    R_mats = sim_C["R_matrices"]
    coords, pops, dists, node_types, meta = city_data
    T, N = inc.shape

    rho_ts   = np.array([R_system(R_mats[t])            for t in range(T)])
    sigma_ts = np.array([reactivity(R_mats[t])["sigma"] for t in range(T)])
    peak     = int(inc.sum(axis=1).argmax())
    t_arr    = np.arange(T)
    loc_lbl  = [f"L{i+1}" for i in range(N)]
    # A₁(1) = max_k R^k_out(t) and E(t) = Σ_j (R^j_out)² / Σ_k R^k_out
    R_out_C = np.array([R_outward(R_mats[t]) for t in range(T)])
    A1_1_C  = np.array([float(np.max(R_out_C[t])) for t in range(T)])
    E_t_C   = np.array([np.sum(R_out_C[t]**2) / (np.sum(R_out_C[t]) + 1e-300)
                         for t in range(T)])

    # ── colour palette ──────────────────────────────────────────────────────
    COL_RHO   = "#0072B2"   # deep blue   — ρ(t)
    COL_SIGMA = "#E69F00"   # amber       — σ(t)
    COL_ZONE  = "#E69F00"   # same amber  — fill
    COL_INC   = "#009E73"   # teal        — incidence overlay
    COL_DROP  = "#CC5500"   # burnt sienna — drop-line annotation

    fig = plt.figure(figsize=(7.2, 6.4))
    gs  = gridspec.GridSpec(
        2, 3,
        height_ratios=[1, 1.6],
        hspace=0.52, wspace=0.50,
        left=0.09, right=0.97, top=0.96, bottom=0.09)

    # ── a: mean mobility matrix ─────────────────────────────────────────────
    ax = fig.add_subplot(gs[0, 0])
    f_mean = f_C.mean(axis=0)
    # Mask diagonal to highlight off-diagonal concentration
    f_off  = f_mean.copy(); np.fill_diagonal(f_off, np.nan)
    im = ax.pcolormesh(np.arange(N+1)-0.5, np.arange(N+1)-0.5,
                       f_mean, cmap="YlOrRd", shading="flat",
                       vmin=0, vmax=float(np.nanpercentile(f_off, 98)))
    ax.set_xlim(-0.5, N-0.5); ax.set_ylim(N-0.5, -0.5)
    ax.set_xticks(range(N)); ax.set_xticklabels(loc_lbl, fontsize=4.5, rotation=45)
    ax.set_yticks(range(N)); ax.set_yticklabels(loc_lbl, fontsize=4.5)
    ax.set_xlabel("Activity location $k$", fontsize=6)
    ax.set_ylabel("Residence $j$", fontsize=6)
    ax.set_title("Hub-concentrated mobility\n$\\bar{f}_{jk}$ (off-diag range)", fontsize=6, pad=3)
    cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.ax.set_title("$\\bar{f}_{jk}$", fontsize=5.5, pad=2)
    cb.ax.tick_params(labelsize=5)
    # mark hub (highest population) on axes
    i_hub = int(np.argmax(pops))
    ax.axhline(i_hub, color="#CC5500", lw=0.8, ls="--", alpha=0.7)
    ax.axvline(i_hub, color="#CC5500", lw=0.8, ls="--", alpha=0.7)
    ax.text(i_hub + 0.15, -0.5, "hub", fontsize=4.2, color="#CC5500", va="top")
    _panel_label(ax, "A")

    # ── b: resident population bar chart ────────────────────────────────────
    ax = fig.add_subplot(gs[0, 1])
    i_hub_b = int(np.argmax(pops))
    bar_colors = [COL_DROP if i == i_hub_b else "0.55" for i in range(N)]
    ax.barh(range(N), pops / 1e6, color=bar_colors, height=0.72, edgecolor="none")
    ax.set_yticks(range(N))
    ax.set_yticklabels(loc_lbl, fontsize=5)
    ax.set_xlabel("Resident population (millions)", fontsize=6)
    ax.set_title("Resident population\nper location", fontsize=6, pad=3)
    ax.invert_yaxis()
    ax.tick_params(labelsize=5)
    ax.text(pops[i_hub_b] / 1e6 + 0.03, i_hub_b, "hub",
            fontsize=5, color=COL_DROP, va="center")
    _panel_label(ax, "B")

    # ── c: R_{kj} heatmap at epidemic peak ──────────────────────────────────
    ax = fig.add_subplot(gs[0, 2])
    R_pk = R_mats[peak]
    vmax_pk = float(np.percentile(R_pk[R_pk > 0], 97)) if (R_pk > 0).any() else 1.0
    im = ax.pcolormesh(np.arange(N+1)-0.5, np.arange(N+1)-0.5,
                       R_pk, cmap="plasma", shading="flat",
                       vmin=0, vmax=vmax_pk)
    ax.set_xlim(-0.5, N-0.5); ax.set_ylim(N-0.5, -0.5)
    ax.set_xticks(range(N)); ax.set_xticklabels(loc_lbl, fontsize=4.5, rotation=45)
    ax.set_yticks(range(N)); ax.set_yticklabels(loc_lbl, fontsize=4.5)
    ax.set_xlabel("Infectee $j$", fontsize=6)
    ax.set_ylabel("Infector $k$", fontsize=6)
    ax.set_title(f"$R_{{kj}}$ at epidemic peak (day {peak})", fontsize=6, pad=3)
    cb3 = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb3.ax.set_title("$R_{kj}$", fontsize=5.5, pad=2)
    cb3.ax.tick_params(labelsize=5)
    _panel_label(ax, "C")

    # ── d: transient amplification zone (wide, bottom row) ──────────────────
    gs_bot = gridspec.GridSpecFromSubplotSpec(
        1, 1, subplot_spec=gs[1, :])
    ax  = fig.add_subplot(gs_bot[0])
    ax2 = ax.twinx()
    ax2.spines["right"].set_visible(True)
    ax2.spines["top"].set_visible(False)

    # incidence on right axis (muted, background)
    total_inc = inc.sum(axis=1)
    ax2.fill_between(t_arr, total_inc / 1e3, color=COL_INC, alpha=0.12)
    ax2.plot(t_arr, total_inc / 1e3, color=COL_INC, lw=0.8, alpha=0.55)
    ax2.set_ylabel("Total incidence ($\\times 10^3$)", color=COL_INC, fontsize=6.5)
    ax2.tick_params(axis="y", labelcolor=COL_INC, labelsize=6)
    ax2.set_ylim(bottom=0)

    # ρ(t), σ(t), A₁(1) and E(t) on left axis
    COL_A1 = "#56B4E9"   # sky blue  — A₁(1)
    COL_ET = "#CC79A7"   # reddish-purple — E(t)
    vR = rho_ts > 0; vS = sigma_ts > 0
    vA1C = A1_1_C > 0;  vEC = E_t_C > 0
    ax.plot(t_arr[vR], rho_ts[vR],   color=COL_RHO,   lw=2.0, zorder=4,
            label="$\\mathcal{R}(t) = \\rho(\\mathbf{R}(t))$  [network reproduction number]")
    ax.plot(t_arr[vS], sigma_ts[vS], color=COL_SIGMA,  lw=2.0, ls="--", zorder=4,
            label="$\\sigma(t) = \\|\\mathbf{R}(t)\\|_2$  [reactivity]")
    ax.plot(t_arr[vA1C], A1_1_C[vA1C], color=COL_A1, lw=1.4, ls="-.", zorder=4,
            label="$\\mathcal{A}_1(1) = \\max_k R^k_{\\rm out}$  [first-gen. epidemicity]")
    ax.plot(t_arr[vEC],  E_t_C[vEC],   color=COL_ET, lw=1.4, ls=":", zorder=4,
            label="$\\mathcal{E}(t) = X(1,t)$  [risk-aware reproduction number]")
    ax.axhline(1.0, color="0.40", ls=":", lw=1.1, zorder=2)

    # Transient zone: σ > 1 AND ρ < 1
    tr_mask = (sigma_ts > 1) & (rho_ts < 1)
    if tr_mask.any():
        # amber fill between σ(t) and threshold=1
        ax.fill_between(t_arr, 1.0, sigma_ts,
                        where=tr_mask,
                        color=COL_ZONE, alpha=0.35, zorder=3,
                        label="Transient amplification zone\n($\\sigma > 1$, $\\mathcal{R} < 1$)")

        idx = np.where(tr_mask)[0]
        t_start, t_end = int(idx[0]), int(idx[-1])
        n_days = t_end - t_start + 1

        # dotted vertical drop-lines at zone boundaries
        for t_bd, lbl_txt in [(t_start, f"day {t_start}"), (t_end, f"day {t_end}")]:
            sig_at = float(sigma_ts[t_bd]) if sigma_ts[t_bd] > 0 else 1.0
            ax.plot([t_bd, t_bd], [0, sig_at],
                    color=COL_DROP, lw=1.1, ls=":", zorder=5, alpha=0.85)
            ax.text(t_bd, -0.04,
                    lbl_txt,
                    transform=ax.get_xaxis_transform(),
                    fontsize=5.5, ha="center", va="top",
                    color=COL_DROP)

        # horizontal brace annotation below threshold line
        t_mid = int((t_start + t_end) / 2)
        ax.annotate(
            "",
            xy=(t_end + 0.5, 0.97), xytext=(t_start - 0.5, 0.97),
            xycoords=("data", "axes fraction"),
            textcoords=("data", "axes fraction"),
            arrowprops=dict(arrowstyle="<->", color=COL_DROP, lw=1.1))
        ax.text(t_mid, 0.93,
                f"$\\Delta t = {n_days}$ days of transient growth\n"
                f"despite $\\mathcal{{R}} < 1$",
                transform=ax.get_xaxis_transform(),
                fontsize=6, ha="center", va="top",
                color=COL_DROP,
                bbox=dict(boxstyle="round,pad=0.25", fc="#FFF8DC",
                          ec=COL_DROP, alpha=0.92, lw=0.7))

        # (σ̄/R̄ metric folded into the stats box below)
    else:
        ax.text(0.5, 0.6, "No transient zone detected\nat these parameters",
                transform=ax.transAxes, fontsize=8, ha="center", color="0.45",
                style="italic")

    # axis labels and formatting
    ax.set_xlabel("Day $t$", fontsize=7)
    ax.set_ylabel("$\\mathcal{R}(t)$  /  $\\sigma(t)$", fontsize=7)
    ax.set_ylim(bottom=0)
    y_top = max(float(rho_ts.max()), float(sigma_ts.max())) * 1.15
    ax.set_ylim(0, max(y_top, 1.4))
    ax.tick_params(axis="both", labelsize=6)
    # stats + gap summary in upper right
    att = float(inc.sum() / pops.sum() * 100)
    R0_val = float(rho_ts[rho_ts > 0][0]) if (rho_ts > 0).any() else 0.0
    _active_c = rho_ts > 0.05
    _sdiff_c  = float(np.nanmean((sigma_ts - rho_ts)[_active_c])) if _active_c.any() else 0.0
    _sratio_c = float(np.nanmean((sigma_ts / (rho_ts + 1e-300))[_active_c])) if _active_c.any() else 1.0
    _ediff_c  = float(np.nanmean((E_t_C - rho_ts)[_active_c])) if _active_c.any() else 0.0
    _eratio_c = float(np.nanmean((E_t_C / (rho_ts + 1e-300))[_active_c])) if _active_c.any() else 1.0
    ax.text(0.985, 0.97,
            (f"$\\mathcal{{R}}_0 = {R0_val:.2f}$\n"
             f"Attack rate = {att:.1f}%\n"
             f"Mean $\\sigma - \\mathcal{{R}}$: ${_sdiff_c:+.3f}$"
             f"  ($\\sigma/\\mathcal{{R}} = {_sratio_c:.3f}$)\n"
             f"Mean $\\mathcal{{X}}(1) - \\mathcal{{R}}$: ${_ediff_c:+.3f}$"
             f"  ($\\mathcal{{X}}(1)/\\mathcal{{R}} = {_eratio_c:.3f}$)"),
            transform=ax.transAxes, fontsize=5.0, ha="right", va="top",
            color="0.2",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.70",
                      alpha=0.90, lw=0.5))
    # peak marker
    ax.axvline(peak, color="0.5", lw=0.7, ls="--", alpha=0.6, zorder=1)
    ax.text(peak + 1, ax.get_ylim()[1] * 0.88,
            f"Epidemic\npeak (d{peak})", fontsize=5, color="0.4", va="top")

    ax.set_title("Transient amplification: $\\sigma(t) > 1$ while $\\mathcal{R}(t) < 1$\n"
                 "(hub-amplified non-normal scenario — near-star topology, $\\lambda_b/\\lambda_w = 0.3$)",
                 fontsize=6.5, pad=4)
    leg = ax.legend(fontsize=5.2, borderpad=0.3, labelspacing=0.15,
                    ncol=2, loc="upper center",
                    bbox_to_anchor=(0.5, -0.12), framealpha=0.92)
    _panel_label(ax, "D")


    plt.savefig(f"{save_prefix}_scenario_C_transience.pdf", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved {save_prefix}_scenario_C_transience.pdf")


def plot_power_mean_spectrum(sim, city_data, scenario_label, save_prefix="fig"):
    """Power-mean reproduction-number spectrum X(α,t) over time.

    For a given simulation, visualise the continuous family
        X(α,t) = Σ_j ω_j(α) R^j_out(t),   ω_j ∝ (R^j_out)^α
    over α ∈ [0, 20] and all simulation days.

    Special cases:
      α = 0  →  arithmetic mean of R^j_out  (equal location weights)
      α = 1  →  risk-aware E(t) = Σ(R^j_out)² / Σ R^j_out
      α → ∞  →  A₁(1) = max_j R^j_out  (first-generation epidemicity)

    Layout (2-row):
      Row 0, col 0–1 (wide):  Heatmap of X(α,t) — spectrum over time
      Row 0, col 2:           X(α, t) at three time points (early/peak/late)
      Row 1 (full width):     Selected α slices as time-series
    """
    inc    = sim["incidence"]
    R_mats = sim["R_matrices"]
    coords, pops, dists, node_types, meta = city_data
    T, N   = inc.shape
    t_arr  = np.arange(T)
    peak   = int(inc.sum(axis=1).argmax())
    early  = max(0, peak // 3)
    late   = min(T - 1, peak + (T - peak) // 2)

    # ── compute spectrum ────────────────────────────────────────────────────
    R_out_series = np.array([R_outward(R_mats[t]) for t in range(T)])
    rho_ts       = np.array([R_system(R_mats[t]) for t in range(T)])
    A1_1_ts      = R_out_series.max(axis=1)
    E_t          = np.array([np.sum(R_out_series[t]**2) / (np.sum(R_out_series[t]) + 1e-300)
                              for t in range(T)])
    arith_mean   = R_out_series.mean(axis=1)

    # α grid: 0 to 20 inclusive, 80 points
    alpha_arr = np.linspace(0, 20, 80)
    X = _compute_power_mean_spectrum(R_out_series, alpha_arr)   # (n_alpha, T)

    # ── figure layout ───────────────────────────────────────────────────────
    fig = plt.figure(figsize=(7.2, 5.6))
    gs  = gridspec.GridSpec(
        2, 3, height_ratios=[1.15, 1],
        hspace=0.62, wspace=0.52,
        left=0.09, right=0.97, top=0.94, bottom=0.10)

    # ── panel A: heatmap X(α,t) ─────────────────────────────────────────────
    ax = fig.add_subplot(gs[0, 0:2])
    da = float(alpha_arr[1] - alpha_arr[0])
    alpha_edges = np.append(alpha_arr - da / 2, alpha_arr[-1] + da / 2)
    t_edges     = np.arange(-0.5, T + 0.5)
    vmax_h = float(np.nanpercentile(X, 98))
    im = ax.pcolormesh(
        t_edges, alpha_edges, X,
        cmap="YlOrRd", shading="flat",
        vmin=0, vmax=max(vmax_h, 0.01))
    cb = plt.colorbar(im, ax=ax, fraction=0.030, pad=0.02)
    cb.ax.set_title("$X(\\alpha,t)$", fontsize=5.5, pad=2)
    cb.ax.tick_params(labelsize=5)
    # horizontal reference lines at key α values
    ax.axhline(0.0, color="0.25", lw=0.8, ls="--", alpha=0.7)
    ax.axhline(1.0, color=OKABE_ITO[6], lw=1.0, ls="--", alpha=0.85)
    ax.text(T * 0.99, 0.08, "$\\alpha=0$\n(arith. mean)", fontsize=4.5,
            color="0.25", ha="right", va="bottom")
    ax.text(T * 0.99, 1.08, "$\\alpha=1$  $\\mathcal{E}(t)$", fontsize=4.5,
            color=OKABE_ITO[6], ha="right", va="bottom")
    # vertical lines at early / peak / late
    for t_pt, lbl, col in [(early, "early", OKABE_ITO[2]),
                            (peak,  "peak",  OKABE_ITO[0]),
                            (late,  "late",  OKABE_ITO[4])]:
        ax.axvline(t_pt, color=col, lw=0.8, ls=":", alpha=0.7)
        ax.text(t_pt + 1, alpha_arr[-1] * 0.97, lbl,
                fontsize=4, color=col, va="top")
    ax.set_xlabel("Day $t$", fontsize=7)
    ax.set_ylabel("Power-mean exponent $\\alpha$", fontsize=6)
    ax.set_title(
        f"Power-mean spectrum $X(\\alpha,t)$ — {scenario_label}\n"
        "$\\alpha{=}0$: arithmetic mean;  $\\alpha{=}1$: risk-aware $\\mathcal{E}(t)$;  "
        "$\\alpha{\\to}\\infty$: $\\mathcal{A}_1(1)$",
        fontsize=5.5, pad=3)
    ax.tick_params(labelsize=5.5)
    _panel_label(ax, "A")

    # ── panel B: spectrum α-slices at three times ────────────────────────────
    ax = fig.add_subplot(gs[0, 2])
    time_pts = [(early, "early",  OKABE_ITO[2]),
                (peak,  "peak",   OKABE_ITO[0]),
                (late,  "late",   OKABE_ITO[4])]
    for t_pt, lbl, col in time_pts:
        x_slice = X[:, t_pt]
        valid   = ~np.isnan(x_slice)
        ax.plot(alpha_arr[valid], x_slice[valid], color=col, lw=1.1, label=f"Day {t_pt} ({lbl})")
        # mark α=0 and α=1 on each curve
        ax.plot(0.0, float(np.interp(0.0, alpha_arr[valid], x_slice[valid])),
                "o", color=col, ms=3, zorder=5)
        ax.plot(1.0, float(np.interp(1.0, alpha_arr[valid], x_slice[valid])),
                "s", color=col, ms=3, zorder=5)
    ax.axhline(1.0, color="0.55", ls="--", lw=0.7)
    ax.set_xlabel("$\\alpha$", fontsize=7)
    ax.set_ylabel("$X(\\alpha, t_*)$", fontsize=7)
    ax.set_title("Spectrum at three\ntime points", fontsize=6, pad=3)
    ax.legend(fontsize=4.8, borderpad=0.3, labelspacing=0.12)
    ax.text(0.04, 0.14, "dot: $\\alpha{=}0$   square: $\\alpha{=}1$",
            transform=ax.transAxes, fontsize=4.5, color="0.4")
    ax.tick_params(labelsize=5.5)
    _panel_label(ax, "B")

    # ── panel C (bottom wide): selected α curves over time ───────────────────
    ax = fig.add_subplot(gs[1, :])
    alpha_sel = [0.0, 0.5, 1.0, 2.0, 4.0, 20.0]
    cmap_sel  = plt.cm.plasma(np.linspace(0.1, 0.85, len(alpha_sel)))
    for alpha_s, col_s in zip(alpha_sel, cmap_sel):
        # Plot the EXACT X(α) at each requested α (the alpha_arr grid does not contain
        # 0.5/1/2/4, so nearest-grid indexing would mislabel e.g. X(1.013) as E(t)).
        if alpha_s == 0.0:
            x_ts = arith_mean          # X(0) = arithmetic mean of R^j_out
        elif alpha_s == 1.0:
            x_ts = E_t                 # X(1) = E(t) = Σ(R^j_out)² / Σ R^j_out
        else:
            x_ts = _compute_power_mean_spectrum(R_out_series, np.array([alpha_s]))[0]
        valid = ~np.isnan(x_ts)
        lbl = (f"$\\alpha={alpha_s:.1f}$"
               + (" ← $\\mathcal{E}(t)$" if alpha_s == 1.0 else "")
               + (" ← arith. mean"        if alpha_s == 0.0 else ""))
        ax.plot(t_arr[valid], x_ts[valid], color=col_s, lw=0.9, label=lbl)
    # overlay ρ(R(t)) and A₁(1) for reference
    vR = rho_ts > 0
    ax.plot(t_arr[vR], rho_ts[vR],  color=OKABE_ITO[4], lw=1.4, ls="--",
            label="$\\mathcal{R}(t)=\\rho(\\mathbf{R})$", zorder=5)
    vA = A1_1_ts > 0
    ax.plot(t_arr[vA], A1_1_ts[vA], color=OKABE_ITO[1], lw=1.1, ls="-.",
            label="$\\mathcal{A}_1(1)=\\max_k R^k_{\\rm out}$", zorder=5)
    ax.axhline(1.0, color="0.4", ls=":", lw=0.8)
    ax.axvline(peak, color="0.55", lw=0.7, ls="--", alpha=0.6)
    ax.text(peak + 1, 0.97, f"peak (d{peak})",
            transform=ax.get_xaxis_transform(),
            fontsize=4.5, color="0.4", va="top")
    ax.set_xlabel("Day $t$", fontsize=7)
    ax.set_ylabel("$X(\\alpha,t)$", fontsize=7)
    ax.set_title("Power-mean time series at selected $\\alpha$ values", fontsize=6, pad=3)
    ax.legend(fontsize=4.5, borderpad=0.3, labelspacing=0.10, ncol=4,
              loc="upper right")
    ax.tick_params(labelsize=5.5)
    _panel_label(ax, "C")

    fig.suptitle(
        f"Power-mean reproduction-number spectrum $X(\\alpha,t)$ — {scenario_label}",
        fontsize=7, y=0.985)

    fname = f"{save_prefix}_power_mean_spectrum_{scenario_label.replace(' ', '_')}.pdf"
    plt.savefig(fname, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved {fname}")
