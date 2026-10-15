"""mobility_rt.plotting.figures_main — manuscript Fig 4, the per-scheme SI figure set
(one figure per aggregation scheme) and Figs SI28-SI33."""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
from mobility_rt.config import OKABE_ITO
from mobility_rt.error_tables import (ESTIMATOR_WINDOW, RHO_FLOOR, SCHEMES, _period_masks,
                                      references, scored_crossing_day,
                                      threshold_crossing_day)
from mobility_rt.estimators import _compute_naive_suite
from mobility_rt.kernel import R_outward, R_system
from mobility_rt.plotting.style import _bar3d_Rkj, _bar3d_inc, _panel_label, _panel_label_3d
from mobility_rt.simulation import phase_days
from mobility_rt.spectral import reactivity
from mobility_rt.transmissibility import lehmer_mean_spectrum, risk_averse_R, within_fraction
from mobility_rt.type_reproduction import (_group_indices, type_reproduction_number_group,
                                           type_reproduction_numbers)

# Non-breaking hyphen for tick labels.  Held in a constant because a backslash escape
# inside an f-string expression is a SyntaxError before Python 3.12.
NB_HYPHEN = "\u2011"


def plot_fig5(sim_A, sim_B, f_B, save_prefix="fig"):
    """Fig SI31: dense urban (Scenario A) and sparse national (Scenario B) compared.

    Panels:
      A  mean mobility matrix f̄_{jk} of Scenario B (log colour scale)
      B  pairwise new infections E^{kj} at Scenario B's epidemic peak
      C  R^{kj} at Scenario B's epidemic peak
      D  ℛ(t) and σ(t) for both scenarios
      E  ℛ(t) for both scenarios, with normalised incidence on a twin axis
      F  overall within-location fraction π̄(t) = Σ_j E^{jj} / Σ_{k,j} E^{kj}
    """
    import matplotlib.colors as mcolors
    inc_A, inc_B = sim_A["incidence"], sim_B["incidence"]
    Rm_A,  Rm_B  = sim_A["R_matrices"], sim_B["R_matrices"]
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

    # ── A: mean mobility matrix — Scenario B (log scale to reveal movement) ──
    ax = fig.add_subplot(gs[0, 0])
    f_B_mean = f_B.mean(axis=0)
    lognorm = mcolors.LogNorm(vmin=max(f_B_mean[f_B_mean > 0].min(), 1e-4),
                              vmax=f_B_mean.max())
    im = ax.imshow(f_B_mean, cmap="Oranges", aspect="auto", norm=lognorm)
    ax.set_xticks(range(N)); ax.set_xticklabels(loc, fontsize=5, rotation=45)
    ax.set_yticks(range(N)); ax.set_yticklabels(loc, fontsize=5)
    ax.set_xlabel("Activity location $k$"); ax.set_ylabel("Residence $j$")
    cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.ax.set_title("$\\bar{f}_{jk}$\n(log)", fontsize=6, pad=3)
    _panel_label(ax, "A")

    # ── B: 3D E_kj at peak — Scenario B ───────────────────────────────────
    ax_b = fig.add_subplot(gs[0, 1], projection="3d")
    _bar3d_inc(ax_b, sim_B["incidence_matrix"][pk_B], f"Sparse national — peak (day {pk_B})")
    ax_b.text2D(-0.10, 1.05, "B", transform=ax_b.transAxes,
                fontsize=10, fontweight="bold", va="top", ha="left")

    # ── C: 3D R_kj at peak — Scenario B ───────────────────────────────────
    ax_c = fig.add_subplot(gs[1, 0], projection="3d")
    _bar3d_Rkj(ax_c, Rm_B[pk_B], f"Sparse national — peak (day {pk_B})")
    ax_c.text2D(-0.10, 1.05, "C", transform=ax_c.transAxes,
                fontsize=10, fontweight="bold", va="top", ha="left")

    # ── D: R(t) and σ(t) for both scenarios ───────────────────────────────
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

    # ── E: system R(t) with normalised incidence on twin axis ──────────────
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

    # ── F: overall within-location fraction π̄(t) for both scenarios ────────
    ax = fig.add_subplot(gs[2, 1])
    for sim_xy, col_xy, lbl_xy in [(sim_A, col_A, "Dense urban"),
                                   (sim_B, col_B, "Sparse national")]:
        _, pi_xy = within_fraction(sim_xy["incidence_matrix"])
        vm_xy = np.isfinite(pi_xy)
        ax.plot(t_arr[vm_xy], pi_xy[vm_xy], color=col_xy, lw=1.0, label=lbl_xy)
    ax.axhline(0.5, color="0.55", ls=":", lw=0.7, alpha=0.7)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Day $t$")
    ax.set_ylabel(r"Overall $\bar{\pi}(t) = \sum_j E_{jj}/\sum_{k,j} E_{kj}$")
    ax.legend(fontsize=6, borderpad=0.3)
    _panel_label(ax, "F")

    plt.savefig(f"{save_prefix}_05_comparison.pdf", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved {save_prefix}_05_comparison.pdf")


# ── Spatial-aggregation-bias figures: Fig 4 and the per-scheme SI set ─────

DEFAULT_BIAS_ROW_LABELS = ("Dense urban (Scenario A)",
                           "Sparse national (Scenario B)",
                           "Hub-amplified (Scenario C)")

# Network quantities a scheme can be scored against: suite key → (symbol, file tag).
_TARGETS = {"rho_t": (r"\mathcal{R}", "Rt"),
            "E_t":   (r"\mathcal{E}", "Et")}

# One colour per network quantity, the same in every figure.
_TARGET_COLOURS = {"rho_t": "#333333", "E_t": OKABE_ITO[6]}


def _compute_main_bias_data(sim_A, sim_B, sim_C, city_A, city_B, city_C,
                            gen_time_pmf, labels=None):
    """Naive-estimator suites for the three scenario rows of the bias figures.

    labels overrides the row titles; it must give exactly three labels in row order.

    Returns [(suite_dict, scenario_label), ...] in row order A, B, C.
    """
    labs = DEFAULT_BIAS_ROW_LABELS if labels is None else tuple(labels)
    if len(labs) != 3:
        raise ValueError(f"bias figures need exactly 3 row labels, got {len(labs)}")
    scenarios = [
        (sim_A, city_A, labs[0]),
        (sim_B, city_B, labs[1]),
        (sim_C, city_C, labs[2]),
    ]
    return [(_compute_naive_suite(sim, city, gen_time_pmf), label)
            for sim, city, label in scenarios]


def _scheme(est_key):
    """The error_tables.SCHEMES entry for est_key; KeyError if there is none."""
    for sch in SCHEMES:
        if sch.key == est_key:
            return sch
    raise KeyError(f"{est_key!r} has no entry in error_tables.SCHEMES, so its "
                   f"smoothing is unknown and it cannot be scored")


def _plot_main_bias_grid(all_data, est_key, fname, target="rho_t",
                        show_benchmark=True):
    """One 3-row spatial-aggregation-bias figure (Fig 4 and the per-scheme SI set).

    all_data : [(suite, row_label), ...] for Scenarios A, B, C (_compute_main_bias_data)
    est_key  : the aggregation scheme shown, a key of error_tables.SCHEMES
    fname    : output path (.pdf)
    target   : "rho_t" scores the scheme against ℛ(t) = ρ(R(t)); "E_t" against E(t)
    show_benchmark : draw the third column.  The statistics it summarises are
         returned either way, so suppressing it changes the layout only.  Fig 4 and
         its heterogeneous twin keep it (main-text layout, panels A-I); the
         scheme-by-scheme sets drop it, because with one figure per scheme the
         out_Rw figure IS E(t) against ℛ(t) and repeating it in every other figure
         adds nothing.

    Columns, one row per scenario:
      0  the target (solid), the other network quantity (dotted), the scheme
         (dashed) and, for a sliding-window scheme, the target's trailing mean
      1  the scheme minus its matched reference
      2  E(t) − ℛ(t), the same in every figure, so every scheme is read against
         one benchmark (only when show_benchmark)

    Matched reference.  A sliding-window (Cori/EpiEstim) scheme summarises the
    ESTIMATOR_WINDOW days ending at t, so it is compared with the target's trailing
    mean over those days; a scheme without a window is compared with the target on
    day t.  Whether a scheme is windowed comes from error_tables.SCHEMES.

    Scored days.  Statistics use error_tables._period_masks: a day counts only when
    ℛ(t) and every reference exceed the noise floor and every scheme is defined;
    growth and decline are ℛ(t) > 1 and ℛ(t) <= 1.  For target "rho_t" these are
    the days of Tables SI1-SI2, so each MRE and bias equals its table cell.  Bias
    traces are drawn on the scored days only.

    Statistics box, per period: MRE = mean |difference| / reference and bias = mean
    difference / reference, in per cent, with the number of days.  "threshold" is
    the settled crossing of 1 by the scheme minus that of the target on day t, both
    counted on the scored days, so it includes any delay due to the window.

    Returns {row_label: {"n_days": {...}, "scheme": {...}, "E": {...}}}, where
    "scheme" and "E" hold (MRE, bias) for "growth" and "decline" and the
    "threshold_days" lag (None if either series never settles below 1).

    Panel labels A–I (A–F without the benchmark column), row-major.
    """
    if len(all_data) != 3:
        raise ValueError(
            f"_plot_main_bias_grid expects exactly 3 scenario rows, got {len(all_data)}")
    sch = _scheme(est_key)
    if target not in _TARGETS:
        raise KeyError(f"unknown target {target!r}; use one of {sorted(_TARGETS)}")
    other_key = "E_t" if target == "rho_t" else "rho_t"
    tgt_sym = _TARGETS[target][0]
    tgt_tex = tgt_sym + "(t)"
    other_tex = _TARGETS[other_key][0] + "(t)"
    est_tex = sch.tex.strip("$")
    ref_tex = (rf"\overline{{{tgt_sym}}}_{{{ESTIMATOR_WINDOW}}}(t)"
               if sch.windowed else tgt_tex)

    COL_EST    = OKABE_ITO[0]    # orange         — the scheme
    COL_OVER   = "#D55E00"       # vermillion     — positive difference
    COL_UNDER  = "#56B4E9"       # sky blue       — negative difference
    COL_LINE   = "0.30"          # dark grey difference trace
    COL_GROWTH = "#F0F0F0"       # growth-phase band, ℛ(t) > 1
    COL_REF    = "#7E9BB5"       # muted blue-grey — trailing-mean reference
    PANEL_IDS  = list("ABCDEFGHI")
    SC_COLS    = [OKABE_ITO[4], OKABE_ITO[2], OKABE_ITO[5]]

    def _rel(diff, ref, mask):
        """(MRE %, bias %) of diff relative to ref over mask; (None, None) if empty."""
        if not mask.any():
            return None, None
        r = diff[mask] / ref[mask]
        return float(np.mean(np.abs(r)) * 100.0), float(np.mean(r) * 100.0)

    def _lag(series, reference, mask):
        c_s = scored_crossing_day(series, mask)
        c_r = scored_crossing_day(reference, mask)
        return None if c_s is None or c_r is None else int(c_s - c_r)

    def _lag_txt(days):
        if days is None:
            return "n/a"
        if days == 0:
            return "on time"
        return f"{abs(days)} d {'early' if days < 0 else 'late'}"

    def _num(v, signed):
        # inside mathtext so a negative renders as a true minus sign
        if v is None:
            return "n/a"
        return f"${v:+.1f}$%" if signed else f"${v:.1f}$%"

    # Panel geometry is pinned in INCHES, so a panel is exactly the same size with
    # and without the benchmark column: only the figure width changes.  The
    # three-column case reproduces the original 7.2 in figure and its 0.11/0.97
    # margins exactly, so the SI figures are unchanged by the two-column option.
    FIG_W_3COL, MARGIN_L, MARGIN_R, WSPACE = 7.2, 0.792, 0.216, 0.52
    n_cols  = 3 if show_benchmark else 2
    panel_w = (FIG_W_3COL - MARGIN_L - MARGIN_R) / (3 + 2 * WSPACE)
    fig_w   = MARGIN_L + panel_w * (n_cols + (n_cols - 1) * WSPACE) + MARGIN_R
    # No panel titles, so the rows need only clear the x labels and panel letters;
    # the figure is shortened to keep each panel the size it was with titles.
    fig = plt.figure(figsize=(fig_w, 6.3))
    gs  = gridspec.GridSpec(3, n_cols, hspace=0.45, wspace=WSPACE,
                            left=MARGIN_L / fig_w, right=1.0 - MARGIN_R / fig_w,
                            top=0.95, bottom=0.06)

    stats = {}
    panel_idx = 0
    for row, (d, sc_label) in enumerate(all_data):
        t_arr  = np.arange(d["T"])
        rho_t  = np.asarray(d["rho_t"], dtype=float)
        E_t    = np.asarray(d["E_t"], dtype=float)
        tgt    = np.asarray(d[target], dtype=float)
        other  = np.asarray(d[other_key], dtype=float)
        est_t  = np.asarray(d[est_key], dtype=float)
        peak_t = d["peak_t"]
        sc_col = SC_COLS[row]

        masks = _period_masks(d, targets=tuple(dict.fromkeys(("rho_t", target))))
        full_m, growth_m, decline_m = masks["full"], masks["growth"], masks["decline"]
        ref_est  = references(tgt, ESTIMATOR_WINDOW)[sch.windowed]
        diff_est = np.where(full_m, est_t - ref_est, np.nan)
        diff_E   = np.where(full_m, E_t - rho_t, np.nan)
        growth_band = (rho_t > RHO_FLOOR) & (rho_t > 1.0)

        n_days = {"full": int(full_m.sum()), "growth": int(growth_m.sum()),
                  "decline": int(decline_m.sum())}
        row_stats = {
            "n_days": n_days,
            "scheme": {"growth":  _rel(diff_est, ref_est, growth_m),
                       "decline": _rel(diff_est, ref_est, decline_m),
                       "threshold_days": _lag(est_t, tgt, full_m)},
            "E":      {"growth":  _rel(diff_E, rho_t, growth_m),
                       "decline": _rel(diff_E, rho_t, decline_m),
                       "threshold_days": _lag(E_t, rho_t, full_m)},
        }
        stats[sc_label] = row_stats

        # ── Col 0: time series ─────────────────────────────────────────────
        ax = fig.add_subplot(gs[row, 0])
        ax.fill_between(t_arr, 0, 1, where=growth_band, step="mid",
                        transform=ax.get_xaxis_transform(),
                        color=COL_GROWTH, lw=0, zorder=0)
        active = rho_t > RHO_FLOOR
        v_tgt = active & np.isfinite(tgt)
        ax.plot(t_arr[v_tgt], tgt[v_tgt], color=_TARGET_COLOURS[target], lw=1.5,
                label=f"${tgt_tex}$", zorder=4)
        v_oth = active & np.isfinite(other)
        ax.plot(t_arr[v_oth], other[v_oth], color=_TARGET_COLOURS[other_key], lw=1.0,
                ls=":", label=f"${other_tex}$", zorder=4)
        v_est = np.isfinite(est_t)
        ax.plot(t_arr[v_est], est_t[v_est], color=COL_EST, lw=1.0, ls="--",
                label=f"${est_tex}$", zorder=5)
        if sch.windowed:
            v_ref = active & np.isfinite(ref_est)
            ax.plot(t_arr[v_ref], ref_est[v_ref], color=COL_REF, lw=0.9, ls="-",
                    label=f"${ref_tex}$", zorder=3)
        ax.axhline(1.0, color="0.60", ls=":", lw=0.7, zorder=1)
        ax.axvline(peak_t, color="0.55", ls="--", lw=0.7, alpha=0.45, zorder=1)
        ax.set_xlabel("Day $t$", fontsize=7)
        ax.set_ylabel("Reproduction number", fontsize=7)
        ax.legend(fontsize=4.8, borderpad=0.3, labelspacing=0.10,
                  loc="upper right", ncol=1)
        ax.tick_params(labelsize=6)
        _panel_label(ax, PANEL_IDS[panel_idx], x=-0.17, y=1.03); panel_idx += 1

        def _bias_panel(ax_b, diff, st, ylabel, title_suffix):
            ax_b.fill_between(t_arr, 0, 1, where=growth_band, step="mid",
                              transform=ax_b.get_xaxis_transform(),
                              color=COL_GROWTH, lw=0, zorder=0)
            bv = np.isfinite(diff)
            if bv.any():
                pos = bv & (diff > 0)
                neg = bv & (diff < 0)
                if pos.any():
                    ax_b.fill_between(t_arr, 0, np.where(pos, diff, 0),
                                      color=COL_OVER, alpha=0.38,
                                      label="Over-estimate")
                if neg.any():
                    ax_b.fill_between(t_arr, 0, np.where(neg, diff, 0),
                                      color=COL_UNDER, alpha=0.38,
                                      label="Under-estimate")
                ax_b.plot(t_arr, diff, color=COL_LINE, lw=0.8, zorder=3)
            ax_b.axhline(0, color="0.60", ls="--", lw=0.7)
            ax_b.axvline(peak_t, color="0.55", ls="--", lw=0.7, alpha=0.45)
            g_mre, g_bias = st["growth"]
            d_mre, d_bias = st["decline"]
            stats_str = (
                f"$\\mathcal{{R}}>1$ ({n_days['growth']} d)   MRE {_num(g_mre, False)}"
                f"   bias {_num(g_bias, True)}\n"
                f"$\\mathcal{{R}}\\leq 1$ ({n_days['decline']} d)   MRE {_num(d_mre, False)}"
                f"   bias {_num(d_bias, True)}\n"
                f"threshold   {_lag_txt(st['threshold_days'])}")
            ax_b.text(0.03, 0.97, stats_str,
                      transform=ax_b.transAxes, fontsize=4.6,
                      ha="left", va="top", color="black", linespacing=1.35,
                      bbox=dict(facecolor="white", edgecolor="0.80",
                                linewidth=0.5, alpha=0.88, pad=2.0))
            if bv.any():
                ax_b.legend(fontsize=5.0, borderpad=0.3, labelspacing=0.12,
                            loc="lower right")
            ax_b.set_xlabel("Day $t$", fontsize=7)
            ax_b.set_ylabel(ylabel, fontsize=7)
            ax_b.tick_params(labelsize=6)

        # ── Col 1: the scheme against its matched reference ─────────────────
        ax = fig.add_subplot(gs[row, 1])
        _bias_panel(ax, diff_est, row_stats["scheme"],
                    f"${est_tex} - {ref_tex}$", f"Bias — ${est_tex}$")
        _panel_label(ax, PANEL_IDS[panel_idx], x=-0.17, y=1.03); panel_idx += 1

        # ── Col 2: E(t) against ℛ(t) ────────────────────────────────────────
        if show_benchmark:
            ax = fig.add_subplot(gs[row, 2])
            _bias_panel(ax, diff_E, row_stats["E"],
                        r"$\mathcal{E}(t) - \mathcal{R}(t)$",
                        r"Bias — $\mathcal{E}(t)$ vs $\mathcal{R}(t)$")
            _panel_label(ax, PANEL_IDS[panel_idx], x=-0.17, y=1.03); panel_idx += 1

    plt.savefig(fname, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved {fname}")
    return stats


HET_BIAS_ROW_LABELS = ("Dense urban, heterogeneous (Scenario A-het)",
                       "Sparse national, heterogeneous (Scenario B-het)",
                       "Hub-amplified, heterogeneous (Scenario C-het)")


def plot_main_bias_figure(sim_A, sim_B, sim_C, city_A, city_B, city_C,
                          gen_time_pmf, save_prefix="fig", all_data=None,
                          labels=None, fname_suffix="main_bias_combined"):
    """Fig 4: bias of the relative-transmissibility-weighted independent estimates
    R^{ind,Rw}(t) = Σ_j (R̂^j_ind)² / Σ_j R̂^j_ind (§3.2.2, ref [68]) and of E(t)
    (Eq 42), both against ℛ(t), for Scenarios A, B and C.

    Layout and statistics are documented on _plot_main_bias_grid.  Pass a precomputed
    all_data (from _compute_main_bias_data) to reuse the estimator suites; `labels`
    is then ignored, because all_data carries its own row labels.

    THREE columns, panels A-I: the trajectories, the scheme against its matched
    reference, and the E(t) benchmark.  This is the MAIN-TEXT layout and it keeps
    the benchmark column deliberately, so that Fig 4 shows R^{ind,Rw}(t) and E(t)
    against ℛ(t) side by side.  The per-scheme sets that support it
    (plot_main_bias_suite and plot_estimator_si_suite) drop that column instead,
    because it is identical in every figure of a set and because those sets already
    give E(t) its own figure as the out_Rw scheme.

    Saved as {save_prefix}_{fname_suffix}.pdf.  The heterogeneous twin is the same
    figure over the -het simulations, so it is produced by this same entry point
    with fname_suffix="het_main_bias_combined" and HET_BIAS_ROW_LABELS; keeping one
    plotting path means the two figures cannot drift apart.  Returns all_data.
    """
    if all_data is None:
        all_data = _compute_main_bias_data(sim_A, sim_B, sim_C,
                                           city_A, city_B, city_C, gen_time_pmf,
                                           labels=labels)
    _plot_main_bias_grid(all_data, "ind_Rw", f"{save_prefix}_{fname_suffix}.pdf",
                        target="rho_t", show_benchmark=True)
    return all_data


def plot_main_bias_suite(all_data, save_prefix="fig",
                         fname_suffix="main_bias_combined"):
    """The Fig 4 layout for EVERY aggregation scheme, scored against ℛ(t).

    all_data : [(suite, row_label), ...] from _compute_main_bias_data, reused across
               all schemes so the nine figures share one set of simulations.

    The scheme list is error_tables.SCHEMES itself rather than a hand-kept copy, so
    a scheme cannot be added to the tables and quietly missed here.  That includes
    out_Rw, which IS E(t): with one figure per scheme, E(t) against ℛ(t) is that
    figure, which is why no figure carries an E(t) benchmark column.

    Writes {save_prefix}_{fname_suffix}_{scheme}.pdf.  Returns {path: statistics}.
    """
    written = {}
    for sch in SCHEMES:
        fname = f"{save_prefix}_{fname_suffix}_{sch.key}.pdf"
        written[fname] = _plot_main_bias_grid(all_data, sch.key, fname,
                                             target="rho_t", show_benchmark=False)
    return written


def plot_estimator_si_suite(all_data, save_prefix="fig"):
    """Every aggregation scheme in error_tables.SCHEMES scored against ℛ(t), in the
    layout of Fig 4: one figure per scheme, two columns each.

    NO E(t) BENCHMARK COLUMN.  It was the same three panels in every figure of the
    set, and E(t) against ℛ(t) is already the out_Rw figure of this same set.

    SCORED AGAINST ℛ(t) ONLY.  ℛ(t) is the quantity with the threshold property;
    E(t) explicitly does not have one (§3.1.5).  Scoring the same schemes against
    E(t) as well adds little: retargeting moves every scheme in a scenario by
    roughly the same amount, about -(E(t) - ℛ(t))/ℛ(t), which over the growth phase
    averages -3.4%, -0.1% and +0.9% in Scenarios A, B and C.  It is NOT a single
    constant — the daily gap varies about that mean (sd 0.7, 1.3 and 0.3 points
    respectively), and the target enters each day's denominator — so rankings are
    preserved in practice here rather than by construction.  The gap itself is what
    the out_Rw figure of this set reports.

    Writes {save_prefix}_naive_{scheme}_vs_Rt.pdf, one per scheme.
    Returns {path: statistics returned by _plot_main_bias_grid}.
    """
    tag = _TARGETS["rho_t"][1]
    written = {}
    for sch in SCHEMES:
        fname = f"{save_prefix}_naive_{sch.key}_vs_{tag}.pdf"
        written[fname] = _plot_main_bias_grid(all_data, sch.key, fname,
                                              target="rho_t", show_benchmark=False)
    return written


# ── Type reproduction numbers: Figs SI28-SI30 ─────────────────────────────

def plot_type_repro(sim_A, sim_B, city_A, city_B, save_prefix):
    """
    Figs SI28-SI30: type reproduction numbers T_j(t) (Eqs 53-55) and group type
    reproduction numbers T^P(t) (Eq 56) for Scenarios A and B.

        T_j(t) = R_jj + R_jJ (I - R_JJ)^{-1} R_Jj,          J = every location but j
        T^P(t) = ρ(R_PP + R_PQ (I - R_QQ)^{-1} R_QP),        Q = every location not in P

    T_j is undefined when ρ(R_JJ) >= 1, i.e. when the network without j can sustain
    the epidemic on its own, and T^P likewise when ρ(R_QQ) >= 1.  Because
    ρ(R_JJ) <= ρ(R), T_j is defined whenever ρ(R) < 1, and it can be defined earlier.
    Wherever it is defined, T_j > 1 exactly when ρ(R) > 1 (Eq 55).  A location j
    with R_jj > 1 leaves T_i undefined for every other location i, because the
    network without i contains j and so has spectral radius at least R_jj.

    Files
      *_type_heatmaps.png  (Fig SI29) T_j(t) heatmaps and R_jj(t)
      *_type_surfaces.png  (Fig SI28) T_j(t) and R_jj(t) surfaces
      *_type_groups.png    (Fig SI30) T^P against ρ(R(t)); R_jj(t) by type;
                           T_j against ρ(R(t)); T^P(t) for the locations of each type
    The dashed vertical line marks the day ρ(R(t)) settles below 1
    (error_tables.threshold_crossing_day).

    Reference: Roberts & Heesterbeek 2003, Proc R Soc B 270:1359.
    """
    from matplotlib.colors import Normalize, TwoSlopeNorm
    from matplotlib.patches import Patch

    types_A = city_A[3]
    types_B = city_B[3]

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
    lab_B = [f"{types_B[j].replace('-', NB_HYPHEN).title()} #{j+1}" for j in idx_B]

    def _settled_day(rho):
        """Day ρ(R(t)) settles below 1; T (the right edge) if it never does."""
        day = threshold_crossing_day(rho)
        return T if day is None else day

    cross_A = _settled_day(rho_A)
    cross_B = _settled_day(rho_B)

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
    # Figure 1 (Fig SI29) — T_j heatmaps (top) + R_jj(t) (bottom)
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    fig = plt.figure(figsize=(15, 10))
    gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.48, wspace=0.32)

    norm_div = TwoSlopeNorm(vmin=0.0, vcenter=1.0, vmax=5.0)
    cmap_div = plt.cm.RdBu_r

    # ── row 0: T_j heatmaps ──────────────────────────────────────────────
    for col, (T_ser, idx, labs, xday, title, panel_lbl) in enumerate([
        (T_ser_A, idx_A, lab_A, cross_A, "Scenario A: Dense urban", "a"),
        (T_ser_B, idx_B, lab_B, cross_B, "Scenario B: Sparse national", "b"),
    ]):
        ax = fig.add_subplot(gs[0, col])
        data   = T_ser[:, idx].T          # (N, T)
        masked = np.ma.masked_invalid(data)

        X, Y = np.meshgrid(np.arange(T + 1), np.arange(N + 1))
        im = ax.pcolormesh(X, Y - 0.5, masked, cmap=cmap_div, norm=norm_div,
                           shading="flat")
        # Grey overlay where T_j is undefined (ρ(R_JJ) >= 1)
        nan_mask_pm = np.where(np.isnan(data), 1.0, np.nan)
        ax.pcolormesh(X, Y - 0.5, nan_mask_pm, cmap="Greys_r",
                      vmin=0.5, vmax=1.5, shading="flat", alpha=0.55)
        ax.axvline(xday, color="#111111", lw=1.4, ls="--", alpha=0.9)

        ax.text(xday * 0.30, N * 0.82,
                r"$R^j_{\rm type}$ undefined (grey)" + "\n"
                r"$\rho(\mathbf{R}_{JJ})\geq 1$: the network" + "\n"
                r"without $j$ is self-sustaining",
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

        handles = [
            Line2D([0], [0], color="#111111", lw=1.4, ls="--",
                   label=f"$\\rho(t)$ settles below 1, day {xday}"),
            Patch(facecolor="#aaaaaa", edgecolor="none",
                  label=r"$R^j_{\rm type}$ undefined"),
        ]
        ax.legend(handles=handles, fontsize=7, loc="lower right",
                  framealpha=0.80, handlelength=1.2)
        cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.03, extend="max")
        cb.set_label("$R^j_{\\rm type}(t)$", fontsize=8.5)
        cb.ax.axhline(1.0, color="#333", lw=0.9, ls=":")   # colorbar axes are in data units
        _panel_label(ax, panel_lbl, x=-0.16)

    # ── row 1: R_jj(t) ───────────────────────────────────────────────────
    for col, (Rjj, idx, labs, xday, title, panel_lbl) in enumerate([
        (Rjj_A, idx_A, lab_A, cross_A,
         r"$R_{jj}(t)$: within-location reproduction — Scenario A", "c"),
        (Rjj_B, idx_B, lab_B, cross_B,
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

        ax.text(0.02, 0.97,
                r"$R_{jj}(t)>1$: location $j$ self-sustaining" + "\n"
                r"$\Rightarrow R^i_{\rm type}$ undefined for every $i\neq j$",
                transform=ax.transAxes, fontsize=6, va="top", ha="left",
                color="#555555",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="0.97",
                          edgecolor="0.70", linewidth=0.5))

        ax.set_xlabel("Day", fontsize=9)
        ax.set_ylabel("$R_{jj}(t)$", fontsize=9)
        ax.set_ylim(bottom=0)
        ax.legend(fontsize=6.5, ncol=2, loc="upper right",
                  framealpha=0.80, handlelength=1.2)
        _panel_label(ax, panel_lbl)

    fname1 = f"{save_prefix}_type_heatmaps.png"
    plt.savefig(fname1, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved {fname1}")

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Figure 2 (Fig SI28) — 3-D surfaces: T_j(t) and R_jj(t) side by side
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    fig = plt.figure(figsize=(16, 9))
    gs2 = gridspec.GridSpec(2, 2, figure=fig, hspace=0.10, wspace=0.02)

    surf_cmap_T   = plt.cm.plasma
    surf_cmap_Rjj = plt.cm.viridis
    panel_it = iter(["a", "b", "c", "d"])

    for row, (T_ser, Rjj, idx, title_scen) in enumerate([
        (T_ser_A, Rjj_A, idx_A, "Scenario A: Dense urban"),
        (T_ser_B, Rjj_B, idx_B, "Scenario B: Sparse national"),
    ]):
        XX, YY = np.meshgrid(days, np.arange(N))

        # left: T_j(t) surface (undefined values are not drawn)
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
        ax2.view_init(elev=26, azim=-52)
        ax2.tick_params(labelsize=6)
        cb2 = fig.colorbar(surf2, ax=ax2, fraction=0.022, pad=0.08, shrink=0.60)
        cb2.set_label("$R_{jj}(t)$", fontsize=7)
        _panel_label_3d(ax2, next(panel_it))

    fname2 = f"{save_prefix}_type_surfaces.png"
    plt.savefig(fname2, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved {fname2}")

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Figure 3 (Fig SI30) — group T^P vs ρ(R), R_jj by type, threshold check,
    # T^P for each node type
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    cmap_t = plt.cm.cividis

    fig, axes = plt.subplots(4, 2, figsize=(14, 17),
                             gridspec_kw={"hspace": 0.55, "wspace": 0.35})

    # ── Row 0: T^P against ρ(R(t)) ──────────────────────────────────────────
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
                # Mark the first day T^P is defined
                first = int(np.where(mask)[0][0])
                ax.scatter(rho_s[first], vals[first],
                           color=oi[k % len(oi)], s=35, zorder=5,
                           edgecolors="white", linewidths=0.6)

        ax.axhline(1.0, color="#333333", lw=1.0, ls="--", alpha=0.8,
                   label="$R^{\\mathcal{P}}_{\\rm type}=1$", zorder=4)
        ax.axvline(1.0, color="#333333", lw=1.0, ls=":", alpha=0.8,
                   label="$\\rho=1$", zorder=4)
        # Scenario B's curves run up to the top-right corner, so its note goes in the
        # empty bottom-right instead
        note_y, note_va = (0.03, "bottom") if panel_lbl == "b" else (0.97, "top")
        ax.text(0.97, note_y,
                r"Dots: first day $R^{\mathcal{P}}_{\rm type}$ is defined ($\rho(R_{QQ})<1$)",
                transform=ax.transAxes, fontsize=6, va=note_va, ha="right",
                color="#444444", style="italic",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="0.96",
                          edgecolor="0.65", linewidth=0.5))
        ax.set_xlabel(r"$\rho\!\left(\mathbf{R}(t)\right)$", fontsize=9)
        ax.set_ylabel(r"$R^{\mathcal{P}}_{\rm type}(t)$", fontsize=9)
        ax.set_xlim(0, None)
        ax.set_ylim(0, None)
        ax.legend(fontsize=6.5, loc="upper left", framealpha=0.80,
                  handlelength=1.2, ncol=1)
        _panel_label(ax, panel_lbl)

    # ── Row 1: R_jj(t) by node type ─────────────────────────────────────────
    for ax, Rjj, types, xday, title, panel_lbl in [
        (axes[1, 0], Rjj_A, types_A, cross_A,
         r"$R_{jj}(t)$ by node type — Scenario A: Dense urban", "c"),
        (axes[1, 1], Rjj_B, types_B, cross_B,
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
                r"$R_{jj}>1$: location $j$ sustains transmission alone" + "\n"
                r"$\Rightarrow R^i_{\rm type}$ undefined for every $i\neq j$" + "\n"
                r"$R_{jj}<1$: transmission within $j$ alone is subcritical",
                transform=ax.transAxes, fontsize=6, va="top", ha="left",
                color="#555555",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="0.97",
                          edgecolor="0.68", linewidth=0.5))
        ax.set_xlabel("Day", fontsize=9)
        ax.set_ylabel("$R_{jj}(t)$  (mean ± range by type)", fontsize=8.5)
        ax.set_ylim(bottom=0)
        ax.legend(fontsize=7, loc="upper right", framealpha=0.80,
                  handlelength=1.2)
        _panel_label(ax, panel_lbl)

    # ── Row 2: threshold property, T_j against ρ(R(t)) ──────────────────────
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

        # Computed, not asserted: how many defined (day, location) pairs break Eq 55.
        defined = np.isfinite(T_ser)
        violations = defined & ((T_ser > 1.0) != (rho_s[:, None] > 1.0))
        ax.text(0.02, 0.97,
                f"$R^j_{{\\rm type}}$ defined on {int(defined.sum())} (day, location) pairs\n"
                f"pairs with $(R^j_{{\\rm type}}>1)\\neq(\\rho>1)$: {int(violations.sum())}",
                transform=ax.transAxes, fontsize=6.5, va="top", ha="left",
                color="#444444",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="0.96",
                          edgecolor="0.68", linewidth=0.5))

        ax.set_xlabel(r"$\rho\!\left(\mathbf{R}(t)\right)$", fontsize=9)
        ax.set_ylabel("$R^j_{\\rm type}(t)$", fontsize=9)
        ax.legend(fontsize=7.5, loc="lower right", framealpha=0.80,
                  handlelength=1.2)
        cb = fig.colorbar(
            plt.cm.ScalarMappable(cmap=cmap_t, norm=norm_t),
            ax=ax, fraction=0.032, pad=0.03)
        cb.set_label("Day $t$", fontsize=8)
        _panel_label(ax, panel_lbl)

    # ── Row 3: T^P(t) for the group of all locations of each node type ──────
    for ax, R_mats, types, t_order, xday, title, panel_lbl in [
        (axes[3, 0], sim_A["R_matrices"], types_A, type_order_A, cross_A,
         r"$R^{\mathcal{P}}_{\rm type}(t)$, $\mathcal{P}$ = locations of one type — Scenario A", "g"),
        (axes[3, 1], sim_B["R_matrices"], types_B, type_order_B, cross_B,
         r"$R^{\mathcal{P}}_{\rm type}(t)$, $\mathcal{P}$ = locations of one type — Scenario B", "h"),
    ]:
        n_types = len(t_order)
        type_data = np.full((n_types, T), np.nan)
        for ti, ntype in enumerate(t_order):
            P = _group_indices(types, {ntype})
            if P:
                type_data[ti] = [type_reproduction_number_group(R_mats[t], P)
                                 for t in range(T)]
        # values above 5 are drawn at 5 (the colorbar extends)
        data_plot = np.where(np.isfinite(type_data),
                             np.minimum(type_data, 5.0), np.nan)
        masked = np.ma.masked_invalid(data_plot)
        norm_tile = TwoSlopeNorm(vmin=0.0, vcenter=1.0, vmax=5.0)
        im = ax.pcolormesh(np.arange(T + 1), np.arange(n_types + 1) - 0.5,
                           masked, cmap="RdBu_r", norm=norm_tile,
                           shading="flat")
        # Grey overlay where T^P is undefined (ρ(R_QQ) >= 1)
        nan_overlay = np.where(np.isnan(type_data), 1.0, np.nan)
        ax.pcolormesh(np.arange(T + 1), np.arange(n_types + 1) - 0.5,
                      nan_overlay, cmap="Greys_r",
                      vmin=0.5, vmax=1.5, shading="flat", alpha=0.50)
        ax.axvline(xday, color="#111111", lw=1.2, ls="--", alpha=0.85)
        ax.set_yticks(np.arange(n_types))
        ax.set_yticklabels([t.replace("-", NB_HYPHEN).title()
                            for t in t_order], fontsize=7)
        ax.set_xlabel("Day $t$", fontsize=9)
        cb_t = fig.colorbar(im, ax=ax, fraction=0.032, pad=0.03, extend="max")
        cb_t.set_label(r"$R^{\mathcal{P}}_{\rm type}(t)$", fontsize=7.5)
        cb_t.ax.axhline(1.0, color="#333", lw=0.8, ls=":")   # data units
        handles_t = [
            Line2D([0], [0], color="#111111", lw=1.2, ls="--",
                   label=f"$\\rho(t)$ settles below 1, day {xday}"),
            Patch(facecolor="#aaaaaa", edgecolor="none",
                  label=r"$R^{\mathcal{P}}_{\rm type}$ undefined"),
        ]
        ax.legend(handles=handles_t, fontsize=6.5, loc="lower right",
                  framealpha=0.80, handlelength=1.2)
        _panel_label(ax, panel_lbl)

    fname3 = f"{save_prefix}_type_groups.png"
    plt.savefig(fname3, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved {fname3}")


# ── Scenario C transience: Fig SI32 ───────────────────────────────────────

def plot_fig_C_transience(sim_C, city_data, f_C, save_prefix="fig"):
    """Fig SI32: Scenario C (hub-amplified), transient amplification σ(t) > 1 while
    ℛ(t) < 1.

    Layout (2 rows):
      Top row (3 panels): A mean mobility | B resident populations | C R^{kj} at peak
      Bottom (1 wide):    D ℛ(t), σ(t), A₁(1) and E(t), with the σ > 1, ℛ < 1 days shaded
    """
    inc    = sim_C["incidence"]
    R_mats = sim_C["R_matrices"]
    pops   = np.asarray(city_data[1], dtype=float)
    T, N = inc.shape

    rho_ts   = np.array([R_system(R_mats[t])            for t in range(T)])
    sigma_ts = np.array([reactivity(R_mats[t])["sigma"] for t in range(T)])
    peak     = int(inc.sum(axis=1).argmax())
    t_arr    = np.arange(T)
    loc_lbl  = [f"L{i+1}" for i in range(N)]
    # A₁(1) = max_k R^k_out(t) and E(t) = Σ_j (R^j_out)² / Σ_k R^k_out (Eq 42)
    R_out_C = np.array([R_outward(R_mats[t]) for t in range(T)])
    A1_1_C  = R_out_C.max(axis=1)
    E_t_C   = risk_averse_R(R_out_C)
    lb_over_lw = sim_C["lambda_between_scaled"] / sim_C["lambda_within_scaled"]

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

    # ── A: mean mobility matrix ─────────────────────────────────────────────
    ax = fig.add_subplot(gs[0, 0])
    f_mean = f_C.mean(axis=0)
    # Colour range from the off-diagonal entries, so the hub column is visible
    f_off  = f_mean.copy(); np.fill_diagonal(f_off, np.nan)
    im = ax.pcolormesh(np.arange(N+1)-0.5, np.arange(N+1)-0.5,
                       f_mean, cmap="YlOrRd", shading="flat",
                       vmin=0, vmax=float(np.nanpercentile(f_off, 98)))
    ax.set_xlim(-0.5, N-0.5); ax.set_ylim(N-0.5, -0.5)
    ax.set_xticks(range(N)); ax.set_xticklabels(loc_lbl, fontsize=4.5, rotation=45)
    ax.set_yticks(range(N)); ax.set_yticklabels(loc_lbl, fontsize=4.5)
    ax.set_xlabel("Activity location $k$", fontsize=6)
    ax.set_ylabel("Residence $j$", fontsize=6)
    cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.ax.set_title("$\\bar{f}_{jk}$", fontsize=5.5, pad=2)
    cb.ax.tick_params(labelsize=5)
    i_hub = int(np.argmax(pops))
    ax.axhline(i_hub, color="#CC5500", lw=0.8, ls="--", alpha=0.7)
    ax.axvline(i_hub, color="#CC5500", lw=0.8, ls="--", alpha=0.7)
    ax.text(i_hub + 0.15, -0.5, "hub", fontsize=4.2, color="#CC5500", va="top")
    _panel_label(ax, "A")

    # ── B: resident population bar chart ────────────────────────────────────
    ax = fig.add_subplot(gs[0, 1])
    bar_colors = [COL_DROP if i == i_hub else "0.55" for i in range(N)]
    ax.barh(range(N), pops / 1e6, color=bar_colors, height=0.72, edgecolor="none")
    ax.set_yticks(range(N))
    ax.set_yticklabels(loc_lbl, fontsize=5)
    ax.set_xlabel("Resident population (millions)", fontsize=6)
    ax.invert_yaxis()
    ax.tick_params(labelsize=5)
    ax.text(pops[i_hub] / 1e6 + 0.03, i_hub, "hub",
            fontsize=5, color=COL_DROP, va="center")
    _panel_label(ax, "B")

    # ── C: R_{kj} heatmap at epidemic peak ──────────────────────────────────
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
    cb3 = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb3.ax.set_title("$R_{kj}$", fontsize=5.5, pad=2)
    cb3.ax.tick_params(labelsize=5)
    _panel_label(ax, "C")

    # ── D: transient amplification (wide, bottom row) ───────────────────────
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
    vA1C = A1_1_C > 0;  vEC = np.isfinite(E_t_C) & (E_t_C > 0)
    ax.plot(t_arr[vR], rho_ts[vR],   color=COL_RHO,   lw=2.0, zorder=4,
            label="$\\mathcal{R}(t) = \\rho(\\mathbf{R}(t))$  [network reproduction number]")
    ax.plot(t_arr[vS], sigma_ts[vS], color=COL_SIGMA,  lw=2.0, ls="--", zorder=4,
            label="$\\sigma(t) = \\|\\mathbf{R}(t)\\|_2$  [reactivity]")
    ax.plot(t_arr[vA1C], A1_1_C[vA1C], color=COL_A1, lw=1.4, ls="-.", zorder=4,
            label="$\\mathcal{A}_1(1) = \\max_k R^k_{\\rm out}$  [first-generation epidemicity]")
    ax.plot(t_arr[vEC],  E_t_C[vEC],   color=COL_ET, lw=1.4, ls=":", zorder=4,
            label="$\\mathcal{E}(t) = X(1,t)$  [risk-averse reproduction number]")
    ax.axhline(1.0, color="0.40", ls=":", lw=1.1, zorder=2)

    # Transient zone: σ > 1 AND ρ < 1
    tr_mask = (sigma_ts > 1) & (rho_ts < 1)
    if tr_mask.any():
        ax.fill_between(t_arr, 1.0, sigma_ts,
                        where=tr_mask,
                        color=COL_ZONE, alpha=0.35, zorder=3,
                        label="Transient amplification zone\n($\\sigma > 1$, $\\mathcal{R} < 1$)")

        idx = np.where(tr_mask)[0]
        t_start, t_end = int(idx[0]), int(idx[-1])
        n_days = int(tr_mask.sum())                           # days in the zone
        n_runs = int(1 + np.count_nonzero(np.diff(idx) > 1))  # separate stretches

        # dotted vertical drop-lines at the first and last zone day
        for t_bd, lbl_txt in [(t_start, f"day {t_start}"), (t_end, f"day {t_end}")]:
            sig_at = float(sigma_ts[t_bd]) if sigma_ts[t_bd] > 0 else 1.0
            ax.plot([t_bd, t_bd], [0, sig_at],
                    color=COL_DROP, lw=1.1, ls=":", zorder=5, alpha=0.85)
            ax.text(t_bd + 1, 0.02,
                    lbl_txt,
                    transform=ax.get_xaxis_transform(),
                    fontsize=5.5, ha="left", va="bottom",
                    color=COL_DROP)

        t_mid = int((t_start + t_end) / 2)
        ax.annotate(
            "",
            xy=(t_end + 0.5, 0.97), xytext=(t_start - 0.5, 0.97),
            xycoords=("data", "axes fraction"),
            textcoords=("data", "axes fraction"),
            arrowprops=dict(arrowstyle="<->", color=COL_DROP, lw=1.1))
        span_txt = (f"one stretch, days {t_start}–{t_end}" if n_runs == 1
                    else f"{n_runs} stretches between days {t_start} and {t_end}")
        ax.text(t_mid, 0.93,
                f"n = {n_days} days with $\\sigma > 1$ and $\\mathcal{{R}} < 1$\n"
                f"({span_txt})",
                transform=ax.get_xaxis_transform(),
                fontsize=6, ha="center", va="top",
                color=COL_DROP,
                bbox=dict(boxstyle="round,pad=0.25", fc="#FFF8DC",
                          ec=COL_DROP, alpha=0.92, lw=0.7))
    else:
        ax.text(0.5, 0.6, "No day with $\\sigma > 1$ and $\\mathcal{R} < 1$",
                transform=ax.transAxes, fontsize=8, ha="center", color="0.45",
                style="italic")

    # axis labels and formatting
    ax.set_xlabel("Day $t$", fontsize=7)
    ax.set_ylabel("$\\mathcal{R}(t)$  /  $\\sigma(t)$", fontsize=7)
    y_top = max(float(rho_ts.max()), float(sigma_ts.max())) * 1.15
    ax.set_ylim(0, max(y_top, 1.4))
    ax.tick_params(axis="both", labelsize=6)
    # summary box: means over the days with ℛ(t) above the noise floor
    att = float(inc.sum() / pops.sum() * 100)
    active = rho_ts > RHO_FLOOR
    _sdiff_c  = float(np.mean((sigma_ts - rho_ts)[active]))
    _sratio_c = float(np.mean((sigma_ts / rho_ts)[active]))
    _ediff_c  = float(np.mean((E_t_C - rho_ts)[active]))
    _eratio_c = float(np.mean((E_t_C / rho_ts)[active]))
    ax.text(0.985, 0.97,
            (f"$\\mathcal{{R}}(0) = {rho_ts[0]:.2f}$\n"
             f"Attack rate = {att:.1f}%\n"
             f"Mean $\\sigma - \\mathcal{{R}}$: ${_sdiff_c:+.3f}$"
             f"  ($\\sigma/\\mathcal{{R}} = {_sratio_c:.3f}$)\n"
             f"Mean $\\mathcal{{E}}(t) - \\mathcal{{R}}$: ${_ediff_c:+.3f}$"
             f"  ($\\mathcal{{E}}(t)/\\mathcal{{R}} = {_eratio_c:.3f}$)"),
            transform=ax.transAxes, fontsize=5.0, ha="right", va="top",
            color="0.2",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.70",
                      alpha=0.90, lw=0.5))
    # peak marker
    ax.axvline(peak, color="0.5", lw=0.7, ls="--", alpha=0.6, zorder=1)
    ax.text(peak - 1, 0.10, f"Epidemic\npeak (d{peak})",
            transform=ax.get_xaxis_transform(), fontsize=5, color="0.4",
            ha="right", va="bottom")

    ax.legend(fontsize=5.2, borderpad=0.3, labelspacing=0.15,
              ncol=2, loc="upper center",
              bbox_to_anchor=(0.5, -0.12), framealpha=0.92)
    _panel_label(ax, "D")

    plt.savefig(f"{save_prefix}_scenario_C_transience.pdf", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved {save_prefix}_scenario_C_transience.pdf")


# ── Lehmer-mean spectrum: Fig SI33 ────────────────────────────────────────

def plot_lehmer_mean_spectrum(sim, scenario_label, save_prefix="fig"):
    """Fig SI33: Lehmer-mean spectrum X(α,t) of the outward reproduction numbers (Eq 41).

        X(α,t) = Σ_j ω_j(α) R^j_out(t),   ω_j(α) ∝ (R^j_out)^α,   α ∈ [0, 20]

    Special cases:
      α = 0  →  arithmetic mean of R^j_out
      α = 1  →  risk-averse reproduction number E(t) = Σ(R^j_out)² / Σ R^j_out (Eq 42)
      α → ∞  →  A₁(1) = max_j R^j_out

    Layout (2 rows):
      Row 0, cols 0–1: heatmap of X(α,t)
      Row 0, col 2:    X(α, t) against α at the early / peak / late days (phase_days)
      Row 1:           X(α, t) against t at selected α, with ℛ(t) and A₁(1)
    """
    inc    = sim["incidence"]
    R_mats = sim["R_matrices"]
    T, N   = inc.shape
    t_arr  = np.arange(T)
    phases = phase_days(inc)
    early, peak, late = phases["early"], phases["peak"], phases["late"]

    # ── compute spectrum ────────────────────────────────────────────────────
    R_out_series = np.array([R_outward(R_mats[t]) for t in range(T)])
    rho_ts       = np.array([R_system(R_mats[t]) for t in range(T)])
    A1_1_ts      = R_out_series.max(axis=1)
    E_t          = risk_averse_R(R_out_series)
    arith_mean   = R_out_series.mean(axis=1)

    # α grid: 0 to 20 inclusive, 80 points
    alpha_arr = np.linspace(0, 20, 80)
    X = lehmer_mean_spectrum(R_out_series, alpha_arr)   # (n_alpha, T)

    fig = plt.figure(figsize=(7.2, 5.6))
    gs  = gridspec.GridSpec(
        2, 3, height_ratios=[1.15, 1],
        hspace=0.42, wspace=0.52,
        left=0.09, right=0.97, top=0.94, bottom=0.10)

    # ── A: heatmap X(α,t) ───────────────────────────────────────────────────
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
    ax.axhline(0.0, color="0.25", lw=0.8, ls="--", alpha=0.7)
    ax.axhline(1.0, color=OKABE_ITO[6], lw=1.0, ls="--", alpha=0.85)
    for t_pt, lbl, col in [(early, "early", OKABE_ITO[2]),
                            (peak,  "peak",  OKABE_ITO[0]),
                            (late,  "late",  OKABE_ITO[4])]:
        ax.axvline(t_pt, color=col, lw=0.8, ls=":", alpha=0.7)
        ax.text(t_pt + 1, alpha_arr[-1] * 0.97, lbl,
                fontsize=4, color=col, va="top")
    ax.set_xlabel("Day $t$", fontsize=7)
    ax.set_ylabel("Lehmer-mean exponent $\\alpha$", fontsize=6)
    ax.tick_params(labelsize=5.5)
    _panel_label(ax, "A")

    # ── B: spectrum against α at three days ─────────────────────────────────
    ax = fig.add_subplot(gs[0, 2])
    for t_pt, lbl, col in [(early, "early", OKABE_ITO[2]),
                           (peak,  "peak",  OKABE_ITO[0]),
                           (late,  "late",  OKABE_ITO[4])]:
        x_slice = X[:, t_pt]
        valid   = ~np.isnan(x_slice)
        ax.plot(alpha_arr[valid], x_slice[valid], color=col, lw=1.1, label=f"Day {t_pt} ({lbl})")
        # α = 0 and α = 1 evaluated exactly, not read off the α grid
        x01 = lehmer_mean_spectrum(R_out_series[t_pt][None, :], np.array([0.0, 1.0]))[:, 0]
        ax.plot(0.0, x01[0], "o", color=col, ms=3, zorder=5)
        ax.plot(1.0, x01[1], "s", color=col, ms=3, zorder=5)
    ax.axhline(1.0, color="0.55", ls="--", lw=0.7)
    ax.set_xlabel("$\\alpha$", fontsize=7)
    ax.set_ylabel("$X(\\alpha, t_*)$", fontsize=7)
    ax.legend(fontsize=4.8, borderpad=0.3, labelspacing=0.12)
    ax.text(0.04, 0.14, "dot: $\\alpha{=}0$   square: $\\alpha{=}1$",
            transform=ax.transAxes, fontsize=4.5, color="0.4")
    ax.tick_params(labelsize=5.5)
    _panel_label(ax, "B")

    # ── C: selected α over time ─────────────────────────────────────────────
    ax = fig.add_subplot(gs[1, :])
    alpha_sel = [0.0, 0.5, 1.0, 2.0, 4.0, 20.0]
    cmap_sel  = plt.cm.plasma(np.linspace(0.1, 0.85, len(alpha_sel)))
    for alpha_s, col_s in zip(alpha_sel, cmap_sel):
        # X(α) evaluated exactly at each α (the α grid does not contain 0.5, 1, 2, 4)
        if alpha_s == 0.0:
            x_ts = arith_mean
        elif alpha_s == 1.0:
            x_ts = E_t
        else:
            x_ts = lehmer_mean_spectrum(R_out_series, np.array([alpha_s]))[0]
        valid = ~np.isnan(x_ts)
        lbl = (f"$\\alpha={alpha_s:.1f}$"
               + (" = $\\mathcal{E}(t)$" if alpha_s == 1.0 else "")
               + (" = arith. mean"        if alpha_s == 0.0 else ""))
        ax.plot(t_arr[valid], x_ts[valid], color=col_s, lw=0.9, label=lbl)
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
    ax.legend(fontsize=4.5, borderpad=0.3, labelspacing=0.10, ncol=4,
              loc="upper right")
    ax.tick_params(labelsize=5.5)
    _panel_label(ax, "C")


    fname = f"{save_prefix}_lehmer_mean_spectrum_{scenario_label.replace(' ', '_')}.pdf"
    plt.savefig(fname, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved {fname}")
