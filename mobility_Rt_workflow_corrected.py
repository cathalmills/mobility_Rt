#!/usr/bin/env python3
"""
Corrected manuscript figure workflow for mobility-informed R(t).

Imports the numerical core from directly_transmitted.py (unchanged) and
regenerates Figure 2, 3, and 4 with:
  - manuscript-aligned notation (f^{jk} superscripts in labels);
  - fixed elasticity axis semantics + infectee marginal panel;
  - new 3×3 Figure 2 layout (network, mobility, incidence, prevalence, …);
  - new 3×4 Figure 3 layout with eigenvector trajectories and 3D-surface variants;
  - Figure 4 twin-axis spacing and plain tick formatting.

Outputs default to ./out_figs/corrected/
"""

from __future__ import annotations

import os
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
import matplotlib.patheffects as pe
from matplotlib.ticker import FormatStrFormatter, FuncFormatter, LogLocator, MaxNLocator, ScalarFormatter

import directly_transmitted as dt

# ── publication style (Nature-friendly) ────────────────────────────────────


def set_modern_style():
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": [
                "Inter",
                "Helvetica Neue",
                "Helvetica",
                "Arial",
                "DejaVu Sans",
            ],
            "font.size": 9,
            "axes.titlesize": 9,
            "axes.labelsize": 9,
            "axes.titleweight": "600",
            "axes.edgecolor": "#2b2b2b",
            "axes.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "xtick.major.width": 0.65,
            "ytick.major.width": 0.65,
            "xtick.major.size": 3.2,
            "ytick.major.size": 3.2,
            "legend.frameon": False,
            "legend.fontsize": 8,
            "figure.facecolor": "white",
            "axes.facecolor": "#fafafa",
            "grid.color": "#d0d0d0",
            "grid.linestyle": "--",
            "grid.linewidth": 0.45,
            "lines.linewidth": 1.15,
            "image.cmap": "cividis",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.03,
        }
    )


def panel_label(ax, letter: str, x=-0.12, y=1.02):
    ax.text(
        x,
        y,
        letter,
        transform=ax.transAxes,
        fontsize=11,
        fontweight="700",
        va="top",
        ha="left",
        color="#111111",
        path_effects=[pe.withStroke(linewidth=2.5, foreground="white")],
    )


OK = dt.OKABE_ITO


# ── derived quantities ─────────────────────────────────────────────────────


def activity_location_prevalence(f_jk: np.ndarray, E_pde: np.ndarray) -> np.ndarray:
    """
    Activity-location prevalence: sum_a sum_j f^{jl}(t) E^j(t,a).

    Code stores f[t,j,l]=f^{jl} and E_pde[t,j,a]=E^j(t,a).
    """
    T, N, _ = E_pde.shape
    out = np.zeros((T, N))
    for t in range(T):
        E_sum = E_pde[t].sum(axis=1)
        out[t] = f_jk[t].T @ E_sum
    return out


def home_mobility_series(f_jk: np.ndarray) -> np.ndarray:
    """Diagonal home fractions f^{jj}(t), shape (T, N)."""
    T, N, _ = f_jk.shape
    return np.array([[float(f_jk[t, j, j]) for j in range(N)] for t in range(T)])


def infector_infectee_elasticity_series(R_mats: np.ndarray):
    """Returns (T,N) ε^k_out=sum_j ε^{kj} and (T,N) ε^j_in=sum_k ε^{kj}."""
    T, N, _ = R_mats.shape
    inf_k = np.zeros((T, N))
    inf_j = np.zeros((T, N))
    for t in range(T):
        Em = dt.sensitivity_elasticity(R_mats[t])["elasticity"]
        inf_k[t] = Em.sum(axis=1)
        inf_j[t] = Em.sum(axis=0)
    return inf_k, inf_j


def eigenvector_timeseries(R_mats: np.ndarray):
    """Dominant v_j (reproductive value) and v*_j (stable distribution) per manuscript.

    Manuscript convention:
      v   = left  eigvec of R(manuscript) = reproductive value   = right_eigvec in code
      v*  = right eigvec of R(manuscript) = stable distribution  = left_eigvec  in code

    Code's R_mat is the transpose of the manuscript's R, so the roles are flipped.
    Returns (v_t, vstar_t) as (T,N) arrays, sum-normalised.
    """
    T, N, _ = R_mats.shape
    v_t = np.zeros((T, N))
    vstar_t = np.zeros((T, N))
    for t in range(T):
        sp = dt.spectral_analysis(R_mats[t])
        v_t[t] = sp["right_eigvec"]    # right eigvec of code = left eigvec of manuscript = v (reproductive value)
        vstar_t[t] = sp["left_eigvec"] # left eigvec of code = right eigvec of manuscript = v* (stable distribution)
    return v_t, vstar_t


def _panel_label_3d(ax, letter: str):
    ax.text2D(
        -0.08,
        1.05,
        letter,
        transform=ax.transAxes,
        fontsize=11,
        fontweight="700",
        va="top",
        ha="left",
        color="#111111",
        path_effects=[pe.withStroke(linewidth=2.5, foreground="white")],
    )


def _style_3d_ax(ax, *, elev=28, azim=-55):
    for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
        pane.fill = False
        pane.set_edgecolor("#cccccc")
    ax.xaxis._axinfo["grid"]["color"] = (0, 0, 0, 0)
    ax.yaxis._axinfo["grid"]["color"] = (0, 0, 0, 0)
    ax.zaxis._axinfo["grid"]["color"] = (0, 0, 0, 0)
    ax.view_init(elev=elev, azim=azim)
    ax.tick_params(labelsize=6)


# ── network panel (static, mean mobility on edges) ──────────────────────────


def plot_static_mobility_network(ax, coords, pops, f_mean: np.ndarray, *, threshold: float):
    """Geographic network: edge colour/width ∝ mean f^{jk}; node ∝ resident pop."""
    N = coords.shape[0]
    x, y = coords[:, 0], coords[:, 1]
    off = ~np.eye(N, dtype=bool)
    fmax = float(f_mean[off].max()) if f_mean[off].max() > 1e-12 else 1.0

    segments, widths, colours = [], [], []
    cmap = plt.get_cmap("Blues")
    for j in range(N):
        for k in range(N):
            if j == k:
                continue
            w = float(f_mean[j, k])
            if w < threshold:
                continue
            segments.append(np.column_stack([[x[j], x[k]], [y[j], y[k]]]))
            frac = w / fmax
            widths.append(0.35 + 2.4 * frac)
            colours.append(cmap(0.25 + 0.75 * frac))

    if segments:
        lc = LineCollection(
            segments, colors=colours, linewidths=widths, alpha=0.72, zorder=1
        )
        ax.add_collection(lc)

    pop = np.asarray(pops, float)
    s_node = 40 + 520 * (np.sqrt(pop / pop.max()))
    sc = ax.scatter(
        x,
        y,
        s=s_node,
        c=np.log10(pop),
        cmap="YlOrBr",
        edgecolors="#1a1a1a",
        linewidths=0.65,
        zorder=3,
        vmin=np.log10(pop.min()),
        vmax=np.log10(pop.max()),
    )
    for j in range(N):
        ax.text(
            x[j],
            y[j],
            f"{j + 1}",
            ha="center",
            va="center",
            fontsize=7,
            color="#1a1a1a",
            fontweight="700",
            zorder=5,
            path_effects=[
                pe.withStroke(linewidth=2.8, foreground="white", alpha=0.95),
                pe.withStroke(linewidth=1.0, foreground="#f5f5f5", alpha=0.6),
            ],
        )
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    _colorbar_right(ax, sc, r"$\log_{10} N^j$")


def _colorbar_right(ax, mappable, label: str, *, shrink=0.82, pad=0.04):
    """Vertical colorbar to the right of a 2D axes (label as a title above the bar)."""
    cb = plt.colorbar(mappable, ax=ax, fraction=0.046, pad=pad, shrink=shrink)
    cb.ax.set_title(label, fontsize=8, pad=4, loc="center")
    cb.ax.tick_params(labelsize=7, length=2.5)
    cb.outline.set_linewidth(0.5)
    return cb


def _colorbar_right_3d(fig, ax, mappable, label: str, *, shrink=0.68, pad=0.14):
    """Vertical colorbar to the right of a 3D axes (label as a title above the bar)."""
    cb = fig.colorbar(mappable, ax=ax, shrink=shrink, pad=pad, fraction=0.03)
    cb.ax.set_title(label, fontsize=8, pad=4, loc="center")
    cb.ax.tick_params(labelsize=7, length=2.5)
    cb.outline.set_linewidth(0.5)
    return cb


# ── tile heatmaps and 3D surfaces ─────────────────────────────────────────────


def _plot_matrix_tile(ax, Z, *, xlabels, ylabels, cbar_title, letter, cmap="Blues"):
    im = ax.imshow(Z, aspect="auto", origin="upper", cmap=cmap, interpolation="nearest")
    ax.set_xticks(range(len(xlabels)))
    ax.set_xticklabels(xlabels, fontsize=7, rotation=45, ha="right")
    ax.set_yticks(range(len(ylabels)))
    ax.set_yticklabels(ylabels, fontsize=7)
    _colorbar_right(ax, im, cbar_title)
    panel_label(ax, letter)
    ax.grid(False)


def _plot_matrix_surface(fig, gs_cell, Z, *, cbar_title, letter, cmap="viridis", zlabel=""):
    """3D surface for matrix Z[k,j] = R^{kj} (rows infector k, cols infectee j)."""
    ax = fig.add_subplot(gs_cell, projection="3d")
    Nk, Nj = Z.shape
    X, Y = np.meshgrid(np.arange(Nj), np.arange(Nk))
    zc = np.maximum(Z, 0.0)
    surf = ax.plot_surface(
        X,
        Y,
        zc,
        cmap=cmap,
        linewidth=0,
        antialiased=True,
        alpha=0.92,
        rstride=1,
        cstride=1,
    )
    ax.set_xlabel(r"Infectee $j$", fontsize=7, labelpad=4)
    ax.set_ylabel(r"Infector $k$", fontsize=7, labelpad=4)
    if zlabel:
        ax.set_zlabel(zlabel, fontsize=7, labelpad=6)
    ax.set_xticks(range(Nj))
    ax.set_yticks(range(Nk))
    ax.tick_params(axis="z", pad=2)
    _style_3d_ax(ax)
    _colorbar_right_3d(fig, ax, surf, cbar_title)
    _panel_label_3d(ax, letter)
    return ax


def _plot_time_location_tile(
    ax,
    Z,
    T: int,
    Nloc: int,
    *,
    cbar_title: str,
    letter: str,
    cmap="viridis",
    y_label=r"Location",
    vmin=None,
    vmax=None,
):
    """Z shape (T, Nloc)."""
    Zt = Z.T
    kw = dict(aspect="auto", origin="upper", extent=[-0.5, T - 0.5, Nloc - 0.5, -0.5], cmap=cmap, interpolation="nearest")
    if vmin is not None:
        kw["vmin"] = vmin
    if vmax is not None:
        kw["vmax"] = vmax
    im = ax.imshow(Zt, **kw)
    loc = [f"L{i+1}" for i in range(Nloc)]
    ax.set_xlim(-0.5, T - 0.5)
    ax.set_ylim(Nloc - 0.5, -0.5)
    ax.set_yticks(range(Nloc))
    ax.set_yticklabels(loc, fontsize=7)
    ax.set_xlabel("Day $t$", fontsize=9)
    ax.set_ylabel(y_label, fontsize=9)
    _colorbar_right(ax, im, cbar_title)
    panel_label(ax, letter)


def _plot_time_location_surface(
    fig,
    gs_cell,
    Z,
    T: int,
    Nloc: int,
    *,
    cbar_title: str,
    letter: str,
    cmap="viridis",
    y_label=r"Location $j$",
    zlabel="",
    vmin=None,
    vmax=None,
):
    """3D surface: x=day t, y=location, z=Z(t,·)."""
    ax = fig.add_subplot(gs_cell, projection="3d")
    TT, LL = np.meshgrid(np.arange(T), np.arange(Nloc))
    Zt = Z.T
    if vmin is None or vmax is None:
        zmin, zmax = np.nanpercentile(Z, [2, 98])
    else:
        zmin, zmax = vmin, vmax
    if not np.isfinite(zmax) or zmax <= zmin:
        zmax = zmin + 1e-9
    surf = ax.plot_surface(
        TT,
        LL,
        Zt,
        cmap=cmap,
        linewidth=0,
        antialiased=True,
        alpha=0.9,
        vmin=zmin,
        vmax=zmax,
        rstride=max(1, T // 80),
        cstride=1,
    )
    ax.set_xlabel("Day $t$", fontsize=7, labelpad=4)
    ax.set_ylabel(y_label, fontsize=7, labelpad=4)
    if zlabel:
        ax.set_zlabel(zlabel, fontsize=7, labelpad=6)
    ax.set_yticks(range(Nloc))
    ax.set_yticklabels([f"L{i+1}" for i in range(Nloc)], fontsize=5)
    ax.tick_params(axis="z", pad=2)
    _style_3d_ax(ax, elev=32, azim=-58)
    _colorbar_right_3d(fig, ax, surf, cbar_title)
    _panel_label_3d(ax, letter)
    return ax


# ── Figure 2 (3 × 3) ────────────────────────────────────────────────────────


def plot_fig02_overview(
    sim: dict,
    city_data: tuple,
    f_jk: np.ndarray,
    gen_time_pmf: np.ndarray,
    max_days: int,
    scenario_name: str,
    save_prefix: str,
    *,
    surface_3d: bool = False,
):
    coords, pops, _, node_types, meta = city_data
    inc = sim["incidence"]
    R_mats = sim["R_matrices"]
    S_series = sim["susceptibles"]
    E_pde = sim["E_pde_state"]
    T, N = inc.shape
    loc = [f"L{i+1}" for i in range(N)]

    S_eff_s, _ = dt.compute_effective_populations_series(
        f_jk, S_series, inc, gen_time_pmf, max_days
    )
    prev_act = activity_location_prevalence(f_jk, E_pde)
    R_sys = np.array([dt.R_system(R_mats[t]) for t in range(T)])
    eps_out, eps_in = infector_infectee_elasticity_series(R_mats)
    f_mean = f_jk.mean(axis=0)
    f_diag = home_mobility_series(f_jk)

    sfx = "_surface" if surface_3d else ""
    fig = plt.figure(figsize=(11.4, 10.8), constrained_layout=False)
    gs = gridspec.GridSpec(
        3, 3, figure=fig, hspace=0.52, wspace=0.48, left=0.06, right=0.98, top=0.96, bottom=0.06
    )

    # A — network
    ax = fig.add_subplot(gs[0, 0])
    thr = 0.004 if meta.get("scenario") == "lagos" else 0.0012
    plot_static_mobility_network(ax, coords, pops, f_mean, threshold=thr)
    panel_label(ax, "A")

    # B — mean f^{jk} (always tile, including in surface layout)
    ax = fig.add_subplot(gs[0, 1])
    _plot_matrix_tile(ax, f_mean, xlabels=loc, ylabels=loc, cbar_title=r"$\bar{f}^{jk}$", letter="B", cmap="Blues")
    ax.set_xlabel(r"Activity $k$")
    ax.set_ylabel(r"Residence $j$")

    # C — home fraction f^{jj}(t) (always tile)
    ax = fig.add_subplot(gs[0, 2])
    _plot_time_location_tile(
        ax,
        f_diag,
        T,
        N,
        cbar_title=r"$f^{jj}(t)$",
        letter="C",
        cmap="RdYlGn",
        y_label=r"Residence $j$",
        vmin=0.3,
        vmax=1.0,
    )

    # D — residence incidence E^j(t,0)
    if surface_3d:
        _plot_time_location_surface(
            fig, gs[1, 0], inc, T, N, cbar_title=r"$E^j(t,0)$", letter="D", cmap="YlOrRd", y_label=r"Residence $j$"
        )
    else:
        ax = fig.add_subplot(gs[1, 0])
        _plot_time_location_tile(ax, inc, T, N, cbar_title=r"$E^j(t,0)$", letter="D", cmap="YlOrRd", y_label=r"Residence $j$")

    # E — activity prevalence
    if surface_3d:
        _plot_time_location_surface(
            fig,
            gs[1, 1],
            prev_act,
            T,
            N,
            cbar_title=r"$\sum_a\sum_j f^{jl}E^j(t, a)$",
            letter="E",
            cmap="magma",
            y_label=r"Activity $l$",
        )
    else:
        ax = fig.add_subplot(gs[1, 1])
        _plot_time_location_tile(
            ax, prev_act, T, N, cbar_title=r"$\sum_a\sum_j f^{jl}E^j(t, a)$", letter="E", cmap="magma", y_label=r"Activity $l$"
        )

    # F — S^l_eff
    if surface_3d:
        _plot_time_location_surface(
            fig,
            gs[1, 2],
            S_eff_s / 1e3,
            T,
            N,
            cbar_title=r"$S^l_{\mathrm{eff}}(t)$ ($10^3$)",
            letter="F",
            cmap="Blues",
            y_label=r"Activity $l$",
        )
    else:
        ax = fig.add_subplot(gs[1, 2])
        _plot_time_location_tile(
            ax,
            S_eff_s / 1e3,
            T,
            N,
            cbar_title=r"$S^l_{\mathrm{eff}}(t)$ ($10^3$)",
            letter="F",
            cmap="Blues",
            y_label=r"Activity $l$",
        )

    # G — R(t), E(t), incidence
    ax = fig.add_subplot(gs[2, 0])
    ax2 = ax.twinx()
    ax2.spines["right"].set_visible(True)
    ax2.spines["top"].set_visible(False)
    vld = R_sys > 0
    R_out_series = np.array([dt.R_outward(R_mats[t]) for t in range(T)])
    E_t = np.array(
        [np.sum(R_out_series[t] ** 2) / (np.sum(R_out_series[t]) + 1e-300) for t in range(T)]
    )
    vE = E_t > 0
    ax.plot(np.where(vld)[0], R_sys[vld], color=OK[4], lw=1.2, label=r"$\mathcal{R}(t)$")
    ax.plot(np.where(vE)[0], E_t[vE], color=OK[6], lw=1.1, ls="--", label=r"$\mathcal{E}(t)$")
    ax.axhline(1.0, color="#555", ls="--", lw=0.85)
    total_inc = inc.sum(axis=1)
    ax2.fill_between(np.arange(T), total_inc / 1e3, alpha=0.18, color=OK[1])
    ax2.plot(np.arange(T), total_inc / 1e3, color=OK[1], lw=1.05, label=r"$\sum_j E^j(t, 0)$")
    ax.set_ylabel(r"$\mathcal{R}(t)$, $\mathcal{E}(t)$", color=OK[4])
    ax2.set_ylabel(r"$\sum_j E^j(t,0)$ ($10^3$)", color=OK[1])
    ax.set_xlabel("Day $t$")
    ax.tick_params(axis="y", labelcolor=OK[4])
    ax2.tick_params(axis="y", labelcolor=OK[1])
    ax.set_ylim(0, max(2.0, float(R_sys[vld].max()) * 1.12) if vld.any() else 3.5)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=7, loc="upper right")
    panel_label(ax, "G")

    # H — infector elasticity ε^k_out(t) (always tile)
    vmax_e = np.percentile(eps_out[eps_out > 0], 97) if (eps_out > 0).any() else 1.0
    ax = fig.add_subplot(gs[2, 1])
    _plot_time_location_tile(
        ax,
        eps_out,
        T,
        N,
        cbar_title=r"$\varepsilon^k_{\mathrm{out}}(t)$",
        letter="H",
        cmap="YlOrRd",
        y_label=r"Infector $k$",
        vmin=0,
        vmax=vmax_e,
    )

    # I — infectee elasticity ε^j_in(t) (always tile)
    vmax_i = np.percentile(eps_in[eps_in > 0], 97) if (eps_in > 0).any() else 1.0
    ax = fig.add_subplot(gs[2, 2])
    _plot_time_location_tile(
        ax,
        eps_in,
        T,
        N,
        cbar_title=r"$\varepsilon^j_{\mathrm{in}}(t)$",
        letter="I",
        cmap="magma",
        y_label=r"Infectee $j$",
        vmin=0,
        vmax=vmax_i,
    )
    os.makedirs(os.path.dirname(save_prefix) or ".", exist_ok=True)
    path = f"{save_prefix}_02_overview{sfx}.pdf"
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved {path}")


# ── Figure 3 (3 × 4) ────────────────────────────────────────────────────────


def plot_fig03_taxonomy(
    sim: dict,
    city_data: tuple,
    R_independent: np.ndarray,
    gt_snaps: dict,
    w_within: np.ndarray,
    w_between: np.ndarray,
    scenario_name: str,
    save_prefix: str,
    *,
    surface_3d: bool = False,
):
    inc = sim["incidence"]
    inc_mat = sim["incidence_matrix"]
    R_mats = sim["R_matrices"]
    R_meet_s = sim["R_meeting_series"]
    coords, pops, _, node_types, meta = city_data
    T, N = inc.shape
    loc = [f"L{i+1}" for i in range(N)]

    R_out_s = np.array([dt.R_outward(R_mats[t]) for t in range(T)])
    R_in_s = np.array([dt.R_inward(R_mats[t]) for t in range(T)])
    peak = int(inc.sum(axis=1).argmax())

    i_hub, i_mid, i_per, show_locs, show_lbls = dt.representative_locs(city_data)

    gt_p = gt_snaps["peak"]
    g_pw, g_out, g_in = gt_p["g_pairwise"], gt_p["g_outward"], gt_p["g_inward"]
    days = np.arange(len(w_within))
    g_univ = w_within / w_within.sum() if w_within.sum() > 0 else w_within.copy()
    GT_univ = float(np.sum(days * g_univ))

    eps_out, eps_in = infector_infectee_elasticity_series(R_mats)
    mean_eps_out = np.nanmean(eps_out, axis=0)
    mean_eps_in = np.nanmean(eps_in, axis=0)
    v_t, vstar_t = eigenvector_timeseries(R_mats)

    T_ser = np.zeros((T, N))
    for t_idx in range(T):
        T_ser[t_idx] = dt.type_reproduction_numbers(R_mats[t_idx])

    sfx = "_surface" if surface_3d else ""
    fig = plt.figure(figsize=(14.5, 10.2))
    gs = gridspec.GridSpec(
        3, 4, figure=fig, hspace=0.50, wspace=0.48, left=0.05, right=0.98, top=0.96, bottom=0.06
    )

    # A — GT
    ax = fig.add_subplot(gs[0, 0])
    g_jj = g_pw[:, i_hub, i_hub]
    if g_jj.sum() > 0.5:
        ax.plot(days, g_jj, color=OK[0], lw=1.4, label=r"$g^{jj}$ (within)")
    if g_out[:, i_hub].sum() > 0.5:
        ax.plot(days, g_out[:, i_hub], color=OK[1], lw=1.1, ls="--", label=r"$g^k_{\mathrm{out}}$")
    if g_in[:, i_hub].sum() > 0.5:
        ax.plot(days, g_in[:, i_hub], color=OK[5], lw=1.1, ls=":", label=r"$g^j_{\mathrm{in}}$")
    ax.plot(days, g_univ, color="#555", lw=1.5, ls="-", alpha=0.45, label=rf"$p/\int p$")
    ax.set_xlabel("Age $a_E$ (days)")
    ax.set_ylabel("Probability")
    ax.legend(fontsize=6.5, ncol=2)
    panel_label(ax, "A")

    # B — R^k_out(t)
    pos = R_out_s[R_out_s > 0]
    vmin_o = np.percentile(pos, 2) if pos.size else 0.0
    vmax_o = np.percentile(pos, 97) if pos.size else 3.0
    if surface_3d:
        _plot_time_location_surface(
            fig,
            gs[0, 1],
            R_out_s,
            T,
            N,
            cbar_title=r"$R^k_{\mathrm{out}}$",
            letter="B",
            cmap="plasma",
            y_label=r"Infector $k$",
            vmin=vmin_o,
            vmax=vmax_o,
        )
    else:
        ax = fig.add_subplot(gs[0, 1])
        _plot_time_location_tile(
            ax,
            R_out_s,
            T,
            N,
            cbar_title=r"$R^k_{\mathrm{out}}$",
            letter="B",
            cmap="plasma",
            y_label=r"Infector $k$",
            vmin=vmin_o,
            vmax=vmax_o,
        )

    # C — R^j_in(t)
    posi = R_in_s[R_in_s > 0]
    vmin_i = np.percentile(posi, 2) if posi.size else 0.0
    vmax_i = np.percentile(posi, 97) if posi.size else 3.0
    if surface_3d:
        _plot_time_location_surface(
            fig,
            gs[0, 2],
            R_in_s,
            T,
            N,
            cbar_title=r"$R^j_{\mathrm{in}}$",
            letter="C",
            cmap="viridis",
            y_label=r"Infectee $j$",
            vmin=vmin_i,
            vmax=vmax_i,
        )
    else:
        ax = fig.add_subplot(gs[0, 2])
        _plot_time_location_tile(
            ax,
            R_in_s,
            T,
            N,
            cbar_title=r"$R^j_{\mathrm{in}}$",
            letter="C",
            cmap="viridis",
            y_label=r"Infectee $j$",
            vmin=vmin_i,
            vmax=vmax_i,
        )

    # D — elasticity rankings (time-mean marginals)
    ax = fig.add_subplot(gs[0, 3])
    ax.scatter(mean_eps_out, mean_eps_in, s=55, c=np.arange(N), cmap="tab10", edgecolors="#222", linewidths=0.5)
    for j in range(N):
        ax.annotate(
            f"L{j+1}",
            (mean_eps_out[j], mean_eps_in[j]),
            textcoords="offset points",
            xytext=(4, 3),
            fontsize=7,
            color="#222",
        )
    mx = max(mean_eps_out.max(), mean_eps_in.max(), 1e-9)
    ax.plot([0, mx], [0, mx], color="#999", ls="--", lw=0.85, label="equal marginals")
    ax.set_xlabel(r"time-mean $\varepsilon^k_{\mathrm{out}}$")
    ax.set_ylabel(r"time-mean $\varepsilon^j_{\mathrm{in}}$")
    ax.legend(fontsize=6.5, loc="lower right")
    panel_label(ax, "D")

    # E — left eigenvector v_j (always tile)
    ax = fig.add_subplot(gs[1, 0])
    _plot_time_location_tile(
        ax, v_t, T, N, cbar_title=r"$v_j$", letter="E", cmap="YlGnBu", y_label=r"Location $j$", vmin=0
    )

    # F — right eigenvector v^{*}_j (always tile)
    ax = fig.add_subplot(gs[1, 1])
    _plot_time_location_tile(
        ax,
        vstar_t,
        T,
        N,
        cbar_title=r"$v^{*}_j$",
        letter="F",
        cmap="OrRd",
        y_label=r"Location $j$",
        vmin=0,
    )

    # G — R^{kj} at peak (always tile)
    Rpk = R_mats[peak]
    axg = fig.add_subplot(gs[1, 2])
    _plot_matrix_tile(
        axg,
        Rpk,
        xlabels=loc,
        ylabels=loc,
        cbar_title=r"$R^{kj}$",
        letter="G",
        cmap="PuOr",
    )
    axg.set_xlabel(r"Infectee $j$")
    axg.set_ylabel(r"Infector $k$")

    # H — E^{kj} at peak (always tile)
    Epk = inc_mat[peak]
    axh = fig.add_subplot(gs[1, 3])
    _plot_matrix_tile(
        axh,
        Epk,
        xlabels=loc,
        ylabels=loc,
        cbar_title=r"$E^{kj}$",
        letter="H",
        cmap="YlGn",
    )
    axh.set_xlabel(r"Infectee $j$")
    axh.set_ylabel(r"Infector $k$")

    # I — R̂_ind vs R^j_in
    ax = fig.add_subplot(gs[2, 0])
    for j, lbl, col in zip(show_locs, show_lbls, [OK[0], OK[2], OK[5]]):
        vi = ~np.isnan(R_independent[:, j])
        vm = R_in_s[:, j] > 0
        b = vi & vm
        if b.sum() > 3:
            ax.plot(np.where(b)[0], R_independent[b, j], "--", color=col, lw=1.0, alpha=0.75)
            ax.plot(np.where(b)[0], R_in_s[b, j], "-", color=col, lw=1.0, label=lbl)
    ax.axhline(1, color="#666", ls="--", lw=0.85)
    h, lab = ax.get_legend_handles_labels()
    h += [
        Line2D([0], [0], color="0.35", lw=1.0, ls="-"),
        Line2D([0], [0], color="0.35", lw=1.0, ls="--"),
    ]
    lab += [r"$R^j_{\mathrm{in}}$ (solid)", r"$\hat R^j_{\mathrm{ind}}$ (dashed)"]
    ax.legend(handles=h, labels=lab, fontsize=6.5, ncol=1)
    ax.set_xlabel("Day $t$")
    ax.set_ylabel(r"$R$")
    panel_label(ax, "I")

    # J — R̂_ind vs R^k_out
    ax = fig.add_subplot(gs[2, 1])
    for j, lbl, col in zip(show_locs, show_lbls, [OK[0], OK[2], OK[5]]):
        vi = ~np.isnan(R_independent[:, j])
        vo = R_out_s[:, j] > 0
        b = vi & vo
        if b.sum() > 3:
            ax.plot(np.where(b)[0], R_independent[b, j], "--", color=col, lw=1.0, alpha=0.75)
            ax.plot(np.where(b)[0], R_out_s[b, j], "-", color=col, lw=1.0, label=lbl)
    ax.axhline(1, color="#666", ls="--", lw=0.85)
    h2, lab2 = ax.get_legend_handles_labels()
    h2 += [
        Line2D([0], [0], color="0.35", lw=1.0, ls="-"),
        Line2D([0], [0], color="0.35", lw=1.0, ls="--"),
    ]
    lab2 += [r"$R^k_{\mathrm{out}}$ (solid)", r"$\hat R^j_{\mathrm{ind}}$ (dashed)"]
    ax.legend(handles=h2, labels=lab2, fontsize=6.5, ncol=1)
    ax.set_xlabel("Day $t$")
    ax.set_ylabel(r"$R$")
    panel_label(ax, "J")

    # K — type reproduction T_j(t)
    if surface_3d:
        _plot_time_location_surface(
            fig,
            gs[2, 2],
            np.nan_to_num(T_ser, nan=0.0),
            T,
            N,
            cbar_title=r"$T_j$",
            letter="K",
            cmap="plasma",
            y_label=r"Location $j$",
        )
    else:
        ax = fig.add_subplot(gs[2, 2])
        _plot_time_location_tile(
            ax,
            np.nan_to_num(T_ser, nan=0.0),
            T,
            N,
            cbar_title=r"$T_j$",
            letter="K",
            cmap="plasma",
            y_label=r"Location $j$",
        )

    # L — R^l_meeting(t)
    posm = R_meet_s[R_meet_s > 0]
    vmin_m = float(np.percentile(posm, 2)) if posm.size else 0.0
    vmax_m = float(np.percentile(posm, 97)) if posm.size else 3.0
    if surface_3d:
        _plot_time_location_surface(
            fig,
            gs[2, 3],
            R_meet_s,
            T,
            N,
            cbar_title=r"$R^l_{\mathrm{meeting}}$",
            letter="L",
            cmap="cividis",
            y_label=r"Meeting $l$",
            vmin=vmin_m,
            vmax=vmax_m,
        )
    else:
        ax = fig.add_subplot(gs[2, 3])
        _plot_time_location_tile(
            ax,
            R_meet_s,
            T,
            N,
            cbar_title=r"$R^l_{\mathrm{meeting}}$",
            letter="L",
            cmap="cividis",
            y_label=r"Meeting $l$",
            vmin=vmin_m,
            vmax=vmax_m,
        )
    path = f"{save_prefix}_03_taxonomy{sfx}.pdf"
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved {path}")


# ── Figure 4 (patched layout + formatters) ─────────────────────────────────


def plot_fig04_spectral(sim: dict, city_data: tuple, R_mat_alt_t0: np.ndarray, scenario_name: str, w_within, w_between, max_days: int, save_prefix: str):
    """Spectral figure with wider gutter for panel A twin axis and plain ticks on D/F."""
    inc = sim["incidence"]
    R_mats = sim["R_matrices"]
    coords, pops, dists, node_types, meta = city_data
    T, N = inc.shape

    R_sys = np.array([dt.R_system(R_mats[t]) for t in range(T)])
    specs = [dt.spectral_analysis(R_mats[t]) for t in range(T)]
    mix_ts = np.array([s["mixing_ratio"] for s in specs])
    sigma_ts = np.array([dt.reactivity(R_mats[t])["sigma"] for t in range(T)])
    peak = int(inc.sum(axis=1).argmax())
    early = max(1, peak // 3)
    late = min(T - 1, peak + 30)

    fig = plt.figure(figsize=(7.4, 7.8))
    # Extra column between A and B for gutter; taller middle row for C–E separation
    gs = gridspec.GridSpec(
        3,
        7,
        figure=fig,
        hspace=0.82,
        wspace=0.58,
        height_ratios=[1.0, 1.22, 1.05],
        left=0.07,
        right=0.97,
        top=0.96,
        bottom=0.08,
    )

    # A — columns 0–2, B — columns 4–6 (column 3 empty)
    ax = fig.add_subplot(gs[0, 0:3])
    ax2 = ax.twinx()
    ax2.spines["right"].set_visible(True)
    ax2.spines["right"].set_position(("axes", 1.02))
    ax2.spines["top"].set_visible(False)
    vm = mix_ts > 0
    ax.plot(np.where(vm)[0], mix_ts[vm], color=OK[2], lw=1.05, label=r"$s(t)$", zorder=5)
    dow_scale = np.array([1.00, 1.00, 1.00, 1.00, 0.95, 0.90, 0.75])
    dow_pattern = np.array([dow_scale[t % 7] for t in range(T)])
    ax2.plot(range(T), dow_pattern, color=OK[0], lw=1.05, ls="--", alpha=0.42, label="DoW scale", zorder=3)
    ax2.set_ylabel("DoW scale", color=OK[0], fontsize=8)
    ax2.tick_params(axis="y", labelcolor=OK[0], labelsize=7)
    # Equally spaced right-hand ticks over the meaningful DoW range (0.75–1.0)
    ax2.set_ylim(0.7, 1.0)
    ax2.set_yticks([0.70, 0.80, 0.90, 1.00])
    ax.set_xlabel("Day $t$")
    ax.set_ylabel(r"$s(t)=|\lambda_2|/\mathcal{R}$")
    ax.set_ylim(0, 1.05)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=7, loc="upper left")
    panel_label(ax, "A")

    ax = fig.add_subplot(gs[0, 4:7])
    t_arr = np.arange(T)
    vR = R_sys > 0
    vsig = sigma_ts > 0
    A1_1_ts = np.array([float(np.max(dt.R_outward(R_mats[t]))) for t in range(T)])
    R_out_b = np.array([dt.R_outward(R_mats[t]) for t in range(T)])
    E_t_b = np.array([np.sum(R_out_b[t] ** 2) / (np.sum(R_out_b[t]) + 1e-300) for t in range(T)])
    vEb = E_t_b > 0
    vA1 = A1_1_ts > 0
    ax.plot(t_arr[vR], R_sys[vR], color=OK[4], lw=1.05, label=r"$\mathcal{R}(t)$")
    ax.plot(t_arr[vsig], sigma_ts[vsig], color=OK[5], lw=1.0, label=r"$\sigma(t)$")
    ax.plot(t_arr[vA1], A1_1_ts[vA1], color=OK[1], lw=1.0, ls="-.", label=r"$\mathcal{A}_1(1)$")
    ax.plot(t_arr[vEb], E_t_b[vEb], color=OK[6], lw=1.0, ls="--", label=r"$\mathcal{E}(t)$")
    ax.axhline(1.0, color="#666", ls="--", lw=0.85)
    transient_mask = (sigma_ts > 1) & (R_sys < 1)
    if transient_mask.any():
        ax.fill_between(
            t_arr,
            1.0,
            sigma_ts,
            where=transient_mask,
            color="orange",
            alpha=0.14,
            label=r"$\sigma>1$, $\mathcal{R}<1$",
        )
    ax.set_xlabel("Day $t$")
    ax.set_ylabel("Value")
    ax.legend(fontsize=6.5, ncol=2, loc="upper right")
    yfmt = ScalarFormatter(useOffset=False)
    yfmt.set_scientific(False)
    ax.yaxis.set_major_formatter(yfmt)
    panel_label(ax, "B")

    # C — amplification (span 0:2): ℓ2-norm A(n) solid + ℓ1-norm A1(n) dotted
    ax = fig.add_subplot(gs[1, 0:2])
    n_max_env = 20
    for t_phase, phase_name, col in [(early, "early", OK[2]), (peak, "peak", OK[5]), (late, "late", OK[0])]:
        env = dt.amplification_envelope(R_mats[t_phase], n_max=n_max_env)
        rho_n = env["rho_n"]
        # ℓ2-norm envelope A(n) = ‖R^n‖_2
        ax.plot(env["n"], env["A"] / (rho_n + 1e-300), color=col, lw=1.05,
                label=f"$A(n)$ {phase_name}")
        # ℓ1-norm envelope A1(n) = max row sum of R^n
        Rn = np.eye(R_mats[t_phase].shape[0])
        A1_n = np.zeros(n_max_env + 1)
        for n in range(n_max_env + 1):
            A1_n[n] = float(np.max(Rn.sum(axis=1)))
            Rn = Rn @ R_mats[t_phase]
        ax.plot(env["n"], A1_n / (rho_n + 1e-300), color=col, lw=0.8, ls=":",
                label=f"$\\mathcal{{A}}_1(n)$ {phase_name}")
    ax.axhline(1.0, color="#888", ls="--", lw=0.75)
    ax.set_xlabel("$n$ (generations)")
    ax.set_ylabel(r"Envelope$/\mathcal{R}^n$")
    ax.set_title(r"Amplification envelopes", fontsize=7, pad=3)
    ax.legend(fontsize=5.5, borderpad=0.3, labelspacing=0.10, ncol=2,
              loc="upper right")
    panel_label(ax, "C")

    # D — eigenvalues (plain decimal ticks); column 2 left empty as a gutter so
    # panel C's title/legend no longer overlaps panel D.
    ax = fig.add_subplot(gs[1, 3:5])
    lam1 = np.array([float(np.abs(specs[t]["eigenvalues"][0])) for t in range(T)])
    lam2 = np.array(
        [float(np.abs(specs[t]["eigenvalues"][1])) if len(specs[t]["eigenvalues"]) > 1 else 0.0 for t in range(T)]
    )
    lam3 = np.array(
        [float(np.abs(specs[t]["eigenvalues"][2])) if len(specs[t]["eigenvalues"]) > 2 else 0.0 for t in range(T)]
    )
    ax.plot(t_arr, lam1, color=OK[4], lw=1.05, label=r"$|\lambda_1|$")
    ax.plot(t_arr, lam2, color=OK[2], lw=1.0, ls="--", label=r"$|\lambda_2|$")
    ax.plot(t_arr, lam3, color=OK[0], lw=0.95, ls=":", label=r"$|\lambda_3|$")
    ax.axvline(peak, color="#777", ls="--", lw=0.8, alpha=0.75)
    ax.set_xlabel("Day $t$")
    ax.set_ylabel("Eigenvalue magnitude")
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6, prune=None))
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.3f"))
    ax.legend(fontsize=7)
    panel_label(ax, "D")

    # E — condition number
    ax = fig.add_subplot(gs[1, 5:7])

    def _cond_eigvec(R_m):
        ev_r, evec_r = np.linalg.eig(R_m)
        ev_l, evec_l = np.linalg.eig(R_m.T)
        idx_r = int(np.argmax(np.abs(ev_r)))
        idx_l = int(np.argmax(np.abs(ev_l)))
        w = evec_r[:, idx_r]
        v = evec_l[:, idx_l]
        return 1.0 / (abs(float(v @ w)) + 1e-300)

    cond_ts = np.minimum(np.array([_cond_eigvec(R_mats[t]) for t in range(T)]), 1e4)

    def _kappa_log_tick(y, _pos):
        if y <= 0:
            return ""
        if y >= 1000:
            return f"{int(round(y))}"
        if y >= 10:
            return f"{int(round(y))}"
        return f"{y:.1f}"

    ax.semilogy(np.arange(T), cond_ts, color=OK[1], lw=1.0, label=r"$\kappa(\mathbf{R})$")
    ax.axvline(peak, color="#777", ls="--", lw=0.8, alpha=0.75)
    ax.set_xlabel("Day $t$")
    ax.set_ylabel(r"$\kappa$ (log)")
    ax.yaxis.set_major_locator(LogLocator(base=10, subs=(1.0, 2.0, 5.0), numticks=8))
    ax.yaxis.set_major_formatter(FuncFormatter(_kappa_log_tick))
    ax.yaxis.set_minor_formatter(FuncFormatter(_kappa_log_tick))
    ot = ax.yaxis.get_offset_text()
    ot.set_visible(False)
    ax.legend(fontsize=7)
    panel_label(ax, "E")

    # F — within fraction (twin right axis plain)
    # Wider gutter (wspace) so the heatmap colorbar / twin-axis legend clear the
    # right-hand bar panel; slightly narrower bar panel to compensate.
    gs_e = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[2, :], width_ratios=[3, 1.05], wspace=0.9)
    ax = fig.add_subplot(gs_e[0, 0])
    ax_r = fig.add_subplot(gs_e[0, 1])
    inc_mat = sim["incidence_matrix"]
    col_sum = inc_mat.sum(axis=1)
    diag = np.array([inc_mat[t].diagonal() for t in range(T)])
    pi_within = np.where(col_sum > 0, diag / col_sum, np.nan)
    total_mat = inc_mat.sum(axis=(1, 2))
    total_diag = np.array([inc_mat[t].trace() for t in range(T)])
    pi_overall = np.where(total_mat > 0, total_diag / total_mat, np.nan)
    im = ax.imshow(
        pi_within.T,
        aspect="auto",
        origin="upper",
        cmap="RdYlGn",
        vmin=0,
        vmax=1,
        interpolation="nearest",
        extent=[0, T, N + 0.5, 0.5],
    )
    ax.set_xlabel("Day $t$")
    ax.set_ylabel(r"Location $j$")
    ax.set_yticks(range(1, N + 1))
    ax.set_yticklabels([f"L{i}" for i in range(1, N + 1)], fontsize=7)
    ax2 = ax.twinx()
    ax2.spines["right"].set_visible(True)
    ax2.spines["top"].set_visible(False)
    mask_o = ~np.isnan(pi_overall)
    ax2.plot(np.where(mask_o)[0], pi_overall[mask_o], color="#1a237e", lw=1.3, ls="-.", alpha=0.9, label=r"$\bar\pi(t)$")
    ax2.set_ylim(0, 1.0)
    ax2.set_ylabel(r"Overall $\bar\pi(t)$", color="#1a237e", fontsize=8)
    ax2.tick_params(axis="y", labelcolor="#1a237e", labelsize=7)
    yfmt2 = ScalarFormatter(useOffset=False)
    yfmt2.set_scientific(False)
    ax2.yaxis.set_major_formatter(yfmt2)
    # π̄(t) legend inset INSIDE the heatmap (upper-right corner, where the dash-dot
    # π̄(t) trace does not run), framed so it reads over the cells.
    ax2.legend(loc="upper right", fontsize=7, frameon=True, facecolor="white",
               framealpha=0.9, edgecolor="0.8", borderpad=0.3, handlelength=1.4)
    # Colorbar in its own inset in the gutter, to the RIGHT of the heatmap's twin
    # y-axis (so it no longer overlaps the "Overall π̄(t)" axis), left of the bar.
    cax = ax.inset_axes([1.26, 0.06, 0.05, 0.88])
    cbar = fig.colorbar(im, cax=cax)
    cbar.ax.set_title(r"$\pi_j(t)$", fontsize=8, pad=4)
    cbar.ax.tick_params(labelsize=7, length=2.5)
    panel_label(ax, "F")
    pi_j_avg = np.nanmean(pi_within, axis=0)
    pi_overall_avg = float(np.nanmean(pi_overall[mask_o])) if mask_o.any() else 0.5
    bar_clrs = [plt.cm.RdYlGn(float(np.clip(v, 0, 1))) for v in pi_j_avg]
    ax_r.barh(range(1, N + 1), pi_j_avg, color=bar_clrs, height=0.72, edgecolor="none")
    ax_r.axvline(pi_overall_avg, color="#1a237e", lw=1.1, ls="-.")
    ax_r.set_xlim(0, 1.05)
    ax_r.set_xlabel(r"$\bar\pi_j$", fontsize=8)
    ax_r.set_yticks(range(1, N + 1))
    ax_r.set_yticklabels([f"L{i}" for i in range(1, N + 1)], fontsize=6)

    path = f"{save_prefix}_04_spectral.pdf"
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved {path}")


# ── driver ───────────────────────────────────────────────────────────────────


def run_corrected_figure_suite(
    *,
    out_dir: str = "out_figs/corrected",
    N_LOC: int = 10,
    T: int = 365,
    SEED: int = 42,
):
    os.makedirs(out_dir, exist_ok=True)
    set_modern_style()

    params = dt.COVID_PARAMS
    max_days = params["max_gen_time"]
    gen_time_pmf = dt.discretise_gamma(params["gen_time_mean"], params["gen_time_sd"], max_days)
    infect_profile = gen_time_pmf.copy()
    w_within = w_between = infect_profile
    LW = params["base_contact_rate"]
    LB = params["base_contact_rate"] * 0.30

    print("─── Scenario A (dense urban) ───")
    city_A = dt.generate_city(N_LOC, scenario="lagos", seed=SEED)
    coords_A, pops_A, dists_A, types_A, meta_A = city_A
    f_A, base_fA = dt.generate_mobility(N_LOC, T, pops_A, dists_A, types_A, meta_A, seed=SEED)
    initial_A = np.zeros(N_LOC)
    initial_A[0] = 10
    sim_A = dt.simulate_epidemic_pde(
        T,
        N_LOC,
        pops_A,
        f_A,
        params["prob_transmission_peak"],
        infect_profile,
        max_days,
        params["R0_target"],
        initial_A,
        LW,
        LB,
        w_within,
        w_between,
        birth_rate=3e-5,
        death_rate=3e-5,
        stochastic=False,
        susceptible_depletion=True,
        seed=SEED,
    )

    f_static = np.stack([base_fA] * T, axis=0)
    sim_static = dt.simulate_epidemic_pde(
        T,
        N_LOC,
        pops_A,
        f_static,
        params["prob_transmission_peak"],
        infect_profile,
        max_days,
        params["R0_target"],
        initial_A,
        LW,
        LB,
        w_within,
        w_between,
        birth_rate=3e-5,
        death_rate=3e-5,
        stochastic=False,
        susceptible_depletion=True,
        seed=SEED,
    )

    R_ind_A = dt.estimate_R_independent(sim_A["incidence"], gen_time_pmf, window=7)
    R_ind_s = dt.estimate_R_independent(sim_static["incidence"], gen_time_pmf, window=7)

    pk_A = int(sim_A["incidence"].sum(axis=1).argmax())
    early_A = max(1, pk_A // 3)
    late_A = min(T - 1, pk_A + 30)
    gt_snaps_A = {}
    lw_A, lb_A = sim_A["lambda_within_scaled"], sim_A["lambda_between_scaled"]
    for nm, day in [("early", early_A), ("peak", pk_A), ("late", late_A)]:
        gt_snaps_A[nm] = dt.compute_generation_times(
            f_A[day],
            sim_A["susceptibles"][day],
            pops_A,
            params["prob_transmission_peak"],
            w_within,
            max_days,
            lw_A,
            lb_A,
        )

    pk_s = int(sim_static["incidence"].sum(axis=1).argmax())
    early_s = max(1, pk_s // 3)
    late_s = min(T - 1, pk_s + 30)
    gt_snaps_s = {}
    lw_s, lb_s = sim_static["lambda_within_scaled"], sim_static["lambda_between_scaled"]
    for nm, day in [("early", early_s), ("peak", pk_s), ("late", late_s)]:
        gt_snaps_s[nm] = dt.compute_generation_times(
            f_static[day],
            sim_static["susceptibles"][day],
            pops_A,
            params["prob_transmission_peak"],
            w_within,
            max_days,
            lw_s,
            lb_s,
        )

    city_B = dt.generate_city(N_LOC, scenario="zambia", seed=SEED)
    coords_B, pops_B, dists_B, types_B, meta_B = city_B
    f_B, _ = dt.generate_mobility(N_LOC, T, pops_B, dists_B, types_B, meta_B, seed=SEED)
    initial_B = np.zeros(N_LOC)
    initial_B[0] = 10
    sim_B = dt.simulate_epidemic_pde(
        T,
        N_LOC,
        pops_B,
        f_B,
        params["prob_transmission_peak"],
        infect_profile,
        max_days,
        params["R0_target"],
        initial_B,
        LW,
        LB,
        w_within,
        w_between,
        birth_rate=3e-5,
        death_rate=3e-5,
        stochastic=False,
        susceptible_depletion=True,
        seed=SEED,
    )
    R_t0_B = dt.compute_R_matrix(
        f_B[0],
        pops_B,
        pops_B,
        params["prob_transmission_peak"],
        infect_profile,
        sim_B["lambda_within_scaled"],
        sim_B["lambda_between_scaled"],
    )

    pfx = os.path.join(out_dir, "fig")
    spfx = os.path.join(out_dir, "fig_SI_static")

    print("\n─── Figure 2 (tile heatmaps + 3D surfaces) ───")
    plot_fig02_overview(sim_A, city_A, f_A, gen_time_pmf, max_days, "Dense urban", pfx, surface_3d=False)
    plot_fig02_overview(sim_A, city_A, f_A, gen_time_pmf, max_days, "Dense urban", pfx, surface_3d=True)
    plot_fig02_overview(sim_static, city_A, f_static, gen_time_pmf, max_days, "Dense urban (static mobility)", spfx, surface_3d=False)
    plot_fig02_overview(sim_static, city_A, f_static, gen_time_pmf, max_days, "Dense urban (static mobility)", spfx, surface_3d=True)

    print("\n─── Figure 3 (tile heatmaps + 3D surfaces) ───")
    plot_fig03_taxonomy(sim_A, city_A, R_ind_A, gt_snaps_A, w_within, w_between, "Dense urban", pfx, surface_3d=False)
    plot_fig03_taxonomy(sim_A, city_A, R_ind_A, gt_snaps_A, w_within, w_between, "Dense urban", pfx, surface_3d=True)
    plot_fig03_taxonomy(sim_static, city_A, R_ind_s, gt_snaps_s, w_within, w_between, "Dense urban (static mobility)", spfx, surface_3d=False)
    plot_fig03_taxonomy(sim_static, city_A, R_ind_s, gt_snaps_s, w_within, w_between, "Dense urban (static mobility)", spfx, surface_3d=True)

    print("\n─── Corrected Figure 4 ───")
    plot_fig04_spectral(sim_A, city_A, R_t0_B, "Dense urban", w_within, w_between, max_days, pfx)
    plot_fig04_spectral(sim_static, city_A, R_t0_B, "Dense urban (static mobility)", w_within, w_between, max_days, spfx)

    print("\nDone. Outputs in:", os.path.abspath(out_dir))


if __name__ == "__main__":
    run_corrected_figure_suite()
