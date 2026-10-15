"""mobility_rt.plotting.figures_corrected — manuscript Figures 2, 3 and 5.

Drawn from the same Scenario A simulation (and its static-mobility twin, Figs SI7-SI9)
as every other figure in mobility_rt.scripts.run_all.

Provides:
  plot_fig02_overview  manuscript Fig 2 / SI7, tile-heatmap and 3-D surface variants
  plot_fig03_taxonomy  manuscript Fig 3 / SI8, tile-heatmap and 3-D surface variants
  plot_fig04_spectral  manuscript Fig 5 / SI9 (file name kept for continuity)
  set_modern_style     the Nature-friendly rcParams these three figures assume

set_modern_style mutates the global matplotlib rcParams, so run_corrected_figures
restores the caller's style afterwards; call these functions through it rather
than directly if other figures follow in the same process.
"""
from __future__ import annotations

import os
import numpy as np
import matplotlib
# Backend: mobility_rt.plotting.__init__ already selects Agg, and importing this
# submodule always runs that package __init__ first, so no matplotlib.use() here.
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
import matplotlib.patheffects as pe
from matplotlib.ticker import FormatStrFormatter, FuncFormatter, LogLocator, MaxNLocator, ScalarFormatter
from scipy.stats import spearmanr

import mobility_rt as dt
from mobility_rt.config import DOW_SCALE
from mobility_rt.distributions import renewal_gt
from mobility_rt.error_tables import ESTIMATOR_WINDOW, trailing_mean
from mobility_rt.simulation import phase_days
from mobility_rt.transmissibility import risk_averse_R, within_fraction

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


def panel_label(ax, letter: str, x=-0.12, y=1.04):
    """Bold panel letter just above the top-left corner of the axes, clear of the
    top tick label (the letter's baseline sits above the axes frame)."""
    ax.text(
        x,
        y,
        letter,
        transform=ax.transAxes,
        fontsize=11,
        fontweight="700",
        va="bottom",
        ha="left",
        color="#111111",
        path_effects=[pe.withStroke(linewidth=2.5, foreground="white")],
    )


OK = dt.OKABE_ITO


# ── derived quantities ─────────────────────────────────────────────────────


def activity_location_prevalence(f_jk: np.ndarray, E_pde: np.ndarray) -> np.ndarray:
    """
    Activity-location prevalence: sum_a sum_j f^{jl}(t) E^j(t,a), over the stored ages
    a = 0..max_days-1.

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


def location_elasticity_series(R_mats: np.ndarray):
    """Location elasticity ε^j(t) and the within-location share of total elasticity.

    Returns (eps_loc, within_share):
      eps_loc[t, j]    ε^j(t) = Σ_k ε^{jk}(t) = Σ_k ε^{kj}(t) = v_j v*_j / (vᵀv*)
      within_share[t]  Σ_j ε^{jj}(t), the share of the total elasticity (which sums to
                       one) carried by within-location corridors k = j

    The infector sum ε^j_out and the infectee sum ε^j_in are the same vector for every
    matrix (R v = ℛ v and v*ᵀ R = ℛ v*ᵀ), so they are drawn once, as ε^j.  The
    identity is checked here rather than shown in a figure.
    """
    T, N, _ = R_mats.shape
    eps_loc = np.zeros((T, N))
    within_share = np.zeros(T)
    for t in range(T):
        Em = dt.sensitivity_elasticity(R_mats[t])["elasticity"]
        eps_out, eps_in = Em.sum(axis=1), Em.sum(axis=0)
        if not np.allclose(eps_out, eps_in, rtol=0.0, atol=1e-9):
            raise AssertionError(f"day {t}: infector and infectee elasticities differ "
                                 f"(max |diff| {np.abs(eps_out - eps_in).max():.2e})")
        eps_loc[t] = eps_out
        within_share[t] = float(np.trace(Em))
    return eps_loc, within_share


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
    """Z shape (T, Nloc); cells are centred on each day t.  NaN cells are drawn light grey."""
    Zt = Z.T
    cm = plt.get_cmap(cmap).copy()
    cm.set_bad("0.85")
    kw = dict(aspect="auto", origin="upper", extent=[-0.5, T - 0.5, Nloc - 0.5, -0.5], cmap=cm, interpolation="nearest")
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
    return im


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
    """3D surface: x=day t, y=location, z=Z(t,·).  NaN values are not drawn.

    The surface array Zt = Z.T has one row per location and one column per day, so
    every location is kept (rstride=1) and days are thinned to about 80 (cstride).
    """
    ax = fig.add_subplot(gs_cell, projection="3d")
    TT, LL = np.meshgrid(np.arange(T), np.arange(Nloc))
    Zt = Z.T
    if vmin is None or vmax is None:
        finite = np.asarray(Z, dtype=float)
        finite = finite[np.isfinite(finite)]
        # An all-NaN panel (e.g. T_j undefined on every day of a short run) draws
        # nothing; give it a nominal colour range instead of a NaN percentile.
        zmin, zmax = np.percentile(finite, [2, 98]) if finite.size else (0.0, 1.0)
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
        rstride=1,
        cstride=max(1, T // 80),
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


# ── elasticity panels (Fig 2 I, Fig 3 D) ─────────────────────────────────────

COL_WITHIN = OK[2]    # bluish green: within-location corridors (k = j)
COL_BETWEEN = OK[0]   # orange: between-location corridors (k ≠ j)


def _plot_elasticity_share(ax, within_share: np.ndarray, *, peak_day: int):
    """Stacked shares of the total elasticity: within-location corridors Σ_j ε^{jj}(t)
    at the bottom, between-location corridors 1 − Σ_j ε^{jj}(t) above.  Each day's
    elasticities sum to one, so the two areas fill the unit band exactly."""
    t = np.arange(len(within_share))
    w = np.clip(np.asarray(within_share, dtype=float), 0.0, 1.0)
    ax.fill_between(t, 0.0, w, color=COL_WITHIN, alpha=0.55, lw=0,
                    label=r"Within-location ($k=j$)")
    ax.fill_between(t, w, 1.0, color=COL_BETWEEN, alpha=0.40, lw=0,
                    label=r"Between-location ($k\neq j$)")
    ax.plot(t, w, color=COL_WITHIN, lw=1.1)
    ax.axvline(peak_day, color="#555", ls="--", lw=0.85, label="Incidence peak")
    ax.set_xlim(0, len(w) - 1)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("Day $t$")
    ax.set_ylabel("Share of total elasticity")
    ax.legend(fontsize=7, loc="lower right", frameon=True, facecolor="white",
              framealpha=0.9, edgecolor="0.8", borderpad=0.4, handlelength=1.4)
    ax.grid(False)


def _label_points(ax, x, y, labels, *, fontsize=7, box=(0.075, 0.05), gap=0.014):
    """Label scatter points without overlap.

    Each label first tries four positions next to its point (right-above,
    right-below, left-above, left-below).  If none is free, as happens when many
    points crowd together, it is stacked at the first free level above (then below)
    the point and joined to it by a thin leader line.  A position is free when its
    box stays inside the axes and clears every other point, every label already
    placed and the correlation note in the top-right corner.  Works in axes-fraction
    units, so the axis limits must be set before calling."""
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    fx = (np.asarray(x, float) - x0) / (x1 - x0)
    fy = (np.asarray(y, float) - y0) / (y1 - y0)
    w, h = box
    r = 0.018                                    # clearance around a marker
    placed = [(0.62, 0.90, 1.0, 1.0)]            # reserved: the correlation note

    def clear(b, j):
        bx0, by0, bx1, by1 = b
        if bx0 < 0 or by0 < 0 or bx1 > 1 or by1 > 1:
            return False
        if any(not (bx1 <= c0 or bx0 >= c2 or by1 <= c1 or by0 >= c3)
               for c0, c1, c2, c3 in placed):
            return False
        return not any(bx0 - r < fx[k] < bx1 + r and by0 - r < fy[k] < by1 + r
                       for k in range(len(fx)) if k != j)

    # top points first, then left to right, so stacked labels keep the points' order
    for j in np.lexsort((fx, -fy)):
        px, py = fx[j], fy[j]
        near = [(px + gap, py + 0.6 * gap), (px + gap, py - 0.6 * gap - h),
                (px - gap - w, py + 0.6 * gap), (px - gap - w, py - 0.6 * gap - h)]
        step = h + 0.01
        stacked = ([(px - w / 2, py + 2 * gap + k * step) for k in range(1, 14)]
                   + [(px - w / 2, py - 2 * gap - h - k * step) for k in range(1, 14)])
        choice = next(((cx, cy, False) for cx, cy in near
                       if clear((cx, cy, cx + w, cy + h), j)), None)
        if choice is None:
            choice = next(((cx, cy, True) for cx, cy in stacked
                           if clear((cx, cy, cx + w, cy + h), j)),
                          (near[0][0], near[0][1], False))
        cx, cy, leader = choice
        placed.append((cx, cy, cx + w, cy + h))
        ax.text(cx + (w / 2 if leader else 0.0), cy, labels[j], transform=ax.transAxes,
                fontsize=fontsize, color="#222", ha="center" if leader else "left",
                va="bottom")
        if leader:
            end_y = cy if cy > py else cy + h
            ax.plot([px, cx + w / 2], [py, end_y], transform=ax.transAxes,
                    color="0.6", lw=0.5, zorder=2)


def _plot_elasticity_vs_Rout(ax, mean_Rout: np.ndarray, mean_eps: np.ndarray):
    """Each location's time-averaged location elasticity ε^j against its
    time-averaged outward reproduction number R^j_out, with the Spearman rank
    correlation between the two across locations."""
    N = len(mean_eps)
    ax.scatter(mean_Rout, mean_eps, s=46, color=OK[4], edgecolors="white",
               linewidths=0.8, zorder=3)
    rho_s = float(spearmanr(mean_Rout, mean_eps).statistic)   # average ranks for ties
    ax.text(0.97, 0.97, rf"Spearman $\rho_s = {rho_s:+.2f}$", transform=ax.transAxes,
            fontsize=7, ha="right", va="top", color="#333")
    xpad = 0.08 * (mean_Rout.max() - mean_Rout.min() + 1e-12)
    ax.set_xlim(mean_Rout.min() - xpad, mean_Rout.max() + 2 * xpad)
    ax.set_ylim(0.0, mean_eps.max() * 1.18)
    _label_points(ax, mean_Rout, mean_eps, [f"L{j + 1}" for j in range(N)])
    ax.set_xlabel(r"Time-averaged $R^j_{\mathrm{out}}$")
    ax.set_ylabel(r"Time-averaged $\varepsilon^j$")
    ax.grid(False)
    return rho_s


# ── Figure 2 (3 × 3) ────────────────────────────────────────────────────────


def plot_fig02_overview(
    sim: dict,
    city_data: tuple,
    f_jk: np.ndarray,
    gen_time_pmf: np.ndarray,
    max_days: int,
    save_prefix: str,
    *,
    surface_3d: bool = False,
):
    """Fig 2 (time-varying mobility) / Fig SI7 (static mobility)."""
    coords, pops, _, node_types, meta = city_data
    inc = sim["incidence"]
    R_mats = sim["R_matrices"]
    S_series = sim["S_start"]
    E_pde = sim["E_pde_state"]
    T, N = inc.shape
    loc = [f"L{i+1}" for i in range(N)]

    S_eff_s, _ = dt.compute_effective_populations_series(
        f_jk, S_series, inc, gen_time_pmf, max_days
    )
    prev_act = activity_location_prevalence(f_jk, E_pde)
    R_sys = np.array([dt.R_system(R_mats[t]) for t in range(T)])
    eps_loc, within_share = location_elasticity_series(R_mats)
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

    # F — S^l_eff at the start of day t (the S that R(t) uses)
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
    E_t = risk_averse_R(R_out_series)
    vE = np.isfinite(E_t) & (E_t > 0)
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

    # H — location elasticity ε^j(t) (always tile)
    vmax_e = np.percentile(eps_loc[eps_loc > 0], 97) if (eps_loc > 0).any() else 1.0
    ax = fig.add_subplot(gs[2, 1])
    _plot_time_location_tile(
        ax,
        eps_loc,
        T,
        N,
        cbar_title=r"$\varepsilon^j(t)$",
        letter="H",
        cmap="YlOrRd",
        y_label=r"Location $j$",
        vmin=0,
        vmax=vmax_e,
    )

    # I — share of the total elasticity in within-location corridors (k = j) and in
    # between-location corridors (k ≠ j); the two add up to one on every day
    ax = fig.add_subplot(gs[2, 2])
    _plot_elasticity_share(ax, within_share, peak_day=int(inc.sum(axis=1).argmax()))
    panel_label(ax, "I")
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
    gen_time_pmf: np.ndarray,
    save_prefix: str,
    *,
    surface_3d: bool = False,
):
    """Fig 3 (time-varying mobility) / Fig SI8 (static mobility)."""
    inc = sim["incidence"]
    inc_mat = sim["incidence_matrix"]
    R_mats = sim["R_matrices"]
    R_meet_s = sim["R_meeting_series"]
    T, N = inc.shape
    loc = [f"L{i+1}" for i in range(N)]

    R_out_s = np.array([dt.R_outward(R_mats[t]) for t in range(T)])
    R_in_s = np.array([dt.R_inward(R_mats[t]) for t in range(T)])
    peak = int(inc.sum(axis=1).argmax())

    i_hub, i_mid, i_per, show_locs, show_lbls = dt.representative_locs(city_data)

    gt_p = gt_snaps["peak"]
    g_pw, g_out, g_in = gt_p["g_pairwise"], gt_p["g_outward"], gt_p["g_inward"]
    days = np.arange(len(gen_time_pmf))
    g_univ = renewal_gt(gen_time_pmf)   # p/Σ_{a≥1} p, the shape every GT reduces to

    eps_loc, _ = location_elasticity_series(R_mats)
    mean_eps = np.nanmean(eps_loc, axis=0)
    mean_Rout = np.nanmean(R_out_s, axis=0)
    v_t, vstar_t = eigenvector_timeseries(R_mats)

    # T_j(t); NaN where undefined (ρ(R_JJ) >= 1), and kept NaN when drawn
    T_ser = np.array([dt.type_reproduction_numbers(R_mats[t_idx]) for t_idx in range(T)])

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
    ax.plot(days, g_univ, color="#555", lw=1.5, ls="-", alpha=0.45, label=r"$p/\sum_{a\geq 1}p$")
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

    # D — where control matters (time-averaged location elasticity ε^j) against
    # where residents transmit most (time-averaged R^j_out)
    ax = fig.add_subplot(gs[0, 3])
    _plot_elasticity_vs_Rout(ax, mean_Rout, mean_eps)
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

    # I, J — R̂_ind (a sliding-window estimate) against the matching trailing means of
    # R^j_in and R^k_out
    win_lbl = f"{ESTIMATOR_WINDOW}-day trailing mean"
    for letter, cell, series, ref_tex in (
        ("I", gs[2, 0], R_in_s, r"\bar{R}^j_{\mathrm{in}}"),
        ("J", gs[2, 1], R_out_s, r"\bar{R}^k_{\mathrm{out}}"),
    ):
        ax = fig.add_subplot(cell)
        for j, lbl, col in zip(show_locs, show_lbls, [OK[0], OK[2], OK[5]]):
            ref_j = trailing_mean(series[:, j], ESTIMATOR_WINDOW)
            vi = np.isfinite(R_independent[:, j])
            vm = np.isfinite(ref_j) & (ref_j > 0)
            b = vi & vm
            if b.sum() > 3:
                ax.plot(np.where(b)[0], R_independent[b, j], "--", color=col, lw=1.0, alpha=0.75)
                ax.plot(np.where(b)[0], ref_j[b], "-", color=col, lw=1.0, label=lbl)
        ax.axhline(1, color="#666", ls="--", lw=0.85)
        h, lab = ax.get_legend_handles_labels()
        h += [
            Line2D([0], [0], color="0.35", lw=1.0, ls="-"),
            Line2D([0], [0], color="0.35", lw=1.0, ls="--"),
        ]
        lab += [rf"${ref_tex}$, {win_lbl} (solid)", r"$\hat R^j_{\mathrm{ind}}$ (dashed)"]
        ax.legend(handles=h, labels=lab, fontsize=6.5, ncol=1)
        ax.set_xlabel("Day $t$")
        ax.set_ylabel(r"$R$")
        panel_label(ax, letter)

    # K — type reproduction numbers T_j(t); grey (tile) or not drawn (surface) where
    # undefined, as the caption states.  T_j diverges as
    # ρ(R_JJ) → 1, so the colour range is taken from the 2nd-98th percentiles of the
    # defined values, as for panels B, C and L.
    T_finite = T_ser[np.isfinite(T_ser)]
    vmin_T, vmax_T = ((float(np.percentile(T_finite, 2)), float(np.percentile(T_finite, 98)))
                      if T_finite.size else (0.0, 1.0))
    if surface_3d:
        _plot_time_location_surface(
            fig,
            gs[2, 2],
            T_ser,
            T,
            N,
            cbar_title=r"$T_j$",
            letter="K",
            cmap="plasma",
            y_label=r"Location $j$",
            vmin=vmin_T,
            vmax=vmax_T,
        )
    else:
        ax = fig.add_subplot(gs[2, 2])
        _plot_time_location_tile(
            ax,
            T_ser,
            T,
            N,
            cbar_title=r"$T_j$",
            letter="K",
            cmap="plasma",
            y_label=r"Location $j$",
            vmin=vmin_T,
            vmax=vmax_T,
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


# ── Figure 5 (file name fig_04_spectral) ───────────────────────────────────


def plot_fig04_spectral(sim: dict, save_prefix: str, *, show_dow: bool = True):
    """Fig 5 (time-varying mobility) / Fig SI9 (static mobility).

    show_dow draws panel A's day-of-week mobility factor on the twin axis (the daily
    lognormal factor ξ_t is not shown).  Set it False for a scenario whose mobility
    series is time-invariant: the static-mobility twin never applies that factor.
    """
    inc = sim["incidence"]
    R_mats = sim["R_matrices"]
    T, N = inc.shape

    R_sys = np.array([dt.R_system(R_mats[t]) for t in range(T)])
    specs = [dt.spectral_analysis(R_mats[t]) for t in range(T)]
    mix_ts = np.array([s["mixing_ratio"] for s in specs])
    sigma_ts = np.array([dt.reactivity(R_mats[t])["sigma"] for t in range(T)])
    phases = phase_days(inc)
    early, peak, late = phases["early"], phases["peak"], phases["late"]

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
    if show_dow:
        dow_pattern = np.array([DOW_SCALE[t % 7] for t in range(T)])
        ax2.plot(range(T), dow_pattern, color=OK[0], lw=1.05, ls="--", alpha=0.42,
                 label="day-of-week factor", zorder=3)
        ax2.set_ylabel(r"Day-of-week factor (no $\xi_t$)", color=OK[0], fontsize=7)
        ax2.tick_params(axis="y", labelcolor=OK[0], labelsize=7)
        # Equally spaced right-hand ticks over the day-of-week range (0.75–1.0)
        ax2.set_ylim(0.7, 1.0)
        ax2.set_yticks([0.70, 0.80, 0.90, 1.00])
    else:
        ax2.set_yticks([])
        ax2.spines["right"].set_visible(False)
    ax.set_xlabel("Day $t$")
    ax.set_ylabel(r"$s(t)=|\lambda_2|/\mathcal{R}$")
    ax.set_ylim(0, 1.05)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=7, loc="lower left", frameon=True,
              facecolor="white", framealpha=0.9, edgecolor="0.8", borderpad=0.4)
    panel_label(ax, "A")

    ax = fig.add_subplot(gs[0, 4:7])
    t_arr = np.arange(T)
    vR = R_sys > 0
    vsig = sigma_ts > 0
    R_out_b = np.array([dt.R_outward(R_mats[t]) for t in range(T)])
    A1_1_ts = R_out_b.max(axis=1)                 # A₁(1) = max_k R^k_out
    E_t_b = risk_averse_R(R_out_b)                # Eq 42
    vEb = np.isfinite(E_t_b) & (E_t_b > 0)
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

    # C — amplification (span 0:2): ℓ2 envelope A(n) solid, ℓ1 envelope A1(n) dotted
    ax = fig.add_subplot(gs[1, 0:2])
    n_max_env = 20
    for t_phase, phase_name, col in [(early, "early", OK[2]), (peak, "peak", OK[5]), (late, "late", OK[0])]:
        env = dt.amplification_envelope(R_mats[t_phase], n_max=n_max_env)
        rho_n = env["rho_n"]
        ax.plot(env["n"], env["A"] / (rho_n + 1e-300), color=col, lw=1.05,
                label=f"$A(n)$ {phase_name}")
        ax.plot(env["n"], env["A1"] / (rho_n + 1e-300), color=col, lw=0.8, ls=":",
                label=f"$\\mathcal{{A}}_1(n)$ {phase_name}")
    ax.axhline(1.0, color="#888", ls="--", lw=0.75)
    ax.set_xlabel("$n$ (generations)")
    ax.set_ylabel(r"Envelope$/\mathcal{R}^n$")
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

    # E — eigenvalue condition number κ (Eq 38), from the core, uncapped
    ax = fig.add_subplot(gs[1, 5:7])
    cond_ts = np.array([dt.sensitivity_elasticity(R_mats[t])["condition_number"] for t in range(T)])

    def _kappa_log_tick(y, _pos):
        if y <= 0:
            return ""
        if y >= 10:
            return f"{int(round(y))}"
        return f"{y:.2f}"

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

    # F — within-location fraction of new infections (twin right axis plain)
    gs_e = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[2, :], width_ratios=[3, 1.05], wspace=0.9)
    ax = fig.add_subplot(gs_e[0, 0])
    ax_r = fig.add_subplot(gs_e[0, 1])
    pi_within, pi_overall = within_fraction(sim["incidence_matrix"])
    im = ax.imshow(
        pi_within.T,
        aspect="auto",
        origin="upper",
        cmap="RdYlGn",
        vmin=0,
        vmax=1,
        interpolation="nearest",
        extent=[-0.5, T - 0.5, N + 0.5, 0.5],   # each cell centred on its day
    )
    ax.set_xlabel("Day $t$")
    ax.set_ylabel(r"Location $j$")
    ax.set_yticks(range(1, N + 1))
    ax.set_yticklabels([f"L{i}" for i in range(1, N + 1)], fontsize=7)
    ax2 = ax.twinx()
    ax2.spines["right"].set_visible(True)
    ax2.spines["top"].set_visible(False)
    mask_o = np.isfinite(pi_overall)
    ax2.plot(np.where(mask_o)[0], pi_overall[mask_o], color="#1a237e", lw=1.3, ls="-.", alpha=0.9, label=r"$\bar\pi(t)$")
    ax2.set_ylim(0, 1.0)
    ax2.set_ylabel(r"Overall $\bar\pi(t)$", color="#1a237e", fontsize=8)
    ax2.tick_params(axis="y", labelcolor="#1a237e", labelsize=7)
    yfmt2 = ScalarFormatter(useOffset=False)
    yfmt2.set_scientific(False)
    ax2.yaxis.set_major_formatter(yfmt2)
    ax2.legend(loc="upper right", fontsize=7, frameon=True, facecolor="white",
               framealpha=0.9, edgecolor="0.8", borderpad=0.3, handlelength=1.4)
    cax = ax.inset_axes([1.26, 0.06, 0.05, 0.88])
    cbar = fig.colorbar(im, cax=cax)
    cbar.ax.set_title(r"$\pi_j(t)$", fontsize=8, pad=4)
    cbar.ax.tick_params(labelsize=7, length=2.5)
    panel_label(ax, "F")
    pi_j_avg = np.nanmean(pi_within, axis=0)
    pi_overall_avg = float(np.mean(pi_overall[mask_o])) if mask_o.any() else np.nan
    bar_clrs = [plt.cm.RdYlGn(float(np.clip(v, 0, 1))) for v in pi_j_avg]
    ax_r.barh(range(1, N + 1), pi_j_avg, color=bar_clrs, height=0.72, edgecolor="none")
    if np.isfinite(pi_overall_avg):
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

def run_corrected_figures(
    sim,
    sim_static,
    city_data,
    f_series,
    f_static,
    R_ind,
    R_ind_static,
    gt_snaps,
    gt_snaps_static,
    gen_time_pmf,
    max_days: int,
    scenario_name: str = "Dense urban",
    out_dir: str = "out_figs/corrected",
    extra_scenarios=None,
):
    """Draw the Fig 2 / Fig 3 / Fig 5 set from ALREADY-simulated inputs.

    Takes the caller's simulations rather than building its own, so these figures
    depict exactly the same Scenario A (and its static-mobility twin) as the rest of
    the pipeline.

    Writes {out_dir}/fig_* for the time-varying run (manuscript Figs 2, 3, 5) and
    {out_dir}/fig_SI_static_* for the static-mobility run (Figs SI7-SI9).  Returns
    the list of prefixes written, in drawing order.

    extra_scenarios draws the SAME three figures for further alternative scenarios —
    the heterogeneous-transmissibility counterparts A-het / B-het / C-het and the
    Scenario B / C baselines they are compared against — so every alternative gets
    the identical panel set rather than a bespoke one.  Each entry is a dict:

        {"sim": ..., "city_data": ..., "f_series": ..., "R_ind": ...,
         "gt_snaps": ..., "name": "Dense urban (A-het)", "prefix": "fig_A_het"}

    with an optional "show_dow" (default True, since every alternative scenario runs
    on a time-varying mobility series), and produces
    {out_dir}/{prefix}_02_overview[_surface].pdf,
    {prefix}_03_taxonomy[_surface].pdf and {prefix}_04_spectral.pdf.  Prefixes must
    be distinct from each other and from the two built-in ones, or one scenario
    would silently overwrite another's figures.

    set_modern_style only sets the rcParams it cares about, so running it on top of
    the pipeline's own style would leave the pipeline's other settings (minor tick
    size, legend handle length, patch line width, figure dpi) in place and change
    these figures.  The body therefore resets rcParams to the values matplotlib
    loaded at import time BEFORE applying set_modern_style, exactly reproducing a
    fresh process, and runs inside an rc_context so the caller's style is restored
    on exit.
    """
    os.makedirs(out_dir, exist_ok=True)

    # One record per scenario, in the order the figures are written.
    scenarios = [
        {"sim": sim, "city_data": city_data, "f_series": f_series, "R_ind": R_ind,
         "gt_snaps": gt_snaps, "name": scenario_name, "prefix": "fig",
         "show_dow": True},
        {"sim": sim_static, "city_data": city_data, "f_series": f_static,
         "R_ind": R_ind_static, "gt_snaps": gt_snaps_static,
         "name": f"{scenario_name} (static mobility)", "prefix": "fig_SI_static",
         "show_dow": False},
    ]
    for extra in (extra_scenarios or []):
        missing = sorted({"sim", "city_data", "f_series", "R_ind", "gt_snaps",
                          "name", "prefix"} - set(extra))
        if missing:
            raise KeyError(f"extra scenario {extra.get('name', '?')!r} is missing "
                           f"key(s): {missing}")
        scenarios.append(dict(extra))

    prefixes = [sc["prefix"] for sc in scenarios]
    dupes = sorted({p for p in prefixes if prefixes.count(p) > 1})
    if dupes:
        raise ValueError(f"duplicate corrected-figure prefix(es) {dupes}: each "
                         f"scenario needs its own, or its figures are overwritten")
    for sc in scenarios:
        sc["path_prefix"] = os.path.join(out_dir, sc["prefix"])

    with plt.rc_context():
        matplotlib.rc_file_defaults()   # discard the caller's style first
        set_modern_style()

        print("\n─── Figure 2 (tile heatmaps + 3D surfaces) ───")
        for surface in (False, True):
            for sc in scenarios:
                print(f"  [{sc['name']}]")
                plot_fig02_overview(sc["sim"], sc["city_data"], sc["f_series"],
                                    gen_time_pmf, max_days, sc["path_prefix"],
                                    surface_3d=surface)

        print("\n─── Figure 3 (tile heatmaps + 3D surfaces) ───")
        for surface in (False, True):
            for sc in scenarios:
                print(f"  [{sc['name']}]")
                plot_fig03_taxonomy(sc["sim"], sc["city_data"], sc["R_ind"],
                                    sc["gt_snaps"], gen_time_pmf, sc["path_prefix"],
                                    surface_3d=surface)

        print("\n─── Figure 5 (spectral) ───")
        for sc in scenarios:
            print(f"  [{sc['name']}]")
            plot_fig04_spectral(sc["sim"], sc["path_prefix"],
                                show_dow=sc.get("show_dow", True))

    print(f"  Figures written to {os.path.abspath(out_dir)}")
    return [sc["path_prefix"] for sc in scenarios]
