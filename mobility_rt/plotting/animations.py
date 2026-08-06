"""mobility_rt.animations."""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mobility_rt.config import OKABE_ITO
from mobility_rt.estimators import estimate_R_independent
from mobility_rt.kernel import R_system
from mobility_rt.plotting.style import _panel_label
from mobility_rt.spectral import spectral_analysis

try:
    from matplotlib.animation import FuncAnimation as _FuncAnimation
    from matplotlib.animation import PillowWriter as _PillowWriter
    from matplotlib.collections import LineCollection as _LineCollection
    from matplotlib.colors import Normalize as _Normalize
    _ANIM_OK = True
except ImportError:
    _ANIM_OK = False


def _anim_type_colors(node_types, scenario):
    """Return Okabe-Ito ring colours per node matched to node type."""
    if scenario == "lagos":
        cm = {"core":             OKABE_ITO[0], "dense":      OKABE_ITO[1],
              "suburban":         OKABE_ITO[2], "peripheral": OKABE_ITO[3]}
    else:
        cm = {"capital":          OKABE_ITO[0], "peri-capital":     OKABE_ITO[1],
              "urban-industrial": OKABE_ITO[2], "semi-urban":       OKABE_ITO[3],
              "rural":            OKABE_ITO[5], "remote-rural":     OKABE_ITO[4]}
    return [cm.get(t, "#888888") for t in node_types]


def _anim_make_tv_edges(ax, coords, base_f, threshold=0.003, rad=0.18, n_pts=12):
    """Build a curved directed-edge LineCollection seeded from base_f for time-varying updates.

    Each edge is sampled as a quadratic Bézier arc (n_pts points) so that
    opposing j→k and k→j arcs bow in opposite directions — matching the look
    of connectionstyle='arc3,rad=0.18'.  Edge topology is fixed to edges where
    base_f[j,k] >= threshold for frame-to-frame stability.

    Colour (Blues cmap) and linewidth encode flow value each frame via
    lc.set_array(vals) and lc.set_linewidths(widths).

    Returns (lc, pairs, f_max):
      lc      LineCollection added to ax (None if no edges pass threshold)
      pairs   list of (j, k) tuples indexing into lc arrays
      f_max   normalisation denominator for linewidth / colour scaling
    """
    N = base_f.shape[0]
    x, y = coords[:, 0], coords[:, 1]
    off_mask = ~np.eye(N, dtype=bool)
    f_max = float(base_f[off_mask].max()) if base_f[off_mask].max() > 1e-12 else 1.0

    t_arr = np.linspace(0, 1, n_pts)

    segments, pairs, init_vals = [], [], []
    for j in range(N):
        for k in range(N):
            if j == k:
                continue
            w = float(base_f[j, k])
            if w < threshold:
                continue
            # Quadratic Bézier: start → control → end
            # Control point is offset perpendicular to chord by rad × chord_length.
            # j→k bows CCW; k→j (reversed direction) bows the other way naturally.
            p0x, p0y = x[j], y[j]
            p2x, p2y = x[k], y[k]
            dx = p2x - p0x; dy = p2y - p0y
            L  = max(np.sqrt(dx * dx + dy * dy), 1e-9)
            px = -dy / L; py = dx / L          # unit perpendicular (CCW)
            c1x = (p0x + p2x) / 2 + rad * L * px
            c1y = (p0y + p2y) / 2 + rad * L * py
            bx = (1 - t_arr)**2 * p0x + 2*(1-t_arr)*t_arr * c1x + t_arr**2 * p2x
            by = (1 - t_arr)**2 * p0y + 2*(1-t_arr)*t_arr * c1y + t_arr**2 * p2y
            segments.append(np.column_stack([bx, by]))
            pairs.append((j, k))
            init_vals.append(w)

    if not segments:
        return None, [], f_max

    init_vals = np.array(init_vals)
    norm_e = _Normalize(vmin=0, vmax=f_max)
    lc = _LineCollection(segments, cmap="Blues", norm=norm_e,
                          linewidths=0.3 + 2.5 * init_vals / f_max,
                          alpha=0.65, zorder=1)
    lc.set_array(init_vals)
    ax.add_collection(lc)
    return lc, pairs, f_max


def _anim_save(anim, path, fps, dpi):
    """Save FuncAnimation as a looping GIF; print status or any error."""
    try:
        writer = _PillowWriter(fps=fps, metadata={"loop": 0})
        anim.save(path, writer=writer, dpi=dpi)
        print(f"  Saved: {path}")
    except Exception as exc:
        print(f"  [GIF save failed for {path}: {exc}]")


def _precompute_anim_data(sim, pops, gen_time_pmf):
    """Precompute all time-varying animation quantities in one pass.

    Returns a dict with:
      rho_t        (T,)      Network reproduction number ρ(R(t))
      sigma_t      (T,)      Reactivity σ(t) = ‖R(t)‖₂
      R_out_t      (T, N)    Outward reproduction numbers R^j_out(t)
      eigvec_t     (T, N)    Stable distribution v*_j(t) = left eigvec of code's R_mat
                             = right eigvec of manuscript's R (sum-normalised)
      elast_node_t (T, N)    Per-location elasticity ε_j(t) = v*_j v_j / (v*^T v)
      elast_edge_t (T, N, N) Edge elasticity E_{kj}(t) = R_{kj} v*_k v_j/(ρ v*^T v)
      E_t          (T,)      Risk-weighted R: E(t) = X(1,t) = Σ(R_out²)/Σ R_out
      agg_R_t      (T,)      Aggregate naive independent R̂ on total incidence (window=7)
      imp_frac_t   (T, N)    Imported fraction at j: Σ_{k≠j} Λ_{kj} / Λ_j
      cum_inc_t    (T, N)    Cumulative attack rate (clipped to [0,1])
    """
    R_mats  = sim["R_matrices"]        # (T, N, N)
    inc     = sim["incidence"]         # (T, N)
    inc_mat = sim["incidence_matrix"]  # (T, N, N)
    T, N    = R_mats.shape[:2]

    rho_t        = np.zeros(T)
    sigma_t      = np.zeros(T)
    R_out_t      = np.zeros((T, N))
    eigvec_t     = np.zeros((T, N))
    elast_node_t = np.zeros((T, N))
    elast_edge_t = np.zeros((T, N, N))
    E_t          = np.zeros(T)

    for t in range(T):
        Rm = R_mats[t]
        rho_t[t]   = R_system(Rm)
        sigma_t[t] = float(np.linalg.svd(Rm, compute_uv=False)[0])
        Rout       = Rm.sum(axis=1)
        R_out_t[t] = Rout
        s = float(Rout.sum())
        E_t[t] = float((Rout ** 2).sum() / s) if s > 1e-10 else 0.0

        sa   = spectral_analysis(Rm)
        v, w = sa["left_eigvec"], sa["right_eigvec"]
        eigvec_t[t] = v
        vw = max(float(v @ w), 1e-300)
        elast_node_t[t] = v * w / vw
        rho = rho_t[t]
        if rho > 1e-10:
            elast_edge_t[t] = Rm * np.outer(v, w) / (rho * vw)

    # Cumulative attack rate
    cum_inc_t = np.clip(np.cumsum(inc, axis=0) / pops[np.newaxis, :], 0.0, 1.0)

    # Aggregate naive independent R on total incidence
    inc_total = inc.sum(axis=1, keepdims=True)              # (T, 1)
    agg_R_t   = estimate_R_independent(inc_total, gen_time_pmf, window=7)[:, 0]

    # Imported fraction  (off-diagonal incidence_matrix / total incidence)
    imp_frac_t = np.zeros((T, N))
    for t in range(T):
        for j in range(N):
            inc_j = float(inc[t, j])
            if inc_j > 1e-10:
                imp_frac_t[t, j] = float(
                    inc_mat[t, :, j].sum() - inc_mat[t, j, j]) / inc_j
    imp_frac_t = np.clip(imp_frac_t, 0.0, 1.0)

    return dict(rho_t=rho_t, sigma_t=sigma_t, R_out_t=R_out_t,
                eigvec_t=eigvec_t, elast_node_t=elast_node_t,
                elast_edge_t=elast_edge_t, E_t=E_t, agg_R_t=agg_R_t,
                imp_frac_t=imp_frac_t, cum_inc_t=cum_inc_t)


def plot_animation_network_metric(
        sim_A, sim_B, city_A, city_B, f_A, f_B,
        data_A, data_B, metric="incidence",
        save_prefix="fig", step=3, fps=10, dpi=100):
    """
    Animated GIF: per-node epidemic metric across the geographic network (A & B).

    One animation per metric; call four times for full coverage.

    All metrics use the perceptually-uniform, colorblind-safe "plasma" colormap
    (see _MCFG) for a consistent look across the animation set:
    metric="incidence"      Node fill = current daily incidence rate (inc/pop).
    metric="cum_incidence"  Node fill = cumulative attack rate [0–1].
    metric="r_out"          Node fill = R^j_out(t).
    metric="eigvec"         Node fill = stable distribution v*_j(t)
                            (right eigvec of manuscript's R = left eigvec of code's R_mat).

    Layout — 2×2 (Scenario A left, Scenario B right):
      Row 0: Network graph.  Node fill = chosen metric.  Ring colour = node type.
             Fixed node size.  Static grey arrows = base mobility flows.
      Row 1: Per-location incidence time series growing to current day.

    Output: {save_prefix}_anim_01{a|b|c|d}_network_{metric}.gif
    """
    if not _ANIM_OK:
        print("  [animation skipped: matplotlib.animation unavailable]")
        return

    _MCFG = {
        "incidence":     ("a", "plasma",   "Current incidence rate",
                          "Current daily incidence rate"),
        "cum_incidence": ("b", "plasma",   "Cumulative attack rate",
                          "Cumulative attack rate"),
        "r_out":         ("c", "plasma",
                          r"$R^j_{\mathrm{out}}(t)$",
                          r"Outward reproduction number $R^j_{\mathrm{out}}(t)$"),
        "eigvec":        ("d", "plasma",
                          r"Stable distribution $v^*_j(t)$",
                          r"Stable distribution $v^*_j(t)$ (eigenvector centrality)"),
    }
    if metric not in _MCFG:
        raise ValueError(f"Unknown metric {metric!r}")
    suffix, cmap_name, cb_label, metric_title = _MCFG[metric]

    coords_A, pops_A, _, types_A, meta_A = city_A
    coords_B, pops_B, _, types_B, meta_B = city_B
    inc_A = sim_A["incidence"]
    inc_B = sim_B["incidence"]
    T, N  = inc_A.shape
    days  = np.arange(T)
    frame_list = list(range(0, T, step))

    rho_A = data_A["rho_t"]
    rho_B = data_B["rho_t"]

    # ── Metric arrays and colour range ────────────────────────────────────────
    if metric == "incidence":
        vals_A = inc_A / pops_A[np.newaxis, :]
        vals_B = inc_B / pops_B[np.newaxis, :]
        vmin   = 0.0
        vmax   = float(max(vals_A.max(), vals_B.max()))
    elif metric == "cum_incidence":
        vals_A = data_A["cum_inc_t"]
        vals_B = data_B["cum_inc_t"]
        vmin, vmax = 0.0, 1.0
    elif metric == "r_out":
        vals_A = data_A["R_out_t"]
        vals_B = data_B["R_out_t"]
        vmin   = 0.0
        vmax   = float(max(vals_A.max(), vals_B.max()))
    else:   # eigvec
        vals_A = data_A["eigvec_t"]
        vals_B = data_B["eigvec_t"]
        vmin, vmax = 0.0, 1.0

    # ── Figure ────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(11.5, 7.5), dpi=dpi)
    fig.patch.set_facecolor("white")
    gs = gridspec.GridSpec(2, 2, figure=fig,
                           hspace=0.42, wspace=0.22,
                           height_ratios=[1.45, 1.0],
                           left=0.06, right=0.97, top=0.93, bottom=0.08)
    ax_nA = fig.add_subplot(gs[0, 0])
    ax_nB = fig.add_subplot(gs[0, 1])
    ax_tA = fig.add_subplot(gs[1, 0])
    ax_tB = fig.add_subplot(gs[1, 1])

    base_fA = f_A.mean(axis=0)
    base_fB = f_B.mean(axis=0)
    lc_nA, pairs_nA, fm_nA = _anim_make_tv_edges(ax_nA, coords_A, base_fA,
                                                   threshold=0.005)
    lc_nB, pairs_nB, fm_nB = _anim_make_tv_edges(ax_nB, coords_B, base_fB,
                                                   threshold=0.001)

    # Ring colour, fill, and size all encode the metric (triple visual encoding)
    _cmap_fn = plt.cm.plasma
    _vrange  = max(vmax - vmin, 1e-10)

    def _norm_vals(v):
        return np.clip((np.asarray(v, dtype=float) - vmin) / _vrange, 0, 1)

    def _ring_rgba(normed):
        rgba = np.array([_cmap_fn(float(x)) for x in normed])
        rgba[:, :3] *= 0.55
        return rgba

    n0_A = _norm_vals(vals_A[0]); n0_B = _norm_vals(vals_B[0])
    sc_A = ax_nA.scatter(coords_A[:, 0], coords_A[:, 1],
                          s=15 + 265 * n0_A, c=vals_A[0], cmap=cmap_name,
                          vmin=vmin, vmax=vmax,
                          edgecolors=_ring_rgba(n0_A), linewidths=2.2, zorder=3)
    sc_B = ax_nB.scatter(coords_B[:, 0], coords_B[:, 1],
                          s=15 + 265 * n0_B, c=vals_B[0], cmap=cmap_name,
                          vmin=vmin, vmax=vmax,
                          edgecolors=_ring_rgba(n0_B), linewidths=2.2, zorder=3)

    for j in range(N):
        ax_nA.text(coords_A[j, 0], coords_A[j, 1], f"{j + 1}",
                   ha="center", va="center", fontsize=4.5,
                   color="white", fontweight="bold", zorder=5)
        ax_nB.text(coords_B[j, 0], coords_B[j, 1], f"{j + 1}",
                   ha="center", va="center", fontsize=4.5,
                   color="white", fontweight="bold", zorder=5)

    for sc, ax in [(sc_A, ax_nA), (sc_B, ax_nB)]:
        cb = fig.colorbar(sc, ax=ax, fraction=0.035, pad=0.03, shrink=0.75)
        cb.set_label(cb_label, fontsize=5.5)
        cb.ax.tick_params(labelsize=5)

    for ax, ttl in [(ax_nA, "Scenario A — Dense urban"),
                    (ax_nB, "Scenario B — Sparse national")]:
        ax.set_title(ttl, fontsize=8, fontweight="bold", pad=4)
        ax.set_aspect("equal", adjustable="datalim")
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)

    day_txt_A = ax_nA.text(0.02, 0.97, "Day 1", transform=ax_nA.transAxes,
                            fontsize=6.5, va="top", ha="left",
                            bbox=dict(fc="white", ec="none", alpha=0.8, pad=1.5))
    day_txt_B = ax_nB.text(0.02, 0.97, "Day 1", transform=ax_nB.transAxes,
                            fontsize=6.5, va="top", ha="left",
                            bbox=dict(fc="white", ec="none", alpha=0.8, pad=1.5))

    # ── Incidence time series (context; consistent across all metric variants) ─
    loc_cmap = plt.cm.tab10
    loc_cols = [loc_cmap(j / max(N - 1, 1)) for j in range(N)]

    for ax, inc in [(ax_tA, inc_A), (ax_tB, inc_B)]:
        for j in range(N):
            ax.plot(days, inc[:, j] / 1e3, color=loc_cols[j],
                    lw=0.35, alpha=0.18, zorder=1)
        ax.set_xlim(0, T - 1)
        ax.set_ylim(0, max(float(inc.max()) / 1e3 * 1.10, 0.01))
        ax.set_xlabel("Day", fontsize=7)
        ax.set_ylabel("Daily incidence (×10³)", fontsize=7)
        ax.tick_params(labelsize=6)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    ax_tA.set_title("Per-location incidence — Scenario A", fontsize=7, pad=2)
    ax_tB.set_title("Per-location incidence — Scenario B", fontsize=7, pad=2)

    lines_A = [ax_tA.plot([], [], color=loc_cols[j], lw=0.85, zorder=2)[0]
               for j in range(N)]
    lines_B = [ax_tB.plot([], [], color=loc_cols[j], lw=0.85, zorder=2)[0]
               for j in range(N)]

    vl_A = ax_tA.axvline(0, color="#333333", lw=0.8, ls="--", zorder=3)
    vl_B = ax_tB.axvline(0, color="#333333", lw=0.8, ls="--", zorder=3)
    rho_ann_A = ax_tA.text(0.98, 0.97, "", transform=ax_tA.transAxes,
                            fontsize=6, va="top", ha="right", color=OKABE_ITO[4])
    rho_ann_B = ax_tB.text(0.98, 0.97, "", transform=ax_tB.transAxes,
                            fontsize=6, va="top", ha="right", color=OKABE_ITO[4])

    _panel_label(ax_nA, "A"); _panel_label(ax_nB, "B")
    _panel_label(ax_tA, "C"); _panel_label(ax_tB, "D")
    fig.suptitle(f"Network epidemic — {metric_title}",
                  fontsize=9, fontweight="bold", y=0.98)

    def _update_nm(t):
        # Time-varying mobility edges
        for lc, pairs, f_max, f_tv in [
                (lc_nA, pairs_nA, fm_nA, f_A),
                (lc_nB, pairs_nB, fm_nB, f_B)]:
            if lc is not None and pairs:
                ev = np.array([float(f_tv[t, j, k]) for (j, k) in pairs])
                lc.set_array(ev)
                lc.set_linewidths(0.3 + 2.5 * ev / f_max)
        nA = _norm_vals(vals_A[t]); nB = _norm_vals(vals_B[t])
        sc_A.set_array(vals_A[t]);       sc_B.set_array(vals_B[t])
        sc_A.set_edgecolors(_ring_rgba(nA)); sc_B.set_edgecolors(_ring_rgba(nB))
        sc_A.set_sizes(15 + 265 * nA);  sc_B.set_sizes(15 + 265 * nB)
        day_txt_A.set_text(
            f"Day {t + 1}   $\\mathcal{{R}}(t) = {rho_A[t]:.2f}$")
        day_txt_B.set_text(
            f"Day {t + 1}   $\\mathcal{{R}}(t) = {rho_B[t]:.2f}$")
        for j in range(N):
            lines_A[j].set_data(days[:t + 1], inc_A[:t + 1, j] / 1e3)
            lines_B[j].set_data(days[:t + 1], inc_B[:t + 1, j] / 1e3)
        vl_A.set_xdata([t, t]); vl_B.set_xdata([t, t])
        rho_ann_A.set_text(f"$\\mathcal{{R}}(t) = {rho_A[t]:.2f}$")
        rho_ann_B.set_text(f"$\\mathcal{{R}}(t) = {rho_B[t]:.2f}$")

    anim = _FuncAnimation(fig, _update_nm, frames=frame_list,
                           blit=False, interval=1000 // fps)
    _anim_save(anim,
               f"{save_prefix}_anim_01{suffix}_network_{metric}.gif", fps, dpi)
    plt.close(fig)


def plot_animation_elasticity(
        sim_A, sim_B, sim_C, city_A, city_B, city_C,
        base_fA, base_fB, base_fC, data_A, data_B, data_C,
        save_prefix="fig", step=3, fps=10, dpi=100):
    """
    Animated GIF: per-location and between-location elasticity (Scenarios A, B, C).

    Elasticity of the network reproduction number ρ(R(t)) to R-matrix elements:

      Per-location:     ε_j(t) = v_j w_j / (v^T w)
      Between-location: E_{kj}(t) = R_{kj} v_k w_j / (ρ v^T w)

    where v = left eigenvector of R(t) and w = right eigenvector (both
    sum-normalised); ε_j sums to 1 over nodes and E_{kj} sums to 1 over edges.

    Layout — 2 rows × 3 columns (one column per scenario):
      Row 0: Geographic network graph.
             Node fill = ε_j(t) [YlOrRd].
             Directed edges coloured and width-scaled by E_{kj}(t) [YlOrRd
             LineCollection]; colour intensity and thickness both encode magnitude.
      Row 1: Animated bar chart of ε_j(t) per location, coloured by node type.

    Output: {save_prefix}_anim_04_elasticity.gif
    """
    if not _ANIM_OK:
        print("  [animation skipped: matplotlib.animation unavailable]")
        return

    cities  = [city_A, city_B, city_C]
    base_fs = [base_fA, base_fB, base_fC]
    datas   = [data_A, data_B, data_C]
    scen_labels = ["Scenario A — Dense urban",
                   "Scenario B — Sparse national",
                   "Scenario C — Hub–satellite"]
    rho_ts  = [data_A["rho_t"], data_B["rho_t"], data_C["rho_t"]]

    coords_A, pops_A, _, types_A, meta_A = city_A
    T, N = sim_A["incidence"].shape
    frame_list = list(range(0, T, step))

    # Global elasticity range (shared colourbar)
    e_node_max = float(max(d["elast_node_t"].max() for d in datas))
    e_edge_max = float(max(d["elast_edge_t"].max() for d in datas))
    e_node_max = max(e_node_max, 1e-10)
    e_edge_max = max(e_edge_max, 1e-10)

    # ── Figure ────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(15.0, 7.5), dpi=dpi)
    fig.patch.set_facecolor("white")
    gs = gridspec.GridSpec(2, 3, figure=fig,
                           hspace=0.42, wspace=0.30,
                           height_ratios=[1.45, 1.0],
                           left=0.05, right=0.96, top=0.91, bottom=0.09)

    scatters       = []
    bar_containers = []
    day_texts      = []
    line_colls     = []
    edge_pair_lists = []

    for col in range(3):
        coords, pops, _, types, meta = cities[col]
        base_f = base_fs[col]
        data   = datas[col]
        ax_n   = fig.add_subplot(gs[0, col])
        ax_b   = fig.add_subplot(gs[1, col])
        ec     = _anim_type_colors(types, meta["scenario"])
        x, y   = coords[:, 0], coords[:, 1]

        # ── Directed edge LineCollection (k→j, slight perpendicular offset) ──
        off_mask = ~np.eye(N, dtype=bool)
        f_max_col = float(base_f[off_mask].max()) if base_f[off_mask].max() > 1e-12 else 1.0
        thresh_f  = f_max_col * 0.02   # include edges with ≥ 2% of max flow
        segments  = []
        pairs     = []
        for k in range(N):
            for j in range(N):
                if k == j:
                    continue
                if float(base_f[k, j]) < thresh_f:
                    continue
                dx = x[j] - x[k]; dy = y[j] - y[k]
                L  = max(np.sqrt(dx * dx + dy * dy), 1e-9)
                ox = -dy / L * 0.30;  oy = dx / L * 0.30   # perp offset
                segments.append([(x[k] + ox, y[k] + oy),
                                  (x[j] + ox, y[j] + oy)])
                pairs.append((k, j))

        if segments:
            init_vals = np.array([float(data["elast_edge_t"][0][k, j])
                                   for (k, j) in pairs])
            norm_e = _Normalize(vmin=0, vmax=e_edge_max)
            lc = _LineCollection(segments, cmap="plasma", norm=norm_e,
                                  linewidths=0.8 + 4.5 * init_vals / e_edge_max,
                                  alpha=0.80, zorder=2)
            lc.set_array(init_vals)
            ax_n.add_collection(lc)
        else:
            lc = None
        line_colls.append(lc)
        edge_pair_lists.append(pairs)

        # ── Node scatter (fill, ring, size all encode per-location elasticity) ──
        _e0   = data["elast_node_t"][0]
        _n0   = np.clip(_e0 / max(e_node_max, 1e-10), 0, 1)
        _ring0 = np.array([plt.cm.plasma(float(v)) for v in _n0])
        _ring0[:, :3] *= 0.55
        sc = ax_n.scatter(x, y, s=15 + 265 * _n0, c=_e0,
                           cmap="plasma", vmin=0, vmax=e_node_max,
                           edgecolors=_ring0, linewidths=2.2, zorder=3)
        scatters.append(sc)

        for j in range(N):
            ax_n.text(x[j], y[j], f"{j + 1}",
                      ha="center", va="center", fontsize=4.5,
                      color="white", fontweight="bold", zorder=5)

        # Shared colourbar on rightmost column only
        if col == 2:
            cb = fig.colorbar(sc, ax=ax_n, fraction=0.035, pad=0.03, shrink=0.75)
            cb.set_label(
                r"$\varepsilon_j(t) = v_j w_j\,/\,(\mathbf{v}^\top\mathbf{w})$",
                fontsize=5.5)
            cb.ax.tick_params(labelsize=5)

        ax_n.set_title(scen_labels[col], fontsize=8, fontweight="bold", pad=4)
        ax_n.set_aspect("equal", adjustable="datalim")
        ax_n.set_xticks([]); ax_n.set_yticks([])
        for sp in ax_n.spines.values():
            sp.set_visible(False)

        dt = ax_n.text(0.02, 0.97, "Day 1", transform=ax_n.transAxes,
                        fontsize=6.5, va="top", ha="left",
                        bbox=dict(fc="white", ec="none", alpha=0.8, pad=1.5))
        day_texts.append(dt)

        # ── Bar chart (per-location elasticity) ───────────────────────────────
        bar_x = np.arange(N)
        bars  = ax_b.bar(bar_x, data["elast_node_t"][0], color=ec,
                          edgecolor="white", linewidth=0.5, zorder=2)
        bar_containers.append(bars)
        ax_b.set_ylim(0, e_node_max * 1.18)
        ax_b.set_xticks(bar_x)
        ax_b.set_xticklabels([f"L{j + 1}" for j in range(N)],
                              fontsize=5.5, rotation=45, ha="right")
        ax_b.set_xlabel("Location $j$", fontsize=7)
        ax_b.set_ylabel(r"$\varepsilon_j(t)$", fontsize=7)
        ax_b.set_title(
            f"Per-location elasticity — {scen_labels[col].split('—')[0].strip()}",
            fontsize=7, pad=2)
        ax_b.tick_params(labelsize=6)
        ax_b.spines["top"].set_visible(False)
        ax_b.spines["right"].set_visible(False)

    for col, letter in enumerate(["A", "B", "C"]):
        _panel_label(fig.axes[col * 2], letter)      # network axes
    for col, letter in enumerate(["D", "E", "F"]):
        _panel_label(fig.axes[col * 2 + 1], letter)  # bar axes

    fig.suptitle(
        r"Elasticity of $\mathcal{R}(t)$: "
        r"node $\varepsilon_j = v_j w_j/(\mathbf{v}^\top\mathbf{w})$, "
        r"edge $E_{kj} = R_{kj} v_k w_j / (\rho\,\mathbf{v}^\top\mathbf{w})$",
        fontsize=8.5, fontweight="bold", y=0.98)

    def _update_elast(t):
        for col in range(3):
            en   = datas[col]["elast_node_t"][t]
            ee   = datas[col]["elast_edge_t"][t]
            en_n = np.clip(en / max(e_node_max, 1e-10), 0, 1)
            scatters[col].set_array(en)
            ring_c = np.array([plt.cm.plasma(float(v)) for v in en_n])
            ring_c[:, :3] *= 0.55
            scatters[col].set_edgecolors(ring_c)
            scatters[col].set_sizes(15 + 265 * en_n)
            lc = line_colls[col]
            if lc is not None and edge_pair_lists[col]:
                e_vals = np.array([float(ee[k, j])
                                   for (k, j) in edge_pair_lists[col]])
                lc.set_array(e_vals)
                lc.set_linewidths(0.8 + 4.5 * e_vals / e_edge_max)
            day_texts[col].set_text(
                f"Day {t + 1}   $\\mathcal{{R}}(t) = {rho_ts[col][t]:.2f}$")
            for bar, h in zip(bar_containers[col], en):
                bar.set_height(float(h))

    anim = _FuncAnimation(fig, _update_elast, frames=frame_list,
                           blit=False, interval=1000 // fps)
    _anim_save(anim, f"{save_prefix}_anim_04_elasticity.gif", fps, dpi)
    plt.close(fig)


def plot_animation_R_heatmap(sim_A, city_A, data_A, gen_time_pmf,
                              save_prefix="fig", step=3, fps=10, dpi=100):
    """
    Animated GIF: R-matrix heatmap, reproduction numbers, R_out, imported fraction.

    2×2 layout:
      A (top-left):  Heatmap of R_{kj}(t) [YlOrRd].  Colour scale fixed to
                     time-maximum for frame-to-frame comparability.
      B (top-right): Network-level ρ(t) = spectral radius, risk-weighted
                     E(t) = X(1,t), and aggregate naive independent R̂_agg(t)
                     applied to total incidence.  Animated scan line tracks current day.
      C (bottom-left):  Animated bar chart of R^j_out(t) per location, with
                        dashed line showing ρ(t).
      D (bottom-right): Animated bar chart of imported fraction per location —
                        the proportion of j's new infections attributed to
                        infectors from other home locations.

    Output: {save_prefix}_anim_02_R_heatmap.gif
    """
    if not _ANIM_OK:
        print("  [animation skipped: matplotlib.animation unavailable]")
        return

    coords_A, pops_A, _, types_A, meta_A = city_A
    R_mats = sim_A["R_matrices"]       # (T, N, N)
    T, N   = R_mats.shape[:2]
    days   = np.arange(T)
    frame_list = list(range(0, T, step))
    ec = _anim_type_colors(types_A, meta_A["scenario"])

    rho_t      = data_A["rho_t"]
    R_out_t    = data_A["R_out_t"]
    E_t        = data_A["E_t"]
    agg_R_t    = data_A["agg_R_t"]
    imp_frac_t = data_A["imp_frac_t"]

    # ── Figure ────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(12.5, 9.0), dpi=dpi)
    fig.patch.set_facecolor("white")
    gs = gridspec.GridSpec(2, 2, figure=fig,
                           hspace=0.48, wspace=0.42,
                           left=0.07, right=0.97, top=0.91, bottom=0.10)
    ax_heat = fig.add_subplot(gs[0, 0])
    ax_spec = fig.add_subplot(gs[0, 1])
    ax_bar  = fig.add_subplot(gs[1, 0])
    ax_imp  = fig.add_subplot(gs[1, 1])

    # ── Panel A: R-matrix heatmap ─────────────────────────────────────────────
    vmax_R = float(R_mats.max())
    im = ax_heat.imshow(R_mats[0], vmin=0, vmax=vmax_R,
                        cmap="YlOrRd", aspect="auto", interpolation="nearest")
    cb_h = fig.colorbar(im, ax=ax_heat, fraction=0.046, pad=0.04)
    cb_h.set_label(r"$R_{kj}(t)$", fontsize=7)
    cb_h.ax.tick_params(labelsize=5.5)
    ax_heat.set_xticks(range(N))
    ax_heat.set_xticklabels([f"L{j + 1}" for j in range(N)], fontsize=5)
    ax_heat.set_yticks(range(N))
    ax_heat.set_yticklabels([f"L{k + 1}" for k in range(N)], fontsize=5)
    ax_heat.set_xlabel("Infectee $j$", fontsize=7)
    ax_heat.set_ylabel("Infector $k$", fontsize=7)
    ax_heat.set_title(r"Next-generation matrix $R_{kj}(t)$", fontsize=8, pad=4)
    heat_day = ax_heat.text(0.97, 0.03, "Day 1",
                             transform=ax_heat.transAxes, fontsize=6,
                             va="bottom", ha="right", color="#333333",
                             bbox=dict(fc="white", ec="none", alpha=0.8, pad=1))

    # ── Panel B: ρ(t), E(t), R̂_agg(t) ────────────────────────────────────────
    valid_agg = ~np.isnan(agg_R_t)
    ax_spec.plot(days, rho_t, color=OKABE_ITO[0], lw=1.3,
                  label=r"$\mathcal{R}(t) = \rho(\mathbf{R})$")
    ax_spec.plot(days, E_t,   color=OKABE_ITO[6], lw=1.3, ls="--",
                  label=r"$\mathcal{E}(t) = X(1,t)$")
    ax_spec.plot(np.where(valid_agg)[0], agg_R_t[valid_agg],
                  color=OKABE_ITO[4], lw=1.0, ls=":",
                  label=r"$R_{\mathrm{ind}}(t)$  [naive independent]")
    ax_spec.axhline(1.0, color="#888888", lw=0.7, ls=":", zorder=0)
    ymax_spec = max(float(np.nanmax(agg_R_t[valid_agg])) if valid_agg.any() else 0,
                    float(rho_t.max()), float(E_t.max())) * 1.12
    ax_spec.set_xlim(0, T - 1)
    ax_spec.set_ylim(0, ymax_spec)
    ax_spec.set_xlabel("Day", fontsize=7)
    ax_spec.set_ylabel("Reproduction number", fontsize=7)
    ax_spec.set_title(
        r"$\mathcal{R}(t)$, risk-weighted $\mathcal{E}(t)$, "
        r"naive $\hat{R}_{\mathrm{agg}}(t)$",
        fontsize=7.5, pad=4)
    ax_spec.legend(fontsize=6, loc="upper right")
    ax_spec.tick_params(labelsize=6)
    ax_spec.spines["top"].set_visible(False)
    ax_spec.spines["right"].set_visible(False)
    vl_spec = ax_spec.axvline(0, color="#222222", lw=1.0, ls="-", zorder=5)

    # ── Panel C: R_out animated bars ──────────────────────────────────────────
    bar_x   = np.arange(N)
    bars_C  = ax_bar.bar(bar_x, R_out_t[0], color=ec,
                          edgecolor="white", linewidth=0.5, zorder=2)
    ymax_C  = float(R_out_t.max()) * 1.15
    ax_bar.set_ylim(0, ymax_C)
    rho_ln  = ax_bar.axhline(float(rho_t[0]), color="#333333", lw=0.9, ls="--",
                              zorder=3, label=r"$\mathcal{R}(t)$")
    ax_bar.set_xticks(bar_x)
    ax_bar.set_xticklabels([f"L{j + 1}" for j in range(N)],
                            fontsize=5.5, rotation=45, ha="right")
    ax_bar.set_xlabel("Location $j$", fontsize=7)
    ax_bar.set_ylabel(r"$R^j_{\mathrm{out}}(t)$", fontsize=7)
    ax_bar.set_title("Outward reproduction numbers", fontsize=8, pad=4)
    ax_bar.tick_params(labelsize=6)
    ax_bar.spines["top"].set_visible(False)
    ax_bar.spines["right"].set_visible(False)
    ax_bar.legend(fontsize=6, loc="upper right")

    # ── Panel D: Imported fraction animated bars ───────────────────────────────
    bars_D  = ax_imp.bar(bar_x, imp_frac_t[0], color=ec,
                          edgecolor="white", linewidth=0.5, zorder=2)
    ax_imp.set_ylim(0, 1.05)
    ax_imp.axhline(0.5, color="#aaaaaa", lw=0.6, ls=":", zorder=0)
    ax_imp.set_xticks(bar_x)
    ax_imp.set_xticklabels([f"L{j + 1}" for j in range(N)],
                            fontsize=5.5, rotation=45, ha="right")
    ax_imp.set_xlabel("Location $j$", fontsize=7)
    ax_imp.set_ylabel("Imported fraction", fontsize=7)
    ax_imp.set_title(
        r"Imported fraction  $\sum_{k \neq j} \Lambda_{kj}(t)\,/\,\Lambda_j(t)$",
        fontsize=7.5, pad=4)
    ax_imp.tick_params(labelsize=6)
    ax_imp.spines["top"].set_visible(False)
    ax_imp.spines["right"].set_visible(False)

    _panel_label(ax_heat, "A"); _panel_label(ax_spec, "B")
    _panel_label(ax_bar,  "C"); _panel_label(ax_imp,  "D")
    fig.suptitle(
        "R-matrix evolution and network-level quantities — Scenario A (Dense urban)",
        fontsize=9, fontweight="bold", y=0.97)

    def _update_heat(t):
        im.set_data(R_mats[t])
        heat_day.set_text(f"Day {t + 1}")
        vl_spec.set_xdata([t, t])
        for bar, h in zip(bars_C, R_out_t[t]):
            bar.set_height(float(h))
        rho_ln.set_ydata([float(rho_t[t]), float(rho_t[t])])
        for bar, h in zip(bars_D, imp_frac_t[t]):
            bar.set_height(float(h))

    anim = _FuncAnimation(fig, _update_heat, frames=frame_list,
                           blit=False, interval=1000 // fps)
    _anim_save(anim, f"{save_prefix}_anim_02_R_heatmap.gif", fps, dpi)
    plt.close(fig)


def plot_animation_scenario_comparison(
        sim_A, sim_C, city_A, city_C, f_A, f_C,
        data_A, data_C,
        save_prefix="fig", step=3, fps=10, dpi=100):
    """
    Animated comparison of Scenario A (symmetric) and Scenario C (hub–satellite).

    Layout — 2×2 grid:
      Row 0: Geographic networks.  Node fill = current daily incidence rate
             (incidence / population), common OrRd colour scale.
             Node ring colour = node type.  Static mobility flow arrows.
      Row 1: Per-location incidence time series with animated day marker
             and ρ(t) annotation, for Scenario A (left) and C (right).

    Output: {save_prefix}_anim_03_scenario_AC.gif
    """
    if not _ANIM_OK:
        print("  [animation skipped: matplotlib.animation unavailable]")
        return

    coords_A, pops_A, _, types_A, meta_A = city_A
    coords_C, pops_C, _, types_C, meta_C = city_C
    inc_A = sim_A["incidence"]
    inc_C = sim_C["incidence"]
    T, N  = inc_A.shape
    days  = np.arange(T)
    frame_list = list(range(0, T, step))

    rho_A = data_A["rho_t"]
    rho_C = data_C["rho_t"]

    # Common incidence-rate colour scale
    inc_rate_A = inc_A / pops_A[np.newaxis, :]
    inc_rate_C = inc_C / pops_C[np.newaxis, :]
    vmax_rate  = float(max(inc_rate_A.max(), inc_rate_C.max()))

    # ── Figure ────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(11.5, 7.5), dpi=dpi)
    fig.patch.set_facecolor("white")
    gs = gridspec.GridSpec(2, 2, figure=fig,
                           hspace=0.42, wspace=0.25,
                           height_ratios=[1.45, 1.0],
                           left=0.07, right=0.97, top=0.93, bottom=0.08)
    ax_nA = fig.add_subplot(gs[0, 0])
    ax_nC = fig.add_subplot(gs[0, 1])
    ax_tA = fig.add_subplot(gs[1, 0])
    ax_tC = fig.add_subplot(gs[1, 1])

    base_fA_sc = f_A.mean(axis=0)
    base_fC_sc = f_C.mean(axis=0)
    lc_scA, pairs_scA, fm_scA = _anim_make_tv_edges(ax_nA, coords_A, base_fA_sc,
                                                      threshold=0.005)
    lc_scC, pairs_scC, fm_scC = _anim_make_tv_edges(ax_nC, coords_C, base_fC_sc,
                                                      threshold=0.005)

    # Ring colour, fill and size all encode incidence rate (plasma, triple encoding)
    _vrange_sc = max(vmax_rate, 1e-10)

    def _nc(v):
        return np.clip(np.asarray(v, dtype=float) / _vrange_sc, 0, 1)

    def _ring_c(normed):
        rgba = np.array([plt.cm.plasma(float(x)) for x in normed])
        rgba[:, :3] *= 0.55
        return rgba

    n0_A = _nc(inc_rate_A[0]); n0_C = _nc(inc_rate_C[0])
    sc_A = ax_nA.scatter(coords_A[:, 0], coords_A[:, 1],
                          s=15 + 265 * n0_A, c=inc_rate_A[0], cmap="plasma",
                          vmin=0, vmax=vmax_rate,
                          edgecolors=_ring_c(n0_A), linewidths=2.2, zorder=3)
    sc_C = ax_nC.scatter(coords_C[:, 0], coords_C[:, 1],
                          s=15 + 265 * n0_C, c=inc_rate_C[0], cmap="plasma",
                          vmin=0, vmax=vmax_rate,
                          edgecolors=_ring_c(n0_C), linewidths=2.2, zorder=3)

    for j in range(N):
        ax_nA.text(coords_A[j, 0], coords_A[j, 1], f"{j + 1}",
                   ha="center", va="center", fontsize=4.5,
                   color="white", fontweight="bold", zorder=5)
        ax_nC.text(coords_C[j, 0], coords_C[j, 1], f"{j + 1}",
                   ha="center", va="center", fontsize=4.5,
                   color="white", fontweight="bold", zorder=5)

    for sc, ax in [(sc_A, ax_nA), (sc_C, ax_nC)]:
        cb = fig.colorbar(sc, ax=ax, fraction=0.035, pad=0.03, shrink=0.75)
        cb.set_label("Daily incidence rate", fontsize=5.5)
        cb.ax.tick_params(labelsize=5)

    for ax, ttl in [(ax_nA, "Scenario A — Symmetric mobility"),
                    (ax_nC, "Scenario C — Hub–satellite mobility")]:
        ax.set_title(ttl, fontsize=8, fontweight="bold", pad=4)
        ax.set_aspect("equal", adjustable="datalim")
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)

    day_txt_A = ax_nA.text(0.02, 0.97, "Day 1", transform=ax_nA.transAxes,
                            fontsize=6.5, va="top", ha="left",
                            bbox=dict(fc="white", ec="none", alpha=0.8, pad=1.5))
    day_txt_C = ax_nC.text(0.02, 0.97, "Day 1", transform=ax_nC.transAxes,
                            fontsize=6.5, va="top", ha="left",
                            bbox=dict(fc="white", ec="none", alpha=0.8, pad=1.5))

    # ── Per-location incidence time series ────────────────────────────────────
    loc_cmap = plt.cm.tab10
    loc_cols = [loc_cmap(j / max(N - 1, 1)) for j in range(N)]

    for ax, inc, lbl in [(ax_tA, inc_A, "Scenario A"),
                          (ax_tC, inc_C, "Scenario C")]:
        for j in range(N):
            ax.plot(days, inc[:, j] / 1e3, color=loc_cols[j],
                    lw=0.35, alpha=0.18, zorder=1)
        ax.set_xlim(0, T - 1)
        ax.set_ylim(0, max(float(inc.max()) / 1e3 * 1.10, 0.01))
        ax.set_xlabel("Day", fontsize=7)
        ax.set_ylabel("Daily incidence (×10³)", fontsize=7)
        ax.tick_params(labelsize=6)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_title(f"Per-location incidence — {lbl}", fontsize=7, pad=2)

    lines_A = [ax_tA.plot([], [], color=loc_cols[j], lw=0.85, zorder=2)[0]
               for j in range(N)]
    lines_C = [ax_tC.plot([], [], color=loc_cols[j], lw=0.85, zorder=2)[0]
               for j in range(N)]

    vl_A = ax_tA.axvline(0, color="#333333", lw=0.8, ls="--", zorder=3)
    vl_C = ax_tC.axvline(0, color="#333333", lw=0.8, ls="--", zorder=3)
    rho_ann_A = ax_tA.text(0.98, 0.97, "", transform=ax_tA.transAxes,
                            fontsize=6, va="top", ha="right", color=OKABE_ITO[4])
    rho_ann_C = ax_tC.text(0.98, 0.97, "", transform=ax_tC.transAxes,
                            fontsize=6, va="top", ha="right", color=OKABE_ITO[4])

    _panel_label(ax_nA, "A"); _panel_label(ax_nC, "B")
    _panel_label(ax_tA, "C"); _panel_label(ax_tC, "D")
    fig.suptitle(
        "Scenario A vs C: symmetric vs hub–satellite mobility\n"
        "(node fill = current daily incidence rate)",
        fontsize=9, fontweight="bold", y=0.98)

    def _update_comp(t):
        # Time-varying mobility edges
        for lc, pairs, f_max, f_tv in [
                (lc_scA, pairs_scA, fm_scA, f_A),
                (lc_scC, pairs_scC, fm_scC, f_C)]:
            if lc is not None and pairs:
                ev = np.array([float(f_tv[t, j, k]) for (j, k) in pairs])
                lc.set_array(ev)
                lc.set_linewidths(0.3 + 2.5 * ev / f_max)
        nA = _nc(inc_rate_A[t]); nC = _nc(inc_rate_C[t])
        sc_A.set_array(inc_rate_A[t]);  sc_C.set_array(inc_rate_C[t])
        sc_A.set_edgecolors(_ring_c(nA)); sc_C.set_edgecolors(_ring_c(nC))
        sc_A.set_sizes(15 + 265 * nA);  sc_C.set_sizes(15 + 265 * nC)
        day_txt_A.set_text(
            f"Day {t + 1}   $\\mathcal{{R}}(t) = {rho_A[t]:.2f}$")
        day_txt_C.set_text(
            f"Day {t + 1}   $\\mathcal{{R}}(t) = {rho_C[t]:.2f}$")
        for j in range(N):
            lines_A[j].set_data(days[:t + 1], inc_A[:t + 1, j] / 1e3)
            lines_C[j].set_data(days[:t + 1], inc_C[:t + 1, j] / 1e3)
        vl_A.set_xdata([t, t]); vl_C.set_xdata([t, t])
        rho_ann_A.set_text(f"$\\mathcal{{R}}(t) = {rho_A[t]:.2f}$")
        rho_ann_C.set_text(f"$\\mathcal{{R}}(t) = {rho_C[t]:.2f}$")

    anim = _FuncAnimation(fig, _update_comp, frames=frame_list,
                           blit=False, interval=1000 // fps)
    _anim_save(anim, f"{save_prefix}_anim_03_scenario_AC.gif", fps, dpi)
    plt.close(fig)
