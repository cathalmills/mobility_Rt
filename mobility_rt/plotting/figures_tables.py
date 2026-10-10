"""mobility_rt.plotting.figures_tables — the three result tables, drawn.

One figure per table written by the pipeline:

  plot_estimator_error_figure       estimator_error_{growth,full}  (error_tables)
  plot_location_discrepancy_figure  location_discrepancy           (location_errors)
  plot_location_diagnostics_figure  location_diagnostics           (location_errors)
  plot_manuscript_figure            manuscript_table               (combined_table)

Every value is read from the table dicts those modules return, so a figure can
never disagree with the table it depicts.  The entry points assert that the keys
they need are present; the scenario ORDER is the dict's own insertion order, which
the producing modules build from their ordered scenario list.  The only arithmetic
here is reducing a per-location array to the quartiles a panel draws, which uses
location_errors._ROWS so that figure and table reduce it identically.

Design rules kept throughout.  Magnitudes use ONE sequential hue, light to dark,
so "darker means worse" needs no key; signed quantities use one diverging pair,
orange for under, blue for over, with white at zero.  Every cell also carries its
number, so a reader never has to decode a colour to get a value.  Colour scales
are logarithmic because the errors span two to three decades and a linear ramp
would flatten everything below ten per cent into one pale box.

NO TITLES.  Like the other main-text figures, none of these carries a figure
title, a strapline or a panel title: panels are identified by their letter and by
their axis and colourbar labels, and everything else belongs in the caption.  That
is also why each axis and colourbar label names its quantity in full.

Drawn in the style of figures_corrected.set_modern_style, which rewrites global
rcParams, so each entry point runs inside an rc_context and restores the caller's
style on exit.
"""
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, LogNorm, SymLogNorm
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter

from mobility_rt.error_tables import SCHEMES
from mobility_rt.plotting.figures_corrected import set_modern_style, panel_label

# Sequential single-hue ramp, light to dark: darker is a larger error.
_SEQ = LinearSegmentedColormap.from_list(
    "err", ["#f2f7fb", "#cfe0ee", "#9dc2df", "#5b97c8", "#2b6ca8", "#123f68"])
# A log norm cannot place zero.  An error of exactly zero is the best possible, so
# it takes the lightest colour rather than matplotlib's transparent default.
_SEQ.set_bad(_SEQ(0.0))

# Diverging ramp for signed bias: orange under-estimates, blue over-estimates.
_DIV = LinearSegmentedColormap.from_list(
    "bias", ["#8a3800", "#d55e00", "#f3b183", "#ffffff",
             "#8fbfe0", "#0072B2", "#00436b"])
_DIV.set_bad("#ffffff")

INK      = "#1a1a1a"     # primary text
INK_SOFT = "#5f6b76"     # secondary text
GOOD     = "#0072B2"     # the informative reading
BAD      = "#D55E00"     # the reading that misleads
CONTEXT  = "#8C8C8C"     # the truth a reading is judged against
RULE     = "#c8ccd0"

# Row blocks of SCHEMES, in the order error_tables declares them.
_FAMILY_NAME = {"agg": "pooled", "ind": "independent", "out": "outward"}


def _family(key):
    """agg / ind / out — the block a scheme belongs to."""
    return key.split("_", 1)[0]


def _short(label):
    """Scenario column headers: 'Scenario A-het' -> 'A-het'."""
    return label.replace("Scenario ", "")


def _blocks(keys):
    """Contiguous runs of one family, as (start, stop_exclusive, family)."""
    fams = [_family(k) for k in keys]
    runs, i = [], 0
    while i < len(fams):
        j = i
        while j + 1 < len(fams) and fams[j + 1] == fams[i]:
            j += 1
        runs.append((i, j + 1, fams[i]))
        i = j + 1
    return runs


def _grid(ax, n_rows, n_cols):
    """White hairlines between cells, no spines, no tick marks."""
    ax.set_xticks(np.arange(-0.5, n_cols, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n_rows, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.1)
    ax.tick_params(which="minor", length=0)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)


def _annotate(ax, values, norm, cmap, fmt, bold=None):
    """Print every cell, in white wherever the fill is too dark for black ink."""
    n_rows, n_cols = values.shape
    for i in range(n_rows):
        for j in range(n_cols):
            v = values[i, j]
            if not np.isfinite(v):
                ax.text(j, i, "--", ha="center", va="center", fontsize=6.2,
                        color=INK_SOFT)
                continue
            # luminance of the fill this cell actually received
            r, g, b, _ = cmap(norm(v))
            lum = 0.299 * r + 0.587 * g + 0.114 * b
            weight = "bold" if (bold is not None and bold[i, j]) else "normal"
            ax.text(j, i, fmt(v), ha="center", va="center", fontsize=6.3,
                    color="white" if lum < 0.55 else INK, fontweight=weight)


def _scheme_rows(ax, keys, tex=None, brackets=True):
    """Row labels plus the family brackets down the left-hand side.

    tex=None leaves the axis unlabelled but still draws the block separators, for
    a second panel sitting beside one that carries the labels.
    """
    if tex is None:
        ax.set_yticks([])
    else:
        if len(tex) != len(keys):
            raise ValueError(f"{len(tex)} row labels for {len(keys)} schemes")
        ax.set_yticks(range(len(keys)))
        ax.set_yticklabels(tex, fontsize=7.4)
    runs = _blocks(keys)
    for lo, _hi, _f in runs[1:]:
        ax.axhline(lo - 0.5, color=INK, lw=1.0, zorder=6)
    if not brackets:
        return
    trans = ax.get_yaxis_transform()
    for lo, hi, fam in runs:
        mid = (lo + hi - 1) / 2.0
        ax.plot([-0.335, -0.335], [lo - 0.35, hi - 1 + 0.35], color=RULE, lw=1.6,
                transform=trans, clip_on=False, solid_capstyle="butt")
        ax.text(-0.355, mid, _FAMILY_NAME[fam], transform=trans, rotation=90,
                ha="center", va="center", fontsize=6.6, color=INK_SOFT)


def _scenario_axis(ax, labels, n_days=None, fontsize=7.4):
    """Categorical x axis of scenarios, optionally carrying the day count."""
    ticks = [_short(l) for l in labels]
    if n_days is not None:
        ticks = [f"{t}\nn={n_days[l]}" for t, l in zip(ticks, labels)]
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(ticks, fontsize=fontsize)


def _log_ticks(ax):
    """Label 1-2-5 within each decade; a two-decade panel with decade ticks only
    would carry two labels and be unreadable."""
    ax.yaxis.set_major_locator(LogLocator(base=10.0, subs=(1.0, 2.0, 5.0),
                                          numticks=20))
    ax.yaxis.set_minor_formatter(NullFormatter())
    # "%g", never a fixed number of decimals: rounding 0.05 to one place would
    # label it 0.1 and put the wrong number on the axis.
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _p: f"{v:g}"))


def _matrix(d_by_scheme, keys, labels):
    """{scheme: {label: value}} -> array in (scheme, scenario) order."""
    missing = [k for k in keys if k not in d_by_scheme]
    if missing:
        raise KeyError(f"the table is missing scheme(s) {missing}")
    for k in keys:
        absent = [l for l in labels if l not in d_by_scheme[k]]
        if absent:
            raise KeyError(f"scheme {k!r} is missing scenario(s) {absent}")
    return np.array([[d_by_scheme[k][l] for l in labels] for k in keys], float)


def _decade_limits(values, lo_floor=1e-3):
    """Whole decades bracketing the data, for a log colour scale.

    lo_floor caps how many decades the ramp may span: below it the differences are
    far smaller than anything the analysis can resolve, and spending colour on them
    would compress the range that matters.  Values under the floor are drawn in the
    lightest colour; every cell still carries its printed number.
    """
    v = values[np.isfinite(values) & (values > 0)]
    if v.size == 0:
        raise ValueError("a log colour scale needs at least one positive value")
    vmin = max(10.0 ** np.floor(np.log10(v.min())), lo_floor)
    vmax = 10.0 ** np.ceil(np.log10(v.max()))
    return vmin, (vmax if vmax > vmin else vmin * 10.0)


# --------------------------------------------------------------------------- #
#  Figure 1: the network-level error table
# --------------------------------------------------------------------------- #

def plot_estimator_error_figure(table, period="growth", save_prefix="fig",
                                fname_suffix=None):
    """estimator_error_<period>, drawn as a pair of heatmaps.

    table  : the dict returned by error_tables.compute_error_table.
    period : which of its periods to draw ("growth" or "full").

    Panel a is the mean relative error, |estimate - reference| / reference averaged
    over the scored days and expressed as a PERCENTAGE, as the colourbar says: how
    far a scheme sits from the true network reproduction number, never cancelling.
    The best scheme in each scenario is set in bold — exactly one cell per column,
    ties broken by scheme order, as error_tables does when it bolds the .tex.
    Panel b is the mean SIGNED error on the same days and the same scale.  Read
    together they separate two failures: a large panel-a value with a panel-b value
    of the same size is a systematic, one-directional error, while a large panel-a
    value next to a near-zero panel-b value is a scheme that is wrong in both
    directions and only looks unbiased on average.

    Untitled by design; the colourbars name the two quantities and the row
    brackets name the three families.

    Returns the path written.
    """
    if period not in table:
        raise KeyError(f"{period!r} is not in the table; have {sorted(table)}")
    d = table[period]
    labels = list(d["labels"])
    if not labels:
        raise ValueError("the error table has no scenarios")
    keys = [s.key for s in SCHEMES]

    mre = _matrix(d["mre"], keys, labels)
    bias = _matrix(d["bias"], keys, labels)
    if not np.all(np.isfinite(mre)):
        raise ValueError("the error table contains a non-finite mean relative error")
    # Exactly one bold cell per scenario, the first minimum in SCHEMES order, the
    # tie-break error_tables.write_error_tables uses for the .tex.  Bolding every
    # tied cell here would let figure and table name different best schemes.
    best = np.zeros_like(mre, dtype=bool)
    best[mre.argmin(axis=0), np.arange(mre.shape[1])] = True

    vmin, vmax = _decade_limits(mre)
    peak = np.max(np.abs(bias[np.isfinite(bias)])) if np.any(np.isfinite(bias)) else 0.0
    # Round the signed scale UP to a 1-2-5 step rather than to a whole decade: a
    # decade can be up to 10x the data (peak 11 would give a +-100 ramp, rendering
    # every cell near-white and understating the bias the numbers report).
    if peak > 0:
        decade = 10.0 ** np.floor(np.log10(peak))
        blim = next(m * decade for m in (1.0, 2.0, 5.0, 10.0)
                    if m * decade >= peak - 1e-12)
    else:
        blim = 1.0
    norm_e = LogNorm(vmin=vmin, vmax=vmax)
    norm_b = SymLogNorm(linthresh=0.1, linscale=0.6, vmin=-blim, vmax=blim, base=10)

    with plt.rc_context():
        set_modern_style()
        fig = plt.figure(figsize=(9.0, 4.3))
        gs = gridspec.GridSpec(2, 2, height_ratios=[1.0, 0.05], hspace=0.40,
                               wspace=0.10, left=0.20, right=0.985,
                               top=0.955, bottom=0.10)

        ax_e = fig.add_subplot(gs[0, 0])
        ax_e.imshow(mre, cmap=_SEQ, norm=norm_e, aspect="auto",
                    interpolation="nearest")
        _annotate(ax_e, mre, norm_e, _SEQ, lambda v: f"{v:.2f}", bold=best)
        _grid(ax_e, len(keys), len(labels))
        _scheme_rows(ax_e, keys, [s.tex for s in SCHEMES])
        _scenario_axis(ax_e, labels, d["n_days"])

        ax_b = fig.add_subplot(gs[0, 1])
        ax_b.imshow(bias, cmap=_DIV, norm=norm_b, aspect="auto",
                    interpolation="nearest")
        _annotate(ax_b, bias, norm_b, _DIV, lambda v: f"{v:+.2f}")
        _grid(ax_b, len(keys), len(labels))
        _scheme_rows(ax_b, keys, None, brackets=False)
        _scenario_axis(ax_b, labels)

        cb_e = fig.colorbar(plt.cm.ScalarMappable(norm=norm_e, cmap=_SEQ),
                            cax=fig.add_subplot(gs[1, 0]), orientation="horizontal")
        cb_e.set_label("mean relative error (%)", fontsize=7.2, color=INK)
        cb_b = fig.colorbar(plt.cm.ScalarMappable(norm=norm_b, cmap=_DIV),
                            cax=fig.add_subplot(gs[1, 1]), orientation="horizontal")
        cb_b.set_label("mean signed error (%):  under-estimates left, "
                       "over-estimates right", fontsize=7.2, color=INK)
        for cb in (cb_e, cb_b):
            cb.ax.tick_params(labelsize=6.2, length=2)
            cb.outline.set_visible(False)

        panel_label(ax_e, "a", x=-0.40, y=1.03)
        panel_label(ax_b, "b", x=-0.035, y=1.03)

        suffix = fname_suffix or f"estimator_error_{period}"
        fname = f"{save_prefix}_{suffix}.pdf"
        fig.savefig(fname, dpi=300, bbox_inches="tight")
        plt.close(fig)
    print(f"  Saved {fname}")
    return fname


# --------------------------------------------------------------------------- #
#  Panels shared by the diagnostics figure and the manuscript figure
# --------------------------------------------------------------------------- #

def _require_positive(name, *series):
    """A log axis cannot show zero or a negative; say which panel, not "nan"."""
    for a in series:
        if not np.all(np.isfinite(a)) or np.any(a <= 0):
            raise ValueError(f"{name}: every value must be finite and positive "
                             f"for a log axis, got {np.asarray(a)}")


def _require_percent(name, *series):
    """Agreement rates are percentages of location-days; outside 0-100 the axis
    would silently clip."""
    for a in series:
        if not np.all(np.isfinite(a)) or np.any(a < 0) or np.any(a > 100):
            raise ValueError(f"{name}: every value must lie in [0, 100], "
                             f"got {np.asarray(a)}")


def _panel_rank(ax, labels, med, q1=None, q3=None):
    """Median across-location rank correlation, with the interquartile range if given.

    A scenario is flagged in the warning colour when its lower quartile reaches down
    to zero or below -- i.e. on at least a quarter of days the ranking carries no
    information.  Where no quartiles are supplied the median alone decides.
    """
    for name, a in (("median", med), ("q1", q1), ("q3", q3)):
        if a is not None and (not np.all(np.isfinite(a)) or np.any(np.abs(a) > 1)):
            raise ValueError(f"rank correlation {name} outside [-1, 1]: {a}")
    x = np.arange(len(labels))
    ax.axhspan(-1.05, 0, color="#fbeee6", zorder=0)
    ax.axhline(0.0, color=INK_SOFT, lw=0.8, zorder=1)
    for i in x:
        col = BAD if (med[i] if q1 is None else q1[i]) <= 0.0 else GOOD
        if q1 is not None and q3 is not None:
            ax.plot([i, i], [q1[i], q3[i]], color=col, lw=3.6, alpha=0.45,
                    solid_capstyle="round", zorder=2)
        else:
            ax.plot([i, i], [0.0, med[i]], color=col, lw=1.6, alpha=0.55,
                    solid_capstyle="round", zorder=2)
        ax.plot(i, med[i], "o", ms=6.0, color=col, mec="white", mew=1.0, zorder=3)
    ax.set_ylim(-1.05, 1.05)
    ax.set_yticks([-1, -0.5, 0, 0.5, 1])
    ax.set_ylabel("rank correlation with "
                  r"$R^{j}_{\mathrm{out}}$ across locations", fontsize=7.4)
    # Name the rule in force: with quartiles the lower quartile decides, without them
    # the median does, and the same label over both would let one scenario be flagged
    # in this figure and not in the other with nothing on the page to explain it.
    rule = "median" if q1 is None else "lower quartile"
    ax.text(0.02, 0.03, f"no better than chance ({rule} at or below zero)",
            transform=ax.transAxes, fontsize=6.4, color=BAD, va="bottom")
    ax.grid(axis="y", zorder=0)
    ax.set_axisbelow(True)


def _panel_spread(ax, labels, est, true):
    """Across-location spread of the estimates against the true spread."""
    _require_positive("coefficient of variation", est, true)
    x = np.arange(len(labels))
    for i in x:
        ax.plot([i, i], [est[i], true[i]], color=RULE, lw=2.2, zorder=1,
                solid_capstyle="round")
    ax.plot(x, true, "o", ms=6.0, color=CONTEXT, mec="white", mew=1.0, zorder=3)
    ax.plot(x, est, "o", ms=6.0, color=GOOD, mec="white", mew=1.0, zorder=3)
    ax.set_yscale("log")
    lo, hi = min(est.min(), true.min()), max(est.max(), true.max())
    ax.set_ylim(lo / 1.8, hi * 3.2)          # headroom for the key
    _log_ticks(ax)
    ax.set_ylabel("spread across locations (CV, %)", fontsize=7.4)
    ax.legend(handles=[
        Line2D([], [], ls="", marker="o", ms=5, color=CONTEXT,
               label=r"true, $R^{j}_{\mathrm{out}}$"),
        Line2D([], [], ls="", marker="o", ms=5, color=GOOD,
               label="estimated")], loc="upper left", fontsize=6.6,
        handletextpad=0.3, borderaxespad=0.2)
    ax.grid(axis="y", zorder=0)
    ax.set_axisbelow(True)


def _panel_agree(ax, labels, growth, own):
    """Which yes/no question the per-location estimate actually answers."""
    _require_percent("agreement", growth, own)
    x = np.arange(len(labels))
    w = 0.36
    ax.bar(x - w / 2, growth, width=w, color=GOOD, lw=0,
           label="local cases rising")
    ax.bar(x + w / 2, own, width=w, color=BAD, lw=0,
           label=r"own $R^{j}_{\mathrm{out}}>1$")
    ax.axhline(50.0, color=INK_SOFT, lw=0.9, ls=(0, (3, 2)), zorder=4)
    ax.set_ylim(0, 142)                      # headroom for the key above the bars
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_ylabel("location-days agreeing (%)", fontsize=7.4)
    ax.legend(loc="upper left", fontsize=6.4, ncol=1, handlelength=1.0,
              handletextpad=0.4, borderaxespad=0.25)
    ax.grid(axis="y", zorder=0)
    ax.set_axisbelow(True)


def _panel_distance(ax, labels, own, net):
    """What a location's own estimate is actually close to."""
    _require_positive("relative distance", own, net)
    x = np.arange(len(labels))
    for i in x:
        ax.plot([i, i], [own[i], net[i]], color=RULE, lw=2.2, zorder=1,
                solid_capstyle="round")
    ax.plot(x, own, "o", ms=6.0, color=BAD, mec="white", mew=1.0, zorder=3)
    ax.plot(x, net, "o", ms=6.0, color=GOOD, mec="white", mew=1.0, zorder=3)
    ax.set_yscale("log")
    lo, hi = min(own.min(), net.min()), max(own.max(), net.max())
    ax.set_ylim(lo / 1.8, hi * 3.2)          # headroom for the key
    _log_ticks(ax)
    ax.set_ylabel("distance from the comparator (%)", fontsize=7.4)
    ax.legend(handles=[
        Line2D([], [], ls="", marker="o", ms=5, color=BAD,
               label=r"own $R^{j}_{\mathrm{out}}$"),
        Line2D([], [], ls="", marker="o", ms=5, color=GOOD,
               label=r"network $\rho(\mathbf{R})$")], loc="upper left",
        fontsize=6.6, handletextpad=0.3, borderaxespad=0.2)
    ax.grid(axis="y", zorder=0)
    ax.set_axisbelow(True)


def _panel_net(ax, labels, net):
    """The control row: how far one location's own estimate sits from the network
    value.

    Small here is only half the statement; the other half, the distance from the
    location's own R_out, is not in the diagnostics figure at all — it is the
    agree_own row here and the paired panel of the discrepancy figure
    (_panel_distance), where the two comparators are drawn together.  Read against
    that, a small distance from the network value is the location-level result: the
    estimate has tracked the network, not the location."""
    _require_positive("distance from the network value", net)
    x = np.arange(len(labels))
    for i in x:
        ax.plot([i, i], [0.0, net[i]], color=GOOD, lw=1.6, alpha=0.55, zorder=1,
                solid_capstyle="round")
    ax.plot(x, net, "o", ms=6.0, color=GOOD, mec="white", mew=1.0, zorder=3)
    ax.set_ylim(0, net.max() * 1.45)
    ax.set_ylabel(r"distance from $\rho(\mathbf{R}(t))$ (%)", fontsize=7.4)
    ax.grid(axis="y", zorder=0)
    ax.set_axisbelow(True)


# --------------------------------------------------------------------------- #
#  Figure 2: the location-level diagnostics table
# --------------------------------------------------------------------------- #

_DIAG_KEYS = ("rank_median", "rank_q1", "rank_q3", "cv_est", "cv_true",
              "agree_growth", "agree_own", "net_median", "n_days")


def plot_location_diagnostics_figure(diag_table, save_prefix="fig",
                                     fname_suffix="location_diagnostics"):
    """location_diagnostics, drawn as the questions it answers.

    diag_table : {scenario label: location_errors.location_diagnostics(...)},
                 in the column order wanted.

    Panel a asks whether ranking locations by their own independent estimates
    recovers the ranking by their true outward reproduction numbers; a median at or
    below zero means it does not.  Panel b asks whether the estimates reproduce how
    unequal transmission really is across locations.  Panel c contrasts the two
    yes/no questions the same estimate could be asked: are local cases rising, and
    is this location's own outward reproduction number above one.  Panel d is the
    table's control row, the distance from the network value.

    Every row of location_diagnostics appears: the three rank-correlation rows in
    a, the two spread rows in b, the two agreement rows in c and the control in d.
    The day count per scenario rides on the x axis, since the figure carries no
    title or strapline to put it in.

    Returns the path written.
    """
    labels = list(diag_table)
    if not labels:
        raise ValueError("the diagnostics table has no scenarios")
    for lab in labels:
        missing = [k for k in _DIAG_KEYS if k not in diag_table[lab]]
        if missing:
            raise KeyError(f"{lab} is missing diagnostic(s) {missing}")

    g = lambda k: np.array([float(diag_table[l][k]) for l in labels])
    med, q1, q3 = g("rank_median"), g("rank_q1"), g("rank_q3")
    if np.any(q1 > med) or np.any(med > q3):
        raise ValueError("the rank-correlation quartiles are not ordered "
                         "q1 <= median <= q3")

    with plt.rc_context():
        set_modern_style()
        fig, axes = plt.subplots(1, 4, figsize=(10.4, 3.1))
        _panel_rank(axes[0], labels, med, q1, q3)
        _panel_spread(axes[1], labels, g("cv_est"), g("cv_true"))
        _panel_agree(axes[2], labels, g("agree_growth"), g("agree_own"))
        _panel_net(axes[3], labels, g("net_median"))
        n_days = {l: diag_table[l]["n_days"] for l in labels}
        # Panel a drops any day on which the across-location correlation is
        # undefined (every location tied), so it can rest on fewer days than the
        # rest of the block; label it with its own count rather than the block's.
        n_rank = {l: int(diag_table[l].get("n_rank", n_days[l])) for l in labels}
        # Panel c's cases-rising bar needs two complete case windows behind a day, so
        # it too can rest on fewer days than the block.
        n_grow = {l: int(diag_table[l].get("n_growth", n_days[l])) for l in labels}
        per_panel = {"a": n_rank, "c": n_grow}
        for ax, letter in zip(axes, "abcd"):
            _scenario_axis(ax, labels, per_panel.get(letter, n_days), fontsize=7.0)
            ax.set_xlim(-0.6, len(labels) - 0.4)
            panel_label(ax, letter, x=-0.17, y=1.06)
        fig.tight_layout()

        fname = f"{save_prefix}_{fname_suffix}.pdf"
        fig.savefig(fname, dpi=300, bbox_inches="tight")
        plt.close(fig)
    print(f"  Saved {fname}")
    return fname


# --------------------------------------------------------------------------- #
#  Figure 3: the per-location discrepancy table
# --------------------------------------------------------------------------- #

_DISC_COMPARATORS = (("own", BAD,  r"vs own $R^{j}_{\mathrm{out}}(t)$", -0.14),
                     ("net", GOOD, r"vs $\rho(\mathbf{R}(t))$",         +0.14))


def _panel_discrepancy(ax, labels, table, metric):
    """One metric of location_discrepancy: median, interquartile range and the
    extreme location, for both comparators, one pair of markers per scenario.

    The table stores a value per location; the four statistics the CSV prints
    (median, both quartiles and the extreme location) are recomputed here from those
    same arrays, by the same reductions, so figure and table cannot drift apart.  The
    .tex carries the mean-relative-error rows only, so for `bias` this panel is the
    only place those four statistics are drawn.
    """
    for ckey, col, _lab, off in _DISC_COMPARATORS:
        x = np.arange(len(labels)) + off
        for i, lab in enumerate(labels):
            v = np.asarray(table[lab][ckey][metric], dtype=float)
            if v.size == 0 or not np.all(np.isfinite(v)):
                raise ValueError(f"{lab}/{ckey}/{metric}: expected a finite value "
                                 f"per location, got {v}")
            q1, q3 = np.percentile(v, 25), np.percentile(v, 75)
            ax.plot([x[i], x[i]], [q1, q3], color=col, lw=3.4, alpha=0.45,
                    solid_capstyle="round", zorder=2)
            ax.plot(x[i], v.max(), marker="_", ms=7.0, mew=1.5, ls="", color=col,
                    alpha=0.95, zorder=3)
            ax.plot(x[i], np.median(v), "o", ms=5.5, color=col, mec="white",
                    mew=1.0, zorder=4)
    ax.grid(axis="y", zorder=0)
    ax.set_axisbelow(True)


def plot_location_discrepancy_figure(loc_table, save_prefix="fig",
                                     fname_suffix="location_discrepancy"):
    """location_discrepancy, drawn: the spread ACROSS LOCATIONS of how far one
    location's own independent estimate sits from each comparator.

    loc_table : {scenario label: location_errors.location_discrepancies(...)},
                in the column order wanted.

    Panel a is the mean relative error, panel b the mean signed bias, each on a
    per-location basis and each summarised by the median, the interquartile range
    and the most extreme location -- the four statistics the table prints.  Orange
    compares a location's estimate with its OWN outward reproduction number, blue
    with the network value.  Orange far above blue is the whole point of the table:
    the estimate is close to the network number and far from the local one.

    Untitled by design; the axis labels and the key carry the reading.

    Returns the path written.
    """
    labels = list(loc_table)
    if not labels:
        raise ValueError("the discrepancy table has no scenarios")
    for lab in labels:
        if "n_days" not in loc_table[lab]:
            raise KeyError(f"{lab} is missing n_days")
        for ckey, _c, _l, _o in _DISC_COMPARATORS:
            if ckey not in loc_table[lab]:
                raise KeyError(f"{lab} is missing comparator {ckey!r}")
            for metric in ("mre", "bias"):
                if metric not in loc_table[lab][ckey]:
                    raise KeyError(f"{lab}/{ckey} is missing {metric!r}")
    mre_all = np.concatenate([np.asarray(loc_table[l][c]["mre"], float).ravel()
                              for l in labels for c, _c, _l, _o in _DISC_COMPARATORS])
    _require_positive("per-location mean relative error", mre_all)

    with plt.rc_context():
        set_modern_style()
        fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.2))

        _panel_discrepancy(axes[0], labels, loc_table, "mre")
        axes[0].set_yscale("log")
        axes[0].set_ylim(mre_all.min() / 1.9, mre_all.max() * 3.4)
        _log_ticks(axes[0])
        axes[0].set_ylabel("mean relative error, per location (%)", fontsize=7.4)
        axes[0].legend(handles=[Line2D([], [], ls="", marker="o", ms=5, color=c,
                                       label=lab)
                                for _k, c, lab, _o in _DISC_COMPARATORS],
                       loc="upper left", fontsize=6.6, handletextpad=0.3,
                       borderaxespad=0.2)

        _panel_discrepancy(axes[1], labels, loc_table, "bias")
        axes[1].axhline(0.0, color=INK_SOFT, lw=0.8, zorder=1)
        axes[1].set_yscale("symlog", linthresh=1.0, linscale=0.7)
        axes[1].set_ylabel("mean signed bias, per location (%)", fontsize=7.4)
        axes[1].legend(handles=[
            Line2D([], [], ls="", marker="o", ms=5, color=CONTEXT,
                   label="median location"),
            Line2D([], [], color=CONTEXT, lw=3.4, alpha=0.45,
                   label="interquartile range"),
            Line2D([], [], ls="", marker="_", ms=7, mew=1.5, color=CONTEXT,
                   label="largest value")], loc="upper left", fontsize=6.6,
            handletextpad=0.4, handlelength=1.2, borderaxespad=0.2)

        n_days = {l: loc_table[l]["n_days"] for l in labels}
        for ax, letter in zip(axes, "ab"):
            _scenario_axis(ax, labels, n_days, fontsize=7.0)
            ax.set_xlim(-0.6, len(labels) - 0.4)
            panel_label(ax, letter, x=-0.16, y=1.06)
        fig.tight_layout()

        fname = f"{save_prefix}_{fname_suffix}.pdf"
        fig.savefig(fname, dpi=300, bbox_inches="tight")
        plt.close(fig)
    print(f"  Saved {fname}")
    return fname


# --------------------------------------------------------------------------- #
#  Figure 4: the single manuscript table
# --------------------------------------------------------------------------- #

_COMBINED_LOC_KEYS = ("rank_median", "cv_est", "cv_true", "agree_growth",
                      "agree_own", "own", "net")


def plot_manuscript_figure(combined, save_prefix="fig",
                           fname_suffix="manuscript_table"):
    """manuscript_table, drawn: both halves of the analysis on one set of days.

    combined : the dict returned by combined_table.compute_combined_table.

    Panel a is the network-level half — how close each aggregation scheme gets to
    rho(R(t)) over the growth phase, with the DIRECTION of its mean signed error
    marked in the corner of each cell.  Only the direction: where a scheme errs one
    way the printed error is also the size of that signed error, and where it does
    not the magnitude is in panel b of the network-error figure.  Panels b to e are the
    location-level half: the same simulations asked what one location's own
    independent estimate is worth.  Because combined_table refuses to assemble
    unless the two halves were scored on identical days, the rows may be read
    against each other.

    Returns the path written.
    """
    labels = list(combined["labels"])
    if not labels:
        raise ValueError("the combined table has no scenarios")
    keys = [s.key for s in SCHEMES]
    loc = combined["location"]
    missing = [k for k in _COMBINED_LOC_KEYS if k not in loc]
    if missing:
        raise KeyError(f"the combined table is missing location row(s) {missing}")

    pair = combined["network"]
    mre = np.array([[float(pair[k][l][0]) for l in labels] for k in keys])
    bias = np.array([[float(pair[k][l][1]) for l in labels] for k in keys])
    if not np.all(np.isfinite(mre)):
        raise ValueError("the combined table contains a non-finite relative error")
    # Exactly one bold cell per scenario, the first minimum in SCHEMES order, the
    # tie-break error_tables.write_error_tables uses for the .tex.  Bolding every
    # tied cell here would let figure and table name different best schemes.
    best = np.zeros_like(mre, dtype=bool)
    best[mre.argmin(axis=0), np.arange(mre.shape[1])] = True
    vmin, vmax = _decade_limits(mre)
    norm_e = LogNorm(vmin=vmin, vmax=vmax)

    L = lambda k: np.array([float(loc[k][l]) for l in labels])

    with plt.rc_context():
        set_modern_style()
        fig = plt.figure(figsize=(9.2, 6.0))
        outer = gridspec.GridSpec(2, 1, height_ratios=[1.0, 0.95], hspace=0.26,
                                  left=0.115, right=0.975, top=0.945, bottom=0.075)
        top = gridspec.GridSpecFromSubplotSpec(
            1, 2, subplot_spec=outer[0], width_ratios=[1.0, 0.035], wspace=0.025)
        bot = gridspec.GridSpecFromSubplotSpec(
            1, 4, subplot_spec=outer[1], wspace=0.42)

        ax = fig.add_subplot(top[0, 0])
        ax.imshow(mre, cmap=_SEQ, norm=norm_e, aspect="auto",
                  interpolation="nearest")
        _annotate(ax, mre, norm_e, _SEQ, lambda v: f"{v:.2f}", bold=best)
        # sign of the mean signed error, as a marker in the corner of each cell
        for i in range(len(keys)):
            for j in range(len(labels)):
                if not np.isfinite(bias[i, j]) or bias[i, j] == 0.0:
                    continue                       # no direction to mark
                ax.plot(j + 0.385, i - 0.30, marker="v" if bias[i, j] < 0 else "^",
                        ms=3.6, color=BAD if bias[i, j] < 0 else GOOD,
                        mec="white", mew=0.5, zorder=5)
        _grid(ax, len(keys), len(labels))
        _scheme_rows(ax, keys, [s.tex for s in SCHEMES])
        _scenario_axis(ax, labels, combined["n_days"])
        cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm_e, cmap=_SEQ),
                          cax=fig.add_subplot(top[0, 1]))
        cb.set_label("mean relative error (%)", fontsize=7.2, color=INK)
        cb.ax.tick_params(labelsize=6.2, length=2)
        cb.outline.set_visible(False)
        ax.legend(handles=[
            Line2D([], [], ls="", marker="^", ms=4, color=GOOD,
                   label="over-estimates"),
            Line2D([], [], ls="", marker="v", ms=4, color=BAD,
                   label="under-estimates")],
            loc="lower right", bbox_to_anchor=(1.0, 1.005), ncol=2, fontsize=6.6,
            frameon=False, handletextpad=0.25, columnspacing=1.1,
            borderaxespad=0.0)
        panel_label(ax, "a", x=-0.125, y=1.055)

        axes = [fig.add_subplot(bot[0, i]) for i in range(4)]
        _panel_rank(axes[0], labels, L("rank_median"))   # no quartiles in this table
        _panel_spread(axes[1], labels, L("cv_est"), L("cv_true"))
        _panel_agree(axes[2], labels, L("agree_growth"), L("agree_own"))
        _panel_distance(axes[3], labels, L("own"), L("net"))
        for ax_i, letter in zip(axes, "bcde"):
            _scenario_axis(ax_i, labels, fontsize=7.0)
            ax_i.set_xlim(-0.6, len(labels) - 0.4)
            panel_label(ax_i, letter, x=-0.28, y=1.07)

        fname = f"{save_prefix}_{fname_suffix}.pdf"
        fig.savefig(fname, dpi=300, bbox_inches="tight")
        plt.close(fig)
    print(f"  Saved {fname}")
    return fname
