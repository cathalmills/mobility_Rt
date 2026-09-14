"""mobility_rt.location_errors — what a single location's own estimate tells you.

The network-level tables in mobility_rt.error_tables ask how well a WEIGHTED
AGGREGATE of per-location numbers recovers rho(R(t)).  This module asks the
location-level question underneath it: when you estimate a reproduction number
independently for one location, what are you actually learning about?

Two comparators, for every location:
  own R_out  that location's own outward reproduction number, the quantity a
             reader naturally assumes a local estimate is measuring.
  network    rho(R(t)), the network value, which is not local at all.

The finding this quantifies is that the second is the closer of the two.  Mobility
mixes the infectious pressure arriving in a location, so a location's renewal ratio
reflects the whole network rather than its own transmissibility.  The numbers here
are therefore a DISCREPANCY, not an estimator error: the independent estimator is
not trying and failing to measure R_out, it is measuring something else.

Conventions match error_tables so the two sets of numbers can be read together:
  - the per-location estimate summarises the preceding ESTIMATOR_WINDOW days, so
    both comparators are taken as their matching trailing means;
  - only growth-phase days count, rho(R(t)) > 1, and only days on which EVERY
    location has an estimate, so the location distribution is over a fixed set.
"""
import csv
import os

import numpy as np

from mobility_rt.error_tables import (ESTIMATOR_WINDOW, RHO_FLOOR, _tex_escape,
                                      trailing_mean)
from mobility_rt.estimators import estimate_R_independent
from mobility_rt.kernel import R_outward, R_system

# (key, plain label, LaTeX label) for the two comparators
COMPARATORS = [
    ("own", "that location's own outward R", r"$R^{j}_{\mathrm{out}}(t)$"),
    ("net", "the network reproduction number", r"$\rho(\mathbf{R}(t))$"),
]


def _prepare(sim, gen_time_pmf):
    """Shared inputs and day mask for both entry points.

    Returns (R_hat, ref_own, ref_net, days).  Defined once so the two tables can
    never end up scored on different days: a day counts only when the epidemic is
    growing, every location has an estimate, and both comparators are usable as a
    relative-error denominator.
    """
    inc = sim["incidence"]
    R_mats = sim["R_matrices"]
    T, N = inc.shape
    rho = np.array([R_system(R_mats[t]) for t in range(T)])
    R_out = np.array([R_outward(R_mats[t]) for t in range(T)])
    R_hat = estimate_R_independent(inc, gen_time_pmf, window=ESTIMATOR_WINDOW)

    # matched references: the estimate summarises a window, so must its comparators
    ref_net = trailing_mean(rho, ESTIMATOR_WINDOW)
    ref_own = np.column_stack([trailing_mean(R_out[:, j], ESTIMATOR_WINDOW)
                               for j in range(N)])

    days = (rho > 1.0) & ~np.isnan(R_hat).any(axis=1)
    days &= np.isfinite(ref_net) & (ref_net > RHO_FLOOR)
    days &= (np.isfinite(ref_own) & (ref_own > RHO_FLOOR)).all(axis=1)
    if not days.any():
        raise ValueError("no growth-phase day has an estimate for every location")
    return R_hat, ref_own, ref_net, days


def location_discrepancies(sim, gen_time_pmf):
    """Per-location mean relative discrepancy and signed bias, in per cent.

    Everything needed comes from the simulation itself, so no city tuple is taken;
    compute_location_table still accepts one per scenario purely so that the same
    [(label, sim, city), ...] list can be handed to it and to
    error_tables.compute_error_table.

    Both comparators are taken as PLAIN trailing means over ESTIMATOR_WINDOW days.
    A window-summed ratio actually targets an infectiousness-weighted average over
    its window, so the plain mean is the conventional approximation to that, the
    same one error_tables uses; the two differ only to the extent the comparator
    moves within a single window.

    Returns {comparator: {"mre": array(N), "bias": array(N)}}, plus "n_days".
    Locations are in their natural order, so the arrays line up across scenarios.
    """
    R_hat, ref_own, ref_net, days = _prepare(sim, gen_time_pmf)
    N = R_hat.shape[1]

    out = {"n_days": int(days.sum())}
    for key, ref in (("own", ref_own), ("net", np.repeat(ref_net[:, None], N, axis=1))):
        rel = (R_hat[days] - ref[days]) / ref[days]
        out[key] = {"mre":  np.mean(np.abs(rel), axis=0) * 100.0,
                    "bias": np.mean(rel, axis=0) * 100.0}
    return out


def compute_location_table(scenarios, gen_time_pmf):
    """{scenario label: location_discrepancies(...)} in the given column order."""
    if not scenarios:
        raise ValueError("compute_location_table needs at least one scenario")
    labels = [lab for lab, _, _ in scenarios]
    if len(set(labels)) != len(labels):
        raise ValueError(f"scenario labels must be unique, got {labels}")
    # `city` is unpacked and discarded: the scenario tuples are shaped for
    # error_tables.compute_error_table, which does need it.
    return {lab: location_discrepancies(sim, gen_time_pmf)
            for lab, sim, _city in scenarios}


# (row key, plain label, LaTeX label, how to reduce the per-location array)
_ROWS = [
    ("median", "median across locations", r"median",
     lambda v: float(np.median(v))),
    ("q1",     "lower quartile",          r"lower quartile",
     lambda v: float(np.percentile(v, 25))),
    ("q3",     "upper quartile",          r"upper quartile",
     lambda v: float(np.percentile(v, 75))),
    ("max",    "worst location",          r"worst location",
     lambda v: float(np.max(v))),
]


def _fmt(x):
    """One decimal place; an empty cell for a missing value, as in error_tables."""
    return "" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.1f}"


def write_location_table(table, out_dir, prefix="location_discrepancy"):
    """One CSV and one .tex summarising the per-location distribution."""
    os.makedirs(out_dir, exist_ok=True)
    labels = list(table)
    csv_path = os.path.join(out_dir, f"{prefix}.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["comparator", "statistic", "quantity"] + labels)
        for ckey, cplain, _ in COMPARATORS:
            for metric in ("mre", "bias"):
                for rkey, rplain, _, fn in _ROWS:
                    w.writerow([cplain, rplain,
                                "mean relative error %" if metric == "mre"
                                else "mean signed bias %"]
                               + [_fmt(fn(table[l][ckey][metric])) for l in labels])
        w.writerow(["", "days used", "count"] + [table[l]["n_days"] for l in labels])

    tex_path = os.path.join(out_dir, f"{prefix}.tex")
    with open(tex_path, "w", encoding="utf-8") as fh:
        fh.write("% Auto-generated by mobility_rt.location_errors - do not edit by hand.\n")
        fh.write("% Requires \\usepackage{booktabs}.\n")
        fh.write("\\begin{table}[htbp]\n\\centering\n\\small\n")
        fh.write("\\caption{Per-location discrepancy of the independent estimate "
                 "$\\hat{R}^{j}_{\\mathrm{ind}}(t)$, over growth-phase days on which "
                 "every location has an estimate. Each comparator is taken as its "
                 f"{ESTIMATOR_WINDOW}-day trailing mean, matching the window the "
                 "estimate summarises. Statistics are over the locations. The "
                 "estimate sits closer to the network value than to the location's "
                 "own outward reproduction number, so these are a discrepancy "
                 "rather than an estimator error. Signed biases are in the "
                 "companion CSV.}\n")
        fh.write(f"\\label{{tab:{prefix}}}\n")
        fh.write("\\begin{tabular}{ll" + "r" * len(labels) + "}\n\\toprule\n")
        fh.write("Compared with & Statistic & "
                 + " & ".join(_tex_escape(l) for l in labels) + " \\\\\n")
        for ckey, _, ctex in COMPARATORS:
            fh.write("\\midrule\n")
            fh.write(f"\\multicolumn{{{2 + len(labels)}}}{{l}}"
                     f"{{\\textit{{Mean relative error vs {ctex} (\\%)}}}} \\\\\n")
            for rkey, _, rtex, fn in _ROWS:
                fh.write(f" & {rtex} & "
                         + " & ".join(_fmt(fn(table[l][ckey]["mre"])) for l in labels)
                         + " \\\\\n")
        fh.write("\\midrule\nDays used & & "
                 + " & ".join(str(table[l]["n_days"]) for l in labels) + " \\\\\n")
        fh.write("\\bottomrule\n\\end{tabular}\n\\end{table}\n")
    return [csv_path, tex_path]


def build_location_table(scenarios, gen_time_pmf, out_dir,
                         prefix="location_discrepancy", verbose=True):
    """Compute and write the location-level table; return (table, paths)."""
    table = compute_location_table(scenarios, gen_time_pmf)
    paths = write_location_table(table, out_dir, prefix=prefix)
    if verbose:
        labels = list(table)
        print("\n  Per-location discrepancy of the independent estimate (%), "
              "growth phase")
        print("    " + " " * 34 + "".join(f"{l[:14]:>16s}" for l in labels))
        for ckey, cplain, _ in COMPARATORS:
            print(f"    vs {cplain}")
            for rkey, rplain, _, fn in _ROWS:
                print(f"      {rplain:<30s}"
                      + "".join(f"{_fmt(fn(table[l][ckey]['mre'])):>15s}%"
                                for l in labels))
        for p in paths:
            print(f"  Saved {p}")
    return table, paths


# ---------------------------------------------------------------------------
# What a panel of per-location estimates is, and is not, good for
# ---------------------------------------------------------------------------
#
# The discrepancy table above says how far a local estimate sits from two
# comparators.  A distance is hard to act on, so these three diagnostics ask the
# questions a public health team actually has, and each has a clear answer:
#
#   rank   Can I rank locations by their estimates to find the worst place?
#          Spearman correlation between the estimates and the locations' true
#          outward reproduction numbers, computed ACROSS locations on each day.
#   cv     Does a panel of estimates convey how heterogeneous the network is?
#          Spread of the estimates against the true spread of R^j_out.
#   agree  Does a location's estimate at least tell that location whether its own
#          cases are rising?  Agreement of sign(Rhat_j - 1) with (a) that
#          location's incidence actually growing week on week, and (b) that
#          location's own outward reproduction number exceeding 1.
#
# Growth is defined week on week on the windowed case totals, the standard
# surveillance comparison, so the first ESTIMATOR_WINDOW days of any window pair
# are excluded because the earlier window is not yet complete.

def _spearman(a, b):
    """Rank correlation, with average ranks for ties; NaN if either side is flat.

    Implemented here rather than pulled from scipy so that the tie handling and
    the degenerate case are explicit: when every location has the same value the
    correlation is undefined, not zero.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.shape != b.shape or a.ndim != 1 or a.size < 2:
        raise ValueError("_spearman needs two 1-D arrays of equal length >= 2")

    def _rank(x):
        """Ranks 1..n, with tied values sharing their average rank."""
        order = np.argsort(x, kind="mergesort")
        r = np.empty(len(x), dtype=float)
        r[order] = np.arange(1, len(x) + 1, dtype=float)
        # average the ranks of tied values
        s = x[order]
        i = 0
        while i < len(s):
            j = i
            while j + 1 < len(s) and s[j + 1] == s[i]:
                j += 1
            if j > i:
                r[order[i:j + 1]] = np.mean(r[order[i:j + 1]])
            i = j + 1
        return r

    ra, rb = _rank(a), _rank(b)
    if ra.std() == 0 or rb.std() == 0:
        return np.nan
    return float(np.corrcoef(ra, rb)[0, 1])


def location_diagnostics(sim, gen_time_pmf):
    """Rank, spread and growth-agreement diagnostics for one simulation.

    Returns a dict with:
      rank_median, rank_q1, rank_q3  quartiles over growth-phase days of the
                                     across-location Spearman correlation
      cv_est, cv_true                median across-location coefficient of
                                     variation of the estimates and of R^j_out
      agree_growth                   % of location-days where sign(Rhat-1)
                                     matches the location's cases actually rising
      agree_own                      % where it matches that location's own
                                     outward reproduction number exceeding 1
      net_median                     median across locations of the distance from
                                     the network value, kept only as a control
      n_days
    """
    inc = sim["incidence"]
    T, N = inc.shape
    if N < 2:
        raise ValueError("rank diagnostics need at least two locations")
    R_hat, ref_own, ref_net, days = _prepare(sim, gen_time_pmf)

    idx = np.where(days)[0]
    rs = np.array([_spearman(R_hat[t], ref_own[t]) for t in idx])
    rs = rs[np.isfinite(rs)]
    if rs.size == 0:
        raise ValueError("the outward reproduction numbers never vary across "
                         "locations, so a ranking correlation is undefined")

    cv_est = float(np.median(R_hat[days].std(axis=1) / R_hat[days].mean(axis=1)))
    cv_true = float(np.median(ref_own[days].std(axis=1) / ref_own[days].mean(axis=1)))

    # week-on-week growth of the windowed case totals
    win = np.array([inc[max(0, t - ESTIMATOR_WINDOW + 1):t + 1].sum(axis=0)
                    for t in range(T)])
    growing = np.zeros((T, N), dtype=bool)
    growing[ESTIMATOR_WINDOW:] = win[ESTIMATOR_WINDOW:] > win[:-ESTIMATOR_WINDOW]
    gdays = days.copy()
    gdays[:2 * ESTIMATOR_WINDOW] = False      # need two complete windows
    if not gdays.any():
        raise ValueError("no day has two complete windows behind it")

    return {
        "rank_median": float(np.median(rs)),
        "rank_q1":     float(np.percentile(rs, 25)),
        "rank_q3":     float(np.percentile(rs, 75)),
        "cv_est":      cv_est * 100.0,
        "cv_true":     cv_true * 100.0,
        "agree_growth": float(np.mean((R_hat[gdays] > 1.0) == growing[gdays]) * 100.0),
        "agree_own":    float(np.mean((R_hat[days] > 1.0)
                                      == (ref_own[days] > 1.0)) * 100.0),
        "net_median":   float(np.median(np.mean(
                            np.abs(R_hat[days] - ref_net[days, None])
                            / ref_net[days, None], axis=0) * 100.0)),
        "n_days":       int(days.sum()),
    }


def compute_diagnostics_table(scenarios, gen_time_pmf):
    """{scenario label: location_diagnostics(...)} in the given column order."""
    if not scenarios:
        raise ValueError("compute_diagnostics_table needs at least one scenario")
    labels = [lab for lab, _, _ in scenarios]
    if len(set(labels)) != len(labels):
        raise ValueError(f"scenario labels must be unique, got {labels}")
    return {lab: location_diagnostics(sim, gen_time_pmf)
            for lab, sim, _city in scenarios}


# (csv key, plain label, LaTeX label)
_DIAG_ROWS = [
    ("rank_median",  "Rank correlation with true R_out, median",
     r"Rank correlation with $R^{j}_{\mathrm{out}}$, median"),
    ("rank_q1",      "Rank correlation, lower quartile", r"\quad lower quartile"),
    ("rank_q3",      "Rank correlation, upper quartile", r"\quad upper quartile"),
    ("cv_est",       "Spread of the estimates (CV %)",
     r"Spread of the estimates (CV, \%)"),
    ("cv_true",      "True spread of R_out (CV %)",
     r"True spread of $R^{j}_{\mathrm{out}}$ (CV, \%)"),
    ("agree_growth", "Agrees with local cases rising (%)",
     r"Agrees with local cases rising (\%)"),
    ("agree_own",    "Agrees with own R_out > 1 (%)",
     r"Agrees with own $R^{j}_{\mathrm{out}}>1$ (\%)"),
    ("net_median",   "Distance from the network value (%), control",
     r"Distance from $\rho(\mathbf{R})$ (\%), control"),
]


def write_diagnostics_table(table, out_dir, prefix="location_diagnostics"):
    """One CSV and one .tex for the three diagnostics.  Returns the paths."""
    os.makedirs(out_dir, exist_ok=True)
    labels = list(table)
    csv_path = os.path.join(out_dir, f"{prefix}.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["diagnostic"] + labels)
        for key, plain, _ in _DIAG_ROWS:
            w.writerow([plain] + [f"{table[l][key]:.2f}" for l in labels])
        w.writerow(["Days used"] + [table[l]["n_days"] for l in labels])

    tex_path = os.path.join(out_dir, f"{prefix}.tex")
    with open(tex_path, "w", encoding="utf-8") as fh:
        fh.write("% Auto-generated by mobility_rt.location_errors - do not edit by hand.\n")
        fh.write("% Requires \\usepackage{booktabs}.\n")
        fh.write("\\begin{table}[htbp]\n\\centering\n\\small\n")
        fh.write("\\caption{What a panel of per-location independent estimates is, and "
                 "is not, good for, over growth-phase days on which every location has "
                 "an estimate. The rank correlation is computed across locations on "
                 "each day and summarised over days. Agreement with local cases rising "
                 "uses the week-on-week change in windowed case totals. A rank "
                 "correlation at or below zero means ranking locations by their "
                 "estimates does not identify where transmission is highest.}\n")
        fh.write(f"\\label{{tab:{prefix}}}\n")
        fh.write("\\begin{tabular}{l" + "r" * len(labels) + "}\n\\toprule\n")
        fh.write("Diagnostic & " + " & ".join(_tex_escape(l) for l in labels)
                 + " \\\\\n\\midrule\n")
        for key, _, tex in _DIAG_ROWS:
            fh.write(f"{tex} & "
                     + " & ".join(f"{table[l][key]:.2f}" for l in labels) + " \\\\\n")
        fh.write("\\midrule\nDays used & "
                 + " & ".join(str(table[l]["n_days"]) for l in labels) + " \\\\\n")
        fh.write("\\bottomrule\n\\end{tabular}\n\\end{table}\n")
    return [csv_path, tex_path]


def build_diagnostics_table(scenarios, gen_time_pmf, out_dir,
                            prefix="location_diagnostics", verbose=True):
    """Compute and write the diagnostics table; return (table, paths)."""
    table = compute_diagnostics_table(scenarios, gen_time_pmf)
    paths = write_diagnostics_table(table, out_dir, prefix=prefix)
    if verbose:
        labels = list(table)
        width = max(len(p) for _, p, _ in _DIAG_ROWS)
        print("\n  What a panel of per-location estimates is good for")
        print("    " + " " * width + "".join(f"{l[:14]:>16s}" for l in labels))
        for key, plain, _ in _DIAG_ROWS:
            print(f"    {plain:<{width}}"
                  + "".join(f"{table[l][key]:15.2f} " for l in labels))
        for p in paths:
            print(f"  Saved {p}")
    return table, paths
