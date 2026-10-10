"""mobility_rt.combined_table — the single manuscript table.

Brings the two halves of the analysis into one table, on one set of days:

  Network level   how well each weighted aggregate of per-location numbers
                  recovers the true rho(R(t)).  Nine schemes: four weightings of
                  the per-location independent estimates, the same four of the
                  model's outward reproduction numbers, and pooled incidence.
  Location level  what a SINGLE location's own independent estimate tells you.
                  It tracks growth but cannot attribute it, so it recovers the
                  network value and not the location's own transmissibility.

Every figure is lifted from error_tables and location_errors, which each verify
their own arithmetic; the only arithmetic here is the median ACROSS LOCATIONS of the
two discrepancy rows, which reduces a per-location array to the single number the
table prints.  The builder compares the day masks of the two halves, not just their
counts, and refuses to emit anything unless they are identical.  That shared day set
is what makes it legitimate to read a network row against a location row.

The table reports the GROWTH PHASE, rho(R(t)) > 1, because that is where an error
can change a decision.  error_tables also writes a full-period version.
"""
import csv
import os

import numpy as np

from mobility_rt.error_tables import (ESTIMATOR_WINDOW, SCHEMES, _tex_escape,
                                      compute_error_table)
from mobility_rt.location_errors import (compute_diagnostics_table,
                                         compute_location_table)

PERIOD = "growth"

# Location-level rows: (key, source, plain label, LaTeX label, decimals)
#   source "diag" -> compute_diagnostics_table, "loc" -> compute_location_table
_LOCATION_ROWS = [
    ("rank_median", "diag",
     "Rank correlation with own R_out (median over days)",
     r"Rank correlation with own $R^{j}_{\mathrm{out}}$", 2),
    ("cv_est", "diag",
     "Spread of the estimates across locations (CV, %)",
     r"Spread of the estimates (CV, \%)", 2),
    ("cv_true", "diag",
     "True spread of R_out across locations (CV, %)",
     r"True spread of $R^{j}_{\mathrm{out}}$ (CV, \%)", 2),
    ("agree_growth", "diag",
     "Agrees with local cases rising (%)",
     r"Agrees with local cases rising (\%)", 2),
    ("agree_own", "diag",
     "Agrees with own R_out above 1 (%)",
     r"Agrees with own $R^{j}_{\mathrm{out}}>1$ (\%)", 2),
    ("own", "loc",
     "Distance from own R_out (median over locations, %)",
     r"Distance from own $R^{j}_{\mathrm{out}}$ (\%)", 2),
    ("net", "loc",
     "Distance from network R(t) (median over locations, %)",
     r"Distance from $\rho(\mathbf{R}(t))$ (\%)", 2),
]


def _check_consistent(labels, net_table, loc_table, diag_table):
    """Refuse to build unless the three sources agree on scenarios AND days.

    Day MASKS are compared where all three carry one, so that two halves which
    happen to score the same NUMBER of different days cannot slip through: the two
    masks are built independently (location_errors additionally requires every
    location's own trailing-mean R_out to clear RHO_FLOOR), so equality is a real
    check and not a tautology.  Counts are compared as a fallback, for a table dict
    assembled by hand in a test.
    """
    for name, tbl in (("network", net_table[PERIOD]["n_days"]),
                      ("location", {k: v["n_days"] for k, v in loc_table.items()}),
                      ("diagnostics", {k: v["n_days"] for k, v in diag_table.items()})):
        if set(tbl) != set(labels):
            raise ValueError(f"the {name} table covers {sorted(tbl)}, "
                             f"expected {sorted(labels)}")
    for lab in labels:
        n_net = net_table[PERIOD]["n_days"][lab]
        n_loc = loc_table[lab]["n_days"]
        n_dia = diag_table[lab]["n_days"]
        if not (n_net == n_loc == n_dia):
            raise ValueError(
                f"{lab}: the halves used different days "
                f"(network {n_net}, location {n_loc}, diagnostics {n_dia}); "
                "a combined table would invite comparisons across different "
                "day sets")
        present = [m for m in (net_table[PERIOD].get("days", {}).get(lab),
                               loc_table[lab].get("days"),
                               diag_table[lab].get("days")) if m is not None]
        if len(present) > 1 and not all(np.array_equal(present[0], m)
                                        for m in present[1:]):
            raise ValueError(
                f"{lab}: the halves scored the same NUMBER of days ({n_net}) but "
                "not the same days; a combined table would invite comparisons "
                "across different day sets")


def compute_combined_table(scenarios, gen_time_pmf, net_table=None,
                           loc_table=None, diag_table=None):
    """Assemble the manuscript table.  Pass precomputed halves to avoid redoing them.

    Returns {"labels": [...], "network": {scheme: {label: (mre, bias)}},
             "location": {row key: {label: value}}, "n_days": {label: int}}.
    """
    if not scenarios:
        raise ValueError("compute_combined_table needs at least one scenario")
    labels = [lab for lab, _, _ in scenarios]
    if len(set(labels)) != len(labels):
        raise ValueError(f"scenario labels must be unique, got {labels}")

    if net_table is None:
        net_table = compute_error_table(scenarios, gen_time_pmf)
    if loc_table is None:
        loc_table = compute_location_table(scenarios, gen_time_pmf)
    if diag_table is None:
        diag_table = compute_diagnostics_table(scenarios, gen_time_pmf)
    _check_consistent(labels, net_table, loc_table, diag_table)

    network = {sch.key: {lab: (net_table[PERIOD]["mre"][sch.key][lab],
                               net_table[PERIOD]["bias"][sch.key][lab])
                         for lab in labels}
               for sch in SCHEMES}
    location = {}
    for key, source, _, _, _ in _LOCATION_ROWS:
        if source == "diag":
            location[key] = {lab: float(diag_table[lab][key]) for lab in labels}
        else:                       # median across locations of the discrepancy
            location[key] = {lab: float(np.median(loc_table[lab][key]["mre"]))
                             for lab in labels}
    return {"labels": labels, "network": network, "location": location,
            "n_days": {lab: int(net_table[PERIOD]["n_days"][lab]) for lab in labels}}


def _num(x, nd=2, signed=False):
    """Plain text, for the CSV."""
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return ""
    return f"{x:+.{nd}f}" if signed else f"{x:.{nd}f}"


def _tex_num(x, nd=2, signed=False):
    """Math mode, for the .tex, so a negative renders as a true minus sign and
    the digits share the same width down a column."""
    t = _num(x, nd, signed)
    return "" if t == "" else f"${t}$"


def write_combined_table(table, out_dir, prefix="manuscript_table"):
    """Write the CSV and the .tex.  Returns the paths."""
    os.makedirs(out_dir, exist_ok=True)
    labels = table["labels"]

    csv_path = os.path.join(out_dir, f"{prefix}.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["section", "metric", "statistic", "unit"] + labels)
        for sch in SCHEMES:
            for stat, i in (("mean relative error", 0), ("mean signed bias", 1)):
                w.writerow(["Network-level aggregation", sch.plain, stat, "%"]
                           + [_num(table["network"][sch.key][l][i],
                                   signed=(i == 1)) for l in labels])
        for key, _, plain, _, nd in _LOCATION_ROWS:
            unit = "" if key == "rank_median" else "%"
            w.writerow(["Location-level estimate", plain, "value", unit]
                       + [_num(table["location"][key][l], nd) for l in labels])
        w.writerow(["", "Days used", "count", ""]
                   + [table["n_days"][l] for l in labels])

    tex_path = os.path.join(out_dir, f"{prefix}.tex")
    ncol = 1 + len(labels)
    with open(tex_path, "w", encoding="utf-8") as fh:
        fh.write("% Auto-generated by mobility_rt.combined_table - do not edit by hand.\n")
        fh.write("% Requires \\usepackage{booktabs}.\n")
        fh.write("\\begin{table}[htbp]\n\\centering\n\\small\n"
                 "\\setlength{\\tabcolsep}{4pt}\n")
        fh.write("\\caption{Recovering the network reproduction number "
                 "$\\rho(\\mathbf{R}(t))$, and what a single location's own estimate "
                 "tells you. All figures are over growth-phase days, "
                 "$\\rho(\\mathbf{R}(t))>1$, on which every location has an estimate; "
                 "both halves use the same days. Sliding-window schemes summarise "
                 f"the preceding {ESTIMATOR_WINDOW} days and are scored against the "
                 f"matching {ESTIMATOR_WINDOW}-day trailing mean of their comparator, "
                 "so that smoothing is not charged as error. In the upper block the "
                 "signed bias is in parentheses; a mean relative error far larger "
                 "than the bias means the scheme errs in both directions. A rank "
                 "correlation at or below zero means ranking locations by their "
                 "estimates does not find where transmission is highest.}\n")
        fh.write(f"\\label{{tab:{prefix}}}\n")
        fh.write("\\begin{tabular}{l" + "r" * len(labels) + "}\n\\toprule\n")
        fh.write(" & " + " & ".join(_tex_escape(l) for l in labels) + " \\\\\n")
        fh.write("\\midrule\n")
        fh.write(f"\\multicolumn{{{ncol}}}{{l}}{{\\textit{{Network-level aggregation: "
                 f"mean relative error, \\% (signed bias)}}}} \\\\\n")
        for sch in SCHEMES:
            cells = []
            for l in labels:
                mre, bias = table["network"][sch.key][l]
                cells.append(f"{_tex_num(mre)} ({_tex_num(bias, signed=True)})"
                             if _num(mre) else "")
            fh.write(f"{sch.tex} & " + " & ".join(cells) + " \\\\\n")
        fh.write("\\midrule\n")
        fh.write(f"\\multicolumn{{{ncol}}}{{l}}{{\\textit{{Location-level: what one "
                 f"location's own estimate tells you}}}} \\\\\n")
        for key, _, _, tex, nd in _LOCATION_ROWS:
            fh.write(f"{tex} & "
                     + " & ".join(_tex_num(table["location"][key][l], nd)
                                  for l in labels)
                     + " \\\\\n")
        fh.write("\\midrule\nDays used & "
                 + " & ".join(f"${table['n_days'][l]}$" for l in labels)
                 + " \\\\\n")
        fh.write("\\bottomrule\n\\end{tabular}\n\\end{table}\n")
    return [csv_path, tex_path]


def build_combined_table(scenarios, gen_time_pmf, out_dir, prefix="manuscript_table",
                         net_table=None, loc_table=None, diag_table=None,
                         verbose=True):
    """Assemble and write the manuscript table; return (table, paths)."""
    table = compute_combined_table(scenarios, gen_time_pmf, net_table,
                                   loc_table, diag_table)
    paths = write_combined_table(table, out_dir, prefix=prefix)
    if verbose:
        labels = table["labels"]
        width = max([len(s.plain) for s in SCHEMES]
                    + [len(r[2]) for r in _LOCATION_ROWS])
        print("\n  Manuscript table (growth phase)")
        print("    " + " " * width + "".join(f"{l[:13]:>15s}" for l in labels))
        print("    Network-level: mean relative error % (signed bias)")
        for sch in SCHEMES:
            print(f"    {sch.plain:<{width}}" + "".join(
                f"{_num(table['network'][sch.key][l][0]):>8s}"
                f"({_num(table['network'][sch.key][l][1], signed=True):>6s})"
                for l in labels))
        print("    Location-level:")
        for key, _, plain, _, nd in _LOCATION_ROWS:
            print(f"    {plain:<{width}}" + "".join(
                f"{_num(table['location'][key][l], nd):>15s}" for l in labels))
        print(f"    {'Days used':<{width}}"
              + "".join(f"{table['n_days'][l]:>15d}" for l in labels))
        for p in paths:
            print(f"  Saved {p}")
    return table, paths
