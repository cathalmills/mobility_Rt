"""mobility_rt.error_tables — mean relative error of every aggregation scheme.

Tabulates how far each naive network-level reproduction-number estimator sits from
the true system value rho(R(t)), for every scenario.

PERIODS scored (TABLE_PERIODS are the ones written to disk):
  growth   days on which the epidemic is actually growing, rho(R(t)) > 1.
           This is the policy-relevant window: it is where an estimator crossing
           the threshold at the wrong time changes a decision.
  decline  the complement, rho(R(t)) <= 1.  Computed and shared with the bias
           figures so figure and table use one definition; not written as its own
           table.
  full     every day above the numerical noise floor.  Roughly half of these are
           the burnt-out tail, where every scheme agrees, so `full` is
           systematically more flattering than `growth`.

MATCHED REFERENCE.  A scheme is compared with the reference at ITS OWN time
resolution.  A Cori/EpiEstim estimate summarises the ESTIMATOR_WINDOW days ending
at t, so it is scored against the matching trailing mean of rho; a scheme carrying
no window is scored against rho itself.  rho oscillates with the day-of-week
commuting cycle, so scoring a smoothed estimate against an unsmoothed reference
would charge it for its smoothing and its delay rather than for spatial aggregation
(decision D8).  Which schemes are windowed is declared once, in SCHEMES.

COMMON DAY MASK.  Within a scenario every scheme is scored on exactly the same
days: a day counts only when every scheme has a finite estimate there AND both
references are finite and above the noise floor.  Without this the windowed
schemes, which are undefined until the windowed incidence clears the reporting
threshold, would be scored on a different and easier subset than the model-based
ones, and the columns would not be comparable.

Metrics per scheme, both in per cent, where `ref` is that scheme's matched
reference:
  mre   mean over the masked days of |estimator - ref| / ref.  Never cancels.
  bias  mean over the masked days of (estimator - ref) / ref.  Signed, so it
        does cancel; reported alongside because a near-zero bias with a large
        mre means the scheme is wrong in both directions rather than accurate.
"""
import csv
import os
from collections import namedtuple

import numpy as np

from mobility_rt.estimators import WINDOW as _EST_WINDOW, _compute_naive_suite


# Numerical noise floor on rho(R(t)); the same value the bias figures use to drop
# days where the epidemic has effectively ended and relative error would explode.
RHO_FLOOR = 0.05

# Width of the sliding window used by the Cori/EpiEstim estimators.  A windowed
# estimate summarises the window [t-w+1, t], so it is scored against the matching
# trailing mean of rho(R(t)), not against rho at day t alone (decision D8).  The
# delay a window introduces is reported separately, as a threshold-crossing lag.
ESTIMATOR_WINDOW = _EST_WINDOW

# windowed=True  -> scored against the ESTIMATOR_WINDOW trailing mean of rho
# windowed=False -> scored against rho(R(t)) itself (no smoothing to match)
Scheme = namedtuple("Scheme", "key plain tex windowed")

SCHEMES = [
    Scheme("agg",    "Aggregate incidence",                    r"$\hat{R}_{\mathrm{agg}}$",                        True),
    Scheme("ind_am", "Arithmetic mean, independent",           r"$R^{j}_{\mathrm{ind},\mathrm{am}}$",              True),
    Scheme("ind_Lw", "Infectiousness-weighted, independent",   r"$R^{j}_{\mathrm{ind},\Lambda\mathrm{w}}$",        True),
    Scheme("ind_pw", "Population-weighted, independent",       r"$R^{j}_{\mathrm{ind},\mathrm{pw}}$",              True),
    Scheme("ind_Rw", "Relative-transmissibility-weighted, independent",
           r"$R^{j}_{\mathrm{ind},\mathrm{Rw}}$",                                                                 True),
    Scheme("out_am", "Arithmetic mean, outward",               r"$\hat{R}^{\mathrm{out}}_{\mathrm{am}}$",          False),
    Scheme("out_Lw", "Infectiousness-weighted, outward",       r"$\hat{R}^{\mathrm{out}}_{\Lambda\mathrm{w}}$",    False),
    Scheme("out_pw", "Population-weighted, outward",           r"$\hat{R}^{\mathrm{out}}_{\mathrm{pw}}$",          False),
    Scheme("out_Rw", "Relative-transmissibility-weighted, outward (= E(t))",
           r"$\hat{R}^{\mathrm{out}}_{\mathrm{Rw}}=\mathcal{E}(t)$",                                              False),
]

# Periods the module can score.  "decline" is not written as its own table but is
# shared with the bias figures so that figure and table use identical definitions.
PERIODS = ("growth", "decline", "full")

# The subset written to disk as CSV/LaTeX.
TABLE_PERIODS = ("growth", "full")


def trailing_mean(x, window=ESTIMATOR_WINDOW):
    """Trailing mean over [t-window+1, t], the quantity a windowed estimate targets.

    Leading days average over fewer observations, matching the estimator, which
    also has less history to draw on there.  NaNs propagate: a window containing a
    NaN yields NaN, so a scheme is never scored against a partly-undefined
    reference.
    """
    x = np.asarray(x, dtype=float)
    if x.ndim != 1:
        raise ValueError(f"trailing_mean expects a 1-D series, got shape {x.shape}")
    if int(window) != window or window < 1:
        raise ValueError(f"window must be a positive whole number, got {window!r}")
    window = int(window)
    out = np.full(x.shape, np.nan)
    for t in range(len(x)):
        out[t] = x[max(0, t - window + 1):t + 1].mean()
    return out


def references(rho, window=ESTIMATOR_WINDOW):
    """{True: trailing mean of rho, False: rho} — the reference each scheme is due."""
    rho = np.asarray(rho, dtype=float)
    return {True: trailing_mean(rho, window), False: rho}


def _period_masks(suite, targets=("rho_t",)):
    """Common per-day masks for one scenario: {period: boolean array}.

    A day is usable when rho(R(t)) is above the noise floor, EVERY scheme has a finite
    estimate there, and, for each target series named in `targets` (keys of the
    suite, "rho_t" and optionally "E_t"), both references a scheme might be scored
    against (the target itself and its trailing mean) are finite and above the noise
    floor.  Requiring all of them up front is what makes the "identical set of days"
    guarantee exact: if a windowed scheme silently dropped days its reference did
    not cover, its column would be scored on an easier subset than the rest and the
    reported day count would overstate what was used.  It also keeps the relative
    error's denominator safely away from zero.

    The periods are always defined by rho(R(t)): growth is rho > 1.  With the default
    targets these are exactly the masks behind Tables SI1-SI2.
    """
    rho    = np.asarray(suite["rho_t"], dtype=float)
    usable = rho > RHO_FLOOR
    for sch in SCHEMES:
        usable &= np.isfinite(np.asarray(suite[sch.key], dtype=float))
    for key in targets:
        for ref in references(suite[key]).values():
            ref = np.asarray(ref, dtype=float)
            usable &= np.isfinite(ref) & (ref > RHO_FLOOR)
    return {"full":    usable,
            "growth":  usable & (rho > 1.0),
            "decline": usable & (rho <= 1.0)}


def threshold_crossing_day(series):
    """First day after which `series` stays at or below 1; None if it never does.

    This is the decision-relevant crossing, not the first dip below 1: day-of-week
    variation makes R(t) flicker across the threshold, and a one-day dip is not the
    moment an epidemic came under control.  Defined as the first FINITE day after
    the last day the series exceeds 1, so NaN gaps in an estimator do not shift it.

    Comparing an estimator's crossing with rho(R(t))'s converts a relative error into
    the quantity a decision-maker cares about: how many days early or late you would
    have declared the epidemic under control.
    """
    x = np.asarray(series, dtype=float)
    finite = np.isfinite(x)
    if not finite.any():
        return None
    above = finite & (x > 1.0)
    if not above.any():
        return int(np.argmax(finite))          # at or below 1 from the first finite day
    last_above = int(np.max(np.where(above)[0]))
    later = np.where(finite[last_above + 1:])[0]
    if later.size == 0:
        return None                            # never settles below 1 in this window
    return int(last_above + 1 + later[0])


def scored_crossing_day(series, mask):
    """threshold_crossing_day of `series`, counting only the days where mask is True.

    Decision D10: an estimator's crossing is taken on the days the tables and figures
    score (every scheme and reference defined, rho above the noise floor).  In the
    far tail a sliding-window estimate rests on fractions of a case, the prior
    dominates and the estimate can drift back above 1; those days would otherwise
    decide when the estimate "settles" below 1.
    """
    x = np.asarray(series, dtype=float)
    return threshold_crossing_day(np.where(np.asarray(mask, dtype=bool), x, np.nan))


def compute_error_table(scenarios, gen_time_pmf):
    """Mean relative error of every scheme, for every scenario, in both periods.

    scenarios : list of (label, sim, city) in the column order wanted.
    Returns {period: {"labels": [...], "n_days": {label: int},
                      "mre": {scheme: {label: pct}},
                      "bias": {scheme: {label: pct}}}}.
    """
    if not scenarios:
        raise ValueError("compute_error_table needs at least one scenario")
    labels = [lab for lab, _, _ in scenarios]
    if len(set(labels)) != len(labels):
        raise ValueError(f"scenario labels must be unique, got {labels}")

    out = {p: {"labels": labels, "n_days": {},
               "mre": {sch.key: {} for sch in SCHEMES},
               "bias": {sch.key: {} for sch in SCHEMES}} for p in PERIODS}

    for label, sim, city in scenarios:
        suite = _compute_naive_suite(sim, city, gen_time_pmf)
        rho   = np.asarray(suite["rho_t"], dtype=float)
        refs  = references(rho)
        masks = _period_masks(suite)
        for period in PERIODS:
            m = masks[period]
            n = int(m.sum())
            out[period]["n_days"][label] = n
            if n == 0:
                print(f"  WARNING: no usable days for {label} ({period}); "
                      f"reporting NaN")
            for sch in SCHEMES:
                # Each scheme is scored against the reference matched to its own
                # smoothing.  The MASK is common to every scheme and already
                # guarantees both references are usable there, so n_days above is
                # exactly the number of days behind each figure in this row.
                ref = refs[sch.windowed]
                if n == 0:
                    mre = bias = np.nan
                else:
                    rel = (np.asarray(suite[sch.key], dtype=float)[m] - ref[m]) / ref[m]
                    mre  = float(np.mean(np.abs(rel)) * 100.0)
                    bias = float(np.mean(rel) * 100.0)
                out[period]["mre"][sch.key][label]  = mre
                out[period]["bias"][sch.key][label] = bias
    return out


def _fmt(x, nd=2):
    return "" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{nd}f}"


# Single-pass character map: sequential str.replace would re-escape the braces
# that \textbackslash{} and friends introduce.
_TEX_ESCAPES = {
    "\\": r"\textbackslash{}",
    "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#", "_": r"\_",
    "{": r"\{", "}": r"\}",
    "~": r"\textasciitilde{}", "^": r"\textasciicircum{}",
}


def _tex_escape(s):
    """Escape a plain-text scenario label for LaTeX (labels carry no math)."""
    return "".join(_TEX_ESCAPES.get(ch, ch) for ch in str(s))


def write_error_tables(table, out_dir, prefix="estimator_error"):
    """Write one CSV and one .tex per period.  Returns the list of paths written."""
    os.makedirs(out_dir, exist_ok=True)
    written = []
    for period in TABLE_PERIODS:
        d      = table[period]
        labels = d["labels"]

        csv_path = os.path.join(out_dir, f"{prefix}_{period}.csv")
        with open(csv_path, "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["scheme", "scheme_label"]
                       + [f"{lab} | MRE %"  for lab in labels]
                       + [f"{lab} | bias %" for lab in labels])
            for sch in SCHEMES:
                w.writerow([sch.key, sch.plain]
                           + [_fmt(d["mre"][sch.key][lab])  for lab in labels]
                           + [_fmt(d["bias"][sch.key][lab]) for lab in labels])
            w.writerow(["n_days", "Days used"]
                       + [d["n_days"][lab] for lab in labels]
                       + [d["n_days"][lab] for lab in labels])
        written.append(csv_path)

        tex_path = os.path.join(out_dir, f"{prefix}_{period}.tex")
        period_txt = ("days on which $\\rho(\\mathbf{R}(t))>1$"
                      if period == "growth"
                      else f"all days on which $\\rho(\\mathbf{{R}}(t))>{RHO_FLOOR}$")
        best = {}
        for lab in labels:
            vals = [(d["mre"][sch.key][lab], sch.key) for sch in SCHEMES
                    if not np.isnan(d["mre"][sch.key][lab])]
            best[lab] = min(vals)[1] if vals else None
        with open(tex_path, "w", encoding="utf-8") as fh:
            fh.write("% Auto-generated by mobility_rt.error_tables - do not edit by hand.\n")
            fh.write("% Requires \\usepackage{booktabs} for \\toprule/\\midrule/\\bottomrule.\n")
            fh.write("\\begin{table}[htbp]\n\\centering\n\\small\n")
            fh.write("\\caption{Mean relative error (\\%) of each spatial aggregation scheme "
                     "against the network reproduction number, over "
                     f"{period_txt}. "
                     "All schemes are scored on an identical set of days within each scenario. "
                     f"Sliding-window schemes (the first {sum(1 for x in SCHEMES if x.windowed)} "
                     "rows) summarise the preceding "
                     f"{ESTIMATOR_WINDOW} days, so they are compared with the matching "
                     f"{ESTIMATOR_WINDOW}-day trailing mean of "
                     "$\\rho(\\mathbf{R}(t))$; the remaining rows carry no window and are "
                     "compared with $\\rho(\\mathbf{R}(t))$ itself. "
                     "The most accurate scheme in each column is in bold.}\n")
            fh.write(f"\\label{{tab:{prefix}_{period}}}\n")
            fh.write("\\begin{tabular}{l" + "r" * len(labels) + "}\n\\toprule\n")
            fh.write("Scheme & " + " & ".join(_tex_escape(l) for l in labels) + " \\\\\n")
            fh.write("\\midrule\n")
            for sch in SCHEMES:
                cells = []
                for lab in labels:
                    v = _fmt(d["mre"][sch.key][lab])
                    cells.append(f"\\textbf{{{v}}}" if sch.key == best[lab] and v else v)
                fh.write(f"{sch.tex} & " + " & ".join(cells) + " \\\\\n")
            fh.write("\\midrule\n")
            fh.write("Days used & " + " & ".join(str(d["n_days"][l]) for l in labels)
                     + " \\\\\n")
            fh.write("\\bottomrule\n\\end{tabular}\n\\end{table}\n")
        written.append(tex_path)
    return written


def build_estimator_error_tables(scenarios, gen_time_pmf, out_dir,
                                 prefix="estimator_error", verbose=True):
    """Compute and write both period tables; return (table, paths)."""
    table = compute_error_table(scenarios, gen_time_pmf)
    paths = write_error_tables(table, out_dir, prefix=prefix)
    if verbose:
        for period in TABLE_PERIODS:
            d = table[period]
            print(f"\n  Mean relative error (%), {period} period "
                  f"[days used: "
                  + ", ".join(f"{l}={d['n_days'][l]}" for l in d["labels"]) + "]")
            width = max(len(sch.plain) for sch in SCHEMES)
            print("    " + " " * width
                  + "".join(f"{l[:14]:>16s}" for l in d["labels"]))
            for sch in SCHEMES:
                print(f"    {sch.plain:<{width}}"
                      + "".join(f"{_fmt(d['mre'][sch.key][l]):>15s}%" for l in d["labels"]))
        for p in paths:
            print(f"  Saved {p}")
    return table, paths
