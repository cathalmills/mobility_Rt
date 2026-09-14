"""mobility_rt.gt_spatial — does the generation time vary across space?

The heterogeneous scenarios (A-het / B-het / C-het) give every location its own
contact rate c[l] and its own per-contact transmissibility m[k].  That raises an
obvious question about the generation time, and the answer has two halves that must
not be confused with each other.

FORWARD (intrinsic) generation time.  The kernel is
    K_{kj}(t,a_E) = m[k] · S_j(t) · base_K^c[k,j] · p(a_E),
and c[l] and m[k] enter only through the factor m[k]·base_K^c[k,j], which carries no
dependence on infection age a_E.  Normalising by R_{kj}(t) therefore cancels them
exactly along with S_j, leaving g_{kj}(t,a_E) = p(a_E)/∫p for every pair and every
time.  Spatial heterogeneity in contact rate and transmissibility changes WHO infects
WHOM and HOW MUCH; it does not change WHEN, so the forward generation time is as
universal in the heterogeneous scenarios as in the homogeneous ones.  It would only
vary across space if the infectiousness PROFILE ITSELF were location-dependent,
p_k(a_E) with a different shape in different places, which this model does not have.

REALISED (backward) generation interval.  What an analyst reconstructs from observed
transmission pairs is the time since the infector's own infection, which reweights
p(a_E) by the infector's infection-age composition.  That composition is a function
of the local epidemic history, so the realised interval IS location-specific: shorter
than the intrinsic mean while a location is growing, longer while it is declining.
This varies across space in the homogeneous scenarios too — heterogeneity modulates
it by desynchronising local epidemics, it does not create it.

This module estimates both halves and writes the result as a table, so the claim of
universality is a measured quantity rather than an assertion.  The arithmetic lives
here (no matplotlib); the companion figure is
mobility_rt.plotting.figures_si.plot_SI_gt_heterogeneity.
"""
import csv
import os

import numpy as np

from mobility_rt.config import R_MIN_WINDOW_CASES
from mobility_rt.error_tables import _tex_escape
from mobility_rt.generation_time import realised_generation_times


def universal_gt(p_aE):
    """The intrinsic forward GT, p(a_E)/∫p renormalised over the a_E >= 1 support.

    a_E = 0 is excluded because compute_R_matrix and the forward simulator both
    integrate over a_E >= 1 (no same-day transmission), so this is the shape the
    kernel actually normalises to — comparing against the full-support pmf instead
    would report a spurious deviation of order p(0) plus a ~2e-5 d mean shift.
    """
    p = np.asarray(p_aE, dtype=float).copy()
    p[0] = 0.0
    total = p.sum()
    if total <= 0:
        raise ValueError("p_aE carries no mass at ages a_E >= 1")
    return p / total


def forward_gt_dispersion(gt_snaps, p_aE):
    """How far the forward GT departs from universality, over a scenario's snapshots.

    gt_snaps is the {"early"/"peak"/"late": compute_generation_times(...)} dict the
    pipeline already builds for each scenario.  Every pair (k,j) with R_{kj} > 0 at
    every snapshot is compared with the universal shape, so a location-dependent
    generation time could not hide in a single pair or a single phase.

    Returns the sup-norm deviation of the pairwise, outward, inward and meeting-site
    GTs, and the range of their means in days (max - min over pairs and snapshots).
    """
    g_univ = universal_gt(p_aE)
    days   = np.arange(len(g_univ), dtype=float)
    mu_univ = float(days @ g_univ)

    sup = 0.0
    mus = []
    n_pairs = 0
    for snap in gt_snaps.values():
        R_mat = snap["R_matrix"]
        N = R_mat.shape[0]
        for k in range(N):
            for j in range(N):
                if R_mat[k, j] <= 1e-15:
                    continue
                g = snap["g_pairwise"][:, k, j]
                sup = max(sup, float(np.abs(g - g_univ).max()))
                mus.append(float(days @ g))
                n_pairs += 1
        # Marginal and meeting-site GTs are separate normalisations of the same
        # kernel, so they are checked too rather than assumed to follow.
        for key, R_marg in (("g_outward", R_mat.sum(axis=1)),
                            ("g_inward",  R_mat.sum(axis=0)),
                            ("g_meeting", snap["R_meeting"])):
            G = snap[key]
            for l in range(G.shape[1]):
                if R_marg[l] > 1e-15:
                    sup = max(sup, float(np.abs(G[:, l] - g_univ).max()))
    if not mus:
        raise ValueError("no (k,j) pair with positive R in any snapshot")
    mus = np.asarray(mus)
    return {"sup_deviation":  sup,
            "mean_gt":        float(mus.mean()),
            "mean_gt_range":  float(mus.max() - mus.min()),
            "universal_mean": mu_univ,
            "n_pairs":        n_pairs}


def epidemic_window(incidence, min_cases=R_MIN_WINDOW_CASES):
    """Boolean mask of the days carrying at least one case network-wide.

    The deterministic solver leaves fractions of a case in the far tail, where the
    realised generation interval is numerically real but epidemiologically
    meaningless.  Reusing the estimator's one-case floor keeps this window the same
    observability threshold used everywhere else rather than a new free choice.
    """
    return np.asarray(incidence, dtype=float).sum(axis=1) >= min_cases


def realised_gt_dispersion(sim, p_aE, phase_days=None):
    """Spatial dispersion of the REALISED (backward) generation interval over time.

    Returns the per-location and network backward GT means, the across-location
    spread (max - min, in days) on each day of the epidemic window, where that
    spread peaks, and the values at the requested phase days.
    """
    rg   = realised_generation_times(sim["E_pde_state"], sim["incidence_matrix"], p_aE)
    mu_j = rg["mean_infectee"]         # (T, N) realised interval of cases arising in j
    mu_k = rg["mean_infector"]         # (T, N) realised interval of transmissions from k
    mu_net = rg["mean_network"]        # (T,)
    mask = epidemic_window(sim["incidence"])

    # max - min across locations, computed without nan-reductions so that a day on
    # which no location has transmitted yet stays NaN instead of raising an all-NaN
    # slice warning and returning a meaningless 0.
    valid = ~np.isnan(mu_j)
    hi = np.where(valid, mu_j, -np.inf).max(axis=1)
    lo = np.where(valid, mu_j,  np.inf).min(axis=1)
    spread = np.where(valid.any(axis=1) & mask, hi - lo, np.nan)
    if np.all(np.isnan(spread)):
        raise ValueError("no day of this run carries a case; nothing to summarise")
    argmax_day = int(np.nanargmax(spread))

    phases = {}
    for name, day in (phase_days or {}).items():
        phases[name] = {"day": int(day),
                        "network_mean": float(mu_net[day]),
                        "spread": float(spread[day])}
    return {"mean_infectee": mu_j, "mean_infector": mu_k, "mean_network": mu_net,
            "spread": spread, "window": mask,
            "max_spread": float(np.nanmax(spread)), "max_spread_day": argmax_day,
            "median_spread": float(np.nanmedian(spread)),
            "phases": phases}


def phase_days(sim, T=None):
    """The early / peak / late days the pipeline uses for every scenario's snapshots."""
    inc = sim["incidence"]
    T   = inc.shape[0] if T is None else T
    pk  = int(inc.sum(axis=1).argmax())
    return {"early": max(1, pk // 3), "peak": pk, "late": min(T - 1, pk + 30)}


# (csv key, plain label, LaTeX label, formatter)
_ROWS = [
    ("forward_mean", "Forward GT mean (d)",
     r"Forward GT mean (d)", lambda v: f"{v:.4f}"),
    ("forward_range", "Forward GT mean, spread over pairs (d)",
     r"\quad spread over pairs (d)", lambda v: f"{v:.2e}"),
    ("forward_sup", "Forward GT max |g - p/int p|",
     r"\quad $\max|g-p/\!\int\!p|$", lambda v: f"{v:.2e}"),
    ("realised_early", "Realised GT mean, early (d)",
     r"Realised GT mean, early (d)", lambda v: f"{v:.3f}"),
    ("realised_peak", "Realised GT mean, peak (d)",
     r"\quad at peak (d)", lambda v: f"{v:.3f}"),
    ("realised_late", "Realised GT mean, late (d)",
     r"\quad late (d)", lambda v: f"{v:.3f}"),
    ("spread_early", "Realised GT spatial spread, early (d)",
     r"Spatial spread, early (d)", lambda v: f"{v:.3f}"),
    ("spread_peak", "Realised GT spatial spread, peak (d)",
     r"\quad at peak (d)", lambda v: f"{v:.3f}"),
    ("spread_late", "Realised GT spatial spread, late (d)",
     r"\quad late (d)", lambda v: f"{v:.3f}"),
    ("spread_max", "Realised GT spatial spread, max over window (d)",
     r"\quad maximum (d)", lambda v: f"{v:.3f}"),
    ("spread_max_day", "Day of maximum spatial spread",
     r"\quad day of maximum", lambda v: f"{v:.0f}"),
]


def build_gt_spatial_table(scenarios, p_aE, out_dir="out_figs/tables",
                           prefix="gt_spatial_heterogeneity", verbose=True):
    """Estimate forward and realised GT dispersion per scenario; write CSV + .tex.

    scenarios is a list of (label, sim, gt_snaps) in the column order wanted, e.g.
    the three baseline runs each followed by its heterogeneous counterpart.  Returns
    (table, paths).
    """
    labels = [lab for lab, _, _ in scenarios]
    dupes  = sorted({l for l in labels if labels.count(l) > 1})
    if dupes:
        raise ValueError(f"duplicate scenario label(s) {dupes}: columns would collide")

    table = {}
    for label, sim, gt_snaps in scenarios:
        fwd = forward_gt_dispersion(gt_snaps, p_aE)
        days = phase_days(sim)
        rea = realised_gt_dispersion(sim, p_aE, phase_days=days)
        table[label] = {
            "forward_mean":   fwd["mean_gt"],
            "forward_range":  fwd["mean_gt_range"],
            "forward_sup":    fwd["sup_deviation"],
            "realised_early": rea["phases"]["early"]["network_mean"],
            "realised_peak":  rea["phases"]["peak"]["network_mean"],
            "realised_late":  rea["phases"]["late"]["network_mean"],
            "spread_early":   rea["phases"]["early"]["spread"],
            "spread_peak":    rea["phases"]["peak"]["spread"],
            "spread_late":    rea["phases"]["late"]["spread"],
            "spread_max":     rea["max_spread"],
            "spread_max_day": float(rea["max_spread_day"]),
        }
        if verbose:
            print(f"  {label:24s} forward GT {fwd['mean_gt']:.4f} d "
                  f"(max |g-p/∫p| = {fwd['sup_deviation']:.1e}, "
                  f"spread over {fwd['n_pairs']} pairs = {fwd['mean_gt_range']:.1e} d)"
                  f"  |  realised GT spatial spread: peak "
                  f"{rea['phases']['peak']['spread']:.3f} d, "
                  f"max {rea['max_spread']:.3f} d on day {rea['max_spread_day']}")

    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, f"{prefix}.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["quantity"] + labels)
        for key, plain, _, fmt in _ROWS:
            w.writerow([plain] + [fmt(table[l][key]) for l in labels])

    tex_path = os.path.join(out_dir, f"{prefix}.tex")
    with open(tex_path, "w", encoding="utf-8") as fh:
        fh.write("% Auto-generated by mobility_rt.gt_spatial - do not edit by hand.\n")
        fh.write("% Requires \\usepackage{booktabs}.\n")
        fh.write("\\begin{table}[htbp]\n\\centering\n\\small\n")
        fh.write("\\caption{Generation time under spatially heterogeneous contact "
                 "rates and transmissibility. The forward generation time built into "
                 "the kernel is universal to machine precision in every scenario, "
                 "heterogeneous or not, because the contact multiplier $c_l$ and the "
                 "transmissibility multiplier $m_k$ carry no dependence on infection "
                 "age and so cancel in $g_{kj}=K_{kj}/R_{kj}$. What varies across "
                 "space is the realised (backward) interval, which reweights "
                 "$p(a_E)$ by the infector's infection-age composition and therefore "
                 "tracks each location's own epidemic phase.}\n")
        fh.write(f"\\label{{tab:{prefix}}}\n")
        fh.write("\\begin{tabular}{l" + "r" * len(labels) + "}\n\\toprule\n")
        fh.write("Quantity & " + " & ".join(_tex_escape(l) for l in labels) + " \\\\\n")
        fh.write("\\midrule\n")
        for i, (key, _, tex, fmt) in enumerate(_ROWS):
            if i in (3, 6):
                fh.write("\\midrule\n")
            fh.write(f"{tex} & "
                     + " & ".join(fmt(table[l][key]) for l in labels) + " \\\\\n")
        fh.write("\\bottomrule\n\\end{tabular}\n\\end{table}\n")

    if verbose:
        print(f"  Wrote {csv_path}")
        print(f"  Wrote {tex_path}")
    return table, [csv_path, tex_path]
