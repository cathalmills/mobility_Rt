#!/usr/bin/env python3
"""Numerical manifest of every pipeline scenario (docs/FIX_PLAN.md, P0.2).

A manifest records, for each scenario, the headline numbers and a SHA-256 hash of
every simulated array, plus every cell of the estimator error tables.  Comparing two
manifests shows exactly which numbers a code change moved, so no change can alter a
result without that being visible.

Usage
-----
  python3 scripts/manifest.py --out out_manifest/final.json
      Build every scenario exactly as mobility_rt.scripts.run_all does and write
      the manifest.
  python3 scripts/manifest.py --compare out_manifest/baseline.json out_manifest/final.json
      Print every entry that differs between two manifests.
"""
import argparse
import contextlib
import hashlib
import io
import json
import os
import sys

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from mobility_rt.error_tables import RHO_FLOOR, compute_error_table, threshold_crossing_day
from mobility_rt.kernel import R_outward, R_system
from mobility_rt.spectral import reactivity, sensitivity_elasticity
from mobility_rt.type_reproduction import type_reproduction_numbers

# Scenarios whose columns appear in Tables SI1-SI2, in the tables' column order.
TABLE_LABELS = ("Scenario A", "Scenario A-het", "Scenario B",
                "Scenario B-het", "Scenario C", "Scenario C-het")


def _sha256(arr):
    """Hash of the exact bytes of a float64 array (any change of any bit shows)."""
    a = np.ascontiguousarray(np.asarray(arr, dtype=np.float64))
    return hashlib.sha256(a.tobytes()).hexdigest()


def _end_of_day_susceptibles(sim):
    """End-of-day S under either simulator output convention.

    Before the F03 fix the simulator returned end-of-day S as "susceptibles"; after
    it, the same array is "S_end".  Both are hashed under one name so a baseline
    and a later manifest can be compared.
    """
    return sim["S_end"] if "S_end" in sim else sim["susceptibles"]


def summarise_scenario(sim, city):
    """Headline numbers and array hashes for one simulated scenario."""
    inc = np.asarray(sim["incidence"], dtype=float)
    R = np.asarray(sim["R_matrices"], dtype=float)
    T, N = inc.shape
    pops = np.asarray(city[1], dtype=float)

    rho = np.array([R_system(R[t]) for t in range(T)])
    sigma = np.array([reactivity(R[t])["sigma"] for t in range(T)])
    R_out = np.array([R_outward(R[t]) for t in range(T)])
    E = (R_out ** 2).sum(axis=1) / R_out.sum(axis=1)

    below = np.where(rho < 1.0)[0]
    zone = np.where((sigma > 1.0) & (rho < 1.0))[0]
    active = rho > RHO_FLOOR

    T_j = np.array([type_reproduction_numbers(R[t]) for t in range(T)])
    any_defined = np.where(np.isfinite(T_j).any(axis=1))[0]
    eps_out = np.array([sensitivity_elasticity(R[t])["elasticity"].sum(axis=1)
                        for t in range(T)])
    top3 = np.argsort(eps_out.mean(axis=0))[::-1][:3]

    return {
        "attack_rate_pct": float(inc.sum() / pops.sum() * 100.0),
        "peak_day": int(inc.sum(axis=1).argmax()),
        "rho_day0": float(rho[0]),
        "sigma_over_rho_day0": float(sigma[0] / rho[0]),
        "first_day_rho_below_1": int(below[0]) if below.size else None,
        "settled_crossing_rho": threshold_crossing_day(rho),
        "zone_sigma_gt1_rho_lt1_days": int(zone.size),
        "zone_first_day": int(zone[0]) if zone.size else None,
        "zone_last_day": int(zone[-1]) if zone.size else None,
        "mean_E_minus_rho_on_rho_gt_floor": float(np.mean((E - rho)[active])),
        "first_day_any_T_j_defined": int(any_defined[0]) if any_defined.size else None,
        "n_locations_R_jj_day0_gt_1": int((np.diag(R[0]) > 1.0).sum()),
        "top3_time_mean_infector_elasticity": [f"L{int(i) + 1}" for i in top3],
        "lambda_within_scaled": float(sim["lambda_within_scaled"]),
        "lambda_between_scaled": float(sim["lambda_between_scaled"]),
        "sums": {
            "incidence": float(inc.sum()),
            "R_matrices": float(R.sum()),
            "R_meeting_series": float(np.asarray(sim["R_meeting_series"]).sum()),
            "S_end": float(np.asarray(_end_of_day_susceptibles(sim)).sum()),
        },
        "sha256": {
            "incidence": _sha256(inc),
            "R_matrices": _sha256(R),
            "incidence_matrix": _sha256(sim["incidence_matrix"]),
            "R_meeting_series": _sha256(sim["R_meeting_series"]),
            "S_end": _sha256(_end_of_day_susceptibles(sim)),
        },
    }


def build_manifest(scenarios, gen_time_pmf):
    """scenarios: {label: (sim, city)}; must contain every label in TABLE_LABELS."""
    missing = [lab for lab in TABLE_LABELS if lab not in scenarios]
    if missing:
        raise KeyError(f"manifest needs scenario(s) {missing}")
    manifest = {"scenarios": {}}
    for label, (sim, city) in scenarios.items():
        print(f"  summarising {label} ...", flush=True)
        manifest["scenarios"][label] = summarise_scenario(sim, city)
    print("  computing estimator error tables ...", flush=True)
    manifest["error_tables"] = compute_error_table(
        [(lab, *scenarios[lab]) for lab in TABLE_LABELS], gen_time_pmf)
    return manifest


def write_manifest(manifest, path):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=1, sort_keys=True)
    print(f"  wrote {path}")


def _diff(a, b, path=""):
    """Yield (path, a_value, b_value) for every leaf that differs."""
    if isinstance(a, dict) and isinstance(b, dict):
        for key in sorted(set(a) | set(b)):
            sub = f"{path}/{key}"
            if key not in a:
                yield sub, "<absent>", b[key]
            elif key not in b:
                yield sub, a[key], "<absent>"
            else:
                yield from _diff(a[key], b[key], sub)
    elif a != b and not (isinstance(a, float) and isinstance(b, float)
                         and np.isnan(a) and np.isnan(b)):
        yield path, a, b


def compare_manifests(path_a, path_b):
    with open(path_a, encoding="utf-8") as fh:
        a = json.load(fh)
    with open(path_b, encoding="utf-8") as fh:
        b = json.load(fh)
    n = 0
    for where, va, vb in _diff(a, b):
        n += 1
        rel = ""
        if isinstance(va, (int, float)) and isinstance(vb, (int, float)) and va != 0:
            rel = f"   (relative change {(vb - va) / abs(va):+.3e})"
        print(f"{where}: {va!r} -> {vb!r}{rel}")
    print(f"{n} differing entr{'y' if n == 1 else 'ies'}")
    return n


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", help="write a manifest of the current code to this path")
    ap.add_argument("--compare", nargs=2, metavar=("A", "B"),
                    help="print every entry that differs between two manifests")
    args = ap.parse_args()
    if args.compare:
        compare_manifests(*args.compare)
    if args.out:
        from mobility_rt.scripts.run_all import build_scenarios
        with contextlib.redirect_stdout(io.StringIO()):
            sc = build_scenarios()
        scenarios = {
            "Scenario A":        (sc["sim_A"],        sc["city_A"]),
            "Scenario A-static": (sc["sim_A_static"], sc["city_A"]),
            "Scenario A-het":    (sc["sim_A_het"],    sc["city_A"]),
            "Scenario B":        (sc["sim_B"],        sc["city_B"]),
            "Scenario B-het":    (sc["sim_B_het"],    sc["city_B"]),
            "Scenario C":        (sc["sim_C"],        sc["city_C"]),
            "Scenario C-het":    (sc["sim_C_het"],    sc["city_C"]),
        }
        write_manifest(build_manifest(scenarios, sc["gen_time_pmf"]), args.out)
    if not (args.out or args.compare):
        ap.print_help()


if __name__ == "__main__":
    main()
