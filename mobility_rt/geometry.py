"""mobility_rt.geometry."""
import numpy as np
from scipy.spatial.distance import cdist


def generate_city(n_locations=10, scenario="lagos", seed=42):
    """
    Generate synthetic city with empirically-grounded structure.

    scenario="lagos"  — dense urban megacity (Lagos-inspired).
        N locations, tight spatial clustering (~40 km span),
        node types: core/dense/suburban/peripheral,
        commuting fractions 18–40% (Wesolowski 2015, Tizzoni 2014).

    scenario="zambia" — sparse national network (Zambia-inspired).
        N locations spread over ~600 km,
        node types: capital/peri-capital/urban-industrial/semi-urban/rural/remote-rural,
        commuting fractions 1.5–8% (Wesolowski 2021 eLife).

    Returns: coords [N,2] (km), populations [N], distances [N,N] (km),
             node_types [N], node_metadata dict.
    """
    rng = np.random.default_rng(seed)
    N   = n_locations

    if scenario == "lagos":
        # Type assignment: ~20% core, 30% dense, 30% suburban, 20% peripheral
        type_seq = (["core"]     * max(1, int(0.20 * N)) +
                    ["dense"]    * max(1, int(0.30 * N)) +
                    ["suburban"] * max(1, int(0.30 * N)) +
                    ["peripheral"] * max(1, N - int(0.20*N) - int(0.30*N) - int(0.30*N)))
        type_seq = type_seq[:N]
        rng.shuffle(type_seq)
        node_types = type_seq

        # Spatial: tight urban spread, ~40 km radius
        radii  = rng.exponential(8.0, N)
        angles = rng.uniform(0, 2*np.pi, N)
        coords = np.column_stack([radii * np.cos(angles),
                                  radii * np.sin(angles)])

        # Population: cores largest, peripherals smallest
        type_pop = {"core": 900_000, "dense": 700_000,
                    "suburban": 400_000, "peripheral": 200_000}
        pop_raw = np.array([type_pop[t] for t in node_types], dtype=float)
        pop_raw *= np.exp(rng.normal(0, 0.25, N))

        # Commuting fractions per type (Wesolowski 2015, Tizzoni 2014)
        cf_map = {"core": 0.40, "dense": 0.35, "suburban": 0.28, "peripheral": 0.18}

        # Distance decay scale: 7 km (most intra-city trips < 10 km)
        decay_scale = 7.0

    elif scenario == "zambia":
        # Types: 1 capital, 1 peri-capital, 1 urban-industrial, 1 semi-urban,
        #        rest split rural/remote
        n_remote = max(1, N // 5)
        n_rural  = max(1, N // 3)
        n_urban  = N - 4 - n_rural - n_remote
        type_seq = (["capital"] + ["peri-capital"] + ["urban-industrial"] +
                    ["semi-urban"] * max(1, n_urban) +
                    ["rural"] * n_rural +
                    ["remote-rural"] * n_remote)
        type_seq = type_seq[:N]
        while len(type_seq) < N:
            type_seq.append("rural")
        node_types = type_seq

        # Spatial: spread over ~600 km
        radii  = rng.exponential(120.0, N)
        radii[0] = 0.0   # capital at centre
        angles = rng.uniform(0, 2*np.pi, N)
        coords = np.column_stack([radii * np.cos(angles),
                                  radii * np.sin(angles)])

        # Population: capital largest
        type_pop = {"capital": 3_000_000, "peri-capital": 1_800_000,
                    "urban-industrial": 2_500_000, "semi-urban": 1_500_000,
                    "rural": 900_000, "remote-rural": 600_000}
        pop_raw = np.array([type_pop[t] for t in node_types], dtype=float)
        pop_raw *= np.exp(rng.normal(0, 0.20, N))

        cf_map = {"capital": 0.06, "peri-capital": 0.08, "urban-industrial": 0.05,
                  "semi-urban": 0.04, "rural": 0.025, "remote-rural": 0.015}

        decay_scale = 200.0

    else:
        raise ValueError(f"Unknown scenario: {scenario!r}. Use 'lagos' or 'zambia'.")

    populations = np.round(pop_raw).astype(float)
    distances   = cdist(coords, coords, metric="euclidean")
    commuting_fracs = np.array([cf_map[t] for t in node_types])

    # Identify representative hub / peripheral / mid indices from node_types.
    # For Lagos: hub = most-populated "core"; peripheral = least-populated "peripheral".
    # For Zambia: hub = "capital"; peripheral = least-populated "remote-rural".
    HUB_TYPES   = {"lagos": ["core"],                       "zambia": ["capital"]}
    PERIPH_TYPES = {"lagos": ["peripheral"],                "zambia": ["remote-rural", "rural"]}
    MID_TYPES   = {"lagos": ["dense", "suburban"],          "zambia": ["semi-urban", "urban-industrial"]}

    def _pick(type_candidates, prefer_high_pop):
        """Return index of node whose type is in type_candidates; ties broken by population."""
        idxs = [i for i, t in enumerate(node_types) if t in type_candidates]
        if not idxs:
            # Fall back to population rank
            return int(np.argmax(populations)) if prefer_high_pop else int(np.argmin(populations))
        pops_cand = populations[idxs]
        return idxs[int(np.argmax(pops_cand))] if prefer_high_pop else idxs[int(np.argmin(pops_cand))]

    hub_idx    = _pick(HUB_TYPES.get(scenario, ["core"]),   prefer_high_pop=True)
    periph_idx = _pick(PERIPH_TYPES.get(scenario, ["peripheral"]), prefer_high_pop=False)
    # Mid: pick a node of mid category, or just the median-population node outside hub/periph
    mid_candidates = [i for i, t in enumerate(node_types)
                      if t in MID_TYPES.get(scenario, ["dense", "suburban"])
                      and i not in (hub_idx, periph_idx)]
    if mid_candidates:
        mid_idx = mid_candidates[len(mid_candidates) // 2]
    else:
        order = np.argsort(populations)
        mids  = [i for i in order if i not in (hub_idx, periph_idx)]
        mid_idx = int(mids[len(mids) // 2]) if mids else int(order[len(order) // 2])

    meta = {
        "node_types":       node_types,
        "commuting_fracs":  commuting_fracs,
        "decay_scale":      decay_scale,
        "scenario":         scenario,
        "cf_map":           cf_map,
        "hub_idx":          hub_idx,
        "periph_idx":       periph_idx,
        "mid_idx":          mid_idx,
    }
    return coords, populations, distances, node_types, meta


def representative_locs(city_data):
    """Return (i_hub, i_mid, i_per, show_locs, show_lbls) from node_types in meta.

    Uses meta["hub_idx"] / "mid_idx" / "periph_idx" set by generate_city, so labels
    match the actual node category (core/dense/suburban/peripheral, capital/remote-rural,
    etc.) rather than distance from the city centroid.
    """
    coords, pops, dists, node_types, meta = city_data
    N = len(pops)
    i_hub  = int(meta.get("hub_idx",   0))
    i_per  = int(meta.get("periph_idx", N - 1))
    i_mid  = int(meta.get("mid_idx",   N // 2))
    show_locs = [i_hub, i_mid, i_per]
    show_lbls = [
        f"L{i_hub+1} ({node_types[i_hub]})",
        f"L{i_mid+1} ({node_types[i_mid]})",
        f"L{i_per+1} ({node_types[i_per]})",
    ]
    return i_hub, i_mid, i_per, show_locs, show_lbls
