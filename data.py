"""
Generate pseudo-realistic mobility networks for two contrasting scenarios,
parameterised from empirical literature rather than gravity models.

Scenario A: Lagos, Nigeria (20 LGAs)
  - Dense megacity, high intra-city mixing
  - Connectivity parameterised from Wesolowski et al. 2015 (Kenya CDR, urban settings),
    Azman et al. 2014 (cholera metapopulation connectivity ranges)

Scenario B: Zambia (10 Provinces)  
  - Sparse country, capital-dominated hub-and-spoke
  - Connectivity parameterised from Wesolowski et al. 2021 (Zambia CDR, eLife),
    Mozambique COVID metapopulation (radiation model, 11 provinces)

Literature sources for connectivity parameterisation:
  - Azman et al. 2014 (Proc Roy Soc B): c_ij in [0, 0.20] for cholera metapopulation
  - Wesolowski et al. 2015 (PLOS Comp Bio): gravity models overestimate spread in SSA;
    empirical CDR shows strong regional clustering, capital dominance
  - Wesolowski et al. 2021 (eLife): adjusted gravity with trip-type heterogeneity;
    urban-urban coupling ~3-5x stronger than rural-rural in Zambia
  - Xia et al. 2004 (Am Nat): measles metapopulation, coupling ~ 0.001-0.01
    for UK cities (lower bound for well-separated populations)
  - Tizzoni et al. 2014 (PLOS Comp Bio): commuting networks, ~5-30% leave home
    region daily in European cities
  - Bichara et al. 2015: metapopulation with residence-time matrix P where
    p_ij = fraction of time residents of i spend in j

Approach: Instead of a gravity model, we directly construct the residence-time 
matrix P (equivalent to transition matrix T) using empirically-grounded rules:

1. Set diagonal (stay-home fraction) per node type from literature
2. Distribute off-diagonal mass using empirical patterns:
   - Adjacency/contiguity effects (neighbours get more flow)
   - Trip-type heterogeneity (urban-urban > rural-urban > rural-rural)
   - Capital/hub attraction multiplier
   - Distance decay (exponential for Zambia per Wesolowski 2021)
3. Add stochastic perturbation to avoid unrealistic symmetry
"""

import numpy as np
import pandas as pd
import json

np.random.seed(42)

# ============================================================
# Helper functions
# ============================================================

def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = np.sin(dlat/2)**2 + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon/2)**2
    return 2 * R * np.arcsin(np.sqrt(a))

def build_distance_matrix(lats, lons):
    n = len(lats)
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                D[i, j] = haversine_km(lats[i], lons[i], lats[j], lons[j])
    return D

def normalise_rows(T):
    """Ensure each row sums to exactly 1."""
    row_sums = T.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums > 0, row_sums, 1)
    return T / row_sums

# ============================================================
# SCENARIO A: Lagos, Nigeria — 20 LGAs
# ============================================================
# Parameterised as a dense urban metapopulation with high mixing

lagos = {
    "names": [
        "Agege", "Ajeromi-Ifelodun", "Alimosho", "Amuwo-Odofin", "Apapa",
        "Badagry", "Epe", "Eti-Osa", "Ibeju-Lekki", "Ifako-Ijaiye",
        "Ikeja", "Ikorodu", "Kosofe", "Lagos-Island", "Lagos-Mainland",
        "Mushin", "Ojo", "Oshodi-Isolo", "Shomolu", "Surulere"
    ],
    "population": np.array([
        1053000, 1435000, 2047000, 524000, 618000,
        442000, 323000, 1065000, 197000, 745000,
        648000, 905000, 934000, 496000, 629000,
        824000, 941000, 1134000, 622000, 676000
    ], dtype=float),
    "lat": [6.619, 6.457, 6.608, 6.464, 6.449,
            6.415, 6.584, 6.458, 6.467, 6.647,
            6.601, 6.656, 6.573, 6.455, 6.497,
            6.526, 6.442, 6.539, 6.543, 6.502],
    "lon": [3.321, 3.345, 3.244, 3.326, 3.360,
            2.882, 3.979, 3.528, 3.783, 3.297,
            3.347, 3.538, 3.407, 3.395, 3.373,
            3.354, 3.234, 3.334, 3.378, 3.347],
    # Node types determine connectivity patterns
    # core: commercial centres (strong attractor)
    # dense: high-density residential/informal settlements
    # suburban: middle-density residential
    # peripheral: outer/rural areas
    "type": [
        "dense", "dense", "suburban", "suburban", "core",
        "peripheral", "peripheral", "core", "peripheral", "suburban",
        "core", "suburban", "suburban", "core", "core",
        "dense", "suburban", "suburban", "dense", "dense"
    ]
}

def build_lagos_matrix():
    """
    Build Lagos transition matrix using empirical patterns.
    
    Key empirical constraints (from literature):
    - Daily commuting fraction: 25-45% in large African cities 
      (Wesolowski 2015, Tizzoni 2014)
    - Intra-city coupling: c_ij up to 0.15-0.20 for adjacent areas
      (Azman et al. 2014)
    - Urban-urban coupling ~3-5x stronger than other trip types
      (Wesolowski 2021 eLife, Zambia/Kenya CDR data)
    - Strong neighbour effect: most trips < 10km in dense cities
    - Commercial cores attract disproportionately
    """
    n = len(lagos["names"])
    dist = build_distance_matrix(lagos["lat"], lagos["lon"])
    types = lagos["type"]
    
    # Step 1: Set commuting fractions by node type
    # Literature: 25-45% daily mobility in large African cities
    cf = {
        "core": 0.40,       # CBD workers + commercial activity
        "dense": 0.35,      # dense residential, commute to cores
        "suburban": 0.28,   # moderate commuting
        "peripheral": 0.18  # mostly self-contained
    }
    commuting = np.array([cf[t] for t in types])
    
    # Step 2: Build off-diagonal weights using empirical patterns
    T = np.zeros((n, n))
    
    # Base coupling: exponential distance decay
    # Wesolowski 2021: exponential kernel works better than power law for short distances
    # Decay scale ~5-10 km for intra-city (most trips < 10km)
    decay_scale = 7.0  # km
    
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            
            # Base: exponential distance decay
            weight = np.exp(-dist[i, j] / decay_scale)
            
            # Trip-type multipliers (from Wesolowski 2021 eLife)
            # Urban↔Urban coupling is 3-5x stronger than Rural↔Rural
            ti, tj = types[i], types[j]
            
            if ti in ("core", "dense") and tj in ("core", "dense"):
                weight *= 4.0   # urban-urban: strongest
            elif ti in ("core", "dense") or tj in ("core", "dense"):
                weight *= 2.0   # urban-rural: moderate  
            else:
                weight *= 1.0   # rural-rural: baseline
            
            # Capital/hub attraction (Wesolowski 2015: mega-city pull 
            # underestimated by standard models)
            # Lagos Island, Ikeja, Apapa are major employment centres
            if tj == "core":
                weight *= 2.5
            
            # Population-proportional attraction (mild, not gravity-model strong)
            # Just a sqrt scaling to reflect that bigger areas have more destinations
            weight *= np.sqrt(lagos["population"][j] / 1e6)
            
            # Stochastic perturbation (±30%) to avoid unrealistic symmetry
            # Real mobility is inherently asymmetric
            weight *= np.exp(np.random.normal(0, 0.15))
            
            T[i, j] = weight
    
    # Step 3: Normalise off-diagonal rows and apply commuting fractions
    for i in range(n):
        row_sum = T[i, :].sum()
        if row_sum > 0:
            T[i, :] = commuting[i] * T[i, :] / row_sum
        T[i, i] = 1.0 - commuting[i]
    
    return T, commuting, dist

# ============================================================
# SCENARIO B: Zambia — 10 Provinces
# ============================================================
# Parameterised as a sparse country-level network with hub-and-spoke structure

zambia = {
    "names": [
        "Central", "Copperbelt", "Eastern", "Luapula", "Lusaka",
        "Muchinga", "Northern", "North-Western", "Southern", "Western"
    ],
    "population": np.array([
        1901883, 2763000, 2220000, 1310000, 3079964,
        1020000, 1440000, 1004000, 2100000, 1100000
    ], dtype=float),
    "lat": [-14.45, -12.80, -13.63, -11.78, -15.39,
            -12.27, -10.22, -12.34, -15.78, -15.50],
    "lon": [28.97, 28.21, 32.17, 28.88, 28.32,
            31.27, 31.13, 25.82, 27.83, 23.13],
    "type": [
        "peri-capital", "urban-industrial", "rural", "rural", "capital",
        "rural", "rural", "remote-rural", "semi-urban", "remote-rural"
    ],
    # Adjacency: which provinces share borders (empirical from map)
    # This is the strongest predictor of inter-provincial flow
    # (Wesolowski 2021: regional clustering dominates)
    "adjacency": {
        "Central":       ["Lusaka", "Copperbelt", "Northern", "Muchinga", "Eastern", "Southern"],
        "Copperbelt":    ["Central", "North-Western", "Northern", "Luapula"],
        "Eastern":       ["Central", "Muchinga", "Lusaka"],
        "Luapula":       ["Copperbelt", "Northern", "Muchinga"],
        "Lusaka":        ["Central", "Southern", "Eastern"],
        "Muchinga":      ["Central", "Eastern", "Northern", "Luapula"],
        "Northern":      ["Central", "Copperbelt", "Luapula", "Muchinga"],
        "North-Western": ["Copperbelt", "Western", "Southern"],
        "Southern":      ["Lusaka", "Central", "Western", "North-Western"],
        "Western":       ["Southern", "North-Western"]
    }
}

def build_zambia_matrix():
    """
    Build Zambia transition matrix using empirical patterns.
    
    Key empirical constraints:
    - Inter-provincial daily mobility: 2-10% 
      (much lower than intra-city; Wesolowski 2021 eLife)
    - Exponential distance decay (best fit for Zambia, Wesolowski 2021)
    - Strong regional clustering: adjacent provinces get ~60-80% of flow
    - Capital dominance: Lusaka attracts 3-5x more than population alone predicts
    - Copperbelt secondary hub: mining economy creates independent attractor
    - Remote provinces (Western, North-Western): very low outflow
    """
    n = len(zambia["names"])
    dist = build_distance_matrix(zambia["lat"], zambia["lon"])
    types = zambia["type"]
    names = zambia["names"]
    
    # Step 1: Inter-provincial commuting fractions
    # Much lower than intra-city — these represent fraction of population
    # that moves to a DIFFERENT PROVINCE on any given day/week
    # Literature: ~2-8% for inter-regional in SSA (Wesolowski 2021)
    cf = {
        "capital": 0.06,         # Lusaka: some outflow to Southern/Central
        "peri-capital": 0.08,    # Central: highest due to Lusaka proximity
        "urban-industrial": 0.05, # Copperbelt: own economy, less need to travel
        "semi-urban": 0.04,      # Southern: moderate
        "rural": 0.025,          # rural provinces: low
        "remote-rural": 0.015    # Western/NW: very low inter-provincial movement
    }
    commuting = np.array([cf[t] for t in types])
    
    # Step 2: Build off-diagonal weights
    T = np.zeros((n, n))
    
    # Exponential distance decay (Wesolowski 2021: best fit for Zambia)
    # Decay scale ~150-250 km for inter-provincial (much larger than intra-city)
    decay_scale = 200.0  # km
    
    # Build adjacency lookup
    adj = zambia["adjacency"]
    
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            
            # Base: exponential distance decay
            weight = np.exp(-dist[i, j] / decay_scale)
            
            # Adjacency bonus: neighbours get 3x more flow
            # (Wesolowski 2021: regional clustering is the dominant feature)
            if names[j] in adj[names[i]]:
                weight *= 3.0
            
            # Capital attraction (Wesolowski 2015: mega-city pull)
            # Lusaka attracts disproportionately from all provinces
            if names[j] == "Lusaka":
                weight *= 4.0
            
            # Copperbelt secondary attraction (mining economy)
            if names[j] == "Copperbelt":
                weight *= 2.0
            
            # Trip-type heterogeneity (Wesolowski 2021)
            ti, tj = types[i], types[j]
            urban_types = ("capital", "peri-capital", "urban-industrial", "semi-urban")
            
            if ti in urban_types and tj in urban_types:
                weight *= 2.5  # urban-urban
            elif ti in urban_types or tj in urban_types:
                weight *= 1.5  # urban-rural
            # else: rural-rural stays at baseline
            
            # Population proportionality (mild)
            weight *= np.sqrt(zambia["population"][j] / 1e6)
            
            # Stochastic perturbation (±25%)
            weight *= np.exp(np.random.normal(0, 0.12))
            
            T[i, j] = weight
    
    # Step 3: Normalise and apply commuting fractions
    for i in range(n):
        row_sum = T[i, :].sum()
        if row_sum > 0:
            T[i, :] = commuting[i] * T[i, :] / row_sum
        T[i, i] = 1.0 - commuting[i]
    
    return T, commuting, dist

# ============================================================
# Build both scenarios
# ============================================================
print("=" * 60)
print("SCENARIO A: Lagos, Nigeria (20 LGAs)")
print("=" * 60)

lagos_T, lagos_cf, lagos_dist = build_lagos_matrix()
print(f"Nodes: {len(lagos['population'])}")
print(f"Total population: {lagos['population'].sum():,.0f}")
print(f"Row sums: min={lagos_T.sum(1).min():.6f}, max={lagos_T.sum(1).max():.6f}")
print(f"Diagonal range: [{lagos_T.diagonal().min():.3f}, {lagos_T.diagonal().max():.3f}]")
print(f"Max off-diagonal: {np.max(lagos_T - np.diag(np.diag(lagos_T))):.4f}")
print(f"Distance range: [{lagos_dist[lagos_dist > 0].min():.1f}, {lagos_dist.max():.1f}] km")

print(f"\nCommuting fractions by type:")
for t in ["core", "dense", "suburban", "peripheral"]:
    idx = [i for i, x in enumerate(lagos["type"]) if x == t]
    print(f"  {t:>12}: {lagos_cf[idx[0]]:.2f}  ({len(idx)} LGAs)")

print(f"\nTop 5 off-diagonal flows:")
offdiag = lagos_T.copy(); np.fill_diagonal(offdiag, 0)
for _ in range(5):
    i, j = np.unravel_index(offdiag.argmax(), offdiag.shape)
    print(f"  {lagos['names'][i]:>20} -> {lagos['names'][j]:<20} T={lagos_T[i,j]:.4f}  ({lagos_dist[i,j]:.1f} km)")
    offdiag[i, j] = 0

print()
print("=" * 60)
print("SCENARIO B: Zambia (10 Provinces)")
print("=" * 60)

zambia_T, zambia_cf, zambia_dist = build_zambia_matrix()
print(f"Nodes: {len(zambia['population'])}")
print(f"Total population: {zambia['population'].sum():,.0f}")
print(f"Row sums: min={zambia_T.sum(1).min():.6f}, max={zambia_T.sum(1).max():.6f}")
print(f"Diagonal range: [{zambia_T.diagonal().min():.3f}, {zambia_T.diagonal().max():.3f}]")
print(f"Max off-diagonal: {np.max(zambia_T - np.diag(np.diag(zambia_T))):.4f}")
print(f"Distance range: [{zambia_dist[zambia_dist > 0].min():.1f}, {zambia_dist.max():.1f}] km")

print(f"\nCommuting fractions by type:")
for t in ["capital", "peri-capital", "urban-industrial", "semi-urban", "rural", "remote-rural"]:
    idx = [i for i, x in enumerate(zambia["type"]) if x == t]
    if idx:
        print(f"  {t:>18}: {zambia_cf[idx[0]]:.3f}  ({len(idx)} provinces)")

print(f"\nTop 5 off-diagonal flows:")
offdiag = zambia_T.copy(); np.fill_diagonal(offdiag, 0)
for _ in range(5):
    i, j = np.unravel_index(offdiag.argmax(), offdiag.shape)
    print(f"  {zambia['names'][i]:>20} -> {zambia['names'][j]:<20} T={zambia_T[i,j]:.4f}  ({zambia_dist[i,j]:.1f} km, adj={'yes' if zambia['names'][j] in zambia['adjacency'][zambia['names'][i]] else 'no'})")
    offdiag[i, j] = 0

# ============================================================
# Comparison
# ============================================================
print("\n" + "=" * 60)
print("SCENARIO COMPARISON")
print("=" * 60)
print(f"{'':>30} {'Lagos':>15} {'Zambia':>15}")
print(f"{'Nodes':>30} {len(lagos['population']):>15} {len(zambia['population']):>15}")
print(f"{'Total population':>30} {lagos['population'].sum():>15,.0f} {zambia['population'].sum():>15,.0f}")
print(f"{'Spatial scale (max km)':>30} {lagos_dist.max():>15.0f} {zambia_dist.max():>15.0f}")
print(f"{'Mean commuting fraction':>30} {lagos_cf.mean():>15.3f} {zambia_cf.mean():>15.3f}")
print(f"{'Min diagonal (max mobility)':>30} {lagos_T.diagonal().min():>15.3f} {zambia_T.diagonal().min():>15.3f}")
print(f"{'Max off-diagonal coupling':>30} {np.max(lagos_T-np.diag(np.diag(lagos_T))):>15.4f} {np.max(zambia_T-np.diag(np.diag(zambia_T))):>15.4f}")
print(f"{'Mixing regime':>30} {'high (urban)':>15} {'low (national)':>15}")
print(f"{'Primary disease context':>30} {'cholera/measles':>15} {'cholera':>15}")

# ============================================================
# Save outputs
# ============================================================
def save_scenario(name, data, T, cf, dist):
    n = len(data["population"])
    labels = data["names"]
    
    pd.DataFrame(T, index=labels, columns=labels).to_csv(f"/Users/reddy/Control_Theory/datasets/{name}_transition_matrix.csv")
    pd.DataFrame(dist, index=labels, columns=labels).to_csv(f"/Users/reddy/Control_Theory/datasets/{name}_distance_km.csv")
    
    pd.DataFrame({
        "node_id": range(n),
        "name": labels,
        "population": data["population"].astype(int),
        "latitude": data["lat"],
        "longitude": data["lon"],
        "type": data["type"],
        "commuting_fraction": cf
    }).to_csv(f"/Users/reddy/Control_Theory/datasets/{name}_nodes.csv", index=False)
    
    summary = {
        "scenario": name,
        "n_nodes": n,
        "total_population": int(data["population"].sum()),
        "parameterisation": "Empirical literature (see script docstring)",
        "key_references": [
            "Wesolowski et al. 2015 PLOS Comp Bio (SSA mobility, gravity model failures)",
            "Wesolowski et al. 2021 eLife (4 SSA countries, trip-type heterogeneity)",
            "Azman et al. 2014 Proc Roy Soc B (cholera metapopulation, connectivity ranges)",
            "Xia et al. 2004 Am Nat (measles metapopulation coupling)",
            "Tizzoni et al. 2014 PLOS Comp Bio (commuting network validation)"
        ],
        "mobility_construction": {
            "method": "Empirical rule-based (NOT gravity/radiation model)",
            "distance_decay": "Exponential (Wesolowski 2021: best fit for Zambia)",
            "trip_type_heterogeneity": "Urban-urban 3-5x > rural-rural (Wesolowski 2021)",
            "capital_attraction": "Explicit multiplier (Wesolowski 2015: standard models underestimate)",
            "adjacency_effect": "Neighbours 3x (Zambia) / distance-implicit (Lagos)",
            "stochastic_asymmetry": "Log-normal perturbation to avoid unrealistic symmetry"
        }
    }
    with open(f"/Users/reddy/Control_Theory/datasets/{name}_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nSaved: {name}_transition_matrix.csv, {name}_distance_km.csv, {name}_nodes.csv, {name}_summary.json")

save_scenario("lagos", lagos, lagos_T, lagos_cf, lagos_dist)
save_scenario("zambia", zambia, zambia_T, zambia_cf, zambia_dist)