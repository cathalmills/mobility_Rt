# Mobility-Informed R(t) Dashboard — Design Plan

**Reference framework:** Mills (2026) *Reproduction numbers for epidemics on human mobility networks*  
**Implementation stack:** Streamlit + Plotly + mobility_rt_framework.py  
**Target users:** Epidemiologists and public health analysts with access to mobility and case data

---

## 1. Overview and design philosophy

The dashboard operationalises the parameter exploration from `mobility_rt_exploration.ipynb`,
giving users an interactive, real-time interface to:

1. **Specify what they know** — biological parameters (β or R₀), contact rates, GT profile, network
2. **Explore what they don't know** — scan parameter uncertainty axes with sliders
3. **Read off actionable outputs** — all reproduction numbers, transience measures, and control targets from Tables 1–2 of the manuscript

Every slider change re-runs the relevant computation and updates all panels instantly.
Heavy computations (full PDE simulation) run asynchronously; lightweight computations
(R₀ from β, spectral quantities from a stored NGM) update synchronously.

---

## 2. Architecture

```
dashboard/
├── app.py                    # Main Streamlit app entry point
├── pages/
│   ├── 01_parameter_setup.py   # Tab 1: Network & biological inputs
│   ├── 02_single_scenario.py   # Tab 2: Full simulation + all outputs
│   ├── 03_parameter_space.py   # Tab 3: 2D surface / contour explorer
│   └── 04_control_planner.py   # Tab 4: Intervention targets
├── compute/
│   ├── runner.py               # Async simulation runner (st.cache_data)
│   ├── sweep.py                # Fast parameter sweep with caching
│   └── precompute.py           # Pre-compute R₀(β) curves at startup
├── viz/
│   ├── plotly_helpers.py       # Reusable Plotly figure builders
│   └── network_plot.py         # Interactive network graph (pyvis/Plotly)
└── mobility_rt_framework.py    # (symlinked from parent directory)
```

**Caching strategy:**
- R₀(β) mapping computed once at startup for the current network — `@st.cache_data(ttl=3600)`
- Full simulation results cached by parameter hash — `@st.cache_data(hash_funcs={...})`
- Parameter sweep results cached and progressively updated

---

## 3. Tab 1 — Parameter Setup

**Purpose:** Define the network and biological inputs. This tab is visited first
and feeds all other tabs.

### Left panel: Network inputs

| Control | Type | Default | Notes |
|---------|------|---------|-------|
| Network topology | Dropdown | Urban megacity | Options: Urban, Sparse rural, Hub-and-spoke, Custom |
| N locations | Slider | 6 | Range 3–20 |
| Population sizes | Editable table | Auto-generated | Manually adjustable per location |
| Mobility matrix source | Radio | Synthetic | Options: Synthetic (decay model), Upload CSV |
| Commuting fractions | Per-location sliders | Type-dependent | Core: 40%, Suburban: 28%, etc. |
| Distance decay scale (km) | Slider | 20 km | Range 5–300 km |
| Hub attraction power | Slider | 0.5 | Range 0.1–3.0 (p in gravity model) |
| Day-of-week variation | Toggle + σ slider | On, σ=0.12 | |
| **Mobility preview** | Heatmap (live) | — | Updates with every slider change |

### Right panel: Biological inputs

| Control | Type | Default | Notes |
|---------|------|---------|-------|
| Pathogen | Dropdown | SARS-CoV-2 (Alpha) | Pre-fills GT, β range, λ_W |
| GT distribution | Radio | Gamma | Options: Gamma, Weibull, empirical upload |
| GT mean (days) | Slider | 5.5 | Range 1–21 |
| GT standard deviation | Slider | 1.8 | Range 0.5–5.0 |
| Max infection age (days) | Slider | 25 | Range 10–50 |
| λ_W (home contacts/day) | Slider | 13.0 | Range 5–25 (POLYMOD reference shown) |
| λ_B/λ_W ratio | Slider | 0.30 | Range 0.05–0.90 |
| **Transmission input mode** | Radio | β-mode | Options: β-mode, R₀-mode |
| β (if β-mode) | Slider | auto | Range 0.005–0.15 |
| R₀ target (if R₀-mode) | Slider | — | Range 0.5–5.0 |
| Doubling time observed | Input | — | Converts to β via Euler-Lotka |
| **Implied R₀** | Live readout | — | β × network → R₀ (no simulation) |

### Bottom: Seeding conditions

| Control | Type | Default |
|---------|------|---------|
| Seed location | Multi-select | L1 (core) |
| Seed size | Number input | 10 |
| Simulation duration (days) | Slider | 200 |
| Demographics | Toggle | On (birth/death = 3×10⁻⁵) |

---

## 4. Tab 2 — Single Scenario Explorer

**Purpose:** Full PDE simulation for the current parameters. Displays all
reproduction numbers, transience measures, and spatial dynamics from Tables 1–2.

**Trigger:** "Run simulation" button (or auto-run if compute is fast enough).

### Layout: 3-row dashboard

```
┌─────────────────────────────────────────────────────────────────┐
│  ROW 1: Epidemic overview (4 panels)                            │
│  [R(t)/σ(t)/E(t) + incidence] [Incidence heatmap] [S_eff] [...]│
├─────────────────────────────────────────────────────────────────┤
│  ROW 2: Reproduction number taxonomy (5 panels)                 │
│  [R_out] [R_in] [R_pairwise@peak] [R_meeting] [Elasticity]     │
├─────────────────────────────────────────────────────────────────┤
│  ROW 3: Transience + eigenvectors (4 panels)                    │
│  [Mixing ratio s(t)] [Amplification A(n)] [Condition κ] [v/v*] │
└─────────────────────────────────────────────────────────────────┘
```

### Interactive elements within Tab 2

- **Time scrubber:** Horizontal slider (day 0 to T). All "at peak" panels update to
  show the selected day instead. R_pairwise heatmap, source-sink bar chart, and
  elasticity scatter all update live.

- **Trajectory overlay toggle:** Show/hide σ(t), E(t), A₁(1) on the R(t) panel.

- **Highlight location:** Click any location in the incidence heatmap → updates
  all per-location panels to highlight that location.

- **Export panel:** Download all outputs as CSV or NetCDF (one row per time step,
  one column per quantity/location).

### Key metrics bar (always visible, top of screen)

```
R₀ = 1.43  |  Peak day = 87  |  Attack rate = 52.3%  |  Max σ = 1.61  |  
False-action zone: 12 days  |  Spatial CV of Rout at peak: 0.28
```

---

## 5. Tab 3 — Parameter Space Explorer

**Purpose:** Real-time 2D parameter surfaces showing how outputs vary across pairs
of uncertain parameters. The core tool for early-outbreak exploration.

### Surface controls (left sidebar)

| Control | Type |
|---------|------|
| X-axis parameter | Dropdown (β, R₀, λ_B/λ_W, GT mean, GT SD, commuting scale) |
| Y-axis parameter | Dropdown (same options) |
| X range | Dual slider (min, max) |
| Y range | Dual slider (min, max) |
| Grid resolution | Slider (5×5 to 20×20) |
| Output to display | Dropdown (R₀, attack rate, peak day, reactivity at peak, E(t)>R(t) days) |

### Main panel: Interactive Plotly surface

- Filled contour plot of the selected output over the (X, Y) parameter space
- **Iso-R₀ contour overlay** (R₀ = 1 always shown in white; additional iso-lines selectable)
- **Cross-hair cursor:** hover over any point → tooltip shows all key outputs
- **Current parameters:** marked with a large star/dot
- **Click to simulate:** clicking any point in the surface runs the full simulation for those
  parameters and opens a mini-version of Tab 2

### Comparison panel (bottom)

Side-by-side time-series for up to 4 selected parameter combinations:
- R(t) trajectories with confidence bands (if uncertainty mode active)
- Peak incidence bar chart
- Attack rate comparison

---

## 6. Tab 4 — Intervention Planner

**Purpose:** Given the current simulation, what interventions are needed and where?

### Input section

- Time slider: select day t at which to evaluate intervention targets
- Intervention type: transmission reduction (uniform / location-targeted / corridor-targeted)
- Efficacy: slider for fractional reduction per targeted element

### Output panels

| Panel | Content |
|-------|---------|
| Type R numbers | T^j_type(t) for all locations; flag undefined with explanation |
| Minimum homogeneous effort | u = 1 − 1/R(t); required uniform reduction |
| Elasticity-ranked targets | Ranked list of (corridor kj, location k) by ε^{kj} |
| Counterfactual R | Simulated R(t) under proposed intervention |
| Duration estimate | Days until R < 1 under intervention |
| Contact tracing priority | Locations ranked by reproductive value v_k (high → prioritise tracing) |

### Intervention scenario builder

"What if I reduce..." interface:
- Select location(s) → set % reduction in R_out
- Select mobility corridor(s) → set % reduction in f^{jk}
- Select meeting location(s) → set % reduction in λ_W or λ_B at that venue
- Dashboard re-simulates and shows counterfactual R(t) overlaid on baseline

---

## 7. Streamlit implementation notes

### Session state management

```python
import streamlit as st

# Initialise with defaults on first load
if "params" not in st.session_state:
    st.session_state.params = default_params()

if "sim_results" not in st.session_state:
    st.session_state.sim_results = None

# Cache heavy simulation with parameter hash
@st.cache_data(hash_funcs={"numpy.ndarray": lambda a: a.tobytes()}, ttl=600)
def run_simulation_cached(params_hash, f_jk, populations, p, lw, lb,
                           R0_target, beta, seed, T):
    from mobility_rt_framework import simulate_epidemic
    return simulate_epidemic(f_jk, populations, p, lw, lb, R0_target,
                             seed, T, _beta_mode=(beta is not None))
```

### Performance targets

| Operation | Target latency |
|-----------|---------------|
| R₀(β) lookup (pre-computed) | < 1 ms |
| Slider → R₀ readout update | < 50 ms |
| Slider → 2D surface re-render | < 500 ms (cached tiles) |
| Full PDE simulation (N=6, T=200) | 2–5 s (background thread) |
| Parameter sweep (10×10 grid) | 30–120 s (progressive update) |

### Progressive surface rendering

For the parameter space surface (Tab 3), use a coarse grid first (5×5) and
refine progressively:

```python
import threading, time

def compute_surface_async(param_grid, placeholder):
    # Pass 1: coarse 5×5 grid → render immediately
    coarse = parameter_sweep(..., resolution=5)
    placeholder.plotly_chart(render_surface(coarse))
    
    # Pass 2: refine to 10×10 in background
    fine = parameter_sweep(..., resolution=10)
    placeholder.plotly_chart(render_surface(fine))
```

---

## 8. Suggested Plotly figure specifications

### R(t) / σ(t) time series (Tab 2, Panel 1)

```python
import plotly.graph_objects as go

fig = go.Figure()
fig.add_trace(go.Scatter(x=t_arr, y=R_network, name="R(t)", line=dict(color="#0072B2")))
fig.add_trace(go.Scatter(x=t_arr, y=reactivity, name="σ(t)", line=dict(color="#D55E00", dash="dash")))
fig.add_trace(go.Scatter(x=t_arr, y=E_risk_averse, name="E(t)", line=dict(color="#CC79A7", dash="dot")))
# False-action zone fill
fig.add_vrect(x0=false_start, x1=false_end, fillcolor="orange", opacity=0.15,
              annotation_text="False-action zone", annotation_position="top right")
fig.add_hline(y=1.0, line_dash="dash", line_color="gray")
fig.update_layout(title="Network reproduction numbers", xaxis_title="Day t",
                  yaxis_title="Value", hovermode="x unified")
```

### Parameter surface (Tab 3)

```python
fig = go.Figure(data=go.Contour(
    x=x_axis, y=y_axis, z=output_grid,
    colorscale="RdBu_r",
    contours=dict(showlabels=True),
    line_smoothing=0.85,
))
# Add R₀=1 iso-line
fig.add_trace(go.Contour(x=x_axis, y=y_axis, z=R0_grid,
                          showscale=False, contours=dict(start=1, end=1, size=0.001),
                          line=dict(color="white", width=2, dash="dash"),
                          name="R₀=1 threshold"))
# Current parameter star
fig.add_trace(go.Scatter(x=[current_x], y=[current_y], mode="markers",
                         marker=dict(symbol="star", size=16, color="yellow"),
                         name="Current params"))
```

### Network graph (Tab 1 preview)

```python
import networkx as nx
import plotly.graph_objects as go

G = nx.DiGraph()
for j in range(N):
    for k in range(N):
        if j != k and f_mean[j,k] > threshold:
            G.add_edge(j, k, weight=float(f_mean[j,k]))

pos = nx.spring_layout(G, seed=42)
edge_x, edge_y = [], []
for u, v in G.edges():
    edge_x += [pos[u][0], pos[v][0], None]
    edge_y += [pos[u][1], pos[v][1], None]

fig = go.Figure()
fig.add_trace(go.Scatter(x=edge_x, y=edge_y, mode="lines",
                         line=dict(width=1.5, color="#aaa"), hoverinfo="none"))
fig.add_trace(go.Scatter(x=[pos[j][0] for j in G.nodes()],
                         y=[pos[j][1] for j in G.nodes()],
                         mode="markers+text",
                         marker=dict(size=[np.sqrt(populations[j])/300 for j in G.nodes()],
                                     color=np.log10(populations), colorscale="YlOrBr"),
                         text=[loc_s[j] for j in G.nodes()], textposition="top center"))
```

---

## 9. Development roadmap

### Phase 1 (MVP — 2 weeks)
- [ ] Tab 1: Parameter setup with live R₀ readout from β slider
- [ ] Tab 2: Single scenario with time scrubber and key metrics bar
- [ ] Tab 3: 2D surface (β × λ_B/λ_W) with coarse grid (5×5)
- [ ] Export: CSV download of simulation outputs

### Phase 2 (Core features — 2 weeks)
- [ ] Tab 3: Full parameter sweep with fine grid + all sweep axes
- [ ] Tab 4: Type R and controllability panel
- [ ] Network CSV upload (custom f_jk)
- [ ] Mobile phone data integration (SafeGraph / Google format parser)
- [ ] Uncertainty mode: sample from doubling-time distribution → R₀ CI

### Phase 3 (Advanced — 2–4 weeks)
- [ ] Tab 4: Counterfactual simulation builder
- [ ] Real-time data connector (epidemiological case reports)
- [ ] Vector-borne disease mode (extends framework Section E.1)
- [ ] Multi-language labels (French, Portuguese for SSA settings)
- [ ] Docker container for deployment

---

## 10. Quick start (local development)

```bash
# Install dependencies
pip install streamlit plotly networkx scipy numpy pandas

# Clone framework
git clone https://github.com/your-org/mobility-rt.git
cd mobility-rt

# Run dashboard
streamlit run dashboard/app.py

# Dashboard opens at http://localhost:8501
```

**Minimal app.py skeleton:**

```python
import streamlit as st
import numpy as np
import sys
sys.path.insert(0, "..")
from mobility_rt_framework import (
    MobilityRtFramework, discretise_gamma, build_synthetic_network,
    compute_R0_from_beta, scan_beta_to_R0,
)

st.set_page_config(page_title="Mobility-Informed R(t)", layout="wide",
                   page_icon="🦠")
st.title("Mobility-Informed Renewal Equation Dashboard")

# Sidebar: key sliders
with st.sidebar:
    st.header("Parameters")
    beta  = st.slider("β (transmission prob.)", 0.005, 0.12, 0.035, 0.001)
    lw    = st.slider("λ_W (home contacts/day)", 5.0, 25.0, 13.0, 0.5)
    ratio = st.slider("λ_B/λ_W ratio", 0.05, 0.80, 0.30, 0.01)
    gm    = st.slider("GT mean (days)", 2.0, 14.0, 5.5, 0.5)
    gs    = st.slider("GT SD (days)", 0.5, 4.0, 1.8, 0.1)
    T     = st.slider("Duration (days)", 50, 365, 200)
    N_loc = st.selectbox("N locations", [4, 6, 8, 10], index=1)

# Build network + profile
p    = discretise_gamma(gm, gs, 25)
f_jk, pops, _ = build_synthetic_network(N=N_loc, T=T)
f0   = f_jk[0]

# Live R₀ readout (no simulation needed)
R0_live = compute_R0_from_beta(f0, pops, p, beta, lw, lw * ratio)
st.metric("R₀ (from β, no simulation)", f"{R0_live:.3f}",
          delta=f"{'↑ growing' if R0_live > 1 else '↓ declining'}")

# Run simulation on button press
if st.button("Run Full Simulation"):
    seed = np.zeros(N_loc); seed[0] = 5.0
    model = MobilityRtFramework(
        f_jk=f_jk, populations=pops, infectiousness_profile=p,
        contact_rate_home=lw, contact_rate_away=lw*ratio,
        initial_infections=seed, T=T, beta=beta, verbose=False,
    )
    with st.spinner("Simulating..."):
        results = model.simulate()
    st.success(f"Done! R₀={results['R0_achieved']:.3f}, "
               f"attack rate={results['incidence'].sum()/pops.sum()*100:.1f}%")
    # ... render plots ...
```
