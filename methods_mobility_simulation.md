# Methods: Simulation of Mobility, Epidemiological Parameters, and Renewal Equation PDE

---

## 1. City geometry

Ten nodes are placed in polar coordinates $(r_i, \theta_i)$ about a common centroid.
Radii are drawn from an exponential distribution whose scale parameter reflects
the spatial extent of the target setting: $\lambda_r = 8$ km (dense urban, Lagos-inspired)
or $\lambda_r = 120$ km (sparse national, Zambia-inspired).
Angles are drawn uniformly on $[0, 2\pi)$.

Each node is assigned a **type** that determines its base population and commuting intensity.

**Dense urban scenario (Scenario A, Lagos-inspired)**

| Type | Share | Base population | Commuting fraction $c_j$ |
|------|-------|----------------|--------------------------|
| Core | 20 % | 900 000 | 0.40 |
| Dense | 30 % | 700 000 | 0.35 |
| Suburban | 30 % | 400 000 | 0.28 |
| Peripheral | 20 % | 200 000 | 0.18 |

**Sparse national scenario (Scenario B, Zambia-inspired)**

| Type | Nodes | Base population | Commuting fraction $c_j$ |
|------|-------|----------------|--------------------------|
| Capital | 1 | 3 000 000 | 0.06 |
| Peri-capital | 1 | 1 800 000 | 0.08 |
| Urban-industrial | 1 | 2 500 000 | 0.05 |
| Semi-urban | variable | 1 500 000 | 0.04 |
| Rural | variable | 900 000 | 0.025 |
| Remote-rural | variable | 600 000 | 0.015 |

Realised node populations are $N_j = N_j^{\text{base}} \exp(\varepsilon_j)$,
where $\varepsilon_j \sim \mathcal{N}(0, \sigma_\varepsilon^2)$
with $\sigma_\varepsilon = 0.25$ (Lagos) or $0.20$ (Zambia).

---

## 2. Base mobility matrix $\bar{f}_{jk}$

The time-invariant base matrix encodes home fraction (diagonal) and
distance–population-weighted off-diagonal flows.

**Home fraction:**
$$\bar{f}_{jj} = 1 - c_j.$$

**Off-diagonal weights** ($k \neq j$):
$$w_{jk} = \exp\!\left(-\frac{d_{jk}}{\delta}\right)
            \left(\frac{N_k}{\bar{N}}\right)^{\!p},$$

where $d_{jk}$ is the Euclidean distance between nodes $j$ and $k$ (km),
$\delta$ is the distance-decay scale ($\delta = 7$ km for Scenario A,
$\delta = 200$ km for Scenario B), and $p$ is the population-attractiveness
exponent (default $p = 0.5$, as in the radiation-model limit of Simini et al.\ 2012).
For Scenario C (hub-amplified transience), $p = 3.0$ concentrates flows toward the
dominant hub node.

Off-diagonal entries are row-normalised and rescaled to the commuting fraction:
$$\bar{f}_{jk} = c_j \cdot \frac{w_{jk}}{\sum_{k' \neq j} w_{jk'}}, \quad k \neq j,$$
ensuring row-stochasticity, $\sum_k \bar{f}_{jk} = 1$.

---

## 3. Time-varying mobility $f_{jk}(t)$

Two empirically motivated sources of day-to-day variation are applied to $\bar{f}_{jk}$.

**Day-of-week scaling.** A weekly scaling vector modulates the away fraction,
reflecting the systematic reduction in commuting observed on weekends in
mobile-phone CDR studies across sub-Saharan Africa (Wesolowski et al.\ 2015;
Tizzoni et al.\ 2014):

$$[s_{\text{Mon}}, s_{\text{Tue}}, s_{\text{Wed}}, s_{\text{Thu}},
   s_{\text{Fri}}, s_{\text{Sat}}, s_{\text{Sun}}]
 = [1.00,\;1.00,\;1.00,\;1.00,\;0.90,\;0.60,\;0.50].$$

**Daily lognormal noise.** A multiplicative lognormal perturbation
$\xi_t \sim \operatorname{LogNormal}(0, \sigma_\xi)$ with $\sigma_\xi = 0.15$,
clipped to $[0.50, 1.80]$, captures day-to-day idiosyncratic variability
(weather, events, etc.).

For each day $t$, the composite scale is $s_t = s_{\text{DoW}(t)} \cdot \xi_t$.
The away fraction of each origin $j$ is scaled by $s_t$, capped at 0.95,
and the home fraction set to $f_{jj}(t) = 1 - \sum_{k \neq j} f_{jk}(t)$
before renormalising each row to sum to 1.

---

## 4. Epidemiological parameters

**Contact rates.** The within-location contact-rate parameter is
$\lambda_W = 13.0$ contacts per person per day, following the POLYMOD
population-representative contact survey for European populations
(Mossong et al.\ 2008). The between-location (away) contact-rate parameter
is $\lambda_B = 0.30 \times \lambda_W$, a modelling assumption reflecting
reduced contact intensity outside the home location; this ratio is broadly
consistent with estimates of community-to-household contact rate ratios
from contact survey data. For Scenario C (transient amplification),
$\lambda_{B,C} = 0.05 \times \lambda_W$.

**Generation-time distribution.** The infectiousness profile $p(a_E)$ is
a discretised Gamma distribution:
$$p(a_E) = \Pr(a_E \leq A < a_E + 1 \mid A \sim \operatorname{Gamma}(\alpha, \beta)),$$
with $\mathbb{E}[A] = 5.5$ days and $\operatorname{SD}[A] = 1.8$ days
(shape $\alpha = (5.5/1.8)^2 \approx 9.34$, scale $\beta = 1.8^2/5.5 \approx 0.59$ days),
truncated at 25 days. These parameters are consistent with estimates of the
intrinsic generation-time distribution for SARS-CoV-2 reported in the literature
(Hart et al.\ 2022).

**Basic reproduction number.** We target $\mathcal{R}_0 = 1.5$ (Scenarios A–C).
A calibration scalar $\kappa$ is computed at $t = 0$ from the spectral radius of
the initial $R$-matrix under full susceptibility, and $\lambda_W$, $\lambda_B$
are multiplied by $\kappa$ so that $\rho(R(t=0)) = 1.5$.

---

## 5. PDE simulation of the epidemic

The epidemic is governed by the age-of-exposure (structured-population) partial
differential equation

$$\frac{\partial E_j}{\partial t} + \frac{\partial E_j}{\partial a_E} = 0,
\quad t > 0,\; a_E > 0, \tag{Eq.\ 2}$$

with boundary condition (the incidence kernel)

$$E_j(t, 0) = \sum_k \sum_{a_E} K_{kj}(t, a_E)\, E_k(t, a_E), \tag{Eq.\ 4}$$

where the kernel is

$$K_{kj}(t, a_E) = \text{prob\_peak} \cdot S_j(t) \cdot \text{base\_K}[k,j] \cdot p(a_E),$$

and $\text{base\_K}[k,j] = \lambda_W f_{jk} f_{kk}/N^k_\text{eff}
+ \lambda_B \sum_{l \neq k} f_{jl} f_{kl}/N^l_\text{eff}$
is the two-component (within/between) infection kernel integrated over
the infectiousness profile.

**Numerical scheme.** Equations (2)–(4) are solved by first-order
upwind finite differences on a uniform grid with $\Delta t = \Delta a_E = 1$ day.
At each step the age profile is shifted one slot (upwind advection),
and new infections $E_j(t, 0)$ are drawn from a Poisson distribution with
mean $\max(S_j \sum_k \text{base\_K}[k,j] \cdot \text{prob\_peak} \cdot
             \sum_{a_E} p(a_E) E_k(t-1, a_E),\; 0)$,
capped at the current susceptible count. Susceptible depletion is tracked
exactly: $S_j(t) = S_j(t-1) - E_j(t,0) + b \, N_j - d \, S_j(t-1)$,
where $b = d = 3 \times 10^{-5}$ day$^{-1}$ are equal birth and death rates
(open population; included for model generality but have negligible effect
over 365 days).

**Instantaneous reproduction matrix.** At each time step, the $L \times L$
matrix $R_{kj}(t)$ is computed from the current susceptible vector:
$$R_{kj}(t) = \text{prob\_peak}\cdot \mathbf{1}^\top p \cdot S_j(t) \cdot \text{base\_K}[k,j].$$
The network reproduction number $\mathcal{R}(t) = \rho(R(t))$ is the spectral
radius (dominant eigenvalue modulus) of this matrix. Reactivity is
$\sigma(t) = \rho\bigl((R(t) + R(t)^\top)/2\bigr) = \|R(t)\|_2$.

---

## 6. Type reproduction numbers

The scalar type reproduction number $T_j(t)$ (Roberts \& Heesterbeek 2003) is

$$T_j(t) = R_{jj} + R_{j\mathcal{J}}\,(I - R_{\mathcal{J}\mathcal{J}})^{-1} R_{\mathcal{J}j},$$

where $\mathcal{J} = \{0,\ldots,L-1\} \setminus \{j\}$.
$T_j$ is defined only when $\rho(R_{\mathcal{J}\mathcal{J}}) < 1$;
when the background network is itself above threshold, $T_j \to \infty$
(and is reported as `NaN`). The group type reproduction number
$T^P_\text{type}(t) = \rho\bigl(R_{PP} + R_{PQ}(I-R_{QQ})^{-1}R_{QP}\bigr)$
aggregates over a set $P$ of locations.

---

## 7. References

- Mossong J et al. (2008) *PLOS Med* 5(3):e74 —
  POLYMOD contact survey; $\bar{c} = 13$ contacts/day used as $\lambda_W$.
- Simini F et al. (2012) *Nature* 484:96–100 —
  Radiation model; population-attractiveness exponent $p = 0.5$ for default scenarios.
- Tizzoni M et al. (2014) *PLOS Comput Biol* 10(7):e1003716 —
  Validation of CDR-based mobility proxies for epidemic modelling; day-of-week patterns.
- Wesolowski A et al. (2015) *PLOS Comput Biol* 11(7):e1004267 —
  Regional mobility modelling in sub-Saharan Africa using CDRs; spatial interaction models.
- Meredith HR, Giles JR, Wesolowski A et al. (2021) *eLife* 10:e68441 —
  Human mobility patterns in rural sub-Saharan Africa including Zambia;
  characterisation of long-distance and rural travel.
- Hart WS et al. (2022) *Lancet Infect Dis* 22(5):603–610 —
  Intrinsic generation-time distribution for SARS-CoV-2 Alpha variant;
  informs choice of mean $\approx 5.5$ days.
- Roberts MG & Heesterbeek JAP (2003) *Proc R Soc B* 270:1359–1364 —
  Type reproduction number $T_j$; Eqs (33) and (35) in this manuscript.
