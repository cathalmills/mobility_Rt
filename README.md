# Multi-scale measures of time-varying epidemic spread on human mobility networks

Code for the paper

> **Multi-scale measures of time-varying epidemic spread on human mobility networks**
> Cathal Mills, Benjamin Reddy, William Hart, Robin Thompson, Kris V. Parag,
> Moritz U. G. Kraemer, Christl A. Donnelly, Ben Lambert.
> arXiv: https://doi.org/10.48550/arXiv.2607.28514

People catch and pass on infections wherever they spend their day, not only where they
live. This code uses data on **how much time residents of each place spend in every other
place** to work out, day by day:

- whether the epidemic across all places is growing or shrinking;
- how many infections residents of each place cause, and how many they receive;
- how much infection flows along each route between two places;
- which places and routes matter most for controlling the epidemic;
- whether infections can briefly rise in some places even while the overall epidemic shrinks.

---

## Install

Requires Python 3.12 or later.

```bash
pip install -r requirements-lock.txt   # the exact package versions used for the paper
pip install -e .
```

## Reproduce the paper

From the repository root:

```bash
python3 -m mobility_rt.scripts.run_all     # every figure and table in the paper
python3 -m pytest -q test_mobility_rt.py    # checks that the code is correct
```

Results are written to `out_figs/` (`main/`, `SI/`, `corrected/`, `tables/`).
Figure 1 is a drawing and is not produced by the code. Some file names differ from the
paper's figure numbers: paper Figure 4 is `fig_main_bias_combined.pdf` and paper
Figure 5 is `fig_04_spectral.pdf`.

The notebooks (`*.ipynb`) are for illustration only.

---

## Use it on your own data

### What you need

| Input | What it is | How to get it |
|---|---|---|
| `population` | Number of residents of each of your `L` places | Census |
| `time_spent` | For each day, a table where row `j`, column `k` is the share of the day that residents of place `j` spend in place `k`. Each row adds up to 1. Shape: `(days, L, L)` | Mobile-phone location data: hours residents of `j` spend in `k`, divided by their total hours that day |
| `infections` | New infections among residents of each place on each day. Shape: `(days, L)`, starting from the first infection | Case counts, scaled up for under-reporting if needed |
| Infectiousness | How infectious a person is on each day after being infected | Published estimates for your disease. The paper uses a mean of 5.5 days and a standard deviation of 1.8 days |
| `away_ratio` | How many contacts a person has away from home, relative to at home | The paper assumes 0.3 |
| `R_start` | How many people one infected person infected at the very start, when everyone could still be infected | Published estimates, or an estimate from early case growth |

### Steps

```python
import numpy as np
import mobility_rt as mr

# 1. Your data
time_spent = np.load("time_spent.npy")    # shape (days, L, L); each row adds up to 1
population = np.load("population.npy")    # shape (L,)
infections = np.load("infections.npy")    # shape (days, L): new infections per day

# 2. Your assumptions
profile = mr.discretise_gamma(mean=5.5, sd=1.8, max_days=25)  # infectiousness by days since infection
away_ratio = 0.3        # contacts away from home, relative to at home
R_start = 1.5           # infections caused per infected person when everyone could be infected

# 3. People who can still be infected, in each place, at the start of each day
susceptible = population - np.vstack([np.zeros(len(population)),
                                      np.cumsum(infections, axis=0)[:-1]])

# 4. Set the overall transmission level so that day 0 matches R_start
unscaled = mr.compute_R_matrix(time_spent[0], population, population, 1.0, profile, 1.0, away_ratio)
scale = R_start / mr.R_system(unscaled)

# 5. Results for each day
for day in range(len(time_spent)):
    R = scale * mr.compute_R_matrix(time_spent[day], susceptible[day], population,
                                    1.0, profile, 1.0, away_ratio)
    network_R  = mr.R_system(R)
    outward    = mr.R_outward(R)
    inward     = mr.R_inward(R)
    targets    = mr.sensitivity_elasticity(R)["elasticity"].sum(axis=1)
    end_up     = mr.spectral_analysis(R)["stable_distribution"]
    short_term = mr.reactivity(R)["sigma"]
    needed     = mr.type_reproduction_numbers(R)
```

### What the results mean

| Result | Meaning |
|---|---|
| `network_R` | Whether the epidemic across all places is growing (above 1) or shrinking (below 1) |
| `R[k, j]` | Infections that one infected resident of place `k` causes among residents of place `j` |
| `outward[k]` | Infections that one infected resident of place `k` causes in total, in any place |
| `inward[j]` | Infections that residents of place `j` would receive if one resident of every place were infected |
| `targets[k]` | How much a small cut in transmission from (or, equivalently, to) residents of place `k` would lower `network_R`, as a share of the total. The values add up to 1; the largest show where control helps most |
| `end_up[j]` | Share of all infections that would eventually be among residents of place `j` if conditions stayed as they are today |
| `short_term` | Above 1: infections can rise for a while in some places, even if `network_R` is below 1 |
| `needed[j]` | If above 1, cutting transmission to (or from) residents of place `j` by more than `1 − 1/needed[j]` would stop the epidemic growing. Empty (`nan`) when controlling place `j` alone cannot stop it |

All of these describe one day, assuming conditions stay as they are on that day.

### Testing "what if" changes

Change `time_spent` (for example, cut travel into one place and add that time to
time at home, so each row still adds up to 1), recompute `R`, and compare `network_R`
before and after. To run a whole epidemic forward under your assumptions, use
`mr.simulate_epidemic_pde` (see `mobility_rt/scripts/run_all.py` for an example).

### Things to check

- `infections` should be infections, not reported cases. If only a fraction of
  infections are reported, scale the counts up.
- Everyone in the same place on the same day is assumed equally likely to meet.
- Results are only as good as the time-spent data. Noisy data, especially for places
  with few residents or few infections, can give misleading values.

---

## What is in the repository

```
mobility_rt/                the package: all model calculations (no plotting)
mobility_rt/plotting/       figure code
mobility_rt/scripts/run_all.py   builds every scenario, figure, and table in the paper
test_mobility_rt.py         tests
requirements-lock.txt       exact package versions used for the paper
*.ipynb                     example notebooks (for illustration only)
```

## Licence

For the Author Accepted Manuscript, a CC BY public copyright licence applies (see the
paper's copyright statement).
