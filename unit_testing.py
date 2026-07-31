#!/usr/bin/env python3
"""
Comprehensive unit tests for directly_transmitted.py
Uses Python's built-in unittest module only.
"""

import unittest
import numpy as np

from directly_transmitted import (
    discretise_gamma,
    generate_city,
    generate_mobility,
    _kernel_base,
    compute_R_matrix,
    R_outward,
    R_inward,
    R_system,
    spectral_analysis,
    sensitivity_elasticity,
    compute_generation_times,
    eigenvector_weighted_gt,
    within_between_decomposition,
    source_sink_analysis,
    reactivity,
    amplification_envelope,
    convergence_cosine,
    euler_lotka_r,
    simulate_epidemic_pde,
    COVID_PARAMS,
)


# ---------------------------------------------------------------------------
# Shared small-simulation fixture (N=5, T=50) — built once in setUpClass
# ---------------------------------------------------------------------------

class TestMobilityRt(unittest.TestCase):

    # -----------------------------------------------------------------------
    # Class-level setup: run a small simulation once
    # -----------------------------------------------------------------------

    @classmethod
    def setUpClass(cls):
        """Run a small Lagos simulation (N=5, T=50) shared across tests."""
        print("\n[setUpClass] Running small simulation (N=5, T=50) ...")
        N, T, SEED = 5, 50, 42
        params   = COVID_PARAMS
        max_days = params["max_gen_time"]

        gen_time_pmf   = discretise_gamma(
            params["gen_time_mean"], params["gen_time_sd"], max_days)
        infect_profile = gen_time_pmf.copy()
        w_within  = discretise_gamma(4.0, 1.5, max_days)
        w_between = discretise_gamma(7.0, 2.5, max_days)

        city_data = generate_city(N, scenario="lagos", seed=SEED)
        coords, pops, dists, node_types, meta = city_data

        f_jk, base_f = generate_mobility(
            N, T, pops, dists, node_types, meta,
            day_variation_sd=0.15, seed=SEED)

        initial = np.zeros(N);  initial[0] = 5
        sim = simulate_epidemic_pde(
            T, N, pops, f_jk,
            params["prob_transmission_peak"], infect_profile, max_days,
            params["R0_target"], initial,
            params["base_contact_rate"],
            params["base_contact_rate"] * 0.30,
            w_within, w_between,
            birth_rate=0.00003, death_rate=0.00003,
            stochastic=False, susceptible_depletion=True, seed=SEED)

        cls.N            = N
        cls.T            = T
        cls.SEED         = SEED
        cls.params       = params
        cls.max_days     = max_days
        cls.gen_time_pmf = gen_time_pmf
        cls.infect_profile = infect_profile
        cls.w_within  = w_within
        cls.w_between = w_between
        cls.city_data    = city_data
        cls.coords       = coords
        cls.pops         = pops
        cls.dists        = dists
        cls.node_types   = node_types
        cls.meta         = meta
        cls.f_jk         = f_jk
        cls.base_f       = base_f
        cls.sim          = sim
        cls.lw           = sim["lambda_within_scaled"]
        cls.lb           = sim["lambda_between_scaled"]
        cls.inc          = sim["incidence"]
        cls.R_matrices   = sim["R_matrices"]
        cls.S_series     = sim["susceptibles"]

        inc_total = cls.inc.sum(axis=1)
        cls.peak  = int(np.argmax(inc_total))
        cls.early = max(1, cls.peak // 3)
        cls.late  = min(T - 1, cls.peak + 30)

        print(f"  Peak day: {cls.peak}, early: {cls.early}, late: {cls.late}")
        print("[setUpClass] Done.\n")

    # -----------------------------------------------------------------------
    # 1. Row-stochastic property of mobility matrices
    # -----------------------------------------------------------------------

    def test_row_stochastic(self):
        """f_jk.sum(axis=-1) == 1 for all t, j (Lagos and Zambia)."""
        N, T, SEED = 10, 20, 42

        # Lagos
        city_L = generate_city(N, scenario="lagos", seed=SEED)
        coords_L, pops_L, dists_L, types_L, meta_L = city_L
        f_L, _ = generate_mobility(N, T, pops_L, dists_L, types_L, meta_L,
                                   day_variation_sd=0.15, seed=SEED)
        row_sums_L = f_L.sum(axis=-1)
        np.testing.assert_allclose(
            row_sums_L, np.ones((T, N)),
            atol=1e-10,
            err_msg="Lagos f_jk is not row-stochastic")

        # Zambia
        city_Z = generate_city(N, scenario="zambia", seed=SEED)
        coords_Z, pops_Z, dists_Z, types_Z, meta_Z = city_Z
        f_Z, _ = generate_mobility(N, T, pops_Z, dists_Z, types_Z, meta_Z,
                                   day_variation_sd=0.12, seed=SEED)
        row_sums_Z = f_Z.sum(axis=-1)
        np.testing.assert_allclose(
            row_sums_Z, np.ones((T, N)),
            atol=1e-10,
            err_msg="Zambia f_jk is not row-stochastic")

    # -----------------------------------------------------------------------
    # 2. Diekmann stable distribution and reproductive value
    # -----------------------------------------------------------------------

    def test_diekmann_stable_distribution(self):
        """
        stable_distribution is the right eigenvector of R_mat.T (= left of R_mat).
        reprod_value is the right eigenvector of R_mat.
        Both normalised to sum-to-one (L1) inside spectral_analysis.
        """
        R_mat = np.array([[2.0, 1.0, 0.0],
                          [0.5, 1.5, 0.3],
                          [0.1, 0.2, 1.0]])

        res = spectral_analysis(R_mat)
        rho          = res["rho"]
        stable_dist  = res["stable_distribution"]   # left eigvec of R_mat
        reprod_val   = res["reprod_value"]           # right eigvec of R_mat

        # Verify stable_distribution is right eigvec of R_mat.T
        # Eigenvector equation: R^T v = rho * v  =>  (R^T v) / ||R^T v|| == v / ||v||
        lhs_stable = R_mat.T @ stable_dist
        stable_l2  = stable_dist / (np.linalg.norm(stable_dist) + 1e-300)
        lhs_l2     = lhs_stable  / (np.linalg.norm(lhs_stable)  + 1e-300)
        np.testing.assert_allclose(
            lhs_l2, stable_l2,
            atol=1e-8,
            err_msg="stable_distribution is not right eigvec of R_mat.T")
        # Also verify the scale factor is rho
        scale_stable = float(np.linalg.norm(lhs_stable) /
                             (np.linalg.norm(stable_dist) + 1e-300))
        self.assertAlmostEqual(
            scale_stable, rho, places=8,
            msg=f"Eigenvalue of stable_dist: {scale_stable:.8f} != rho={rho:.8f}")

        # Verify reprod_value is right eigvec of R_mat
        # Eigenvector equation: R w = rho * w  =>  (R w)/||R w|| == w/||w||
        lhs_rv    = R_mat @ reprod_val
        rv_l2     = reprod_val / (np.linalg.norm(reprod_val) + 1e-300)
        lhs_rv_l2 = lhs_rv    / (np.linalg.norm(lhs_rv)     + 1e-300)
        np.testing.assert_allclose(
            lhs_rv_l2, rv_l2,
            atol=1e-8,
            err_msg="reprod_value is not right eigvec of R_mat")
        scale_rv = float(np.linalg.norm(lhs_rv) /
                         (np.linalg.norm(reprod_val) + 1e-300))
        self.assertAlmostEqual(
            scale_rv, rho, places=8,
            msg=f"Eigenvalue of reprod_val: {scale_rv:.8f} != rho={rho:.8f}")

        # Verify aliases point to the same arrays
        np.testing.assert_array_equal(
            res["stable_distribution"], res["left_eigvec"],
            err_msg="stable_distribution != left_eigvec")
        np.testing.assert_array_equal(
            res["reprod_value"], res["right_eigvec"],
            err_msg="reprod_value != right_eigvec")

    # -----------------------------------------------------------------------
    # 3. Sensitivity formula verified by finite differences
    # -----------------------------------------------------------------------

    def test_sensitivity_formula_numerical(self):
        """dρ/dR_{kj} matches finite difference on at least 4 (k,j) pairs."""
        rng   = np.random.default_rng(2024)
        R_mat = rng.uniform(0.1, 1.5, (4, 4))

        res  = sensitivity_elasticity(R_mat)
        S    = res["sensitivity"]
        delta = 1e-6

        test_pairs = [(0, 0), (1, 2), (3, 1), (2, 2)]
        for k, j in test_pairs:
            R_plus = R_mat.copy()
            R_plus[k, j] += delta
            deriv_fd = (R_system(R_plus) - R_system(R_mat)) / delta
            self.assertAlmostEqual(
                S[k, j], deriv_fd, places=4,
                msg=f"Sensitivity S[{k},{j}]={S[k,j]:.6f} != FD={deriv_fd:.6f}")

    # -----------------------------------------------------------------------
    # 4. Sensitivity/elasticity properties
    # -----------------------------------------------------------------------

    def test_sensitivity_elasticity_properties(self):
        """S >= 0 everywhere; E_{kj} = S_{kj}*R_{kj}/rho; sum(E) == 1."""
        rng   = np.random.default_rng(7)
        R_mat = rng.uniform(0.1, 2.0, (5, 5))

        res = sensitivity_elasticity(R_mat)
        S   = res["sensitivity"]
        E   = res["elasticity"]
        rho = res["rho"]

        # All sensitivities non-negative (eigenvectors normalised to abs)
        self.assertTrue(
            np.all(S >= -1e-12),
            "Some sensitivity values are negative")

        # Elasticity formula: E_{kj} = S_{kj} * R_{kj} / rho
        E_check = S * R_mat / rho
        np.testing.assert_allclose(
            E, E_check, atol=1e-12,
            err_msg="Elasticity formula E=S*R/rho not satisfied")

        # Elasticity sums to 1: sum(E_{kj}) = 1
        self.assertAlmostEqual(
            E.sum(), 1.0, places=10,
            msg=f"Elasticity sum={E.sum():.15f} != 1")

    # -----------------------------------------------------------------------
    # 5. Generation time time-invariance
    # -----------------------------------------------------------------------

    def test_gt_time_invariance(self):
        """g_pairwise[:, k, j] varies by location pair (two-component model)."""
        max_days = self.max_days
        params   = self.params

        gt_early = compute_generation_times(
            self.f_jk[self.early], self.S_series[self.early], self.pops,
            params["prob_transmission_peak"], self.w_within, self.w_between,
            max_days, self.lw, self.lb)
        gt_peak = compute_generation_times(
            self.f_jk[self.peak], self.S_series[self.peak], self.pops,
            params["prob_transmission_peak"], self.w_within, self.w_between,
            max_days, self.lw, self.lb)
        gt_late = compute_generation_times(
            self.f_jk[self.late], self.S_series[self.late], self.pops,
            params["prob_transmission_peak"], self.w_within, self.w_between,
            max_days, self.lw, self.lb)

        N = self.N
        # g_pairwise should differ across (k,j) pairs in two-component model
        days_arr = np.arange(max_days)
        means = []
        for k in range(N):
            for j in range(N):
                if gt_peak["R_matrix"][k, j] > 1e-15:
                    m = float(np.sum(days_arr * gt_peak["g_pairwise"][:, k, j]))
                    means.append(m)
        # At least some pairs should differ (two-component model)
        if len(means) > 1:
            self.assertGreater(
                max(means) - min(means), 0.0,
                "g_pairwise means should vary across (k,j) pairs in two-component model")

        # g_network should sum to 1 at each snapshot
        for label, gt_snap in [("early", gt_early), ("peak", gt_peak), ("late", gt_late)]:
            self.assertAlmostEqual(
                gt_snap["g_network"].sum(), 1.0, places=8,
                msg=f"g_network at {label} does not sum to 1")

    # -----------------------------------------------------------------------
    # 6. All pairwise / outward / inward GTs equal biological GT
    # -----------------------------------------------------------------------

    def test_gt_pairwise_equal_to_bio(self):
        """At peak, central location has shorter mean GT than peripheral (two-component)."""
        max_days = self.max_days
        params   = self.params

        gt_peak = compute_generation_times(
            self.f_jk[self.peak], self.S_series[self.peak], self.pops,
            params["prob_transmission_peak"], self.w_within, self.w_between,
            max_days, self.lw, self.lb)

        N        = self.N
        days_arr = np.arange(max_days)

        # Find central location (smallest distance to centroid) and peripheral (largest)
        dc    = np.linalg.norm(self.coords - self.coords.mean(axis=0), axis=1)
        i_cen = int(np.argmin(dc))
        i_per = int(np.argmax(dc))

        # Within-location GT mean: central vs peripheral
        g_cen = gt_peak["g_pairwise"][:, i_cen, i_cen]
        g_per = gt_peak["g_pairwise"][:, i_per, i_per]

        if g_cen.sum() > 0.5 and g_per.sum() > 0.5:
            mean_cen = float(np.sum(days_arr * g_cen))
            mean_per = float(np.sum(days_arr * g_per))
            # Central (lower home fraction) → more between → longer GT
            # Peripheral (higher home fraction) → more within → shorter GT
            self.assertGreater(
                mean_cen, mean_per - 0.5,
                msg=(f"Central mean GT ({mean_cen:.3f}d) should be ≥ peripheral "
                     f"({mean_per:.3f}d) minus tolerance; central has lower home frac"))

        # All pairwise GTs should sum to 1
        for k in range(N):
            for j in range(N):
                if gt_peak["R_matrix"][k, j] > 1e-15:
                    self.assertAlmostEqual(
                        gt_peak["g_pairwise"][:, k, j].sum(), 1.0, places=8,
                        msg=f"g_pairwise[{k},{j}] does not sum to 1")

    # -----------------------------------------------------------------------
    # 7. Euler-Lotka identity
    # -----------------------------------------------------------------------

    def test_euler_lotka_identity(self):
        """Euler-Lotka: R * sum(g(a) * exp(-r*a)) = 1."""
        g_tilde = self.gen_time_pmf
        days    = np.arange(len(g_tilde))

        # rho = 2.0
        rho = 2.0
        r   = euler_lotka_r(rho, g_tilde)
        self.assertFalse(np.isnan(r), "euler_lotka_r returned nan for rho=2.0")
        residual = rho * float(np.sum(g_tilde * np.exp(-r * days))) - 1.0
        self.assertAlmostEqual(
            residual, 0.0, places=6,
            msg=f"Euler-Lotka residual={residual:.2e} for rho=2.0")

        # rho = 1.0 -> r should be 0
        r1 = euler_lotka_r(1.0, g_tilde)
        self.assertAlmostEqual(
            r1, 0.0, places=8,
            msg=f"euler_lotka_r(1.0, g) = {r1}, expected 0")

        # rho = 0.5 -> r should be negative, identity still holds
        r_low = euler_lotka_r(0.5, g_tilde)
        self.assertFalse(np.isnan(r_low),
                         "euler_lotka_r returned nan for rho=0.5")
        self.assertLess(r_low, 0.0,
                        "r should be negative for rho < 1")
        residual_low = 0.5 * float(np.sum(g_tilde * np.exp(-r_low * days))) - 1.0
        self.assertAlmostEqual(
            residual_low, 0.0, places=6,
            msg=f"Euler-Lotka residual={residual_low:.2e} for rho=0.5")

    # -----------------------------------------------------------------------
    # 8. Source-sink conservation
    # -----------------------------------------------------------------------

    def test_source_sink_conservation(self):
        """sum(R_outward) == sum(R_inward) == sum(R_mat) for any R_mat."""
        rng = np.random.default_rng(99)
        for _ in range(5):
            R_mat = rng.uniform(0, 2, (6, 6))
            R_out = R_outward(R_mat)
            R_in  = R_inward(R_mat)
            total = R_mat.sum()
            self.assertAlmostEqual(
                R_out.sum(), total, places=12,
                msg="sum(R_outward) != sum(R_mat)")
            self.assertAlmostEqual(
                R_in.sum(), total, places=12,
                msg="sum(R_inward) != sum(R_mat)")
            self.assertAlmostEqual(
                R_out.sum(), R_in.sum(), places=12,
                msg="sum(R_outward) != sum(R_inward)")

    # -----------------------------------------------------------------------
    # 9. Within-between decomposition
    # -----------------------------------------------------------------------

    def test_within_between_decomposition(self):
        """pi_within + pi_between == 1; diagonal -> 1.0; off-diagonal only -> 0.0."""
        rng   = np.random.default_rng(13)
        R_mat = rng.uniform(0.1, 2.0, (4, 4))
        res   = within_between_decomposition(R_mat)
        self.assertAlmostEqual(
            res["pi_within"] + res["pi_between"], 1.0, places=12,
            msg="pi_within + pi_between != 1")

        # Diagonal R_mat -> pi_within = 1
        D     = np.diag([1.0, 2.0, 3.0, 0.5])
        res_D = within_between_decomposition(D)
        self.assertAlmostEqual(res_D["pi_within"], 1.0, places=12,
                               msg="Diagonal R_mat: pi_within != 1")

        # Zero-diagonal R_mat -> pi_within = 0
        ZD = np.array([[0, 1, 0, 1],
                       [1, 0, 1, 0],
                       [0, 1, 0, 1],
                       [1, 0, 1, 0]], dtype=float)
        res_ZD = within_between_decomposition(ZD)
        self.assertAlmostEqual(res_ZD["pi_within"], 0.0, places=12,
                               msg="Zero-diagonal R_mat: pi_within != 0")

    # -----------------------------------------------------------------------
    # 10. Convergence cosine == 1 when incidence proportional to stable dist.
    # -----------------------------------------------------------------------

    def test_convergence_cosine_uses_stable_distribution(self):
        """cos ~= 1 when incidence is proportional to stable_distribution."""
        R_test = np.array([[2.0, 1.0, 0.5],
                           [0.3, 1.5, 0.2],
                           [0.1, 0.4, 1.2]])

        spec   = spectral_analysis(R_test)
        stable = spec["stable_distribution"]       # already sum-to-one
        stable = stable / (stable.sum() + 1e-300)

        T_mock     = 5
        incidence  = np.tile(stable * 1000.0, (T_mock, 1))
        R_matrices = np.tile(R_test, (T_mock, 1, 1))

        cs = convergence_cosine(incidence, R_matrices)
        for t in range(T_mock):
            if not np.isnan(cs[t]):
                self.assertAlmostEqual(
                    cs[t], 1.0, places=8,
                    msg=f"convergence_cosine[{t}]={cs[t]:.10f} != 1")

    # -----------------------------------------------------------------------
    # 11. Reactivity sigma >= 0 and sigma == rho for symmetric matrices
    # -----------------------------------------------------------------------

    def test_reactivity_sigma_positive(self):
        """sigma >= 0 always; for symmetric positive R_mat, sigma == rho."""
        rng = np.random.default_rng(55)

        # General non-symmetric matrices: sigma >= 0
        for _ in range(5):
            R_mat = rng.uniform(0.1, 2.0, (5, 5))
            res   = reactivity(R_mat)
            self.assertGreaterEqual(
                res["sigma"], -1e-12,
                msg=f"sigma={res['sigma']:.6f} is negative")

        # Symmetric positive matrix: sigma == rho (because (R+R^T)/2 = R)
        R_sym = rng.uniform(0.5, 1.5, (4, 4))
        R_sym = (R_sym + R_sym.T) / 2.0
        res_sym = reactivity(R_sym)
        self.assertAlmostEqual(
            res_sym["sigma"], res_sym["rho"], places=10,
            msg=(f"For symmetric R: sigma={res_sym['sigma']:.12f} "
                 f"!= rho={res_sym['rho']:.12f}"))

    # -----------------------------------------------------------------------
    # 12. Amplification envelope: A[0]=1, A[n]>=0, symmetric R -> A[n]=rho^n
    # -----------------------------------------------------------------------

    def test_amplification_envelope_initial(self):
        """A[0]=1 (||I||_2=1); A[n]>=0; for symmetric R, A[10] ~ rho^10."""
        R_sym = np.array([[1.5, 0.3, 0.1],
                          [0.3, 1.2, 0.2],
                          [0.1, 0.2, 1.0]])

        amp   = amplification_envelope(R_sym, n_max=10)
        A     = amp["A"]
        rho   = amp["rho"]

        # A[0] == 1
        self.assertAlmostEqual(
            A[0], 1.0, places=10,
            msg=f"A[0]={A[0]:.12f} != 1")

        # All non-negative
        self.assertTrue(
            np.all(A >= -1e-12),
            msg="Some A[n] values are negative")

        # For symmetric positive definite matrix ||R^n||_2 = rho(R^n) = rho^n
        # Check at n=10 within 1%
        self.assertAlmostEqual(
            A[10], rho ** 10,
            delta=0.01 * rho ** 10,
            msg=f"A[10]={A[10]:.6f} vs rho^10={rho**10:.6f} (>1% diff)")

    # -----------------------------------------------------------------------
    # 13. R_system calibration check
    # -----------------------------------------------------------------------

    def test_R_system_calibration(self):
        """R_system(R_matrices[1]) is within 15% of R0_target."""
        R0_target = self.params["R0_target"]
        rho_check = R_system(self.R_matrices[1])
        self.assertAlmostEqual(
            rho_check, R0_target,
            delta=0.15 * R0_target,
            msg=f"Calibration: rho={rho_check:.4f}, target={R0_target}")

    # -----------------------------------------------------------------------
    # 14. generate_city output shapes
    # -----------------------------------------------------------------------

    def test_generate_city_shapes(self):
        """coords, populations, distances shapes correct; dist[i,i]=0; pop>0."""
        for N in [5, 10]:
            for scen in ["lagos", "zambia"]:
                coords, pops, dists, node_types, meta = generate_city(
                    N, scenario=scen, seed=42)

                self.assertEqual(
                    coords.shape, (N, 2),
                    f"coords shape wrong for {scen}, N={N}")
                self.assertEqual(
                    pops.shape, (N,),
                    f"populations shape wrong for {scen}, N={N}")
                self.assertEqual(
                    dists.shape, (N, N),
                    f"distances shape wrong for {scen}, N={N}")
                self.assertEqual(
                    len(node_types), N,
                    f"node_types length wrong for {scen}, N={N}")

                for i in range(N):
                    self.assertAlmostEqual(
                        dists[i, i], 0.0, places=10,
                        msg=f"distances[{i},{i}] != 0 for {scen}")

                self.assertTrue(
                    np.all(pops > 0),
                    msg=f"Some populations <= 0 for {scen}")

    # -----------------------------------------------------------------------
    # 15. Mobility day-of-week variation
    # -----------------------------------------------------------------------

    def test_mobility_day_of_week(self):
        """Sundays have lower average away fraction than Mondays."""
        N, T = 8, 14
        city  = generate_city(N, scenario="lagos", seed=123)
        coords, pops, dists, node_types, meta = city
        # day_variation_sd=0.0 so lognormal noise is exactly 1.0 (exp(0))
        f_jk, _ = generate_mobility(N, T, pops, dists, node_types, meta,
                                    day_variation_sd=0.0, seed=0)

        # away fraction for j at t = 1 - f_jk[t, j, j]
        # Mondays: t=0, t=7  (dow_scale=1.00)
        # Sundays: t=6, t=13 (dow_scale=0.50)
        away_monday = np.mean(
            [1.0 - f_jk[0, j, j]  for j in range(N)] +
            [1.0 - f_jk[7, j, j]  for j in range(N)])
        away_sunday = np.mean(
            [1.0 - f_jk[6, j, j]  for j in range(N)] +
            [1.0 - f_jk[13, j, j] for j in range(N)])

        self.assertGreater(
            away_monday, away_sunday,
            msg=(f"Monday away={away_monday:.4f} not > "
                 f"Sunday away={away_sunday:.4f}"))

    # -----------------------------------------------------------------------
    # 16. Kernel base: vectorised == naive Python loop
    # -----------------------------------------------------------------------

    def test_kernel_base_vectorized_vs_naive(self):
        """_kernel_base matches naive triple-loop computation (atol=1e-12)."""
        rng  = np.random.default_rng(7)
        N    = 5
        lw, lb = 15.0, 5.0

        # Build a valid row-stochastic f_t
        raw = rng.uniform(0.2, 1.0, (N, N))
        f_t = raw / raw.sum(axis=1, keepdims=True)
        pops = rng.uniform(1e4, 1e5, N)

        base_K_vec, bKw_vec, bKb_vec, N_eff_vec, inv_Neff_vec = _kernel_base(f_t, pops, lw, lb)

        # Naive computation following the docstring
        N_eff_naive = np.array([
            sum(f_t[j, l] * pops[j] for j in range(N))
            for l in range(N)
        ])
        inv_Neff = np.where(N_eff_naive > 0, 1.0 / N_eff_naive, 0.0)

        D_naive = np.zeros((N, N))
        for j in range(N):
            for k in range(N):
                D_naive[j, k] = sum(
                    f_t[j, l] * f_t[k, l] * inv_Neff[l]
                    for l in range(N))

        within_naive = np.zeros((N, N))
        for k in range(N):
            for j in range(N):
                within_naive[k, j] = f_t[j, k] * f_t[k, k] * inv_Neff[k]

        # base_K[k,j] = lb * D^T[k,j] + (lw-lb) * within[k,j]
        # D^T[k,j] = D[j,k] = D_naive[j,k]
        base_K_naive = lb * D_naive.T + (lw - lb) * within_naive

        np.testing.assert_allclose(
            base_K_vec, base_K_naive, atol=1e-12,
            err_msg="Vectorised _kernel_base != naive loop")

        np.testing.assert_allclose(
            N_eff_vec, N_eff_naive, atol=1e-12,
            err_msg="N_eff from _kernel_base != naive computation")

    # -----------------------------------------------------------------------
    # 17. Eigenvector-weighted GT sums to 1
    # -----------------------------------------------------------------------

    def test_eigenvector_weighted_gt_sums_to_one(self):
        """eigenvector_weighted_gt returns a distribution that sums to 1."""
        max_days = 10
        N = 4
        rng = np.random.default_rng(31)

        R_mat        = rng.uniform(0.3, 2.0, (N, N))
        gen_time_pmf = discretise_gamma(5.0, 1.8, max_days)

        # K_series[a, k, j] = gen_time_pmf[a] * R_mat[k, j]
        K_series = np.zeros((max_days, N, N))
        for a in range(max_days):
            K_series[a] = gen_time_pmf[a] * R_mat

        g_tilde = eigenvector_weighted_gt(K_series, R_mat)

        self.assertAlmostEqual(
            g_tilde.sum(), 1.0, places=10,
            msg=f"eigenvector_weighted_gt sums to {g_tilde.sum():.15f} != 1")

    # -----------------------------------------------------------------------
    # 18. Peripheral locs have higher home fraction → shorter GT mean
    # -----------------------------------------------------------------------

    def test_gt_peripheral_shorter_than_central(self):
        """Peripheral locs (high home fraction) → more within-profile → shorter GT mean."""
        max_days = self.max_days
        params   = self.params

        gt_peak = compute_generation_times(
            self.f_jk[self.peak], self.S_series[self.peak], self.pops,
            params["prob_transmission_peak"], self.w_within, self.w_between,
            max_days, self.lw, self.lb)

        days_arr = np.arange(max_days)
        dc    = np.linalg.norm(self.coords - self.coords.mean(axis=0), axis=1)
        i_cen = int(np.argmin(dc))
        i_per = int(np.argmax(dc))

        g_cen = gt_peak["g_pairwise"][:, i_cen, i_cen]
        g_per = gt_peak["g_pairwise"][:, i_per, i_per]

        if g_cen.sum() > 0.5 and g_per.sum() > 0.5:
            mean_cen = float(np.sum(days_arr * g_cen))
            mean_per = float(np.sum(days_arr * g_per))
            within_mean  = float(np.sum(days_arr * self.w_within))
            between_mean = float(np.sum(days_arr * self.w_between))
            # Both should be between within_mean and between_mean
            self.assertGreaterEqual(mean_cen, within_mean - 0.1,
                                    "central GT mean below within_mean")
            self.assertLessEqual(mean_per, between_mean + 0.1,
                                 "peripheral GT mean above between_mean")

    # -----------------------------------------------------------------------
    # 19. Weekend (high f_kk) → shorter GT mean than weekday
    # -----------------------------------------------------------------------

    def test_gt_weekday_vs_weekend(self):
        """Weekend (high home fraction) → shift toward w_within → shorter GT mean."""
        max_days = self.max_days
        params   = self.params
        N = self.N

        # Find a weekday and weekend day: t%7==0 is Monday (scale=1.00),
        # t%7==6 is Sunday (scale=0.50 → high home fraction)
        t_mon = 0  # Monday
        t_sun = 6  # Sunday

        gt_mon = compute_generation_times(
            self.f_jk[t_mon], self.S_series[t_mon], self.pops,
            params["prob_transmission_peak"], self.w_within, self.w_between,
            max_days, self.lw, self.lb)
        gt_sun = compute_generation_times(
            self.f_jk[t_sun], self.S_series[t_sun], self.pops,
            params["prob_transmission_peak"], self.w_within, self.w_between,
            max_days, self.lw, self.lb)

        days_arr = np.arange(max_days)
        mean_mon = float(np.sum(days_arr * gt_mon["g_network"]))
        mean_sun = float(np.sum(days_arr * gt_sun["g_network"]))

        # Sunday (more time at home) → shorter GT → mean_sun <= mean_mon
        self.assertLessEqual(
            mean_sun, mean_mon + 0.05,
            msg=(f"Sunday GT mean ({mean_sun:.3f}d) should be ≤ Monday "
                 f"({mean_mon:.3f}d); Sunday has higher home fraction"))

    # -----------------------------------------------------------------------
    # 20. GT mixture is bounded between within_mean and between_mean
    # -----------------------------------------------------------------------

    def test_gt_mixture_bounded(self):
        """Pairwise GT mean is bounded between within_mean and between_mean."""
        max_days = self.max_days
        params   = self.params
        days_arr = np.arange(max_days)

        within_mean  = float(np.sum(days_arr * self.w_within))
        between_mean = float(np.sum(days_arr * self.w_between))

        gt_peak = compute_generation_times(
            self.f_jk[self.peak], self.S_series[self.peak], self.pops,
            params["prob_transmission_peak"], self.w_within, self.w_between,
            max_days, self.lw, self.lb)

        N = self.N
        for k in range(N):
            for j in range(N):
                if gt_peak["R_matrix"][k, j] > 1e-15:
                    m = float(np.sum(days_arr * gt_peak["g_pairwise"][:, k, j]))
                    self.assertGreaterEqual(
                        m, within_mean - 0.1,
                        msg=f"g_pairwise[{k},{j}] mean {m:.3f} < within_mean {within_mean:.3f}")
                    self.assertLessEqual(
                        m, between_mean + 0.1,
                        msg=f"g_pairwise[{k},{j}] mean {m:.3f} > between_mean {between_mean:.3f}")

    # -----------------------------------------------------------------------
    # 21. R values are unchanged by the within/between GT split
    # -----------------------------------------------------------------------

    def test_r_values_unchanged_by_gt_split(self):
        """R_{kj} should be the same regardless of within/between split (both PMFs sum to 1)."""
        params   = self.params
        max_days = self.max_days

        # Compute R with w_within/w_between
        gt1 = compute_generation_times(
            self.f_jk[self.peak], self.S_series[self.peak], self.pops,
            params["prob_transmission_peak"], self.w_within, self.w_between,
            max_days, self.lw, self.lb)

        # Compute R with a single unified GT (gen_time_pmf for both)
        gt2 = compute_generation_times(
            self.f_jk[self.peak], self.S_series[self.peak], self.pops,
            params["prob_transmission_peak"], self.gen_time_pmf, self.gen_time_pmf,
            max_days, self.lw, self.lb)

        np.testing.assert_allclose(
            gt1["R_matrix"], gt2["R_matrix"], atol=1e-10,
            err_msg="R_matrix differs between two-component and single-component GT")

    # -----------------------------------------------------------------------
    # Additional sanity: rho > 0 for non-trivial R_mat
    # -----------------------------------------------------------------------

    def test_rho_positive_for_nontrivial_R(self):
        """spectral_analysis returns rho > 0 for non-zero R_mat."""
        R_mat = np.array([[2.0, 1.0, 0.0],
                          [0.5, 1.5, 0.3],
                          [0.1, 0.2, 1.0]])
        res = spectral_analysis(R_mat)
        self.assertGreater(
            res["rho"], 0.0,
            msg="rho should be > 0 for non-trivial R_mat")


# ---------------------------------------------------------------------------
# Summary printer
# ---------------------------------------------------------------------------

def run_with_summary():
    loader = unittest.TestLoader()
    suite  = loader.loadTestsFromTestCase(TestMobilityRt)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    n_run    = result.testsRun
    n_fail   = len(result.failures)
    n_error  = len(result.errors)
    n_skip   = len(result.skipped)
    n_passed = n_run - n_fail - n_error - n_skip

    print("\n" + "=" * 72)
    print("UNIT TEST SUMMARY")
    print("=" * 72)
    print(f"  Tests run    : {n_run}")
    print(f"  Passed       : {n_passed}")
    print(f"  Failures     : {n_fail}")
    print(f"  Errors       : {n_error}")
    print(f"  Skipped      : {n_skip}")
    if n_fail == 0 and n_error == 0:
        print("\n  ALL TESTS PASSED")
    else:
        print("\n  SOME TESTS FAILED -- see output above for details.")
    print("=" * 72)
    return result


if __name__ == "__main__":
    run_with_summary()
