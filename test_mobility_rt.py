#!/usr/bin/env python3
"""
Unit tests for the mobility_rt package.
Uses Python's built-in unittest module only.
"""

import unittest
import numpy as np

from mobility_rt import (
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
    reactivity,
    amplification_envelope,
    convergence_cosine,
    euler_lotka_r,
    simulate_epidemic_pde,
    estimate_R_independent,
    type_reproduction_number,
    type_reproduction_numbers,
    _compute_power_mean_spectrum,
    R_PRIOR_SHAPE,
    R_PRIOR_RATE,
    COVID_PARAMS,
)
from scipy.stats import gamma as _gamma
import mobility_rt.config as _cfg


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
        """Universal single-profile model: g_network is time-invariant and sums to 1."""
        max_days = self.max_days
        params   = self.params
        p_aE     = self.gen_time_pmf

        gt_early = compute_generation_times(
            self.f_jk[self.early], self.S_series[self.early], self.pops,
            params["prob_transmission_peak"], p_aE, max_days, self.lw, self.lb)
        gt_peak = compute_generation_times(
            self.f_jk[self.peak], self.S_series[self.peak], self.pops,
            params["prob_transmission_peak"], p_aE, max_days, self.lw, self.lb)
        gt_late = compute_generation_times(
            self.f_jk[self.late], self.S_series[self.late], self.pops,
            params["prob_transmission_peak"], p_aE, max_days, self.lw, self.lb)

        # Under the separable Eq-7 form with a shared infectiousness profile the
        # generation time is UNIVERSAL: g_network(t,a) = p(a)/Σp for all t.
        for label, gt_snap in [("early", gt_early), ("peak", gt_peak), ("late", gt_late)]:
            self.assertAlmostEqual(
                gt_snap["g_network"].sum(), 1.0, places=8,
                msg=f"g_network at {label} does not sum to 1")
        np.testing.assert_allclose(
            gt_early["g_network"], gt_late["g_network"], atol=1e-10,
            err_msg="g_network should be time-invariant (universal GT)")
        np.testing.assert_allclose(
            gt_peak["g_network"], gt_early["g_network"], atol=1e-10,
            err_msg="g_network should be time-invariant (universal GT)")

    # -----------------------------------------------------------------------
    # 6. All pairwise / outward / inward GTs equal biological GT
    # -----------------------------------------------------------------------

    def test_gt_pairwise_equal_to_bio(self):
        """All pairwise / outward / inward / network GTs equal the biological profile."""
        max_days = self.max_days
        params   = self.params
        p_aE     = self.gen_time_pmf
        # Renewal GT distribution excludes same-day (a=0) transmission and is
        # renormalised over a ≥ 1 (matching the forward simulator / compute_R_matrix).
        g_bio    = p_aE.copy(); g_bio[0] = 0.0; g_bio = g_bio / g_bio.sum()

        gt_peak = compute_generation_times(
            self.f_jk[self.peak], self.S_series[self.peak], self.pops,
            params["prob_transmission_peak"], p_aE, max_days, self.lw, self.lb)

        N = self.N

        # Every defined pairwise GT equals the (universal) biological profile.
        for k in range(N):
            for j in range(N):
                if gt_peak["R_matrix"][k, j] > 1e-15:
                    np.testing.assert_allclose(
                        gt_peak["g_pairwise"][:, k, j], g_bio, atol=1e-10,
                        err_msg=f"g_pairwise[{k},{j}] != biological p(a)/Σp")
        # Outward / inward / network GTs also collapse to the same universal shape.
        for k in range(N):
            if gt_peak["R_matrix"].sum(axis=1)[k] > 1e-15:
                np.testing.assert_allclose(
                    gt_peak["g_outward"][:, k], g_bio, atol=1e-10,
                    err_msg=f"g_outward[{k}] != biological p(a)/Σp")
            if gt_peak["R_matrix"].sum(axis=0)[k] > 1e-15:
                np.testing.assert_allclose(
                    gt_peak["g_inward"][:, k], g_bio, atol=1e-10,
                    err_msg=f"g_inward[{k}] != biological p(a)/Σp")
        np.testing.assert_allclose(
            gt_peak["g_network"], g_bio, atol=1e-10,
            err_msg="g_network != biological p(a)/Σp")

    # -----------------------------------------------------------------------
    # 7. Euler-Lotka identity
    # -----------------------------------------------------------------------

    def test_euler_lotka_identity(self):
        """Euler-Lotka: R * Σ_{a≥1} g(a) exp(-r*a) = 1 (a=0 excluded, renormalised)."""
        g_tilde = self.gen_time_pmf
        # euler_lotka_r solves over a ≥ 1, so check the identity with the same profile.
        g_r  = g_tilde.copy(); g_r[0] = 0.0; g_r = g_r / g_r.sum()
        days = np.arange(len(g_r))

        # rho = 2.0
        rho = 2.0
        r   = euler_lotka_r(rho, g_tilde)
        self.assertFalse(np.isnan(r), "euler_lotka_r returned nan for rho=2.0")
        residual = rho * float(np.sum(g_r * np.exp(-r * days))) - 1.0
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
        residual_low = 0.5 * float(np.sum(g_r * np.exp(-r_low * days))) - 1.0
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
        # Sundays: t=6, t=13 (dow_scale=0.75)
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

    def test_gt_universal_across_locations(self):
        """GT shape is location-invariant: peripheral and central pairs share one GT."""
        max_days = self.max_days
        params   = self.params
        p_aE     = self.gen_time_pmf

        gt_peak = compute_generation_times(
            self.f_jk[self.peak], self.S_series[self.peak], self.pops,
            params["prob_transmission_peak"], p_aE, max_days, self.lw, self.lb)

        dc    = np.linalg.norm(self.coords - self.coords.mean(axis=0), axis=1)
        i_cen = int(np.argmin(dc))
        i_per = int(np.argmax(dc))

        g_cen = gt_peak["g_pairwise"][:, i_cen, i_cen]
        g_per = gt_peak["g_pairwise"][:, i_per, i_per]

        # Universal-GT model: central and peripheral within-location GTs are identical.
        np.testing.assert_allclose(
            g_cen, g_per, atol=1e-10,
            err_msg="within-location GT should be identical across locations (universal GT)")

    # -----------------------------------------------------------------------
    # 19. Weekend (high f_kk) → shorter GT mean than weekday
    # -----------------------------------------------------------------------

    def test_gt_weekday_vs_weekend(self):
        """GT shape is invariant to day-of-week mobility: Monday and Sunday coincide."""
        max_days = self.max_days
        params   = self.params
        p_aE     = self.gen_time_pmf

        # Monday (t%7==0, scale=1.00) vs Sunday (t%7==6, scale=0.75, higher home fraction).
        t_mon, t_sun = 0, 6

        gt_mon = compute_generation_times(
            self.f_jk[t_mon], self.S_series[t_mon], self.pops,
            params["prob_transmission_peak"], p_aE, max_days, self.lw, self.lb)
        gt_sun = compute_generation_times(
            self.f_jk[t_sun], self.S_series[t_sun], self.pops,
            params["prob_transmission_peak"], p_aE, max_days, self.lw, self.lb)

        # Universal GT: mobility changes affect R magnitudes, not GT shape.
        np.testing.assert_allclose(
            gt_sun["g_network"], gt_mon["g_network"], atol=1e-10,
            err_msg="g_network should not depend on day-of-week mobility (universal GT)")

    # -----------------------------------------------------------------------
    # 20. GT mixture is bounded between within_mean and between_mean
    # -----------------------------------------------------------------------

    def test_gt_pairwise_mean_equals_bio_mean(self):
        """Every pairwise GT mean equals the biological generation-time mean."""
        max_days = self.max_days
        params   = self.params
        p_aE     = self.gen_time_pmf
        days_arr = np.arange(max_days)

        # GT distribution over a ≥ 1 (a=0 excluded, renormalised), as returned.
        g_r      = p_aE.copy(); g_r[0] = 0.0; g_r = g_r / g_r.sum()
        bio_mean = float(np.sum(days_arr * g_r))

        gt_peak = compute_generation_times(
            self.f_jk[self.peak], self.S_series[self.peak], self.pops,
            params["prob_transmission_peak"], p_aE, max_days, self.lw, self.lb)

        N = self.N
        for k in range(N):
            for j in range(N):
                if gt_peak["R_matrix"][k, j] > 1e-15:
                    m = float(np.sum(days_arr * gt_peak["g_pairwise"][:, k, j]))
                    self.assertAlmostEqual(
                        m, bio_mean, places=8,
                        msg=f"g_pairwise[{k},{j}] mean {m:.4f} != bio mean {bio_mean:.4f}")

    # -----------------------------------------------------------------------
    # 21. R values are unchanged by the within/between GT split
    # -----------------------------------------------------------------------

    def test_r_values_independent_of_gt_shape(self):
        """R_{kj} depends only on the total infectiousness Σp(a), not the GT shape."""
        params   = self.params
        max_days = self.max_days

        # Two infectiousness profiles with identical total a≥1 mass but very
        # different shapes; R = prob_peak·S_j·base_K·Σ_{a≥1}p should be identical.
        # Both are zero at a=0 (same-day excluded), so Σ_{a≥1}p = 1 for each.
        p1 = self.gen_time_pmf.copy(); p1[0] = 0.0; p1 = p1 / p1.sum()  # Gamma on a≥1
        p2 = np.zeros(max_days); p2[1:] = 1.0 / (max_days - 1)          # uniform on a≥1

        gt1 = compute_generation_times(
            self.f_jk[self.peak], self.S_series[self.peak], self.pops,
            params["prob_transmission_peak"], p1, max_days, self.lw, self.lb)
        gt2 = compute_generation_times(
            self.f_jk[self.peak], self.S_series[self.peak], self.pops,
            params["prob_transmission_peak"], p2, max_days, self.lw, self.lb)

        np.testing.assert_allclose(
            gt1["R_matrix"], gt2["R_matrix"], atol=1e-10,
            err_msg="R_matrix should be invariant to GT profile shape (Σp equal)")

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

    # =======================================================================
    # Regression tests added after the 2026 review — each pins a class of bug
    # that previously slipped through (discretisation mean, estimator prior /
    # convolution alignment, a=0 lag, spectral identities, type-R threshold).
    # =======================================================================

    def test_discretise_gamma_recovers_continuous_mean(self):
        """Double-censored discretisation recovers the continuous mean (would catch the
        old floor(X) scheme that biased the mean ~0.5 d low)."""
        for mean, sd in [(5.5, 1.8), (3.0, 1.0), (8.0, 2.5)]:
            pmf = discretise_gamma(mean, sd, 40)
            self.assertAlmostEqual(pmf.sum(), 1.0, places=10)
            self.assertTrue(np.all(pmf >= 0.0))
            disc_mean = float(np.sum(np.arange(40) * pmf))
            self.assertAlmostEqual(
                disc_mean, mean, delta=0.02,
                msg=f"discrete mean {disc_mean:.3f} != continuous {mean} (discretisation bias)")

    def test_discretise_gamma_matches_montecarlo(self):
        """Analytic double-censored pmf == Monte-Carlo of floor(U+X)."""
        rng = np.random.default_rng(0)
        a = (5.5 / 1.8) ** 2; scale = 1.8 ** 2 / 5.5
        X = _gamma.rvs(a, scale=scale, size=2_000_000, random_state=rng)
        U = rng.uniform(0, 1, X.size)
        D = np.floor(U + X).astype(int)
        mc = np.bincount(D, minlength=25)[:25] / X.size
        pmf = discretise_gamma(5.5, 1.8, 25)
        np.testing.assert_allclose(pmf, mc / mc.sum(), atol=3e-3)

    def test_estimator_prior_is_epiestim_default(self):
        """Prior must be EpiEstim's default (mean 5), not mean 0.2 — this is C1.
        Posterior at zero data equals the prior mean R_PRIOR_SHAPE/R_PRIOR_RATE."""
        self.assertAlmostEqual(R_PRIOR_SHAPE / R_PRIOR_RATE, 5.0, places=6,
                               msg="independent-estimator prior mean must be 5 (EpiEstim default)")
        # One case is the smallest observable count; anything larger is arbitrary.
        self.assertEqual(_cfg.R_MIN_WINDOW_CASES, 1.0)
        self.assertFalse(hasattr(_cfg, "R_MIN_WINDOW_INCIDENCE"),
                         "the old tuned reporting threshold has been reintroduced")

    def test_estimator_lambda_matches_forward_model(self):
        """The estimator's total-infectiousness Λ uses the SAME lag convention as the
        forward simulator's force of infection (pins the convolution-alignment bug)."""
        inc = self.inc                      # (T, N) from the shared fixture
        pmf = self.gen_time_pmf
        max_s = len(pmf)
        # Forward-model Λ_j(t) = Σ_{a≥1} p(a) I_j(t-a)
        T, N = inc.shape
        lam_fwd = np.zeros((T, N))
        for t in range(T):
            for a in range(1, max_s):
                if t - a >= 0:
                    lam_fwd[t] += pmf[a] * inc[t - a]
        # Estimator Λ over a 1-day window equals the same convolution
        R_est = estimate_R_independent(inc, pmf, window=1)
        # Reconstruct estimator's Λ where it reported: Λ = (R_PRIOR_SHAPE+I)/R̂ - R_PRIOR_RATE
        t, j = self.peak, int(np.argmax(inc[self.peak]))
        if np.isfinite(R_est[t, j]):
            lam_est = (R_PRIOR_SHAPE + inc[t, j]) / R_est[t, j] - R_PRIOR_RATE
            self.assertAlmostEqual(lam_est, lam_fwd[t, j], places=4,
                                   msg="estimator Λ != forward-model Λ (lag misalignment)")

    def test_estimator_recovers_constant_R(self):
        """A constant-R renewal series is recovered to within a few % once incidence
        is well above the prior (pins the prior mis-specification end-to-end)."""
        pmf = self.gen_time_pmf
        R_true = 1.5
        T = 80
        inc = np.zeros((T, 1)); inc[0, 0] = 50.0
        for t in range(1, T):
            lam = sum(pmf[a] * inc[t - a, 0] for a in range(1, len(pmf)) if t - a >= 0)
            inc[t, 0] = R_true * lam
        R_est = estimate_R_independent(inc, pmf, window=7)
        late = R_est[40:70, 0]
        late = late[np.isfinite(late)]
        self.assertTrue(late.size > 0)
        self.assertAlmostEqual(float(np.mean(late)), R_true, delta=0.08,
                               msg=f"estimator recovered {np.mean(late):.3f}, expected {R_true}")

    def test_R_uses_a1_support(self):
        """compute_R_matrix integrates the GT over a≥1 (excludes a=0), matching the
        simulator — R scales with Σ_{a≥1}p, so a profile with large p[0] gives smaller R."""
        f_t = self.f_jk[self.peak]; S = self.S_series[self.peak]
        base = self.gen_time_pmf
        # profile with substantial a=0 mass but same shape on a≥1
        p_big0 = base.copy(); p_big0[0] += 0.5; p_big0 /= p_big0.sum()
        R1 = compute_R_matrix(f_t, S, self.pops, 0.035, base, self.lw, self.lb)
        R2 = compute_R_matrix(f_t, S, self.pops, 0.035, p_big0, self.lw, self.lb)
        ratio = (R2.sum() / R1.sum())
        expected = p_big0[1:].sum() / base[1:].sum()
        self.assertAlmostEqual(ratio, expected, places=6,
                               msg="compute_R_matrix does not integrate over a≥1")

    def test_rho_bounds_eq24(self):
        """min_k R^k_out ≤ ρ(R) ≤ max_k R^k_out (Eq 24)."""
        rng = np.random.default_rng(7)
        for _ in range(20):
            R = rng.uniform(0, 2, (6, 6))
            rho = R_system(R)
            R_out = R_outward(R)
            self.assertLessEqual(R_out.min() - 1e-9, rho)
            self.assertLessEqual(rho, R_out.max() + 1e-9)

    def test_sigma_geq_rho(self):
        """Reactivity σ = ‖R‖₂ ≥ ρ(R) for any matrix (Eq 37); strict for non-normal."""
        rng = np.random.default_rng(11)
        saw_strict = False
        for _ in range(30):
            R = rng.uniform(0, 2, (5, 5))
            rr = reactivity(R)
            self.assertGreaterEqual(rr["sigma"], rr["rho"] - 1e-9)
            if rr["sigma"] > rr["rho"] + 1e-6:
                saw_strict = True
        self.assertTrue(saw_strict, "expected σ > ρ for some non-normal matrix")

    def test_elasticity_matches_finite_difference(self):
        """ε^{kj} = (R_{kj}/ρ) ∂ρ/∂R_{kj} matches a central finite difference."""
        rng = np.random.default_rng(5)
        R = rng.uniform(0.05, 1.5, (5, 5))
        se = sensitivity_elasticity(R)
        rho0 = R_system(R); h = 1e-6
        for k in range(5):
            for j in range(5):
                Rp = R.copy(); Rp[k, j] += h
                Rm = R.copy(); Rm[k, j] -= h
                drho = (R_system(Rp) - R_system(Rm)) / (2 * h)
                eps_fd = (R[k, j] / rho0) * drho
                self.assertAlmostEqual(se["elasticity"][k, j], eps_fd, places=5,
                                       msg=f"elasticity[{k},{j}] != finite diff")

    def test_type_R_threshold_property(self):
        """T_j > 1 ⟺ ρ(R) > 1 wherever T_j is defined (Eq 55)."""
        rng = np.random.default_rng(13)
        checked = 0
        for _ in range(60):
            R = rng.uniform(0, 0.6, (4, 4))
            rho = R_system(R)
            for Tj in type_reproduction_numbers(R):
                if not np.isnan(Tj):
                    self.assertEqual(Tj > 1.0, rho > 1.0,
                                     msg=f"T_j={Tj:.3f} vs rho={rho:.3f} threshold mismatch")
                    checked += 1
        self.assertGreater(checked, 0)

    def test_power_mean_limits(self):
        """X(0)=arithmetic mean, X(α→∞)→max of R^j_out (Eq 41 limits), X(1)=E(t)."""
        # Well-separated values (max ratio ≥ 2) so the α→∞ limit converges fast.
        R_out = np.array([[0.5, 1.0, 2.0],
                          [1.0, 1.5, 3.0]])            # (T=2, N=3)
        X = _compute_power_mean_spectrum(R_out, np.array([0.0, 1.0, 80.0]))
        np.testing.assert_allclose(X[0], R_out.mean(axis=1), atol=1e-9)          # arith mean
        E_t = (R_out ** 2).sum(axis=1) / R_out.sum(axis=1)
        np.testing.assert_allclose(X[1], E_t, atol=1e-9)                          # α=1 → E(t)
        np.testing.assert_allclose(X[2], R_out.max(axis=1), atol=1e-6)           # α→∞ → max
        # monotone non-decreasing in α
        Xs = _compute_power_mean_spectrum(R_out, np.array([0.0, 1.0, 2.0, 5.0]))
        self.assertTrue(np.all(np.diff(Xs, axis=0) >= -1e-9))

    def test_row_stochastic_under_stress(self):
        """Mobility rows sum to 1 under hub amplification and commuting-fraction scaling."""
        for N in (4, 10, 15):
            city = generate_city(N, "lagos", seed=3)
            _, pops, dists, types, meta = city
            f, base = generate_mobility(N, 30, pops, dists, types, meta,
                                        seed=3, commuting_frac_scale=1.2,
                                        hub_attraction_power=8.0)
            self.assertTrue(np.allclose(f.sum(axis=-1), 1.0, atol=1e-9),
                            msg=f"rows not stochastic (N={N}, hub_power=8)")

    def test_type_R_single_location(self):
        """type_reproduction_number does not crash for N=1 (returns R_jj)."""
        self.assertAlmostEqual(type_reproduction_number(np.array([[0.7]]), 0), 0.7, places=10)

# ---------------------------------------------------------------------------
# The single manuscript table, and LaTeX validity of every generated table
# ---------------------------------------------------------------------------

from mobility_rt.combined_table import (
    _LOCATION_ROWS,
    PERIOD,
    compute_combined_table,
    write_combined_table,
)


def _tex_structure_errors(text, n_data_cols=None):
    """Structural problems in a generated LaTeX table: unbalanced braces or math,
    and rows whose cell count does not match the tabular preamble.

    The expected column count is read from the preamble itself when not given, so
    the check cannot be wrong about the table it is checking."""
    import re as _re
    errs = []
    if n_data_cols is None:
        m = _re.search(r"\\begin\{tabular\}\{([^}]*)\}", text)
        if not m:
            return ["no tabular preamble found"]
        n_data_cols = sum(1 for ch in m.group(1) if ch in "lcr")
    body = "\n".join(l for l in text.split("\n") if not l.lstrip().startswith("%"))
    if body.count("{") != body.count("}"):
        errs.append(f"unbalanced braces: {body.count('{')} open, "
                    f"{body.count('}')} close")
    if body.count("$") % 2:
        errs.append("odd number of $ delimiters")
    for env in ("table", "tabular"):
        if body.count("\\begin{" + env + "}") != body.count("\\end{" + env + "}"):
            errs.append(f"{env} not closed")
    for line in body.split("\n"):
        t = line.strip()
        if not t.endswith("\\\\") or "multicolumn" in t:
            continue
        if t.count("&") != n_data_cols - 1:
            errs.append(f"row has {t.count('&') + 1} cells, "
                        f"expected {n_data_cols}: {t[:48]}")
    return errs


class TestCombinedManuscriptTable(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.N, cls.T = 6, 260
        params = COVID_PARAMS
        cls.gt = discretise_gamma(params["gen_time_mean"], params["gen_time_sd"],
                                  params["max_gen_time"])
        cls.city = generate_city(cls.N, scenario="lagos", seed=6)
        f = generate_mobility(cls.N, cls.T, cls.city[1], cls.city[2],
                              cls.city[3], cls.city[4], seed=6)[0]
        init = np.zeros(cls.N); init[0] = 10.0
        LW = params["base_contact_rate"]
        cls.sim = simulate_epidemic_pde(
            cls.T, cls.N, cls.city[1], f, params["prob_transmission_peak"], cls.gt,
            params["max_gen_time"], params["R0_target"], init, LW, LW * 0.30,
            cls.gt, cls.gt, birth_rate=3e-5, death_rate=3e-5, seed=6)
        cls.scen = [("One", cls.sim, cls.city), ("Two", cls.sim, cls.city)]
        cls.table = compute_combined_table(cls.scen, cls.gt)

    def test_values_are_lifted_unchanged_from_the_sources(self):
        """Nothing may be recomputed: every cell must equal its source table."""
        net = compute_error_table(self.scen, self.gt)
        loc = compute_location_table(self.scen, self.gt)
        diag = compute_diagnostics_table(self.scen, self.gt)
        for lab in ("One", "Two"):
            for sch in SCHEMES:
                mre, bias = self.table["network"][sch.key][lab]
                self.assertAlmostEqual(mre, net[PERIOD]["mre"][sch.key][lab], places=12)
                self.assertAlmostEqual(bias, net[PERIOD]["bias"][sch.key][lab],
                                       places=12)
            for key, source, _, _, _ in _LOCATION_ROWS:
                want = (diag[lab][key] if source == "diag"
                        else float(np.median(loc[lab][key]["mre"])))
                self.assertAlmostEqual(self.table["location"][key][lab], want,
                                       places=12)
            self.assertEqual(self.table["n_days"][lab], net[PERIOD]["n_days"][lab])

    def test_refuses_when_the_halves_used_different_days(self):
        """The whole point of one table is one day set; a mismatch must be fatal."""
        import copy
        net = compute_error_table(self.scen, self.gt)
        loc = compute_location_table(self.scen, self.gt)
        diag = compute_diagnostics_table(self.scen, self.gt)
        bad = copy.deepcopy(diag)
        bad["One"]["n_days"] += 1
        with self.assertRaises(ValueError):
            compute_combined_table(self.scen, self.gt, net, loc, bad)

    def test_refuses_mismatched_scenarios(self):
        net = compute_error_table(self.scen, self.gt)
        loc = compute_location_table(self.scen, self.gt)
        diag = compute_diagnostics_table([("One", self.sim, self.city)], self.gt)
        with self.assertRaises(ValueError):
            compute_combined_table(self.scen, self.gt, net, loc, diag)
        with self.assertRaises(ValueError):
            compute_combined_table([], self.gt)
        with self.assertRaises(ValueError):
            compute_combined_table([("X", self.sim, self.city),
                                    ("X", self.sim, self.city)], self.gt)

    def test_files_written_and_well_formed(self):
        with _tempfile.TemporaryDirectory() as d:
            paths = write_combined_table(self.table, d, prefix="mt")
            self.assertEqual(len(paths), 2)
            rows = list(_csv.reader(open(_os.path.join(d, "mt.csv"))))
            self.assertEqual(len({len(r) for r in rows}), 1)
            self.assertEqual(len(rows),
                             1 + 2 * len(SCHEMES) + len(_LOCATION_ROWS) + 1)
            self.assertEqual(rows[-1][1], "Days used")
            for r in rows[1:-1]:
                for c in r[4:]:
                    float(c)
                self.assertNotIn("\\", r[1])
            tex = open(_os.path.join(d, "mt.tex")).read()
            errs = _tex_structure_errors(tex)
            self.assertEqual(errs, [], f"LaTeX problems: {errs}")

    def test_every_generated_tex_table_is_structurally_valid(self):
        """Guards all four writers at once, including the brace balance that a
        mixed f-string / plain-string literal silently breaks."""
        from mobility_rt import error_tables as _et, location_errors as _le
        with _tempfile.TemporaryDirectory() as d:
            n = len(self.table["labels"])
            net = compute_error_table(self.scen, self.gt)
            loc = compute_location_table(self.scen, self.gt)
            diag = compute_diagnostics_table(self.scen, self.gt)
            written = []
            written += _et.write_error_tables(net, d, prefix="e")
            written += _le.write_location_table(loc, d, prefix="l")
            written += _le.write_diagnostics_table(diag, d, prefix="g")
            written += write_combined_table(self.table, d, prefix="m")
            tex_files = [p for p in written if p.endswith(".tex")]
            self.assertEqual(len(tex_files), 5)   # growth, full, loc, diag, combined
            for path in tex_files:
                body = open(path).read()
                errs = _tex_structure_errors(body)
                self.assertEqual(errs, [], f"{_os.path.basename(path)}: {errs}")
                # and the preamble must actually match the scenario count
                import re
                pre = re.search(r"\\begin\{tabular\}\{([^}]*)\}", body).group(1)
                self.assertEqual(sum(1 for c in pre if c == "r"), n,
                                 f"{_os.path.basename(path)}: wrong number of "
                                 f"numeric columns for {n} scenarios")



# ---------------------------------------------------------------------------
# Framework <-> core parity
# ---------------------------------------------------------------------------
#
# mobility_rt.framework carries its own copy of most of the core math, because it is
# the self-contained general-purpose API (the one documented in the README's "Using
# your own data" section and used by the notebooks).  Duplication is the deliberate
# cost of that self-containment; silent DRIFT between the two copies is the thing
# that must never happen, and it is the failure mode behind the earlier GT and
# estimator bugs.
#
# _FW_DUPLICATES pins every framework function that re-implements core math to the
# core callable it must agree with, and every entry has a numerical parity test
# below.  _FW_ONLY lists the framework functions with no core counterpart.
# test_every_framework_function_is_accounted_for fails when a function appears in
# neither, so a NEW duplicate cannot be introduced without a deliberate decision
# about how it is guarded.
#
# ARGUMENT MAPPING.  The two copies are the same equations with different
# signatures, and the differences are the trap:
#
#   core                                               framework
#   ----------------------------------------------     -------------------------
#   compute_R_matrix(f, S, pops, prob_peak, p, lw, lb)  (f, S, pops, p, lw, lb)
#   compute_R_meeting(f, S, pops, prob_peak, p, lw, lb) R_meeting(f, S, pops, p, lw, lb)
#   compute_generation_times(..., prob_peak, p, max_days, lw, lb)  (..., p, lw, lb)
#   simulate_epidemic_pde(T, N, pops, f, prob_peak, p, max_days, R0, seed, lw, lb)
#                                                       simulate_epidemic(f, pops, p,
#                                                       lw, lb, R0, seed, T)
#
# The framework has no prob_peak argument: beta is folded into lw/lb instead, which
# is the same thing because beta enters R multiplicatively.  So every core call
# below passes prob_peak=1.0, and test_beta_may_be_folded_into_either_place pins
# that equivalence.  The framework also drops max_days (it uses len(p)) and the
# contact_mult/infect_mult heterogeneity multipliers (it has none, so core is
# called with the default None).

import mobility_rt.framework as _fw
from mobility_rt import estimators as _core_est
from mobility_rt import generation_time as _core_gt
from mobility_rt import kernel as _core_kernel
from mobility_rt import simulation as _core_sim
from mobility_rt import spectral as _core_spectral
from mobility_rt import transmissibility as _core_trans
from mobility_rt import type_reproduction as _core_type

_FW_DUPLICATES = {
    # framework attribute            core callable it must match
    "_kernel_base":                 _core_kernel._kernel_base,
    "compute_R_matrix":             _core_kernel.compute_R_matrix,
    "R_outward":                    _core_kernel.R_outward,
    "R_inward":                     _core_kernel.R_inward,
    "R_network":                    _core_kernel.R_system,
    "R_meeting":                    _core_kernel.compute_R_meeting,
    "_perron_real":                 _core_spectral._perron_real,
    "spectral_analysis":            _core_spectral.spectral_analysis,
    "sensitivity_elasticity":       _core_spectral.sensitivity_elasticity,
    "amplification_envelope":       _core_spectral.amplification_envelope,
    "type_R_number":                _core_type.type_reproduction_number,
    "type_R_all":                   _core_type.type_reproduction_numbers,
    "group_type_R":                 _core_type.type_reproduction_number_group,
    "compute_generation_times":     _core_gt.compute_generation_times,
    "estimate_R_independent":       _core_est.estimate_R_independent,
    "euler_lotka_r":                _core_trans.euler_lotka_r,
    "empirical_growth_rate":        _core_trans.empirical_growth_rate,
    "source_sink_analysis":         _core_trans.source_sink_analysis,
    "effective_populations_series": _core_sim.compute_effective_populations_series,
    "simulate_epidemic":            _core_sim.simulate_epidemic_pde,
}

_FW_ONLY = frozenset({
    "validate_inputs",                 # input checking, framework-specific
    "compute_R0_from_beta",            # beta -> R0 map, no core equivalent
    "estimate_beta_from_growth_rate",  # the one data-anchored entry point
    "scan_beta_to_R0",
    "parameter_sweep",
    "build_synthetic_network",
    "_scale_commuting",
    "run_all_tests",                   # the framework's own self-check harness
    # E_risk_averse has no core FUNCTION, but the core computes the same quantity
    # inline as E_t in estimators._compute_naive_suite; pinned by
    # test_E_risk_averse_matches_the_inline_core_formula.
    "E_risk_averse",
})


class TestFrameworkCoreConsistency(unittest.TestCase):
    """Guard: MobilityRtFramework and the core must compute IDENTICAL math.

    This institutionalises the single-source-of-truth.  If the framework's math ever
    drifts from the package core, one of these assertions fails.
    """

    @classmethod
    def setUpClass(cls):
        """One valid, fully-positive scenario shared by every parity test.

        Valid matters: the two copies differ in defensive clamping (the framework
        clamps the away-meeting term in R_meeting, the core does not), so parity is
        asserted on the documented input domain -- a row-stochastic f with positive
        populations and susceptibles -- where the clamps are inert and the two are
        genuinely the same function.
        """
        rng = np.random.default_rng(20260913)
        cls.N, cls.T = 5, 60
        W = rng.uniform(0.01, 1.0, (cls.N, cls.N))
        np.fill_diagonal(W, rng.uniform(4.0, 9.0, cls.N))   # home-dominant rows
        cls.f        = W / W.sum(axis=1, keepdims=True)
        cls.f_series = np.tile(cls.f, (cls.T, 1, 1))
        cls.pops     = rng.uniform(5e4, 9e5, cls.N)
        cls.S        = cls.pops * rng.uniform(0.3, 1.0, cls.N)
        cls.p        = discretise_gamma(5.5, 1.8, 20)
        cls.lw, cls.lb = 13.0, 3.9
        cls.inc      = rng.poisson(25, (cls.T, cls.N)).astype(float)
        cls.R        = compute_R_matrix(cls.f, cls.S, cls.pops, 1.0,
                                        cls.p, cls.lw, cls.lb)

    # ── helpers ───────────────────────────────────────────────────────────────

    def _same(self, a, b, what, atol=1e-10):
        np.testing.assert_allclose(a, b, rtol=0, atol=atol, equal_nan=True,
                                   err_msg=f"framework and core disagree on {what}")

    def _same_dicts(self, d_fw, d_core, what, expect_keys, atol=1e-10):
        """Compare every key the two dicts share, and assert the shared set is the
        expected one -- so a key silently disappearing from either side fails too."""
        shared = sorted(set(d_fw) & set(d_core))
        self.assertEqual(shared, sorted(expect_keys),
                         f"{what}: the set of keys shared by framework and core "
                         f"changed (got {shared})")
        for k in shared:
            if isinstance(d_fw[k], (bool, np.bool_)):
                self.assertEqual(bool(d_fw[k]), bool(d_core[k]), f"{what}[{k}]")
            else:
                self._same(d_fw[k], d_core[k], f"{what}[{k}]", atol=atol)

    # ── the registry itself ──────────────────────────────────────────────────

    def test_every_framework_function_is_accounted_for(self):
        """Every function the framework defines is either a pinned duplicate of core
        math or explicitly framework-only.  A new function in neither set fails here,
        which is the point: it forces the author to decide which it is."""
        import inspect
        defined = {n for n, o in vars(_fw).items()
                   if inspect.isfunction(o) and o.__module__ == "mobility_rt.framework"}
        unaccounted = defined - set(_FW_DUPLICATES) - _FW_ONLY
        self.assertFalse(unaccounted,
                         f"mobility_rt.framework defines {sorted(unaccounted)} with no "
                         f"parity test and no _FW_ONLY entry. If it duplicates core "
                         f"math, add it to _FW_DUPLICATES and write a parity test; "
                         f"if it is framework-only, add it to _FW_ONLY.")
        stale = (set(_FW_DUPLICATES) | _FW_ONLY) - defined
        self.assertFalse(stale, f"registry lists functions the framework no longer "
                                f"defines: {sorted(stale)}")

    def test_every_registered_duplicate_has_a_parity_test(self):
        """Registering a duplicate without testing it would be worse than useless."""
        tested = {n for n in dir(self) if n.startswith("test_")}
        # Each duplicate must be named in the source of at least one parity test.
        import inspect
        bodies = "\n".join(inspect.getsource(getattr(type(self), n)) for n in tested)
        missing = [name for name in _FW_DUPLICATES if f'"{name}"' not in bodies
                   and f"_fw.{name}" not in bodies]
        self.assertFalse(missing, f"registered duplicates with no parity test: {missing}")

    def test_discretise_gamma_is_shared(self):
        # Deduped rather than duplicated: the framework re-imports the core function,
        # so this is literally the same object and cannot drift at all.
        self.assertIs(_fw.discretise_gamma, discretise_gamma)

    # ── kernel ───────────────────────────────────────────────────────────────

    def test_kernel_base_matches_core(self):
        """The kernel is the one duplicate that was previously unguarded, and it is
        the one everything else is built on."""
        a = _fw._kernel_base(self.f, self.pops, self.lw, self.lb)
        b = _core_kernel._kernel_base(self.f, self.pops, self.lw, self.lb)
        self.assertEqual(len(a), 4)   # framework: base_K, within, between, N_eff
        self.assertEqual(len(b), 5)   # core adds inv_Neff
        for i, name in enumerate(["base_K", "K_within", "K_between", "N_eff"]):
            self._same(a[i], b[i], f"_kernel_base[{name}]", atol=0.0)

    def test_compute_R_matrix_matches_core(self):
        """Core takes prob_peak as its FOURTH POSITIONAL argument; the framework has
        no such argument.  They agree exactly when core is given prob_peak=1.0."""
        self._same(_fw.compute_R_matrix(self.f, self.S, self.pops, self.p, self.lw, self.lb),
                   compute_R_matrix(self.f, self.S, self.pops, 1.0, self.p, self.lw, self.lb),
                   "compute_R_matrix", atol=0.0)

    def test_beta_may_be_folded_into_either_place(self):
        """beta in core's prob_peak == beta in the framework's lw/lb.  This is what
        makes prob_peak=1.0 the correct translation, and it is the mistake the README
        warns about: supplying beta in BOTH places scales every R by beta squared."""
        beta = 0.037
        via_prob_peak = compute_R_matrix(self.f, self.S, self.pops, beta,
                                         self.p, self.lw, self.lb)
        via_contact_rates = _fw.compute_R_matrix(self.f, self.S, self.pops, self.p,
                                                 beta * self.lw, beta * self.lb)
        self._same(via_prob_peak, via_contact_rates, "beta folding", atol=1e-12)
        # and the double-application really is a beta^2 error, not a harmless one
        doubled = compute_R_matrix(self.f, self.S, self.pops, beta, self.p,
                                   beta * self.lw, beta * self.lb)
        self._same(doubled, beta * via_prob_peak, "beta applied twice", atol=1e-12)

    def test_R_summaries_match_core(self):
        R_fw = _fw.compute_R_matrix(self.f, self.S, self.pops, self.p, self.lw, self.lb)
        self._same(_fw.R_outward(R_fw), R_outward(self.R), "R_outward", atol=0.0)
        self._same(_fw.R_inward(R_fw),  R_inward(self.R),  "R_inward",  atol=0.0)
        self.assertEqual(_fw.R_network(R_fw), R_system(self.R))   # R_network == R_system

    def test_R_meeting_matches_core(self):
        self._same(_fw.R_meeting(self.f, self.S, self.pops, self.p, self.lw, self.lb),
                   _core_kernel.compute_R_meeting(self.f, self.S, self.pops, 1.0,
                                     self.p, self.lw, self.lb),
                   "R_meeting")

    # ── spectral ─────────────────────────────────────────────────────────────

    def test_perron_real_matches_core(self):
        rng = np.random.default_rng(101)
        for _ in range(20):
            M = rng.uniform(0.05, 2.0, (5, 5))
            _, evec = np.linalg.eig(M)
            for col in range(evec.shape[1]):
                self._same(_fw._perron_real(evec[:, col]),
                           _core_spectral._perron_real(evec[:, col]),
                           "_perron_real", atol=0.0)

    def test_spectral_analysis_matches_core(self):
        rng = np.random.default_rng(1)
        shared = ["damping_ratio", "eigenvalues", "lambda2", "left_eigvec",
                  "mixing_ratio", "rho", "right_eigvec"]
        for _ in range(10):
            R = rng.uniform(0.05, 2.0, (6, 6))
            self._same_dicts(_fw.spectral_analysis(R), spectral_analysis(R),
                             "spectral_analysis", shared, atol=1e-9)

    def test_framework_spectral_extras_match_their_core_functions(self):
        """The framework folds sigma, the condition number and the spectral gap into
        spectral_analysis; the core keeps them in reactivity() and
        sensitivity_elasticity().  Same numbers, different homes."""
        rng = np.random.default_rng(11)
        for _ in range(50):
            R  = rng.uniform(0.0, 2.0, (5, 5))
            sf = _fw.spectral_analysis(R)
            rc = reactivity(R)
            se = sensitivity_elasticity(R)
            sc = spectral_analysis(R)
            self._same(sf["sigma"], rc["sigma"], "sigma vs core reactivity", atol=1e-12)
            self._same(sf["amplification_ratio"], rc["amplification_ratio"],
                       "amplification_ratio", atol=1e-12)
            self.assertEqual(bool(sf["transient_possible"]),
                             bool(rc["transient_possible"]),
                             "the false-action-zone flag (sigma>1, rho<1) disagrees")
            self._same(sf["condition_number"], se["condition_number"],
                       "condition_number", atol=1e-9)
            self._same(sf["spectral_gap"], sc["rho"] - sc["lambda2"],
                       "spectral_gap", atol=1e-12)

    def test_sensitivity_elasticity_matches_core(self):
        rng = np.random.default_rng(12)
        for _ in range(10):
            R = rng.uniform(0.05, 2.0, (5, 5))
            self._same_dicts(_fw.sensitivity_elasticity(R), sensitivity_elasticity(R),
                             "sensitivity_elasticity",
                             ["condition_number", "elasticity", "rho", "sensitivity"],
                             atol=1e-9)

    def test_amplification_envelope_matches_core(self):
        rng = np.random.default_rng(13)
        for _ in range(5):
            R = rng.uniform(0.0, 1.2, (5, 5))
            self._same_dicts(_fw.amplification_envelope(R, 20),
                             amplification_envelope(R, 20),
                             "amplification_envelope", ["A", "n", "rho", "rho_n"],
                             atol=1e-9)

    # ── type reproduction numbers ────────────────────────────────────────────

    def test_type_R_matches_core(self):
        rng = np.random.default_rng(3)
        for _ in range(10):
            R = rng.uniform(0.0, 0.5, (4, 4))
            self._same(_fw.type_R_all(R), type_reproduction_numbers(R),
                       "type_R_all", atol=1e-9)
            for j in range(4):
                self._same(_fw.type_R_number(R, j), type_reproduction_number(R, j),
                           f"type_R_number[{j}]", atol=1e-9)
            for P in ([0], [0, 2], [1, 2, 3]):
                self._same(_fw.group_type_R(R, P),
                           _core_type.type_reproduction_number_group(R, P),
                           f"group_type_R{P}", atol=1e-9)

    def test_type_R_agrees_where_it_is_undefined_too(self):
        """T_j is infinite when the background subsystem is self-sustaining.  Both
        copies must return the SAME non-finite value, not merely both be non-finite."""
        R = np.array([[0.2, 0.1, 0.1],       # location 1 alone sustains the epidemic
                      [0.1, 2.6, 0.1],
                      [0.1, 0.1, 0.2]])
        a, b = _fw.type_R_all(R), type_reproduction_numbers(R)
        np.testing.assert_array_equal(np.isfinite(a), np.isfinite(b))
        np.testing.assert_array_equal(np.isnan(a), np.isnan(b))
        self._same(a, b, "type_R_all on a self-sustaining background")

    # ── generation times ─────────────────────────────────────────────────────

    def test_generation_times_match_core(self):
        """Core takes prob_peak and max_days explicitly; the framework uses 1.0 and
        len(p).  Every GT the two both produce must agree."""
        g_fw = _fw.compute_generation_times(self.f, self.S, self.pops,
                                            self.p, self.lw, self.lb)
        g_c  = compute_generation_times(self.f, self.S, self.pops, 1.0, self.p,
                                        len(self.p), self.lw, self.lb)
        self._same_dicts(g_fw, g_c, "compute_generation_times",
                         ["R_matrix", "R_meeting", "g_inward", "g_meeting",
                          "g_network", "g_outward", "g_pairwise"])

    # ── estimators, growth, transmissibility ────────────────────────────────

    def test_estimate_R_independent_matches_core(self):
        rng = np.random.default_rng(2)
        inc = rng.poisson(20, (60, 4)).astype(float)
        pmf = discretise_gamma(5.5, 1.8, 25)
        self._same(_fw.estimate_R_independent(inc, pmf, 7),
                   estimate_R_independent(inc, pmf, 7),
                   "estimate_R_independent", atol=1e-9)

    def test_euler_lotka_matches_core(self):
        pmf = discretise_gamma(5.5, 1.8, 25)
        for rho in (0.7, 1.0, 1.5, 2.5):
            self._same(_fw.euler_lotka_r(rho, pmf), euler_lotka_r(rho, pmf),
                       f"euler_lotka_r({rho})", atol=1e-9)

    def test_empirical_growth_rate_matches_core(self):
        self._same(_fw.empirical_growth_rate(self.inc, 3),
                   _core_trans.empirical_growth_rate(self.inc, 3),
                   "empirical_growth_rate", atol=1e-12)

    def test_source_sink_matches_core(self):
        self._same_dicts(_fw.source_sink_analysis(self.R),
                         _core_trans.source_sink_analysis(self.R),
                         "source_sink_analysis",
                         ["R_inward", "R_outward", "net_export", "sinks", "sources"])

    def test_effective_populations_series_matches_core(self):
        """Framework infers max_days from len(p); core takes it explicitly."""
        S_ser = np.tile(self.S, (self.T, 1))
        a = _fw.effective_populations_series(self.f_series, S_ser, self.inc, self.p)
        b = _core_sim.compute_effective_populations_series(
            self.f_series, S_ser, self.inc, self.p, len(self.p))
        for arr_fw, arr_core, name in zip(a, b, ["S_eff", "I_eff"]):
            self._same(arr_fw, arr_core, f"effective_populations_series[{name}]",
                       atol=1e-9)

    def test_E_risk_averse_matches_the_inline_core_formula(self):
        """E(t) = sum_j (R^j_out)^2 / sum_k R^k_out (Eq 42).  The core has no function
        for it -- estimators._compute_naive_suite computes it inline as E_t -- so this
        pins the framework against the formula that suite uses."""
        Ro = R_outward(self.R)
        self._same(_fw.E_risk_averse(self.R),
                   float(np.sum(Ro ** 2) / (np.sum(Ro) + 1e-300)),
                   "E_risk_averse", atol=1e-9)

    # ── the simulators ───────────────────────────────────────────────────────

    def test_simulators_are_bit_identical(self):
        """The largest duplicate and the one that matters most: the manuscript
        figures come out of simulate_epidemic_pde, while a user of the documented
        MobilityRtFramework API gets simulate_epidemic.  If these two ever diverge,
        the published results and the shipped tool stop being the same model.

        atol=0: not 'close', identical.
        """
        seed0 = np.zeros(self.N); seed0[0] = 100.0
        R0_target, max_days = 1.8, len(self.p)
        a = _fw.simulate_epidemic(self.f_series, self.pops, self.p,
                                  self.lw, self.lb, R0_target, seed0, self.T,
                                  verbose=False)
        b = simulate_epidemic_pde(self.T, self.N, self.pops, self.f_series,
                                  1.0, self.p, max_days, R0_target, seed0,
                                  self.lw, self.lb)
        for key_fw, key_core in [("incidence", "incidence"),
                                 ("susceptibles", "susceptibles"),
                                 ("incidence_matrix", "incidence_matrix"),
                                 ("R_pairwise", "R_matrices"),
                                 ("R_meeting", "R_meeting_series"),
                                 ("E_pde_state", "E_pde_state")]:
            self._same(a[key_fw], b[key_core],
                       f"simulator output {key_fw} vs {key_core}", atol=0.0)
        # and the calibration landed on the same contact rates
        self._same(a["lw"], b["lambda_within_scaled"],  "calibrated lw", atol=0.0)
        self._same(a["lb"], b["lambda_between_scaled"], "calibrated lb", atol=0.0)



# ---------------------------------------------------------------------------
# Spatially heterogeneous contact rate and infectiousness
# ---------------------------------------------------------------------------

from mobility_rt.kernel import _as_multiplier, compute_R_meeting
from mobility_rt.heterogeneity import (
    CONTACT_MULT_BY_TYPE,
    contact_multipliers,
    infectiousness_multipliers,
)


class TestSpatialHeterogeneity(unittest.TestCase):
    """contact_mult c[l] and infect_mult m[k] in the kernel and the simulator."""

    @classmethod
    def setUpClass(cls):
        rng = np.random.default_rng(17)
        cls.N = 6
        raw = rng.uniform(0.2, 1.0, (cls.N, cls.N))
        cls.f_t = raw / raw.sum(axis=1, keepdims=True)
        cls.pops = rng.uniform(1e4, 1e5, cls.N)
        cls.S = cls.pops * rng.uniform(0.4, 1.0, cls.N)
        cls.lw, cls.lb = 15.0, 5.0
        cls.p = discretise_gamma(5.5, 1.8, 20)
        cls.prob_peak = 0.035
        cls.c = np.exp(rng.normal(0, 0.4, cls.N))
        cls.m = np.exp(rng.normal(0, 0.5, cls.N))

    # -- validation ------------------------------------------------------
    def test_as_multiplier_accepts_none_scalar_vector(self):
        np.testing.assert_array_equal(_as_multiplier(None, 4, "x"), np.ones(4))
        np.testing.assert_array_equal(_as_multiplier(2.5, 4, "x"), np.full(4, 2.5))
        np.testing.assert_array_equal(_as_multiplier([1, 2, 3], 3, "x"),
                                      np.array([1.0, 2.0, 3.0]))

    def test_as_multiplier_rejects_bad_input(self):
        for bad in (np.ones(3), [-1.0, 1.0], [np.nan, 1.0], [np.inf, 1.0]):
            with self.assertRaises(ValueError):
                _as_multiplier(bad, 2, "x")

    # -- backward compatibility -----------------------------------------
    def test_none_multipliers_reproduce_homogeneous_exactly(self):
        """Passing None must be bit-identical to the pre-heterogeneity kernel."""
        a = _kernel_base(self.f_t, self.pops, self.lw, self.lb)
        b = _kernel_base(self.f_t, self.pops, self.lw, self.lb, contact_mult=None)
        for x, y in zip(a, b):
            self.assertTrue(np.array_equal(x, y), "None changed _kernel_base output")
        R1 = compute_R_matrix(self.f_t, self.S, self.pops, self.prob_peak,
                              self.p, self.lw, self.lb)
        R2 = compute_R_matrix(self.f_t, self.S, self.pops, self.prob_peak,
                              self.p, self.lw, self.lb, None, None)
        self.assertTrue(np.array_equal(R1, R2))
        M1 = compute_R_meeting(self.f_t, self.S, self.pops, self.prob_peak,
                               self.p, self.lw, self.lb)
        M2 = compute_R_meeting(self.f_t, self.S, self.pops, self.prob_peak,
                               self.p, self.lw, self.lb, None, None)
        self.assertTrue(np.array_equal(M1, M2))

    def test_unit_multipliers_match_homogeneous(self):
        """An explicit vector of ones equals the homogeneous result numerically."""
        one = np.ones(self.N)
        R1 = compute_R_matrix(self.f_t, self.S, self.pops, self.prob_peak,
                              self.p, self.lw, self.lb)
        R2 = compute_R_matrix(self.f_t, self.S, self.pops, self.prob_peak,
                              self.p, self.lw, self.lb, one, one)
        np.testing.assert_allclose(R1, R2, rtol=1e-13, atol=0)
        M1 = compute_R_meeting(self.f_t, self.S, self.pops, self.prob_peak,
                               self.p, self.lw, self.lb)
        M2 = compute_R_meeting(self.f_t, self.S, self.pops, self.prob_peak,
                               self.p, self.lw, self.lb, one, one)
        np.testing.assert_allclose(M1, M2, rtol=1e-13, atol=0)

    # -- the maths -------------------------------------------------------
    def test_kernel_base_heterogeneous_vs_naive(self):
        """Vectorised c[l] kernel matches the naive triple loop over meeting places."""
        N, f, c = self.N, self.f_t, self.c
        base_K, bKw, bKb, N_eff, _ = _kernel_base(f, self.pops, self.lw, self.lb, c)
        N_eff_naive = np.array([sum(f[j, l] * self.pops[j] for j in range(N))
                                for l in range(N)])
        inv = 1.0 / N_eff_naive
        within_naive = np.array([[f[j, k] * f[k, k] * c[k] * inv[k]
                                  for j in range(N)] for k in range(N)])
        between_naive = np.array([[sum(c[l] * f[j, l] * f[k, l] * inv[l]
                                       for l in range(N) if l != k)
                                   for j in range(N)] for k in range(N)])
        np.testing.assert_allclose(N_eff, N_eff_naive, rtol=1e-12)
        np.testing.assert_allclose(bKw, self.lw * within_naive, rtol=1e-11)
        np.testing.assert_allclose(bKb, self.lb * between_naive, rtol=1e-11)
        np.testing.assert_allclose(base_K, self.lw * within_naive
                                   + self.lb * between_naive, rtol=1e-11)

    def test_contact_mult_enters_at_meeting_location(self):
        """Scaling c at ONE location must change every pair that can meet there."""
        c = np.ones(self.N)
        c[2] = 3.0
        base0, _, _, _, _ = _kernel_base(self.f_t, self.pops, self.lw, self.lb)
        base1, _, _, _, _ = _kernel_base(self.f_t, self.pops, self.lw, self.lb, c)
        # row 2 (infector resident at 2) is not the only one affected: anyone who
        # travels to 2 is too, so the change must NOT be confined to row/column 2.
        changed = ~np.isclose(base0, base1)
        self.assertTrue(changed.any())
        off = changed.copy()
        off[2, :] = False
        off[:, 2] = False
        self.assertTrue(off.any(),
                        "contact_mult behaves like a row/column scaling, so it is "
                        "being applied at the infector's home rather than at the "
                        "meeting location")

    def test_infect_mult_is_exactly_a_row_scaling(self):
        """m[k] scales row k of R and nothing else."""
        R0 = compute_R_matrix(self.f_t, self.S, self.pops, self.prob_peak,
                              self.p, self.lw, self.lb)
        Rm = compute_R_matrix(self.f_t, self.S, self.pops, self.prob_peak,
                              self.p, self.lw, self.lb, None, self.m)
        np.testing.assert_allclose(Rm, self.m[:, None] * R0, rtol=1e-12)

    def test_contact_and_infect_multipliers_commute(self):
        """Applying c and m together == applying c then row-scaling by m."""
        Rc = compute_R_matrix(self.f_t, self.S, self.pops, self.prob_peak,
                              self.p, self.lw, self.lb, self.c, None)
        Rcm = compute_R_matrix(self.f_t, self.S, self.pops, self.prob_peak,
                               self.p, self.lw, self.lb, self.c, self.m)
        np.testing.assert_allclose(Rcm, self.m[:, None] * Rc, rtol=1e-12)

    def test_R_meeting_mbar_is_presence_weighted(self):
        """m̄[l] reduces to m[l] when nobody travels (f = identity)."""
        eye = np.eye(self.N)
        M = compute_R_meeting(eye, self.S, self.pops, self.prob_peak, self.p,
                              self.lw, self.lb, None, self.m)
        M1 = compute_R_meeting(eye, self.S, self.pops, self.prob_peak, self.p,
                               self.lw, self.lb, None, None)
        np.testing.assert_allclose(M, self.m * M1, rtol=1e-12)


class TestHeterogeneousSimulation(unittest.TestCase):
    """The forward simulator must stay consistent with its own NGM under c and m."""

    @classmethod
    def setUpClass(cls):
        cls.N, cls.T = 6, 60
        cls.params = COVID_PARAMS
        cls.max_days = cls.params["max_gen_time"]
        cls.p = discretise_gamma(cls.params["gen_time_mean"],
                                 cls.params["gen_time_sd"], cls.max_days)
        cls.city = generate_city(cls.N, scenario="lagos", seed=3)
        cls.pops = cls.city[1]
        cls.f = generate_mobility(cls.N, cls.T, cls.pops, cls.city[2],
                                  cls.city[3], cls.city[4], seed=3)[0]
        cls.init = np.zeros(cls.N); cls.init[0] = 10.0
        cls.LW = cls.params["base_contact_rate"]
        cls.LB = cls.LW * 0.30
        cls.c = contact_multipliers(cls.city[3])
        cls.m = infectiousness_multipliers(cls.N, seed=3)

    def _sim(self, **kw):
        return simulate_epidemic_pde(
            self.T, self.N, self.pops, self.f, self.params["prob_transmission_peak"],
            self.p, self.max_days, self.params["R0_target"], self.init,
            self.LW, self.LB, self.p, self.p, birth_rate=3e-5, death_rate=3e-5,
            stochastic=False, susceptible_depletion=True, seed=3, **kw)

    def test_calibration_holds_under_heterogeneity(self):
        sim = self._sim(contact_mult=self.c, infect_mult=self.m)
        self.assertAlmostEqual(R_system(sim["R_matrices"][0]),
                               self.params["R0_target"], places=8)

    def test_scalar_multipliers_are_absorbed_by_calibration(self):
        """Only RELATIVE variation matters: a common factor is rescaled away."""
        base = self._sim()
        for kw in (dict(contact_mult=4.0), dict(infect_mult=0.25),
                   dict(contact_mult=4.0, infect_mult=0.25)):
            s = self._sim(**kw)
            np.testing.assert_allclose(s["incidence"], base["incidence"],
                                       rtol=1e-10, atol=1e-8,
                                       err_msg=f"scalar multiplier changed the run: {kw}")

    def test_heterogeneity_actually_changes_the_run(self):
        """Guard against the multipliers being silently ignored."""
        base = self._sim()
        het = self._sim(contact_mult=self.c, infect_mult=self.m)
        self.assertGreater(np.abs(het["incidence"] - base["incidence"]).max(),
                           1e-3 * base["incidence"].max())

    def test_incidence_matches_its_own_R_matrix(self):
        """E_j(t,0) = Σ_k R_kj(t) Λ_k(t) / Σ_{a≥1} p(a), exactly, with and without
        heterogeneity — the identity that makes R(t) the NGM of these dynamics."""
        for kw in ({}, dict(contact_mult=self.c),
                   dict(infect_mult=self.m),
                   dict(contact_mult=self.c, infect_mult=self.m)):
            sim = self._sim(**kw)
            E = sim["E_pde_state"]
            sum_w = self.p[1:].sum()
            for t in range(1, 25):     # early days: no susceptible clamping
                lam = E[t, :, 1:] @ self.p[1:]
                pred = sim["R_matrices"][t].T @ lam / sum_w
                np.testing.assert_allclose(
                    sim["incidence"][t], pred, rtol=1e-9, atol=1e-9,
                    err_msg=f"incidence != R^T Λ / Σp at t={t} for {kw}")

    def test_heterogeneity_recorded_in_output(self):
        sim = self._sim(contact_mult=self.c, infect_mult=self.m)
        np.testing.assert_array_equal(sim["contact_mult"], self.c)
        np.testing.assert_array_equal(sim["infect_mult"], self.m)
        base = self._sim()
        self.assertIsNone(base["contact_mult"])
        self.assertIsNone(base["infect_mult"])


class TestHeterogeneityParameters(unittest.TestCase):
    """The multiplier constructors in mobility_rt.heterogeneity."""

    def test_contact_multipliers_geometric_mean_one(self):
        for scen in ("lagos", "zambia"):
            types = generate_city(10, scenario=scen, seed=1)[3]
            c = contact_multipliers(types)
            self.assertAlmostEqual(float(np.exp(np.mean(np.log(c)))), 1.0, places=12)
            self.assertTrue(np.all(c > 0))
            self.assertEqual(len(c), 10)

    def test_contact_multipliers_preserve_type_ordering(self):
        types = ["core", "dense", "suburban", "peripheral"]
        c = contact_multipliers(types)
        self.assertTrue(np.all(np.diff(c) < 0),
                        "denser settlement types must have higher contact rates")

    def test_contact_multipliers_reject_unknown_type(self):
        with self.assertRaises(KeyError):
            contact_multipliers(["core", "not-a-real-type"])

    def test_every_generated_node_type_has_a_multiplier(self):
        """generate_city must never emit a type the table does not cover."""
        for scen in ("lagos", "zambia"):
            for n in (5, 8, 10, 15):
                for t in generate_city(n, scenario=scen, seed=n)[3]:
                    self.assertIn(t, CONTACT_MULT_BY_TYPE,
                                  f"{scen} N={n} emits unmapped node type {t!r}")

    def test_infectiousness_multipliers_geometric_mean_one(self):
        m = infectiousness_multipliers(12, seed=9)
        self.assertAlmostEqual(float(np.exp(np.mean(np.log(m)))), 1.0, places=12)
        self.assertGreater(m.std() / m.mean(), 0.05)

    def test_multiplier_constructors_reject_empty(self):
        with self.assertRaises(ValueError):
            contact_multipliers([])
        with self.assertRaises(ValueError):
            infectiousness_multipliers(0)

    def test_single_location_is_not_heterogeneous(self):
        np.testing.assert_array_equal(infectiousness_multipliers(1, seed=1),
                                      np.ones(1))
        np.testing.assert_allclose(contact_multipliers(["core"]), np.ones(1))

    def test_infectiousness_multipliers_deterministic_and_zero_sd(self):
        np.testing.assert_array_equal(infectiousness_multipliers(7, seed=4),
                                      infectiousness_multipliers(7, seed=4))
        self.assertFalse(np.array_equal(infectiousness_multipliers(7, seed=4),
                                        infectiousness_multipliers(7, seed=5)))
        np.testing.assert_array_equal(infectiousness_multipliers(7, log_sd=0.0),
                                      np.ones(7))
        with self.assertRaises(ValueError):
            infectiousness_multipliers(7, log_sd=-0.1)


# ---------------------------------------------------------------------------
# Per-location discrepancy of the independent estimate
# ---------------------------------------------------------------------------

from mobility_rt.location_errors import (
    COMPARATORS,
    _spearman,
    compute_diagnostics_table,
    compute_location_table,
    location_diagnostics,
    location_discrepancies,
    write_diagnostics_table,
    write_location_table,
)


class TestLocationDiscrepancy(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.N, cls.T = 6, 260
        params = COVID_PARAMS
        cls.gt = discretise_gamma(params["gen_time_mean"], params["gen_time_sd"],
                                  params["max_gen_time"])
        cls.city = generate_city(cls.N, scenario="lagos", seed=6)
        f = generate_mobility(cls.N, cls.T, cls.city[1], cls.city[2],
                              cls.city[3], cls.city[4], seed=6)[0]
        init = np.zeros(cls.N); init[0] = 10.0
        LW = params["base_contact_rate"]
        cls.sim = simulate_epidemic_pde(
            cls.T, cls.N, cls.city[1], f, params["prob_transmission_peak"], cls.gt,
            params["max_gen_time"], params["R0_target"], init, LW, LW * 0.30,
            cls.gt, cls.gt, birth_rate=3e-5, death_rate=3e-5, seed=6)
        cls.d = location_discrepancies(cls.sim, cls.gt)

    def test_no_unused_parameters(self):
        import inspect
        sig = inspect.signature(location_discrepancies)
        self.assertEqual(list(sig.parameters), ["sim", "gen_time_pmf"])
        src = inspect.getsource(location_discrepancies)
        for name in sig.parameters:
            self.assertGreater(src.count(name), 1, f"{name} is never read")

    def test_shapes_and_positivity(self):
        for key, _, _ in COMPARATORS:
            self.assertEqual(len(self.d[key]["mre"]), self.N)
            self.assertEqual(len(self.d[key]["bias"]), self.N)
            self.assertTrue(np.all(self.d[key]["mre"] >= 0.0))
            self.assertTrue(np.all(np.isfinite(self.d[key]["mre"])))
        self.assertGreater(self.d["n_days"], 0)

    def test_mre_dominates_bias_per_location(self):
        for key, _, _ in COMPARATORS:
            self.assertTrue(np.all(self.d[key]["mre"] + 1e-9
                                   >= np.abs(self.d[key]["bias"])),
                            f"{key}: |bias| exceeds mean absolute error")

    def test_recomputed_by_hand(self):
        """Independent re-derivation of one comparator, including the matched
        trailing-mean reference."""
        from mobility_rt.kernel import R_outward, R_system
        inc = self.sim["incidence"]
        Rm = self.sim["R_matrices"]
        T, N = inc.shape
        rho = np.array([R_system(Rm[t]) for t in range(T)])
        Ro = np.array([R_outward(Rm[t]) for t in range(T)])
        Rh = estimate_R_independent(inc, self.gt, window=ESTIMATOR_WINDOW)
        ref_net = trailing_mean(rho, ESTIMATOR_WINDOW)
        ref_own = np.column_stack([trailing_mean(Ro[:, j], ESTIMATOR_WINDOW)
                                   for j in range(N)])
        days = (rho > 1.0) & ~np.isnan(Rh).any(axis=1)
        days &= np.isfinite(ref_net) & (ref_net > RHO_FLOOR)
        days &= (np.isfinite(ref_own) & (ref_own > RHO_FLOOR)).all(axis=1)
        self.assertEqual(int(days.sum()), self.d["n_days"])
        rel = (Rh[days] - ref_own[days]) / ref_own[days]
        np.testing.assert_allclose(self.d["own"]["mre"],
                                   np.mean(np.abs(rel), axis=0) * 100, rtol=1e-12)
        rel_n = (Rh[days] - ref_net[days, None]) / ref_net[days, None]
        np.testing.assert_allclose(self.d["net"]["mre"],
                                   np.mean(np.abs(rel_n), axis=0) * 100, rtol=1e-12)

    def test_estimate_is_closer_to_the_network_than_to_its_own_R_out(self):
        """The finding the table exists to report, as a median statement."""
        self.assertLess(np.median(self.d["net"]["mre"]),
                        np.median(self.d["own"]["mre"]))

    def test_duplicate_labels_and_empty_rejected(self):
        with self.assertRaises(ValueError):
            compute_location_table([], self.gt)
        with self.assertRaises(ValueError):
            compute_location_table([("X", self.sim, self.city),
                                    ("X", self.sim, self.city)], self.gt)

    def test_files_written_and_parseable(self):
        tbl = compute_location_table([("One", self.sim, self.city)], self.gt)
        with _tempfile.TemporaryDirectory() as d:
            paths = write_location_table(tbl, d, prefix="loc")
            self.assertEqual(len(paths), 2)
            rows = list(_csv.reader(open(_os.path.join(d, "loc.csv"))))
            self.assertEqual(len({len(r) for r in rows}), 1)
            self.assertEqual(rows[-1][1], "days used")
            for r in rows[1:-1]:
                float(r[3])                       # every value numeric
            tex = open(_os.path.join(d, "loc.tex")).read()
            self.assertIn("booktabs", tex)
            self.assertEqual(tex.count("\\begin{table}"), 1)

    # -- diagnostics ------------------------------------------------------
    def test_spearman_matches_scipy_and_handles_edges(self):
        from scipy.stats import spearmanr
        rng = np.random.default_rng(3)
        for _ in range(8):
            a, b = rng.normal(size=9), rng.normal(size=9)
            self.assertAlmostEqual(_spearman(a, b), spearmanr(a, b).statistic,
                                   places=12)
        # ties get average ranks, matching scipy
        t, u = np.array([1., 2., 2., 3.]), np.array([4., 3., 3., 1.])
        self.assertAlmostEqual(_spearman(t, u), spearmanr(t, u).statistic, places=12)
        # a flat input has an UNDEFINED correlation, not zero
        self.assertTrue(np.isnan(_spearman(np.ones(5), np.arange(5.))))
        with self.assertRaises(ValueError):
            _spearman(np.ones(3), np.ones(4))

    def test_diagnostics_ranges_are_sane(self):
        d = location_diagnostics(self.sim, self.gt)
        self.assertGreaterEqual(d["rank_q1"], -1.0)
        self.assertLessEqual(d["rank_q3"], 1.0)
        self.assertLessEqual(d["rank_q1"], d["rank_median"])
        self.assertLessEqual(d["rank_median"], d["rank_q3"])
        for k in ("agree_growth", "agree_own"):
            self.assertGreaterEqual(d[k], 0.0)
            self.assertLessEqual(d[k], 100.0)
        for k in ("cv_est", "cv_true", "net_median"):
            self.assertGreater(d[k], 0.0)
        self.assertGreater(d["n_days"], 0)

    def test_diagnostics_recomputed_by_hand(self):
        from mobility_rt.kernel import R_outward, R_system
        inc = self.sim["incidence"]; Rm = self.sim["R_matrices"]
        T, N = inc.shape
        rho = np.array([R_system(Rm[t]) for t in range(T)])
        Ro = np.array([R_outward(Rm[t]) for t in range(T)])
        Rh = estimate_R_independent(inc, self.gt, window=ESTIMATOR_WINDOW)
        rn = trailing_mean(rho, ESTIMATOR_WINDOW)
        ro = np.column_stack([trailing_mean(Ro[:, j], ESTIMATOR_WINDOW)
                              for j in range(N)])
        days = (rho > 1.0) & ~np.isnan(Rh).any(axis=1)
        days &= np.isfinite(rn) & (rn > RHO_FLOOR)
        days &= (np.isfinite(ro) & (ro > RHO_FLOOR)).all(axis=1)
        d = location_diagnostics(self.sim, self.gt)
        self.assertEqual(int(days.sum()), d["n_days"])
        rs = np.array([_spearman(Rh[t], ro[t]) for t in np.where(days)[0]])
        rs = rs[np.isfinite(rs)]
        self.assertAlmostEqual(d["rank_median"], float(np.median(rs)), places=12)
        self.assertAlmostEqual(
            d["cv_true"],
            float(np.median(ro[days].std(axis=1) / ro[days].mean(axis=1))) * 100,
            places=10)
        self.assertAlmostEqual(
            d["agree_own"],
            float(np.mean((Rh[days] > 1.0) == (ro[days] > 1.0)) * 100), places=10)

    def test_both_entry_points_share_one_day_mask(self):
        """The mask must live in exactly one place, or the two tables could end up
        scored on different days without anything failing."""
        import inspect
        from mobility_rt import location_errors as _le
        for fn in (_le.location_discrepancies, _le.location_diagnostics):
            src = inspect.getsource(fn)
            self.assertIn("_prepare(sim, gen_time_pmf)", src,
                          f"{fn.__name__} does not use the shared preparation")
            self.assertNotIn("days &=", src,
                             f"{fn.__name__} rebuilds the day mask")
        self.assertEqual(location_diagnostics(self.sim, self.gt)["n_days"],
                         location_discrepancies(self.sim, self.gt)["n_days"])

    def test_net_median_matches_the_discrepancy_module(self):
        self.assertAlmostEqual(
            location_diagnostics(self.sim, self.gt)["net_median"],
            float(np.median(location_discrepancies(self.sim, self.gt)["net"]["mre"])),
            places=12)

    def test_growth_agreement_uses_two_complete_windows(self):
        """The first two windows cannot be compared week on week, so they must be
        excluded; including them would compare against an incomplete baseline."""
        import inspect
        src = inspect.getsource(location_diagnostics)
        self.assertIn("2 * ESTIMATOR_WINDOW", src)

    def test_diagnostics_files_written(self):
        tbl = compute_diagnostics_table([("One", self.sim, self.city)], self.gt)
        with _tempfile.TemporaryDirectory() as d:
            paths = write_diagnostics_table(tbl, d, prefix="dg")
            self.assertEqual(len(paths), 2)
            rows = list(_csv.reader(open(_os.path.join(d, "dg.csv"))))
            self.assertEqual(len({len(r) for r in rows}), 1)
            self.assertEqual(rows[-1][0], "Days used")
            for r in rows[1:-1]:
                float(r[1])
            tex = open(_os.path.join(d, "dg.tex")).read()
            self.assertIn("booktabs", tex)
            self.assertEqual(tex.count("\\begin{table}"), 1)

    def test_diagnostics_figure_writes_and_validates_keys(self):
        from mobility_rt.plotting.figures_main import plot_location_diagnostics
        tbl = compute_diagnostics_table([("One", self.sim, self.city)], self.gt)
        with _tempfile.TemporaryDirectory() as d:
            out = plot_location_diagnostics(tbl, save_prefix=_os.path.join(d, "f"))
            self.assertTrue(_os.path.exists(out))
            self.assertGreater(_os.path.getsize(out), 0)
        with self.assertRaises(ValueError):
            plot_location_diagnostics({}, save_prefix="unused")
        with self.assertRaises(KeyError):
            plot_location_diagnostics({"X": {"rank_median": 0.0}},
                                      save_prefix="unused")

    def test_figure_writes_and_validates_comparators(self):
        from mobility_rt.plotting.figures_main import plot_location_discrepancy
        tbl = compute_location_table([("One", self.sim, self.city)], self.gt)
        with _tempfile.TemporaryDirectory() as d:
            out = plot_location_discrepancy(tbl, save_prefix=_os.path.join(d, "f"))
            self.assertTrue(_os.path.exists(out))
            self.assertGreater(_os.path.getsize(out), 0)
        with self.assertRaises(ValueError):
            plot_location_discrepancy({}, save_prefix="unused")


# ---------------------------------------------------------------------------
# Estimator mean-relative-error tables
# ---------------------------------------------------------------------------

import csv as _csv
import os as _os
import tempfile as _tempfile

from mobility_rt.error_tables import (
    ESTIMATOR_WINDOW,
    PERIODS,
    RHO_FLOOR,
    SCHEMES,
    TABLE_PERIODS,
    compute_error_table,
    references,
    threshold_crossing_day,
    trailing_mean,
    write_error_tables,
)
from mobility_rt.estimators import _compute_naive_suite


class TestErrorTables(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # T must run past the threshold crossing, or the "decline" period is empty
        # and the period-split metrics are never exercised.
        cls.N, cls.T = 6, 260
        params = COVID_PARAMS
        cls.gt = discretise_gamma(params["gen_time_mean"], params["gen_time_sd"],
                                  params["max_gen_time"])
        cls.city = generate_city(cls.N, scenario="lagos", seed=6)
        f = generate_mobility(cls.N, cls.T, cls.city[1], cls.city[2],
                              cls.city[3], cls.city[4], seed=6)[0]
        init = np.zeros(cls.N); init[0] = 10.0
        LW = params["base_contact_rate"]
        cls.sim = simulate_epidemic_pde(
            cls.T, cls.N, cls.city[1], f, params["prob_transmission_peak"], cls.gt,
            params["max_gen_time"], params["R0_target"], init, LW, LW * 0.30,
            cls.gt, cls.gt, birth_rate=3e-5, death_rate=3e-5, seed=6)
        cls.scen = [("One", cls.sim, cls.city), ("Two", cls.sim, cls.city)]
        cls.table = compute_error_table(cls.scen, cls.gt)

    def test_every_period_is_non_empty_in_the_fixture(self):
        """Guard: a fixture with no decline days would silently skip half the tests."""
        for period in PERIODS:
            self.assertGreater(self.table[period]["n_days"]["One"], 0,
                               f"fixture has no days in the {period!r} period")

    def test_table_periods_are_a_subset_of_periods(self):
        self.assertTrue(set(TABLE_PERIODS) <= set(PERIODS))
        self.assertIn("growth", TABLE_PERIODS)
        self.assertIn("full", TABLE_PERIODS)

    def test_growth_and_decline_partition_full(self):
        for lab in ("One", "Two"):
            self.assertEqual(self.table["growth"]["n_days"][lab]
                             + self.table["decline"]["n_days"][lab],
                             self.table["full"]["n_days"][lab])

    def test_scheme_keys_exist_in_the_suite(self):
        suite = _compute_naive_suite(self.sim, self.city, self.gt)
        for sch in SCHEMES:
            self.assertIn(sch.key, suite)
            self.assertNotIn("\\", sch.plain,
                             f"CSV label for {sch.key} contains LaTeX")

    def test_identical_scenarios_give_identical_columns(self):
        for period in PERIODS:
            d = self.table[period]
            self.assertEqual(d["n_days"]["One"], d["n_days"]["Two"])
            for sch in SCHEMES:
                self.assertAlmostEqual(d["mre"][sch.key]["One"],
                                       d["mre"][sch.key]["Two"], places=12)

    def test_growth_is_a_subset_of_full(self):
        for lab in ("One", "Two"):
            self.assertLessEqual(self.table["growth"]["n_days"][lab],
                                 self.table["full"]["n_days"][lab])
            self.assertGreater(self.table["growth"]["n_days"][lab], 0)

    def test_mask_is_common_across_schemes(self):
        """Every scheme must be scored on exactly the same days."""
        suite = _compute_naive_suite(self.sim, self.city, self.gt)
        rho = np.asarray(suite["rho_t"], dtype=float)
        expected = rho > RHO_FLOOR
        for sch in SCHEMES:
            expected &= np.isfinite(np.asarray(suite[sch.key], dtype=float))
        self.assertEqual(int(expected.sum()), self.table["full"]["n_days"]["One"])
        self.assertEqual(int((expected & (rho > 1.0)).sum()),
                         self.table["growth"]["n_days"]["One"])

    def test_mre_dominates_absolute_signed_bias(self):
        """|mean signed bias| <= mean absolute relative error, by Jensen."""
        for period in PERIODS:
            d = self.table[period]
            for sch in SCHEMES:
                self.assertGreaterEqual(d["mre"][sch.key]["One"] + 1e-9,
                                        abs(d["bias"][sch.key]["One"]))
                self.assertGreaterEqual(d["mre"][sch.key]["One"], 0.0)

    def test_every_location_contributes_on_every_reported_day(self):
        """With no minimum-case rule the aggregate has a FIXED composition: an
        average over a changing subset of locations is not a network aggregate."""
        R_loc = estimate_R_independent(self.sim["incidence"], self.gt, window=7)
        suite = _compute_naive_suite(self.sim, self.city, self.gt)
        rho = np.asarray(suite["rho_t"], dtype=float)
        m = rho > RHO_FLOOR
        for sch in SCHEMES:
            m &= np.isfinite(np.asarray(suite[sch.key], dtype=float))
        for ref in references(rho, ESTIMATOR_WINDOW).values():
            m &= np.isfinite(ref) & (ref > RHO_FLOOR)
        n_contrib = (~np.isnan(R_loc)).sum(axis=1)[m]
        self.assertTrue((n_contrib == self.N).all(),
                        f"composition varies: {n_contrib.min()}..{n_contrib.max()} "
                        f"of {self.N} locations contribute on reported days")

    def test_suite_reproduces_estimate_R_independent_exactly(self):
        """_compute_naive_suite must not hold a second definition of the estimator."""
        R_ref = estimate_R_independent(self.sim["incidence"], self.gt, window=7)
        inc = self.sim["incidence"]
        suite = _compute_naive_suite(self.sim, self.city, self.gt)
        # every day on which the suite reports an aggregate must be a day on which
        # the shared estimator is defined for EVERY location
        for key in ("ind_am", "ind_pw"):
            rep = np.isfinite(np.asarray(suite[key], dtype=float))
            self.assertTrue((~np.isnan(R_ref[rep])).all(),
                            f"{key} reports on a day where a location is undefined")
        # and the arithmetic mean must equal the plain mean of that estimator there
        rep = np.isfinite(np.asarray(suite["ind_am"], dtype=float))
        np.testing.assert_allclose(np.asarray(suite["ind_am"])[rep],
                                   R_ref[rep].mean(axis=1), rtol=1e-12)

    def test_all_estimators_share_the_one_case_floor(self):
        """One floor, applied identically in all three estimator entry points."""
        import inspect
        from mobility_rt import estimators as _est
        import mobility_rt.framework as _fw
        for fn in (_est.estimate_R_independent, _est.compute_naive_R_aggregate,
                   _fw.estimate_R_independent):
            self.assertIn("R_MIN_WINDOW_CASES", inspect.getsource(fn),
                          f"{fn.__name__} does not apply the observability floor")
        # the suite must not re-implement the estimator; it must delegate to it
        suite_src = inspect.getsource(_est._compute_naive_suite)
        self.assertIn("estimate_R_independent(inc", suite_src,
                      "_compute_naive_suite no longer delegates to the shared estimator")
        self.assertNotIn("R_PRIOR_SHAPE, R_PRIOR_RATE", suite_src,
                         "_compute_naive_suite is re-implementing the estimator")
        # sub-one-case windows must be dropped, whole-case windows kept
        inc = np.zeros((40, 1)); inc[:, 0] = 0.01          # never reaches one case
        self.assertTrue(np.isnan(estimate_R_independent(inc, self.gt, 7)).all())
        inc2 = np.zeros((40, 1)); inc2[:, 0] = 5.0
        self.assertTrue(np.isfinite(estimate_R_independent(inc2, self.gt, 7)[10:]).all())

    def test_numerical_guards_are_inert(self):
        """Every epsilon in the chain must be a numerical guard, not a hidden
        parameter: none may fire on a day the tables report."""
        suite = _compute_naive_suite(self.sim, self.city, self.gt)
        inc = self.sim["incidence"]
        T, N = inc.shape
        rho = np.asarray(suite["rho_t"], dtype=float)
        m = rho > RHO_FLOOR
        for sch in SCHEMES:
            m &= np.isfinite(np.asarray(suite[sch.key], dtype=float))
        for ref in references(rho, ESTIMATOR_WINDOW).values():
            m &= np.isfinite(ref) & (ref > RHO_FLOOR)
        self.assertGreater(int(m.sum()), 0, "fixture reports no days at all")
        # rho floor
        self.assertEqual(int((rho <= RHO_FLOOR).sum()), 0,
                         "RHO_FLOOR is excluding days, so it is a parameter")
        # same-day incidence guards behind ind_iw / out_iw
        self.assertEqual(int((m & (inc.sum(axis=1) <= 1e-6)).sum()), 0,
                         "the incidence guard fires on a reported day")
        # windowed infectiousness guard behind the per-location estimator
        lam = np.zeros((T, N))
        for t in range(T):
            for a in range(1, len(self.gt)):
                if t - a >= 0:
                    lam[t] += self.gt[a] * inc[t - a]
        Lw = np.array([lam[max(0, t - ESTIMATOR_WINDOW + 1):t + 1].sum(axis=0)
                       for t in range(T)])
        self.assertEqual(int((m[:, None] & (Lw <= 1e-4)).sum()), 0,
                         "the infectiousness guard fires on a reported day")

    def test_window_constant_is_single_sourced(self):
        from mobility_rt.estimators import WINDOW
        self.assertIs(ESTIMATOR_WINDOW, WINDOW)

    def test_all_four_independent_weightings_share_their_days(self):
        """The four independent schemes must be defined on exactly the same days,
        or a difference between them could be a difference in day sets."""
        suite = _compute_naive_suite(self.sim, self.city, self.gt)
        masks = {k: np.isfinite(np.asarray(suite[k], dtype=float))
                 for k in ("ind_am", "ind_iw", "ind_pw", "ind_Rw")}
        ref = masks["ind_am"]
        for k, m in masks.items():
            np.testing.assert_array_equal(m, ref, err_msg=f"{k} differs from ind_am")

    def test_trailing_mean(self):
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        np.testing.assert_allclose(trailing_mean(x, 1), x)
        np.testing.assert_allclose(trailing_mean(x, 3),
                                   [1.0, 1.5, 2.0, 3.0, 4.0])   # leading partial windows
        np.testing.assert_allclose(trailing_mean(x, 99), np.cumsum(x) / np.arange(1, 6))
        # a NaN anywhere in the window makes that window undefined
        y = np.array([1.0, np.nan, 3.0, 4.0])
        out = trailing_mean(y, 2)
        self.assertTrue(np.isnan(out[1]) and np.isnan(out[2]))
        self.assertAlmostEqual(out[3], 3.5)
        with self.assertRaises(ValueError):
            trailing_mean(x, 0)

    def test_references_are_matched_to_smoothing(self):
        r = np.linspace(2.0, 0.5, 40)
        refs = references(r, ESTIMATOR_WINDOW)
        np.testing.assert_array_equal(refs[False], r)      # unwindowed: rho itself
        np.testing.assert_allclose(refs[True], trailing_mean(r, ESTIMATOR_WINDOW))
        self.assertFalse(np.allclose(refs[True], refs[False]),
                         "the two references must actually differ")

    def test_scheme_windowed_flags_are_right(self):
        """A sliding-window scheme must be flagged, an instantaneous one must not."""
        flags = {sch.key: sch.windowed for sch in SCHEMES}
        for k in ("agg", "ind_am", "ind_iw", "ind_pw", "ind_Rw"):
            self.assertTrue(flags[k], f"{k} uses the Cori window and must be flagged")
        for k in ("out_am", "out_iw", "out_pw", "out_Rw"):
            self.assertFalse(flags[k], f"{k} carries no window and must not be flagged")
        self.assertEqual(len(flags), len(SCHEMES))       # keys unique
        self.assertEqual(sorted(flags), sorted(
            ["agg", "ind_am", "ind_iw", "ind_pw", "ind_Rw",
             "out_am", "out_iw", "out_pw", "out_Rw"]),
            "the scheme set changed; the tables and figures assume these nine")

    def test_windowed_scheme_is_scored_against_the_trailing_mean(self):
        """End-to-end: the table cell for a windowed scheme uses the smoothed
        reference, and would differ if rho were used instead."""
        suite = _compute_naive_suite(self.sim, self.city, self.gt)
        rho = np.asarray(suite["rho_t"], dtype=float)
        tm = trailing_mean(rho, ESTIMATOR_WINDOW)
        usable = rho > RHO_FLOOR
        for sch in SCHEMES:
            usable &= np.isfinite(np.asarray(suite[sch.key], dtype=float))
        m = usable & (rho > 1.0)
        e = np.asarray(suite["agg"], dtype=float)
        via_tm = np.mean(np.abs(e[m] - tm[m]) / tm[m]) * 100
        via_rho = np.mean(np.abs(e[m] - rho[m]) / rho[m]) * 100
        self.assertAlmostEqual(self.table["growth"]["mre"]["agg"]["One"], via_tm,
                               places=9)
        self.assertNotAlmostEqual(via_tm, via_rho, places=3,
                                  msg="fixture has no weekly variation, so this test "
                                      "cannot detect a reference mix-up")

    def test_unwindowed_scheme_is_scored_against_rho(self):
        suite = _compute_naive_suite(self.sim, self.city, self.gt)
        rho = np.asarray(suite["rho_t"], dtype=float)
        usable = rho > RHO_FLOOR
        for sch in SCHEMES:
            usable &= np.isfinite(np.asarray(suite[sch.key], dtype=float))
        m = usable & (rho > 1.0)
        e = np.asarray(suite["out_Rw"], dtype=float)
        self.assertAlmostEqual(self.table["growth"]["mre"]["out_Rw"]["One"],
                               np.mean(np.abs(e[m] - rho[m]) / rho[m]) * 100, places=9)

    def test_threshold_crossing_day(self):
        """The settled crossing, not the first dip below 1."""
        self.assertEqual(threshold_crossing_day([2, 2, 0.9, 1.2, 1.1, 0.8, 0.7]), 5)
        self.assertEqual(threshold_crossing_day([0.5, 0.4, 0.3]), 0)
        self.assertIsNone(threshold_crossing_day([2, 2, 2]))
        self.assertIsNone(threshold_crossing_day([np.nan, np.nan]))
        # NaN gaps must not shift the answer
        self.assertEqual(
            threshold_crossing_day([2, 2, 1.5, np.nan, np.nan, 0.8, 0.7]), 5)
        # a leading NaN run is skipped, not counted as "below 1"
        self.assertEqual(threshold_crossing_day([np.nan, np.nan, 0.4]), 2)

    def test_bias_figure_rejects_a_scheme_with_unknown_smoothing(self):
        """A middle-column key absent from SCHEMES has no known reference, so the
        figure must refuse rather than silently pick one."""
        from mobility_rt.plotting.figures_main import _plot_main_bias_3x3
        d = _compute_naive_suite(self.sim, self.city, self.gt)
        with self.assertRaises(KeyError):
            _plot_main_bias_3x3([(d, "a"), (d, "b"), (d, "c")],
                                "not_a_scheme", "x", "y", "unused.pdf")

    def test_duplicate_scenario_labels_rejected(self):
        with self.assertRaises(ValueError):
            compute_error_table([("Same", self.sim, self.city),
                                 ("Same", self.sim, self.city)], self.gt)
        with self.assertRaises(ValueError):
            compute_error_table([], self.gt)

    def test_tex_escape_does_not_double_escape(self):
        """Sequential str.replace would turn \\textbackslash{} into \\textbackslash\\{\\}."""
        from mobility_rt.error_tables import _tex_escape
        self.assertEqual(_tex_escape("a_b"), r"a\_b")
        self.assertEqual(_tex_escape("50% & $x$"), r"50\% \& \$x\$")
        self.assertEqual(_tex_escape("c:\\path"), r"c:\textbackslash{}path")
        self.assertEqual(_tex_escape("{x}"), r"\{x\}")
        self.assertEqual(_tex_escape("Scenario A-het"), "Scenario A-het")

    def test_summary_figure_writes_and_validates_labels(self):
        from mobility_rt.plotting.figures_main import plot_estimator_error_summary
        pairs = [("One", "Two", "Panel")]
        with _tempfile.TemporaryDirectory() as d:
            out = plot_estimator_error_summary(self.table, pairs,
                                               save_prefix=_os.path.join(d, "f"))
            self.assertTrue(_os.path.exists(out))
            self.assertGreater(_os.path.getsize(out), 0)
            with self.assertRaises(KeyError):
                plot_estimator_error_summary(
                    self.table, [("One", "Nope", "Panel")],
                    save_prefix=_os.path.join(d, "g"))

    def test_files_written_and_parseable(self):
        with _tempfile.TemporaryDirectory() as d:
            paths = write_error_tables(self.table, d, prefix="tbl")
            self.assertEqual(len(paths), 2 * len(TABLE_PERIODS))
            self.assertFalse(_os.path.exists(_os.path.join(d, "tbl_decline.csv")),
                             "decline is a computation period, not a written table")
            for path in paths:
                self.assertTrue(_os.path.exists(path))
                self.assertGreater(_os.path.getsize(path), 0)
            for period in TABLE_PERIODS:
                with open(_os.path.join(d, f"tbl_{period}.csv")) as fh:
                    rows = list(_csv.reader(fh))
                self.assertEqual(len(rows), len(SCHEMES) + 2)   # header + schemes + n_days
                self.assertEqual(len(set(len(r) for r in rows)), 1)
                self.assertEqual(rows[-1][0], "n_days")
                for r in rows[1:-1]:
                    for cell in r[2:]:
                        float(cell)                              # every value numeric
                tex = open(_os.path.join(d, f"tbl_{period}.tex")).read()
                self.assertEqual(tex.count("\\\\"), len(SCHEMES) + 2)
                self.assertIn("booktabs", tex)          # the rules need that package
                self.assertTrue(tex.startswith("%"))    # generated-file banner
                self.assertIn("\\begin{tabular}{lrr}", tex)      # 2 scenario columns
                self.assertEqual(tex.count("\\begin{table}"), 1)
                self.assertEqual(tex.count("\\end{table}"), 1)



# ---------------------------------------------------------------------------
# Generation time under spatially heterogeneous contact and transmissibility
# ---------------------------------------------------------------------------

from mobility_rt.generation_time import realised_generation_times
from mobility_rt.gt_spatial import (
    build_gt_spatial_table,
    epidemic_window,
    forward_gt_dispersion,
    realised_gt_dispersion,
    universal_gt,
)


class TestForwardGTUnderHeterogeneity(unittest.TestCase):
    """c[l] and m[k] carry no infection age, so the forward GT stays universal."""

    @classmethod
    def setUpClass(cls):
        rng = np.random.default_rng(23)
        cls.N = 6
        raw = rng.uniform(0.2, 1.0, (cls.N, cls.N))
        cls.f_t = raw / raw.sum(axis=1, keepdims=True)
        cls.pops = rng.uniform(1e4, 1e5, cls.N)
        cls.S = cls.pops * rng.uniform(0.4, 1.0, cls.N)
        cls.lw, cls.lb = 14.0, 4.2
        cls.max_days = 20
        cls.p = discretise_gamma(5.5, 1.8, cls.max_days)
        cls.prob_peak = 0.035
        cls.c = np.exp(rng.normal(0, 0.4, cls.N))
        cls.m = np.exp(rng.normal(0, 0.5, cls.N))

    def _gt(self, cm=None, im=None):
        return compute_generation_times(self.f_t, self.S, self.pops, self.prob_peak,
                                        self.p, self.max_days, self.lw, self.lb,
                                        cm, im)

    def test_none_multipliers_reproduce_the_homogeneous_gt_exactly(self):
        """The heterogeneity arguments must be a no-op when they are not supplied."""
        a = compute_generation_times(self.f_t, self.S, self.pops, self.prob_peak,
                                     self.p, self.max_days, self.lw, self.lb)
        b = self._gt(None, None)
        for key in a:
            np.testing.assert_array_equal(a[key], b[key])

    def test_every_gt_is_the_universal_shape_under_heterogeneity(self):
        """Pairwise, outward, inward and meeting-site GTs all collapse onto p/int p."""
        g_univ = universal_gt(self.p)
        gt = self._gt(self.c, self.m)
        R = gt["R_matrix"]
        for k in range(self.N):
            for j in range(self.N):
                if R[k, j] > 1e-15:
                    np.testing.assert_allclose(gt["g_pairwise"][:, k, j], g_univ,
                                               atol=1e-14)
        for key, marg in (("g_outward", R.sum(axis=1)),
                          ("g_inward", R.sum(axis=0)),
                          ("g_meeting", gt["R_meeting"])):
            for l in range(self.N):
                if marg[l] > 1e-15:
                    np.testing.assert_allclose(gt[key][:, l], g_univ, atol=1e-14)
        np.testing.assert_allclose(gt["g_network"], g_univ, atol=1e-14)

    def test_snapshot_R_matrix_matches_the_kernel_module(self):
        """The GT snapshot must carry the same R^{kj} the rest of the code computes."""
        gt = self._gt(self.c, self.m)
        ref = compute_R_matrix(self.f_t, self.S, self.pops, self.prob_peak, self.p,
                               self.lw, self.lb, self.c, self.m)
        np.testing.assert_allclose(gt["R_matrix"], ref, rtol=1e-12)

    def test_snapshot_R_meeting_matches_the_kernel_module(self):
        """R^l_meeting is taken from compute_R_meeting, not re-derived, so it agrees."""
        gt = self._gt(self.c, self.m)
        ref = compute_R_meeting(self.f_t, self.S, self.pops, self.prob_peak, self.p,
                                self.lw, self.lb, self.c, self.m)
        np.testing.assert_allclose(gt["R_meeting"], ref, rtol=1e-12)

    def test_heterogeneity_does_change_R_even_though_the_gt_is_unchanged(self):
        """Guards against the multipliers being silently dropped altogether."""
        hom = self._gt()
        het = self._gt(self.c, self.m)
        self.assertFalse(np.allclose(hom["R_matrix"], het["R_matrix"]))

    def test_universal_gt_is_normalised_over_ages_one_and_up(self):
        g = universal_gt(self.p)
        self.assertEqual(g[0], 0.0)
        self.assertAlmostEqual(float(g.sum()), 1.0, places=12)

    def test_forward_dispersion_reports_machine_precision(self):
        snaps = {"peak": self._gt(self.c, self.m)}
        d = forward_gt_dispersion(snaps, self.p)
        self.assertLess(d["sup_deviation"], 1e-13)
        self.assertLess(d["mean_gt_range"], 1e-9)
        self.assertAlmostEqual(d["mean_gt"], d["universal_mean"], places=10)
        self.assertEqual(d["n_pairs"], self.N * self.N)


class TestRealisedGenerationTimes(unittest.TestCase):
    """The backward interval is a reweighting of p(a_E) by infection-age structure."""

    @classmethod
    def setUpClass(cls):
        cls.N, cls.T = 5, 70
        cls.params = COVID_PARAMS
        cls.max_days = cls.params["max_gen_time"]
        cls.p = discretise_gamma(cls.params["gen_time_mean"],
                                 cls.params["gen_time_sd"], cls.max_days)
        cls.city = generate_city(cls.N, scenario="lagos", seed=5)
        cls.pops = cls.city[1]
        cls.f = generate_mobility(cls.N, cls.T, cls.pops, cls.city[2],
                                  cls.city[3], cls.city[4], seed=5)[0]
        init = np.zeros(cls.N); init[0] = 10.0
        LW = cls.params["base_contact_rate"]
        cls.c = contact_multipliers(cls.city[3])
        cls.m = infectiousness_multipliers(cls.N, seed=5)
        common = dict(prob_peak=cls.params["prob_transmission_peak"],
                      infect_profile=cls.p, max_days=cls.max_days,
                      R0_target=cls.params["R0_target"],
                      initial_infections=init, lambda_within=LW,
                      lambda_between=LW * 0.30, birth_rate=3e-5, death_rate=3e-5,
                      stochastic=False, susceptible_depletion=True, seed=5)
        cls.sim_hom = simulate_epidemic_pde(cls.T, cls.N, cls.pops, cls.f, **common)
        cls.sim_het = simulate_epidemic_pde(cls.T, cls.N, cls.pops, cls.f,
                                            contact_mult=cls.c, infect_mult=cls.m,
                                            **common)
        cls.rg = realised_generation_times(cls.sim_het["E_pde_state"],
                                           cls.sim_het["incidence_matrix"], cls.p)

    def test_every_distribution_is_a_pmf_or_empty(self):
        for key in ("g_infector", "g_infectee", "g_network"):
            tot = self.rg[key].sum(axis=-1)
            near_one = np.isclose(tot, 1.0, atol=1e-12)
            near_zero = np.isclose(tot, 0.0, atol=1e-12)
            self.assertTrue(np.all(near_one | near_zero), key)
            self.assertTrue(near_one.any(), f"{key} is empty on every day")
            self.assertTrue(np.all(self.rg[key] >= 0.0), key)

    def test_age_zero_never_carries_mass(self):
        """a_E = 0 is excluded from transmission, so no infector can be age zero."""
        for key in ("g_infector", "g_infectee", "g_network"):
            self.assertEqual(float(np.abs(self.rg[key][..., 0]).max()), 0.0, key)

    def test_a_flat_age_profile_returns_the_intrinsic_shape(self):
        """With E^k(t,a) constant in a, the reweighting is the identity."""
        T, N, A = 3, 2, self.max_days
        E = np.ones((T, N, A))
        inc = np.ones((T, N, N))
        out = realised_generation_times(E, inc, self.p)
        g_univ = universal_gt(self.p)
        for key in ("g_infector", "g_infectee"):
            for t in range(T):
                for n in range(N):
                    np.testing.assert_allclose(out[key][t, n], g_univ, atol=1e-14)
        np.testing.assert_allclose(out["g_network"][0], g_univ, atol=1e-14)

    def test_infectee_distribution_is_the_pairwise_incidence_mixture(self):
        """Recomputed by hand on one day, straight from the stored arrays."""
        inc_mat = self.sim_het["incidence_matrix"]
        t = int(self.sim_het["incidence"].sum(axis=1).argmax())
        for j in range(self.N):
            w = inc_mat[t, :, j]
            if w.sum() <= 0:
                continue
            expect = (w @ self.rg["g_infector"][t]) / w.sum()
            np.testing.assert_allclose(self.rg["g_infectee"][t, j], expect, atol=1e-14)

    def test_means_are_consistent_with_the_distributions(self):
        days = np.arange(self.max_days, dtype=float)
        mu = self.rg["mean_infectee"]
        ok = ~np.isnan(mu)
        np.testing.assert_allclose(mu[ok], (self.rg["g_infectee"] @ days)[ok],
                                   atol=1e-12)
        self.assertTrue(np.isnan(mu[~ok]).all())

    def test_growth_phase_interval_is_shorter_than_the_intrinsic_mean(self):
        """Backward intervals are early-shifted while incidence is still growing."""
        mu_univ = float(np.arange(self.max_days) @ universal_gt(self.p))
        peak = int(self.sim_het["incidence"].sum(axis=1).argmax())
        grow = self.rg["mean_network"][max(10, peak // 3):max(11, peak // 2)]
        grow = grow[~np.isnan(grow)]
        self.assertTrue(grow.size > 0)
        self.assertLess(float(grow.max()), mu_univ)

    def test_the_realised_interval_varies_across_space_and_the_forward_one_does_not(self):
        """The question this whole analysis answers, as a single assertion."""
        dis = realised_gt_dispersion(self.sim_het, self.p)
        self.assertGreater(dis["max_spread"], 1e-3)   # backward: genuinely spatial
        snaps = {}
        for name, day in (("early", 20), ("peak", 45)):
            snaps[name] = compute_generation_times(
                self.f[day], self.sim_het["susceptibles"][day], self.pops,
                self.params["prob_transmission_peak"], self.p, self.max_days,
                self.sim_het["lambda_within_scaled"],
                self.sim_het["lambda_between_scaled"], self.c, self.m)
        self.assertLess(forward_gt_dispersion(snaps, self.p)["sup_deviation"], 1e-13)

    def test_dispersion_is_restricted_to_the_observable_window(self):
        dis = realised_gt_dispersion(self.sim_het, self.p)
        mask = epidemic_window(self.sim_het["incidence"])
        np.testing.assert_array_equal(mask, dis["window"])
        self.assertTrue(np.all(np.isnan(dis["spread"][~mask])))

    def test_rejects_mismatched_shapes(self):
        E = self.sim_het["E_pde_state"]
        inc = self.sim_het["incidence_matrix"]
        with self.assertRaises(ValueError):
            realised_generation_times(E, inc, self.p[:-1])
        with self.assertRaises(ValueError):
            realised_generation_times(E, inc[:, :-1], self.p)
        with self.assertRaises(ValueError):
            realised_generation_times(E[0], inc, self.p)

    def test_table_is_written_and_every_value_parses(self):
        import os as _os, csv as _csv, tempfile
        snaps_hom = {"peak": compute_generation_times(
            self.f[40], self.sim_hom["susceptibles"][40], self.pops,
            self.params["prob_transmission_peak"], self.p, self.max_days,
            self.sim_hom["lambda_within_scaled"],
            self.sim_hom["lambda_between_scaled"])}
        snaps_het = {"peak": compute_generation_times(
            self.f[40], self.sim_het["susceptibles"][40], self.pops,
            self.params["prob_transmission_peak"], self.p, self.max_days,
            self.sim_het["lambda_within_scaled"],
            self.sim_het["lambda_between_scaled"], self.c, self.m)}
        scen = [("Dense urban (A)", self.sim_hom, snaps_hom),
                ("Dense urban (A-het)", self.sim_het, snaps_het)]
        with tempfile.TemporaryDirectory() as d:
            table, paths = build_gt_spatial_table(scen, self.p, out_dir=d,
                                                  prefix="gt", verbose=False)
            self.assertEqual(len(paths), 2)
            rows = list(_csv.reader(open(_os.path.join(d, "gt.csv"))))
            self.assertEqual(rows[0], ["quantity", "Dense urban (A)",
                                       "Dense urban (A-het)"])
            for r in rows[1:]:
                for cell in r[1:]:
                    float(cell)
            tex = open(_os.path.join(d, "gt.tex")).read()
            self.assertTrue(tex.startswith("%"))
            self.assertIn("booktabs", tex)
            self.assertIn("\\begin{tabular}{lrr}", tex)
            self.assertEqual(tex.count("\\begin{table}"), 1)
        # the forward half is universal in BOTH columns; the realised half is not
        for lab in ("Dense urban (A)", "Dense urban (A-het)"):
            self.assertLess(table[lab]["forward_sup"], 1e-13)
            self.assertGreater(table[lab]["spread_max"], 0.0)

    def test_table_rejects_duplicate_scenario_labels(self):
        snaps = {"peak": compute_generation_times(
            self.f[40], self.sim_hom["susceptibles"][40], self.pops,
            self.params["prob_transmission_peak"], self.p, self.max_days,
            self.sim_hom["lambda_within_scaled"],
            self.sim_hom["lambda_between_scaled"])}
        with self.assertRaises(ValueError):
            build_gt_spatial_table([("A", self.sim_hom, snaps),
                                    ("A", self.sim_het, snaps)],
                                   self.p, out_dir="unused", verbose=False)


class TestCorrectedFigureScenarios(unittest.TestCase):
    """Every alternative scenario gets the same corrected fig_02/03/04 set."""

    @classmethod
    def setUpClass(cls):
        import matplotlib
        matplotlib.use("Agg")
        cls.N, cls.T = 4, 40
        params = COVID_PARAMS
        cls.max_days = params["max_gen_time"]
        cls.p = discretise_gamma(params["gen_time_mean"], params["gen_time_sd"],
                                 cls.max_days)
        cls.city = generate_city(cls.N, scenario="lagos", seed=9)
        pops = cls.city[1]
        cls.f, base_f = generate_mobility(cls.N, cls.T, pops, cls.city[2],
                                          cls.city[3], cls.city[4], seed=9)
        cls.f_static = np.stack([base_f] * cls.T, axis=0)
        init = np.zeros(cls.N); init[0] = 10.0
        LW = params["base_contact_rate"]
        common = dict(prob_peak=params["prob_transmission_peak"], infect_profile=cls.p,
                      max_days=cls.max_days, R0_target=params["R0_target"],
                      initial_infections=init, lambda_within=LW,
                      lambda_between=LW * 0.30, seed=9)
        cls.sim = simulate_epidemic_pde(cls.T, cls.N, pops, cls.f, **common)
        cls.sim_s = simulate_epidemic_pde(cls.T, cls.N, pops, cls.f_static, **common)
        cls.sim_h = simulate_epidemic_pde(
            cls.T, cls.N, pops, cls.f,
            contact_mult=contact_multipliers(cls.city[3]),
            infect_mult=infectiousness_multipliers(cls.N, seed=9), **common)
        cls.gt = {"peak": compute_generation_times(
            cls.f[20], cls.sim["susceptibles"][20], pops,
            params["prob_transmission_peak"], cls.p, cls.max_days,
            cls.sim["lambda_within_scaled"], cls.sim["lambda_between_scaled"])}
        cls.R_ind = estimate_R_independent(cls.sim["incidence"], cls.p, window=7)

    def _args(self, **kw):
        from mobility_rt.plotting.figures_corrected import run_corrected_figures
        base = dict(sim=self.sim, sim_static=self.sim_s, city_data=self.city,
                    f_series=self.f, f_static=self.f_static, R_ind=self.R_ind,
                    R_ind_static=self.R_ind, gt_snaps=self.gt,
                    gt_snaps_static=self.gt, gen_time_pmf=self.p, w_within=self.p,
                    w_between=self.p, max_days=self.max_days)
        base.update(kw)
        return run_corrected_figures, base

    def test_each_extra_scenario_gets_the_full_file_set(self):
        import os as _os, tempfile
        run, kw = self._args()
        extra = {"sim": self.sim_h, "city_data": self.city, "f_series": self.f,
                 "R_ind": self.R_ind, "gt_snaps": self.gt,
                 "name": "Dense urban (A-het)", "prefix": "fig_A_het"}
        with tempfile.TemporaryDirectory() as d:
            run(out_dir=d, extra_scenarios=[extra], **kw)
            names = set(_os.listdir(d))
            for pfx in ("fig", "fig_SI_static", "fig_A_het"):
                for suffix in ("_02_overview.pdf", "_02_overview_surface.pdf",
                               "_03_taxonomy.pdf", "_03_taxonomy_surface.pdf",
                               "_04_spectral.pdf"):
                    self.assertIn(pfx + suffix, names)
            # three scenarios x five files, and nothing else
            self.assertEqual(len(names), 15)

    def test_a_duplicate_prefix_is_refused_rather_than_overwriting(self):
        import tempfile
        run, kw = self._args()
        extra = {"sim": self.sim_h, "city_data": self.city, "f_series": self.f,
                 "R_ind": self.R_ind, "gt_snaps": self.gt,
                 "name": "clash", "prefix": "fig_SI_static"}
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(ValueError):
                run(out_dir=d, extra_scenarios=[extra], **kw)

    def test_an_incomplete_extra_scenario_is_refused(self):
        import tempfile
        run, kw = self._args()
        extra = {"sim": self.sim_h, "city_data": self.city, "f_series": self.f,
                 "name": "incomplete", "prefix": "fig_x"}
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(KeyError):
                run(out_dir=d, extra_scenarios=[extra], **kw)

    def test_no_extras_still_writes_exactly_the_original_two_sets(self):
        import os as _os, tempfile
        run, kw = self._args()
        with tempfile.TemporaryDirectory() as d:
            prefixes = run(out_dir=d, **kw)
            self.assertEqual(len(prefixes), 2)
            self.assertEqual(len(_os.listdir(d)), 10)



# ---------------------------------------------------------------------------
# Summary printer
# ---------------------------------------------------------------------------

def run_with_summary():
    import sys
    loader = unittest.TestLoader()
    suite  = loader.loadTestsFromModule(sys.modules[__name__])   # all TestCase classes
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
