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
    reactivity,
    amplification_envelope,
    euler_lotka_r,
    simulate_epidemic_pde,
    estimate_R_independent,
    type_reproduction_number,
    type_reproduction_numbers,
    lehmer_mean_spectrum,
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

        city_data = generate_city(N, scenario="lagos", seed=SEED)
        coords, pops, dists, node_types, meta = city_data

        f_jk, base_f = generate_mobility(
            N, T, pops, dists, meta,
            day_variation_sd=0.15, seed=SEED)

        initial = np.zeros(N);  initial[0] = 5
        sim = simulate_epidemic_pde(
            T, N, pops, f_jk,
            params["prob_transmission_peak"], infect_profile, max_days,
            params["R0_target"], initial,
            params["base_contact_rate"],
            params["base_contact_rate"] * 0.30,
            f_calibration=base_f)

        cls.N            = N
        cls.T            = T
        cls.SEED         = SEED
        cls.params       = params
        cls.max_days     = max_days
        cls.gen_time_pmf = gen_time_pmf
        cls.infect_profile = infect_profile
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
        cls.S_series     = sim["S_start"]

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
        f_L, _ = generate_mobility(N, T, pops_L, dists_L, meta_L,
                                   day_variation_sd=0.15, seed=SEED)
        row_sums_L = f_L.sum(axis=-1)
        np.testing.assert_allclose(
            row_sums_L, np.ones((T, N)),
            atol=1e-10,
            err_msg="Lagos f_jk is not row-stochastic")

        # Zambia
        city_Z = generate_city(N, scenario="zambia", seed=SEED)
        coords_Z, pops_Z, dists_Z, types_Z, meta_Z = city_Z
        f_Z, _ = generate_mobility(N, T, pops_Z, dists_Z, meta_Z,
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

    def test_euler_lotka_reference_values_and_no_silent_except(self):
        """Known roots for the model pmf; NaN (not an exception) when the root lies
        outside the bracket |r| <= 2 per day; and no exception handler that could turn
        a solver failure into a silent NaN."""
        import inspect
        p = discretise_gamma(5.5, 1.8, 25)
        self.assertTrue(np.isnan(euler_lotka_r(1e4, p)))
        self.assertAlmostEqual(euler_lotka_r(1.5, p), 0.07543477, places=8)
        self.assertAlmostEqual(euler_lotka_r(1.5384, p), 0.08025335, places=8)
        self.assertNotIn("except", inspect.getsource(euler_lotka_r))

    def test_importing_the_package_leaves_warning_filters_unchanged(self):
        """Importing mobility_rt must not silence warnings process-wide.

        The filter list is emptied first: the package is already imported by the time
        this runs, and re-adding an identical "ignore" filter would leave a non-empty
        list unchanged, so comparing against the current list could not fail."""
        import importlib
        import warnings
        import mobility_rt
        with warnings.catch_warnings():
            warnings.resetwarnings()
            importlib.reload(mobility_rt)
            after = list(warnings.filters)
        self.assertEqual(after, [],
                         "importing mobility_rt changed the global warning filters")

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

        # Symmetric positive matrix: sigma == rho, because the singular values of a
        # symmetric matrix are the absolute values of its eigenvalues, so the largest
        # singular value (Eq 37) equals the spectral radius.
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
        f_jk, _ = generate_mobility(N, T, pops, dists, meta,
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

    def test_estimator_prior_has_mean_one(self):
        """Prior mean must be 1, NOT EpiEstim's default of 5.

        A prior mean of 1 is the only one that cannot move a threshold crossing:
        the posterior mean is a weighted average of the prior mean and the data
        ratio, so centring it at 1 leaves sign(R-1) to the data alone.
        """
        self.assertAlmostEqual(R_PRIOR_SHAPE / R_PRIOR_RATE, 1.0, places=6,
                               msg="independent-estimator prior mean must be 1")
        # the shape is one pseudo-case, as in EpiEstim: only the centre moved
        self.assertAlmostEqual(R_PRIOR_SHAPE, 1.0, places=12)
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

    def test_lehmer_mean_limits(self):
        """X(0)=arithmetic mean, X(α→∞)→max of R^j_out (Eq 41 limits), X(1)=E(t)."""
        # Well-separated values (max ratio ≥ 2) so the α→∞ limit converges fast.
        R_out = np.array([[0.5, 1.0, 2.0],
                          [1.0, 1.5, 3.0]])            # (T=2, N=3)
        X = lehmer_mean_spectrum(R_out, np.array([0.0, 1.0, 80.0]))
        np.testing.assert_allclose(X[0], R_out.mean(axis=1), atol=1e-9)          # arith mean
        E_t = (R_out ** 2).sum(axis=1) / R_out.sum(axis=1)
        np.testing.assert_allclose(X[1], E_t, atol=1e-9)                          # α=1 → E(t)
        np.testing.assert_allclose(X[2], R_out.max(axis=1), atol=1e-6)           # α→∞ → max
        # monotone non-decreasing in α
        Xs = lehmer_mean_spectrum(R_out, np.array([0.0, 1.0, 2.0, 5.0]))
        self.assertTrue(np.all(np.diff(Xs, axis=0) >= -1e-9))

    def test_row_stochastic_under_stress(self):
        """Mobility rows sum to 1 under strong hub amplification."""
        for N in (4, 10, 15):
            city = generate_city(N, "lagos", seed=3)
            _, pops, dists, types, meta = city
            f, base = generate_mobility(N, 30, pops, dists, meta,
                                        seed=3, hub_attraction_power=8.0)
            self.assertTrue(np.allclose(f.sum(axis=-1), 1.0, atol=1e-9),
                            msg=f"rows not stochastic (N={N}, hub_power=8)")

    def test_type_R_single_location(self):
        """type_reproduction_number does not crash for N=1 (returns R_jj)."""
        self.assertAlmostEqual(type_reproduction_number(np.array([[0.7]]), 0), 0.7, places=10)


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
        cls.f, cls.base_f = generate_mobility(cls.N, cls.T, cls.pops, cls.city[2],
                                              cls.city[4], seed=3)
        cls.init = np.zeros(cls.N); cls.init[0] = 10.0
        cls.LW = cls.params["base_contact_rate"]
        cls.LB = cls.LW * 0.30
        cls.c = contact_multipliers(cls.city[3])
        cls.m = infectiousness_multipliers(cls.N, seed=3)

    def _sim(self, **kw):
        return simulate_epidemic_pde(
            self.T, self.N, self.pops, self.f, self.params["prob_transmission_peak"],
            self.p, self.max_days, self.params["R0_target"], self.init,
            self.LW, self.LB, f_calibration=self.base_f, **kw)

    def test_calibration_holds_under_heterogeneity(self):
        sim = self._sim(contact_mult=self.c, infect_mult=self.m)
        R_cal = compute_R_matrix(self.base_f, self.pops, self.pops,
                                 self.params["prob_transmission_peak"], self.p,
                                 sim["lambda_within_scaled"],
                                 sim["lambda_between_scaled"], self.c, self.m)
        self.assertAlmostEqual(R_system(R_cal), self.params["R0_target"], places=8)

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
        """E_j(t,0) = Σ_k R_kj(t) Λ_k(t) / Σ_{a≥1} p(a), with and without
        heterogeneity — the identity that makes R(t) the NGM of these dynamics.

        Exact for update="euler", whose boundary condition is linear in S.  The
        default exponential update depletes S over the day, so there the identity
        holds only to first order in the daily force of infection φ: the relative
        shortfall of S(1−e^{−φ}) against Sφ is φ/2 + O(φ²), checked against the φ
        each run actually produced."""
        sum_w = self.p[1:].sum()
        for kw in ({}, dict(contact_mult=self.c),
                   dict(infect_mult=self.m),
                   dict(contact_mult=self.c, infect_mult=self.m)):
            sim = self._sim(update="euler", **kw)
            E = sim["E_pde_state"]
            for t in range(1, 25):     # early days: no susceptible clamping
                lam = E[t, :, 1:] @ self.p[1:]
                pred = sim["R_matrices"][t].T @ lam / sum_w
                np.testing.assert_allclose(
                    sim["incidence"][t], pred, rtol=1e-9, atol=1e-9,
                    err_msg=f"incidence != R^T Λ / Σp at t={t} for {kw}")

            exp_sim = self._sim(**kw)          # default: exponential escape
            E = exp_sim["E_pde_state"]
            for t in range(1, 25):
                lam  = E[t, :, 1:] @ self.p[1:]
                pred = exp_sim["R_matrices"][t].T @ lam / sum_w        # = S_j φ_j
                S    = np.maximum(exp_sim["S_start"][t], 1e-30)
                phi  = pred / S
                rel  = np.where(pred > 0, (pred - exp_sim["incidence"][t])
                                / np.maximum(pred, 1e-30), 0.0)
                # RELATIVE tolerance: on this fixture φ/2 is O(1e-5), so an absolute
                # tolerance of 1e-3 would be satisfied by rel ≡ 0 as well, i.e. by a
                # linear update.  The residual must track φ/2 itself, to O(φ).
                np.testing.assert_allclose(
                    rel, phi / 2, rtol=1e-2, atol=1e-12,
                    err_msg=f"exponential residual is not φ/2 at t={t} for {kw}")

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
# Estimator mean-relative-error tables
# ---------------------------------------------------------------------------

import csv as _csv
import os as _os
import tempfile as _tempfile

import numpy as _np_for_fixture

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
        f, base_f = generate_mobility(cls.N, cls.T, cls.city[1], cls.city[2],
                                      cls.city[4], seed=6)
        init = np.zeros(cls.N); init[0] = 10.0
        LW = params["base_contact_rate"]
        cls.sim = simulate_epidemic_pde(
            cls.T, cls.N, cls.city[1], f, params["prob_transmission_peak"], cls.gt,
            params["max_gen_time"], params["R0_target"], init, LW, LW * 0.30,
            f_calibration=base_f)
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
        """One floor, applied identically in both estimator entry points."""
        import inspect
        from mobility_rt import estimators as _est
        for fn in (_est.estimate_R_independent, _est.compute_naive_R_aggregate):
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
        # windowed infectiousness guard behind the per-location estimator, and the
        # total-infectiousness weight guards behind ind_Lw (windowed) and out_Lw
        lam = np.zeros((T, N))
        for t in range(T):
            for a in range(1, len(self.gt)):
                if t - a >= 0:
                    lam[t] += self.gt[a] * inc[t - a]
        Lw = np.array([lam[max(0, t - ESTIMATOR_WINDOW + 1):t + 1].sum(axis=0)
                       for t in range(T)])
        self.assertEqual(int((m[:, None] & (Lw <= 1e-4)).sum()), 0,
                         "the infectiousness guard fires on a reported day")
        self.assertEqual(int((m & (Lw.sum(axis=1) <= 1e-6)).sum()), 0,
                         "the ind_Lw weight guard fires on a reported day")
        self.assertEqual(int((m & (lam.sum(axis=1) <= 1e-6)).sum()), 0,
                         "the out_Lw weight guard fires on a reported day")

    def test_window_constant_is_single_sourced(self):
        from mobility_rt.estimators import WINDOW
        self.assertIs(ESTIMATOR_WINDOW, WINDOW)

    def test_all_four_independent_weightings_share_their_days(self):
        """The four independent schemes must be defined on exactly the same days,
        or a difference between them could be a difference in day sets."""
        suite = _compute_naive_suite(self.sim, self.city, self.gt)
        masks = {k: np.isfinite(np.asarray(suite[k], dtype=float))
                 for k in ("ind_am", "ind_Lw", "ind_pw", "ind_Rw")}
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
        for k in ("agg", "ind_am", "ind_Lw", "ind_pw", "ind_Rw"):
            self.assertTrue(flags[k], f"{k} uses the Cori window and must be flagged")
        for k in ("out_am", "out_Lw", "out_pw", "out_Rw"):
            self.assertFalse(flags[k], f"{k} carries no window and must not be flagged")
        self.assertEqual(len(flags), len(SCHEMES))       # keys unique
        self.assertEqual(sorted(flags), sorted(
            ["agg", "ind_am", "ind_Lw", "ind_pw", "ind_Rw",
             "out_am", "out_Lw", "out_pw", "out_Rw"]),
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
        from mobility_rt.plotting.figures_main import _plot_main_bias_grid
        d = _compute_naive_suite(self.sim, self.city, self.gt)
        with self.assertRaises(KeyError):
            _plot_main_bias_grid([(d, "a"), (d, "b"), (d, "c")],
                                "not_a_scheme", "unused.pdf")

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

    def test_error_table_tex_is_structurally_valid(self):
        """Balanced braces and math, closed environments, one cell per column in every
        row, and one numeric column per scenario, for every written period.  This is
        the only structural check on the .tex of Tables SI1-SI2."""
        import re
        with _tempfile.TemporaryDirectory() as d:
            paths = write_error_tables(self.table, d, prefix="e")
            tex_files = [p for p in paths if p.endswith(".tex")]
            self.assertEqual(len(tex_files), len(TABLE_PERIODS))
            n = len(self.table["full"]["labels"])
            for path in tex_files:
                with open(path, encoding="utf-8") as fh:
                    body = fh.read()
                errs = _tex_structure_errors(body)
                self.assertEqual(errs, [], f"{_os.path.basename(path)}: {errs}")
                pre = re.search(r"\\begin\{tabular\}\{([^}]*)\}", body).group(1)
                self.assertEqual(sum(1 for c in pre if c == "r"), n,
                                 f"{_os.path.basename(path)}: wrong number of "
                                 f"numeric columns for {n} scenarios")


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
                with open(_os.path.join(d, f"tbl_{period}.tex"), encoding="utf-8") as fh:
                    tex = fh.read()
                self.assertEqual(tex.count("\\\\"), len(SCHEMES) + 2)
                self.assertIn("booktabs", tex)          # the rules need that package
                self.assertTrue(tex.startswith("%"))    # generated-file banner
                self.assertIn("\\begin{tabular}{lrr}", tex)      # 2 scenario columns
                self.assertEqual(tex.count("\\begin{table}"), 1)
                self.assertEqual(tex.count("\\end{table}"), 1)


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
        f, base_f = generate_mobility(cls.N, cls.T, cls.city[1], cls.city[2],
                                      cls.city[4], seed=6)
        init = np.zeros(cls.N); init[0] = 10.0
        LW = params["base_contact_rate"]
        cls.sim = simulate_epidemic_pde(
            cls.T, cls.N, cls.city[1], f, params["prob_transmission_peak"], cls.gt,
            params["max_gen_time"], params["R0_target"], init, LW, LW * 0.30,
            f_calibration=base_f)
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
        excluded; including them would compare against an incomplete baseline.

        Checked behaviourally, on a mask forced to include day 0: grepping the
        source for "2 * ESTIMATOR_WINDOW" would also pass for an off-by-one.
        """
        from unittest import mock
        from mobility_rt import location_errors as _le
        R_hat, ref_own, ref_net, days = _le._prepare(self.sim, self.gt)
        wide = np.ones_like(days)                 # every day "scored"
        with mock.patch.object(_le, "_prepare",
                               return_value=(R_hat, ref_own, ref_net, wide)):
            d = _le.location_diagnostics(self.sim, self.gt)
        # 2W - 1 = 13 is the first day with two complete windows behind it, and the
        # exclusion applies to the cases-rising row ALONE: narrowing the whole block
        # would put this half on a different day set from the network half.
        self.assertEqual(d["n_growth"], len(wide) - (2 * ESTIMATOR_WINDOW - 1))
        self.assertEqual(d["n_days"], len(wide))

    def test_the_day_mask_admits_exactly_the_days_that_qualify(self):
        """The mask's DEFINITION, not a copy of its code.

        The two "recomputed by hand" tests rebuild _prepare's mask line for line, so
        they would reproduce a wrong mask.  This one states the four conditions
        independently and requires every scored day to meet all of them and every
        excluded day to fail at least one.
        """
        from mobility_rt.kernel import R_outward, R_system
        from mobility_rt.location_errors import _prepare
        R_hat, ref_own, ref_net, days = _prepare(self.sim, self.gt)
        Rm = self.sim["R_matrices"]
        T = len(Rm)
        rho = np.array([R_system(Rm[t]) for t in range(T)])
        Ro = np.array([R_outward(Rm[t]) for t in range(T)])
        ref_n = trailing_mean(rho, ESTIMATOR_WINDOW)
        ref_o = np.column_stack([trailing_mean(Ro[:, j], ESTIMATOR_WINDOW)
                                 for j in range(Ro.shape[1])])
        for t in range(T):
            qualifies = (rho[t] > 1.0
                         and np.all(np.isfinite(R_hat[t]))
                         and np.isfinite(ref_n[t]) and ref_n[t] > RHO_FLOOR
                         and np.all(np.isfinite(ref_o[t]))
                         and np.all(ref_o[t] > RHO_FLOOR))
            self.assertEqual(bool(days[t]), bool(qualifies),
                             f"day {t}: mask says {days[t]}, conditions say {qualifies}")
        self.assertGreater(int(days.sum()), 0)
        self.assertLess(int(days.sum()), T, "no day was excluded; mask untested")

    def test_growth_agreement_value_recomputed_independently(self):
        """agree_growth is a manuscript-table row and had no numerical test."""
        from mobility_rt.location_errors import _prepare
        R_hat, _ref_own, _ref_net, days = _prepare(self.sim, self.gt)
        inc = self.sim["incidence"]
        T, N = inc.shape
        days = days.copy()
        days[:2 * ESTIMATOR_WINDOW - 1] = False
        hits = total = 0
        for t in np.where(days)[0]:
            for j in range(N):
                this_week = inc[t - ESTIMATOR_WINDOW + 1:t + 1, j].sum()
                last_week = inc[t - 2 * ESTIMATOR_WINDOW + 1:t - ESTIMATOR_WINDOW + 1, j].sum()
                hits += int((R_hat[t, j] > 1.0) == (this_week > last_week))
                total += 1
        want = 100.0 * hits / total
        self.assertAlmostEqual(location_diagnostics(self.sim, self.gt)["agree_growth"],
                               want, places=10)

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


# ---------------------------------------------------------------------------
# Table SI3: the combined network- and location-level table
# ---------------------------------------------------------------------------

from mobility_rt.combined_table import (
    _LOCATION_ROWS,
    PERIOD,
    compute_combined_table,
    write_combined_table,
)


class TestCombinedManuscriptTable(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.N, cls.T = 6, 260
        params = COVID_PARAMS
        cls.gt = discretise_gamma(params["gen_time_mean"], params["gen_time_sd"],
                                  params["max_gen_time"])
        cls.city = generate_city(cls.N, scenario="lagos", seed=6)
        f, base_f = generate_mobility(cls.N, cls.T, cls.city[1], cls.city[2],
                                      cls.city[4], seed=6)
        init = np.zeros(cls.N); init[0] = 10.0
        LW = params["base_contact_rate"]
        cls.sim = simulate_epidemic_pde(
            cls.T, cls.N, cls.city[1], f, params["prob_transmission_peak"], cls.gt,
            params["max_gen_time"], params["R0_target"], init, LW, LW * 0.30,
            f_calibration=base_f)
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
# Generation time under spatially heterogeneous contact and transmissibility
# ---------------------------------------------------------------------------


def _renewal_gt(p):
    """p(a_E) with the a_E = 0 entry zeroed and renormalised over a_E >= 1: the shape
    every kernel normalisation in the model reduces to."""
    g = np.asarray(p, dtype=float).copy()
    g[0] = 0.0
    return g / g.sum()


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
        """Pairwise, outward, inward, meeting-site and network GTs all equal p/int p."""
        g_univ = _renewal_gt(self.p)
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


class TestCorrectedFigureScenarios(unittest.TestCase):
    """The corrected Fig 2 / Fig 3 / Fig 5 set, its static-mobility twin, and the
    identical set every alternative scenario (A-het, B, B-het, C, C-het) gets."""

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
                                          cls.city[4], seed=9)
        cls.f_static = np.stack([base_f] * cls.T, axis=0)
        init = np.zeros(cls.N); init[0] = 10.0
        LW = params["base_contact_rate"]
        common = dict(prob_peak=params["prob_transmission_peak"], infect_profile=cls.p,
                      max_days=cls.max_days, R0_target=params["R0_target"],
                      initial_infections=init, lambda_within=LW,
                      lambda_between=LW * 0.30, f_calibration=base_f)
        cls.sim = simulate_epidemic_pde(cls.T, cls.N, pops, cls.f, **common)
        cls.sim_s = simulate_epidemic_pde(cls.T, cls.N, pops, cls.f_static, **common)
        cls.sim_h = simulate_epidemic_pde(
            cls.T, cls.N, pops, cls.f,
            contact_mult=contact_multipliers(cls.city[3]),
            infect_mult=infectiousness_multipliers(cls.N, seed=9), **common)
        cls.gt = {"peak": compute_generation_times(
            cls.f[20], cls.sim["S_start"][20], pops,
            params["prob_transmission_peak"], cls.p, cls.max_days,
            cls.sim["lambda_within_scaled"], cls.sim["lambda_between_scaled"])}
        cls.R_ind = estimate_R_independent(cls.sim["incidence"], cls.p, window=7)

    def _args(self, **kw):
        from mobility_rt.plotting.figures_corrected import run_corrected_figures
        base = dict(sim=self.sim, sim_static=self.sim_s, city_data=self.city,
                    f_series=self.f, f_static=self.f_static, R_ind=self.R_ind,
                    R_ind_static=self.R_ind, gt_snaps=self.gt,
                    gt_snaps_static=self.gt, gen_time_pmf=self.p,
                    max_days=self.max_days)
        base.update(kw)
        return run_corrected_figures, base

    def test_writes_exactly_the_two_manuscript_sets(self):
        import os as _os, tempfile
        run, kw = self._args()
        with tempfile.TemporaryDirectory() as d:
            prefixes = run(out_dir=d, **kw)
            self.assertEqual(len(prefixes), 2)
            names = set(_os.listdir(d))
            for pfx in ("fig", "fig_SI_static"):
                for suffix in ("_02_overview.pdf", "_02_overview_surface.pdf",
                               "_03_taxonomy.pdf", "_03_taxonomy_surface.pdf",
                               "_04_spectral.pdf"):
                    self.assertIn(pfx + suffix, names)
            self.assertEqual(len(names), 10)

    def test_each_extra_scenario_gets_the_full_file_set(self):
        import os as _os, tempfile
        run, kw = self._args()
        extra = {"sim": self.sim_h, "city_data": self.city, "f_series": self.f,
                 "R_ind": self.R_ind, "gt_snaps": self.gt,
                 "name": "Dense urban (A-het)", "prefix": "fig_A_het"}
        with tempfile.TemporaryDirectory() as d:
            prefixes = run(out_dir=d, extra_scenarios=[extra], **kw)
            self.assertEqual(len(prefixes), 3)
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




# ---------------------------------------------------------------------------
# Oracle tests: each manuscript quantity against an independent computation
# ---------------------------------------------------------------------------
#
# Each class compares one manuscript quantity with an independent computation
# (brute force, closed form, a hand-worked example or scipy).  New names are
# imported inside the tests so that this module still imports on code that
# predates them, which is how each test was seen to fail before its fix.


def _small_run(contact_mult=None, infect_mult=None, T=60, N=6, seed=3):
    """A short run on the pipeline's conventions (calibrated on base_f)."""
    from mobility_rt.config import BETWEEN_FRACTION
    params = COVID_PARAMS
    max_days = params["max_gen_time"]
    p = discretise_gamma(params["gen_time_mean"], params["gen_time_sd"], max_days)
    city = generate_city(N, scenario="lagos", seed=seed)
    f, base_f = generate_mobility(N, T, city[1], city[2], city[4],
                                  day_variation_sd=0.15, seed=seed)
    init = np.zeros(N)
    init[0] = 10.0
    LW = params["base_contact_rate"]
    sim = simulate_epidemic_pde(T, N, city[1], f, params["prob_transmission_peak"], p,
                                max_days, params["R0_target"], init, LW,
                                LW * BETWEEN_FRACTION, f_calibration=base_f,
                                contact_mult=contact_mult, infect_mult=infect_mult)
    return sim, city, f, base_f, p


class TestStartOfDaySusceptibles(unittest.TestCase):
    """R_matrices[t] and the day-t force of infection use START-of-day S."""

    @classmethod
    def setUpClass(cls):
        cls.runs = {}
        for tag, kw in (("hom", {}), ("het", "het")):
            if kw == "het":
                city = generate_city(6, scenario="lagos", seed=3)
                kw = dict(contact_mult=contact_multipliers(city[3]),
                          infect_mult=infectiousness_multipliers(6, seed=3))
            cls.runs[tag] = (_small_run(**kw), kw)

    def test_R_matrices_use_start_of_day_S(self):
        pp = COVID_PARAMS["prob_transmission_peak"]
        for tag, ((sim, city, f, base_f, p), kw) in self.runs.items():
            self.assertNotIn("susceptibles", sim)
            pops = city[1]
            S0, S1 = sim["S_start"], sim["S_end"]
            np.testing.assert_array_equal(S0[0], pops)
            np.testing.assert_array_equal(S0[1:], S1[:-1])
            lw, lb = sim["lambda_within_scaled"], sim["lambda_between_scaled"]
            c, m = kw.get("contact_mult"), kw.get("infect_mult")
            for t in range(sim["incidence"].shape[0]):
                np.testing.assert_allclose(
                    compute_R_matrix(f[t], S0[t], pops, pp, p, lw, lb, c, m),
                    sim["R_matrices"][t], rtol=1e-12, atol=0,
                    err_msg=f"{tag}: R_matrices[{t}] is not built from S_start[{t}]")
                np.testing.assert_allclose(
                    compute_R_meeting(f[t], S0[t], pops, pp, p, lw, lb, c, m),
                    sim["R_meeting_series"][t], rtol=1e-12, atol=0,
                    err_msg=f"{tag}: R_meeting_series[{t}] is not built from S_start[{t}]")

    def test_effective_susceptibles_align_with_R(self):
        from mobility_rt.simulation import compute_effective_populations_series
        sim, city, f, base_f, p = self.runs["hom"][0]
        S_eff, _ = compute_effective_populations_series(
            f, sim["S_start"], sim["incidence"], p, len(p))
        for t in range(sim["incidence"].shape[0]):
            np.testing.assert_allclose(S_eff[t], f[t].T @ sim["S_start"][t], rtol=1e-14)

    def test_calibration_is_exact_on_the_calibration_matrix(self):
        """ρ(R) = R0 for a fully susceptible population under f_calibration."""
        pp = COVID_PARAMS["prob_transmission_peak"]
        for tag, ((sim, city, f, base_f, p), kw) in self.runs.items():
            R_cal = compute_R_matrix(base_f, city[1], city[1], pp, p,
                                     sim["lambda_within_scaled"],
                                     sim["lambda_between_scaled"],
                                     kw.get("contact_mult"), kw.get("infect_mult"))
            self.assertAlmostEqual(R_system(R_cal), COVID_PARAMS["R0_target"],
                                   places=10, msg=tag)


class TestMeetingLocationEq50(unittest.TestCase):
    """R^l_meeting is Eq 50, Σ_j Σ_k f^{jl} S_j f^{kl} λ^{kl} over a_E >= 1."""

    @classmethod
    def setUpClass(cls):
        rng = np.random.default_rng(0)
        cls.N = 6
        raw = rng.uniform(0.2, 1.0, (cls.N, cls.N))
        cls.f = raw / raw.sum(axis=1, keepdims=True)
        cls.pops = rng.uniform(1e4, 1e5, cls.N)
        cls.S = cls.pops * rng.uniform(0.3, 1.0, cls.N)
        cls.c = np.exp(rng.normal(0, 0.4, cls.N))
        cls.m = np.exp(rng.normal(0, 0.5, cls.N))
        cls.p = discretise_gamma(5.5, 1.8, 20)
        cls.pp, cls.lw, cls.lb = 0.035, 14.0, 4.2

    def test_R_meeting_sums_to_total_R_under_heterogeneity(self):
        M = compute_R_meeting(self.f, self.S, self.pops, self.pp, self.p,
                              self.lw, self.lb, self.c, self.m)
        R = compute_R_matrix(self.f, self.S, self.pops, self.pp, self.p,
                             self.lw, self.lb, self.c, self.m)
        self.assertAlmostEqual(M.sum() / R.sum(), 1.0, places=12)

    def test_R_meeting_matches_eq50_brute_force(self):
        N, f = self.N, self.f
        N_eff = np.array([sum(f[j, l] * self.pops[j] for j in range(N)) for l in range(N)])
        sum_p = self.p[1:].sum()
        want = np.zeros(N)
        for l in range(N):
            for j in range(N):
                for k in range(N):
                    chi = self.c[l] * (self.lw if k == l else self.lb)
                    want[l] += (f[j, l] * self.S[j] * f[k, l] * chi
                                * self.m[k] * self.pp * sum_p / N_eff[l])
        got = compute_R_meeting(f, self.S, self.pops, self.pp, self.p,
                                self.lw, self.lb, self.c, self.m)
        np.testing.assert_allclose(got, want, rtol=1e-12)

    def test_R_meeting_reduces_to_own_infectiousness_without_travel(self):
        """With f = I every contact at l is between residents of l, so m enters as m_l."""
        eye = np.eye(self.N)
        M = compute_R_meeting(eye, self.S, self.pops, self.pp, self.p,
                              self.lw, self.lb, None, self.m)
        M1 = compute_R_meeting(eye, self.S, self.pops, self.pp, self.p,
                               self.lw, self.lb, None, None)
        np.testing.assert_allclose(M, self.m * M1, rtol=1e-12)


class TestNetworkGenerationTimeEq27(unittest.TestCase):
    """g_network is Eqs 27-28, eigenvector-weighted, not the unweighted sum."""

    def test_eigenvector_weighted_gt_is_eqs_27_28_for_a_nonseparable_kernel(self):
        import scipy.linalg
        rng = np.random.default_rng(42)
        N, A = 5, 12
        R = rng.uniform(0.05, 1.0, (N, N))              # code orientation: R[k, j]
        K = np.zeros((A, N, N))
        for k in range(N):
            for j in range(N):
                q = np.zeros(A)
                q[1:] = rng.gamma(1.0 + 3.0 * rng.uniform(), 1.0, A - 1)
                K[:, k, j] = R[k, j] * q / q.sum()        # a different shape per pair
        R_ms = R.T                                         # manuscript NGM: R_ms[j, k] = R^{kj}
        w, vl, vr = scipy.linalg.eig(R_ms, left=True, right=True)
        i = int(np.argmax(w.real))
        rho = float(w[i].real)
        vstar = np.abs(vr[:, i].real)                      # R_ms v* = rho v*  (Eq 25)
        v = np.abs(vl[:, i].real)                          # v^T R_ms = rho v^T (Eq 26)
        K_net = np.zeros(A)
        for j in range(N):
            for k in range(N):
                K_net += v[j] * vstar[k] * K[:, k, j] / (v @ vstar)
        want = K_net / rho
        got = eigenvector_weighted_gt(K, R)
        np.testing.assert_allclose(got, want, rtol=1e-10, atol=1e-15)
        self.assertAlmostEqual(got.sum(), 1.0, places=12)
        self.assertFalse(np.allclose(got, K.sum(axis=(1, 2)) / R.sum(), atol=1e-6),
                         "fixture is too close to separable to tell Eq 27 from the plain sum")

    def test_compute_generation_times_uses_eqs_27_28(self):
        from unittest import mock
        import mobility_rt.generation_time as gt_mod
        rng = np.random.default_rng(1)
        raw = rng.uniform(0.2, 1.0, (4, 4))
        f = raw / raw.sum(axis=1, keepdims=True)
        pops = rng.uniform(1e4, 1e5, 4)
        p = discretise_gamma(5.5, 1.8, 15)
        sentinel = np.full(15, 0.123)
        with mock.patch.object(gt_mod, "eigenvector_weighted_gt", return_value=sentinel) as spy:
            out = gt_mod.compute_generation_times(f, pops, pops, 0.035, p, 15, 14.0, 4.2)
        spy.assert_called_once()
        np.testing.assert_array_equal(out["g_network"], sentinel)

    def test_profile_length_must_equal_max_days(self):
        p = discretise_gamma(5.5, 1.8, 15)
        f = np.eye(3)
        pops = np.full(3, 1e4)
        with self.assertRaises(ValueError):
            compute_generation_times(f, pops, pops, 0.035, p, 20, 14.0, 4.2)


class TestInfectiousnessWeighting(unittest.TestCase):
    """Network aggregates weighted by total infectiousness Λ_j (ref [68])."""

    def test_total_infectiousness_matches_the_forward_model(self):
        from mobility_rt.estimators import total_infectiousness
        rng = np.random.default_rng(4)
        inc = rng.uniform(0, 50, (40, 3))
        p = discretise_gamma(5.5, 1.8, 25)
        want = np.zeros_like(inc)
        for t in range(40):
            for a in range(1, 25):
                if t - a >= 0:
                    want[t] += p[a] * inc[t - a]
        np.testing.assert_allclose(total_infectiousness(inc, p), want, rtol=1e-13)

    def test_hand_computed_two_location_example(self):
        """Weights, by hand, on constant incidence (10 and 30 per day); R_out = (3, 7).

        The profile puts weight 0.4 on lag 1 and 0.2 on lag 2, so Λ_j(t) = 0.6 I_j and
        each window carries Σ I = 7 I_j against Σ Λ = 4.2 I_j.  It deliberately does
        NOT sum to 1: with a normalised profile and constant incidence the window sums
        would satisfy ΣI = ΣΛ, every R̂_j would be exactly (1+ΣI)/(1+ΣI) = 1 under the
        Gamma(1, 1) prior, and the fixture could no longer tell one weighting from
        another.  The final assertion pins that discrimination.
        """
        T = 20
        inc = np.zeros((T, 2))
        inc[:, 0], inc[:, 1] = 10.0, 30.0
        p = np.array([0.0, 0.4, 0.2])
        sim = {"incidence": inc,
               "R_matrices": np.tile(np.array([[1.0, 2.0], [3.0, 4.0]]), (T, 1, 1))}
        city = (None, np.array([1.0, 1.0]), None, ["core", "core"], {})
        s = _compute_naive_suite(sim, city, p)
        self.assertNotIn("ind_iw", s)
        self.assertNotIn("out_iw", s)
        t = 10
        # window sums: Σ I = 7 I_j, Σ Λ = 7 * 0.6 * I_j
        R1 = (R_PRIOR_SHAPE + 70.0) / (R_PRIOR_RATE + 42.0)
        R2 = (R_PRIOR_SHAPE + 210.0) / (R_PRIOR_RATE + 126.0)
        self.assertAlmostEqual(s["ind_Lw"][t], (42.0 * R1 + 126.0 * R2) / 168.0, places=12)
        self.assertAlmostEqual(s["out_Lw"][t], (10.0 * 3.0 + 30.0 * 7.0) / 40.0, places=12)
        self.assertAlmostEqual(s["out_pw"][t], 5.0, places=12)
        # the two locations must actually differ, or no weighting could be told
        # apart from any other on this fixture
        self.assertGreater(abs(R2 - R1), 1e-3)
        self.assertNotAlmostEqual(s["ind_Lw"][t], s["ind_am"][t], places=6)


class TestSpectralHelpers(unittest.TestCase):

    def _positive_matrices(self):
        rng = np.random.default_rng(8)
        mats = [rng.uniform(0.01, 1.0, (6, 6)) for _ in range(10)]
        # positive, with a complex-conjugate pair of non-Perron eigenvalues
        mats.append(np.array([[0.1, 1.0, 0.1], [0.1, 0.1, 1.0], [1.0, 0.1, 0.1]]))
        return mats

    def test_condition_number_matches_scipy(self):
        """κ = ||v|| ||v*|| / |vᵀv*| (Eq 38), independent of eigenvector scaling."""
        import scipy.linalg
        for R in self._positive_matrices():
            w, vl, vr = scipy.linalg.eig(R, left=True, right=True)
            i = int(np.argmax(np.abs(w)))
            a, b = vl[:, i], vr[:, i]
            want = np.linalg.norm(a) * np.linalg.norm(b) / abs(np.vdot(a, b))
            self.assertAlmostEqual(sensitivity_elasticity(R)["condition_number"] / want,
                                   1.0, places=10)

    def test_condition_number_raises_no_warning(self):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            for R in self._positive_matrices():
                sensitivity_elasticity(R)

    def test_infector_and_infectee_elasticities_coincide(self):
        """Σ_j ε^{kj} and Σ_k ε^{kj} are the same vector for any matrix."""
        for R in self._positive_matrices():
            E = sensitivity_elasticity(R)["elasticity"]
            np.testing.assert_allclose(E.sum(axis=1), E.sum(axis=0), rtol=0, atol=1e-12)

    def test_A1_is_the_manuscript_l1_norm(self):
        """Eq 40: A1(n) = ||R_ms^n||_1 and the code's R_mat is R_ms transposed."""
        rng = np.random.default_rng(9)
        R = rng.uniform(0.0, 0.6, (5, 5))
        env = amplification_envelope(R, n_max=8)
        for n in range(9):
            self.assertAlmostEqual(
                env["A1"][n], np.linalg.norm(np.linalg.matrix_power(R.T, n), 1), places=12)


class TestScenarioDecisions(unittest.TestCase):
    """Calibration on base_f, the closed population, and Scenario C."""

    @classmethod
    def setUpClass(cls):
        import contextlib
        import io
        from mobility_rt.scripts.run_all import build_scenarios
        with contextlib.redirect_stdout(io.StringIO()):
            cls.sc = build_scenarios()

    def _runs(self):
        sc = self.sc
        return (("A", sc["sim_A"], sc["city_A"], sc["base_fA"]),
                ("A-static", sc["sim_A_static"], sc["city_A"], sc["base_fA"]),
                ("A-het", sc["sim_A_het"], sc["city_A"], sc["base_fA"]),
                ("B", sc["sim_B"], sc["city_B"], sc["base_fB"]),
                ("B-het", sc["sim_B_het"], sc["city_B"], sc["base_fB"]),
                ("C", sc["sim_C"], sc["city_C"], sc["base_fC"]),
                ("C-het", sc["sim_C_het"], sc["city_C"], sc["base_fC"]))

    def test_every_scenario_is_calibrated_on_its_base_mobility(self):
        pp = self.sc["params"]["prob_transmission_peak"]
        for tag, sim, city, base_f in self._runs():
            R_cal = compute_R_matrix(base_f, city[1], city[1], pp, self.sc["infect_profile"],
                                     sim["lambda_within_scaled"], sim["lambda_between_scaled"],
                                     sim["contact_mult"], sim["infect_mult"])
            self.assertAlmostEqual(R_system(R_cal), self.sc["params"]["R0_target"],
                                   places=10, msg=tag)

    def test_population_is_closed(self):
        """No births or deaths: S_end(t) + cumulative incidence = N on every day."""
        import inspect
        from mobility_rt.simulation import _simulate_heun, _simulate_rk4
        for tag, sim, city, _ in self._runs():
            lhs = sim["S_end"] + np.cumsum(sim["incidence"], axis=0)
            np.testing.assert_allclose(lhs, np.broadcast_to(city[1], lhs.shape),
                                       rtol=1e-12, atol=0, err_msg=tag)
        removed = {"birth_rate", "death_rate", "w_within", "w_between", "stochastic",
                   "seed", "susceptible_depletion"}
        for fn in (simulate_epidemic_pde, _simulate_heun, _simulate_rk4):
            self.assertFalse(set(inspect.signature(fn).parameters) & removed, fn.__name__)

    def test_scenario_C_differs_from_A_only_in_populations_and_hub_power(self):
        sc = self.sc
        f_A, f_C = sc["f_A"], sc["f_C"]
        # same commuting fractions and the same daily noise draws, so the same home
        # fraction on every day; only the destinations of the away fraction differ
        np.testing.assert_allclose(np.diagonal(f_C, axis1=1, axis2=2),
                                   np.diagonal(f_A, axis1=1, axis2=2), rtol=0, atol=1e-12)
        np.testing.assert_array_equal(sc["initial_C"], sc["initial_A"])
        np.testing.assert_array_equal(sc["city_C"][0], sc["city_A"][0])
        np.testing.assert_array_equal(sc["city_C"][2], sc["city_A"][2])
        types_C = sc["city_C"][3]
        self.assertEqual(types_C, ["core"] + ["peripheral"] * 9)
        self.assertEqual(sc["city_C"][4]["node_types"], types_C)
        meta = sc["city_C"][4]
        idx = (meta["hub_idx"], meta["mid_idx"], meta["periph_idx"])
        self.assertEqual(meta["hub_idx"], int(np.argmax(sc["city_C"][1])))
        self.assertEqual(len(set(idx)), 3, f"representative locations not distinct: {idx}")

    def test_hub_representative_locations_rule(self):
        """periph = satellite farthest from the hub; mid = lower-median distance."""
        from mobility_rt.scripts.run_all import hub_representative_locations
        d = np.array([[0.0, 5.0, 9.0, 2.0],
                      [5.0, 0.0, 4.0, 3.0],
                      [9.0, 4.0, 0.0, 7.0],
                      [2.0, 3.0, 7.0, 0.0]])
        # satellites 1, 2, 3 lie 5, 9 and 2 km from the hub
        self.assertEqual(hub_representative_locations(d, 0), (0, 1, 2))
        tied = d.copy()
        tied[3, 0] = tied[0, 3] = 9.0
        with self.assertRaises(ValueError):
            hub_representative_locations(tied, 0)


class TestCoreHelpers(unittest.TestCase):
    """Single sources for quantities the figures used to compute inline (P3.6)."""

    def test_risk_averse_R_is_the_alpha_one_lehmer_mean(self):
        from mobility_rt.transmissibility import risk_averse_R
        rng = np.random.default_rng(10)
        R_out = rng.uniform(0.1, 3.0, (7, 4))
        np.testing.assert_allclose(risk_averse_R(R_out),
                                   lehmer_mean_spectrum(R_out, np.array([1.0]))[0],
                                   rtol=1e-13)
        self.assertAlmostEqual(risk_averse_R(np.array([1.0, 3.0])), 10.0 / 4.0, places=14)
        self.assertTrue(np.isnan(risk_averse_R(np.zeros(3))))

    def test_within_fraction_hand_example(self):
        import warnings
        from mobility_rt.transmissibility import within_fraction
        M = np.zeros((2, 2, 2))
        M[0] = [[3.0, 1.0], [2.0, 4.0]]     # [k infector, j infectee]
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            pi_j, pi_bar = within_fraction(M)
        np.testing.assert_allclose(pi_j[0], [3.0 / 5.0, 4.0 / 5.0])
        self.assertAlmostEqual(pi_bar[0], 7.0 / 10.0)
        self.assertTrue(np.isnan(pi_j[1]).all() and np.isnan(pi_bar[1]))

    def test_renewal_gt(self):
        from mobility_rt.distributions import renewal_gt
        p = discretise_gamma(5.5, 1.8, 25)
        g = renewal_gt(p)
        self.assertEqual(g[0], 0.0)
        self.assertAlmostEqual(g.sum(), 1.0, places=14)
        f = np.full((3, 3), 1.0 / 3.0)
        pops = np.full(3, 1e4)
        snap = compute_generation_times(f, pops, pops, 0.035, p, 25, 14.0, 4.2)
        np.testing.assert_allclose(snap["g_pairwise"][:, 0, 1], g, atol=1e-15)

    def test_phase_days(self):
        from mobility_rt.simulation import phase_days
        inc = np.zeros((100, 2))
        inc[30, 0] = 5.0
        self.assertEqual(phase_days(inc), {"early": 10, "peak": 30, "late": 60})
        inc = np.zeros((100, 2))
        inc[1, 1] = 5.0
        self.assertEqual(phase_days(inc)["early"], 1)
        inc = np.zeros((100, 2))
        inc[90, 1] = 5.0
        self.assertEqual(phase_days(inc)["late"], 99)


class TestFigureFixes(unittest.TestCase):
    """Plotting invariants — labels, panels and statistics — on one small run."""

    @classmethod
    def setUpClass(cls):
        import matplotlib
        matplotlib.use("Agg")
        cls.sim, cls.city, cls.f, cls.base_f, cls.p = _small_run(T=260, N=6)
        cls.suite = _compute_naive_suite(cls.sim, cls.city, cls.p)
        cls.R_ind = estimate_R_independent(cls.sim["incidence"], cls.p, window=7)
        cls.snaps = {"peak": compute_generation_times(
            cls.f[50], cls.sim["S_start"][50], cls.city[1],
            COVID_PARAMS["prob_transmission_peak"], cls.p, len(cls.p),
            cls.sim["lambda_within_scaled"], cls.sim["lambda_between_scaled"])}

    def tearDown(self):
        import matplotlib.pyplot as plt
        plt.close("all")

    @staticmethod
    def _capture(fn, *args, **kwargs):
        """Run a plotting function with savefig and close disabled; return its figures."""
        import matplotlib.pyplot as plt
        from unittest import mock
        before = set(plt.get_fignums())
        with mock.patch.object(plt, "savefig"), mock.patch.object(plt, "close"):
            fn(*args, **kwargs)
        return [plt.figure(n) for n in plt.get_fignums() if n not in before]

    @staticmethod
    def _axis_with_label(fig, letter):
        hits = [ax for ax in fig.axes if any(t.get_text() == letter for t in ax.texts)]
        assert len(hits) == 1, f"expected one axes labelled {letter!r}, found {len(hits)}"
        return hits[0]

    def test_bias_figure_statistics_equal_the_table_cells(self):
        """P4.1: every MRE, bias and day count in a statistics box is its table cell."""
        import matplotlib.pyplot as plt
        from unittest import mock
        from mobility_rt.error_tables import _period_masks, scored_crossing_day
        from mobility_rt.plotting.figures_main import _plot_main_bias_grid
        table = compute_error_table([("One", self.sim, self.city)], self.p)
        for period in ("growth", "decline"):
            self.assertGreater(table[period]["n_days"]["One"], 0,
                               f"fixture has no {period} days")
        all_data = [(self.suite, "a"), (self.suite, "b"), (self.suite, "c")]
        masks = _period_masks(self.suite)
        with mock.patch.object(plt, "savefig"):
            for sch in SCHEMES:
                stats = _plot_main_bias_grid(all_data, sch.key, "unused.pdf")
                row = stats["a"]
                for period in ("growth", "decline"):
                    mre, bias = row["scheme"][period]
                    self.assertAlmostEqual(mre, table[period]["mre"][sch.key]["One"],
                                           places=9, msg=(sch.key, period))
                    self.assertAlmostEqual(bias, table[period]["bias"][sch.key]["One"],
                                           places=9, msg=(sch.key, period))
                    self.assertEqual(row["n_days"][period],
                                     table[period]["n_days"]["One"])
                # crossing lag on the scored days, against rho on day t
                c_s = scored_crossing_day(self.suite[sch.key], masks["full"])
                c_r = scored_crossing_day(self.suite["rho_t"], masks["full"])
                want = None if c_s is None or c_r is None else c_s - c_r
                self.assertEqual(row["scheme"]["threshold_days"], want, sch.key)

    def test_benchmark_column_changes_the_layout_and_nothing_else(self):
        """Dropping the E(t) column must move no number: same statistics, one
        column fewer, panel letters A-F instead of A-I."""
        import matplotlib.pyplot as plt
        from unittest import mock
        from mobility_rt.plotting.figures_main import _plot_main_bias_grid
        all_data = [(self.suite, "a"), (self.suite, "b"), (self.suite, "c")]
        seen = {}
        for flag in (True, False):
            n_axes = []
            with mock.patch.object(
                    plt, "savefig",
                    side_effect=lambda *a, **k: n_axes.append(len(plt.gcf().axes))):
                seen[flag] = _plot_main_bias_grid(all_data, "ind_Rw", "unused.pdf",
                                                  show_benchmark=flag)
            self.assertEqual(n_axes, [9 if flag else 6], f"show_benchmark={flag}")
        self.assertEqual(seen[True], seen[False])
        # the suppressed column's statistics are still returned
        for row in seen[False].values():
            self.assertIn("E", row)
            self.assertIsNotNone(row["E"]["growth"][0])

    def test_si_suite_is_one_untargeted_figure_per_scheme(self):
        """The SI set scores against R(t) only, with no E(t) benchmark column:
        one figure per scheme, six panels, no "_vs_Et" file."""
        import matplotlib.pyplot as plt
        from unittest import mock
        from mobility_rt.plotting.figures_main import plot_estimator_si_suite
        all_data = [(self.suite, "a"), (self.suite, "b"), (self.suite, "c")]
        n_axes = []
        with mock.patch.object(
                plt, "savefig",
                side_effect=lambda *a, **k: n_axes.append(len(plt.gcf().axes))):
            written = plot_estimator_si_suite(all_data, save_prefix="p")
        self.assertEqual(sorted(written),
                         sorted(f"p_naive_{sch.key}_vs_Rt.pdf" for sch in SCHEMES))
        self.assertFalse([f for f in written if f.endswith("_vs_Et.pdf")])
        self.assertEqual(n_axes, [6] * len(SCHEMES))

    def test_bias_suite_writes_one_figure_per_scheme(self):
        """SCHEMES is the single source of truth: a scheme added to the tables must
        appear in the main-text set without anyone editing a second list."""
        import matplotlib.pyplot as plt
        from unittest import mock
        from mobility_rt.plotting.figures_main import plot_main_bias_suite
        all_data = [(self.suite, "a"), (self.suite, "b"), (self.suite, "c")]
        with mock.patch.object(plt, "savefig"):
            written = plot_main_bias_suite(all_data, save_prefix="p",
                                           fname_suffix="sfx")
        self.assertEqual(sorted(written),
                         sorted(f"p_sfx_{sch.key}.pdf" for sch in SCHEMES))

    def test_no_literal_part_tilde_titles(self):
        import inspect
        import mobility_rt.plotting.figures_main as fm
        self.assertNotIn("Part~", inspect.getsource(fm))

    def test_fig3_I_J_compare_with_trailing_means(self):
        """P4.2: the solid curves in Fig 3 I and J are 7-day trailing means."""
        from mobility_rt.plotting.figures_corrected import plot_fig03_taxonomy
        from mobility_rt.geometry import representative_locs
        figs = self._capture(plot_fig03_taxonomy, self.sim, self.city, self.R_ind,
                             self.snaps, self.p, "unused")
        R_mats = self.sim["R_matrices"]
        series = {"I": np.array([R_inward(R) for R in R_mats]),
                  "J": np.array([R_outward(R) for R in R_mats])}
        _, _, _, show_locs, show_lbls = representative_locs(self.city)
        for letter, ser in series.items():
            ax = self._axis_with_label(figs[-1], letter)
            solid = {ln.get_label(): ln for ln in ax.get_lines()
                     if ln.get_linestyle() == "-" and ln.get_label() in show_lbls}
            self.assertEqual(set(solid), set(show_lbls))
            for j, lbl in zip(show_locs, show_lbls):
                x = np.asarray(solid[lbl].get_xdata(), dtype=int)
                np.testing.assert_allclose(solid[lbl].get_ydata(),
                                           trailing_mean(ser[:, j], 7)[x], rtol=1e-12)

    def test_fig3_K_keeps_undefined_T_j_as_nan(self):
        """Undefined T_j reaches the tile and the surface as NaN, not 0."""
        from unittest import mock
        import mobility_rt.plotting.figures_corrected as fc
        T_ser = np.array([type_reproduction_numbers(R) for R in self.sim["R_matrices"]])
        self.assertTrue(np.isnan(T_ser).any(), "fixture has no undefined T_j")
        for surface, name in ((False, "_plot_time_location_tile"),
                              (True, "_plot_time_location_surface")):
            real = getattr(fc, name)
            recorded = {}

            def spy(*args, letter, **kwargs):
                Z = args[1] if name == "_plot_time_location_tile" else args[2]
                recorded[letter] = np.array(Z, dtype=float, copy=True)
                out = real(*args, letter=letter, **kwargs)
                if name == "_plot_time_location_tile":
                    recorded[letter + "_masked"] = int(np.ma.count_masked(out.get_array()))
                return out

            with mock.patch.object(fc, name, side_effect=spy):
                self._capture(fc.plot_fig03_taxonomy, self.sim, self.city, self.R_ind,
                              self.snaps, self.p, "unused", surface_3d=surface)
            np.testing.assert_array_equal(np.isnan(recorded["K"]), np.isnan(T_ser))
            if not surface:
                self.assertEqual(recorded["K_masked"], int(np.isnan(T_ser).sum()))

    def test_surface_draws_every_location(self):
        """Rows of the surface are locations, so none may be skipped."""
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec
        from mobility_rt.plotting.figures_corrected import _plot_time_location_surface
        fig = plt.figure()
        gs = gridspec.GridSpec(1, 1, figure=fig)
        Z = np.random.default_rng(0).uniform(size=(365, 10))
        ax = _plot_time_location_surface(fig, gs[0, 0], Z, 365, 10,
                                         cbar_title="z", letter="X")
        self.assertEqual(ax.collections[0].get_array().size, 9 * 91)

    def test_type_figures_mark_one_on_the_colorbar(self):
        """The dotted threshold line on each T colorbar is at 1, not norm(1)."""
        from mobility_rt.plotting.figures_main import plot_type_repro
        figs = self._capture(plot_type_repro, self.sim, self.sim, self.city, self.city,
                             "unused")
        dotted = [ln for fig in figs for ax in fig.axes
                  if getattr(ax, "_colorbar", None) is not None
                  for ln in ax.get_lines() if ln.get_linestyle() == ":"]
        self.assertEqual(len(dotted), 4)     # two heatmaps, two type-group tiles
        for ln in dotted:
            np.testing.assert_array_equal(np.asarray(ln.get_ydata(), dtype=float), [1.0, 1.0])

    def test_SI26_accepts_an_all_nan_estimate(self):
        """P4.3: no NameError from an unbound colorbar mappable."""
        from mobility_rt.plotting.figures_si import plot_SI_R_comparison
        R_nan = np.full(self.sim["incidence"].shape, np.nan)
        self._capture(plot_SI_R_comparison, self.sim, self.city, R_nan, "unused")


# ---------------------------------------------------------------------------
# Summary printer
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
#  Figures drawn from the three result tables
# ---------------------------------------------------------------------------

class TestTableFigures(unittest.TestCase):
    """The three table figures must depict their tables and nothing else.

    They are pure renderers: they take the dicts the table modules return, assert
    the keys they need, and draw.  These tests pin that contract -- a missing
    scheme, a missing scenario or an impossible quartile ordering must raise
    rather than produce a figure that quietly disagrees with the table beside it.
    """

    @classmethod
    def setUpClass(cls):
        cls.labels = ["One", "Two", "Three"]
        keys = [sch.key for sch in SCHEMES]
        # deliberately spanning three decades, as the real tables do
        cls.mre = {k: {l: 0.2 + 3.0 * i + 7.0 * j
                       for j, l in enumerate(cls.labels)}
                   for i, k in enumerate(keys)}
        cls.bias = {k: {l: (-1.0) ** (i + j) * cls.mre[k][l] * 0.5
                        for j, l in enumerate(cls.labels)}
                    for i, k in enumerate(keys)}
        cls.net = {"growth": {"labels": list(cls.labels),
                              "n_days": {l: 100 + j for j, l in enumerate(cls.labels)},
                              "mre": cls.mre, "bias": cls.bias}}
        cls.diag = {l: {"rank_median": -0.2 + 0.3 * j, "rank_q1": -0.6 + 0.3 * j,
                        "rank_q3": 0.4 + 0.2 * j, "cv_est": 2.0 + j,
                        "cv_true": 6.0 + 12.0 * j, "agree_growth": 99.5 + 0.1 * j,
                        "agree_own": 90.0 - 30.0 * j, "net_median": 2.0 + j,
                        "n_days": 100 + j}
                    for j, l in enumerate(cls.labels)}
        rng = _np_for_fixture.random.default_rng(0)
        cls.disc = {l: {c: {"mre": 2.0 + 8.0 * rng.random(7) + 5.0 * j,
                            "bias": (-1.0) ** j * (1.0 + 9.0 * rng.random(7))}
                        for c in ("own", "net")}
                    for j, l in enumerate(cls.labels)}
        for j, l in enumerate(cls.labels):
            cls.disc[l]["n_days"] = 100 + j
        cls.comb = {
            "labels": list(cls.labels),
            "network": {k: {l: (cls.mre[k][l], cls.bias[k][l]) for l in cls.labels}
                        for k in keys},
            "location": {key: {l: cls.diag[l].get(key, 5.0 + j)
                               for j, l in enumerate(cls.labels)}
                         for key, _s, _p, _t, _n in _LOCATION_ROWS},
            "n_days": {l: 100 + j for j, l in enumerate(cls.labels)},
        }

    def _render(self, fn, *a, **kw):
        with _tempfile.TemporaryDirectory() as d:
            path = fn(*a, save_prefix=_os.path.join(d, "t"), **kw)
            self.assertTrue(_os.path.exists(path) and _os.path.getsize(path) > 0)
            return _os.path.basename(path)

    def test_all_three_figures_render(self):
        from mobility_rt.plotting.figures_tables import (
            plot_estimator_error_figure, plot_location_diagnostics_figure,
            plot_location_discrepancy_figure, plot_manuscript_figure)
        self.assertEqual(self._render(plot_estimator_error_figure, self.net,
                                      "growth"), "t_estimator_error_growth.pdf")
        self.assertEqual(self._render(plot_location_diagnostics_figure, self.diag),
                         "t_location_diagnostics.pdf")
        self.assertEqual(self._render(plot_manuscript_figure, self.comb),
                         "t_manuscript_table.pdf")
        self.assertEqual(self._render(plot_location_discrepancy_figure, self.disc),
                         "t_location_discrepancy.pdf")

    def test_a_missing_scheme_or_scenario_is_fatal(self):
        """A silently dropped row would make the figure contradict the table."""
        import copy
        from mobility_rt.plotting.figures_tables import (
            plot_manuscript_figure, plot_estimator_error_figure)
        bad = copy.deepcopy(self.net)
        del bad["growth"]["mre"][SCHEMES[3].key]
        with self.assertRaises(KeyError):
            plot_estimator_error_figure(bad, "growth")
        bad = copy.deepcopy(self.net)
        del bad["growth"]["mre"][SCHEMES[3].key]["Two"]
        with self.assertRaises(KeyError):
            plot_estimator_error_figure(bad, "growth")
        with self.assertRaises(KeyError):
            plot_estimator_error_figure(self.net, "decline")
        bad = copy.deepcopy(self.comb)
        del bad["location"]["agree_own"]
        with self.assertRaises(KeyError):
            plot_manuscript_figure(bad)

    def test_unordered_quartiles_are_fatal(self):
        """q1 <= median <= q3 is what the interquartile bar draws; violate it and
        the bar would be drawn upside down rather than flagged."""
        import copy
        from mobility_rt.plotting.figures_tables import \
            plot_location_diagnostics_figure
        bad = copy.deepcopy(self.diag)
        bad["Two"]["rank_q1"] = bad["Two"]["rank_q3"] + 0.1
        with self.assertRaises(ValueError):
            plot_location_diagnostics_figure(bad)

    def test_the_bold_cell_is_the_scheme_the_tex_table_bolds(self):
        """Figure and .tex must name the SAME best scheme, ties included.

        The previous version of this test compared a column minimum with itself,
        which holds for every matrix and tested numpy rather than the code.  What
        matters is that _matrix is oriented (scheme, scenario) and that the figure's
        tie-break agrees with error_tables.write_error_tables: both take the first
        minimum in SCHEMES order.  The tie below is between "ind_am" and "ind_Lw",
        which sort one way by SCHEMES order and the other by ASCII, so a writer that
        sorted the key strings would fail here.
        """
        import numpy as _np
        from mobility_rt.plotting.figures_tables import _matrix
        keys = [sch.key for sch in SCHEMES]
        self.assertEqual(keys[1:3], ["ind_am", "ind_Lw"],
                         "this test relies on the scheme order it names")
        mre = {k: dict(v) for k, v in self.mre.items()}
        mre[keys[1]][self.labels[0]] = mre[keys[2]][self.labels[0]] = 0.01
        m = _matrix(mre, keys, self.labels)
        self.assertEqual(m.shape, (len(keys), len(self.labels)),
                         "_matrix must be (scheme, scenario)")
        for i, k in enumerate(keys):            # orientation, not just shape
            for j, lab in enumerate(self.labels):
                self.assertAlmostEqual(m[i, j], mre[k][lab], places=12)

        best = _np.zeros_like(m, dtype=bool)     # the figure's rule
        best[m.argmin(axis=0), _np.arange(m.shape[1])] = True
        self.assertTrue(_np.all(best.sum(axis=0) == 1), "one bold cell per scenario")
        for j, lab in enumerate(self.labels):    # the .tex's rule
            tex_best = min((mre[k][lab], i, k) for i, k in enumerate(keys))[2]
            self.assertEqual(keys[int(_np.argmax(best[:, j]))], tex_best,
                             f"{lab}: figure and .tex disagree on the best scheme")

    def test_family_blocks_are_contiguous_runs(self):
        """The block rules are drawn from these runs; a non-contiguous SCHEMES
        order would put a rule in the wrong place instead of erroring."""
        from mobility_rt.plotting.figures_tables import _blocks, _family
        keys = [sch.key for sch in SCHEMES]
        runs = _blocks(keys)
        self.assertEqual([f for _, _, f in runs], ["agg", "ind", "out"])
        self.assertEqual(runs[0][0], 0)
        self.assertEqual(runs[-1][1], len(keys))
        for lo, hi, fam in runs:
            self.assertTrue(all(_family(k) == fam for k in keys[lo:hi]))

    def test_the_log_colour_scale_brackets_the_data_in_whole_decades(self):
        import numpy as _np
        from mobility_rt.plotting.figures_tables import _decade_limits
        vmin, vmax = _decade_limits(_np.array([[0.13, 4.0, 48.27]]))
        self.assertAlmostEqual(vmin, 0.1)
        self.assertAlmostEqual(vmax, 100.0)
        self.assertLess(vmin, 0.13)
        self.assertGreater(vmax, 48.27)
        with self.assertRaises(ValueError):
            _decade_limits(_np.array([[0.0, -1.0]]))

    def test_the_figures_depict_every_row_of_their_tables(self):
        """A row added to a table but not to its figure would be silently lost."""
        from mobility_rt.location_errors import _DIAG_ROWS, COMPARATORS
        from mobility_rt.plotting.figures_tables import _DISC_COMPARATORS
        self.assertEqual([k for k, _c, _l, _o in _DISC_COMPARATORS],
                         [k for k, _p, _t in COMPARATORS])
        from mobility_rt.plotting.figures_tables import (_COMBINED_LOC_KEYS,
                                                         _DIAG_KEYS)
        self.assertEqual(set(_DIAG_KEYS),
                         {k for k, _p, _t in _DIAG_ROWS} | {"n_days"})
        self.assertEqual(set(_COMBINED_LOC_KEYS),
                         {k for k, _s, _p, _t, _n in _LOCATION_ROWS})

    def test_impossible_values_raise_rather_than_draw(self):
        """A log panel cannot show zero, agreement cannot exceed 100 per cent and
        a rank correlation cannot leave [-1, 1]; each must be fatal."""
        import copy
        from mobility_rt.plotting.figures_tables import (
            plot_location_diagnostics_figure, plot_manuscript_figure)
        for key, value in (("cv_est", 0.0), ("agree_own", 101.0),
                           ("rank_median", 1.4)):
            bad = copy.deepcopy(self.diag)
            bad["Two"][key] = value
            if key == "rank_median":          # keep the quartile ordering valid
                bad["Two"]["rank_q3"] = 1.5
            with self.assertRaises(ValueError, msg=f"{key}={value} was accepted"):
                plot_location_diagnostics_figure(bad)
        bad = copy.deepcopy(self.comb)
        bad["location"]["own"]["Two"] = -1.0
        with self.assertRaises(ValueError):
            plot_manuscript_figure(bad)

    def test_a_log_tick_is_never_labelled_with_a_rounded_value(self):
        """0.05 must not be labelled 0.1."""
        import matplotlib.pyplot as _plt
        from mobility_rt.plotting.figures_tables import _log_ticks
        fig, ax = _plt.subplots()
        try:
            ax.set_yscale("log")
            ax.set_ylim(0.04, 60)
            _log_ticks(ax)
            fig.canvas.draw()
            texts = [t.get_text() for t in ax.get_yticklabels() if t.get_text()]
            for label, tick in zip(texts, ax.get_yticks()):
                self.assertAlmostEqual(float(label), tick, places=10)
        finally:
            _plt.close(fig)

    def test_the_discrepancy_panels_draw_the_statistics_the_table_prints(self):
        """The figure reduces the per-location arrays itself; it must reduce them
        exactly as location_errors._ROWS does, or figure and table disagree."""
        import matplotlib.pyplot as _plt
        from mobility_rt.location_errors import _ROWS
        from mobility_rt.plotting.figures_tables import (_DISC_COMPARATORS,
                                                         _panel_discrepancy)
        reduce = {k: fn for k, _p, _t, fn in _ROWS}
        for metric in ("mre", "bias"):
            fig, ax = _plt.subplots()
            try:
                _panel_discrepancy(ax, self.labels, self.disc, metric)
                # three artists per (comparator, scenario): IQR bar, max, median
                lines = [ln for ln in ax.get_lines()]
                self.assertEqual(len(lines),
                                 3 * len(_DISC_COMPARATORS) * len(self.labels))
                k = 0
                for ckey, _c, _l, _o in _DISC_COMPARATORS:
                    for lab in self.labels:
                        v = self.disc[lab][ckey][metric]
                        bar, mx, med = lines[k], lines[k + 1], lines[k + 2]
                        k += 3
                        self.assertAlmostEqual(min(bar.get_ydata()),
                                               reduce["q1"](v), places=12)
                        self.assertAlmostEqual(max(bar.get_ydata()),
                                               reduce["q3"](v), places=12)
                        self.assertAlmostEqual(mx.get_ydata()[0],
                                               reduce["max"](v), places=12)
                        self.assertAlmostEqual(med.get_ydata()[0],
                                               reduce["median"](v), places=12)
            finally:
                _plt.close(fig)

    def test_no_table_figure_carries_a_title(self):
        """These follow the main-text convention: panel letters and axis labels
        only, everything else in the caption."""
        import matplotlib.pyplot as _plt
        from mobility_rt.plotting import figures_tables as _ft
        src = open(_ft.__file__).read()
        for banned in ("set_title(", "suptitle("):
            self.assertNotIn(banned, src, f"{banned} is back in figures_tables")

    def test_rendering_restores_the_caller_style(self):
        """set_modern_style rewrites global rcParams; an rc_context must undo it."""
        import matplotlib.pyplot as _plt
        from mobility_rt.plotting.figures_tables import plot_estimator_error_figure
        _plt.rcParams["font.size"] = 17.0
        try:
            self._render(plot_estimator_error_figure, self.net, "growth")
            self.assertEqual(_plt.rcParams["font.size"], 17.0)
        finally:
            _plt.rcdefaults()


# ---------------------------------------------------------------------------
# The daily solver against a refined grid (the numbers in simulate_epidemic_pde's
# docstring)
# ---------------------------------------------------------------------------

class TestRefinedGridLimit(unittest.TestCase):
    """Δt = Δa_E = 1 d against a Δt = Δa_E = 1/m d reference, Scenario A geometry.

    The reference is an INDEPENDENT solver written here, not the module's: it steps
    the same age-structured renewal PDE on a finer grid, binning the continuous
    Gamma(5.5, 1.8) profile at the refined step, so it shares no code path with
    simulate_epidemic_pde beyond the calibration and the mobility kernel.  This
    pins the accuracy claims in that function's docstring; without it they are
    assertions no one can check.
    """

    @classmethod
    def setUpClass(cls):
        from mobility_rt.config import BETWEEN_FRACTION, DAY_VARIATION_SD
        cls.params = COVID_PARAMS
        cls.N, cls.T, cls.SEED = 10, 365, 42
        city = generate_city(cls.N, scenario="lagos", seed=cls.SEED)
        cls.pops = city[1]
        cls.f, cls.base_f = generate_mobility(
            cls.N, cls.T, cls.pops, city[2], city[4],
            day_variation_sd=DAY_VARIATION_SD["A"], seed=cls.SEED)
        cls.init = np.zeros(cls.N)
        cls.init[0] = 10.0
        cls.LW = cls.params["base_contact_rate"]
        cls.LB = cls.LW * BETWEEN_FRACTION

    def _refined(self, R0, m):
        """Attack rate (%) and peak daily incidence at Δt = Δa_E = 1/m day."""
        from mobility_rt.simulation import _calibrate
        p, h = self.params, 1.0 / m
        shape = (p["gen_time_mean"] / p["gen_time_sd"]) ** 2
        scale = p["gen_time_sd"] ** 2 / p["gen_time_mean"]
        n_age = int(p["max_gen_time"] * m)
        p_h = np.diff(_gamma.cdf(np.arange(n_age + 1) * h, a=shape, scale=scale))
        p_h = np.concatenate([[0.0], p_h])          # no transmission at age 0
        lw, lb, _, _ = _calibrate(self.base_f, self.pops,
                                  p["prob_transmission_peak"], p_h, R0,
                                  self.LW, self.LB, None, None)
        S = self.pops.astype(float).copy()
        E = np.zeros((self.N, len(p_h)))
        E[:, 0] = self.init
        S -= self.init
        daily = np.zeros((self.T, self.N))
        daily[0] = self.init
        for t in range(1, self.T):
            base_K, _, _, _, _ = _kernel_base(self.f[min(t, len(self.f) - 1)],
                                              self.pops, lw, lb, None)
            for _ in range(m):
                E[:, 1:] = E[:, :-1]
                # φ is a per-DAY rate: p_h[i]*E[:, i] are counts, /h makes a rate
                wE = p["prob_transmission_peak"] * (E[:, 1:] @ p_h[1:]) / h
                phi = np.maximum((base_K * wE[:, None]).sum(axis=0), 0.0)
                new = S * (-np.expm1(-phi * h))
                S -= new
                E[:, 0] = new
                daily[t] += new
        return (100.0 * daily.sum() / self.pops.sum(), daily.sum(axis=1).max())

    def _daily(self, R0, update):
        prof = discretise_gamma(self.params["gen_time_mean"],
                                self.params["gen_time_sd"],
                                self.params["max_gen_time"])
        sim = simulate_epidemic_pde(
            self.T, self.N, self.pops, self.f,
            self.params["prob_transmission_peak"], prof,
            self.params["max_gen_time"], R0, self.init, self.LW, self.LB,
            f_calibration=self.base_f, update=update)
        inc = sim["incidence"]
        return (100.0 * inc.sum() / self.pops.sum(), inc.sum(axis=1).max())

    def test_refined_grid_limit(self):
        """The docstring's table: exponential tracks the refined limit, Euler does not."""
        for R0, euler_attack_min, euler_peak_min in ((1.5, 0.4, 2.0), (3.0, 0.9, 6.0)):
            ar_ref, pk_ref = self._refined(R0, 64)
            ar_16, _ = self._refined(R0, 16)
            self.assertLess(abs(ar_16 - ar_ref), 0.02,
                            f"R0={R0}: refined grid is not itself converged")

            ar_exp, pk_exp = self._daily(R0, "exponential")
            self.assertLess(abs(ar_exp - ar_ref), 0.05,
                            f"R0={R0}: exponential attack rate {ar_exp:.4f} is not "
                            f"within 0.05 pp of the refined limit {ar_ref:.4f}")

            ar_eul, pk_eul = self._daily(R0, "euler")
            self.assertGreater(ar_eul - ar_ref, euler_attack_min,
                               f"R0={R0}: Euler no longer overstates the attack rate "
                               "as the docstring says")
            self.assertGreater(100.0 * (pk_eul - pk_ref) / pk_ref, euler_peak_min,
                               f"R0={R0}: Euler no longer overstates the peak")
            # and the default is the accurate one
            self.assertLess(abs(ar_exp - ar_ref), abs(ar_eul - ar_ref))


# ---------------------------------------------------------------------------
# The pipeline's own entry points, and the alternative solvers behind Fig SI4C
# ---------------------------------------------------------------------------

class TestPipelineWrappersAndAltSolvers(unittest.TestCase):
    """run_all calls the build_* wrappers and Fig SI4C draws _simulate_heun/_rk4.

    Both were reachable only through a full pipeline run, so a break in the verbose
    printing or in either solver would have surfaced as a broken figure rather than
    as a failing test.
    """

    @classmethod
    def setUpClass(cls):
        params = COVID_PARAMS
        cls.N, cls.T = 6, 200
        cls.gt = discretise_gamma(params["gen_time_mean"], params["gen_time_sd"],
                                  params["max_gen_time"])
        city = generate_city(cls.N, scenario="lagos", seed=11)
        cls.city = city
        cls.pops = city[1]
        cls.f, cls.base_f = generate_mobility(cls.N, cls.T, cls.pops, city[2],
                                              city[4], seed=11)
        cls.init = np.zeros(cls.N)
        cls.init[0] = 10.0
        cls.LW = params["base_contact_rate"]
        cls.LB = cls.LW * 0.30
        cls.common = dict(prob_peak=params["prob_transmission_peak"],
                          max_days=params["max_gen_time"],
                          R0_target=params["R0_target"])
        cls.sim = simulate_epidemic_pde(
            cls.T, cls.N, cls.pops, cls.f, cls.common["prob_peak"], cls.gt,
            cls.common["max_days"], cls.common["R0_target"], cls.init,
            cls.LW, cls.LB, f_calibration=cls.base_f)
        cls.scen = [("One", cls.sim, city), ("Two", cls.sim, city)]

    def test_every_build_wrapper_runs_and_writes_its_files(self):
        from mobility_rt.combined_table import build_combined_table
        from mobility_rt.error_tables import build_estimator_error_tables
        from mobility_rt.location_errors import (build_diagnostics_table,
                                                 build_location_table)
        with _tempfile.TemporaryDirectory() as d:
            err, err_paths = build_estimator_error_tables(self.scen, self.gt, d,
                                                          prefix="e")
            loc, loc_paths = build_location_table(self.scen, self.gt, d, prefix="l")
            dia, dia_paths = build_diagnostics_table(self.scen, self.gt, d, prefix="g")
            comb, comb_paths = build_combined_table(self.scen, self.gt, d, prefix="m",
                                                    net_table=err, loc_table=loc,
                                                    diag_table=dia)
            for paths in (err_paths, loc_paths, dia_paths, comb_paths):
                for p in paths:
                    self.assertTrue(_os.path.getsize(p) > 0, f"{p} is empty")
            self.assertEqual(comb["labels"], ["One", "Two"])

    def test_the_alternative_solvers_track_the_exponential_update(self):
        """Heun and RK4 approximate the same within-day depletion as the default, so
        they must land far closer to it than forward Euler does — the claim Fig SI4C's
        docstring makes for panel C."""
        from mobility_rt.simulation import _simulate_heun, _simulate_rk4
        kw = dict(f_calibration=self.base_f)
        ref = self.sim["incidence"].sum()
        eul = simulate_epidemic_pde(
            self.T, self.N, self.pops, self.f, self.common["prob_peak"], self.gt,
            self.common["max_days"], self.common["R0_target"], self.init,
            self.LW, self.LB, update="euler", **kw)["incidence"].sum()
        d_eul = abs(eul - ref) / ref
        for name, fn in (("heun", _simulate_heun), ("rk4", _simulate_rk4)):
            got = fn(self.T, self.N, self.pops, self.f, self.common["prob_peak"],
                     self.gt, self.common["max_days"], self.common["R0_target"],
                     self.init, self.LW, self.LB, **kw)["incidence"].sum()
            self.assertTrue(np.isfinite(got) and got > 0, f"{name} produced nothing")
            self.assertLess(abs(got - ref) / ref, d_eul,
                            f"{name} is no closer to the exponential update than "
                            "forward Euler is")


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
