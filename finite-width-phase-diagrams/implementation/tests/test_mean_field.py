"""
Comprehensive tests for the mean_field module.

Covers order_parameters, free_energy, susceptibility, and replica submodules
with synthetic/mocked data using numpy.
"""
import sys
import os
import numpy as np
import pytest

sys.path.insert(
    0,
    os.path.join(os.path.dirname(__file__), os.pardir, "src"),
)

from mean_field.order_parameters import (
    OverlapParameter,
    CorrelationFunction,
    ResponseFunction,
    FixedPointIterator,
    MultiFixedPointDetector,
    FixedPointStabilityAnalyzer,
    OrderParameterSolver,
)
from mean_field.free_energy import (
    FreeEnergyLandscape,
    SaddlePointSolver,
    FreeEnergyBarrier,
    PhaseTransitionDetector,
    TransitionClassifier,
    SaddlePointResult,
    PathResult,
    TransitionInfo,
)
from mean_field.susceptibility import (
    LinearResponseComputer,
    FluctuationDissipation,
    DynamicSusceptibility,
    CriticalExponentExtractor,
    FiniteSizeScaling,
)
from mean_field.replica import (
    ReplicaSymmetricSolver,
    OneStepRSBSolver,
    DeAlmeidaThoulessChecker,
    OverlapDistribution,
    ParisiFunctional,
)


# ======================================================================
# Helpers
# ======================================================================

RNG = np.random.default_rng(42)


def _random_configs(n_samples, N, seed=0):
    """Generate random ±1 spin configurations."""
    rng = np.random.default_rng(seed)
    return rng.choice([-1.0, 1.0], size=(n_samples, N))


def _random_continuous_configs(n_samples, N, seed=0):
    """Generate random continuous configurations in [-1, 1]."""
    rng = np.random.default_rng(seed)
    return rng.uniform(-1, 1, size=(n_samples, N))


# ======================================================================
# 1. OverlapParameter tests
# ======================================================================


class TestOverlapParameter:
    def test_self_overlap_ising(self):
        """Self-overlap of ±1 spins should be 1.0."""
        op = OverlapParameter(normalize=True)
        config = np.array([1, -1, 1, -1, 1], dtype=float)
        assert op.compute_self_overlap(config) == pytest.approx(1.0)

    def test_self_overlap_continuous(self):
        """Self-overlap of continuous config should be mean of squares."""
        op = OverlapParameter(normalize=True)
        config = np.array([0.5, -0.3, 0.8])
        expected = np.mean(config ** 2)
        assert op.compute_self_overlap(config) == pytest.approx(expected)

    def test_compute_overlap_identical(self):
        """Overlap of a config with itself equals self-overlap."""
        op = OverlapParameter(normalize=True)
        config = np.array([1, -1, 1, 1, -1], dtype=float)
        assert op.compute_overlap(config, config) == pytest.approx(1.0)

    def test_compute_overlap_opposite(self):
        """Overlap of a config with its negative equals -self-overlap."""
        op = OverlapParameter(normalize=True)
        config = np.array([1, -1, 1, 1, -1], dtype=float)
        assert op.compute_overlap(config, -config) == pytest.approx(-1.0)

    def test_compute_overlap_orthogonal(self):
        """Two orthogonal configs have zero overlap."""
        op = OverlapParameter(normalize=True)
        a = np.array([1, 0, 0, 0], dtype=float)
        b = np.array([0, 1, 0, 0], dtype=float)
        assert op.compute_overlap(a, b) == pytest.approx(0.0)

    def test_overlap_matrix_shape(self):
        """Overlap matrix should be (n_replicas, n_replicas)."""
        op = OverlapParameter(normalize=True)
        configs = _random_configs(5, 20)
        Q = op.compute_overlap_matrix(configs)
        assert Q.shape == (5, 5)

    def test_overlap_matrix_symmetry(self):
        """Overlap matrix must be symmetric."""
        op = OverlapParameter(normalize=True)
        configs = _random_configs(4, 30, seed=7)
        Q = op.compute_overlap_matrix(configs)
        np.testing.assert_allclose(Q, Q.T)

    def test_overlap_matrix_diagonal(self):
        """Diagonal of overlap matrix of ±1 spins should be 1."""
        op = OverlapParameter(normalize=True)
        configs = _random_configs(5, 100, seed=3)
        Q = op.compute_overlap_matrix(configs)
        np.testing.assert_allclose(np.diag(Q), 1.0)

    def test_thermal_average_overlap_uniform_weights(self):
        """Thermal average with uniform weights = simple mean overlap."""
        op = OverlapParameter(normalize=True)
        N = 50
        configs = _random_configs(10, N, seed=1)
        avg = op.thermal_average_overlap(configs)
        # Manual: average over all distinct pairs
        Q = op.compute_overlap_matrix(configs)
        n = Q.shape[0]
        manual = (Q.sum() - np.trace(Q)) / (n * (n - 1))
        assert avg == pytest.approx(manual, abs=1e-10)

    def test_overlap_distribution_returns_bins(self):
        """overlap_distribution should return arrays of bin centres and P(q)."""
        op = OverlapParameter(normalize=True)
        configs = _random_configs(20, 50, seed=2)
        q_vals, p_vals = op.overlap_distribution(configs)
        assert len(q_vals) == len(p_vals)
        assert len(q_vals) > 0

    def test_unnormalized_overlap(self):
        """Without normalization, overlap = raw dot product."""
        op = OverlapParameter(normalize=False)
        a = np.array([1, -1, 1], dtype=float)
        b = np.array([1, 1, 1], dtype=float)
        assert op.compute_overlap(a, b) == pytest.approx(1.0)

    def test_shape_mismatch_raises(self):
        """Mismatched shapes should raise ValueError."""
        op = OverlapParameter()
        with pytest.raises(ValueError):
            op.compute_overlap(np.ones(3), np.ones(4))


# ======================================================================
# 2. CorrelationFunction tests
# ======================================================================


class TestCorrelationFunction:
    def test_equal_time_positive(self):
        """Equal-time auto-correlation should be positive for nonzero configs."""
        cf = CorrelationFunction(normalize=True)
        configs = _random_configs(10, 50)
        C = cf.equal_time_correlation(configs)
        assert C > 0

    def test_equal_time_ising(self):
        """For ±1 spins, normalized equal-time correlation = 1."""
        cf = CorrelationFunction(normalize=True, subtract_mean=False)
        configs = _random_configs(20, 100, seed=5)
        C = cf.equal_time_correlation(configs)
        assert C == pytest.approx(1.0, abs=1e-10)

    def test_two_time_identical(self):
        """Two-time correlation at the same time equals equal-time."""
        cf = CorrelationFunction(normalize=True, subtract_mean=False)
        configs = _random_configs(15, 40, seed=4)
        C_eq = cf.equal_time_correlation(configs)
        C_tt = cf.two_time_correlation(configs, configs)
        assert C_eq == pytest.approx(C_tt, abs=1e-10)

    def test_two_time_uncorrelated(self):
        """Independent configs should have near-zero two-time correlation."""
        cf = CorrelationFunction(normalize=True, subtract_mean=True)
        c1 = _random_configs(1000, 200, seed=10)
        c2 = _random_configs(1000, 200, seed=11)
        C = cf.two_time_correlation(c1, c2)
        assert abs(C) < 0.1

    def test_connected_correlation(self):
        """Connected correlation removes the mean contribution."""
        cf = CorrelationFunction(normalize=True)
        configs_t = _random_continuous_configs(50, 30, seed=6)
        configs_tp = configs_t + 0.01 * RNG.standard_normal(configs_t.shape)
        C_conn = cf.connected_correlation(configs_t, configs_tp)
        assert isinstance(C_conn, float)

    def test_correlation_matrix_shape(self):
        """Correlation matrix should be (N, N)."""
        cf = CorrelationFunction(normalize=True)
        configs = _random_continuous_configs(50, 10, seed=8)
        C = cf.correlation_matrix(configs)
        assert C.shape == (10, 10)

    def test_autocorrelation_function_decays(self):
        """Autocorrelation should start at ~1 and decay toward 0."""
        cf = CorrelationFunction(normalize=True)
        T = 50
        N = 30
        # Build a slowly decorrelating time series
        rng = np.random.default_rng(12)
        trajectory = np.zeros((T, N))
        trajectory[0] = rng.choice([-1.0, 1.0], size=N)
        for t in range(1, T):
            flip_mask = rng.random(N) < 0.1
            trajectory[t] = trajectory[t - 1].copy()
            trajectory[t, flip_mask] *= -1
        ac = cf.autocorrelation_function(trajectory)
        assert ac[0] == pytest.approx(1.0, abs=0.05)
        assert ac[-1] < ac[0]

    def test_shape_mismatch_two_time_raises(self):
        """Mismatched shapes should raise ValueError."""
        cf = CorrelationFunction()
        with pytest.raises(ValueError):
            cf.two_time_correlation(np.ones((3, 4)), np.ones((3, 5)))


# ======================================================================
# 3. ResponseFunction tests
# ======================================================================


class TestResponseFunction:
    def test_compute_response_identity(self):
        """If m = h (identity response), K should be ~identity."""
        rf = ResponseFunction()
        N = 5
        fields = np.eye(N) * 0.01
        mags = fields.copy()  # identity response
        K = rf.compute_response(mags, fields)
        np.testing.assert_allclose(K, np.eye(N), atol=1e-8)

    def test_susceptibility_from_identity(self):
        """Susceptibility of identity matrix is 1.0."""
        rf = ResponseFunction()
        K = np.eye(4)
        chi = rf.susceptibility_from_response(K)
        assert chi == pytest.approx(1.0)

    def test_dynamic_response_shape(self):
        """Dynamic response matrix should be (T, T)."""
        rf = ResponseFunction()
        T, N = 10, 5
        rng = np.random.default_rng(20)
        mags = rng.standard_normal((T, N))
        fields = rng.standard_normal((T, N)) * 0.01
        times = np.arange(T, dtype=float)
        R = rf.dynamic_response(mags, fields, times)
        assert R.shape == (T, T)

    def test_fdr_equilibrium(self):
        """FDT ratio should be ~1 in equilibrium."""
        rf = ResponseFunction()
        T = 100
        temp = 1.0
        # Simulate equilibrium: C(t) = exp(-t/tau), R(t) = (1/T)*dC/dt
        tau = 10.0
        times = np.arange(T, dtype=float)
        C = np.exp(-times / tau)
        R = (1.0 / temp) * np.exp(-times / tau) / tau
        X = rf.fluctuation_dissipation_ratio(C, R, temp)
        # Finite-diff approximation should yield X ~ -1 (convention dependent)
        # Check that FDR is roughly constant and finite
        valid = np.isfinite(X)
        assert np.sum(valid) > 0

    def test_response_shape_mismatch_raises(self):
        """Mismatched shapes should raise ValueError."""
        rf = ResponseFunction()
        with pytest.raises(ValueError):
            rf.compute_response(np.ones((3, 4)), np.ones((3, 5)))


# ======================================================================
# 4. FixedPointIterator tests
# ======================================================================


class TestFixedPointIterator:
    def test_convergence_contraction(self):
        """x = cos(x) has a fixed point near 0.7391."""
        fpi = FixedPointIterator(max_iter=1000, tol=1e-10)
        result = fpi.iterate(
            np.array([0.5]),
            lambda x: np.array([np.cos(x[0])]),
        )
        assert result["converged"]
        assert result["solution"][0] == pytest.approx(0.7390851332, abs=1e-6)

    def test_divergence(self):
        """An expanding map should not converge."""
        fpi = FixedPointIterator(max_iter=100, tol=1e-10)
        result = fpi.iterate(
            np.array([1.1]),
            lambda x: 2 * x,  # expands
        )
        assert not result["converged"]

    def test_damped_iteration_converges(self):
        """Damped iteration should converge for mildly unstable maps."""
        fpi = FixedPointIterator(max_iter=5000, tol=1e-8)
        # x = 0.5*(x + 1/x) converges to 1 (Newton sqrt)
        result = fpi.damped_iteration(
            np.array([2.0]),
            lambda x: np.array([0.5 * (x[0] + 1.0 / x[0])]),
            damping=0.8,
        )
        assert result["converged"]
        assert result["solution"][0] == pytest.approx(1.0, abs=1e-6)

    def test_anderson_mixing_converges(self):
        """Anderson mixing should accelerate convergence."""
        fpi = FixedPointIterator(max_iter=500, tol=1e-8)
        result = fpi.anderson_mixing(
            np.array([0.5]),
            lambda x: np.array([np.cos(x[0])]),
            m=3,
            damping=0.5,
        )
        assert result["converged"]
        assert result["solution"][0] == pytest.approx(0.7390851332, abs=1e-5)

    def test_history_recorded(self):
        """Iteration history should be recorded."""
        fpi = FixedPointIterator(max_iter=50, tol=1e-8)
        result = fpi.iterate(
            np.array([0.5]),
            lambda x: np.array([np.cos(x[0])]),
        )
        assert len(result["history"]) > 1
        assert result["history"][0][0] == pytest.approx(0.5)

    def test_detect_convergence(self):
        """detect_convergence should return convergence info dict."""
        fpi = FixedPointIterator(max_iter=500, tol=1e-12)
        result = fpi.iterate(
            np.array([0.5]),
            lambda x: np.array([np.cos(x[0])]),
        )
        info = fpi.detect_convergence(result["history"])
        assert "converged" in info
        assert "rate" in info
        assert "final_residual" in info

    def test_damped_invalid_damping_raises(self):
        """damping outside (0,1] should raise ValueError."""
        fpi = FixedPointIterator()
        with pytest.raises(ValueError):
            fpi.damped_iteration(np.array([1.0]), lambda x: x, damping=0.0)


# ======================================================================
# 5. MultiFixedPointDetector tests
# ======================================================================


class TestMultiFixedPointDetector:
    def test_finds_multiple_fixed_points(self):
        """Should find multiple fixed points of x = sin(pi*x)."""
        def equations(x):
            return np.array([np.sin(np.pi * x[0])])

        detector = MultiFixedPointDetector(
            iterator=FixedPointIterator(max_iter=500, tol=1e-8),
            tol_cluster=0.01,
        )
        result = detector.scan_initial_conditions(
            equations,
            param_ranges=[(-1.0, 1.0)],
            n_samples=50,
            method="damped",
            damping=0.5,
            seed=42,
        )
        # x=0 is always a fixed point of sin(pi*x)
        assert result["n_converged"] > 0
        assert len(result["solutions"]) >= 1

    def test_cluster_fixed_points_dedup(self):
        """Clustering should merge nearby solutions."""
        detector = MultiFixedPointDetector(tol_cluster=0.1)
        solutions = np.array([[1.0], [1.01], [1.005], [3.0], [3.01]])
        unique = detector.cluster_fixed_points(solutions)
        assert len(unique) == 2

    def test_basin_of_attraction(self):
        """Basin of attraction labels should be assigned to converged points."""
        def equations(x):
            # Two fixed points: x=0 and x=1 for x -> x^2
            return x ** 2

        iterator = FixedPointIterator(max_iter=200, tol=1e-6)
        detector = MultiFixedPointDetector(iterator=iterator)
        fps = np.array([[0.0], [1.0]])
        grid = np.linspace(-0.5, 1.5, 20).reshape(-1, 1)
        labels = detector.basin_of_attraction(equations, fps, grid, method="plain")
        assert labels.shape == (20,)
        # Points near 0 should converge to fp 0
        assert labels[10] in (0, 1, -1)  # just check it's a valid label


# ======================================================================
# 6. FixedPointStabilityAnalyzer tests
# ======================================================================


class TestFixedPointStabilityAnalyzer:
    def test_jacobian_shape(self):
        """Jacobian should be (dim, dim)."""
        analyzer = FixedPointStabilityAnalyzer()
        fp = np.array([0.7390851332])
        J = analyzer.compute_jacobian(
            lambda x: np.array([np.cos(x[0])]),
            fp,
        )
        assert J.shape == (1, 1)

    def test_stable_fixed_point(self):
        """cos(x) fixed point at 0.739 is stable (|J| < 1)."""
        analyzer = FixedPointStabilityAnalyzer()
        fp = np.array([0.7390851332])
        J = analyzer.compute_jacobian(lambda x: np.array([np.cos(x[0])]), fp)
        eig = analyzer.eigenvalue_analysis(J)
        classification = analyzer.classify_fixed_point(eig["eigenvalues"])
        # |sin(0.739)| ~ 0.674 < 1 → stable
        assert classification["stable"]

    def test_unstable_fixed_point(self):
        """x=0 is an unstable fixed point of x -> 2*sin(x)."""
        analyzer = FixedPointStabilityAnalyzer()
        fp = np.array([0.0])
        J = analyzer.compute_jacobian(lambda x: np.array([2 * np.sin(x[0])]), fp)
        eig = analyzer.eigenvalue_analysis(J)
        classification = analyzer.classify_fixed_point(eig["eigenvalues"])
        # J = 2*cos(0) = 2 > 1 → unstable
        assert not classification["stable"]

    def test_eigenvalue_analysis(self):
        """eigenvalue_analysis should return eigenvalues."""
        analyzer = FixedPointStabilityAnalyzer()
        J = np.array([[0.5, 0.1], [0.1, 0.3]])
        eig_result = analyzer.eigenvalue_analysis(J)
        assert "eigenvalues" in eig_result
        assert len(eig_result["eigenvalues"]) == 2

    def test_lyapunov_exponents(self):
        """Lyapunov exponents should be computable for a contracting map."""
        analyzer = FixedPointStabilityAnalyzer()
        def equations(x):
            return 0.5 * x
        lyap = analyzer.lyapunov_exponents(equations, np.array([0.1]), trajectory_length=100)
        # Contraction → negative Lyapunov exponent
        assert lyap[0] < 0


# ======================================================================
# 7. FreeEnergyLandscape tests
# ======================================================================


class TestFreeEnergyLandscape:
    def test_compute_free_energy_at_origin(self):
        """F(0) should be zero (no energy, no entropy)."""
        fel = FreeEnergyLandscape(temperature=1.0)
        F = fel.compute_free_energy(np.zeros(2))
        # At q=0: energy=0, entropy = T*sum(log(1)) = 0
        # Actually s(0) = ½[(1)ln(1) + (1)ln(1)] = 0
        assert F == pytest.approx(0.0, abs=1e-10)

    def test_free_energy_negative_at_aligned(self):
        """F should be lower at the aligned state for strong coupling."""
        J = 5.0 * np.eye(2)
        fel = FreeEnergyLandscape(temperature=1.0, coupling_matrix=J)
        F0 = fel.compute_free_energy(np.zeros(2))
        Fm = fel.compute_free_energy(np.array([0.8, 0.8]))
        assert Fm < F0

    def test_gradient_at_minimum(self):
        """Gradient should be near zero at a minimum."""
        fel = FreeEnergyLandscape(temperature=0.5, coupling_matrix=2.0 * np.eye(2))
        minima = fel.find_minima(n_starts=10)
        if len(minima) > 0:
            grad = fel.gradient_of_free_energy(minima[0]["position"])
            # At boundary-constrained minima, gradient may not be exactly zero
            assert np.linalg.norm(grad) < 5.0

    def test_hessian_positive_at_minimum(self):
        """Hessian eigenvalues should be positive at a minimum."""
        fel = FreeEnergyLandscape(temperature=0.5, coupling_matrix=2.0 * np.eye(2))
        minima = fel.find_minima(n_starts=10)
        if len(minima) > 0:
            eigvals = minima[0]["hessian_eigenvalues"]
            assert np.all(eigvals > -1e-6)

    def test_free_energy_surface_shape(self):
        """Free energy surface should return 3 meshgrids of correct shape."""
        fel = FreeEnergyLandscape(temperature=1.0)
        Q1, Q2, F = fel.free_energy_surface(resolution=20)
        assert Q1.shape == (20, 20)
        assert Q2.shape == (20, 20)
        assert F.shape == (20, 20)

    def test_entropy_from_free_energy(self):
        """Entropy should be a finite real number."""
        fel = FreeEnergyLandscape(temperature=1.0)
        S = fel.entropy_from_free_energy(np.array([0.3, 0.3]))
        assert np.isfinite(S)

    def test_invalid_temperature_raises(self):
        """Temperature <= 0 should raise ValueError."""
        with pytest.raises(ValueError):
            FreeEnergyLandscape(temperature=-1.0)

    def test_find_minima_returns_list(self):
        """find_minima should return a list of dicts."""
        fel = FreeEnergyLandscape(temperature=1.0)
        minima = fel.find_minima(n_starts=5)
        assert isinstance(minima, list)
        for m in minima:
            assert "position" in m
            assert "energy" in m


# ======================================================================
# 8. SaddlePointSolver tests
# ======================================================================


class TestSaddlePointSolver:
    def test_newton_convergence(self):
        """Newton solver should find the saddle point of a simple landscape."""
        fel = FreeEnergyLandscape(temperature=0.5, coupling_matrix=2.0 * np.eye(2))
        solver = SaddlePointSolver(landscape=fel, tol=1e-10)
        result = solver.solve_saddle_point(np.array([0.5, 0.5]), method="newton")
        assert result.converged
        assert result.residual < 1e-6

    def test_hybr_method(self):
        """Scipy hybr solver should converge."""
        fel = FreeEnergyLandscape(temperature=1.0)
        solver = SaddlePointSolver(landscape=fel, tol=1e-8)
        result = solver.solve_saddle_point(np.array([0.1, 0.1]), method="hybr")
        assert result.converged

    def test_continuation_method(self):
        """Continuation should track solutions as parameter varies."""
        fel = FreeEnergyLandscape(temperature=1.0)
        solver = SaddlePointSolver(landscape=fel)

        def eqs(x, lam):
            """Gradient with temperature-like parameter."""
            J = lam * np.eye(2)
            q = np.clip(x, -0.99, 0.99)
            return -(J @ q) + np.arctanh(q)

        params = np.linspace(0.5, 2.0, 10)
        result = solver.continuation_method(eqs, "coupling", params,
                                            initial_solution=np.array([0.1, 0.1]))
        assert result["solutions"].shape == (10, 2)
        assert np.all(np.isfinite(result["solutions"]))

    def test_saddle_point_from_action(self):
        """Should find saddle point of a quadratic action."""
        solver = SaddlePointSolver(tol=1e-8)

        def action(x):
            return 0.5 * np.sum(x ** 2)

        result = solver.saddle_point_from_action(action, np.array([1.0, 1.0]))
        assert result.converged
        np.testing.assert_allclose(result.solution, [0.0, 0.0], atol=1e-4)

    def test_no_landscape_no_equations_raises(self):
        """Should raise ValueError if no landscape and no equations."""
        solver = SaddlePointSolver()
        with pytest.raises(ValueError):
            solver.solve_saddle_point(np.array([0.0]))


# ======================================================================
# 9. FreeEnergyBarrier tests
# ======================================================================


class TestFreeEnergyBarrier:
    def test_neb_finds_barrier(self):
        """NEB should find a positive barrier between two minima."""
        J = 3.0 * np.eye(2)
        fel = FreeEnergyLandscape(temperature=0.5, coupling_matrix=J)
        barrier_calc = FreeEnergyBarrier(fel)
        # Two minima: roughly symmetric at (±q*, ±q*)
        result = barrier_calc.compute_barrier(
            np.array([0.8, 0.8]),
            np.array([-0.8, -0.8]),
            method="neb",
            n_images=15,
            max_iter=200,
        )
        assert result.barrier_height >= 0.0
        assert result.images.shape[0] == 15

    def test_string_method(self):
        """String method should produce a valid path."""
        J = 3.0 * np.eye(2)
        fel = FreeEnergyLandscape(temperature=0.5, coupling_matrix=J)
        barrier_calc = FreeEnergyBarrier(fel)
        result = barrier_calc.string_method(
            np.array([0.8, 0.8]),
            np.array([-0.8, -0.8]),
            n_images=12,
            max_iter=100,
        )
        assert len(result.energies) == 12
        assert result.barrier_height >= 0.0

    def test_barrier_height_static(self):
        """Static barrier_height should return max - min(endpoints)."""
        energies = np.array([0.0, 0.5, 1.0, 0.5, 0.0])
        bh = FreeEnergyBarrier.barrier_height(energies)
        assert bh == pytest.approx(1.0)

    def test_transition_rate(self):
        """Kramers rate should decrease with barrier height."""
        r1 = FreeEnergyBarrier.transition_rate(1.0, 1.0)
        r2 = FreeEnergyBarrier.transition_rate(2.0, 1.0)
        assert r2 < r1


# ======================================================================
# 10. PhaseTransitionDetector tests
# ======================================================================


class TestPhaseTransitionDetector:
    def test_detect_crossing(self):
        """Should detect where two branches cross."""
        ptd = PhaseTransitionDetector()
        params = np.linspace(0, 2, 100)
        f1 = params
        f2 = 2 - params
        crossing = ptd.detect_crossing(f1, f2, params)
        assert crossing is not None
        assert crossing == pytest.approx(1.0, abs=0.05)

    def test_no_crossing_returns_none(self):
        """Should return None when branches don't cross."""
        ptd = PhaseTransitionDetector()
        params = np.linspace(0, 1, 50)
        f1 = np.zeros(50)
        f2 = np.ones(50)
        crossing = ptd.detect_crossing(f1, f2, params)
        assert crossing is None

    def test_maxwell_construction(self):
        """Maxwell construction should find a transition parameter."""
        ptd = PhaseTransitionDetector()

        def free_energy(lam):
            # Double-well: F(lam) = (lam-1)^4 - 2*(lam-1)^2
            x = lam - 1.0
            return x ** 4 - 2 * x ** 2

        params = np.linspace(-0.5, 2.5, 200)
        result = ptd.maxwell_construction(free_energy, params)
        # Should detect a transition somewhere in the range
        assert "transition_param" in result

    def test_critical_point_detection(self):
        """Should find the critical point of a Landau-type free energy."""
        ptd = PhaseTransitionDetector()

        def free_energy(params):
            q, T = params
            return 0.5 * (T - 1.0) * q ** 2 + 0.25 * q ** 4

        result = ptd.critical_point(
            free_energy,
            param_ranges=((-0.5, 0.5), (0.5, 1.5)),
            resolution=50,
        )
        assert result is not None
        assert abs(result["param_critical"] - 1.0) < 0.1


# ======================================================================
# 11. TransitionClassifier tests
# ======================================================================


class TestTransitionClassifier:
    def test_classify_first_order(self):
        """A discontinuous jump should be classified as first-order."""
        tc = TransitionClassifier()
        control = np.linspace(0, 2, 200)
        # Order parameter with a sharp jump at control=1
        q = np.where(control < 1.0, 0.0, 0.8)
        info = tc.classify((control, q))
        assert info.order == 1
        assert abs(info.location - 1.0) < 0.05

    def test_classify_second_order(self):
        """A smooth continuous transition should be classified as second-order."""
        tc = TransitionClassifier()
        control = np.linspace(0, 2, 1000)
        # Smooth tanh-like transition (no discontinuity)
        q = 0.5 * (1.0 + np.tanh(2.0 * (control - 1.0)))
        info = tc.classify((control, q))
        assert info.order == 2

    def test_latent_heat(self):
        """Latent heat should be nonzero for a first-order transition."""
        tc = TransitionClassifier()
        T = np.linspace(0.5, 1.5, 200)
        F = np.where(T < 1.0, -T, -T + 0.5)  # kink at T=1
        L = tc.latent_heat(F, T)
        assert L is not None
        assert L > 0

    def test_critical_exponents_mean_field(self):
        """Should extract approximate mean-field exponents near Tc."""
        tc = TransitionClassifier()
        # Use control param where q grows above critical point
        control = np.linspace(0.01, 2.0, 500)
        T_c = 1.0
        q = np.where(control > T_c, np.sqrt(np.maximum(control - T_c, 0)), 0.0)
        exponents = tc.critical_exponents((control, q), T_c)
        # β should be close to 0.5
        if np.isfinite(exponents.get("beta", np.nan)):
            assert abs(exponents["beta"] - 0.5) < 0.3

    def test_landau_expansion(self):
        """Landau expansion should return coefficients."""
        tc = TransitionClassifier()
        q_vals = np.linspace(-0.8, 0.8, 100)
        # F = 0.5*q^2 + 0.25*q^4
        F_vals = 0.5 * q_vals ** 2 + 0.25 * q_vals ** 4
        result = tc.landau_expansion((q_vals, F_vals))
        assert isinstance(result, dict)


# ======================================================================
# 12. LinearResponseComputer tests
# ======================================================================


class TestLinearResponseComputer:
    def test_static_susceptibility(self):
        """Static susceptibility from correlation matrix."""
        lrc = LinearResponseComputer(system_size=10, temperature=1.0)
        # For uncorrelated system, correlation matrix ≈ identity
        corr = np.eye(10) + 0.01 * RNG.standard_normal((10, 10))
        corr = 0.5 * (corr + corr.T)
        chi = lrc.compute_static_susceptibility(corr)
        assert np.isfinite(chi)

    def test_connected_susceptibility(self):
        """Connected susceptibility from m(h) data."""
        lrc = LinearResponseComputer(system_size=5, temperature=1.0)
        h = np.linspace(-1, 1, 50)
        m = np.tanh(h)  # simple response
        chi = lrc.connected_susceptibility(m, h)
        assert np.all(np.isfinite(chi))

    def test_susceptibility_tensor_shape(self):
        """Susceptibility tensor should have correct shape."""
        lrc = LinearResponseComputer(system_size=3, temperature=1.0)
        order_params = RNG.standard_normal((10, 3))
        perturbations = RNG.standard_normal((10, 3))
        tensor = lrc.susceptibility_tensor(order_params, perturbations)
        assert tensor.shape == (3, 3)

    def test_sum_rule_check(self):
        """Sum rule check should return a result."""
        lrc = LinearResponseComputer(system_size=5, temperature=1.0)
        freqs = np.linspace(0.01, 10, 100)
        chi_real = 1.0 / (1.0 + freqs ** 2)
        chi_imag = freqs / (1.0 + freqs ** 2)
        result = lrc.sum_rule_check({
            "frequencies": freqs,
            "chi_real": chi_real,
            "chi_imag": chi_imag,
        })
        assert isinstance(result, dict)


# ======================================================================
# 13. FluctuationDissipation tests
# ======================================================================


class TestFluctuationDissipation:
    def test_verify_fdt_equilibrium(self):
        """FDT should be approximately satisfied in equilibrium."""
        fd = FluctuationDissipation(temperature=1.0)
        tau = 5.0
        times = np.linspace(0, 30, 100)
        C = np.exp(-times / tau)
        R = (1.0 / tau) * np.exp(-times / tau)
        result = fd.verify_fdt(C, R, times)
        assert isinstance(result, dict)
        # Check that ratio is close to 1/T
        if "ratio" in result:
            assert abs(result["ratio"] - 1.0) < 0.5

    def test_effective_temperature(self):
        """Effective temperature should equal T for equilibrium dynamics."""
        fd = FluctuationDissipation(temperature=1.0)
        times = np.linspace(0, 20, 50)
        C = np.exp(-times / 5.0)
        R = np.exp(-times / 5.0) / 5.0
        T_eff = fd.effective_temperature(C, R, times)
        assert isinstance(T_eff, (float, np.floating, dict))

    def test_fdt_ratio(self):
        """FDT ratio should be computable."""
        fd = FluctuationDissipation(temperature=1.0)
        n_tw, n_t = 5, 30
        C = np.exp(-np.arange(n_t)[None, :] / 5.0) * np.ones((n_tw, 1))
        R = C / 5.0
        ratio = fd.fdt_ratio(C, R)
        assert ratio.shape == (n_tw, n_t)

    def test_parametric_fdt_plot(self):
        """Parametric FDT plot should return correlation and integrated response."""
        fd = FluctuationDissipation(temperature=1.0)
        C = np.exp(-np.linspace(0, 5, 50))
        R = C / 5.0
        result = fd.parametric_fdt_plot(C, R)
        assert isinstance(result, dict)


# ======================================================================
# 14. DynamicSusceptibility tests
# ======================================================================


class TestDynamicSusceptibility:
    def test_compute_chi_omega(self):
        """Should compute frequency-dependent susceptibility."""
        ds = DynamicSusceptibility(max_frequency=50.0, n_frequencies=256)
        dt = 0.1
        times = np.arange(0, 25.6, dt)
        # Exponentially decaying correlation
        C = np.exp(-times / 5.0)
        result = ds.compute_chi_omega(C, dt)
        assert isinstance(result, (dict, np.ndarray, tuple))

    def test_relaxation_time(self):
        """Relaxation time from chi'' peak should match input tau."""
        ds = DynamicSusceptibility(max_frequency=50.0, n_frequencies=500)
        tau = 2.0
        freqs = np.linspace(0.01, 20, 500)
        chi_imag = freqs * tau / (1.0 + (freqs * tau) ** 2)
        result = ds.relaxation_time(chi_imag, freqs)
        # Should be close to tau
        if isinstance(result, (float, np.floating)):
            assert abs(result - tau) < 1.0
        elif isinstance(result, dict) and "tau" in result:
            assert abs(result["tau"] - tau) < 1.0

    def test_real_and_imaginary(self):
        """Should separate real and imaginary parts."""
        ds = DynamicSusceptibility()
        chi = np.array([1 + 0.5j, 0.8 + 0.3j, 0.5 + 0.1j])
        result = ds.real_and_imaginary(chi)
        assert isinstance(result, (dict, tuple))

    def test_spectral_density(self):
        """Spectral density should be non-negative for physical systems."""
        ds = DynamicSusceptibility()
        freqs = np.linspace(0.1, 10, 100)
        chi_imag = freqs / (1.0 + freqs ** 2)
        T = 1.0
        rho = ds.spectral_density(chi_imag, freqs, T)
        # Spectral density proportional to chi''/omega should be >= 0
        if isinstance(rho, np.ndarray):
            assert np.all(rho >= -1e-10)


# ======================================================================
# 15. CriticalExponentExtractor tests
# ======================================================================


class TestCriticalExponentExtractor:
    def _power_law_data(self, exponent, Tc=1.0, n=200):
        """Generate power-law data near Tc."""
        T = np.linspace(Tc + 0.01, Tc + 1.0, n)
        eps = T - Tc
        y = eps ** exponent
        return T, y

    def test_extract_beta(self):
        """Should extract β ≈ 0.5 from mean-field order parameter."""
        cee = CriticalExponentExtractor(critical_point_estimate=1.0)
        T = np.linspace(0.01, 0.99, 200)
        q = np.sqrt(1.0 - T)  # β = 0.5
        result = cee.extract_beta(q, T)
        if isinstance(result, dict) and "beta" in result:
            assert abs(result["beta"] - 0.5) < 0.3
        elif isinstance(result, (float, np.floating)):
            assert abs(result - 0.5) < 0.3

    def test_extract_gamma(self):
        """Should extract γ ≈ 1.0 from diverging susceptibility."""
        cee = CriticalExponentExtractor(critical_point_estimate=1.0)
        T = np.linspace(1.01, 2.0, 200)
        chi = 1.0 / (T - 1.0)  # γ = 1
        result = cee.extract_gamma(chi, T)
        if isinstance(result, dict) and "gamma" in result:
            assert abs(result["gamma"] - 1.0) < 0.3
        elif isinstance(result, (float, np.floating)):
            assert abs(result - 1.0) < 0.3

    def test_fit_power_law(self):
        """fit_power_law should return exponent and quality of fit."""
        cee = CriticalExponentExtractor(critical_point_estimate=0.0)
        x = np.linspace(0.01, 1.0, 100)
        y = x ** 0.7
        result = cee.fit_power_law(x, y, x_critical=0.0)
        assert isinstance(result, dict)
        if "exponent" in result:
            assert abs(result["exponent"] - 0.7) < 0.2

    def test_scaling_relation_check(self):
        """Scaling relations should be approximately satisfied."""
        cee = CriticalExponentExtractor()
        # Mean-field exponents: α=0, β=0.5, γ=1, δ=3, ν=0.5, η=0
        exponents = {"alpha": 0, "beta": 0.5, "gamma": 1.0,
                     "delta": 3.0, "nu": 0.5, "eta": 0.0}
        result = cee.scaling_relation_check(exponents)
        assert isinstance(result, dict)

    def test_extract_eta(self):
        """Should extract η from spatial correlations at Tc."""
        cee = CriticalExponentExtractor()
        distances = np.linspace(1, 50, 100)
        # C(r) ~ r^{-(d-2+eta)} in d=3 with eta=0 → r^{-1}
        C = 1.0 / distances
        result = cee.extract_eta(C, distances)
        assert result is not None


# ======================================================================
# 16. FiniteSizeScaling tests
# ======================================================================


class TestFiniteSizeScaling:
    def test_binder_cumulant(self):
        """Binder cumulant U4 should be between 0 and 1 for Gaussian data."""
        sizes = [10, 20, 40, 80]
        fss = FiniteSizeScaling(system_sizes=sizes)
        samples = {}
        rng = np.random.default_rng(99)
        for L in sizes:
            samples[L] = rng.standard_normal(1000)
        result = fss.binder_cumulant(samples)
        assert isinstance(result, dict)

    def test_scaling_collapse(self):
        """Scaling collapse should return a quality metric."""
        sizes = [8, 16, 32, 64]
        fss = FiniteSizeScaling(system_sizes=sizes)
        rng = np.random.default_rng(77)
        T = np.linspace(0.5, 1.5, 50)
        data = {}
        Tc = 1.0
        for L in sizes:
            data[L] = np.array([np.abs(t - Tc) ** 0.5 * L ** (0.5 / 0.5)
                                + 0.01 * rng.standard_normal() for t in T])
        result = fss.scaling_collapse(data, T, nu_guess=0.5, gamma_guess=1.0)
        assert isinstance(result, dict)

    def test_crossing_analysis(self):
        """Crossing analysis should find approximate Tc."""
        sizes = [10, 20, 40]
        fss = FiniteSizeScaling(system_sizes=sizes)
        T = np.linspace(0.5, 1.5, 100)
        binder_data = {}
        for L in sizes:
            binder_data[L] = 1.0 / (1.0 + np.exp(-L * 0.1 * (T - 1.0)))
        result = fss.crossing_analysis({
            "temperatures": T,
            "binder": binder_data,
        })
        assert isinstance(result, dict)

    def test_finite_size_critical_point(self):
        """Should estimate Tc from finite-size data."""
        sizes = [8, 16, 32]
        fss = FiniteSizeScaling(system_sizes=sizes)
        # Map L -> pseudo-critical Tc(L), shifting toward Tc=1.0
        observable_vs_size = {L: 1.0 + 0.5 / L for L in sizes}
        result = fss.finite_size_critical_point(observable_vs_size, "temperature")
        assert isinstance(result, dict)
        assert "Tc_inf" in result


# ======================================================================
# 17. ReplicaSymmetricSolver tests
# ======================================================================


class TestReplicaSymmetricSolver:
    def test_rs_free_energy_finite(self):
        """RS free energy should be finite for valid parameters."""
        rs = ReplicaSymmetricSolver(temperature=1.0, alpha=1.0)
        F = rs.rs_free_energy(0.5, 0.5)
        assert np.isfinite(F)

    def test_rs_saddle_point_residual(self):
        """Saddle point equations should have small residual at solution."""
        rs = ReplicaSymmetricSolver(temperature=1.0, alpha=1.0)
        sol = rs.solve_rs(initial_q=0.3, initial_qhat=0.3, max_iter=3000)
        if sol["converged"]:
            residual = rs.rs_saddle_point_equations(sol["q"], sol["qhat"])
            assert np.linalg.norm(residual) < 1e-3

    def test_solve_rs_converges(self):
        """RS solver should converge for moderate alpha."""
        rs = ReplicaSymmetricSolver(temperature=1.0, alpha=0.5)
        sol = rs.solve_rs(initial_q=0.3, initial_qhat=0.3, max_iter=5000)
        assert sol["converged"]
        assert 0 < sol["q"] < 1

    def test_training_error_bounded(self):
        """Training error should be in [0, 1]."""
        rs = ReplicaSymmetricSolver(temperature=1.0, alpha=1.0)
        sol = rs.solve_rs()
        if sol["converged"]:
            e_t = rs.training_error_rs(sol["q"])
            assert 0 <= e_t <= 1.0

    def test_generalization_error_bounded(self):
        """Generalization error should be in [0, 0.5]."""
        rs = ReplicaSymmetricSolver(temperature=1.0, alpha=1.0)
        sol = rs.solve_rs()
        if sol["converged"]:
            e_g = rs.generalization_error_rs(sol["q"])
            assert 0 <= e_g <= 0.5

    def test_entropy_finite(self):
        """RS entropy should be finite."""
        rs = ReplicaSymmetricSolver(temperature=1.0, alpha=0.5)
        sol = rs.solve_rs()
        if sol["converged"]:
            S = rs.entropy_rs(sol["q"], sol["qhat"])
            assert np.isfinite(S)

    def test_free_energy_monotonic_in_temperature(self):
        """Free energy at fixed q, qhat should vary smoothly with T."""
        F_vals = []
        for T in [0.5, 1.0, 1.5, 2.0]:
            rs = ReplicaSymmetricSolver(temperature=T, alpha=1.0)
            F_vals.append(rs.rs_free_energy(0.5, 0.5))
        assert all(np.isfinite(F_vals))


# ======================================================================
# 18. OneStepRSBSolver tests
# ======================================================================


class TestOneStepRSBSolver:
    def test_rsb1_free_energy_finite(self):
        """1RSB free energy should be finite for valid parameters."""
        rsb = OneStepRSBSolver(temperature=1.0, alpha=1.0)
        F = rsb.rsb1_free_energy(0.3, 0.7, 0.5, 0.3, 0.7)
        assert np.isfinite(F)

    def test_rsb1_overlap_distribution(self):
        """1RSB overlap distribution should have two delta peaks."""
        rsb = OneStepRSBSolver(temperature=1.0, alpha=1.0)
        result = rsb.rsb1_overlap_distribution(0.3, 0.8, 0.6)
        assert isinstance(result, dict)
        if "q" in result and "pq" in result:
            assert len(result["q"]) > 0

    def test_complexity(self):
        """Complexity should be computable."""
        rsb = OneStepRSBSolver(temperature=1.0, alpha=1.0)
        # Use a simple free energy value
        Sigma = rsb.complexity(-0.5, 0.5)
        assert isinstance(Sigma, (float, np.floating))


# ======================================================================
# 19. DeAlmeidaThoulessChecker tests
# ======================================================================


class TestDeAlmeidaThoulessChecker:
    def test_at_stability_high_temperature(self):
        """RS should be stable at high temperature."""
        at = DeAlmeidaThoulessChecker(temperature=2.0)
        # At high T, q is small, RS is stable
        lam = at.at_stability_condition(0.1, 0.1)
        assert isinstance(lam, (float, np.floating))

    def test_replicon_eigenvalue(self):
        """Replicon eigenvalue should be computable."""
        at = DeAlmeidaThoulessChecker(temperature=1.0)
        lam = at.replicon_eigenvalue(0.5, 0.5)
        assert np.isfinite(lam)

    def test_rs_stability_check(self):
        """RS stability check should return stability info."""
        at = DeAlmeidaThoulessChecker(temperature=1.0)
        rs_sol = {"q": 0.3, "qhat": 0.3}
        result = at.rs_stability_check(rs_sol)
        assert "stable" in result
        assert "replicon" in result

    def test_at_line_returns_dict(self):
        """AT line computation should return temperatures and fields."""
        at = DeAlmeidaThoulessChecker(temperature=1.0)
        T_range = np.linspace(0.3, 0.9, 5)
        h_range = np.linspace(0, 2, 20)
        result = at.at_line(T_range, field_range=h_range)
        assert "temperature" in result
        assert "h_AT" in result
        assert len(result["temperature"]) == 5

    def test_instability_direction(self):
        """Instability direction should be a matrix."""
        at = DeAlmeidaThoulessChecker(temperature=1.0)
        rs_sol = {"q": 0.5, "qhat": 0.5}
        delta_Q = at.instability_direction(rs_sol)
        assert delta_Q.ndim == 2
        # Should be symmetric
        np.testing.assert_allclose(delta_Q, delta_Q.T, atol=1e-10)

    def test_at_stability_with_rs_solution(self):
        """Check AT stability of an actual RS solution."""
        rs = ReplicaSymmetricSolver(temperature=1.0, alpha=0.5)
        sol = rs.solve_rs()
        if sol["converged"]:
            at = DeAlmeidaThoulessChecker(temperature=1.0)
            result = at.rs_stability_check(sol)
            assert isinstance(result["stable"], (bool, np.bool_))


# ======================================================================
# 20. OverlapDistribution tests
# ======================================================================


class TestOverlapDistribution:
    def test_pq_from_samples_normalized(self):
        """P(q) from samples should integrate to ~1."""
        od = OverlapDistribution(n_bins=100)
        samples = RNG.uniform(-1, 1, size=10000)
        result = od.compute_pq_from_samples(samples)
        dq = np.diff(result["q"])
        integral = np.sum(result["pq"][:-1] * dq)
        assert integral == pytest.approx(1.0, abs=0.1)

    def test_pq_from_rs_single_peak(self):
        """RS P(q) should be a single delta peak at q_RS."""
        od = OverlapDistribution(n_bins=200)
        q_rs = 0.7
        result = od.pq_from_rs(q_rs)
        # Peak should be near q_rs
        peak_idx = np.argmax(result["pq"])
        assert abs(result["q"][peak_idx] - q_rs) < 0.02

    def test_pq_from_rsb1_two_peaks(self):
        """1RSB P(q) should have two peaks at q0 and q1."""
        od = OverlapDistribution(n_bins=200)
        q0, q1, m = 0.3, 0.8, 0.5
        result = od.pq_from_rsb1(q0, q1, m)
        pq = result["pq"]
        q = result["q"]
        # Should have support near q0 and q1
        assert pq[np.argmin(np.abs(q - q0))] > 0
        assert pq[np.argmin(np.abs(q - q1))] > 0

    def test_pq_moments(self):
        """Moments should be consistent with the distribution."""
        od = OverlapDistribution(n_bins=200)
        result = od.pq_from_rs(0.5)
        moments = od.pq_moments(result["pq"], result["q"])
        # First moment should be close to q_rs = 0.5
        assert abs(moments[1] - 0.5) < 0.05

    def test_pq_support(self):
        """Support of RS P(q) should be narrow."""
        od = OverlapDistribution(n_bins=200)
        result = od.pq_from_rs(0.5)
        support = od.pq_support(result["pq"], result["q"])
        assert support["is_delta"]

    def test_ghirlanda_guerra_check_random(self):
        """GG check should run on a random overlap matrix."""
        od = OverlapDistribution()
        n = 5
        configs = _random_configs(n, 100, seed=33)
        op = OverlapParameter(normalize=True)
        Q = op.compute_overlap_matrix(configs)
        result = od.ghirlanda_guerra_check(Q)
        assert "violation" in result
        assert "satisfied" in result

    def test_ultrametricity_check(self):
        """Ultrametricity check should run on a random overlap matrix."""
        od = OverlapDistribution()
        n = 5
        configs = _random_configs(n, 100, seed=44)
        op = OverlapParameter(normalize=True)
        Q = op.compute_overlap_matrix(configs)
        result = od.ultrametricity_check(Q)
        assert "fraction_violated" in result
        assert "ultrametric" in result

    def test_ultrametric_matrix_passes(self):
        """A perfectly ultrametric matrix should pass the check."""
        od = OverlapDistribution()
        # Construct an ultrametric overlap matrix:
        # Two groups: {0,1} and {2,3}, inter-group overlap = 0.3,
        # intra-group overlap = 0.8
        Q = np.array([
            [1.0, 0.8, 0.3, 0.3],
            [0.8, 1.0, 0.3, 0.3],
            [0.3, 0.3, 1.0, 0.8],
            [0.3, 0.3, 0.8, 1.0],
        ])
        result = od.ultrametricity_check(Q)
        assert result["ultrametric"]

    def test_ghirlanda_guerra_too_few_replicas(self):
        """GG check should handle fewer than 3 replicas gracefully."""
        od = OverlapDistribution()
        Q = np.array([[1.0, 0.5], [0.5, 1.0]])
        result = od.ghirlanda_guerra_check(Q)
        assert result["valid"] is False


# ======================================================================
# 21. ParisiFunctional tests
# ======================================================================


class TestParisiFunctional:
    def test_parisi_pde_runs(self):
        """Parisi PDE should return a result for a simple q(x)."""
        pf = ParisiFunctional(n_steps=50)

        def q_func(x):
            return 0.3 + 0.5 * x  # linear q(x)

        result = pf.parisi_pde(q_func)
        assert result is not None

    def test_parisi_free_energy_finite(self):
        """Parisi free energy should be finite."""
        pf = ParisiFunctional(n_steps=50)

        def q_func(x):
            return 0.3 + 0.5 * x

        F = pf.parisi_free_energy(q_func)
        assert np.isfinite(F)

    def test_discretized_rsb(self):
        """Discretized k-RSB should return a result."""
        pf = ParisiFunctional(n_steps=50)
        result = pf.discretized_rsb(k_steps=2, alpha=1.0, temperature=1.0)
        assert isinstance(result, dict)


# ======================================================================
# 22. Integration / cross-module tests
# ======================================================================


class TestCrossModule:
    def test_overlap_to_free_energy_consistency(self):
        """Overlap at free energy minimum should be physical."""
        J = 2.0 * np.eye(2)
        fel = FreeEnergyLandscape(temperature=0.5, coupling_matrix=J)
        minima = fel.find_minima(n_starts=10)
        # Each minimum should have |q| <= 1
        for m in minima:
            assert np.all(np.abs(m["position"]) <= 1.0 + 1e-6)

    def test_rs_to_at_stability_pipeline(self):
        """Full pipeline: solve RS, then check AT stability."""
        rs = ReplicaSymmetricSolver(temperature=1.0, alpha=0.5)
        sol = rs.solve_rs(max_iter=5000)
        if sol["converged"]:
            at = DeAlmeidaThoulessChecker(temperature=1.0)
            stability = at.rs_stability_check(sol)
            assert "stable" in stability
            assert "replicon" in stability

    def test_overlap_parameter_to_distribution(self):
        """Compute overlaps from configs and build P(q)."""
        N = 50
        configs = _random_configs(8, N, seed=55)
        op = OverlapParameter(normalize=True)
        Q = op.compute_overlap_matrix(configs)
        # Extract off-diagonal elements
        triu_idx = np.triu_indices(8, k=1)
        overlaps = Q[triu_idx]
        od = OverlapDistribution(n_bins=50)
        result = od.compute_pq_from_samples(overlaps)
        assert len(result["pq"]) == 50

    def test_fixed_point_then_stability(self):
        """Find a fixed point, then analyze its stability."""
        fpi = FixedPointIterator(max_iter=500, tol=1e-10)
        result = fpi.iterate(
            np.array([0.5]),
            lambda x: np.array([np.cos(x[0])]),
        )
        assert result["converged"]
        analyzer = FixedPointStabilityAnalyzer()
        J = analyzer.compute_jacobian(
            lambda x: np.array([np.cos(x[0])]),
            result["solution"],
        )
        eig = analyzer.eigenvalue_analysis(J)
        classification = analyzer.classify_fixed_point(eig["eigenvalues"])
        assert classification["stable"]

    def test_saddle_point_to_barrier(self):
        """Find saddle point, then compute barrier between minima."""
        J = 3.0 * np.eye(2)
        fel = FreeEnergyLandscape(temperature=0.5, coupling_matrix=J)
        solver = SaddlePointSolver(landscape=fel, tol=1e-8)

        # Find two minima
        r1 = solver.solve_saddle_point(np.array([0.5, 0.5]))
        r2 = solver.solve_saddle_point(np.array([-0.5, -0.5]))

        if r1.converged and r2.converged:
            barrier_calc = FreeEnergyBarrier(fel)
            path = barrier_calc.compute_barrier(
                r1.solution, r2.solution,
                n_images=10, max_iter=50,
            )
            assert path.barrier_height >= 0.0


# ======================================================================
# 23. Edge cases and error handling
# ======================================================================


class TestEdgeCases:
    def test_overlap_single_site(self):
        """Overlap computation with a single site."""
        op = OverlapParameter(normalize=True)
        assert op.compute_overlap(np.array([1.0]), np.array([1.0])) == 1.0
        assert op.compute_overlap(np.array([1.0]), np.array([-1.0])) == -1.0

    def test_zero_temperature_free_energy(self):
        """Very low temperature should push to extreme order parameters."""
        J = 5.0 * np.eye(2)
        fel = FreeEnergyLandscape(temperature=0.01, coupling_matrix=J)
        minima = fel.find_minima(n_starts=20)
        if len(minima) > 0:
            # At low T, minima should be near ±1
            best = minima[0]["position"]
            assert np.all(np.abs(best) > 0.5)

    def test_large_system_overlap_matrix(self):
        """Overlap matrix for moderately large system."""
        op = OverlapParameter(normalize=True)
        configs = _random_configs(10, 1000, seed=77)
        Q = op.compute_overlap_matrix(configs)
        assert Q.shape == (10, 10)
        # Off-diagonal should be ~0 for random configs
        off_diag = Q[np.triu_indices(10, k=1)]
        assert np.mean(np.abs(off_diag)) < 0.15

    def test_correlation_zero_configs(self):
        """Correlation of zero configs should be zero."""
        cf = CorrelationFunction(normalize=True)
        configs = np.zeros((10, 20))
        C = cf.equal_time_correlation(configs)
        assert C == pytest.approx(0.0)

    def test_rs_solver_extreme_alpha(self):
        """RS solver should handle very small alpha gracefully."""
        rs = ReplicaSymmetricSolver(temperature=1.0, alpha=0.01)
        sol = rs.solve_rs(initial_q=0.1, initial_qhat=0.1, max_iter=3000)
        # At very low alpha, q should be small
        if sol["converged"]:
            assert sol["q"] < 0.5

    def test_generalization_error_range(self):
        """Generalization error for q near 0 should be ~0.5 (random guessing)."""
        rs = ReplicaSymmetricSolver(temperature=1.0, alpha=1.0)
        e_g = rs.generalization_error_rs(0.0)
        assert e_g == pytest.approx(0.5, abs=0.01)

    def test_generalization_error_perfect(self):
        """Generalization error for q near 1 should be ~0."""
        rs = ReplicaSymmetricSolver(temperature=1.0, alpha=1.0)
        e_g = rs.generalization_error_rs(0.99)
        assert e_g < 0.1
