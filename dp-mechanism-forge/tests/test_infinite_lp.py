"""
Comprehensive tests for dp_forge.infinite module (cutting-plane solver).

Tests cover convergence monitoring, dual oracle, continuous relaxation,
optimal transport, duality certification, and the main InfiniteLPSolver.
"""

from __future__ import annotations

import math
from typing import List

import numpy as np
import pytest
from hypothesis import given, settings, assume
from hypothesis import strategies as st

from dp_forge.infinite.convergence_monitor import (
    ConvergenceMonitor,
    ConvergenceSnapshot,
)
from dp_forge.infinite.dual_oracle import (
    DualOracle,
    OracleResult,
    _golden_section_max,
)
from dp_forge.infinite.continuous_relaxation import ContinuousMechanism
from dp_forge.infinite.optimal_transport import (
    DPTransport,
    TransportPlan,
    _ground_cost_matrix,
)
from dp_forge.infinite.duality import (
    InfiniteDualityChecker,
    SlaterPoint,
    GapCertificate,
)
from dp_forge.infinite.cutting_plane import (
    InfiniteLPSolver,
    InfiniteLPResult,
    solve_infinite_lp,
    _build_initial_grid,
    _insert_point,
)
from dp_forge.types import QuerySpec, LossFunction


# =========================================================================
# Fixtures
# =========================================================================


@pytest.fixture
def small_spec():
    """Small QuerySpec for testing (2 databases, 10 bins)."""
    return QuerySpec(
        query_values=np.array([0.0, 1.0]),
        domain="test",
        sensitivity=1.0,
        epsilon=1.0,
        delta=0.0,
        k=10,
        loss_fn=LossFunction.L2,
    )


@pytest.fixture
def medium_spec():
    """Medium QuerySpec for testing (3 databases, 20 bins)."""
    return QuerySpec(
        query_values=np.array([0.0, 1.0, 2.0]),
        domain="test",
        sensitivity=1.0,
        epsilon=1.0,
        delta=0.0,
        k=20,
        loss_fn=LossFunction.L2,
    )


@pytest.fixture
def rng():
    return np.random.default_rng(42)


# =========================================================================
# Section 1: Convergence Monitor
# =========================================================================


class TestConvergenceMonitor:
    """Tests for the convergence monitor."""

    def test_gap_decreases(self):
        """Gap should decrease monotonically with good updates."""
        mon = ConvergenceMonitor(target_tol=1e-6, max_iter=100)
        prev_gap = float("inf")
        for i in range(10):
            ub = 1.0 + 1.0 / (i + 1)
            lb = 1.0 - 0.5 / (i + 1)
            snap = mon.update(ub, lb, grid_size=10 + i, violation=0.1 / (i + 1))
            assert snap.gap <= prev_gap + 1e-12
            prev_gap = snap.gap

    def test_converged_below_tol(self):
        """Monitor reports convergence when gap < tolerance."""
        mon = ConvergenceMonitor(target_tol=0.1)
        mon.update(1.05, 1.0, grid_size=10, violation=0.01)
        assert mon.should_terminate()

    def test_not_converged_above_tol(self):
        """Monitor does not terminate when gap > tolerance."""
        mon = ConvergenceMonitor(target_tol=1e-6, max_iter=100)
        mon.update(2.0, 1.0, grid_size=10, violation=0.5)
        assert not mon.should_terminate()

    def test_max_iter_terminates(self):
        """Terminates after max_iter even without convergence."""
        mon = ConvergenceMonitor(target_tol=1e-10, max_iter=5)
        for i in range(6):
            mon.update(2.0, 1.0, grid_size=10, violation=0.5)
        assert mon.should_terminate()

    def test_snapshot_fields(self):
        """ConvergenceSnapshot has all expected fields."""
        mon = ConvergenceMonitor(target_tol=1e-6)
        snap = mon.update(2.0, 1.0, grid_size=10, violation=0.5)
        assert snap.iteration == 0
        assert abs(snap.gap - 1.0) < 1e-10
        assert snap.grid_size == 10
        assert abs(snap.violation - 0.5) < 1e-10

    def test_gap_history(self):
        """gap_history returns array of all recorded gaps."""
        mon = ConvergenceMonitor(target_tol=1e-6, max_iter=100)
        for i in range(5):
            mon.update(2.0 - i * 0.1, 1.0 + i * 0.05, grid_size=10, violation=0.1)
        hist = mon.gap_history()
        assert len(hist) == 5

    def test_convergence_rate(self):
        """convergence_rate returns (alpha, C) pair."""
        mon = ConvergenceMonitor(target_tol=1e-6, max_iter=100)
        for i in range(20):
            gap = 1.0 / (i + 1)
            mon.update(1.0 + gap, 1.0, grid_size=10 + i, violation=gap)
        alpha, C = mon.convergence_rate()
        assert alpha > 0  # Should detect some convergence rate
        assert C > 0

    def test_predict_iterations(self):
        """predict_iterations_remaining returns reasonable estimate."""
        mon = ConvergenceMonitor(target_tol=0.01, max_iter=1000)
        for i in range(10):
            gap = 1.0 / (i + 1)
            mon.update(1.0 + gap, 1.0, grid_size=10 + i, violation=gap)
        remaining = mon.predict_iterations_remaining()
        if remaining is not None:
            assert remaining >= 0

    def test_summary_string(self):
        """summary() returns a non-empty string."""
        mon = ConvergenceMonitor(target_tol=1e-6)
        mon.update(2.0, 1.0, grid_size=10, violation=0.5)
        s = mon.summary()
        assert isinstance(s, str)
        assert len(s) > 0

    def test_stall_detection(self):
        """Stall detection when gap doesn't improve."""
        mon = ConvergenceMonitor(
            target_tol=1e-10, max_iter=100,
            min_improvement=1e-8, stall_window=3,
        )
        for i in range(10):
            mon.update(2.0, 1.0, grid_size=10, violation=0.5)
        assert mon.should_terminate()


# =========================================================================
# Section 2: Dual Oracle
# =========================================================================


class TestDualOracle:
    """Tests for the dual oracle (separation oracle)."""

    def test_golden_section_finds_max(self):
        """Golden section search finds maximum of simple function."""
        f = lambda x: -(x - 3.0) ** 2 + 9.0  # max at x=3
        x_star, f_star, n_eval = _golden_section_max(f, 0.0, 6.0, tol=1e-10)
        assert abs(x_star - 3.0) < 1e-6
        assert abs(f_star - 9.0) < 1e-6

    def test_oracle_from_spec(self, small_spec):
        """DualOracle.from_spec creates valid oracle."""
        oracle = DualOracle.from_spec(small_spec, margin=1.0)
        assert oracle.domain[0] < oracle.domain[1]

    def test_oracle_finds_violation(self, small_spec):
        """Oracle finds point with positive reduced cost (violation)."""
        oracle = DualOracle.from_spec(small_spec, margin=2.0)
        # Create artificial dual variables that aren't optimal
        n = small_spec.n
        # Number of dual vars depends on LP structure; use simple ones
        # λ for epigraph, μ for simplex
        dual_vars = np.zeros(2 * n)
        dual_vars[:n] = 1.0  # λ = 1
        dual_vars[n:] = 0.0  # μ = 0
        result = oracle.find_most_violated(dual_vars, small_spec)
        assert isinstance(result, OracleResult)
        assert math.isfinite(result.y_star)
        assert result.n_evaluations > 0

    def test_oracle_domain(self, small_spec):
        """Oracle domain covers query values."""
        oracle = DualOracle.from_spec(small_spec, margin=1.0)
        q_min = small_spec.query_values.min()
        q_max = small_spec.query_values.max()
        assert oracle.domain[0] <= q_min
        assert oracle.domain[1] >= q_max

    def test_oracle_custom_bounds(self):
        """Custom domain bounds are respected."""
        oracle = DualOracle(domain_lower=-5.0, domain_upper=5.0)
        assert oracle.domain == (-5.0, 5.0)


# =========================================================================
# Section 3: Continuous Mechanism
# =========================================================================


class TestContinuousMechanism:
    """Tests for continuous mechanism density representation."""

    def test_density_integrates_to_one(self):
        """Density approximately integrates to 1."""
        grid = np.linspace(-3, 3, 50)
        weights = np.ones(50) / 50
        mech = ContinuousMechanism(y_grid=grid, weights=weights)
        # Integrate density over wide range
        y_eval = np.linspace(-5, 5, 2000)
        density_vals = np.array([mech.density(y) for y in y_eval])
        integral = np.trapz(density_vals, y_eval)
        assert abs(integral - 1.0) < 0.05  # KDE integral ≈ 1

    def test_cdf_monotone(self):
        """CDF is monotonically non-decreasing."""
        grid = np.linspace(0, 1, 20)
        weights = np.ones(20) / 20
        mech = ContinuousMechanism(y_grid=grid, weights=weights)
        y_eval = np.linspace(-1, 2, 100)
        cdf_vals = np.array([mech.cdf(y) for y in y_eval])
        for i in range(len(cdf_vals) - 1):
            assert cdf_vals[i] <= cdf_vals[i + 1] + 1e-10

    def test_cdf_range(self):
        """CDF goes from ~0 to ~1."""
        grid = np.linspace(0, 1, 20)
        weights = np.ones(20) / 20
        mech = ContinuousMechanism(y_grid=grid, weights=weights)
        assert mech.cdf(-10.0) < 0.01
        assert mech.cdf(10.0) > 0.99

    def test_density_nonnegative(self):
        """Density is non-negative."""
        grid = np.linspace(0, 1, 10)
        weights = np.ones(10) / 10
        mech = ContinuousMechanism(y_grid=grid, weights=weights)
        y_eval = np.linspace(-2, 3, 100)
        for y in y_eval:
            assert mech.density(y) >= -1e-15

    def test_expected_loss_nonneg(self):
        """Expected loss is non-negative."""
        grid = np.linspace(0, 1, 20)
        weights = np.ones(20) / 20
        mech = ContinuousMechanism(y_grid=grid, weights=weights)
        loss = mech.expected_loss(0.5, loss_fn=LossFunction.L2)
        assert loss >= 0

    def test_sample_shape(self, rng):
        """Sampling returns correct number of samples."""
        grid = np.linspace(0, 1, 10)
        weights = np.ones(10) / 10
        mech = ContinuousMechanism(y_grid=grid, weights=weights)
        samples = mech.sample(100, rng=rng)
        assert samples.shape == (100,)

    def test_total_variation_self_zero(self):
        """TV(M, M) = 0."""
        grid = np.linspace(0, 1, 20)
        weights = np.ones(20) / 20
        mech = ContinuousMechanism(y_grid=grid, weights=weights)
        tv = mech.total_variation(mech, n_points=500)
        assert tv < 0.01

    def test_total_variation_different(self):
        """TV between different mechanisms > 0."""
        grid = np.linspace(0, 1, 20)
        w1 = np.zeros(20)
        w1[0] = 1.0  # point mass at 0
        w2 = np.zeros(20)
        w2[-1] = 1.0  # point mass at 1
        m1 = ContinuousMechanism(y_grid=grid, weights=w1)
        m2 = ContinuousMechanism(y_grid=grid, weights=w2)
        tv = m1.total_variation(m2, n_points=500)
        assert tv > 0.5  # Should be close to 1.0

    def test_from_lp_solution(self):
        """from_lp_solution creates list of ContinuousMechanisms."""
        table = np.array([
            [0.5, 0.3, 0.2],
            [0.2, 0.3, 0.5],
        ])
        grid = np.array([0.0, 0.5, 1.0])
        mechanisms = ContinuousMechanism.from_lp_solution(table, grid)
        assert len(mechanisms) == 2
        for m in mechanisms:
            assert abs(m.weights.sum() - 1.0) < 1e-10

    def test_piecewise_constant_density(self):
        """Piecewise constant density covers grid."""
        grid = np.linspace(0, 1, 10)
        weights = np.ones(10) / 10
        mech = ContinuousMechanism(y_grid=grid, weights=weights)
        # Evaluate at midpoint of first bin
        mid = (grid[0] + grid[1]) / 2
        d_pc = mech.density_piecewise_constant(mid)
        assert d_pc >= 0


# =========================================================================
# Section 4: Optimal Transport
# =========================================================================


class TestOptimalTransport:
    """Tests for Wasserstein distance and optimal transport."""

    def test_wasserstein_identical_zero(self):
        """Wasserstein between identical distributions = 0."""
        transport = DPTransport(p=1.0)
        grid = np.linspace(0, 1, 10)
        dist = np.ones(10) / 10
        w = transport.wasserstein(dist, dist, grid)
        assert abs(w) < 1e-10

    def test_wasserstein_shifted(self):
        """Wasserstein between shifted distributions ≈ shift distance."""
        transport = DPTransport(p=1.0)
        grid_a = np.linspace(0, 1, 20)
        grid_b = np.linspace(0.5, 1.5, 20)
        dist = np.ones(20) / 20
        w = transport.wasserstein(dist, dist, grid_a, grid_b)
        assert abs(w - 0.5) < 0.1  # ≈ shift of 0.5

    def test_wasserstein_nonneg(self):
        """Wasserstein distance is non-negative."""
        transport = DPTransport(p=1.0)
        grid = np.linspace(0, 1, 10)
        rng = np.random.default_rng(42)
        d1 = rng.dirichlet(np.ones(10))
        d2 = rng.dirichlet(np.ones(10))
        w = transport.wasserstein(d1, d2, grid)
        assert w >= -1e-10

    def test_transport_plan_marginals(self):
        """Transport plan marginals match source and target."""
        transport = DPTransport(p=1.0)
        grid = np.linspace(0, 1, 5)
        source = np.array([0.3, 0.2, 0.2, 0.2, 0.1])
        target = np.array([0.1, 0.2, 0.3, 0.2, 0.2])
        plan = transport.transport_plan(source, target, grid)
        np.testing.assert_array_almost_equal(
            plan.source_marginal, source, decimal=6
        )
        np.testing.assert_array_almost_equal(
            plan.target_marginal, target, decimal=6
        )

    def test_earth_movers_distance(self):
        """EMD equals W₁ on same grid."""
        transport = DPTransport(p=1.0)
        grid = np.linspace(0, 1, 10)
        rng = np.random.default_rng(42)
        d1 = rng.dirichlet(np.ones(10))
        d2 = rng.dirichlet(np.ones(10))
        emd = transport.earth_movers_distance(d1, d2, grid)
        w1 = transport.wasserstein(d1, d2, grid)
        assert abs(emd - w1) < 1e-6

    def test_compare_mechanisms(self):
        """compare_mechanisms returns per-row distances."""
        transport = DPTransport(p=1.0)
        grid = np.linspace(0, 1, 5)
        mech_a = np.array([
            [0.5, 0.3, 0.1, 0.05, 0.05],
            [0.05, 0.05, 0.1, 0.3, 0.5],
        ])
        mech_b = np.array([
            [0.5, 0.3, 0.1, 0.05, 0.05],
            [0.5, 0.3, 0.1, 0.05, 0.05],
        ])
        dists = transport.compare_mechanisms(mech_a, mech_b, grid)
        assert dists.shape == (2,)
        assert dists[0] < 1e-10  # Same row
        assert dists[1] > 0  # Different rows

    def test_ground_cost_matrix(self):
        """Ground cost matrix has correct shape and values."""
        grid = np.array([0.0, 1.0, 2.0])
        C = _ground_cost_matrix(grid, grid, p=1.0)
        assert C.shape == (3, 3)
        assert C[0, 0] == 0
        assert abs(C[0, 2] - 2.0) < 1e-10

    def test_sinkhorn_transport(self):
        """Sinkhorn transport gives approximately correct result."""
        transport = DPTransport(p=1.0, use_sinkhorn=True, sinkhorn_reg=0.01)
        grid = np.linspace(0, 1, 10)
        dist = np.ones(10) / 10
        w = transport.wasserstein(dist, dist, grid)
        assert abs(w) < 0.1  # Sinkhorn is approximate


# =========================================================================
# Section 5: Duality Checker
# =========================================================================


class TestDualityChecker:
    """Tests for LP duality checking."""

    def test_slater_point_uniform(self, small_spec):
        """Uniform mechanism is a valid Slater (strictly feasible) point."""
        checker = InfiniteDualityChecker()
        y_grid = np.linspace(-1, 2, 10)
        slater = checker.slater_point(small_spec, y_grid)
        assert isinstance(slater, SlaterPoint)
        assert slater.is_strictly_feasible
        # Uniform mechanism: p_i(y) = 1/k for all i, y
        assert slater.mechanism.shape == (small_spec.n, len(y_grid))
        np.testing.assert_array_almost_equal(
            slater.mechanism.sum(axis=1),
            np.ones(small_spec.n),
            decimal=10,
        )

    def test_certify_gap_tight(self):
        """Certify gap when primal ≈ dual."""
        checker = InfiniteDualityChecker()
        cert = checker.certify_gap(
            primal_obj=1.0001, dual_obj=1.0, tolerance=0.01
        )
        assert isinstance(cert, GapCertificate)
        assert cert.is_certified
        assert cert.gap < 0.01

    def test_certify_gap_not_tight(self):
        """Non-certified gap when difference is large."""
        checker = InfiniteDualityChecker()
        cert = checker.certify_gap(
            primal_obj=2.0, dual_obj=1.0, tolerance=0.01
        )
        assert not cert.is_certified
        assert abs(cert.gap - 1.0) < 1e-10

    def test_gap_from_grid_size(self, small_spec):
        """Gap bound from grid size is positive."""
        checker = InfiniteDualityChecker()
        y_grid = np.linspace(-1, 2, 10)
        bound = checker.bound_gap_from_grid_size(small_spec, y_grid)
        assert bound >= 0

    def test_gap_bound_decreases_with_finer_grid(self, small_spec):
        """Finer grid → smaller gap bound."""
        checker = InfiniteDualityChecker()
        b1 = checker.bound_gap_from_grid_size(
            small_spec, np.linspace(-1, 2, 10)
        )
        b2 = checker.bound_gap_from_grid_size(
            small_spec, np.linspace(-1, 2, 100)
        )
        assert b2 <= b1 + 1e-10


# =========================================================================
# Section 6: Grid Operations
# =========================================================================


class TestGridOperations:
    """Tests for grid construction and manipulation."""

    def test_initial_grid(self, small_spec):
        """Initial grid covers query range with margin."""
        grid = _build_initial_grid(small_spec, k0=10, margin=1.0)
        assert len(grid) == 10
        assert grid[0] <= small_spec.query_values.min()
        assert grid[-1] >= small_spec.query_values.max()
        # Sorted
        assert all(grid[i] < grid[i + 1] for i in range(len(grid) - 1))

    def test_insert_point_new(self):
        """Inserting a new point increases grid size."""
        grid = np.array([0.0, 1.0, 2.0])
        new_grid, inserted = _insert_point(grid, 1.5)
        assert inserted
        assert len(new_grid) == 4
        assert 1.5 in new_grid

    def test_insert_point_duplicate(self):
        """Inserting existing point (too close) doesn't increase grid."""
        grid = np.array([0.0, 1.0, 2.0])
        new_grid, inserted = _insert_point(grid, 1.0, min_spacing=0.1)
        assert not inserted
        assert len(new_grid) == 3

    def test_insert_maintains_sorting(self):
        """Grid remains sorted after insertion."""
        grid = np.array([0.0, 1.0, 3.0, 5.0])
        new_grid, _ = _insert_point(grid, 2.0)
        assert all(new_grid[i] < new_grid[i + 1] for i in range(len(new_grid) - 1))


# =========================================================================
# Section 7: Infinite LP Solver (small-scale)
# =========================================================================


@pytest.mark.slow
class TestInfiniteLPSolver:
    """Tests for the full cutting-plane solver on small instances."""

    def test_solver_returns_result(self, small_spec):
        """Solver returns InfiniteLPResult."""
        solver = InfiniteLPSolver(
            initial_k=5, max_iter=10, target_tol=1e-2, verbose=0,
        )
        result = solver.solve(small_spec)
        assert isinstance(result, InfiniteLPResult)
        assert result.mechanism.shape[0] == small_spec.n
        assert result.iterations >= 1

    def test_mechanism_normalized(self, small_spec):
        """Solver output mechanism rows sum to 1."""
        solver = InfiniteLPSolver(
            initial_k=5, max_iter=10, target_tol=1e-2, verbose=0,
        )
        result = solver.solve(small_spec)
        row_sums = result.mechanism.sum(axis=1)
        np.testing.assert_array_almost_equal(row_sums, np.ones(small_spec.n), decimal=4)

    def test_duality_gap_decreases(self, small_spec):
        """Duality gap should be non-negative and small after solving."""
        solver = InfiniteLPSolver(
            initial_k=5, max_iter=20, target_tol=1e-3, verbose=0,
        )
        result = solver.solve(small_spec)
        assert result.duality_gap >= -1e-6
        if result.iterations > 5:
            assert result.duality_gap < 1.0  # Should make some progress

    def test_grid_grows(self, small_spec):
        """Grid size grows across iterations."""
        solver = InfiniteLPSolver(
            initial_k=5, max_iter=5, target_tol=1e-10, verbose=0,
        )
        result = solver.solve(small_spec)
        if len(result.grid_history) > 1:
            assert result.grid_history[-1] >= result.grid_history[0]

    def test_solve_functional_api(self, small_spec):
        """solve_infinite_lp functional API works."""
        result = solve_infinite_lp(
            small_spec, target_tol=1e-2, initial_k=5, max_iter=10, verbose=0,
        )
        assert isinstance(result, InfiniteLPResult)


# =========================================================================
# Section 8: Edge Cases
# =========================================================================


class TestInfiniteEdgeCases:
    """Edge case tests for infinite-dimensional module."""

    def test_1point_grid(self):
        """1-point grid still produces a valid initial grid."""
        spec = QuerySpec(
            query_values=np.array([0.0, 1.0]),
            domain="test", sensitivity=1.0,
            epsilon=1.0, k=2,
        )
        grid = _build_initial_grid(spec, k0=2, margin=1.0)
        assert len(grid) >= 2

    def test_tight_tolerance_monitor(self):
        """Very tight tolerance still terminates via max_iter."""
        mon = ConvergenceMonitor(target_tol=1e-15, max_iter=3)
        for _ in range(4):
            mon.update(2.0, 1.0, 10, 0.5)
        assert mon.should_terminate()

    def test_wasserstein_p2(self):
        """Wasserstein with p=2."""
        transport = DPTransport(p=2.0)
        grid = np.linspace(0, 1, 10)
        d1 = np.ones(10) / 10
        d2 = np.zeros(10)
        d2[-1] = 1.0
        w = transport.wasserstein(d1, d2, grid)
        assert w >= 0

    def test_continuous_mechanism_bandwidth(self):
        """Custom bandwidth is respected."""
        grid = np.linspace(0, 1, 10)
        weights = np.ones(10) / 10
        mech = ContinuousMechanism(y_grid=grid, weights=weights, bandwidth=0.5)
        assert abs(mech.bandwidth - 0.5) < 1e-15  # type: ignore
