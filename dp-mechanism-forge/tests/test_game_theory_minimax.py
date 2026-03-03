"""Tests for game_theory.minimax module."""
import math
import numpy as np
import pytest

from dp_forge.game_theory.minimax import (
    MinimaxSolver,
    SaddlePointComputation,
    MaxMinFair,
    WorstCaseDataset,
    RobustOptimization,
    MinimaxLPFormulation,
)
from dp_forge.types import AdjacencyRelation, GameMatrix


# ── MinimaxSolver ──────────────────────────────────────────────────


class TestMinimaxSolver:
    """Test MinimaxSolver finds saddle points and game values."""

    def test_identity_game(self):
        """Identity matrix: value = 1, uniform strategies."""
        A = np.eye(3)
        solver = MinimaxSolver()
        result = solver.solve(GameMatrix(payoffs=A))
        # For identity game, minimax value should be 1/3
        # (uniform strategy gives expected payoff 1/3)
        assert result.equilibrium.game_value == pytest.approx(1.0 / 3.0, abs=1e-4)

    def test_zero_game(self):
        """All-zeros game: value = 0."""
        A = np.zeros((3, 3))
        solver = MinimaxSolver()
        result = solver.solve(GameMatrix(payoffs=A))
        assert result.equilibrium.game_value == pytest.approx(0.0, abs=1e-6)

    def test_constant_game(self):
        """Constant game: value = constant."""
        c = 5.0
        A = np.full((4, 4), c)
        solver = MinimaxSolver()
        result = solver.solve(GameMatrix(payoffs=A))
        assert result.equilibrium.game_value == pytest.approx(c, abs=1e-4)

    def test_dominated_strategy_game(self):
        """Game with a dominated strategy: solver avoids it."""
        A = np.array([[3.0, 0.0], [5.0, 1.0]])
        solver = MinimaxSolver()
        result = solver.solve(GameMatrix(payoffs=A))
        # Row 1 dominates row 0 (5>3, 1>0), so adversary puts weight on row 1
        assert result.equilibrium.game_value >= 1.0 - 1e-6

    def test_2x2_known_game(self):
        """Known 2x2 game: [[3,1],[0,2]]."""
        A = np.array([[3.0, 1.0], [0.0, 2.0]])
        solver = MinimaxSolver()
        result = solver.solve(GameMatrix(payoffs=A))
        # Value of this game = 3/2 (computed analytically)
        assert result.equilibrium.game_value == pytest.approx(1.5, abs=0.1)

    def test_strategies_are_distributions(self):
        """Verify strategies sum to 1 and are nonnegative."""
        A = np.array([[1.0, 3.0], [2.0, 0.0]])
        solver = MinimaxSolver()
        result = solver.solve(GameMatrix(payoffs=A))
        d = result.equilibrium.designer_strategy.probabilities
        a = result.equilibrium.adversary_strategy.probabilities
        assert np.all(d >= -1e-8)
        assert np.all(a >= -1e-8)
        assert np.sum(d) == pytest.approx(1.0, abs=1e-6)
        assert np.sum(a) == pytest.approx(1.0, abs=1e-6)

    def test_minimax_theorem_numerical(self):
        """Verify max_min = min_max (von Neumann's minimax theorem)."""
        A = np.array([[4.0, 1.0, 2.0], [3.0, 5.0, 0.0], [1.0, 2.0, 3.0]])
        solver = MinimaxSolver()
        result = solver.solve(GameMatrix(payoffs=A))
        cert = result.optimality_certificate
        assert cert is not None
        # Duality gap should be small
        assert cert.duality_gap < 1e-4
        assert abs(cert.primal_obj - cert.dual_obj) < 1e-4

    def test_worst_case_pair_returned(self):
        """Worst-case pair should be a valid tuple of row indices."""
        A = np.array([[1.0, 2.0], [3.0, 4.0], [0.0, 5.0]])
        solver = MinimaxSolver()
        result = solver.solve(GameMatrix(payoffs=A))
        i, j = result.worst_case_pair
        assert 0 <= i < 3
        assert 0 <= j < 3
        assert i != j

    def test_non_square_game(self):
        """Non-square payoff matrix (more rows than columns)."""
        A = np.array([[1.0, 2.0], [3.0, 0.0], [2.0, 1.0]])
        solver = MinimaxSolver()
        result = solver.solve(GameMatrix(payoffs=A))
        assert isinstance(result.equilibrium.game_value, float)

    def test_negative_payoffs(self):
        """Game with negative entries."""
        A = np.array([[-1.0, -3.0], [-2.0, -4.0]])
        solver = MinimaxSolver()
        result = solver.solve(GameMatrix(payoffs=A))
        assert result.equilibrium.game_value < 0

    def test_large_game(self):
        """Larger game (10x10) should solve without error."""
        rng = np.random.default_rng(42)
        A = rng.standard_normal((10, 10))
        solver = MinimaxSolver()
        result = solver.solve(GameMatrix(payoffs=A))
        assert abs(result.optimality_certificate.duality_gap) < 0.01


# ── SaddlePointComputation ─────────────────────────────────────────


class TestSaddlePointComputation:
    """Test SaddlePointComputation LP duality and pure saddle points."""

    def test_pure_saddle_point_exists(self):
        """Matrix with a clear saddle point."""
        A = np.array([[3.0, 1.0, 2.0], [4.0, 2.0, 3.0], [1.0, 0.0, 1.0]])
        sp = SaddlePointComputation()
        points = sp.find_pure_saddle_points(A)
        # (1,1) is a saddle: row 1 min is 2 at col 1, col 1 max is 2 at row 1
        assert (1, 1) in points

    def test_no_pure_saddle_point(self):
        """Matching pennies: no pure saddle point."""
        A = np.array([[1.0, -1.0], [-1.0, 1.0]])
        sp = SaddlePointComputation()
        points = sp.find_pure_saddle_points(A)
        assert len(points) == 0

    def test_mixed_saddle_point(self):
        """Mixed saddle point for matching pennies: value = 0."""
        A = np.array([[1.0, -1.0], [-1.0, 1.0]])
        sp = SaddlePointComputation()
        x, y, val = sp.find_mixed_saddle_point(A)
        assert val == pytest.approx(0.0, abs=0.1)
        assert np.sum(x) == pytest.approx(1.0, abs=1e-6)
        assert np.sum(y) == pytest.approx(1.0, abs=1e-6)

    def test_verify_saddle_point_valid(self):
        """Verify known saddle point is accepted."""
        A = np.array([[1.0, -1.0], [-1.0, 1.0]])
        x = np.array([0.5, 0.5])
        y = np.array([0.5, 0.5])
        sp = SaddlePointComputation()
        assert sp.verify_saddle_point(A, x, y) is True

    def test_verify_saddle_point_invalid(self):
        """Verify non-saddle-point is rejected."""
        A = np.array([[1.0, -1.0], [-1.0, 1.0]])
        x = np.array([1.0, 0.0])
        y = np.array([1.0, 0.0])
        sp = SaddlePointComputation()
        # x^T A y = 1, but max_i (Ay)_i = 1 and min_j (x^T A)_j = -1
        assert sp.verify_saddle_point(A, x, y, tol=0.01) is False


# ── MaxMinFair ──────────────────────────────────────────────────────


class TestMaxMinFair:
    """Test MaxMinFair allocation."""

    def test_equal_sensitivities(self):
        """Equal sensitivities → equal allocations."""
        s = np.array([1.0, 1.0, 1.0])
        alloc = MaxMinFair().allocate(s, total_epsilon=3.0)
        np.testing.assert_allclose(alloc, [1.0, 1.0, 1.0], atol=1e-6)

    def test_budget_sums_correctly(self):
        """Allocations should sum to total_epsilon."""
        s = np.array([1.0, 2.0, 3.0])
        alloc = MaxMinFair().allocate(s, total_epsilon=6.0)
        assert np.sum(alloc) == pytest.approx(6.0, abs=1e-6)

    def test_proportional_to_sensitivity(self):
        """Allocation proportional to sensitivity."""
        s = np.array([1.0, 2.0])
        alloc = MaxMinFair().allocate(s, total_epsilon=3.0)
        assert alloc[1] / alloc[0] == pytest.approx(2.0, abs=1e-6)

    def test_weighted_allocation(self):
        """Weighted allocation sums to total epsilon."""
        s = np.array([1.0, 1.0])
        w = np.array([2.0, 1.0])
        alloc = MaxMinFair().allocate_with_weights(s, w, total_epsilon=3.0)
        assert np.sum(alloc) == pytest.approx(3.0, abs=1e-6)
        assert alloc[0] > alloc[1]  # higher weight gets more budget

    def test_fairness_index_equal(self):
        """Equal allocations have Jain index 1."""
        alloc = np.array([1.0, 1.0, 1.0])
        idx = MaxMinFair().compute_fairness_index(alloc)
        assert idx == pytest.approx(1.0, abs=1e-10)

    def test_fairness_index_unequal(self):
        """Unequal allocations have Jain index < 1."""
        alloc = np.array([1.0, 0.0, 0.0])
        idx = MaxMinFair().compute_fairness_index(alloc)
        assert idx < 1.0
        assert idx > 0.0

    def test_empty_sensitivities(self):
        """Empty sensitivities return empty allocation."""
        alloc = MaxMinFair().allocate(np.array([]), total_epsilon=1.0)
        assert len(alloc) == 0


# ── WorstCaseDataset ────────────────────────────────────────────────


class TestWorstCaseDataset:
    """Test WorstCaseDataset identification."""

    def test_find_worst_pair_uniform(self):
        """Uniform mechanism: worst pair should have measurable loss."""
        P = np.array([[0.5, 0.5], [0.5, 0.5]])
        adj = AdjacencyRelation(edges=[(0, 1)], n=2, symmetric=True)
        wcd = WorstCaseDataset()
        pair, loss = wcd.find_worst_pair(P, adj)
        assert loss == pytest.approx(0.0, abs=1e-6)

    def test_find_worst_pair_skewed(self):
        """Skewed mechanism: worst pair has positive loss."""
        P = np.array([[0.9, 0.1], [0.1, 0.9]])
        adj = AdjacencyRelation(edges=[(0, 1)], n=2, symmetric=True)
        wcd = WorstCaseDataset()
        pair, loss = wcd.find_worst_pair(P, adj)
        assert loss > 0

    def test_privacy_loss_distribution(self):
        """PLD shape matches output dimension."""
        P = np.array([[0.7, 0.3], [0.4, 0.6]])
        wcd = WorstCaseDataset()
        pld = wcd.privacy_loss_distribution(P, 0, 1)
        assert pld.shape == (2,)

    def test_hockey_stick_divergence_nonneg(self):
        """Hockey-stick divergence >= 0."""
        P = np.array([[0.6, 0.4], [0.5, 0.5]])
        wcd = WorstCaseDataset()
        hsd = wcd.hockey_stick_divergence(P, 0, 1, epsilon=1.0)
        assert hsd >= -1e-10

    def test_hockey_stick_divergence_identical(self):
        """Identical rows: HSD = 0 for any eps."""
        P = np.array([[0.5, 0.5], [0.5, 0.5]])
        wcd = WorstCaseDataset()
        hsd = wcd.hockey_stick_divergence(P, 0, 1, epsilon=0.0)
        assert hsd == pytest.approx(0.0, abs=1e-10)


# ── RobustOptimization ─────────────────────────────────────────────


class TestRobustOptimization:
    """Test RobustOptimization with different uncertainty sets."""

    def test_box_robust_feasible(self):
        """Box-robust solution exists."""
        A = np.array([[2.0, 1.0], [1.0, 3.0]])
        delta = np.ones_like(A) * 0.1
        rob = RobustOptimization()
        strat, val = rob.solve_box_robust(A, delta)
        assert np.sum(strat) == pytest.approx(1.0, abs=1e-6)
        assert np.all(strat >= -1e-8)

    def test_box_robust_higher_than_nominal(self):
        """Robust value should be >= nominal value."""
        A = np.array([[2.0, 1.0], [1.0, 3.0]])
        delta = np.ones_like(A) * 0.5
        rob = RobustOptimization()
        _, robust_val = rob.solve_box_robust(A, delta)
        solver = MinimaxSolver()
        nominal = solver.solve(GameMatrix(payoffs=A)).equilibrium.game_value
        assert robust_val >= nominal - 1e-4

    def test_budget_robust(self):
        """Budget-robust solution returns valid strategy."""
        A = np.array([[2.0, 1.0], [1.0, 2.0]])
        dev = np.ones_like(A) * 0.3
        rob = RobustOptimization()
        strat, val = rob.solve_budget_robust(A, dev, gamma=1.0)
        assert np.all(strat >= -1e-8)
        assert np.sum(strat) == pytest.approx(1.0, abs=1e-6)

    def test_ellipsoidal_robust(self):
        """Ellipsoidal robust value increases with rho."""
        A = np.array([[2.0, 1.0], [1.0, 2.0]])
        rob = RobustOptimization()
        _, val_small = rob.solve_ellipsoidal_robust(A, rho=0.01)
        _, val_large = rob.solve_ellipsoidal_robust(A, rho=1.0)
        assert val_large >= val_small - 1e-6

    def test_sensitivity_analysis(self):
        """Sensitivity analysis returns reasonable statistics."""
        A = np.array([[2.0, 1.0], [1.0, 2.0]])
        rob = RobustOptimization()
        stats = rob.sensitivity_analysis(A, perturbation_range=0.1, n_samples=10)
        assert "mean" in stats
        assert "std" in stats
        assert stats["std"] >= 0
        assert len(stats["values"]) == 10


# ── MinimaxLPFormulation ────────────────────────────────────────────


class TestMinimaxLPFormulation:
    """Test MinimaxLPFormulation builds and solves correct LPs."""

    def test_build_lp_dimensions(self):
        """LP dimensions match expected sizes."""
        adj = AdjacencyRelation(edges=[(0, 1)], n=2, symmetric=True)
        lp_form = MinimaxLPFormulation(epsilon=1.0)
        data = lp_form.build(
            query_values=np.array([0.0, 1.0]),
            output_grid=np.array([0.0, 0.5, 1.0]),
            adjacency=adj,
        )
        assert data["n_databases"] == 2
        assert data["n_outputs"] == 3

    def test_solve_returns_mechanism(self):
        """Solve returns a valid mechanism (rows sum to 1)."""
        adj = AdjacencyRelation(edges=[(0, 1)], n=2, symmetric=True)
        lp_form = MinimaxLPFormulation(epsilon=1.0)
        P, loss = lp_form.solve(
            query_values=np.array([0.0, 1.0]),
            output_grid=np.array([0.0, 0.5, 1.0]),
            adjacency=adj,
        )
        assert P.shape == (2, 3)
        np.testing.assert_allclose(P.sum(axis=1), [1.0, 1.0], atol=1e-4)
        assert np.all(P >= -1e-6)

    def test_dp_constraint_satisfied(self):
        """Solved mechanism satisfies ε-DP constraint."""
        eps = 1.0
        adj = AdjacencyRelation(edges=[(0, 1)], n=2, symmetric=True)
        lp_form = MinimaxLPFormulation(epsilon=eps)
        P, _ = lp_form.solve(
            query_values=np.array([0.0, 1.0]),
            output_grid=np.array([0.0, 0.5, 1.0]),
            adjacency=adj,
        )
        e_eps = np.exp(eps)
        for j in range(P.shape[1]):
            if P[1, j] > 1e-10:
                assert P[0, j] / P[1, j] <= e_eps + 0.01
            if P[0, j] > 1e-10:
                assert P[1, j] / P[0, j] <= e_eps + 0.01
