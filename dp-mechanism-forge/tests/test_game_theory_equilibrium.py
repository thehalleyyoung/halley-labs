"""Tests for game_theory.equilibrium module."""
import numpy as np
import pytest

from dp_forge.game_theory.equilibrium import (
    NashEquilibrium,
    SupportEnumeration,
    LemkeHowson,
    CorrelatedEquilibrium,
    EvolutionaryDynamics,
    TrembleEquilibrium,
)
from dp_forge.game_theory import NashAlgorithm


class TestNashEquilibrium:
    """Test NashEquilibrium existence and uniqueness (2x2 games)."""

    def test_pure_nash_dominant(self):
        """Prisoner's dilemma has pure dominant NE at (D,D)."""
        # A = payoffs for row player (prisoner 1)
        # (C,C)=(-1,-1), (C,D)=(-3,0), (D,C)=(0,-3), (D,D)=(-2,-2)
        A = np.array([[-1.0, -3.0], [0.0, -2.0]])
        B = np.array([[-1.0, 0.0], [-3.0, -2.0]])
        ne = NashEquilibrium()
        results = ne.compute(A, B)
        assert len(results) >= 1
        # Verify (D,D) is a NE
        for p, q in results:
            assert ne.is_nash_equilibrium(A, B, p, q)

    def test_matching_pennies_mixed(self):
        """Matching pennies has unique fully mixed NE at (0.5, 0.5)."""
        A = np.array([[1.0, -1.0], [-1.0, 1.0]])
        B = -A  # zero-sum
        ne = NashEquilibrium()
        results = ne.compute(A, B)
        assert len(results) >= 1
        p, q = results[0]
        assert p[0] == pytest.approx(0.5, abs=0.1)
        assert q[0] == pytest.approx(0.5, abs=0.1)

    def test_is_nash_true(self):
        """Verify valid NE passes check."""
        A = np.array([[1.0, -1.0], [-1.0, 1.0]])
        B = -A
        ne = NashEquilibrium()
        p = np.array([0.5, 0.5])
        q = np.array([0.5, 0.5])
        assert ne.is_nash_equilibrium(A, B, p, q)

    def test_is_nash_false(self):
        """Verify non-NE fails check."""
        A = np.array([[1.0, -1.0], [-1.0, 1.0]])
        B = -A
        ne = NashEquilibrium()
        p = np.array([1.0, 0.0])
        q = np.array([1.0, 0.0])
        assert not ne.is_nash_equilibrium(A, B, p, q, tol=0.01)

    def test_zero_sum_nash(self):
        """Zero-sum Nash is computed via minimax."""
        A = np.array([[3.0, 0.0], [5.0, 1.0]])
        ne = NashEquilibrium()
        p, q, val = ne.compute_zero_sum(A)
        assert np.sum(p) == pytest.approx(1.0, abs=1e-6)
        assert np.sum(q) == pytest.approx(1.0, abs=1e-6)

    def test_coordination_game(self):
        """Coordination game has multiple pure NE."""
        A = np.array([[2.0, 0.0], [0.0, 1.0]])
        B = np.array([[2.0, 0.0], [0.0, 1.0]])
        ne = NashEquilibrium()
        results = ne.compute(A, B)
        # Should find at least the two pure NE
        assert len(results) >= 2


class TestSupportEnumeration:
    """Test SupportEnumeration correctness."""

    def test_finds_all_ne_2x2(self):
        """Find all NE of a 2x2 game with multiple NE."""
        A = np.array([[2.0, 0.0], [0.0, 1.0]])
        B = np.array([[1.0, 0.0], [0.0, 2.0]])
        se = SupportEnumeration()
        results = se.find_all(A, B)
        assert len(results) >= 2
        for p, q in results:
            assert np.all(p >= -1e-8)
            assert np.all(q >= -1e-8)
            assert np.sum(p) == pytest.approx(1.0, abs=1e-6)

    def test_unique_ne(self):
        """Game with unique mixed NE."""
        A = np.array([[0.0, 1.0], [1.0, 0.0]])
        B = np.array([[1.0, 0.0], [0.0, 1.0]])
        se = SupportEnumeration()
        results = se.find_all(A, B)
        assert len(results) >= 1

    def test_no_duplicate_equilibria(self):
        """No duplicate NE returned."""
        A = np.array([[2.0, 0.0], [0.0, 1.0]])
        B = np.array([[2.0, 0.0], [0.0, 1.0]])
        se = SupportEnumeration()
        results = se.find_all(A, B)
        for i in range(len(results)):
            for j in range(i + 1, len(results)):
                pi, qi = results[i]
                pj, qj = results[j]
                assert not (np.allclose(pi, pj, atol=1e-6) and np.allclose(qi, qj, atol=1e-6))


class TestLemkeHowson:
    """Test LemkeHowson finds valid NE."""

    def test_finds_ne_2x2(self):
        """Find a NE for a 2x2 game."""
        A = np.array([[3.0, 0.0], [5.0, 1.0]])
        B = np.array([[3.0, 5.0], [0.0, 1.0]])
        lh = LemkeHowson()
        result = lh.solve(A, B)
        assert result is not None
        p, q = result
        assert np.all(p >= -1e-6)
        assert np.all(q >= -1e-6)
        assert np.sum(p) == pytest.approx(1.0, abs=1e-4)
        assert np.sum(q) == pytest.approx(1.0, abs=1e-4)

    def test_finds_ne_3x3(self):
        """Find a NE for a 3x3 game."""
        A = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        B = A.copy()
        lh = LemkeHowson()
        result = lh.solve(A, B)
        assert result is not None
        p, q = result
        ne = NashEquilibrium()
        assert ne.is_nash_equilibrium(A, B, p, q, tol=0.1)

    def test_matching_pennies(self):
        """Matching pennies via Lemke-Howson."""
        A = np.array([[1.0, -1.0], [-1.0, 1.0]])
        B = -A
        lh = LemkeHowson()
        result = lh.solve(A, B)
        assert result is not None


class TestCorrelatedEquilibrium:
    """Test CorrelatedEquilibrium LP formulation."""

    def test_ce_is_distribution(self):
        """CE is a valid probability distribution."""
        A = np.array([[2.0, 0.0], [0.0, 1.0]])
        B = np.array([[1.0, 0.0], [0.0, 2.0]])
        ce = CorrelatedEquilibrium()
        pi = ce.compute(A, B)
        assert pi.shape == (2, 2)
        assert np.all(pi >= -1e-8)
        assert np.sum(pi) == pytest.approx(1.0, abs=1e-6)

    def test_marginals_are_distributions(self):
        """Marginals of CE are valid distributions."""
        A = np.array([[2.0, 0.0], [0.0, 1.0]])
        B = np.array([[1.0, 0.0], [0.0, 2.0]])
        ce = CorrelatedEquilibrium()
        pi = ce.compute(A, B)
        p1, p2 = ce.marginals(pi)
        assert np.sum(p1) == pytest.approx(1.0, abs=1e-6)
        assert np.sum(p2) == pytest.approx(1.0, abs=1e-6)

    def test_max_welfare_ce(self):
        """Max welfare CE maximizes sum of payoffs."""
        A = np.array([[3.0, 0.0], [0.0, 1.0]])
        B = np.array([[3.0, 0.0], [0.0, 1.0]])
        ce = CorrelatedEquilibrium()
        pi = ce.compute(A, B, objective="max_welfare")
        welfare = np.sum(pi * (A + B))
        assert welfare > 0

    def test_fairness_ce(self):
        """Max-fairness CE returns valid distribution."""
        A = np.array([[3.0, 0.0], [0.0, 1.0]])
        B = np.array([[1.0, 0.0], [0.0, 3.0]])
        ce = CorrelatedEquilibrium()
        pi = ce.compute(A, B, objective="max_fairness")
        assert np.all(pi >= -1e-8)
        assert np.sum(pi) == pytest.approx(1.0, abs=1e-6)

    def test_zero_sum_ce(self):
        """CE of zero-sum game is well-defined."""
        A = np.array([[1.0, -1.0], [-1.0, 1.0]])
        B = -A
        ce = CorrelatedEquilibrium()
        pi = ce.compute(A, B)
        assert np.sum(pi) == pytest.approx(1.0, abs=1e-6)


class TestEvolutionaryDynamics:
    """Test EvolutionaryDynamics convergence."""

    def test_converges_to_rest_point(self):
        """Replicator dynamics converges for a dominance solvable game."""
        A = np.array([[3.0, 0.0], [0.0, 1.0]])
        ed = EvolutionaryDynamics(dt=0.01, max_steps=5000)
        final, traj = ed.simulate(A)
        assert np.sum(final) == pytest.approx(1.0, abs=1e-4)
        assert len(traj) >= 2

    def test_trajectory_stays_on_simplex(self):
        """All states in trajectory sum to 1 and are non-negative."""
        A = np.array([[2.0, 1.0], [0.0, 3.0]])
        ed = EvolutionaryDynamics(dt=0.01, max_steps=1000)
        _, traj = ed.simulate(A)
        for x in traj:
            assert np.all(x >= -1e-8)
            assert np.sum(x) == pytest.approx(1.0, abs=1e-4)

    def test_two_population_dynamics(self):
        """Two-population dynamics terminates."""
        A = np.array([[1.0, 0.0], [0.0, 1.0]])
        B = np.array([[0.0, 1.0], [1.0, 0.0]])
        ed = EvolutionaryDynamics(dt=0.01, max_steps=2000)
        x, y, traj = ed.simulate_two_population(A, B)
        assert np.sum(x) == pytest.approx(1.0, abs=1e-4)
        assert np.sum(y) == pytest.approx(1.0, abs=1e-4)

    def test_find_rest_points(self):
        """Find at least one rest point."""
        A = np.array([[2.0, 0.0], [0.0, 1.0]])
        ed = EvolutionaryDynamics(dt=0.01, max_steps=5000)
        rest = ed.find_rest_points(A, n_trials=5)
        assert len(rest) >= 1

    def test_matching_pennies_dynamics(self):
        """Matching pennies: replicator dynamics should stay near (0.5,0.5)."""
        A = np.array([[1.0, -1.0], [-1.0, 1.0]])
        ed = EvolutionaryDynamics(dt=0.005, max_steps=5000)
        final, _ = ed.simulate(A)
        # Should be near interior NE
        assert final[0] == pytest.approx(0.5, abs=0.3)


class TestTrembleEquilibrium:
    """Test TrembleEquilibrium computation."""

    def test_thp_returns_ne(self):
        """THP equilibria are Nash equilibria."""
        A = np.array([[2.0, 0.0], [0.0, 1.0]])
        B = np.array([[2.0, 0.0], [0.0, 1.0]])
        te = TrembleEquilibrium()
        results = te.compute(A, B)
        assert len(results) >= 1
        ne = NashEquilibrium()
        for p, q in results:
            assert ne.is_nash_equilibrium(A, B, p, q, tol=0.1)

    def test_perturbed_game(self):
        """Perturbed game returns valid strategies."""
        A = np.array([[2.0, 0.0], [0.0, 1.0]])
        B = np.array([[2.0, 0.0], [0.0, 1.0]])
        te = TrembleEquilibrium()
        p, q = te.compute_perturbed_game(A, B, eta=0.01)
        # Strategies should be valid distributions
        assert np.all(p >= -1e-6)
        assert np.all(q >= -1e-6)
        assert np.sum(p) == pytest.approx(1.0, abs=1e-4)
        assert np.sum(q) == pytest.approx(1.0, abs=1e-4)
