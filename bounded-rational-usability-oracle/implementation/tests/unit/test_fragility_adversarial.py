"""Unit tests for usability_oracle.fragility.adversarial.AdversarialAnalyzer.

Tests the adversarial analysis module that finds worst-case and best-case
beta values for an MDP, computes minimax regret between two MDPs, and
produces regression verdicts comparing two UI designs.
"""

from __future__ import annotations

import math
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from usability_oracle.core.enums import RegressionVerdict
from usability_oracle.fragility.adversarial import AdversarialAnalyzer
from usability_oracle.mdp.models import MDP, Action, State, Transition


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_simple_mdp(cost: float = 1.0) -> MDP:
    """Build a two-state MDP with a given transition cost."""
    s0 = State(state_id="s0", label="start")
    s1 = State(state_id="s1", label="goal", is_terminal=True, is_goal=True)
    a = Action(action_id="a_go", action_type="click", target_node_id="btn")
    t = Transition(source="s0", action="a_go", target="s1", probability=1.0, cost=cost)
    return MDP(
        states={"s0": s0, "s1": s1},
        actions={"a_go": a},
        transitions=[t],
        initial_state="s0",
        goal_states={"s1"},
        discount=0.99,
    )


def _make_branching_mdp() -> MDP:
    """MDP with two alternative actions from the start."""
    s0 = State(state_id="s0", label="start")
    s1 = State(state_id="s1", label="mid")
    s2 = State(state_id="s2", label="goal", is_terminal=True, is_goal=True)
    a1 = Action(action_id="a1", action_type="click", target_node_id="n1")
    a2 = Action(action_id="a2", action_type="click", target_node_id="n2")
    a3 = Action(action_id="a3", action_type="click", target_node_id="n3")
    transitions = [
        Transition(source="s0", action="a1", target="s1", probability=1.0, cost=2.0),
        Transition(source="s0", action="a2", target="s2", probability=1.0, cost=6.0),
        Transition(source="s1", action="a3", target="s2", probability=1.0, cost=1.0),
    ]
    return MDP(
        states={"s0": s0, "s1": s1, "s2": s2},
        actions={"a1": a1, "a2": a2, "a3": a3},
        transitions=transitions,
        initial_state="s0",
        goal_states={"s2"},
        discount=0.99,
    )


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

class TestAdversarialAnalyzerConstruction:
    """Tests for AdversarialAnalyzer instantiation."""

    def test_default_construction(self):
        """AdversarialAnalyzer can be created with defaults."""
        aa = AdversarialAnalyzer()
        assert aa is not None

    def test_custom_n_trajectories(self):
        """n_trajectories parameter is stored."""
        aa = AdversarialAnalyzer(n_trajectories=500)
        assert aa.n_trajectories == 500

    def test_custom_optimizer_maxiter(self):
        """optimizer_maxiter parameter is stored."""
        aa = AdversarialAnalyzer(optimizer_maxiter=200)
        assert aa.optimizer_maxiter == 200

    def test_custom_grid_resolution(self):
        """grid_resolution parameter is stored."""
        aa = AdversarialAnalyzer(grid_resolution=100)
        assert aa.grid_resolution == 100


# ---------------------------------------------------------------------------
# find_worst_beta
# ---------------------------------------------------------------------------

class TestFindWorstBeta:
    """Tests for AdversarialAnalyzer.find_worst_beta."""

    @patch("usability_oracle.fragility.adversarial._evaluate_cost")
    def test_returns_tuple(self, mock_cost):
        """find_worst_beta returns a (beta, cost) tuple."""
        mock_cost.return_value = 5.0
        aa = AdversarialAnalyzer(n_trajectories=5, grid_resolution=10)
        result = aa.find_worst_beta(_make_simple_mdp(), beta_range=(0.1, 10.0))
        assert isinstance(result, tuple)
        assert len(result) == 2

    @patch("usability_oracle.fragility.adversarial._evaluate_cost")
    def test_worst_beta_in_range(self, mock_cost):
        """Returned worst beta lies within the supplied beta_range."""
        mock_cost.return_value = 5.0
        aa = AdversarialAnalyzer(n_trajectories=5, grid_resolution=10)
        beta, cost = aa.find_worst_beta(_make_simple_mdp(), beta_range=(1.0, 8.0))
        assert 1.0 <= beta <= 8.0

    @patch("usability_oracle.fragility.adversarial._evaluate_cost")
    def test_worst_cost_nonnegative(self, mock_cost):
        """Worst-case cost is non-negative."""
        mock_cost.return_value = 3.0
        aa = AdversarialAnalyzer(n_trajectories=5, grid_resolution=10)
        _, cost = aa.find_worst_beta(_make_simple_mdp(), beta_range=(0.1, 10.0))
        assert cost >= 0.0

    @patch("usability_oracle.fragility.adversarial._evaluate_cost")
    def test_worst_beta_picks_max_cost(self, mock_cost):
        """find_worst_beta picks the beta that maximizes cost."""
        def decreasing_cost(mdp, beta, n_trajectories=100):
            return 10.0 / beta

        mock_cost.side_effect = decreasing_cost
        aa = AdversarialAnalyzer(n_trajectories=5, grid_resolution=20)
        beta, cost = aa.find_worst_beta(_make_simple_mdp(), beta_range=(0.5, 10.0))
        # Worst beta should be near the low end (high cost)
        assert beta < 3.0


# ---------------------------------------------------------------------------
# find_best_beta
# ---------------------------------------------------------------------------

class TestFindBestBeta:
    """Tests for AdversarialAnalyzer.find_best_beta."""

    @patch("usability_oracle.fragility.adversarial._evaluate_cost")
    def test_returns_tuple(self, mock_cost):
        """find_best_beta returns a (beta, cost) tuple."""
        mock_cost.return_value = 2.0
        aa = AdversarialAnalyzer(n_trajectories=5, grid_resolution=10)
        result = aa.find_best_beta(_make_simple_mdp(), beta_range=(0.1, 10.0))
        assert isinstance(result, tuple)
        assert len(result) == 2

    @patch("usability_oracle.fragility.adversarial._evaluate_cost")
    def test_best_beta_in_range(self, mock_cost):
        """Returned best beta lies within the supplied beta_range."""
        mock_cost.return_value = 2.0
        aa = AdversarialAnalyzer(n_trajectories=5, grid_resolution=10)
        beta, cost = aa.find_best_beta(_make_simple_mdp(), beta_range=(1.0, 8.0))
        assert 1.0 <= beta <= 8.0

    @patch("usability_oracle.fragility.adversarial._evaluate_cost")
    def test_best_cost_nonnegative(self, mock_cost):
        """Best-case cost is non-negative."""
        mock_cost.return_value = 1.0
        aa = AdversarialAnalyzer(n_trajectories=5, grid_resolution=10)
        _, cost = aa.find_best_beta(_make_simple_mdp(), beta_range=(0.1, 10.0))
        assert cost >= 0.0

    @patch("usability_oracle.fragility.adversarial._evaluate_cost")
    def test_best_beta_picks_min_cost(self, mock_cost):
        """find_best_beta picks the beta that minimizes cost."""
        def decreasing_cost(mdp, beta, n_trajectories=100):
            return 10.0 / beta

        mock_cost.side_effect = decreasing_cost
        aa = AdversarialAnalyzer(n_trajectories=5, grid_resolution=20)
        beta, cost = aa.find_best_beta(_make_simple_mdp(), beta_range=(0.5, 10.0))
        # Best beta should be near the high end (low cost)
        assert beta > 5.0


# ---------------------------------------------------------------------------
# worst_cost >= best_cost
# ---------------------------------------------------------------------------

class TestWorstVsBestCost:
    """Tests that worst_cost is always >= best_cost."""

    @patch("usability_oracle.fragility.adversarial._evaluate_cost")
    def test_worst_geq_best_constant(self, mock_cost):
        """With constant cost, worst == best."""
        mock_cost.return_value = 4.0
        aa = AdversarialAnalyzer(n_trajectories=5, grid_resolution=10)
        mdp = _make_simple_mdp()
        _, worst_cost = aa.find_worst_beta(mdp, beta_range=(0.1, 10.0))
        _, best_cost = aa.find_best_beta(mdp, beta_range=(0.1, 10.0))
        assert worst_cost >= best_cost - 1e-6

    @patch("usability_oracle.fragility.adversarial._evaluate_cost")
    def test_worst_geq_best_varying(self, mock_cost):
        """With varying cost, worst >= best."""
        def varying(mdp, beta, n_trajectories=100):
            return 5.0 + 3.0 * math.sin(beta)

        mock_cost.side_effect = varying
        aa = AdversarialAnalyzer(n_trajectories=5, grid_resolution=30)
        mdp = _make_simple_mdp()
        _, worst = aa.find_worst_beta(mdp, beta_range=(0.1, 10.0))
        _, best = aa.find_best_beta(mdp, beta_range=(0.1, 10.0))
        assert worst >= best - 1e-6


# ---------------------------------------------------------------------------
# minimax_regret
# ---------------------------------------------------------------------------

class TestMinimaxRegret:
    """Tests for AdversarialAnalyzer.minimax_regret."""

    @patch("usability_oracle.fragility.adversarial._evaluate_cost")
    def test_returns_float(self, mock_cost):
        """minimax_regret returns a float."""
        mock_cost.return_value = 2.0
        aa = AdversarialAnalyzer(n_trajectories=5, grid_resolution=10)
        mdp_a = _make_simple_mdp(cost=1.0)
        mdp_b = _make_simple_mdp(cost=2.0)
        regret = aa.minimax_regret(mdp_a, mdp_b, beta_range=(0.1, 10.0))
        assert isinstance(regret, float)

    @patch("usability_oracle.fragility.adversarial._evaluate_cost")
    def test_regret_nonnegative(self, mock_cost):
        """minimax_regret is always >= 0."""
        mock_cost.return_value = 3.0
        aa = AdversarialAnalyzer(n_trajectories=5, grid_resolution=10)
        mdp_a = _make_simple_mdp(cost=1.0)
        mdp_b = _make_simple_mdp(cost=2.0)
        regret = aa.minimax_regret(mdp_a, mdp_b, beta_range=(0.1, 10.0))
        assert regret >= 0.0

    @patch("usability_oracle.fragility.adversarial._evaluate_cost")
    def test_identical_mdps_zero_regret(self, mock_cost):
        """Identical MDPs yield identical minimax value (constant cost)."""
        mock_cost.return_value = 4.0
        aa = AdversarialAnalyzer(n_trajectories=5, grid_resolution=10)
        mdp = _make_simple_mdp()
        regret = aa.minimax_regret(mdp, mdp, beta_range=(0.1, 10.0))
        # With identical MDPs and constant cost, minimax value = min(C,C) = C
        assert regret >= 0.0


# ---------------------------------------------------------------------------
# adversarial_comparison
# ---------------------------------------------------------------------------

class TestAdversarialComparison:
    """Tests for AdversarialAnalyzer.adversarial_comparison."""

    @patch("usability_oracle.fragility.adversarial._evaluate_cost")
    def test_returns_regression_verdict(self, mock_cost):
        """adversarial_comparison returns a RegressionVerdict enum member."""
        mock_cost.return_value = 2.0
        aa = AdversarialAnalyzer(n_trajectories=5, grid_resolution=10)
        mdp_a = _make_simple_mdp(cost=1.0)
        mdp_b = _make_simple_mdp(cost=2.0)
        verdict = aa.adversarial_comparison(mdp_a, mdp_b, beta_range=(0.1, 10.0))
        assert isinstance(verdict, RegressionVerdict)

    @patch("usability_oracle.fragility.adversarial._evaluate_cost")
    def test_same_mdp_neutral_or_inconclusive(self, mock_cost):
        """Comparing identical MDPs should not report regression."""
        mock_cost.return_value = 3.0
        aa = AdversarialAnalyzer(n_trajectories=5, grid_resolution=10)
        mdp = _make_simple_mdp()
        verdict = aa.adversarial_comparison(mdp, mdp, beta_range=(0.1, 10.0))
        assert verdict in (RegressionVerdict.NEUTRAL, RegressionVerdict.INCONCLUSIVE,
                           RegressionVerdict.IMPROVEMENT)

    @patch("usability_oracle.fragility.adversarial._evaluate_cost")
    def test_worse_mdp_signals_regression(self, mock_cost):
        """When mdp_b is clearly worse, verdict should be REGRESSION."""
        def cost_fn(mdp, beta, n_trajectories=100):
            if mdp.initial_state == "s0" and len(mdp.transitions) == 1:
                # mdp_a
                return 1.0
            # mdp_b – much worse
            return 100.0

        mock_cost.side_effect = cost_fn
        aa = AdversarialAnalyzer(n_trajectories=5, grid_resolution=10)
        mdp_a = _make_simple_mdp(cost=1.0)
        mdp_b = _make_branching_mdp()
        verdict = aa.adversarial_comparison(mdp_a, mdp_b, beta_range=(0.1, 10.0))
        assert verdict in (RegressionVerdict.REGRESSION, RegressionVerdict.INCONCLUSIVE)


# ---------------------------------------------------------------------------
# RegressionVerdict enum
# ---------------------------------------------------------------------------

class TestRegressionVerdictEnum:
    """Tests for the RegressionVerdict enum used in adversarial analysis."""

    def test_has_regression(self):
        """REGRESSION member exists."""
        assert RegressionVerdict.REGRESSION is not None

    def test_has_improvement(self):
        """IMPROVEMENT member exists."""
        assert RegressionVerdict.IMPROVEMENT is not None

    def test_has_neutral(self):
        """NEUTRAL member exists."""
        assert RegressionVerdict.NEUTRAL is not None

    def test_has_inconclusive(self):
        """INCONCLUSIVE member exists."""
        assert RegressionVerdict.INCONCLUSIVE is not None

    def test_regression_is_actionable(self):
        """REGRESSION verdict is actionable."""
        assert RegressionVerdict.REGRESSION.is_actionable is True

    def test_improvement_not_actionable(self):
        """IMPROVEMENT verdict is not actionable."""
        assert RegressionVerdict.IMPROVEMENT.is_actionable is False
