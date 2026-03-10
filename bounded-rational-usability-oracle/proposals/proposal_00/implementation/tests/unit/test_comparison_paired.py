"""Unit tests for usability_oracle.comparison.paired — PairedComparator.

Tests the paired usability comparison pipeline that constructs bounded-rational
policies for two MDP versions (before/after), evaluates task-completion costs,
and determines a regression verdict with statistical guarantees.

References
----------
- Ortega & Braun (2013). *Proc. R. Soc. A*, 469.
- Cohen, J. (1988). *Statistical Power Analysis*.
"""

from __future__ import annotations

import math
import pytest
import numpy as np

from usability_oracle.comparison.paired import PairedComparator, _solve_softmax_policy
from usability_oracle.comparison.models import (
    AlignmentResult,
    ComparisonResult,
    StateMapping,
    Partition,
    PartitionBlock,
    ComparisonContext,
)
from usability_oracle.core.enums import RegressionVerdict
from usability_oracle.cognitive.models import CostElement
from usability_oracle.mdp.models import MDP, State, Action, Transition
from usability_oracle.taskspec.models import TaskSpec

from tests.fixtures.sample_mdps import (
    make_two_state_mdp,
    make_cyclic_mdp,
    make_large_chain_mdp,
    make_choice_mdp,
)
from tests.fixtures.sample_tasks import make_login_task, make_search_task


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _identity_alignment(mdp: MDP) -> AlignmentResult:
    """Build an identity alignment mapping every state to itself."""
    mappings = [
        StateMapping(state_a=sid, state_b=sid, similarity=1.0, mapping_type="exact")
        for sid in mdp.states
    ]
    return AlignmentResult(
        mappings=mappings,
        unmapped_a=[],
        unmapped_b=[],
        overall_similarity=1.0,
    )


def _make_higher_cost_mdp() -> MDP:
    """Create a 2-state MDP with transition cost 5.0 (vs 1.0 baseline)."""
    states = {
        "start": State(state_id="start", features={"x": 0.0}, label="start",
                       is_terminal=False, is_goal=False, metadata={}),
        "mid": State(state_id="mid", features={"x": 0.5}, label="mid",
                     is_terminal=False, is_goal=False, metadata={}),
        "goal": State(state_id="goal", features={"x": 1.0}, label="goal",
                      is_terminal=True, is_goal=True, metadata={}),
    }
    actions = {
        "go": Action(action_id="go", action_type="click",
                     target_node_id="g", description="go", preconditions=[]),
    }
    transitions = [
        Transition(source="start", action="go", target="mid",
                   probability=0.5, cost=4.0),
        Transition(source="start", action="go", target="goal",
                   probability=0.5, cost=6.0),
        Transition(source="mid", action="go", target="goal",
                   probability=1.0, cost=5.0),
    ]
    return MDP(states=states, actions=actions, transitions=transitions,
               initial_state="start", goal_states={"goal"}, discount=0.99)


def _make_lower_cost_mdp() -> MDP:
    """Create a 2-state MDP with transition cost 0.1 (vs 1.0 baseline)."""
    states = {
        "start": State(state_id="start", features={"x": 0.0}, label="start",
                       is_terminal=False, is_goal=False, metadata={}),
        "mid": State(state_id="mid", features={"x": 0.5}, label="mid",
                     is_terminal=False, is_goal=False, metadata={}),
        "goal": State(state_id="goal", features={"x": 1.0}, label="goal",
                      is_terminal=True, is_goal=True, metadata={}),
    }
    actions = {
        "go": Action(action_id="go", action_type="click",
                     target_node_id="g", description="go", preconditions=[]),
    }
    transitions = [
        Transition(source="start", action="go", target="mid",
                   probability=0.5, cost=0.05),
        Transition(source="start", action="go", target="goal",
                   probability=0.5, cost=0.30),
        Transition(source="mid", action="go", target="goal",
                   probability=1.0, cost=0.02),
    ]
    return MDP(states=states, actions=actions, transitions=transitions,
               initial_state="start", goal_states={"goal"}, discount=0.99)


# ---------------------------------------------------------------------------
# Tests: PairedComparator instantiation
# ---------------------------------------------------------------------------


class TestPairedComparatorInit:
    """Tests for PairedComparator constructor and parameter handling."""

    def test_default_parameters(self):
        """PairedComparator uses sensible defaults: β=1.0, n_trajectories=500, α=0.05."""
        comp = PairedComparator()
        assert comp.beta == 1.0
        assert comp.n_trajectories == 500
        assert comp.significance_level == 0.05
        assert comp.min_effect_size == 0.2

    def test_custom_beta(self):
        """PairedComparator should accept a custom rationality parameter β."""
        comp = PairedComparator(beta=5.0)
        assert comp.beta == 5.0

    def test_custom_n_trajectories(self):
        """PairedComparator should accept a custom number of Monte Carlo trajectories."""
        comp = PairedComparator(n_trajectories=1000)
        assert comp.n_trajectories == 1000

    def test_custom_significance_level(self):
        """A smaller α requires stronger evidence to declare a regression."""
        comp = PairedComparator(significance_level=0.01)
        assert comp.significance_level == 0.01


# ---------------------------------------------------------------------------
# Tests: compare() — identical MDPs
# ---------------------------------------------------------------------------


class TestPairedComparatorIdentical:
    """Tests for comparing identical MDPs, which should yield NEUTRAL."""

    def test_identical_two_state_mdps_verdict(self):
        """Comparing identical two-state MDPs should yield NEUTRAL verdict."""
        mdp = make_two_state_mdp()
        alignment = _identity_alignment(mdp)
        task = make_login_task()
        comp = PairedComparator(beta=1.0, n_trajectories=200)
        result = comp.compare(mdp, mdp, alignment, task)

        assert isinstance(result, ComparisonResult)
        assert result.verdict in (RegressionVerdict.NEUTRAL, RegressionVerdict.INCONCLUSIVE)

    def test_identical_mdps_delta_near_zero(self):
        """For identical MDPs the delta cost mean_time should be approximately zero."""
        mdp = make_two_state_mdp()
        alignment = _identity_alignment(mdp)
        task = make_login_task()
        comp = PairedComparator(beta=1.0, n_trajectories=200)
        result = comp.compare(mdp, mdp, alignment, task)

        assert abs(result.delta_cost.mean_time) < 0.5

    def test_identical_mdps_p_value_high(self):
        """For identical MDPs the p-value should be high (fail to reject H₀)."""
        mdp = make_two_state_mdp()
        alignment = _identity_alignment(mdp)
        task = make_login_task()
        comp = PairedComparator(beta=1.0, n_trajectories=200)
        result = comp.compare(mdp, mdp, alignment, task)

        assert result.p_value > 0.01

    def test_identical_cyclic_mdps(self):
        """Comparing identical cyclic MDPs should also yield NEUTRAL."""
        mdp = make_cyclic_mdp()
        alignment = _identity_alignment(mdp)
        task = make_login_task()
        comp = PairedComparator(beta=2.0, n_trajectories=200)
        result = comp.compare(mdp, mdp, alignment, task)

        assert result.verdict in (RegressionVerdict.NEUTRAL, RegressionVerdict.INCONCLUSIVE)


# ---------------------------------------------------------------------------
# Tests: compare() — regression detection
# ---------------------------------------------------------------------------


class TestPairedComparatorRegression:
    """Tests for detecting regressions when the after-MDP has higher costs."""

    def test_higher_cost_yields_regression(self):
        """After-MDP with 5× cost should yield REGRESSION verdict."""
        mdp_before = make_two_state_mdp()
        mdp_after = _make_higher_cost_mdp()
        alignment = AlignmentResult(
            mappings=[
                StateMapping(state_a="start", state_b="start", similarity=1.0),
                StateMapping(state_a="goal", state_b="goal", similarity=1.0),
            ],
            unmapped_a=[],
            unmapped_b=["mid"],
            overall_similarity=0.8,
        )
        task = make_login_task()
        comp = PairedComparator(beta=1.0, n_trajectories=300)
        result = comp.compare(mdp_before, mdp_after, alignment, task)

        assert result.verdict == RegressionVerdict.REGRESSION

    def test_regression_positive_delta(self):
        """A regression should produce a positive delta cost (after > before)."""
        mdp_before = make_two_state_mdp()
        mdp_after = _make_higher_cost_mdp()
        alignment = AlignmentResult(
            mappings=[
                StateMapping(state_a="start", state_b="start", similarity=1.0),
                StateMapping(state_a="goal", state_b="goal", similarity=1.0),
            ],
            unmapped_a=[],
            unmapped_b=["mid"],
            overall_similarity=0.8,
        )
        task = make_login_task()
        comp = PairedComparator(beta=1.0, n_trajectories=300)
        result = comp.compare(mdp_before, mdp_after, alignment, task)

        assert result.delta_cost.mean_time > 0

    def test_regression_is_regression_property(self):
        """is_regression should be True when verdict is REGRESSION."""
        mdp_before = make_two_state_mdp()
        mdp_after = _make_higher_cost_mdp()
        alignment = AlignmentResult(
            mappings=[
                StateMapping(state_a="start", state_b="start", similarity=1.0),
                StateMapping(state_a="goal", state_b="goal", similarity=1.0),
            ],
            unmapped_a=[],
            unmapped_b=["mid"],
            overall_similarity=0.8,
        )
        task = make_login_task()
        comp = PairedComparator(beta=1.0, n_trajectories=300)
        result = comp.compare(mdp_before, mdp_after, alignment, task)

        assert result.is_regression is True
        assert result.is_improvement is False


# ---------------------------------------------------------------------------
# Tests: compare() — improvement detection
# ---------------------------------------------------------------------------


class TestPairedComparatorImprovement:
    """Tests for detecting improvements when the after-MDP has lower costs."""

    def test_lower_cost_yields_neutral_not_regression(self):
        """Lower after-cost yields NEUTRAL (one-sided test for regression only)."""
        mdp_before = _make_higher_cost_mdp()  # high-cost stochastic MDP
        mdp_after = _make_lower_cost_mdp()    # low-cost stochastic MDP
        alignment = AlignmentResult(
            mappings=[
                StateMapping(state_a="start", state_b="start", similarity=1.0),
                StateMapping(state_a="mid", state_b="mid", similarity=1.0),
                StateMapping(state_a="goal", state_b="goal", similarity=1.0),
            ],
            overall_similarity=1.0,
        )
        task = make_login_task()
        comp = PairedComparator(beta=1.0, n_trajectories=300)
        result = comp.compare(mdp_before, mdp_after, alignment, task)

        assert result.verdict == RegressionVerdict.NEUTRAL

    def test_improvement_negative_delta(self):
        """When after-cost < before-cost, delta should be negative."""
        mdp_before = _make_higher_cost_mdp()
        mdp_after = _make_lower_cost_mdp()
        alignment = AlignmentResult(
            mappings=[
                StateMapping(state_a="start", state_b="start", similarity=1.0),
                StateMapping(state_a="mid", state_b="mid", similarity=1.0),
                StateMapping(state_a="goal", state_b="goal", similarity=1.0),
            ],
            overall_similarity=1.0,
        )
        task = make_login_task()
        comp = PairedComparator(beta=1.0, n_trajectories=300)
        result = comp.compare(mdp_before, mdp_after, alignment, task)

        assert result.delta_cost.mean_time < 0


# ---------------------------------------------------------------------------
# Tests: effect size and confidence interval
# ---------------------------------------------------------------------------


class TestPairedComparatorEffectSize:
    """Tests for Cohen's d effect size and confidence interval computation."""

    def test_effect_size_nonzero_for_different_mdps(self):
        """Cohen's d should be non-zero for MDPs with different costs."""
        mdp_before = make_two_state_mdp()
        mdp_after = _make_higher_cost_mdp()
        alignment = AlignmentResult(
            mappings=[
                StateMapping(state_a="start", state_b="start", similarity=1.0),
                StateMapping(state_a="goal", state_b="goal", similarity=1.0),
            ],
            unmapped_a=[], unmapped_b=["mid"], overall_similarity=0.8,
        )
        task = make_login_task()
        comp = PairedComparator(beta=1.0, n_trajectories=300)
        result = comp.compare(mdp_before, mdp_after, alignment, task)

        assert result.effect_size != 0.0

    def test_effect_size_large_for_big_difference(self):
        """A large cost increase should produce a large effect size (d ≥ 0.8)."""
        mdp_before = make_two_state_mdp()
        mdp_after = _make_higher_cost_mdp()
        alignment = AlignmentResult(
            mappings=[
                StateMapping(state_a="start", state_b="start", similarity=1.0),
                StateMapping(state_a="goal", state_b="goal", similarity=1.0),
            ],
            unmapped_a=[], unmapped_b=["mid"], overall_similarity=0.8,
        )
        task = make_login_task()
        comp = PairedComparator(beta=1.0, n_trajectories=300)
        result = comp.compare(mdp_before, mdp_after, alignment, task)

        assert abs(result.effect_size) >= 0.8

    def test_confidence_level(self):
        """confidence should be 1 − α; with α=0.05 → confidence=0.95."""
        mdp = make_two_state_mdp()
        alignment = _identity_alignment(mdp)
        task = make_login_task()
        comp = PairedComparator(beta=1.0, significance_level=0.05, n_trajectories=200)
        result = comp.compare(mdp, mdp, alignment, task)

        assert result.confidence == pytest.approx(0.95)

    def test_cost_before_and_after_are_cost_elements(self):
        """cost_before and cost_after should be CostElement instances."""
        mdp = make_two_state_mdp()
        alignment = _identity_alignment(mdp)
        task = make_login_task()
        comp = PairedComparator(beta=1.0, n_trajectories=200)
        result = comp.compare(mdp, mdp, alignment, task)

        assert isinstance(result.cost_before, CostElement)
        assert isinstance(result.cost_after, CostElement)
        assert result.cost_before.mean_time >= 0
        assert result.cost_after.mean_time >= 0


# ---------------------------------------------------------------------------
# Tests: alignment handling
# ---------------------------------------------------------------------------


class TestPairedComparatorAlignment:
    """Tests for alignment parameter handling and partial mappings."""

    def test_partial_alignment(self):
        """The comparator should handle alignments with unmapped states."""
        mdp_a = make_cyclic_mdp()
        mdp_b = make_two_state_mdp()
        alignment = AlignmentResult(
            mappings=[
                StateMapping(state_a="s0", state_b="start", similarity=0.8),
            ],
            unmapped_a=["s1", "s2"],
            unmapped_b=[],
            overall_similarity=0.5,
        )
        task = make_login_task()
        comp = PairedComparator(beta=1.0, n_trajectories=200)
        result = comp.compare(mdp_a, mdp_b, alignment, task)

        assert isinstance(result, ComparisonResult)
        assert result.verdict in list(RegressionVerdict)

    def test_empty_alignment(self):
        """An alignment with no mappings should still produce a valid result."""
        mdp = make_two_state_mdp()
        alignment = AlignmentResult(
            mappings=[],
            unmapped_a=list(mdp.states.keys()),
            unmapped_b=list(mdp.states.keys()),
            overall_similarity=0.0,
        )
        task = make_login_task()
        comp = PairedComparator(beta=1.0, n_trajectories=200)
        result = comp.compare(mdp, mdp, alignment, task)

        assert isinstance(result, ComparisonResult)


# ---------------------------------------------------------------------------
# Tests: config override
# ---------------------------------------------------------------------------


class TestPairedComparatorConfig:
    """Tests for configuration override via the config parameter."""

    def test_config_overrides_beta(self):
        """config={'beta': 10.0} should override the constructor β."""
        mdp = make_two_state_mdp()
        alignment = _identity_alignment(mdp)
        task = make_login_task()
        comp = PairedComparator(beta=1.0, n_trajectories=200)
        result = comp.compare(mdp, mdp, alignment, task, config={"beta": 10.0})

        assert isinstance(result, ComparisonResult)
        assert result.parameter_sensitivity.get("beta") == 10.0

    def test_result_has_description(self):
        """ComparisonResult should include a human-readable description."""
        mdp = make_two_state_mdp()
        alignment = _identity_alignment(mdp)
        task = make_login_task()
        comp = PairedComparator(beta=1.0, n_trajectories=200)
        result = comp.compare(mdp, mdp, alignment, task)

        assert isinstance(result.description, str)
        assert len(result.description) > 0


# ---------------------------------------------------------------------------
# Tests: softmax policy solver
# ---------------------------------------------------------------------------


class TestSoftmaxPolicySolver:
    """Tests for the _solve_softmax_policy helper function."""

    def test_policy_sums_to_one(self):
        """The bounded-rational policy action probs should sum to 1.0."""
        mdp = make_choice_mdp(n_choices=5)
        policy = _solve_softmax_policy(mdp, beta=1.0)

        probs = policy.action_probs("start")
        assert probs is not None
        total = sum(probs.values())
        assert total == pytest.approx(1.0, abs=1e-6)

    def test_higher_beta_concentrates_policy(self):
        """Higher β should concentrate the policy on the cheapest action."""
        mdp = make_choice_mdp(n_choices=5)
        policy_low = _solve_softmax_policy(mdp, beta=1.0)
        policy_high = _solve_softmax_policy(mdp, beta=10.0)

        prob_low = policy_low.action_probs("start").get("choice_0", 0)
        prob_high = policy_high.action_probs("start").get("choice_0", 0)

        assert prob_high > prob_low

    def test_empty_mdp_returns_empty_policy(self):
        """Solving an MDP with no states should return an empty policy."""
        mdp = MDP()
        policy = _solve_softmax_policy(mdp, beta=1.0)
        assert policy.beta == 1.0
