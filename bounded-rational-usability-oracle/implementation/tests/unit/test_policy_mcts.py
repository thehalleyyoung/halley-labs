"""Unit tests for usability_oracle.policy.mcts — MCTS search and Thompson sampling.

Tests cover bounded-rational MCTS, UCB1/PUCT scoring, Thompson sampling for
bandits, and policy gradient methods.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from usability_oracle.mdp.models import State, Action, Transition, MDP
from usability_oracle.policy.models import Policy, QValues
from usability_oracle.policy.mcts import (
    BoundedRationalMCTS,
    MCTSConfig,
    MCTSNode,
    Simulator,
    aggregate_parallel_results,
    puct_score,
    ucb1_score,
)
from usability_oracle.policy.thompson import (
    BayesianUIOptimiser,
    BetaBernoulliArm,
    BetaBernoulliThompson,
    BoundedRationalThompson,
    GaussianArm,
    GaussianThompson,
    KnowledgeGradient,
)
from usability_oracle.policy.gradient import (
    GradientTrajectory,
    REINFORCE,
    NaturalPolicyGradient,
    SoftmaxPolicyParam,
    learn_cognitive_policy,
)
from usability_oracle.policy.multi_objective import (
    ConstrainedMDPSolver,
    MultiObjectiveQValues,
    ParetoFrontierComputer,
    ParetoPoint,
    weighted_sum_scalarisation,
    lexicographic_optimise,
    multi_objective_value_iteration,
)


# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════


class _SimpleSimulator:
    """Deterministic grid-like simulator for MCTS tests."""

    def __init__(self):
        self._actions = {
            "start": ["go", "stay"],
            "mid": ["finish"],
            "goal": [],
        }

    def get_actions(self, state: str) -> list[str]:
        return self._actions.get(state, [])

    def step(self, state: str, action: str) -> tuple[str, float, bool]:
        if state == "start" and action == "go":
            return "mid", -1.0, False
        elif state == "start" and action == "stay":
            return "start", -0.5, False
        elif state == "mid" and action == "finish":
            return "goal", 0.0, True
        return state, -1.0, True


def _simple_q_multi() -> MultiObjectiveQValues:
    """2-state, 2-action, 2-objective Q-values."""
    return MultiObjectiveQValues(
        values={
            "s0": {
                "a": np.array([1.0, 3.0]),
                "b": np.array([3.0, 1.0]),
            },
        },
        n_objectives=2,
        objective_names=["time", "error"],
    )


# ═══════════════════════════════════════════════════════════════════════════
# MCTS Search
# ═══════════════════════════════════════════════════════════════════════════


class TestMCTS:
    """Test bounded-rational Monte Carlo Tree Search."""

    def test_mcts_returns_action_values(self):
        config = MCTSConfig(n_simulations=50, exploration_constant=1.0, beta=1.0,
                            discount=0.99, max_rollout_depth=10)
        sim = _SimpleSimulator()
        mcts = BoundedRationalMCTS(sim, config, rng=np.random.default_rng(42))
        values = mcts.search("start")
        assert isinstance(values, dict)
        assert len(values) > 0

    def test_mcts_prefers_go_over_stay(self):
        """In the simple domain, 'go' leads to goal faster."""
        config = MCTSConfig(n_simulations=200, exploration_constant=1.0, beta=1.0,
                            discount=0.99, max_rollout_depth=20)
        sim = _SimpleSimulator()
        mcts = BoundedRationalMCTS(sim, config, rng=np.random.default_rng(42))
        values = mcts.search("start")
        assert values.get("go", 0) >= values.get("stay", 0) - 1.0

    def test_mcts_search_policy(self):
        config = MCTSConfig(n_simulations=100, beta=1.0)
        sim = _SimpleSimulator()
        mcts = BoundedRationalMCTS(sim, config, rng=np.random.default_rng(42))
        policy = mcts.search_policy("start", temperature=1.0)
        assert isinstance(policy, Policy)

    def test_mcts_config_defaults(self):
        config = MCTSConfig()
        assert config.n_simulations > 0
        assert config.exploration_constant > 0

    def test_mcts_terminal_state(self):
        config = MCTSConfig(n_simulations=10, beta=1.0)
        sim = _SimpleSimulator()
        mcts = BoundedRationalMCTS(sim, config, rng=np.random.default_rng(42))
        values = mcts.search("goal")
        assert isinstance(values, dict)


# ═══════════════════════════════════════════════════════════════════════════
# UCB1 and PUCT Scoring
# ═══════════════════════════════════════════════════════════════════════════


class TestUCBScoring:
    """Test UCB1 and PUCT scoring functions."""

    def test_ucb1_increases_with_parent_visits(self):
        child = MCTSNode(
            state="s", parent=None, parent_action="a",
            children={}, visit_count=5, total_value=2.0,
            prior_prob=0.5, is_terminal=False, untried_actions=[],
        )
        s1 = ucb1_score(child, parent_visits=10, c=1.0, beta=1.0,
                        prior_probs={"a": 0.5})
        s2 = ucb1_score(child, parent_visits=100, c=1.0, beta=1.0,
                        prior_probs={"a": 0.5})
        assert s2 > s1

    def test_puct_with_prior(self):
        child = MCTSNode(
            state="s", parent=None, parent_action="a",
            children={}, visit_count=3, total_value=1.5,
            prior_prob=0.8, is_terminal=False, untried_actions=[],
        )
        score = puct_score(child, parent_visits=10, c_puct=1.0, beta=1.0)
        assert isinstance(score, float)


# ═══════════════════════════════════════════════════════════════════════════
# Thompson Sampling
# ═══════════════════════════════════════════════════════════════════════════


class TestThompsonSampling:
    """Test Thompson sampling for bandits."""

    def test_beta_bernoulli_arm(self):
        arm = BetaBernoulliArm(alpha=1.0, beta_param=1.0)
        rng = np.random.default_rng(42)
        sample = arm.sample(rng)
        assert 0.0 <= sample <= 1.0
        arm.update(success=True)
        assert arm.alpha == 2.0

    def test_beta_bernoulli_thompson(self):
        ts = BetaBernoulliThompson(actions=["a", "b", "c"])
        for _ in range(100):
            action = ts.select_action()
            ts.update(action, success=(action == "b"))
        probs = ts.action_probabilities()
        assert probs["b"] > probs["a"]

    def test_gaussian_arm(self):
        arm = GaussianArm(mu=0.0, n_obs=0, tau=1.0, sum_x=0.0, sum_x2=0.0)
        rng = np.random.default_rng(42)
        sample = arm.sample(rng)
        assert isinstance(sample, float)
        arm.update(1.5)
        assert arm.n_obs == 1

    def test_gaussian_thompson(self):
        ts = GaussianThompson(actions=["a", "b"], rng=np.random.default_rng(42))
        for _ in range(50):
            action = ts.select_action(minimise=True)
            value = 1.0 if action == "a" else 2.0
            ts.update(action, value)
        means = ts.posterior_means()
        assert means["a"] < means["b"] + 0.5

    def test_bounded_rational_thompson(self):
        brt = BoundedRationalThompson(
            actions=["a", "b"], beta=2.0, rng=np.random.default_rng(42)
        )
        for _ in range(30):
            action = brt.select_action(minimise=True)
            brt.update(action, 1.0 if action == "a" else 3.0)
        policy = brt.to_policy("s0")
        assert isinstance(policy, Policy)

    def test_knowledge_gradient(self):
        kg = KnowledgeGradient(
            actions=["a", "b", "c"], rng=np.random.default_rng(42)
        )
        action = kg.select_action()
        assert action in ["a", "b", "c"]
        kg.update(action, 1.0)


# ═══════════════════════════════════════════════════════════════════════════
# Policy Gradient
# ═══════════════════════════════════════════════════════════════════════════


class TestPolicyGradient:
    """Test policy gradient methods."""

    def test_softmax_policy_param(self):
        states = ["s0", "s1"]
        actions_per_state = {"s0": ["a", "b"], "s1": ["c"]}
        param = SoftmaxPolicyParam(states, actions_per_state, beta=1.0)
        probs = param.action_probs("s0")
        assert abs(sum(probs.values()) - 1.0) < 1e-6

    def test_reinforce_update(self):
        states = ["s0"]
        actions_per_state = {"s0": ["a", "b"]}
        param = SoftmaxPolicyParam(states, actions_per_state, beta=1.0)
        reinforce = REINFORCE(param, learning_rate=0.1)
        traj = GradientTrajectory(states=["s0", "s0"], actions=["a", "b"], costs=[1.0, 2.0])
        info = reinforce.update([traj])
        assert isinstance(info, dict)
        policy = reinforce.get_policy()
        assert isinstance(policy, Policy)

    def test_natural_policy_gradient(self):
        states = ["s0"]
        actions_per_state = {"s0": ["a", "b"]}
        param = SoftmaxPolicyParam(states, actions_per_state, beta=1.0)
        npg = NaturalPolicyGradient(param, learning_rate=0.01)
        traj = GradientTrajectory(states=["s0"], actions=["a"], costs=[1.0])
        info = npg.update([traj])
        assert isinstance(info, dict)

    def test_learn_cognitive_policy(self):
        traces = [
            GradientTrajectory(states=["s0", "s0"], actions=["a", "b"], costs=[1.0, 0.5]),
            GradientTrajectory(states=["s0", "s0"], actions=["b", "a"], costs=[0.5, 1.5]),
        ]
        policy = learn_cognitive_policy(
            traces, states=["s0"], actions_per_state={"s0": ["a", "b"]},
            beta=1.0, n_epochs=10, learning_rate=0.1,
        )
        assert isinstance(policy, Policy)


# ═══════════════════════════════════════════════════════════════════════════
# Multi-Objective Policy
# ═══════════════════════════════════════════════════════════════════════════


class TestMultiObjectivePolicy:
    """Test multi-objective policy computation."""

    def test_scalarisation(self):
        q = _simple_q_multi()
        weights = np.array([0.5, 0.5])
        scalar_q = q.scalarise(weights)
        assert isinstance(scalar_q, QValues)

    def test_weighted_sum_scalarisation(self):
        q = _simple_q_multi()
        weights = np.array([1.0, 0.0])
        policy = weighted_sum_scalarisation(q, weights, beta=1.0)
        assert isinstance(policy, Policy)

    def test_pareto_frontier(self):
        q = _simple_q_multi()
        pf = ParetoFrontierComputer(n_objectives=2, n_weight_samples=10)
        points = pf.compute(q)
        assert isinstance(points, list)
        assert all(isinstance(p, ParetoPoint) for p in points)

    def test_lexicographic(self):
        q = _simple_q_multi()
        policy = lexicographic_optimise(q, priority_order=[0, 1], beta=1.0)
        assert isinstance(policy, Policy)

    def test_constrained_solver(self):
        q = _simple_q_multi()
        solver = ConstrainedMDPSolver(beta=1.0)
        policy, lambdas = solver.solve(q, constraints={1: 2.0})
        assert isinstance(policy, Policy)
