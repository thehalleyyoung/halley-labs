"""
Tests for MCTS search module.

Tests cover: UCB computation, MCTS search, causal UCB pruning,
convergence monitoring, and d-separation pruner.
"""
import math
import pytest
import numpy as np
import networkx as nx

from causalbound.mcts.tree_node import MCTSNode, NodeStatistics
from causalbound.mcts.search import MCTSSearch, SearchConfig, SearchResult
from causalbound.mcts.causal_ucb import CausalUCB, ArmStatistics
from causalbound.mcts.convergence import ConvergenceMonitor, ArmTracker
from causalbound.mcts.pruning import DSeparationPruner
from causalbound.mcts.rollout import RolloutScheduler, RolloutPolicy


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def simple_dag():
    G = nx.DiGraph()
    G.add_edges_from([("A", "B"), ("B", "C"), ("A", "C")])
    return G


@pytest.fixture
def diamond_dag():
    G = nx.DiGraph()
    G.add_edges_from([("A", "B"), ("A", "C"), ("B", "D"), ("C", "D")])
    return G


@pytest.fixture
def star_dag():
    G = nx.DiGraph()
    for i in range(5):
        G.add_edge("center", f"leaf_{i}")
    return G


# ---------------------------------------------------------------------------
# NodeStatistics
# ---------------------------------------------------------------------------

class TestNodeStatistics:
    """Test MCTS node statistics tracking."""

    def test_initial_state(self):
        stats = NodeStatistics()
        assert stats.visit_count == 0
        assert stats.mean_value == 0.0

    def test_single_update(self):
        stats = NodeStatistics()
        stats.update(0.75)
        assert stats.visit_count == 1
        assert abs(stats.mean_value - 0.75) < 1e-10

    def test_multiple_updates(self):
        stats = NodeStatistics()
        values = [0.5, 0.7, 0.3, 0.9, 0.1]
        for v in values:
            stats.update(v)
        assert stats.visit_count == 5
        assert abs(stats.mean_value - np.mean(values)) < 1e-10
        assert abs(stats.min_value - 0.1) < 1e-10
        assert abs(stats.max_value - 0.9) < 1e-10

    def test_variance(self):
        stats = NodeStatistics()
        stats.update(0.0)
        stats.update(1.0)
        var = stats.variance
        assert var >= 0
        # Var([0, 1]) = E[X^2] - E[X]^2 = 0.5 - 0.25 = 0.25
        assert abs(var - 0.25) < 1e-10

    def test_std_dev(self):
        stats = NodeStatistics()
        stats.update(0.0)
        stats.update(1.0)
        assert abs(stats.std_dev - 0.5) < 1e-10

    def test_to_dict(self):
        stats = NodeStatistics()
        stats.update(0.5)
        d = stats.to_dict()
        assert d["visit_count"] == 1
        assert d["mean_value"] == 0.5


# ---------------------------------------------------------------------------
# MCTSNode
# ---------------------------------------------------------------------------

class TestMCTSNode:
    """Test MCTS tree node operations."""

    def test_root_node(self):
        root = MCTSNode()
        assert root.is_root
        assert root.depth == 0
        assert root.visit_count == 0

    def test_child_creation(self):
        root = MCTSNode()
        actions = [("X", 0.0), ("X", 1.0)]
        root.set_available_actions(actions)
        child = root.expand(action=("X", 1.0))
        assert not child.is_root
        assert child.depth == 1
        assert child.action == ("X", 1.0)
        assert child.parent is root
        assert root.n_children == 1

    def test_ucb_unvisited(self):
        """Unvisited node should have infinite UCB."""
        root = MCTSNode()
        child = root.expand(action=("X", 0.0))
        score = child.get_ucb_score()
        assert score == float("inf")

    def test_ucb_computation(self):
        """Test UCB1 formula: mean + c * sqrt(ln(N)/n)."""
        root = MCTSNode()
        root.stats.update(0.5)
        root.stats.update(0.6)
        child = root.expand(action=("X", 0.0))
        child.stats.update(0.4)
        c = 1.414
        score = child.get_ucb_score(exploration_constant=c)
        expected_exploit = 0.4
        expected_explore = c * math.sqrt(math.log(2) / 1)
        expected = expected_exploit + expected_explore
        assert abs(score - expected) < 1e-6

    def test_ucb_tuned(self):
        root = MCTSNode()
        for _ in range(10):
            root.stats.update(0.5)
        child = root.expand(action=("X", 0.0))
        for _ in range(5):
            child.stats.update(0.3)
        score = child.get_ucb_tuned_score()
        assert isinstance(score, float)
        assert not math.isinf(score)

    def test_backpropagate(self):
        root = MCTSNode()
        c1 = root.expand(action=("X", 0.0))
        c2 = c1.expand(action=("Y", 1.0))
        c2.backpropagate(0.8)
        assert c2.visit_count == 1
        assert c1.visit_count == 1
        assert root.visit_count == 1
        assert abs(root.mean_value - 0.8) < 1e-10

    def test_best_child(self):
        root = MCTSNode()
        c1 = root.expand(action=("X", 0.0))
        c2 = root.expand(action=("X", 1.0))
        c1.stats.update(0.3)
        c2.stats.update(0.7)
        best = root.best_child(maximize=True)
        assert best is c2

    def test_most_visited_child(self):
        root = MCTSNode()
        c1 = root.expand(action=("X", 0.0))
        c2 = root.expand(action=("X", 1.0))
        for _ in range(10):
            c1.stats.update(0.3)
        for _ in range(5):
            c2.stats.update(0.7)
        mvc = root.most_visited_child()
        assert mvc is c1

    def test_is_leaf(self):
        node = MCTSNode()
        assert node.is_leaf()
        node.expand(action=("X", 0.0))
        assert not node.is_leaf()

    def test_tree_size(self):
        root = MCTSNode()
        c1 = root.expand(action=("X", 0.0))
        c2 = root.expand(action=("X", 1.0))
        c11 = c1.expand(action=("Y", 0.0))
        assert root.tree_size() == 4

    def test_tree_depth(self):
        root = MCTSNode()
        c1 = root.expand(action=("X", 0.0))
        c11 = c1.expand(action=("Y", 0.0))
        assert root.tree_depth() == 2

    def test_path_from_root(self):
        root = MCTSNode()
        c1 = root.expand(action=("X", 0.0))
        c11 = c1.expand(action=("Y", 0.0))
        path = c11.get_path_from_root()
        assert len(path) == 3
        assert path[0] is root
        assert path[-1] is c11

    def test_action_sequence(self):
        root = MCTSNode()
        c1 = root.expand(action=("X", 0.0))
        c11 = c1.expand(action=("Y", 1.0))
        seq = c11.get_action_sequence()
        assert len(seq) == 2
        assert seq[0] == ("X", 0.0)
        assert seq[1] == ("Y", 1.0)

    def test_pruning(self):
        root = MCTSNode()
        c1 = root.expand(action=("X", 0.0))
        c1.mark_pruned(reason="d-separated")
        assert c1.pruned

    def test_progressive_expand(self):
        root = MCTSNode()
        actions = [("X", float(i)) for i in range(5)]
        root.set_available_actions(actions)
        # progressive widening needs visit_count > 0 to compute max_children
        root.stats.update(0.5)
        def action_gen(state, expanded):
            for a in actions:
                if a not in expanded:
                    return a
            return None
        child = root.expand_progressive(action_gen)
        assert child is not None
        assert root.n_children >= 1

    def test_subtree_statistics(self):
        root = MCTSNode()
        c1 = root.expand(action=("X", 0.0))
        c1.stats.update(0.5)
        stats = root.subtree_statistics()
        assert "total_nodes" in stats

    def test_to_dict(self):
        root = MCTSNode()
        root.stats.update(0.5)
        d = root.to_dict(max_depth=1)
        assert "node_id" in d

    def test_select_child_ucb(self):
        root = MCTSNode()
        for i in range(3):
            c = root.expand(action=("X", float(i)))
            for _ in range(i + 1):
                c.stats.update(0.5)
            root.stats.update(0.5)
        selected = root.select_child()
        assert selected is not None


# ---------------------------------------------------------------------------
# CausalUCB
# ---------------------------------------------------------------------------

class TestCausalUCB:
    """Test CausalUCB score computation and pruning."""

    def test_compute_score(self, simple_dag):
        cucb = CausalUCB()
        root = MCTSNode()
        child = root.expand(action=("B", 1.0))
        child.stats.update(0.5)
        child.stats.update(0.7)
        root.stats.update(0.5)
        root.stats.update(0.7)
        score = cucb.compute_score(
            node=child,
            parent_visits=10,
        )
        assert isinstance(score, float)
        assert not math.isnan(score)

    def test_prune_irrelevant(self, diamond_dag):
        cucb = CausalUCB()
        root = MCTSNode()
        actions = [("A", 0.0), ("B", 1.0), ("C", 0.0), ("D", 1.0)]
        for a in actions:
            root.expand(action=a)
        target = "D"
        pruned_count = cucb.prune_irrelevant(root, diamond_dag, target)
        assert isinstance(pruned_count, int)

    def test_select_action(self, simple_dag):
        cucb = CausalUCB()
        root = MCTSNode()
        actions = [("A", 0.0), ("A", 1.0), ("B", 0.0), ("B", 1.0)]
        for a in actions:
            child = root.expand(action=a)
            child.stats.update(np.random.random())
        root.stats.update(0.5)
        selected = cucb.select_action(
            node=root,
            available_actions=actions,
        )
        assert selected in actions

    def test_update_arm(self):
        cucb = CausalUCB()
        cucb.update_arm(("A", 1.0), 0.6)
        cucb.update_arm(("A", 1.0), 0.8)
        ci = cucb.get_confidence_interval(("A", 1.0))
        assert ci is not None
        assert ci[0] <= ci[1]

    def test_information_gain_bonus(self, simple_dag):
        cucb = CausalUCB()
        bonus = cucb.compute_information_gain_bonus(
            action=("B", 1.0),
            dag=simple_dag,
            target="C",
            evidence={},
        )
        assert isinstance(bonus, float)
        assert bonus >= 0

    def test_effective_arms(self, simple_dag):
        cucb = CausalUCB()
        effective = cucb.get_effective_arms(
            dag=simple_dag,
            target="C",
            evidence={},
        )
        assert len(effective) > 0

    def test_arm_summary(self):
        cucb = CausalUCB()
        cucb.update_arm(("A", 0.0), 0.5)
        summary = cucb.get_arm_summary()
        assert isinstance(summary, dict)

    def test_pac_bound(self):
        cucb = CausalUCB()
        result = cucb.compute_pac_bound(
            n_arms=5,
            epsilon=0.05,
            delta=0.05,
        )
        assert result.required_rollouts > 0

    def test_reset(self):
        cucb = CausalUCB()
        cucb.update_arm(("A", 1.0), 0.5)
        cucb.reset()
        summary = cucb.get_arm_summary()
        assert summary["n_arms_tracked"] == 0


# ---------------------------------------------------------------------------
# ConvergenceMonitor
# ---------------------------------------------------------------------------

class TestConvergenceMonitor:
    """Test convergence monitoring."""

    def test_basic_tracking(self):
        monitor = ConvergenceMonitor()
        for i in range(100):
            monitor.update("arm_0", 0.5 + 0.01 * np.random.randn())
        ci = monitor.get_confidence_interval("arm_0")
        assert ci is not None
        assert ci[0] <= ci[1]

    def test_convergence_detection(self):
        monitor = ConvergenceMonitor(default_epsilon=0.01)
        # Feed very consistent values
        for i in range(200):
            monitor.update("arm_a", 0.5 + 0.001 * np.random.randn())
        converged = monitor.is_converged()
        assert isinstance(converged, bool)

    def test_best_arm(self):
        monitor = ConvergenceMonitor()
        for _ in range(50):
            monitor.update("good", 0.9 + 0.01 * np.random.randn())
            monitor.update("bad", 0.1 + 0.01 * np.random.randn())
        best = monitor.get_best_arm(maximize=True)
        assert best == "good"

    def test_arm_ranking(self):
        monitor = ConvergenceMonitor()
        for _ in range(30):
            monitor.update("A", 0.8)
            monitor.update("B", 0.5)
            monitor.update("C", 0.3)
        ranking = monitor.get_arm_ranking(maximize=True)
        assert ranking[0][0] == "A"

    def test_batch_update(self):
        monitor = ConvergenceMonitor()
        obs = [("arm_1", 0.5), ("arm_2", 0.6), ("arm_1", 0.55)]
        monitor.batch_update(obs)
        ci = monitor.get_confidence_interval("arm_1")
        assert ci is not None

    def test_convergence_curve(self):
        monitor = ConvergenceMonitor()
        for i in range(50):
            monitor.update("arm_0", 0.5 + 0.1 / (i + 1))
        curve = monitor.get_convergence_curve()
        assert isinstance(curve, list)

    def test_successive_elimination(self):
        monitor = ConvergenceMonitor(default_epsilon=0.01)
        for _ in range(100):
            monitor.update("winner", 0.9)
            monitor.update("loser", 0.1)
        converged, eliminated = monitor.is_converged_successive_elimination()
        assert isinstance(converged, bool)
        assert isinstance(eliminated, list)

    def test_global_statistics(self):
        monitor = ConvergenceMonitor()
        for _ in range(20):
            monitor.update("a", 0.5)
        stats = monitor.get_global_statistics()
        assert isinstance(stats, dict)

    def test_estimate_remaining_rollouts(self):
        monitor = ConvergenceMonitor(default_epsilon=0.05)
        for _ in range(50):
            monitor.update("arm", 0.5 + 0.1 * np.random.randn())
        remaining = monitor.estimate_remaining_rollouts()
        assert remaining >= 0

    def test_convergence_stats(self):
        monitor = ConvergenceMonitor()
        for _ in range(30):
            monitor.update("arm", 0.5)
        stats = monitor.get_convergence_stats()
        assert isinstance(stats, dict)

    def test_reset(self):
        monitor = ConvergenceMonitor()
        monitor.update("a", 0.5)
        monitor.reset()
        stats = monitor.get_global_statistics()
        assert stats["n"] == 0


# ---------------------------------------------------------------------------
# ArmTracker
# ---------------------------------------------------------------------------

class TestArmTracker:
    """Test arm tracker statistics."""

    def test_basic(self):
        tracker = ArmTracker()
        tracker.add(0.5)
        tracker.add(0.7)
        assert tracker.n == 2
        assert abs(tracker.mean - 0.6) < 1e-10

    def test_variance(self):
        tracker = ArmTracker()
        tracker.add(0.0)
        tracker.add(1.0)
        # sample variance ddof=1: var = 0.5
        assert tracker.variance >= 0

    def test_std_error(self):
        tracker = ArmTracker()
        for _ in range(100):
            tracker.add(np.random.random())
        se = tracker.std_error
        assert se >= 0
        assert se < 1.0  # Should be small with 100 samples


# ---------------------------------------------------------------------------
# DSeparationPruner
# ---------------------------------------------------------------------------

class TestDSeparationPruner:
    """Test d-separation-based pruning."""

    def test_set_dag(self, simple_dag):
        pruner = DSeparationPruner()
        pruner.set_dag(simple_dag)
        assert pruner is not None

    def test_is_prunable(self, diamond_dag):
        pruner = DSeparationPruner()
        pruner.set_dag(diamond_dag)
        # is_prunable expects evidence as a dict
        result = pruner.is_prunable(
            variable="A",
            target="D",
            evidence={"B": 0.0, "C": 0.0},
        )
        assert isinstance(result, bool)

    def test_relevant_variables(self, diamond_dag):
        pruner = DSeparationPruner()
        pruner.set_dag(diamond_dag)
        relevant = pruner.get_relevant_variables(
            target="D",
            evidence={},
        )
        assert isinstance(relevant, list)

    def test_active_trails(self, simple_dag):
        pruner = DSeparationPruner()
        pruner.set_dag(simple_dag)
        # get_active_trails returns dict mapping reachable node -> list of trails
        trails = pruner.get_active_trails(
            source="A",
            evidence={},
        )
        assert isinstance(trails, dict)
        assert len(trails) > 0  # A->B, A->C paths exist

    def test_pruning_ratio(self, diamond_dag):
        pruner = DSeparationPruner()
        pruner.set_dag(diamond_dag)
        result = pruner.compute_pruning_ratio(
            dag=diamond_dag,
            target="D",
            evidence={"B": 0.0, "C": 0.0},
        )
        assert isinstance(result, dict)
        assert 0.0 <= result["pruning_ratio"] <= 1.0

    def test_markov_blanket(self, diamond_dag):
        pruner = DSeparationPruner()
        pruner.set_dag(diamond_dag)
        mb = pruner.get_markov_blanket("B")
        assert isinstance(mb, set)
        assert "A" in mb  # parent
        assert "D" in mb  # child

    def test_pruning_summary(self, simple_dag):
        pruner = DSeparationPruner()
        pruner.set_dag(simple_dag)
        pruner.is_prunable("A", "C", {"B": 0.0})
        summary = pruner.get_pruning_summary()
        assert isinstance(summary, dict)

    def test_precompute_oracle(self, diamond_dag):
        pruner = DSeparationPruner()
        pruner.precompute_dsep_oracle(diamond_dag)
        # After precomputing, queries should be fast
        result = pruner.is_prunable("A", "D", {"B": 0.0, "C": 0.0})
        assert isinstance(result, bool)


# ---------------------------------------------------------------------------
# RolloutScheduler
# ---------------------------------------------------------------------------

class TestRolloutScheduler:
    """Test rollout scheduling and evaluation."""

    def test_set_policy(self):
        sched = RolloutScheduler()
        sched.set_rollout_policy(RolloutPolicy.RANDOM)
        assert sched is not None

    def test_variable_domains(self):
        sched = RolloutScheduler()
        sched.set_variable_domains({"X": (0.0, 1.0), "Y": (0.0, 1.0)})
        assert sched is not None

    def test_memoization_stats(self):
        sched = RolloutScheduler()
        stats = sched.get_memoization_stats()
        assert isinstance(stats, dict)

    def test_clear_cache(self):
        sched = RolloutScheduler()
        sched.clear_cache()
        assert sched.get_memoization_stats()["cache_size"] == 0


# ---------------------------------------------------------------------------
# MCTSSearch: integration
# ---------------------------------------------------------------------------

class TestMCTSSearchIntegration:
    """Integration tests for MCTS search."""

    def test_search_config(self):
        config = SearchConfig(
            n_rollouts=10,
            exploration_constant=1.414,
        )
        assert config.n_rollouts == 10

    def test_search_tree_stats(self):
        config = SearchConfig(n_rollouts=5)
        search = MCTSSearch(config=config)
        stats = search.get_tree_stats()
        assert isinstance(stats, dict)

    def test_search_summary(self):
        config = SearchConfig(n_rollouts=5)
        search = MCTSSearch(config=config)
        summary = search.get_search_summary()
        assert isinstance(summary, dict)
