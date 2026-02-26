"""Tests for MCTS search engine."""

import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from marace.search.mcts import (
    MCTS,
    MCTSNode,
    MCTSTree,
    ScheduleAction,
    SearchBudget,
    SearchResult,
)
from marace.search.ucb import (
    UCB1Standard,
    UCB1Safety,
    ExplorationBonus,
)
from marace.search.pruning import (
    SafetyMarginPruner,
    HBConsistencyPruner,
    CompositePruner,
)
from marace.search.schedule_optimizer import (
    LocalScheduleSearch,
    GeneticScheduleSearch,
)


class TestMCTSNode:
    """Test MCTS node operations."""

    def test_node_creation(self):
        """Test creating a node."""
        node = MCTSNode(
            schedule=[],
            abstract_state=np.zeros(4),
        )
        assert node.visit_count == 0
        assert node.value_sum == 0.0
        assert len(node.children) == 0

    def test_node_expansion(self):
        """Test expanding a node with children."""
        root = MCTSNode(schedule=[], abstract_state=np.zeros(2))
        actions = [
            ScheduleAction(agent_id="a", timing_offset=0.0),
            ScheduleAction(agent_id="b", timing_offset=0.0),
        ]
        for a in actions:
            child = MCTSNode(
                schedule=[a],
                abstract_state=np.zeros(2),
                parent=root,
                depth=1,
            )
            root.children[a] = child
        assert len(root.children) == 2

    def test_node_value(self):
        """Test node value computation."""
        node = MCTSNode(schedule=[], abstract_state=np.zeros(2))
        node.visit_count = 10
        node.value_sum = 7.5
        assert np.isclose(node.value, 0.75)

    def test_backpropagation(self):
        """Test value backpropagation."""
        root = MCTSNode(schedule=[], abstract_state=np.zeros(2))
        action_a = ScheduleAction("a", 0.0)
        child = MCTSNode(schedule=[action_a], abstract_state=np.zeros(2), parent=root, depth=1)
        root.children[action_a] = child
        action_b = ScheduleAction("b", 0.0)
        grandchild = MCTSNode(schedule=[action_a, action_b], abstract_state=np.zeros(2), parent=child, depth=2)
        child.children[action_b] = grandchild

        # Backpropagate from grandchild
        value = 0.8
        node = grandchild
        while node is not None:
            node.visit_count += 1
            node.value_sum += value
            node = node.parent

        assert root.visit_count == 1
        assert child.visit_count == 1
        assert grandchild.visit_count == 1
        assert np.isclose(root.value, 0.8)


class TestUCB1:
    """Test UCB1 selection strategies."""

    def test_ucb1_standard(self):
        """Test standard UCB1 selection."""
        ucb = UCB1Standard(bonus=ExplorationBonus(exploration_constant=1.414))
        # Test scoring directly
        score = ucb.score(value=0.5, parent_visits=100, child_visits=10)
        assert score > 0.5  # value + exploration bonus

    def test_ucb1_selects_unexplored(self):
        """Test UCB1 gives inf score for unexplored nodes."""
        bonus = ExplorationBonus(exploration_constant=1.414)
        b = bonus.compute(parent_visits=50, child_visits=0)
        assert b == float("inf")

    def test_ucb1_safety(self):
        """Test UCB1 with safety margin."""
        ucb = UCB1Safety(
            bonus=ExplorationBonus(exploration_constant=1.414),
            safety_weight=0.5,
        )
        score = ucb.score(value=0.5, parent_visits=100, child_visits=10)
        assert score > 0  # Should produce a valid score


class TestExplorationBonus:
    """Test exploration bonus computation."""

    def test_basic_bonus(self):
        """Test basic exploration bonus."""
        bonus = ExplorationBonus(exploration_constant=1.414)
        b = bonus.compute(parent_visits=100, child_visits=10)
        assert b > 0

    def test_bonus_decreases_with_visits(self):
        """Test bonus decreases with more visits."""
        bonus = ExplorationBonus(exploration_constant=1.414)
        b1 = bonus.compute(parent_visits=100, child_visits=1)
        b2 = bonus.compute(parent_visits=100, child_visits=50)
        assert b1 > b2


class TestPruning:
    """Test pruning strategies."""

    def test_safety_margin_pruner(self):
        """Test safety margin pruning."""
        pruner = SafetyMarginPruner(
            margin_threshold=0.5,
            safety_direction=np.array([1.0, 0.0]),
        )
        # Without zonotope set, can_prune returns False
        assert not pruner.can_prune(["e1", "e2"], np.array([0.8, 0.0]))

    def test_composite_pruner(self):
        """Test composite pruner with multiple strategies."""
        pruner1 = SafetyMarginPruner(
            margin_threshold=0.5,
            safety_direction=np.array([1.0, 0.0]),
        )
        composite = CompositePruner(pruners=[pruner1])
        # Without zonotope, no pruning
        assert not composite.can_prune(["e1"], np.array([0.8, 0.0]))

    def test_no_pruning_without_zonotope(self):
        """Test no pruning without zonotope configured."""
        pruner = SafetyMarginPruner(
            margin_threshold=0.5,
            safety_direction=np.array([1.0, 0.0]),
        )
        assert not pruner.can_prune(["e1"], np.array([0.1, 0.0]))


class TestScheduleOptimizer:
    """Test schedule optimization."""

    def test_local_search(self):
        """Test local schedule search."""
        def evaluator(schedule, state):
            return sum(abs(e.get("timing_offset", 0.0)) for e in schedule)

        initial = [{"agent_id": "a", "timing_offset": 0.0},
                   {"agent_id": "b", "timing_offset": 0.1},
                   {"agent_id": "c", "timing_offset": 0.2}]
        search = LocalScheduleSearch(
            safety_evaluator=evaluator,
            max_iterations=10,
        )
        result = search.search(initial, np.zeros(2))
        assert result is not None

    def test_genetic_search(self):
        """Test genetic schedule search."""
        def evaluator(schedule, state):
            return len(schedule)

        initial_schedules = [
            [{"agent_id": "a", "timing_offset": 0.0},
             {"agent_id": "b", "timing_offset": 0.1}]
        ]
        search = GeneticScheduleSearch(
            safety_evaluator=evaluator,
            population_size=20,
            generations=5,
            seed=42,
        )
        result = search.search(initial_schedules, np.zeros(2))
        assert result is not None


class TestMCTSTree:
    """Test MCTS tree operations."""

    def test_tree_creation(self):
        """Test creating MCTS tree."""
        tree = MCTSTree(
            root_state=np.zeros(4),
        )
        assert tree.root is not None
        assert tree.root.visit_count == 0

    def test_tree_size(self):
        """Test tree size tracking."""
        tree = MCTSTree(root_state=np.zeros(2))
        assert tree.size == 1
        action = ScheduleAction("a", 0.0)
        child = MCTSNode(schedule=[action], abstract_state=np.zeros(2),
                         parent=tree.root, depth=1)
        tree.root.children[action] = child
        tree.register_node()
        assert tree.size == 2


class TestSearchBudget:
    """Test search budget configuration."""

    def test_iteration_budget(self):
        """Test iteration-based budget."""
        budget = SearchBudget(iteration_count=100)
        assert not budget.is_exhausted(iterations_done=50, time_elapsed=0, nodes_created=0)
        assert budget.is_exhausted(iterations_done=100, time_elapsed=0, nodes_created=0)

    def test_node_budget(self):
        """Test node-based budget."""
        budget = SearchBudget(max_nodes=1000)
        assert not budget.is_exhausted(iterations_done=0, time_elapsed=0, nodes_created=500)
        assert budget.is_exhausted(iterations_done=0, time_elapsed=0, nodes_created=1000)
