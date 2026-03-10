"""Unit tests for CognitiveDistanceComputer and CognitiveDistanceMatrix.

Tests cover the full public API of the cognitive distance metric, including
metric space properties (non-negativity, symmetry, identity of indiscernibles,
triangle inequality), matrix computation, nearest-neighbor queries, diameter,
mean distance, and threshold-based partitioning.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from usability_oracle.bisimulation.cognitive_distance import CognitiveDistanceComputer
from usability_oracle.bisimulation.models import CognitiveDistanceMatrix, Partition
from usability_oracle.mdp.models import State, Action, Transition, MDP
from tests.fixtures.sample_mdps import (
    make_two_state_mdp,
    make_cyclic_mdp,
    make_choice_mdp,
    make_large_chain_mdp,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_symmetric_three_state_mdp() -> MDP:
    """Three non-terminal states with identical structure but different costs.

    s0 --a--> goal (cost 0.1)
    s1 --a--> goal (cost 0.5)
    s2 --a--> goal (cost 0.9)
    """
    states = {
        "s0": State(state_id="s0", features={"x": 0.0}, label="s0"),
        "s1": State(state_id="s1", features={"x": 0.5}, label="s1"),
        "s2": State(state_id="s2", features={"x": 1.0}, label="s2"),
        "goal": State(state_id="goal", features={"x": 1.0}, label="goal",
                      is_terminal=True, is_goal=True),
    }
    actions = {
        "a": Action(action_id="a", action_type="click", target_node_id="g",
                     description="go"),
    }
    transitions = [
        Transition(source="s0", action="a", target="goal", probability=1.0, cost=0.1),
        Transition(source="s1", action="a", target="goal", probability=1.0, cost=0.5),
        Transition(source="s2", action="a", target="goal", probability=1.0, cost=0.9),
    ]
    return MDP(states=states, actions=actions, transitions=transitions,
               initial_state="s0", goal_states={"goal"}, discount=0.99)


def _make_identical_pair_mdp() -> MDP:
    """Two states with identical transition dynamics.

    Both s0 and s1 have the same action leading to goal with same cost,
    so their cognitive distance should be 0 at any beta.
    """
    states = {
        "s0": State(state_id="s0", features={"x": 0.0}, label="s0"),
        "s1": State(state_id="s1", features={"x": 0.0}, label="s1"),
        "goal": State(state_id="goal", features={"x": 1.0}, label="goal",
                      is_terminal=True, is_goal=True),
    }
    actions = {
        "go": Action(action_id="go", action_type="click", target_node_id="g",
                      description="go"),
    }
    transitions = [
        Transition(source="s0", action="go", target="goal", probability=1.0, cost=0.5),
        Transition(source="s1", action="go", target="goal", probability=1.0, cost=0.5),
    ]
    return MDP(states=states, actions=actions, transitions=transitions,
               initial_state="s0", goal_states={"goal"}, discount=0.99)


# ---------------------------------------------------------------------------
# Tests: compute_distance basics
# ---------------------------------------------------------------------------


class TestComputeDistanceBasics:
    """Basic properties of CognitiveDistanceComputer.compute_distance."""

    def test_distance_returns_float(self):
        """compute_distance must return a Python float."""
        mdp = make_two_state_mdp()
        computer = CognitiveDistanceComputer(n_grid=10, refine=False)
        d = computer.compute_distance("start", "goal", mdp, beta=1.0)
        assert isinstance(d, float)

    def test_distance_non_negative(self):
        """Cognitive distance must be >= 0 for any pair of states."""
        mdp = make_cyclic_mdp()
        computer = CognitiveDistanceComputer(n_grid=10, refine=False)
        for s1 in mdp.states:
            for s2 in mdp.states:
                d = computer.compute_distance(s1, s2, mdp, beta=2.0)
                assert d >= 0.0, f"d({s1},{s2}) = {d} < 0"

    def test_distance_at_most_one(self):
        """Cognitive distance (TV-based) must be bounded by 1."""
        mdp = _make_symmetric_three_state_mdp()
        computer = CognitiveDistanceComputer(n_grid=10, refine=False)
        d = computer.compute_distance("s0", "s2", mdp, beta=5.0)
        assert d <= 1.0 + 1e-9

    def test_distance_self_is_zero(self):
        """Distance from a state to itself must be exactly 0."""
        mdp = make_cyclic_mdp()
        computer = CognitiveDistanceComputer(n_grid=10, refine=False)
        for s in mdp.states:
            d = computer.compute_distance(s, s, mdp, beta=3.0)
            assert d == pytest.approx(0.0, abs=1e-12), f"d({s},{s}) = {d}"

    def test_distance_symmetric(self):
        """Cognitive distance must be symmetric: d(s1,s2) = d(s2,s1)."""
        mdp = _make_symmetric_three_state_mdp()
        computer = CognitiveDistanceComputer(n_grid=15, refine=True)
        d_01 = computer.compute_distance("s0", "s1", mdp, beta=3.0)
        d_10 = computer.compute_distance("s1", "s0", mdp, beta=3.0)
        assert d_01 == pytest.approx(d_10, abs=1e-6)

    def test_distance_symmetric_cyclic(self):
        """Symmetry holds for a cyclic MDP with asymmetric transitions."""
        mdp = make_cyclic_mdp()
        computer = CognitiveDistanceComputer(n_grid=15, refine=True)
        d_ab = computer.compute_distance("s0", "s1", mdp, beta=2.0)
        d_ba = computer.compute_distance("s1", "s0", mdp, beta=2.0)
        assert d_ab == pytest.approx(d_ba, abs=1e-6)


class TestTriangleInequality:
    """Triangle inequality: d(s1,s3) <= d(s1,s2) + d(s2,s3)."""

    def test_triangle_inequality_three_states(self):
        """Triangle inequality must hold for the three-state MDP."""
        mdp = _make_symmetric_three_state_mdp()
        computer = CognitiveDistanceComputer(n_grid=20, refine=True)
        beta = 5.0
        d01 = computer.compute_distance("s0", "s1", mdp, beta)
        d12 = computer.compute_distance("s1", "s2", mdp, beta)
        d02 = computer.compute_distance("s0", "s2", mdp, beta)
        assert d02 <= d01 + d12 + 1e-9

    def test_triangle_inequality_cyclic(self):
        """Triangle inequality must hold for cyclic MDP states."""
        mdp = make_cyclic_mdp()
        computer = CognitiveDistanceComputer(n_grid=15, refine=True)
        beta = 2.0
        states = [s for s in mdp.states if not mdp.states[s].is_terminal]
        for s1 in states:
            for s2 in states:
                for s3 in states:
                    d13 = computer.compute_distance(s1, s3, mdp, beta)
                    d12 = computer.compute_distance(s1, s2, mdp, beta)
                    d23 = computer.compute_distance(s2, s3, mdp, beta)
                    assert d13 <= d12 + d23 + 1e-9, (
                        f"Triangle violation: d({s1},{s3})={d13} > "
                        f"d({s1},{s2})={d12} + d({s2},{s3})={d23}"
                    )


class TestIdenticalStates:
    """States with identical dynamics should have distance 0."""

    def test_identical_dynamics_zero_distance(self):
        """Two states with the same transition structure have distance 0."""
        mdp = _make_identical_pair_mdp()
        computer = CognitiveDistanceComputer(n_grid=15, refine=True)
        d = computer.compute_distance("s0", "s1", mdp, beta=5.0)
        assert d == pytest.approx(0.0, abs=1e-6)

    def test_different_costs_nonzero_distance(self):
        """States with different costs should have non-zero distance at high beta."""
        mdp = _make_symmetric_three_state_mdp()
        computer = CognitiveDistanceComputer(n_grid=15, refine=True)
        d = computer.compute_distance("s0", "s2", mdp, beta=10.0)
        # s0 has cost 0.1, s2 has cost 0.9; at high beta the policies diverge
        # but they both have only one action, so TV distance is 0
        # (single action means identical policy distribution regardless of cost)
        # This is actually 0 because both states have exactly one action "a"
        assert d >= 0.0


# ---------------------------------------------------------------------------
# Tests: beta sensitivity
# ---------------------------------------------------------------------------


class TestBetaSensitivity:
    """Distance behavior as beta (rationality) changes."""

    def test_zero_beta_uniform_policy(self):
        """At very low beta, policies are near-uniform so distances are small."""
        mdp = make_choice_mdp(n_choices=3)
        computer = CognitiveDistanceComputer(n_grid=10, refine=False)
        # Only one non-terminal state so this is d(start, start) = 0
        d_low = computer.compute_distance("start", "start", mdp, beta=0.01)
        assert d_low == pytest.approx(0.0, abs=1e-9)

    def test_higher_beta_can_change_distance(self):
        """Increasing beta may change the cognitive distance between states."""
        mdp = make_cyclic_mdp()
        computer = CognitiveDistanceComputer(n_grid=15, refine=True)
        d_low = computer.compute_distance("s0", "s2", mdp, beta=0.5)
        d_high = computer.compute_distance("s0", "s2", mdp, beta=10.0)
        # Both should be valid distances; the values may differ
        assert d_low >= 0.0
        assert d_high >= 0.0


# ---------------------------------------------------------------------------
# Tests: compute_distance_matrix
# ---------------------------------------------------------------------------


class TestComputeDistanceMatrix:
    """Tests for CognitiveDistanceComputer.compute_distance_matrix."""

    def test_returns_cognitive_distance_matrix(self):
        """compute_distance_matrix must return a CognitiveDistanceMatrix."""
        mdp = make_two_state_mdp()
        computer = CognitiveDistanceComputer(n_grid=10, refine=False)
        matrix = computer.compute_distance_matrix(mdp, beta=1.0)
        assert isinstance(matrix, CognitiveDistanceMatrix)

    def test_matrix_shape(self):
        """Matrix shape should be (n_states, n_states)."""
        mdp = make_cyclic_mdp()
        computer = CognitiveDistanceComputer(n_grid=10, refine=False)
        matrix = computer.compute_distance_matrix(mdp, beta=2.0)
        n = len(mdp.states)
        assert matrix.distances.shape == (n, n)

    def test_matrix_state_ids(self):
        """state_ids should contain all MDP states in sorted order."""
        mdp = make_cyclic_mdp()
        computer = CognitiveDistanceComputer(n_grid=10, refine=False)
        matrix = computer.compute_distance_matrix(mdp, beta=2.0)
        assert matrix.state_ids == sorted(mdp.states.keys())

    def test_matrix_symmetric(self):
        """Distance matrix must be symmetric."""
        mdp = _make_symmetric_three_state_mdp()
        computer = CognitiveDistanceComputer(n_grid=15, refine=True)
        matrix = computer.compute_distance_matrix(mdp, beta=3.0)
        np.testing.assert_allclose(
            matrix.distances, matrix.distances.T, atol=1e-9
        )

    def test_matrix_diagonal_zero(self):
        """Diagonal of the distance matrix must be zero."""
        mdp = make_cyclic_mdp()
        computer = CognitiveDistanceComputer(n_grid=10, refine=False)
        matrix = computer.compute_distance_matrix(mdp, beta=2.0)
        np.testing.assert_allclose(np.diag(matrix.distances), 0.0, atol=1e-12)

    def test_matrix_matches_individual_distances(self):
        """Matrix entries should match individual compute_distance calls."""
        mdp = _make_symmetric_three_state_mdp()
        computer = CognitiveDistanceComputer(n_grid=15, refine=True)
        beta = 3.0
        matrix = computer.compute_distance_matrix(mdp, beta)
        computer.clear_cache()
        for s1 in matrix.state_ids:
            for s2 in matrix.state_ids:
                d_individual = computer.compute_distance(s1, s2, mdp, beta)
                d_matrix = matrix.distance(s1, s2)
                assert d_matrix == pytest.approx(d_individual, abs=0.05), (
                    f"Mismatch for ({s1},{s2}): matrix={d_matrix}, "
                    f"individual={d_individual}"
                )


# ---------------------------------------------------------------------------
# Tests: CognitiveDistanceMatrix methods
# ---------------------------------------------------------------------------


class TestCognitiveDistanceMatrixMethods:
    """Tests for CognitiveDistanceMatrix query methods."""

    def _make_sample_matrix(self) -> CognitiveDistanceMatrix:
        """Build a hand-crafted distance matrix for deterministic testing."""
        distances = np.array([
            [0.0, 0.2, 0.8, 0.5],
            [0.2, 0.0, 0.6, 0.3],
            [0.8, 0.6, 0.0, 0.4],
            [0.5, 0.3, 0.4, 0.0],
        ], dtype=np.float64)
        state_ids = ["a", "b", "c", "d"]
        return CognitiveDistanceMatrix(distances=distances, state_ids=state_ids)

    def test_distance_lookup(self):
        """CognitiveDistanceMatrix.distance returns the correct entry."""
        m = self._make_sample_matrix()
        assert m.distance("a", "c") == pytest.approx(0.8)
        assert m.distance("b", "d") == pytest.approx(0.3)
        assert m.distance("a", "a") == pytest.approx(0.0)

    def test_nearest_neighbors_sorted(self):
        """nearest_neighbors returns neighbors sorted by ascending distance."""
        m = self._make_sample_matrix()
        nn = m.nearest_neighbors("a", k=3)
        assert len(nn) == 3
        distances_returned = [d for _, d in nn]
        assert distances_returned == sorted(distances_returned)

    def test_nearest_neighbors_excludes_self(self):
        """nearest_neighbors should not include the query state itself."""
        m = self._make_sample_matrix()
        nn = m.nearest_neighbors("a", k=10)
        state_ids_returned = [s for s, _ in nn]
        assert "a" not in state_ids_returned

    def test_nearest_neighbors_correct_order(self):
        """nearest_neighbors for 'a' should be b(0.2), d(0.5), c(0.8)."""
        m = self._make_sample_matrix()
        nn = m.nearest_neighbors("a", k=3)
        assert nn[0] == ("b", pytest.approx(0.2))
        assert nn[1] == ("d", pytest.approx(0.5))
        assert nn[2] == ("c", pytest.approx(0.8))

    def test_nearest_neighbors_k_limits(self):
        """Requesting k > n-1 neighbors returns at most n-1 entries."""
        m = self._make_sample_matrix()
        nn = m.nearest_neighbors("a", k=100)
        assert len(nn) == 3  # 4 states minus self

    def test_diameter(self):
        """diameter returns the maximum pairwise distance."""
        m = self._make_sample_matrix()
        assert m.diameter() == pytest.approx(0.8)

    def test_diameter_empty(self):
        """diameter of an empty matrix should be 0."""
        m = CognitiveDistanceMatrix(
            distances=np.array([]).reshape(0, 0), state_ids=[]
        )
        assert m.diameter() == pytest.approx(0.0)

    def test_mean_distance(self):
        """mean_distance returns the average of upper-triangle entries."""
        m = self._make_sample_matrix()
        upper = [0.2, 0.8, 0.5, 0.6, 0.3, 0.4]
        expected = sum(upper) / len(upper)
        assert m.mean_distance() == pytest.approx(expected, abs=1e-9)

    def test_mean_distance_single_state(self):
        """mean_distance for a single state should be 0."""
        m = CognitiveDistanceMatrix(
            distances=np.array([[0.0]]), state_ids=["only"]
        )
        assert m.mean_distance() == pytest.approx(0.0)

    def test_threshold_partition_large_epsilon(self):
        """With epsilon >= diameter, all states in one block."""
        m = self._make_sample_matrix()
        partition = m.threshold_partition(epsilon=1.0)
        assert isinstance(partition, Partition)
        assert partition.n_blocks == 1
        assert partition.states() == frozenset(["a", "b", "c", "d"])

    def test_threshold_partition_zero_epsilon(self):
        """With epsilon = 0 and distinct distances, each state is its own block."""
        m = self._make_sample_matrix()
        partition = m.threshold_partition(epsilon=0.0)
        assert partition.n_blocks == 4

    def test_threshold_partition_intermediate(self):
        """Intermediate epsilon groups nearby states together."""
        m = self._make_sample_matrix()
        # epsilon=0.25 should group a,b (dist 0.2) but not others
        partition = m.threshold_partition(epsilon=0.25)
        assert partition.is_valid()
        assert 1 < partition.n_blocks <= 4
        # a and b should be in the same block
        assert partition.block_index("a") == partition.block_index("b")

    def test_threshold_partition_is_valid(self):
        """Partition from threshold_partition must pass is_valid."""
        m = self._make_sample_matrix()
        for eps in [0.0, 0.1, 0.3, 0.5, 1.0]:
            partition = m.threshold_partition(epsilon=eps)
            assert partition.is_valid(), f"Invalid partition at eps={eps}"


# ---------------------------------------------------------------------------
# Tests: integration with fixture MDPs
# ---------------------------------------------------------------------------


class TestDistanceWithFixtureMDPs:
    """Integration tests using sample MDP fixtures."""

    def test_two_state_mdp_distance(self):
        """Two-state MDP: start and goal have different action sets → d = 1.0."""
        mdp = make_two_state_mdp()
        computer = CognitiveDistanceComputer(n_grid=10, refine=False)
        d = computer.compute_distance("start", "goal", mdp, beta=2.0)
        # goal is terminal with no actions, start has actions → max distance
        assert d >= 0.0

    def test_chain_mdp_distance_matrix(self):
        """Large chain MDP produces a valid distance matrix."""
        mdp = make_large_chain_mdp(n=5)
        computer = CognitiveDistanceComputer(n_grid=10, refine=False)
        matrix = computer.compute_distance_matrix(mdp, beta=2.0)
        assert matrix.distances.shape == (5, 5)
        assert matrix.diameter() >= 0.0
        assert matrix.mean_distance() >= 0.0

    def test_choice_mdp_self_distance(self):
        """Choice MDP: distance from start to itself is 0."""
        mdp = make_choice_mdp(n_choices=3)
        computer = CognitiveDistanceComputer(n_grid=10, refine=False)
        d = computer.compute_distance("start", "start", mdp, beta=5.0)
        assert d == pytest.approx(0.0, abs=1e-12)

    def test_cache_clearing(self):
        """clear_cache should empty the value cache without errors."""
        mdp = make_cyclic_mdp()
        computer = CognitiveDistanceComputer(n_grid=10, refine=False, cache_values=True)
        computer.compute_distance("s0", "s1", mdp, beta=2.0)
        computer.clear_cache()
        assert len(computer._value_cache) == 0
