"""Tests for decomposition module."""

import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from marace.decomposition.interaction_graph import (
    InteractionGraph,
    InteractionEdge,
    InteractionType,
    InteractionStrengthMetrics,
)
from marace.decomposition.contracts import (
    Contract,
    LinearContract,
    LinearPredicate,
    ConjunctivePredicate,
    ContractChecker,
)
from marace.decomposition.partitioning import (
    Partition,
    SpectralPartitioner,
    MinCutPartitioner,
    ConstrainedPartitioner,
    PartitionQualityMetrics,
)


def _edge(source, target, itype_str, strength):
    """Helper to create an InteractionEdge with string interaction type."""
    type_map = {
        "observation": InteractionType.OBSERVATION,
        "obs": InteractionType.OBSERVATION,
        "communication": InteractionType.COMMUNICATION,
        "physics": InteractionType.PHYSICS,
    }
    return InteractionEdge(
        source_agent=source,
        target_agent=target,
        interaction_type=type_map[itype_str],
        strength=strength,
    )


class TestInteractionGraph:
    """Test interaction graph construction."""

    def test_empty_graph(self):
        """Test empty interaction graph."""
        g = InteractionGraph()
        assert g.num_agents == 0
        assert g.num_edges == 0

    def test_add_agents(self):
        """Test adding agents."""
        g = InteractionGraph()
        g.add_agent("a")
        g.add_agent("b")
        g.add_agent("c")
        assert g.num_agents == 3

    def test_add_interaction(self):
        """Test adding interaction edge."""
        g = InteractionGraph()
        g.add_agent("a")
        g.add_agent("b")
        edge = InteractionEdge(
            source_agent="a",
            target_agent="b",
            interaction_type=InteractionType.OBSERVATION,
            strength=0.8,
        )
        g.add_interaction(edge)
        assert g.num_edges == 1

    def test_get_neighbors(self):
        """Test getting interacting neighbors."""
        g = InteractionGraph()
        for a in ["a", "b", "c", "d"]:
            g.add_agent(a)
        g.add_interaction(_edge("a", "b", "observation", 0.5))
        g.add_interaction(_edge("a", "c", "communication", 0.3))
        neighbors = g.neighbours("a")
        assert "b" in neighbors
        assert "c" in neighbors
        assert "d" not in neighbors

    def test_interaction_strength(self):
        """Test interaction strength retrieval."""
        g = InteractionGraph()
        g.add_agent("a")
        g.add_agent("b")
        g.add_interaction(_edge("a", "b", "observation", 0.75))
        strength = g.coupling_strength("a", "b")
        assert np.isclose(strength, 0.75)

    def test_connected_groups(self):
        """Test extracting connected groups."""
        g = InteractionGraph()
        for a in ["a", "b", "c", "d"]:
            g.add_agent(a)
        g.add_interaction(_edge("a", "b", "observation", 0.5))
        g.add_interaction(_edge("c", "d", "communication", 0.5))
        groups = g.connected_components()
        assert len(groups) == 2


class TestContracts:
    """Test contract generation and checking."""

    def test_linear_contract_creation(self):
        """Test creating a linear contract."""
        assume = ConjunctivePredicate()
        assume.add(LinearPredicate({"x0": 1.0, "x2": -1.0}, 10.0))
        guarantee = ConjunctivePredicate()
        guarantee.add(LinearPredicate({"x0": 1.0, "x2": -1.0}, 2.0))
        contract = LinearContract(
            name="separation",
            assumption=assume,
            guarantee=guarantee,
        )
        assert contract.name == "separation"

    def test_contract_satisfaction(self):
        """Test checking contract satisfaction on a state."""
        assume = ConjunctivePredicate()
        assume.add(LinearPredicate({"x": 1.0}, 5.0))
        guarantee = ConjunctivePredicate()
        guarantee.add(LinearPredicate({"y": 1.0}, 3.0))
        contract = LinearContract(
            name="test",
            assumption=assume,
            guarantee=guarantee,
        )
        assignment = {"x": 2.0, "y": 1.0}
        # Assume: x <= 5 -> True; Guarantee: y <= 3 -> True
        assert contract.assumption.evaluate(assignment)
        assert contract.guarantee.evaluate(assignment)

    def test_contract_violation(self):
        """Test contract violation detection."""
        assume = ConjunctivePredicate()
        assume.add(LinearPredicate({"x": 1.0}, 5.0))
        guarantee = ConjunctivePredicate()
        guarantee.add(LinearPredicate({"y": 1.0}, 3.0))
        contract = LinearContract(
            name="test",
            assumption=assume,
            guarantee=guarantee,
        )
        assignment = {"x": 2.0, "y": 10.0}
        assert contract.assumption.evaluate(assignment)
        assert not contract.guarantee.evaluate(assignment)


class TestPartitioning:
    """Test agent partitioning algorithms."""

    def test_spectral_two_groups(self):
        """Test spectral partitioning into two groups."""
        g = InteractionGraph()
        for i in range(6):
            g.add_agent(f"a{i}")
        # Two clusters
        for i in range(3):
            for j in range(i + 1, 3):
                g.add_interaction(_edge(f"a{i}", f"a{j}", "obs", 1.0))
        for i in range(3, 6):
            for j in range(i + 1, 6):
                g.add_interaction(_edge(f"a{i}", f"a{j}", "obs", 1.0))
        # Weak link between clusters
        g.add_interaction(_edge("a2", "a3", "obs", 0.1))

        partitioner = SpectralPartitioner(num_groups=2)
        partition = partitioner.partition(g)
        assert partition.num_groups == 2
        assert len(partition.all_agents) == 6

    def test_constrained_max_size(self):
        """Test constrained partitioning with max group size."""
        g = InteractionGraph()
        for i in range(8):
            g.add_agent(f"a{i}")
        for i in range(7):
            g.add_interaction(_edge(f"a{i}", f"a{i+1}", "obs", 0.5))

        partitioner = ConstrainedPartitioner(max_group_size=3)
        partition = partitioner.partition(g)
        for group in partition.groups.values():
            assert len(group) <= 3

    def test_partition_quality(self):
        """Test partition quality metrics."""
        g = InteractionGraph()
        for i in range(4):
            g.add_agent(f"a{i}")
        g.add_interaction(_edge("a0", "a1", "obs", 1.0))
        g.add_interaction(_edge("a2", "a3", "obs", 1.0))

        partition = Partition(groups={
            "g0": frozenset({"a0", "a1"}),
            "g1": frozenset({"a2", "a3"}),
        })
        cross = PartitionQualityMetrics.cross_group_coupling(g, partition)
        assert cross == 0.0  # No cross-group edges

    def test_min_cut_partitioner(self):
        """Test min-cut partitioning."""
        g = InteractionGraph()
        for i in range(4):
            g.add_agent(f"a{i}")
        g.add_interaction(_edge("a0", "a1", "obs", 1.0))
        g.add_interaction(_edge("a1", "a2", "obs", 0.1))
        g.add_interaction(_edge("a2", "a3", "obs", 1.0))

        partitioner = MinCutPartitioner(num_groups=2)
        partition = partitioner.partition(g)
        assert partition.num_groups == 2


class TestInteractionStrengthMetrics:
    """Test interaction strength computation."""

    def test_hb_density(self):
        """Test HB edge density metric."""
        # 5 HB edges from "a" to "b" out of 10 total events
        hb_edges = [("a", "b")] * 5 + [("b", "c")] * 5
        strength = InteractionStrengthMetrics.hb_edge_density(
            hb_edges=hb_edges, source="a", target="b", total_events=10
        )
        assert 0.0 <= strength <= 1.0

    def test_observation_overlap(self):
        """Test observation overlap metric."""
        obs_i = np.array([[1.0], [2.0], [3.0], [4.0], [5.0]])
        obs_j = np.array([[1.05], [2.05], [3.05], [4.05], [5.05]])
        overlap = InteractionStrengthMetrics.observation_overlap(obs_i, obs_j)
        assert overlap > 0.8  # Very similar observations

    def test_no_overlap(self):
        """Test zero overlap for independent observations."""
        obs_i = np.array([[1.0], [0.0], [0.0]])
        obs_j = np.array([[0.0], [0.0], [100.0]])
        overlap = InteractionStrengthMetrics.observation_overlap(obs_i, obs_j)
        assert overlap < 0.5
