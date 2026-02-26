"""Tests for marace.decomposition.group_size_theory — interaction graph analysis."""

import math
import pytest
import numpy as np
import networkx as nx
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from marace.decomposition.group_size_theory import (
    InteractionGraphModel,
    BoundedDegreeCondition,
    SpatialLocalityCondition,
    TransitiveClosure,
    ComponentStats,
    TransitivityReport,
)


# ======================================================================
# Helpers
# ======================================================================

def _make_path_graph(n=5):
    """Path graph: 0-1-2-..-(n-1), one connected component."""
    return nx.path_graph(n)


def _make_disconnected_graph():
    """Two disconnected components: {0,1,2} and {3,4}."""
    G = nx.Graph()
    G.add_edges_from([(0, 1), (1, 2), (3, 4)])
    return G


def _make_complete_graph(n=4):
    return nx.complete_graph(n)


# ======================================================================
# InteractionGraphModel
# ======================================================================

class TestInteractionGraphModel:
    """Test InteractionGraphModel construction and properties."""

    def test_from_shared_variables(self):
        agents = [0, 1, 2]
        writes = {0: {"x"}, 1: {"y"}, 2: {"z"}}
        reads = {0: {"y"}, 1: {"z"}, 2: set()}
        model = InteractionGraphModel.from_shared_variables(agents, writes, reads)
        assert model.graph.number_of_nodes() == 3
        # 0 reads y (written by 1) → edge(0,1)
        # 1 reads z (written by 2) → edge(1,2)
        assert model.graph.has_edge(0, 1)
        assert model.graph.has_edge(1, 2)

    def test_from_networkx(self):
        G = _make_path_graph(4)
        model = InteractionGraphModel.from_networkx(G)
        assert model.graph.number_of_nodes() == 4
        assert model.graph.number_of_edges() == 3

    def test_component_stats(self):
        G = _make_disconnected_graph()
        model = InteractionGraphModel.from_networkx(G)
        stats = model.component_stats()
        assert isinstance(stats, ComponentStats)
        assert stats.count == 2
        assert stats.max_size == 3
        assert stats.total_nodes == 5

    def test_interaction_groups(self):
        G = _make_disconnected_graph()
        model = InteractionGraphModel.from_networkx(G)
        groups = model.interaction_groups()
        assert len(groups) == 2
        sizes = sorted([len(g) for g in groups], reverse=True)
        assert sizes == [3, 2]

    def test_max_degree(self):
        G = _make_complete_graph(4)
        model = InteractionGraphModel.from_networkx(G)
        assert model.max_degree() == 3

    def test_density(self):
        G = _make_complete_graph(4)
        model = InteractionGraphModel.from_networkx(G)
        assert model.density() == pytest.approx(1.0, abs=1e-6)


# ======================================================================
# BoundedDegreeCondition
# ======================================================================

class TestBoundedDegreeCondition:
    """Test bounded-degree condition on graphs."""

    def test_check_on_path_graph(self):
        """Path graph has max degree 2, should satisfy condition."""
        G = _make_path_graph(10)
        bdc = BoundedDegreeCondition()
        satisfied, max_deg, comp_bound = bdc.check_condition(G)
        assert max_deg == 2
        assert satisfied  # 2 < sqrt(10) ≈ 3.16

    def test_check_on_complete_graph(self):
        """Complete graph K_10 has max degree 9 ≈ n-1, likely unsatisfied."""
        G = _make_complete_graph(10)
        bdc = BoundedDegreeCondition()
        satisfied, max_deg, comp_bound = bdc.check_condition(G)
        assert max_deg == 9
        assert not satisfied  # 9 > sqrt(10) ≈ 3.16

    def test_unit_ball_volume(self):
        """Check unit ball volumes for known dimensions."""
        # 1D: length 2
        assert BoundedDegreeCondition.unit_ball_volume(1) == pytest.approx(2.0, abs=1e-6)
        # 2D: π
        assert BoundedDegreeCondition.unit_ball_volume(2) == pytest.approx(math.pi, abs=1e-6)
        # 3D: 4π/3
        assert BoundedDegreeCondition.unit_ball_volume(3) == pytest.approx(4 * math.pi / 3, abs=1e-4)

    def test_ball_packing_bound(self):
        bdc = BoundedDegreeCondition()
        bound = bdc.ball_packing_bound(radius=1.0, dim=2, min_separation=1.0)
        assert bound > 0
        assert isinstance(bound, int)

    def test_predicted_component_size(self):
        bdc = BoundedDegreeCondition()
        size = bdc.predicted_component_size(
            max_degree=3, spatial_dim=2, density=0.1, radius=1.0,
        )
        assert size > 0

    def test_empty_graph(self):
        G = nx.Graph()
        bdc = BoundedDegreeCondition()
        satisfied, max_deg, comp_bound = bdc.check_condition(G)
        assert satisfied
        assert max_deg == 0


# ======================================================================
# SpatialLocalityCondition
# ======================================================================

class TestSpatialLocalityCondition:
    """Test spatial locality and percolation thresholds."""

    def test_critical_density_2d(self):
        """Check critical density for 2D."""
        slc = SpatialLocalityCondition()
        rho_c = slc.critical_density(R=1.0, dim=2)
        assert rho_c > 0
        # λ_c ≈ 4.512, C_2 = π, R=1 → ρ_c = 4.512/π ≈ 1.435
        assert rho_c == pytest.approx(4.512 / math.pi, rel=0.01)

    def test_subcritical_detection(self):
        """Low density should be sub-critical."""
        slc = SpatialLocalityCondition()
        assert slc.is_subcritical(density=0.1, R=1.0, dim=2)

    def test_supercritical_detection(self):
        """High density should be super-critical."""
        slc = SpatialLocalityCondition()
        assert not slc.is_subcritical(density=100.0, R=1.0, dim=2)

    def test_expected_degree(self):
        slc = SpatialLocalityCondition()
        deg = slc.expected_degree(density=1.0, R=1.0, dim=2)
        # λ = ρ · C_2 · R^2 = 1 · π · 1 = π
        assert deg == pytest.approx(math.pi, abs=1e-4)

    def test_subcritical_component_bound(self):
        """In sub-critical regime, bound should be finite."""
        slc = SpatialLocalityCondition()
        bound = slc.subcritical_component_bound(
            n_agents=100, density=0.1, R=1.0, dim=2,
        )
        assert bound is not None
        assert bound > 0

    def test_supercritical_no_bound(self):
        """In super-critical regime, no sub-linear bound exists."""
        slc = SpatialLocalityCondition()
        bound = slc.subcritical_component_bound(
            n_agents=100, density=100.0, R=1.0, dim=2,
        )
        assert bound is None


# ======================================================================
# TransitiveClosure
# ======================================================================

class TestTransitiveClosure:
    """Test transitive closure analysis."""

    def test_path_graph_closure(self):
        """Path 0-1-2: closure has 3 edges (full clique)."""
        G = _make_path_graph(3)
        tc = TransitiveClosure()
        report = tc.analyze_transitivity(G)
        assert isinstance(report, TransitivityReport)
        assert report.original_edges == 2
        assert report.closure_edges == 3  # C(3,2) = 3

    def test_complete_graph_closure_same(self):
        """Complete graph: closure = original."""
        G = _make_complete_graph(4)
        tc = TransitiveClosure()
        report = tc.analyze_transitivity(G)
        assert report.original_edges == report.closure_edges
        assert report.transitive_fraction == pytest.approx(1.0, abs=1e-6)

    def test_disconnected_graph_closure(self):
        """Disconnected components: closure is union of cliques."""
        G = _make_disconnected_graph()  # {0,1,2} and {3,4}
        tc = TransitiveClosure()
        report = tc.analyze_transitivity(G)
        # Component 1: 3 nodes → 3 edges, Component 2: 2 nodes → 1 edge
        assert report.closure_edges == 3 + 1

    def test_transitivity_growth_ratio(self):
        """Growth ratio should be >= 1."""
        G = _make_path_graph(5)
        tc = TransitiveClosure()
        report = tc.analyze_transitivity(G)
        assert report.transitivity_growth_ratio >= 1.0

    def test_clustering_coefficient(self):
        G = _make_complete_graph(4)
        tc = TransitiveClosure()
        report = tc.analyze_transitivity(G)
        assert report.clustering_coefficient == pytest.approx(1.0, abs=1e-6)

    def test_hb_interaction_note(self):
        note = TransitiveClosure.hb_interaction_note()
        assert isinstance(note, str)
        assert len(note) > 0

    def test_empty_graph(self):
        G = nx.Graph()
        G.add_node(0)
        tc = TransitiveClosure()
        report = tc.analyze_transitivity(G)
        assert report.original_edges == 0
        assert report.closure_edges == 0
