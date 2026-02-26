"""Tests for HB graph construction and operations."""

import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from marace.hb.hb_graph import HBGraph, HBRelation
from marace.hb.vector_clock import VectorClock


class TestHBGraphConstruction:
    """Test HB graph construction."""

    def test_empty_graph(self):
        """Test creating an empty HB graph."""
        g = HBGraph()
        assert g.num_events == 0
        assert g.num_edges == 0

    def test_add_events(self):
        """Test adding events to the graph."""
        g = HBGraph()
        g.add_event("e1", agent_id="a", timestamp=0.0)
        g.add_event("e2", agent_id="a", timestamp=0.1)
        g.add_event("e3", agent_id="b", timestamp=0.05)
        assert g.num_events == 3

    def test_add_hb_edge(self):
        """Test adding happens-before edge."""
        g = HBGraph()
        g.add_event("e1", agent_id="a", timestamp=0.0)
        g.add_event("e2", agent_id="b", timestamp=0.1)
        g.add_hb_edge("e1", "e2")
        assert g.num_edges == 1
        result = g.query_hb("e1", "e2")
        assert result == HBRelation.BEFORE

    def test_query_concurrent(self):
        """Test querying concurrent events."""
        g = HBGraph()
        g.add_event("e1", agent_id="a", timestamp=0.0)
        g.add_event("e2", agent_id="b", timestamp=0.0)
        result = g.query_hb("e1", "e2")
        assert result == HBRelation.CONCURRENT

    def test_query_after(self):
        """Test querying AFTER relation."""
        g = HBGraph()
        g.add_event("e1", agent_id="a", timestamp=0.0)
        g.add_event("e2", agent_id="b", timestamp=0.1)
        g.add_hb_edge("e1", "e2")
        result = g.query_hb("e2", "e1")
        assert result == HBRelation.AFTER

    def test_transitive_closure(self):
        """Test transitive closure computation."""
        g = HBGraph()
        g.add_event("e1", agent_id="a", timestamp=0.0)
        g.add_event("e2", agent_id="b", timestamp=0.1)
        g.add_event("e3", agent_id="c", timestamp=0.2)
        g.add_hb_edge("e1", "e2")
        g.add_hb_edge("e2", "e3")
        g.compute_transitive_closure()
        result = g.query_hb("e1", "e3")
        assert result == HBRelation.BEFORE

    def test_chain_of_events(self):
        """Test long chain of HB edges."""
        g = HBGraph()
        n = 10
        for i in range(n):
            g.add_event(f"e{i}", agent_id=f"a{i % 3}", timestamp=float(i) * 0.1)
        for i in range(n - 1):
            g.add_hb_edge(f"e{i}", f"e{i+1}")
        g.compute_transitive_closure()
        assert g.query_hb("e0", f"e{n-1}") == HBRelation.BEFORE
        assert g.query_hb(f"e{n-1}", "e0") == HBRelation.AFTER


class TestHBGraphComponents:
    """Test connected component extraction."""

    def test_single_component(self):
        """Test graph with single connected component."""
        g = HBGraph()
        g.add_event("e1", agent_id="a", timestamp=0.0)
        g.add_event("e2", agent_id="b", timestamp=0.1)
        g.add_event("e3", agent_id="a", timestamp=0.2)
        g.add_hb_edge("e1", "e2")
        g.add_hb_edge("e2", "e3")
        components = g.connected_components()
        assert len(components) == 1

    def test_two_components(self):
        """Test graph with two disconnected components."""
        g = HBGraph()
        g.add_event("e1", agent_id="a", timestamp=0.0)
        g.add_event("e2", agent_id="a", timestamp=0.1)
        g.add_event("e3", agent_id="b", timestamp=0.0)
        g.add_event("e4", agent_id="b", timestamp=0.1)
        g.add_hb_edge("e1", "e2")
        g.add_hb_edge("e3", "e4")
        components = g.connected_components()
        assert len(components) == 2

    def test_component_agents(self):
        """Test extracting agents from components."""
        g = HBGraph()
        g.add_event("e1", agent_id="a", timestamp=0.0)
        g.add_event("e2", agent_id="b", timestamp=0.1)
        g.add_event("e3", agent_id="c", timestamp=0.0)
        g.add_hb_edge("e1", "e2")
        components = g.connected_components()
        comp_agents = []
        for comp in components:
            agents = set()
            for eid in comp:
                node_data = g.graph.nodes[eid]
                agents.add(node_data.get("agent_id", ""))
            comp_agents.append(agents)
        assert any({"a", "b"}.issubset(a) for a in comp_agents)


class TestHBGraphProperties:
    """Test HB graph structural properties."""

    def test_no_cycles(self):
        """Test that valid HB graph has no cycles."""
        g = HBGraph()
        g.add_event("e1", agent_id="a", timestamp=0.0)
        g.add_event("e2", agent_id="b", timestamp=0.1)
        g.add_event("e3", agent_id="a", timestamp=0.2)
        g.add_hb_edge("e1", "e2")
        g.add_hb_edge("e2", "e3")
        assert not g.has_cycles()

    def test_graph_depth(self):
        """Test computation of graph depth (longest path)."""
        g = HBGraph()
        for i in range(5):
            g.add_event(f"e{i}", agent_id="a", timestamp=float(i))
        for i in range(4):
            g.add_hb_edge(f"e{i}", f"e{i+1}")
        stats = g.statistics()
        assert stats["depth"] == 4

    def test_subgraph_extraction(self):
        """Test extracting subgraph for subset of agents."""
        g = HBGraph()
        g.add_event("e1", agent_id="a", timestamp=0.0)
        g.add_event("e2", agent_id="b", timestamp=0.1)
        g.add_event("e3", agent_id="c", timestamp=0.2)
        g.add_hb_edge("e1", "e2")
        g.add_hb_edge("e2", "e3")
        sub = g.subgraph_for_agents({"a", "b"})
        assert sub.num_events == 2

    def test_concurrent_pairs(self):
        """Test finding concurrent event pairs."""
        g = HBGraph()
        g.add_event("e1", agent_id="a", timestamp=0.0)
        g.add_event("e2", agent_id="b", timestamp=0.0)
        g.add_event("e3", agent_id="c", timestamp=0.0)
        concurrent = g.concurrent_pairs()
        assert len(concurrent) == 3  # (e1,e2), (e1,e3), (e2,e3)

    def test_graph_density(self):
        """Test graph density computation."""
        g = HBGraph()
        for i in range(4):
            g.add_event(f"e{i}", agent_id=f"a{i}", timestamp=float(i))
        g.add_hb_edge("e0", "e1")
        g.add_hb_edge("e0", "e2")
        stats = g.statistics()
        assert "density" in stats
        assert 0.0 <= stats["density"] <= 1.0

    def test_transitive_reduction(self):
        """Test transitive reduction removes redundant edges."""
        g = HBGraph()
        g.add_event("e1", agent_id="a", timestamp=0.0)
        g.add_event("e2", agent_id="b", timestamp=0.1)
        g.add_event("e3", agent_id="c", timestamp=0.2)
        g.add_hb_edge("e1", "e2")
        g.add_hb_edge("e2", "e3")
        g.add_hb_edge("e1", "e3")  # Redundant: transitive
        reduced = g.compute_transitive_reduction()
        assert reduced.num_edges == 2


class TestHBGraphLargeScale:
    """Test HB graph with larger number of events."""

    def test_many_events(self):
        """Test with many events."""
        g = HBGraph()
        n_agents = 5
        n_steps = 20
        for step in range(n_steps):
            for agent in range(n_agents):
                eid = f"e_{agent}_{step}"
                g.add_event(eid, agent_id=f"a{agent}", timestamp=float(step) * 0.1)
                if step > 0:
                    prev_eid = f"e_{agent}_{step-1}"
                    g.add_hb_edge(prev_eid, eid)
        assert g.num_events == n_agents * n_steps
        assert not g.has_cycles()

    def test_many_interactions(self):
        """Test with many cross-agent interactions."""
        g = HBGraph()
        n_agents = 4
        n_steps = 10
        for step in range(n_steps):
            for agent in range(n_agents):
                eid = f"e_{agent}_{step}"
                g.add_event(eid, agent_id=f"a{agent}", timestamp=float(step) * 0.1)
                if step > 0:
                    g.add_hb_edge(f"e_{agent}_{step-1}", eid)
                    other = (agent + 1) % n_agents
                    g.add_hb_edge(f"e_{other}_{step-1}", eid)
        g.compute_transitive_closure()
        assert g.query_hb("e_0_0", f"e_0_{n_steps-1}") == HBRelation.BEFORE
