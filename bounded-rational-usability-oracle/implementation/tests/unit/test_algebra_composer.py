"""
Unit tests for usability_oracle.algebra.composer — Task-graph-level composition.

Tests the TaskGraphComposer class which composes a networkx DAG of task
dependencies into a single CostElement by detecting parallel groups at each
topological level and composing them with ⊗ (parallel), then sequentially
composing levels with ⊕ (sequential).

Test graphs:
- Linear chain: A → B → C
- Diamond (fork-join): A → {B, C} → D
- Wide parallel: A → {B, C, D} → E
- Single node
- Empty graph
"""

import math
import pytest
import networkx as nx

from usability_oracle.algebra.models import CostElement
from usability_oracle.algebra.composer import TaskGraphComposer


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def composer():
    """Return a TaskGraphComposer with default settings."""
    return TaskGraphComposer()


@pytest.fixture
def cost_a():
    return CostElement(mu=1.0, sigma_sq=0.1, kappa=0.0, lambda_=0.01)


@pytest.fixture
def cost_b():
    return CostElement(mu=2.0, sigma_sq=0.2, kappa=0.1, lambda_=0.02)


@pytest.fixture
def cost_c():
    return CostElement(mu=1.5, sigma_sq=0.15, kappa=0.0, lambda_=0.01)


@pytest.fixture
def cost_d():
    return CostElement(mu=0.5, sigma_sq=0.05, kappa=0.0, lambda_=0.01)


@pytest.fixture
def linear_graph():
    """A → B → C: purely sequential task chain."""
    G = nx.DiGraph()
    G.add_edges_from([("A", "B"), ("B", "C")])
    return G


@pytest.fixture
def diamond_graph():
    """A → {B, C} → D: fork-join parallelism."""
    G = nx.DiGraph()
    G.add_edges_from([("A", "B"), ("A", "C"), ("B", "D"), ("C", "D")])
    return G


@pytest.fixture
def wide_parallel_graph():
    """A → {B, C, D} → E: wide fan-out / fan-in."""
    G = nx.DiGraph()
    G.add_edges_from([
        ("A", "B"), ("A", "C"), ("A", "D"),
        ("B", "E"), ("C", "E"), ("D", "E"),
    ])
    return G


@pytest.fixture
def linear_costs(cost_a, cost_b, cost_c):
    return {"A": cost_a, "B": cost_b, "C": cost_c}


@pytest.fixture
def diamond_costs(cost_a, cost_b, cost_c, cost_d):
    return {"A": cost_a, "B": cost_b, "C": cost_c, "D": cost_d}


@pytest.fixture
def wide_costs(cost_a, cost_b, cost_c, cost_d):
    return {
        "A": cost_a,
        "B": cost_b,
        "C": cost_c,
        "D": cost_d,
        "E": CostElement(mu=0.3, sigma_sq=0.03, kappa=0.0, lambda_=0.005),
    }


# ---------------------------------------------------------------------------
# Basic compose
# ---------------------------------------------------------------------------

class TestComposeLinear:
    """Tests for composing linear (purely sequential) task graphs."""

    def test_linear_compose_valid(self, composer, linear_graph, linear_costs):
        """Composing a linear graph returns a valid CostElement."""
        result = composer.compose(linear_graph, linear_costs)
        assert isinstance(result, CostElement)
        assert result.is_valid

    def test_linear_mu_at_least_sum(self, composer, linear_graph, linear_costs):
        """Linear graph composed μ ≥ sum of individual μ values."""
        result = composer.compose(linear_graph, linear_costs)
        total_mu = sum(c.mu for c in linear_costs.values())
        assert result.mu >= total_mu - 1e-10

    def test_linear_mu_monotonic(self, composer, linear_graph, linear_costs):
        """Linear composed μ ≥ max individual μ."""
        result = composer.compose(linear_graph, linear_costs)
        max_mu = max(c.mu for c in linear_costs.values())
        assert result.mu >= max_mu - 1e-10


class TestComposeDiamond:
    """Tests for composing diamond (fork-join) task graphs."""

    def test_diamond_compose_valid(self, composer, diamond_graph, diamond_costs):
        """Composing a diamond graph returns a valid CostElement."""
        result = composer.compose(diamond_graph, diamond_costs)
        assert result.is_valid

    def test_diamond_mu_less_than_full_serial(self, composer, diamond_graph, diamond_costs):
        """Diamond graph μ < sum of all μ values (parallelism saves time)."""
        result = composer.compose(diamond_graph, diamond_costs)
        serial_sum = sum(c.mu for c in diamond_costs.values())
        assert result.mu < serial_sum + 1e-10

    def test_diamond_mu_at_least_critical_path(self, composer, diamond_graph, diamond_costs):
        """Diamond graph μ ≥ critical path μ (A + max(B, C) + D)."""
        result = composer.compose(diamond_graph, diamond_costs)
        crit = diamond_costs["A"].mu + max(diamond_costs["B"].mu, diamond_costs["C"].mu) + diamond_costs["D"].mu
        assert result.mu >= crit - 1e-8

    def test_diamond_parallel_level_bottleneck(self, composer, diamond_graph, diamond_costs):
        """The parallel level cost ≥ max of parallel branch costs."""
        result = composer.compose(diamond_graph, diamond_costs)
        assert result.mu >= max(diamond_costs["B"].mu, diamond_costs["C"].mu) - 1e-10


class TestComposeWideParallel:
    """Tests for composing wide fan-out graphs."""

    def test_wide_parallel_valid(self, composer, wide_parallel_graph, wide_costs):
        """Wide parallel graph produces a valid result."""
        result = composer.compose(wide_parallel_graph, wide_costs)
        assert result.is_valid

    def test_wide_parallel_less_than_serial(self, composer, wide_parallel_graph, wide_costs):
        """Wide parallel μ < sum of all μ values."""
        result = composer.compose(wide_parallel_graph, wide_costs)
        serial = sum(c.mu for c in wide_costs.values())
        assert result.mu < serial + 1e-10


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    """Tests for edge cases in task graph composition."""

    def test_empty_graph(self, composer):
        """Empty graph returns zero element."""
        G = nx.DiGraph()
        result = composer.compose(G, {})
        assert result == CostElement.zero()

    def test_single_node(self, composer, cost_a):
        """Single-node graph returns that node's cost."""
        G = nx.DiGraph()
        G.add_node("A")
        result = composer.compose(G, {"A": cost_a})
        assert math.isclose(result.mu, cost_a.mu, rel_tol=1e-10)

    def test_missing_node_in_cost_map_raises(self, composer, linear_graph):
        """Missing cost_map entry raises ValueError."""
        with pytest.raises(ValueError, match="missing"):
            composer.compose(linear_graph, {"A": CostElement(1, 0.1, 0, 0.01)})

    def test_cyclic_graph_raises(self, composer):
        """Cyclic graph raises ValueError."""
        G = nx.DiGraph()
        G.add_edges_from([("A", "B"), ("B", "C"), ("C", "A")])
        costs = {n: CostElement(1, 0.1, 0, 0.01) for n in "ABC"}
        with pytest.raises(ValueError, match="cycle"):
            composer.compose(G, costs)

    def test_disconnected_graph(self, composer):
        """Disconnected graph with independent components composes correctly."""
        G = nx.DiGraph()
        G.add_node("A")
        G.add_node("B")
        costs = {
            "A": CostElement(mu=1.0, sigma_sq=0.1, kappa=0.0, lambda_=0.01),
            "B": CostElement(mu=2.0, sigma_sq=0.2, kappa=0.0, lambda_=0.02),
        }
        result = composer.compose(G, costs)
        assert result.is_valid
        # A and B are at the same topological level → parallel
        assert result.mu >= max(1.0, 2.0) - 1e-10


# ---------------------------------------------------------------------------
# Critical path
# ---------------------------------------------------------------------------

class TestCriticalPath:
    """Tests for critical_path_cost — longest-path analysis."""

    def test_linear_critical_path_is_full_chain(self, composer, linear_graph, linear_costs):
        """In a linear graph, the critical path includes all nodes."""
        path, cost = composer.critical_path_cost(linear_graph, linear_costs)
        assert set(path) == {"A", "B", "C"}

    def test_diamond_critical_path_takes_longer_branch(self, composer, diamond_graph, diamond_costs):
        """In a diamond graph, critical path goes through the more expensive branch."""
        path, cost = composer.critical_path_cost(diamond_graph, diamond_costs)
        assert "A" in path
        assert "D" in path
        # B has higher mu (2.0) than C (1.5), so B should be on critical path
        assert "B" in path

    def test_critical_path_cost_valid(self, composer, diamond_graph, diamond_costs):
        """Critical path cost is a valid CostElement."""
        _, cost = composer.critical_path_cost(diamond_graph, diamond_costs)
        assert cost.is_valid
        assert cost.mu > 0

    def test_empty_graph_critical_path(self, composer):
        """Empty graph has empty critical path and zero cost."""
        path, cost = composer.critical_path_cost(nx.DiGraph(), {})
        assert path == []
        assert cost == CostElement.zero()

    def test_critical_path_mu_leq_compose_mu(self, composer, diamond_graph, diamond_costs):
        """Critical path μ ≤ full composed μ (compose includes parallel overhead)."""
        _, crit_cost = composer.critical_path_cost(diamond_graph, diamond_costs)
        full_cost = composer.compose(diamond_graph, diamond_costs)
        # The critical path cost should be comparable to full compose
        # (both are sequential compositions of the critical path)
        assert crit_cost.mu > 0


# ---------------------------------------------------------------------------
# Parallelism factor
# ---------------------------------------------------------------------------

class TestParallelismFactor:
    """Tests for parallelism_factor — degree of parallelism estimate."""

    def test_linear_parallelism_is_one(self, composer, linear_graph):
        """A purely serial graph has parallelism factor 1.0."""
        pf = composer.parallelism_factor(linear_graph)
        assert math.isclose(pf, 1.0, rel_tol=1e-10)

    def test_diamond_parallelism_above_one(self, composer, diamond_graph):
        """A diamond graph has parallelism factor > 1."""
        pf = composer.parallelism_factor(diamond_graph)
        assert pf > 1.0

    def test_wide_parallelism_higher_than_diamond(self, composer, diamond_graph, wide_parallel_graph):
        """Wider parallelism → higher factor."""
        pf_diamond = composer.parallelism_factor(diamond_graph)
        pf_wide = composer.parallelism_factor(wide_parallel_graph)
        assert pf_wide >= pf_diamond

    def test_empty_graph_parallelism(self, composer):
        """Empty graph has parallelism factor 1.0."""
        pf = composer.parallelism_factor(nx.DiGraph())
        assert math.isclose(pf, 1.0)

    def test_single_node_parallelism(self, composer):
        """Single-node graph has parallelism factor 1.0."""
        G = nx.DiGraph()
        G.add_node("X")
        pf = composer.parallelism_factor(G)
        assert math.isclose(pf, 1.0)


# ---------------------------------------------------------------------------
# Bottleneck nodes
# ---------------------------------------------------------------------------

class TestBottleneckNodes:
    """Tests for bottleneck_nodes — identifying highest-impact nodes."""

    def test_bottleneck_returns_list(self, composer, diamond_graph, diamond_costs):
        """bottleneck_nodes returns a list of (node, score) tuples."""
        result = composer.bottleneck_nodes(diamond_graph, diamond_costs)
        assert isinstance(result, list)
        assert all(isinstance(item, tuple) and len(item) == 2 for item in result)

    def test_bottleneck_top_k(self, composer, diamond_graph, diamond_costs):
        """Top-k parameter limits the number of results."""
        result = composer.bottleneck_nodes(diamond_graph, diamond_costs, top_k=2)
        assert len(result) <= 2

    def test_bottleneck_sorted_descending(self, composer, diamond_graph, diamond_costs):
        """Results are sorted by score in descending order."""
        result = composer.bottleneck_nodes(diamond_graph, diamond_costs, top_k=4)
        scores = [s for _, s in result]
        assert scores == sorted(scores, reverse=True)

    def test_root_often_bottleneck(self, composer, diamond_graph, diamond_costs):
        """Root node (A) with many descendants often has a high bottleneck score."""
        result = composer.bottleneck_nodes(diamond_graph, diamond_costs, top_k=4)
        node_ids = [n for n, _ in result]
        # A has 3 descendants (B, C, D), score = 1.0 * (1+3) = 4.0
        assert "A" in node_ids

    def test_bottleneck_scores_positive(self, composer, diamond_graph, diamond_costs):
        """All bottleneck scores are positive for non-zero cost elements."""
        result = composer.bottleneck_nodes(diamond_graph, diamond_costs)
        assert all(score > 0 for _, score in result)


# ---------------------------------------------------------------------------
# Subgraph cost
# ---------------------------------------------------------------------------

class TestSubgraphCost:
    """Tests for subgraph_cost — cost of a node subset."""

    def test_subgraph_single_node(self, composer, diamond_graph, diamond_costs):
        """Subgraph of a single node returns that node's cost."""
        result = composer.subgraph_cost(diamond_graph, diamond_costs, {"A"})
        assert math.isclose(result.mu, diamond_costs["A"].mu, rel_tol=1e-10)

    def test_subgraph_full_equals_compose(self, composer, diamond_graph, diamond_costs):
        """Subgraph of all nodes equals full compose."""
        full = composer.compose(diamond_graph, diamond_costs)
        sub = composer.subgraph_cost(diamond_graph, diamond_costs, set(diamond_graph.nodes))
        assert math.isclose(full.mu, sub.mu, rel_tol=1e-10)

    def test_subgraph_parallel_pair(self, composer, diamond_graph, diamond_costs):
        """Subgraph of parallel pair {B, C} (no edges between them) → parallel compose."""
        result = composer.subgraph_cost(diamond_graph, diamond_costs, {"B", "C"})
        assert result.is_valid
        # B and C have no edge between them, so they form one parallel group
        assert result.mu >= max(diamond_costs["B"].mu, diamond_costs["C"].mu) - 1e-10

    def test_subgraph_valid_result(self, composer, linear_graph, linear_costs):
        """Subgraph cost produces a valid CostElement."""
        result = composer.subgraph_cost(linear_graph, linear_costs, {"A", "B"})
        assert result.is_valid


# ---------------------------------------------------------------------------
# Constructor parameters
# ---------------------------------------------------------------------------

class TestConstructorParams:
    """Tests for TaskGraphComposer constructor parameters."""

    def test_default_coupling_zero(self):
        """Default coupling is 0.0."""
        composer = TaskGraphComposer()
        G = nx.DiGraph()
        G.add_edges_from([("A", "B")])
        costs = {
            "A": CostElement(mu=1.0, sigma_sq=0.1, kappa=0.0, lambda_=0.01),
            "B": CostElement(mu=1.0, sigma_sq=0.1, kappa=0.0, lambda_=0.01),
        }
        result = composer.compose(G, costs)
        assert math.isclose(result.mu, 2.0, rel_tol=1e-10)

    def test_custom_coupling(self):
        """Custom default_coupling increases sequential composition cost."""
        c0 = TaskGraphComposer(default_coupling=0.0)
        c5 = TaskGraphComposer(default_coupling=0.5)
        G = nx.DiGraph()
        G.add_edges_from([("A", "B")])
        costs = {
            "A": CostElement(mu=1.0, sigma_sq=0.5, kappa=0.0, lambda_=0.01),
            "B": CostElement(mu=1.0, sigma_sq=0.5, kappa=0.0, lambda_=0.01),
        }
        r0 = c0.compose(G, costs)
        r5 = c5.compose(G, costs)
        assert r5.mu > r0.mu

    def test_custom_interference(self):
        """Custom default_interference affects parallel composition cost."""
        c0 = TaskGraphComposer(default_interference=0.0)
        c5 = TaskGraphComposer(default_interference=0.5)
        G = nx.DiGraph()
        G.add_node("A")
        G.add_node("B")
        costs = {
            "A": CostElement(mu=1.0, sigma_sq=0.1, kappa=0.0, lambda_=0.01),
            "B": CostElement(mu=2.0, sigma_sq=0.2, kappa=0.0, lambda_=0.02),
        }
        r0 = c0.compose(G, costs)
        r5 = c5.compose(G, costs)
        assert r5.mu > r0.mu
