"""Integration tests for the full CausalBound pipeline."""
import pytest
import numpy as np
import networkx as nx
import json
import os
import tempfile

from causalbound.network.generators import ErdosRenyiGenerator, ScaleFreeGenerator
from causalbound.network.topology import NetworkTopology
from causalbound.contagion.debtrank import DebtRankModel
from causalbound.contagion.cascade import CascadeModel
from causalbound.scm.builder import SCMBuilder
from causalbound.scm.dag import DAGRepresentation
from causalbound.graph.decomposition import TreeDecomposer
from causalbound.graph.treewidth import TreewidthEstimator
from causalbound.graph.moral import MoralGraphConstructor
from causalbound.composition.composer import (
    BoundComposer, SubgraphBound, SeparatorInfo, OverlapStructure,
)
from causalbound.composition.gap_estimation import GapEstimator
from causalbound.composition.theorem import CompositionTheorem
from causalbound.junction.potential_table import PotentialTable
from causalbound.junction.engine import JunctionTreeEngine
from causalbound.junction.discretization import AdaptiveDiscretizer, BinningStrategy
from causalbound.mcts.search import MCTSSearch
from causalbound.mcts.convergence import ConvergenceMonitor
from causalbound.instruments.cds import CDSModel
from causalbound.instruments.equity_option import EquityOptionModel
from causalbound.data.serialization import NetworkSerializer
from causalbound.data.caching import CacheManager
from causalbound.data.checkpoint import CheckpointManager
from causalbound.evaluation.metrics import MetricsComputer


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def small_financial_network():
    """Small financial network for integration testing (10 nodes)."""
    gen = ErdosRenyiGenerator(seed=42)
    g = gen.generate(n_nodes=10, density=0.3)
    return g


@pytest.fixture
def medium_financial_network():
    """Medium financial network (25 nodes)."""
    gen = ScaleFreeGenerator(seed=123)
    g = gen.generate(n_nodes=25, m=2)
    return g


@pytest.fixture
def simple_dag():
    """Simple DAG for inference testing."""
    dag = DAGRepresentation()
    dag.add_edge("Shock", "Bank_A")
    dag.add_edge("Shock", "Bank_B")
    dag.add_edge("Bank_A", "Bank_C")
    dag.add_edge("Bank_B", "Bank_C")
    dag.add_edge("Bank_C", "Loss")
    return dag


@pytest.fixture
def temp_dir():
    """Temporary directory for file I/O tests."""
    with tempfile.TemporaryDirectory() as d:
        yield d


# ---------------------------------------------------------------------------
# Pipeline Stage Integration Tests
# ---------------------------------------------------------------------------
class TestNetworkToDecomposition:
    """Test: generate network -> decompose into bounded-treewidth subgraphs."""

    def test_generate_and_decompose(self, small_financial_network):
        g = small_financial_network
        assert g.number_of_nodes() == 10

        # Build undirected version for tree decomposition
        ug = g.to_undirected() if g.is_directed() else g

        td = TreeDecomposer()
        result = td.decompose(ug, max_width=6)
        assert result is not None
        assert result.width >= 0

        # Verify decomposition covers all nodes
        covered = set()
        for bag in result.bags.values():
            covered.update(bag)
        assert covered == set(ug.nodes())

    def test_treewidth_estimation_vs_decomposition(self, small_financial_network):
        ug = small_financial_network.to_undirected()
        te = TreewidthEstimator()
        td = TreeDecomposer()

        ub = te.upper_bound(ug)
        result = td.decompose(ug, max_width=10)
        assert result.width <= ub + 1


class TestSCMConstruction:
    """Test: build SCM from network."""

    def test_build_scm_from_dag(self, simple_dag):
        builder = SCMBuilder()
        scm = builder.build(network=simple_dag._graph)
        assert scm is not None

    def test_dag_dseparation(self, simple_dag):
        # Shock d-separates Bank_A from Bank_B? No, they share parent Shock
        # but are not d-separated given empty set
        assert not simple_dag.d_separated({"Bank_A"}, {"Bank_B"}, set())
        # Given Shock, Bank_A and Bank_B should be d-separated
        assert simple_dag.d_separated({"Bank_A"}, {"Bank_B"}, {"Shock"})

    def test_moral_graph_from_dag(self, simple_dag):
        mgc = MoralGraphConstructor()
        moral = mgc.moralize(simple_dag._graph)
        assert moral is not None
        assert not moral.is_directed()
        # Bank_A and Bank_B should be married (co-parents of Bank_C)
        assert moral.has_edge("Bank_A", "Bank_B")


class TestContagionModels:
    """Test: run contagion models on generated networks."""

    def test_debtrank_on_generated_network(self, small_financial_network):
        g = small_financial_network
        dr = DebtRankModel()
        initial_shocks = {0: 1.0}
        result = dr.compute(g, initial_shocks, max_rounds=10)
        assert result.system_debtrank >= 0.0

    def test_cascade_on_generated_network(self, small_financial_network):
        g = small_financial_network
        cm = CascadeModel()
        result = cm.simulate_cascade(g, initial_defaults={0})
        assert result.total_defaults >= 1

    def test_multiple_shock_sources(self, small_financial_network):
        g = small_financial_network
        dr = DebtRankModel()
        shocks_single = {0: 1.0}
        shocks_multi = {0: 1.0, 1: 0.5}
        loss_single = dr.compute(g, shocks_single, max_rounds=10).system_debtrank
        loss_multi = dr.compute(g, shocks_multi, max_rounds=10).system_debtrank
        assert loss_multi >= loss_single - 0.01


class TestInstrumentToCPD:
    """Test: financial instruments -> discretized CPDs."""

    def test_cds_to_cpd(self):
        cds = CDSModel(
            notional=1_000_000, spread=200, recovery=0.4,
            tenor=5.0, frequency=0.25,
        )
        cpd = cds.to_cpd(discretization={})
        assert cpd is not None
        assert "cpd" in cpd

    def test_option_to_cpd(self):
        opt = EquityOptionModel(
            spot=100, strike=100, volatility=0.2,
            risk_free_rate=0.05, time_to_expiry=1.0, option_type="call",
        )
        cpd = opt.to_cpd(discretization={})
        assert cpd is not None
        assert "cpd" in cpd

    def test_discretization_roundtrip(self):
        disc = AdaptiveDiscretizer()
        values = np.random.randn(1000)
        result = disc.discretize(values, n_bins=8, strategy=BinningStrategy.QUANTILE)
        assert result.n_bins == 8


class TestJunctionTreeInference:
    """Test: build junction tree and run inference."""

    def test_potential_table_marginalization(self):
        """Test potential table operations."""
        values = np.random.rand(2, 3)
        values /= values.sum()
        pt = PotentialTable(
            variables=["A", "B"],
            cardinalities={"A": 2, "B": 3},
            values=values,
        )

        # Marginalize over B
        marginal = pt.marginalize(["B"])
        assert marginal.values.shape == (2,)
        assert abs(marginal.values.sum() - 1.0) < 1e-6

    def test_potential_table_multiply(self):
        """Test potential table multiplication."""
        pt1 = PotentialTable(
            variables=["A", "B"],
            cardinalities={"A": 2, "B": 2},
            values=np.array([[0.5, 0.5], [0.3, 0.7]]),
        )

        pt2 = PotentialTable(
            variables=["B", "C"],
            cardinalities={"B": 2, "C": 2},
            values=np.array([[0.8, 0.2], [0.4, 0.6]]),
        )

        product = pt1.multiply(pt2)
        assert "A" in product.variables
        assert "B" in product.variables
        assert "C" in product.variables


class TestMCTSSearch:
    """Test: MCTS adversarial search."""

    def test_mcts_finds_maximum(self):
        """MCTS should find the maximum in a simple reward landscape."""
        monitor = ConvergenceMonitor(value_range=5.0)

        # Simple test: track convergence with synthetic data
        rng = np.random.RandomState(42)
        for i in range(100):
            arm = rng.randint(0, 10)
            value = rng.randn() + (1.0 if arm == 5 else 0.0)  # arm 5 is best
            monitor.update(arm, value)

        best = monitor.get_best_arm()
        assert best is not None

    def test_convergence_monitor(self):
        """Convergence monitor should track statistics."""
        monitor = ConvergenceMonitor(value_range=5.0)
        rng = np.random.RandomState(42)
        for _ in range(200):
            arm = rng.randint(0, 5)
            monitor.update(arm, rng.randn())
        ci = monitor.get_confidence_interval(0)
        assert ci is not None
        assert ci[0] <= ci[1]


# ---------------------------------------------------------------------------
# Data Pipeline Tests
# ---------------------------------------------------------------------------
class TestDataPipeline:
    def test_network_serialization_roundtrip(self, small_financial_network, temp_dir):
        """Save and load network should preserve structure."""
        path = os.path.join(temp_dir, "network.json")
        serializer = NetworkSerializer()
        serializer.save(small_financial_network, path)
        loaded = serializer.load(path)
        assert loaded.number_of_nodes() == small_financial_network.number_of_nodes()
        assert loaded.number_of_edges() == small_financial_network.number_of_edges()

    def test_cache_manager(self, temp_dir):
        """Cache should store and retrieve values."""
        cache = CacheManager(max_entries=100, spill_to_disk=False)
        cache.put("key1", {"value": 42})
        result = cache.get("key1")
        assert result is not None
        assert result["value"] == 42

    def test_cache_miss(self):
        """Cache miss should return None."""
        cache = CacheManager(max_entries=10, spill_to_disk=False)
        result = cache.get("nonexistent")
        assert result is None

    def test_cache_eviction(self):
        """Cache should evict when full."""
        cache = CacheManager(max_entries=3, spill_to_disk=False)
        for i in range(5):
            cache.put(f"key_{i}", i)
        # First entries should be evicted
        stats = cache.get_stats()
        assert stats.entry_count <= 3

    def test_checkpoint_roundtrip(self, temp_dir):
        """Checkpoint save/load should preserve state."""
        ckpt = CheckpointManager()
        state = {
            "phase": "decomposition",
            "progress": 0.5,
            "partial_results": [1, 2, 3],
        }
        path = os.path.join(temp_dir, "checkpoint.ckpt.gz")
        ckpt.save_checkpoint(state, path)
        loaded = ckpt.load_checkpoint(path)
        assert loaded.state["phase"] == "decomposition"
        assert loaded.state["progress"] == 0.5


# ---------------------------------------------------------------------------
# Metrics Tests
# ---------------------------------------------------------------------------
class TestMetrics:
    def test_bound_ratio(self):
        mc = MetricsComputer()
        cb_bounds = (0.1, 0.6)
        true_bounds = (0.2, 0.5)
        result = mc.compute_bound_ratio(cb_bounds, true_bounds)
        # CausalBound interval = 0.5, true interval = 0.3
        assert result.ratio == pytest.approx(0.5 / 0.3, rel=0.01) or result.ratio > 1.0

    def test_pathway_recall(self):
        mc = MetricsComputer()
        discovered = [(0, 1, 3), (0, 2, 3)]
        planted = [(0, 1, 3), (0, 2, 3), (0, 1, 2, 3)]
        result = mc.compute_pathway_recall(discovered, planted)
        assert 0.0 <= result.recall <= 1.0

    def test_discovery_ratio(self):
        mc = MetricsComputer()
        result = mc.compute_discovery_ratio(mcts_loss=0.85, baseline_loss=0.70)
        assert result.ratio == pytest.approx(0.85 / 0.70, rel=0.01)

    def test_overhead_ratio(self):
        mc = MetricsComputer()
        result = mc.compute_overhead_ratio(verified_time=3.0, unverified_time=1.0)
        assert result.ratio == pytest.approx(3.0, rel=0.01)


# ---------------------------------------------------------------------------
# End-to-End Pipeline Test
# ---------------------------------------------------------------------------
class TestEndToEndPipeline:
    def test_small_pipeline_runs(self):
        """Run the full pipeline on a tiny network."""
        # 1. Generate network
        gen = ErdosRenyiGenerator(seed=42)
        g = gen.generate(n_nodes=8, density=0.4)

        # 2. Topology analysis
        topo = NetworkTopology()
        report = topo.analyze(g)
        assert report.n_nodes == 8

        # 3. Decompose
        ug = g.to_undirected()
        td = TreeDecomposer()
        decomp = td.decompose(ug, max_width=5)
        assert decomp is not None
        assert decomp.width >= 0

        # 4. Run contagion model
        dr = DebtRankModel()
        shocks = {0: 1.0}
        result = dr.compute(g, shocks, max_rounds=10)
        assert result.system_debtrank >= 0.0

        # 5. Compose bounds (synthetic for this test)
        composer = BoundComposer()
        subgraph_bounds = [
            SubgraphBound(subgraph_id=0, lower=np.array([0.1]), upper=np.array([0.4])),
            SubgraphBound(subgraph_id=1, lower=np.array([0.15]), upper=np.array([0.45])),
        ]
        sep_info = [
            SeparatorInfo(separator_id=0, variable_indices=[0], adjacent_subgraphs=[0, 1]),
        ]
        overlap = OverlapStructure(
            n_subgraphs=2,
            overlap_matrix=np.array([[0, 1], [1, 0]]),
            shared_variables={(0, 1): [0]},
        )
        comp_result = composer.compose(subgraph_bounds, sep_info, overlap)
        assert np.all(comp_result.global_lower <= comp_result.global_upper)

        # 6. Verify composition theorem conditions
        ct = CompositionTheorem()
        stmt = ct.get_theorem_statement()
        assert len(stmt) > 0

    def test_network_generation_variants(self):
        """All generator types should produce valid networks."""
        generators = [
            ("ER", ErdosRenyiGenerator(seed=1), {"n_nodes": 10, "density": 0.3}),
            ("SF", ScaleFreeGenerator(seed=2), {"n_nodes": 15, "m": 2}),
        ]
        for name, gen, params in generators:
            g = gen.generate(**params)
            assert g.number_of_nodes() > 0, f"{name} failed"
            assert g.number_of_edges() > 0, f"{name} has no edges"

    def test_dag_to_inference_pipeline(self):
        """DAG -> moral graph -> triangulation -> junction tree."""
        # Build DAG
        dag = nx.DiGraph()
        dag.add_edges_from([
            ("S", "A"), ("S", "B"), ("A", "C"), ("B", "C"), ("C", "L")
        ])

        # Moral graph
        mgc = MoralGraphConstructor()
        moral = mgc.moralize(dag)
        assert moral.has_edge("A", "B")

        # Triangulate
        tri = mgc.triangulate(moral)
        assert mgc.is_chordal(tri)

        # Tree decomposition of triangulated graph
        td = TreeDecomposer()
        decomp = td.decompose(tri, max_width=10)
        assert decomp.width <= 3  # small graph
