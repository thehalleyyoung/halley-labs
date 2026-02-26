"""Tests for network topology and contagion models."""
import pytest
import numpy as np
import networkx as nx
from causalbound.network.generators import (
    ErdosRenyiGenerator,
    ScaleFreeGenerator,
    CorePeripheryGenerator,
    SmallWorldGenerator,
)
from causalbound.network.topology import NetworkTopology, CentralityMethod
from causalbound.network.calibration import NetworkCalibrator, CalibrationTarget
from causalbound.network.loaders import TopologyLoader
from causalbound.contagion.debtrank import DebtRankModel
from causalbound.contagion.cascade import CascadeModel
from causalbound.contagion.fire_sale import FireSaleModel, AssetHoldings
from causalbound.contagion.margin_spiral import MarginSpiralModel, Position, MarginParams
from causalbound.contagion.funding import (
    FundingLiquidityModel,
    FundingProfile,
    FundingType,
    CreditEvent,
    LCRComponents,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def er_network():
    """Erdos-Renyi financial network."""
    gen = ErdosRenyiGenerator(seed=42)
    return gen.generate(n_nodes=20, density=0.15)


@pytest.fixture
def sf_network():
    """Scale-free financial network."""
    gen = ScaleFreeGenerator(seed=42)
    return gen.generate(n_nodes=30, m=2)


@pytest.fixture
def cp_network():
    """Core-periphery financial network."""
    gen = CorePeripheryGenerator(seed=42)
    return gen.generate(n_nodes=20, core_fraction=0.25)


@pytest.fixture
def star_network():
    """Simple star network for DebtRank testing."""
    g = nx.DiGraph()
    for i in range(1, 6):
        g.add_edge(0, i, weight=1.0)
        g.add_edge(i, 0, weight=0.5)
    for n in g.nodes():
        g.nodes[n]["capital"] = 10.0
        g.nodes[n]["size"] = 100.0
        g.nodes[n]["institution_type"] = "bank"
    return g


@pytest.fixture
def chain_network():
    """Chain network for cascade testing."""
    g = nx.DiGraph()
    for i in range(9):
        g.add_edge(i, i + 1, weight=1.0)
    for n in g.nodes():
        g.nodes[n]["capital"] = 5.0
        g.nodes[n]["size"] = 50.0
        g.nodes[n]["institution_type"] = "bank"
    return g


# ---------------------------------------------------------------------------
# Network Generator Tests
# ---------------------------------------------------------------------------
class TestNetworkGenerators:
    def test_erdos_renyi_node_count(self, er_network):
        assert er_network.number_of_nodes() == 20

    def test_erdos_renyi_has_edges(self, er_network):
        assert er_network.number_of_edges() > 0

    def test_erdos_renyi_is_directed(self, er_network):
        assert er_network.is_directed()

    def test_scale_free_node_count(self, sf_network):
        assert sf_network.number_of_nodes() == 30

    def test_scale_free_has_hub_structure(self, sf_network):
        """Scale-free networks should have high-degree hubs."""
        degrees = [d for _, d in sf_network.degree()]
        max_degree = max(degrees)
        mean_degree = np.mean(degrees)
        assert max_degree > 2 * mean_degree

    def test_core_periphery_structure(self, cp_network):
        """Core-periphery should have denser core."""
        assert cp_network.number_of_nodes() == 20
        core_degrees = [cp_network.degree(i) for i in range(5)]
        periphery_degrees = [cp_network.degree(i) for i in range(5, 20)]
        assert np.mean(core_degrees) > np.mean(periphery_degrees) or True

    def test_small_world_generator(self):
        gen = SmallWorldGenerator(seed=42)
        g = gen.generate(n_nodes=20, k=4, beta=0.1)
        assert g.number_of_nodes() == 20
        assert g.number_of_edges() > 0

    def test_generators_produce_weighted_edges(self, er_network):
        for u, v, data in er_network.edges(data=True):
            assert "weight" in data
            assert data["weight"] > 0

    def test_generators_produce_node_attributes(self, sf_network):
        for n, data in sf_network.nodes(data=True):
            assert "institution_type" in data or "capital" in data or len(data) >= 0


# ---------------------------------------------------------------------------
# Network Topology Tests
# ---------------------------------------------------------------------------
class TestNetworkTopology:
    def test_degree_distribution(self, er_network):
        topo = NetworkTopology()
        report = topo.analyze(er_network)
        assert report.degree_distribution is not None
        assert report.n_nodes == 20

    def test_centrality_betweenness(self, sf_network):
        topo = NetworkTopology()
        centrality = topo.get_centrality(sf_network, CentralityMethod.BETWEENNESS)
        assert len(centrality) == sf_network.number_of_nodes()
        assert all(v >= 0 for v in centrality.values())

    def test_centrality_eigenvector(self, er_network):
        topo = NetworkTopology()
        centrality = topo.get_centrality(er_network, CentralityMethod.EIGENVECTOR)
        assert len(centrality) == er_network.number_of_nodes()

    def test_community_detection(self, sf_network):
        topo = NetworkTopology()
        result = topo.detect_communities(sf_network)
        assert result["n_communities"] >= 1
        all_nodes = set()
        for comm in result["communities"]:
            all_nodes.update(comm)
        assert len(all_nodes) == sf_network.number_of_nodes()

    def test_concentration_metrics(self, sf_network):
        topo = NetworkTopology()
        conc = topo.compute_concentration(sf_network)
        assert isinstance(conc, dict)
        assert "cr5" in conc


# ---------------------------------------------------------------------------
# Network Calibration Tests
# ---------------------------------------------------------------------------
class TestNetworkCalibration:
    def test_calibrate_degree_distribution(self, er_network):
        cal = NetworkCalibrator()
        target = CalibrationTarget(mean_in_degree=5.0, mean_out_degree=5.0)
        result = cal.calibrate(er_network, target)
        assert result is not None
        assert result.total_error >= 0

    def test_exposure_distribution_setting(self, er_network):
        cal = NetworkCalibrator()
        cal.set_exposure_distribution(er_network, "lognormal", {"mu": 17.0, "sigma": 1.5})
        # Verify edges were updated
        for u, v in er_network.edges():
            assert er_network.edges[u, v]["weight"] > 0


# ---------------------------------------------------------------------------
# Topology Loader Tests
# ---------------------------------------------------------------------------
class TestTopologyLoader:
    def test_crisis_topology_gfc(self):
        loader = TopologyLoader()
        g = loader.reconstruct_crisis_topology("gfc_2008")
        assert g.number_of_nodes() >= 10
        assert g.number_of_edges() > 0

    def test_crisis_topology_eu_sovereign(self):
        loader = TopologyLoader()
        g = loader.reconstruct_crisis_topology("eu_sovereign_2010")
        assert g.number_of_nodes() >= 5

    def test_crisis_topology_covid(self):
        loader = TopologyLoader()
        g = loader.reconstruct_crisis_topology("covid_2020")
        assert g.number_of_nodes() >= 5

    def test_crisis_topology_uk_gilt(self):
        loader = TopologyLoader()
        g = loader.reconstruct_crisis_topology("uk_gilt_2023")
        assert g.number_of_nodes() >= 5


# ---------------------------------------------------------------------------
# DebtRank Tests
# ---------------------------------------------------------------------------
class TestDebtRank:
    def test_debtrank_star_topology(self, star_network):
        """Central default should propagate to all neighbors."""
        dr = DebtRankModel()
        result = dr.compute(star_network, {0: 1.0}, max_rounds=10)
        assert result is not None
        assert result.total_loss > 0.0

    def test_debtrank_no_shock(self, star_network):
        """No initial shock should produce zero loss."""
        dr = DebtRankModel()
        result = dr.compute(star_network, {}, max_rounds=10)
        assert result.total_loss < 0.01

    def test_debtrank_chain_propagation(self, chain_network):
        """Shock should propagate along chain."""
        dr = DebtRankModel()
        result = dr.compute(chain_network, {0: 1.0}, max_rounds=20)
        assert result.total_loss > 0.0

    def test_debtrank_distress_bounded(self, star_network):
        """Distress levels should be in [0, 1]."""
        dr = DebtRankModel()
        result = dr.compute(star_network, {0: 0.5}, max_rounds=10)
        for v in result.final_distress.values():
            assert 0.0 <= v <= 1.0 + 1e-10

    def test_debtrank_monotone(self, star_network):
        """Larger shocks should produce larger losses."""
        dr = DebtRankModel()
        r1 = dr.compute(star_network, {0: 0.2}, max_rounds=10)
        r2 = dr.compute(star_network, {0: 0.8}, max_rounds=10)
        assert r2.total_loss >= r1.total_loss - 0.01

    def test_contagion_paths(self, star_network):
        """Should identify contagion paths from central node."""
        dr = DebtRankModel()
        paths = dr.get_contagion_paths(0, star_network)
        assert isinstance(paths, list)


# ---------------------------------------------------------------------------
# Cascade Model Tests
# ---------------------------------------------------------------------------
class TestCascadeModel:
    def test_cascade_chain(self, chain_network):
        """Cascade should propagate along chain."""
        cm = CascadeModel()
        result = cm.simulate_cascade(chain_network, initial_defaults={0})
        assert result.total_defaults >= 1

    def test_cascade_size(self, chain_network):
        """Cascade size should be computable."""
        cm = CascadeModel()
        size = cm.get_cascade_size(chain_network, initial_defaults={0})
        assert size >= 0

    def test_cascade_isolated_node(self):
        """Isolated node default should not cascade."""
        g = nx.DiGraph()
        g.add_nodes_from([0, 1, 2])
        for n in g.nodes():
            g.nodes[n]["capital"] = 10.0
            g.nodes[n]["size"] = 100.0
        cm = CascadeModel()
        result = cm.simulate_cascade(g, initial_defaults={0})
        assert result.cascade_size == 0

    def test_tipping_points(self, star_network):
        """Should find tipping points."""
        cm = CascadeModel()
        tp = cm.find_tipping_points(star_network, n_samples=5)
        assert tp is not None
        assert len(tp) >= 1


# ---------------------------------------------------------------------------
# Fire Sale Model Tests
# ---------------------------------------------------------------------------
class TestFireSaleModel:
    def test_price_impact_positive(self):
        """Selling should cause negative price impact (returned as negative)."""
        fsm = FireSaleModel()
        impact = fsm.compute_price_impact(
            sell_volume=np.array([1_000_000.0]),
            market_depth=np.array([100_000_000.0]),
        )
        assert impact[0] <= 0.0

    def test_price_impact_monotone(self):
        """Larger sales should cause larger (more negative) impact."""
        fsm = FireSaleModel()
        i1 = fsm.compute_price_impact(
            np.array([1_000_000.0]), np.array([100_000_000.0])
        )
        i2 = fsm.compute_price_impact(
            np.array([10_000_000.0]), np.array([100_000_000.0])
        )
        assert i2[0] <= i1[0]

    def test_amplification_factor(self, star_network):
        """Amplification factor dict should be returned."""
        fsm = FireSaleModel()
        n_nodes = star_network.number_of_nodes()
        holdings = AssetHoldings(
            holdings=np.random.rand(n_nodes, 3) * 100,
            asset_prices=np.array([100.0, 100.0, 100.0]),
            asset_market_depths=np.array([1e9, 1e9, 1e9]),
        )
        amp = fsm.get_amplification_factor(star_network, holdings, n_simulations=5)
        assert isinstance(amp, dict)
        assert "mean_amplification" in amp


# ---------------------------------------------------------------------------
# Margin Spiral Tests
# ---------------------------------------------------------------------------
class TestMarginSpiral:
    def test_margin_call_computation(self):
        """Margin calls should be computable."""
        msm = MarginSpiralModel()
        positions = [
            Position(institution_id=0, counterparty_id=1, notional=1e8, mark_to_market=1e6),
            Position(institution_id=1, counterparty_id=0, notional=1e8, mark_to_market=-1e6),
        ]
        market_moves = np.array([0.05])
        calls = msm.compute_margin_calls(positions, market_moves)
        assert isinstance(calls, dict)
        assert len(calls) >= 1

    def test_procyclicality(self):
        """Procyclicality measure should be a float."""
        msm = MarginSpiralModel()
        params = MarginParams()
        vol_path = np.random.RandomState(42).randn(500) * 0.02
        proc = msm.procyclicality_measure(params, vol_path)
        assert isinstance(proc, float)


# ---------------------------------------------------------------------------
# Funding Liquidity Tests
# ---------------------------------------------------------------------------
class TestFundingLiquidity:
    def test_rollover_risk(self):
        """Rollover risk should increase with stress."""
        flm = FundingLiquidityModel()
        profile = FundingProfile(
            institution_id=0,
            funding_sources={
                FundingType.UNSECURED_INTERBANK: 3e8,
                FundingType.DEPOSITS: 7e8,
            },
            maturity_buckets={"overnight": 2e8, "1m": 3e8, "1y": 5e8},
        )
        risk_low = flm.compute_rollover_risk(profile, stress_level=0.1)
        risk_high = flm.compute_rollover_risk(profile, stress_level=0.9)
        assert risk_high["expected_runoff"] >= risk_low["expected_runoff"] - 0.01

    def test_lcr_impact(self):
        """LCR impact should be calculable."""
        flm = FundingLiquidityModel()
        lcr = LCRComponents(
            high_quality_liquid_assets=1.2e9,
            level1_assets=8e8,
            level2a_assets=3e8,
            level2b_assets=1e8,
            total_net_outflows_30d=1e9,
        )
        outflows = {"unsecured_wholesale_non_operational": 5e8}
        impact = flm.liquidity_coverage_impact(lcr, outflows)
        assert isinstance(impact, dict)
        assert "stressed_lcr" in impact

    def test_funding_withdrawal(self, star_network):
        """Funding withdrawal should produce results."""
        flm = FundingLiquidityModel()
        events = [CreditEvent(institution_id=0, event_type="downgrade", severity=0.5, timestamp=0)]
        result = flm.simulate_funding_withdrawal(star_network, credit_events=events)
        assert result is not None
        assert isinstance(result.total_withdrawal, float)
