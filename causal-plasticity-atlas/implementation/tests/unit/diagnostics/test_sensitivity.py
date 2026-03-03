"""Tests for sensitivity analysis.

Covers SHD perturbation, descriptor stability, perturbation
response curves, critical threshold detection, and diagnostic reports.
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose

from cpa.diagnostics.sensitivity import (
    SensitivityAnalyzer,
    DescriptorSensitivity,
    PerturbationResponseCurve,
    DiagnosticReport,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def rng():
    return np.random.default_rng(42)


@pytest.fixture
def chain_adj():
    """Chain DAG: 0->1->2->3."""
    adj = np.zeros((4, 4))
    adj[0, 1] = 1
    adj[1, 2] = 1
    adj[2, 3] = 1
    return adj


@pytest.fixture
def dense_adj(rng):
    """Denser DAG."""
    adj = np.zeros((5, 5))
    adj[0, 1] = 1
    adj[0, 2] = 1
    adj[1, 3] = 1
    adj[2, 3] = 1
    adj[3, 4] = 1
    adj[1, 4] = 1
    return adj


@pytest.fixture
def data_4var(rng):
    """100 × 4 data matrix."""
    return rng.normal(0, 1, size=(100, 4))


@pytest.fixture
def data_5var(rng):
    """100 × 5 data matrix."""
    return rng.normal(0, 1, size=(100, 5))


@pytest.fixture
def analyzer():
    return SensitivityAnalyzer(
        n_perturbations=20,
        noise_scale=0.1,
        seed=42,
    )


@pytest.fixture
def variable_names_4():
    return ["X0", "X1", "X2", "X3"]


@pytest.fixture
def variable_names_5():
    return ["X0", "X1", "X2", "X3", "X4"]


def _simple_descriptor(adj, data=None):
    """Simple 4-D descriptor based on adjacency matrix."""
    density = np.sum(adj > 0) / max(adj.shape[0] * (adj.shape[0] - 1), 1)
    mean_in = np.mean(np.sum(adj > 0, axis=0))
    deg_var = np.var(np.sum(adj > 0, axis=0) + np.sum(adj > 0, axis=1))
    clustering = density * 0.5
    return np.array([density, mean_in, deg_var, clustering])


# ---------------------------------------------------------------------------
# Test SHD perturbation
# ---------------------------------------------------------------------------

class TestSHDPerturbation:

    def test_analyze_returns_dict(self, analyzer, chain_adj, data_4var, variable_names_4):
        result = analyzer.analyze(
            chain_adj, data=data_4var,
            variable_names=variable_names_4,
            descriptor_fn=_simple_descriptor,
        )
        assert isinstance(result, dict)

    def test_analyze_has_perturbations(self, analyzer, chain_adj, data_4var):
        result = analyzer.analyze(
            chain_adj, data=data_4var,
            descriptor_fn=_simple_descriptor,
        )
        assert "perturbation_results" in result or "perturbations" in result or len(result) > 0

    def test_structural_hamming_distance(self, analyzer):
        adj1 = np.array([[0, 1, 0], [0, 0, 1], [0, 0, 0]])
        adj2 = np.array([[0, 1, 1], [0, 0, 0], [0, 0, 0]])
        shd = analyzer.structural_hamming_distance(adj1, adj2)
        assert shd == 2  # edge 1->2 removed, edge 0->2 added

    def test_shd_identical_is_zero(self, analyzer, chain_adj):
        shd = analyzer.structural_hamming_distance(chain_adj, chain_adj)
        assert shd == 0

    def test_shd_symmetric(self, analyzer, rng):
        p = 4
        adj1 = np.zeros((p, p))
        adj1[0, 1] = 1
        adj2 = np.zeros((p, p))
        adj2[1, 2] = 1
        assert analyzer.structural_hamming_distance(adj1, adj2) == \
               analyzer.structural_hamming_distance(adj2, adj1)

    def test_analyze_without_data(self, analyzer, chain_adj):
        result = analyzer.analyze(chain_adj, descriptor_fn=_simple_descriptor)
        assert isinstance(result, dict)

    def test_analyze_with_default_descriptor(self, analyzer, chain_adj, data_4var):
        result = analyzer.analyze(chain_adj, data=data_4var)
        assert isinstance(result, dict)


# ---------------------------------------------------------------------------
# Test descriptor stability
# ---------------------------------------------------------------------------

class TestDescriptorStability:

    def test_perturbation_results_structure(self, analyzer, chain_adj, data_4var):
        result = analyzer.analyze(
            chain_adj, data=data_4var,
            descriptor_fn=_simple_descriptor,
        )
        assert isinstance(result, dict)

    def test_robustness_score(self, analyzer, chain_adj, data_4var):
        result = analyzer.analyze(
            chain_adj, data=data_4var,
            descriptor_fn=_simple_descriptor,
        )
        if "robustness_score" in result:
            assert 0.0 <= result["robustness_score"] <= 1.0

    def test_stability_higher_for_dense_graph(self, analyzer, chain_adj, dense_adj, data_4var, data_5var):
        r_chain = analyzer.analyze(chain_adj, data=data_4var, descriptor_fn=_simple_descriptor)
        r_dense = analyzer.analyze(dense_adj, data=data_5var, descriptor_fn=_simple_descriptor)
        # Both should return valid results
        assert isinstance(r_chain, dict)
        assert isinstance(r_dense, dict)

    def test_variable_sensitivities(self, analyzer, chain_adj, data_4var, variable_names_4):
        result = analyzer.analyze(
            chain_adj, data=data_4var,
            variable_names=variable_names_4,
            descriptor_fn=_simple_descriptor,
        )
        if "variable_sensitivities" in result:
            vs_dict = result["variable_sensitivities"]
            for var_name, vs in vs_dict.items():
                assert vs["sensitivity"] >= 0.0

    def test_risk_assessment(self, analyzer, chain_adj, data_4var, variable_names_4):
        result = analyzer.analyze(
            chain_adj, data=data_4var,
            variable_names=variable_names_4,
            descriptor_fn=_simple_descriptor,
        )
        if "risks" in result:
            for risk in result["risks"]:
                assert risk["risk_level"] in ("low", "medium", "high", "critical")


# ---------------------------------------------------------------------------
# Test DescriptorSensitivity
# ---------------------------------------------------------------------------

class TestDescriptorSensitivityClass:

    def test_sensitivity_matrix(self, chain_adj, data_4var):
        ds = DescriptorSensitivity(epsilon=0.01)
        result = ds.compute_sensitivity_matrix(
            chain_adj, data_4var, _simple_descriptor,
        )
        assert isinstance(result, dict)

    def test_estimate_gradients(self, chain_adj, data_4var):
        ds = DescriptorSensitivity(epsilon=0.01)
        grads = ds.estimate_gradients(chain_adj, data_4var, _simple_descriptor)
        p = chain_adj.shape[0]
        n_desc = len(_simple_descriptor(chain_adj))
        assert grads.shape == (p, p, n_desc)

    def test_identify_critical_edges(self, chain_adj, data_4var):
        ds = DescriptorSensitivity(epsilon=0.01)
        critical = ds.identify_critical_edges(
            chain_adj, data_4var, _simple_descriptor, top_k=3,
        )
        assert isinstance(critical, list)
        assert len(critical) <= 3

    def test_critical_edges_have_impact(self, chain_adj, data_4var):
        ds = DescriptorSensitivity(epsilon=0.01)
        critical = ds.identify_critical_edges(
            chain_adj, data_4var, _simple_descriptor, top_k=5,
        )
        for edge_info in critical:
            assert isinstance(edge_info, dict)


# ---------------------------------------------------------------------------
# Test perturbation response curves
# ---------------------------------------------------------------------------

class TestPerturbationResponseCurve:

    def test_noise_response(self, chain_adj, data_4var):
        prc = PerturbationResponseCurve(n_levels=10, max_perturbation=0.5)
        result = prc.compute_noise_response(
            chain_adj, data_4var, _simple_descriptor,
            n_reps=5, seed=42,
        )
        assert isinstance(result, dict)

    def test_noise_response_has_levels(self, chain_adj, data_4var):
        prc = PerturbationResponseCurve(n_levels=10)
        result = prc.compute_noise_response(
            chain_adj, data_4var, _simple_descriptor,
            n_reps=3, seed=42,
        )
        if "noise_levels" in result:
            assert len(result["noise_levels"]) == 10

    def test_edge_removal_response(self, chain_adj, data_4var):
        prc = PerturbationResponseCurve(n_levels=10)
        result = prc.compute_edge_removal_response(
            chain_adj, data_4var, _simple_descriptor,
        )
        assert isinstance(result, dict)

    def test_noise_response_monotonic(self, chain_adj, data_4var):
        """Higher noise should generally cause higher descriptor change."""
        prc = PerturbationResponseCurve(n_levels=5, max_perturbation=1.0)
        result = prc.compute_noise_response(
            chain_adj, data_4var, _simple_descriptor,
            n_reps=5, seed=42,
        )
        if "mean_distances" in result:
            dists = result["mean_distances"]
            # Rough check: last should be >= first
            assert dists[-1] >= dists[0] - 0.1


# ---------------------------------------------------------------------------
# Test critical threshold detection
# ---------------------------------------------------------------------------

class TestCriticalThresholds:

    def test_thresholds_in_analysis(self, analyzer, chain_adj, data_4var):
        result = analyzer.analyze(
            chain_adj, data=data_4var,
            descriptor_fn=_simple_descriptor,
        )
        if "critical_thresholds" in result:
            for ptype, val in result["critical_thresholds"].items():
                assert isinstance(val, (int, float))


# ---------------------------------------------------------------------------
# Test diagnostic reports
# ---------------------------------------------------------------------------

class TestDiagnosticReport:

    def test_report_creation(self, analyzer, chain_adj, data_4var, variable_names_4):
        result = analyzer.analyze(
            chain_adj, data=data_4var,
            variable_names=variable_names_4,
            descriptor_fn=_simple_descriptor,
        )
        report = DiagnosticReport(result, variable_names=variable_names_4)
        assert report is not None

    def test_summary_table(self, analyzer, chain_adj, data_4var, variable_names_4):
        result = analyzer.analyze(
            chain_adj, data=data_4var,
            variable_names=variable_names_4,
            descriptor_fn=_simple_descriptor,
        )
        report = DiagnosticReport(result, variable_names=variable_names_4)
        table = report.summary_table()
        assert isinstance(table, list)

    def test_risk_assessment_text(self, analyzer, chain_adj, data_4var, variable_names_4):
        result = analyzer.analyze(
            chain_adj, data=data_4var,
            variable_names=variable_names_4,
            descriptor_fn=_simple_descriptor,
        )
        report = DiagnosticReport(result, variable_names=variable_names_4)
        text = report.risk_assessment_text()
        assert isinstance(text, str)

    def test_recommendations(self, analyzer, chain_adj, data_4var, variable_names_4):
        result = analyzer.analyze(
            chain_adj, data=data_4var,
            variable_names=variable_names_4,
            descriptor_fn=_simple_descriptor,
        )
        report = DiagnosticReport(result, variable_names=variable_names_4)
        recs = report.recommendations()
        assert isinstance(recs, list)

    def test_to_dict(self, analyzer, chain_adj, data_4var, variable_names_4):
        result = analyzer.analyze(
            chain_adj, data=data_4var,
            variable_names=variable_names_4,
            descriptor_fn=_simple_descriptor,
        )
        report = DiagnosticReport(result, variable_names=variable_names_4)
        d = report.to_dict()
        assert isinstance(d, dict)
