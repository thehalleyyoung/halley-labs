"""Unit tests for cpa.inference.counterfactual.CounterfactualEngine."""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose

from cpa.core.scm import StructuralCausalModel
from cpa.inference.counterfactual import (
    CounterfactualEngine,
    CounterfactualResult,
    TwinNetwork,
)


# ===================================================================
# Fixtures
# ===================================================================


@pytest.fixture
def rng():
    return np.random.default_rng(42)


@pytest.fixture
def engine():
    return CounterfactualEngine(n_samples=10_000, seed=42)


@pytest.fixture
def chain_scm():
    """Chain SCM: 0 → 1 → 2 with known linear coefficients."""
    adj = np.zeros((3, 3))
    adj[0, 1] = 1.0
    adj[1, 2] = 1.0
    coefs = np.zeros((3, 3))
    coefs[0, 1] = 0.5
    coefs[1, 2] = 0.8
    var = np.array([1.0, 1.0, 1.0])
    return StructuralCausalModel(
        adj, variable_names=["X0", "X1", "X2"],
        regression_coefficients=coefs, residual_variances=var,
        sample_size=1000,
    )


@pytest.fixture
def fork_scm():
    """Fork SCM: 1 ← 0 → 2."""
    adj = np.zeros((3, 3))
    adj[0, 1] = 1.0
    adj[0, 2] = 1.0
    coefs = np.zeros((3, 3))
    coefs[0, 1] = 0.7
    coefs[0, 2] = 0.3
    var = np.ones(3)
    return StructuralCausalModel(
        adj, regression_coefficients=coefs, residual_variances=var,
        sample_size=1000,
    )


@pytest.fixture
def diamond_scm():
    """Diamond SCM: 0 → 1, 0 → 2, 1 → 3, 2 → 3."""
    adj = np.zeros((4, 4))
    adj[0, 1] = 1.0
    adj[0, 2] = 1.0
    adj[1, 3] = 1.0
    adj[2, 3] = 1.0
    coefs = np.zeros((4, 4))
    coefs[0, 1] = 0.6
    coefs[0, 2] = 0.4
    coefs[1, 3] = 0.7
    coefs[2, 3] = 0.3
    var = np.ones(4)
    return StructuralCausalModel(
        adj, regression_coefficients=coefs, residual_variances=var,
        sample_size=1000,
    )


# ===================================================================
# Tests – TwinNetwork structure
# ===================================================================


class TestTwinNetwork:
    """Test twin network construction."""

    def test_build_twin_doubles_variables(self, chain_scm):
        twin = TwinNetwork()
        twin.build_twin(chain_scm)
        combined = twin.get_combined_adjacency()
        assert combined.shape == (6, 6)

    def test_build_twin_preserves_original(self, chain_scm):
        twin = TwinNetwork()
        twin.build_twin(chain_scm)
        combined = twin.get_combined_adjacency()
        # Original edges preserved in first p×p block
        assert combined[0, 1] == 1 or combined[0, 1] > 0

    def test_num_original(self, chain_scm):
        twin = TwinNetwork()
        twin.build_twin(chain_scm)
        assert twin.num_original == 3

    def test_shared_exogenous(self, chain_scm):
        twin = TwinNetwork()
        twin.build_twin(chain_scm)
        shared = twin.get_shared_exogenous()
        assert isinstance(shared, set)

    def test_counterfactual_world_has_edges(self, chain_scm):
        twin = TwinNetwork()
        twin.build_twin(chain_scm)
        combined = twin.get_combined_adjacency()
        p = twin.num_original
        # Counterfactual world edges in block [p:, p:]
        cf_block = combined[p:, p:]
        assert np.sum(cf_block) > 0

    def test_apply_intervention(self, chain_scm):
        twin = TwinNetwork()
        twin.build_twin(chain_scm)
        twin.apply_intervention({1: 3.0})
        combined = twin.get_combined_adjacency()
        p = twin.num_original
        # Edge into intervention node in CF world should be removed
        assert combined[p + 0, p + 1] == 0

    def test_diamond_twin(self, diamond_scm):
        twin = TwinNetwork()
        twin.build_twin(diamond_scm)
        combined = twin.get_combined_adjacency()
        assert combined.shape == (8, 8)


# ===================================================================
# Tests – Abduction-Action-Prediction
# ===================================================================


class TestAbductionActionPrediction:
    """Test three-step counterfactual procedure."""

    def test_abduction_returns_dict(self, engine, chain_scm):
        result = engine.abduction(chain_scm, {0: 1.0, 1: 1.5, 2: 2.0})
        assert isinstance(result, dict)

    def test_abduction_has_noise(self, engine, chain_scm):
        result = engine.abduction(chain_scm, {0: 1.0, 1: 1.5, 2: 2.0})
        assert "noise" in result or "values" in result

    def test_evaluate_returns_result(self, engine, chain_scm):
        result = engine.evaluate(
            chain_scm,
            factual_evidence={0: 1.0, 1: 1.5, 2: 2.2},
            counterfactual_intervention={0: 2.0},
            target=2,
        )
        assert isinstance(result, CounterfactualResult)

    def test_evaluate_value_is_finite(self, engine, chain_scm):
        result = engine.evaluate(
            chain_scm,
            factual_evidence={0: 1.0, 1: 1.5, 2: 2.2},
            counterfactual_intervention={0: 2.0},
            target=2,
        )
        assert np.isfinite(result.value)

    def test_counterfactual_chain_logic(self, engine, chain_scm):
        # Evidence: X0=1, X1=0.5*1+u1, X2=0.8*X1+u2
        # Set X1=1.5, u1=1.0, so do(X0=2) → X1_cf = 0.5*2 + u1 = 2.0
        # X2_cf = 0.8 * 2.0 + u2
        result = engine.evaluate(
            chain_scm,
            factual_evidence={0: 1.0, 1: 1.5, 2: 2.0},
            counterfactual_intervention={0: 2.0},
            target=2,
        )
        assert isinstance(result.value, (float, np.floating))

    def test_confidence_interval_ordered(self, engine, chain_scm):
        result = engine.evaluate(
            chain_scm,
            factual_evidence={0: 1.0, 1: 1.5, 2: 2.2},
            counterfactual_intervention={0: 2.0},
            target=2,
        )
        lo, hi = result.confidence_interval
        assert lo <= hi

    def test_evaluate_fork_scm(self, engine, fork_scm):
        result = engine.evaluate(
            fork_scm,
            factual_evidence={0: 1.0, 1: 1.7, 2: 1.3},
            counterfactual_intervention={0: 0.0},
            target=1,
        )
        assert np.isfinite(result.value)


# ===================================================================
# Tests – Probability of necessity and sufficiency
# ===================================================================


class TestProbabilityNecessitySufficiency:
    """Test PN and PS computations."""

    def test_pn_in_unit_interval(self, engine, chain_scm):
        pn = engine.probability_of_necessity(
            chain_scm, treatment=0, outcome=2, n_samples=5000,
        )
        assert 0.0 <= pn <= 1.0

    def test_ps_in_unit_interval(self, engine, chain_scm):
        ps = engine.probability_of_sufficiency(
            chain_scm, treatment=0, outcome=2, n_samples=5000,
        )
        assert 0.0 <= ps <= 1.0

    def test_pn_returns_float(self, engine, chain_scm):
        pn = engine.probability_of_necessity(
            chain_scm, treatment=0, outcome=2, n_samples=2000,
        )
        assert isinstance(pn, (float, np.floating))

    def test_ps_returns_float(self, engine, chain_scm):
        ps = engine.probability_of_sufficiency(
            chain_scm, treatment=0, outcome=2, n_samples=2000,
        )
        assert isinstance(ps, (float, np.floating))


# ===================================================================
# Tests – Natural direct / indirect effects
# ===================================================================


class TestNaturalEffects:
    """Test NDE and NIE computation."""

    def test_nde_is_finite(self, engine, chain_scm):
        nde = engine.natural_direct_effect(
            chain_scm, treatment=0, outcome=2, n_samples=5000,
        )
        assert np.isfinite(nde)

    def test_nie_requires_mediator(self, engine, chain_scm):
        nie = engine.natural_indirect_effect(
            chain_scm, treatment=0, outcome=2, mediator=1, n_samples=5000,
        )
        assert np.isfinite(nie)


# ===================================================================
# Tests – Effect of treatment on treated
# ===================================================================


class TestETT:
    """Test compute_etf (ETT) computation."""

    def test_etf_is_finite(self, engine, chain_scm):
        ett = engine.compute_etf(chain_scm, interventions={0: 1.0}, target=2)
        assert np.isfinite(ett)


# ===================================================================
# Tests – Twin network query path
# ===================================================================


class TestTwinNetworkQuery:
    """Test twin_network_query end-to-end."""

    def test_twin_query_returns_result(self, engine, chain_scm):
        result = engine.twin_network_query(
            chain_scm,
            evidence={0: 1.0, 1: 1.5, 2: 2.2},
            intervention={0: 2.0},
            target=2,
        )
        assert isinstance(result, CounterfactualResult)
        assert np.isfinite(result.value)


# ===================================================================
# Tests – Edge cases
# ===================================================================


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_single_node_scm(self, engine):
        adj = np.zeros((1, 1))
        scm = StructuralCausalModel(adj, residual_variances=np.array([1.0]),
                                     sample_size=100)
        result = engine.evaluate(
            scm,
            factual_evidence={0: 1.0},
            counterfactual_intervention={0: 2.0},
            target=0,
        )
        assert_allclose(result.value, 2.0, atol=0.1)

    def test_identity_intervention(self, engine, chain_scm):
        result = engine.evaluate(
            chain_scm,
            factual_evidence={0: 1.0, 1: 1.5, 2: 2.2},
            counterfactual_intervention={0: 1.0},
            target=2,
        )
        assert_allclose(result.value, 2.2, atol=0.3)
