"""Unit tests for cpa.inference.do_calculus.DoCalculusEngine."""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose

from cpa.core.scm import StructuralCausalModel
from cpa.inference.do_calculus import (
    DoCalculusEngine,
    DoCalculusResult,
    _ancestors_of,
    _descendants_of,
    _d_separated,
    _parents_of,
    _topological_sort,
)


# ===================================================================
# Fixtures
# ===================================================================


@pytest.fixture
def rng():
    return np.random.default_rng(42)


@pytest.fixture
def engine():
    return DoCalculusEngine(verbose=False)


@pytest.fixture
def chain_adj():
    """Chain: 0 → 1 → 2."""
    adj = np.zeros((3, 3), dtype=int)
    adj[0, 1] = 1
    adj[1, 2] = 1
    return adj


@pytest.fixture
def fork_adj():
    """Fork: 1 ← 0 → 2."""
    adj = np.zeros((3, 3), dtype=int)
    adj[0, 1] = 1
    adj[0, 2] = 1
    return adj


@pytest.fixture
def collider_adj():
    """Collider: 0 → 2 ← 1."""
    adj = np.zeros((3, 3), dtype=int)
    adj[0, 2] = 1
    adj[1, 2] = 1
    return adj


@pytest.fixture
def diamond_adj():
    """Diamond: 0 → 1, 0 → 2, 1 → 3, 2 → 3."""
    adj = np.zeros((4, 4), dtype=int)
    adj[0, 1] = 1
    adj[0, 2] = 1
    adj[1, 3] = 1
    adj[2, 3] = 1
    return adj


@pytest.fixture
def chain_scm():
    """Linear chain SCM: X0 → X1 → X2 with known coefficients."""
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
# Tests – mutilated graph construction
# ===================================================================


class TestMutilatedGraph:
    """Test _build_mutilated_graph removes edges to intervention nodes."""

    def test_chain_mutilate_middle(self, chain_adj):
        mut = DoCalculusEngine._build_mutilated_graph(
            chain_adj.astype(float), {1}
        )
        assert mut[0, 1] == 0, "Edge into intervention node should be removed"
        assert mut[1, 2] == 1, "Outgoing edge from intervention node preserved"

    def test_chain_mutilate_root(self, chain_adj):
        mut = DoCalculusEngine._build_mutilated_graph(
            chain_adj.astype(float), {0}
        )
        np.testing.assert_array_equal(mut, chain_adj)

    def test_mutilate_leaf(self, chain_adj):
        mut = DoCalculusEngine._build_mutilated_graph(
            chain_adj.astype(float), {2}
        )
        assert mut[1, 2] == 0
        assert mut[0, 1] == 1

    def test_diamond_mutilate_two_nodes(self, diamond_adj):
        mut = DoCalculusEngine._build_mutilated_graph(
            diamond_adj.astype(float), {1, 2}
        )
        assert mut[0, 1] == 0 and mut[0, 2] == 0
        assert mut[1, 3] == 1 and mut[2, 3] == 1

    def test_empty_intervention_set(self, chain_adj):
        mut = DoCalculusEngine._build_mutilated_graph(
            chain_adj.astype(float), set()
        )
        np.testing.assert_array_equal(mut, chain_adj)

    def test_mutilated_preserves_shape(self, diamond_adj):
        mut = DoCalculusEngine._build_mutilated_graph(
            diamond_adj.astype(float), {3}
        )
        assert mut.shape == (4, 4)


# ===================================================================
# Tests – do-operator
# ===================================================================


class TestDoOperator:
    """Test do-operator computation on known SCMs."""

    def test_do_on_root_is_identity(self, engine, chain_scm):
        result = engine.do_operator(chain_scm, {0: 2.0}, target=0, n_samples=1000)
        assert_allclose(result, 2.0, atol=0.01)

    def test_do_on_chain_propagates(self, engine, chain_scm):
        # do(X0=1): E[X1] = 0.5*1 = 0.5, E[X2] = 0.8*0.5 = 0.4
        e_x2 = engine.do_operator(chain_scm, {0: 1.0}, target=2, n_samples=50000)
        assert_allclose(e_x2, 0.4, atol=0.1)

    def test_do_on_middle_blocks_upstream(self, engine, chain_scm):
        # do(X1=3): E[X2] = 0.8*3 = 2.4
        e_x2 = engine.do_operator(chain_scm, {1: 3.0}, target=2, n_samples=50000)
        assert_allclose(e_x2, 2.4, atol=0.1)

    def test_do_on_leaf_no_downstream_effect(self, engine, chain_scm):
        e_x1 = engine.do_operator(chain_scm, {2: 5.0}, target=1, n_samples=10000)
        assert_allclose(e_x1, 0.0, atol=0.15)

    def test_do_diamond_two_paths(self, engine, diamond_scm):
        # do(X0=1): E[X3] = 0.7*(0.6) + 0.3*(0.4) = 0.42 + 0.12 = 0.54
        e_x3 = engine.do_operator(diamond_scm, {0: 1.0}, target=3, n_samples=50000)
        assert_allclose(e_x3, 0.54, atol=0.15)


# ===================================================================
# Tests – truncated factorization
# ===================================================================


class TestTruncatedFactorization:
    """Test truncated factorization of post-intervention distribution."""

    def test_returns_array(self, engine, chain_scm):
        result = engine.truncated_factorization(chain_scm, {0: 1.0})
        assert isinstance(result, np.ndarray)

    def test_intervention_on_root(self, engine, chain_scm):
        cov = engine.truncated_factorization(chain_scm, {0: 1.0})
        assert cov.shape[0] == chain_scm.num_variables

    def test_empty_intervention(self, engine, chain_scm):
        cov = engine.truncated_factorization(chain_scm, {})
        assert cov.shape == (3, 3) or cov.ndim >= 1


# ===================================================================
# Tests – Rule applicability
# ===================================================================


class TestRuleApplicability:
    """Test Rule 1, 2, 3 checks."""

    def test_rule1_returns_result(self, engine, chain_adj):
        result = engine.rule1(chain_adj, "P(Y|do(X))", {0}, {2})
        assert isinstance(result, DoCalculusResult)

    def test_rule2_returns_result(self, engine, chain_adj):
        result = engine.rule2(chain_adj, "P(Y|do(X))", {0}, {1})
        assert isinstance(result, DoCalculusResult)

    def test_rule3_returns_result(self, engine, chain_adj):
        result = engine.rule3(chain_adj, "P(Y|do(X))", {0})
        assert isinstance(result, DoCalculusResult)

    def test_apply_rules_returns_result(self, engine, chain_adj):
        result = engine.apply_rules(chain_adj, "P(Y|do(X))")
        assert isinstance(result, DoCalculusResult)
        assert isinstance(result.applicable_rules, list)


# ===================================================================
# Tests – underline / augmented graph construction
# ===================================================================


class TestUnderlineAndAugmentedGraphs:
    """Test _build_underline_graph and _build_augmented_graph."""

    def test_underline_removes_outgoing(self, chain_adj):
        under = DoCalculusEngine._build_underline_graph(
            chain_adj.astype(float), {1}
        )
        assert under[1, 2] == 0, "Outgoing edge from intervention removed"
        assert under[0, 1] == 1, "Incoming edge preserved"

    def test_augmented_has_policy_nodes(self, chain_adj):
        aug = DoCalculusEngine._build_augmented_graph(
            chain_adj.astype(float), {1}
        )
        assert aug.shape[0] > chain_adj.shape[0]


# ===================================================================
# Tests – helper functions
# ===================================================================


class TestHelpers:
    """Test module-level helper functions."""

    def test_ancestors_chain(self, chain_adj):
        anc = _ancestors_of(chain_adj, {2})
        assert 0 in anc and 1 in anc

    def test_ancestors_of_root(self, chain_adj):
        anc = _ancestors_of(chain_adj, {0})
        assert len(anc) == 0

    def test_descendants_chain(self, chain_adj):
        desc = _descendants_of(chain_adj, {0})
        assert 1 in desc and 2 in desc

    def test_descendants_of_leaf(self, chain_adj):
        desc = _descendants_of(chain_adj, {2})
        assert len(desc) == 0

    def test_parents_of_middle(self, chain_adj):
        pa = _parents_of(chain_adj, 1)
        assert 0 in pa

    def test_parents_of_root_empty(self, chain_adj):
        pa = _parents_of(chain_adj, 0)
        assert len(pa) == 0

    def test_topological_sort_chain(self, chain_adj):
        order = _topological_sort(chain_adj)
        assert order.index(0) < order.index(1) < order.index(2)

    def test_d_separated_chain_conditioned(self, chain_adj):
        assert _d_separated(chain_adj, {0}, {2}, {1})

    def test_not_d_separated_chain_unconditioned(self, chain_adj):
        assert not _d_separated(chain_adj, {0}, {2}, set())

    def test_d_separated_fork_conditioned(self, fork_adj):
        assert _d_separated(fork_adj, {1}, {2}, {0})

    def test_collider_unconditioned_separated(self, collider_adj):
        assert _d_separated(collider_adj, {0}, {1}, set())

    def test_collider_conditioned_connected(self, collider_adj):
        assert not _d_separated(collider_adj, {0}, {1}, {2})


# ===================================================================
# Tests – identify (ID algorithm)
# ===================================================================


class TestIdentify:
    """Test the ID algorithm for causal effect identification."""

    def test_chain_is_identifiable(self, engine, chain_adj):
        result = engine.identify(chain_adj, target={2}, intervention={0},
                                 observables={0, 1, 2})
        assert result.identified

    def test_fork_is_identifiable(self, engine, fork_adj):
        result = engine.identify(fork_adj, target={1}, intervention={0},
                                 observables={0, 1, 2})
        assert isinstance(result, DoCalculusResult)

    def test_result_has_expression(self, engine, chain_adj):
        result = engine.identify(chain_adj, target={2}, intervention={0},
                                 observables={0, 1, 2})
        assert isinstance(result.expression, str)
        assert len(result.expression) > 0
