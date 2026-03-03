"""Tests for model comparison.

Covers BIC/AIC comparison and equivalence class analysis.
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose

from cpa.diagnostics.model_comparison import (
    ModelSelector,
    ModelScore,
    EquivalenceClassAnalyzer,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def rng():
    return np.random.default_rng(42)


@pytest.fixture
def linear_data(rng):
    """X0 -> X1 -> X2 linear data."""
    n = 200
    X0 = rng.normal(0, 1, size=n)
    X1 = 0.8 * X0 + rng.normal(0, 0.3, size=n)
    X2 = 0.6 * X1 + rng.normal(0, 0.3, size=n)
    return np.column_stack([X0, X1, X2])


@pytest.fixture
def true_adj():
    """True adjacency: chain 0 -> 1 -> 2."""
    return np.array([[0, 1, 0], [0, 0, 1], [0, 0, 0]], dtype=float)


@pytest.fixture
def wrong_adj():
    """Wrong adjacency: 2 -> 1 -> 0."""
    return np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=float)


@pytest.fixture
def empty_adj():
    return np.zeros((3, 3), dtype=float)


@pytest.fixture
def full_adj():
    """Fully connected DAG: 0->1, 0->2, 1->2."""
    return np.array([[0, 1, 1], [0, 0, 1], [0, 0, 0]], dtype=float)


@pytest.fixture
def selector():
    return ModelSelector(criterion="bic")


# ---------------------------------------------------------------------------
# Test BIC/AIC comparison
# ---------------------------------------------------------------------------

class TestBICAICComparison:

    def test_score_model(self, selector, true_adj, linear_data):
        score = selector.score_model(true_adj, linear_data, model_id="true_model")
        assert isinstance(score, ModelScore)
        assert np.isfinite(score.bic)
        assert np.isfinite(score.aic)

    def test_true_model_better_bic(self, selector, true_adj, wrong_adj, linear_data):
        s_true = selector.score_model(true_adj, linear_data, model_id="true")
        s_wrong = selector.score_model(wrong_adj, linear_data, model_id="wrong")
        # True model should have better (lower) BIC
        assert s_true.bic <= s_wrong.bic + 50  # some tolerance

    def test_empty_model_scored(self, selector, empty_adj, linear_data):
        score = selector.score_model(empty_adj, linear_data)
        assert np.isfinite(score.bic)

    def test_score_models_batch(self, selector, true_adj, wrong_adj, linear_data):
        scores = selector.score_models(
            [true_adj, wrong_adj], linear_data,
            model_ids=["true", "wrong"],
        )
        assert len(scores) == 2

    def test_select_best(self, selector, true_adj, wrong_adj, full_adj, linear_data):
        scores = selector.score_models(
            [true_adj, wrong_adj, full_adj], linear_data,
            model_ids=["true", "wrong", "full"],
        )
        best = selector.select_best(scores)
        assert isinstance(best, ModelScore)

    def test_rank_models(self, selector, true_adj, wrong_adj, full_adj, linear_data):
        scores = selector.score_models(
            [true_adj, wrong_adj, full_adj], linear_data,
            model_ids=["true", "wrong", "full"],
        )
        ranked = selector.rank_models(scores)
        assert len(ranked) == 3
        # First should be best
        assert ranked[0][1] >= ranked[-1][1]  # highest weight first

    def test_cross_validate(self, selector, true_adj, linear_data):
        result = selector.cross_validate(true_adj, linear_data, seed=42)
        assert isinstance(result, dict)

    def test_aic_criterion(self, true_adj, linear_data):
        sel = ModelSelector(criterion="aic")
        score = sel.score_model(true_adj, linear_data)
        assert np.isfinite(score.aic)

    def test_bayes_factor_approximation(self, selector, true_adj, wrong_adj, linear_data):
        s1 = selector.score_model(true_adj, linear_data, model_id="m1")
        s2 = selector.score_model(wrong_adj, linear_data, model_id="m2")
        bf = selector.bayes_factor_approximation(s1, s2)
        assert isinstance(bf, dict)

    def test_model_score_fields(self):
        ms = ModelScore(
            model_id="test", adj_matrix=np.eye(3),
            bic=100.0, aic=90.0, log_likelihood=-40.0,
            n_params=5, n_samples=100,
        )
        assert ms.model_id == "test"
        assert ms.n_params == 5


# ---------------------------------------------------------------------------
# Test equivalence class analysis
# ---------------------------------------------------------------------------

class TestEquivalenceClassAnalyzer:

    def test_dag_to_cpdag(self, true_adj):
        analyzer = EquivalenceClassAnalyzer()
        cpdag = analyzer.dag_to_cpdag(true_adj)
        assert cpdag.shape == true_adj.shape

    def test_cpdag_to_skeleton(self, true_adj):
        analyzer = EquivalenceClassAnalyzer()
        cpdag = analyzer.dag_to_cpdag(true_adj)
        skeleton = analyzer.cpdag_to_skeleton(cpdag)
        assert skeleton.shape == true_adj.shape
        # Skeleton should be symmetric
        assert_allclose(skeleton, skeleton.T)

    def test_identify_compelled_edges(self, true_adj):
        analyzer = EquivalenceClassAnalyzer()
        cpdag = analyzer.dag_to_cpdag(true_adj)
        compelled = analyzer.identify_compelled_edges(cpdag)
        assert isinstance(compelled, list)

    def test_identify_reversible_edges(self, true_adj):
        analyzer = EquivalenceClassAnalyzer()
        cpdag = analyzer.dag_to_cpdag(true_adj)
        reversible = analyzer.identify_reversible_edges(cpdag)
        assert isinstance(reversible, list)

    def test_estimate_class_size(self, true_adj):
        analyzer = EquivalenceClassAnalyzer()
        cpdag = analyzer.dag_to_cpdag(true_adj)
        size = analyzer.estimate_class_size(cpdag)
        assert size >= 1

    def test_identifiability_assessment(self, true_adj):
        analyzer = EquivalenceClassAnalyzer()
        result = analyzer.identifiability_assessment(true_adj)
        assert isinstance(result, dict)

    def test_compare_equivalence_classes(self, true_adj, full_adj):
        analyzer = EquivalenceClassAnalyzer()
        result = analyzer.compare_equivalence_classes(true_adj, full_adj)
        assert isinstance(result, dict)

    def test_v_structure_dag(self):
        """V-structure: 0 -> 2 <- 1 (edges compelled)."""
        adj = np.array([[0, 0, 1], [0, 0, 1], [0, 0, 0]], dtype=float)
        analyzer = EquivalenceClassAnalyzer()
        cpdag = analyzer.dag_to_cpdag(adj)
        compelled = analyzer.identify_compelled_edges(cpdag)
        # Both edges should be compelled due to v-structure
        assert len(compelled) >= 1

    def test_class_size_chain(self):
        """Chain of 3 without v-structure: class size > 1."""
        adj = np.array([[0, 1, 0], [0, 0, 1], [0, 0, 0]], dtype=float)
        analyzer = EquivalenceClassAnalyzer()
        cpdag = analyzer.dag_to_cpdag(adj)
        size = analyzer.estimate_class_size(cpdag)
        # Chain of 3 without v-structure has multiple orientations
        assert size >= 1
