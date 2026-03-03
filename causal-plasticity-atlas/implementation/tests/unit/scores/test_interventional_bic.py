"""Unit tests for cpa.scores.interventional_bic."""

from __future__ import annotations

import math

import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal

from cpa.scores.interventional_bic import InterventionalBICScore


# ── helpers ─────────────────────────────────────────────────────────

def _invariant_contexts(n_per_ctx: int = 200, seed: int = 42):
    """Three contexts with the SAME mechanism for X1 | X0."""
    rng = np.random.default_rng(seed)
    datasets = []
    for i in range(3):
        x0 = rng.standard_normal(n_per_ctx)
        x1 = 0.8 * x0 + 0.3 * rng.standard_normal(n_per_ctx)
        x2 = 0.5 * x1 + 0.3 * rng.standard_normal(n_per_ctx)
        datasets.append(np.column_stack([x0, x1, x2]))
    targets = [set(), set(), set()]  # all observational
    return datasets, targets


def _heterogeneous_contexts(n_per_ctx: int = 200, seed: int = 42):
    """Three contexts where X1|X0 mechanism CHANGES."""
    rng = np.random.default_rng(seed)
    coeffs = [0.3, 0.8, -0.5]  # different mechanisms
    datasets = []
    for i, c in enumerate(coeffs):
        x0 = rng.standard_normal(n_per_ctx)
        x1 = c * x0 + 0.2 * rng.standard_normal(n_per_ctx)
        x2 = 0.5 * x1 + 0.3 * rng.standard_normal(n_per_ctx)
        datasets.append(np.column_stack([x0, x1, x2]))
    targets = [set(), set(), set()]
    return datasets, targets


def _intervention_contexts(n_per_ctx: int = 200, seed: int = 42):
    """Two observational + one interventional context on X1."""
    rng = np.random.default_rng(seed)
    datasets = []
    # Obs context 1
    x0 = rng.standard_normal(n_per_ctx)
    x1 = 0.8 * x0 + 0.3 * rng.standard_normal(n_per_ctx)
    x2 = 0.5 * x1 + 0.3 * rng.standard_normal(n_per_ctx)
    datasets.append(np.column_stack([x0, x1, x2]))
    # Obs context 2
    x0 = rng.standard_normal(n_per_ctx)
    x1 = 0.8 * x0 + 0.3 * rng.standard_normal(n_per_ctx)
    x2 = 0.5 * x1 + 0.3 * rng.standard_normal(n_per_ctx)
    datasets.append(np.column_stack([x0, x1, x2]))
    # Interventional: X1 is set randomly (not a function of X0)
    x0 = rng.standard_normal(n_per_ctx)
    x1 = rng.standard_normal(n_per_ctx) * 2.0  # intervention
    x2 = 0.5 * x1 + 0.3 * rng.standard_normal(n_per_ctx)
    datasets.append(np.column_stack([x0, x1, x2]))
    targets = [set(), set(), {1}]
    return datasets, targets


def _multi_intervention(n_per_ctx: int = 200, seed: int = 42):
    """Four contexts: 2 obs, 1 intervene on X1, 1 intervene on X2."""
    rng = np.random.default_rng(seed)
    datasets = []
    for _ in range(2):
        x0 = rng.standard_normal(n_per_ctx)
        x1 = 0.7 * x0 + 0.3 * rng.standard_normal(n_per_ctx)
        x2 = 0.6 * x1 + 0.3 * rng.standard_normal(n_per_ctx)
        datasets.append(np.column_stack([x0, x1, x2]))
    # Intervene on X1
    x0 = rng.standard_normal(n_per_ctx)
    x1 = rng.standard_normal(n_per_ctx)
    x2 = 0.6 * x1 + 0.3 * rng.standard_normal(n_per_ctx)
    datasets.append(np.column_stack([x0, x1, x2]))
    # Intervene on X2
    x0 = rng.standard_normal(n_per_ctx)
    x1 = 0.7 * x0 + 0.3 * rng.standard_normal(n_per_ctx)
    x2 = rng.standard_normal(n_per_ctx)
    datasets.append(np.column_stack([x0, x1, x2]))
    targets = [set(), set(), {1}, {2}]
    return datasets, targets


# ── Construction tests ─────────────────────────────────────────────

class TestInterventionalBICConstruction:
    def test_basic_construction(self):
        ds, tgt = _invariant_contexts()
        scorer = InterventionalBICScore(ds, tgt)
        assert scorer.n_contexts == 3
        assert scorer.n_variables == 3

    def test_mismatched_targets_raises(self):
        ds, _ = _invariant_contexts()
        with pytest.raises(ValueError, match="target"):
            InterventionalBICScore(ds, [set()])  # wrong count

    def test_mismatched_variables_raises(self):
        ds, tgt = _invariant_contexts()
        ds[1] = ds[1][:, :2]  # wrong number of columns
        with pytest.raises(ValueError, match="variable"):
            InterventionalBICScore(ds, tgt)

    def test_custom_labels(self):
        ds, tgt = _invariant_contexts()
        scorer = InterventionalBICScore(ds, tgt, context_labels=["a", "b", "c"])
        assert scorer.context_labels == ["a", "b", "c"]

    def test_repr(self):
        ds, tgt = _invariant_contexts()
        scorer = InterventionalBICScore(ds, tgt)
        r = repr(scorer)
        assert "InterventionalBICScore" in r
        assert "3" in r


# ── Invariant mechanism tests ──────────────────────────────────────

class TestInvariantMechanisms:
    def test_pooled_score_finite(self):
        ds, tgt = _invariant_contexts()
        scorer = InterventionalBICScore(ds, tgt)
        s = scorer.pooled_score(1, [0])
        assert np.isfinite(s)

    def test_pooled_beats_heterogeneous_for_invariant(self):
        """When mechanism is the same, pooled should win or tie."""
        ds, tgt = _invariant_contexts(n_per_ctx=500, seed=42)
        scorer = InterventionalBICScore(ds, tgt)
        pooled = scorer.pooled_score(1, [0])
        hetero = scorer._heterogeneous_score(1, [0])
        assert pooled >= hetero - 5.0  # allow small tolerance

    def test_model_selection_picks_pooled(self):
        ds, tgt = _invariant_contexts(n_per_ctx=500, seed=42)
        scorer = InterventionalBICScore(ds, tgt)
        model, score = scorer._model_selection(1, [0])
        assert model == "pooled"


# ── Heterogeneous mechanism tests ──────────────────────────────────

class TestHeterogeneousMechanisms:
    def test_heterogeneous_beats_pooled(self):
        """When mechanisms differ, heterogeneous should score higher."""
        ds, tgt = _heterogeneous_contexts(n_per_ctx=500, seed=42)
        scorer = InterventionalBICScore(ds, tgt)
        pooled = scorer.pooled_score(1, [0])
        hetero = scorer._heterogeneous_score(1, [0])
        assert hetero > pooled

    def test_model_selection_picks_heterogeneous(self):
        ds, tgt = _heterogeneous_contexts(n_per_ctx=500, seed=42)
        scorer = InterventionalBICScore(ds, tgt)
        model, _ = scorer._model_selection(1, [0])
        assert model == "heterogeneous"


# ── Intervention target tests ──────────────────────────────────────

class TestInterventionTargets:
    def test_obs_and_int_contexts(self):
        ds, tgt = _intervention_contexts()
        scorer = InterventionalBICScore(ds, tgt)
        obs = scorer._obs_contexts(1)
        int_ = scorer._int_contexts(1)
        assert obs == [0, 1]
        assert int_ == [2]

    def test_non_intervened_node_all_obs(self):
        ds, tgt = _intervention_contexts()
        scorer = InterventionalBICScore(ds, tgt)
        obs = scorer._obs_contexts(0)
        assert obs == [0, 1, 2]

    def test_context_specific_score_finite(self):
        ds, tgt = _intervention_contexts()
        scorer = InterventionalBICScore(ds, tgt)
        for c in range(3):
            s = scorer.context_specific_score(1, [0], c)
            assert np.isfinite(s)

    def test_local_score_single_context(self):
        ds, tgt = _intervention_contexts()
        scorer = InterventionalBICScore(ds, tgt)
        s = scorer.local_score(1, [0], context=0)
        assert np.isfinite(s)

    def test_local_score_combined(self):
        ds, tgt = _intervention_contexts()
        scorer = InterventionalBICScore(ds, tgt)
        s = scorer.local_score(1, [0])
        assert np.isfinite(s)


# ── Mechanism change detection ─────────────────────────────────────

class TestMechanismChangeDetection:
    def test_detect_returns_dict(self):
        ds, tgt = _intervention_contexts()
        scorer = InterventionalBICScore(ds, tgt)
        result = scorer.detect_intervention_targets(1, [0])
        assert "model" in result
        assert "pooled_score" in result
        assert "heterogeneous_score" in result
        assert "context_scores" in result
        assert "changed_contexts" in result
        assert "mechanism_params" in result

    def test_intervention_context_in_changed(self):
        ds, tgt = _intervention_contexts(n_per_ctx=300, seed=42)
        scorer = InterventionalBICScore(ds, tgt)
        result = scorer.detect_intervention_targets(1, [0])
        assert 2 in result["changed_contexts"]

    def test_heterogeneous_detects_changes(self):
        ds, tgt = _heterogeneous_contexts(n_per_ctx=500, seed=42)
        scorer = InterventionalBICScore(ds, tgt)
        result = scorer.detect_intervention_targets(1, [0])
        # Should detect that mechanisms differ across contexts
        assert len(result["changed_contexts"]) > 0 or result["model"] == "heterogeneous"

    def test_invariant_no_changes(self):
        ds, tgt = _invariant_contexts(n_per_ctx=500, seed=42)
        scorer = InterventionalBICScore(ds, tgt)
        result = scorer.detect_intervention_targets(1, [0])
        assert result["model"] == "pooled"

    def test_mechanism_params_have_coefficients(self):
        ds, tgt = _intervention_contexts()
        scorer = InterventionalBICScore(ds, tgt)
        result = scorer.detect_intervention_targets(1, [0])
        for mp in result["mechanism_params"]:
            assert "coefficients" in mp
            assert "variance" in mp
            assert "n_samples" in mp

    def test_multi_intervention(self):
        ds, tgt = _multi_intervention(n_per_ctx=300, seed=42)
        scorer = InterventionalBICScore(ds, tgt)
        # X1 is intervened in context 2
        result1 = scorer.detect_intervention_targets(1, [0])
        assert 2 in result1["changed_contexts"]
        # X2 is intervened in context 3
        result2 = scorer.detect_intervention_targets(2, [1])
        assert 3 in result2["changed_contexts"]


# ── score_dag tests ───────────────────────────────────────────────

class TestScoreDAG:
    def test_score_dag_finite(self):
        ds, tgt = _invariant_contexts()
        scorer = InterventionalBICScore(ds, tgt)
        adj = np.array([[0, 1, 0], [0, 0, 1], [0, 0, 0]], dtype=float)
        s = scorer.score_dag(adj)
        assert np.isfinite(s)

    def test_true_dag_beats_wrong(self):
        ds, tgt = _invariant_contexts(n_per_ctx=500, seed=42)
        scorer = InterventionalBICScore(ds, tgt)
        adj_true = np.array([[0, 1, 0], [0, 0, 1], [0, 0, 0]], dtype=float)
        adj_wrong = np.array([[0, 0, 0], [1, 0, 0], [1, 0, 0]], dtype=float)
        assert scorer.score_dag(adj_true) > scorer.score_dag(adj_wrong)

    def test_score_dag_range(self):
        ds, tgt = _invariant_contexts()
        scorer = InterventionalBICScore(ds, tgt)
        adj = np.array([[0, 1, 0], [0, 0, 1], [0, 0, 0]], dtype=float)
        results = scorer.score_dag_range(adj, [0.5, 1.0, 2.0])
        assert len(results) == 3
        assert all(np.isfinite(v) for v in results.values())


# ── Validation tests ──────────────────────────────────────────────

class TestValidation:
    def test_node_out_of_range(self):
        ds, tgt = _invariant_contexts()
        scorer = InterventionalBICScore(ds, tgt)
        with pytest.raises(ValueError, match="out of range"):
            scorer.local_score(10, [0])

    def test_parent_out_of_range(self):
        ds, tgt = _invariant_contexts()
        scorer = InterventionalBICScore(ds, tgt)
        with pytest.raises(ValueError, match="out of range"):
            scorer.local_score(1, [10])

    def test_self_parent_raises(self):
        ds, tgt = _invariant_contexts()
        scorer = InterventionalBICScore(ds, tgt)
        with pytest.raises(ValueError, match="own parent"):
            scorer.local_score(1, [1])


# ── Edge cases ────────────────────────────────────────────────────

class TestEdgeCases:
    def test_no_obs_contexts(self):
        """All contexts intervene on node 1."""
        rng = np.random.default_rng(42)
        ds = [rng.standard_normal((100, 3)) for _ in range(2)]
        tgt = [{1}, {1}]
        scorer = InterventionalBICScore(ds, tgt)
        s = scorer.pooled_score(1, [0])
        assert s == 0.0

    def test_single_context(self):
        rng = np.random.default_rng(42)
        x0 = rng.standard_normal(200)
        x1 = 0.8 * x0 + 0.3 * rng.standard_normal(200)
        x2 = 0.5 * x1 + 0.3 * rng.standard_normal(200)
        ds = [np.column_stack([x0, x1, x2])]
        tgt = [set()]
        scorer = InterventionalBICScore(ds, tgt)
        s = scorer.local_score(1, [0])
        assert np.isfinite(s)
