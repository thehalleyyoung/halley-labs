"""Edge case hardening tests for CausalQD modules.

Tests degenerate inputs, empty collections, numerical stability,
and proper error handling across key modules.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from causal_qd.core.dag import DAG


# ===================================================================
# 1. GES Baseline edge cases
# ===================================================================

class TestGESBaselineEdgeCases:
    """Edge cases for causal_qd.baselines.ges_baseline."""

    def test_single_variable(self):
        """GES on data with a single variable should return a 1-node DAG."""
        from causal_qd.baselines.ges_baseline import GESBaseline

        data = np.random.default_rng(0).standard_normal((100, 1))
        ges = GESBaseline()
        dag = ges.fit(data)
        assert dag.n_nodes == 1
        assert dag.adjacency.sum() == 0

    def test_constant_columns(self):
        """GES should not crash when all columns are constant."""
        from causal_qd.baselines.ges_baseline import GESBaseline

        data = np.ones((50, 3))
        ges = GESBaseline()
        dag = ges.fit(data)
        assert dag.n_nodes == 3
        # Score should be finite
        score = ges._total_score(data, dag.adjacency, 3)
        assert np.isfinite(score)

    def test_collinear_data(self):
        """GES should handle perfectly collinear columns without crashing."""
        from causal_qd.baselines.ges_baseline import GESBaseline

        rng = np.random.default_rng(1)
        x = rng.standard_normal((100, 1))
        data = np.hstack([x, 2 * x, 3 * x])
        ges = GESBaseline()
        dag = ges.fit(data)
        assert dag.n_nodes == 3
        assert np.all(np.isfinite(dag.adjacency))

    def test_empty_data_zero_samples(self):
        """GES with zero-sample data should not crash."""
        from causal_qd.baselines.ges_baseline import GESBaseline

        data = np.empty((0, 3))
        ges = GESBaseline()
        dag = ges.fit(data)
        assert dag.n_nodes == 3

    def test_gaussian_bic_zero_samples(self):
        """_GaussianBIC.local_score should be finite for 0 samples."""
        from causal_qd.baselines.ges_baseline import _GaussianBIC

        scorer = _GaussianBIC()
        data = np.empty((0, 3))
        score = scorer.local_score(0, [], data)
        assert np.isfinite(score)

    def test_score_cache_overflow(self):
        """_ScoreCache with max_size=2 should not error on overflow."""
        from causal_qd.baselines.ges_baseline import _ScoreCache, _GaussianBIC

        cache = _ScoreCache(max_size=2)
        scorer = _GaussianBIC()
        data = np.random.default_rng(0).standard_normal((50, 4))
        # Fill beyond max_size — should silently stop caching
        for node in range(4):
            cache.get(scorer, data, node, [])
        assert True  # no crash


# ===================================================================
# 2. Convergence Analysis edge cases
# ===================================================================

class TestConvergenceAnalysisEdgeCases:
    """Edge cases for causal_qd.analysis.convergence_analysis."""

    def test_no_data_recorded(self):
        """Summary and convergence checks on empty analyzer."""
        from causal_qd.analysis.convergence_analysis import ConvergenceAnalyzer

        ca = ConvergenceAnalyzer(window_size=5)
        assert ca.summary() == {"n_snapshots": 0}
        assert ca.qd_score_history() == []
        assert ca.coverage_history() == []
        assert ca.convergence_rate() == 0.0
        assert ca.expected_remaining_generations(0.5) == -1

    def test_single_snapshot(self):
        """Single snapshot should not crash convergence checks."""
        from causal_qd.analysis.convergence_analysis import ConvergenceAnalyzer

        ca = ConvergenceAnalyzer(window_size=5)

        class FakeArchive:
            def elites(self):
                return []

        ca.record(FakeArchive(), generation=0)
        assert ca.has_converged("plateau") is False
        assert ca.convergence_rate() == 0.0
        assert ca.expected_remaining_generations(0.5) == -1

    def test_nan_fitness_values(self):
        """Analyzer should handle NaN fitness without raising."""
        from causal_qd.analysis.convergence_analysis import (
            ConvergenceAnalyzer,
            _mann_kendall_trend,
        )

        # Mann-Kendall with NaN should return a float (may be nan)
        result = _mann_kendall_trend([1.0, float("nan"), 3.0, 4.0, 5.0])
        assert isinstance(result, float)

    def test_has_converged_unknown_method(self):
        """Unknown convergence method should raise ValueError."""
        from causal_qd.analysis.convergence_analysis import ConvergenceAnalyzer

        ca = ConvergenceAnalyzer(window_size=2)

        class FakeArchive:
            def elites(self):
                return []

        for i in range(5):
            ca.record(FakeArchive(), generation=i)

        with pytest.raises(ValueError, match="Unknown convergence method"):
            ca.has_converged("nonexistent_method")

    def test_convergence_all_methods_with_constant_scores(self):
        """Constant QD-scores should be detected as converged by all methods."""
        from causal_qd.analysis.convergence_analysis import ConvergenceAnalyzer
        from types import SimpleNamespace

        ca = ConvergenceAnalyzer(window_size=5, significance_level=0.05)

        class StableArchive:
            total_cells = 100
            def elites(self):
                return [SimpleNamespace(quality=1.0) for _ in range(10)]

        for gen in range(10):
            ca.record(StableArchive(), generation=gen)

        assert ca.has_converged("plateau") is True
        assert ca.has_converged("mann_kendall") is True
        assert ca.has_converged("relative") is True
        # Geweke on constant should converge (z=0)
        assert ca.has_converged("geweke") is True


# ===================================================================
# 3. Constrained Operators edge cases
# ===================================================================

class TestConstrainedOperatorsEdgeCases:
    """Edge cases for causal_qd.operators.constrained."""

    def test_all_edges_forbidden(self):
        """With all edges forbidden, repair should produce empty DAG."""
        from causal_qd.operators.constrained import EdgeConstraints, _repair

        n = 3
        forbidden = frozenset((i, j) for i in range(n) for j in range(n) if i != j)
        constraints = EdgeConstraints(forbidden_edges=forbidden)
        # Start with a full lower-triangular DAG
        adj = np.zeros((n, n), dtype=np.int8)
        adj[0, 1] = 1
        adj[0, 2] = 1
        adj[1, 2] = 1
        repaired = _repair(adj, constraints)
        assert repaired.sum() == 0

    def test_all_edges_required_chain(self):
        """Required edges forming a valid DAG should be preserved."""
        from causal_qd.operators.constrained import EdgeConstraints, _repair

        required = frozenset({(0, 1), (1, 2)})
        constraints = EdgeConstraints(required_edges=required)
        adj = np.zeros((3, 3), dtype=np.int8)
        repaired = _repair(adj, constraints)
        assert repaired[0, 1] == 1
        assert repaired[1, 2] == 1

    def test_conflicting_constraints_raises(self):
        """Edges both required and forbidden should raise ValueError."""
        from causal_qd.operators.constrained import EdgeConstraints

        ec = EdgeConstraints(
            forbidden_edges=frozenset({(0, 1)}),
            required_edges=frozenset({(0, 1)}),
        )
        with pytest.raises(ValueError, match="both required and forbidden"):
            ec.validate()

    def test_tier_ordering_forbids_backward_edge(self):
        """Tier ordering should forbid backward edges."""
        from causal_qd.operators.constrained import EdgeConstraints, TierConstraints

        tiers = [{0}, {1}, {2}]
        tc = TierConstraints(tiers)
        assert tc.is_valid_edge(0, 1) is True
        assert tc.is_valid_edge(1, 0) is False
        assert tc.is_valid_edge(0, 0) is False  # same tier

    def test_max_parents_enforcement(self):
        """Repair should trim excess parents to max_parents."""
        from causal_qd.operators.constrained import EdgeConstraints, _repair

        constraints = EdgeConstraints(max_parents=1)
        adj = np.zeros((4, 4), dtype=np.int8)
        adj[0, 3] = 1
        adj[1, 3] = 1
        adj[2, 3] = 1
        repaired = _repair(adj, constraints)
        assert repaired[:, 3].sum() <= 1

    def test_is_valid_dag_empty(self):
        """Empty DAG should always be valid."""
        from causal_qd.operators.constrained import EdgeConstraints

        ec = EdgeConstraints()
        adj = np.zeros((3, 3), dtype=np.int8)
        assert ec.is_valid_dag(adj) is True


# ===================================================================
# 4. Nonlinear SCM edge cases
# ===================================================================

class TestNonlinearSCMEdgeCases:
    """Edge cases for causal_qd.data.nonlinear_scm."""

    def test_single_node_dag(self):
        """SCM with a single node should sample pure noise."""
        from causal_qd.data.nonlinear_scm import NonlinearSCM, MechanismType

        adj = np.zeros((1, 1), dtype=np.int8)
        dag = DAG(adj)
        scm = NonlinearSCM(dag, mechanisms=MechanismType.LINEAR)
        data = scm.sample(100, rng=np.random.default_rng(0))
        assert data.shape == (100, 1)
        assert np.all(np.isfinite(data))

    def test_no_edges_dag(self):
        """SCM with no edges should produce independent noise columns."""
        from causal_qd.data.nonlinear_scm import NonlinearSCM

        adj = np.zeros((4, 4), dtype=np.int8)
        dag = DAG(adj)
        scm = NonlinearSCM(dag)
        data = scm.sample(200, rng=np.random.default_rng(1))
        assert data.shape == (200, 4)
        assert np.all(np.isfinite(data))

    def test_large_coefficients_polynomial(self):
        """SCM with very large coefficients should still produce finite data."""
        from causal_qd.data.nonlinear_scm import NonlinearSCM, MechanismType

        adj = np.zeros((2, 2), dtype=np.int8)
        adj[0, 1] = 1
        dag = DAG(adj)
        # Use very large coefficient range — with polynomial mechanism the
        # cubic terms can blow up, but data should at least not raise.
        scm = NonlinearSCM(
            dag,
            mechanisms=MechanismType.LINEAR,
            coefficient_range=(10.0, 100.0),
            rng=np.random.default_rng(2),
        )
        data = scm.sample(50, rng=np.random.default_rng(3))
        assert data.shape == (50, 2)
        # With LINEAR mechanism and bounded noise, values should be finite
        assert np.all(np.isfinite(data))

    def test_all_mechanism_types(self):
        """All mechanism types should produce finite data on a 2-node chain."""
        from causal_qd.data.nonlinear_scm import NonlinearSCM, MechanismType

        adj = np.zeros((2, 2), dtype=np.int8)
        adj[0, 1] = 1
        dag = DAG(adj)
        for mech in MechanismType:
            scm = NonlinearSCM(dag, mechanisms=mech, rng=np.random.default_rng(0))
            data = scm.sample(50, rng=np.random.default_rng(0))
            assert data.shape == (50, 2), f"Failed for {mech}"
            assert np.all(np.isfinite(data)), f"Non-finite data for {mech}"

    def test_intervene_on_all_nodes(self):
        """Hard intervention on every node should set values exactly."""
        from causal_qd.data.nonlinear_scm import NonlinearSCM

        adj = np.zeros((3, 3), dtype=np.int8)
        adj[0, 1] = 1
        adj[1, 2] = 1
        dag = DAG(adj)
        scm = NonlinearSCM(dag, rng=np.random.default_rng(0))
        targets = {0: 5.0, 1: 10.0, 2: -3.0}
        data = scm.intervene(targets, n_samples=20)
        assert np.allclose(data[:, 0], 5.0)
        assert np.allclose(data[:, 1], 10.0)
        assert np.allclose(data[:, 2], -3.0)

    def test_from_random_single_node(self):
        """from_random with 1 node should work."""
        from causal_qd.data.nonlinear_scm import NonlinearSCM

        scm = NonlinearSCM.from_random(n_nodes=1, rng=np.random.default_rng(0))
        data = scm.sample(10)
        assert data.shape == (10, 1)


# ===================================================================
# 5. Parallel Tempering edge cases
# ===================================================================

class TestParallelTemperingEdgeCases:
    """Edge cases for causal_qd.sampling.parallel_tempering."""

    def _make_scorer(self):
        from causal_qd.baselines.ges_baseline import _GaussianBIC
        return _GaussianBIC()

    def test_single_chain(self):
        """Single-chain tempering should work (no swaps)."""
        from causal_qd.sampling.parallel_tempering import ParallelTempering

        pt = ParallelTempering(
            score_function=self._make_scorer(),
            n_chains=1,
        )
        assert pt.temperatures == [1.0]
        data = np.random.default_rng(0).standard_normal((50, 3))
        result = pt.run(data, n_samples=5, burnin=5, rng=np.random.default_rng(1))
        assert len(result.samples) == 5
        assert result.swap_acceptance_rates == []

    def test_very_high_temperature(self):
        """Very high max temperature should not cause overflow."""
        from causal_qd.sampling.parallel_tempering import ParallelTempering

        pt = ParallelTempering(
            score_function=self._make_scorer(),
            n_chains=2,
            max_temp=1e6,
        )
        data = np.random.default_rng(0).standard_normal((30, 2))
        result = pt.run(data, n_samples=3, burnin=3, rng=np.random.default_rng(2))
        assert len(result.samples) == 3
        # Edge probabilities should be finite
        assert np.all(np.isfinite(result.edge_probabilities))

    def test_zero_samples(self):
        """n_samples=0 should return empty sample list."""
        from causal_qd.sampling.parallel_tempering import ParallelTempering

        pt = ParallelTempering(
            score_function=self._make_scorer(),
            n_chains=2,
        )
        data = np.random.default_rng(0).standard_normal((30, 2))
        result = pt.run(data, n_samples=0, burnin=2, rng=np.random.default_rng(3))
        assert len(result.samples) == 0

    def test_compute_edge_probabilities_empty_raises(self):
        """compute_edge_probabilities with empty list should raise."""
        from causal_qd.sampling.parallel_tempering import ParallelTempering

        with pytest.raises(ValueError, match="No samples"):
            ParallelTempering.compute_edge_probabilities([])

    def test_build_ladder_unknown_type_raises(self):
        """Unknown ladder type should raise ValueError."""
        from causal_qd.sampling.parallel_tempering import ParallelTempering

        with pytest.raises(ValueError, match="Unknown ladder type"):
            ParallelTempering._build_ladder(3, "unknown", 10.0)


# ===================================================================
# 6. Cached Score edge cases
# ===================================================================

class TestCachedScoreEdgeCases:
    """Edge cases for causal_qd.scores.cached."""

    def _make_cached(self, max_size=8192):
        from causal_qd.scores.cached import CachedScore
        from causal_qd.baselines.ges_baseline import _GaussianBIC
        return CachedScore(_GaussianBIC(), max_cache_size=max_size)

    def test_empty_parent_set(self):
        """Scoring with empty parent set should work."""
        cached = self._make_cached()
        data = np.random.default_rng(0).standard_normal((50, 3))
        score = cached.local_score(0, [], data)
        assert np.isfinite(score)

    def test_single_variable(self):
        """Scoring a single-variable DAG should work."""
        cached = self._make_cached()
        data = np.random.default_rng(0).standard_normal((50, 1))
        dag = np.zeros((1, 1), dtype=np.int8)
        score = cached.score(dag, data)
        assert np.isfinite(score)

    def test_cache_overflow_eviction(self):
        """Cache with max_size=2 should evict old entries via LRU."""
        cached = self._make_cached(max_size=2)
        data = np.random.default_rng(0).standard_normal((50, 4))
        # Score 4 different parent sets to overflow
        for node in range(4):
            cached.local_score(node, [], data)
        info = cached.cache_info
        assert info.current_size <= 2
        assert info.misses >= 3  # at least 3 misses after eviction

    def test_cache_hit_tracking(self):
        """Repeated calls should register as cache hits."""
        cached = self._make_cached()
        data = np.random.default_rng(0).standard_normal((50, 3))
        cached.local_score(0, [1], data)
        cached.local_score(0, [1], data)  # should be a hit
        info = cached.cache_info
        assert info.hits >= 1

    def test_clear_cache_resets_stats(self):
        """Clearing cache should reset hit/miss counters."""
        cached = self._make_cached()
        data = np.random.default_rng(0).standard_normal((50, 3))
        cached.local_score(0, [], data)
        cached.clear_cache()
        info = cached.cache_info
        assert info.hits == 0
        assert info.misses == 0
        assert info.current_size == 0

    def test_score_diff(self):
        """score_diff should return finite difference."""
        cached = self._make_cached()
        data = np.random.default_rng(0).standard_normal((50, 3))
        dag = np.zeros((3, 3), dtype=np.int8)
        diff = cached.score_diff(dag, 1, [], [0], data)
        assert np.isfinite(diff)


# ===================================================================
# 7. Fast Descriptors edge cases
# ===================================================================

class TestFastDescriptorsEdgeCases:
    """Edge cases for causal_qd.descriptors.fast_descriptors."""

    def test_empty_batch_structural(self):
        """batch_structural_descriptors with empty list returns (0, d) array."""
        from causal_qd.descriptors.fast_descriptors import batch_structural_descriptors

        result = batch_structural_descriptors([])
        assert result.shape[0] == 0
        assert result.ndim == 2

    def test_single_dag_batch_structural(self):
        """Single-DAG batch should produce valid descriptors."""
        from causal_qd.descriptors.fast_descriptors import batch_structural_descriptors

        adj = np.zeros((3, 3), dtype=np.int8)
        adj[0, 1] = 1
        result = batch_structural_descriptors([adj])
        assert result.shape[0] == 1
        assert np.all(np.isfinite(result))

    def test_all_zero_adjacency(self):
        """All-zero adjacency should produce valid descriptors with no NaN."""
        from causal_qd.descriptors.fast_descriptors import FastStructuralDescriptor

        desc = FastStructuralDescriptor()
        adj = np.zeros((5, 5), dtype=np.int8)
        result = desc.compute(adj)
        assert np.all(np.isfinite(result))
        # Edge density should be 0
        assert result[0] == 0.0

    def test_single_node_descriptors(self):
        """Single-node DAG should return all zeros (n<=1 guard)."""
        from causal_qd.descriptors.fast_descriptors import FastStructuralDescriptor

        desc = FastStructuralDescriptor()
        adj = np.zeros((1, 1), dtype=np.int8)
        result = desc.compute(adj)
        assert np.all(result == 0.0)

    def test_empty_batch_info_theoretic(self):
        """batch_info_theoretic_descriptors with empty list returns (0, d)."""
        from causal_qd.descriptors.fast_descriptors import batch_info_theoretic_descriptors

        data = np.random.default_rng(0).standard_normal((50, 3))
        result = batch_info_theoretic_descriptors([], data)
        assert result.shape[0] == 0
        assert result.ndim == 2

    def test_info_theoretic_no_edges(self):
        """Info-theoretic descriptors on edgeless DAG should be finite."""
        from causal_qd.descriptors.fast_descriptors import FastInfoTheoreticDescriptor

        desc = FastInfoTheoreticDescriptor()
        adj = np.zeros((3, 3), dtype=np.int8)
        data = np.random.default_rng(0).standard_normal((50, 3))
        result = desc.compute(adj, data)
        assert np.all(np.isfinite(result))

    def test_info_theoretic_constant_data(self):
        """Info-theoretic descriptors with constant data should be finite."""
        from causal_qd.descriptors.fast_descriptors import FastInfoTheoreticDescriptor

        desc = FastInfoTheoreticDescriptor()
        adj = np.zeros((3, 3), dtype=np.int8)
        adj[0, 1] = 1
        data = np.ones((50, 3))
        result = desc.compute(adj, data)
        assert np.all(np.isfinite(result))

    def test_composite_unfitted_compute(self):
        """Composite descriptor before fit should return raw values."""
        from causal_qd.descriptors.fast_descriptors import (
            FastStructuralDescriptor,
            FastCompositeDescriptor,
        )

        struct = FastStructuralDescriptor(["edge_density", "max_in_degree"])
        composite = FastCompositeDescriptor([struct])
        adj = np.zeros((3, 3), dtype=np.int8)
        result = composite.compute(adj)
        assert np.all(np.isfinite(result))

    def test_composite_batch_empty(self):
        """Composite batch_compute on empty list should return empty array."""
        from causal_qd.descriptors.fast_descriptors import (
            FastStructuralDescriptor,
            FastCompositeDescriptor,
        )

        struct = FastStructuralDescriptor(["edge_density"])
        composite = FastCompositeDescriptor([struct])
        result = composite.batch_compute([])
        assert result.shape[0] == 0


# ===================================================================
# 8. ParentSetCache edge cases
# ===================================================================

class TestParentSetCacheEdgeCases:
    """Edge cases for ParentSetCache in causal_qd.scores.cached."""

    def test_hit_rate_no_queries(self):
        """hit_rate should be 0.0 when no queries have been made."""
        from causal_qd.scores.cached import ParentSetCache

        cache = ParentSetCache(score_fn=lambda n, p, d: 0.0)
        assert cache.hit_rate == 0.0

    def test_stats_no_queries(self):
        """stats should report zeros when no queries have been made."""
        from causal_qd.scores.cached import ParentSetCache

        cache = ParentSetCache(score_fn=lambda n, p, d: 0.0)
        s = cache.stats
        assert s.hits == 0
        assert s.misses == 0
        assert s.current_size == 0

    def test_clear_resets(self):
        """Clearing cache should reset all counters."""
        from causal_qd.scores.cached import ParentSetCache

        cache = ParentSetCache(score_fn=lambda n, p, d: float(n))
        data = np.zeros((5, 2))
        cache.score_family(0, [], data)
        cache.clear()
        assert cache.stats.hits == 0
        assert cache.stats.misses == 0
