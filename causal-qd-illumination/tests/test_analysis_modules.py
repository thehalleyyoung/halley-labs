"""Tests for analysis modules: sensitivity, diagnostics, causal_inference, comparison,
supermartingale, ergodicity, and interventional BIC scoring."""

from __future__ import annotations

import numpy as np
import pytest

from causal_qd.analysis.sensitivity import (
    DataSensitivityAnalyzer,
    InfluenceResult,
    ScoreSensitivityAnalyzer,
    SensitivityResult,
)
from causal_qd.analysis.diagnostics import (
    ArchiveDiagnostics,
    ArchiveHealthReport,
    OperatorDiagnostics,
    ScoreDiagnostics,
)
from causal_qd.analysis.causal_inference import (
    ArchiveCausalInference,
    CausalQueryEngine,
    InterventionEstimator,
)
from causal_qd.analysis.comparison import (
    AlgorithmComparator,
    BenchmarkSuite,
    edge_precision_recall_f1,
    structural_hamming_distance,
)
from causal_qd.analysis.supermartingale import SupermartingaleTracker
from causal_qd.analysis.ergodicity import ErgodicityChecker
from causal_qd.scores.interventional_bic import InterventionalBICScore
from causal_qd.scores.bic import BICScore


# ===================================================================
# Helpers
# ===================================================================

def _chain_adj(n: int = 5) -> np.ndarray:
    """0→1→2→...→(n-1)."""
    adj = np.zeros((n, n), dtype=np.int8)
    for i in range(n - 1):
        adj[i, i + 1] = 1
    return adj


def _fork_adj(n: int = 5) -> np.ndarray:
    """0→1, 0→2, ..., 0→(n-1)."""
    adj = np.zeros((n, n), dtype=np.int8)
    for i in range(1, n):
        adj[0, i] = 1
    return adj


def _linear_gaussian_data(adj: np.ndarray, n_samples: int = 100,
                           seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = adj.shape[0]
    data = np.zeros((n_samples, n))
    # topological order for upper-triangular adj
    for j in range(n):
        parents = np.where(adj[:, j])[0]
        noise = rng.standard_normal(n_samples) * 0.5
        if len(parents) > 0:
            weights = rng.uniform(0.5, 1.5, size=len(parents))
            data[:, j] = data[:, parents] @ weights + noise
        else:
            data[:, j] = noise
    return data


# ===================================================================
# TestSensitivityAnalysis
# ===================================================================


class TestSensitivityAnalysis:
    """Tests for causal_qd.analysis.sensitivity."""

    def test_leave_one_out_returns_results(self):
        scorer = BICScore()
        adj = _chain_adj(5)
        data = _linear_gaussian_data(adj, n_samples=50)
        # Create a second DAG (fork) for ranking comparison
        adj2 = _fork_adj(5)
        dags = [adj, adj2]

        analyzer = DataSensitivityAnalyzer(scorer)
        results = analyzer.leave_one_out_influence(dags, data)

        assert len(results) == 50
        assert all(isinstance(r, InfluenceResult) for r in results)
        assert all(r.influence_on_score >= 0.0 for r in results)
        assert all(0.0 <= r.influence_on_ranking <= 1.0 for r in results)

    def test_jackknife_stability(self):
        scorer = BICScore()
        adj = _chain_adj(5)
        data = _linear_gaussian_data(adj, n_samples=50)
        dags = [adj, _fork_adj(5)]

        analyzer = DataSensitivityAnalyzer(scorer)
        result = analyzer.jackknife_stability(dags, data)

        assert isinstance(result, SensitivityResult)
        assert result.metric_name == "jackknife_qd_score"
        assert result.baseline_value != 0.0
        assert len(result.perturbed_values) == 50
        assert result.std_change >= 0.0
        assert "jackknife_se" in result.details

    def test_score_perturbation(self):
        scorer = BICScore()
        adj = _chain_adj(5)
        data = _linear_gaussian_data(adj, n_samples=50)
        dags = [adj, _fork_adj(5)]

        analyzer = ScoreSensitivityAnalyzer({"bic": scorer})
        results = analyzer.score_perturbation(dags, data, noise_std=0.1,
                                              n_perturbations=5)

        assert "bic" in results
        sr = results["bic"]
        assert isinstance(sr, SensitivityResult)
        assert sr.metric_name == "score_perturbation_bic"
        assert len(sr.perturbed_values) == 5
        assert sr.mean_change >= 0.0

    def test_compare_rankings(self):
        scorer1 = BICScore()
        scorer2 = BICScore(penalty_multiplier=2.0)
        adj = _chain_adj(5)
        data = _linear_gaussian_data(adj, n_samples=80)
        dags = [adj, _fork_adj(5)]

        analyzer = ScoreSensitivityAnalyzer({"bic1": scorer1, "bic2": scorer2})
        result = analyzer.compare_rankings(dags, data)

        assert "bic1" in result and "bic2" in result
        assert result["bic1"]["bic1"] == 0.0
        assert result["bic2"]["bic2"] == 0.0
        # Cross distance should be in [0, 1]
        assert 0.0 <= result["bic1"]["bic2"] <= 1.0


# ===================================================================
# TestDiagnostics
# ===================================================================


class TestDiagnostics:
    """Tests for causal_qd.analysis.diagnostics."""

    def test_archive_diagnostics_health_check(self, sample_archive):
        diag = ArchiveDiagnostics(stagnation_window=5)
        for i in range(10):
            diag.record_iteration(qd_score=float(i * 10), n_improvements=max(0, 3 - i))

        qualities = [float(i) for i in range(1, 11)]
        report = diag.health_check(qualities, n_cells=25)

        assert isinstance(report, ArchiveHealthReport)
        assert report.n_elites == 10
        assert report.coverage == 10 / 25
        assert report.qd_score == sum(range(1, 11))
        assert report.best_quality == 10.0
        assert report.worst_quality == 1.0
        assert report.std_quality >= 0.0

    def test_operator_diagnostics(self):
        diag = OperatorDiagnostics()
        # Record successes and failures for two operators
        for _ in range(8):
            diag.record("edge_add", True, quality_gain=1.5)
        for _ in range(2):
            diag.record("edge_add", False)
        for _ in range(3):
            diag.record("edge_del", True, quality_gain=0.5)
        for _ in range(7):
            diag.record("edge_del", False)

        report = diag.report()
        assert "edge_add" in report and "edge_del" in report
        assert report["edge_add"].success_rate == pytest.approx(0.8)
        assert report["edge_del"].success_rate == pytest.approx(0.3)
        assert report["edge_add"].n_applications == 10
        assert report["edge_add"].mean_quality_gain == pytest.approx(1.5)

        assert diag.best_operator() == "edge_add"
        assert diag.worst_operator() == "edge_del"

    def test_score_diagnostics_analyze(self):
        scores = [1.0, 2.0, 3.0, 4.0, 5.0, 5.0, 5.0, 10.0, 20.0, 30.0]
        diag = ScoreDiagnostics()
        report = diag.analyze_distribution(scores)

        assert report.mean == pytest.approx(np.mean(scores))
        assert report.std == pytest.approx(np.std(scores))
        assert report.median == pytest.approx(np.median(scores))
        assert report.entropy >= 0.0


# ===================================================================
# TestCausalInference
# ===================================================================


class TestCausalInference:
    """Tests for causal_qd.analysis.causal_inference."""

    def test_edge_probabilities(self):
        # 5 small DAGs, all have edge 0→1; only some have 1→2
        dags = []
        for i in range(5):
            adj = np.zeros((3, 3), dtype=np.int8)
            adj[0, 1] = 1
            if i < 3:
                adj[1, 2] = 1
            dags.append(adj)

        ci = ArchiveCausalInference(weight_by_quality=False)
        probs = ci.edge_probabilities(dags)

        assert probs.shape == (3, 3)
        assert probs[0, 1] == pytest.approx(1.0)
        assert probs[1, 2] == pytest.approx(3.0 / 5.0)
        assert 0.0 <= probs.min() and probs.max() <= 1.0

    def test_consensus_structure(self):
        # All 5 DAGs have 0→1, 3/5 have 1→2
        dags = []
        for i in range(5):
            adj = np.zeros((3, 3), dtype=np.int8)
            adj[0, 1] = 1
            if i < 3:
                adj[1, 2] = 1
            dags.append(adj)

        ci = ArchiveCausalInference(weight_by_quality=False)
        consensus = ci.consensus_structure(dags, threshold=0.5)

        assert consensus.shape == (3, 3)
        assert consensus[0, 1] == 1  # present in all
        assert consensus[1, 2] == 1  # present in 3/5 > 0.5

    def test_causal_query_does_cause(self):
        # Chain DAG: 0→1→2
        chain = np.zeros((3, 3), dtype=np.int8)
        chain[0, 1] = 1
        chain[1, 2] = 1
        dags = [chain] * 5

        engine = CausalQueryEngine(confidence_threshold=0.5)
        result = engine.does_cause(dags, source=0, target=2)

        assert result.answer is True
        assert result.confidence >= 0.5
        assert result.supporting_dags == 5
        assert result.total_dags == 5


# ===================================================================
# TestComparison
# ===================================================================


class TestComparison:
    """Tests for causal_qd.analysis.comparison."""

    def test_shd_known_pair(self):
        chain = _chain_adj(5)
        # Modify: remove edge 3→4
        modified = chain.copy()
        modified[3, 4] = 0
        assert structural_hamming_distance(modified, chain) == 1

    def test_precision_recall_f1(self):
        true_dag = _chain_adj(5)
        pred = true_dag.copy()
        # Remove one true edge -> FN, add one false edge -> FP
        pred[3, 4] = 0
        pred[0, 3] = 1

        p, r, f = edge_precision_recall_f1(pred, true_dag)
        # true positives = 3, FP = 1, FN = 1
        assert p == pytest.approx(3.0 / 4.0)
        assert r == pytest.approx(3.0 / 4.0)
        assert f == pytest.approx(3.0 / 4.0)

    def test_benchmark_suite_asia(self):
        adj, n = BenchmarkSuite.asia_network()
        assert n == 8
        assert adj.shape == (8, 8)
        assert adj.sum() == 8  # 8 edges in Asia network
        # Check acyclicity via topological sort
        in_deg = adj.sum(axis=0).copy()
        queue = [i for i in range(n) if in_deg[i] == 0]
        order = []
        while queue:
            v = queue.pop(0)
            order.append(v)
            for c in range(n):
                if adj[v, c]:
                    in_deg[c] -= 1
                    if in_deg[c] == 0:
                        queue.append(c)
        assert len(order) == n  # all nodes reachable => DAG


# ===================================================================
# TestSupermartingaleTracker
# ===================================================================


class TestSupermartingaleTracker:
    """Tests for causal_qd.analysis.supermartingale."""

    def test_tracker_records_and_converges(self):
        tracker = SupermartingaleTracker(epsilon=0.01)
        # Simulate improving qualities that converge
        for t in range(20):
            tracker.record(t, {"A": 10.0 + t, "B": 5.0 + t * 0.5})

        diag = tracker.convergence_diagnostic()
        assert diag["converged_fraction"] >= 0.0
        # Since quality only improves and we always pass the running best,
        # residuals should be 0
        assert diag["mean_residual"] == pytest.approx(0.0)
        assert tracker.all_converged()

    def test_empty_tracker(self):
        tracker = SupermartingaleTracker(epsilon=1e-3)
        assert tracker.all_converged() is False
        diag = tracker.convergence_diagnostic()
        assert diag["converged_fraction"] == 0.0

    def test_residuals_nondecreasing(self):
        tracker = SupermartingaleTracker(epsilon=0.01)
        # Record improving then stable qualities
        qualities_sequence = [1.0, 2.0, 3.0, 4.0, 5.0, 5.0, 5.0]
        for t, q in enumerate(qualities_sequence):
            tracker.record(t, {"cell_0": q})

        diag = tracker.convergence_diagnostic()
        residuals = diag["per_cell"]["cell_0"]
        # M_t should be non-increasing
        for i in range(1, len(residuals)):
            assert residuals[i] <= residuals[i - 1] + 1e-12


# ===================================================================
# TestErgodicityChecker
# ===================================================================


class TestErgodicityChecker:
    """Tests for causal_qd.analysis.ergodicity."""

    def test_coverage_increases(self):
        checker = ErgodicityChecker(total_cells=10)
        checker.record_occupied_cells(0, {0, 1})
        checker.record_occupied_cells(1, {0, 1, 2, 3})
        checker.record_occupied_cells(2, {0, 1, 2, 3, 4, 5})

        iters, cov = checker.coverage_curve()
        assert len(iters) == 3
        # Coverage should be non-decreasing
        for i in range(1, len(cov)):
            assert cov[i] >= cov[i - 1]
        assert cov[-1] == pytest.approx(6.0 / 10.0)

    def test_mixing_time_estimation(self):
        checker = ErgodicityChecker(total_cells=100)
        # Simulate gradual exploration that saturates
        rng = np.random.default_rng(7)
        occupied = set()
        for t in range(50):
            new_cells = set(rng.integers(0, 100, size=5).tolist())
            occupied = occupied | new_cells
            checker.record_occupied_cells(t, occupied)

        tau = checker.estimate_mixing_time()
        # Should return a positive mixing time
        assert tau is not None
        assert tau > 0.0

    def test_empty_checker(self):
        checker = ErgodicityChecker(total_cells=10)
        assert checker.is_ergodic() is False


# ===================================================================
# TestInterventionalBIC
# ===================================================================


class TestInterventionalBIC:
    """Tests for causal_qd.scores.interventional_bic."""

    def test_true_dag_scores_higher(self):
        # Generate data from chain SCM with an intervention on node 2
        true_dag = _chain_adj(5)
        rng = np.random.default_rng(42)

        # Observational data
        obs_data = _linear_gaussian_data(true_dag, n_samples=100, seed=10)
        # Interventional data: fix node 2 to constant
        int_data = obs_data.copy()
        int_data[:, 2] = 0.0  # do(X2 = 0)

        regimes = [
            (obs_data, set()),       # observational
            (int_data, {2}),         # intervention on node 2
        ]
        scorer = InterventionalBICScore(regimes)

        true_score = scorer.score(true_dag)
        # Wrong DAG: reversed chain
        wrong_dag = np.zeros((5, 5), dtype=np.int8)
        for i in range(1, 5):
            wrong_dag[i, i - 1] = 1
        wrong_score = scorer.score(wrong_dag)

        assert true_score > wrong_score

    def test_fully_intervened_returns_zero(self):
        adj = _chain_adj(3)
        data = _linear_gaussian_data(adj, n_samples=50)
        # Node 1 is always an intervention target
        regimes = [(data, {0, 1, 2})]
        scorer = InterventionalBICScore(regimes)
        # All nodes intervened -> local_score should be 0 for each
        for node in range(3):
            parents = np.where(adj[:, node])[0]
            assert scorer.local_score(node, parents) == 0.0

    def test_observational_only(self):
        adj = _chain_adj(5)
        data = _linear_gaussian_data(adj, n_samples=100, seed=7)
        # No intervention targets
        regimes = [(data, set())]
        ibic = InterventionalBICScore(regimes)
        bic = BICScore()

        ibic_score = ibic.score(adj)
        bic_score = bic.score(adj, data)
        # Should be equivalent (both use full data, same BIC formula)
        assert ibic_score == pytest.approx(bic_score, rel=0.05)

    def test_empty_regimes_raises(self):
        with pytest.raises(ValueError, match="non-empty"):
            InterventionalBICScore(data_regimes=[])
