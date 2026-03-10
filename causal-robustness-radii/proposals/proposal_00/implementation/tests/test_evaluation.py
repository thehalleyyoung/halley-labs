"""Tests for causalcert.evaluation – DGP, published DAGs, metrics, ablation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from causalcert.evaluation.dgp import (
    SyntheticDGP,
    DGPInstance,
    chain_dag,
    diamond_dag,
    fork_dag,
    collider_dag,
    random_dag_erdos_renyi,
    generate_linear_gaussian as dgp_generate_linear,
)
from causalcert.evaluation.published_dags import (
    list_published_dags,
    get_published_dag,
    get_all_published_dags,
    get_small_dags,
    PublishedDAG,
)
from causalcert.evaluation.metrics import (
    coverage_rate,
    exact_match_rate,
    within_one_rate,
    mean_absolute_error,
    fragility_auc,
    fragility_precision_at_k,
    runtime_summary,
    compute_all_metrics,
)
from causalcert.evaluation.ablation import AblationHarness, AblationResult
from causalcert.dag.validation import is_dag
from causalcert.types import (
    EditType,
    FragilityChannel,
    FragilityScore,
    RobustnessRadius,
    SolverStrategy,
    StructuralEdit,
)

# ═══════════════════════════════════════════════════════════════════════════
# DGP generation
# ═══════════════════════════════════════════════════════════════════════════


class TestDGPGeneration:
    def test_chain_dag(self) -> None:
        adj = chain_dag(5)
        assert is_dag(adj)
        assert adj.shape == (5, 5)
        assert adj.sum() == 4

    def test_diamond_dag(self) -> None:
        adj = diamond_dag()
        assert is_dag(adj)
        assert adj.shape == (4, 4)

    def test_fork_dag(self) -> None:
        adj = fork_dag()
        assert is_dag(adj)

    def test_collider_dag(self) -> None:
        adj = collider_dag()
        assert is_dag(adj)

    def test_erdos_renyi_dag(self) -> None:
        adj = random_dag_erdos_renyi(10, density=0.2, rng=np.random.RandomState(42))
        assert is_dag(adj)
        assert adj.shape == (10, 10)

    def test_generate_linear(self) -> None:
        adj = chain_dag(4)
        df, weights, ate = dgp_generate_linear(adj, n_samples=200, rng=np.random.RandomState(42))
        assert df.shape[0] == 200
        assert df.shape[1] == 4
        assert isinstance(ate, float)


# ═══════════════════════════════════════════════════════════════════════════
# SyntheticDGP class
# ═══════════════════════════════════════════════════════════════════════════


class TestSyntheticDGP:
    def test_generate_batch(self) -> None:
        dgp = SyntheticDGP(seed=42)
        instances = dgp.generate(n_instances=5, n_samples=100)
        assert len(instances) == 5
        for inst in instances:
            assert isinstance(inst, DGPInstance)
            assert is_dag(inst.adj)
            assert inst.data.shape[0] == 100

    def test_generate_chain(self) -> None:
        dgp = SyntheticDGP(seed=42)
        inst = dgp.generate_chain(length=5, n_samples=200, seed=42)
        assert inst.adj.shape == (5, 5)
        assert inst.data.shape[0] == 200

    def test_generate_diamond(self) -> None:
        dgp = SyntheticDGP(seed=42)
        inst = dgp.generate_diamond(n_samples=200, seed=42)
        assert is_dag(inst.adj)

    def test_generate_with_known_radius(self) -> None:
        dgp = SyntheticDGP(seed=42)
        inst = dgp.generate_with_known_radius(n_nodes=6, target_radius=2, n_samples=200, seed=42)
        assert isinstance(inst, DGPInstance)
        assert inst.true_radius == 2

    def test_varying_sample_sizes(self) -> None:
        dgp = SyntheticDGP(seed=42)
        instances = dgp.generate_varying_sample_sizes(
            n_nodes=5, target_radius=1, sample_sizes=(100, 500)
        )
        assert len(instances) == 2
        assert instances[0].data.shape[0] == 100
        assert instances[1].data.shape[0] == 500


# ═══════════════════════════════════════════════════════════════════════════
# Published DAGs
# ═══════════════════════════════════════════════════════════════════════════


class TestPublishedDAGs:
    def test_list_not_empty(self) -> None:
        names = list_published_dags()
        assert len(names) >= 1

    def test_get_published_dag(self) -> None:
        names = list_published_dags()
        dag = get_published_dag(names[0])
        assert isinstance(dag, PublishedDAG)
        assert dag.adj.shape[0] == dag.n_nodes

    def test_all_published_valid(self) -> None:
        for dag in get_all_published_dags():
            assert dag.n_nodes == dag.adj.shape[0]
            assert dag.n_edges == int(dag.adj.sum())

    def test_small_dags_filtered(self) -> None:
        small = get_small_dags(max_nodes=10)
        for dag in small:
            assert dag.n_nodes <= 10

    def test_dag_fields(self) -> None:
        names = list_published_dags()
        dag = get_published_dag(names[0])
        assert isinstance(dag.name, str)
        assert isinstance(dag.description, str)
        assert isinstance(dag.node_names, list)


# ═══════════════════════════════════════════════════════════════════════════
# Evaluation metrics
# ═══════════════════════════════════════════════════════════════════════════


class TestMetrics:
    @pytest.fixture
    def instances_and_results(self):
        dgp = SyntheticDGP(seed=42)
        instances = [
            dgp.generate_chain(length=4, n_samples=100, seed=i)
            for i in range(5)
        ]
        results = [
            RobustnessRadius(
                lower_bound=inst.true_radius,
                upper_bound=inst.true_radius,
                certified=True,
            )
            for inst in instances
        ]
        return instances, results

    def test_coverage_rate(self, instances_and_results) -> None:
        instances, results = instances_and_results
        rate = coverage_rate(instances, results)
        assert 0.0 <= rate <= 1.0

    def test_exact_match_rate(self, instances_and_results) -> None:
        instances, results = instances_and_results
        rate = exact_match_rate(instances, results)
        assert 0.0 <= rate <= 1.0

    def test_within_one_rate(self, instances_and_results) -> None:
        instances, results = instances_and_results
        rate = within_one_rate(instances, results)
        assert 0.0 <= rate <= 1.0

    def test_mean_absolute_error(self, instances_and_results) -> None:
        instances, results = instances_and_results
        mae = mean_absolute_error(instances, results)
        assert mae >= 0.0

    def test_runtime_summary(self) -> None:
        timings = [0.1, 0.2, 0.5, 1.0]
        summary = runtime_summary(timings)
        assert "mean" in summary
        assert summary["mean"] > 0

    def test_fragility_auc(self) -> None:
        scores = [
            FragilityScore(edge=(0, 1), total_score=0.9),
            FragilityScore(edge=(1, 2), total_score=0.3),
            FragilityScore(edge=(2, 3), total_score=0.1),
        ]
        true_fragile = {(0, 1)}
        auc = fragility_auc(scores, true_fragile)
        assert 0.0 <= auc <= 1.0

    def test_fragility_precision(self) -> None:
        scores = [
            FragilityScore(edge=(0, 1), total_score=0.9),
            FragilityScore(edge=(1, 2), total_score=0.3),
        ]
        prec = fragility_precision_at_k(scores, {(0, 1)}, k=1)
        assert 0.0 <= prec <= 1.0

    def test_compute_all_metrics(self, instances_and_results) -> None:
        instances, results = instances_and_results
        metrics = compute_all_metrics(instances, results, timings=[0.1] * 5)
        assert isinstance(metrics, dict)


# ═══════════════════════════════════════════════════════════════════════════
# Ablation harness
# ═══════════════════════════════════════════════════════════════════════════


class TestAblation:
    def test_ablation_result_fields(self) -> None:
        ar = AblationResult(condition="no_kernel", dgp_name="chain", coverage=0.9, runtime_s=1.0)
        assert ar.condition == "no_kernel"
        assert ar.coverage == 0.9

    def test_ablation_harness_creation(self) -> None:
        harness = AblationHarness()
        assert isinstance(harness, AblationHarness)

    def test_summary_table(self) -> None:
        results = [
            AblationResult(condition="baseline", dgp_name="chain", coverage=0.9, runtime_s=1.0),
            AblationResult(condition="no_kernel", dgp_name="chain", coverage=0.8, runtime_s=0.5),
        ]
        harness = AblationHarness()
        df = harness.summary_table(results)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2

    def test_aggregate_summary(self) -> None:
        results = [
            AblationResult(condition="baseline", dgp_name="chain", coverage=0.9, runtime_s=1.0),
            AblationResult(condition="baseline", dgp_name="diamond", coverage=0.85, runtime_s=1.5),
            AblationResult(condition="no_kernel", dgp_name="chain", coverage=0.8, runtime_s=0.5),
        ]
        harness = AblationHarness()
        agg = harness.aggregate_summary(results)
        assert isinstance(agg, pd.DataFrame)


# ═══════════════════════════════════════════════════════════════════════════
# DGP instance fields
# ═══════════════════════════════════════════════════════════════════════════


class TestDGPInstanceFields:
    def test_instance_fields(self) -> None:
        dgp = SyntheticDGP(seed=42)
        inst = dgp.generate_chain(length=4, n_samples=100, seed=42)
        assert isinstance(inst.adj, np.ndarray)
        assert isinstance(inst.data, pd.DataFrame)
        assert isinstance(inst.treatment, int)
        assert isinstance(inst.outcome, int)
        assert isinstance(inst.true_radius, int)

    def test_instance_with_name(self) -> None:
        dgp = SyntheticDGP(seed=42)
        inst = dgp.generate_chain(length=4, n_samples=100, seed=42)
        assert isinstance(inst.name, str)


# ═══════════════════════════════════════════════════════════════════════════
# Nonlinear DGP
# ═══════════════════════════════════════════════════════════════════════════


class TestNonlinearDGP:
    def test_nonlinear_batch(self) -> None:
        dgp = SyntheticDGP(seed=42, n_nodes_range=(5, 8))
        instances = dgp.generate_nonlinear_batch(n_instances=3, n_samples=100)
        assert len(instances) == 3
        for inst in instances:
            assert is_dag(inst.adj)
            assert inst.data.shape[0] == 100

    def test_varying_effect_sizes(self) -> None:
        dgp = SyntheticDGP(seed=42)
        instances = dgp.generate_varying_effect_sizes(
            n_nodes=5, n_samples=200,
            effect_ranges=[(0.1, 0.5), (1.0, 2.0)],
        )
        assert len(instances) == 2


# ═══════════════════════════════════════════════════════════════════════════
# Metrics edge cases
# ═══════════════════════════════════════════════════════════════════════════


class TestMetricsEdgeCases:
    def test_single_instance_metrics(self) -> None:
        dgp = SyntheticDGP(seed=42)
        inst = dgp.generate_chain(length=3, n_samples=100, seed=42)
        result = RobustnessRadius(lower_bound=1, upper_bound=1, certified=True)
        rate = coverage_rate([inst], [result])
        assert 0.0 <= rate <= 1.0

    def test_perfect_predictions(self) -> None:
        dgp = SyntheticDGP(seed=42)
        instances = [dgp.generate_chain(length=3, n_samples=100, seed=i) for i in range(5)]
        results = [
            RobustnessRadius(
                lower_bound=inst.true_radius, upper_bound=inst.true_radius, certified=True,
            )
            for inst in instances
        ]
        assert exact_match_rate(instances, results) == 1.0
        assert mean_absolute_error(instances, results) == 0.0

    def test_empty_timings(self) -> None:
        summary = runtime_summary([])
        assert isinstance(summary, dict)

    def test_fragility_recall_at_k(self) -> None:
        from causalcert.evaluation.metrics import fragility_recall_at_k
        scores = [
            FragilityScore(edge=(0, 1), total_score=0.9),
            FragilityScore(edge=(1, 2), total_score=0.3),
            FragilityScore(edge=(2, 3), total_score=0.1),
        ]
        recall = fragility_recall_at_k(scores, {(0, 1), (2, 3)}, k=2)
        assert 0.0 <= recall <= 1.0
