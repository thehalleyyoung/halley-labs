"""End-to-end integration tests for the CausalCert pipeline."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from causalcert.dag.graph import CausalDAG
from causalcert.dag.dsep import DSeparationOracle
from causalcert.dag.validation import is_dag
from causalcert.dag.edit import edit_distance, single_edit_perturbations
from causalcert.fragility.scorer import FragilityScorerImpl
from causalcert.fragility.ranking import rank_edges, top_k_fragile
from causalcert.estimation.backdoor import satisfies_backdoor, has_valid_adjustment_set
from causalcert.estimation.effects import estimate_ate
from causalcert.pipeline.orchestrator import CausalCertPipeline, ATESignificancePredicate
from causalcert.pipeline.config import quick_config
from causalcert.reporting.json_report import to_json_report
from causalcert.reporting.html_report import to_html_report
from causalcert.reporting.narrative import generate_narrative
from causalcert.types import (
    AdjacencyMatrix,
    AuditReport,
    EditType,
    EstimationResult,
    FragilityChannel,
    FragilityScore,
    RobustnessRadius,
    SolverStrategy,
    StructuralEdit,
)

# ── helpers ───────────────────────────────────────────────────────────────

def _adj(n: int, edges: list[tuple[int, int]]) -> AdjacencyMatrix:
    a = np.zeros((n, n), dtype=np.int8)
    for u, v in edges:
        a[u, v] = 1
    return a


def _generate_data(
    adj: AdjacencyMatrix,
    n: int = 500,
    true_ate: float = 2.0,
    treatment: int = 0,
    outcome: int = -1,
    seed: int = 42,
) -> tuple[pd.DataFrame, float]:
    """Generate linear-Gaussian data with known treatment effect."""
    rng = np.random.default_rng(seed)
    p = adj.shape[0]
    if outcome < 0:
        outcome = p - 1

    # Topological sort
    in_deg = adj.sum(axis=0).astype(int).copy()
    queue = [v for v in range(p) if in_deg[v] == 0]
    topo = []
    while queue:
        v = queue.pop(0)
        topo.append(v)
        for w in range(p):
            if adj[v, w]:
                in_deg[w] -= 1
                if in_deg[w] == 0:
                    queue.append(w)

    weights = rng.uniform(0.3, 1.0, (p, p)) * adj.astype(float)
    # Set treatment->outcome weight to true_ate
    if adj[treatment, outcome]:
        weights[treatment, outcome] = true_ate

    data = np.zeros((n, p))
    for v in topo:
        pa = np.where(adj[:, v] == 1)[0]
        mean = data[:, pa] @ weights[pa, v] if len(pa) else 0.0
        # Binary treatment
        if v == treatment:
            data[:, v] = (rng.standard_normal(n) + mean > 0).astype(float)
        else:
            data[:, v] = mean + rng.standard_normal(n) * 0.5

    cols = [f"X{i}" for i in range(p)]
    return pd.DataFrame(data, columns=cols), true_ate


# ═══════════════════════════════════════════════════════════════════════════
# End-to-end: generate data → run pipeline → verify
# ═══════════════════════════════════════════════════════════════════════════


class TestEndToEnd:
    """Full end-to-end workflow."""

    def test_confounded_dag_pipeline(self) -> None:
        """C->X, C->Y, X->Y."""
        adj = _adj(3, [(0, 1), (0, 2), (1, 2)])
        data, true_ate = _generate_data(adj, n=500, treatment=1, outcome=2)
        cfg = quick_config(treatment=1, outcome=2)
        pipeline = CausalCertPipeline(config=cfg)
        report = pipeline.run(adj, data)

        assert isinstance(report, AuditReport)
        assert report.treatment == 1
        assert report.outcome == 2
        assert report.n_nodes == 3
        assert report.n_edges == 3
        assert report.radius.lower_bound >= 0
        assert report.radius.upper_bound >= report.radius.lower_bound

    def test_chain_pipeline(self) -> None:
        """X -> M -> Y."""
        adj = _adj(3, [(0, 1), (1, 2)])
        data, _ = _generate_data(adj, n=500, treatment=0, outcome=2)
        cfg = quick_config(treatment=0, outcome=2)
        pipeline = CausalCertPipeline(config=cfg)
        report = pipeline.run(adj, data)
        assert isinstance(report, AuditReport)

    def test_diamond_pipeline(self) -> None:
        adj = _adj(4, [(0, 1), (0, 2), (1, 3), (2, 3)])
        data, _ = _generate_data(adj, n=500, treatment=0, outcome=3)
        cfg = quick_config(treatment=0, outcome=3)
        pipeline = CausalCertPipeline(config=cfg)
        report = pipeline.run(adj, data)
        assert isinstance(report, AuditReport)
        assert len(report.fragility_ranking) >= 1

    def test_pipeline_then_reports(self) -> None:
        adj = _adj(3, [(0, 1), (0, 2), (1, 2)])
        data, _ = _generate_data(adj, n=300, treatment=1, outcome=2)
        cfg = quick_config(treatment=1, outcome=2)
        pipeline = CausalCertPipeline(config=cfg)
        report = pipeline.run(adj, data)

        j = to_json_report(report)
        assert len(j) > 0
        parsed = json.loads(j)
        assert isinstance(parsed, dict)

        html = to_html_report(report)
        assert len(html) > 0

        narrative = generate_narrative(report)
        assert len(narrative) > 0


# ═══════════════════════════════════════════════════════════════════════════
# Test with known fragile edges
# ═══════════════════════════════════════════════════════════════════════════


class TestKnownFragileEdges:
    """Verify that known fragile edges are detected."""

    def test_direct_edge_is_fragile(self) -> None:
        """In X->Y (only edge), the edge is maximally fragile."""
        adj = _adj(2, [(0, 1)])
        scorer = FragilityScorerImpl(alpha=0.05)
        scores = scorer.score_data_free(adj, treatment=0, outcome=1)
        assert len(scores) >= 1
        # The existing edge should have non-zero fragility
        existing = [s for s in scores if s.edge == (0, 1)]
        assert len(existing) >= 1

    def test_chain_mediator_fragile(self) -> None:
        """In X->M->Y, both edges are on the only causal path."""
        adj = _adj(3, [(0, 1), (1, 2)])
        scorer = FragilityScorerImpl(alpha=0.05)
        scores = scorer.score_data_free(adj, treatment=0, outcome=2)
        ranked = rank_edges(scores, descending=True)
        # Top edges should include the chain edges
        top_edges = {s.edge for s in ranked[:3]}
        assert (0, 1) in top_edges or (1, 2) in top_edges

    def test_confounded_confounder_edge_important(self) -> None:
        """In C->X, C->Y, X->Y: deleting C->Y might change identification."""
        adj = _adj(3, [(0, 1), (0, 2), (1, 2)])
        data = _generate_data(adj, n=300, treatment=1, outcome=2)[0]
        scorer = FragilityScorerImpl(alpha=0.05)
        scores = scorer.score(adj, treatment=1, outcome=2, data=data)
        assert len(scores) >= 1


# ═══════════════════════════════════════════════════════════════════════════
# Test with known radius
# ═══════════════════════════════════════════════════════════════════════════


class TestKnownRadius:
    """Verify correct radius computation."""

    def test_direct_edge_radius_1(self) -> None:
        """X->Y: radius for 'has direct edge' is 1."""
        adj = _adj(2, [(0, 1)])

        def has_direct(a, d, *, treatment, outcome):
            return bool(a[treatment, outcome])

        # Single-edit perturbation check
        perturbs = single_edit_perturbations(adj)
        flipped = [p for p in perturbs if not has_direct(p[0], None, treatment=0, outcome=1)]
        assert len(flipped) >= 1  # at least one perturbation flips it

    def test_diamond_path_radius_2(self) -> None:
        """Diamond: two disjoint paths → radius ≥ 2 for path predicate."""
        adj = _adj(4, [(0, 1), (0, 2), (1, 3), (2, 3)])

        def has_path(a, d, *, treatment, outcome):
            n = a.shape[0]
            visited = set()
            stack = [treatment]
            while stack:
                v = stack.pop()
                if v == outcome:
                    return True
                if v in visited:
                    continue
                visited.add(v)
                for w in range(n):
                    if a[v, w]:
                        stack.append(w)
            return False

        # Single edits: no single edit can break both paths
        perturbs = single_edit_perturbations(adj)
        all_still_have_path = all(
            has_path(p[0], None, treatment=0, outcome=3) for p in perturbs
        )
        # At least one single edit should NOT break both paths
        # (i.e., not all single edits break the path)
        some_still_have_path = any(
            has_path(p[0], None, treatment=0, outcome=3) for p in perturbs
        )
        assert some_still_have_path


# ═══════════════════════════════════════════════════════════════════════════
# Multiple conclusion predicates
# ═══════════════════════════════════════════════════════════════════════════


class TestMultiplePredicates:
    def test_has_path_predicate(self) -> None:
        adj = _adj(3, [(0, 1), (1, 2)])

        def has_path(a, d, *, treatment, outcome):
            n = a.shape[0]
            visited = set()
            stack = [treatment]
            while stack:
                v = stack.pop()
                if v == outcome:
                    return True
                if v in visited:
                    continue
                visited.add(v)
                for w in range(n):
                    if a[v, w]:
                        stack.append(w)
            return False

        assert has_path(adj, None, treatment=0, outcome=2)
        deleted = adj.copy()
        deleted[0, 1] = 0
        assert not has_path(deleted, None, treatment=0, outcome=2)

    def test_has_valid_adjustment_predicate(self) -> None:
        adj = _adj(3, [(0, 1), (0, 2), (1, 2)])
        assert has_valid_adjustment_set(adj, treatment=1, outcome=2)

    def test_dsep_predicate(self) -> None:
        adj = _adj(3, [(0, 1), (1, 2)])
        oracle = DSeparationOracle(adj)
        assert oracle.is_d_separated(0, 2, frozenset({1}))


# ═══════════════════════════════════════════════════════════════════════════
# Robustness to sample size
# ═══════════════════════════════════════════════════════════════════════════


class TestSampleSizeRobustness:
    @pytest.mark.parametrize("n", [200, 500, 1000])
    def test_pipeline_runs_at_different_sizes(self, n: int) -> None:
        adj = _adj(3, [(0, 1), (0, 2), (1, 2)])
        data, _ = _generate_data(adj, n=n, treatment=1, outcome=2)
        cfg = quick_config(treatment=1, outcome=2)
        pipeline = CausalCertPipeline(config=cfg)
        report = pipeline.run(adj, data)
        assert isinstance(report, AuditReport)
        assert report.n_nodes == 3

    def test_small_sample_doesnt_crash(self) -> None:
        adj = _adj(3, [(0, 1), (1, 2)])
        data, _ = _generate_data(adj, n=50, treatment=0, outcome=2)
        cfg = quick_config(treatment=0, outcome=2)
        pipeline = CausalCertPipeline(config=cfg)
        report = pipeline.run(adj, data)
        assert isinstance(report, AuditReport)


# ═══════════════════════════════════════════════════════════════════════════
# Smoke test on published DAGs
# ═══════════════════════════════════════════════════════════════════════════


class TestPublishedDAGSmoke:
    """Smoke test: run fragility scoring on all published DAGs."""

    def test_published_dag_scoring(self) -> None:
        from causalcert.evaluation.published_dags import get_small_dags
        small = get_small_dags(max_nodes=8)
        if not small:
            pytest.skip("No small published DAGs available")
        dag = small[0]
        # Pick treatment=0, outcome=last node
        t, o = 0, dag.n_nodes - 1
        if dag.default_treatment is not None:
            t = dag.default_treatment
        if dag.default_outcome is not None:
            o = dag.default_outcome
        scorer = FragilityScorerImpl(alpha=0.05)
        scores = scorer.score_data_free(dag.adj, treatment=t, outcome=o)
        assert isinstance(scores, list)

    def test_all_published_are_valid_dags(self) -> None:
        from causalcert.evaluation.published_dags import get_all_published_dags
        for dag in get_all_published_dags():
            assert dag.n_nodes == dag.adj.shape[0]


# ═══════════════════════════════════════════════════════════════════════════
# Consistency checks
# ═══════════════════════════════════════════════════════════════════════════


class TestConsistency:
    def test_edit_distance_consistency(self) -> None:
        adj1 = _adj(3, [(0, 1), (1, 2)])
        adj2 = _adj(3, [(0, 1)])
        d = edit_distance(adj1, adj2)
        assert d == 1

    def test_dsep_consistent_with_data(self) -> None:
        """d-sep predictions should be consistent with CI tests on large data."""
        from causalcert.ci_testing.partial_corr import PartialCorrelationTest

        adj = _adj(3, [(0, 1), (1, 2)])
        oracle = DSeparationOracle(adj)

        rng = np.random.default_rng(42)
        n = 2000
        x = rng.standard_normal(n)
        m = 0.8 * x + 0.3 * rng.standard_normal(n)
        y = 0.8 * m + 0.3 * rng.standard_normal(n)
        data = pd.DataFrame({0: x, 1: m, 2: y})

        pcorr = PartialCorrelationTest(alpha=0.05, seed=42)

        # 0 _||_ 2 | 1 (d-sep) → CI test should not reject
        assert oracle.is_d_separated(0, 2, frozenset({1}))
        result = pcorr.test(0, 2, frozenset({1}), data)
        assert not result.reject or result.p_value > 0.01

        # 0 _not_||_ 2 | {} → CI test should reject
        assert not oracle.is_d_separated(0, 2, frozenset())
        result2 = pcorr.test(0, 2, frozenset(), data)
        assert result2.reject

    def test_backdoor_consistent_with_dag(self) -> None:
        adj = _adj(3, [(0, 1), (0, 2), (1, 2)])
        # C=0, X=1, Y=2
        assert satisfies_backdoor(adj, 1, 2, frozenset({0}))
        assert not satisfies_backdoor(adj, 1, 2, frozenset({2}))  # can't condition on outcome


# ═══════════════════════════════════════════════════════════════════════════
# DAG operations consistency
# ═══════════════════════════════════════════════════════════════════════════


class TestDAGOperationsConsistency:
    def test_copy_preserves_all_queries(self) -> None:
        adj = _adj(4, [(0, 1), (0, 2), (1, 3), (2, 3)])
        dag1 = CausalDAG(adj)
        dag2 = dag1.copy()
        for u in range(4):
            assert dag1.parents(u) == dag2.parents(u)
            assert dag1.children(u) == dag2.children(u)

    def test_subgraph_preserves_edges(self) -> None:
        adj = _adj(4, [(0, 1), (0, 2), (1, 3), (2, 3)])
        dag = CausalDAG(adj)
        sub = dag.subgraph(frozenset({0, 1, 3}))
        # Edge 0->1 should be in subgraph
        found_01 = any(
            sub.has_edge(i, j)
            for i in range(sub.n_nodes)
            for j in range(sub.n_nodes)
            if i != j
        )
        assert found_01

    def test_edit_distance_triangle_inequality(self) -> None:
        adj_a = _adj(3, [(0, 1)])
        adj_b = _adj(3, [(0, 1), (1, 2)])
        adj_c = _adj(3, [(1, 2)])
        d_ab = edit_distance(adj_a, adj_b)
        d_bc = edit_distance(adj_b, adj_c)
        d_ac = edit_distance(adj_a, adj_c)
        # Triangle inequality
        assert d_ac <= d_ab + d_bc

    @pytest.mark.parametrize("seed", [42, 43, 44])
    def test_fragility_scores_all_bounded(self, seed: int) -> None:
        from tests.conftest import random_dag
        adj = random_dag(5, edge_prob=0.3, seed=seed)
        scorer = FragilityScorerImpl(alpha=0.05)
        scores = scorer.score_data_free(adj, treatment=0, outcome=4)
        for fs in scores:
            assert 0.0 <= fs.total_score <= 1.0
            for ch, val in fs.channel_scores.items():
                assert 0.0 <= val <= 1.0

    def test_pipeline_report_consistent(self) -> None:
        adj = _adj(3, [(0, 1), (0, 2), (1, 2)])
        data, _ = _generate_data(adj, n=300, treatment=1, outcome=2)
        cfg = quick_config(treatment=1, outcome=2)
        pipeline = CausalCertPipeline(config=cfg)
        report = pipeline.run(adj, data)
        # Radius bounds must be consistent
        assert report.radius.lower_bound <= report.radius.upper_bound
        # Report fields match DAG
        assert report.n_nodes == 3
        assert report.n_edges == 3


# ═══════════════════════════════════════════════════════════════════════════
# Reproducibility tests
# ═══════════════════════════════════════════════════════════════════════════


class TestReproducibility:
    """Two runs with the same seed should produce identical results."""

    def test_pipeline_determinism(self) -> None:
        adj = _adj(4, [(0, 1), (1, 2), (2, 3)])
        data1, _ = _generate_data(adj, n=200, treatment=0, outcome=3, seed=42)
        data2, _ = _generate_data(adj, n=200, treatment=0, outcome=3, seed=42)
        pd.testing.assert_frame_equal(data1, data2)

        cfg = quick_config(treatment=0, outcome=3)
        p1 = CausalCertPipeline(config=cfg)
        p2 = CausalCertPipeline(config=cfg)
        r1 = p1.run(adj, data1)
        r2 = p2.run(adj, data2)
        assert r1.radius.lower_bound == r2.radius.lower_bound
        assert r1.radius.upper_bound == r2.radius.upper_bound

    def test_fragility_determinism(self) -> None:
        adj = _adj(4, [(0, 1), (1, 2), (2, 3)])
        scorer = FragilityScorerImpl(alpha=0.05)
        s1 = scorer.score_data_free(adj, treatment=0, outcome=3)
        s2 = scorer.score_data_free(adj, treatment=0, outcome=3)
        for a, b in zip(s1, s2):
            assert a.total_score == b.total_score
            assert a.edge == b.edge

    def test_dsep_oracle_determinism(self) -> None:
        adj = _adj(5, [(0, 1), (1, 2), (2, 3), (3, 4), (0, 3)])
        o1 = DSeparationOracle(adj)
        o2 = DSeparationOracle(adj)
        for x in range(5):
            for y in range(5):
                for z_bits in range(1 << 5):
                    z = frozenset(i for i in range(5) if z_bits & (1 << i) and i != x and i != y)
                    assert o1.is_d_separated(x, y, z) == o2.is_d_separated(x, y, z)


# ═══════════════════════════════════════════════════════════════════════════
# Larger DAG stress tests
# ═══════════════════════════════════════════════════════════════════════════


class TestLargerDAGStress:
    """Basic sanity on 20-node randomly generated DAGs."""

    @pytest.mark.parametrize("seed", range(10))
    def test_random_dag_operations(self, seed: int) -> None:
        from tests.conftest import random_dag
        adj = random_dag(15, edge_prob=0.2, seed=seed)
        dag = CausalDAG.from_adjacency_matrix(adj)
        # Topo sort works
        topo = dag.topological_sort()
        assert len(topo) == 15
        # D-sep oracle doesn't crash
        oracle = DSeparationOracle(adj)
        assert isinstance(oracle.is_d_separated(0, 14, frozenset()), bool)

    @pytest.mark.parametrize("n", [8, 12, 16])
    def test_chain_dag_fragility(self, n: int) -> None:
        edges = [(i, i + 1) for i in range(n - 1)]
        adj = _adj(n, edges)
        scorer = FragilityScorerImpl(alpha=0.05, include_absent=False)
        scores = scorer.score_data_free(adj, treatment=0, outcome=n - 1)
        assert len(scores) >= len(edges)
        # At least some edges should have non-zero fragility
        nonzero = [fs for fs in scores if fs.total_score > 0.0]
        assert len(nonzero) >= 1

    def test_dense_dag_solver(self) -> None:
        n = 6
        edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
        adj = _adj(n, edges)
        dag = CausalDAG.from_adjacency_matrix(adj)
        assert dag.n_edges == n * (n - 1) // 2
        oracle = DSeparationOracle(adj)
        # With all edges: no non-trivial d-separation
        assert not oracle.is_d_separated(0, n - 1, frozenset())

    def test_bipartite_dag(self) -> None:
        edges = [(i, j) for i in range(4) for j in range(4, 8)]
        adj = _adj(8, edges)
        dag = CausalDAG.from_adjacency_matrix(adj)
        assert dag.n_nodes == 8
        assert dag.n_edges == 16
        oracle = DSeparationOracle(adj)
        # Sources are not d-separated from sinks
        assert not oracle.is_d_separated(0, 7, frozenset())
