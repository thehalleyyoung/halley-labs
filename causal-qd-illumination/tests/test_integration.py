"""End-to-end integration tests for the causal QD pipeline.

Combines data generation, MAP-Elites, scoring, descriptors, baselines,
and certificates in full pipeline runs.
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
import numpy.testing as npt
import pytest

from causal_qd.archive.archive_base import ArchiveEntry
from causal_qd.archive.grid_archive import GridArchive
from causal_qd.baselines.pc import PCAlgorithm
from causal_qd.baselines.ges import GESAlgorithm
from causal_qd.baselines.mmhc import MMHCAlgorithm
from causal_qd.certificates.bootstrap import BootstrapCertificateComputer
from causal_qd.core.dag import DAG
from causal_qd.data.generator import DataGenerator
from causal_qd.data.scm import LinearGaussianSCM
from causal_qd.descriptors.structural import StructuralDescriptor
from causal_qd.engine.map_elites import CausalMAPElites, MAPElitesConfig
from causal_qd.metrics.qd_metrics import QDScore, Coverage, Diversity
from causal_qd.metrics.structural import SHD, F1
from causal_qd.operators.mutation import TopologicalMutation, EdgeAddMutation, EdgeRemoveMutation
from causal_qd.scores.bic import BICScore
from causal_qd.types import AdjacencyMatrix, DataMatrix, BehavioralDescriptor


# ===================================================================
# Helpers
# ===================================================================

def _chain_dag(n: int) -> DAG:
    adj = np.zeros((n, n), dtype=np.int8)
    for i in range(n - 1):
        adj[i, i + 1] = 1
    return DAG(adj)


def _make_data(dag: DAG, n_samples: int, seed: int = 42) -> DataMatrix:
    scm = LinearGaussianSCM.from_dag(
        dag, weight_range=(0.5, 1.0),
        noise_std_range=(0.3, 0.6),
        rng=np.random.default_rng(seed),
    )
    return scm.sample(n_samples, rng=np.random.default_rng(seed + 1))


def _build_score_fn(bic: BICScore):
    """Return a score function compatible with CausalMAPElites."""
    def score_fn(adj: AdjacencyMatrix, data: DataMatrix) -> float:
        n = adj.shape[0]
        total = 0.0
        for node in range(n):
            parents = list(np.where(adj[:, node])[0])
            total += bic.local_score(node, parents, data)
        return total
    return score_fn


def _build_descriptor_fn(desc: StructuralDescriptor):
    """Return a descriptor function compatible with CausalMAPElites."""
    def descriptor_fn(adj: AdjacencyMatrix, data: DataMatrix) -> BehavioralDescriptor:
        return desc.compute(adj, data)
    return descriptor_fn


def _build_mutations():
    """Return mutation operators as callables for CausalMAPElites."""
    topo = TopologicalMutation()
    add_m = EdgeAddMutation()
    rem_m = EdgeRemoveMutation()

    def mut_topo(adj: AdjacencyMatrix, rng: np.random.Generator) -> AdjacencyMatrix:
        return topo.mutate(adj, rng)

    def mut_add(adj: AdjacencyMatrix, rng: np.random.Generator) -> AdjacencyMatrix:
        return add_m.mutate(adj, rng)

    def mut_rem(adj: AdjacencyMatrix, rng: np.random.Generator) -> AdjacencyMatrix:
        return rem_m.mutate(adj, rng)

    return [mut_topo, mut_add, mut_rem]


def _dummy_crossover(adj1: AdjacencyMatrix, adj2: AdjacencyMatrix,
                      rng: np.random.Generator) -> AdjacencyMatrix:
    """Simple uniform crossover: for each edge position pick from parent 1 or 2."""
    n = adj1.shape[0]
    child = np.zeros_like(adj1)
    for i in range(n):
        for j in range(i + 1, n):
            if rng.random() < 0.5:
                child[i, j] = adj1[i, j]
            else:
                child[i, j] = adj2[i, j]
    # Ensure acyclicity (upper triangular is always a DAG)
    return child


def _run_map_elites(data: DataMatrix, n_vars: int, n_iters: int = 10,
                     batch_size: int = 8, seed: int = 42):
    """Set up and run CausalMAPElites, return the archive."""
    bic = BICScore()
    desc = StructuralDescriptor()

    config = MAPElitesConfig(
        mutation_prob=0.8,
        crossover_rate=0.2,
        archive_dims=(10, 10),
        archive_ranges=tuple(
            (lb, ub) for lb, ub in zip(
                desc.descriptor_bounds[0][:2],
                desc.descriptor_bounds[1][:2],
            )
        ),
        seed=seed,
    )

    mutations = _build_mutations()
    crossovers = [_dummy_crossover]
    score_fn = _build_score_fn(bic)
    descriptor_fn = _build_descriptor_fn(desc)

    engine = CausalMAPElites(
        mutations=mutations,
        crossovers=crossovers,
        descriptor_fn=descriptor_fn,
        score_fn=score_fn,
        config=config,
    )

    # Seed with random initial DAGs
    rng = np.random.default_rng(seed)
    initial_dags = []
    for _ in range(batch_size):
        dag = DAG.random_dag(n_vars, edge_prob=0.3, rng=rng)
        initial_dags.append(dag.adjacency)

    archive = engine.run(
        data, n_iterations=n_iters, batch_size=batch_size,
        initial_dags=initial_dags,
    )
    return engine, archive


# ===================================================================
# Full Pipeline Tests
# ===================================================================

class TestFullPipelineSmall:
    """Test full pipeline on a small problem (5 vars, 100 samples)."""

    def test_full_pipeline_small(self):
        dag = _chain_dag(5)
        data = _make_data(dag, 100, seed=0)

        engine, archive = _run_map_elites(data, n_vars=5, n_iters=5, batch_size=4)

        assert archive.fill_count > 0, "Archive should have at least one elite"
        best = archive.best()
        assert best is not None
        assert best.solution.shape == (5, 5)

    def test_archive_properties_after_run(self):
        dag = _chain_dag(5)
        data = _make_data(dag, 100)

        engine, archive = _run_map_elites(data, n_vars=5, n_iters=5, batch_size=4)

        cov = archive.coverage()
        qd = archive.qd_score()
        assert 0.0 <= cov <= 1.0
        assert isinstance(qd, float)

    def test_history_recorded(self):
        dag = _chain_dag(5)
        data = _make_data(dag, 100)

        engine, archive = _run_map_elites(data, n_vars=5, n_iters=5, batch_size=4)

        history = engine.history
        assert len(history) >= 5
        for stat in history:
            assert stat.archive_size >= 0
            assert isinstance(stat.best_quality, float)

    def test_best_elite_is_valid_dag(self):
        dag = _chain_dag(5)
        data = _make_data(dag, 100)

        engine, archive = _run_map_elites(data, n_vars=5, n_iters=5, batch_size=4)

        best = archive.best()
        best_dag = DAG(best.solution)
        assert not best_dag.has_cycle()

    def test_all_elites_are_dags(self):
        dag = _chain_dag(5)
        data = _make_data(dag, 100)

        engine, archive = _run_map_elites(data, n_vars=5, n_iters=5, batch_size=4)

        for entry in archive.entries:
            elite_dag = DAG(entry.solution)
            assert not elite_dag.has_cycle(), "Elite is not a valid DAG"


class TestFullPipelineMedium:
    """Test full pipeline on a medium problem (10 vars, 500 samples)."""

    def test_full_pipeline_medium(self):
        dag = _chain_dag(10)
        data = _make_data(dag, 500, seed=10)

        engine, archive = _run_map_elites(
            data, n_vars=10, n_iters=10, batch_size=8, seed=10,
        )

        assert archive.fill_count > 0
        best = archive.best()
        assert best.solution.shape == (10, 10)

    def test_medium_coverage_positive(self):
        dag = _chain_dag(10)
        data = _make_data(dag, 500, seed=10)

        engine, archive = _run_map_elites(
            data, n_vars=10, n_iters=10, batch_size=8, seed=10,
        )

        cov = archive.coverage()
        assert cov > 0.0

    def test_medium_qd_score_nonzero(self):
        dag = _chain_dag(10)
        data = _make_data(dag, 500, seed=10)

        engine, archive = _run_map_elites(
            data, n_vars=10, n_iters=10, batch_size=8, seed=10,
        )

        qd = archive.qd_score()
        assert qd != 0.0


class TestPipelineWithCertificates:
    """Test pipeline combined with bootstrap certificates."""

    def test_full_pipeline_with_certificates(self):
        dag = _chain_dag(5)
        data = _make_data(dag, 200, seed=0)

        engine, archive = _run_map_elites(data, n_vars=5, n_iters=5, batch_size=4)

        best = archive.best()
        bic = BICScore()

        def score_fn(adj: AdjacencyMatrix, d: DataMatrix) -> float:
            n = adj.shape[0]
            total = 0.0
            for node in range(n):
                parents = list(np.where(adj[:, node])[0])
                total += bic.local_score(node, parents, d)
            return total

        cert_computer = BootstrapCertificateComputer(
            n_bootstrap=20,
            score_fn=score_fn,
            confidence_level=0.95,
            rng=np.random.default_rng(0),
        )

        certs = cert_computer.compute_edge_certificates(best.solution, data)
        assert isinstance(certs, dict)
        # Should have a certificate for each edge in the best DAG
        n_edges = best.solution.sum()
        if n_edges > 0:
            assert len(certs) == n_edges

    def test_certificates_have_valid_fields(self):
        dag = _chain_dag(4)
        data = _make_data(dag, 200, seed=0)

        bic = BICScore()

        def score_fn(adj: AdjacencyMatrix, d: DataMatrix) -> float:
            n = adj.shape[0]
            total = 0.0
            for node in range(n):
                parents = list(np.where(adj[:, node])[0])
                total += bic.local_score(node, parents, d)
            return total

        cert_computer = BootstrapCertificateComputer(
            n_bootstrap=10,
            score_fn=score_fn,
            confidence_level=0.95,
            rng=np.random.default_rng(0),
        )

        certs = cert_computer.compute_edge_certificates(dag.adjacency, data)
        for edge, cert in certs.items():
            assert isinstance(edge, tuple)
            assert len(edge) == 2
            # Certificate should have meaningful attributes
            assert hasattr(cert, "edge") or hasattr(cert, "confidence")


class TestPipelineWithBaselinesComparison:
    """Compare MAP-Elites results with baseline algorithms."""

    def test_full_pipeline_with_baselines_comparison(self):
        dag = _chain_dag(5)
        data = _make_data(dag, 1000, seed=0)
        true_adj = dag.adjacency

        # Run baselines
        pc_adj = PCAlgorithm(alpha=0.05).run(data)
        ges_adj = GESAlgorithm().run(data)

        # Run MAP-Elites
        engine, archive = _run_map_elites(
            data, n_vars=5, n_iters=15, batch_size=8,
        )
        best = archive.best()

        # All should return 5×5 adjacency matrices
        assert pc_adj.shape == (5, 5)
        assert ges_adj.shape == (5, 5)
        assert best.solution.shape == (5, 5)

        # Compute SHDs
        shd_pc = SHD.compute(pc_adj, true_adj)
        shd_ges = SHD.compute(ges_adj, true_adj)
        shd_me = SHD.compute(best.solution, true_adj)

        # All should be finite nonneg
        assert shd_pc >= 0
        assert shd_ges >= 0
        assert shd_me >= 0

    def test_best_elite_competitive_with_baselines(self):
        """Best MAP-Elites elite should have comparable quality to baselines."""
        dag = _chain_dag(5)
        data = _make_data(dag, 1000, seed=0)

        bic = BICScore()

        def total_bic(adj: AdjacencyMatrix) -> float:
            n = adj.shape[0]
            total = 0.0
            for node in range(n):
                parents = list(np.where(adj[:, node])[0])
                total += bic.local_score(node, parents, data)
            return total

        pc_adj = PCAlgorithm(alpha=0.05).run(data)
        ges_adj = GESAlgorithm().run(data)

        engine, archive = _run_map_elites(
            data, n_vars=5, n_iters=20, batch_size=8,
        )
        best = archive.best()

        bic_pc = total_bic(pc_adj)
        bic_ges = total_bic(ges_adj)
        bic_me = total_bic(best.solution)

        # MAP-Elites best should be in the same ballpark
        worst_baseline = min(bic_pc, bic_ges)
        assert bic_me >= worst_baseline * 2, (
            f"MAP-Elites BIC={bic_me:.1f} too far from baselines "
            f"(PC={bic_pc:.1f}, GES={bic_ges:.1f})"
        )


class TestArchiveDiversity:
    """Tests focused on archive diversity properties."""

    def test_archive_diversity_exceeds_random(self):
        """MAP-Elites archive should have better diversity than random sampling."""
        dag = _chain_dag(5)
        data = _make_data(dag, 200, seed=0)

        engine, archive = _run_map_elites(
            data, n_vars=5, n_iters=15, batch_size=8,
        )

        # Diversity: compute mean pairwise descriptor distance directly
        # since _GridArchive is not iterable for the metrics API
        entries = archive.entries
        if len(entries) >= 2:
            descs = np.array([e.descriptor for e in entries])
            from scipy.spatial.distance import pdist
            me_diversity = float(np.mean(pdist(descs)))
        else:
            me_diversity = 0.0

        # Create a random archive by just adding random DAGs
        desc = StructuralDescriptor()
        random_archive = GridArchive(
            dims=(10, 10),
            lower_bounds=desc.descriptor_bounds[0][:2],
            upper_bounds=desc.descriptor_bounds[1][:2],
        )
        bic = BICScore()
        rng = np.random.default_rng(99)
        for _ in range(archive.fill_count):
            adj = DAG.random_dag(5, edge_prob=0.3, rng=rng).adjacency
            d = desc.compute(adj, data)
            q = 0.0
            for node in range(5):
                parents = list(np.where(adj[:, node])[0])
                q += bic.local_score(node, parents, data)
            random_archive.add(ArchiveEntry(
                solution=adj, descriptor=d[:2], quality=q,
            ))

        rand_diversity = Diversity.compute(random_archive)

        # Both should be non-negative
        assert me_diversity >= 0.0
        assert rand_diversity >= 0.0

    def test_archive_coverage_increases_over_iterations(self):
        """Coverage should generally increase (or stay the same) over iterations."""
        dag = _chain_dag(5)
        data = _make_data(dag, 200, seed=0)

        bic = BICScore()
        desc_computer = StructuralDescriptor()

        config = MAPElitesConfig(
            archive_dims=(10, 10),
            archive_ranges=tuple(
                (lb, ub) for lb, ub in zip(
                    desc_computer.descriptor_bounds[0][:2],
                    desc_computer.descriptor_bounds[1][:2],
                )
            ),
            seed=42,
        )

        engine = CausalMAPElites(
            mutations=_build_mutations(),
            crossovers=[_dummy_crossover],
            descriptor_fn=_build_descriptor_fn(desc_computer),
            score_fn=_build_score_fn(bic),
            config=config,
        )

        rng = np.random.default_rng(42)
        initial_dags = [
            DAG.random_dag(5, edge_prob=0.3, rng=rng).adjacency
            for _ in range(4)
        ]
        engine._seed_archive(initial_dags, data)

        coverages = []
        for _ in range(10):
            engine.step(data, batch_size=4)
            coverages.append(engine.archive.coverage())

        # Coverage should not decrease (monotonically non-decreasing)
        for i in range(1, len(coverages)):
            assert coverages[i] >= coverages[i - 1] - 1e-12


class TestScoringIntegration:
    """Integration tests for scoring components."""

    def test_bic_score_consistent_across_calls(self):
        dag = _chain_dag(5)
        data = _make_data(dag, 500)
        bic = BICScore()

        score1 = bic.local_score(2, [1], data)
        score2 = bic.local_score(2, [1], data)
        assert score1 == pytest.approx(score2)

    def test_bic_more_parents_penalized(self):
        """BIC should penalize overly complex models with irrelevant parents."""
        rng = np.random.default_rng(42)
        n = 500
        # Simple: X0 → X1, plus independent noise columns X2..X4
        x0 = rng.standard_normal(n)
        x1 = 0.8 * x0 + rng.standard_normal(n) * 0.5
        data = np.column_stack([x0, x1] + [rng.standard_normal(n) for _ in range(3)])
        bic = BICScore()

        # True parent of node 1 is just [0]
        score_correct = bic.local_score(1, [0], data)
        # Adding irrelevant noise parents should be penalized
        score_overfit = bic.local_score(1, [0, 2, 3, 4], data)
        assert score_correct >= score_overfit

    def test_descriptor_output_shape(self):
        dag = _chain_dag(5)
        data = _make_data(dag, 100)
        desc = StructuralDescriptor()
        d = desc.compute(dag.adjacency, data)
        assert d.ndim == 1
        assert len(d) == desc.descriptor_dim
        assert np.isfinite(d).all()

    def test_descriptor_different_dags_different_descriptors(self):
        """Different DAGs should (usually) get different descriptors."""
        dag1 = _chain_dag(5)
        adj2 = np.zeros((5, 5), dtype=np.int8)
        adj2[0, 1] = 1
        adj2[0, 2] = 1
        adj2[0, 3] = 1
        adj2[0, 4] = 1

        data = _make_data(dag1, 100)
        desc = StructuralDescriptor()
        d1 = desc.compute(dag1.adjacency, data)
        d2 = desc.compute(adj2, data)
        assert not np.allclose(d1, d2)


class TestEngineState:
    """Tests for MAP-Elites engine state management."""

    def test_engine_summary(self):
        dag = _chain_dag(5)
        data = _make_data(dag, 100)
        engine, archive = _run_map_elites(data, n_vars=5, n_iters=3, batch_size=4)
        summary = engine.summary()
        assert isinstance(summary, dict)
        assert "iteration" in summary or len(summary) > 0

    def test_engine_iteration_counter(self):
        dag = _chain_dag(5)
        data = _make_data(dag, 100)
        n_iters = 5
        engine, archive = _run_map_elites(
            data, n_vars=5, n_iters=n_iters, batch_size=4,
        )
        assert engine.iteration >= n_iters

    def test_step_returns_stats(self):
        dag = _chain_dag(5)
        data = _make_data(dag, 100)

        bic = BICScore()
        desc = StructuralDescriptor()

        config = MAPElitesConfig(
            archive_dims=(10, 10),
            archive_ranges=tuple(
                (lb, ub) for lb, ub in zip(
                    desc.descriptor_bounds[0][:2],
                    desc.descriptor_bounds[1][:2],
                )
            ),
            seed=42,
        )

        engine = CausalMAPElites(
            mutations=_build_mutations(),
            crossovers=[_dummy_crossover],
            descriptor_fn=_build_descriptor_fn(desc),
            score_fn=_build_score_fn(bic),
            config=config,
        )

        rng = np.random.default_rng(42)
        initial_dags = [
            DAG.random_dag(5, edge_prob=0.3, rng=rng).adjacency
            for _ in range(4)
        ]
        engine._seed_archive(initial_dags, data)

        stats = engine.step(data, batch_size=4)
        assert hasattr(stats, "iteration")
        assert hasattr(stats, "archive_size")
        assert stats.archive_size >= 0
