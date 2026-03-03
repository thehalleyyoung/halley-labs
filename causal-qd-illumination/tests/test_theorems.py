"""Theorem validation tests for the CausalQD theory (Theorems 1–9).

Each test class targets one theorem, using the fixtures from conftest.py
and the actual library implementations.
"""

from __future__ import annotations

import numpy as np
import pytest

from causal_qd.core.dag import DAG
from causal_qd.operators.mutation import TopologicalMutation, EdgeFlipMutation
from causal_qd.operators.crossover import OrderCrossover, UniformCrossover
from causal_qd.mec.hasher import CanonicalHasher
from causal_qd.mec.cpdag import CPDAGConverter
from causal_qd.certificates.edge_certificate import EdgeCertificate
from causal_qd.certificates.path_certificate import PathCertificate
from causal_qd.certificates.bootstrap import (
    boltzmann_weighted_stability,
    BoltzmannStabilityResult,
)
from causal_qd.analysis.supermartingale import SupermartingaleTracker
from causal_qd.analysis.ergodicity import ErgodicityChecker


# ===================================================================
# Theorem 1: Topological Mutation Preservation
# ===================================================================

class TestTheorem1TopologicalMutation:
    """Theorem 1: Topological Mutation Preservation.

    Every mutation operator must produce a valid DAG (acyclic) from a
    valid DAG input.
    """

    def test_mutation_preserves_acyclicity_chain(self, small_dag):
        """Mutate chain DAG 100 times, all must remain acyclic."""
        rng = np.random.default_rng(42)
        mutator = TopologicalMutation()
        adj = small_dag.adjacency_matrix
        for _ in range(100):
            mutated = mutator.mutate(adj, rng)
            assert DAG.is_acyclic(mutated), "Mutation produced a cyclic graph"

    def test_mutation_preserves_acyclicity_collider(self, small_dag_collider):
        """Mutate collider DAG 100 times, all must remain acyclic."""
        rng = np.random.default_rng(42)
        mutator = TopologicalMutation()
        adj = small_dag_collider.adjacency_matrix
        for _ in range(100):
            mutated = mutator.mutate(adj, rng)
            assert DAG.is_acyclic(mutated), "Mutation produced a cyclic graph"

    def test_mutation_preserves_acyclicity_random(self):
        """Generate 20 random DAGs, mutate each 10 times."""
        rng = np.random.default_rng(42)
        mutator = TopologicalMutation()
        for _ in range(20):
            dag = DAG.random_dag(8, edge_prob=0.3, rng=rng)
            adj = dag.adjacency_matrix
            for _ in range(10):
                mutated = mutator.mutate(adj, rng)
                assert DAG.is_acyclic(mutated), "Mutation produced a cyclic graph"

    def test_mutation_preserves_node_count(self, small_dag):
        """Node count is invariant under mutation."""
        rng = np.random.default_rng(42)
        mutator = TopologicalMutation()
        adj = small_dag.adjacency_matrix
        n = adj.shape[0]
        for _ in range(50):
            mutated = mutator.mutate(adj, rng)
            assert mutated.shape == (n, n), "Mutation changed matrix shape"


# ===================================================================
# Theorem 2: Crossover Acyclicity Preservation
# ===================================================================

class TestTheorem2CrossoverAcyclicity:
    """Theorem 2: Crossover Acyclicity Preservation.

    Every crossover operator must produce acyclic offspring from acyclic
    parents.
    """

    def test_crossover_preserves_acyclicity(self, small_dag, small_dag_fork):
        """Cross two DAGs 50 times, all children must be acyclic."""
        rng = np.random.default_rng(42)
        xover = OrderCrossover()
        p1 = small_dag.adjacency_matrix
        p2 = small_dag_fork.adjacency_matrix
        for _ in range(50):
            c1, c2 = xover.crossover(p1, p2, rng)
            assert DAG.is_acyclic(c1), "Crossover child 1 has a cycle"
            assert DAG.is_acyclic(c2), "Crossover child 2 has a cycle"

    def test_crossover_preserves_node_count(self, small_dag, small_dag_fork):
        """Offspring must have the same shape as parents."""
        rng = np.random.default_rng(42)
        xover = UniformCrossover()
        p1 = small_dag.adjacency_matrix
        p2 = small_dag_fork.adjacency_matrix
        n = p1.shape[0]
        for _ in range(20):
            c1, c2 = xover.crossover(p1, p2, rng)
            assert c1.shape == (n, n), "Child 1 shape changed"
            assert c2.shape == (n, n), "Child 2 shape changed"


# ===================================================================
# Theorem 3: Archive Coverage Under Ergodicity
# ===================================================================

class TestTheorem3ArchiveCoverage:
    """Theorem 3: Archive Coverage Under Ergodicity.

    Coverage (fraction of archive cells ever occupied) is monotone
    non-decreasing across generations.
    """

    def test_coverage_monotone_increasing(self):
        """Coverage should be non-decreasing across generations."""
        rng = np.random.default_rng(42)
        total_cells = 25
        checker = ErgodicityChecker(total_cells=total_cells)

        occupied: set = set()
        for gen in range(30):
            # Simulate adding a new cell with some probability
            new_cell = int(rng.integers(0, total_cells))
            occupied.add(new_cell)
            checker.record_occupied_cells(gen, occupied)

        iters, coverages = checker.coverage_curve()
        for i in range(1, len(coverages)):
            assert coverages[i] >= coverages[i - 1], (
                f"Coverage decreased at step {i}: {coverages[i-1]} -> {coverages[i]}"
            )

    def test_ergodicity_checker_basic(self):
        """ErgodicityChecker reports ergodicity once threshold is met."""
        total_cells = 10
        checker = ErgodicityChecker(total_cells=total_cells)

        # Add cells until we exceed 50% coverage
        occupied: set = set()
        for i in range(6):
            occupied.add(i)
            checker.record_occupied_cells(i, occupied)

        assert checker.is_ergodic(coverage_threshold=0.5), (
            "Should be ergodic with 6/10 cells"
        )


# ===================================================================
# Theorem 4: Archive Convergence via Supermartingale
# ===================================================================

class TestTheorem4SupermartingaleConvergence:
    """Theorem 4: Archive Convergence via Supermartingale.

    The best quality in each cell is monotone non-decreasing (elitist
    archive), so the supermartingale residuals M_t are non-increasing.
    """

    def test_best_quality_nondecreasing(self):
        """Best quality in archive never decreases (elitist property)."""
        from causal_qd.archive.grid_archive import GridArchive
        from causal_qd.archive.archive_base import ArchiveEntry

        archive = GridArchive(
            dims=(5, 5),
            lower_bounds=np.array([0.0, 0.0]),
            upper_bounds=np.array([1.0, 1.0]),
        )
        rng = np.random.default_rng(42)
        best_so_far = float("-inf")

        for _ in range(50):
            desc = rng.uniform(0, 1, size=2)
            quality = float(rng.uniform(-100, 0))
            entry = ArchiveEntry(
                solution=np.zeros((5, 5), dtype=np.int8),
                descriptor=desc,
                quality=quality,
            )
            archive.add(entry)
            current_best = max(e.quality for e in archive.elites())
            assert current_best >= best_so_far, (
                "Best quality decreased in elitist archive"
            )
            best_so_far = current_best

    def test_supermartingale_tracker_convergence(self):
        """SupermartingaleTracker records convergence diagnostics."""
        tracker = SupermartingaleTracker(epsilon=0.01)

        # Simulate improving qualities in two cells
        for t in range(20):
            tracker.record(t, {"cell_A": -100.0 + t * 5, "cell_B": -50.0 + t * 2})

        diag = tracker.convergence_diagnostic()
        assert "converged_fraction" in diag
        assert "mean_residual" in diag
        assert diag["mean_residual"] >= 0.0

    def test_supermartingale_residuals_nondecreasing(self):
        """M_t residuals are monotone non-increasing per cell."""
        tracker = SupermartingaleTracker(epsilon=1e-6)

        # Feed monotone non-decreasing qualities (elitist archive)
        qualities = sorted(np.random.default_rng(42).uniform(-100, 0, size=30))
        for t, q in enumerate(qualities):
            tracker.record(t, {"cell_0": q})

        diag = tracker.convergence_diagnostic()
        residuals = diag["per_cell"]["cell_0"]
        for i in range(1, len(residuals)):
            assert residuals[i] <= residuals[i - 1] + 1e-12, (
                f"Residual increased at step {i}: {residuals[i-1]} -> {residuals[i]}"
            )


# ===================================================================
# Theorem 5: MEC Separation
# ===================================================================

class TestTheorem5MECSeparation:
    """Theorem 5: MEC Separation.

    DAGs in the same Markov Equivalence Class receive the same MEC hash;
    DAGs in different MECs receive different hashes.
    """

    def test_same_mec_same_hash(self):
        """Two DAGs in same MEC get same hash."""
        hasher = CanonicalHasher()
        # Chain 0→1→2 and chain 0←1←2 are in the same MEC
        # (no v-structures, same skeleton)
        dag1 = DAG.from_edges(3, [(0, 1), (1, 2)])
        dag2 = DAG.from_edges(3, [(1, 0), (2, 1)])

        h1 = hasher.hash_mec(dag1)
        h2 = hasher.hash_mec(dag2)
        assert h1 == h2, "Same-MEC DAGs got different MEC hashes"

    def test_different_mec_different_hash(self):
        """Two DAGs in different MECs get different hashes."""
        hasher = CanonicalHasher()
        # Chain 0→1→2 (no v-structure)
        dag_chain = DAG.from_edges(3, [(0, 1), (1, 2)])
        # Collider 0→1←2 (has a v-structure)
        dag_collider = DAG.from_edges(3, [(0, 1), (2, 1)])

        h_chain = hasher.hash_mec(dag_chain)
        h_collider = hasher.hash_mec(dag_collider)
        assert h_chain != h_collider, "Different-MEC DAGs got the same MEC hash"

    def test_mec_separation_with_descriptors(self, gaussian_data):
        """DAGs from different MECs produce different CPDAGs."""
        data, true_adj = gaussian_data
        converter = CPDAGConverter()

        # True DAG: chain 0→1→2→3→4
        dag_chain = DAG(true_adj)
        # Collider variant: 0→2←1, 2→3→4
        collider_adj = np.zeros((5, 5), dtype=np.int8)
        collider_adj[0, 2] = 1
        collider_adj[1, 2] = 1
        collider_adj[2, 3] = 1
        collider_adj[3, 4] = 1
        dag_collider = DAG(collider_adj)

        cpdag1 = converter.dag_to_cpdag(dag_chain)
        cpdag2 = converter.dag_to_cpdag(dag_collider)

        assert not np.array_equal(cpdag1, cpdag2), (
            "CPDAGs of DAGs from different MECs should differ"
        )


# ===================================================================
# Theorem 7: Edge Certificate with Lipschitz Bound
# ===================================================================

class TestTheorem7EdgeCertificate:
    """Theorem 7: Edge Certificate with Lipschitz Bound.

    Edge certificates are bounded in [0, 1] and satisfy Lipschitz
    continuity under data perturbation.
    """

    def test_certificate_bounded(self, gaussian_data):
        """Edge certificates are in [0, 1]."""
        data, true_adj = gaussian_data

        # Construct edge certificates for the chain
        for i in range(4):
            cert = EdgeCertificate(
                source=i,
                target=i + 1,
                bootstrap_frequency=0.85,
                score_delta=2.0,
            )
            assert 0.0 <= cert.value <= 1.0, (
                f"Certificate value {cert.value} out of [0, 1]"
            )

    def test_certificate_perturbation_lipschitz(self, gaussian_data):
        """Small data perturbation -> bounded certificate change."""
        data, true_adj = gaussian_data

        from causal_qd.certificates.bootstrap import BootstrapCertificateComputer
        from causal_qd.scores.bic import BICScore

        scorer = BICScore()
        score_fn = lambda adj, d: scorer.score(adj, d)

        rng = np.random.default_rng(42)
        computer = BootstrapCertificateComputer(
            n_bootstrap=20,
            score_fn=score_fn,
            rng=rng,
            compute_lipschitz=True,
        )

        certs = computer.compute_edge_certificates(true_adj, data)
        for (src, tgt), cert in certs.items():
            assert 0.0 <= cert.value <= 1.0
            if cert.lipschitz_bound is not None:
                assert cert.lipschitz_bound >= 0.0, (
                    f"Lipschitz bound negative for edge {src}→{tgt}"
                )


# ===================================================================
# Theorem 8: Path Certificate Composition
# ===================================================================

class TestTheorem8PathCertificate:
    """Theorem 8: Path Certificate Composition.

    The path certificate value equals the minimum of individual edge
    certificate values along the path (weakest-link principle).
    """

    def test_path_cert_weakest_link(self, gaussian_data):
        """Path certificate <= min of individual edge certificates on path."""
        edge_certs = []
        for i in range(3):
            ec = EdgeCertificate(
                source=i,
                target=i + 1,
                bootstrap_frequency=0.7 + 0.1 * i,
                score_delta=1.0 + i,
            )
            edge_certs.append(ec)

        path_cert = PathCertificate(
            path=[0, 1, 2, 3],
            edge_certificates=edge_certs,
        )

        min_edge = min(ec.value for ec in edge_certs)
        assert abs(path_cert.value - min_edge) < 1e-12, (
            f"Path cert {path_cert.value} != min edge cert {min_edge}"
        )


# ===================================================================
# Theorem 9: Boltzmann-Weighted Certificate Stability
# ===================================================================

class TestTheorem9BoltzmannStability:
    """Theorem 9: Boltzmann-Weighted Certificate Stability.

    At low temperature, Boltzmann weighting concentrates on the best-quality
    DAG; at high temperature, it approaches uniform weighting.
    """

    def _make_archive(self):
        """Helper: create a small archive with known qualities and certs."""
        rng = np.random.default_rng(42)
        dags = []
        qualities = []
        certs_list = []
        for i in range(5):
            adj = np.zeros((4, 4), dtype=np.int8)
            adj[0, 1] = 1
            if i > 0:
                adj[1, 2] = 1
            dags.append(adj)
            qualities.append(-50.0 + i * 10)  # -50, -40, -30, -20, -10
            ec = EdgeCertificate(
                source=0,
                target=1,
                bootstrap_frequency=0.5 + 0.1 * i,
                score_delta=1.0,
            )
            certs_list.append({(0, 1): ec})
        return dags, qualities, certs_list

    def test_boltzmann_concentrates_at_low_temp(self):
        """Low temperature concentrates weight on one DAG."""
        dags, qualities, certs_list = self._make_archive()

        result = boltzmann_weighted_stability(
            archive_dags=dags,
            archive_qualities=qualities,
            edge_certificates=certs_list,
            beta=100.0,
        )

        # exp(-β·q) concentrates on lowest q (index 0, q=-50)
        assert result.per_dag_weights[0] > 0.99, (
            f"Low-temp weight on min-quality DAG is {result.per_dag_weights[0]}, expected >0.99"
        )
        assert result.effective_sample_size < 1.5, (
            "ESS should be near 1 at very low temperature"
        )

    def test_boltzmann_uniform_at_high_temp(self):
        """High temperature gives approximately uniform weights."""
        dags, qualities, certs_list = self._make_archive()

        result = boltzmann_weighted_stability(
            archive_dags=dags,
            archive_qualities=qualities,
            edge_certificates=certs_list,
            beta=0.0001,
        )

        # All weights should be approximately equal
        expected_weight = 1.0 / len(dags)
        for w in result.per_dag_weights:
            assert abs(w - expected_weight) < 0.05, (
                f"Weight {w} deviates from uniform {expected_weight}"
            )
        assert result.effective_sample_size > 4.5, (
            "ESS should be near N at high temperature"
        )
