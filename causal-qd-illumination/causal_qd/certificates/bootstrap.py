"""Bootstrap-based certificate computation for DAG edges.

Provides :class:`BootstrapCertificateComputer` which resamples the data
and re-evaluates edges to compute :class:`EdgeCertificate` instances
with bootstrap frequencies, score deltas, confidence intervals, and
optional Lipschitz bounds.

Also provides Boltzmann-weighted certificate stability (Theorem 9):
:func:`boltzmann_weighted_stability`, :func:`boltzmann_edge_probabilities`,
and :func:`optimal_beta`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Dict, List, Optional, Tuple

import numpy as np

from causal_qd.certificates.edge_certificate import EdgeCertificate
from causal_qd.types import BootstrapSample, DataMatrix

if TYPE_CHECKING:
    from causal_qd.types import AdjacencyMatrix

ScoreFunction = Callable[["AdjacencyMatrix", DataMatrix], float]


# ---------------------------------------------------------------------------
# Boltzmann-weighted certificate stability (Theorem 9)
# ---------------------------------------------------------------------------

@dataclass
class BoltzmannStabilityResult:
    """Result of Boltzmann-weighted archive certificate stability.

    Attributes
    ----------
    stability_score : float
        S_β(A) — the Boltzmann-weighted average certificate quality.
    partition_function : float
        Z_β = Σ exp(-β·q(G)).
    per_dag_weights : np.ndarray
        Normalised Boltzmann weights for each DAG.
    per_dag_avg_certs : np.ndarray
        C_avg(G) for each DAG.
    effective_sample_size : float
        ESS = (Σw_i)² / Σw_i² to diagnose weight concentration.
    beta : float
        Inverse temperature used.
    """

    stability_score: float
    partition_function: float
    per_dag_weights: np.ndarray
    per_dag_avg_certs: np.ndarray
    effective_sample_size: float
    beta: float


def boltzmann_weighted_stability(
    archive_dags: List[np.ndarray],
    archive_qualities: List[float],
    edge_certificates: List[Dict[Tuple[int, int], EdgeCertificate]],
    beta: float = 1.0,
) -> BoltzmannStabilityResult:
    """Compute Boltzmann-weighted certificate stability (Theorem 9).

    S_β(A) = Σ_G exp(-β·q(G)) · C_avg(G) / Z_β

    Parameters
    ----------
    archive_dags : list of np.ndarray
        Adjacency matrices for each DAG in the archive.
    archive_qualities : list of float
        Quality score q(G, D) for each DAG.
    edge_certificates : list of dict
        Per-DAG mapping ``{(src, tgt): EdgeCertificate}``.
    beta : float
        Inverse temperature (≥ 0).

    Returns
    -------
    BoltzmannStabilityResult
    """
    n = len(archive_dags)
    if n == 0:
        raise ValueError("archive must contain at least one DAG")
    if len(archive_qualities) != n or len(edge_certificates) != n:
        raise ValueError("archive_dags, archive_qualities, and edge_certificates must have the same length")

    qualities = np.asarray(archive_qualities, dtype=np.float64)

    # Log-sum-exp trick for numerical stability
    log_weights = -beta * qualities
    max_lw = np.max(log_weights)
    shifted = log_weights - max_lw
    exp_shifted = np.exp(shifted)
    log_partition = max_lw + np.log(np.sum(exp_shifted))
    partition_function = float(np.exp(np.clip(log_partition, -700, 700)))
    weights = exp_shifted / np.sum(exp_shifted)  # normalised weights

    # Compute C_avg(G) for each DAG
    avg_certs = np.empty(n, dtype=np.float64)
    for i, certs in enumerate(edge_certificates):
        if certs:
            avg_certs[i] = float(np.mean([c.value for c in certs.values()]))
        else:
            avg_certs[i] = 0.0

    stability_score = float(np.dot(weights, avg_certs))

    # ESS = (Σw_i)² / Σ(w_i²);  weights already sum to 1
    ess = 1.0 / float(np.sum(weights ** 2))

    return BoltzmannStabilityResult(
        stability_score=stability_score,
        partition_function=partition_function,
        per_dag_weights=weights,
        per_dag_avg_certs=avg_certs,
        effective_sample_size=ess,
        beta=beta,
    )


def boltzmann_edge_probabilities(
    archive_dags: List[np.ndarray],
    archive_qualities: List[float],
    edge_certificates: List[Dict[Tuple[int, int], EdgeCertificate]],
    beta: float = 1.0,
) -> np.ndarray:
    """Boltzmann-weighted edge probability matrix.

    For every possible edge (i, j), computes the weighted average of
    the edge certificate value across the archive, using Boltzmann
    weights derived from quality scores.

    Returns
    -------
    np.ndarray
        Square matrix P where P[i, j] is the Boltzmann-weighted
        certificate value for edge i→j (0 if the edge never appears).
    """
    if not archive_dags:
        raise ValueError("archive must contain at least one DAG")

    result = boltzmann_weighted_stability(
        archive_dags, archive_qualities, edge_certificates, beta
    )
    weights = result.per_dag_weights

    n_nodes = archive_dags[0].shape[0]
    prob_matrix = np.zeros((n_nodes, n_nodes), dtype=np.float64)

    for idx, certs in enumerate(edge_certificates):
        for (src, tgt), cert in certs.items():
            prob_matrix[src, tgt] += weights[idx] * cert.value

    return prob_matrix


def optimal_beta(archive_qualities: List[float]) -> float:
    """Heuristic for choosing the inverse temperature β.

    Uses β = 1 / std(qualities).  Falls back to 1.0 when the standard
    deviation is zero or the archive is empty.

    Parameters
    ----------
    archive_qualities : list of float
        Quality scores for each DAG in the archive.

    Returns
    -------
    float
        Recommended β value.
    """
    if len(archive_qualities) < 2:
        return 1.0
    s = float(np.std(archive_qualities))
    if s < 1e-15:
        return 1.0
    return 1.0 / s


class BootstrapCertificateComputer:
    """Compute edge certificates via bootstrap resampling.

    For each bootstrap sample the score function is re-evaluated for the
    original DAG and for the DAG with each edge removed.  The bootstrap
    frequency of an edge is the fraction of samples where including the
    edge improves the score, and the score delta is the average
    improvement.

    Parameters
    ----------
    n_bootstrap : int
        Number of bootstrap resamples to draw.
    score_fn : ScoreFunction
        Scoring function ``(adjacency_matrix, data) -> float``.
    confidence_level : float
        Confidence level attached to the resulting certificates.
    rng : np.random.Generator or None
        Optional random generator for reproducibility.
    compute_lipschitz : bool
        If ``True``, compute an empirical Lipschitz bound for each edge.
    lipschitz_perturbation_scale : float
        Scale for Lipschitz perturbation (fraction of std).
    """

    def __init__(
        self,
        n_bootstrap: int,
        score_fn: ScoreFunction,
        confidence_level: float = 0.95,
        rng: np.random.Generator | None = None,
        compute_lipschitz: bool = False,
        lipschitz_perturbation_scale: float = 0.01,
    ) -> None:
        if n_bootstrap < 1:
            raise ValueError(f"n_bootstrap must be >= 1, got {n_bootstrap}")
        if not 0.0 < confidence_level < 1.0:
            raise ValueError(
                f"confidence_level must be in (0, 1), got {confidence_level}"
            )

        self._n_bootstrap = n_bootstrap
        self._score_fn = score_fn
        self._confidence_level = confidence_level
        self._rng = rng if rng is not None else np.random.default_rng()
        self._compute_lipschitz = compute_lipschitz
        self._lip_scale = lipschitz_perturbation_scale

    # -- public API ----------------------------------------------------------

    def compute_edge_certificates(
        self,
        dag: "AdjacencyMatrix",
        data: DataMatrix,
    ) -> Dict[Tuple[int, int], EdgeCertificate]:
        """Compute an :class:`EdgeCertificate` for every edge in *dag*.

        Parameters
        ----------
        dag : AdjacencyMatrix
            Adjacency matrix (``dag[i, j] == 1`` iff i → j).
        data : DataMatrix
            Observed data matrix.

        Returns
        -------
        Dict[Tuple[int, int], EdgeCertificate]
            Mapping from ``(source, target)`` to its certificate.
        """
        edges = list(zip(*np.nonzero(dag)))
        if not edges:
            return {}

        n_obs = data.shape[0]
        edge_hits: Dict[Tuple[int, int], int] = {e: 0 for e in edges}
        edge_deltas: Dict[Tuple[int, int], List[float]] = {
            e: [] for e in edges
        }

        for _ in range(self._n_bootstrap):
            indices: BootstrapSample = self._rng.integers(
                0, n_obs, size=n_obs
            ).astype(np.int64)
            boot_data = data[indices]

            base_score = self._score_fn(dag, boot_data)

            for src, tgt in edges:
                modified = dag.copy()
                modified[src, tgt] = 0
                reduced_score = self._score_fn(modified, boot_data)
                delta = base_score - reduced_score
                if delta > 0:
                    edge_hits[(src, tgt)] += 1
                edge_deltas[(src, tgt)].append(float(delta))

        # Compute Lipschitz bounds if requested
        lip_bounds: Dict[Tuple[int, int], float] = {}
        if self._compute_lipschitz:
            lip_bounds = self._compute_edge_lipschitz(dag, data, edges)

        result: Dict[Tuple[int, int], EdgeCertificate] = {}
        for src, tgt in edges:
            freq = edge_hits[(src, tgt)] / self._n_bootstrap
            avg_delta = float(np.mean(edge_deltas[(src, tgt)]))
            result[(src, tgt)] = EdgeCertificate(
                source=src,
                target=tgt,
                bootstrap_frequency=freq,
                score_delta=avg_delta,
                confidence=self._confidence_level,
                lipschitz_bound=lip_bounds.get((src, tgt)),
                bootstrap_deltas=edge_deltas[(src, tgt)],
                n_bootstrap=self._n_bootstrap,
            )
        return result

    def compute_all_certificates(
        self,
        dag: "AdjacencyMatrix",
        data: DataMatrix,
    ) -> List[EdgeCertificate]:
        """Return edge certificates sorted by ``(source, target)``.

        Convenience wrapper around :meth:`compute_edge_certificates`.
        """
        certs = self.compute_edge_certificates(dag, data)
        return [certs[k] for k in sorted(certs)]

    def compute_nonedge_certificates(
        self,
        dag: "AdjacencyMatrix",
        data: DataMatrix,
        candidate_edges: Optional[List[Tuple[int, int]]] = None,
    ) -> Dict[Tuple[int, int], EdgeCertificate]:
        """Compute certificates for absent edges (evidence for non-existence).

        For each non-edge, measures how often adding it *worsens* the
        score across bootstrap samples.

        Parameters
        ----------
        dag : AdjacencyMatrix
        data : DataMatrix
        candidate_edges : List of (i, j), optional
            Non-edges to evaluate. If ``None``, all non-edges are tested.

        Returns
        -------
        Dict[Tuple[int, int], EdgeCertificate]
        """
        n = dag.shape[0]
        existing = set(zip(*np.nonzero(dag)))
        if candidate_edges is None:
            candidate_edges = [
                (i, j) for i in range(n) for j in range(n)
                if i != j and (i, j) not in existing
            ]

        n_obs = data.shape[0]
        edge_absent_hits: Dict[Tuple[int, int], int] = {
            e: 0 for e in candidate_edges
        }
        edge_deltas: Dict[Tuple[int, int], List[float]] = {
            e: [] for e in candidate_edges
        }

        for _ in range(self._n_bootstrap):
            idx = self._rng.integers(0, n_obs, size=n_obs).astype(np.int64)
            boot_data = data[idx]
            base_score = self._score_fn(dag, boot_data)

            for src, tgt in candidate_edges:
                modified = dag.copy()
                modified[src, tgt] = 1
                # Check acyclicity quickly
                if not self._is_acyclic_fast(modified):
                    edge_absent_hits[(src, tgt)] += 1
                    edge_deltas[(src, tgt)].append(0.0)
                    continue
                modified_score = self._score_fn(modified, boot_data)
                delta = base_score - modified_score  # positive = absence preferred
                if delta >= 0:
                    edge_absent_hits[(src, tgt)] += 1
                edge_deltas[(src, tgt)].append(float(delta))

        result: Dict[Tuple[int, int], EdgeCertificate] = {}
        for src, tgt in candidate_edges:
            freq = edge_absent_hits[(src, tgt)] / self._n_bootstrap
            avg_delta = float(np.mean(edge_deltas[(src, tgt)]))
            result[(src, tgt)] = EdgeCertificate(
                source=src,
                target=tgt,
                bootstrap_frequency=freq,
                score_delta=avg_delta,
                confidence=self._confidence_level,
                bootstrap_deltas=edge_deltas[(src, tgt)],
                n_bootstrap=self._n_bootstrap,
            )
        return result

    # -- internal ------------------------------------------------------------

    def _compute_edge_lipschitz(
        self,
        dag: "AdjacencyMatrix",
        data: DataMatrix,
        edges: List[Tuple[int, int]],
    ) -> Dict[Tuple[int, int], float]:
        """Estimate per-edge Lipschitz bounds via finite differences."""
        stds = np.std(data, axis=0)
        stds[stds == 0] = 1.0
        result: Dict[Tuple[int, int], float] = {}

        for src, tgt in edges:
            dag_mod = dag.copy()
            dag_mod[src, tgt] = 0

            base_with = self._score_fn(dag, data)
            base_without = self._score_fn(dag_mod, data)
            base_delta = abs(base_with - base_without)

            max_ratio = 0.0
            for _ in range(10):
                perturbation = (
                    self._rng.standard_normal(data.shape) * stds * self._lip_scale
                )
                pert_data = data + perturbation
                pert_with = self._score_fn(dag, pert_data)
                pert_without = self._score_fn(dag_mod, pert_data)
                pert_delta = abs(pert_with - pert_without)
                diff = abs(pert_delta - base_delta)
                norm = float(np.linalg.norm(perturbation))
                if norm > 1e-15:
                    max_ratio = max(max_ratio, diff / norm)

            result[(src, tgt)] = max_ratio
        return result

    @staticmethod
    def _is_acyclic_fast(adj: "AdjacencyMatrix") -> bool:
        """Quick acyclicity check using Kahn's algorithm."""
        from collections import deque

        n = adj.shape[0]
        in_degree = adj.sum(axis=0).copy()
        queue = deque(int(i) for i in range(n) if in_degree[i] == 0)
        visited = 0
        while queue:
            node = queue.popleft()
            visited += 1
            for child in range(n):
                if adj[node, child]:
                    in_degree[child] -= 1
                    if in_degree[child] == 0:
                        queue.append(child)
        return visited == n
