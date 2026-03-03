"""Parallelised bootstrap resampling for DAG quality estimation.

Provides:
  - BootstrapParallel: distribute bootstrap samples across processes
  - Parallel score computation for bootstrap
  - Result aggregation with proper statistics (mean, CI, std)
"""
from __future__ import annotations

import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from causal_qd.types import AdjacencyMatrix, DataMatrix, QualityScore

if TYPE_CHECKING:
    from causal_qd.core.dag import DAG

logger = logging.getLogger(__name__)


# ======================================================================
# Worker functions
# ======================================================================


def _bootstrap_worker(
    args: Tuple[Any, DataMatrix, Callable[..., QualityScore], int],
) -> QualityScore:
    """Evaluate a single bootstrap resample in a child process."""
    dag, data, score_fn, seed = args
    rng = np.random.default_rng(seed)
    n = data.shape[0]
    indices = rng.integers(0, n, size=n)
    return score_fn(dag, data[indices])


def _bootstrap_batch_worker(
    args: Tuple[Any, DataMatrix, Callable[..., QualityScore], List[int]],
) -> List[QualityScore]:
    """Evaluate multiple bootstrap resamples in one worker call."""
    dag, data, score_fn, seeds = args
    n = data.shape[0]
    results = []
    for seed in seeds:
        rng = np.random.default_rng(seed)
        indices = rng.integers(0, n, size=n)
        try:
            score = score_fn(dag, data[indices])
        except Exception:
            score = float("nan")
        results.append(score)
    return results


def _bootstrap_edge_stability_worker(
    args: Tuple[AdjacencyMatrix, DataMatrix, Callable, int],
) -> AdjacencyMatrix:
    """Evaluate which edges are significant in a bootstrap resample."""
    adj, data, learn_fn, seed = args
    rng = np.random.default_rng(seed)
    n = data.shape[0]
    indices = rng.integers(0, n, size=n)
    try:
        learned_adj = learn_fn(data[indices])
        return np.asarray(learned_adj, dtype=np.int8)
    except Exception:
        return np.zeros_like(adj)


# ======================================================================
# Bootstrap statistics
# ======================================================================


class BootstrapStatistics:
    """Aggregated bootstrap statistics.

    Attributes
    ----------
    scores : list of float
        Individual bootstrap scores.
    mean : float
        Mean of bootstrap scores.
    std : float
        Standard deviation of bootstrap scores.
    ci_lower : float
        Lower bound of confidence interval.
    ci_upper : float
        Upper bound of confidence interval.
    confidence_level : float
        Confidence level used for the CI.
    """

    def __init__(
        self,
        scores: List[float],
        confidence_level: float = 0.95,
    ) -> None:
        self.scores = scores
        self.confidence_level = confidence_level
        valid = [s for s in scores if np.isfinite(s)]
        self.n_valid = len(valid)
        self.n_failed = len(scores) - len(valid)

        if valid:
            self.mean = float(np.mean(valid))
            self.std = float(np.std(valid, ddof=1)) if len(valid) > 1 else 0.0
            alpha = 1.0 - confidence_level
            self.ci_lower = float(np.percentile(valid, 100 * alpha / 2))
            self.ci_upper = float(np.percentile(valid, 100 * (1 - alpha / 2)))
            self.median = float(np.median(valid))
            self.se = self.std / np.sqrt(len(valid)) if len(valid) > 0 else 0.0
        else:
            self.mean = self.std = self.median = self.se = 0.0
            self.ci_lower = self.ci_upper = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "mean": self.mean,
            "std": self.std,
            "median": self.median,
            "se": self.se,
            "ci_lower": self.ci_lower,
            "ci_upper": self.ci_upper,
            "confidence_level": self.confidence_level,
            "n_valid": self.n_valid,
            "n_failed": self.n_failed,
        }


# ======================================================================
# BootstrapParallel
# ======================================================================


class BootstrapParallel:
    """Parallelise bootstrap resampling across multiple processes.

    Supports both single-DAG score bootstrapping and edge-stability
    analysis.
    """

    # ------------------------------------------------------------------
    # Score bootstrapping
    # ------------------------------------------------------------------

    @staticmethod
    def run(
        dag: Any,
        data: DataMatrix,
        n_bootstrap: int,
        score_fn: Callable[..., QualityScore],
        n_workers: int = 1,
        base_seed: int = 0,
    ) -> List[QualityScore]:
        """Run *n_bootstrap* bootstrap evaluations of *dag*.

        Parameters
        ----------
        dag :
            The DAG to evaluate.
        data :
            ``N × p`` data matrix.
        n_bootstrap :
            Number of bootstrap resamples.
        score_fn :
            Callable ``(dag, data) -> QualityScore``.
        n_workers :
            Number of worker processes (1 = sequential).
        base_seed :
            Base seed for reproducibility.

        Returns
        -------
        list of QualityScore
        """
        seeds = list(range(base_seed, base_seed + n_bootstrap))

        if n_workers > 1:
            # Batch seeds across workers
            chunk_size = max(1, n_bootstrap // n_workers)
            seed_chunks = [seeds[i:i + chunk_size]
                           for i in range(0, n_bootstrap, chunk_size)]
            batch_args = [(dag, data, score_fn, chunk) for chunk in seed_chunks]

            with ProcessPoolExecutor(max_workers=n_workers) as pool:
                futures = [pool.submit(_bootstrap_batch_worker, a)
                           for a in batch_args]
                results: List[QualityScore] = []
                for f in as_completed(futures):
                    try:
                        results.extend(f.result())
                    except Exception as exc:
                        logger.warning("Bootstrap worker failed: %s", exc)
                        results.extend([float("nan")] * chunk_size)
        else:
            args_list = [(dag, data, score_fn, s) for s in seeds]
            results = [_bootstrap_worker(a) for a in args_list]

        return results[:n_bootstrap]

    @staticmethod
    def run_with_statistics(
        dag: Any,
        data: DataMatrix,
        n_bootstrap: int,
        score_fn: Callable[..., QualityScore],
        n_workers: int = 1,
        confidence_level: float = 0.95,
        base_seed: int = 0,
    ) -> BootstrapStatistics:
        """Run bootstrap and return aggregated statistics.

        Parameters
        ----------
        dag :
            DAG to evaluate.
        data :
            Data matrix.
        n_bootstrap :
            Number of resamples.
        score_fn :
            Score function.
        n_workers :
            Worker count.
        confidence_level :
            CI confidence level.
        base_seed :
            Base seed.

        Returns
        -------
        BootstrapStatistics
        """
        scores = BootstrapParallel.run(
            dag, data, n_bootstrap, score_fn, n_workers, base_seed,
        )
        return BootstrapStatistics(scores, confidence_level)

    # ------------------------------------------------------------------
    # Edge stability bootstrapping
    # ------------------------------------------------------------------

    @staticmethod
    def edge_stability(
        adj: AdjacencyMatrix,
        data: DataMatrix,
        learn_fn: Callable[[DataMatrix], AdjacencyMatrix],
        n_bootstrap: int = 100,
        n_workers: int = 1,
        base_seed: int = 0,
    ) -> np.ndarray:
        """Compute edge stability via bootstrap.

        For each bootstrap resample, learn a DAG using *learn_fn* and
        count how often each edge appears.

        Parameters
        ----------
        adj :
            Reference adjacency matrix (used for shape).
        data :
            Data matrix.
        learn_fn :
            Callable ``data -> adjacency_matrix``.
        n_bootstrap :
            Number of bootstrap resamples.
        n_workers :
            Worker count.
        base_seed :
            Base seed.

        Returns
        -------
        np.ndarray
            ``n × n`` matrix of edge frequencies in ``[0, 1]``.
        """
        seeds = list(range(base_seed, base_seed + n_bootstrap))
        adj = np.asarray(adj, dtype=np.int8)
        n = adj.shape[0]
        edge_counts = np.zeros((n, n), dtype=np.float64)

        if n_workers > 1:
            args_list = [(adj, data, learn_fn, s) for s in seeds]
            with ProcessPoolExecutor(max_workers=n_workers) as pool:
                futures = [pool.submit(_bootstrap_edge_stability_worker, a)
                           for a in args_list]
                for f in as_completed(futures):
                    try:
                        learned = f.result()
                        edge_counts += (learned > 0).astype(np.float64)
                    except Exception as exc:
                        logger.warning("Edge stability worker failed: %s", exc)
        else:
            for seed in seeds:
                result = _bootstrap_edge_stability_worker(
                    (adj, data, learn_fn, seed),
                )
                edge_counts += (result > 0).astype(np.float64)

        return edge_counts / max(n_bootstrap, 1)

    # ------------------------------------------------------------------
    # Multiple DAGs
    # ------------------------------------------------------------------

    @staticmethod
    def run_multiple(
        dags: List[Any],
        data: DataMatrix,
        n_bootstrap: int,
        score_fn: Callable[..., QualityScore],
        n_workers: int = 1,
        base_seed: int = 0,
    ) -> List[BootstrapStatistics]:
        """Run bootstrap for multiple DAGs.

        Parameters
        ----------
        dags :
            List of DAGs.
        data :
            Data matrix.
        n_bootstrap :
            Resamples per DAG.
        score_fn :
            Score function.
        n_workers :
            Worker count.
        base_seed :
            Base seed.

        Returns
        -------
        list of BootstrapStatistics
        """
        results = []
        for idx, dag in enumerate(dags):
            stats = BootstrapParallel.run_with_statistics(
                dag, data, n_bootstrap, score_fn, n_workers,
                base_seed=base_seed + idx * n_bootstrap,
            )
            results.append(stats)
        return results
