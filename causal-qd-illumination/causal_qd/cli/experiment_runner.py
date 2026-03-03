"""Orchestrates full CausalQD MAP-Elites experiments.

Provides:
  - ExperimentRunner: set up and run a single experiment
  - Grid search over hyperparameters
  - Result aggregation across multiple runs
  - Reproducibility via seed management
"""
from __future__ import annotations

import copy
import itertools
import json
import logging
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from causal_qd.config.config import CausalQDConfig
from causal_qd.streaming.online_archive import OnlineArchive
from causal_qd.types import DataMatrix
from causal_qd.utils.graph_utils import is_dag
from causal_qd.utils.random_utils import set_seed

logger = logging.getLogger(__name__)


class ExperimentRunner:
    """Set up and execute a CausalQD experiment.

    Parameters
    ----------
    config :
        Full experiment configuration.
    progress_callback :
        Optional callable invoked once per generation for progress tracking.
    """

    def __init__(
        self,
        config: CausalQDConfig,
        progress_callback: Optional[Callable[[], None]] = None,
    ) -> None:
        self._cfg = config
        self._progress_callback = progress_callback

    def run(self, data: DataMatrix) -> Dict[str, Any]:
        """Execute the MAP-Elites loop and return aggregated results.

        Parameters
        ----------
        data :
            ``N × p`` observational data matrix.

        Returns
        -------
        dict
            Dictionary containing:
            - ``best_quality`` – best quality found
            - ``best_adjacency`` – adjacency matrix of the best DAG
            - ``qd_score`` – final QD-score
            - ``coverage`` – final archive coverage
            - ``qd_score_history`` – QD-score per generation
            - ``coverage_history`` – coverage per generation
            - ``best_quality_history`` – best quality per generation
            - ``elapsed_seconds`` – wall-clock time
            - ``n_elites`` – number of filled cells
            - ``n_evaluated`` – total DAGs evaluated
        """
        set_seed(self._cfg.experiment.seed)
        rng = np.random.default_rng(self._cfg.experiment.seed)

        n = self._cfg.n_nodes
        archive = OnlineArchive(
            dims=self._cfg.archive.dims,
            bounds=self._cfg.archive.descriptor_bounds,
        )

        qd_score_history: List[float] = []
        coverage_history: List[float] = []
        best_quality_history: List[float] = []

        best_quality: float = -np.inf
        best_adj: np.ndarray = np.zeros((n, n), dtype=np.int8)
        n_evaluated = 0

        # Build score function
        score_fn = self._build_score_fn(data)

        t0 = time.time()

        for iteration in range(self._cfg.experiment.n_iterations):
            for _ in range(self._cfg.experiment.batch_size):
                # Generate a candidate DAG
                adj = self._generate_candidate(n, rng, archive)

                if not is_dag(adj):
                    continue

                n_evaluated += 1

                # Score the DAG
                quality = self._evaluate_quality(data, adj, n, score_fn)

                # Compute descriptor
                descriptor = self._compute_descriptor(adj, n)

                archive.add(adj, quality, descriptor)
                if quality > best_quality:
                    best_quality = quality
                    best_adj = adj.copy()

            # Record history
            qd_total = sum(e.quality for e in archive.elites.values())
            cov = len(archive.elites) / max(archive.total_cells, 1)
            qd_score_history.append(qd_total)
            coverage_history.append(cov)
            best_quality_history.append(best_quality)

            if self._progress_callback is not None:
                self._progress_callback()

        elapsed = time.time() - t0

        return {
            "best_quality": best_quality,
            "best_adjacency": best_adj.tolist(),
            "qd_score": qd_score_history[-1] if qd_score_history else 0.0,
            "coverage": coverage_history[-1] if coverage_history else 0.0,
            "qd_score_history": qd_score_history,
            "coverage_history": coverage_history,
            "best_quality_history": best_quality_history,
            "elapsed_seconds": elapsed,
            "n_elites": len(archive.elites),
            "n_evaluated": n_evaluated,
            "n_iterations": self._cfg.experiment.n_iterations,
            "batch_size": self._cfg.experiment.batch_size,
            "seed": self._cfg.experiment.seed,
        }

    # ------------------------------------------------------------------
    # Grid search
    # ------------------------------------------------------------------

    @staticmethod
    def grid_search(
        data: DataMatrix,
        base_config: CausalQDConfig,
        param_grid: Dict[str, List[Any]],
        n_repeats: int = 1,
    ) -> List[Dict[str, Any]]:
        """Run experiments over a grid of hyperparameters.

        Parameters
        ----------
        data :
            Data matrix.
        base_config :
            Base configuration (will be copied and modified).
        param_grid :
            Dict mapping parameter paths to lists of values.
            Example: ``{"experiment.n_iterations": [100, 500],
                        "experiment.batch_size": [10, 50]}``
        n_repeats :
            Number of repeated runs per configuration.

        Returns
        -------
        list of dict
            Results for each configuration and repeat.
        """
        keys = list(param_grid.keys())
        values = list(param_grid.values())
        all_results: List[Dict[str, Any]] = []

        for combo in itertools.product(*values):
            for repeat in range(n_repeats):
                cfg = copy.deepcopy(base_config)
                params = {}
                for key, val in zip(keys, combo):
                    parts = key.split(".")
                    obj = cfg
                    for part in parts[:-1]:
                        obj = getattr(obj, part)
                    setattr(obj, parts[-1], val)
                    params[key] = val

                cfg.experiment.seed = base_config.experiment.seed + repeat

                runner = ExperimentRunner(cfg)
                result = runner.run(data)
                result["params"] = params
                result["repeat"] = repeat
                all_results.append(result)

                logger.info("Grid search: params=%s, repeat=%d, qd=%.2f, cov=%.4f",
                            params, repeat, result["qd_score"], result["coverage"])

        return all_results

    @staticmethod
    def aggregate_results(results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate results from multiple runs.

        Parameters
        ----------
        results :
            List of result dicts from multiple runs.

        Returns
        -------
        dict
            Aggregated statistics (mean, std of key metrics).
        """
        if not results:
            return {}

        metrics = ["qd_score", "coverage", "best_quality", "elapsed_seconds", "n_elites"]
        agg: Dict[str, Any] = {"n_runs": len(results)}

        for metric in metrics:
            values = [r.get(metric, 0) for r in results if metric in r]
            if values:
                agg[f"{metric}_mean"] = float(np.mean(values))
                agg[f"{metric}_std"] = float(np.std(values))
                agg[f"{metric}_min"] = float(np.min(values))
                agg[f"{metric}_max"] = float(np.max(values))

        return agg

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _generate_candidate(
        self, n: int, rng: np.random.Generator, archive: OnlineArchive,
    ) -> np.ndarray:
        """Generate a candidate DAG (random or mutated from archive)."""
        # With 50% probability, mutate an existing elite if archive is non-empty
        if archive.elites and rng.random() < 0.5:
            elites = list(archive.elites.values())
            parent = elites[rng.integers(len(elites))]
            adj = parent.dag.copy() if isinstance(parent.dag, np.ndarray) else np.array(parent.dag, dtype=np.int8)
            # Apply random mutation
            op = rng.choice(["add", "remove", "reverse"])
            if op == "add":
                zeros = list(zip(*np.where((adj == 0) & (np.eye(n, dtype=np.int8) == 0))))
                if zeros:
                    i, j = zeros[rng.integers(len(zeros))]
                    adj[i, j] = 1
            elif op == "remove":
                edges = list(zip(*np.nonzero(adj)))
                if edges:
                    i, j = edges[rng.integers(len(edges))]
                    adj[i, j] = 0
            else:
                edges = list(zip(*np.nonzero(adj)))
                if edges:
                    i, j = edges[rng.integers(len(edges))]
                    adj[i, j] = 0
                    adj[j, i] = 1
            return adj

        # Random DAG
        adj = np.zeros((n, n), dtype=np.int8)
        order = rng.permutation(n)
        for idx_i in range(n):
            for idx_j in range(idx_i + 1, n):
                if rng.random() < 0.3:
                    adj[order[idx_i], order[idx_j]] = 1
        return adj

    def _build_score_fn(self, data: DataMatrix) -> Any:
        """Build a scoring function based on config."""
        return None  # Use built-in BIC

    def _evaluate_quality(
        self, data: DataMatrix, adj: np.ndarray, n: int, score_fn: Any,
    ) -> float:
        """Evaluate the quality of a DAG using BIC scoring."""
        n_samples = data.shape[0]
        log_n = np.log(max(n_samples, 1))
        total = 0.0
        for j in range(n):
            parents = sorted(int(i) for i in np.nonzero(adj[:, j])[0])
            y = data[:, j]
            k = len(parents)
            if k > 0:
                X = data[:, parents]
                try:
                    coef, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
                    residuals = y - X @ coef
                except np.linalg.LinAlgError:
                    residuals = y - y.mean()
            else:
                residuals = y - y.mean()
            rss = max(float(np.sum(residuals ** 2)), 1e-12)
            total += -n_samples / 2.0 * np.log(rss / n_samples) - (k + 1) * log_n / 2.0
        return total

    @staticmethod
    def _compute_descriptor(adj: np.ndarray, n: int) -> np.ndarray:
        """Compute a behavioral descriptor from a DAG."""
        out_degree_std = float(np.sum(adj, axis=1).std()) if n > 1 else 0.0
        in_degree_std = float(np.sum(adj, axis=0).std()) if n > 1 else 0.0
        return np.array([out_degree_std, in_degree_std], dtype=np.float64)
