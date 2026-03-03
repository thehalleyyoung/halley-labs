"""Pooled data baseline (BL2) – ignore context heterogeneity.

Pools all context-specific datasets into a single dataset and runs
a standard structure learner.  By construction, every edge is classified
as INVARIANT because only a single DAG is produced.  This provides the
lower bound on detecting plasticity when context is entirely ignored.

References
----------
Spirtes, Glymour & Scheines (2000).  *Causation, Prediction, and Search*.
"""

from __future__ import annotations

import itertools
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
from numpy.typing import NDArray
from scipy import stats

from cpa.core.types import PlasticityClass

# Re-use PC helpers from ind_phc to avoid duplication
from cpa.baselines.ind_phc import (
    _pc_algorithm,
    _structural_hamming_distance,
    _collect_edges,
    _partial_correlation,
    _fisher_z_test,
)


# -------------------------------------------------------------------
# Pooling utilities
# -------------------------------------------------------------------


def _pool_datasets(datasets: Dict[str, NDArray]) -> NDArray:
    """Concatenate all context datasets into a single matrix.

    Parameters
    ----------
    datasets : Dict[str, NDArray]
        ``{context_label: (n_i, p)}`` arrays.

    Returns
    -------
    NDArray, shape (sum(n_i), p)
    """
    arrays = [datasets[k] for k in sorted(datasets.keys())]
    return np.vstack(arrays)


def _compute_bic(data: NDArray, adj: NDArray) -> float:
    """Compute the BIC score for a DAG adjacency on *data*.

    Uses linear-Gaussian local score: for each node, regress on parents.
    BIC = -2 * log-likelihood + k * log(n)
    """
    n, p = data.shape
    total_bic = 0.0
    for j in range(p):
        parents = np.where(adj[:, j] != 0)[0]
        k = len(parents) + 1  # number of parameters (coefficients + variance)
        if len(parents) == 0:
            residuals = data[:, j] - np.mean(data[:, j])
        else:
            X = data[:, parents]
            X_aug = np.column_stack([X, np.ones(n)])
            beta, _, _, _ = np.linalg.lstsq(X_aug, data[:, j], rcond=None)
            residuals = data[:, j] - X_aug @ beta
        rss = float(np.sum(residuals ** 2))
        sigma2 = rss / max(n, 1)
        if sigma2 < 1e-15:
            sigma2 = 1e-15
        ll = -0.5 * n * (np.log(2 * np.pi * sigma2) + 1.0)
        total_bic += -2.0 * ll + k * np.log(n)
    return total_bic


def _estimate_coefficients(data: NDArray, adj: NDArray) -> NDArray:
    """Estimate linear regression coefficients given a DAG structure.

    Returns a coefficient matrix B where B[i, j] is the coefficient
    of variable i in the regression for variable j.
    """
    n, p = data.shape
    B = np.zeros((p, p), dtype=np.float64)
    for j in range(p):
        parents = np.where(adj[:, j] != 0)[0]
        if len(parents) == 0:
            continue
        X = data[:, parents]
        X_aug = np.column_stack([X, np.ones(n)])
        beta, _, _, _ = np.linalg.lstsq(X_aug, data[:, j], rcond=None)
        for idx, pa in enumerate(parents):
            B[pa, j] = beta[idx]
    return B


def _estimate_residual_variances(
    data: NDArray, adj: NDArray, B: NDArray,
) -> NDArray:
    """Estimate residual variances for each variable."""
    n, p = data.shape
    variances = np.zeros(p, dtype=np.float64)
    for j in range(p):
        parents = np.where(adj[:, j] != 0)[0]
        if len(parents) == 0:
            variances[j] = float(np.var(data[:, j], ddof=1))
        else:
            predicted = data[:, parents] @ B[parents, j]
            residuals = data[:, j] - predicted
            variances[j] = float(np.var(residuals, ddof=1))
    return variances


# -------------------------------------------------------------------
# Main class
# -------------------------------------------------------------------


class PooledBaseline:
    """Pooled-data baseline that ignores context labels (BL2).

    Pools all datasets, learns a single DAG, and classifies every edge
    as INVARIANT.  Optionally compares the pooled DAG against per-context
    DAGs for diagnostic purposes.

    Parameters
    ----------
    learner : str
        Structure learning algorithm (``"pc"``).
    significance_level : float
        Alpha for conditional independence tests.
    """

    def __init__(
        self,
        learner: str = "pc",
        significance_level: float = 0.05,
    ) -> None:
        if learner not in ("pc",):
            raise ValueError(f"Unsupported learner: {learner!r}")
        self._learner_name = learner
        self._alpha = significance_level
        self._dag: Optional[NDArray] = None
        self._coefficients: Optional[NDArray] = None
        self._residual_vars: Optional[NDArray] = None
        self._bic: Optional[float] = None
        self._n_vars: int = 0
        self._pooled_data: Optional[NDArray] = None
        self._datasets: Dict[str, NDArray] = {}
        self._fitted: bool = False

    # ---------------------------------------------------------------
    # Public API
    # ---------------------------------------------------------------

    def fit(
        self,
        datasets: Dict[str, NDArray],
        context_labels: Optional[List[str]] = None,
    ) -> "PooledBaseline":
        """Pool all datasets and learn a single DAG.

        Parameters
        ----------
        datasets : Dict[str, NDArray]
            ``{context_label: (n_samples, n_vars)}`` arrays.
        context_labels : list of str, optional
            Ignored; retained for API compatibility.

        Returns
        -------
        self
        """
        if not datasets:
            raise ValueError("datasets must be non-empty")
        if isinstance(datasets, list):
            datasets = {f"ctx_{i}": d for i, d in enumerate(datasets)}
        first = next(iter(datasets.values()))
        self._n_vars = first.shape[1]
        for key, data in datasets.items():
            if data.shape[1] != self._n_vars:
                raise ValueError(
                    f"Context {key!r} has {data.shape[1]} variables, "
                    f"expected {self._n_vars}"
                )
        self._datasets = dict(datasets)
        self._pooled_data = self._pool_data(datasets)
        self._dag = self._learn_single_dag(self._pooled_data)
        self._coefficients = _estimate_coefficients(
            self._pooled_data, self._dag,
        )
        self._residual_vars = _estimate_residual_variances(
            self._pooled_data, self._dag, self._coefficients,
        )
        self._bic = _compute_bic(self._pooled_data, self._dag)
        self._fitted = True
        return self

    def predict_plasticity(self) -> Dict[Tuple[int, int], PlasticityClass]:
        """Return all-invariant plasticity classifications.

        Since a single DAG is learned from pooled data, every edge is
        classified as INVARIANT by construction.
        """
        if not self._fitted:
            raise RuntimeError("Call fit() first.")
        return self._classify_all_invariant()

    def pooled_dag(self) -> NDArray:
        """Return the learned DAG adjacency matrix."""
        if not self._fitted:
            raise RuntimeError("Call fit() first.")
        assert self._dag is not None
        return self._dag.copy()

    def coefficient_matrix(self) -> NDArray:
        """Return estimated regression coefficients on pooled data."""
        if not self._fitted:
            raise RuntimeError("Call fit() first.")
        assert self._coefficients is not None
        return self._coefficients.copy()

    def residual_variances(self) -> NDArray:
        """Return estimated residual variances on pooled data."""
        if not self._fitted:
            raise RuntimeError("Call fit() first.")
        assert self._residual_vars is not None
        return self._residual_vars.copy()

    def bic_score(self) -> float:
        """Return the BIC score of the pooled DAG."""
        if not self._fitted:
            raise RuntimeError("Call fit() first.")
        assert self._bic is not None
        return self._bic

    def compare_to_contextual(
        self,
        context_dags: Dict[str, NDArray],
    ) -> Dict[str, float]:
        """Compare pooled DAG against per-context DAGs via SHD.

        Parameters
        ----------
        context_dags : Dict[str, NDArray]
            Per-context DAG adjacency matrices.

        Returns
        -------
        Dict[str, float]
            SHD per context.
        """
        if not self._fitted:
            raise RuntimeError("Call fit() first.")
        assert self._dag is not None
        results: Dict[str, float] = {}
        for ctx, dag in context_dags.items():
            results[ctx] = float(
                _structural_hamming_distance(self._dag, dag)
            )
        return results

    def summary(self) -> Dict[str, Any]:
        """Return summary statistics of the pooled model."""
        if not self._fitted:
            raise RuntimeError("Call fit() first.")
        assert self._dag is not None
        n_edges = int(np.sum(self._dag != 0))
        return {
            "n_variables": self._n_vars,
            "n_edges": n_edges,
            "density": n_edges / max(self._n_vars * (self._n_vars - 1), 1),
            "bic": self._bic,
            "n_pooled_samples": (
                self._pooled_data.shape[0] if self._pooled_data is not None
                else 0
            ),
        }

    # ---------------------------------------------------------------
    # Internal methods
    # ---------------------------------------------------------------

    def _pool_data(self, datasets: Dict[str, NDArray]) -> NDArray:
        """Concatenate all context datasets."""
        return _pool_datasets(datasets)

    def _learn_single_dag(self, pooled_data: NDArray) -> NDArray:
        """Learn one DAG from the pooled data."""
        return _pc_algorithm(pooled_data, self._alpha)

    def _classify_all_invariant(
        self,
    ) -> Dict[Tuple[int, int], PlasticityClass]:
        """Classify every edge as invariant (by construction)."""
        assert self._dag is not None
        edges = _collect_edges(self._dag)
        classifications: Dict[Tuple[int, int], PlasticityClass] = {}
        for i, j in edges:
            if (i, j) not in classifications and (j, i) not in classifications:
                classifications[(i, j)] = PlasticityClass.INVARIANT
        return classifications
