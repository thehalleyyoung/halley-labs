"""GES-based multi-context baseline (BL6).

Implements the Greedy Equivalence Search (GES) algorithm of Chickering
(2002).  GES searches the space of CPDAGs via two phases:

1. **Forward phase** – greedily add single-edge insertions that
   maximally improve the BIC score.
2. **Backward phase** – greedily remove single edges that improve
   the BIC score.

Supports per-context learning, pooled learning, and a merged-graph
heuristic for multi-context comparison.

References
----------
Chickering, D. M. (2002).  Optimal Structure Identification with
Greedy Search.  *JMLR*, 3, 507-554.
"""

from __future__ import annotations

import itertools
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
from numpy.typing import NDArray
from scipy import stats as sp_stats

from cpa.core.types import PlasticityClass
from cpa.baselines.ind_phc import (
    _collect_edges,
    _structural_hamming_distance,
)


# -------------------------------------------------------------------
# BIC score computation
# -------------------------------------------------------------------


def _local_bic_score(
    data: NDArray, node: int, parents: List[int], penalty: float = 1.0,
) -> float:
    """Compute the local BIC score for *node* given *parents*.

    BIC_local = -n * log(sigma^2_hat) - |parents| * log(n)
    Higher is better.
    """
    n = data.shape[0]
    if n < 2:
        return -np.inf

    y = data[:, node]
    k = len(parents)

    if k == 0:
        sigma2 = float(np.var(y, ddof=1))
    else:
        X = data[:, parents]
        X_aug = np.column_stack([X, np.ones(n)])
        beta, _, _, _ = np.linalg.lstsq(X_aug, y, rcond=None)
        residuals = y - X_aug @ beta
        sigma2 = float(np.sum(residuals ** 2) / max(n - k - 1, 1))

    if sigma2 < 1e-15:
        sigma2 = 1e-15

    # BIC = -n * log(sigma^2) - k * log(n)  (higher = better)
    return -n * np.log(sigma2) - penalty * k * np.log(n)


def _total_bic(
    data: NDArray, adj: NDArray, penalty: float = 1.0,
) -> float:
    """Compute total BIC score for a DAG."""
    p = adj.shape[0]
    total = 0.0
    for j in range(p):
        parents = [i for i in range(p) if adj[i, j] != 0]
        total += _local_bic_score(data, j, parents, penalty)
    return total


def _get_parents(adj: NDArray, node: int) -> List[int]:
    """Get parent list for *node* from adjacency matrix."""
    p = adj.shape[0]
    return [i for i in range(p) if adj[i, node] != 0 and i != node]


# -------------------------------------------------------------------
# GES forward and backward phases
# -------------------------------------------------------------------


def _ges_forward_phase(
    adj: NDArray, data: NDArray, penalty: float = 1.0, max_parents: int = 5,
) -> NDArray:
    """GES forward phase: greedily add edges that improve BIC.

    Iterates until no single edge insertion improves the score.
    """
    p = adj.shape[0]
    current = adj.copy()
    current_score = _total_bic(data, current, penalty)

    improved = True
    while improved:
        improved = False
        best_gain = 0.0
        best_edge: Optional[Tuple[int, int]] = None

        for i in range(p):
            for j in range(p):
                if i == j or current[i, j] != 0:
                    continue
                # Check max parents constraint
                parents_j = _get_parents(current, j)
                if len(parents_j) >= max_parents:
                    continue
                # Check acyclicity: would adding i -> j create a cycle?
                if _would_create_cycle(current, i, j):
                    continue

                # Score gain from adding i -> j
                old_score_j = _local_bic_score(
                    data, j, parents_j, penalty,
                )
                new_parents_j = parents_j + [i]
                new_score_j = _local_bic_score(
                    data, j, new_parents_j, penalty,
                )
                gain = new_score_j - old_score_j
                if gain > best_gain:
                    best_gain = gain
                    best_edge = (i, j)

        if best_edge is not None and best_gain > 0:
            i, j = best_edge
            current[i, j] = 1.0
            current_score += best_gain
            improved = True

    return current


def _ges_backward_phase(
    adj: NDArray, data: NDArray, penalty: float = 1.0,
) -> NDArray:
    """GES backward phase: greedily remove edges that improve BIC.

    Iterates until no single edge removal improves the score.
    """
    p = adj.shape[0]
    current = adj.copy()

    improved = True
    while improved:
        improved = False
        best_gain = 0.0
        best_edge: Optional[Tuple[int, int]] = None

        for i in range(p):
            for j in range(p):
                if i == j or current[i, j] == 0:
                    continue

                parents_j = _get_parents(current, j)
                old_score_j = _local_bic_score(
                    data, j, parents_j, penalty,
                )
                new_parents_j = [pa for pa in parents_j if pa != i]
                new_score_j = _local_bic_score(
                    data, j, new_parents_j, penalty,
                )
                gain = new_score_j - old_score_j
                if gain > best_gain:
                    best_gain = gain
                    best_edge = (i, j)

        if best_edge is not None and best_gain > 0:
            i, j = best_edge
            current[i, j] = 0.0
            improved = True

    return current


def _would_create_cycle(adj: NDArray, source: int, target: int) -> bool:
    """Check whether adding source -> target would create a cycle.

    Performs BFS from target to see if source is reachable.
    """
    p = adj.shape[0]
    visited = set()
    queue = [target]
    while queue:
        node = queue.pop(0)
        if node == source:
            return True
        if node in visited:
            continue
        visited.add(node)
        for child in range(p):
            if adj[node, child] != 0 and child not in visited:
                queue.append(child)
    return False


def _run_ges(
    data: NDArray, penalty: float = 1.0, max_parents: int = 5,
) -> NDArray:
    """Run the full GES algorithm (forward + backward)."""
    p = data.shape[1]
    empty = np.zeros((p, p), dtype=np.float64)
    after_forward = _ges_forward_phase(empty, data, penalty, max_parents)
    after_backward = _ges_backward_phase(after_forward, data, penalty)
    return after_backward


# -------------------------------------------------------------------
# Multi-context comparison
# -------------------------------------------------------------------


def _compare_ges_results(
    dags: Dict[str, NDArray],
) -> Dict[Tuple[int, int], PlasticityClass]:
    """Compare GES results across contexts for plasticity classification.

    Uses edge presence/absence and weight similarity.
    """
    ctx_keys = sorted(dags.keys())
    n_ctx = len(ctx_keys)

    all_edges: Set[Tuple[int, int]] = set()
    for dag in dags.values():
        all_edges |= _collect_edges(dag)

    classifications: Dict[Tuple[int, int], PlasticityClass] = {}

    for i, j in all_edges:
        if (i, j) in classifications or (j, i) in classifications:
            continue

        present = [dags[k][i, j] != 0 for k in ctx_keys]
        n_present = sum(present)

        if n_present == n_ctx:
            classifications[(i, j)] = PlasticityClass.INVARIANT
        elif n_present == 1:
            classifications[(i, j)] = PlasticityClass.EMERGENT
        elif n_present == 0:
            continue
        else:
            classifications[(i, j)] = PlasticityClass.STRUCTURAL_PLASTIC

    return classifications


# -------------------------------------------------------------------
# Main class
# -------------------------------------------------------------------


class GESBaseline:
    """GES baseline for single- and multi-context structure learning (BL6).

    Runs GES per context and on pooled data.  Compares per-context
    CPDAGs to classify edge plasticity.

    Parameters
    ----------
    score : str
        Scoring function (``"bic"``).
    penalty_weight : float
        Multiplier for the BIC complexity penalty.
    max_parents : int
        Maximum number of parents per node.
    """

    def __init__(
        self,
        score: str = "bic",
        penalty_weight: float = 1.0,
        max_parents: int = 5,
    ) -> None:
        if score not in ("bic",):
            raise ValueError(f"Unsupported score: {score!r}")
        self._score = score
        self._penalty_weight = penalty_weight
        self._max_parents = max_parents
        self._per_ctx: Dict[str, NDArray] = {}
        self._pooled: Optional[NDArray] = None
        self._plasticity: Dict[Tuple[int, int], PlasticityClass] = {}
        self._n_vars: int = 0
        self._datasets: Dict[str, NDArray] = {}
        self._fitted: bool = False

    # ---------------------------------------------------------------
    # Public API
    # ---------------------------------------------------------------

    def fit(
        self,
        datasets: Dict[str, NDArray],
        context_labels: Optional[List[str]] = None,
    ) -> "GESBaseline":
        """Run GES per context and on pooled data.

        Parameters
        ----------
        datasets : Dict[str, NDArray]
            ``{context_label: (n_samples, n_vars)}`` arrays.
        context_labels : list of str, optional

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
        self._datasets = dict(datasets)

        for k, d in datasets.items():
            if d.shape[1] != self._n_vars:
                raise ValueError(
                    f"Context {k!r}: {d.shape[1]} vars, "
                    f"expected {self._n_vars}"
                )

        # Per-context GES
        for ctx_key, data in datasets.items():
            self._per_ctx[ctx_key] = self._run_ges(data)

        # Pooled GES
        pooled = np.vstack([datasets[k] for k in sorted(datasets.keys())])
        self._pooled = self._run_ges(pooled)

        # Classify edges
        self._plasticity = self._compare_results(self._per_ctx)

        self._fitted = True
        return self

    def predict_plasticity(self) -> Dict[Tuple[int, int], PlasticityClass]:
        """Return edge plasticity classifications."""
        if not self._fitted:
            raise RuntimeError("Call fit() first.")
        return dict(self._plasticity)

    def per_context_cpdag(self) -> Dict[str, NDArray]:
        """Return the CPDAG learned per context."""
        if not self._fitted:
            raise RuntimeError("Call fit() first.")
        return {k: v.copy() for k, v in self._per_ctx.items()}

    def pooled_cpdag(self) -> NDArray:
        """Return the CPDAG learned from pooled data."""
        if not self._fitted:
            raise RuntimeError("Call fit() first.")
        assert self._pooled is not None
        return self._pooled.copy()

    def merged_graph(self) -> NDArray:
        """Merge per-context CPDAGs using union heuristic.

        An edge is present in the merged graph if it appears in at
        least half of the per-context CPDAGs.
        """
        if not self._fitted:
            raise RuntimeError("Call fit() first.")
        p = self._n_vars
        count = np.zeros((p, p), dtype=np.float64)
        for dag in self._per_ctx.values():
            count += (dag != 0).astype(np.float64)
        threshold = len(self._per_ctx) / 2.0
        return (count >= threshold).astype(np.float64)

    def forward_phase(self, data: NDArray) -> NDArray:
        """Execute the GES forward phase."""
        p = data.shape[1]
        empty = np.zeros((p, p), dtype=np.float64)
        return _ges_forward_phase(
            empty, data, self._penalty_weight, self._max_parents,
        )

    def backward_phase(self, cpdag: NDArray, data: NDArray) -> NDArray:
        """Execute the GES backward phase."""
        return _ges_backward_phase(cpdag, data, self._penalty_weight)

    def bic_scores(self) -> Dict[str, float]:
        """Return BIC scores for per-context and pooled models."""
        if not self._fitted:
            raise RuntimeError("Call fit() first.")
        scores: Dict[str, float] = {}
        for key, dag in self._per_ctx.items():
            scores[f"ctx_{key}"] = _total_bic(
                self._datasets[key], dag, self._penalty_weight,
            )
        if self._pooled is not None:
            pooled = np.vstack(
                [self._datasets[k] for k in sorted(self._datasets.keys())]
            )
            scores["pooled"] = _total_bic(
                pooled, self._pooled, self._penalty_weight,
            )
        return scores

    # ---------------------------------------------------------------
    # Internal methods
    # ---------------------------------------------------------------

    def _run_ges(self, data: NDArray) -> NDArray:
        """Run full GES on a single dataset."""
        return _run_ges(data, self._penalty_weight, self._max_parents)

    def _compare_results(
        self, dags: Dict[str, NDArray],
    ) -> Dict[Tuple[int, int], PlasticityClass]:
        """Compare GES results across contexts."""
        return _compare_ges_results(dags)
