"""Structural accuracy metrics for causal discovery evaluation.

Provides:
  - SHD: Structural Hamming Distance
  - F1: F1 score for directed edges (with precision and recall)
  - skeleton_f1: F1 for skeleton (undirected) recovery
  - orientation_f1: F1 for edge orientation correctness
  - sid: Structural Intervention Distance (approximation)
  - edge_precision / edge_recall
  - mec_recall: fraction of archive DAGs in the true MEC
"""
from __future__ import annotations

from collections import deque
from typing import TYPE_CHECKING, List, Optional, Set, Tuple

import numpy as np

from causal_qd.types import AdjacencyMatrix

if TYPE_CHECKING:
    from causal_qd.archive.archive_base import Archive
    from causal_qd.core.dag import DAG


class SHD:
    """Structural Hamming Distance metric.

    Counts the minimum number of edge additions, deletions, and reversals
    needed to transform the predicted graph into the true graph.
    """

    @staticmethod
    def compute(predicted: AdjacencyMatrix, true: AdjacencyMatrix) -> int:
        """Compute the SHD between *predicted* and *true* adjacency matrices.

        For directed graphs, this counts:
        - Extra edges (FP)
        - Missing edges (FN)
        - Reversed edges (counted as 1, not 2)

        Parameters
        ----------
        predicted :
            Predicted adjacency matrix.
        true :
            Ground-truth adjacency matrix.

        Returns
        -------
        int
            Non-negative SHD value.
        """
        pred = np.asarray(predicted, dtype=np.int8)
        true_ = np.asarray(true, dtype=np.int8)
        n = pred.shape[0]

        shd = 0
        counted = set()

        for i in range(n):
            for j in range(n):
                if (i, j) in counted:
                    continue

                p_ij = bool(pred[i, j])
                p_ji = bool(pred[j, i])
                t_ij = bool(true_[i, j])
                t_ji = bool(true_[j, i])

                if p_ij == t_ij and p_ji == t_ji:
                    continue

                # Check for reversal: pred has i→j but true has j→i
                if p_ij and not p_ji and t_ji and not t_ij:
                    shd += 1
                    counted.add((i, j))
                    counted.add((j, i))
                elif p_ji and not p_ij and t_ij and not t_ji:
                    shd += 1
                    counted.add((i, j))
                    counted.add((j, i))
                else:
                    # Addition or deletion
                    if p_ij != t_ij:
                        shd += 1
                        counted.add((i, j))

        return shd

    @staticmethod
    def compute_simple(predicted: AdjacencyMatrix, true: AdjacencyMatrix) -> int:
        """Simple entry-wise SHD (no reversal handling)."""
        return int(np.sum(np.asarray(predicted) != np.asarray(true)))


class F1:
    """F1-score metric for directed edges, with precision and recall."""

    def __init__(self) -> None:
        self._tp: int = 0
        self._fp: int = 0
        self._fn: int = 0

    def _compute_counts(
        self,
        predicted: AdjacencyMatrix,
        true: AdjacencyMatrix,
    ) -> None:
        """Compute TP, FP, FN counts."""
        pred_flat = np.asarray(predicted).ravel().astype(bool)
        true_flat = np.asarray(true).ravel().astype(bool)
        self._tp = int(np.sum(pred_flat & true_flat))
        self._fp = int(np.sum(pred_flat & ~true_flat))
        self._fn = int(np.sum(~pred_flat & true_flat))

    def compute(
        self,
        predicted: AdjacencyMatrix,
        true: AdjacencyMatrix,
    ) -> float:
        """Compute the F1-score between *predicted* and *true*.

        Returns
        -------
        float
            F1 score in ``[0, 1]``.
        """
        self._compute_counts(predicted, true)
        denom = 2 * self._tp + self._fp + self._fn
        if denom == 0:
            return 1.0
        return 2 * self._tp / denom

    def precision(self) -> float:
        """Precision of the most recent :meth:`compute` call."""
        denom = self._tp + self._fp
        return self._tp / denom if denom > 0 else 1.0

    def recall(self) -> float:
        """Recall of the most recent :meth:`compute` call."""
        denom = self._tp + self._fn
        return self._tp / denom if denom > 0 else 1.0


# ======================================================================
# Skeleton F1
# ======================================================================


def skeleton_f1(predicted: AdjacencyMatrix, true: AdjacencyMatrix) -> float:
    """F1 score for undirected skeleton recovery.

    Ignores edge orientation: only checks whether an edge exists
    between each pair of nodes (in either direction).

    Parameters
    ----------
    predicted :
        Predicted adjacency matrix.
    true :
        True adjacency matrix.

    Returns
    -------
    float
        Skeleton F1 score in ``[0, 1]``.
    """
    pred = np.asarray(predicted, dtype=np.int8)
    true_ = np.asarray(true, dtype=np.int8)

    pred_skel = (pred | pred.T).astype(bool)
    true_skel = (true_ | true_.T).astype(bool)

    # Only upper triangle (avoid double counting)
    n = pred.shape[0]
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)

    pred_upper = pred_skel[mask]
    true_upper = true_skel[mask]

    tp = int(np.sum(pred_upper & true_upper))
    fp = int(np.sum(pred_upper & ~true_upper))
    fn = int(np.sum(~pred_upper & true_upper))

    denom = 2 * tp + fp + fn
    return 2 * tp / denom if denom > 0 else 1.0


# ======================================================================
# Orientation F1
# ======================================================================


def orientation_f1(predicted: AdjacencyMatrix, true: AdjacencyMatrix) -> float:
    """F1 score for edge orientation correctness.

    Only considers edges that exist in both the predicted and true
    skeletons.  An orientation is correct if the directed edge matches
    exactly.

    Parameters
    ----------
    predicted :
        Predicted adjacency matrix.
    true :
        True adjacency matrix.

    Returns
    -------
    float
        Orientation F1 in ``[0, 1]``.
    """
    pred = np.asarray(predicted, dtype=np.int8)
    true_ = np.asarray(true, dtype=np.int8)
    n = pred.shape[0]

    tp = fp = fn = 0
    seen = set()

    for i in range(n):
        for j in range(n):
            if i == j or (i, j) in seen:
                continue
            # Check if edge exists in both skeletons
            pred_has = bool(pred[i, j]) or bool(pred[j, i])
            true_has = bool(true_[i, j]) or bool(true_[j, i])

            if pred_has and true_has:
                # Check orientation match
                if pred[i, j] == true_[i, j] and pred[j, i] == true_[j, i]:
                    tp += 1
                else:
                    fp += 1
                    fn += 1
                seen.add((i, j))
                seen.add((j, i))

    denom = 2 * tp + fp + fn
    return 2 * tp / denom if denom > 0 else 1.0


# ======================================================================
# Edge precision and recall (standalone functions)
# ======================================================================


def edge_precision(predicted: AdjacencyMatrix, true: AdjacencyMatrix) -> float:
    """Precision: fraction of predicted edges that are correct."""
    pred = np.asarray(predicted).ravel().astype(bool)
    true_ = np.asarray(true).ravel().astype(bool)
    tp = int(np.sum(pred & true_))
    fp = int(np.sum(pred & ~true_))
    return tp / (tp + fp) if (tp + fp) > 0 else 1.0


def edge_recall(predicted: AdjacencyMatrix, true: AdjacencyMatrix) -> float:
    """Recall: fraction of true edges that are predicted."""
    pred = np.asarray(predicted).ravel().astype(bool)
    true_ = np.asarray(true).ravel().astype(bool)
    tp = int(np.sum(pred & true_))
    fn = int(np.sum(~pred & true_))
    return tp / (tp + fn) if (tp + fn) > 0 else 1.0


# ======================================================================
# Structural Intervention Distance (SID) — approximation
# ======================================================================


def sid(predicted: AdjacencyMatrix, true: AdjacencyMatrix) -> int:
    """Structural Intervention Distance (approximation).

    Counts the number of ordered pairs (i, j) where the set of nodes
    reachable from j after intervening on i differs between the
    predicted and true DAGs.

    This is an approximation that compares the causal ancestors of
    each node between the two graphs.

    Parameters
    ----------
    predicted :
        Predicted adjacency matrix.
    true :
        True adjacency matrix.

    Returns
    -------
    int
        SID value (lower is better).
    """
    pred = np.asarray(predicted, dtype=np.int8)
    true_ = np.asarray(true, dtype=np.int8)
    n = pred.shape[0]

    def _ancestors(adj: np.ndarray, node: int) -> Set[int]:
        """BFS to find all ancestors of a node."""
        visited: Set[int] = set()
        queue = deque([node])
        while queue:
            cur = queue.popleft()
            parents = [i for i in range(n) if adj[i, cur] and i not in visited]
            for p in parents:
                visited.add(p)
                queue.append(p)
        return visited

    count = 0
    for j in range(n):
        pred_anc = _ancestors(pred, j)
        true_anc = _ancestors(true_, j)
        # Count interventional differences
        for i in range(n):
            if i == j:
                continue
            pred_effect = i in pred_anc or pred[i, j]
            true_effect = i in true_anc or true_[i, j]
            if pred_effect != true_effect:
                count += 1

    return count


# ======================================================================
# All-in-one evaluation
# ======================================================================


def evaluate_all(
    predicted: AdjacencyMatrix,
    true: AdjacencyMatrix,
) -> dict:
    """Compute all structural metrics at once.

    Parameters
    ----------
    predicted :
        Predicted adjacency matrix.
    true :
        True adjacency matrix.

    Returns
    -------
    dict
        Dictionary with keys: shd, f1, precision, recall,
        skeleton_f1, orientation_f1, sid.
    """
    f1_metric = F1()
    f1_val = f1_metric.compute(predicted, true)

    return {
        "shd": SHD.compute(predicted, true),
        "f1": f1_val,
        "precision": f1_metric.precision(),
        "recall": f1_metric.recall(),
        "skeleton_f1": skeleton_f1(predicted, true),
        "orientation_f1": orientation_f1(predicted, true),
        "sid": sid(predicted, true),
    }
