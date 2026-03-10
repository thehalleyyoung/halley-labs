"""
Multiplicity correction for families of CI tests.

Implements Benjamini–Yekutieli (BY) FDR control, which is valid under
arbitrary dependence, and ancestral-set pruning to reduce the testing
burden by restricting attention to ancestors of the treatment/outcome pair.

Also provides Benjamini–Hochberg, Bonferroni, Holm–Bonferroni, and a
selective inference framework for adjusted p-value computation.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Sequence

import numpy as np

from causalcert.types import AdjacencyMatrix, CITestResult, NodeId, NodeSet


# ---------------------------------------------------------------------------
# BFS / DFS helpers for ancestral sets
# ---------------------------------------------------------------------------


def _ancestors(adj: AdjacencyMatrix, node: NodeId) -> set[NodeId]:
    """Return the set of ancestors of *node* (including *node*).

    An ancestor of *node* is any node from which there exists a directed
    path to *node* in the DAG represented by *adj*.

    Parameters
    ----------
    adj : AdjacencyMatrix
        DAG adjacency matrix (``adj[i,j]=1`` iff ``i → j``).
    node : NodeId
        Target node.

    Returns
    -------
    set[NodeId]
        Ancestor set including *node* itself.
    """
    p = adj.shape[0]
    visited: set[NodeId] = set()
    stack = [node]
    while stack:
        v = stack.pop()
        if v in visited:
            continue
        visited.add(v)
        # Parents of v: all i s.t. adj[i, v] == 1
        for i in range(p):
            if adj[i, v] and i not in visited:
                stack.append(i)
    return visited


def _descendants(adj: AdjacencyMatrix, node: NodeId) -> set[NodeId]:
    """Return the set of descendants of *node* (including *node*).

    Parameters
    ----------
    adj : AdjacencyMatrix
        DAG adjacency matrix.
    node : NodeId
        Source node.

    Returns
    -------
    set[NodeId]
    """
    p = adj.shape[0]
    visited: set[NodeId] = set()
    stack = [node]
    while stack:
        v = stack.pop()
        if v in visited:
            continue
        visited.add(v)
        for j in range(p):
            if adj[v, j] and j not in visited:
                stack.append(j)
    return visited


# ---------------------------------------------------------------------------
# Adjusted p-value computation
# ---------------------------------------------------------------------------


def _adjusted_pvalues_bh(raw_pvalues: np.ndarray) -> np.ndarray:
    """Compute Benjamini–Hochberg adjusted p-values.

    Parameters
    ----------
    raw_pvalues : np.ndarray
        Raw (unadjusted) p-values, shape ``(m,)``.

    Returns
    -------
    np.ndarray
        Adjusted p-values.
    """
    m = len(raw_pvalues)
    if m == 0:
        return np.array([], dtype=np.float64)

    order = np.argsort(raw_pvalues)
    sorted_p = raw_pvalues[order]

    # Adjusted p_values: p_adj[i] = min over j>=i of (m / (j+1)) * p_sorted[j]
    adjusted = np.empty(m, dtype=np.float64)
    adjusted[-1] = sorted_p[-1]
    for i in range(m - 2, -1, -1):
        adjusted[i] = min(adjusted[i + 1], m / (i + 1) * sorted_p[i])
    adjusted = np.minimum(adjusted, 1.0)

    # Unsort
    result = np.empty(m, dtype=np.float64)
    result[order] = adjusted
    return result


def _adjusted_pvalues_by(raw_pvalues: np.ndarray) -> np.ndarray:
    """Compute Benjamini–Yekutieli adjusted p-values.

    The BY correction multiplies the BH threshold by the harmonic number
    ``c(m) = sum_{k=1}^{m} 1/k``, making it valid under arbitrary dependence.

    Parameters
    ----------
    raw_pvalues : np.ndarray
        Raw (unadjusted) p-values.

    Returns
    -------
    np.ndarray
        BY-adjusted p-values.
    """
    m = len(raw_pvalues)
    if m == 0:
        return np.array([], dtype=np.float64)

    harmonic = sum(1.0 / k for k in range(1, m + 1))

    order = np.argsort(raw_pvalues)
    sorted_p = raw_pvalues[order]

    adjusted = np.empty(m, dtype=np.float64)
    adjusted[-1] = min(sorted_p[-1] * harmonic, 1.0)
    for i in range(m - 2, -1, -1):
        adjusted[i] = min(
            adjusted[i + 1],
            m * harmonic / (i + 1) * sorted_p[i],
        )
    adjusted = np.minimum(adjusted, 1.0)

    result = np.empty(m, dtype=np.float64)
    result[order] = adjusted
    return result


def _adjusted_pvalues_bonferroni(raw_pvalues: np.ndarray) -> np.ndarray:
    """Compute Bonferroni-adjusted p-values.

    Parameters
    ----------
    raw_pvalues : np.ndarray
        Raw p-values.

    Returns
    -------
    np.ndarray
        Bonferroni-adjusted p-values (= min(m * p, 1)).
    """
    m = len(raw_pvalues)
    return np.minimum(raw_pvalues * m, 1.0)


def _adjusted_pvalues_holm(raw_pvalues: np.ndarray) -> np.ndarray:
    """Compute Holm–Bonferroni adjusted p-values (step-down).

    Parameters
    ----------
    raw_pvalues : np.ndarray
        Raw p-values.

    Returns
    -------
    np.ndarray
        Holm-adjusted p-values.
    """
    m = len(raw_pvalues)
    if m == 0:
        return np.array([], dtype=np.float64)

    order = np.argsort(raw_pvalues)
    sorted_p = raw_pvalues[order]

    adjusted = np.empty(m, dtype=np.float64)
    adjusted[0] = min(sorted_p[0] * m, 1.0)
    for i in range(1, m):
        adjusted[i] = max(adjusted[i - 1], min(sorted_p[i] * (m - i), 1.0))

    result = np.empty(m, dtype=np.float64)
    result[order] = adjusted
    return result


# ---------------------------------------------------------------------------
# Benjamini–Yekutieli FDR control
# ---------------------------------------------------------------------------


class BenjaminiYekutieli:
    """Benjamini–Yekutieli FDR control procedure.

    Valid under arbitrary dependence between the test statistics, unlike the
    simpler Benjamini–Hochberg procedure which requires positive dependence.

    Parameters
    ----------
    alpha : float
        Target FDR level.
    """

    def __init__(self, alpha: float = 0.05) -> None:
        self.alpha = alpha

    def adjust(self, results: Sequence[CITestResult]) -> list[CITestResult]:
        """Apply BY correction and return updated results with adjusted rejections.

        Parameters
        ----------
        results : Sequence[CITestResult]
            Raw (unadjusted) CI test results.

        Returns
        -------
        list[CITestResult]
            New results with ``reject`` fields updated according to BY-adjusted
            thresholds.
        """
        if len(results) == 0:
            return []

        m = len(results)
        raw_p = np.array([r.p_value for r in results], dtype=np.float64)
        adjusted_p = _adjusted_pvalues_by(raw_p)

        out: list[CITestResult] = []
        for i, r in enumerate(results):
            out.append(
                CITestResult(
                    x=r.x,
                    y=r.y,
                    conditioning_set=r.conditioning_set,
                    statistic=r.statistic,
                    p_value=float(adjusted_p[i]),
                    method=r.method,
                    reject=bool(adjusted_p[i] < self.alpha),
                    alpha=self.alpha,
                )
            )
        return out

    @staticmethod
    def _by_threshold(alpha: float, m: int) -> float:
        """Compute the BY threshold for *m* tests at level *alpha*.

        Parameters
        ----------
        alpha : float
            Target FDR level.
        m : int
            Number of tests.

        Returns
        -------
        float
            Adjusted significance threshold.
        """
        harmonic = sum(1.0 / k for k in range(1, m + 1))
        return alpha / (m * harmonic)


class BenjaminiHochberg:
    """Benjamini–Hochberg FDR control (valid under PRDS).

    Parameters
    ----------
    alpha : float
        Target FDR level.
    """

    def __init__(self, alpha: float = 0.05) -> None:
        self.alpha = alpha

    def adjust(self, results: Sequence[CITestResult]) -> list[CITestResult]:
        """Apply BH correction.

        Parameters
        ----------
        results : Sequence[CITestResult]
            Raw CI test results.

        Returns
        -------
        list[CITestResult]
            Adjusted results.
        """
        if len(results) == 0:
            return []

        raw_p = np.array([r.p_value for r in results], dtype=np.float64)
        adjusted_p = _adjusted_pvalues_bh(raw_p)

        out: list[CITestResult] = []
        for i, r in enumerate(results):
            out.append(
                CITestResult(
                    x=r.x, y=r.y,
                    conditioning_set=r.conditioning_set,
                    statistic=r.statistic,
                    p_value=float(adjusted_p[i]),
                    method=r.method,
                    reject=bool(adjusted_p[i] < self.alpha),
                    alpha=self.alpha,
                )
            )
        return out


class Bonferroni:
    """Bonferroni correction.

    Parameters
    ----------
    alpha : float
        Target FWER level.
    """

    def __init__(self, alpha: float = 0.05) -> None:
        self.alpha = alpha

    def adjust(self, results: Sequence[CITestResult]) -> list[CITestResult]:
        """Apply Bonferroni correction.

        Parameters
        ----------
        results : Sequence[CITestResult]
            Raw CI test results.

        Returns
        -------
        list[CITestResult]
            Adjusted results.
        """
        if len(results) == 0:
            return []

        raw_p = np.array([r.p_value for r in results], dtype=np.float64)
        adjusted_p = _adjusted_pvalues_bonferroni(raw_p)

        out: list[CITestResult] = []
        for i, r in enumerate(results):
            out.append(
                CITestResult(
                    x=r.x, y=r.y,
                    conditioning_set=r.conditioning_set,
                    statistic=r.statistic,
                    p_value=float(adjusted_p[i]),
                    method=r.method,
                    reject=bool(adjusted_p[i] < self.alpha),
                    alpha=self.alpha,
                )
            )
        return out


class HolmBonferroni:
    """Holm–Bonferroni step-down procedure.

    Parameters
    ----------
    alpha : float
        Target FWER level.
    """

    def __init__(self, alpha: float = 0.05) -> None:
        self.alpha = alpha

    def adjust(self, results: Sequence[CITestResult]) -> list[CITestResult]:
        """Apply Holm–Bonferroni step-down correction.

        Parameters
        ----------
        results : Sequence[CITestResult]
            Raw CI test results.

        Returns
        -------
        list[CITestResult]
            Adjusted results.
        """
        if len(results) == 0:
            return []

        raw_p = np.array([r.p_value for r in results], dtype=np.float64)
        adjusted_p = _adjusted_pvalues_holm(raw_p)

        out: list[CITestResult] = []
        for i, r in enumerate(results):
            out.append(
                CITestResult(
                    x=r.x, y=r.y,
                    conditioning_set=r.conditioning_set,
                    statistic=r.statistic,
                    p_value=float(adjusted_p[i]),
                    method=r.method,
                    reject=bool(adjusted_p[i] < self.alpha),
                    alpha=self.alpha,
                )
            )
        return out


# ---------------------------------------------------------------------------
# Ancestral set pruning
# ---------------------------------------------------------------------------


def ancestral_pruning(
    adj: AdjacencyMatrix,
    treatment: NodeId,
    outcome: NodeId,
    all_triples: Sequence[tuple[NodeId, NodeId, NodeSet]],
) -> list[tuple[NodeId, NodeId, NodeSet]]:
    """Prune CI test triples to those involving ancestors of treatment/outcome.

    By restricting to the ancestral set of ``{treatment, outcome}``, we
    reduce the multiplicity burden without sacrificing soundness.

    A triple ``(x, y, S)`` is kept if and only if *x*, *y*, and every
    member of *S* are in the ancestral closure of ``{treatment, outcome}``.

    Parameters
    ----------
    adj : AdjacencyMatrix
        DAG adjacency matrix.
    treatment, outcome : NodeId
        Treatment and outcome nodes.
    all_triples : Sequence[tuple[NodeId, NodeId, NodeSet]]
        Full list of CI test triples.

    Returns
    -------
    list[tuple[NodeId, NodeId, NodeSet]]
        Pruned triples involving only ancestral nodes.
    """
    anc_set = _ancestors(adj, treatment) | _ancestors(adj, outcome)

    pruned: list[tuple[NodeId, NodeId, NodeSet]] = []
    for x, y, s in all_triples:
        if x in anc_set and y in anc_set and all(v in anc_set for v in s):
            pruned.append((x, y, s))

    return pruned


# ---------------------------------------------------------------------------
# Selective inference framework
# ---------------------------------------------------------------------------


class SelectiveInference:
    """Framework for selective inference on CI tests.

    When tests are selected adaptively (e.g. based on preliminary data
    exploration), standard multiplicity corrections can be anti-conservative.
    This class provides a simple conditioning-based correction.

    Parameters
    ----------
    alpha : float
        Target selective type-I error rate.
    selection_fraction : float
        Fraction of data used for selection (the remaining fraction is
        used for inference).  Must be in ``(0, 1)``.
    """

    def __init__(
        self,
        alpha: float = 0.05,
        selection_fraction: float = 0.5,
    ) -> None:
        if not 0 < selection_fraction < 1:
            raise ValueError("selection_fraction must be in (0, 1).")
        self.alpha = alpha
        self.selection_fraction = selection_fraction

    def adjust(self, results: Sequence[CITestResult]) -> list[CITestResult]:
        """Apply selective-inference correction.

        Scales p-values by ``1 / (1 - selection_fraction)`` to account
        for the selection event, then applies Bonferroni on the selected
        subset.

        Parameters
        ----------
        results : Sequence[CITestResult]
            Results from the inference split.

        Returns
        -------
        list[CITestResult]
            Adjusted results.
        """
        if len(results) == 0:
            return []

        m = len(results)
        factor = 1.0 / (1.0 - self.selection_fraction)
        raw_p = np.array([r.p_value for r in results], dtype=np.float64)
        adjusted_p = np.minimum(raw_p * factor * m, 1.0)

        out: list[CITestResult] = []
        for i, r in enumerate(results):
            out.append(
                CITestResult(
                    x=r.x, y=r.y,
                    conditioning_set=r.conditioning_set,
                    statistic=r.statistic,
                    p_value=float(adjusted_p[i]),
                    method=r.method,
                    reject=bool(adjusted_p[i] < self.alpha),
                    alpha=self.alpha,
                )
            )
        return out
