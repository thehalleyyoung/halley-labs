"""
Faithfulness analysis for causal DAGs.

Detects and quantifies violations of the faithfulness assumption —
situations where conditional independencies in the data do not correspond
to d-separations in the DAG, typically caused by exact or near-exact
path cancellations in the parameterised SCM.

Classes
-------
- :class:`FaithfulnessReport` — Summary of faithfulness assessment.
- :class:`FaithfulnessChecker` — Main analysis engine.
"""

from __future__ import annotations

import itertools
from collections import deque
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy import stats as sp_stats

from causalcert.types import AdjacencyMatrix, NodeId, NodeSet


# ============================================================================
# Data structures
# ============================================================================


@dataclass(frozen=True, slots=True)
class PathCancellation:
    """A detected or near-detected path cancellation.

    Attributes
    ----------
    source : NodeId
        Start of the paths.
    target : NodeId
        End of the paths.
    paths : tuple[tuple[NodeId, ...], ...]
        The individual directed paths.
    path_effects : tuple[float, ...]
        Effect magnitude along each path.
    total_effect : float
        Sum of path effects (near-zero indicates cancellation).
    severity : float
        Cancellation severity in [0, 1]; higher means more severe.
    """

    source: NodeId
    target: NodeId
    paths: tuple[tuple[NodeId, ...], ...] = ()
    path_effects: tuple[float, ...] = ()
    total_effect: float = 0.0
    severity: float = 0.0


@dataclass(frozen=True, slots=True)
class FaithfulnessViolation:
    """A single faithfulness violation instance.

    Attributes
    ----------
    x : NodeId
        First variable.
    y : NodeId
        Second variable.
    conditioning_set : NodeSet
        Conditioning set.
    d_separated : bool
        Whether x and y are d-separated given the conditioning set.
    ci_p_value : float
        p-value from the CI test (low = dependent).
    partial_corr : float
        Partial correlation estimate.
    violation_type : str
        ``"unfaithful"`` (d-connected but CI) or ``"extra_dep"``
        (d-separated but dependent).
    """

    x: NodeId
    y: NodeId
    conditioning_set: NodeSet
    d_separated: bool
    ci_p_value: float
    partial_corr: float
    violation_type: str


@dataclass(frozen=True, slots=True)
class FaithfulnessReport:
    """Summary of faithfulness assessment.

    Attributes
    ----------
    is_faithful : bool
        ``True`` if no violations were detected.
    violations : tuple[FaithfulnessViolation, ...]
        Detected violations.
    cancellations : tuple[PathCancellation, ...]
        Detected path cancellations.
    overall_severity : float
        Aggregate faithfulness violation severity in [0, 1].
    n_tests_performed : int
        Total number of CI tests performed.
    n_violations : int
        Number of violations detected.
    """

    is_faithful: bool
    violations: tuple[FaithfulnessViolation, ...] = ()
    cancellations: tuple[PathCancellation, ...] = ()
    overall_severity: float = 0.0
    n_tests_performed: int = 0
    n_violations: int = 0


# ============================================================================
# Internal helpers
# ============================================================================


def _parents(adj: NDArray, v: int) -> list[int]:
    return [int(p) for p in np.nonzero(adj[:, v])[0]]


def _d_separated(
    adj: NDArray, x: int, y: int, conditioning: frozenset[int]
) -> bool:
    """Check d-separation via the Bayes-Ball algorithm.

    Parameters
    ----------
    adj : NDArray
        DAG adjacency matrix.
    x, y : int
        Query variables.
    conditioning : frozenset[int]
        Conditioning set.

    Returns
    -------
    bool
        ``True`` if x ⊥ y | conditioning in the DAG.
    """
    n = adj.shape[0]
    cond = set(conditioning)

    # Ancestors of conditioning set (needed for v-structures)
    anc_cond: set[int] = set(cond)
    queue = deque(cond)
    while queue:
        v = queue.popleft()
        for p in np.nonzero(adj[:, v])[0]:
            p = int(p)
            if p not in anc_cond:
                anc_cond.add(p)
                queue.append(p)

    # Bayes-Ball: BFS with direction tracking
    # State: (node, direction) where direction is "up" (from child) or "down" (from parent)
    visited: set[tuple[int, str]] = set()
    reachable: set[int] = set()
    start_queue: deque[tuple[int, str]] = deque()

    # Start from x in both directions
    for child in np.nonzero(adj[x])[0]:
        start_queue.append((int(child), "down"))
    for parent in np.nonzero(adj[:, x])[0]:
        start_queue.append((int(parent), "up"))

    while start_queue:
        node, direction = start_queue.popleft()
        if (node, direction) in visited:
            continue
        visited.add((node, direction))

        if node == y:
            return False  # x and y are d-connected

        if direction == "down":
            # Arrived via a parent → child edge
            if node not in cond:
                # Pass through non-conditioned: continue down to children
                for child in np.nonzero(adj[node])[0]:
                    start_queue.append((int(child), "down"))
                # Also pass up to parents
                for parent in np.nonzero(adj[:, node])[0]:
                    start_queue.append((int(parent), "up"))
            else:
                # Node is conditioned: can only go up (collider opening)
                pass
        else:  # direction == "up"
            # Arrived via a child → parent edge
            if node not in cond:
                # Non-conditioned: continue up to parents
                for parent in np.nonzero(adj[:, node])[0]:
                    start_queue.append((int(parent), "up"))
            # If node (or descendant) is conditioned, collider is open: go down
            if node in anc_cond:
                for child in np.nonzero(adj[node])[0]:
                    start_queue.append((int(child), "down"))

    return True


def _partial_correlation(
    data: NDArray[np.float64], x: int, y: int, z_set: frozenset[int]
) -> tuple[float, float]:
    """Compute partial correlation and Fisher-z p-value.

    Returns
    -------
    tuple[float, float]
        (partial_corr, p_value)
    """
    n, p = data.shape
    z_list = sorted(z_set)

    if not z_list:
        r = np.corrcoef(data[:, x], data[:, y])[0, 1]
    else:
        # Residualize x and y on z
        Z = data[:, z_list]
        # Add intercept
        Z_aug = np.column_stack([np.ones(n), Z])
        try:
            proj = Z_aug @ np.linalg.lstsq(Z_aug, data[:, [x, y]], rcond=None)[0]
        except np.linalg.LinAlgError:
            return 0.0, 1.0
        resid = data[:, [x, y]] - proj
        if resid.std(axis=0).min() < 1e-12:
            return 0.0, 1.0
        r = np.corrcoef(resid[:, 0], resid[:, 1])[0, 1]

    r = np.clip(r, -0.9999, 0.9999)

    # Fisher z-transform
    z_stat = 0.5 * np.log((1 + r) / (1 - r))
    df = max(n - len(z_list) - 3, 1)
    se = 1.0 / np.sqrt(df)
    p_value = 2.0 * sp_stats.norm.sf(abs(z_stat / se))

    return float(r), float(p_value)


def _enumerate_directed_paths(
    adj: NDArray, source: int, target: int, max_length: int = 10
) -> list[tuple[int, ...]]:
    """Enumerate all directed paths from *source* to *target*."""
    paths: list[tuple[int, ...]] = []
    stack: list[tuple[int, list[int]]] = [(source, [source])]
    while stack:
        node, path = stack.pop()
        if len(path) > max_length + 1:
            continue
        for child in np.nonzero(adj[node])[0]:
            child = int(child)
            if child == target:
                paths.append(tuple(path + [child]))
            elif child not in path:
                stack.append((child, path + [child]))
    return paths


def _path_effect(
    path: tuple[int, ...], weights: NDArray[np.float64]
) -> float:
    """Compute the product of edge weights along a directed path."""
    effect = 1.0
    for i in range(len(path) - 1):
        effect *= weights[path[i], path[i + 1]]
    return effect


# ============================================================================
# FaithfulnessChecker
# ============================================================================


class FaithfulnessChecker:
    """Analyse faithfulness of a DAG with respect to data.

    Parameters
    ----------
    alpha : float
        Significance level for CI tests.
    max_conditioning_size : int
        Maximum conditioning set size to enumerate.
    cancellation_threshold : float
        Ratio threshold for path-cancellation detection.
        A pair is flagged if ``|total| / max(|paths|) < threshold``.
    """

    def __init__(
        self,
        alpha: float = 0.05,
        max_conditioning_size: int = 3,
        cancellation_threshold: float = 0.1,
    ) -> None:
        self.alpha = alpha
        self.max_conditioning_size = max_conditioning_size
        self.cancellation_threshold = cancellation_threshold

    # -- Main entry point ----------------------------------------------------

    def assess_faithfulness(
        self,
        adj: AdjacencyMatrix,
        data: NDArray[np.float64] | pd.DataFrame,
        weights: NDArray[np.float64] | None = None,
    ) -> FaithfulnessReport:
        """Assess faithfulness of the DAG given data.

        Parameters
        ----------
        adj : AdjacencyMatrix
            The causal DAG.
        data : NDArray | DataFrame
            Observational data (n_samples × n_variables).
        weights : NDArray | None
            If provided, also perform parameter-based analysis.

        Returns
        -------
        FaithfulnessReport
        """
        adj = np.asarray(adj, dtype=np.int8)
        if isinstance(data, pd.DataFrame):
            data = data.values
        data = np.asarray(data, dtype=np.float64)
        n_vars = adj.shape[0]

        violations: list[FaithfulnessViolation] = []
        n_tests = 0

        # Test all pairs with conditioning sets up to max size
        for x in range(n_vars):
            for y in range(x + 1, n_vars):
                others = [
                    i for i in range(n_vars) if i != x and i != y
                ]
                for size in range(self.max_conditioning_size + 1):
                    for z_tuple in itertools.combinations(others, size):
                        z_set = frozenset(z_tuple)
                        n_tests += 1

                        d_sep = _d_separated(adj, x, y, z_set)
                        pcorr, pval = _partial_correlation(data, x, y, z_set)

                        ci_holds = pval > self.alpha
                        if d_sep and not ci_holds:
                            violations.append(FaithfulnessViolation(
                                x=x, y=y,
                                conditioning_set=z_set,
                                d_separated=True,
                                ci_p_value=pval,
                                partial_corr=pcorr,
                                violation_type="extra_dep",
                            ))
                        elif not d_sep and ci_holds:
                            violations.append(FaithfulnessViolation(
                                x=x, y=y,
                                conditioning_set=z_set,
                                d_separated=False,
                                ci_p_value=pval,
                                partial_corr=pcorr,
                                violation_type="unfaithful",
                            ))

        # Path cancellation detection
        cancellations: list[PathCancellation] = []
        if weights is not None:
            cancellations = self._detect_cancellations(adj, weights)

        # Overall severity
        if violations:
            severities = [
                1.0 - v.ci_p_value if v.violation_type == "extra_dep"
                else v.ci_p_value / self.alpha
                for v in violations
            ]
            overall = float(np.mean(severities))
        else:
            overall = 0.0

        return FaithfulnessReport(
            is_faithful=len(violations) == 0,
            violations=tuple(violations),
            cancellations=tuple(cancellations),
            overall_severity=overall,
            n_tests_performed=n_tests,
            n_violations=len(violations),
        )

    # -- Parameter-based faithfulness ----------------------------------------

    def assess_parameter_faithfulness(
        self,
        adj: AdjacencyMatrix,
        weights: NDArray[np.float64],
    ) -> list[PathCancellation]:
        """Detect near-cancellations from the weight matrix alone.

        Parameters
        ----------
        adj : AdjacencyMatrix
            The causal DAG.
        weights : NDArray
            Weight matrix of shape ``(n, n)``.

        Returns
        -------
        list[PathCancellation]
        """
        return self._detect_cancellations(
            np.asarray(adj, dtype=np.int8), weights
        )

    # -- Distribution-based faithfulness (CI testing) ------------------------

    def assess_distribution_faithfulness(
        self,
        adj: AdjacencyMatrix,
        data: NDArray[np.float64] | pd.DataFrame,
    ) -> list[FaithfulnessViolation]:
        """Detect unfaithfulness purely from data via CI tests.

        Parameters
        ----------
        adj : AdjacencyMatrix
            The causal DAG.
        data : NDArray | DataFrame
            Observational data.

        Returns
        -------
        list[FaithfulnessViolation]
        """
        report = self.assess_faithfulness(adj, data)
        return list(report.violations)

    # -- Internals -----------------------------------------------------------

    def _detect_cancellations(
        self,
        adj: NDArray,
        weights: NDArray[np.float64],
    ) -> list[PathCancellation]:
        """Detect path cancellations from edge weights."""
        n = adj.shape[0]
        cancellations: list[PathCancellation] = []

        for source in range(n):
            for target in range(n):
                if source == target:
                    continue
                paths = _enumerate_directed_paths(adj, source, target)
                if len(paths) < 2:
                    continue

                effects = [_path_effect(p, weights) for p in paths]
                total = sum(effects)
                max_abs = max(abs(e) for e in effects)

                if max_abs < 1e-10:
                    continue

                ratio = abs(total) / max_abs
                if ratio < self.cancellation_threshold:
                    severity = 1.0 - ratio
                    cancellations.append(PathCancellation(
                        source=source,
                        target=target,
                        paths=tuple(paths),
                        path_effects=tuple(effects),
                        total_effect=total,
                        severity=severity,
                    ))

        # Sort by severity descending
        cancellations.sort(key=lambda c: c.severity, reverse=True)
        return cancellations

    # -- Convenience ---------------------------------------------------------

    def measure_violation_severity(
        self,
        adj: AdjacencyMatrix,
        data: NDArray[np.float64] | pd.DataFrame,
    ) -> float:
        """Return a scalar measure of faithfulness violation severity.

        Parameters
        ----------
        adj : AdjacencyMatrix
            The causal DAG.
        data : NDArray | DataFrame
            Observational data.

        Returns
        -------
        float
            Severity in [0, 1]; 0 = perfectly faithful.
        """
        report = self.assess_faithfulness(adj, data)
        return report.overall_severity
