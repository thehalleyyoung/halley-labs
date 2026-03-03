"""
Query sensitivity computation for DP-Forge.

This module provides certified global sensitivity calculations for all
supported query types, as well as local and smooth sensitivity variants.
Sensitivity is the core parameter that governs the noise scale in any
differentially private mechanism: it measures the maximum change in query
output when one record is added, removed, or substituted.

Key Components:
    - ``sensitivity_l1``, ``sensitivity_l2``, ``sensitivity_linf``:
      Compute the global sensitivity of a linear query matrix under the
      respective norm.
    - ``adjacency_graph``: Build an explicit adjacency graph for a query
      function over a finite domain.
    - ``QuerySensitivityAnalyzer``: Unified interface for sensitivity
      analysis across query types.
    - Query-specific sensitivity classes (``CountingQuerySensitivity``,
      ``HistogramQuerySensitivity``, etc.) for tight, closed-form bounds.
    - Adjacency-graph builders for standard DP neighbour relations.

Mathematical Background:
    For a query function f: X → R^d and an adjacency relation ~ on X,
    the global sensitivity under L_p norm is:

        GS_p(f) = max_{x ~ x'} ||f(x) - f(x')||_p

    For a linear workload matrix A ∈ R^{m×d}, the global sensitivity is:

        GS_1(A) = max_j ||A e_j||_1   (max column L1 norm)
        GS_2(A) = max_j ||A e_j||_2   (max column L2 norm, = spectral norm)

    where e_j is the j-th standard basis vector.

All functions raise :class:`~dp_forge.exceptions.SensitivityError` when
sensitivity cannot be computed (unbounded domain, non-finite outputs, etc.).
"""

from __future__ import annotations

import itertools
import math
import warnings
from dataclasses import dataclass, field
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Sequence,
    Set,
    Tuple,
    Union,
)

import numpy as np
import numpy.typing as npt
from scipy import sparse

from .exceptions import ConfigurationError, SensitivityError
from .types import (
    AdjacencyRelation,
    QuerySpec,
    QueryType,
    WorkloadSpec,
)


# ---------------------------------------------------------------------------
# Top-level sensitivity functions for linear queries (matrix workloads)
# ---------------------------------------------------------------------------


def sensitivity_l1(
    A: npt.NDArray[np.float64],
    *,
    adjacency: Optional[AdjacencyRelation] = None,
) -> float:
    """Compute the L1 global sensitivity of a linear query matrix.

    For a linear query ``q(x) = A x`` under the substitution/add-remove
    adjacency (where neighbouring databases differ in one coordinate by ±1),
    the L1 global sensitivity is the maximum column L1 norm of A:

        GS_1(A) = max_j  sum_i |A_{ij}|

    When an explicit ``adjacency`` is provided with custom edge structure,
    the sensitivity is computed over all adjacent pairs by enumerating
    differences.

    Args:
        A: Query matrix of shape ``(m, d)`` where *m* is the number of
            queries and *d* is the domain dimension.
        adjacency: Optional adjacency relation.  When ``None``, the
            standard unit-change adjacency is assumed (sensitivity equals
            the max column L1 norm).

    Returns:
        The L1 global sensitivity as a non-negative float.

    Raises:
        SensitivityError: If the matrix contains non-finite values.

    Examples:
        >>> import numpy as np
        >>> A = np.eye(3)
        >>> sensitivity_l1(A)
        1.0
        >>> A = np.array([[1, 1], [0, 1]])
        >>> sensitivity_l1(A)
        2.0
    """
    A = np.asarray(A, dtype=np.float64)
    _validate_matrix(A, "sensitivity_l1")

    if adjacency is None:
        # Standard unit-change: max column L1 norm
        col_l1 = np.sum(np.abs(A), axis=0)
        return float(np.max(col_l1))

    # Custom adjacency: enumerate adjacent pairs
    return _sensitivity_over_adjacency(A, adjacency, ord=1)


def sensitivity_l2(
    A: npt.NDArray[np.float64],
    *,
    adjacency: Optional[AdjacencyRelation] = None,
) -> float:
    """Compute the L2 global sensitivity of a linear query matrix.

    For standard unit-change adjacency:

        GS_2(A) = max_j  ||A_j||_2  =  max column L2 norm

    which equals the spectral norm (largest singular value) when the
    adjacency corresponds to unit coordinate changes.

    Args:
        A: Query matrix of shape ``(m, d)``.
        adjacency: Optional adjacency relation.

    Returns:
        The L2 global sensitivity as a non-negative float.

    Raises:
        SensitivityError: If the matrix contains non-finite values.

    Examples:
        >>> import numpy as np
        >>> A = np.eye(3)
        >>> sensitivity_l2(A)
        1.0
    """
    A = np.asarray(A, dtype=np.float64)
    _validate_matrix(A, "sensitivity_l2")

    if adjacency is None:
        col_l2 = np.sqrt(np.sum(A ** 2, axis=0))
        return float(np.max(col_l2))

    return _sensitivity_over_adjacency(A, adjacency, ord=2)


def sensitivity_linf(
    A: npt.NDArray[np.float64],
    *,
    adjacency: Optional[AdjacencyRelation] = None,
) -> float:
    """Compute the Linf global sensitivity of a linear query matrix.

    For standard unit-change adjacency:

        GS_∞(A) = max_j  max_i |A_{ij}|  =  max column Linf norm

    Args:
        A: Query matrix of shape ``(m, d)``.
        adjacency: Optional adjacency relation.

    Returns:
        The Linf global sensitivity as a non-negative float.

    Raises:
        SensitivityError: If the matrix contains non-finite values.

    Examples:
        >>> import numpy as np
        >>> A = np.array([[1, 2], [3, 4]])
        >>> sensitivity_linf(A)
        4.0
    """
    A = np.asarray(A, dtype=np.float64)
    _validate_matrix(A, "sensitivity_linf")

    if adjacency is None:
        col_linf = np.max(np.abs(A), axis=0)
        return float(np.max(col_linf))

    return _sensitivity_over_adjacency(A, adjacency, ord=np.inf)


# ---------------------------------------------------------------------------
# Adjacency graph construction
# ---------------------------------------------------------------------------


def adjacency_graph(
    f: Callable[[Any], npt.NDArray[np.float64]],
    domain: Sequence[Any],
    *,
    adjacency: Optional[AdjacencyRelation] = None,
) -> Dict[str, Any]:
    """Build an adjacency graph for a query function over a finite domain.

    Evaluates the query function on every domain element and computes
    pairwise differences for all adjacent pairs.  The result is a
    dictionary containing the graph structure and sensitivity metrics.

    Args:
        f: Query function mapping a domain element to a float array.
        domain: Finite sequence of domain elements.
        adjacency: Adjacency relation.  If ``None``, uses complete
            adjacency (every pair is adjacent).

    Returns:
        Dictionary with keys:

        - ``"nodes"``: List of domain elements.
        - ``"values"``: Array of query values, shape ``(n, d)``.
        - ``"edges"``: List of ``(i, j)`` adjacent pairs.
        - ``"diffs"``: Dict mapping ``(i, j)`` to ``f(x_i) - f(x_j)``.
        - ``"sensitivity_l1"``: Global L1 sensitivity.
        - ``"sensitivity_l2"``: Global L2 sensitivity.
        - ``"sensitivity_linf"``: Global Linf sensitivity.

    Raises:
        SensitivityError: If the query function returns non-finite values.

    Examples:
        >>> def f(x):
        ...     return np.array([x, x**2])
        >>> result = adjacency_graph(f, [0, 1, 2])
        >>> result["sensitivity_l1"]
        5.0
    """
    n = len(domain)
    if n == 0:
        raise SensitivityError(
            "Cannot build adjacency graph for empty domain",
            query_type="custom",
            domain_size=0,
        )

    # Evaluate f on all domain elements
    values = []
    for x in domain:
        val = np.asarray(f(x), dtype=np.float64)
        if not np.all(np.isfinite(val)):
            raise SensitivityError(
                f"Query function returned non-finite value for input {x!r}",
                query_type="custom",
            )
        values.append(val)

    values_arr = np.array(values)

    # Determine edges
    if adjacency is not None:
        edges = list(adjacency.edges)
        if adjacency.symmetric:
            edges = edges + [(j, i) for i, j in adjacency.edges]
    else:
        edges = [(i, j) for i in range(n) for j in range(n) if i != j]

    # Compute differences
    diffs: Dict[Tuple[int, int], npt.NDArray[np.float64]] = {}
    max_l1 = 0.0
    max_l2 = 0.0
    max_linf = 0.0

    for i, j in edges:
        diff = values_arr[i] - values_arr[j]
        diffs[(i, j)] = diff

        l1 = float(np.sum(np.abs(diff)))
        l2 = float(np.sqrt(np.sum(diff ** 2)))
        linf = float(np.max(np.abs(diff))) if diff.size > 0 else 0.0

        max_l1 = max(max_l1, l1)
        max_l2 = max(max_l2, l2)
        max_linf = max(max_linf, linf)

    return {
        "nodes": list(domain),
        "values": values_arr,
        "edges": edges,
        "diffs": diffs,
        "sensitivity_l1": max_l1,
        "sensitivity_l2": max_l2,
        "sensitivity_linf": max_linf,
    }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _validate_matrix(A: npt.NDArray[np.float64], caller: str) -> None:
    """Validate that a query matrix is well-formed.

    Args:
        A: Matrix to validate.
        caller: Name of the calling function for error messages.

    Raises:
        SensitivityError: If the matrix is not 2-D or contains non-finite values.
    """
    if A.ndim == 1:
        # Treat 1-D arrays as row vectors
        A = A.reshape(1, -1)
    if A.ndim != 2:
        raise SensitivityError(
            f"{caller}: expected 2-D matrix, got shape {A.shape}",
            sensitivity_norm=caller.replace("sensitivity_", ""),
        )
    if not np.all(np.isfinite(A)):
        raise SensitivityError(
            f"{caller}: matrix contains non-finite values",
            sensitivity_norm=caller.replace("sensitivity_", ""),
        )


def _sensitivity_over_adjacency(
    A: npt.NDArray[np.float64],
    adjacency: AdjacencyRelation,
    ord: Union[int, float],
) -> float:
    """Compute global sensitivity by enumerating adjacent pairs.

    For each adjacent pair (i, j), computes ||A (e_i - e_j)||_p and
    returns the maximum.

    Args:
        A: Query matrix of shape ``(m, d)``.
        adjacency: Adjacency relation defining neighbour pairs.
        ord: Norm order (1, 2, or np.inf).

    Returns:
        Global sensitivity under the specified norm.
    """
    m, d = A.shape
    max_sens = 0.0

    edges = list(adjacency.edges)
    if adjacency.symmetric:
        # For symmetric relations, (i,j) and (j,i) yield the same norm
        pass

    for i, j in edges:
        if i >= d or j >= d:
            # Edge refers to indices beyond matrix columns — skip
            continue
        diff = A[:, i] - A[:, j]
        norm_val = float(np.linalg.norm(diff, ord=ord))
        max_sens = max(max_sens, norm_val)

    return max_sens


# ---------------------------------------------------------------------------
# SensitivityResult dataclass
# ---------------------------------------------------------------------------


@dataclass
class SensitivityResult:
    """Container for sensitivity analysis results.

    Attributes:
        l1: Global L1 sensitivity.
        l2: Global L2 sensitivity.
        linf: Global Linf sensitivity.
        query_type: Type of query analyzed.
        adjacency_type: Description of the adjacency relation used.
        is_tight: Whether the computed bounds are tight (exact).
        details: Additional analysis details.
    """

    l1: float
    l2: float
    linf: float
    query_type: str = "unknown"
    adjacency_type: str = "standard"
    is_tight: bool = True
    details: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name, val in [("l1", self.l1), ("l2", self.l2), ("linf", self.linf)]:
            if val < 0:
                raise ValueError(f"Sensitivity {name} must be non-negative, got {val}")
            if not math.isfinite(val):
                raise ValueError(f"Sensitivity {name} must be finite, got {val}")

    def max_sensitivity(self) -> float:
        """Return the maximum sensitivity across all norms."""
        return max(self.l1, self.l2, self.linf)

    def __repr__(self) -> str:
        return (
            f"SensitivityResult(l1={self.l1:.4f}, l2={self.l2:.4f}, "
            f"linf={self.linf:.4f}, type={self.query_type})"
        )


# ---------------------------------------------------------------------------
# QuerySensitivityAnalyzer
# ---------------------------------------------------------------------------


class QuerySensitivityAnalyzer:
    """Unified sensitivity analysis for all query types.

    This class dispatches sensitivity computation to the appropriate
    query-specific handler based on the ``QueryType`` of the input
    specification, and provides methods for smooth and local sensitivity.

    Examples:
        >>> analyzer = QuerySensitivityAnalyzer()
        >>> spec = QuerySpec.counting(n=5, epsilon=1.0)
        >>> result = analyzer.analyze(spec)
        >>> result.l1
        1.0
    """

    _dispatch: Dict[QueryType, Callable[..., SensitivityResult]]

    def __init__(self) -> None:
        self._dispatch = {
            QueryType.COUNTING: self._analyze_counting,
            QueryType.HISTOGRAM: self._analyze_histogram,
            QueryType.RANGE: self._analyze_range,
            QueryType.LINEAR_WORKLOAD: self._analyze_linear_workload,
            QueryType.MARGINAL: self._analyze_marginal,
            QueryType.CUSTOM: self._analyze_custom,
        }

    def analyze(self, query_spec: QuerySpec) -> SensitivityResult:
        """Compute all sensitivity metrics for a query specification.

        Dispatches to the appropriate handler based on
        ``query_spec.query_type``.

        Args:
            query_spec: Full query specification including query values,
                adjacency, and query type.

        Returns:
            A :class:`SensitivityResult` with L1, L2, and Linf sensitivity.

        Raises:
            SensitivityError: If sensitivity cannot be computed.
        """
        handler = self._dispatch.get(query_spec.query_type)
        if handler is None:
            raise SensitivityError(
                f"No sensitivity handler for query type {query_spec.query_type}",
                query_type=query_spec.query_type.name,
            )
        return handler(query_spec)

    def compute_smooth_sensitivity(
        self,
        f: Callable[[Any], float],
        beta: float,
        domain: Sequence[Any],
        adjacency: Optional[AdjacencyRelation] = None,
    ) -> float:
        """Compute the smooth sensitivity of a function.

        Smooth sensitivity (Nissim, Raskhodnikova, Smith 2007) is defined as:

            S*_{f,β}(x) = max_{y ∈ X} ( LS_f(y) · exp(-β · dist(x, y)) )

        where LS_f(y) is the local sensitivity at y and dist(x, y) is the
        graph distance in the adjacency graph.  For the global smooth
        sensitivity (used here), we take the maximum over all x:

            SS_{f,β} = max_{x ∈ X} S*_{f,β}(x)

        This function computes SS_{f,β} by exhaustive enumeration, which
        is exact but requires O(n²) time.

        Args:
            f: Scalar query function.
            beta: Smoothness parameter β > 0.  Smaller β gives tighter
                bounds but weaker privacy.
            domain: Finite domain of the query.
            adjacency: Adjacency relation.  If ``None``, uses the
                Hamming-1 relation (consecutive elements).

        Returns:
            The global smooth sensitivity as a float.

        Raises:
            SensitivityError: If computation fails.
            ConfigurationError: If ``beta <= 0``.
        """
        if beta <= 0:
            raise ConfigurationError(
                f"beta must be > 0, got {beta}",
                parameter="beta",
                value=beta,
                constraint="beta > 0",
            )

        n = len(domain)
        if n == 0:
            raise SensitivityError(
                "Cannot compute smooth sensitivity for empty domain",
                domain_size=0,
            )

        # Compute all function values
        values = np.array([f(x) for x in domain], dtype=np.float64)
        if not np.all(np.isfinite(values)):
            raise SensitivityError(
                "Query function returned non-finite values",
                query_type="custom",
            )

        # Build adjacency for distance computation
        if adjacency is None:
            adjacency = AdjacencyRelation.hamming_distance_1(n)

        # Compute shortest-path distances (BFS on adjacency graph)
        distances = _compute_all_distances(adjacency)

        # Compute local sensitivities
        local_sens = np.zeros(n)
        all_edges = list(adjacency.edges)
        if adjacency.symmetric:
            all_edges += [(j, i) for i, j in adjacency.edges]

        for i, j in all_edges:
            diff = abs(values[i] - values[j])
            local_sens[i] = max(local_sens[i], diff)

        # Compute smooth sensitivity: max over x of max over y of LS(y)*exp(-β*d(x,y))
        smooth_sens = 0.0
        for x_idx in range(n):
            s_star = 0.0
            for y_idx in range(n):
                d = distances[x_idx, y_idx]
                s_star = max(s_star, local_sens[y_idx] * math.exp(-beta * d))
            smooth_sens = max(smooth_sens, s_star)

        return smooth_sens

    def compute_local_sensitivity(
        self,
        f: Callable[[Any], float],
        x: Any,
        domain: Sequence[Any],
        x_index: int,
        adjacency: Optional[AdjacencyRelation] = None,
    ) -> float:
        """Compute the local sensitivity of a function at a specific point.

        Local sensitivity at x is:

            LS_f(x) = max_{x' ~ x} |f(x) - f(x')|

        Args:
            f: Scalar query function.
            x: The specific input point.
            domain: Finite domain of the query.
            x_index: Index of x in the domain.
            adjacency: Adjacency relation.

        Returns:
            The local sensitivity at x.

        Raises:
            SensitivityError: If computation fails.
        """
        n = len(domain)
        if adjacency is None:
            adjacency = AdjacencyRelation.hamming_distance_1(n)

        fx = f(x)
        if not math.isfinite(fx):
            raise SensitivityError(
                f"Query function returned non-finite value at x={x!r}",
                query_type="custom",
            )

        # Find neighbours of x_index
        neighbours = _get_neighbours(adjacency, x_index)

        max_diff = 0.0
        for j in neighbours:
            fxj = f(domain[j])
            if not math.isfinite(fxj):
                raise SensitivityError(
                    f"Query function returned non-finite value at domain[{j}]",
                    query_type="custom",
                )
            max_diff = max(max_diff, abs(fx - fxj))

        return max_diff

    def lipschitz_sensitivity(
        self,
        f: Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]],
        domain: npt.NDArray[np.float64],
        *,
        n_samples: int = 1000,
        seed: Optional[int] = None,
    ) -> SensitivityResult:
        """Estimate sensitivity of a Lipschitz-continuous query via sampling.

        For Lipschitz-continuous functions f: R^d → R^m with Lipschitz
        constant L, the global sensitivity under unit-change adjacency is
        bounded by L.  This method estimates L by sampling pairs from
        the domain.

        Args:
            f: Lipschitz-continuous query function.
            domain: Array of domain points, shape ``(n, d)``.
            n_samples: Number of random pairs to sample for Lipschitz
                constant estimation.
            seed: Random seed for reproducibility.

        Returns:
            A :class:`SensitivityResult` with estimated sensitivities.
            The ``is_tight`` field is set to ``False`` since this is
            an estimate.

        Raises:
            SensitivityError: If estimation fails.
        """
        domain = np.asarray(domain, dtype=np.float64)
        if domain.ndim == 1:
            domain = domain.reshape(-1, 1)
        n = domain.shape[0]

        if n < 2:
            raise SensitivityError(
                "Domain must have at least 2 points for Lipschitz estimation",
                query_type="lipschitz",
                domain_size=n,
            )

        rng = np.random.default_rng(seed)

        max_ratio_l1 = 0.0
        max_ratio_l2 = 0.0
        max_ratio_linf = 0.0

        # Evaluate f on all domain points
        f_values = []
        for i in range(n):
            val = np.asarray(f(domain[i]), dtype=np.float64)
            if not np.all(np.isfinite(val)):
                raise SensitivityError(
                    f"Query function returned non-finite value at domain[{i}]",
                    query_type="lipschitz",
                )
            f_values.append(val)
        f_arr = np.array(f_values)

        # Sample random pairs and estimate Lipschitz constant
        effective_samples = min(n_samples, n * (n - 1) // 2)

        if n * (n - 1) // 2 <= n_samples:
            # Enumerate all pairs
            pairs = list(itertools.combinations(range(n), 2))
        else:
            idx_i = rng.integers(0, n, size=n_samples)
            idx_j = rng.integers(0, n, size=n_samples)
            pairs = [
                (int(i), int(j)) for i, j in zip(idx_i, idx_j) if i != j
            ]

        for i, j in pairs:
            input_diff = domain[i] - domain[j]
            input_norm = float(np.linalg.norm(input_diff))
            if input_norm < 1e-15:
                continue

            output_diff = f_arr[i] - f_arr[j]
            out_l1 = float(np.sum(np.abs(output_diff)))
            out_l2 = float(np.linalg.norm(output_diff))
            out_linf = float(np.max(np.abs(output_diff))) if output_diff.size > 0 else 0.0

            max_ratio_l1 = max(max_ratio_l1, out_l1 / input_norm)
            max_ratio_l2 = max(max_ratio_l2, out_l2 / input_norm)
            max_ratio_linf = max(max_ratio_linf, out_linf / input_norm)

        return SensitivityResult(
            l1=max_ratio_l1,
            l2=max_ratio_l2,
            linf=max_ratio_linf,
            query_type="lipschitz",
            adjacency_type="continuous_domain",
            is_tight=False,
            details={
                "n_samples": effective_samples,
                "n_domain": n,
                "method": "lipschitz_sampling",
            },
        )

    # -----------------------------------------------------------------------
    # Private dispatch handlers
    # -----------------------------------------------------------------------

    def _analyze_counting(self, spec: QuerySpec) -> SensitivityResult:
        """Sensitivity for counting queries.

        A counting query counts records satisfying a predicate.  Under
        add/remove adjacency, GS = 1 for all norms.
        """
        return SensitivityResult(
            l1=1.0,
            l2=1.0,
            linf=1.0,
            query_type="counting",
            adjacency_type="add_remove",
            is_tight=True,
            details={"n": spec.n, "closed_form": True},
        )

    def _analyze_histogram(self, spec: QuerySpec) -> SensitivityResult:
        """Sensitivity for histogram queries.

        A histogram query returns a vector of bin counts.  Under add/remove
        adjacency, one bin changes by 1, so:
            GS_1 = 1 (add/remove: one bin ±1)
            GS_2 = 1
            GS_∞ = 1

        Under substitution adjacency, one record moves from bin i to bin j:
            GS_1 = 2 (one bin -1, another +1)
            GS_2 = sqrt(2)
            GS_∞ = 1
        """
        adj_type = spec.metadata.get("adjacency_type", "add_remove")

        if adj_type == "substitution":
            return SensitivityResult(
                l1=2.0,
                l2=math.sqrt(2.0),
                linf=1.0,
                query_type="histogram",
                adjacency_type="substitution",
                is_tight=True,
                details={
                    "n_bins": spec.n,
                    "closed_form": True,
                    "note": "Substitution: one record moves between bins",
                },
            )
        else:
            return SensitivityResult(
                l1=1.0,
                l2=1.0,
                linf=1.0,
                query_type="histogram",
                adjacency_type="add_remove",
                is_tight=True,
                details={
                    "n_bins": spec.n,
                    "closed_form": True,
                    "note": "Add/remove: one bin changes by ±1",
                },
            )

    def _analyze_range(self, spec: QuerySpec) -> SensitivityResult:
        """Sensitivity for range queries.

        A range query sums records in an interval [a, b].  For prefix-sum
        (cumulative) queries over d elements:
            - Under add/remove, each prefix sum changes by at most 1,
              and up to d prefix sums are affected.
            GS_1 = d (all prefix sums after the changed element)
            GS_2 = sqrt(d)
            GS_∞ = 1

        For general range queries, sensitivity depends on how many ranges
        contain any single element.
        """
        d = spec.n
        max_ranges_per_element = spec.metadata.get("max_ranges_per_element", d)

        return SensitivityResult(
            l1=float(max_ranges_per_element),
            l2=math.sqrt(float(max_ranges_per_element)),
            linf=1.0,
            query_type="range",
            adjacency_type="add_remove",
            is_tight=True,
            details={
                "d": d,
                "max_ranges_per_element": max_ranges_per_element,
                "closed_form": True,
            },
        )

    def _analyze_linear_workload(self, spec: QuerySpec) -> SensitivityResult:
        """Sensitivity for general linear workload queries.

        For a workload matrix A, the sensitivity is the maximum column norm.
        This requires the workload matrix to be stored in metadata.
        """
        A = spec.metadata.get("workload_matrix")
        if A is None:
            # Fall back to spec.sensitivity
            return SensitivityResult(
                l1=spec.sensitivity,
                l2=spec.sensitivity,
                linf=spec.sensitivity,
                query_type="linear_workload",
                adjacency_type="unit_change",
                is_tight=False,
                details={"note": "Using pre-computed sensitivity from spec"},
            )

        A = np.asarray(A, dtype=np.float64)
        _validate_matrix(A, "linear_workload")

        return SensitivityResult(
            l1=sensitivity_l1(A),
            l2=sensitivity_l2(A),
            linf=sensitivity_linf(A),
            query_type="linear_workload",
            adjacency_type="unit_change",
            is_tight=True,
            details={
                "m": A.shape[0],
                "d": A.shape[1],
                "closed_form": True,
            },
        )

    def _analyze_marginal(self, spec: QuerySpec) -> SensitivityResult:
        """Sensitivity for marginal queries.

        A k-way marginal query over d binary attributes computes the
        joint histogram of k out of d attributes.  Under add/remove:
            GS_1 = 1 (exactly one cell of the marginal changes)
            GS_2 = 1
            GS_∞ = 1

        Under substitution:
            GS_1 = 2
            GS_2 = sqrt(2)
            GS_∞ = 1
        """
        k = spec.metadata.get("k", 1)
        adj_type = spec.metadata.get("adjacency_type", "add_remove")

        if adj_type == "substitution":
            return SensitivityResult(
                l1=2.0,
                l2=math.sqrt(2.0),
                linf=1.0,
                query_type="marginal",
                adjacency_type="substitution",
                is_tight=True,
                details={"k": k, "closed_form": True},
            )
        else:
            return SensitivityResult(
                l1=1.0,
                l2=1.0,
                linf=1.0,
                query_type="marginal",
                adjacency_type="add_remove",
                is_tight=True,
                details={"k": k, "closed_form": True},
            )

    def _analyze_custom(self, spec: QuerySpec) -> SensitivityResult:
        """Sensitivity for custom queries.

        Computes sensitivity by enumerating all adjacent pairs in the
        specification's adjacency relation.
        """
        values = spec.query_values
        edges = spec.edges
        assert edges is not None

        max_l1 = 0.0
        max_l2 = 0.0
        max_linf = 0.0

        all_edges = list(edges.edges)
        if edges.symmetric:
            all_edges += [(j, i) for i, j in edges.edges]

        for i, j in all_edges:
            diff = abs(values[i] - values[j])
            max_l1 = max(max_l1, diff)
            max_l2 = max(max_l2, diff)
            max_linf = max(max_linf, diff)

        return SensitivityResult(
            l1=max_l1,
            l2=max_l2,
            linf=max_linf,
            query_type="custom",
            adjacency_type=edges.description or "custom",
            is_tight=True,
            details={
                "n": spec.n,
                "n_edges": len(all_edges),
                "method": "enumeration",
            },
        )


# ---------------------------------------------------------------------------
# Query-specific sensitivity classes
# ---------------------------------------------------------------------------


class CountingQuerySensitivity:
    """Certified sensitivity for counting queries.

    A counting query counts the number of records satisfying some predicate
    φ: X → {0, 1}.  The global sensitivity is always 1 under add/remove
    adjacency, regardless of the predicate.

    Attributes:
        n: Domain size (number of possible database states).
        adjacency_type: Type of adjacency relation ("add_remove" or
            "substitution").

    Examples:
        >>> cqs = CountingQuerySensitivity(n=100)
        >>> cqs.global_sensitivity_l1()
        1.0
        >>> cqs.global_sensitivity_l2()
        1.0
    """

    def __init__(
        self,
        n: int = 1,
        *,
        adjacency_type: str = "add_remove",
    ) -> None:
        if n < 1:
            raise ConfigurationError(
                f"n must be >= 1, got {n}",
                parameter="n",
                value=n,
                constraint="n >= 1",
            )
        self.n = n
        self.adjacency_type = adjacency_type

    def global_sensitivity_l1(self) -> float:
        """L1 global sensitivity = 1 for counting queries."""
        return 1.0

    def global_sensitivity_l2(self) -> float:
        """L2 global sensitivity = 1 for counting queries."""
        return 1.0

    def global_sensitivity_linf(self) -> float:
        """Linf global sensitivity = 1 for counting queries."""
        return 1.0

    def local_sensitivity(self, x: int) -> float:
        """Local sensitivity at any point is 1 for counting queries.

        Args:
            x: Database index (unused for counting queries).

        Returns:
            Always 1.0.
        """
        return 1.0

    def analyze(self) -> SensitivityResult:
        """Full sensitivity analysis for counting queries.

        Returns:
            A :class:`SensitivityResult` with all metrics.
        """
        return SensitivityResult(
            l1=self.global_sensitivity_l1(),
            l2=self.global_sensitivity_l2(),
            linf=self.global_sensitivity_linf(),
            query_type="counting",
            adjacency_type=self.adjacency_type,
            is_tight=True,
            details={
                "n": self.n,
                "closed_form": True,
                "reference": "Dwork & Roth, Algorithmic Foundations of DP, Sec 3.3",
            },
        )

    def __repr__(self) -> str:
        return f"CountingQuerySensitivity(n={self.n}, GS=1)"


class HistogramQuerySensitivity:
    """Certified sensitivity for histogram queries.

    A histogram query partitions records into d bins and returns the bin
    counts.  The sensitivity depends on the adjacency relation:

    - **Add/remove**: One record is added or removed. Exactly one bin
      changes by ±1.
        GS_1 = 1, GS_2 = 1, GS_∞ = 1

    - **Substitution**: One record changes its value.  One bin loses a
      record and another gains one.
        GS_1 = 2, GS_2 = √2, GS_∞ = 1

    Attributes:
        n_bins: Number of histogram bins.
        adjacency_type: "add_remove" or "substitution".

    Examples:
        >>> hqs = HistogramQuerySensitivity(n_bins=10)
        >>> hqs.global_sensitivity_l1()
        1.0
        >>> hqs = HistogramQuerySensitivity(n_bins=10, adjacency_type="substitution")
        >>> hqs.global_sensitivity_l1()
        2.0
    """

    def __init__(
        self,
        n_bins: int,
        *,
        adjacency_type: str = "add_remove",
    ) -> None:
        if n_bins < 1:
            raise ConfigurationError(
                f"n_bins must be >= 1, got {n_bins}",
                parameter="n_bins",
                value=n_bins,
                constraint="n_bins >= 1",
            )
        if adjacency_type not in ("add_remove", "substitution"):
            raise ConfigurationError(
                f"adjacency_type must be 'add_remove' or 'substitution', "
                f"got {adjacency_type!r}",
                parameter="adjacency_type",
                value=adjacency_type,
            )
        self.n_bins = n_bins
        self.adjacency_type = adjacency_type

    def global_sensitivity_l1(self) -> float:
        """L1 global sensitivity of the histogram query."""
        if self.adjacency_type == "substitution":
            return 2.0
        return 1.0

    def global_sensitivity_l2(self) -> float:
        """L2 global sensitivity of the histogram query."""
        if self.adjacency_type == "substitution":
            return math.sqrt(2.0)
        return 1.0

    def global_sensitivity_linf(self) -> float:
        """Linf global sensitivity of the histogram query."""
        return 1.0

    def local_sensitivity(self, counts: npt.NDArray[np.int64]) -> float:
        """Local sensitivity at a specific histogram.

        For add/remove, the local sensitivity is always 1.
        For substitution, it is 2 if any bin is non-empty and there are
        at least 2 bins, otherwise 0.

        Args:
            counts: Current bin counts, shape ``(n_bins,)``.

        Returns:
            Local sensitivity at the given histogram.
        """
        counts = np.asarray(counts, dtype=np.int64)
        if self.adjacency_type == "add_remove":
            return 1.0
        # Substitution: need at least one non-empty bin to move a record
        non_empty = np.sum(counts > 0)
        if non_empty >= 1 and self.n_bins >= 2:
            return 2.0
        return 0.0

    def build_adjacency(self, n_records: int) -> AdjacencyRelation:
        """Build the adjacency graph for the histogram query.

        For a histogram with d bins and n records under add/remove,
        adjacent states differ by ±1 in exactly one bin.

        Args:
            n_records: Number of records in the database.

        Returns:
            An :class:`AdjacencyRelation` for the histogram.
        """
        # For small domains, build explicit adjacency
        # Each state is a d-dimensional count vector summing to n_records (or n±1)
        if self.n_bins > 5 or n_records > 10:
            warnings.warn(
                f"Explicit adjacency for histogram({self.n_bins}, {n_records}) "
                f"may be very large; consider using sensitivity bounds directly.",
                stacklevel=2,
            )

        if self.adjacency_type == "add_remove":
            # States summing to n_records are adjacent to states summing to n_records±1
            # For simplicity, use Hamming-1 adjacency on the count vector space
            d = self.n_bins
            edges: List[Tuple[int, int]] = []
            # This is simplified — in practice the state space is combinatorial
            n = (n_records + 1) * d
            for i in range(min(n - 1, 1000)):
                edges.append((i, i + 1))
            return AdjacencyRelation(
                edges=edges,
                n=min(n, 1001),
                symmetric=True,
                description=f"histogram_add_remove(bins={d}, n={n_records})",
            )
        else:
            d = self.n_bins
            n = n_records * d
            edges_list: List[Tuple[int, int]] = []
            for i in range(min(n - 1, 1000)):
                edges_list.append((i, i + 1))
            return AdjacencyRelation(
                edges=edges_list,
                n=min(n, 1001),
                symmetric=True,
                description=f"histogram_substitution(bins={d}, n={n_records})",
            )

    def analyze(self) -> SensitivityResult:
        """Full sensitivity analysis.

        Returns:
            A :class:`SensitivityResult` with all metrics.
        """
        return SensitivityResult(
            l1=self.global_sensitivity_l1(),
            l2=self.global_sensitivity_l2(),
            linf=self.global_sensitivity_linf(),
            query_type="histogram",
            adjacency_type=self.adjacency_type,
            is_tight=True,
            details={
                "n_bins": self.n_bins,
                "closed_form": True,
            },
        )

    def __repr__(self) -> str:
        gs1 = self.global_sensitivity_l1()
        return (
            f"HistogramQuerySensitivity(bins={self.n_bins}, "
            f"adj={self.adjacency_type}, GS_1={gs1})"
        )


class RangeQuerySensitivity:
    """Certified sensitivity for range queries.

    A range query sums elements in an interval [a, b] of a 1-D array.
    The sensitivity depends on the range structure:

    - **Single range**: GS_1 = GS_2 = GS_∞ = 1
    - **All prefix sums** (d queries): GS_1 = d, GS_2 = √d, GS_∞ = 1
    - **All range queries** (d(d+1)/2 queries): GS_1 = d, GS_2 = √d, GS_∞ = 1

    The key quantity is the maximum number of ranges containing any
    single element.

    Attributes:
        d: Domain size (number of elements).
        range_type: "single", "prefix", or "all_range".

    Examples:
        >>> rqs = RangeQuerySensitivity(d=10, range_type="prefix")
        >>> rqs.global_sensitivity_l1()
        10.0
        >>> rqs.global_sensitivity_l2()  # doctest: +ELLIPSIS
        3.162...
    """

    def __init__(
        self,
        d: int,
        *,
        range_type: str = "prefix",
    ) -> None:
        if d < 1:
            raise ConfigurationError(
                f"d must be >= 1, got {d}",
                parameter="d",
                value=d,
                constraint="d >= 1",
            )
        if range_type not in ("single", "prefix", "all_range"):
            raise ConfigurationError(
                f"range_type must be 'single', 'prefix', or 'all_range', "
                f"got {range_type!r}",
                parameter="range_type",
                value=range_type,
            )
        self.d = d
        self.range_type = range_type

    @property
    def max_ranges_per_element(self) -> int:
        """Maximum number of ranges containing any single element.

        For prefix sums, element i is contained in ranges [0,i], [0,i+1],
        ..., [0,d-1], giving d - i ranges.  The maximum is d (for i=0).

        For all range queries, element i is in all ranges [a,b] where
        a <= i <= b.  There are (i+1)(d-i) such ranges, maximized at
        i = d//2, giving approximately d²/4.  But for L1 sensitivity of
        the full workload matrix, it's the max column L1 norm.
        """
        if self.range_type == "single":
            return 1
        elif self.range_type == "prefix":
            return self.d
        else:
            # all_range: max column sum of the workload matrix
            # Element i is in (i+1)*(d-i) ranges
            return max((i + 1) * (self.d - i) for i in range(self.d))

    def global_sensitivity_l1(self) -> float:
        """L1 global sensitivity."""
        return float(self.max_ranges_per_element)

    def global_sensitivity_l2(self) -> float:
        """L2 global sensitivity.

        For prefix sums: sqrt(d).
        For all range queries: computed from the workload matrix.
        """
        if self.range_type == "single":
            return 1.0
        elif self.range_type == "prefix":
            return math.sqrt(float(self.d))
        else:
            # Compute exact L2 sensitivity from workload matrix column norms
            # Column j has entries 1 for rows (a, b) where a <= j <= b
            # Number of such rows = (j+1)(d-j), so L2 norm of column j is sqrt((j+1)(d-j))
            max_col_l2 = 0.0
            for j in range(self.d):
                col_nnz = (j + 1) * (self.d - j)
                max_col_l2 = max(max_col_l2, math.sqrt(float(col_nnz)))
            return max_col_l2

    def global_sensitivity_linf(self) -> float:
        """Linf global sensitivity = 1 for all range query types."""
        return 1.0

    def analyze(self) -> SensitivityResult:
        """Full sensitivity analysis.

        Returns:
            A :class:`SensitivityResult` with all metrics.
        """
        return SensitivityResult(
            l1=self.global_sensitivity_l1(),
            l2=self.global_sensitivity_l2(),
            linf=self.global_sensitivity_linf(),
            query_type="range",
            adjacency_type="add_remove",
            is_tight=True,
            details={
                "d": self.d,
                "range_type": self.range_type,
                "max_ranges_per_element": self.max_ranges_per_element,
                "closed_form": True,
            },
        )

    def __repr__(self) -> str:
        return (
            f"RangeQuerySensitivity(d={self.d}, type={self.range_type}, "
            f"GS_1={self.global_sensitivity_l1():.1f})"
        )


class LinearWorkloadSensitivity:
    """Certified sensitivity for linear workload queries.

    For a workload matrix A ∈ R^{m×d}, the query is q(x) = A x where
    x ∈ R^d is the data histogram.  Under unit-change adjacency (one
    coordinate of x changes by ±1):

        GS_1(A) = max_j ||A_{:,j}||_1  (max column L1 norm)
        GS_2(A) = max_j ||A_{:,j}||_2  (max column L2 norm)
        GS_∞(A) = max_j ||A_{:,j}||_∞  (max column Linf norm)

    Attributes:
        matrix: The workload matrix A, shape ``(m, d)``.

    Examples:
        >>> import numpy as np
        >>> A = np.eye(5)
        >>> lws = LinearWorkloadSensitivity(A)
        >>> lws.global_sensitivity_l1()
        1.0
    """

    def __init__(
        self,
        matrix: npt.NDArray[np.float64],
    ) -> None:
        self.matrix = np.asarray(matrix, dtype=np.float64)
        _validate_matrix(self.matrix, "LinearWorkloadSensitivity")
        if self.matrix.ndim == 1:
            self.matrix = self.matrix.reshape(1, -1)

    @property
    def m(self) -> int:
        """Number of queries."""
        return self.matrix.shape[0]

    @property
    def d(self) -> int:
        """Data domain dimension."""
        return self.matrix.shape[1]

    def global_sensitivity_l1(self) -> float:
        """L1 global sensitivity = max column L1 norm."""
        return sensitivity_l1(self.matrix)

    def global_sensitivity_l2(self) -> float:
        """L2 global sensitivity = max column L2 norm."""
        return sensitivity_l2(self.matrix)

    def global_sensitivity_linf(self) -> float:
        """Linf global sensitivity = max column Linf norm."""
        return sensitivity_linf(self.matrix)

    def column_sensitivities(self, norm_ord: Union[int, float] = 1) -> npt.NDArray[np.float64]:
        """Compute per-column sensitivities.

        Args:
            norm_ord: Norm order (1, 2, or np.inf).

        Returns:
            Array of shape ``(d,)`` with the norm of each column.
        """
        result = np.zeros(self.d)
        for j in range(self.d):
            result[j] = float(np.linalg.norm(self.matrix[:, j], ord=norm_ord))
        return result

    def analyze(self) -> SensitivityResult:
        """Full sensitivity analysis.

        Returns:
            A :class:`SensitivityResult` with all metrics.
        """
        col_l1 = self.column_sensitivities(1)
        col_l2 = self.column_sensitivities(2)
        col_linf = self.column_sensitivities(np.inf)

        return SensitivityResult(
            l1=float(np.max(col_l1)),
            l2=float(np.max(col_l2)),
            linf=float(np.max(col_linf)),
            query_type="linear_workload",
            adjacency_type="unit_change",
            is_tight=True,
            details={
                "m": self.m,
                "d": self.d,
                "col_l1_stats": {
                    "min": float(np.min(col_l1)),
                    "max": float(np.max(col_l1)),
                    "mean": float(np.mean(col_l1)),
                },
                "col_l2_stats": {
                    "min": float(np.min(col_l2)),
                    "max": float(np.max(col_l2)),
                    "mean": float(np.mean(col_l2)),
                },
                "closed_form": True,
            },
        )

    def sensitivity_from_workload_spec(self, spec: WorkloadSpec) -> SensitivityResult:
        """Analyze sensitivity from a :class:`WorkloadSpec`.

        Args:
            spec: Workload specification.

        Returns:
            A :class:`SensitivityResult` with all metrics.
        """
        self.matrix = spec.matrix
        return self.analyze()

    def __repr__(self) -> str:
        return (
            f"LinearWorkloadSensitivity(m={self.m}, d={self.d}, "
            f"GS_1={self.global_sensitivity_l1():.4f})"
        )


class MarginalQuerySensitivity:
    """Certified sensitivity for marginal queries.

    A k-way marginal query over d binary attributes computes the joint
    frequency table for a subset of k attributes.  The sensitivity
    depends on the adjacency relation:

    - **Add/remove**: Exactly one cell of the marginal changes by 1.
        GS_1 = 1, GS_2 = 1, GS_∞ = 1

    - **Substitution**: One cell decreases by 1, another increases by 1.
        GS_1 = 2, GS_2 = √2, GS_∞ = 1

    When answering *all* k-way marginals (C(d,k) marginals, each with 2^k
    cells), the full workload has sensitivity that can be computed from the
    workload matrix structure.

    Attributes:
        d: Number of binary attributes.
        k: Marginal order (number of attributes in each marginal).
        adjacency_type: "add_remove" or "substitution".
        all_marginals: Whether computing all C(d,k) marginals.

    Examples:
        >>> mqs = MarginalQuerySensitivity(d=5, k=2)
        >>> mqs.global_sensitivity_l1()
        1.0
        >>> mqs = MarginalQuerySensitivity(d=5, k=2, adjacency_type="substitution")
        >>> mqs.global_sensitivity_l1()
        2.0
    """

    def __init__(
        self,
        d: int,
        k: int,
        *,
        adjacency_type: str = "add_remove",
        all_marginals: bool = False,
    ) -> None:
        if d < 1:
            raise ConfigurationError(
                f"d must be >= 1, got {d}",
                parameter="d",
                value=d,
                constraint="d >= 1",
            )
        if k < 1 or k > d:
            raise ConfigurationError(
                f"k must be in [1, d={d}], got {k}",
                parameter="k",
                value=k,
                constraint=f"1 <= k <= {d}",
            )
        if adjacency_type not in ("add_remove", "substitution"):
            raise ConfigurationError(
                f"adjacency_type must be 'add_remove' or 'substitution', "
                f"got {adjacency_type!r}",
                parameter="adjacency_type",
                value=adjacency_type,
            )
        self.d = d
        self.k = k
        self.adjacency_type = adjacency_type
        self.all_marginals = all_marginals

    @property
    def n_marginals(self) -> int:
        """Number of k-way marginals: C(d, k)."""
        return math.comb(self.d, self.k)

    @property
    def cells_per_marginal(self) -> int:
        """Number of cells in each marginal: 2^k."""
        return 2 ** self.k

    def global_sensitivity_l1(self) -> float:
        """L1 global sensitivity.

        For a single marginal under add/remove: 1.
        For all marginals under add/remove: C(d-1, k-1) (each record
        contributes to C(d-1, k-1) marginals).
        For substitution: multiply by 2.
        """
        if not self.all_marginals:
            base = 1.0
        else:
            # Each record affects C(d-1, k-1) of the C(d,k) marginals
            base = float(math.comb(self.d - 1, self.k - 1))

        if self.adjacency_type == "substitution":
            return 2.0 * base
        return base

    def global_sensitivity_l2(self) -> float:
        """L2 global sensitivity.

        For a single marginal under add/remove: 1.
        For all marginals under add/remove: sqrt(C(d-1, k-1)).
        For substitution: sqrt(2) × the add/remove sensitivity.
        """
        if not self.all_marginals:
            base = 1.0
        else:
            base = math.sqrt(float(math.comb(self.d - 1, self.k - 1)))

        if self.adjacency_type == "substitution":
            return math.sqrt(2.0) * base
        return base

    def global_sensitivity_linf(self) -> float:
        """Linf global sensitivity = 1 for marginal queries."""
        return 1.0

    def analyze(self) -> SensitivityResult:
        """Full sensitivity analysis.

        Returns:
            A :class:`SensitivityResult` with all metrics.
        """
        return SensitivityResult(
            l1=self.global_sensitivity_l1(),
            l2=self.global_sensitivity_l2(),
            linf=self.global_sensitivity_linf(),
            query_type="marginal",
            adjacency_type=self.adjacency_type,
            is_tight=True,
            details={
                "d": self.d,
                "k": self.k,
                "all_marginals": self.all_marginals,
                "n_marginals": self.n_marginals,
                "cells_per_marginal": self.cells_per_marginal,
                "closed_form": True,
            },
        )

    def __repr__(self) -> str:
        return (
            f"MarginalQuerySensitivity(d={self.d}, k={self.k}, "
            f"adj={self.adjacency_type}, GS_1={self.global_sensitivity_l1():.1f})"
        )


class CustomFunctionSensitivity:
    """Sensitivity estimation for arbitrary query functions.

    Provides two approaches:

    1. **Exact enumeration**: When the domain is finite and small,
       enumerate all adjacent pairs and compute exact sensitivity.
    2. **Sampling estimation**: For large or continuous domains, sample
       pairs and estimate sensitivity with statistical guarantees.

    Attributes:
        f: The query function.
        domain: Finite domain (list of elements) or None for sampling.
        adjacency: Adjacency relation (required for enumeration).

    Examples:
        >>> def f(x):
        ...     return np.array([x ** 2])
        >>> cfs = CustomFunctionSensitivity(f, domain=list(range(5)))
        >>> result = cfs.analyze_exact()
        >>> result.l1  # max |x^2 - (x±1)^2| over adjacents
        7.0
    """

    def __init__(
        self,
        f: Callable[[Any], npt.NDArray[np.float64]],
        domain: Optional[Sequence[Any]] = None,
        adjacency: Optional[AdjacencyRelation] = None,
    ) -> None:
        self.f = f
        self.domain = list(domain) if domain is not None else None
        self.adjacency = adjacency

    def analyze_exact(self) -> SensitivityResult:
        """Compute exact sensitivity by enumerating all adjacent pairs.

        Requires a finite domain and adjacency relation.

        Returns:
            Exact :class:`SensitivityResult`.

        Raises:
            SensitivityError: If domain or adjacency is not set.
        """
        if self.domain is None:
            raise SensitivityError(
                "Exact sensitivity requires a finite domain",
                query_type="custom",
            )

        n = len(self.domain)
        if self.adjacency is None:
            self.adjacency = AdjacencyRelation.hamming_distance_1(n)

        graph = adjacency_graph(self.f, self.domain, adjacency=self.adjacency)

        return SensitivityResult(
            l1=graph["sensitivity_l1"],
            l2=graph["sensitivity_l2"],
            linf=graph["sensitivity_linf"],
            query_type="custom",
            adjacency_type=self.adjacency.description or "custom",
            is_tight=True,
            details={
                "n": n,
                "n_edges": len(graph["edges"]),
                "method": "exact_enumeration",
            },
        )

    def analyze_sampling(
        self,
        n_samples: int = 10000,
        seed: Optional[int] = None,
    ) -> SensitivityResult:
        """Estimate sensitivity by sampling adjacent pairs.

        Randomly samples pairs from the adjacency relation and computes
        the maximum observed difference as a lower bound on sensitivity.

        Args:
            n_samples: Number of pairs to sample.
            seed: Random seed.

        Returns:
            Estimated :class:`SensitivityResult` with ``is_tight=False``.

        Raises:
            SensitivityError: If domain or adjacency is not set.
        """
        if self.domain is None:
            raise SensitivityError(
                "Sampling sensitivity requires a finite domain",
                query_type="custom",
            )

        n = len(self.domain)
        if self.adjacency is None:
            self.adjacency = AdjacencyRelation.hamming_distance_1(n)

        rng = np.random.default_rng(seed)
        all_edges = list(self.adjacency.edges)
        if self.adjacency.symmetric:
            all_edges += [(j, i) for i, j in self.adjacency.edges]

        if len(all_edges) == 0:
            return SensitivityResult(
                l1=0.0, l2=0.0, linf=0.0,
                query_type="custom",
                is_tight=True,
                details={"method": "sampling", "n_edges": 0},
            )

        # Evaluate f on all domain points (caching)
        f_values: Dict[int, npt.NDArray[np.float64]] = {}
        for idx in range(n):
            f_values[idx] = np.asarray(self.f(self.domain[idx]), dtype=np.float64)

        effective_samples = min(n_samples, len(all_edges))
        if effective_samples >= len(all_edges):
            sample_edges = all_edges
        else:
            indices = rng.choice(len(all_edges), size=effective_samples, replace=False)
            sample_edges = [all_edges[i] for i in indices]

        max_l1 = 0.0
        max_l2 = 0.0
        max_linf = 0.0

        for i, j in sample_edges:
            diff = f_values[i] - f_values[j]
            l1 = float(np.sum(np.abs(diff)))
            l2 = float(np.linalg.norm(diff))
            linf = float(np.max(np.abs(diff))) if diff.size > 0 else 0.0

            max_l1 = max(max_l1, l1)
            max_l2 = max(max_l2, l2)
            max_linf = max(max_linf, linf)

        return SensitivityResult(
            l1=max_l1,
            l2=max_l2,
            linf=max_linf,
            query_type="custom",
            adjacency_type=self.adjacency.description or "custom",
            is_tight=(effective_samples >= len(all_edges)),
            details={
                "n": n,
                "n_edges_total": len(all_edges),
                "n_samples": effective_samples,
                "method": "sampling",
            },
        )

    def analyze_symbolic(
        self,
        gradient: Optional[Callable[[Any], npt.NDArray[np.float64]]] = None,
    ) -> SensitivityResult:
        """Estimate sensitivity using gradient information.

        If the gradient (Jacobian) of the query function is available,
        uses it to bound the Lipschitz constant and hence the sensitivity.

        For f: R^d → R^m with Jacobian J(x), the Lipschitz constant is
        bounded by max_x ||J(x)||_{p→q}, and hence:

            GS_p(f) <= max_x ||J(x)||_{p→q}

        Args:
            gradient: Function returning the Jacobian at a point, shape
                ``(m, d)``.

        Returns:
            Estimated :class:`SensitivityResult` with ``is_tight=False``.

        Raises:
            SensitivityError: If no gradient is provided and no domain
                is available for numerical differentiation.
        """
        if self.domain is None:
            raise SensitivityError(
                "Symbolic sensitivity requires a domain for evaluation",
                query_type="custom",
            )

        n = len(self.domain)
        max_l1 = 0.0
        max_l2 = 0.0
        max_linf = 0.0

        for idx in range(n):
            x = self.domain[idx]

            if gradient is not None:
                J = np.asarray(gradient(x), dtype=np.float64)
            else:
                # Numerical gradient via finite differences
                J = self._numerical_jacobian(x)

            if J.ndim == 1:
                J = J.reshape(1, -1)

            if not np.all(np.isfinite(J)):
                continue

            # Column norms of the Jacobian give local sensitivity bounds
            for col_idx in range(J.shape[1]):
                col = J[:, col_idx]
                max_l1 = max(max_l1, float(np.sum(np.abs(col))))
                max_l2 = max(max_l2, float(np.linalg.norm(col)))
                max_linf = max(max_linf, float(np.max(np.abs(col))))

        return SensitivityResult(
            l1=max_l1,
            l2=max_l2,
            linf=max_linf,
            query_type="custom",
            adjacency_type="lipschitz_bound",
            is_tight=False,
            details={
                "n_points": n,
                "method": "symbolic_gradient" if gradient else "numerical_gradient",
            },
        )

    def _numerical_jacobian(
        self,
        x: Any,
        h: float = 1e-7,
    ) -> npt.NDArray[np.float64]:
        """Compute the Jacobian via central finite differences.

        Args:
            x: Point at which to compute the Jacobian.
            h: Step size for finite differences.

        Returns:
            Jacobian matrix of shape ``(m, d)``.
        """
        x_arr = np.asarray(x, dtype=np.float64).ravel()
        d = len(x_arr)
        f0 = np.asarray(self.f(x_arr), dtype=np.float64).ravel()
        m = len(f0)

        J = np.zeros((m, d))
        for j in range(d):
            x_plus = x_arr.copy()
            x_minus = x_arr.copy()
            x_plus[j] += h
            x_minus[j] -= h
            f_plus = np.asarray(self.f(x_plus), dtype=np.float64).ravel()
            f_minus = np.asarray(self.f(x_minus), dtype=np.float64).ravel()
            J[:, j] = (f_plus - f_minus) / (2 * h)

        return J

    def __repr__(self) -> str:
        n = len(self.domain) if self.domain is not None else "?"
        return f"CustomFunctionSensitivity(n={n})"


# ---------------------------------------------------------------------------
# Adjacency graph builders
# ---------------------------------------------------------------------------


def bounded_adjacency(n: int) -> AdjacencyRelation:
    """Build adjacency for counting queries on domain {0, ..., n-1}.

    Adjacent states differ by exactly 1:  i ~ i+1  for all valid i.

    Args:
        n: Domain size.

    Returns:
        Hamming-1 adjacency relation on consecutive integers.

    Examples:
        >>> adj = bounded_adjacency(5)
        >>> adj.n
        5
        >>> len(adj.edges)
        4
    """
    if n < 1:
        raise ConfigurationError(
            f"n must be >= 1, got {n}",
            parameter="n",
            value=n,
            constraint="n >= 1",
        )
    return AdjacencyRelation.hamming_distance_1(n)


def hamming_adjacency(d: int, k: int = 1) -> AdjacencyRelation:
    """Build adjacency for vectors at Hamming distance exactly k.

    Each domain element is a binary vector in {0, 1}^d.  Two vectors
    are adjacent if they differ in exactly k positions.

    Args:
        d: Dimension of binary vectors.
        k: Hamming distance for adjacency.

    Returns:
        Adjacency relation on the 2^d binary vectors.

    Raises:
        ConfigurationError: If d is too large (2^d > 2^16) or k > d.

    Examples:
        >>> adj = hamming_adjacency(3, k=1)
        >>> adj.n
        8
    """
    if d < 1:
        raise ConfigurationError(
            f"d must be >= 1, got {d}",
            parameter="d",
            value=d,
            constraint="d >= 1",
        )
    if k < 1 or k > d:
        raise ConfigurationError(
            f"k must be in [1, d={d}], got {k}",
            parameter="k",
            value=k,
            constraint=f"1 <= k <= {d}",
        )
    if d > 16:
        raise ConfigurationError(
            f"d={d} too large for explicit Hamming adjacency (2^{d} = {2**d} vertices)",
            parameter="d",
            value=d,
            constraint="d <= 16",
        )

    n = 2 ** d
    edges: List[Tuple[int, int]] = []

    for i in range(n):
        for j in range(i + 1, n):
            # Count differing bits
            xor = i ^ j
            if bin(xor).count("1") == k:
                edges.append((i, j))

    return AdjacencyRelation(
        edges=edges,
        n=n,
        symmetric=True,
        description=f"Hamming-{k} on {{0,1}}^{d}",
    )


def substitution_adjacency(n: int) -> AdjacencyRelation:
    """Build substitution adjacency on domain {0, ..., n-1}.

    Under substitution, every pair is adjacent (any element can be
    replaced by any other).  This yields complete adjacency.

    Args:
        n: Domain size.

    Returns:
        Complete adjacency relation (every pair is adjacent).

    Examples:
        >>> adj = substitution_adjacency(4)
        >>> adj.n
        4
        >>> len(adj.edges)
        6
    """
    if n < 1:
        raise ConfigurationError(
            f"n must be >= 1, got {n}",
            parameter="n",
            value=n,
            constraint="n >= 1",
        )
    return AdjacencyRelation.complete(n)


def add_remove_adjacency(n: int) -> AdjacencyRelation:
    """Build add/remove adjacency on domain {0, ..., n-1}.

    Under add/remove, adjacent databases differ by exactly one record.
    When modelling this on count vectors, adjacent states differ by ±1
    in exactly one coordinate, i.e., Hamming distance 1 on the counts.

    Args:
        n: Domain size (number of possible count values per coordinate).

    Returns:
        Hamming-1 adjacency (consecutive integers).

    Examples:
        >>> adj = add_remove_adjacency(5)
        >>> adj.n
        5
        >>> len(adj.edges)
        4
    """
    if n < 1:
        raise ConfigurationError(
            f"n must be >= 1, got {n}",
            parameter="n",
            value=n,
            constraint="n >= 1",
        )
    return AdjacencyRelation.hamming_distance_1(n)


def generic_adjacency(
    domain: Sequence[Any],
    relation: Callable[[Any, Any], bool],
) -> AdjacencyRelation:
    """Build adjacency from a custom binary relation.

    Enumerates all pairs and tests the relation to construct the edge
    list.  Automatically detects whether the relation is symmetric.

    Args:
        domain: Finite sequence of domain elements.
        relation: Binary predicate ``(x, y) -> bool`` indicating
            adjacency.

    Returns:
        An :class:`AdjacencyRelation` built from the relation.

    Examples:
        >>> adj = generic_adjacency(
        ...     [0, 1, 2, 3],
        ...     lambda x, y: abs(x - y) == 1,
        ... )
        >>> adj.n
        4
        >>> len(adj.edges)
        3
    """
    n = len(domain)
    if n < 1:
        raise ConfigurationError(
            f"Domain must be non-empty",
            parameter="domain",
            value=n,
            constraint="len(domain) >= 1",
        )

    edges: List[Tuple[int, int]] = []
    is_symmetric = True

    for i in range(n):
        for j in range(i + 1, n):
            fwd = relation(domain[i], domain[j])
            bwd = relation(domain[j], domain[i])

            if fwd and bwd:
                edges.append((i, j))
            elif fwd or bwd:
                is_symmetric = False
                if fwd:
                    edges.append((i, j))
                if bwd:
                    edges.append((j, i))

    return AdjacencyRelation(
        edges=edges,
        n=n,
        symmetric=is_symmetric,
        description="custom_relation",
    )


# ---------------------------------------------------------------------------
# Utility: graph distances
# ---------------------------------------------------------------------------


def _compute_all_distances(adjacency: AdjacencyRelation) -> npt.NDArray[np.float64]:
    """Compute all-pairs shortest-path distances via BFS.

    Args:
        adjacency: Adjacency relation.

    Returns:
        Distance matrix of shape ``(n, n)`` where ``dist[i][j]`` is the
        shortest-path distance from i to j.  ``np.inf`` if disconnected.
    """
    n = adjacency.n
    dist = np.full((n, n), np.inf)
    np.fill_diagonal(dist, 0.0)

    # Build adjacency list
    adj_list: Dict[int, List[int]] = {i: [] for i in range(n)}
    for i, j in adjacency.edges:
        adj_list[i].append(j)
        if adjacency.symmetric:
            adj_list[j].append(i)

    # BFS from each node
    for source in range(n):
        visited = {source}
        queue = [source]
        level = 0
        while queue:
            next_queue: List[int] = []
            for node in queue:
                dist[source, node] = level
                for neighbour in adj_list[node]:
                    if neighbour not in visited:
                        visited.add(neighbour)
                        next_queue.append(neighbour)
            queue = next_queue
            level += 1

    return dist


def _get_neighbours(adjacency: AdjacencyRelation, idx: int) -> List[int]:
    """Get all neighbours of a node in the adjacency graph.

    Args:
        adjacency: Adjacency relation.
        idx: Node index.

    Returns:
        List of neighbour indices.
    """
    neighbours: List[int] = []
    for i, j in adjacency.edges:
        if i == idx:
            neighbours.append(j)
        if adjacency.symmetric and j == idx:
            neighbours.append(i)
    return neighbours


# ---------------------------------------------------------------------------
# Convenience: sensitivity from WorkloadSpec
# ---------------------------------------------------------------------------


def workload_sensitivity(spec: WorkloadSpec) -> SensitivityResult:
    """Compute sensitivity of a :class:`WorkloadSpec`.

    Convenience function that wraps :class:`LinearWorkloadSensitivity`.

    Args:
        spec: Workload specification.

    Returns:
        A :class:`SensitivityResult` with all metrics.

    Examples:
        >>> import numpy as np
        >>> from dp_forge.types import WorkloadSpec
        >>> spec = WorkloadSpec.identity(5)
        >>> result = workload_sensitivity(spec)
        >>> result.l1
        1.0
    """
    analyzer = LinearWorkloadSensitivity(spec.matrix)
    return analyzer.analyze()


def query_spec_sensitivity(spec: QuerySpec) -> SensitivityResult:
    """Compute sensitivity of a :class:`QuerySpec`.

    Convenience function that wraps :class:`QuerySensitivityAnalyzer`.

    Args:
        spec: Query specification.

    Returns:
        A :class:`SensitivityResult` with all metrics.

    Examples:
        >>> spec = QuerySpec.counting(n=5, epsilon=1.0)
        >>> result = query_spec_sensitivity(spec)
        >>> result.l1
        1.0
    """
    analyzer = QuerySensitivityAnalyzer()
    return analyzer.analyze(spec)


# ---------------------------------------------------------------------------
# Sensitivity validation
# ---------------------------------------------------------------------------


def validate_sensitivity(
    claimed: float,
    computed: SensitivityResult,
    *,
    norm: str = "l1",
    tolerance: float = 1e-10,
) -> bool:
    """Validate that a claimed sensitivity matches the computed value.

    Args:
        claimed: The sensitivity value claimed by the user or spec.
        computed: The computed sensitivity result.
        norm: Which norm to check ("l1", "l2", or "linf").
        tolerance: Absolute tolerance for comparison.

    Returns:
        ``True`` if the claimed sensitivity is >= the computed sensitivity
        (within tolerance).

    Raises:
        SensitivityError: If the claimed sensitivity is strictly less than
            the computed value (would violate DP guarantees).
    """
    actual = getattr(computed, norm)
    if claimed < actual - tolerance:
        raise SensitivityError(
            f"Claimed {norm} sensitivity {claimed} is less than computed "
            f"sensitivity {actual:.6f}. Using a sensitivity that is too low "
            f"would violate differential privacy guarantees.",
            sensitivity_norm=norm,
        )
    return True


def sensitivity_from_function(
    f: Callable[[Any], npt.NDArray[np.float64]],
    domain: Sequence[Any],
    *,
    adjacency: Optional[AdjacencyRelation] = None,
) -> SensitivityResult:
    """Compute sensitivity of an arbitrary function over a finite domain.

    This is a convenience wrapper around :func:`adjacency_graph` that
    returns a :class:`SensitivityResult`.

    Args:
        f: Query function.
        domain: Finite domain.
        adjacency: Optional adjacency relation.

    Returns:
        A :class:`SensitivityResult`.

    Examples:
        >>> def f(x):
        ...     return np.array([x])
        >>> result = sensitivity_from_function(f, [0, 1, 2, 3])
        >>> result.l1
        1.0
    """
    graph = adjacency_graph(f, domain, adjacency=adjacency)

    adj_desc = "complete"
    if adjacency is not None:
        adj_desc = adjacency.description or "custom"

    return SensitivityResult(
        l1=graph["sensitivity_l1"],
        l2=graph["sensitivity_l2"],
        linf=graph["sensitivity_linf"],
        query_type="custom",
        adjacency_type=adj_desc,
        is_tight=True,
        details={
            "n": len(domain),
            "n_edges": len(graph["edges"]),
            "method": "exhaustive_enumeration",
        },
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    # Top-level functions
    "sensitivity_l1",
    "sensitivity_l2",
    "sensitivity_linf",
    "adjacency_graph",
    "workload_sensitivity",
    "query_spec_sensitivity",
    "validate_sensitivity",
    "sensitivity_from_function",
    # Result type
    "SensitivityResult",
    # Analyzer
    "QuerySensitivityAnalyzer",
    # Query-specific classes
    "CountingQuerySensitivity",
    "HistogramQuerySensitivity",
    "RangeQuerySensitivity",
    "LinearWorkloadSensitivity",
    "MarginalQuerySensitivity",
    "CustomFunctionSensitivity",
    # Adjacency builders
    "bounded_adjacency",
    "hamming_adjacency",
    "substitution_adjacency",
    "add_remove_adjacency",
    "generic_adjacency",
]
