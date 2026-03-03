"""
Sensitivity polytope computation for DP-Forge.

Computes the convex hull of sensitivity vectors ``{q(x) − q(x') : x ~ x'}``
for general query functions over finite domains.  The geometry of this
polytope determines the optimal noise distribution for differential privacy.

Key Components:
    - ``SensitivityHull``: Exact convex hull computation using
      ``scipy.spatial.ConvexHull`` for vertex enumeration.
    - ``HullApproximation``: Sampling-based approach for high-dimensional
      cases where exact computation is infeasible.

Mathematical Background:
    For a query function ``q: X → R^d`` and adjacency relation ``~``,
    the sensitivity polytope is:

        P = conv({q(x) − q(x') : x ~ x'})

    The L1 sensitivity is the L1 diameter of P, the L2 sensitivity
    is the L2 diameter, etc.  The volume of P in d dimensions is
    related to the complexity of the optimal mechanism: larger polytopes
    require more noise.

    For workload queries ``q(x) = Ax``, sensitivity vectors are
    columns of A (or differences of columns under substitution
    adjacency), and P has the structure of a zonotope.

Usage::

    from dp_forge.sensitivity_hull import SensitivityHull, HullApproximation

    hull = SensitivityHull(sensitivity_vectors)
    hull.compute()
    verts = hull.vertices()
    vol = hull.volume()
    proj = hull.project([0, 1])

    approx = HullApproximation(query_fn, domain, adjacency, dim=10)
    approx.compute(n_samples=10000)
"""

from __future__ import annotations

import logging
import math
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
from scipy import spatial

from .exceptions import (
    ConfigurationError,
    DPForgeError,
    SensitivityError,
)
from .types import AdjacencyRelation, QuerySpec, WorkloadSpec

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# HullResult dataclass
# ---------------------------------------------------------------------------


@dataclass
class HullResult:
    """Result of sensitivity hull computation.

    Attributes:
        vertices: Array of hull vertices, shape (n_vertices, d).
        n_vertices: Number of vertices.
        dimension: Ambient dimension d.
        volume: Volume of the polytope (d-dimensional).
        surface_area: Surface area (only for 2-D and 3-D).
        equations: Halfspace representation (normal, offset) per facet.
        simplices: Simplicial facets of the hull.
        sensitivity_l1: L1 diameter of the hull.
        sensitivity_l2: L2 diameter of the hull.
        sensitivity_linf: Linf diameter of the hull.
        is_exact: Whether this is an exact or approximate hull.
    """

    vertices: npt.NDArray[np.float64]
    n_vertices: int
    dimension: int
    volume: float
    surface_area: float = 0.0
    equations: Optional[npt.NDArray[np.float64]] = None
    simplices: Optional[npt.NDArray[np.intp]] = None
    sensitivity_l1: float = 0.0
    sensitivity_l2: float = 0.0
    sensitivity_linf: float = 0.0
    is_exact: bool = True

    def __repr__(self) -> str:
        exact = "exact" if self.is_exact else "approx"
        return (
            f"HullResult(verts={self.n_vertices}, dim={self.dimension}, "
            f"vol={self.volume:.4e}, {exact})"
        )


# ---------------------------------------------------------------------------
# SensitivityHull — exact hull computation
# ---------------------------------------------------------------------------


class SensitivityHull:
    """Compute the convex hull of sensitivity vectors.

    Given a set of sensitivity vectors ``{q(x) − q(x') : x ~ x'}``,
    computes the exact convex hull using ``scipy.spatial.ConvexHull``
    and derives geometric properties.

    Args:
        vectors: Sensitivity difference vectors, shape (n_pairs, d).
            Each row is ``q(x_i) − q(x_j)`` for some adjacent pair.
        include_negations: Whether to also include negated vectors
            ``q(x') − q(x)`` (for symmetric adjacency).
        include_origin: Whether to include the origin in the hull.

    Raises:
        ConfigurationError: If vectors are empty or invalid.
    """

    def __init__(
        self,
        vectors: npt.NDArray[np.float64],
        *,
        include_negations: bool = True,
        include_origin: bool = True,
    ) -> None:
        vectors = np.asarray(vectors, dtype=np.float64)
        if vectors.ndim == 1:
            vectors = vectors.reshape(-1, 1)
        if vectors.ndim != 2:
            raise ConfigurationError(
                "vectors must be 2-D",
                parameter="vectors",
                value=f"shape={vectors.shape}",
            )
        if len(vectors) == 0:
            raise ConfigurationError(
                "vectors must be non-empty",
                parameter="vectors",
            )
        if not np.all(np.isfinite(vectors)):
            raise SensitivityError(
                "Sensitivity vectors contain non-finite values",
                sensitivity_norm="hull",
            )

        self._raw_vectors = vectors
        self._include_negations = include_negations
        self._include_origin = include_origin
        self._hull_result: Optional[HullResult] = None

        # Build the full point set
        points = [vectors]
        if include_negations:
            points.append(-vectors)
        if include_origin:
            d = vectors.shape[1]
            points.append(np.zeros((1, d), dtype=np.float64))

        self._points = np.unique(np.vstack(points), axis=0)
        self._dimension = vectors.shape[1]

    def compute(self) -> HullResult:
        """Compute the convex hull.

        Returns:
            A :class:`HullResult` with vertices, volume, and sensitivities.

        Raises:
            DPForgeError: If the hull computation fails (e.g., degenerate).
        """
        points = self._points

        if self._dimension == 1:
            return self._compute_1d(points)

        if len(points) < self._dimension + 1:
            return self._compute_degenerate(points)

        # Check for degeneracy: if all points lie in a lower-dim subspace
        centered = points - np.mean(points, axis=0)
        try:
            _, s, _ = np.linalg.svd(centered, full_matrices=False)
        except np.linalg.LinAlgError:
            return self._compute_degenerate(points)

        effective_dim = int(np.sum(s > 1e-10))
        if effective_dim < self._dimension:
            logger.info(
                "Points span %d-dim subspace of R^%d; computing in subspace",
                effective_dim, self._dimension,
            )
            return self._compute_degenerate(points)

        try:
            hull = spatial.ConvexHull(points)
        except spatial.QhullError as e:
            logger.warning("ConvexHull failed: %s; falling back to degenerate", e)
            return self._compute_degenerate(points)

        hull_vertices = points[hull.vertices]

        # Compute sensitivities from pairwise distances of hull vertices
        sens_l1, sens_l2, sens_linf = self._compute_sensitivities(hull_vertices)

        result = HullResult(
            vertices=hull_vertices,
            n_vertices=len(hull_vertices),
            dimension=self._dimension,
            volume=hull.volume,
            surface_area=hull.area if self._dimension >= 2 else 0.0,
            equations=hull.equations,
            simplices=hull.simplices,
            sensitivity_l1=sens_l1,
            sensitivity_l2=sens_l2,
            sensitivity_linf=sens_linf,
            is_exact=True,
        )
        self._hull_result = result

        logger.info(
            "Hull computed: %d vertices, dim=%d, vol=%.4e, "
            "L1=%.4f, L2=%.4f, Linf=%.4f",
            result.n_vertices, result.dimension, result.volume,
            result.sensitivity_l1, result.sensitivity_l2,
            result.sensitivity_linf,
        )

        return result

    def _compute_1d(self, points: npt.NDArray[np.float64]) -> HullResult:
        """Handle 1-D case (interval)."""
        lo = float(np.min(points))
        hi = float(np.max(points))
        vertices = np.array([[lo], [hi]], dtype=np.float64)
        vol = hi - lo

        result = HullResult(
            vertices=vertices,
            n_vertices=2,
            dimension=1,
            volume=vol,
            sensitivity_l1=vol,
            sensitivity_l2=vol,
            sensitivity_linf=vol,
            is_exact=True,
        )
        self._hull_result = result
        return result

    def _compute_degenerate(
        self, points: npt.NDArray[np.float64]
    ) -> HullResult:
        """Handle degenerate cases (coplanar, too few points, etc.)."""
        unique_pts = np.unique(points, axis=0)
        sens_l1, sens_l2, sens_linf = self._compute_sensitivities(unique_pts)

        result = HullResult(
            vertices=unique_pts,
            n_vertices=len(unique_pts),
            dimension=self._dimension,
            volume=0.0,  # Degenerate polytope has zero full-dim volume
            sensitivity_l1=sens_l1,
            sensitivity_l2=sens_l2,
            sensitivity_linf=sens_linf,
            is_exact=True,
        )
        self._hull_result = result
        return result

    def _compute_sensitivities(
        self, vertices: npt.NDArray[np.float64]
    ) -> Tuple[float, float, float]:
        """Compute L1, L2, Linf diameters of vertex set."""
        if len(vertices) <= 1:
            return 0.0, 0.0, 0.0

        # For efficiency, only compute pairwise on hull vertices
        n = len(vertices)
        if n > 5000:
            # Subsample for very large vertex sets
            idx = np.random.default_rng(42).choice(n, min(n, 5000), replace=False)
            vertices = vertices[idx]
            n = len(vertices)

        max_l1 = 0.0
        max_l2 = 0.0
        max_linf = 0.0

        for i in range(n):
            diffs = vertices[i] - vertices[i + 1:]
            if len(diffs) == 0:
                continue
            l1_dists = np.sum(np.abs(diffs), axis=1)
            l2_dists = np.sqrt(np.sum(diffs ** 2, axis=1))
            linf_dists = np.max(np.abs(diffs), axis=1)

            max_l1 = max(max_l1, float(np.max(l1_dists)))
            max_l2 = max(max_l2, float(np.max(l2_dists)))
            max_linf = max(max_linf, float(np.max(linf_dists)))

        return max_l1, max_l2, max_linf

    def vertices(self) -> npt.NDArray[np.float64]:
        """Return hull vertices.

        Returns:
            Array of shape (n_vertices, d).

        Raises:
            DPForgeError: If compute() has not been called.
        """
        if self._hull_result is None:
            self.compute()
        assert self._hull_result is not None
        return self._hull_result.vertices.copy()

    def volume(self) -> float:
        """Return hull volume.

        Returns:
            Volume of the convex hull.

        Raises:
            DPForgeError: If compute() has not been called.
        """
        if self._hull_result is None:
            self.compute()
        assert self._hull_result is not None
        return self._hull_result.volume

    def project(
        self, axes: Sequence[int]
    ) -> HullResult:
        """Project the hull onto a coordinate subspace.

        Computes the convex hull of the projected points.

        Args:
            axes: Indices of coordinates to keep. E.g., [0, 1] for
                projection onto the first two dimensions.

        Returns:
            A :class:`HullResult` for the projected hull.

        Raises:
            ConfigurationError: If axes are out of range.
        """
        axes_arr = np.asarray(axes, dtype=int)
        if np.any(axes_arr < 0) or np.any(axes_arr >= self._dimension):
            raise ConfigurationError(
                f"axes {axes} out of range for dimension {self._dimension}",
                parameter="axes",
            )

        projected = self._points[:, axes_arr]

        if projected.shape[1] == 1:
            lo = float(np.min(projected))
            hi = float(np.max(projected))
            return HullResult(
                vertices=np.array([[lo], [hi]]),
                n_vertices=2,
                dimension=1,
                volume=hi - lo,
                sensitivity_l1=hi - lo,
                sensitivity_l2=hi - lo,
                sensitivity_linf=hi - lo,
                is_exact=True,
            )

        sub_hull = SensitivityHull(
            projected,
            include_negations=False,
            include_origin=False,
        )
        return sub_hull.compute()

    def contains(self, point: npt.NDArray[np.float64]) -> bool:
        """Test whether a point is inside the hull.

        Uses the halfspace representation: x is inside iff
        ``equations @ [x; 1] <= 0`` for all facets.

        Args:
            point: Point of shape (d,).

        Returns:
            True if the point is inside the hull (within tolerance).
        """
        if self._hull_result is None:
            self.compute()
        assert self._hull_result is not None

        point = np.asarray(point, dtype=np.float64)
        if self._hull_result.equations is None:
            # Degenerate case: check if point is in convex combination
            return self._contains_degenerate(point)

        # equations has shape (n_facets, d+1), each row is [normal | offset]
        eqs = self._hull_result.equations
        residuals = eqs[:, :-1] @ point + eqs[:, -1]
        return bool(np.all(residuals <= 1e-8))

    def _contains_degenerate(self, point: npt.NDArray[np.float64]) -> bool:
        """Membership test for degenerate hull (no facets)."""
        verts = self._hull_result.vertices  # type: ignore[union-attr]
        # Check if point is close to any vertex
        dists = np.sqrt(np.sum((verts - point) ** 2, axis=1))
        return bool(np.min(dists) < 1e-8)

    @classmethod
    def from_query(
        cls,
        query_fn: Callable[[Any], npt.NDArray[np.float64]],
        domain: Sequence[Any],
        adjacency: AdjacencyRelation,
    ) -> SensitivityHull:
        """Construct from a query function and adjacency relation.

        Evaluates the query on all domain elements and computes
        pairwise differences for adjacent pairs.

        Args:
            query_fn: Query function mapping domain element to R^d.
            domain: Finite domain elements.
            adjacency: Adjacency relation defining neighbour pairs.

        Returns:
            A SensitivityHull ready for computation.

        Raises:
            SensitivityError: If query_fn returns non-finite values.
        """
        n = len(domain)
        if n == 0:
            raise SensitivityError(
                "Cannot build sensitivity hull for empty domain",
                domain_size=0,
            )

        # Evaluate query on all domain elements
        values = []
        for x in domain:
            v = np.asarray(query_fn(x), dtype=np.float64)
            if not np.all(np.isfinite(v)):
                raise SensitivityError(
                    f"Query returned non-finite for input {x!r}",
                    query_type="custom",
                )
            values.append(v)

        values_arr = np.array(values)

        # Compute difference vectors for adjacent pairs
        diffs = []
        for i, j in adjacency.edges:
            diffs.append(values_arr[i] - values_arr[j])
            if adjacency.symmetric:
                diffs.append(values_arr[j] - values_arr[i])

        if not diffs:
            raise SensitivityError(
                "No adjacent pairs to compute sensitivity vectors",
                domain_size=n,
            )

        return cls(np.array(diffs), include_negations=False, include_origin=True)

    @classmethod
    def from_workload(cls, spec: WorkloadSpec) -> SensitivityHull:
        """Construct from a WorkloadSpec.

        For workload ``q(x) = Ax``, sensitivity vectors under standard
        unit-change adjacency are the columns of A (and their negations).

        Args:
            spec: Workload specification with matrix A.

        Returns:
            A SensitivityHull of the column vectors of A.
        """
        # Columns of A are sensitivity vectors under unit-change adjacency
        columns = spec.matrix.T  # shape (d, m) -> each row is a column of A
        return cls(columns, include_negations=True, include_origin=True)


# ---------------------------------------------------------------------------
# HullApproximation — sampling-based for high dimensions
# ---------------------------------------------------------------------------


class HullApproximation:
    """Approximate sensitivity hull via random sampling.

    For high-dimensional query functions where exact hull computation
    is infeasible (d > ~15), this class samples random adjacent pairs
    and builds an approximate hull from the sampled difference vectors.

    The approximation provides lower bounds on sensitivities and an
    outer approximation of the true hull.

    Args:
        query_fn: Query function mapping domain element to R^d.
        domain: Finite domain.
        adjacency: Adjacency relation.
        seed: Random seed for reproducibility.

    Raises:
        ConfigurationError: If domain is empty.
    """

    def __init__(
        self,
        query_fn: Callable[[Any], npt.NDArray[np.float64]],
        domain: Sequence[Any],
        adjacency: AdjacencyRelation,
        *,
        seed: Optional[int] = None,
    ) -> None:
        if len(domain) == 0:
            raise ConfigurationError(
                "Domain must be non-empty",
                parameter="domain",
            )

        self.query_fn = query_fn
        self.domain = list(domain)
        self.adjacency = adjacency
        self._rng = np.random.default_rng(seed)

        # Pre-evaluate query values
        self._values: Optional[npt.NDArray[np.float64]] = None
        self._sampled_vectors: Optional[npt.NDArray[np.float64]] = None
        self._hull_result: Optional[HullResult] = None

    def _ensure_values(self) -> npt.NDArray[np.float64]:
        """Evaluate query on all domain elements (cached)."""
        if self._values is None:
            vals = []
            for x in self.domain:
                v = np.asarray(self.query_fn(x), dtype=np.float64)
                if not np.all(np.isfinite(v)):
                    raise SensitivityError(
                        f"Query returned non-finite for input {x!r}",
                        query_type="custom",
                    )
                vals.append(v)
            self._values = np.array(vals)
        return self._values

    def compute(
        self,
        n_samples: int = 10000,
        *,
        include_origin: bool = True,
    ) -> HullResult:
        """Compute approximate hull via sampling.

        Randomly samples ``n_samples`` adjacent pairs and computes
        their difference vectors, then builds a convex hull of the
        sampled points.

        Args:
            n_samples: Number of random adjacent pairs to sample.
            include_origin: Whether to include the origin.

        Returns:
            A :class:`HullResult` (with ``is_exact=False``).
        """
        values = self._ensure_values()
        edges = self.adjacency.edges
        if self.adjacency.symmetric:
            edges = edges + [(j, i) for i, j in self.adjacency.edges]

        n_edges = len(edges)
        if n_edges == 0:
            raise SensitivityError(
                "No adjacent pairs available for sampling",
                domain_size=len(self.domain),
            )

        # Sample with replacement
        n_actual = min(n_samples, n_edges)
        if n_actual == n_edges:
            sampled_indices = np.arange(n_edges)
        else:
            sampled_indices = self._rng.choice(n_edges, n_actual, replace=True)

        diffs = []
        for idx in sampled_indices:
            i, j = edges[idx]
            diffs.append(values[i] - values[j])

        diff_arr = np.array(diffs)
        self._sampled_vectors = diff_arr

        # Build hull
        points = [diff_arr]
        if include_origin:
            d = diff_arr.shape[1]
            points.append(np.zeros((1, d), dtype=np.float64))

        all_points = np.unique(np.vstack(points), axis=0)

        if diff_arr.shape[1] == 1:
            lo = float(np.min(all_points))
            hi = float(np.max(all_points))
            result = HullResult(
                vertices=np.array([[lo], [hi]]),
                n_vertices=2,
                dimension=1,
                volume=hi - lo,
                sensitivity_l1=hi - lo,
                sensitivity_l2=hi - lo,
                sensitivity_linf=hi - lo,
                is_exact=False,
            )
            self._hull_result = result
            return result

        # Try computing hull; may fail in high dimensions
        try:
            hull = spatial.ConvexHull(all_points)
            hull_verts = all_points[hull.vertices]
            vol = hull.volume
            area = hull.area if diff_arr.shape[1] >= 2 else 0.0
            equations = hull.equations
            simplices = hull.simplices
        except spatial.QhullError:
            hull_verts = np.unique(all_points, axis=0)
            vol = 0.0
            area = 0.0
            equations = None
            simplices = None

        # Compute sensitivities from all sampled vectors
        sens_l1 = float(np.max(np.sum(np.abs(diff_arr), axis=1)))
        sens_l2 = float(np.max(np.sqrt(np.sum(diff_arr ** 2, axis=1))))
        sens_linf = float(np.max(np.max(np.abs(diff_arr), axis=1)))

        result = HullResult(
            vertices=hull_verts,
            n_vertices=len(hull_verts),
            dimension=diff_arr.shape[1],
            volume=vol,
            surface_area=area,
            equations=equations,
            simplices=simplices,
            sensitivity_l1=sens_l1,
            sensitivity_l2=sens_l2,
            sensitivity_linf=sens_linf,
            is_exact=False,
        )
        self._hull_result = result

        logger.info(
            "Approximate hull: %d vertices from %d samples, dim=%d, "
            "L1=%.4f, L2=%.4f",
            result.n_vertices, n_actual, result.dimension,
            result.sensitivity_l1, result.sensitivity_l2,
        )

        return result

    def vertices(self) -> npt.NDArray[np.float64]:
        """Return approximate hull vertices.

        Returns:
            Array of shape (n_vertices, d).
        """
        if self._hull_result is None:
            self.compute()
        assert self._hull_result is not None
        return self._hull_result.vertices.copy()

    def volume(self) -> float:
        """Return approximate hull volume.

        Returns:
            Volume of the approximate hull (lower bound on true volume).
        """
        if self._hull_result is None:
            self.compute()
        assert self._hull_result is not None
        return self._hull_result.volume

    def coverage_estimate(self, n_bootstrap: int = 100) -> float:
        """Estimate how well the sample covers the true hull.

        Uses bootstrap resampling to estimate the fraction of the
        full sensitivity polytope captured by the current sample.

        Args:
            n_bootstrap: Number of bootstrap iterations.

        Returns:
            Estimated coverage fraction in [0, 1].
        """
        if self._sampled_vectors is None:
            self.compute()
        assert self._sampled_vectors is not None

        vecs = self._sampled_vectors
        n = len(vecs)

        full_l2 = float(np.max(np.sqrt(np.sum(vecs ** 2, axis=1))))
        if full_l2 < 1e-15:
            return 1.0

        bootstrap_maxes = []
        for _ in range(n_bootstrap):
            idx = self._rng.choice(n, n // 2, replace=True)
            sub = vecs[idx]
            sub_l2 = float(np.max(np.sqrt(np.sum(sub ** 2, axis=1))))
            bootstrap_maxes.append(sub_l2)

        mean_sub = np.mean(bootstrap_maxes)
        return min(1.0, mean_sub / full_l2)
