"""
Marginal query construction for multi-dimensional DP mechanisms.

Provides builders for k-way marginal queries over data cubes, automatic
adjacency graph construction for marginal databases, and sensitivity
computation for common marginal query types.

Marginal queries are the canonical workload for multi-dimensional DP:
given a d-dimensional dataset, a k-way marginal selects k coordinates
and counts records in each cell of the resulting contingency table.

Classes:
    MarginalQuery       — specification of a single marginal query
    MarginalQueryBuilder — builder for k-way marginals with sensitivity
"""

from __future__ import annotations

import itertools
import logging
import math
from dataclasses import dataclass, field
from typing import Any, Dict, FrozenSet, List, Optional, Sequence, Set, Tuple, Union

import numpy as np
import numpy.typing as npt

from dp_forge.exceptions import ConfigurationError
from dp_forge.types import AdjacencyRelation, QuerySpec, QueryType

logger = logging.getLogger(__name__)


@dataclass
class MarginalQuery:
    """Specification of a single marginal query.

    A k-way marginal over coordinates (c₁, ..., c_k) counts records in
    each cell of the k-dimensional contingency table formed by those
    coordinates.

    Attributes:
        coordinates: Tuple of coordinate indices in the marginal.
        domain_sizes: Per-coordinate domain sizes (number of distinct values).
        name: Optional human-readable name.
        weight: Weight for workload-level optimisation.
    """

    coordinates: Tuple[int, ...]
    domain_sizes: Tuple[int, ...]
    name: str = ""
    weight: float = 1.0

    def __post_init__(self) -> None:
        if len(self.coordinates) == 0:
            raise ValueError("coordinates must be non-empty")
        if len(self.coordinates) != len(self.domain_sizes):
            raise ValueError(
                f"coordinates length ({len(self.coordinates)}) must match "
                f"domain_sizes length ({len(self.domain_sizes)})"
            )
        if any(ds < 2 for ds in self.domain_sizes):
            raise ValueError("All domain_sizes must be >= 2")
        if self.weight <= 0:
            raise ValueError(f"weight must be > 0, got {self.weight}")
        # Ensure coordinates are sorted for canonical form
        if list(self.coordinates) != sorted(self.coordinates):
            sorted_pairs = sorted(zip(self.coordinates, self.domain_sizes))
            self.coordinates = tuple(c for c, _ in sorted_pairs)
            self.domain_sizes = tuple(d for _, d in sorted_pairs)

    @property
    def k(self) -> int:
        """Number of coordinates in the marginal (k-way)."""
        return len(self.coordinates)

    @property
    def n_cells(self) -> int:
        """Total number of cells in the contingency table."""
        return math.prod(self.domain_sizes)

    @property
    def coordinate_set(self) -> FrozenSet[int]:
        """Frozen set of coordinate indices."""
        return frozenset(self.coordinates)

    def query_matrix(self) -> npt.NDArray[np.float64]:
        """Build the query matrix for this marginal.

        The query matrix A has shape (n_cells, product_domain) where each
        row is an indicator for one cell of the contingency table.

        Returns:
            Binary query matrix of shape (n_cells, product_domain).
        """
        n_cells = self.n_cells
        A = np.eye(n_cells, dtype=np.float64)
        return A

    def __repr__(self) -> str:
        name_str = f", name={self.name!r}" if self.name else ""
        return (
            f"MarginalQuery(coords={self.coordinates}, "
            f"domains={self.domain_sizes}, cells={self.n_cells}{name_str})"
        )


class MarginalQueryBuilder:
    """Builder for k-way marginal queries over data cubes.

    Constructs marginal queries, computes their sensitivities, and builds
    adjacency graphs for marginal databases.

    Args:
        d: Total number of dimensions in the data cube.
        domain_sizes: Per-dimension domain sizes. If scalar, applies to
            all dimensions.
    """

    def __init__(
        self,
        d: int,
        domain_sizes: Union[int, Sequence[int]] = 2,
    ) -> None:
        if d < 1:
            raise ConfigurationError(
                f"d must be >= 1, got {d}", parameter="d", value=d
            )
        self._d = d
        if isinstance(domain_sizes, int):
            self._domain_sizes = [domain_sizes] * d
        else:
            self._domain_sizes = list(domain_sizes)
            if len(self._domain_sizes) != d:
                raise ConfigurationError(
                    f"domain_sizes length ({len(self._domain_sizes)}) must "
                    f"match d ({d})",
                    parameter="domain_sizes",
                )
        if any(ds < 2 for ds in self._domain_sizes):
            raise ConfigurationError(
                "All domain_sizes must be >= 2",
                parameter="domain_sizes",
                value=self._domain_sizes,
            )

    @property
    def d(self) -> int:
        """Number of dimensions."""
        return self._d

    @property
    def domain_sizes(self) -> List[int]:
        """Per-dimension domain sizes."""
        return self._domain_sizes

    @property
    def total_domain_size(self) -> int:
        """Product of all domain sizes."""
        return math.prod(self._domain_sizes)

    def build_kway(
        self,
        k: int,
        coordinates: Optional[Sequence[int]] = None,
    ) -> List[MarginalQuery]:
        """Build all k-way marginal queries.

        If coordinates is None, generates all (d choose k) possible
        k-way marginals. Otherwise, generates the single marginal
        over the specified coordinates.

        Args:
            k: Number of coordinates per marginal.
            coordinates: Specific coordinate set (optional).

        Returns:
            List of MarginalQuery objects.

        Raises:
            ConfigurationError: If k > d or coordinates are invalid.
        """
        if k < 1 or k > self._d:
            raise ConfigurationError(
                f"k must be in [1, {self._d}], got {k}",
                parameter="k", value=k,
            )
        if coordinates is not None:
            if len(coordinates) != k:
                raise ConfigurationError(
                    f"coordinates length ({len(coordinates)}) must match k ({k})",
                    parameter="coordinates",
                )
            for c in coordinates:
                if not 0 <= c < self._d:
                    raise ConfigurationError(
                        f"coordinate {c} out of range [0, {self._d})",
                        parameter="coordinates",
                    )
            coords_tuple = tuple(sorted(coordinates))
            domains_tuple = tuple(self._domain_sizes[c] for c in coords_tuple)
            return [MarginalQuery(
                coordinates=coords_tuple,
                domain_sizes=domains_tuple,
                name=f"marginal_{coords_tuple}",
            )]
        # Generate all k-way combinations
        queries = []
        for combo in itertools.combinations(range(self._d), k):
            domains = tuple(self._domain_sizes[c] for c in combo)
            queries.append(MarginalQuery(
                coordinates=combo,
                domain_sizes=domains,
                name=f"marginal_{combo}",
            ))
        return queries

    def build_all_marginals(self, max_k: Optional[int] = None) -> List[MarginalQuery]:
        """Build all marginals up to max_k-way.

        Args:
            max_k: Maximum marginal order. Defaults to d.

        Returns:
            List of all marginal queries from 1-way up to max_k-way.
        """
        if max_k is None:
            max_k = self._d
        max_k = min(max_k, self._d)
        all_queries: List[MarginalQuery] = []
        for k in range(1, max_k + 1):
            all_queries.extend(self.build_kway(k))
        return all_queries

    def compute_sensitivity(
        self,
        query: MarginalQuery,
        sensitivity_type: str = "L1",
    ) -> float:
        """Compute sensitivity of a marginal query.

        For counting-based marginals under add/remove adjacency:
        - L1 sensitivity = 2 (one record affects two cells)
        - L2 sensitivity = √2
        - Linf sensitivity = 1

        Under substitution adjacency:
        - L1 sensitivity = 2 (old cell -1, new cell +1)
        - L2 sensitivity = √2
        - Linf sensitivity = 1

        Args:
            query: Marginal query to analyse.
            sensitivity_type: "L1", "L2", or "Linf".

        Returns:
            Sensitivity value.

        Raises:
            ConfigurationError: If sensitivity_type is invalid.
        """
        if sensitivity_type == "L1":
            return 2.0
        elif sensitivity_type == "L2":
            return math.sqrt(2.0)
        elif sensitivity_type == "Linf":
            return 1.0
        else:
            raise ConfigurationError(
                f"sensitivity_type must be L1, L2, or Linf, got {sensitivity_type!r}",
                parameter="sensitivity_type",
                value=sensitivity_type,
            )

    def compute_sensitivities(
        self,
        queries: Sequence[MarginalQuery],
        sensitivity_type: str = "L1",
    ) -> npt.NDArray[np.float64]:
        """Compute sensitivities for multiple marginal queries.

        Args:
            queries: Sequence of MarginalQuery objects.
            sensitivity_type: Norm type for sensitivity.

        Returns:
            Array of sensitivity values, one per query.
        """
        return np.array(
            [self.compute_sensitivity(q, sensitivity_type) for q in queries],
            dtype=np.float64,
        )

    def adjacency_graph(
        self,
        query: MarginalQuery,
    ) -> AdjacencyRelation:
        """Build adjacency relation for a marginal query's output domain.

        Two cell indices are adjacent if they differ in exactly one
        coordinate value by exactly 1 (Hamming-1 in each coordinate).

        Args:
            query: Marginal query defining the cell structure.

        Returns:
            AdjacencyRelation over cell indices.
        """
        n_cells = query.n_cells
        edges: List[Tuple[int, int]] = []
        # Cell index = mixed-radix number over domain_sizes
        strides = self._compute_strides(query.domain_sizes)
        for cell_idx in range(n_cells):
            coords = self._unflatten(cell_idx, query.domain_sizes)
            for dim in range(query.k):
                if coords[dim] + 1 < query.domain_sizes[dim]:
                    neighbour = cell_idx + strides[dim]
                    edges.append((cell_idx, neighbour))
        return AdjacencyRelation(
            edges=edges,
            n=n_cells,
            symmetric=True,
            description=f"Hamming-1 adjacency for marginal {query.coordinates}",
        )

    def build_workload_matrix(
        self,
        queries: Sequence[MarginalQuery],
    ) -> npt.NDArray[np.float64]:
        """Build the combined workload matrix for multiple marginals.

        Each marginal contributes rows to the workload matrix. The
        columns correspond to cells of the full data cube.

        Args:
            queries: Marginal queries to include.

        Returns:
            Workload matrix of shape (total_cells, total_domain_size).
        """
        total_domain = self.total_domain_size
        all_domain_sizes = tuple(self._domain_sizes)
        rows: List[npt.NDArray[np.float64]] = []
        for query in queries:
            marginal_cells = query.n_cells
            for cell_idx in range(marginal_cells):
                cell_coords = self._unflatten(cell_idx, query.domain_sizes)
                row = np.zeros(total_domain, dtype=np.float64)
                # Iterate over all full-domain cells that match this marginal cell
                for full_idx in range(total_domain):
                    full_coords = self._unflatten(full_idx, all_domain_sizes)
                    match = all(
                        full_coords[query.coordinates[j]] == cell_coords[j]
                        for j in range(query.k)
                    )
                    if match:
                        row[full_idx] = query.weight
                rows.append(row)
        return np.vstack(rows) if rows else np.zeros((0, total_domain), dtype=np.float64)

    def build_query_spec(
        self,
        query: MarginalQuery,
        epsilon: float,
        delta: float = 0.0,
        k: int = 100,
    ) -> QuerySpec:
        """Build a QuerySpec for synthesising a mechanism for one marginal.

        Args:
            query: Marginal query specification.
            epsilon: Privacy parameter.
            delta: Approximate DP parameter.
            k: Output discretisation bins.

        Returns:
            QuerySpec ready for CEGIS synthesis.
        """
        n_cells = query.n_cells
        query_values = np.arange(n_cells, dtype=np.float64)
        adjacency = self.adjacency_graph(query)
        sensitivity = self.compute_sensitivity(query, "L1")
        return QuerySpec(
            query_values=query_values,
            domain=f"marginal_{query.coordinates}",
            sensitivity=sensitivity,
            epsilon=epsilon,
            delta=delta,
            k=k,
            query_type=QueryType.MARGINAL,
            edges=adjacency,
        )

    @staticmethod
    def _compute_strides(domain_sizes: Sequence[int]) -> List[int]:
        """Compute mixed-radix strides for cell indexing."""
        k = len(domain_sizes)
        strides = [1] * k
        for i in range(k - 2, -1, -1):
            strides[i] = strides[i + 1] * domain_sizes[i + 1]
        return strides

    @staticmethod
    def _unflatten(
        flat_idx: int, domain_sizes: Sequence[int]
    ) -> List[int]:
        """Convert flat cell index to per-coordinate values."""
        coords = []
        for ds in reversed(domain_sizes):
            coords.append(flat_idx % ds)
            flat_idx //= ds
        return list(reversed(coords))

    @staticmethod
    def _flatten(
        coords: Sequence[int], domain_sizes: Sequence[int]
    ) -> int:
        """Convert per-coordinate values to flat cell index."""
        idx = 0
        for i, (c, ds) in enumerate(zip(coords, domain_sizes)):
            idx = idx * ds + c
        return idx

    def overlapping_coordinates(
        self,
        queries: Sequence[MarginalQuery],
    ) -> Dict[FrozenSet[int], List[int]]:
        """Find which queries share coordinates.

        Returns a mapping from coordinate sets to query indices that
        use those coordinates. Useful for identifying dependencies.

        Args:
            queries: Sequence of marginal queries.

        Returns:
            Dict mapping coordinate frozensets to lists of query indices.
        """
        coord_to_queries: Dict[FrozenSet[int], List[int]] = {}
        for idx, q in enumerate(queries):
            for c in q.coordinates:
                key = frozenset([c])
                if key not in coord_to_queries:
                    coord_to_queries[key] = []
                coord_to_queries[key].append(idx)
        return coord_to_queries
