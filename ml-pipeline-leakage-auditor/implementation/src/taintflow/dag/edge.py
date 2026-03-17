"""
taintflow.dag.edge -- DAG edge types for the Pipeline Information DAG (PI-DAG).

Edges in the PI-DAG represent data flows, fit dependencies, parameter flows,
and control-flow relationships between pipeline operations.  Each edge carries
the set of columns that flow along it, a capacity bound (in bits), and a
provenance fraction indicating how much test-origin data flows along it.

The :class:`EdgeSet` provides efficient indexed storage and lookup of edges
by source, target, column, and kind.
"""

from __future__ import annotations

import copy
import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import (
    Any,
    Dict,
    FrozenSet,
    Iterable,
    Iterator,
    List,
    Mapping,
    Optional,
    Sequence,
    Set,
    Tuple,
)

from taintflow.core.types import (
    ColumnSchema,
    EdgeKind,
    OpType,
    Origin,
    ProvenanceInfo,
    ShapeMetadata,
)


# ===================================================================
#  Pipeline edge
# ===================================================================


@dataclass(frozen=True)
class PipelineEdge:
    """A directed edge in the PI-DAG.

    Parameters
    ----------
    source_id : str
        Node ID of the edge's origin.
    target_id : str
        Node ID of the edge's destination.
    columns : frozenset[str]
        Names of columns that flow along this edge.
    edge_kind : EdgeKind
        The semantic kind of this edge.
    capacity : float
        Upper bound on channel capacity (bits) along this edge.
    provenance_fraction : float
        Fraction of data along this edge from test-origin (ρ ∈ [0, 1]).
    metadata : dict[str, Any]
        Arbitrary extra information.
    """

    source_id: str
    target_id: str
    columns: frozenset[str] = field(default_factory=frozenset)
    edge_kind: EdgeKind = EdgeKind.DATA_FLOW
    capacity: float = 0.0
    provenance_fraction: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict, hash=False, compare=False)

    # -- identity ------------------------------------------------------------

    @property
    def key(self) -> Tuple[str, str, EdgeKind]:
        """Unique key for this edge (source, target, kind)."""
        return (self.source_id, self.target_id, self.edge_kind)

    @property
    def pair(self) -> Tuple[str, str]:
        """Source-target pair (ignoring kind)."""
        return (self.source_id, self.target_id)

    # -- query helpers -------------------------------------------------------

    @property
    def is_data_flow(self) -> bool:
        return self.edge_kind == EdgeKind.DATA_FLOW

    @property
    def is_fit_dependency(self) -> bool:
        return self.edge_kind == EdgeKind.FIT_DEPENDENCY

    @property
    def is_parameter_flow(self) -> bool:
        return self.edge_kind == EdgeKind.PARAMETER_FLOW

    @property
    def is_control_flow(self) -> bool:
        return self.edge_kind == EdgeKind.CONTROL_FLOW

    @property
    def has_test_provenance(self) -> bool:
        """True if any test-origin data flows along this edge."""
        return self.provenance_fraction > 0.0

    @property
    def n_columns(self) -> int:
        return len(self.columns)

    # -- capacity helpers ----------------------------------------------------

    def with_capacity(self, capacity: float) -> "PipelineEdge":
        """Return a copy with updated capacity."""
        return PipelineEdge(
            source_id=self.source_id,
            target_id=self.target_id,
            columns=self.columns,
            edge_kind=self.edge_kind,
            capacity=capacity,
            provenance_fraction=self.provenance_fraction,
            metadata=dict(self.metadata),
        )

    def with_provenance(self, fraction: float) -> "PipelineEdge":
        """Return a copy with updated provenance fraction."""
        return PipelineEdge(
            source_id=self.source_id,
            target_id=self.target_id,
            columns=self.columns,
            edge_kind=self.edge_kind,
            capacity=self.capacity,
            provenance_fraction=max(0.0, min(1.0, fraction)),
            metadata=dict(self.metadata),
        )

    def with_columns(self, columns: frozenset[str]) -> "PipelineEdge":
        """Return a copy with updated column set."""
        return PipelineEdge(
            source_id=self.source_id,
            target_id=self.target_id,
            columns=columns,
            edge_kind=self.edge_kind,
            capacity=self.capacity,
            provenance_fraction=self.provenance_fraction,
            metadata=dict(self.metadata),
        )

    def restrict_columns(self, keep: frozenset[str]) -> "PipelineEdge":
        """Return a copy with columns restricted to *keep*."""
        new_cols = self.columns & keep
        n_original = max(len(self.columns), 1)
        fraction_kept = len(new_cols) / n_original
        return PipelineEdge(
            source_id=self.source_id,
            target_id=self.target_id,
            columns=new_cols,
            edge_kind=self.edge_kind,
            capacity=self.capacity * fraction_kept,
            provenance_fraction=self.provenance_fraction,
            metadata=dict(self.metadata),
        )

    # -- merging -------------------------------------------------------------

    def merge(self, other: "PipelineEdge") -> "PipelineEdge":
        """Merge two edges between the same (source, target) pair.

        Columns are unioned, capacity is summed, provenance fraction is
        the weighted average.
        """
        if self.pair != other.pair:
            raise ValueError(
                f"Cannot merge edges with different endpoints: "
                f"{self.pair} vs {other.pair}"
            )
        merged_cols = self.columns | other.columns
        merged_cap = self.capacity + other.capacity
        total_cols = len(self.columns) + len(other.columns)
        if total_cols > 0:
            merged_prov = (
                self.provenance_fraction * len(self.columns)
                + other.provenance_fraction * len(other.columns)
            ) / total_cols
        else:
            merged_prov = max(self.provenance_fraction, other.provenance_fraction)
        merged_kind = self.edge_kind if self.edge_kind == other.edge_kind else EdgeKind.DATA_FLOW
        merged_meta = {**self.metadata, **other.metadata}
        return PipelineEdge(
            source_id=self.source_id,
            target_id=self.target_id,
            columns=merged_cols,
            edge_kind=merged_kind,
            capacity=merged_cap,
            provenance_fraction=merged_prov,
            metadata=merged_meta,
        )

    # -- validation ----------------------------------------------------------

    def validate(self) -> list[str]:
        """Validate edge invariants."""
        errors: list[str] = []
        if not self.source_id:
            errors.append("source_id must be non-empty")
        if not self.target_id:
            errors.append("target_id must be non-empty")
        if self.source_id == self.target_id:
            errors.append(f"Self-loop: source_id == target_id == {self.source_id!r}")
        if self.capacity < 0.0:
            errors.append(f"capacity must be >= 0, got {self.capacity}")
        if math.isnan(self.capacity):
            errors.append("capacity must not be NaN")
        if not (0.0 <= self.provenance_fraction <= 1.0):
            errors.append(
                f"provenance_fraction must be in [0,1], got {self.provenance_fraction}"
            )
        return errors

    # -- serialization -------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "source_id": self.source_id,
            "target_id": self.target_id,
            "edge_kind": self.edge_kind.value,
            "capacity": self.capacity,
            "provenance_fraction": self.provenance_fraction,
        }
        if self.columns:
            d["columns"] = sorted(self.columns)
        if self.metadata:
            d["metadata"] = dict(self.metadata)
        return d

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "PipelineEdge":
        columns = frozenset(data.get("columns", []))
        kind = EdgeKind(data.get("edge_kind", "data_flow"))
        return cls(
            source_id=str(data["source_id"]),
            target_id=str(data["target_id"]),
            columns=columns,
            edge_kind=kind,
            capacity=float(data.get("capacity", 0.0)),
            provenance_fraction=float(data.get("provenance_fraction", 0.0)),
            metadata=dict(data.get("metadata", {})),
        )

    def __repr__(self) -> str:
        return (
            f"PipelineEdge({self.source_id!r} -> {self.target_id!r}, "
            f"kind={self.edge_kind.name}, cols={len(self.columns)}, "
            f"cap={self.capacity:.2f}bits, ρ={self.provenance_fraction:.3f})"
        )


# ===================================================================
#  Edge capacity computation helpers
# ===================================================================


def compute_data_flow_capacity(
    source_schema: Sequence[ColumnSchema],
    target_schema: Sequence[ColumnSchema],
    columns: frozenset[str],
    n_rows: int = 1,
) -> float:
    """Compute the capacity of a data-flow edge from column entropies.

    Capacity is bounded by the sum of per-column entropy bounds for the
    columns that flow along the edge, multiplied by the number of rows.
    """
    source_map = {c.name: c for c in source_schema}
    target_map = {c.name: c for c in target_schema}
    total = 0.0
    for col_name in columns:
        col = source_map.get(col_name) or target_map.get(col_name)
        if col is not None:
            total += col.entropy_bound()
        else:
            total += 64.0
    return total * max(n_rows, 1)


def compute_fit_dependency_capacity(
    n_features: int,
    n_samples: int,
    estimator_class: str = "",
) -> float:
    """Estimate capacity flowing through a fit-dependency edge.

    For linear models, the capacity is bounded by the number of
    learned parameters times their precision (64 bits each).
    """
    if estimator_class in ("LinearRegression", "Ridge", "Lasso"):
        return float(n_features + 1) * 64.0
    if estimator_class in ("LogisticRegression",):
        return float(n_features + 1) * 64.0
    if estimator_class in ("DecisionTreeClassifier", "DecisionTreeRegressor"):
        max_depth = min(n_samples, 2 ** 20)
        return float(max_depth) * math.log2(max(n_features, 2)) * 64.0
    if estimator_class in ("RandomForestClassifier", "RandomForestRegressor"):
        return float(n_features) * float(n_samples) * 0.1
    return float(n_features) * 64.0


def compute_parameter_flow_capacity(n_params: int) -> float:
    """Capacity of a parameter-flow edge: each parameter is 64 bits."""
    return float(max(n_params, 0)) * 64.0


def compute_control_flow_capacity() -> float:
    """Control-flow edges carry at most 1 bit (branch taken or not)."""
    return 1.0


def estimate_edge_capacity(
    edge: PipelineEdge,
    source_schema: Sequence[ColumnSchema] | None = None,
    target_schema: Sequence[ColumnSchema] | None = None,
    n_rows: int = 1,
    n_features: int = 0,
    n_samples: int = 0,
    estimator_class: str = "",
) -> float:
    """Estimate capacity for an edge based on its kind and available schema info."""
    if edge.edge_kind == EdgeKind.DATA_FLOW:
        return compute_data_flow_capacity(
            source_schema or [],
            target_schema or [],
            edge.columns,
            n_rows,
        )
    if edge.edge_kind == EdgeKind.FIT_DEPENDENCY:
        return compute_fit_dependency_capacity(
            n_features or len(edge.columns),
            n_samples or n_rows,
            estimator_class,
        )
    if edge.edge_kind == EdgeKind.PARAMETER_FLOW:
        return compute_parameter_flow_capacity(len(edge.columns))
    if edge.edge_kind == EdgeKind.CONTROL_FLOW:
        return compute_control_flow_capacity()
    if edge.edge_kind == EdgeKind.LABEL_FLOW:
        return 64.0
    if edge.edge_kind == EdgeKind.INDEX_FLOW:
        return math.log2(max(n_rows, 2))
    return compute_data_flow_capacity(
        source_schema or [], target_schema or [], edge.columns, n_rows,
    )


# ===================================================================
#  EdgeSet: indexed collection of edges
# ===================================================================


class EdgeSet:
    """Efficient indexed storage and lookup of :class:`PipelineEdge` objects.

    Maintains secondary indices for fast lookup by source, target, column,
    and edge kind.
    """

    __slots__ = ("_edges", "_by_source", "_by_target", "_by_column", "_by_kind", "_by_pair")

    def __init__(self, edges: Iterable[PipelineEdge] | None = None) -> None:
        self._edges: dict[Tuple[str, str, EdgeKind], PipelineEdge] = {}
        self._by_source: dict[str, set[Tuple[str, str, EdgeKind]]] = defaultdict(set)
        self._by_target: dict[str, set[Tuple[str, str, EdgeKind]]] = defaultdict(set)
        self._by_column: dict[str, set[Tuple[str, str, EdgeKind]]] = defaultdict(set)
        self._by_kind: dict[EdgeKind, set[Tuple[str, str, EdgeKind]]] = defaultdict(set)
        self._by_pair: dict[Tuple[str, str], set[Tuple[str, str, EdgeKind]]] = defaultdict(set)
        if edges is not None:
            for e in edges:
                self.add(e)

    # -- mutation ------------------------------------------------------------

    def add(self, edge: PipelineEdge) -> None:
        """Add an edge to the set.  Replaces any existing edge with the same key."""
        old = self._edges.get(edge.key)
        if old is not None:
            self._remove_from_indices(old)
        self._edges[edge.key] = edge
        self._add_to_indices(edge)

    def remove(self, edge: PipelineEdge) -> None:
        """Remove an edge from the set.  Raises KeyError if not present."""
        key = edge.key
        if key not in self._edges:
            raise KeyError(f"Edge not found: {key}")
        self._remove_from_indices(self._edges[key])
        del self._edges[key]

    def discard(self, edge: PipelineEdge) -> None:
        """Remove an edge if present; do nothing if absent."""
        key = edge.key
        if key in self._edges:
            self._remove_from_indices(self._edges[key])
            del self._edges[key]

    def remove_by_node(self, node_id: str) -> list[PipelineEdge]:
        """Remove all edges incident to *node_id*, returning removed edges."""
        removed: list[PipelineEdge] = []
        keys_to_remove: set[Tuple[str, str, EdgeKind]] = set()
        keys_to_remove.update(self._by_source.get(node_id, set()))
        keys_to_remove.update(self._by_target.get(node_id, set()))
        for key in keys_to_remove:
            if key in self._edges:
                removed.append(self._edges[key])
                self._remove_from_indices(self._edges[key])
                del self._edges[key]
        return removed

    def clear(self) -> None:
        """Remove all edges."""
        self._edges.clear()
        self._by_source.clear()
        self._by_target.clear()
        self._by_column.clear()
        self._by_kind.clear()
        self._by_pair.clear()

    def update_edge(self, edge: PipelineEdge) -> None:
        """Replace the edge with matching key, or add if new."""
        self.add(edge)

    # -- index maintenance ---------------------------------------------------

    def _add_to_indices(self, edge: PipelineEdge) -> None:
        key = edge.key
        self._by_source[edge.source_id].add(key)
        self._by_target[edge.target_id].add(key)
        self._by_kind[edge.edge_kind].add(key)
        self._by_pair[edge.pair].add(key)
        for col in edge.columns:
            self._by_column[col].add(key)

    def _remove_from_indices(self, edge: PipelineEdge) -> None:
        key = edge.key
        self._by_source[edge.source_id].discard(key)
        self._by_target[edge.target_id].discard(key)
        self._by_kind[edge.edge_kind].discard(key)
        self._by_pair[edge.pair].discard(key)
        for col in edge.columns:
            self._by_column[col].discard(key)

    # -- lookup --------------------------------------------------------------

    def by_source(self, source_id: str) -> list[PipelineEdge]:
        """Return all edges originating from *source_id*."""
        return [self._edges[k] for k in self._by_source.get(source_id, set()) if k in self._edges]

    def by_target(self, target_id: str) -> list[PipelineEdge]:
        """Return all edges targeting *target_id*."""
        return [self._edges[k] for k in self._by_target.get(target_id, set()) if k in self._edges]

    def by_column(self, column: str) -> list[PipelineEdge]:
        """Return all edges carrying *column*."""
        return [self._edges[k] for k in self._by_column.get(column, set()) if k in self._edges]

    def filter_by_kind(self, kind: EdgeKind) -> list[PipelineEdge]:
        """Return all edges of the given kind."""
        return [self._edges[k] for k in self._by_kind.get(kind, set()) if k in self._edges]

    def by_pair(self, source_id: str, target_id: str) -> list[PipelineEdge]:
        """Return all edges between a specific source and target."""
        pair = (source_id, target_id)
        return [self._edges[k] for k in self._by_pair.get(pair, set()) if k in self._edges]

    def get(self, source_id: str, target_id: str, kind: EdgeKind = EdgeKind.DATA_FLOW) -> PipelineEdge | None:
        """Get a specific edge by its key, or None."""
        return self._edges.get((source_id, target_id, kind))

    def contains(self, source_id: str, target_id: str, kind: EdgeKind | None = None) -> bool:
        """Check if an edge exists between source and target."""
        if kind is not None:
            return (source_id, target_id, kind) in self._edges
        pair = (source_id, target_id)
        return bool(self._by_pair.get(pair))

    def has_edge(self, edge: PipelineEdge) -> bool:
        """Check if this exact edge (by key) is in the set."""
        return edge.key in self._edges

    # -- iteration -----------------------------------------------------------

    def __iter__(self) -> Iterator[PipelineEdge]:
        return iter(self._edges.values())

    def __len__(self) -> int:
        return len(self._edges)

    def __bool__(self) -> bool:
        return bool(self._edges)

    def __contains__(self, edge: object) -> bool:
        if isinstance(edge, PipelineEdge):
            return edge.key in self._edges
        return False

    def edges(self) -> list[PipelineEdge]:
        """Return all edges as a list."""
        return list(self._edges.values())

    def source_ids(self) -> set[str]:
        """Return all distinct source node IDs."""
        return {e.source_id for e in self._edges.values()}

    def target_ids(self) -> set[str]:
        """Return all distinct target node IDs."""
        return {e.target_id for e in self._edges.values()}

    def all_node_ids(self) -> set[str]:
        """Return all node IDs referenced by edges."""
        return self.source_ids() | self.target_ids()

    def all_columns(self) -> set[str]:
        """Return all column names flowing through any edge."""
        cols: set[str] = set()
        for e in self._edges.values():
            cols.update(e.columns)
        return cols

    # -- aggregation ---------------------------------------------------------

    def total_capacity(self) -> float:
        """Sum of capacity across all edges."""
        return sum(e.capacity for e in self._edges.values())

    def max_provenance_fraction(self) -> float:
        """Maximum provenance fraction across all edges."""
        if not self._edges:
            return 0.0
        return max(e.provenance_fraction for e in self._edges.values())

    def data_flow_edges(self) -> list[PipelineEdge]:
        """Shorthand: all DATA_FLOW edges."""
        return self.filter_by_kind(EdgeKind.DATA_FLOW)

    def fit_dependency_edges(self) -> list[PipelineEdge]:
        """Shorthand: all FIT_DEPENDENCY edges."""
        return self.filter_by_kind(EdgeKind.FIT_DEPENDENCY)

    # -- bulk operations -----------------------------------------------------

    def restrict_to_nodes(self, node_ids: set[str]) -> "EdgeSet":
        """Return a new EdgeSet with only edges between nodes in *node_ids*."""
        restricted = EdgeSet()
        for e in self._edges.values():
            if e.source_id in node_ids and e.target_id in node_ids:
                restricted.add(e)
        return restricted

    def restrict_to_columns(self, columns: frozenset[str]) -> "EdgeSet":
        """Return new EdgeSet with edge columns restricted to *columns*."""
        restricted = EdgeSet()
        for e in self._edges.values():
            new_e = e.restrict_columns(columns)
            if new_e.columns:
                restricted.add(new_e)
        return restricted

    def merge_parallel_edges(self) -> "EdgeSet":
        """Merge edges between the same (source, target) pair into one per pair."""
        merged = EdgeSet()
        pair_groups: dict[Tuple[str, str], list[PipelineEdge]] = defaultdict(list)
        for e in self._edges.values():
            pair_groups[e.pair].append(e)
        for pair, group in pair_groups.items():
            result = group[0]
            for e in group[1:]:
                result = PipelineEdge(
                    source_id=result.source_id,
                    target_id=result.target_id,
                    columns=result.columns | e.columns,
                    edge_kind=EdgeKind.DATA_FLOW,
                    capacity=result.capacity + e.capacity,
                    provenance_fraction=max(result.provenance_fraction, e.provenance_fraction),
                    metadata={**result.metadata, **e.metadata},
                )
            merged.add(result)
        return merged

    # -- serialization -------------------------------------------------------

    def to_list(self) -> list[dict[str, Any]]:
        """Serialize all edges to a list of dicts."""
        return [e.to_dict() for e in self._edges.values()]

    @classmethod
    def from_list(cls, data: Sequence[Mapping[str, Any]]) -> "EdgeSet":
        """Deserialize from a list of dicts."""
        es = cls()
        for d in data:
            es.add(PipelineEdge.from_dict(d))
        return es

    def clone(self) -> "EdgeSet":
        """Return a deep copy."""
        return EdgeSet(copy.deepcopy(e) for e in self._edges.values())

    # -- repr ----------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"EdgeSet({len(self)} edges, "
            f"{len(self.source_ids())} sources, "
            f"{len(self.target_ids())} targets)"
        )

    def summary(self) -> str:
        """Human-readable summary of the edge set."""
        lines = [f"EdgeSet: {len(self)} edges"]
        for kind in EdgeKind:
            count = len(self.filter_by_kind(kind))
            if count > 0:
                lines.append(f"  {kind.name}: {count}")
        lines.append(f"  Total capacity: {self.total_capacity():.2f} bits")
        lines.append(f"  Max ρ: {self.max_provenance_fraction():.4f}")
        return "\n".join(lines)
