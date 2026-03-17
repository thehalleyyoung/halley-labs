"""
taintflow.dag.edges -- Edge representations for the Pipeline Information DAG.

Edges in the PI-DAG encode the data-flow, fit-dependency, parameter-flow,
and control-flow relationships between pipeline operations.  Each edge
carries a column mapping (source columns → target columns), an optional
weight representing channel capacity in bits, and arbitrary metadata.

The :class:`EdgeSet` provides indexed storage and efficient lookup of edges
by source node, target node, edge kind, or arbitrary predicate.
"""

from __future__ import annotations

import copy
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from typing import (
    Any,
    Callable,
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
    Union,
)

from taintflow.core.types import EdgeKind, OpType, Origin, ShapeMetadata, ProvenanceInfo


# ===================================================================
#  Column mapping
# ===================================================================


@dataclass(frozen=True)
class ColumnMapping:
    """Describes how columns map from the source node to the target node.

    Attributes:
        mapping: Dict mapping source column names to target column names.
            An empty mapping means *identity* (all columns pass through).
        passthrough: Columns that pass through unchanged (not renamed).
        created: Columns created at the target that have no source.
        dropped: Source columns that are consumed and not forwarded.
    """

    mapping: Dict[str, str] = field(default_factory=dict)
    passthrough: FrozenSet[str] = field(default_factory=frozenset)
    created: FrozenSet[str] = field(default_factory=frozenset)
    dropped: FrozenSet[str] = field(default_factory=frozenset)

    @property
    def is_identity(self) -> bool:
        """Return True if every source column maps to itself."""
        return (
            not self.mapping
            and not self.created
            and not self.dropped
        )

    @property
    def source_columns(self) -> FrozenSet[str]:
        """All columns referenced on the source side."""
        return frozenset(self.mapping.keys()) | self.passthrough | self.dropped

    @property
    def target_columns(self) -> FrozenSet[str]:
        """All columns present on the target side."""
        return frozenset(self.mapping.values()) | self.passthrough | self.created

    def resolve(self, column: str) -> Optional[str]:
        """Resolve a source column name to its target column name.

        Returns None if the column is dropped.
        """
        if column in self.dropped:
            return None
        if column in self.mapping:
            return self.mapping[column]
        if column in self.passthrough or self.is_identity:
            return column
        return None

    def validate(self) -> List[str]:
        """Return a list of validation errors, empty if valid."""
        errors: List[str] = []
        overlap = self.passthrough & frozenset(self.mapping.keys())
        if overlap:
            errors.append(
                f"Columns in both passthrough and mapping: {sorted(overlap)}"
            )
        overlap_drop = self.passthrough & self.dropped
        if overlap_drop:
            errors.append(
                f"Columns in both passthrough and dropped: {sorted(overlap_drop)}"
            )
        overlap_create_pass = self.created & self.passthrough
        if overlap_create_pass:
            errors.append(
                f"Columns in both created and passthrough: {sorted(overlap_create_pass)}"
            )
        return errors

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {}
        if self.mapping:
            d["mapping"] = dict(self.mapping)
        if self.passthrough:
            d["passthrough"] = sorted(self.passthrough)
        if self.created:
            d["created"] = sorted(self.created)
        if self.dropped:
            d["dropped"] = sorted(self.dropped)
        return d

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> ColumnMapping:
        return cls(
            mapping=dict(d.get("mapping", {})),
            passthrough=frozenset(d.get("passthrough", [])),
            created=frozenset(d.get("created", [])),
            dropped=frozenset(d.get("dropped", [])),
        )

    def __repr__(self) -> str:
        if self.is_identity:
            return "ColumnMapping(identity)"
        parts: List[str] = []
        if self.mapping:
            parts.append(f"rename={len(self.mapping)}")
        if self.passthrough:
            parts.append(f"pass={len(self.passthrough)}")
        if self.created:
            parts.append(f"new={len(self.created)}")
        if self.dropped:
            parts.append(f"drop={len(self.dropped)}")
        return f"ColumnMapping({', '.join(parts)})"


# ===================================================================
#  DAGEdge base
# ===================================================================


@dataclass
class DAGEdge:
    """A directed edge in the Pipeline Information DAG.

    Attributes:
        source_id: Node ID of the edge's origin.
        target_id: Node ID of the edge's destination.
        edge_kind: Kind of relationship this edge represents.
        column_mapping: How columns map from source to target.
        weight: Channel capacity bound in bits (≥ 0).
        metadata: Arbitrary key-value metadata.
        edge_id: Unique identifier for this edge.
        label: Human-readable label for visualisation.
    """

    source_id: str
    target_id: str
    edge_kind: EdgeKind = EdgeKind.DATA_FLOW
    column_mapping: Optional[ColumnMapping] = None
    weight: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    edge_id: str = field(default_factory=lambda: str(uuid.uuid4())[:12])
    label: str = ""

    def __post_init__(self) -> None:
        if not self.label:
            self.label = f"{self.edge_kind.value}:{self.source_id[:6]}→{self.target_id[:6]}"

    # -- properties ----------------------------------------------------------

    @property
    def key(self) -> Tuple[str, str, EdgeKind]:
        """Return the natural key ``(source_id, target_id, edge_kind)``."""
        return (self.source_id, self.target_id, self.edge_kind)

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

    # -- validation ----------------------------------------------------------

    def validate(self) -> List[str]:
        """Validate edge consistency, returning a list of error messages."""
        errors: List[str] = []
        if not self.source_id:
            errors.append("Edge must have a non-empty source_id.")
        if not self.target_id:
            errors.append("Edge must have a non-empty target_id.")
        if self.source_id == self.target_id:
            errors.append("Self-loops are not permitted in a DAG.")
        if self.weight < 0.0:
            errors.append(f"Edge weight must be non-negative, got {self.weight}.")
        if self.column_mapping is not None:
            errors.extend(self.column_mapping.validate())
        return errors

    # -- serialisation -------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a JSON-compatible dictionary."""
        result: Dict[str, Any] = {
            "edge_id": self.edge_id,
            "source_id": self.source_id,
            "target_id": self.target_id,
            "edge_kind": self.edge_kind.value,
            "weight": self.weight,
            "label": self.label,
        }
        if self.column_mapping is not None:
            result["column_mapping"] = self.column_mapping.to_dict()
        if self.metadata:
            result["metadata"] = dict(self.metadata)
        return result

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> DAGEdge:
        """Deserialize from a dictionary.

        Dispatches to specialized subclasses based on ``edge_kind`` when
        the dictionary matches a known specialised type.
        """
        kind = EdgeKind(d["edge_kind"])
        col_map = None
        if "column_mapping" in d:
            col_map = ColumnMapping.from_dict(d["column_mapping"])

        base_kwargs: Dict[str, Any] = dict(
            source_id=d["source_id"],
            target_id=d["target_id"],
            edge_kind=kind,
            column_mapping=col_map,
            weight=float(d.get("weight", 0.0)),
            metadata=dict(d.get("metadata", {})),
            edge_id=d.get("edge_id", str(uuid.uuid4())[:12]),
            label=d.get("label", ""),
        )

        dispatch: Dict[EdgeKind, type] = {
            EdgeKind.DATA_FLOW: DataFlowEdge,
            EdgeKind.FIT_DEPENDENCY: FitDependencyEdge,
            EdgeKind.PARAMETER_FLOW: ParameterFlowEdge,
            EdgeKind.CONTROL_FLOW: ControlFlowEdge,
        }
        target_cls = dispatch.get(kind, cls)

        if target_cls is DataFlowEdge:
            return DataFlowEdge(
                **base_kwargs,
                provenance=ProvenanceInfo.from_dict(d["provenance"]) if "provenance" in d else None,
            )
        if target_cls is FitDependencyEdge:
            return FitDependencyEdge(
                **base_kwargs,
                fitted_params=list(d.get("fitted_params", [])),
                estimator_class=d.get("estimator_class", ""),
            )
        if target_cls is ParameterFlowEdge:
            return ParameterFlowEdge(
                **base_kwargs,
                param_names=list(d.get("param_names", [])),
            )
        if target_cls is ControlFlowEdge:
            return ControlFlowEdge(
                **base_kwargs,
                condition=d.get("condition", ""),
            )
        return cls(**base_kwargs)

    def reversed(self) -> DAGEdge:
        """Return a copy with source and target swapped."""
        new = copy.copy(self)
        object.__setattr__(new, "source_id", self.target_id) if False else None
        new_edge = DAGEdge(
            source_id=self.target_id,
            target_id=self.source_id,
            edge_kind=self.edge_kind,
            column_mapping=self.column_mapping,
            weight=self.weight,
            metadata=dict(self.metadata),
            edge_id=self.edge_id,
            label=self.label,
        )
        return new_edge

    def __repr__(self) -> str:
        return (
            f"DAGEdge(id={self.edge_id!r}, {self.source_id!r}→{self.target_id!r}, "
            f"kind={self.edge_kind.value}, w={self.weight:.2f})"
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, DAGEdge):
            return NotImplemented
        return self.edge_id == other.edge_id

    def __hash__(self) -> int:
        return hash(self.edge_id)


# ===================================================================
#  Specialised edge types
# ===================================================================


@dataclass
class DataFlowEdge(DAGEdge):
    """Edge representing direct data flow between nodes.

    This is the primary edge type, representing the passage of DataFrame
    rows/columns from one operation to the next.

    Attributes:
        provenance: Row-level provenance information for data flowing
            along this edge (train/test fractions).
    """

    provenance: Optional[ProvenanceInfo] = None

    def __post_init__(self) -> None:
        self.edge_kind = EdgeKind.DATA_FLOW
        super().__post_init__()

    def to_dict(self) -> Dict[str, Any]:
        d = super().to_dict()
        if self.provenance is not None:
            d["provenance"] = self.provenance.to_dict()
        return d

    def __repr__(self) -> str:
        prov = f", ρ={self.provenance.test_fraction:.3f}" if self.provenance else ""
        return (
            f"DataFlowEdge({self.source_id!r}→{self.target_id!r}"
            f"{prov}, w={self.weight:.2f})"
        )


@dataclass
class FitDependencyEdge(DAGEdge):
    """Edge representing a fit dependency.

    Connects a ``fit()`` call to a subsequent ``transform()`` or
    ``predict()`` call, encoding the information channel through fitted
    parameters (the fit-transform decomposition lemma).

    Attributes:
        fitted_params: Names of the fitted parameters transferred.
        estimator_class: Fully qualified class name of the estimator.
    """

    fitted_params: List[str] = field(default_factory=list)
    estimator_class: str = ""

    def __post_init__(self) -> None:
        self.edge_kind = EdgeKind.FIT_DEPENDENCY
        super().__post_init__()

    def to_dict(self) -> Dict[str, Any]:
        d = super().to_dict()
        if self.fitted_params:
            d["fitted_params"] = list(self.fitted_params)
        if self.estimator_class:
            d["estimator_class"] = self.estimator_class
        return d

    def __repr__(self) -> str:
        cls_name = self.estimator_class.rsplit(".", 1)[-1] if self.estimator_class else "?"
        return (
            f"FitDependencyEdge({self.source_id!r}→{self.target_id!r}, "
            f"estimator={cls_name}, params={len(self.fitted_params)})"
        )


@dataclass
class ParameterFlowEdge(DAGEdge):
    """Edge representing parameter/hyperparameter flow.

    Connects a node that produces parameters (e.g. grid search) to a node
    that consumes them.

    Attributes:
        param_names: Names of the parameters being transferred.
    """

    param_names: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.edge_kind = EdgeKind.PARAMETER_FLOW
        super().__post_init__()

    def to_dict(self) -> Dict[str, Any]:
        d = super().to_dict()
        if self.param_names:
            d["param_names"] = list(self.param_names)
        return d

    def __repr__(self) -> str:
        return (
            f"ParameterFlowEdge({self.source_id!r}→{self.target_id!r}, "
            f"params={self.param_names})"
        )


@dataclass
class ControlFlowEdge(DAGEdge):
    """Edge representing a control-flow dependency.

    Encodes ordering constraints that do not carry data (e.g. a train/test
    split must happen before either branch is processed).

    Attributes:
        condition: Optional human-readable condition description.
    """

    condition: str = ""

    def __post_init__(self) -> None:
        self.edge_kind = EdgeKind.CONTROL_FLOW
        super().__post_init__()

    def to_dict(self) -> Dict[str, Any]:
        d = super().to_dict()
        if self.condition:
            d["condition"] = self.condition
        return d

    def __repr__(self) -> str:
        cond = f", cond={self.condition!r}" if self.condition else ""
        return (
            f"ControlFlowEdge({self.source_id!r}→{self.target_id!r}{cond})"
        )


# ===================================================================
#  EdgeSet: indexed collection of edges
# ===================================================================


class EdgeSet:
    """Indexed, iterable collection of :class:`DAGEdge` objects.

    Maintains secondary indexes on ``source_id``, ``target_id``, and
    ``edge_kind`` for O(1) lookup of edge subsets.
    """

    def __init__(self, edges: Optional[Iterable[DAGEdge]] = None) -> None:
        self._edges: Dict[str, DAGEdge] = {}
        self._by_source: Dict[str, Set[str]] = defaultdict(set)
        self._by_target: Dict[str, Set[str]] = defaultdict(set)
        self._by_kind: Dict[EdgeKind, Set[str]] = defaultdict(set)
        self._by_key: Dict[Tuple[str, str, EdgeKind], str] = {}

        if edges is not None:
            for e in edges:
                self.add(e)

    # -- mutators ------------------------------------------------------------

    def add(self, edge: DAGEdge) -> None:
        """Add an edge to the set.

        Raises ValueError if an edge with the same ``edge_id`` already exists
        or if the edge fails validation.
        """
        if edge.edge_id in self._edges:
            raise ValueError(f"Duplicate edge ID: {edge.edge_id!r}")
        errors = edge.validate()
        if errors:
            raise ValueError(
                f"Invalid edge {edge.edge_id!r}: {'; '.join(errors)}"
            )
        self._edges[edge.edge_id] = edge
        self._by_source[edge.source_id].add(edge.edge_id)
        self._by_target[edge.target_id].add(edge.edge_id)
        self._by_kind[edge.edge_kind].add(edge.edge_id)
        self._by_key[edge.key] = edge.edge_id

    def remove(self, edge_id: str) -> DAGEdge:
        """Remove and return an edge by its ID.

        Raises KeyError if the edge does not exist.
        """
        edge = self._edges.pop(edge_id)
        self._by_source[edge.source_id].discard(edge_id)
        if not self._by_source[edge.source_id]:
            del self._by_source[edge.source_id]
        self._by_target[edge.target_id].discard(edge_id)
        if not self._by_target[edge.target_id]:
            del self._by_target[edge.target_id]
        self._by_kind[edge.edge_kind].discard(edge_id)
        if not self._by_kind[edge.edge_kind]:
            del self._by_kind[edge.edge_kind]
        self._by_key.pop(edge.key, None)
        return edge

    def remove_by_node(self, node_id: str) -> List[DAGEdge]:
        """Remove all edges incident to a node (as source or target).

        Returns the list of removed edges.
        """
        ids_to_remove: Set[str] = set()
        ids_to_remove.update(self._by_source.get(node_id, set()))
        ids_to_remove.update(self._by_target.get(node_id, set()))
        removed: List[DAGEdge] = []
        for eid in ids_to_remove:
            if eid in self._edges:
                removed.append(self.remove(eid))
        return removed

    def clear(self) -> None:
        """Remove all edges."""
        self._edges.clear()
        self._by_source.clear()
        self._by_target.clear()
        self._by_kind.clear()
        self._by_key.clear()

    # -- queries -------------------------------------------------------------

    def get(self, edge_id: str) -> Optional[DAGEdge]:
        """Return an edge by its ID, or None."""
        return self._edges.get(edge_id)

    def __getitem__(self, edge_id: str) -> DAGEdge:
        return self._edges[edge_id]

    def __contains__(self, edge_id: str) -> bool:
        return edge_id in self._edges

    def has_edge(self, source_id: str, target_id: str, kind: Optional[EdgeKind] = None) -> bool:
        """Return True if an edge from *source_id* to *target_id* exists.

        If *kind* is given, only that edge kind is checked.
        """
        if kind is not None:
            return (source_id, target_id, kind) in self._by_key
        for ek in EdgeKind:
            if (source_id, target_id, ek) in self._by_key:
                return True
        return False

    def lookup_by_key(
        self, source_id: str, target_id: str, kind: EdgeKind,
    ) -> Optional[DAGEdge]:
        """Return the edge matching the natural key, or None."""
        eid = self._by_key.get((source_id, target_id, kind))
        if eid is not None:
            return self._edges.get(eid)
        return None

    def from_source(self, source_id: str) -> List[DAGEdge]:
        """Return all edges originating from *source_id*."""
        return [
            self._edges[eid]
            for eid in self._by_source.get(source_id, set())
            if eid in self._edges
        ]

    def to_target(self, target_id: str) -> List[DAGEdge]:
        """Return all edges arriving at *target_id*."""
        return [
            self._edges[eid]
            for eid in self._by_target.get(target_id, set())
            if eid in self._edges
        ]

    def by_kind(self, kind: EdgeKind) -> List[DAGEdge]:
        """Return all edges of a given kind."""
        return [
            self._edges[eid]
            for eid in self._by_kind.get(kind, set())
            if eid in self._edges
        ]

    def filter(
        self,
        *,
        source_id: Optional[str] = None,
        target_id: Optional[str] = None,
        kind: Optional[EdgeKind] = None,
        predicate: Optional[Callable[[DAGEdge], bool]] = None,
    ) -> List[DAGEdge]:
        """Return edges matching all supplied criteria.

        Parameters
        ----------
        source_id : str, optional
            Filter by source node.
        target_id : str, optional
            Filter by target node.
        kind : EdgeKind, optional
            Filter by edge kind.
        predicate : callable, optional
            Arbitrary filter function.

        Returns
        -------
        list[DAGEdge]
        """
        candidates: Optional[Set[str]] = None

        if source_id is not None:
            src_set = self._by_source.get(source_id, set())
            candidates = set(src_set)
        if target_id is not None:
            tgt_set = self._by_target.get(target_id, set())
            candidates = candidates & tgt_set if candidates is not None else set(tgt_set)
        if kind is not None:
            kind_set = self._by_kind.get(kind, set())
            candidates = candidates & kind_set if candidates is not None else set(kind_set)

        if candidates is None:
            pool = self._edges.values()
        else:
            pool = [self._edges[eid] for eid in candidates if eid in self._edges]

        if predicate is not None:
            return [e for e in pool if predicate(e)]
        return list(pool)

    # -- aggregate queries ---------------------------------------------------

    def source_ids(self) -> Set[str]:
        """Return the set of all source node IDs."""
        return set(self._by_source.keys())

    def target_ids(self) -> Set[str]:
        """Return the set of all target node IDs."""
        return set(self._by_target.keys())

    def node_ids(self) -> Set[str]:
        """Return the set of all node IDs mentioned in any edge."""
        return self.source_ids() | self.target_ids()

    def out_degree(self, node_id: str) -> int:
        """Return the number of outgoing edges from *node_id*."""
        return len(self._by_source.get(node_id, set()))

    def in_degree(self, node_id: str) -> int:
        """Return the number of incoming edges to *node_id*."""
        return len(self._by_target.get(node_id, set()))

    # -- iteration -----------------------------------------------------------

    def __iter__(self) -> Iterator[DAGEdge]:
        return iter(self._edges.values())

    def __len__(self) -> int:
        return len(self._edges)

    def __bool__(self) -> bool:
        return bool(self._edges)

    # -- serialisation -------------------------------------------------------

    def to_list(self) -> List[Dict[str, Any]]:
        """Serialize all edges to a list of dicts."""
        return [e.to_dict() for e in self._edges.values()]

    @classmethod
    def from_list(cls, data: Sequence[Mapping[str, Any]]) -> EdgeSet:
        """Deserialize from a list of dicts."""
        edge_set = cls()
        for d in data:
            edge_set.add(DAGEdge.from_dict(d))
        return edge_set

    def __repr__(self) -> str:
        kind_counts = {k.value: len(ids) for k, ids in self._by_kind.items()}
        return f"EdgeSet(n={len(self)}, kinds={kind_counts})"


# ===================================================================
#  Public API
# ===================================================================

__all__ = [
    "ColumnMapping",
    "DAGEdge",
    "DataFlowEdge",
    "FitDependencyEdge",
    "ParameterFlowEdge",
    "ControlFlowEdge",
    "EdgeSet",
]
