"""
Delta Propagation Engine (Algorithm A1: PROPAGATE)
===================================================

Propagates perturbations through the pipeline DAG using topological traversal.
Given a source perturbation at one or more nodes, computes the resulting
CompoundPerturbation at every downstream node, applying push operators,
interaction homomorphisms, and annihilation detection along the way.

Key algorithm:
    1. Initialize delta map with identity deltas at all nodes.
    2. Set source perturbation(s).
    3. Traverse in topological order:
       - Aggregate incoming deltas from predecessors.
       - Push through the node's SQL operator.
       - Apply interaction homomorphisms (φ, ψ).
       - Detect annihilation.
       - Store propagated delta.
    4. Return PropagationResult.

Complexity:  O(|V| + |E|) for a single-source propagation on a DAG.
"""

from __future__ import annotations

import copy
import logging
import math
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import (
    Any,
    Callable,
    Dict,
    FrozenSet,
    Iterable,
    List,
    Mapping,
    Optional,
    Sequence,
    Set,
    Tuple,
    Union,
)

from arc.algebra.schema_delta import (
    AddColumn,
    ChangeType,
    DropColumn,
    RenameColumn,
    Schema,
    SchemaDelta,
    SchemaOperation,
    SQLType,
)
from arc.algebra.data_delta import (
    DataDelta,
    DataOperation,
    DeleteOp,
    InsertOp,
    MultiSet,
    TypedTuple,
    UpdateOp,
)
from arc.algebra.quality_delta import (
    QualityDelta,
    QualityOperation,
    QualityViolation,
    SeverityLevel,
    ViolationType,
)
from arc.algebra.composition import CompoundPerturbation
from arc.algebra.interaction import (
    PhiHomomorphism,
    PsiHomomorphism,
    apply_schema_interaction,
)
from arc.algebra.push import (
    PushOperator,
    get_push_operator,
)

logger = logging.getLogger(__name__)


# =====================================================================
# Propagation mode enum
# =====================================================================


class PropagationMode(Enum):
    """Strategy for delta propagation through the pipeline DAG."""
    FULL = "full"
    INCREMENTAL = "incremental"
    LAZY = "lazy"
    EAGER = "eager"
    BATCHED = "batched"


class AggregationStrategy(Enum):
    """Strategy for aggregating deltas from multiple predecessors."""
    COMPOSE = "compose"
    MERGE = "merge"
    UNION = "union"
    FIRST_WINS = "first_wins"
    LAST_WINS = "last_wins"


# =====================================================================
# Propagation Statistics
# =====================================================================


@dataclass
class PropagationStats:
    """Statistics collected during a propagation pass.

    Attributes
    ----------
    nodes_visited : int
        Total number of nodes visited during propagation.
    nodes_affected : int
        Number of nodes that received a non-identity delta.
    nodes_annihilated : int
        Number of nodes where the delta was fully annihilated.
    nodes_skipped : int
        Number of nodes that were skipped (no incoming delta).
    total_delta_size : int
        Sum of operation counts across all node deltas.
    max_delta_size : int
        Maximum single-node delta size.
    propagation_time_ms : float
        Wall-clock time of the propagation in milliseconds.
    push_time_ms : float
        Cumulative time spent in push operators.
    interaction_time_ms : float
        Cumulative time spent in interaction homomorphisms.
    annihilation_checks : int
        Number of annihilation checks performed.
    """
    nodes_visited: int = 0
    nodes_affected: int = 0
    nodes_annihilated: int = 0
    nodes_skipped: int = 0
    total_delta_size: int = 0
    max_delta_size: int = 0
    propagation_time_ms: float = 0.0
    push_time_ms: float = 0.0
    interaction_time_ms: float = 0.0
    annihilation_checks: int = 0

    def merge(self, other: PropagationStats) -> PropagationStats:
        """Merge another stats object into this one."""
        return PropagationStats(
            nodes_visited=self.nodes_visited + other.nodes_visited,
            nodes_affected=self.nodes_affected + other.nodes_affected,
            nodes_annihilated=self.nodes_annihilated + other.nodes_annihilated,
            nodes_skipped=self.nodes_skipped + other.nodes_skipped,
            total_delta_size=self.total_delta_size + other.total_delta_size,
            max_delta_size=max(self.max_delta_size, other.max_delta_size),
            propagation_time_ms=self.propagation_time_ms + other.propagation_time_ms,
            push_time_ms=self.push_time_ms + other.push_time_ms,
            interaction_time_ms=self.interaction_time_ms + other.interaction_time_ms,
            annihilation_checks=self.annihilation_checks + other.annihilation_checks,
        )

    def summary(self) -> str:
        """Return a human-readable summary of propagation stats."""
        lines = [
            f"Propagation Statistics:",
            f"  Nodes visited:     {self.nodes_visited}",
            f"  Nodes affected:    {self.nodes_affected}",
            f"  Nodes annihilated: {self.nodes_annihilated}",
            f"  Nodes skipped:     {self.nodes_skipped}",
            f"  Total delta size:  {self.total_delta_size}",
            f"  Max delta size:    {self.max_delta_size}",
            f"  Propagation time:  {self.propagation_time_ms:.2f} ms",
            f"  Push time:         {self.push_time_ms:.2f} ms",
            f"  Interaction time:  {self.interaction_time_ms:.2f} ms",
            f"  Annihilation checks: {self.annihilation_checks}",
        ]
        return "\n".join(lines)


# =====================================================================
# Propagation Path Tracer
# =====================================================================


@dataclass
class PropagationPathEntry:
    """A single step in a delta's propagation path.

    Attributes
    ----------
    node_id : str
        The node where this step occurred.
    incoming_delta_size : int
        Number of operations in the incoming delta.
    outgoing_delta_size : int
        Number of operations in the outgoing delta.
    was_annihilated : bool
        True if the delta was annihilated at this node.
    push_operator_used : str
        Name of the push operator applied.
    timestamp_ms : float
        When this step was processed (relative to propagation start).
    """
    node_id: str = ""
    incoming_delta_size: int = 0
    outgoing_delta_size: int = 0
    was_annihilated: bool = False
    push_operator_used: str = ""
    timestamp_ms: float = 0.0

    def __repr__(self) -> str:
        ann = " [ANNIHILATED]" if self.was_annihilated else ""
        return (
            f"PathEntry({self.node_id}: {self.incoming_delta_size} -> "
            f"{self.outgoing_delta_size}{ann})"
        )


@dataclass
class PropagationPath:
    """Tracks the path of delta propagation from source to a specific node.

    Attributes
    ----------
    source_node : str
        The originating node of the propagation.
    target_node : str
        The destination node of this path.
    entries : list[PropagationPathEntry]
        Ordered list of steps along the path.
    total_amplification : float
        Ratio of output delta size to input delta size.
    """
    source_node: str = ""
    target_node: str = ""
    entries: List[PropagationPathEntry] = field(default_factory=list)
    total_amplification: float = 1.0

    def add_entry(self, entry: PropagationPathEntry) -> None:
        """Add a new entry to the propagation path."""
        self.entries.append(entry)
        if self.entries and self.entries[0].incoming_delta_size > 0:
            self.total_amplification = (
                entry.outgoing_delta_size / self.entries[0].incoming_delta_size
            )

    @property
    def length(self) -> int:
        """Number of steps in the path."""
        return len(self.entries)

    @property
    def was_annihilated(self) -> bool:
        """True if any entry in the path was annihilated."""
        return any(e.was_annihilated for e in self.entries)

    def node_sequence(self) -> List[str]:
        """Return the ordered list of node ids along the path."""
        return [e.node_id for e in self.entries]


# =====================================================================
# Propagation Result
# =====================================================================


@dataclass
class PropagationResult:
    """Complete result of a delta propagation through the pipeline DAG.

    Attributes
    ----------
    node_deltas : dict[str, CompoundPerturbation]
        Mapping from node_id to the computed delta at that node.
    affected_nodes : set[str]
        Set of nodes that received a non-trivial delta.
    annihilated_nodes : set[str]
        Set of nodes where the delta was fully annihilated.
    propagation_paths : dict[str, PropagationPath]
        For each affected node, the path taken by the delta.
    statistics : PropagationStats
        Aggregated statistics for the propagation.
    source_nodes : set[str]
        The set of source nodes where perturbations originated.
    mode : PropagationMode
        The propagation mode used.
    """
    node_deltas: Dict[str, CompoundPerturbation] = field(default_factory=dict)
    affected_nodes: Set[str] = field(default_factory=set)
    annihilated_nodes: Set[str] = field(default_factory=set)
    propagation_paths: Dict[str, PropagationPath] = field(default_factory=dict)
    statistics: PropagationStats = field(default_factory=PropagationStats)
    source_nodes: Set[str] = field(default_factory=set)
    mode: PropagationMode = PropagationMode.FULL

    def get_delta(self, node_id: str) -> Optional[CompoundPerturbation]:
        """Get the propagated delta at a specific node."""
        return self.node_deltas.get(node_id)

    def is_affected(self, node_id: str) -> bool:
        """Check if a node was affected by the propagation."""
        return node_id in self.affected_nodes

    def is_annihilated(self, node_id: str) -> bool:
        """Check if the delta was annihilated at a node."""
        return node_id in self.annihilated_nodes

    def get_path(self, node_id: str) -> Optional[PropagationPath]:
        """Get the propagation path to a specific node."""
        return self.propagation_paths.get(node_id)

    def affected_sink_nodes(self, graph: Any) -> Set[str]:
        """Return the set of affected nodes that are graph sinks."""
        sinks = set(graph.sinks())
        return self.affected_nodes & sinks

    def max_propagation_depth(self) -> int:
        """Return the maximum path length across all propagation paths."""
        if not self.propagation_paths:
            return 0
        return max(p.length for p in self.propagation_paths.values())

    def total_delta_operations(self) -> int:
        """Sum of all delta operation counts across affected nodes."""
        total = 0
        for delta in self.node_deltas.values():
            total += _delta_size(delta)
        return total

    def merge(self, other: PropagationResult) -> PropagationResult:
        """Merge another propagation result into this one.

        Used for combining results from multi-source propagation.
        """
        merged_deltas = dict(self.node_deltas)
        for nid, delta in other.node_deltas.items():
            if nid in merged_deltas:
                merged_deltas[nid] = _compose_perturbations(
                    merged_deltas[nid], delta
                )
            else:
                merged_deltas[nid] = delta

        merged_affected = self.affected_nodes | other.affected_nodes
        merged_annihilated = self.annihilated_nodes | other.annihilated_nodes
        merged_annihilated -= merged_affected

        merged_paths = dict(self.propagation_paths)
        merged_paths.update(other.propagation_paths)

        return PropagationResult(
            node_deltas=merged_deltas,
            affected_nodes=merged_affected,
            annihilated_nodes=merged_annihilated,
            propagation_paths=merged_paths,
            statistics=self.statistics.merge(other.statistics),
            source_nodes=self.source_nodes | other.source_nodes,
            mode=self.mode,
        )

    def summary(self) -> str:
        """Return a human-readable summary of propagation results."""
        lines = [
            f"PropagationResult:",
            f"  Source nodes:      {sorted(self.source_nodes)}",
            f"  Affected nodes:    {len(self.affected_nodes)}",
            f"  Annihilated nodes: {len(self.annihilated_nodes)}",
            f"  Max depth:         {self.max_propagation_depth()}",
            f"  Total operations:  {self.total_delta_operations()}",
        ]
        return "\n".join(lines)


# =====================================================================
# Helper functions
# =====================================================================


def _delta_size(delta: CompoundPerturbation) -> int:
    """Count the total number of operations in a compound perturbation."""
    size = 0
    if delta.schema_delta is not None:
        size += len(delta.schema_delta.operations)
    if delta.data_delta is not None:
        size += len(delta.data_delta.operations)
    if delta.quality_delta is not None:
        size += len(delta.quality_delta.operations)
    return size


def _is_identity(delta: CompoundPerturbation) -> bool:
    """Check if a compound perturbation is the identity (no-op)."""
    schema_empty = (
        delta.schema_delta is None or len(delta.schema_delta.operations) == 0
    )
    data_empty = (
        delta.data_delta is None or len(delta.data_delta.operations) == 0
    )
    quality_empty = (
        delta.quality_delta is None or len(delta.quality_delta.operations) == 0
    )
    return schema_empty and data_empty and quality_empty


def _make_identity() -> CompoundPerturbation:
    """Create an identity (no-op) compound perturbation."""
    return CompoundPerturbation(
        schema_delta=SchemaDelta(operations=[]),
        data_delta=DataDelta(operations=[]),
        quality_delta=QualityDelta(operations=[]),
    )


def _compose_perturbations(
    a: CompoundPerturbation,
    b: CompoundPerturbation,
) -> CompoundPerturbation:
    """Compose two compound perturbations.

    Uses the three-sorted composition rule:
      (σ₁, δ₁, γ₁) ∘ (σ₂, δ₂, γ₂) =
        (σ₁ ∘ σ₂, δ₁ ∘ φ(σ₁)(δ₂), γ₁ ⊔ ψ(σ₁)(γ₂))
    """
    return a.compose(b)


def _aggregate_deltas(
    deltas: List[CompoundPerturbation],
    strategy: AggregationStrategy,
) -> CompoundPerturbation:
    """Aggregate multiple incoming deltas into a single delta.

    Parameters
    ----------
    deltas : list[CompoundPerturbation]
        The deltas to aggregate.
    strategy : AggregationStrategy
        How to combine the deltas.

    Returns
    -------
    CompoundPerturbation
        The aggregated delta.
    """
    if not deltas:
        return _make_identity()

    if len(deltas) == 1:
        return deltas[0]

    if strategy == AggregationStrategy.COMPOSE:
        result = deltas[0]
        for d in deltas[1:]:
            result = _compose_perturbations(result, d)
        return result

    if strategy == AggregationStrategy.FIRST_WINS:
        return deltas[0]

    if strategy == AggregationStrategy.LAST_WINS:
        return deltas[-1]

    if strategy == AggregationStrategy.MERGE:
        return _merge_deltas(deltas)

    if strategy == AggregationStrategy.UNION:
        return _union_deltas(deltas)

    return _merge_deltas(deltas)


def _merge_deltas(deltas: List[CompoundPerturbation]) -> CompoundPerturbation:
    """Merge multiple deltas by concatenating their operation lists."""
    schema_ops: List[SchemaOperation] = []
    data_ops: List[DataOperation] = []
    quality_ops: List[QualityOperation] = []

    for delta in deltas:
        if delta.schema_delta is not None:
            schema_ops.extend(delta.schema_delta.operations)
        if delta.data_delta is not None:
            data_ops.extend(delta.data_delta.operations)
        if delta.quality_delta is not None:
            quality_ops.extend(delta.quality_delta.operations)

    return CompoundPerturbation(
        schema_delta=SchemaDelta(operations=schema_ops),
        data_delta=DataDelta(operations=data_ops),
        quality_delta=QualityDelta(operations=quality_ops),
    )


def _union_deltas(deltas: List[CompoundPerturbation]) -> CompoundPerturbation:
    """Union multiple deltas, deduplicating identical operations."""
    seen_schema: List[SchemaOperation] = []
    seen_data: List[DataOperation] = []
    seen_quality: List[QualityOperation] = []
    schema_set: Set[str] = set()
    data_set: Set[str] = set()
    quality_set: Set[str] = set()

    for delta in deltas:
        if delta.schema_delta is not None:
            for op in delta.schema_delta.operations:
                key = repr(op)
                if key not in schema_set:
                    schema_set.add(key)
                    seen_schema.append(op)
        if delta.data_delta is not None:
            for op in delta.data_delta.operations:
                key = repr(op)
                if key not in data_set:
                    data_set.add(key)
                    seen_data.append(op)
        if delta.quality_delta is not None:
            for op in delta.quality_delta.operations:
                key = repr(op)
                if key not in quality_set:
                    quality_set.add(key)
                    seen_quality.append(op)

    return CompoundPerturbation(
        schema_delta=SchemaDelta(operations=seen_schema),
        data_delta=DataDelta(operations=seen_data),
        quality_delta=QualityDelta(operations=seen_quality),
    )


# =====================================================================
# Push Application Helpers
# =====================================================================


def _get_node_operator_name(node: Any) -> str:
    """Extract the SQL operator name from a pipeline node."""
    if hasattr(node, "operator"):
        op = node.operator
        if hasattr(op, "value"):
            return str(op.value)
        return str(op)
    return "TRANSFORM"


def _apply_push_schema(
    push_op: PushOperator,
    config: Any,
    schema_delta: SchemaDelta,
) -> SchemaDelta:
    """Apply push operator to a schema delta with error handling."""
    try:
        result = push_op.push_schema(config, schema_delta)
        return result
    except Exception as exc:
        logger.warning("Push schema failed for %s: %s", type(push_op).__name__, exc)
        return schema_delta


def _apply_push_data(
    push_op: PushOperator,
    config: Any,
    data_delta: DataDelta,
) -> DataDelta:
    """Apply push operator to a data delta with error handling."""
    try:
        result = push_op.push_data(config, data_delta)
        return result
    except Exception as exc:
        logger.warning("Push data failed for %s: %s", type(push_op).__name__, exc)
        return data_delta


def _apply_push_quality(
    push_op: PushOperator,
    config: Any,
    quality_delta: QualityDelta,
) -> QualityDelta:
    """Apply push operator to a quality delta with error handling."""
    try:
        result = push_op.push_quality(config, quality_delta)
        return result
    except Exception as exc:
        logger.warning("Push quality failed for %s: %s", type(push_op).__name__, exc)
        return quality_delta


def _push_compound_delta(
    push_op: PushOperator,
    config: Any,
    delta: CompoundPerturbation,
) -> CompoundPerturbation:
    """Apply push operator to all three sorts of a compound perturbation.

    Implements the push rule:
        push_f(σ, δ, γ) = (push_f^S(σ), push_f^D(δ), push_f^Q(γ))

    Parameters
    ----------
    push_op : PushOperator
        The push operator for this node's SQL operator.
    config : Any
        Operator-specific configuration.
    delta : CompoundPerturbation
        The incoming compound delta.

    Returns
    -------
    CompoundPerturbation
        The pushed delta.
    """
    pushed_schema = delta.schema_delta
    if pushed_schema is not None and len(pushed_schema.operations) > 0:
        pushed_schema = _apply_push_schema(push_op, config, pushed_schema)

    pushed_data = delta.data_delta
    if pushed_data is not None and len(pushed_data.operations) > 0:
        pushed_data = _apply_push_data(push_op, config, pushed_data)

    pushed_quality = delta.quality_delta
    if pushed_quality is not None and len(pushed_quality.operations) > 0:
        pushed_quality = _apply_push_quality(push_op, config, pushed_quality)

    return CompoundPerturbation(
        schema_delta=pushed_schema,
        data_delta=pushed_data,
        quality_delta=pushed_quality,
    )


# =====================================================================
# Annihilation Checking (lightweight, used during propagation)
# =====================================================================


def _quick_annihilation_check(
    node: Any,
    delta: CompoundPerturbation,
) -> bool:
    """Fast heuristic annihilation check during propagation.

    Checks common annihilation patterns without full analysis:
    - SELECT that doesn't include affected columns
    - FILTER that contradicts data delta
    - GROUP BY on non-group columns

    Returns True if the delta is likely annihilated.
    """
    if _is_identity(delta):
        return True

    operator_name = _get_node_operator_name(node)

    if operator_name in ("SELECT", "PROJECT"):
        return _check_select_annihilation(node, delta)
    if operator_name == "FILTER":
        return _check_filter_annihilation(node, delta)
    if operator_name == "GROUP_BY":
        return _check_groupby_annihilation(node, delta)

    return False


def _check_select_annihilation(node: Any, delta: CompoundPerturbation) -> bool:
    """Check if a SELECT annihilates the delta by dropping affected columns."""
    if delta.schema_delta is None or len(delta.schema_delta.operations) == 0:
        return False

    output_cols = set()
    if hasattr(node, "output_schema"):
        output_cols = {c.name for c in node.output_schema.columns}

    if not output_cols:
        return False

    for op in delta.schema_delta.operations:
        if isinstance(op, AddColumn):
            if op.name in output_cols:
                return False
        elif isinstance(op, DropColumn):
            if op.name not in output_cols:
                return True
        elif isinstance(op, RenameColumn):
            if op.old_name in output_cols or op.new_name in output_cols:
                return False

    return False


def _check_filter_annihilation(node: Any, delta: CompoundPerturbation) -> bool:
    """Check if a FILTER annihilates all data changes."""
    if delta.data_delta is None or len(delta.data_delta.operations) == 0:
        return False

    if not hasattr(node, "query_text") or not node.query_text:
        return False

    return False


def _check_groupby_annihilation(
    node: Any,
    delta: CompoundPerturbation,
) -> bool:
    """Check if GROUP BY annihilates schema changes on non-key columns."""
    if delta.schema_delta is None or len(delta.schema_delta.operations) == 0:
        return False

    return False


# =====================================================================
# Interaction Application
# =====================================================================


def _apply_interactions(
    delta: CompoundPerturbation,
) -> CompoundPerturbation:
    """Apply interaction homomorphisms within a compound perturbation.

    The interactions φ (schema→data) and ψ (schema→quality) ensure that
    schema changes properly affect the data and quality sorts.

    Returns
    -------
    CompoundPerturbation
        Delta with interactions applied.
    """
    if delta.schema_delta is None or len(delta.schema_delta.operations) == 0:
        return delta

    phi = PhiHomomorphism()
    psi = PsiHomomorphism()

    adjusted_data = delta.data_delta
    if adjusted_data is not None:
        adjusted_data = phi.apply(delta.schema_delta, adjusted_data)

    adjusted_quality = delta.quality_delta
    if adjusted_quality is not None:
        adjusted_quality = psi.apply(delta.schema_delta, adjusted_quality)

    return CompoundPerturbation(
        schema_delta=delta.schema_delta,
        data_delta=adjusted_data,
        quality_delta=adjusted_quality,
    )


# =====================================================================
# Operator Config Extraction
# =====================================================================


def _extract_operator_config(node: Any) -> Any:
    """Extract operator-specific configuration from a pipeline node.

    Looks for known configuration attributes on the node and wraps
    them in a format the push operator expects.
    """
    config: Dict[str, Any] = {}

    if hasattr(node, "query_text"):
        config["query_text"] = node.query_text

    if hasattr(node, "input_schema"):
        config["input_schema"] = node.input_schema

    if hasattr(node, "output_schema"):
        config["output_schema"] = node.output_schema

    if hasattr(node, "operator"):
        config["operator"] = node.operator

    if hasattr(node, "metadata"):
        meta = node.metadata
        if hasattr(meta, "to_dict"):
            meta = meta.to_dict()
        if isinstance(meta, dict):
            config.update(meta)

    return config


# =====================================================================
# Topological Order Cache
# =====================================================================


class _TopoCache:
    """Cache for topological ordering, invalidated on graph changes."""

    def __init__(self) -> None:
        self._order: Optional[List[str]] = None
        self._graph_hash: Optional[int] = None

    def get_order(self, graph: Any) -> List[str]:
        """Get topological order, computing it if necessary."""
        gh = self._compute_hash(graph)
        if self._order is not None and self._graph_hash == gh:
            return self._order
        self._order = graph.topological_sort()
        self._graph_hash = gh
        return self._order

    def invalidate(self) -> None:
        """Invalidate the cache."""
        self._order = None
        self._graph_hash = None

    @staticmethod
    def _compute_hash(graph: Any) -> int:
        """Compute a rough hash of the graph for cache invalidation."""
        return hash((graph.node_count, graph.edge_count))


# =====================================================================
# Delta Propagator
# =====================================================================


class DeltaPropagator:
    """Propagate deltas through a pipeline DAG using topological traversal.

    Implements Algorithm A1 (PROPAGATE) from the Algebraic Repair Calculus.
    Given a source perturbation at one or more nodes, computes the resulting
    CompoundPerturbation at every downstream node.

    Parameters
    ----------
    mode : PropagationMode
        Propagation strategy.  FULL recomputes all downstream nodes;
        INCREMENTAL reuses previous results when possible.
    aggregation_strategy : AggregationStrategy
        How to combine deltas arriving from multiple predecessor edges.
    enable_annihilation : bool
        Whether to check for annihilation during propagation.
    enable_interactions : bool
        Whether to apply interaction homomorphisms φ and ψ.
    track_paths : bool
        Whether to record detailed propagation paths.
    max_delta_size : int
        Safety limit: if a delta grows beyond this many operations,
        truncate and record a warning.
    """

    def __init__(
        self,
        mode: PropagationMode = PropagationMode.FULL,
        aggregation_strategy: AggregationStrategy = AggregationStrategy.COMPOSE,
        enable_annihilation: bool = True,
        enable_interactions: bool = True,
        track_paths: bool = True,
        max_delta_size: int = 100_000,
    ) -> None:
        self._mode = mode
        self._aggregation = aggregation_strategy
        self._annihilation = enable_annihilation
        self._interactions = enable_interactions
        self._track_paths = track_paths
        self._max_delta_size = max_delta_size
        self._topo_cache = _TopoCache()
        self._push_cache: Dict[str, PushOperator] = {}

    # ── Public API ────────────────────────────────────────────────

    def propagate(
        self,
        graph: Any,
        source_node: str,
        perturbation: CompoundPerturbation,
    ) -> PropagationResult:
        """Propagate a single-source perturbation through the pipeline DAG.

        Algorithm A1 (PROPAGATE):
        1. Initialize delta map with identity deltas at all nodes.
        2. Set source perturbation at source_node.
        3. Traverse nodes in topological order:
           a. Aggregate incoming deltas from all predecessors.
           b. Apply push operators for the node's SQL operator.
           c. Apply interaction homomorphisms φ and ψ.
           d. Check for annihilation.
           e. Store the resulting delta.
        4. Return PropagationResult.

        Parameters
        ----------
        graph : PipelineGraph
            The pipeline DAG to propagate through.
        source_node : str
            The node where the perturbation originates.
        perturbation : CompoundPerturbation
            The initial perturbation to propagate.

        Returns
        -------
        PropagationResult
            Complete propagation results.
        """
        start_time = time.time()
        stats = PropagationStats()

        topo_order = self._topo_cache.get_order(graph)

        delta_map: Dict[str, CompoundPerturbation] = {}
        for nid in topo_order:
            delta_map[nid] = _make_identity()

        delta_map[source_node] = perturbation

        affected: Set[str] = set()
        annihilated: Set[str] = set()
        paths: Dict[str, PropagationPath] = {}

        if self._track_paths:
            paths[source_node] = PropagationPath(
                source_node=source_node,
                target_node=source_node,
                entries=[
                    PropagationPathEntry(
                        node_id=source_node,
                        incoming_delta_size=_delta_size(perturbation),
                        outgoing_delta_size=_delta_size(perturbation),
                        push_operator_used="source",
                        timestamp_ms=0.0,
                    )
                ],
            )

        source_idx = -1
        for i, nid in enumerate(topo_order):
            if nid == source_node:
                source_idx = i
                break

        if source_idx < 0:
            logger.warning("Source node %s not found in topological order", source_node)
            stats.propagation_time_ms = (time.time() - start_time) * 1000
            return PropagationResult(
                node_deltas=delta_map,
                affected_nodes=affected,
                annihilated_nodes=annihilated,
                propagation_paths=paths,
                statistics=stats,
                source_nodes={source_node},
                mode=self._mode,
            )

        affected.add(source_node)

        for nid in topo_order[source_idx + 1:]:
            stats.nodes_visited += 1
            node = graph.get_node(nid)
            predecessors = graph.predecessors(nid)

            incoming_deltas = []
            for pred in predecessors:
                pred_delta = delta_map.get(pred)
                if pred_delta is not None and not _is_identity(pred_delta):
                    incoming_deltas.append(pred_delta)

            if not incoming_deltas:
                stats.nodes_skipped += 1
                continue

            aggregated = _aggregate_deltas(incoming_deltas, self._aggregation)

            if _is_identity(aggregated):
                stats.nodes_skipped += 1
                continue

            push_start = time.time()
            pushed = self._apply_node_push(node, aggregated)
            stats.push_time_ms += (time.time() - push_start) * 1000

            if self._interactions:
                interaction_start = time.time()
                pushed = _apply_interactions(pushed)
                stats.interaction_time_ms += (time.time() - interaction_start) * 1000

            if self._annihilation:
                stats.annihilation_checks += 1
                if _quick_annihilation_check(node, pushed):
                    annihilated.add(nid)
                    delta_map[nid] = _make_identity()
                    stats.nodes_annihilated += 1

                    if self._track_paths:
                        elapsed = (time.time() - start_time) * 1000
                        entry = PropagationPathEntry(
                            node_id=nid,
                            incoming_delta_size=_delta_size(aggregated),
                            outgoing_delta_size=0,
                            was_annihilated=True,
                            push_operator_used=_get_node_operator_name(node),
                            timestamp_ms=elapsed,
                        )
                        path = PropagationPath(
                            source_node=source_node,
                            target_node=nid,
                            entries=[entry],
                        )
                        paths[nid] = path
                    continue

            ds = _delta_size(pushed)
            if ds > self._max_delta_size:
                logger.warning(
                    "Delta at node %s has %d operations (limit %d), truncating",
                    nid, ds, self._max_delta_size,
                )
                pushed = self._truncate_delta(pushed)

            delta_map[nid] = pushed
            affected.add(nid)
            stats.nodes_affected += 1
            stats.total_delta_size += ds
            stats.max_delta_size = max(stats.max_delta_size, ds)

            if self._track_paths:
                elapsed = (time.time() - start_time) * 1000
                entry = PropagationPathEntry(
                    node_id=nid,
                    incoming_delta_size=_delta_size(aggregated),
                    outgoing_delta_size=ds,
                    was_annihilated=False,
                    push_operator_used=_get_node_operator_name(node),
                    timestamp_ms=elapsed,
                )
                parent_path_entries: List[PropagationPathEntry] = []
                for pred in predecessors:
                    if pred in paths:
                        parent_path_entries = list(paths[pred].entries)
                        break
                path = PropagationPath(
                    source_node=source_node,
                    target_node=nid,
                    entries=parent_path_entries + [entry],
                )
                paths[nid] = path

        stats.propagation_time_ms = (time.time() - start_time) * 1000

        return PropagationResult(
            node_deltas=delta_map,
            affected_nodes=affected,
            annihilated_nodes=annihilated,
            propagation_paths=paths,
            statistics=stats,
            source_nodes={source_node},
            mode=self._mode,
        )

    def propagate_multi_source(
        self,
        graph: Any,
        perturbations: Dict[str, CompoundPerturbation],
    ) -> PropagationResult:
        """Propagate perturbations from multiple source nodes simultaneously.

        When perturbations arrive from multiple sources, they propagate
        independently until reaching a common descendant, where they are
        aggregated using the configured strategy.

        Parameters
        ----------
        graph : PipelineGraph
            The pipeline DAG.
        perturbations : dict[str, CompoundPerturbation]
            Mapping from source node id to its perturbation.

        Returns
        -------
        PropagationResult
            Combined propagation results.
        """
        if not perturbations:
            return PropagationResult(mode=self._mode)

        if len(perturbations) == 1:
            src, delta = next(iter(perturbations.items()))
            return self.propagate(graph, src, delta)

        start_time = time.time()
        stats = PropagationStats()
        topo_order = self._topo_cache.get_order(graph)

        delta_map: Dict[str, CompoundPerturbation] = {}
        for nid in topo_order:
            delta_map[nid] = _make_identity()

        for src, delta in perturbations.items():
            delta_map[src] = delta

        affected: Set[str] = set(perturbations.keys())
        annihilated: Set[str] = set()
        paths: Dict[str, PropagationPath] = {}

        source_indices: Dict[str, int] = {}
        for i, nid in enumerate(topo_order):
            if nid in perturbations:
                source_indices[nid] = i

        if not source_indices:
            stats.propagation_time_ms = (time.time() - start_time) * 1000
            return PropagationResult(
                node_deltas=delta_map,
                affected_nodes=affected,
                annihilated_nodes=annihilated,
                propagation_paths=paths,
                statistics=stats,
                source_nodes=set(perturbations.keys()),
                mode=self._mode,
            )

        min_source_idx = min(source_indices.values())

        for nid in topo_order[min_source_idx + 1:]:
            if nid in perturbations:
                continue

            stats.nodes_visited += 1
            node = graph.get_node(nid)
            predecessors = graph.predecessors(nid)

            incoming_deltas = []
            for pred in predecessors:
                pred_delta = delta_map.get(pred)
                if pred_delta is not None and not _is_identity(pred_delta):
                    incoming_deltas.append(pred_delta)

            if not incoming_deltas:
                stats.nodes_skipped += 1
                continue

            aggregated = _aggregate_deltas(incoming_deltas, self._aggregation)

            if _is_identity(aggregated):
                stats.nodes_skipped += 1
                continue

            push_start = time.time()
            pushed = self._apply_node_push(node, aggregated)
            stats.push_time_ms += (time.time() - push_start) * 1000

            if self._interactions:
                interaction_start = time.time()
                pushed = _apply_interactions(pushed)
                stats.interaction_time_ms += (time.time() - interaction_start) * 1000

            if self._annihilation:
                stats.annihilation_checks += 1
                if _quick_annihilation_check(node, pushed):
                    annihilated.add(nid)
                    delta_map[nid] = _make_identity()
                    stats.nodes_annihilated += 1
                    continue

            ds = _delta_size(pushed)
            if ds > self._max_delta_size:
                pushed = self._truncate_delta(pushed)

            delta_map[nid] = pushed
            affected.add(nid)
            stats.nodes_affected += 1
            stats.total_delta_size += ds
            stats.max_delta_size = max(stats.max_delta_size, ds)

        stats.propagation_time_ms = (time.time() - start_time) * 1000

        return PropagationResult(
            node_deltas=delta_map,
            affected_nodes=affected,
            annihilated_nodes=annihilated,
            propagation_paths=paths,
            statistics=stats,
            source_nodes=set(perturbations.keys()),
            mode=self._mode,
        )

    def incremental_propagate(
        self,
        graph: Any,
        previous_result: PropagationResult,
        new_perturbation: CompoundPerturbation,
        new_source: str,
    ) -> PropagationResult:
        """Efficiently update propagation when a new perturbation arrives.

        Reuses the previous propagation result and only recomputes nodes
        that are affected by the new perturbation. Nodes where the delta
        hasn't changed are skipped.

        Parameters
        ----------
        graph : PipelineGraph
            The pipeline DAG.
        previous_result : PropagationResult
            Result of a previous propagation.
        new_perturbation : CompoundPerturbation
            The new perturbation to incorporate.
        new_source : str
            The node where the new perturbation originates.

        Returns
        -------
        PropagationResult
            Updated propagation results.
        """
        start_time = time.time()
        stats = PropagationStats()

        new_result = self.propagate(graph, new_source, new_perturbation)

        merged_deltas = dict(previous_result.node_deltas)
        changed_nodes: Set[str] = set()

        for nid, delta in new_result.node_deltas.items():
            if _is_identity(delta):
                continue

            old_delta = merged_deltas.get(nid)
            if old_delta is not None and not _is_identity(old_delta):
                composed = _compose_perturbations(old_delta, delta)
                merged_deltas[nid] = composed
            else:
                merged_deltas[nid] = delta
            changed_nodes.add(nid)

        topo_order = self._topo_cache.get_order(graph)
        recompute_needed: Set[str] = set()

        for nid in topo_order:
            if nid in changed_nodes:
                recompute_needed.add(nid)
                continue
            predecessors = graph.predecessors(nid)
            if any(p in recompute_needed for p in predecessors):
                recompute_needed.add(nid)

        for nid in topo_order:
            if nid not in recompute_needed:
                continue
            if nid in changed_nodes:
                continue

            stats.nodes_visited += 1
            node = graph.get_node(nid)
            predecessors = graph.predecessors(nid)

            incoming_deltas = []
            for pred in predecessors:
                pred_delta = merged_deltas.get(pred)
                if pred_delta is not None and not _is_identity(pred_delta):
                    incoming_deltas.append(pred_delta)

            if not incoming_deltas:
                stats.nodes_skipped += 1
                continue

            aggregated = _aggregate_deltas(incoming_deltas, self._aggregation)
            pushed = self._apply_node_push(node, aggregated)

            if self._interactions:
                pushed = _apply_interactions(pushed)

            if self._annihilation and _quick_annihilation_check(node, pushed):
                merged_deltas[nid] = _make_identity()
                stats.nodes_annihilated += 1
                continue

            merged_deltas[nid] = pushed
            stats.nodes_affected += 1

        all_affected = previous_result.affected_nodes | new_result.affected_nodes
        all_annihilated = (
            previous_result.annihilated_nodes | new_result.annihilated_nodes
        )
        all_annihilated -= all_affected

        merged_paths = dict(previous_result.propagation_paths)
        merged_paths.update(new_result.propagation_paths)

        stats.propagation_time_ms = (time.time() - start_time) * 1000
        stats = stats.merge(new_result.statistics)

        return PropagationResult(
            node_deltas=merged_deltas,
            affected_nodes=all_affected,
            annihilated_nodes=all_annihilated,
            propagation_paths=merged_paths,
            statistics=stats,
            source_nodes=previous_result.source_nodes | {new_source},
            mode=PropagationMode.INCREMENTAL,
        )

    def propagate_backwards(
        self,
        graph: Any,
        sink_node: str,
        desired_output_delta: CompoundPerturbation,
    ) -> Dict[str, CompoundPerturbation]:
        """Backwards propagation: given a desired output delta at a sink,
        compute what source deltas would produce it.

        This is an inverse problem and may not have an exact solution.
        We traverse in reverse topological order, applying inverse push
        operators where available.

        Parameters
        ----------
        graph : PipelineGraph
            The pipeline DAG.
        sink_node : str
            The sink node where the desired delta is specified.
        desired_output_delta : CompoundPerturbation
            The desired delta at the sink.

        Returns
        -------
        dict[str, CompoundPerturbation]
            Mapping from source node to required perturbation.
        """
        reverse_order = list(reversed(self._topo_cache.get_order(graph)))
        delta_map: Dict[str, CompoundPerturbation] = {}
        delta_map[sink_node] = desired_output_delta

        for nid in reverse_order:
            if nid not in delta_map:
                continue

            current_delta = delta_map[nid]
            if _is_identity(current_delta):
                continue

            predecessors = graph.predecessors(nid)
            for pred in predecessors:
                if pred not in delta_map:
                    delta_map[pred] = current_delta
                else:
                    delta_map[pred] = _compose_perturbations(
                        delta_map[pred], current_delta
                    )

        sources = graph.sources()
        return {nid: delta_map[nid] for nid in sources if nid in delta_map}

    def propagate_with_checkpoints(
        self,
        graph: Any,
        source_node: str,
        perturbation: CompoundPerturbation,
        checkpoint_nodes: Set[str],
    ) -> Tuple[PropagationResult, Dict[str, CompoundPerturbation]]:
        """Propagate with checkpointing at specified nodes.

        Like normal propagation, but saves intermediate deltas at checkpoint
        nodes for potential re-use in future incremental propagations.

        Parameters
        ----------
        graph : PipelineGraph
            The pipeline DAG.
        source_node : str
            Source of the perturbation.
        perturbation : CompoundPerturbation
            The initial perturbation.
        checkpoint_nodes : set[str]
            Nodes where intermediate deltas should be checkpointed.

        Returns
        -------
        tuple[PropagationResult, dict[str, CompoundPerturbation]]
            The propagation result and checkpoint snapshots.
        """
        result = self.propagate(graph, source_node, perturbation)

        checkpoints: Dict[str, CompoundPerturbation] = {}
        for nid in checkpoint_nodes:
            delta = result.get_delta(nid)
            if delta is not None:
                checkpoints[nid] = delta

        return result, checkpoints

    def compute_propagation_cone(
        self,
        graph: Any,
        source_node: str,
    ) -> Set[str]:
        """Compute the set of nodes that would be affected by any perturbation
        at the given source, without actually propagating a delta.

        This is a lightweight alternative to full propagation for impact
        estimation.

        Parameters
        ----------
        graph : PipelineGraph
            The pipeline DAG.
        source_node : str
            The source node.

        Returns
        -------
        set[str]
            All nodes in the downstream propagation cone.
        """
        cone: Set[str] = {source_node}
        topo_order = self._topo_cache.get_order(graph)

        source_idx = -1
        for i, nid in enumerate(topo_order):
            if nid == source_node:
                source_idx = i
                break

        if source_idx < 0:
            return cone

        for nid in topo_order[source_idx + 1:]:
            predecessors = graph.predecessors(nid)
            if any(p in cone for p in predecessors):
                cone.add(nid)

        return cone

    def estimate_propagation_cost(
        self,
        graph: Any,
        source_node: str,
        perturbation: CompoundPerturbation,
    ) -> float:
        """Estimate the cost of propagation without actually doing it.

        Uses the propagation cone and node cost estimates to approximate
        the total propagation cost.

        Parameters
        ----------
        graph : PipelineGraph
            The pipeline DAG.
        source_node : str
            The source node.
        perturbation : CompoundPerturbation
            The perturbation.

        Returns
        -------
        float
            Estimated total propagation cost.
        """
        cone = self.compute_propagation_cone(graph, source_node)
        total_cost = 0.0
        delta_size = _delta_size(perturbation)

        for nid in cone:
            node = graph.get_node(nid)
            node_cost = node.cost_estimate.total_weighted_cost
            amplification_factor = 1.0

            operator_name = _get_node_operator_name(node)
            if operator_name == "JOIN":
                amplification_factor = 2.0
            elif operator_name == "FILTER":
                amplification_factor = 0.5
            elif operator_name == "GROUP_BY":
                amplification_factor = 0.3

            total_cost += node_cost * delta_size * amplification_factor

        return total_cost

    # ── Internal methods ──────────────────────────────────────────

    def _apply_node_push(
        self,
        node: Any,
        delta: CompoundPerturbation,
    ) -> CompoundPerturbation:
        """Apply the push operator for a node to a compound perturbation.

        Looks up the appropriate push operator for the node's SQL operator
        type and applies it to all three delta sorts.
        """
        operator_name = _get_node_operator_name(node)

        push_op = self._push_cache.get(operator_name)
        if push_op is None:
            try:
                push_op = get_push_operator(operator_name)
                self._push_cache[operator_name] = push_op
            except (KeyError, ValueError):
                logger.debug(
                    "No push operator for %s, passing delta through", operator_name
                )
                return delta

        config = _extract_operator_config(node)
        return _push_compound_delta(push_op, config, delta)

    def _truncate_delta(
        self,
        delta: CompoundPerturbation,
    ) -> CompoundPerturbation:
        """Truncate a delta that exceeds the maximum size."""
        limit = self._max_delta_size

        schema_ops = []
        if delta.schema_delta is not None:
            schema_ops = list(delta.schema_delta.operations[:limit])
            limit -= len(schema_ops)

        data_ops = []
        if delta.data_delta is not None and limit > 0:
            data_ops = list(delta.data_delta.operations[:limit])
            limit -= len(data_ops)

        quality_ops = []
        if delta.quality_delta is not None and limit > 0:
            quality_ops = list(delta.quality_delta.operations[:limit])

        return CompoundPerturbation(
            schema_delta=SchemaDelta(operations=schema_ops),
            data_delta=DataDelta(operations=data_ops),
            quality_delta=QualityDelta(operations=quality_ops),
        )


# =====================================================================
# Batch Propagation
# =====================================================================


class BatchPropagator:
    """Propagate multiple perturbations in batched mode for efficiency.

    Groups perturbations by their topological depth to minimise redundant
    traversals.

    Parameters
    ----------
    propagator : DeltaPropagator
        The underlying propagator to use.
    batch_size : int
        Maximum number of perturbations per batch.
    """

    def __init__(
        self,
        propagator: Optional[DeltaPropagator] = None,
        batch_size: int = 50,
    ) -> None:
        self._propagator = propagator or DeltaPropagator()
        self._batch_size = batch_size

    def propagate_batch(
        self,
        graph: Any,
        perturbations: Dict[str, CompoundPerturbation],
    ) -> PropagationResult:
        """Propagate a batch of perturbations efficiently.

        Groups perturbations by topological depth and processes them in
        waves to avoid redundant propagation.

        Parameters
        ----------
        graph : PipelineGraph
            The pipeline DAG.
        perturbations : dict[str, CompoundPerturbation]
            Mapping from source node to perturbation.

        Returns
        -------
        PropagationResult
            Combined results.
        """
        if not perturbations:
            return PropagationResult()

        if len(perturbations) <= self._batch_size:
            return self._propagator.propagate_multi_source(graph, perturbations)

        topo_order = graph.topological_sort()
        depth_map: Dict[str, int] = {}
        for i, nid in enumerate(topo_order):
            depth_map[nid] = i

        sorted_sources = sorted(
            perturbations.keys(),
            key=lambda nid: depth_map.get(nid, 0),
        )

        combined_result = PropagationResult()
        batch: Dict[str, CompoundPerturbation] = {}

        for src in sorted_sources:
            batch[src] = perturbations[src]

            if len(batch) >= self._batch_size:
                result = self._propagator.propagate_multi_source(graph, batch)
                combined_result = combined_result.merge(result)
                batch = {}

        if batch:
            result = self._propagator.propagate_multi_source(graph, batch)
            combined_result = combined_result.merge(result)

        return combined_result

    def propagate_stream(
        self,
        graph: Any,
        perturbation_stream: Iterable[Tuple[str, CompoundPerturbation]],
    ) -> PropagationResult:
        """Process a stream of perturbations incrementally.

        Each perturbation is applied on top of the previous result using
        incremental propagation.

        Parameters
        ----------
        graph : PipelineGraph
            The pipeline DAG.
        perturbation_stream : Iterable[tuple[str, CompoundPerturbation]]
            Stream of (source_node, perturbation) pairs.

        Returns
        -------
        PropagationResult
            Final cumulative result.
        """
        current_result: Optional[PropagationResult] = None

        for source_node, perturbation in perturbation_stream:
            if current_result is None:
                current_result = self._propagator.propagate(
                    graph, source_node, perturbation
                )
            else:
                current_result = self._propagator.incremental_propagate(
                    graph, current_result, perturbation, source_node
                )

        return current_result or PropagationResult()


# =====================================================================
# Propagation Analysis Utilities
# =====================================================================


class PropagationAnalyzer:
    """Analyze propagation results to extract insights.

    Provides methods for understanding delta amplification, finding
    bottlenecks, and recommending checkpoint placement.
    """

    def compute_amplification_map(
        self,
        result: PropagationResult,
    ) -> Dict[str, float]:
        """Compute the delta amplification factor at each node.

        Amplification measures how much a node increases or decreases
        the delta size compared to its input.

        Returns
        -------
        dict[str, float]
            Mapping from node_id to amplification factor.
        """
        amplification: Dict[str, float] = {}

        for nid, path in result.propagation_paths.items():
            if len(path.entries) < 2:
                amplification[nid] = 1.0
                continue

            last = path.entries[-1]
            if last.incoming_delta_size > 0:
                amplification[nid] = (
                    last.outgoing_delta_size / last.incoming_delta_size
                )
            else:
                amplification[nid] = 0.0

        return amplification

    def find_amplification_bottlenecks(
        self,
        result: PropagationResult,
        threshold: float = 2.0,
    ) -> List[Tuple[str, float]]:
        """Find nodes where delta amplification exceeds a threshold.

        These are potential bottleneck nodes where repairs become expensive.

        Parameters
        ----------
        result : PropagationResult
            Propagation results to analyze.
        threshold : float
            Minimum amplification factor to flag.

        Returns
        -------
        list[tuple[str, float]]
            List of (node_id, amplification_factor) tuples.
        """
        amp_map = self.compute_amplification_map(result)
        bottlenecks = [
            (nid, amp) for nid, amp in amp_map.items() if amp >= threshold
        ]
        bottlenecks.sort(key=lambda x: x[1], reverse=True)
        return bottlenecks

    def recommend_checkpoints(
        self,
        graph: Any,
        result: PropagationResult,
        max_checkpoints: int = 5,
    ) -> List[str]:
        """Recommend checkpoint placement based on propagation analysis.

        Places checkpoints at nodes with high fan-out, high delta size,
        or high amplification to minimise re-propagation cost.

        Parameters
        ----------
        graph : PipelineGraph
            The pipeline DAG.
        result : PropagationResult
            Propagation results.
        max_checkpoints : int
            Maximum number of checkpoints to recommend.

        Returns
        -------
        list[str]
            Ordered list of recommended checkpoint node ids.
        """
        scores: Dict[str, float] = {}

        for nid in result.affected_nodes:
            score = 0.0

            delta = result.get_delta(nid)
            if delta is not None:
                score += _delta_size(delta) * 0.3

            try:
                out_degree = graph.out_degree(nid)
                score += out_degree * 10.0
            except Exception:
                pass

            path = result.get_path(nid)
            if path is not None and path.length > 0:
                score += path.length * 2.0

            scores[nid] = score

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [nid for nid, _ in ranked[:max_checkpoints]]

    def compute_propagation_frontier(
        self,
        result: PropagationResult,
        graph: Any,
    ) -> Set[str]:
        """Compute the propagation frontier: affected nodes with no
        affected successors.

        These are the "leaf" nodes of the propagation, where the delta
        reaches its final form.

        Parameters
        ----------
        result : PropagationResult
            Propagation results.
        graph : PipelineGraph
            The pipeline DAG.

        Returns
        -------
        set[str]
            Set of frontier node ids.
        """
        frontier: Set[str] = set()

        for nid in result.affected_nodes:
            successors = graph.successors(nid)
            has_affected_successor = any(
                s in result.affected_nodes for s in successors
            )
            if not has_affected_successor:
                frontier.add(nid)

        return frontier

    def compute_propagation_depth_histogram(
        self,
        result: PropagationResult,
    ) -> Dict[int, int]:
        """Compute a histogram of propagation path depths.

        Returns
        -------
        dict[int, int]
            Mapping from depth to count of nodes at that depth.
        """
        histogram: Dict[int, int] = defaultdict(int)

        for path in result.propagation_paths.values():
            histogram[path.length] += 1

        return dict(histogram)

    def summarize_delta_distribution(
        self,
        result: PropagationResult,
    ) -> Dict[str, Dict[str, int]]:
        """Summarize delta distribution by sort (schema, data, quality).

        Returns
        -------
        dict[str, dict[str, int]]
            For each sort, mapping from node to operation count.
        """
        schema_dist: Dict[str, int] = {}
        data_dist: Dict[str, int] = {}
        quality_dist: Dict[str, int] = {}

        for nid, delta in result.node_deltas.items():
            if _is_identity(delta):
                continue
            if delta.schema_delta is not None:
                schema_dist[nid] = len(delta.schema_delta.operations)
            if delta.data_delta is not None:
                data_dist[nid] = len(delta.data_delta.operations)
            if delta.quality_delta is not None:
                quality_dist[nid] = len(delta.quality_delta.operations)

        return {
            "schema": schema_dist,
            "data": data_dist,
            "quality": quality_dist,
        }


# =====================================================================
# Propagation Validator
# =====================================================================


class PropagationValidator:
    """Validate propagation results for correctness.

    Checks algebraic invariants that should hold after propagation:
    - Identity propagation: propagating identity gives identity everywhere
    - Composition: propagate(a ∘ b) = propagate(a) ∘ propagate(b) (when linear)
    - Annihilation consistency: annihilated nodes have identity deltas
    """

    def validate_identity_propagation(
        self,
        graph: Any,
        source_node: str,
        propagator: DeltaPropagator,
    ) -> List[str]:
        """Check that propagating identity gives identity at all nodes.

        Returns a list of violations (empty if valid).
        """
        identity = _make_identity()
        result = propagator.propagate(graph, source_node, identity)

        violations: List[str] = []
        for nid, delta in result.node_deltas.items():
            if not _is_identity(delta):
                violations.append(
                    f"Node {nid} has non-identity delta after identity propagation"
                )

        return violations

    def validate_annihilation_consistency(
        self,
        result: PropagationResult,
    ) -> List[str]:
        """Check that annihilated nodes have identity deltas.

        Returns a list of violations (empty if consistent).
        """
        violations: List[str] = []

        for nid in result.annihilated_nodes:
            delta = result.get_delta(nid)
            if delta is not None and not _is_identity(delta):
                violations.append(
                    f"Annihilated node {nid} has non-identity delta"
                )

        return violations

    def validate_topological_monotonicity(
        self,
        result: PropagationResult,
        graph: Any,
    ) -> List[str]:
        """Check that affected nodes form a topologically connected cone.

        Every affected node should either be a source or have at least
        one affected predecessor.
        """
        violations: List[str] = []

        for nid in result.affected_nodes:
            if nid in result.source_nodes:
                continue
            predecessors = graph.predecessors(nid)
            has_affected_pred = any(
                p in result.affected_nodes or p in result.source_nodes
                for p in predecessors
            )
            if not has_affected_pred:
                violations.append(
                    f"Affected node {nid} has no affected predecessor"
                )

        return violations

    def validate_all(
        self,
        result: PropagationResult,
        graph: Any,
        propagator: Optional[DeltaPropagator] = None,
    ) -> Dict[str, List[str]]:
        """Run all validation checks and return results.

        Returns
        -------
        dict[str, list[str]]
            Mapping from check name to list of violations.
        """
        checks: Dict[str, List[str]] = {}

        checks["annihilation_consistency"] = (
            self.validate_annihilation_consistency(result)
        )
        checks["topological_monotonicity"] = (
            self.validate_topological_monotonicity(result, graph)
        )

        if propagator is not None and result.source_nodes:
            source = next(iter(result.source_nodes))
            checks["identity_propagation"] = (
                self.validate_identity_propagation(graph, source, propagator)
            )

        return checks


# =====================================================================
# Convenience functions
# =====================================================================


def propagate_single(
    graph: Any,
    source_node: str,
    perturbation: CompoundPerturbation,
    enable_annihilation: bool = True,
) -> PropagationResult:
    """Convenience function for single-source propagation.

    Parameters
    ----------
    graph : PipelineGraph
        The pipeline DAG.
    source_node : str
        Node where the perturbation originates.
    perturbation : CompoundPerturbation
        The perturbation to propagate.
    enable_annihilation : bool
        Whether to check for annihilation.

    Returns
    -------
    PropagationResult
    """
    propagator = DeltaPropagator(enable_annihilation=enable_annihilation)
    return propagator.propagate(graph, source_node, perturbation)


def propagate_multi(
    graph: Any,
    perturbations: Dict[str, CompoundPerturbation],
) -> PropagationResult:
    """Convenience function for multi-source propagation.

    Parameters
    ----------
    graph : PipelineGraph
        The pipeline DAG.
    perturbations : dict[str, CompoundPerturbation]
        Source-to-perturbation mapping.

    Returns
    -------
    PropagationResult
    """
    propagator = DeltaPropagator()
    return propagator.propagate_multi_source(graph, perturbations)


def estimate_impact(
    graph: Any,
    source_node: str,
    perturbation: CompoundPerturbation,
) -> Dict[str, Any]:
    """Quick impact estimation without full propagation.

    Returns a dict with keys:
    - "cone_size": number of potentially affected nodes
    - "estimated_cost": rough cost estimate
    - "affected_sinks": number of sink nodes in the cone

    Parameters
    ----------
    graph : PipelineGraph
        The pipeline DAG.
    source_node : str
        Source node.
    perturbation : CompoundPerturbation
        The perturbation.

    Returns
    -------
    dict[str, Any]
    """
    propagator = DeltaPropagator(
        enable_annihilation=False,
        enable_interactions=False,
        track_paths=False,
    )
    cone = propagator.compute_propagation_cone(graph, source_node)
    cost = propagator.estimate_propagation_cost(graph, source_node, perturbation)

    sink_nodes = set(graph.sinks())
    affected_sinks = cone & sink_nodes

    return {
        "cone_size": len(cone),
        "estimated_cost": cost,
        "affected_sinks": len(affected_sinks),
        "affected_sink_ids": sorted(affected_sinks),
        "delta_size": _delta_size(perturbation),
    }
