"""
taintflow.dag.builder -- DAG construction from instrumentation data.

The :class:`DAGBuilder` constructs a :class:`PIDAG` from various sources:
trace events collected by the instrumentation layer, sklearn Pipeline
objects, or pandas operation logs.  It handles:

* Automatic node-ID generation with collision avoidance
* Schema inference from observed data
* Provenance propagation during construction
* sklearn Pipeline / ColumnTransformer / FeatureUnion unrolling
* Cross-validation unrolling (GridSearchCV, cross_val_score)
* Column tracking through renames, drops, and adds
* Merging DAG fragments from different code paths
"""

from __future__ import annotations

import copy
import hashlib
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from typing import (
    Any,
    Callable,
    Dict,
    FrozenSet,
    List,
    Mapping,
    Optional,
    Sequence,
    Set,
    Tuple,
    Type,
)

from taintflow.core.types import (
    ColumnSchema,
    EdgeKind,
    NodeKind,
    OpType,
    Origin,
    ProvenanceInfo,
    ShapeMetadata,
)
from taintflow.core.errors import DAGConstructionError, MissingNodeError
from taintflow.dag.node import (
    PipelineNode,
    DataSourceNode,
    PartitionNode,
    TransformNode,
    FitNode,
    PredictNode,
    PandasOpNode,
    AggregationNode,
    FeatureEngineeringNode,
    SelectionNode,
    CustomNode,
    SinkNode,
    SourceLocation,
    NodeFactory,
    _generate_node_id,
)
from taintflow.dag.edge import PipelineEdge, EdgeSet
from taintflow.dag.pidag import PIDAG


# ===================================================================
#  Trace event data structures
# ===================================================================


@dataclass
class TraceEvent:
    """A single event captured by the instrumentation layer.

    Represents one observed operation (function call, method invocation)
    with the schemas of its arguments and return value.
    """

    timestamp: float = 0.0
    event_type: str = "call"
    function: str = ""
    module: str = ""
    class_name: str = ""
    file: str = ""
    line: int = 0
    col: int = 0
    args_schema: list[dict[str, Any]] = field(default_factory=list)
    return_schema: list[dict[str, Any]] = field(default_factory=list)
    args_shape: Tuple[int, ...] | None = None
    return_shape: Tuple[int, ...] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    # -- convenience properties ----------------------------------------------

    @property
    def source_location(self) -> SourceLocation:
        return SourceLocation(
            file=self.file,
            line=self.line,
            col=self.col,
            function_name=self.function,
            class_name=self.class_name,
        )

    @property
    def op_type(self) -> OpType:
        """Infer OpType from the event's function name and metadata."""
        return _infer_op_type(self.function, self.module, self.class_name, self.metadata)

    @property
    def input_columns(self) -> list[ColumnSchema]:
        return [ColumnSchema.from_dict(s) for s in self.args_schema]

    @property
    def output_columns(self) -> list[ColumnSchema]:
        return [ColumnSchema.from_dict(s) for s in self.return_schema]

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "event_type": self.event_type,
            "function": self.function,
            "module": self.module,
            "class_name": self.class_name,
            "file": self.file,
            "line": self.line,
            "col": self.col,
            "args_schema": self.args_schema,
            "return_schema": self.return_schema,
            "args_shape": list(self.args_shape) if self.args_shape else None,
            "return_shape": list(self.return_shape) if self.return_shape else None,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "TraceEvent":
        args_shape = tuple(data["args_shape"]) if data.get("args_shape") else None
        return_shape = tuple(data["return_shape"]) if data.get("return_shape") else None
        return cls(
            timestamp=float(data.get("timestamp", 0.0)),
            event_type=str(data.get("event_type", "call")),
            function=str(data.get("function", "")),
            module=str(data.get("module", "")),
            class_name=str(data.get("class_name", "")),
            file=str(data.get("file", "")),
            line=int(data.get("line", 0)),
            col=int(data.get("col", 0)),
            args_schema=list(data.get("args_schema", [])),
            return_schema=list(data.get("return_schema", [])),
            args_shape=args_shape,
            return_shape=return_shape,
            metadata=dict(data.get("metadata", {})),
        )


@dataclass
class OperationLog:
    """Sequence of :class:`TraceEvent` objects with context information."""

    events: list[TraceEvent] = field(default_factory=list)
    session_id: str = ""
    start_time: float = 0.0
    end_time: float = 0.0
    source_file: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def add_event(self, event: TraceEvent) -> None:
        """Append an event, updating time bounds."""
        self.events.append(event)
        if not self.start_time or event.timestamp < self.start_time:
            self.start_time = event.timestamp
        if event.timestamp > self.end_time:
            self.end_time = event.timestamp

    def sorted_events(self) -> list[TraceEvent]:
        """Return events sorted by timestamp."""
        return sorted(self.events, key=lambda e: e.timestamp)

    def filter_by_module(self, module: str) -> "OperationLog":
        """Return a new log with only events from the given module."""
        filtered = OperationLog(
            session_id=self.session_id,
            source_file=self.source_file,
            metadata=dict(self.metadata),
        )
        for ev in self.events:
            if ev.module == module or ev.module.startswith(module + "."):
                filtered.add_event(ev)
        return filtered

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "source_file": self.source_file,
            "metadata": self.metadata,
            "events": [e.to_dict() for e in self.events],
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "OperationLog":
        log = cls(
            session_id=str(data.get("session_id", "")),
            start_time=float(data.get("start_time", 0.0)),
            end_time=float(data.get("end_time", 0.0)),
            source_file=str(data.get("source_file", "")),
            metadata=dict(data.get("metadata", {})),
        )
        for edata in data.get("events", []):
            log.events.append(TraceEvent.from_dict(edata))
        return log

    def __len__(self) -> int:
        return len(self.events)

    def __repr__(self) -> str:
        return f"OperationLog({len(self.events)} events, session={self.session_id!r})"


# ===================================================================
#  OpType inference
# ===================================================================

_FUNCTION_TO_OPTYPE: dict[str, OpType] = {
    "read_csv": OpType.READ_CSV,
    "read_parquet": OpType.READ_PARQUET,
    "read_json": OpType.READ_JSON,
    "read_excel": OpType.READ_EXCEL,
    "read_sql": OpType.READ_SQL,
    "read_hdf": OpType.READ_HDF,
    "read_feather": OpType.READ_FEATHER,
    "to_csv": OpType.TO_CSV,
    "to_parquet": OpType.TO_PARQUET,
    "merge": OpType.MERGE,
    "join": OpType.JOIN,
    "concat": OpType.CONCAT,
    "append": OpType.APPEND,
    "groupby": OpType.GROUPBY,
    "agg": OpType.AGG,
    "aggregate": OpType.AGGREGATE,
    "transform": OpType.TRANSFORM,
    "apply": OpType.APPLY,
    "applymap": OpType.APPLYMAP,
    "map": OpType.MAP,
    "rolling": OpType.ROLLING,
    "expanding": OpType.EXPANDING,
    "ewm": OpType.EWM,
    "resample": OpType.RESAMPLE,
    "value_counts": OpType.VALUE_COUNTS,
    "describe": OpType.DESCRIBE,
    "corr": OpType.CORR,
    "cov": OpType.COV,
    "fillna": OpType.FILLNA,
    "dropna": OpType.DROPNA,
    "interpolate": OpType.INTERPOLATE,
    "drop": OpType.DROP,
    "drop_duplicates": OpType.DROP_DUPLICATES,
    "rename": OpType.RENAME,
    "assign": OpType.ASSIGN,
    "sort_values": OpType.SORT_VALUES,
    "sort_index": OpType.SORT_INDEX,
    "set_index": OpType.SET_INDEX,
    "reset_index": OpType.RESET_INDEX,
    "head": OpType.HEAD,
    "tail": OpType.TAIL,
    "sample": OpType.SAMPLE,
    "query": OpType.QUERY,
    "pivot": OpType.PIVOT,
    "pivot_table": OpType.PIVOT_TABLE,
    "melt": OpType.MELT,
    "stack": OpType.STACK,
    "unstack": OpType.UNSTACK,
    "explode": OpType.EXPLODE,
    "get_dummies": OpType.GET_DUMMIES,
    "astype": OpType.ASTYPE,
    "replace": OpType.REPLACE,
    "clip": OpType.CLIP,
    "copy": OpType.COPY,
    "fit": OpType.FIT,
    "predict": OpType.PREDICT,
    "fit_transform": OpType.FIT_TRANSFORM,
    "predict_proba": OpType.PREDICT_PROBA,
    "predict_log_proba": OpType.PREDICT_LOG_PROBA,
    "decision_function": OpType.DECISION_FUNCTION,
    "score": OpType.SCORE,
    "fit_predict": OpType.FIT_PREDICT,
    "inverse_transform": OpType.INVERSE_TRANSFORM,
    "train_test_split": OpType.TRAIN_TEST_SPLIT,
    "cross_val_score": OpType.CROSS_VAL_SCORE,
    "cross_validate": OpType.CROSS_VALIDATE,
}


def _infer_op_type(
    function: str,
    module: str,
    class_name: str,
    metadata: dict[str, Any],
) -> OpType:
    """Map a traced function call to its :class:`OpType`."""
    if function in _FUNCTION_TO_OPTYPE:
        return _FUNCTION_TO_OPTYPE[function]

    if "sklearn" in module:
        if function == "transform":
            return OpType.TRANSFORM_SK

    if module.startswith("numpy") or module.startswith("np"):
        np_map = {
            "array": OpType.NP_ARRAY,
            "concatenate": OpType.NP_CONCATENATE,
            "vstack": OpType.NP_VSTACK,
            "hstack": OpType.NP_HSTACK,
            "column_stack": OpType.NP_COLUMN_STACK,
            "split": OpType.NP_SPLIT,
            "where": OpType.NP_WHERE,
            "unique": OpType.NP_UNIQUE,
            "argsort": OpType.NP_ARGSORT,
            "log": OpType.NP_LOG,
            "exp": OpType.NP_EXP,
            "matmul": OpType.NP_MATMUL,
            "dot": OpType.NP_DOT,
            "mean": OpType.NP_MEAN,
            "std": OpType.NP_STD,
            "var": OpType.NP_VAR,
            "median": OpType.NP_MEDIAN,
            "sum": OpType.NP_SUM,
        }
        if function in np_map:
            return np_map[function]

    if function == "__getitem__":
        return OpType.GETITEM
    if function == "__setitem__":
        return OpType.SETITEM

    if metadata.get("is_custom", False):
        return OpType.CUSTOM

    return OpType.UNKNOWN


# ===================================================================
#  Schema inference helpers
# ===================================================================


def _infer_schema_from_dict(schema_dicts: list[dict[str, Any]]) -> list[ColumnSchema]:
    """Convert a list of column-info dicts to ColumnSchema objects."""
    return [ColumnSchema.from_dict(s) for s in schema_dicts]


def _infer_shape(
    args_shape: Tuple[int, ...] | None,
    return_shape: Tuple[int, ...] | None,
    in_schema: list[ColumnSchema],
    out_schema: list[ColumnSchema],
) -> ShapeMetadata:
    """Build ShapeMetadata from available shape and schema information."""
    if return_shape and len(return_shape) >= 2:
        n_rows, n_cols = return_shape[0], return_shape[1]
    elif out_schema:
        n_rows = 0
        n_cols = len(out_schema)
    elif args_shape and len(args_shape) >= 2:
        n_rows, n_cols = args_shape[0], args_shape[1]
    else:
        n_rows, n_cols = 0, len(in_schema) if in_schema else 0
    return ShapeMetadata(n_rows=n_rows, n_cols=n_cols)


def _track_column_changes(
    in_cols: set[str],
    op_type: OpType,
    metadata: dict[str, Any],
    out_schema: list[ColumnSchema],
) -> Tuple[set[str], dict[str, str]]:
    """Track how columns change through an operation.

    Returns (new_column_set, rename_map) where rename_map maps
    old_name -> new_name for renamed columns.
    """
    rename_map: dict[str, str] = {}
    out_cols = {c.name for c in out_schema} if out_schema else set(in_cols)

    if op_type == OpType.DROP:
        dropped = set(metadata.get("columns", []))
        out_cols = in_cols - dropped
    elif op_type == OpType.RENAME:
        raw_map = metadata.get("rename_map", metadata.get("columns", {}))
        if isinstance(raw_map, dict):
            rename_map = dict(raw_map)
            out_cols = set()
            for c in in_cols:
                if c in rename_map:
                    out_cols.add(rename_map[c])
                else:
                    out_cols.add(c)
    elif op_type == OpType.ASSIGN:
        new = set(metadata.get("new_columns", []))
        out_cols = in_cols | new
    elif op_type in {OpType.GETITEM, OpType.LOC, OpType.ILOC, OpType.FILTER}:
        selected = set(metadata.get("columns", []))
        if selected:
            out_cols = in_cols & selected
    elif op_type == OpType.SET_INDEX:
        idx_col = metadata.get("index_column", "")
        if idx_col:
            out_cols = in_cols - {idx_col}
    elif op_type == OpType.RESET_INDEX:
        idx_col = metadata.get("index_column", "")
        if idx_col:
            out_cols = in_cols | {idx_col}
    elif op_type in {OpType.MERGE, OpType.JOIN, OpType.CONCAT}:
        right_cols = set(metadata.get("right_columns", []))
        out_cols = in_cols | right_cols
    elif op_type == OpType.GET_DUMMIES:
        prefix_cols = set(metadata.get("generated_columns", []))
        dummy_source = set(metadata.get("dummy_source_columns", []))
        out_cols = (in_cols - dummy_source) | prefix_cols

    return out_cols, rename_map


# ===================================================================
#  DAGBuilder
# ===================================================================


class DAGBuilder:
    """Construct a :class:`PIDAG` from various input sources.

    The builder maintains state during construction (node registry,
    column state, provenance tracking) and produces a validated DAG.
    """

    def __init__(self) -> None:
        self._dag: PIDAG = PIDAG()
        self._node_counter: int = 0
        self._column_state: dict[str, set[str]] = {}
        self._last_node_for_column: dict[str, str] = {}
        self._last_output_node: str | None = None
        self._id_set: set[str] = set()

    # -- ID generation -------------------------------------------------------

    def _next_id(self, prefix: str = "node") -> str:
        """Generate a unique node ID."""
        self._node_counter += 1
        nid = f"{prefix}_{self._node_counter:04d}"
        while nid in self._id_set:
            self._node_counter += 1
            nid = f"{prefix}_{self._node_counter:04d}"
        self._id_set.add(nid)
        return nid

    # -- public build methods ------------------------------------------------

    def from_trace_events(self, events: Sequence[TraceEvent]) -> PIDAG:
        """Build a PIDAG from a sequence of trace events.

        Events are processed in timestamp order.  Each event creates a node
        and edges connecting it to the previous node(s) based on column overlap.
        """
        sorted_events = sorted(events, key=lambda e: e.timestamp)

        for ev in sorted_events:
            op = ev.op_type
            in_schema = ev.input_columns
            out_schema = ev.output_columns
            shape = _infer_shape(ev.args_shape, ev.return_shape, in_schema, out_schema)

            node = NodeFactory.create(
                op_type=op,
                node_id=self._next_id(op.value),
                source_location=ev.source_location,
                input_schema=in_schema,
                output_schema=out_schema,
                shape=shape,
                metadata=dict(ev.metadata),
                timestamp=ev.timestamp,
            )
            self._dag.add_node(node)

            in_col_names = {c.name for c in in_schema}
            self._connect_to_predecessors(node.node_id, in_col_names, op)

            out_col_names = {c.name for c in out_schema}
            for col in out_col_names:
                self._last_node_for_column[col] = node.node_id
            self._column_state[node.node_id] = out_col_names
            self._last_output_node = node.node_id

        self._dag.propagate_provenance()
        self._dag.propagate_edge_provenance()
        return self._dag

    def from_sklearn_pipeline(self, pipeline: Any) -> PIDAG:
        """Build a PIDAG from an sklearn Pipeline object.

        Handles ``Pipeline``, ``ColumnTransformer``, ``FeatureUnion``,
        and nested combinations thereof.
        """
        steps = _extract_sklearn_steps(pipeline)

        prev_node_id: str | None = None
        prev_out_cols: set[str] = set()

        for step_name, estimator, step_type in steps:
            op = _sklearn_estimator_to_optype(estimator, step_type)
            est_class = type(estimator).__name__ if estimator is not None else "Unknown"

            n_features = getattr(estimator, "n_features_in_", 0)
            feature_names = list(getattr(estimator, "feature_names_in_", []))

            in_schema: list[ColumnSchema] = []
            out_schema: list[ColumnSchema] = []
            if feature_names:
                in_schema = [ColumnSchema(name=f) for f in feature_names]
            out_features = _infer_sklearn_output_features(estimator, feature_names)
            out_schema = [ColumnSchema(name=f) for f in out_features]

            shape = ShapeMetadata(n_rows=0, n_cols=len(out_schema))

            if step_type == "fit":
                node = FitNode(
                    node_id=self._next_id(f"fit_{step_name}"),
                    op_type=OpType.FIT,
                    input_schema=in_schema,
                    output_schema=out_schema,
                    shape=shape,
                    estimator_class=est_class,
                    n_features_in=n_features or len(in_schema),
                    source_location=SourceLocation(function_name=f"{step_name}.fit"),
                )
            elif step_type == "transform":
                node = TransformNode(
                    node_id=self._next_id(f"transform_{step_name}"),
                    op_type=OpType.TRANSFORM_SK,
                    input_schema=in_schema,
                    output_schema=out_schema,
                    shape=shape,
                    estimator_class=est_class,
                    is_fitted=True,
                    source_location=SourceLocation(function_name=f"{step_name}.transform"),
                )
            elif step_type == "fit_transform":
                node = TransformNode(
                    node_id=self._next_id(f"fit_transform_{step_name}"),
                    op_type=OpType.FIT_TRANSFORM,
                    input_schema=in_schema,
                    output_schema=out_schema,
                    shape=shape,
                    estimator_class=est_class,
                    is_fitted=True,
                    source_location=SourceLocation(function_name=f"{step_name}.fit_transform"),
                )
            elif step_type == "predict":
                node = PredictNode(
                    node_id=self._next_id(f"predict_{step_name}"),
                    op_type=OpType.PREDICT,
                    input_schema=in_schema,
                    output_schema=out_schema,
                    shape=shape,
                    estimator_class=est_class,
                    source_location=SourceLocation(function_name=f"{step_name}.predict"),
                )
            else:
                node = NodeFactory.create(
                    op_type=op,
                    node_id=self._next_id(step_name),
                    input_schema=in_schema,
                    output_schema=out_schema,
                    shape=shape,
                    metadata={"estimator_class": est_class, "step_name": step_name},
                )

            self._dag.add_node(node)

            if prev_node_id is not None:
                cols = prev_out_cols if prev_out_cols else frozenset()
                edge = PipelineEdge(
                    source_id=prev_node_id,
                    target_id=node.node_id,
                    columns=frozenset(cols),
                    edge_kind=EdgeKind.DATA_FLOW,
                )
                self._dag.add_edge(edge)

            prev_node_id = node.node_id
            prev_out_cols = {c.name for c in out_schema}

        return self._dag

    def from_pandas_operations(self, op_log: OperationLog) -> PIDAG:
        """Build a PIDAG from a pandas operation log.

        Each event in the log becomes a node.  Edges are inferred from
        column overlap between consecutive operations.
        """
        return self.from_trace_events(op_log.sorted_events())

    def from_cross_validation(
        self,
        cv_events: Sequence[TraceEvent],
        n_splits: int = 5,
    ) -> PIDAG:
        """Unroll cross-validation into the DAG.

        Creates a PartitionNode for each fold and replicates the inner
        pipeline for each split.
        """
        inner_events = [e for e in cv_events if e.function not in ("cross_val_score", "cross_validate")]

        cv_node = PartitionNode(
            node_id=self._next_id("cv_split"),
            op_type=OpType.KFOLD_SPLIT,
            n_splits=n_splits,
            test_size=1.0 / n_splits,
            source_location=SourceLocation(function_name="cross_validation"),
        )
        self._dag.add_node(cv_node)

        fold_sink_ids: list[str] = []
        for fold_idx in range(n_splits):
            fold_builder = DAGBuilder()
            fold_builder._node_counter = self._node_counter + fold_idx * 1000
            fold_dag = fold_builder.from_trace_events(inner_events)

            prefix = f"fold{fold_idx}_"
            remapped: dict[str, str] = {}
            for old_nid, node in list(fold_dag._nodes.items()):
                new_nid = prefix + old_nid
                node.node_id = new_nid
                remapped[old_nid] = new_nid

            renamed_nodes: dict[str, PipelineNode] = {}
            for old_nid, new_nid in remapped.items():
                renamed_nodes[new_nid] = fold_dag._nodes[old_nid]
            fold_dag._nodes = renamed_nodes

            new_edge_set = EdgeSet()
            for e in fold_dag._edges:
                new_src = remapped.get(e.source_id, e.source_id)
                new_tgt = remapped.get(e.target_id, e.target_id)
                new_edge_set.add(PipelineEdge(
                    source_id=new_src,
                    target_id=new_tgt,
                    columns=e.columns,
                    edge_kind=e.edge_kind,
                    capacity=e.capacity,
                    provenance_fraction=e.provenance_fraction,
                    metadata=dict(e.metadata),
                ))
            fold_dag._edges = new_edge_set

            self._dag.merge(fold_dag)

            fold_sources = [nid for nid in fold_dag._nodes
                           if not fold_dag._edges.by_target(nid)]
            for src_nid in fold_sources:
                if src_nid in self._dag._nodes:
                    self._dag.add_edge(PipelineEdge(
                        source_id=cv_node.node_id,
                        target_id=src_nid,
                        edge_kind=EdgeKind.DATA_FLOW,
                    ))

            fold_sinks = [nid for nid in fold_dag._nodes
                         if not fold_dag._edges.by_source(nid)]
            fold_sink_ids.extend(fold_sinks)

        if fold_sink_ids:
            agg_node = PipelineNode(
                node_id=self._next_id("cv_aggregate"),
                op_type=OpType.AGG,
                source_location=SourceLocation(function_name="cv_aggregate"),
                metadata={"description": "Aggregate cross-validation results"},
            )
            self._dag.add_node(agg_node)
            for sink_id in fold_sink_ids:
                if sink_id in self._dag._nodes:
                    self._dag.add_edge(PipelineEdge(
                        source_id=sink_id,
                        target_id=agg_node.node_id,
                        edge_kind=EdgeKind.DATA_FLOW,
                    ))

        return self._dag

    # -- internal edge connection logic --------------------------------------

    def _connect_to_predecessors(
        self,
        node_id: str,
        in_columns: set[str],
        op_type: OpType,
    ) -> None:
        """Connect a new node to its predecessor(s) based on column overlap."""
        predecessors_found: set[str] = set()

        if in_columns:
            for col in in_columns:
                pred_id = self._last_node_for_column.get(col)
                if pred_id is not None and pred_id != node_id:
                    predecessors_found.add(pred_id)

        if not predecessors_found and self._last_output_node is not None:
            if self._last_output_node != node_id:
                predecessors_found.add(self._last_output_node)

        for pred_id in predecessors_found:
            pred_out_cols = self._column_state.get(pred_id, set())
            shared_cols = in_columns & pred_out_cols if in_columns else pred_out_cols

            edge_kind = EdgeKind.DATA_FLOW
            if op_type in {OpType.FIT, OpType.FIT_TRANSFORM, OpType.FIT_PREDICT}:
                pred_node = self._dag._nodes.get(pred_id)
                if pred_node and pred_node.has_fit:
                    edge_kind = EdgeKind.FIT_DEPENDENCY

            edge = PipelineEdge(
                source_id=pred_id,
                target_id=node_id,
                columns=frozenset(shared_cols),
                edge_kind=edge_kind,
            )
            self._dag.add_edge(edge)

    # -- fragment merging ----------------------------------------------------

    def merge_fragment(self, fragment: PIDAG, connect_from: str | None = None, connect_to: str | None = None) -> None:
        """Merge a DAG fragment into the builder's DAG.

        Optionally connect the fragment's sources to *connect_from* or
        the fragment's sinks to *connect_to*.
        """
        self._dag.merge(fragment)

        if connect_from and connect_from in self._dag._nodes:
            for src in fragment.sources:
                if src.node_id in self._dag._nodes:
                    from_node = self._dag._nodes[connect_from]
                    from_cols = {c.name for c in from_node.output_schema}
                    to_cols = {c.name for c in src.input_schema}
                    shared = from_cols & to_cols if to_cols else from_cols
                    self._dag.add_edge(PipelineEdge(
                        source_id=connect_from,
                        target_id=src.node_id,
                        columns=frozenset(shared),
                        edge_kind=EdgeKind.DATA_FLOW,
                    ))

        if connect_to and connect_to in self._dag._nodes:
            for sink in fragment.sinks:
                if sink.node_id in self._dag._nodes:
                    sink_node_out = {c.name for c in sink.output_schema}
                    to_node = self._dag._nodes[connect_to]
                    to_in = {c.name for c in to_node.input_schema}
                    shared = sink_node_out & to_in if to_in else sink_node_out
                    self._dag.add_edge(PipelineEdge(
                        source_id=sink.node_id,
                        target_id=connect_to,
                        columns=frozenset(shared),
                        edge_kind=EdgeKind.DATA_FLOW,
                    ))

    # -- current DAG access --------------------------------------------------

    @property
    def dag(self) -> PIDAG:
        """The DAG under construction."""
        return self._dag

    def validate(self) -> list[str]:
        """Validate the constructed DAG."""
        return self._dag.validate()

    def build(self) -> PIDAG:
        """Finalize and return the constructed DAG (frozen)."""
        errors = self._dag.validate()
        if errors:
            raise DAGConstructionError(
                f"DAG validation failed with {len(errors)} error(s): {errors[0]}"
            )
        return self._dag.freeze()

    def build_unchecked(self) -> PIDAG:
        """Return the DAG without validation."""
        return self._dag


# ===================================================================
#  sklearn introspection helpers
# ===================================================================


def _extract_sklearn_steps(
    pipeline: Any,
) -> list[Tuple[str, Any, str]]:
    """Extract (name, estimator, step_type) triples from an sklearn pipeline.

    Handles Pipeline, ColumnTransformer, and FeatureUnion.
    """
    steps: list[Tuple[str, Any, str]] = []

    if hasattr(pipeline, "steps"):
        for name, estimator in pipeline.steps:
            if _is_column_transformer(estimator):
                steps.extend(_extract_column_transformer_steps(name, estimator))
            elif _is_feature_union(estimator):
                steps.extend(_extract_feature_union_steps(name, estimator))
            elif hasattr(estimator, "predict"):
                steps.append((name, estimator, "fit"))
                steps.append((name, estimator, "predict"))
            elif hasattr(estimator, "transform"):
                steps.append((name, estimator, "fit_transform"))
            else:
                steps.append((name, estimator, "fit_transform"))
    elif _is_column_transformer(pipeline):
        steps.extend(_extract_column_transformer_steps("ct", pipeline))
    elif _is_feature_union(pipeline):
        steps.extend(_extract_feature_union_steps("fu", pipeline))
    elif hasattr(pipeline, "fit"):
        name = type(pipeline).__name__.lower()
        if hasattr(pipeline, "predict"):
            steps.append((name, pipeline, "fit"))
            steps.append((name, pipeline, "predict"))
        else:
            steps.append((name, pipeline, "fit_transform"))
    return steps


def _is_column_transformer(obj: Any) -> bool:
    cls_name = type(obj).__name__
    return cls_name == "ColumnTransformer" or (
        hasattr(obj, "transformers") and hasattr(obj, "remainder")
    )


def _is_feature_union(obj: Any) -> bool:
    cls_name = type(obj).__name__
    return cls_name == "FeatureUnion" or (
        hasattr(obj, "transformer_list") and not hasattr(obj, "steps")
    )


def _extract_column_transformer_steps(
    parent_name: str,
    ct: Any,
) -> list[Tuple[str, Any, str]]:
    """Extract steps from a ColumnTransformer."""
    steps: list[Tuple[str, Any, str]] = []
    transformers = getattr(ct, "transformers", [])
    for item in transformers:
        if len(item) >= 3:
            name, estimator, columns = item[0], item[1], item[2]
        elif len(item) == 2:
            name, estimator = item
            columns = []
        else:
            continue
        step_name = f"{parent_name}__{name}"
        if hasattr(estimator, "steps"):
            sub_steps = _extract_sklearn_steps(estimator)
            for sn, se, st in sub_steps:
                steps.append((f"{step_name}__{sn}", se, st))
        else:
            steps.append((step_name, estimator, "fit_transform"))
    return steps


def _extract_feature_union_steps(
    parent_name: str,
    fu: Any,
) -> list[Tuple[str, Any, str]]:
    """Extract steps from a FeatureUnion."""
    steps: list[Tuple[str, Any, str]] = []
    transformer_list = getattr(fu, "transformer_list", [])
    for name, estimator in transformer_list:
        step_name = f"{parent_name}__{name}"
        if hasattr(estimator, "steps"):
            sub_steps = _extract_sklearn_steps(estimator)
            for sn, se, st in sub_steps:
                steps.append((f"{step_name}__{sn}", se, st))
        else:
            steps.append((step_name, estimator, "fit_transform"))
    return steps


def _sklearn_estimator_to_optype(estimator: Any, step_type: str) -> OpType:
    """Map an sklearn estimator + step type to an OpType."""
    if step_type == "fit":
        return OpType.FIT
    if step_type == "predict":
        return OpType.PREDICT
    if step_type == "fit_transform":
        return OpType.FIT_TRANSFORM
    if step_type == "transform":
        return OpType.TRANSFORM_SK

    cls_name = type(estimator).__name__ if estimator is not None else ""
    scaler_map = {
        "StandardScaler": OpType.STANDARD_SCALER,
        "MinMaxScaler": OpType.MINMAX_SCALER,
        "RobustScaler": OpType.ROBUST_SCALER,
        "Normalizer": OpType.NORMALIZER,
        "LabelEncoder": OpType.LABEL_ENCODER,
        "OrdinalEncoder": OpType.ORDINAL_ENCODER,
        "OneHotEncoder": OpType.ONEHOT_ENCODER,
        "TargetEncoder": OpType.TARGET_ENCODER,
        "PolynomialFeatures": OpType.POLYNOMIAL_FEATURES,
        "KBinsDiscretizer": OpType.KBINS_DISCRETIZER,
        "Binarizer": OpType.BINARIZER,
        "SimpleImputer": OpType.IMPUTER,
        "KNNImputer": OpType.KNN_IMPUTER,
    }
    if cls_name in scaler_map:
        return scaler_map[cls_name]

    return OpType.FIT_TRANSFORM


def _infer_sklearn_output_features(
    estimator: Any,
    input_features: list[str],
) -> list[str]:
    """Infer output feature names from an sklearn estimator."""
    if hasattr(estimator, "get_feature_names_out"):
        try:
            names = estimator.get_feature_names_out()
            return list(names)
        except Exception:
            pass

    if hasattr(estimator, "get_feature_names"):
        try:
            names = estimator.get_feature_names()
            return list(names)
        except Exception:
            pass

    cls_name = type(estimator).__name__ if estimator is not None else ""

    if cls_name == "OneHotEncoder":
        categories = getattr(estimator, "categories_", None)
        if categories and input_features:
            out: list[str] = []
            for i, cats in enumerate(categories):
                feat = input_features[i] if i < len(input_features) else f"x{i}"
                for cat in cats:
                    out.append(f"{feat}_{cat}")
            return out

    if cls_name == "PolynomialFeatures":
        n_features = getattr(estimator, "n_output_features_", None)
        if n_features:
            return [f"poly_{i}" for i in range(n_features)]

    if cls_name in ("PCA", "TruncatedSVD"):
        n_components = getattr(estimator, "n_components_", getattr(estimator, "n_components", None))
        if n_components:
            prefix = "pca" if cls_name == "PCA" else "svd"
            return [f"{prefix}_{i}" for i in range(n_components)]

    return list(input_features)
