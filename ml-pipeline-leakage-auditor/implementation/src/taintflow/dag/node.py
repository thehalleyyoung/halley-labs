"""
taintflow.dag.node -- DAG node types for the Pipeline Information DAG (PI-DAG).

Every operation intercepted by the TaintFlow instrumentation layer is
represented as a :class:`PipelineNode` (or one of its specialised subclasses).
Nodes carry their operation type, source location, input/output schemas,
shape metadata, and per-column provenance information.

Specialised subclasses encode additional semantics:

* :class:`DataSourceNode` -- data ingestion (``read_csv``, ``read_parquet``, ...)
* :class:`PartitionNode` -- train/test splitting (``train_test_split``, k-fold)
* :class:`TransformNode` -- sklearn transformers (scalers, encoders, ...)
* :class:`FitNode` -- estimator fitting
* :class:`PredictNode` -- estimator prediction
* :class:`PandasOpNode` -- pandas DataFrame operations
* :class:`AggregationNode` -- groupby / rolling / expanding aggregations
* :class:`FeatureEngineeringNode` -- feature engineering (polynomial, target enc.)
* :class:`SelectionNode` -- column / row selection
* :class:`CustomNode` -- user-defined operations (gets ``B_max`` bound)
* :class:`SinkNode` -- terminal output nodes
"""

from __future__ import annotations

import copy
import hashlib
import math
import time
import uuid
from dataclasses import dataclass, field
from typing import (
    Any,
    ClassVar,
    Dict,
    FrozenSet,
    List,
    Mapping,
    Optional,
    Sequence,
    Set,
    Tuple,
    Type,
    Union,
)

from taintflow.core.types import (
    ColumnSchema,
    EdgeKind,
    NodeKind,
    OpType,
    Origin,
    ProvenanceInfo,
    Severity,
    ShapeMetadata,
)


# ===================================================================
#  Source location
# ===================================================================


@dataclass(frozen=True)
class SourceLocation:
    """Location in user source code where an operation was invoked."""

    file: str = "<unknown>"
    line: int = 0
    col: int = 0
    function_name: str = ""
    class_name: str = ""

    # -- helpers ------------------------------------------------------------

    @property
    def qualified_name(self) -> str:
        """Return ``ClassName.function_name`` or just ``function_name``."""
        if self.class_name:
            return f"{self.class_name}.{self.function_name}"
        return self.function_name

    @property
    def short_str(self) -> str:
        """Compact ``file:line`` representation."""
        return f"{self.file}:{self.line}"

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"file": self.file, "line": self.line}
        if self.col:
            d["col"] = self.col
        if self.function_name:
            d["function_name"] = self.function_name
        if self.class_name:
            d["class_name"] = self.class_name
        return d

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "SourceLocation":
        return cls(
            file=str(data.get("file", "<unknown>")),
            line=int(data.get("line", 0)),
            col=int(data.get("col", 0)),
            function_name=str(data.get("function_name", "")),
            class_name=str(data.get("class_name", "")),
        )

    def __repr__(self) -> str:
        return f"SourceLocation({self.short_str}, fn={self.qualified_name!r})"

    def validate(self) -> list[str]:
        errors: list[str] = []
        if self.line < 0:
            errors.append(f"line must be >= 0, got {self.line}")
        if self.col < 0:
            errors.append(f"col must be >= 0, got {self.col}")
        return errors


# ===================================================================
#  Base pipeline node
# ===================================================================

_ESTIMATOR_OPS: frozenset[OpType] = frozenset({
    OpType.FIT,
    OpType.PREDICT,
    OpType.FIT_TRANSFORM,
    OpType.TRANSFORM_SK,
    OpType.PREDICT_PROBA,
    OpType.PREDICT_LOG_PROBA,
    OpType.DECISION_FUNCTION,
    OpType.SCORE,
    OpType.FIT_PREDICT,
    OpType.INVERSE_TRANSFORM,
})

_FIT_OPS: frozenset[OpType] = frozenset({
    OpType.FIT,
    OpType.FIT_TRANSFORM,
    OpType.FIT_PREDICT,
})

_AGGREGATION_OPS: frozenset[OpType] = frozenset({
    OpType.GROUPBY,
    OpType.AGG,
    OpType.AGGREGATE,
    OpType.ROLLING,
    OpType.EXPANDING,
    OpType.EWM,
    OpType.RESAMPLE,
    OpType.VALUE_COUNTS,
    OpType.DESCRIBE,
    OpType.CORR,
    OpType.COV,
    OpType.CUMSUM,
    OpType.CUMPROD,
    OpType.CUMMAX,
    OpType.CUMMIN,
    OpType.NP_MEAN,
    OpType.NP_STD,
    OpType.NP_VAR,
    OpType.NP_MEDIAN,
    OpType.NP_SUM,
})

_SCHEMA_MODIFYING_OPS: frozenset[OpType] = frozenset({
    OpType.DROP,
    OpType.RENAME,
    OpType.ASSIGN,
    OpType.INSERT,
    OpType.POP,
    OpType.SET_INDEX,
    OpType.RESET_INDEX,
    OpType.GET_DUMMIES,
    OpType.PIVOT,
    OpType.PIVOT_TABLE,
    OpType.MELT,
    OpType.STACK,
    OpType.UNSTACK,
    OpType.EXPLODE,
    OpType.MERGE,
    OpType.JOIN,
    OpType.CONCAT,
    OpType.ONEHOT_ENCODER,
    OpType.POLYNOMIAL_FEATURES,
})


def _generate_node_id(prefix: str = "node") -> str:
    """Generate a short unique node identifier."""
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


def _deterministic_node_id(op_type: OpType, source: SourceLocation) -> str:
    """Deterministic ID from operation + location for deduplication."""
    raw = f"{op_type.value}:{source.file}:{source.line}:{source.col}"
    digest = hashlib.sha256(raw.encode()).hexdigest()[:12]
    return f"{op_type.value}_{digest}"


@dataclass
class PipelineNode:
    """Base class for all PI-DAG nodes.

    Parameters
    ----------
    node_id : str
        Unique identifier within the DAG.
    op_type : OpType
        The kind of operation this node represents.
    source_location : SourceLocation
        Where in user code the operation was invoked.
    input_schema : list[ColumnSchema]
        Schema of columns flowing *into* this node.
    output_schema : list[ColumnSchema]
        Schema of columns flowing *out of* this node.
    shape : ShapeMetadata
        Row/column counts at this point.
    provenance : dict[str, ProvenanceInfo]
        Per-column provenance (column name -> provenance info).
    metadata : dict[str, Any]
        Arbitrary additional data attached by instrumentation.
    timestamp : float
        Wall-clock time at which the operation was observed.
    """

    node_id: str
    op_type: OpType
    source_location: SourceLocation = field(default_factory=SourceLocation)
    input_schema: list[ColumnSchema] = field(default_factory=list)
    output_schema: list[ColumnSchema] = field(default_factory=list)
    shape: ShapeMetadata = field(default_factory=lambda: ShapeMetadata(n_rows=0, n_cols=0))
    provenance: dict[str, ProvenanceInfo] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

    # Class-level constant for node kind mapping
    NODE_KIND: ClassVar[NodeKind] = NodeKind.UNKNOWN

    # -- query helpers -------------------------------------------------------

    @property
    def node_kind(self) -> NodeKind:
        """Derive the :class:`NodeKind` from :attr:`op_type`."""
        return self._infer_node_kind()

    def _infer_node_kind(self) -> NodeKind:
        """Map OpType to NodeKind."""
        op = self.op_type
        if op in {OpType.READ_CSV, OpType.READ_PARQUET, OpType.READ_JSON,
                  OpType.READ_EXCEL, OpType.READ_SQL, OpType.READ_HDF,
                  OpType.READ_FEATHER}:
            return NodeKind.DATA_SOURCE
        if op in {OpType.TRAIN_TEST_SPLIT, OpType.KFOLD_SPLIT,
                  OpType.STRATIFIED_KFOLD, OpType.GROUP_KFOLD,
                  OpType.LEAVE_ONE_OUT}:
            return NodeKind.SPLIT
        if op in {OpType.FIT}:
            return NodeKind.ESTIMATOR_FIT
        if op in {OpType.PREDICT, OpType.PREDICT_PROBA,
                  OpType.PREDICT_LOG_PROBA, OpType.DECISION_FUNCTION}:
            return NodeKind.ESTIMATOR_PREDICT
        if op in {OpType.TRANSFORM_SK, OpType.FIT_TRANSFORM}:
            return NodeKind.ESTIMATOR_TRANSFORM
        if op in {OpType.MERGE, OpType.JOIN, OpType.CONCAT, OpType.APPEND}:
            return NodeKind.MERGE
        if op in {OpType.SCORE}:
            return NodeKind.EVALUATION
        if op in {OpType.TO_CSV, OpType.TO_PARQUET}:
            return NodeKind.SINK
        if op in {OpType.POLYNOMIAL_FEATURES, OpType.GET_DUMMIES,
                  OpType.TARGET_ENCODER, OpType.ONEHOT_ENCODER}:
            return NodeKind.FEATURE_ENGINEERING
        if op.is_aggregation:
            return NodeKind.TRANSFORM
        return NodeKind.TRANSFORM

    @property
    def is_estimator(self) -> bool:
        """True if this node involves an sklearn estimator operation."""
        return self.op_type in _ESTIMATOR_OPS

    @property
    def is_aggregation(self) -> bool:
        """True if this node performs aggregation."""
        return self.op_type in _AGGREGATION_OPS

    @property
    def has_fit(self) -> bool:
        """True if this node involves a fit step."""
        return self.op_type in _FIT_OPS

    @property
    def modifies_schema(self) -> bool:
        """True if the output schema may differ from the input schema."""
        return self.op_type in _SCHEMA_MODIFYING_OPS

    @property
    def input_columns(self) -> frozenset[str]:
        """Names of columns in the input schema."""
        return frozenset(c.name for c in self.input_schema)

    @property
    def output_columns(self) -> frozenset[str]:
        """Names of columns in the output schema."""
        return frozenset(c.name for c in self.output_schema)

    def column_mapping(self) -> dict[str, set[str]]:
        """Map each output column to the set of input columns it depends on.

        Returns a conservative over-approximation: for operations we don't
        have detailed column-lineage for, every output depends on every input.
        """
        in_cols = self.input_columns
        out_cols = self.output_columns

        if self.op_type == OpType.DROP:
            dropped = in_cols - out_cols
            return {oc: {oc} for oc in out_cols if oc in in_cols}

        if self.op_type == OpType.RENAME:
            rename_map = self.metadata.get("rename_map", {})
            result: dict[str, set[str]] = {}
            for oc in out_cols:
                reverse_map = {v: k for k, v in rename_map.items()}
                if oc in reverse_map:
                    result[oc] = {reverse_map[oc]}
                elif oc in in_cols:
                    result[oc] = {oc}
                else:
                    result[oc] = set(in_cols)
            return result

        if self.op_type in {OpType.GETITEM, OpType.LOC, OpType.ILOC,
                            OpType.FILTER, OpType.HEAD, OpType.TAIL}:
            return {oc: {oc} for oc in out_cols if oc in in_cols}

        if self.op_type == OpType.ASSIGN:
            new_cols = self.metadata.get("new_columns", set())
            result = {}
            for oc in out_cols:
                if oc in new_cols:
                    result[oc] = set(in_cols)
                elif oc in in_cols:
                    result[oc] = {oc}
                else:
                    result[oc] = set(in_cols)
            return result

        if self.op_type in {OpType.MERGE, OpType.JOIN}:
            left_cols = self.metadata.get("left_columns", set())
            right_cols = self.metadata.get("right_columns", set())
            key_cols = self.metadata.get("key_columns", set())
            result = {}
            for oc in out_cols:
                deps: set[str] = set()
                if oc in left_cols:
                    deps.add(oc)
                if oc in right_cols:
                    deps.add(oc)
                if oc in key_cols:
                    deps.add(oc)
                if not deps:
                    deps = set(in_cols)
                result[oc] = deps
            return result

        if self.op_type in {OpType.PIVOT, OpType.PIVOT_TABLE,
                            OpType.MELT, OpType.STACK, OpType.UNSTACK}:
            return {oc: set(in_cols) for oc in out_cols}

        if self.op_type == OpType.COPY or self.op_type == OpType.DEEPCOPY:
            return {oc: {oc} for oc in out_cols if oc in in_cols}

        if self.op_type == OpType.IDENTITY:
            return {oc: {oc} for oc in out_cols if oc in in_cols}

        passed = out_cols & in_cols
        new_out = out_cols - in_cols
        result = {oc: {oc} for oc in passed}
        for oc in new_out:
            result[oc] = set(in_cols)
        return result

    # -- provenance helpers --------------------------------------------------

    def get_provenance(self, column: str) -> ProvenanceInfo:
        """Get provenance for a specific column, defaulting to pure-train."""
        return self.provenance.get(
            column,
            ProvenanceInfo(test_fraction=0.0, origin_set=frozenset({Origin.TRAIN})),
        )

    def max_test_fraction(self) -> float:
        """Maximum test-fraction (ρ) across all columns."""
        if not self.provenance:
            return 0.0
        return max(p.test_fraction for p in self.provenance.values())

    def mixed_columns(self) -> set[str]:
        """Columns whose provenance includes both TRAIN and TEST origins."""
        return {
            col for col, prov in self.provenance.items()
            if prov.is_mixed
        }

    def test_columns(self) -> set[str]:
        """Columns whose provenance is purely TEST."""
        return {
            col for col, prov in self.provenance.items()
            if prov.is_pure_test
        }

    # -- capacity / bit bounds -----------------------------------------------

    def capacity_bound(self) -> float:
        """Upper bound on channel capacity in bits for this operation.

        Subclasses override to provide tighter bounds.
        """
        if self.op_type == OpType.IDENTITY or self.op_type == OpType.COPY:
            return 0.0
        total_bits = 0.0
        for col in self.output_schema:
            total_bits += col.entropy_bound()
        return total_bits

    # -- validation ----------------------------------------------------------

    def validate(self) -> list[str]:
        """Validate node invariants, returning a list of error messages."""
        errors: list[str] = []
        if not self.node_id:
            errors.append("node_id must be non-empty")
        errors.extend(self.source_location.validate())
        for cs in self.input_schema:
            for err in cs.validate():
                errors.append(f"input_schema[{cs.name}]: {err}")
        for cs in self.output_schema:
            for err in cs.validate():
                errors.append(f"output_schema[{cs.name}]: {err}")
        errors.extend(self.shape.validate())
        for col, prov in self.provenance.items():
            for err in prov.validate():
                errors.append(f"provenance[{col}]: {err}")
        return errors

    # -- serialization -------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dictionary."""
        return {
            "node_type": type(self).__name__,
            "node_id": self.node_id,
            "op_type": self.op_type.value,
            "source_location": self.source_location.to_dict(),
            "input_schema": [c.to_dict() for c in self.input_schema],
            "output_schema": [c.to_dict() for c in self.output_schema],
            "shape": self.shape.to_dict(),
            "provenance": {k: v.to_dict() for k, v in self.provenance.items()},
            "metadata": _sanitize_metadata(self.metadata),
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "PipelineNode":
        """Deserialize from a dictionary, dispatching to the right subclass."""
        node_type_name = data.get("node_type", "PipelineNode")
        node_cls = _NODE_REGISTRY.get(node_type_name, cls)
        op_type = OpType(data["op_type"])
        source_location = SourceLocation.from_dict(data.get("source_location", {}))
        input_schema = [ColumnSchema.from_dict(c) for c in data.get("input_schema", [])]
        output_schema = [ColumnSchema.from_dict(c) for c in data.get("output_schema", [])]
        shape = ShapeMetadata.from_dict(data["shape"]) if "shape" in data else ShapeMetadata(n_rows=0, n_cols=0)
        provenance = {
            k: ProvenanceInfo.from_dict(v)
            for k, v in data.get("provenance", {}).items()
        }
        metadata = dict(data.get("metadata", {}))
        timestamp = float(data.get("timestamp", 0.0))

        base_kwargs: dict[str, Any] = dict(
            node_id=str(data["node_id"]),
            op_type=op_type,
            source_location=source_location,
            input_schema=input_schema,
            output_schema=output_schema,
            shape=shape,
            provenance=provenance,
            metadata=metadata,
            timestamp=timestamp,
        )

        if node_cls is DataSourceNode:
            return DataSourceNode(
                **base_kwargs,
                file_path=metadata.get("file_path", ""),
                format=metadata.get("format", "csv"),
                origin=Origin.from_str(metadata.get("origin", "train")),
            )
        if node_cls is PartitionNode:
            return PartitionNode(
                **base_kwargs,
                test_size=float(metadata.get("test_size", 0.25)),
                random_state=metadata.get("random_state"),
                stratify_column=metadata.get("stratify_column"),
                n_splits=int(metadata.get("n_splits", 1)),
            )
        if node_cls is TransformNode:
            return TransformNode(
                **base_kwargs,
                estimator_class=str(metadata.get("estimator_class", "")),
                is_fitted=bool(metadata.get("is_fitted", False)),
                params=dict(metadata.get("params", {})),
            )
        if node_cls is FitNode:
            return FitNode(
                **base_kwargs,
                estimator_class=str(metadata.get("estimator_class", "")),
                params=dict(metadata.get("params", {})),
                n_features_in=metadata.get("n_features_in"),
                n_samples=metadata.get("n_samples"),
            )
        if node_cls is PredictNode:
            return PredictNode(
                **base_kwargs,
                estimator_class=str(metadata.get("estimator_class", "")),
                predict_type=str(metadata.get("predict_type", "predict")),
            )
        if node_cls is PandasOpNode:
            return PandasOpNode(
                **base_kwargs,
                pandas_method=str(metadata.get("pandas_method", "")),
                axis=metadata.get("axis"),
                inplace=bool(metadata.get("inplace", False)),
            )
        if node_cls is AggregationNode:
            return AggregationNode(
                **base_kwargs,
                groupby_columns=list(metadata.get("groupby_columns", [])),
                agg_functions=list(metadata.get("agg_functions", [])),
                as_index=bool(metadata.get("as_index", True)),
            )
        if node_cls is FeatureEngineeringNode:
            return FeatureEngineeringNode(
                **base_kwargs,
                technique=str(metadata.get("technique", "")),
                source_columns=list(metadata.get("source_columns", [])),
                generated_columns=list(metadata.get("generated_columns", [])),
            )
        if node_cls is SelectionNode:
            return SelectionNode(
                **base_kwargs,
                selected_columns=list(metadata.get("selected_columns", [])),
                selection_method=str(metadata.get("selection_method", "manual")),
                row_mask_description=str(metadata.get("row_mask_description", "")),
            )
        if node_cls is CustomNode:
            return CustomNode(
                **base_kwargs,
                function_name=str(metadata.get("function_name", "")),
                source_code_hash=str(metadata.get("source_code_hash", "")),
            )
        if node_cls is SinkNode:
            return SinkNode(
                **base_kwargs,
                sink_type=str(metadata.get("sink_type", "return")),
                sink_target=str(metadata.get("sink_target", "")),
            )
        return PipelineNode(**base_kwargs)

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(id={self.node_id!r}, "
            f"op={self.op_type.name}, "
            f"in={len(self.input_schema)}cols, "
            f"out={len(self.output_schema)}cols)"
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, PipelineNode):
            return NotImplemented
        return self.node_id == other.node_id

    def __hash__(self) -> int:
        return hash(self.node_id)

    def clone(self, new_id: str | None = None) -> "PipelineNode":
        """Return a deep copy, optionally with a new node_id."""
        cloned = copy.deepcopy(self)
        if new_id is not None:
            cloned.node_id = new_id
        return cloned


# ===================================================================
#  Specialised node subclasses
# ===================================================================


@dataclass
class DataSourceNode(PipelineNode):
    """Node representing data ingestion (read_csv, read_parquet, etc.)."""

    NODE_KIND: ClassVar[NodeKind] = NodeKind.DATA_SOURCE

    file_path: str = ""
    format: str = "csv"
    origin: Origin = Origin.TRAIN

    def _infer_node_kind(self) -> NodeKind:
        return NodeKind.DATA_SOURCE

    def capacity_bound(self) -> float:
        if self.origin == Origin.TRAIN:
            return 0.0
        total = 0.0
        for col in self.output_schema:
            total += col.entropy_bound()
        n = max(self.shape.n_rows, 1)
        return total * n

    def to_dict(self) -> dict[str, Any]:
        d = super().to_dict()
        d["metadata"]["file_path"] = self.file_path
        d["metadata"]["format"] = self.format
        d["metadata"]["origin"] = self.origin.value
        return d

    def validate(self) -> list[str]:
        errors = super().validate()
        if self.format not in {"csv", "parquet", "json", "excel", "sql", "hdf", "feather"}:
            errors.append(f"Unknown data source format: {self.format!r}")
        return errors

    def __repr__(self) -> str:
        return (
            f"DataSourceNode(id={self.node_id!r}, "
            f"file={self.file_path!r}, origin={self.origin.name})"
        )


@dataclass
class PartitionNode(PipelineNode):
    """Node representing train/test splitting."""

    NODE_KIND: ClassVar[NodeKind] = NodeKind.SPLIT

    test_size: float = 0.25
    random_state: int | None = None
    stratify_column: str | None = None
    n_splits: int = 1

    def _infer_node_kind(self) -> NodeKind:
        return NodeKind.SPLIT

    def capacity_bound(self) -> float:
        n = max(self.shape.n_rows, 1)
        return math.log2(math.comb(n, max(int(n * self.test_size), 1))) if n > 1 else 0.0

    def to_dict(self) -> dict[str, Any]:
        d = super().to_dict()
        d["metadata"]["test_size"] = self.test_size
        d["metadata"]["random_state"] = self.random_state
        d["metadata"]["stratify_column"] = self.stratify_column
        d["metadata"]["n_splits"] = self.n_splits
        return d

    def validate(self) -> list[str]:
        errors = super().validate()
        if not (0.0 < self.test_size < 1.0):
            errors.append(f"test_size must be in (0,1), got {self.test_size}")
        if self.n_splits < 1:
            errors.append(f"n_splits must be >= 1, got {self.n_splits}")
        return errors

    def __repr__(self) -> str:
        return (
            f"PartitionNode(id={self.node_id!r}, "
            f"test_size={self.test_size}, n_splits={self.n_splits})"
        )


@dataclass
class TransformNode(PipelineNode):
    """Node representing an sklearn transformer application."""

    NODE_KIND: ClassVar[NodeKind] = NodeKind.ESTIMATOR_TRANSFORM

    estimator_class: str = ""
    is_fitted: bool = False
    params: dict[str, Any] = field(default_factory=dict)

    def _infer_node_kind(self) -> NodeKind:
        return NodeKind.ESTIMATOR_TRANSFORM

    def capacity_bound(self) -> float:
        n_out = len(self.output_schema)
        n_rows = max(self.shape.n_rows, 1)
        if self.estimator_class in ("StandardScaler", "MinMaxScaler", "RobustScaler"):
            return float(n_out) * 2.0
        if self.estimator_class in ("OneHotEncoder", "OrdinalEncoder"):
            total = 0.0
            for col in self.output_schema:
                total += col.entropy_bound()
            return total
        return float(n_out) * 64.0

    def to_dict(self) -> dict[str, Any]:
        d = super().to_dict()
        d["metadata"]["estimator_class"] = self.estimator_class
        d["metadata"]["is_fitted"] = self.is_fitted
        d["metadata"]["params"] = self.params
        return d

    def __repr__(self) -> str:
        return (
            f"TransformNode(id={self.node_id!r}, "
            f"estimator={self.estimator_class!r}, fitted={self.is_fitted})"
        )


@dataclass
class FitNode(PipelineNode):
    """Node representing estimator fitting."""

    NODE_KIND: ClassVar[NodeKind] = NodeKind.ESTIMATOR_FIT

    estimator_class: str = ""
    params: dict[str, Any] = field(default_factory=dict)
    n_features_in: int | None = None
    n_samples: int | None = None

    def _infer_node_kind(self) -> NodeKind:
        return NodeKind.ESTIMATOR_FIT

    def capacity_bound(self) -> float:
        n_features = self.n_features_in or len(self.input_schema)
        n_samples = self.n_samples or max(self.shape.n_rows, 1)
        if self.estimator_class in ("LinearRegression", "Ridge", "Lasso", "LogisticRegression"):
            return float(n_features) * 64.0
        if self.estimator_class in ("DecisionTreeClassifier", "DecisionTreeRegressor"):
            return float(n_samples) * math.log2(max(n_features, 2))
        return float(n_features) * float(n_samples)

    def to_dict(self) -> dict[str, Any]:
        d = super().to_dict()
        d["metadata"]["estimator_class"] = self.estimator_class
        d["metadata"]["params"] = self.params
        d["metadata"]["n_features_in"] = self.n_features_in
        d["metadata"]["n_samples"] = self.n_samples
        return d

    def __repr__(self) -> str:
        return (
            f"FitNode(id={self.node_id!r}, "
            f"estimator={self.estimator_class!r}, "
            f"features={self.n_features_in})"
        )


@dataclass
class PredictNode(PipelineNode):
    """Node representing estimator prediction."""

    NODE_KIND: ClassVar[NodeKind] = NodeKind.ESTIMATOR_PREDICT

    estimator_class: str = ""
    predict_type: str = "predict"

    def _infer_node_kind(self) -> NodeKind:
        return NodeKind.ESTIMATOR_PREDICT

    def capacity_bound(self) -> float:
        if self.predict_type == "predict":
            return 64.0
        if self.predict_type in ("predict_proba", "predict_log_proba"):
            n_classes = self.metadata.get("n_classes", 2)
            return float(n_classes) * 64.0
        if self.predict_type == "decision_function":
            return 64.0
        return 64.0

    def to_dict(self) -> dict[str, Any]:
        d = super().to_dict()
        d["metadata"]["estimator_class"] = self.estimator_class
        d["metadata"]["predict_type"] = self.predict_type
        return d

    def __repr__(self) -> str:
        return (
            f"PredictNode(id={self.node_id!r}, "
            f"estimator={self.estimator_class!r}, "
            f"type={self.predict_type!r})"
        )


@dataclass
class PandasOpNode(PipelineNode):
    """Node representing a pandas DataFrame operation."""

    NODE_KIND: ClassVar[NodeKind] = NodeKind.TRANSFORM

    pandas_method: str = ""
    axis: int | None = None
    inplace: bool = False

    def _infer_node_kind(self) -> NodeKind:
        if self.op_type in {OpType.MERGE, OpType.JOIN, OpType.CONCAT, OpType.APPEND}:
            return NodeKind.MERGE
        return NodeKind.TRANSFORM

    def capacity_bound(self) -> float:
        if self.op_type in {OpType.COPY, OpType.DEEPCOPY, OpType.IDENTITY,
                            OpType.SORT_VALUES, OpType.SORT_INDEX, OpType.RESET_INDEX}:
            return 0.0
        if self.op_type in {OpType.DROP, OpType.DROPNA, OpType.DROP_DUPLICATES}:
            return 0.0
        if self.op_type in {OpType.FILLNA, OpType.INTERPOLATE, OpType.REPLACE}:
            n_affected = len(self.output_schema)
            return float(n_affected) * 64.0
        if self.op_type in {OpType.MERGE, OpType.JOIN}:
            left_n = self.metadata.get("left_rows", self.shape.n_rows)
            right_n = self.metadata.get("right_rows", 0)
            n_cols = len(self.output_schema)
            return float(n_cols) * math.log2(max(left_n + right_n, 2))
        total = 0.0
        for col in self.output_schema:
            total += col.entropy_bound()
        return total

    def to_dict(self) -> dict[str, Any]:
        d = super().to_dict()
        d["metadata"]["pandas_method"] = self.pandas_method
        d["metadata"]["axis"] = self.axis
        d["metadata"]["inplace"] = self.inplace
        return d

    def __repr__(self) -> str:
        return (
            f"PandasOpNode(id={self.node_id!r}, "
            f"method={self.pandas_method!r}, "
            f"op={self.op_type.name})"
        )


@dataclass
class AggregationNode(PipelineNode):
    """Node representing a groupby / rolling / expanding aggregation."""

    NODE_KIND: ClassVar[NodeKind] = NodeKind.TRANSFORM

    groupby_columns: list[str] = field(default_factory=list)
    agg_functions: list[str] = field(default_factory=list)
    as_index: bool = True

    def _infer_node_kind(self) -> NodeKind:
        return NodeKind.TRANSFORM

    def capacity_bound(self) -> float:
        n_groups = self.metadata.get("n_groups", max(self.shape.n_rows, 1))
        n_agg = max(len(self.agg_functions), 1)
        n_cols = max(len(self.output_schema), 1)
        return float(n_cols) * math.log2(max(n_groups, 2)) * n_agg

    def column_mapping(self) -> dict[str, set[str]]:
        in_cols = self.input_columns
        result: dict[str, set[str]] = {}
        for oc in self.output_columns:
            base_name = oc.split("_")[0] if "_" in oc else oc
            if base_name in in_cols:
                deps = {base_name} | set(self.groupby_columns)
                result[oc] = deps
            else:
                result[oc] = set(in_cols)
        return result

    def to_dict(self) -> dict[str, Any]:
        d = super().to_dict()
        d["metadata"]["groupby_columns"] = self.groupby_columns
        d["metadata"]["agg_functions"] = self.agg_functions
        d["metadata"]["as_index"] = self.as_index
        return d

    def __repr__(self) -> str:
        return (
            f"AggregationNode(id={self.node_id!r}, "
            f"by={self.groupby_columns!r}, "
            f"agg={self.agg_functions!r})"
        )


@dataclass
class FeatureEngineeringNode(PipelineNode):
    """Node for feature engineering operations."""

    NODE_KIND: ClassVar[NodeKind] = NodeKind.FEATURE_ENGINEERING

    technique: str = ""
    source_columns: list[str] = field(default_factory=list)
    generated_columns: list[str] = field(default_factory=list)

    def _infer_node_kind(self) -> NodeKind:
        return NodeKind.FEATURE_ENGINEERING

    def capacity_bound(self) -> float:
        n_source = max(len(self.source_columns), 1)
        n_generated = max(len(self.generated_columns), 1)
        if self.technique in ("polynomial", "polynomial_features"):
            degree = self.metadata.get("degree", 2)
            return float(n_generated) * float(degree) * 64.0
        if self.technique in ("target_encoding", "target_encoder"):
            return float(n_generated) * 64.0
        if self.technique in ("onehot", "one_hot", "get_dummies"):
            total = 0.0
            for col in self.output_schema:
                total += col.entropy_bound()
            return total
        return float(n_generated) * 64.0

    def column_mapping(self) -> dict[str, set[str]]:
        result: dict[str, set[str]] = {}
        source_set = set(self.source_columns)
        generated_set = set(self.generated_columns)
        for oc in self.output_columns:
            if oc in generated_set:
                result[oc] = source_set if source_set else set(self.input_columns)
            elif oc in self.input_columns:
                result[oc] = {oc}
            else:
                result[oc] = set(self.input_columns)
        return result

    def to_dict(self) -> dict[str, Any]:
        d = super().to_dict()
        d["metadata"]["technique"] = self.technique
        d["metadata"]["source_columns"] = self.source_columns
        d["metadata"]["generated_columns"] = self.generated_columns
        return d

    def __repr__(self) -> str:
        return (
            f"FeatureEngineeringNode(id={self.node_id!r}, "
            f"technique={self.technique!r}, "
            f"generated={len(self.generated_columns)}cols)"
        )


@dataclass
class SelectionNode(PipelineNode):
    """Node representing column/row selection."""

    NODE_KIND: ClassVar[NodeKind] = NodeKind.TRANSFORM

    selected_columns: list[str] = field(default_factory=list)
    selection_method: str = "manual"
    row_mask_description: str = ""

    def _infer_node_kind(self) -> NodeKind:
        return NodeKind.TRANSFORM

    def capacity_bound(self) -> float:
        return 0.0

    def column_mapping(self) -> dict[str, set[str]]:
        return {oc: {oc} for oc in self.output_columns if oc in self.input_columns}

    def to_dict(self) -> dict[str, Any]:
        d = super().to_dict()
        d["metadata"]["selected_columns"] = self.selected_columns
        d["metadata"]["selection_method"] = self.selection_method
        d["metadata"]["row_mask_description"] = self.row_mask_description
        return d

    def __repr__(self) -> str:
        return (
            f"SelectionNode(id={self.node_id!r}, "
            f"columns={self.selected_columns!r})"
        )


@dataclass
class CustomNode(PipelineNode):
    """User-defined operation node.

    Custom nodes are conservatively assigned ``B_max`` capacity because
    we cannot statically analyse what they do.
    """

    NODE_KIND: ClassVar[NodeKind] = NodeKind.UNKNOWN

    function_name: str = ""
    source_code_hash: str = ""
    _B_MAX: ClassVar[float] = 64.0

    def _infer_node_kind(self) -> NodeKind:
        return NodeKind.UNKNOWN

    def capacity_bound(self) -> float:
        n_cols = max(len(self.output_schema), 1)
        n_rows = max(self.shape.n_rows, 1)
        return float(n_cols) * self._B_MAX

    def column_mapping(self) -> dict[str, set[str]]:
        return {oc: set(self.input_columns) for oc in self.output_columns}

    def to_dict(self) -> dict[str, Any]:
        d = super().to_dict()
        d["metadata"]["function_name"] = self.function_name
        d["metadata"]["source_code_hash"] = self.source_code_hash
        return d

    def __repr__(self) -> str:
        return (
            f"CustomNode(id={self.node_id!r}, "
            f"fn={self.function_name!r})"
        )


@dataclass
class SinkNode(PipelineNode):
    """Terminal output node (model save, to_csv, return value, etc.)."""

    NODE_KIND: ClassVar[NodeKind] = NodeKind.SINK

    sink_type: str = "return"
    sink_target: str = ""

    def _infer_node_kind(self) -> NodeKind:
        return NodeKind.SINK

    def capacity_bound(self) -> float:
        return 0.0

    def to_dict(self) -> dict[str, Any]:
        d = super().to_dict()
        d["metadata"]["sink_type"] = self.sink_type
        d["metadata"]["sink_target"] = self.sink_target
        return d

    def __repr__(self) -> str:
        return (
            f"SinkNode(id={self.node_id!r}, "
            f"type={self.sink_type!r}, target={self.sink_target!r})"
        )


# ===================================================================
#  Node registry & factory
# ===================================================================

_NODE_REGISTRY: dict[str, Type[PipelineNode]] = {
    "PipelineNode": PipelineNode,
    "DataSourceNode": DataSourceNode,
    "PartitionNode": PartitionNode,
    "TransformNode": TransformNode,
    "FitNode": FitNode,
    "PredictNode": PredictNode,
    "PandasOpNode": PandasOpNode,
    "AggregationNode": AggregationNode,
    "FeatureEngineeringNode": FeatureEngineeringNode,
    "SelectionNode": SelectionNode,
    "CustomNode": CustomNode,
    "SinkNode": SinkNode,
}

# Mapping from OpType to the most appropriate node subclass
_OPTYPE_TO_NODE_CLASS: dict[OpType, Type[PipelineNode]] = {}
for _op in (OpType.READ_CSV, OpType.READ_PARQUET, OpType.READ_JSON,
            OpType.READ_EXCEL, OpType.READ_SQL, OpType.READ_HDF,
            OpType.READ_FEATHER):
    _OPTYPE_TO_NODE_CLASS[_op] = DataSourceNode

for _op in (OpType.TRAIN_TEST_SPLIT, OpType.KFOLD_SPLIT,
            OpType.STRATIFIED_KFOLD, OpType.GROUP_KFOLD,
            OpType.LEAVE_ONE_OUT):
    _OPTYPE_TO_NODE_CLASS[_op] = PartitionNode

for _op in (OpType.TRANSFORM_SK, OpType.FIT_TRANSFORM,
            OpType.STANDARD_SCALER, OpType.MINMAX_SCALER,
            OpType.ROBUST_SCALER, OpType.NORMALIZER,
            OpType.LABEL_ENCODER, OpType.ORDINAL_ENCODER,
            OpType.ONEHOT_ENCODER, OpType.TARGET_ENCODER,
            OpType.KBINS_DISCRETIZER, OpType.BINARIZER,
            OpType.IMPUTER, OpType.KNN_IMPUTER,
            OpType.INVERSE_TRANSFORM):
    _OPTYPE_TO_NODE_CLASS[_op] = TransformNode

for _op in (OpType.FIT, OpType.FIT_PREDICT):
    _OPTYPE_TO_NODE_CLASS[_op] = FitNode

for _op in (OpType.PREDICT, OpType.PREDICT_PROBA,
            OpType.PREDICT_LOG_PROBA, OpType.DECISION_FUNCTION,
            OpType.SCORE):
    _OPTYPE_TO_NODE_CLASS[_op] = PredictNode

for _op in (OpType.GROUPBY, OpType.AGG, OpType.AGGREGATE,
            OpType.ROLLING, OpType.EXPANDING, OpType.EWM,
            OpType.RESAMPLE, OpType.VALUE_COUNTS, OpType.DESCRIBE,
            OpType.CORR, OpType.COV):
    _OPTYPE_TO_NODE_CLASS[_op] = AggregationNode

for _op in (OpType.POLYNOMIAL_FEATURES, OpType.GET_DUMMIES,
            OpType.FACTORIZE, OpType.CUT, OpType.QCUT):
    _OPTYPE_TO_NODE_CLASS[_op] = FeatureEngineeringNode

for _op in (OpType.GETITEM, OpType.LOC, OpType.ILOC, OpType.AT,
            OpType.IAT, OpType.HEAD, OpType.TAIL, OpType.SAMPLE,
            OpType.NLARGEST, OpType.NSMALLEST, OpType.QUERY,
            OpType.FILTER, OpType.WHERE, OpType.MASK):
    _OPTYPE_TO_NODE_CLASS[_op] = SelectionNode

for _op in (OpType.TO_CSV, OpType.TO_PARQUET):
    _OPTYPE_TO_NODE_CLASS[_op] = SinkNode

_OPTYPE_TO_NODE_CLASS[OpType.CUSTOM] = CustomNode


class NodeFactory:
    """Create the appropriate :class:`PipelineNode` subclass from an OpType.

    The factory uses a static mapping from ``OpType`` to subclass.
    For unmapped ``OpType`` values, it returns a plain :class:`PandasOpNode`
    if the op is pandas, or :class:`PipelineNode` as a fallback.
    """

    @staticmethod
    def create(
        op_type: OpType,
        node_id: str | None = None,
        source_location: SourceLocation | None = None,
        input_schema: list[ColumnSchema] | None = None,
        output_schema: list[ColumnSchema] | None = None,
        shape: ShapeMetadata | None = None,
        provenance: dict[str, ProvenanceInfo] | None = None,
        metadata: dict[str, Any] | None = None,
        timestamp: float | None = None,
        **extra_kwargs: Any,
    ) -> PipelineNode:
        """Create a node of the appropriate subclass for *op_type*."""
        nid = node_id or _generate_node_id(op_type.value)
        loc = source_location or SourceLocation()
        in_schema = input_schema or []
        out_schema = output_schema or []
        shp = shape or ShapeMetadata(n_rows=0, n_cols=0)
        prov = provenance or {}
        meta = metadata or {}
        ts = timestamp if timestamp is not None else time.time()

        base = dict(
            node_id=nid,
            op_type=op_type,
            source_location=loc,
            input_schema=in_schema,
            output_schema=out_schema,
            shape=shp,
            provenance=prov,
            metadata=meta,
            timestamp=ts,
        )

        node_cls = _OPTYPE_TO_NODE_CLASS.get(op_type)

        if node_cls is DataSourceNode:
            return DataSourceNode(
                **base,
                file_path=extra_kwargs.get("file_path", meta.get("file_path", "")),
                format=extra_kwargs.get("format", meta.get("format", "csv")),
                origin=extra_kwargs.get("origin", Origin.TRAIN),
            )

        if node_cls is PartitionNode:
            return PartitionNode(
                **base,
                test_size=float(extra_kwargs.get("test_size", meta.get("test_size", 0.25))),
                random_state=extra_kwargs.get("random_state", meta.get("random_state")),
                stratify_column=extra_kwargs.get("stratify_column", meta.get("stratify_column")),
                n_splits=int(extra_kwargs.get("n_splits", meta.get("n_splits", 1))),
            )

        if node_cls is TransformNode:
            return TransformNode(
                **base,
                estimator_class=str(extra_kwargs.get("estimator_class", meta.get("estimator_class", ""))),
                is_fitted=bool(extra_kwargs.get("is_fitted", meta.get("is_fitted", False))),
                params=dict(extra_kwargs.get("params", meta.get("params", {}))),
            )

        if node_cls is FitNode:
            return FitNode(
                **base,
                estimator_class=str(extra_kwargs.get("estimator_class", meta.get("estimator_class", ""))),
                params=dict(extra_kwargs.get("params", meta.get("params", {}))),
                n_features_in=extra_kwargs.get("n_features_in", meta.get("n_features_in")),
                n_samples=extra_kwargs.get("n_samples", meta.get("n_samples")),
            )

        if node_cls is PredictNode:
            return PredictNode(
                **base,
                estimator_class=str(extra_kwargs.get("estimator_class", meta.get("estimator_class", ""))),
                predict_type=str(extra_kwargs.get("predict_type", meta.get("predict_type", "predict"))),
            )

        if node_cls is AggregationNode:
            return AggregationNode(
                **base,
                groupby_columns=list(extra_kwargs.get("groupby_columns", meta.get("groupby_columns", []))),
                agg_functions=list(extra_kwargs.get("agg_functions", meta.get("agg_functions", []))),
                as_index=bool(extra_kwargs.get("as_index", meta.get("as_index", True))),
            )

        if node_cls is FeatureEngineeringNode:
            return FeatureEngineeringNode(
                **base,
                technique=str(extra_kwargs.get("technique", meta.get("technique", ""))),
                source_columns=list(extra_kwargs.get("source_columns", meta.get("source_columns", []))),
                generated_columns=list(extra_kwargs.get("generated_columns", meta.get("generated_columns", []))),
            )

        if node_cls is SelectionNode:
            return SelectionNode(
                **base,
                selected_columns=list(extra_kwargs.get("selected_columns", meta.get("selected_columns", []))),
                selection_method=str(extra_kwargs.get("selection_method", meta.get("selection_method", "manual"))),
                row_mask_description=str(extra_kwargs.get("row_mask_description", meta.get("row_mask_description", ""))),
            )

        if node_cls is CustomNode:
            return CustomNode(
                **base,
                function_name=str(extra_kwargs.get("function_name", meta.get("function_name", ""))),
                source_code_hash=str(extra_kwargs.get("source_code_hash", meta.get("source_code_hash", ""))),
            )

        if node_cls is SinkNode:
            return SinkNode(
                **base,
                sink_type=str(extra_kwargs.get("sink_type", meta.get("sink_type", "return"))),
                sink_target=str(extra_kwargs.get("sink_target", meta.get("sink_target", ""))),
            )

        if op_type.is_pandas:
            return PandasOpNode(
                **base,
                pandas_method=str(extra_kwargs.get("pandas_method", meta.get("pandas_method", op_type.value))),
                axis=extra_kwargs.get("axis", meta.get("axis")),
                inplace=bool(extra_kwargs.get("inplace", meta.get("inplace", False))),
            )

        return PipelineNode(**base)

    @staticmethod
    def from_dict(data: Mapping[str, Any]) -> PipelineNode:
        """Deserialize a node from a dictionary."""
        return PipelineNode.from_dict(data)

    @staticmethod
    def registered_types() -> dict[str, Type[PipelineNode]]:
        """Return a copy of the node type registry."""
        return dict(_NODE_REGISTRY)


# ===================================================================
#  Utility helpers
# ===================================================================


def _sanitize_metadata(meta: dict[str, Any]) -> dict[str, Any]:
    """Ensure metadata values are JSON-serializable."""
    sanitized: dict[str, Any] = {}
    for k, v in meta.items():
        if isinstance(v, (str, int, float, bool, type(None))):
            sanitized[k] = v
        elif isinstance(v, (list, tuple)):
            sanitized[k] = [_sanitize_value(x) for x in v]
        elif isinstance(v, dict):
            sanitized[k] = _sanitize_metadata(v)
        elif isinstance(v, set):
            sanitized[k] = sorted(str(x) for x in v)
        elif isinstance(v, frozenset):
            sanitized[k] = sorted(str(x) for x in v)
        else:
            sanitized[k] = str(v)
    return sanitized


def _sanitize_value(v: Any) -> Any:
    """Convert a single value to JSON-serializable form."""
    if isinstance(v, (str, int, float, bool, type(None))):
        return v
    if isinstance(v, (list, tuple)):
        return [_sanitize_value(x) for x in v]
    if isinstance(v, dict):
        return _sanitize_metadata(v)
    return str(v)


def nodes_by_kind(nodes: Sequence[PipelineNode], kind: NodeKind) -> list[PipelineNode]:
    """Filter nodes by their inferred NodeKind."""
    return [n for n in nodes if n.node_kind == kind]


def nodes_by_op(nodes: Sequence[PipelineNode], op: OpType) -> list[PipelineNode]:
    """Filter nodes by their OpType."""
    return [n for n in nodes if n.op_type == op]


def leakage_risk_nodes(nodes: Sequence[PipelineNode]) -> list[PipelineNode]:
    """Return nodes whose OpType has ``may_leak == True``."""
    return [n for n in nodes if n.op_type.may_leak]


def sort_nodes_by_timestamp(nodes: Sequence[PipelineNode]) -> list[PipelineNode]:
    """Return nodes sorted by their observation timestamp."""
    return sorted(nodes, key=lambda n: n.timestamp)
