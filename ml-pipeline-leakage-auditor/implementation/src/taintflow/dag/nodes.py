"""
taintflow.dag.nodes -- DAG node representations for pipeline operations.

Each node in the Pipeline Information DAG (PI-DAG) represents a single
operation in the ML pipeline: a pandas transformation, a scikit-learn
estimator method call, a data split, or a user-defined function.
"""

from __future__ import annotations

import copy
import hashlib
import uuid
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import (
    Any,
    Dict,
    FrozenSet,
    List,
    Mapping,
    Optional,
    Sequence,
    Set,
    Tuple,
    Union,
)

from taintflow.core.types import (
    ColumnSchema,
    OpType,
    Origin,
    ProvenanceInfo,
    ShapeMetadata,
)


# -------------------------------------------------------------------
#  Node status tracking
# -------------------------------------------------------------------

class NodeStatus(Enum):
    """Execution status of a DAG node."""
    PENDING = auto()
    EXECUTING = auto()
    COMPLETED = auto()
    FAILED = auto()
    SKIPPED = auto()

    def is_terminal(self) -> bool:
        """Return True if this status represents a final state."""
        return self in (NodeStatus.COMPLETED, NodeStatus.FAILED, NodeStatus.SKIPPED)


class DataFlowDirection(Enum):
    """Direction of data flow through a node."""
    FORWARD = auto()
    BACKWARD = auto()
    BIDIRECTIONAL = auto()


# -------------------------------------------------------------------
#  Source location
# -------------------------------------------------------------------

@dataclass(frozen=True)
class SourceLocation:
    """Location in user source code.

    Attributes:
        file_path: Absolute or relative path to the source file.
        line_number: 1-indexed line number.
        column_offset: 0-indexed column offset within the line.
        end_line: Optional end line for multi-line expressions.
        end_column: Optional end column offset.
        function_name: Enclosing function or method name.
        class_name: Enclosing class name, if any.
    """
    file_path: str
    line_number: int
    column_offset: int = 0
    end_line: Optional[int] = None
    end_column: Optional[int] = None
    function_name: Optional[str] = None
    class_name: Optional[str] = None

    def __str__(self) -> str:
        loc = f"{self.file_path}:{self.line_number}"
        if self.column_offset > 0:
            loc += f":{self.column_offset}"
        if self.function_name:
            prefix = f"{self.class_name}." if self.class_name else ""
            loc += f" in {prefix}{self.function_name}"
        return loc

    def to_dict(self) -> Dict[str, Any]:
        return {
            "file_path": self.file_path,
            "line_number": self.line_number,
            "column_offset": self.column_offset,
            "end_line": self.end_line,
            "end_column": self.end_column,
            "function_name": self.function_name,
            "class_name": self.class_name,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> SourceLocation:
        return cls(
            file_path=d["file_path"],
            line_number=d["line_number"],
            column_offset=d.get("column_offset", 0),
            end_line=d.get("end_line"),
            end_column=d.get("end_column"),
            function_name=d.get("function_name"),
            class_name=d.get("class_name"),
        )


# -------------------------------------------------------------------
#  Column provenance at a node
# -------------------------------------------------------------------

@dataclass
class ColumnProvenance:
    """Tracks which columns flow into and out of a node.

    Attributes:
        input_columns: Columns consumed by this node.
        output_columns: Columns produced by this node.
        passthrough_columns: Columns that pass through unchanged.
        created_columns: Columns newly created by this node.
        dropped_columns: Columns removed by this node.
        renamed_map: Mapping from old column names to new names.
    """
    input_columns: FrozenSet[str] = field(default_factory=frozenset)
    output_columns: FrozenSet[str] = field(default_factory=frozenset)
    passthrough_columns: FrozenSet[str] = field(default_factory=frozenset)
    created_columns: FrozenSet[str] = field(default_factory=frozenset)
    dropped_columns: FrozenSet[str] = field(default_factory=frozenset)
    renamed_map: Dict[str, str] = field(default_factory=dict)

    def validate(self) -> List[str]:
        """Return a list of validation errors, empty if valid."""
        errors: List[str] = []
        if not self.passthrough_columns.issubset(self.input_columns):
            errors.append(
                "Passthrough columns must be a subset of input columns."
            )
        if not self.passthrough_columns.issubset(self.output_columns):
            errors.append(
                "Passthrough columns must be a subset of output columns."
            )
        if not self.created_columns.issubset(self.output_columns):
            errors.append(
                "Created columns must be a subset of output columns."
            )
        if not self.dropped_columns.issubset(self.input_columns):
            errors.append(
                "Dropped columns must be a subset of input columns."
            )
        overlap = self.passthrough_columns & self.created_columns
        if overlap:
            errors.append(
                f"Columns cannot be both passthrough and created: {overlap}"
            )
        return errors

    def to_dict(self) -> Dict[str, Any]:
        return {
            "input_columns": sorted(self.input_columns),
            "output_columns": sorted(self.output_columns),
            "passthrough_columns": sorted(self.passthrough_columns),
            "created_columns": sorted(self.created_columns),
            "dropped_columns": sorted(self.dropped_columns),
            "renamed_map": dict(self.renamed_map),
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> ColumnProvenance:
        return cls(
            input_columns=frozenset(d.get("input_columns", [])),
            output_columns=frozenset(d.get("output_columns", [])),
            passthrough_columns=frozenset(d.get("passthrough_columns", [])),
            created_columns=frozenset(d.get("created_columns", [])),
            dropped_columns=frozenset(d.get("dropped_columns", [])),
            renamed_map=d.get("renamed_map", {}),
        )


# -------------------------------------------------------------------
#  Execution metadata
# -------------------------------------------------------------------

@dataclass
class ExecutionMetadata:
    """Runtime metadata captured during pipeline execution.

    Attributes:
        wall_time_ms: Wall-clock execution time in milliseconds.
        memory_bytes: Peak memory usage in bytes.
        call_count: Number of times this operation was invoked.
        is_fitted: Whether this node represents a fitted estimator.
        estimator_class: Fully qualified class name of the estimator.
        estimator_params: Hyperparameters of the estimator.
        api_method: The API method called (fit, transform, predict, etc.).
    """
    wall_time_ms: float = 0.0
    memory_bytes: int = 0
    call_count: int = 1
    is_fitted: bool = False
    estimator_class: Optional[str] = None
    estimator_params: Dict[str, Any] = field(default_factory=dict)
    api_method: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "wall_time_ms": self.wall_time_ms,
            "memory_bytes": self.memory_bytes,
            "call_count": self.call_count,
            "is_fitted": self.is_fitted,
            "estimator_class": self.estimator_class,
            "estimator_params": self.estimator_params,
            "api_method": self.api_method,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> ExecutionMetadata:
        return cls(
            wall_time_ms=d.get("wall_time_ms", 0.0),
            memory_bytes=d.get("memory_bytes", 0),
            call_count=d.get("call_count", 1),
            is_fitted=d.get("is_fitted", False),
            estimator_class=d.get("estimator_class"),
            estimator_params=d.get("estimator_params", {}),
            api_method=d.get("api_method"),
        )


# -------------------------------------------------------------------
#  DAG Node
# -------------------------------------------------------------------

@dataclass
class DAGNode:
    """A node in the Pipeline Information DAG (PI-DAG).

    Each node represents a single operation observed during pipeline
    execution.  It carries structural metadata (operation type, source
    location), provenance information (train/test row fractions), and
    column-level data flow information.

    Attributes:
        node_id: Unique identifier for this node.
        op_type: The type of operation (from the OpType catalog).
        source_location: Where in the user's code this operation occurs.
        shape: Shape metadata (n_rows, n_cols, n_test_rows).
        provenance: Row provenance information (train/test fractions).
        column_provenance: Column-level input/output tracking.
        execution_meta: Runtime execution metadata.
        status: Current execution status of this node.
        label: Human-readable label for visualization.
        annotations: Arbitrary key-value annotations.
    """
    node_id: str = field(default_factory=lambda: str(uuid.uuid4())[:12])
    op_type: OpType = OpType.UNKNOWN
    source_location: Optional[SourceLocation] = None
    shape: Optional[ShapeMetadata] = None
    provenance: Optional[ProvenanceInfo] = None
    column_provenance: Optional[ColumnProvenance] = None
    execution_meta: ExecutionMetadata = field(default_factory=ExecutionMetadata)
    status: NodeStatus = NodeStatus.PENDING
    label: str = ""
    annotations: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.label:
            self.label = f"{self.op_type.value}@{self.node_id[:8]}"

    @property
    def test_fraction(self) -> float:
        """Return the fraction of test rows at this node."""
        if self.provenance is not None:
            return self.provenance.test_fraction
        if self.shape is not None and self.shape.n_rows > 0:
            return self.shape.n_test_rows / self.shape.n_rows
        return 0.0

    @property
    def is_estimator(self) -> bool:
        """Return True if this node represents a fitted estimator."""
        return self.execution_meta.is_fitted

    @property
    def is_split_point(self) -> bool:
        """Return True if this node is a train/test split."""
        return self.op_type in (
            OpType.TRAIN_TEST_SPLIT,
            OpType.CROSS_VAL_SPLIT,
        )

    @property
    def fingerprint(self) -> str:
        """Compute a content-based fingerprint for deduplication."""
        parts = [
            self.op_type.value,
            str(self.source_location) if self.source_location else "?",
            str(self.shape) if self.shape else "?",
        ]
        content = "|".join(parts)
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def with_status(self, status: NodeStatus) -> DAGNode:
        """Return a copy of this node with updated status."""
        new_node = copy.copy(self)
        new_node.status = status
        return new_node

    def validate(self) -> List[str]:
        """Validate node consistency, returning a list of error messages."""
        errors: List[str] = []
        if not self.node_id:
            errors.append("Node must have a non-empty node_id.")
        if self.shape is not None:
            if self.shape.n_rows < 0:
                errors.append("n_rows must be non-negative.")
            if self.shape.n_cols < 0:
                errors.append("n_cols must be non-negative.")
            if self.shape.n_test_rows < 0:
                errors.append("n_test_rows must be non-negative.")
            if self.shape.n_test_rows > self.shape.n_rows:
                errors.append("n_test_rows cannot exceed n_rows.")
        if self.column_provenance is not None:
            errors.extend(self.column_provenance.validate())
        return errors

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a JSON-compatible dictionary."""
        result: Dict[str, Any] = {
            "node_id": self.node_id,
            "op_type": self.op_type.value,
            "status": self.status.name,
            "label": self.label,
            "annotations": self.annotations,
        }
        if self.source_location is not None:
            result["source_location"] = self.source_location.to_dict()
        if self.shape is not None:
            result["shape"] = {
                "n_rows": self.shape.n_rows,
                "n_cols": self.shape.n_cols,
                "n_test_rows": self.shape.n_test_rows,
            }
        if self.provenance is not None:
            result["provenance"] = {
                "test_fraction": self.provenance.test_fraction,
                "origins": [o.value for o in self.provenance.origin_set],
            }
        if self.column_provenance is not None:
            result["column_provenance"] = self.column_provenance.to_dict()
        result["execution_meta"] = self.execution_meta.to_dict()
        return result

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> DAGNode:
        """Deserialize from a dictionary."""
        source_loc = None
        if "source_location" in d:
            source_loc = SourceLocation.from_dict(d["source_location"])

        shape = None
        if "shape" in d:
            s = d["shape"]
            shape = ShapeMetadata(
                n_rows=s["n_rows"],
                n_cols=s["n_cols"],
                n_test_rows=s["n_test_rows"],
            )

        provenance = None
        if "provenance" in d:
            p = d["provenance"]
            provenance = ProvenanceInfo(
                test_fraction=p["test_fraction"],
                origin_set=frozenset(Origin.from_str(o) for o in p["origins"]),
            )

        col_prov = None
        if "column_provenance" in d:
            col_prov = ColumnProvenance.from_dict(d["column_provenance"])

        exec_meta = ExecutionMetadata.from_dict(d.get("execution_meta", {}))

        return cls(
            node_id=d["node_id"],
            op_type=OpType(d["op_type"]),
            source_location=source_loc,
            shape=shape,
            provenance=provenance,
            column_provenance=col_prov,
            execution_meta=exec_meta,
            status=NodeStatus[d.get("status", "PENDING")],
            label=d.get("label", ""),
            annotations=d.get("annotations", {}),
        )

    def __repr__(self) -> str:
        return (
            f"DAGNode(id={self.node_id!r}, op={self.op_type.value}, "
            f"label={self.label!r})"
        )


# -------------------------------------------------------------------
#  Specialized node types
# -------------------------------------------------------------------

@dataclass
class DataSourceNode(DAGNode):
    """Node representing a data source (CSV load, database query, etc.).

    Attributes:
        source_path: Path or URI of the data source.
        format_type: Data format (csv, parquet, sql, etc.).
        row_count: Number of rows loaded.
        column_names: List of column names in the data source.
    """
    source_path: Optional[str] = None
    format_type: str = "unknown"
    row_count: int = 0
    column_names: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.op_type == OpType.UNKNOWN:
            self.op_type = OpType.DATA_SOURCE
        super().__post_init__()


@dataclass
class SplitNode(DAGNode):
    """Node representing a train/test split operation.

    Attributes:
        split_ratio: Test size ratio (0.0 to 1.0).
        random_state: Random seed used for splitting.
        stratify_column: Column used for stratified splitting.
        shuffle: Whether data was shuffled before splitting.
    """
    split_ratio: float = 0.2
    random_state: Optional[int] = None
    stratify_column: Optional[str] = None
    shuffle: bool = True

    def __post_init__(self) -> None:
        if self.op_type == OpType.UNKNOWN:
            self.op_type = OpType.TRAIN_TEST_SPLIT
        super().__post_init__()

    @property
    def train_ratio(self) -> float:
        return 1.0 - self.split_ratio


@dataclass
class TransformNode(DAGNode):
    """Node representing a data transformation.

    Attributes:
        transformer_class: Fully qualified class name of the transformer.
        is_fit: Whether this is a fit, transform, or fit_transform call.
        fitted_params: Dictionary of fitted parameters.
    """
    transformer_class: str = ""
    is_fit: bool = False
    fitted_params: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.op_type == OpType.UNKNOWN:
            self.op_type = OpType.TRANSFORM
        super().__post_init__()


@dataclass
class EstimatorNode(DAGNode):
    """Node representing a fitted scikit-learn estimator.

    Attributes:
        estimator_type: Type of estimator (classifier, regressor, etc.).
        n_features_in: Number of input features.
        n_features_out: Number of output features.
        fitted_attributes: Names of fitted attributes.
    """
    estimator_type: str = "unknown"
    n_features_in: int = 0
    n_features_out: int = 0
    fitted_attributes: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.op_type == OpType.UNKNOWN:
            self.op_type = OpType.ESTIMATOR_FIT
        self.execution_meta.is_fitted = True
        super().__post_init__()


@dataclass
class AggregationNode(DAGNode):
    """Node representing an aggregation operation (groupby, etc.).

    Attributes:
        group_columns: Columns used for grouping.
        agg_function: Aggregation function name.
        n_groups: Number of groups.
    """
    group_columns: List[str] = field(default_factory=list)
    agg_function: str = ""
    n_groups: int = 0

    def __post_init__(self) -> None:
        if self.op_type == OpType.UNKNOWN:
            self.op_type = OpType.AGGREGATION
        super().__post_init__()


@dataclass
class MergeNode(DAGNode):
    """Node representing a merge/join operation.

    Attributes:
        join_type: Type of join (inner, left, right, outer, cross).
        left_on: Column(s) from left DataFrame.
        right_on: Column(s) from right DataFrame.
        left_shape: Shape of left input.
        right_shape: Shape of right input.
    """
    join_type: str = "inner"
    left_on: List[str] = field(default_factory=list)
    right_on: List[str] = field(default_factory=list)
    left_shape: Optional[Tuple[int, int]] = None
    right_shape: Optional[Tuple[int, int]] = None

    def __post_init__(self) -> None:
        if self.op_type == OpType.UNKNOWN:
            self.op_type = OpType.MERGE
        super().__post_init__()


@dataclass
class PipelineStageNode(DAGNode):
    """Node representing a stage within an sklearn Pipeline.

    Attributes:
        pipeline_name: Name of the enclosing Pipeline.
        stage_index: Index of this stage within the Pipeline.
        stage_name: Name given to this stage.
        sub_nodes: Nodes representing sub-operations of this stage.
    """
    pipeline_name: str = ""
    stage_index: int = 0
    stage_name: str = ""
    sub_nodes: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.op_type == OpType.UNKNOWN:
            self.op_type = OpType.PIPELINE_STAGE
        super().__post_init__()


@dataclass
class CrossValidationNode(DAGNode):
    """Node representing a cross-validation wrapper.

    Attributes:
        n_splits: Number of CV folds.
        cv_type: Type of cross-validation (KFold, StratifiedKFold, etc.).
        scoring: Scoring metric used.
        fold_nodes: Node IDs for each fold's operations.
    """
    n_splits: int = 5
    cv_type: str = "KFold"
    scoring: Optional[str] = None
    fold_nodes: List[List[str]] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.op_type == OpType.UNKNOWN:
            self.op_type = OpType.CROSS_VAL_SPLIT
        super().__post_init__()


@dataclass
class ColumnTransformerNode(DAGNode):
    """Node representing a ColumnTransformer.

    Attributes:
        transformer_specs: List of (name, transformer_class, columns) tuples.
        remainder: What to do with remaining columns ('drop' or 'passthrough').
        n_transformers: Number of sub-transformers.
    """
    transformer_specs: List[Tuple[str, str, List[str]]] = field(
        default_factory=list
    )
    remainder: str = "drop"
    n_transformers: int = 0

    def __post_init__(self) -> None:
        if self.op_type == OpType.UNKNOWN:
            self.op_type = OpType.COLUMN_TRANSFORMER
        self.n_transformers = len(self.transformer_specs)
        super().__post_init__()


@dataclass
class FeatureUnionNode(DAGNode):
    """Node representing a FeatureUnion.

    Attributes:
        transformer_names: Names of the combined transformers.
        n_transformers: Number of sub-transformers.
    """
    transformer_names: List[str] = field(default_factory=list)
    n_transformers: int = 0

    def __post_init__(self) -> None:
        if self.op_type == OpType.UNKNOWN:
            self.op_type = OpType.FEATURE_UNION
        self.n_transformers = len(self.transformer_names)
        super().__post_init__()


@dataclass
class ImputerNode(DAGNode):
    """Node representing an imputation operation.

    Attributes:
        strategy: Imputation strategy (mean, median, most_frequent, constant).
        fill_value: Fill value when strategy is 'constant'.
        missing_indicator: Whether a missing indicator was added.
    """
    strategy: str = "mean"
    fill_value: Optional[Any] = None
    missing_indicator: bool = False

    def __post_init__(self) -> None:
        if self.op_type == OpType.UNKNOWN:
            self.op_type = OpType.IMPUTATION
        super().__post_init__()


@dataclass
class EncoderNode(DAGNode):
    """Node representing a categorical encoding operation.

    Attributes:
        encoding_type: Type of encoding (onehot, ordinal, target, label).
        categories: Categories per feature.
        n_categories: Total number of categories across all features.
    """
    encoding_type: str = "unknown"
    categories: Dict[str, List[str]] = field(default_factory=dict)
    n_categories: int = 0

    def __post_init__(self) -> None:
        if self.op_type == OpType.UNKNOWN:
            self.op_type = OpType.ENCODING
        super().__post_init__()


# -------------------------------------------------------------------
#  Node factory
# -------------------------------------------------------------------

class NodeFactory:
    """Factory for creating DAG nodes from observed operations.

    The factory maintains a registry of operation type -> node class
    mappings and provides methods for creating nodes with appropriate
    defaults based on the observed operation.
    """

    _registry: Dict[OpType, type] = {
        OpType.DATA_SOURCE: DataSourceNode,
        OpType.TRAIN_TEST_SPLIT: SplitNode,
        OpType.CROSS_VAL_SPLIT: CrossValidationNode,
        OpType.TRANSFORM: TransformNode,
        OpType.FIT_TRANSFORM: TransformNode,
        OpType.ESTIMATOR_FIT: EstimatorNode,
        OpType.AGGREGATION: AggregationNode,
        OpType.MERGE: MergeNode,
        OpType.PIPELINE_STAGE: PipelineStageNode,
        OpType.COLUMN_TRANSFORMER: ColumnTransformerNode,
        OpType.FEATURE_UNION: FeatureUnionNode,
        OpType.IMPUTATION: ImputerNode,
        OpType.ENCODING: EncoderNode,
    }

    @classmethod
    def create(
        cls,
        op_type: OpType,
        source_location: Optional[SourceLocation] = None,
        shape: Optional[ShapeMetadata] = None,
        provenance: Optional[ProvenanceInfo] = None,
        **kwargs: Any,
    ) -> DAGNode:
        """Create a node of the appropriate type for the given operation.

        Parameters
        ----------
        op_type : OpType
            The type of operation this node represents.
        source_location : SourceLocation, optional
            Where in the user's source code the operation occurs.
        shape : ShapeMetadata, optional
            Shape metadata for the operation.
        provenance : ProvenanceInfo, optional
            Row provenance information.
        **kwargs
            Additional keyword arguments passed to the node constructor.

        Returns
        -------
        DAGNode
            A node instance of the appropriate subclass.
        """
        node_class = cls._registry.get(op_type, DAGNode)
        return node_class(
            op_type=op_type,
            source_location=source_location,
            shape=shape,
            provenance=provenance,
            **kwargs,
        )

    @classmethod
    def register(cls, op_type: OpType, node_class: type) -> None:
        """Register a custom node class for an operation type."""
        if not issubclass(node_class, DAGNode):
            raise TypeError(
                f"Node class must be a subclass of DAGNode, got {node_class}"
            )
        cls._registry[op_type] = node_class

    @classmethod
    def supported_types(cls) -> Set[OpType]:
        """Return the set of operation types with registered node classes."""
        return set(cls._registry.keys())

    @classmethod
    def from_execution_trace(
        cls,
        op_type: OpType,
        class_name: str,
        method_name: str,
        file_path: str,
        line_number: int,
        input_shape: Optional[Tuple[int, int]] = None,
        output_shape: Optional[Tuple[int, int]] = None,
        n_test_rows: int = 0,
        **kwargs: Any,
    ) -> DAGNode:
        """Create a node from execution trace information.

        This is the primary entry point used by the instrumentation
        layer during pipeline execution.
        """
        source_loc = SourceLocation(
            file_path=file_path,
            line_number=line_number,
            function_name=method_name,
            class_name=class_name,
        )

        shape = None
        if output_shape is not None:
            shape = ShapeMetadata(
                n_rows=output_shape[0],
                n_cols=output_shape[1],
                n_test_rows=n_test_rows,
            )

        exec_meta = ExecutionMetadata(
            estimator_class=class_name,
            api_method=method_name,
            is_fitted=method_name in ("fit", "fit_transform"),
        )

        node = cls.create(
            op_type=op_type,
            source_location=source_loc,
            shape=shape,
            execution_meta=exec_meta,
            **kwargs,
        )
        return node
